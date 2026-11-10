"""Matched continuation with causal stagnation-triggered Beta broadening."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from myo.arm_exploration import (ArmExplorationEnv, ExplorationSpec, evaluation_job,
                                load_exploration, make_exploration_model)
from myo.arm_reward_curriculum import RewardStage, load_reward_curriculum
from myo.arm_training_audit import ConditionMonitor, EpisodeAudit
from myo.arm_workload import save_workload, scenarios, spec_from_dict, stage_passed, validation_rank
from myo.arm_workload_evaluation import merge_results
from myo.train_vertical_ppo import write_json


def worker_env(spec, reward, exploration, worker, workers):
    return ConditionMonitor(ArmExplorationEnv(spec, reward, exploration, worker=worker, workers=workers))


def evaluate(path, conditions):
    indices = [list(range(i, len(conditions), 2)) for i in range(min(2, len(conditions)))]
    chunks = [[conditions[j] for j in group] for group in indices]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluation_job, path, chunk) for chunk in chunks]
        rows = [None]*len(conditions)
        for group, job in zip(indices, jobs):
            for index, row in zip(group, job.result()['cases']):
                rows[index] = row
        return merge_results([{'cases': rows}])


def tracking_improved(result, reference):
    return (result['failures'] < reference['failures'] or (
        result['failures'] == reference['failures'] and (
        (result['passed'] > reference['passed'] and result['score_deg'] <= 1.05*reference['score_deg'])
        or (result['passed'] >= reference['passed'] and result['score_deg'] <= .98*reference['score_deg']))))


def full_stage_passed(validation, training):
    return (stage_passed(validation) and training['failures'] == 0
            and training['passed'] == training['total'])


def effort_rank(validation, training, initial_validation, initial_training):
    """A cheap failed pose must never displace a precise parent."""
    eligible = all(current['failures'] == 0 and current['passed'] == current['total']
        and current['score_deg'] <= baseline['score_deg']+.25
        for current, baseline in ((validation, initial_validation), (training, initial_training)))
    effort = .5*(validation['activation_squared_mean']+training['activation_squared_mean'])
    return (not eligible, effort, *validation_rank(validation), training['score_deg'])


def reconstruct_stopping_state(history):
    reference, blocks, progress = history[0]['validation'], 0, False
    for row in history[1:]:
        if tracking_improved(row['validation'], reference):
            reference, progress = row['validation'], True
        if row['added_transitions'] % 32768 == 0:
            blocks = 0 if progress else blocks + 1
            progress = False
    return reference, blocks


class ExplorationAudit(BaseCallback):
    def __init__(self):
        super().__init__()
        self.count = self.pressured = 0
        self.high = np.zeros(6)
        self.high_pressured = np.zeros(6)
        self.sum_action = np.zeros(6)
        self.sum_activation = np.zeros(6)

    def _on_step(self):
        actions = np.asarray(self.locals['actions'])
        pressured = self.locals['obs_tensor'][:, -1].detach().cpu().numpy() > .5
        self.count += len(actions)
        self.pressured += int(pressured.sum())
        self.high += (actions > .95).sum(axis=0)
        self.high_pressured += (actions[pressured] > .95).sum(axis=0)
        self.sum_action += actions.sum(axis=0)
        self.sum_activation += np.array([i['activation'] for i in self.locals['infos']]).sum(axis=0)
        return True

    def report(self):
        return {'transitions': self.count, 'pressured_transitions': self.pressured,
            'strong_action_fraction': (self.high / max(1, self.count)).tolist(),
            'strong_action_fraction_pressured': (self.high_pressured / max(1, self.pressured)).tolist(),
            'mean_action': (self.sum_action / max(1, self.count)).tolist(),
            'mean_activation': (self.sum_activation / max(1, self.count)).tolist()}


def train(args):
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root/'checkpoints').mkdir()
    (args.root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    module = Path(__file__).resolve().parents[1]/'src/myo/arm_exploration.py'
    (args.root/'exploration_source.py').write_bytes(module.read_bytes())
    audit_source = Path(__file__).resolve().parents[1]/'src/myo/arm_training_audit.py'
    (args.root/'audit_source.py').write_bytes(audit_source.read_bytes())
    parent_metadata = json.loads(Path(str(args.parent)+'.metadata.json').read_text())
    loader = (load_exploration if parent_metadata['contract']['schema'] == 'myo.arm_exploration.v1'
              else load_reward_curriculum)
    parent, checked, parent_meta = loader(args.parent)
    checked.close()
    spec = spec_from_dict(parent_meta['contract']['spec'])
    reward = RewardStage(**parent_meta['contract']['reward_stage'])
    if args.reward_mode is not None:
        reward = RewardStage(args.reward_mode)
    overrides = {name: value for name, value in (
        ('activation_squared_weight', args.activation_squared_weight),
        ('active_fraction_weight', args.active_fraction_weight)) if value is not None}
    spec = replace(spec, **overrides)
    if reward.mode == 'position_only' and (spec.activation_squared_weight or spec.active_fraction_weight):
        raise ValueError('Position-only reward must not silently discard requested effort weights')
    exploration = ExplorationSpec(mode=args.mode, maximum_temperature=args.temperature)
    validation = json.loads(args.validation.read_text())
    train_conditions = (scenarios(spec.cases, spec.masses, spec.tilts, spec.capacities, spec.seconds)
                        if args.full_train_gate else [])
    checking = ArmExplorationEnv(spec, reward, exploration)
    contract = checking.contract()
    vector = SubprocVecEnv([partial(worker_env, spec, reward, exploration, i, args.workers)
                           for i in range(args.workers)], start_method='spawn')
    started = time.monotonic()
    try:
        model = make_exploration_model(vector, parent, contract,
            learning_rate=args.lr, n_steps=1024//args.workers)
        training = {'schema': 'myo.arm_exploration_training.v1', 'seed': 0,
            'parent': str(args.parent), 'parent_sha256': parent_meta['checkpoint_sha256'],
            'source_steps': parent.num_timesteps, 'source_optimizer_steps': parent.optimizer_steps,
            'transfer': 'same tensors/RMSprop; new pressure columns zero; environment/RNG restarted once',
            'workers': args.workers, 'budget': args.steps, 'validation': validation,
            'adaptive_stop': args.adaptive_stop, 'minimum_steps': args.minimum_steps,
            'full_train_gate': args.full_train_gate,
            'episode_audit': args.audit_episodes,
            'reward_stage': asdict(reward),
            'reward_spec': {'activation_squared_weight': spec.activation_squared_weight,
                            'active_fraction_weight': spec.active_fraction_weight},
            'effort_selection': ('all train/validation passes; mean errors <= initial+.25deg; '
                'then minimum equally weighted train/validation activation squared proxy'
                if args.efficiency_selection else None),
            'audit_sha256': hashlib.sha256(audit_source.read_bytes()).hexdigest(),
            'stop_rule': ('current train100% plus validation gate or two32768 blocks '
                          'without accumulated2% train/validation progress; cap budget'
                          if args.full_train_gate else
                          'validation gate or two32768 blocks without accumulated2% progress; cap budget'),
            'validation_sha256': hashlib.sha256(args.validation.read_bytes()).hexdigest(),
            'exploration': asdict(exploration), 'evaluation': 'deterministic Beta means; no added noise',
            'ppo': {'lr': args.lr, 'steps_per_env': 1024//args.workers, 'batch': 128, 'epochs': 10,
                'gamma': .99, 'gae_lambda': .95, 'clip': .2, 'entropy': .001,
                'value': .5, 'grad_norm': .5, 'target_kl': .03, 'optimizer': 'RMSprop'},
            'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')},
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        source_steps, source_sgd = model.num_timesteps, model.optimizer_steps
        summary = {'status': 'running', 'training': training, 'history': [], 'updates': []}
        best = None
        reference = None
        train_reference = None
        initial_result = initial_train = None
        progress_in_block = False
        no_progress_blocks = 0
        if args.resume_stop_reference is not None:
            if (args.parent.name != 'last.zip' or args.resume_stop_reference.resolve()
                    != (args.parent.parent/'summary.json').resolve()):
                raise ValueError('Carry stopping state only from the same last checkpoint phase')
            previous = json.loads(args.resume_stop_reference.read_text())
            if (previous['status'] != 'training_complete' or previous['added_transitions'] % 32768
                    or previous['training']['exploration'] != asdict(exploration)
                    or previous['training']['validation_sha256'] != training['validation_sha256']
                    or parent_meta['steps'] != previous['training']['source_steps']+previous['added_transitions']):
                raise ValueError('Stopping reference requires a completed compatible block boundary')
            if 'progress_reference' in previous:
                reference = previous['progress_reference']
            else:
                reference, reconstructed = reconstruct_stopping_state(previous['history'])
                if reconstructed != previous['blocks_without_progress']:
                    raise ValueError('Stopping history disagrees with its saved counter')
            no_progress_blocks = previous.get('blocks_without_progress',
                                               reconstruct_stopping_state(previous['history'])[1])
            if args.full_train_gate:
                train_reference = previous.get('training_progress_reference')
            training['inherited_stop_reference'] = {
                'path': str(args.resume_stop_reference),
                'sha256': hashlib.sha256(args.resume_stop_reference.read_bytes()).hexdigest(),
                'blocks_without_progress': no_progress_blocks,
                'rule': 'retain previous references and block counter; initial check is not progress; '
                        'initialize train reference if absent in parent phase'}
        summary['blocks_without_progress'] = no_progress_blocks
        audit = ExplorationAudit()
        episodes = EpisodeAudit(args.root) if args.audit_episodes else None

        def check():
            nonlocal best, reference, train_reference, progress_in_block, initial_result, initial_train
            save_workload(model, checking, args.root/'candidate.zip', training)
            combined = evaluate(args.root/'candidate.zip', validation+train_conditions)
            result = merge_results([{'cases': combined['cases'][:len(validation)]}])
            train_result = (merge_results([{'cases': combined['cases'][len(validation):]}])
                            if train_conditions else None)
            if initial_result is None:
                initial_result, initial_train = result, train_result
            rank = validation_rank(result)
            if train_result is not None:
                rank = (not full_stage_passed(result, train_result),
                        train_result['failures'], -train_result['passed']/train_result['total'],
                        *rank, train_result['score_deg'])
            if args.efficiency_selection:
                rank = effort_rank(result, train_result, initial_result, initial_train)
            if best is None or rank < best:
                best = rank
                save_workload(model, checking, args.root/'best.zip', training)
                summary.update(best_added=model.num_timesteps-source_steps, best_validation=result)
                if train_result is not None:
                    summary['best_training'] = train_result
            if reference is None:
                reference = result
            elif summary['history'] and tracking_improved(result, reference):
                reference = result
                progress_in_block = True
            if train_result is not None:
                if train_reference is None:
                    train_reference = train_result
                elif summary['history'] and tracking_improved(train_result, train_reference):
                    train_reference = train_result
                    progress_in_block = True
                summary['training_progress_reference'] = train_reference
            summary['progress_reference'] = reference
            added = model.num_timesteps-source_steps
            summary['history'].append({'added_transitions': added, 'validation': result,
                **({'training_assessment': train_result} if train_result is not None else {}),
                'added_optimizer_steps': model.optimizer_steps-source_sgd, 'exploration': audit.report()})
            save_workload(model, checking, args.root/'checkpoints'/f'added_{added}.zip', training)
            write_json(args.root/'summary.json', summary)
            print(json.dumps({'mode': args.mode, 'added': added, 'passed': result['passed'],
                'total': result['total'], 'error_deg': result['score_deg'], 'failures': result['failures'],
                **({'train_passed': train_result['passed'], 'train_total': train_result['total'],
                    'train_error_deg': train_result['score_deg']} if train_result is not None else {}),
                'activation_squared_mean': result['activation_squared_mean'],
                'sgd': model.optimizer_steps-source_sgd, 'elapsed_s': time.monotonic()-started}), flush=True)

        save_workload(model, checking, args.root/'initial.zip', training)
        model.set_random_seed(0)
        check()
        while args.steps == 0 or model.num_timesteps-source_steps < args.steps:
            model.learn(1024, reset_num_timesteps=False,
                        callback=[audit, episodes] if episodes is not None else audit)
            added = model.num_timesteps-source_steps
            summary['updates'].append({'added': added, **model.last_audit})
            if added % 4096 == 0:
                summary.update(added_transitions=added, exploration=audit.report(),
                               elapsed_s=time.monotonic()-started)
                write_json(args.root/'summary.json', summary)
                print(json.dumps({'mode': args.mode, 'progress': added,
                    'pressure_fraction': audit.pressured/max(1,audit.count)}), flush=True)
            if added % args.check_steps == 0:
                check()
            if added % 32768 == 0:
                no_progress_blocks = 0 if progress_in_block else no_progress_blocks + 1
                progress_in_block = False
                summary['blocks_without_progress'] = no_progress_blocks
            if args.adaptive_stop and added >= args.minimum_steps:
                current = summary['history'][-1]
                passed = (full_stage_passed(current['validation'], current['training_assessment'])
                          if args.full_train_gate else stage_passed(summary['best_validation']))
                if added % args.check_steps == 0 and passed:
                    summary['stop_reason'] = ('train_and_validation_stage_passed'
                                              if args.full_train_gate else 'validation_stage_passed')
                    break
                if added % 32768 == 0 and no_progress_blocks >= 2:
                    summary['stop_reason'] = ('train_validation_plateau_two_blocks'
                                              if args.full_train_gate else 'validation_plateau_two_blocks')
                    break
        save_workload(model, checking, args.root/'last.zip', training)
        summary.update(status='training_complete',
            added_transitions=model.num_timesteps-source_steps,
            added_optimizer_steps=model.optimizer_steps-source_sgd,
            exploration=audit.report(), elapsed_s=time.monotonic()-started)
        if episodes is not None:
            summary['training_episode_audit'] = {'episodes': episodes.episode_count,
                'passed': episodes.passed_count, 'steps': 'training_steps.jsonl.gz',
                'results': 'training_episodes.jsonl',
                'advantages': 'raw GAE, before PPO minibatch normalization',
                'logp_change': 'net same-input logp after minus before each PPO rollout update'}
        summary.setdefault('stop_reason', 'budget_cap' if args.adaptive_stop else 'fixed_comparison_budget')
        write_json(args.root/'summary.json', summary)
    finally:
        if 'episodes' in locals() and episodes is not None:
            episodes.close()
        vector.close()
        checking.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('off', 'stall'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=Path('runs/arm_reward_curriculum/position_continuation/best.zip'))
    p.add_argument('--validation', type=Path, default=Path('runs/arm_reward_curriculum/validation.json'))
    p.add_argument('--steps', type=int, default=65536,
                   help='Transition cap;0 means adaptive gate/plateau stopping without a fixed cap')
    p.add_argument('--check-steps', type=int, default=16384)
    p.add_argument('--workers', type=int, choices=(1,2,4), default=4)
    p.add_argument('--temperature', type=float, default=16.)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--adaptive-stop', action='store_true')
    p.add_argument('--minimum-steps', type=int, default=65536)
    p.add_argument('--resume-stop-reference', type=Path)
    p.add_argument('--full-train-gate', action='store_true',
                   help='Require all deterministic training cases plus validation; track progress on both')
    p.add_argument('--audit-episodes', action='store_true',
                   help='Record training trajectories, raw GAE and before/after action likelihoods')
    p.add_argument('--reward-mode', choices=('position_only', 'position_efficiency'))
    p.add_argument('--activation-squared-weight', type=float)
    p.add_argument('--active-fraction-weight', type=float)
    p.add_argument('--efficiency-selection', action='store_true',
                   help='Fixed-budget study: preserve all initial train/validation passes before minimizing effort')
    args = p.parse_args()
    if (args.steps < 0 or (args.steps == 0 and not args.adaptive_stop) or args.check_steps <= 0
            or args.steps % args.check_steps or args.check_steps % 1024):
        p.error('Use complete validation intervals, divisible by1024')
    if args.adaptive_stop and (32768 % args.check_steps or args.minimum_steps % 32768):
        p.error('Adaptive checks must divide32768; minimum steps must be divisible by32768')
    if any(value is not None and (not np.isfinite(value) or value < 0)
           for value in (args.activation_squared_weight, args.active_fraction_weight)):
        p.error('Effort weights must be finite and nonnegative')
    if args.efficiency_selection and (not args.full_train_gate or args.adaptive_stop):
        p.error('Effort selection requires a fixed budget and full train assessment')
    if args.resume_stop_reference is not None and (args.reward_mode is not None or
            args.activation_squared_weight is not None or args.active_fraction_weight is not None):
        p.error('Changed rewards start a new progress reference')
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()
