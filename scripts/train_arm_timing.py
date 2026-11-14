"""Fresh PPO with 10ms commands/transport and previously selected arm settings."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from myo.arm_exploration import ExplorationSpec
from myo.arm_timing import ArmTimingEnv, evaluation_job, load_timing, make_fresh_model
from myo.arm_workload import save_workload, scenarios, spec_from_dict, validation_rank
from myo.arm_workload_evaluation import merge_results


class TrainingAudit(BaseCallback):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.count = self.pressured = 0
        self.action_sum = np.zeros(6)
        self.strong = np.zeros(6)

    def _on_step(self):
        actions = np.asarray(self.locals['actions'])
        self.count += len(actions)
        self.pressured += int((self.locals['obs_tensor'][:, -1] > .5).sum())
        self.action_sum += actions.sum(axis=0)
        self.strong += (actions > .95).sum(axis=0)
        return True

    def report(self):
        return {'transitions': self.count, 'pressure_fraction': self.pressured/max(1, self.count),
            'mean_action': (self.action_sum/max(1, self.count)).tolist(),
            'strong_action_fraction': (self.strong/max(1, self.count)).tolist()}


def worker(spec, exploration, i, count):
    torch.set_num_threads(1)
    return Monitor(ArmTimingEnv(spec, exploration, worker=i, workers=count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
        default=Path('runs/arm_exploration/stall_audited_continuation/best.zip.metadata.json'))
    parser.add_argument('--steps', type=int, default=1048576)
    parser.add_argument('--minimum-steps', type=int, default=262144)
    parser.add_argument('--check-every', type=int, default=32768)
    parser.add_argument('--plateau-checks', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or 1024 % args.workers or args.steps <= 0 or args.check_every % 1024:
        parser.error('Require positive budget, workers dividing 1024, and rollout-aligned checks')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    protocol = json.loads(args.protocol.read_text())
    spec = spec_from_dict(protocol['contract']['spec'])
    exploration = ExplorationSpec(**protocol['contract']['exploration'])
    validation = protocol['training']['validation']
    training_cases = scenarios(spec.cases, spec.masses, spec.tilts, spec.capacities, spec.seconds)
    known_test = json.loads(Path('runs/arm_exploration/audited_comparison.json').read_text())[
        'roots'][0]['checkpoints']['best']['test']['cases']
    fields = ('case', 'payload_mass', 'tilt_deg', 'capacity', 'seconds')
    test_cases = [{key: row[key] for key in fields} for row in known_test]
    checking = ArmTimingEnv(spec, exploration)
    vector = SubprocVecEnv([partial(worker, spec, exploration, i, args.workers)
                           for i in range(args.workers)], start_method='spawn')
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
    started = time.monotonic()
    training = {'schema': 'myo.arm_timing_training.v1', 'seed': 0,
        'initialization': 'fresh random weights and empty RMSprop state; no demonstrations or transferred weights',
        'protocol_reference': str(args.protocol),
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'steps_cap': args.steps, 'minimum_steps': args.minimum_steps,
        'check_every': args.check_every, 'plateau_checks': args.plateau_checks,
        'stop': 'all train and validation pass, or after minimum_steps four checks without accumulated 2% validation progress; budget cap',
        'ppo': {'lr': 1e-5, 'optimizer': 'RMSprop', 'n_steps': 1024//args.workers,
            'workers': args.workers, 'batch_size': 128, 'n_epochs': 10, 'gamma': .99,
            'gae_lambda': .95, 'clip': .2, 'entropy': .001, 'vf_coef': .5, 'target_kl': .03},
        'time_semantics': 'PPO numbers unchanged per decision; history 2 frames now 10/20ms; gamma/lambda/rollout horizons in seconds shrink by half',
        'reward': 'unchanged position-only and physical-failure reward per decision; no effort costs',
        'evaluation': 'deterministic Beta means; whole 4s episodes; last 1s angle<=3deg and speed<=5deg/s for >=90%',
        'versions': {key: version(key) for key in ('torch', 'mujoco', 'stable-baselines3', 'numpy')},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary = {'status': 'running', 'training': training, 'history': [], 'updates': []}
    audit = TrainingAudit(args.out)
    best, reference, stale = None, None, 0
    stop = 'budget_cap'
    try:
        if args.resume:
            model, restored, meta = load_timing(args.resume)
            if restored.contract() != checking.contract():
                raise ValueError('Resume protocol mismatch')
            restored.close()
            model.set_env(vector)
            training['initialization'] = 'continuation from last; tensors and RMSprop retained; environments/RNG restarted'
            training['parent'] = str(args.resume)
            training['parent_sha256'] = meta['checkpoint_sha256']
        else:
            model = make_fresh_model(vector, checking.contract(), n_steps=1024//args.workers)
            if model.num_timesteps or model.optimizer_steps or model.policy.optimizer.state:
                raise AssertionError('Fresh PPO unexpectedly has training state')
        source_steps = model.num_timesteps
        training['source_steps'] = source_steps
        model.set_logger(configure(str(args.out), ['csv']))

        def check():
            nonlocal best, reference, stale
            save_workload(model, checking, args.out/'candidate.zip', training)
            combined = pool.submit(evaluation_job, args.out/'candidate.zip', validation+training_cases).result()
            result = merge_results([{'cases': combined['cases'][:len(validation)]}])
            train = merge_results([{'cases': combined['cases'][len(validation):]}])
            gate = (train['passed'] == train['total'] and result['passed'] == result['total']
                    and combined['failures'] == 0)
            rank = (not gate, *validation_rank(result), train['failures'], -train['passed'], train['score_deg'])
            if best is None or rank < best:
                best = rank
                save_workload(model, checking, args.out/'best.zip', training)
            progressed = (reference is None or result['failures'] < reference['failures']
                or (result['failures'] == reference['failures'] and (
                    (result['passed'] > reference['passed'] and result['score_deg'] <= 1.05*reference['score_deg'])
                    or (result['passed'] >= reference['passed'] and result['score_deg'] <= .98*reference['score_deg']))))
            if progressed:
                reference, stale = result, 0
            else:
                stale += 1
            row = {'transitions': model.num_timesteps, 'added_transitions': model.num_timesteps-source_steps,
                'validation': result, 'train': train, 'stale_checks': stale,
                'elapsed_s': time.monotonic()-started, 'audit': model.last_audit}
            summary['history'].append(row)
            summary['exploration'] = audit.report()
            (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps({'steps': model.num_timesteps, 'train_pass': train['passed'],
                'validation_pass': result['passed'], 'validation_rmse_deg': result['score_deg'],
                'train_rmse_deg': train['score_deg'], 'failures': combined['failures'],
                'stale_checks': stale, 'elapsed_s': row['elapsed_s']}), flush=True)
            return gate

        check()
        while model.num_timesteps-source_steps < args.steps:
            remaining = args.steps-(model.num_timesteps-source_steps)
            model.learn(min(1024, remaining), reset_num_timesteps=False, callback=audit)
            summary['updates'].append({'transitions': model.num_timesteps, **model.last_audit})
            if ((model.num_timesteps-source_steps) % args.check_every == 0
                    or model.num_timesteps-source_steps >= args.steps):
                gate = check()
                save_workload(model, checking, args.out/'last.zip', training)
                if gate:
                    stop = 'all_train_validation_passed'
                    break
                if model.num_timesteps-source_steps >= args.minimum_steps and stale >= args.plateau_checks:
                    stop = 'validation_plateau'
                    break
        save_workload(model, checking, args.out/'last.zip', training)
        summary.update(status='training_complete', stop_reason=stop,
            added_transitions=model.num_timesteps-source_steps, elapsed_s=time.monotonic()-started)
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        for label in ('best', 'last'):
            report = {name: pool.submit(evaluation_job, args.out/f'{label}.zip', cases).result()
                for name, cases in (('train', training_cases), ('validation', validation), ('test', test_cases))}
            (args.out/f'{label}_assessment.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'checkpoint': label, 'results': {name: {
                key: result[key] for key in ('passed', 'total', 'score_deg', 'failures')}
                for name, result in report.items()}}), flush=True)
    finally:
        pool.shutdown(wait=True)
        vector.close()
        checking.close()


if __name__ == '__main__':
    main()
