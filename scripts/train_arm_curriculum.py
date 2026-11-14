"""Fresh 10/10-ms arm PPO with task/LR stages and deterministic retention gates."""
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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

import myo.arm_curriculum as protocol_module
from myo.arm_curriculum import (curriculum, passed_gate, set_learning_rate,
                                unique_conditions, update_progress)
from myo.arm_exploration import ExplorationSpec
from myo.arm_timing import ArmTimingEnv, evaluation_job, load_timing, make_fresh_model
from myo.arm_workload import save_workload, scenarios, validation_rank
from myo.arm_workload_evaluation import merge_results
from train_arm_timing import TrainingAudit, worker


def evaluate(path, cases):
    # Training is idle while these two processes run. Do not keep extra idle
    # PyTorch interpreters in RAM between checks on the working laptop.
    parts = [list(part) for part in np.array_split(np.arange(len(cases)), min(2, len(cases)))]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(evaluation_job, path, [cases[i] for i in part]) for part in parts]
        return merge_results([future.result() for future in futures])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check-every', type=int, default=16384)
    parser.add_argument('--plateau-checks', type=int, default=4)
    parser.add_argument('--resume', type=Path,
        help='Continue the same curriculum stage from its strict timing checkpoint')
    parser.add_argument('--learning-rate', type=float,
        help='Override LR of the resumed stage, preserving RMSprop moments')
    args = parser.parse_args()
    if args.workers < 1 or 1024 % args.workers or args.check_every <= 0 or args.check_every % 1024:
        parser.error('Workers must divide 1024; checks must be positive whole rollouts')
    if args.learning_rate is not None and (not args.resume or not np.isfinite(args.learning_rate)
                                          or args.learning_rate <= 0):
        parser.error('A positive finite LR override requires --resume')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'curriculum_source.py').write_bytes(Path(protocol_module.__file__).read_bytes())
    stages = curriculum()
    start_index = 0
    parent, retained = None, []
    resume_identity = None
    if args.resume:
        restored, native, meta = load_timing(args.resume)
        native.close()
        if meta['training']['schema'] != 'myo.arm_curriculum_training.v1':
            raise ValueError('Resume requires a curriculum checkpoint')
        start_index = meta['training']['stage_index']
        if meta['training']['stage_name'] != stages[start_index].name:
            raise ValueError('Curriculum stage identity mismatch')
        if meta['contract']['spec'] != json.loads(json.dumps(asdict(stages[start_index].spec))):
            raise ValueError('Resume must preserve the task scope of its curriculum stage')
        parent = args.resume
        retained = meta['training']['retention_conditions']
        resume_identity = {'path': str(parent), 'checkpoint_sha256': meta['checkpoint_sha256'],
                           'steps': restored.num_timesteps, 'training': meta['training']}
        del restored
        if args.learning_rate is not None:
            stages = tuple(replace(stage, learning_rate=args.learning_rate) if i == start_index else stage
                           for i, stage in enumerate(stages))
    protocol = {'schema': 'myo.arm_curriculum_training.v1', 'seed': 0,
        'initialization': 'fresh random policy, empty RMSprop; no pretrained policy, predictor or teacher',
        'stages': [stage.manifest() for stage in stages],
        'ppo': {'n_steps': 1024//args.workers, 'workers': args.workers, 'batch_size': 128,
            'n_epochs': 10, 'gamma': .99, 'gae_lambda': .95, 'entropy': .001,
            'target_kl': .03, 'optimizer': 'RMSprop', 'clip': .2, 'vf_coef': .5, 'max_grad_norm': .5},
        'reward': 'unchanged position-only plus physical failure; no speed/effort costs',
        'timing': '10ms transport, 10ms control, 1ms physics, activation/deactivation 15/50ms',
        'time_semantics': 'same per-decision PPO constants as previous fresh 10/10 run; no horizon rescaling',
        'check_every': args.check_every, 'plateau_checks': args.plateau_checks,
        'gate': 'local: all train/validation; later: >=90% and mean tail RMSE<=3deg; zero failures; all retained conditions pass',
        'stop': 'after minimum steps, four checks without accumulated 2% improvement or more passes in validation or available full train audit; stage cap; stop before harder task if gate fails',
        'precision_fallback': 'local_precision allowed if acquisition passed, found any pass, or reduced initial validation RMSE by >=20%; otherwise stop',
        'evaluation': 'deterministic means, full 4s cold-reset episodes, >=90% of last second angle<=3deg and speed<=5deg/s; test only after training',
        'versions': {name: version(name) for name in ('torch','mujoco','stable-baselines3','numpy')},
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ('runner', Path(__file__)), ('protocol', Path(protocol_module.__file__)))}}
    if resume_identity:
        protocol.update(initialization='continuation from own learned curriculum checkpoint; weights and RMSprop retained',
                        resume=resume_identity, resumed_stage_index=start_index)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summary = {'status': 'running', 'protocol': protocol, 'stages': []}
    started = time.monotonic()
    model, vector, checking = None, None, None
    stop = 'all_stages_passed'

    def write_summary():
        summary['elapsed_s'] = time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    try:
        for index, stage in enumerate(stages):
            if index < start_index:
                continue
            root = args.out/stage.name
            root.mkdir()
            checking = ArmTimingEnv(stage.spec, ExplorationSpec())
            vector = SubprocVecEnv([partial(worker, stage.spec, ExplorationSpec(), i, args.workers)
                                   for i in range(args.workers)], start_method='spawn')
            if parent is None:
                model = make_fresh_model(vector, checking.contract(), n_steps=1024//args.workers)
                assert model.num_timesteps == model.optimizer_steps == 0 and not model.policy.optimizer.state
            else:
                model, restored, metadata = load_timing(parent)
                if (model.observation_space != checking.observation_space
                        or metadata['contract']['timing'] != checking.contract()['timing']):
                    raise ValueError('Curriculum must preserve physics/timing and observation space')
                restored.close()
                model.set_env(vector, force_reset=True)
            set_learning_rate(model, stage.learning_rate)
            model.set_logger(configure(str(root), ['csv']))
            source_steps = model.num_timesteps
            training = {**protocol, 'stage_index': index, 'stage_name': stage.name,
                'learning_rate': stage.learning_rate, 'source_steps': source_steps,
                'parent': str(parent) if parent else None,
                'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest() if parent else None,
                'continuation': 'best weights and RMSprop moments retained; new episode schedule and RNG at each stage',
                'retention_conditions': retained}
            state = {'name': stage.name, 'index': index, 'learning_rate': stage.learning_rate, 'source_steps': source_steps,
                     'status': 'running', 'history': [], 'updates': []}
            summary['stages'].append(state)
            audit = TrainingAudit(root)
            training_cases = scenarios(stage.spec.cases, stage.spec.masses, stage.spec.tilts,
                                       stage.spec.capacities, stage.spec.seconds)
            best_rank = reference = train_reference = None
            stale = 0
            strict = index < 2
            stage_stop = 'stage_budget'

            def check():
                nonlocal best_rank, reference, train_reference, stale
                save_workload(model, checking, root/'candidate.zip', training)
                combined = evaluate(root/'candidate.zip', list(stage.validation)+retained)
                validation = merge_results([{'cases': combined['cases'][:len(stage.validation)]}])
                retention_rows = combined['cases'][len(stage.validation):]
                retention_ok = all(row['passed'] and not row['failure'] for row in retention_rows)
                promising = passed_gate(validation, strict=strict) and retention_ok
                train = evaluate(root/'candidate.zip', training_cases) if promising else None
                gate = promising and passed_gate(train, strict=strict)
                rank = (not gate, not retention_ok, *validation_rank(validation))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    save_workload(model, checking, root/'best.zip', training)
                reference, train_reference, stale = update_progress(
                    validation, train, reference, train_reference, stale)
                row = {'transitions': model.num_timesteps, 'added_transitions': model.num_timesteps-source_steps,
                    'validation': validation, 'retention_cases': retention_rows, 'retention_ok': retention_ok,
                    'train_audit': train, 'gate': bool(gate), 'stale_checks': stale,
                    'elapsed_s': time.monotonic()-started, 'optimizer': model.last_audit}
                state['history'].append(row)
                state['exploration'] = audit.report()
                write_summary()
                print(json.dumps({'stage': stage.name, 'steps': model.num_timesteps,
                    'validation': f"{validation['passed']}/{validation['total']}",
                    'rmse_deg': validation['score_deg'], 'failures': combined['failures'],
                    'retention_ok': retention_ok, 'gate': bool(gate), 'stale': stale,
                    'elapsed_s': row['elapsed_s']}), flush=True)
                return gate

            gate = check()
            while not gate and model.num_timesteps-source_steps < stage.maximum_steps:
                model.learn(1024, reset_num_timesteps=False, callback=audit)
                state['updates'].append({'transitions': model.num_timesteps, **model.last_audit})
                if (model.num_timesteps-source_steps) % args.check_every == 0:
                    gate = check()
                    save_workload(model, checking, root/'last.zip', training)
                    if (not gate and model.num_timesteps-source_steps >= stage.minimum_steps
                            and stale >= args.plateau_checks):
                        stage_stop = 'validation_plateau'
                        break
            if gate:
                stage_stop = 'train_validation_retention_gate'
            save_workload(model, checking, root/'last.zip', training)
            # Include all train cells even when validation never became promising.
            # Test is not consulted for selection, LR changes or stage advancement.
            best_path = root/'best.zip'
            assessment = {name: evaluate(best_path, rows) for name, rows in (
                ('train', training_cases), ('validation', list(stage.validation)))}
            (root/'best_assessment.json').write_text(json.dumps(assessment, indent=2)+'\n')
            state.update(status='passed' if gate else 'not_passed', stop_reason=stage_stop,
                added_transitions=model.num_timesteps-source_steps,
                best_transitions=json.loads(Path(str(best_path)+'.metadata.json').read_text())['steps'],
                assessment=assessment)
            parent = best_path
            vector.close()
            checking.close()
            vector = checking = None
            write_summary()
            if not gate:
                first = state['history'][0]['validation']
                best = assessment['validation']
                fallback = (index == 0 and (best['passed'] > 0 or best['score_deg'] <= .8*first['score_deg']))
                if not fallback:
                    stop = 'stage_failed:'+stage.name
                    break
            else:
                # Replay a compact set of conditions that this checkpoint actually
                # passed. Gate never forgets previous successful validation cases.
                retained = unique_conditions(retained+[dict(row) for row in stage.validation
                    if any(tuple(r['case']) == tuple(row['case']) and r['payload_mass'] == row['payload_mass']
                           and r['tilt_deg'] == row['tilt_deg'] and r['passed']
                           for r in assessment['validation']['cases'])])
        # Independent holdout of each task reached; report only, never tune from it.
        for state in summary['stages']:
            index = state.get('index', start_index)
            root = args.out/state['name']
            test = evaluate(root/'best.zip', list(stages[index].test))
            (root/'test_assessment.json').write_text(json.dumps(test, indent=2)+'\n')
            state['test'] = test
        summary.update(status='complete', stop_reason=stop, final_checkpoint=str(parent))
        write_summary()
        print(json.dumps({'status': summary['status'], 'stop': stop, 'checkpoint': str(parent),
            'stages': [{key: state[key] for key in ('name','status','added_transitions','best_transitions')}
                       for state in summary['stages']]}), flush=True)
    finally:
        if vector is not None:
            vector.close()
        if checking is not None:
            checking.close()


if __name__ == '__main__':
    main()
