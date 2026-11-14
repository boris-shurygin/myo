"""Matched PPO: identical marginal probes, independent vs shared relaxation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import myo.arm_relaxation as relaxation_module
from myo.arm_curriculum import conditions
from myo.arm_efficiency import EfficiencyReward, load_parent, quality_gate
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from myo.arm_relaxation import (ArmRelaxationEnv, RelaxationSpec, load_relaxation,
                                make_relaxation_model)
from myo.arm_tension_diagnostics import antagonist_metrics
from myo.arm_exploration import ExplorationSpec
from myo.arm_timing import ArmTiming
from myo.arm_workload import WorkloadSpec, save_workload
from myo.arm_workload_evaluation import merge_results
from train_arm_timing import TrainingAudit


class RelaxationAudit(TrainingAudit):
    def __init__(self, root):
        super().__init__(root)
        self.activation_squared_sum = self.coactivation_sum = 0.

    def _on_step(self):
        actions = np.asarray(self.locals['actions'])
        self.count += len(actions)
        # Six causal mechanical gains now follow the original pressure channel.
        self.pressured += int((self.locals['obs_tensor'][:, -7] > .5).sum())
        self.action_sum += actions.sum(axis=0)
        self.strong += (actions > .95).sum(axis=0)
        for info in self.locals['infos']:
            self.activation_squared_sum += info['workload']['activation_squared_mean']
            self.coactivation_sum += info['workload']['coactivation_proxy']
        return True

    def report(self):
        return {**super().report(),
            'mean_actual_activation_squared': self.activation_squared_sum/max(1, self.count),
            'mean_actual_coactivation': self.coactivation_sum/max(1, self.count)}


def evaluate(path, cases, trace_path=None):
    torch.set_num_threads(1)
    schema = json.loads(Path(str(path)+'.metadata.json').read_text())['contract']['schema']
    if schema == 'myo.arm_accumulated_economy.v1':
        from myo.arm_economy_curriculum import load_economy
        model, env, _ = load_economy(path)
    elif schema == 'myo.arm_weakening.v1':
        from myo.arm_weakening import load_weakening
        model, env, _ = load_weakening(path)
    elif schema == 'myo.arm_correlated_relaxation.v1':
        from myo.arm_correlated_relaxation import load_correlated
        model, env, _ = load_correlated(path)
    else:
        model, env, _ = load_relaxation(path) if schema == 'myo.arm_relaxation.v1' else load_parent(path)
    rows, traces = [], {}
    try:
        for index, condition in enumerate(cases):
            obs, initial = env.reset(seed=0, options=condition)
            trace = []
            while True:
                obs, _, failed, done, info = env.step(model.predict(obs, deterministic=True)[0])
                trace.append([info['t'], info['q'], info['qvel'], *info['activation'],
                    *info['tendon_force_N'], *info['moment_arm_m'], *info['capacity']])
                if failed or done:
                    break
            a = np.asarray(trace)
            traces[f'case_{index:03d}'] = a
            error = np.rad2deg(a[:, 1])-condition['case'][1]
            speed = np.rad2deg(a[:, 2])
            tail = a[:, 0] > condition['seconds']-1.-1e-8
            settled = (abs(error) <= 3.) & (abs(speed) <= 5.)
            stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
            full = not failed and len(a) == round(condition['seconds']/env.unwrapped.control_dt)
            direction = np.sign(condition['case'][1]-condition['case'][0])
            metrics = antagonist_metrics(a[:, 3:9], a[:, 9:15], a[:, 15:21], a[:, 2])
            row = {**condition, 'failure': bool(failed), 'passed': bool(full and settled[tail].mean() >= .9),
                'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
                'tail_velocity_rms_deg_s': float(np.sqrt(np.mean(speed[tail]**2))),
                'settling_time_s': float(a[stable[0], 0]) if full and len(stable) else None,
                'overshoot_deg': float(max(0., np.max(direction*error))) if direction else None,
                'activation_squared_mean': float(metrics['activation_squared_mean'].mean()),
                'active_fraction_mean': float(np.mean(a[:, 3:9]*a[:, 21:27])),
                'capacity_net_loss_by_muscle': (np.asarray(initial['capacity'])-a[-1, 21:27]).tolist()}
            for key, value in metrics.items():
                row['full_'+key] = float(value.mean())
                row['tail_'+key] = float(value[tail].mean())
            rows.append(row)
            if (index+1) % 5 == 0:
                print(json.dumps({'assessment_cases': index+1, 'of': len(cases),
                    'checkpoint': str(path)}), flush=True)
    finally:
        env.close()
    if trace_path:
        np.savez_compressed(trace_path, **traces)
    result = merge_results([{'cases': rows}])
    for key in ('tail_velocity_rms_deg_s', 'tail_activation_squared_mean', 'tail_activation_coactivation',
                'tail_opposing_muscle_torque_Nm'):
        result[key] = float(np.mean([r[key] for r in rows]))
    return result


def worker(spec, exploration, timing, reward, relaxation, index):
    return Monitor(ArmRelaxationEnv(spec, exploration, timing, reward, relaxation, worker=index, workers=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path,
        default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--check-every', type=int, default=16384)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    args = parser.parse_args()
    if (min(args.steps, args.check_every) <= 0 or args.steps % 1024 or args.check_every % 1024
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Positive whole-rollout budgets and positive finite learning rate required')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'distribution_source.py').write_bytes(Path(relaxation_module.__file__).read_bytes())
    local = tuple((goal+offset, goal) for goal in (55., 75., 95.) for offset in (-20., 0., 20.))
    retention = ((20., 75.), (75., 20.), (15., 15.), (125., 125.), (135., 135.))
    train = tuple(dict.fromkeys(TRAIN_CASES+local+retention))
    pilot = conditions(local)+conditions(tuple((g+o, g) for g in (65., 85.) for o in (-15., 0., 15.)))
    pilot += conditions(retention)
    tests = conditions(tuple((g+o, g) for g in (60., 90.) for o in (-10., 0., 10.)))
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=train)
    source, native, meta = load_parent(args.parent)
    exploration = ExplorationSpec(**meta['contract']['exploration'])
    timing = ArmTiming(**meta['contract']['timing'])
    native.close()
    del source
    reward = EfficiencyReward(activation_squared_weight=.1)
    protocol = {'schema': 'myo.arm_relaxation_training.v1', 'parent': str(args.parent),
        'parent_sha256': meta['checkpoint_sha256'], 'seed': 0, 'steps_per_branch': args.steps,
        'check_every': args.check_every, 'learning_rate': args.learning_rate,
        'train': conditions(train), 'pilot': pilot, 'test': tests,
        'reward': asdict(reward), 'relaxation': asdict(RelaxationSpec()),
        'comparison': 'same parent weights/RMSprop and per-muscle marginal distributions; '
            'independent versus one shared mixture component for all muscles; '
            '16x concentration in both arms; all old training angles replayed',
        'ppo': {'workers': 4, 'vector_backend': 'DummyVecEnv', 'n_steps': 256, 'batch_size': 128,
            'n_epochs': 10, 'gamma': .99, 'gae_lambda': .95, 'entropy': .001, 'target_kl': .03},
        'selection': 'same 20-case quality gate as earlier efficiency study; original included; '
            'among eligible choose lowest full activation_squared_mean; tests only after selection frozen',
        'success': 'joint best at least15% cheaper than original and5% than independent, '
            'pilot quality gate and retention of originally passed full-range/known-test cases, '
            'additional test6/6, long holds3/3, no physical failures',
        'evaluation': 'nominal mean only; relaxation and random sampling disabled',
        'entropy': '4 fresh on-policy samples; unbiased score-function entropy gradient',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summary = {'status': 'running', 'protocol': protocol, 'branches': []}
    started = time.monotonic()

    def write():
        summary['elapsed_s'] = time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        def assess(path, cases, traces=None):
            return pool.submit(evaluate, path, cases, traces).result()

        baseline = assess(args.parent, pilot, args.out/'original_pilot_traces.npz')
        summary['original_pilot'] = baseline
        write()
        for mode in ('independent', 'joint'):
            root = args.out/mode
            root.mkdir()
            relaxation = RelaxationSpec(mode=mode)
            checking = ArmRelaxationEnv(spec, exploration, timing, reward, relaxation)
            vector = DummyVecEnv([partial(worker, spec, exploration, timing, reward, relaxation, i)
                                  for i in range(4)])
            parent, restored, _ = load_parent(args.parent)
            restored.close()
            model = make_relaxation_model(vector, parent, checking.contract(), learning_rate=args.learning_rate)
            del parent
            set_random_seed(0)
            vector.seed(0)
            model.set_logger(configure(str(root), ['csv']))
            source_steps = model.num_timesteps
            training = {**protocol, 'stage': mode, 'relaxation': asdict(relaxation), 'source_steps': source_steps}
            state = {'mode': mode, 'history': [], 'updates': [], 'best_added_steps': 0}
            summary['branches'].append(state)
            audit = RelaxationAudit(root)
            best_rank = None
            save_workload(model, checking, root/'initial.zip', training)

            def check():
                nonlocal best_rank
                added = model.num_timesteps-source_steps
                path = root/'candidate.zip'
                save_workload(model, checking, path, training)
                if added == 0:
                    # Equality is verified by transferred tensors and unit test;
                    # also execute the complete initial pilot once per branch.
                    result = assess(path, pilot, root/'candidate_traces.npz')
                    assert result == baseline
                else:
                    result = assess(path, pilot, root/'candidate_traces.npz')
                eligible, reasons = quality_gate(result, baseline)
                rank = ((0, result['activation_squared_mean'], result['score_deg']) if eligible else
                    (1, -result['passed'], result['score_deg'], result['activation_squared_mean']))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    save_workload(model, checking, root/'best.zip', training)
                    shutil.copyfile(root/'candidate_traces.npz', root/'best_pilot_traces.npz')
                    state.update(best_added_steps=added, best_eligible=eligible)
                state['history'].append({'added_steps': added, 'eligible': eligible, 'reasons': reasons,
                    'result': result})
                write()
                print(json.dumps({'mode': mode, 'steps': added, 'passed': result['passed'],
                    'rmse_deg': result['score_deg'], 'a2': result['activation_squared_mean'],
                    'coactivation': result['tail_activation_coactivation'], 'eligible': eligible,
                    'elapsed_s': summary['elapsed_s']}), flush=True)

            try:
                check()
                while model.num_timesteps-source_steps < args.steps:
                    model.learn(1024, reset_num_timesteps=False, callback=audit)
                    state['updates'].append({'added_steps': model.num_timesteps-source_steps, **model.last_audit})
                    if (model.num_timesteps-source_steps) % 4096 == 0:
                        print(json.dumps({'training': mode, 'steps': model.num_timesteps-source_steps,
                            'audit': model.last_audit, 'exploration': audit.report(),
                            'elapsed_s': time.monotonic()-started}), flush=True)
                    if ((model.num_timesteps-source_steps) % args.check_every == 0
                            or model.num_timesteps-source_steps >= args.steps):
                        check()
                        save_workload(model, checking, root/'last.zip', training)
                state.update(status='complete', exploration=audit.report())
                state['selected'] = next(row['result'] for row in state['history']
                                         if row['added_steps'] == state['best_added_steps'])
                state['last'] = state['history'][-1]['result']
                write()
            finally:
                vector.close()
                checking.close()
            del model
        # Freeze selection before all extra reports. Do not replay unchanged
        # selected parents twice; initial equality was checked above.
        groups = {'test': tests, 'full_range': conditions(TRAIN_CASES)+conditions(VALIDATION_CASES),
            'known_test': conditions(TEST_CASES),
            'long_holds': [{**c, 'seconds': 20.} for c in conditions(((55., 55.), (75., 75.), (95., 95.)))]}
        reports = {'original': {name: assess(args.parent, cases) for name, cases in groups.items()}}
        for state in summary['branches']:
            mode = state['mode']
            reports[mode] = (reports['original'] if state['best_added_steps'] == 0 else
                {name: assess(args.out/mode/'best.zip', cases) for name, cases in groups.items()})
        summary['reports'] = reports
        joint, independent = (state['selected'] for state in summary['branches'][::-1])
        retained = all(not old['passed'] or (new['passed'] and not new['failure'])
            for group in ('full_range', 'known_test')
            for old, new in zip(reports['original'][group]['cases'], reports['joint'][group]['cases']))
        summary.update(status='complete', success=bool(retained
            and reports['joint']['test']['passed'] == len(tests)
            and reports['joint']['long_holds']['passed'] == 3
            and all(result['failures'] == 0 for result in reports['joint'].values())
            and joint['activation_squared_mean'] <= .85*baseline['activation_squared_mean']
            and joint['activation_squared_mean'] <= .95*independent['activation_squared_mean']),
            full_range_and_known_test_retained=retained,
            effort_reduction_vs_original=1-joint['activation_squared_mean']/baseline['activation_squared_mean'],
            effort_reduction_vs_independent=1-joint['activation_squared_mean']/independent['activation_squared_mean'])
        write()
        print(json.dumps({'status': summary['status'], 'success': summary['success'],
            'elapsed_s': summary['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
