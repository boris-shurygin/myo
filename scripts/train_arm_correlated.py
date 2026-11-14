"""Short economy/accuracy PPO rounds with spatial exploration and rollback."""
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
import zipfile

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import myo.arm_correlated_relaxation as correlated_module
from myo.arm_correlated_relaxation import (ArmCorrelatedEnv, CorrelatedSpec,
                                          load_correlated, make_correlated_model)
from myo.arm_curriculum import conditions
from myo.arm_efficiency import EfficiencyReward, load_parent, quality_gate
from myo.arm_exploration import ExplorationSpec
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from myo.arm_timing import ArmTiming
from myo.arm_workload import WorkloadSpec, save_workload
from train_arm_relaxation import RelaxationAudit, evaluate


class CorrelatedAudit(RelaxationAudit):
    def __init__(self, root):
        super().__init__(root)
        self.rollouts = []

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        observations = buffer.observations.reshape(-1, buffer.observations.shape[-1])
        actions = buffer.actions.reshape(-1, 6)
        advantage = buffer.advantages.reshape(-1)
        with torch.no_grad():
            d = self.model.policy.get_distribution(torch.as_tensor(observations))
            mu = d.nominal_mean.numpy()
        # Descriptive correlation, not a causal estimate of a relaxation bonus.
        reduced = (mu-actions).mean(-1)
        corr = np.corrcoef(reduced, advantage)[0, 1] if np.std(advantage)*np.std(reduced) else 0.
        self.rollouts.append({'steps': self.model.num_timesteps,
            'advantage_std': float(advantage.std()), 'advantage_mean': float(advantage.mean()),
            'mean_command_reduction': float(reduced.mean()),
            'command_reduction_advantage_correlation': float(corr),
            'return_minus_value_rms': float(np.sqrt(np.mean((buffer.returns-buffer.values)**2)))})


def worker(spec, exploration, timing, reward, correlated, index):
    return Monitor(ArmCorrelatedEnv(spec, exploration, timing, reward, correlated, worker=index, workers=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--parent', type=Path,
        default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--local-pilot', action='store_true',
        help='Allow a diagnostic pilot when55-95deg probes pass but125deg does not; retain full gate')
    parser.add_argument('--reference', type=Path,
        help='Reuse a completed baseline and identical initial execution from an earlier pilot')
    parser.add_argument('--minimum-economy-reduction', type=float, default=.15,
        help='Require practical pilot economy before spending time on extended promotion checks')
    args = parser.parse_args()
    if (args.cycles < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 < args.minimum_economy_reduction < 1):
        parser.error('Positive cycle count and finite learning rate required')
    probe = json.loads(args.probe.read_text())
    local_rows = [r for r in probe['calibration']+probe['validation']
                  if 55 <= r['goal'] <= 95 and r['distribution'] == probe['selected']]
    locally_qualified = len(local_rows) >= 5 and all(
        r['beneficial_count'] >= .35*probe['samples'] and r['mean_return_delta'] > 0
        and r['mean_activation_reduction'] > 0 and r['quiet_count'] == probe['samples'] for r in local_rows)
    if not probe['proceed'] and not (args.local_pilot and locally_qualified):
        parser.error('Native probe gate did not pass; do not start PPO')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'distribution_source.py').write_bytes(Path(correlated_module.__file__).read_bytes())
    local = tuple((g+o, g) for g in (55.,75.,95.) for o in (-20.,0.,20.))
    retention = ((20.,75.), (75.,20.), (15.,15.), (125.,125.), (135.,135.))
    train_cases = tuple(dict.fromkeys(TRAIN_CASES+local+retention))
    old = {'train': conditions(train_cases), 'pilot': conditions(local)
        +conditions(tuple((g+o,g) for g in (65.,85.) for o in (-15.,0.,15.)))+conditions(retention),
        'test': conditions(tuple((g+o,g) for g in (60.,90.) for o in (-10.,0.,10.)))}
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=train_cases)
    pilot = old['pilot']
    parent, restored, meta = load_parent(args.parent)
    if probe['parent_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('Probe parent differs from PPO parent')
    exploration = ExplorationSpec(**meta['contract']['exploration'])
    timing = ArmTiming(**meta['contract']['timing'])
    restored.close()
    del parent
    correlated = CorrelatedSpec(**probe['selected'])
    protocol = {'parent': str(args.parent), 'parent_sha256': meta['checkpoint_sha256'],
        'reference': str(args.reference) if args.reference else None,
        'probe': str(args.probe), 'probe_sha256': hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        'distribution': asdict(correlated), 'train': old['train'], 'pilot': pilot,
        'local_pilot': args.local_pilot, 'universal_probe_gate_passed': probe['proceed'],
        'probe_limitation': '125deg negative mean return; local pilot justified by five55-95deg '
            'positive-mean probes, not a claim of universal beneficial relaxation; retain125/135deg gates',
        'learning_rate': args.learning_rate, 'cycles': args.cycles, 'block_transitions': 4096,
        'minimum_economy_reduction': args.minimum_economy_reduction,
        'branches': {'continuous': [.1, .1], 'alternating': [.1, .025]},
        'task_reward': 'identical position and near-goal reference-velocity costs in both phases',
        'reset': 'rollouts continue within cycles; new cycle starts native cold states from '
            'accepted weights+RMSprop; seed0, episode schedule offset6*cycle; no random-seed sweep',
        'acceptance': 'after two blocks (8192), same per-condition pilot gate relative to original '
            'and lower full activation squared than previous accepted; else rollback weights AND RMSprop; '
            'every intermediate candidate recorded; no rollback of accounting or attempted transitions',
        'evaluation': 'deterministic nominal actor output; no relaxation or random exploration',
        'ppo': {'workers': 4, 'n_steps': 256, 'batch_size': 128, 'n_epochs': 10,
            'gamma': .99, 'gae_lambda': .95, 'clip': .2, 'entropy': .001, 'target_kl': .03},
        'resource': 'nice19,ionice idle,CPU12-13,Torch/BLAS1; one evaluation worker'}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summary = {'status': 'running', 'protocol': protocol, 'branches': []}
    started = time.monotonic()

    def write():
        summary['elapsed_s'] = time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        def assess(path, cases, trace=None):
            return pool.submit(evaluate, path, cases, trace).result()

        if args.reference:
            reference = json.loads((args.reference/'summary.json').read_text())
            if (reference['protocol']['parent_sha256'] != meta['checkpoint_sha256']
                    or reference['protocol']['pilot'] != json.loads(json.dumps(pilot))):
                raise ValueError('Reference baseline identity mismatch')
            baseline = reference['original']
        else:
            baseline = assess(args.parent, pilot, args.out/'original_traces.npz')
        summary['original'] = baseline
        write()
        for branch, costs in protocol['branches'].items():
            root = args.out/branch
            root.mkdir()
            initial_reward = EfficiencyReward(activation_squared_weight=.1)
            checking = ArmCorrelatedEnv(spec, exploration, timing, initial_reward, correlated)
            parent, check, _ = load_parent(args.parent)
            check.close()
            initial = make_correlated_model(checking, parent, checking.contract(), learning_rate=args.learning_rate)
            save_workload(initial, checking, root/'initial.zip', {**protocol, 'branch': branch, 'phase': 'initial'})
            if args.reference:
                reference_path = args.reference/'continuous'/'initial.zip'
                previous = json.loads(Path(str(reference_path)+'.metadata.json').read_text())
                if previous['contract'] != checking.contract():
                    raise ValueError('Reference initial contract mismatch')
                with zipfile.ZipFile(reference_path) as old_zip, zipfile.ZipFile(root/'initial.zip') as new_zip:
                    if old_zip.read('policy.pth') != new_zip.read('policy.pth'):
                        raise ValueError('Reference initial weights mismatch')
            else:
                initial_result = assess(root/'initial.zip', pilot)
                assert initial_result == baseline
            checking.close()
            del initial, parent
            accepted_path = root/'initial.zip'
            accepted_result = baseline
            state = {'branch': branch, 'history': [], 'accepted_cycles': [], 'attempted_transitions': 0}
            summary['branches'].append(state)
            for cycle in range(args.cycles):
                reward = EfficiencyReward(activation_squared_weight=costs[0])
                checking = ArmCorrelatedEnv(spec, exploration, timing, reward, correlated)
                vector = DummyVecEnv([partial(worker, spec, exploration, timing, reward, correlated, i)
                                      for i in range(4)])
                for env in vector.envs:
                    env.env.episode_index = cycle*6
                source, check, _ = load_correlated(accepted_path)
                check.close()
                model = make_correlated_model(vector, source, checking.contract(), learning_rate=args.learning_rate)
                del source
                set_random_seed(0)
                vector.seed(0)
                model.set_logger(configure(str(root/f'cycle_{cycle}'), ['csv']))
                audit = CorrelatedAudit(root)
                updates = []
                try:
                    for phase, cost in enumerate(costs):
                        reward = EfficiencyReward(activation_squared_weight=cost)
                        checking.efficiency_reward = reward
                        for env in vector.envs:
                            env.env.efficiency_reward = reward
                        for _ in range(4):
                            model.learn(1024, reset_num_timesteps=False, callback=audit)
                            state['attempted_transitions'] += 1024
                            updates.append({**model.last_audit, 'phase': phase,
                                'attempted_transitions': state['attempted_transitions']})
                        path = root/f'cycle_{cycle}_phase_{phase}.zip'
                        training = {**protocol, 'branch': branch, 'cycle': cycle, 'phase': phase,
                            'activation_squared_weight': cost, 'attempted_transitions': state['attempted_transitions'],
                            'accepted_parent': str(accepted_path)}
                        save_workload(model, checking, path, training)
                        result = assess(path, pilot, root/f'cycle_{cycle}_phase_{phase}_traces.npz')
                        eligible, reasons = quality_gate(result, baseline)
                        row = {'cycle': cycle, 'phase': phase, 'cost': cost, 'result': result,
                            'eligible': eligible, 'reasons': reasons,
                            'attempted_transitions': state['attempted_transitions']}
                        state['history'].append(row)
                        write()
                        print(json.dumps({'branch': branch, 'cycle': cycle, 'phase': phase,
                            'passed': result['passed'], 'rmse': result['score_deg'],
                            'a2': result['activation_squared_mean'], 'eligible': eligible,
                            'optimizer_steps': sum(u['optimizer_steps'] for u in updates),
                            'elapsed_s': summary['elapsed_s']}), flush=True)
                    accepted = eligible and result['activation_squared_mean'] < accepted_result['activation_squared_mean']
                    if accepted:
                        accepted_path, accepted_result = path, result
                    state['accepted_cycles'].append({'cycle': cycle, 'accepted': accepted,
                        'updates': updates, 'exploration': audit.report(), 'rollout_audit': audit.rollouts})
                    state.update(selected_path=str(accepted_path), selected=accepted_result)
                    write()
                finally:
                    checking.close()
                    vector.close()
                del model
            for suffix in ('', '.metadata.json'):
                shutil.copyfile(str(accepted_path)+suffix, str(root/'best.zip')+suffix)
            state['status'] = 'complete'
            write()
        # Tiny deterministic differences are not a useful economy result.
        # Preserve their pilot measurements without rerunning extended checks
        # of unchanged or practically unchanged demo behavior.
        summary['pilot_acceptance_found'] = any(any(r['accepted'] for r in state['accepted_cycles'])
                                               for state in summary['branches'])
        practical = [state for state in summary['branches'] if state['selected']['activation_squared_mean']
                     <= (1-args.minimum_economy_reduction)*baseline['activation_squared_mean']]
        if not practical:
            summary.update(status='complete', success=False, reports={},
                extended_checks='skipped: no practical economy candidate; original demo unchanged')
            write()
            print(json.dumps({'status': 'complete', 'success': False,
                'extended_checks': summary['extended_checks'], 'elapsed_s': summary['elapsed_s']}), flush=True)
            return
        # Freeze selection before reporting unused local test and known failures.
        groups = {'local_test': old['test'], 'old_train_validation': conditions(TRAIN_CASES)+conditions(VALIDATION_CASES),
            'known_test': conditions(TEST_CASES),
            'long_holds': [{**c, 'seconds': 20.} for c in conditions(((55.,55.), (75.,75.), (95.,95.)))]}
        summary['reports'] = {'original': {key: assess(args.parent, cases) for key, cases in groups.items()}}
        successful = []
        for state in practical:
            reports = {key: assess(state['selected_path'], cases) for key, cases in groups.items()}
            summary['reports'][state['branch']] = reports
            retained = all(not old_row['passed'] or new_row['passed']
                for group in ('old_train_validation', 'known_test')
                for old_row, new_row in zip(summary['reports']['original'][group]['cases'], reports[group]['cases']))
            if (retained and reports['local_test']['passed'] == 6 and reports['long_holds']['passed'] == 3
                    and all(r['failures'] == 0 for r in reports.values())):
                successful.append(state['branch'])
        summary.update(status='complete', success=bool(successful), successful_branches=successful)
        write()
        print(json.dumps({'status': summary['status'], 'success': summary['success'],
                          'elapsed_s': summary['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
