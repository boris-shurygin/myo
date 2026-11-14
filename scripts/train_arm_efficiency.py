"""Matched stable-hold and effort continuations of the existing 10/10-ms arm."""
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
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import myo.arm_efficiency as reward_module
from myo.arm_curriculum import conditions, set_learning_rate
from myo.arm_efficiency import (ArmEfficiencyEnv, EfficiencyReward, evaluate_job,
    load_parent, quality_gate)
from myo.arm_exploration import ExplorationSpec
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from myo.arm_timing import ArmTiming
from myo.arm_workload import WorkloadSpec, save_workload
from train_arm_timing import TrainingAudit
from finalize_arm_efficiency import execution_identity


def worker(spec, exploration, timing, reward, index, count):
    return Monitor(ArmEfficiencyEnv(spec, exploration, timing, reward, worker=index, workers=count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path,
        default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--stability-steps', type=int, default=32768)
    parser.add_argument('--branch-steps', type=int, default=65536)
    parser.add_argument('--check-every', type=int, default=16384)
    args = parser.parse_args()
    if any(n <= 0 or n % 1024 for n in (args.stability_steps, args.branch_steps, args.check_every)):
        parser.error('Budgets/checks must be positive whole 1024-transition rollouts')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'reward_source.py').write_bytes(Path(reward_module.__file__).read_bytes())
    train = conditions(tuple((goal+offset,goal) for goal in (55.,75.,95.) for offset in (-20.,0.,20.)))
    validation = conditions(tuple((goal+offset,goal) for goal in (65.,85.) for offset in (-15.,0.,15.)))
    retention = conditions(((20.,75.),(75.,20.),(15.,15.),(125.,125.),(135.,135.)))
    pilot = train+validation+retention
    test = conditions(tuple((goal+offset,goal) for goal in (60.,90.) for offset in (-10.,0.,10.)))
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=tuple(row['case'] for row in train))
    source, native, meta = load_parent(args.parent)
    exploration = ExplorationSpec(**meta['contract']['exploration'])
    timing = ArmTiming(**meta['contract']['timing'])
    native.close()
    protocol = {'schema': 'myo.arm_efficiency_training.v1', 'parent': str(args.parent),
        'parent_sha256': meta['checkpoint_sha256'], 'seed': 0,
        'spec': asdict(spec), 'train': train, 'validation': validation, 'retention': retention,
        'test': test, 'stability_steps': args.stability_steps, 'branch_steps': args.branch_steps,
        'check_every': args.check_every, 'learning_rate': 1e-5,
        'reward': asdict(EfficiencyReward()), 'effort_weight': .1,
        'ppo': {'workers': 4, 'vector_backend': 'DummyVecEnv; sequential physics to limit RAM',
            'n_steps': 256, 'batch_size': 128, 'n_epochs': 10, 'gamma': .99, 'gae_lambda': .95,
            'entropy': .001, 'target_kl': .03, 'optimizer': 'RMSprop'},
        'quality_gate': 'all local and 5 retention cases pass; per-case RMSE<=original+.25deg; '
            'speed RMS<=max(5deg/s,1.1*original); settling<=original+.2s; overshoot<=original+1deg; zero failures',
        'selection': 'stability: eligible minimum velocity RMS; matched branches: eligible minimum full activation^2; '
            'initial always considered; test and full-range reports only after selections frozen',
        'success': 'effort chosen activation^2 at least15% below original and5% below control, '
            'local quality gate and no loss of original passed full-range conditions; zero failures; '
            '20s interior holds reported separately',
        'actor_and_physics': 'unchanged; own learned weights and RMSprop retained; no predictor/teacher',
        'rng': 'restart seed0 and environment schedule once per branch; evaluations in separate process',
        'versions': {name: version(name) for name in ('torch','mujoco','stable-baselines3','numpy')},
        'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key,path in (
            ('runner',Path(__file__)),('reward',Path(reward_module.__file__)))}}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summary = {'status': 'running', 'protocol': protocol, 'stages': []}
    started = time.monotonic()

    def write_summary():
        summary['elapsed_s'] = time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    # Only one additional Torch process, idle during training. No env workers.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        def evaluate(path, cases, traces=None):
            return pool.submit(evaluate_job,path,cases,traces).result()

        baseline = evaluate(args.parent, pilot, args.out/'original_pilot_traces.npz')
        summary['original_pilot'] = baseline
        write_summary()
        print(json.dumps({'stage':'original','pass':f"{baseline['passed']}/{baseline['total']}",
            'rmse':baseline['score_deg'],'a2':baseline['activation_squared_mean']}),flush=True)
        stable_parent = args.parent
        del source
        for name, weight, budget in (('stability',0.,args.stability_steps),
                                    ('control',0.,args.branch_steps),('effort',.1,args.branch_steps)):
            parent = args.parent if name == 'stability' else stable_parent
            root = args.out/name
            root.mkdir()
            reward = EfficiencyReward(activation_squared_weight=weight)
            checking = ArmEfficiencyEnv(spec,exploration,timing,reward)
            vector = DummyVecEnv([partial(worker,spec,exploration,timing,reward,i,4) for i in range(4)])
            model, restored, parent_meta = load_parent(parent)
            if (model.observation_space != checking.observation_space
                    or parent_meta['contract']['timing'] != checking.contract()['timing']):
                raise ValueError('Efficiency continuation cannot alter physical timing/inputs')
            restored.close()
            if model.n_steps != 256 or model.batch_size != 128 or model.n_epochs != 10:
                raise ValueError('Unexpected parent PPO rollout/optimizer settings')
            model.set_env(vector,force_reset=True)
            set_learning_rate(model,1e-5)
            set_random_seed(0)
            vector.seed(0)
            model.set_logger(configure(str(root),['csv']))
            source_steps = model.num_timesteps
            training = {**protocol,'stage':name,'reward':asdict(reward),'parent':str(parent),
                'parent_sha256':parent_meta['checkpoint_sha256'],'source_steps':source_steps}
            state = {'name':name,'status':'running','history':[],'updates':[],'best_added_steps':0}
            summary['stages'].append(state)
            best_rank = None
            audit = TrainingAudit(root)
            save_workload(model,checking,root/'initial.zip',training)

            def check():
                nonlocal best_rank
                added = model.num_timesteps-source_steps
                path = root/'candidate.zip'
                save_workload(model,checking,path,training)
                result = evaluate(path,pilot,root/'candidate_traces.npz')
                eligible,reasons = quality_gate(result,baseline)
                metric = ('tail_velocity_rms_deg_s' if name == 'stability' else 'activation_squared_mean')
                # If no eligible candidate exists, retain the most successful one
                # for diagnosis, but do not advance to efficiency training.
                rank = ((0,result[metric],result['score_deg']) if eligible else
                    (1,-result['passed'],result['score_deg'],result[metric]))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    save_workload(model,checking,root/'best.zip',training)
                    shutil.copyfile(root/'candidate_traces.npz',root/'selected_pilot_traces.npz')
                    state.update(best_added_steps=added,best_eligible=eligible)
                state['history'].append({'added_steps':added,'eligible':eligible,
                    'reasons':reasons,'result':result,'optimizer':model.last_audit})
                write_summary()
                print(json.dumps({'stage':name,'steps':added,'pass':f"{result['passed']}/{result['total']}",
                    'eligible':eligible,'reasons':reasons,'rmse':result['score_deg'],
                    'speed':result['tail_velocity_rms_deg_s'],'a2':result['activation_squared_mean'],
                    'coactivation':result['tail_activation_coactivation'],
                    'opposing_Nm':result['tail_opposing_muscle_torque_Nm'],
                    'elapsed_s':summary['elapsed_s']}),flush=True)

            try:
                check()
                while model.num_timesteps-source_steps < budget:
                    model.learn(1024,reset_num_timesteps=False,callback=audit)
                    state['updates'].append({'added_steps':model.num_timesteps-source_steps,**model.last_audit})
                    if ((model.num_timesteps-source_steps)%args.check_every == 0
                            or model.num_timesteps-source_steps >= budget):
                        check()
                        save_workload(model,checking,root/'last.zip',training)
                save_workload(model,checking,root/'last.zip',training)
                state.update(status='complete',added_steps=model.num_timesteps-source_steps,
                    exploration=audit.report(),stop_reason='preregistered_finite_budget')
                state['selected'] = next(h['result'] for h in state['history']
                    if h['added_steps'] == state['best_added_steps'])
                state['last'] = state['history'][-1]['result']
                write_summary()
            finally:
                vector.close()
                checking.close()
            if name == 'stability':
                stable_parent = root/'best.zip'
                if not state['best_eligible']:
                    summary.update(status='stopped',stop_reason='stability_quality_gate_failed')
                    write_summary()
                    break
            del model
        # Freeze all choices before looking at the extra scenarios.
        selections = {'original':args.parent,**{s['name']:args.out/s['name']/'best.zip' for s in summary['stages']}}
        full_range = conditions(TRAIN_CASES)+conditions(VALIDATION_CASES)
        long_holds = [{**row,'seconds':20.} for row in conditions(((55.,55.),(75.,75.),(95.,95.)))]
        reports = {}
        identities = {label:execution_identity(path) for label,path in selections.items()}
        summary['report_reuse'] = {}
        for label,path in selections.items():
            donor = next((other for other in reports if identities[other] == identities[label]),None)
            if donor is not None:
                reports[label] = reports[donor]
                summary['report_reuse'][label] = donor
            else:
                reports[label] = {group:evaluate(path,rows,args.out/f'{label}_{group}_traces.npz')
                    for group,rows in (('test',test),('full_range',full_range),('long_holds',long_holds))}
                # Previously used test remains a known diagnostic, never a new holdout.
                reports[label]['known_test'] = evaluate(path,conditions(TEST_CASES))
            (args.out/'reports.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(json.dumps({'final':label,'groups':{k:{key:v[key] for key in (
                'passed','total','score_deg','activation_squared_mean')} for k,v in reports[label].items()}}),flush=True)
        summary['reports'] = reports
        summary['status'] = 'complete' if len(summary['stages']) == 3 else 'stopped'
        if len(summary['stages']) == 3:
            effort,control = summary['stages'][2]['selected'],summary['stages'][1]['selected']
            full_old,full_new = reports['original']['full_range'],reports['effort']['full_range']
            retained = all(not old['passed'] or (new['passed'] and not new['failure'])
                for old,new in zip(full_old['cases'],full_new['cases']))
            known_retained = all(not old['passed'] or (new['passed'] and not new['failure'])
                for old,new in zip(reports['original']['known_test']['cases'],
                                   reports['effort']['known_test']['cases']))
            summary['success'] = bool(summary['stages'][2]['best_eligible'] and retained
                and known_retained and reports['effort']['known_test']['failures'] == 0
                and reports['effort']['test']['passed'] == len(test)
                and reports['effort']['long_holds']['passed'] == len(long_holds)
                and full_new['failures'] == 0
                and effort['activation_squared_mean'] <= .85*baseline['activation_squared_mean']
                and effort['activation_squared_mean'] <= .95*control['activation_squared_mean'])
            summary['effort_reduction_vs_original'] = 1-effort['activation_squared_mean']/baseline['activation_squared_mean']
            summary['effort_reduction_vs_control'] = 1-effort['activation_squared_mean']/control['activation_squared_mean']
            summary['full_range_retained'] = retained
        write_summary()
        print(json.dumps({'status':summary['status'],'success':summary.get('success'),
            'elapsed_s':summary['elapsed_s']}),flush=True)


if __name__ == '__main__':
    main()
