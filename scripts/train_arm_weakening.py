"""Matched task-only PPO with/without explicit learning of weaker commands."""
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

import myo.arm_weakening as weakening_module
from myo.arm_curriculum import conditions, set_learning_rate
from myo.arm_efficiency import EfficiencyReward, load_parent, quality_gate
from myo.arm_exploration import ExplorationSpec
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from myo.arm_timing import ArmTiming
from myo.arm_weakening import (ArmWeakeningEnv, WeakeningSpec, load_weakening,
    make_weakening_model, weaken_actor)
from myo.arm_workload import WorkloadSpec, save_workload
from train_arm_relaxation import RelaxationAudit, evaluate


def worker(spec, exploration, timing, reward, weakening, index):
    return Monitor(ArmWeakeningEnv(spec, exploration, timing, reward, weakening, worker=index, workers=4))


def collect_corpus(path, cases, output):
    torch.set_num_threads(1)
    model, env, _ = load_weakening(path)
    observations = []
    try:
        for index, condition in enumerate(cases):
            obs, _ = env.reset(seed=0, options=condition)
            while True:
                observations.append(obs)
                obs, _, failed, done, _ = env.step(model.predict(obs, deterministic=True)[0])
                if failed or done:
                    break
            if (index+1)%5 == 0:
                print(json.dumps({'corpus_cases':index+1,'of':len(cases)}),flush=True)
    finally:
        env.close()
    array=np.asarray(observations)
    np.savez_compressed(output,observations=array)
    return array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--aux-updates',type=int,default=64)
    parser.add_argument('--fraction',type=float,default=.05)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--aux-learning-rate',type=float,default=3e-5)
    parser.add_argument('--branches',nargs='+',choices=('control','weakening'),default=['control','weakening'])
    parser.add_argument('--reference',type=Path,help='Reuse verified initial assessment/corpus and reference control')
    parser.add_argument('--early-stop-stale',type=int,default=0,
        help='Stop after this many cycles without any accepted lower-effort policy;0 disables')
    parser.add_argument('--moment-scale',type=float,help='Add auxiliary torque-proxy retention loss; SI Nm scale')
    args=parser.parse_args()
    if min(args.cycles,args.aux_updates)<=0 or min(args.learning_rate,args.aux_learning_rate)<=0:
        parser.error('Positive budgets and learning rates required')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'weakening_source.py').write_bytes(Path(weakening_module.__file__).read_bytes())
    local=tuple((g+o,g) for g in (55.,75.,95.) for o in (-20.,0.,20.))
    retention=((20.,75.),(75.,20.),(15.,15.),(125.,125.),(135.,135.))
    train=tuple(dict.fromkeys(TRAIN_CASES+local+retention))
    pilot=conditions(local)+conditions(tuple((g+o,g) for g in (65.,85.) for o in (-15.,0.,15.)))+conditions(retention)
    # Cached reports have JSON lists, whereas scenarios() returns tuple cases.
    # Compare identical serialized conditions, not Python container types.
    pilot=json.loads(json.dumps(pilot))
    spec=WorkloadSpec(masses=(.5,),tilts=(30.,),cases=train)
    parent,native,meta=load_parent(args.parent)
    exploration=ExplorationSpec(**meta['contract']['exploration']);timing=ArmTiming(**meta['contract']['timing'])
    native.close()
    reward=EfficiencyReward(activation_squared_weight=0.)
    weakening=WeakeningSpec(fraction=args.fraction)
    checking=ArmWeakeningEnv(spec,exploration,timing,reward,weakening)
    initial=make_weakening_model(checking,parent,checking.contract(),learning_rate=args.learning_rate)
    protocol={'parent':str(args.parent),'parent_sha256':meta['checkpoint_sha256'],'seed':0,
        'branches':args.branches,'reference':str(args.reference) if args.reference else None,
        'early_stop_stale_cycles':args.early_stop_stale,
        'auxiliary_moment_scale_Nm':args.moment_scale,
        'train':conditions(train),'pilot':pilot,'cycles':args.cycles,'auxiliary_updates_per_cycle':args.aux_updates,
        'ppo_transitions_per_cycle':2048,'weakening':asdict(weakening),'reward':asdict(reward),
        'learning_rate':args.learning_rate,'aux_learning_rate':args.aux_learning_rate,
        'comparison':'identical initial weights and RMSprop; task-only PPO in both; explicit self-labelled '
            'weakening only in intervention; same16x independent Beta exploration in both; no reward effort term',
        'labels':'current own mean, settled error<=3deg and speed<=5deg/s; remove5% (configured fraction) '
            'of smaller opposing active torque proxy, preserve current gain@mean; moving states anchor old means',
        'corpus':'initial native deterministic33-case replay; later add real PPO rollout observations; '
            'one seed0 permutation, max8192 rows per auxiliary phase; all gains causal in stored observations',
        'optimizer':'RMSprop retained; auxiliary actor-only supervised MSE scaled by.05, global norm.5; '
            'per-step KL check on up to2048 corpus rows with exact weights+RMSprop rollback',
        'selection':'initial included; after every auxiliary and PPO phase use original20-case quality gate; '
            'among eligible choose lowest actual mean(a²); next cycle starts selected weights AND RMSprop',
        'success':'at least15% lower pilot mean(a²), lower coactivation, task gate, preserved originally '
            'passed train/validation/known-test cases, local test6/6,long holds3/3; no physical failures',
        'evaluation':'nominal network means; no auxiliary rule applied at inference',
        'resource':'nice19,ionice idle,CPU12-13,Torch/BLAS1;4 sequential environments,1 evaluation process'}
    if args.moment_scale is not None:
        import myo.arm_weakening_objective as objective_module
        protocol['auxiliary_objective_sha256']=hashlib.sha256(Path(objective_module.__file__).read_bytes()).hexdigest()
    save_workload(initial,checking,args.out/'initial.zip',protocol)
    del initial,parent
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if args.moment_scale is not None:
        import myo.arm_weakening_objective as objective_module
        (args.out/'objective_source.py').write_bytes(Path(objective_module.__file__).read_bytes())
    started=time.monotonic();summary={'status':'running','protocol':protocol,'branches':[]}

    def write():
        summary['elapsed_s']=time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        def assess(path,cases,traces=None):
            return pool.submit(evaluate,path,cases,traces).result()

        reference=json.loads(Path('runs/arm_correlated_10ms/low_lr_rounds/summary.json').read_text())
        if (reference['protocol']['parent_sha256']!=meta['checkpoint_sha256']
                or reference['protocol']['pilot']!=json.loads(json.dumps(pilot))):
            raise ValueError('Cached baseline identity mismatch')
        baseline=reference['original']
        if args.reference:
            old_meta=json.loads(Path(str(args.reference/'initial.zip')+'.metadata.json').read_text())
            if old_meta['contract']!=checking.contract():
                raise ValueError('Reference initial contract mismatch')
            with zipfile.ZipFile(args.reference/'initial.zip') as old,zipfile.ZipFile(args.out/'initial.zip') as new:
                if any(old.read(k)!=new.read(k) for k in ('policy.pth','policy.optimizer.pth')):
                    raise ValueError('Reference initial weights/RMSprop mismatch')
            initial_result=json.loads((args.reference/'initial_assessment.json').read_text())
        else:
            initial_result=assess(args.out/'initial.zip',pilot)
        (args.out/'initial_assessment.json').write_text(json.dumps(initial_result,indent=2)+'\n')
        assert initial_result==baseline
        summary.update(original=baseline,initial=initial_result,baseline_reference='runs/arm_correlated_10ms/low_lr_rounds/summary.json')
        write()
        if args.reference:
            corpus=np.load(args.reference/'initial_corpus.npz')['observations']
            shutil.copyfile(args.reference/'initial_corpus.npz',args.out/'initial_corpus.npz')
        else:
            corpus=pool.submit(collect_corpus,args.out/'initial.zip',protocol['train'],args.out/'initial_corpus.npz').result()
        print(json.dumps({'corpus_rows':len(corpus)}),flush=True)
        for branch in args.branches:
            root=args.out/branch;root.mkdir()
            selected_path=args.out/'initial.zip';selected=baseline
            replay=corpus.copy()
            state={'branch':branch,'history':[],'cycles':[],'ppo_transitions':0,'auxiliary_optimizer_steps':0}
            summary['branches'].append(state)
            stale=0
            for cycle in range(args.cycles):
                vector=DummyVecEnv([partial(worker,spec,exploration,timing,reward,weakening,i) for i in range(4)])
                for env in vector.envs:
                    env.env.episode_index=cycle*2
                parent,restored,_=load_weakening(selected_path);restored.close()
                model=make_weakening_model(vector,parent,checking.contract(),learning_rate=args.learning_rate)
                del parent
                set_random_seed(0);vector.seed(0)
                model.set_logger(configure(str(root/f'cycle_{cycle}'),['csv']))
                audit=RelaxationAudit(root);cycle_row={'cycle':cycle,'selected_parent':str(selected_path),'updates':[]}
                try:
                    phases=('weakening','ppo') if branch=='weakening' else ('ppo',)
                    for phase in phases:
                        if phase=='weakening':
                            subset=np.random.default_rng(0).permutation(len(replay))[:8192]
                            set_learning_rate(model,args.aux_learning_rate)
                            if args.moment_scale is None:
                                report=weaken_actor(model,replay[subset],checking.contract(),updates=args.aux_updates)
                            else:
                                from myo.arm_weakening_objective import weaken_with_moment_constraint
                                report=weaken_with_moment_constraint(model,replay[subset],checking.contract(),
                                    updates=args.aux_updates,moment_scale_Nm=args.moment_scale)
                            state['auxiliary_optimizer_steps']+=report['attempted_updates']
                            cycle_row['weakening']=report
                            set_learning_rate(model,args.learning_rate)
                        else:
                            for _ in range(2):
                                model.learn(1024,reset_num_timesteps=False,callback=audit)
                                state['ppo_transitions']+=1024
                                cycle_row['updates'].append(model.last_audit)
                                new_obs=model.rollout_buffer.observations.reshape(-1,156)
                                replay=np.concatenate((replay,new_obs))[-24576:]
                        path=root/f'cycle_{cycle}_{phase}.zip'
                        training={**protocol,'branch':branch,'cycle':cycle,'phase':phase,
                            'parent':cycle_row['selected_parent'],'ppo_transitions':state['ppo_transitions'],
                            'auxiliary_optimizer_steps':state['auxiliary_optimizer_steps']}
                        save_workload(model,checking,path,training)
                        result=assess(path,pilot,root/f'cycle_{cycle}_{phase}_traces.npz')
                        eligible,reasons=quality_gate(result,baseline)
                        accepted=eligible and result['activation_squared_mean']<selected['activation_squared_mean']
                        if accepted:
                            selected_path,selected=path,result
                        row={'cycle':cycle,'phase':phase,'eligible':eligible,'reasons':reasons,'accepted':accepted,'result':result}
                        state['history'].append(row)
                        state.update(selected_path=str(selected_path),selected=selected)
                        write()
                        print(json.dumps({'branch':branch,'cycle':cycle,'phase':phase,'passed':result['passed'],
                            'rmse':result['score_deg'],'a2':result['activation_squared_mean'],'coactivation':result['tail_activation_coactivation'],
                            'eligible':eligible,'accepted':accepted,'aux':cycle_row.get('weakening'),
                            'elapsed_s':summary['elapsed_s']}),flush=True)
                    cycle_row['exploration']=audit.report();state['cycles'].append(cycle_row)
                    stale=stale+1 if str(selected_path)==cycle_row['selected_parent'] else 0
                    write()
                finally:
                    vector.close()
                del model
                if args.early_stop_stale and stale>=args.early_stop_stale:
                    state['stop_reason']='no accepted effort improvement for configured consecutive cycles'
                    break
            for suffix in ('','.metadata.json'):
                shutil.copyfile(str(selected_path)+suffix,str(root/'best.zip')+suffix)
            state.update(status='complete',reduction=1-selected['activation_squared_mean']/baseline['activation_squared_mean'])
            write()
        promising=[b for b in summary['branches'] if b['reduction']>=.15
            and b['selected']['tail_activation_coactivation']<baseline['tail_activation_coactivation']]
        summary['reports']={}
        groups={'local_test':conditions(tuple((g+o,g) for g in (60.,90.) for o in (-10.,0.,10.))),
            'old_train_validation':conditions(TRAIN_CASES)+conditions(VALIDATION_CASES),
            'known_test':conditions(TEST_CASES),
            'long_holds':[{**c,'seconds':20.} for c in conditions(((55.,55.),(75.,75.),(95.,95.)))]}
        successful=[]
        if promising:
            old=json.loads(Path('runs/arm_relaxation_10ms/comparison/summary.json').read_text())['reports']['original']
            for state in promising:
                reports={name:assess(state['selected_path'],cases) for name,cases in groups.items()}
                summary['reports'][state['branch']]=reports
                retained=all(not before['passed'] or after['passed']
                    for new_name,old_name in (('old_train_validation','full_range'),('known_test','known_test'))
                    for before,after in zip(old[old_name]['cases'],reports[new_name]['cases']))
                if (retained and reports['local_test']['passed']==6 and reports['long_holds']['passed']==3
                        and all(r['failures']==0 for r in reports.values())):
                    successful.append(state['branch'])
        summary.update(status='complete',success=bool(successful),successful_branches=successful,
            extended_checks='completed for promising candidates' if promising else 'skipped: no15% economy candidate')
        write();print(json.dumps({'status':'complete','success':summary['success'],'elapsed_s':summary['elapsed_s']}),flush=True)
    checking.close()


if __name__=='__main__':
    main()
