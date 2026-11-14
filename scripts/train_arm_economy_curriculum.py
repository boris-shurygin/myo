"""Accumulate small learned weakening steps, allow task-and-effort PPO recovery."""
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
import zipfile

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import myo.arm_economy_curriculum as economy_module
import myo.arm_weakening_objective as objective_module
from myo.arm_curriculum import conditions, set_learning_rate
from myo.arm_economy_curriculum import ArmEconomyEnv, economy_gate, load_economy
from myo.arm_efficiency import EfficiencyReward, load_parent, quality_gate
from myo.arm_exploration import ExplorationSpec
from myo.arm_timing import ArmTiming
from myo.arm_weakening import WeakeningSpec, make_weakening_model
from myo.arm_weakening_objective import weaken_with_moment_constraint
from myo.arm_workload import WorkloadSpec, save_workload
from train_arm_relaxation import RelaxationAudit, evaluate


def worker(spec, exploration, timing, reward, weakening, index):
    return Monitor(ArmEconomyEnv(spec, exploration, timing, reward, weakening, worker=index, workers=4))


def collect_economy_corpus(path, cases, output):
    torch.set_num_threads(1)
    model,env,_=load_economy(path)
    observations=[]
    try:
        for condition in cases:
            obs,_=env.reset(seed=0,options=condition)
            while True:
                observations.append(obs)
                obs,_,failed,done,_=env.step(model.predict(obs,deterministic=True)[0])
                if failed or done:break
    finally:
        env.close()
    array=np.asarray(observations)
    np.savez_compressed(output,observations=array)
    return array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--cycles',type=int,default=12)
    parser.add_argument('--minimum-rollouts',type=int,default=2)
    parser.add_argument('--maximum-rollouts',type=int,default=8)
    parser.add_argument('--aux-updates',type=int,default=64)
    parser.add_argument('--fraction',type=float,default=.02)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--aux-learning-rate',type=float,default=3e-6)
    parser.add_argument('--effort-weight',type=float,default=.1)
    args=parser.parse_args()
    if min(args.cycles,args.minimum_rollouts,args.maximum_rollouts,args.aux_updates)<=0 or args.maximum_rollouts<args.minimum_rollouts:
        parser.error('Positive budgets and maximum>=minimum required')
    if not 0<args.fraction<1 or min(args.learning_rate,args.aux_learning_rate)<=0 or args.effort_weight<0:
        parser.error('Positive finite rates/fraction and nonnegative effort required')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,path in [('runner_source.py',Path(__file__)),('economy_source.py',Path(economy_module.__file__)),
                      ('objective_source.py',Path(objective_module.__file__))]:
        (args.out/name).write_bytes(path.read_bytes())
    cases=((70.,75.),(75.,75.),(80.,75.))
    pilot=json.loads(json.dumps(conditions(cases)))
    spec=WorkloadSpec(masses=(.5,),tilts=(30.,),cases=cases)
    source,native,meta=load_parent(args.parent);native.close()
    exploration=ExplorationSpec(**meta['contract']['exploration']);timing=ArmTiming(**meta['contract']['timing'])
    reward=EfficiencyReward(activation_squared_weight=args.effort_weight)
    checking=ArmEconomyEnv(spec,exploration,timing,reward,WeakeningSpec(fraction=args.fraction))
    initial=make_weakening_model(checking,source,checking.contract(),learning_rate=args.learning_rate)
    protocol={'schema':'myo.arm_economy_curriculum_training.v1','parent':str(args.parent),
        'parent_sha256':meta['checkpoint_sha256'],'seed':0,'pilot':pilot,'spec':asdict(spec),
        'cycles_cap':args.cycles,'minimum_rollouts':args.minimum_rollouts,'maximum_rollouts':args.maximum_rollouts,
        'transitions_per_rollout':1024,'aux_updates':args.aux_updates,'initial_fraction':args.fraction,
        'minimum_fraction':.0025,'aux_learning_rate':args.aux_learning_rate,'learning_rate':args.learning_rate,
        'reward':asdict(reward),'moment_scale_Nm':1.,'minimum_economy_step':.001,
        'comparison':'same initial weights/RMSprop, reward, independent Beta16x, seed0; '
            'control gets exactly the intervention PPO rollout count per cycle; auxiliary steps extra',
        'acceptance':'fixed original per-case quality gate AND full/tail activation^2 each decrease>=0.1% '
            'from current selected; select minimum full activation^2; all phases recorded',
        'lineage':'accepted weights AND RMSprop become next parent; rejected cycles restore selected '
            'and halve auxiliary fraction to .0025; no rollback between PPO recovery blocks',
        'stop':'up to12 cycles (configured);3 failed cycles in a row, or>=3 accepted cycles and '
            'full/tail cost>=15% below original and lower coactivation; finite budget is not plateau',
        'corpus':'causal native deterministic initial3 cases, add real PPO observations; max16384 replay, '
            'seed0 permutation at most8192 observations per auxiliary phase',
        'evaluation':'nominal means, no output weakening; evaluation process isolated from training RNG',
        'resource':'nice19,ionice idle,CPU12-13,Torch/BLAS1,4 sequential DummyVecEnv,1 evaluation process',
        'versions':{name:version(name) for name in ('torch','mujoco','stable-baselines3','numpy','gymnasium')},
        'source_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in (
            ('runner',Path(__file__)),('economy',Path(economy_module.__file__)),('objective',Path(objective_module.__file__)))}}
    save_workload(initial,checking,args.out/'initial.zip',protocol)
    with zipfile.ZipFile(args.parent) as old,zipfile.ZipFile(args.out/'initial.zip') as new:
        if any(old.read(key)!=new.read(key) for key in ('policy.pth','policy.optimizer.pth')):
            # Learning-rate assignment is allowed; require all tensor states and
            # all weights instead of assuming identical pickle bytes across envs.
            assert all(torch.equal(value,initial.policy.state_dict()[name]) for name,value in source.policy.state_dict().items())
            old_state=source.policy.optimizer.state_dict()['state'];new_state=initial.policy.optimizer.state_dict()['state']
            assert old_state.keys()==new_state.keys()
            assert all(torch.equal(v,new_state[k][f]) for k,row in old_state.items() for f,v in row.items() if isinstance(v,torch.Tensor))
    del source,initial
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();summary={'status':'running','protocol':protocol,'branches':[]}

    def write():
        summary['elapsed_s']=time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        def assess(path,traces=None):return pool.submit(evaluate,path,pilot,traces).result()
        baseline=assess(args.parent,args.out/'original_traces.npz')
        initial_result=assess(args.out/'initial.zip',args.out/'initial_traces.npz')
        if initial_result!=baseline:raise ValueError('Initial nominal/native behavior did not match source')
        summary['original']=baseline;write()
        corpus=pool.submit(collect_economy_corpus,args.out/'initial.zip',pilot,args.out/'initial_corpus.npz').result()
        rollout_schedule=[]
        for branch in ('weakening','control'):
            root=args.out/branch;root.mkdir()
            selected_path=args.out/'initial.zip';selected=baseline
            replay=corpus.copy();fraction=args.fraction;stale=accepted_cycles=0
            state={'branch':branch,'status':'running','cycles':[],'history':[],
                'ppo_transitions':0,'ppo_sgd':0,'auxiliary_attempts':0,'auxiliary_accepted':0,'accepted_cycles':0,
                'selected_path':str(selected_path),'selected':selected}
            summary['branches'].append(state);write()
            cycle_count=args.cycles if branch=='weakening' else len(rollout_schedule)
            for cycle in range(cycle_count):
                weakening=WeakeningSpec(fraction=fraction)
                env=ArmEconomyEnv(spec,exploration,timing,reward,weakening)
                vector=DummyVecEnv([partial(worker,spec,exploration,timing,reward,weakening,i) for i in range(4)])
                for worker_env in vector.envs:worker_env.env.episode_index=cycle*7
                parent,restored,parent_meta=load_economy(selected_path);restored.close()
                model=make_weakening_model(vector,parent,env.contract(),learning_rate=args.learning_rate);del parent
                set_random_seed(0);vector.seed(0)
                model.set_logger(configure(str(root/f'cycle_{cycle}'),['csv']))
                audit=RelaxationAudit(root)
                row={'cycle':cycle,'parent_path':str(selected_path),'parent_sha256':parent_meta['checkpoint_sha256'],
                     'fraction':fraction if branch=='weakening' else None,'updates':[],'accepted':False}
                trial_path=selected_path;trial=selected;trial_phase='parent'

                def checkpoint(phase):
                    nonlocal trial,trial_path,trial_phase
                    path=root/f'cycle_{cycle}_{phase}.zip'
                    training={**protocol,'branch':branch,'cycle':cycle,'phase':phase,
                        'lineage_parent':row['parent_path'],'lineage_parent_sha256':row['parent_sha256'],
                        'current_fraction':fraction,'ppo_transitions':state['ppo_transitions'],
                        'auxiliary_attempts':state['auxiliary_attempts']}
                    save_workload(model,env,path,training)
                    result=assess(path,root/f'cycle_{cycle}_{phase}_traces.npz')
                    eligible,reasons=economy_gate(result,baseline,selected)
                    quality,quality_reasons=quality_gate(result,baseline)
                    phase_row={'cycle':cycle,'phase':phase,'eligible':eligible,'reasons':reasons,
                        'quality_eligible':quality,'quality_reasons':quality_reasons,'path':str(path),'result':result}
                    state['history'].append(phase_row)
                    if eligible and result['activation_squared_mean']<trial['activation_squared_mean']:
                        trial,trial_path,trial_phase=result,path,phase
                    write()
                    print(json.dumps({'branch':branch,'cycle':cycle,'phase':phase,'eligible':eligible,
                        'quality':quality,'rmse':result['score_deg'],'a2':result['activation_squared_mean'],
                        'tail_a2':result['tail_activation_squared_mean'],'coactivation':result['tail_activation_coactivation'],
                        'reduction':1-result['activation_squared_mean']/baseline['activation_squared_mean'],
                        'reasons':reasons,'elapsed_s':summary['elapsed_s']}),flush=True)

                used_rollouts=0
                try:
                    if branch=='weakening':
                        subset=np.random.default_rng(0).permutation(len(replay))[:8192]
                        set_learning_rate(model,args.aux_learning_rate)
                        report=weaken_with_moment_constraint(model,replay[subset],env.contract(),updates=args.aux_updates)
                        state['auxiliary_attempts']+=report['attempted_updates'];state['auxiliary_accepted']+=report['accepted_updates']
                        row['auxiliary']=report
                        set_learning_rate(model,args.learning_rate)
                        checkpoint('auxiliary')
                    limit=args.maximum_rollouts if branch=='weakening' else rollout_schedule[cycle]
                    for block in range(limit):
                        model.learn(1024,reset_num_timesteps=False,callback=audit)
                        used_rollouts+=1;state['ppo_transitions']+=1024
                        state['ppo_sgd']+=model.last_audit['optimizer_steps']
                        row['updates'].append(dict(model.last_audit))
                        replay=np.concatenate((replay,model.rollout_buffer.observations.reshape(-1,156)))[-16384:]
                        checkpoint(f'ppo_{block+1}')
                        if branch=='weakening' and used_rollouts>=args.minimum_rollouts and trial_path!=selected_path:
                            break
                    if branch=='weakening':rollout_schedule.append(used_rollouts)
                    row.update(used_rollouts=used_rollouts,exploration=audit.report(),selected_phase=trial_phase)
                    if trial_path!=selected_path:
                        selected_path,selected=trial_path,trial;stale=0;accepted_cycles+=1;row['accepted']=True
                    else:
                        stale+=1
                        if branch=='weakening':fraction=max(.0025,fraction/2)
                    row.update(selected_path=str(selected_path),selected=selected)
                    state['cycles'].append(row)
                    state.update(selected_path=str(selected_path),selected=selected,accepted_cycles=accepted_cycles,
                        reduction=1-selected['activation_squared_mean']/baseline['activation_squared_mean'])
                    write()
                    print(json.dumps({'cycle_done':cycle,'branch':branch,'accepted':row['accepted'],
                        'selected_phase':trial_phase,'selected':str(selected_path),'accepted_cycles':accepted_cycles,
                        'reduction':state['reduction'],'rollouts':used_rollouts,'next_fraction':fraction}),flush=True)
                finally:
                    vector.close();env.close()
                if branch=='weakening':
                    local_success=(accepted_cycles>=3 and state['reduction']>=.15
                        and selected['tail_activation_squared_mean']<=.85*baseline['tail_activation_squared_mean']
                        and selected['tail_activation_coactivation']<baseline['tail_activation_coactivation'])
                    if local_success or stale>=3:
                        state['stop_reason']='local_economy_goal' if local_success else 'three_rejected_cycles';break
            state.update(status='complete',stop_reason=state.get('stop_reason','cycle_budget_cap'))
            shutil.copyfile(selected_path,root/'best.zip')
            shutil.copyfile(Path(str(selected_path)+'.metadata.json'),root/'best.zip.metadata.json')
            write()
        summary.update(status='complete',selection_frozen=True,rollout_schedule=rollout_schedule,
            accepted_default=False,stop_reason='local_stage_complete_transfer_not_yet_evaluated')
        write()
    checking.close()


if __name__=='__main__':main()
