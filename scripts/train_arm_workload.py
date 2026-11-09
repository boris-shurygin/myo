"""Adaptive staged PPO: mass, periodic arm mount, activation and fatigue proxies."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from functools import partial
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from myo.arm_learning import load_arm
from myo.arm_precision import VALIDATION_CASES, TEST_CASES
from myo.arm_workload import (ArmWorkloadEnv, WorkloadSpec, make_workload_model,
    load_workload, save_workload, scenarios, evaluate_workload, validation_rank, stage_passed)
from myo.train_vertical_ppo import write_json
from myo.arm_workload_evaluation import evaluate_parallel


HOLDS = tuple((q,q) for q in (30,60,90,120))
MOTIONS = ((20,75),(75,20),(25,105),(105,25),(60,125))
CIRCLE = tuple(float(q) for q in range(0,360,30))


def profile(stage):
    spec=WorkloadSpec()
    if stage=='fixed':
        return replace(spec,masses=(1.,)),scenarios(VALIDATION_CASES,masses=(.5,1.,2.))
    if stage=='mass':
        return spec,scenarios(VALIDATION_CASES,masses=(.5,1.,2.))
    if stage=='hold_near':
        return replace(spec,masses=(1.,),tilts=(0.,30.,60.),cases=HOLDS),scenarios(
            ((q,q) for q in (40,70,100,125)),tilts=(15.,30.,45.))
    if stage=='hold_circle':
        return replace(spec,masses=(1.,),tilts=CIRCLE,cases=HOLDS),scenarios(
            ((q,q) for q in (40,70,100,125)),tilts=(15.,75.,135.,195.,255.,315.))
    if stage=='motion_circle':
        return replace(spec,masses=(1.,),tilts=CIRCLE,cases=HOLDS+MOTIONS),scenarios(
            VALIDATION_CASES,tilts=(15.,75.,135.,195.,255.,315.))
    if stage in ('joint','efficiency','fatigue'):
        capacities=(1.,.7) if stage=='fatigue' else (1.,)
        spec=replace(spec,tilts=CIRCLE,cases=HOLDS+MOTIONS,capacities=capacities,
                     seconds=8. if stage=='fatigue' else 4.)
        validation=[]
        tilts=(15.,75.,135.,195.,255.,315.)
        for i,case in enumerate(VALIDATION_CASES):
            for j in range(4):
                validation+=scenarios((case,),masses=((.75,1.5,2.)[(i+j)%3],),
                    tilts=(tilts[(i+2*j)%6],),capacities=((1.,.85)[j%2],) if stage=='fatigue' else (1.,))
        return spec,validation
    raise ValueError('Unknown stage')


def worker_env(spec,worker,workers):
    return Monitor(ArmWorkloadEnv(spec,worker=worker,workers=workers))


def load_parent(path):
    meta=json.loads(Path(str(path)+'.metadata.json').read_text())
    return (load_workload if meta['contract']['schema']=='myo.arm_workload.v1' else load_arm)(path)


def energy_value(result):
    # Report both physical proxies separately. This fixed scalar ranking uses
    # raw proxy units; it is not a measured metabolic cost.
    return result['activation_squared_mean']+result['active_fraction_mean']*.1


def worst_group(result):
    return max(g['score_deg'] for groups in result['groups'].values() for g in groups.values())


def train(args):
    args.root.mkdir(exist_ok=False,parents=True)
    (args.root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.root/'checkpoints').mkdir()
    spec,validation=profile(args.stage)
    overrides={}
    if args.masses is not None:overrides['masses']=tuple(args.masses)
    if args.tilts is not None:overrides['tilts']=tuple(args.tilts)
    if args.capacities is not None:overrides['capacities']=tuple(args.capacities)
    if args.hold_targets is not None:overrides['cases']=tuple((q,q) for q in args.hold_targets)
    if args.energy is not None:overrides['activation_squared_weight']=args.energy
    if args.fatigue is not None:overrides['active_fraction_weight']=args.fatigue
    spec=replace(spec,**overrides)
    if (args.validation_tilts is not None or args.validation_masses is not None
            or args.validation_capacities is not None):
        cases=tuple(dict.fromkeys(tuple(c['case']) for c in validation))
        masses=tuple(args.validation_masses) if args.validation_masses is not None else tuple(sorted({c['payload_mass'] for c in validation}))
        tilts=tuple(args.validation_tilts) if args.validation_tilts is not None else tuple(sorted({c['tilt_deg'] for c in validation}))
        capacities=(tuple(args.validation_capacities) if args.validation_capacities is not None
                    else tuple(sorted({c['capacity'] for c in validation})))
        validation=scenarios(cases,masses=masses,tilts=tilts,capacities=capacities)
    checking=ArmWorkloadEnv(spec)
    contract=checking.contract()
    parent,parent_env,parent_meta=load_parent(args.parent);parent_env.close()
    assert parent.policy.feature_mode=='none'
    vector=SubprocVecEnv([partial(worker_env,spec,i,args.workers) for i in range(args.workers)],start_method='spawn')
    evaluation_pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'))
    try:
        model=make_workload_model(vector,parent,contract,learning_rate=args.lr,n_steps=1024//args.workers)
        model.ent_coef=args.entropy
        training={'schema':'myo.arm_workload_training.v1','stage':args.stage,'seed':0,
            'parent':str(args.parent),'parent_sha256':parent_meta['checkpoint_sha256'],
            'source_steps':parent.num_timesteps,'source_optimizer_steps':parent.optimizer_steps,
            'transfer':'equal named weights/moments; new orientation columns zero; environment and RNG restarted once',
            'workers':args.workers,'ppo':{'lr':args.lr,'rollout_total':1024,'steps_per_env':1024//args.workers,
                'batch':128,'epochs':10,'gamma':.99,'gae_lambda':.95,'clip':.2,'entropy':args.entropy,
                'value':.5,'grad_norm':.5,'target_kl':.03,'optimizer':'RMSprop'},
            'validation':validation,'minimum_transitions':args.minimum_steps,'fixed_steps':args.fixed_steps,
            'stop':'stage passes after minimum experience, or two32768-transition blocks without accumulated2% progress',
            'efficiency_selection':('weighted added activation cost; >=10% reduction without losing initial passes '
                'and mean error <=initial+0.25deg; native costs unchanged'
                if spec.activation_squared_weight>0 or spec.active_fraction_weight>0
                else 'tracking rank; auxiliary combined effort proxy does not determine stopping'),
            'spec':asdict(spec),'versions':{k:version(k) for k in ('torch','stable-baselines3','mujoco','numpy')},
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        source_steps,source_sgd=model.num_timesteps,model.optimizer_steps
        summary={'status':'running','training':training,'contract':contract,'history':[],'updates':[],
            'source_steps':source_steps,'source_optimizer_steps':source_sgd,'blocks_without_progress':0}
        save_workload(model,checking,args.root/'initial.zip',training)
        model.set_random_seed(0)
        started=time.monotonic()
        best=None;reference=None;progress_in_block=False;best_energy=None;initial=None
        optimizing=spec.activation_squared_weight>0 or spec.active_fraction_weight>0

        def effort_value(result):
            if not optimizing:
                return energy_value(result)
            return (spec.activation_squared_weight*result['activation_squared_mean']
                    +spec.active_fraction_weight*result['active_fraction_mean'])

        def check():
            nonlocal best,reference,progress_in_block,best_energy,initial
            save_workload(model,checking,args.root/'candidate.zip',training)
            result=evaluate_parallel(evaluation_pool,args.root/'candidate.zip',validation,args.workers)
            if initial is None:initial=result
            current=validation_rank(result)
            if best is None or current<best:
                best=current
                summary.update(best_steps=model.num_timesteps,best_validation=result)
                save_workload(model,checking,args.root/'best.zip',training)
            eligible=(result['failures']==0 and result['passed']>=initial['passed']
                      and result['score_deg']<=initial['score_deg']+.25
                      and (not optimizing or stage_passed(result)))
            if eligible and (best_energy is None or effort_value(result)<best_energy):
                best_energy=effort_value(result)
                summary.update(energy_best_steps=model.num_timesteps,energy_best_validation=result)
                save_workload(model,checking,args.root/'energy_best.zip',training)
            if reference is None or (result['failures']<reference['failures']
                or (result['failures']==reference['failures'] and (
                    (result['passed']>reference['passed'] and result['score_deg']<=1.05*reference['score_deg'])
                    or (result['passed']>=reference['passed'] and result['score_deg']<=.98*reference['score_deg'])
                    or (result['passed']>=reference['passed'] and worst_group(result)<=.98*worst_group(reference))
                    or (optimizing and eligible and effort_value(result)<=.98*effort_value(reference))))):
                reference=result;progress_in_block=True
            summary['history'].append({'steps':model.num_timesteps,'added_transitions':model.num_timesteps-source_steps,
                'validation':result,'optimizer_steps':model.optimizer_steps,'energy_eligible':eligible})
            summary.update(steps=model.num_timesteps,elapsed_s=time.monotonic()-started)
            save_workload(model,checking,args.root/'checkpoints'/f'step_{model.num_timesteps}.zip',training)
            write_json(args.root/'summary.json',summary)
            print(json.dumps({'stage':args.stage,'added':model.num_timesteps-source_steps,
                'validation_deg':result['score_deg'],'passed':result['passed'],'total':result['total'],
                'failures':result['failures'],'a2':result['activation_squared_mean'],
                'active_fraction':result['active_fraction_mean'],'sgd_added':model.optimizer_steps-source_sgd,
                'elapsed_s':summary['elapsed_s']}),flush=True)

        check()
        while True:
            model.learn(1024,reset_num_timesteps=False)
            summary['updates'].append({'steps':model.num_timesteps,**model.last_audit})
            added=model.num_timesteps-source_steps
            if added%args.check_steps==0:
                check()
                if added>=args.minimum_steps:
                    achieved=(stage_passed(summary['best_validation']) if not optimizing else
                        best_energy is not None and best_energy<=.9*effort_value(initial)
                        and stage_passed(summary['energy_best_validation']))
                    if achieved and not args.fixed_steps:
                        summary['stop_reason']='validation_stage_passed';break
            if args.fixed_steps and added>=args.fixed_steps:
                summary['stop_reason']='fixed_comparison_budget';break
            if added%32768==0:
                summary['blocks_without_progress']=0 if progress_in_block else summary['blocks_without_progress']+1
                progress_in_block=False
                if not args.fixed_steps and added>=args.minimum_steps and summary['blocks_without_progress']>=2:
                    summary['stop_reason']='validation_plateau_two_blocks';break
        save_workload(model,checking,args.root/'last.zip',training)
        summary.update(status='training_complete',added_transitions=model.num_timesteps-source_steps,
            added_optimizer_steps=model.optimizer_steps-source_sgd,elapsed_s=time.monotonic()-started)
        write_json(args.root/'summary.json',summary)
        print(json.dumps({'status':summary['status'],'stop':summary['stop_reason'],
                          'added':summary['added_transitions'],'best_deg':summary['best_validation']['score_deg']}),flush=True)
    finally:
        vector.close();evaluation_pool.shutdown();checking.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('fixed','mass','hold_near','hold_circle','motion_circle','joint','efficiency','fatigue'),required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=Path('runs/arm_continuation/best.zip'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--entropy',type=float,default=.001)
    p.add_argument('--minimum-steps',type=int,default=65536)
    p.add_argument('--fixed-steps',type=int)
    p.add_argument('--check-steps',type=int,default=16384)
    p.add_argument('--masses',type=float,nargs='+')
    p.add_argument('--tilts',type=float,nargs='+')
    p.add_argument('--capacities',type=float,nargs='+')
    p.add_argument('--hold-targets',type=float,nargs='+')
    p.add_argument('--validation-tilts',type=float,nargs='+')
    p.add_argument('--validation-masses',type=float,nargs='+')
    p.add_argument('--validation-capacities',type=float,nargs='+')
    p.add_argument('--energy',type=float)
    p.add_argument('--fatigue',type=float)
    args=p.parse_args()
    if args.workers not in (1,2,4) or args.minimum_steps%32768 or args.check_steps%1024:
        p.error('Use1/2/4 workers, minimum budget divisible by32768 and check interval by1024')
    if args.fixed_steps is not None and (args.fixed_steps<=0 or args.fixed_steps%args.check_steps):
        p.error('Fixed budget must consist of complete validation intervals')
    torch.set_num_threads(1)
    train(args)


if __name__=='__main__':main()
