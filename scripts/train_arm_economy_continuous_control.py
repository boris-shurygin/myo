"""Distinguish continuous PPO from repeated gated short PPO control attempts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from myo.arm_economy_curriculum import economy_gate, load_economy
from myo.arm_weakening import make_weakening_model
from myo.arm_workload import save_workload
from train_arm_economy_curriculum import worker
from train_arm_relaxation import RelaxationAudit, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    reference=json.loads((args.reference/'summary.json').read_text())
    if reference['status']!='complete' or not reference.get('selection_frozen'):
        raise ValueError('Reference training must finish before matching actual budget')
    schedule=reference['rollout_schedule'];baseline=reference['original'];pilot=reference['protocol']['pilot']
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    model,checking,meta=load_economy(args.reference/'initial.zip');del model
    initial=args.reference/'initial.zip'
    protocol={**reference['protocol'],'method':'continuous_ppo_without_auxiliary',
        'comparison':'same initial, reward, noise, LR, scenario/RNG restart each cycle and exact rollout schedule; '
            'continue LAST weights AND RMSprop even if quality gate rejected; best tracked only for reporting',
        'reference':str(args.reference),'rollout_schedule':schedule,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summary={'status':'running','protocol':protocol,'original':baseline,'history':[],
        'ppo_transitions':0,'ppo_sgd':0,'selected_path':str(initial),'selected':baseline,'reduction':0.}
    started=time.monotonic();current=selected_path=initial;selected=baseline

    def write():
        summary['elapsed_s']=time.monotonic()-started
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        initial_report=pool.submit(evaluate,initial,pilot).result()
        if initial_report!=baseline:raise ValueError('Continuous-control initial baseline mismatch')
        write()
        for cycle,rollouts in enumerate(schedule):
            vector=DummyVecEnv([partial(worker,checking.workload,checking.exploration,checking.timing,
                checking.efficiency_reward,checking.weakening,i) for i in range(4)])
            for env in vector.envs:env.env.episode_index=cycle*7
            parent,restored,parent_meta=load_economy(current);restored.close()
            model=make_weakening_model(vector,parent,checking.contract(),learning_rate=protocol['learning_rate']);del parent
            set_random_seed(0);vector.seed(0)
            model.set_logger(configure(str(args.out/f'cycle_{cycle}'),['csv']))
            audit=RelaxationAudit(args.out)
            try:
                for block in range(rollouts):
                    model.learn(1024,reset_num_timesteps=False,callback=audit)
                    summary['ppo_transitions']+=1024;summary['ppo_sgd']+=model.last_audit['optimizer_steps']
                    path=args.out/f'cycle_{cycle}_ppo_{block+1}.zip'
                    training={**protocol,'cycle':cycle,'phase':f'ppo_{block+1}',
                        'lineage_parent':str(current),'lineage_parent_sha256':parent_meta['checkpoint_sha256'],
                        'ppo_transitions':summary['ppo_transitions']}
                    save_workload(model,checking,path,training)
                    result=pool.submit(evaluate,path,pilot,args.out/f'cycle_{cycle}_ppo_{block+1}_traces.npz').result()
                    if cycle==0:
                        control=next(b for b in reference['branches'] if b['branch']=='control')
                        if result!=control['history'][block]['result']:
                            raise ValueError('First continuous-control cycle differs from matched reset control')
                    eligible,reasons=economy_gate(result,baseline,selected)
                    accepted=eligible and result['activation_squared_mean']<selected['activation_squared_mean']
                    if accepted:selected_path,selected=path,result
                    summary['history'].append({'cycle':cycle,'block':block+1,'path':str(path),
                        'eligible':eligible,'reasons':reasons,'accepted':accepted,'result':result,'optimizer':model.last_audit})
                    summary.update(selected_path=str(selected_path),selected=selected,
                        reduction=1-selected['activation_squared_mean']/baseline['activation_squared_mean'])
                    write()
                    print(json.dumps({'cycle':cycle,'block':block+1,'eligible':eligible,'accepted':accepted,
                        'rmse':result['score_deg'],'a2':result['activation_squared_mean'],
                        'reduction':1-result['activation_squared_mean']/baseline['activation_squared_mean'],
                        'selected_reduction':summary['reduction'],'elapsed_s':summary['elapsed_s']}),flush=True)
                current=path
            finally:
                vector.close()
        assert summary['ppo_transitions']==1024*sum(schedule)
        shutil.copyfile(selected_path,args.out/'best.zip')
        shutil.copyfile(Path(str(selected_path)+'.metadata.json'),args.out/'best.zip.metadata.json')
        summary.update(status='complete',selection_frozen=True,last_path=str(current),last=result)
        write()
    checking.close()


if __name__=='__main__':main()
