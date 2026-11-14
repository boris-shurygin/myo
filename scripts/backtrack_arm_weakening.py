"""Shorten an actual learned actor update using the native per-case task gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from myo.arm_efficiency import quality_gate
from myo.arm_weakening import load_weakening
from myo.arm_workload import save_workload
from train_arm_relaxation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    protocol=json.loads((args.root/'protocol.json').read_text())
    baseline=json.loads((args.root/'summary.json').read_text())['original']
    parent,env,meta=load_weakening(args.root/'initial.zip')
    candidate,check,trained_meta=load_weakening(args.candidate);check.close()
    if trained_meta['contract']!=meta['contract']:
        raise ValueError('Backtracking cannot alter policy, physics or observations')
    old=copy.deepcopy(parent.policy.state_dict());new=candidate.policy.state_dict()
    # This experiment shortens an auxiliary actor update, before PPO. Critic and
    # calibration buffers must be unchanged; do not interpolate a different task.
    for name,value in old.items():
        if 'critic' in name or 'value_' in name or name not in dict(parent.policy.named_parameters()):
            if not torch.equal(value,new[name]):raise ValueError('Expected an actor-only update')
    summary={'protocol':{'parent':str(args.root/'initial.zip'),'parent_sha256':meta['checkpoint_sha256'],
        'learned_candidate':str(args.candidate),'candidate_sha256':trained_meta['checkpoint_sha256'],
        'rule':'interpolate learned actor weights theta=parent+s*(candidate-parent); '
            'keep parent RMSprop moments and calibration; no command scaling at runtime, no extra SGD; '
            'try s=.5 then.25, choose largest passing native20-case task gate with lower actual a²',
        'pilot':protocol['pilot']},'history':[],'selected_path':None,'status':'running'}
    try:
        for fraction in (.5,.25):
            parent.policy.load_state_dict({name: value+fraction*(new[name]-value)
                if value.is_floating_point() else value for name,value in old.items()})
            path=args.out/f'fraction_{fraction}.zip'
            training={**protocol,'method':'native_quality_backtracking_of_auxiliary_actor_update',
                'weight_fraction':fraction,'learned_candidate':str(args.candidate),
                'learned_candidate_sha256':trained_meta['checkpoint_sha256'],
                'rmsprop':'restored parent state, not interpolated; extra SGD0',
                'source_auxiliary_training':trained_meta['training']}
            save_workload(parent,env,path,training)
            result=evaluate(path,protocol['pilot'],args.out/f'fraction_{fraction}_traces.npz')
            eligible,reasons=quality_gate(result,baseline)
            row={'fraction':fraction,'eligible':eligible,'reasons':reasons,'result':result,
                'reduction':1-result['activation_squared_mean']/baseline['activation_squared_mean']}
            summary['history'].append(row)
            if eligible and row['reduction']>0:
                summary.update(selected_path=str(path),selected=result,reduction=row['reduction'])
            (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({'fraction':fraction,'eligible':eligible,'reduction':row['reduction'],
                'rmse':result['score_deg'],'a2':result['activation_squared_mean'],'reasons':reasons}),flush=True)
            if summary['selected_path']:
                break
        summary['status']='complete'
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        env.close()


if __name__=='__main__':
    main()
