"""Compare workload policies with one validation-only effort selection rule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import torch

from myo.arm_precision import TEST_CASES
from myo.arm_workload import scenarios,stage_passed
from myo.arm_workload_evaluation import evaluate_parallel
from myo.train_vertical_ppo import write_json


def select(root,metric):
    summary=json.loads((root/'summary.json').read_text())
    if summary['status'] not in ('training_complete','complete'):
        raise ValueError('Finish training before freezing effort candidates')
    initial=summary['history'][0]['validation']
    eligible=[h for h in summary['history'] if stage_passed(h['validation'])
        and h['validation']['passed']>=initial['passed']
        and h['validation']['score_deg']<=initial['score_deg']+.25]
    chosen=min(eligible,key=lambda h:(h['validation'][metric],h['steps']))
    source=root/'checkpoints'/f"step_{chosen['steps']}.zip"
    meta=json.loads(Path(str(source)+'.metadata.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==meta['checkpoint_sha256']
    selection={'metric':metric,'constraints':'stage_passed; passes>=initial; mean<=initial+0.25deg',
        'steps':chosen['steps'],'checkpoint_sha256':digest,'validation':chosen['validation']}
    frozen=root/'matched_efficiency_selection.json'
    if frozen.exists():assert json.loads(frozen.read_text())==selection
    else:write_json(frozen,selection)
    target=root/'matched_efficiency.zip'
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    else:
        shutil.copyfile(source,target)
        shutil.copyfile(Path(str(source)+'.metadata.json'),Path(str(target)+'.metadata.json'))
    return target,selection,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    parser.add_argument('--metric',choices=('active_fraction_mean','activation_squared_mean'),default='active_fraction_mean')
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--output',type=Path,default=Path('runs/arm_workload_efficiency/comparison.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    selected={root:select(root,args.metric) for root in args.roots}
    first=next(iter(selected.values()))[2]['training']
    assert all(s['training']['parent_sha256']==first['parent_sha256']
        and s['training']['validation']==first['validation'] for _,_,s in selected.values())
    def nonreward_spec(spec):
        return {k:v for k,v in spec.items() if k not in ('activation_squared_weight','active_fraction_weight')}
    assert all(nonreward_spec(s['training']['spec'])==nonreward_spec(first['spec'])
        and s['training']['ppo']==first['ppo'] and s['training']['workers']==first['workers']
        and s['training']['seed']==first['seed'] for _,_,s in selected.values())
    assert len({tuple(h['added_transitions'] for h in s['history']) for _,_,s in selected.values()})==1
    test=scenarios(TEST_CASES,masses=(.75,1.5,2.5))
    long=scenarios(((60,60),(120,120)),masses=(.75,2.),capacities=(.85,),seconds=20.)
    report={'selection':'same validation-only metric and quality constraints for every branch','metric':args.metric,'models':{}}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for root,(path,selection,summary) in selected.items():
            replay=evaluate_parallel(pool,path,summary['training']['validation'],args.workers)
            assert replay==selection['validation']
            result={'selection':selection,'strict_validation_replay':True}
            for name,conditions in (('test',test),('long_holds',long)):
                value=evaluate_parallel(pool,path,conditions,args.workers)
                result[name]=value
                write_json(root/('matched_efficiency_'+name+'.json'),value)
                print(json.dumps({'root':str(root),'set':name,'score_deg':value['score_deg'],
                    'passed':value['passed'],'total':value['total'],'a2':value['activation_squared_mean'],
                    'active_fraction':value['active_fraction_mean'],
                    'maximum_capacity_loss':float(np.max([r['capacity_loss_by_muscle'] for r in value['cases']]))}),flush=True)
            report['models'][str(root)]=result
            write_json(args.output,report)


if __name__=='__main__':main()
