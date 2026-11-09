"""Freeze trained candidates, replay validation and evaluate withheld workload cases."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import torch

from myo.arm_precision import TEST_CASES
from myo.arm_workload import scenarios
from myo.arm_workload_evaluation import evaluate_parallel
from myo.train_vertical_ppo import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--kind',choices=('mass','circle','joint'),default='mass')
    p.add_argument('--long',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    summaries={root:json.loads((root/'summary.json').read_text()) for root in args.roots}
    assert all(s['status'] in ('training_complete','complete') for s in summaries.values())
    manifests={}
    for root,s in summaries.items():
        labels=['initial','best','last']+(['energy_best'] if 'energy_best_steps' in s else [])
        manifest={label:json.loads((root/(label+'.zip.metadata.json')).read_text())['checkpoint_sha256'] for label in labels}
        frozen=root/'frozen_selection.json'
        if frozen.exists():assert json.loads(frozen.read_text())==manifest
        else:write_json(frozen,manifest)
        manifests[root]=manifest
    if args.kind=='mass':
        conditions=scenarios(TEST_CASES,masses=(.75,1.5,2.5))
    elif args.kind=='circle':
        conditions=scenarios(((q,q) for q in (30,60,90,120)),tilts=(45.,105.,165.,225.,285.,345.))
    else:
        conditions=[]
        for i,case in enumerate(TEST_CASES):
            for j in range(6):
                conditions+=scenarios((case,),masses=((.75,1.5,2.5)[(i+j)%3],),
                    tilts=((45.,105.,165.,225.,285.,345.)[j],),capacities=((1.,.85)[j%2],))
    long=scenarios(((60,60),(120,120)),masses=(.75,2.),
                   tilts=(30.,120.,210.,300.) if args.kind!='mass' else (30.,),capacities=(.85,),seconds=20.)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for root,s in summaries.items():
            for label in manifests[root]:
                path=root/(label+'.zip')
                assert hashlib.sha256(path.read_bytes()).hexdigest()==manifests[root][label]
                if label in ('best','last'):
                    replay=evaluate_parallel(pool,path,s['training']['validation'],4)
                    expected=s['best_validation'] if label=='best' else s['history'][-1]['validation']
                    assert replay==expected
                result=evaluate_parallel(pool,path,conditions,4)
                write_json(root/(label+'_test.json'),result)
                s[label+'_test']=result
                if args.long and label in ('best','energy_best'):
                    s[label+'_long_holds']=evaluate_parallel(pool,path,long,4)
                    write_json(root/(label+'_long_holds.json'),s[label+'_long_holds'])
                print(json.dumps({'root':str(root),'label':label,'test_deg':result['score_deg'],
                    'pass':result['passed'],'total':result['total'],'failures':result['failures']}),flush=True)
            s.update(status='complete',strict_validation_replay=True,test_selection_frozen=True,
                     test_kind=args.kind)
            write_json(root/'summary.json',s)


if __name__=='__main__':main()
