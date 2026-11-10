"""Compare preselected reward-stage policies on training cells and local holdout."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import torch

from myo.arm_workload import scenarios
from myo.arm_workload_evaluation import evaluate_parallel
from myo.train_vertical_ppo import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--include-test',action='store_true')
    parser.add_argument('--test-file',type=Path,default=Path('runs/arm_reward_curriculum/test.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for root in args.roots:
            summary=json.loads((root/'summary.json').read_text())
            if summary['status'] not in ('training_complete','complete'):
                raise ValueError('Finish training before assessment')
            meta=json.loads((root/'best.zip.metadata.json').read_text())
            digest=hashlib.sha256((root/'best.zip').read_bytes()).hexdigest()
            assert digest==meta['checkpoint_sha256']
            spec=meta['training']['spec']
            conditions={'train':scenarios(spec['cases'],masses=spec['masses'],tilts=spec['tilts'],
                                          capacities=spec['capacities'],seconds=spec['seconds'])}
            if args.include_test:
                conditions['test']=json.loads(args.test_file.read_text())
            assessment={'checkpoint_sha256':digest,'selected_by':'saved validation rank, before this assessment',
                        'reward_stage':meta['training']['reward_stage'],'validation':summary['best_validation']}
            for name,cases in conditions.items():
                result=evaluate_parallel(pool,root/'best.zip',cases,args.workers)
                assessment[name]=result
                write_json(root/'assessment.json',assessment)
                print(json.dumps({'root':str(root),'set':name,'passed':result['passed'],
                    'total':result['total'],'rmse_deg':result['score_deg']}),flush=True)
            assessment['status']='complete'
            if args.include_test:
                assessment['test_manifest_sha256']=hashlib.sha256(args.test_file.read_bytes()).hexdigest()
            write_json(root/'assessment.json',assessment)


if __name__=='__main__':
    main()
