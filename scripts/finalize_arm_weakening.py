"""Frozen-candidate holdouts for the weakening study; no more policy selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from myo.arm_curriculum import conditions
from myo.arm_efficiency import quality_gate
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from train_arm_relaxation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    old_path=Path('runs/arm_relaxation_10ms/comparison/summary.json')
    old=json.loads(old_path.read_text())
    meta=json.loads(Path(str(args.candidate)+'.metadata.json').read_text())
    if meta['training']['parent_sha256']!=old['protocol']['parent_sha256']:
        raise ValueError('Original baseline identity mismatch')
    groups={'local_test':conditions(tuple((g+o,g) for g in (60.,90.) for o in (-10.,0.,10.))),
        'old_train_validation':conditions(TRAIN_CASES)+conditions(VALIDATION_CASES),
        'known_test':conditions(TEST_CASES),
        'long_holds':[{**c,'seconds':20.} for c in conditions(((55.,55.),(75.,75.),(95.,95.)))]}
    summary={'candidate':str(args.candidate),'candidate_sha256':meta['checkpoint_sha256'],
        'selection_frozen':True,'baseline_reference':str(old_path),
        'baseline_reference_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'reports':{},'status':'running'}
    for name,cases in groups.items():
        summary['reports'][name]=evaluate(args.candidate,cases,args.out/f'{name}_traces.npz')
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        r=summary['reports'][name]
        print(json.dumps({'group':name,'passed':r['passed'],'total':r['total'],'rmse':r['score_deg'],
            'a2':r['activation_squared_mean'],'failures':r['failures']}),flush=True)
    gates={};retained=True
    for new_name,old_name in (('old_train_validation','full_range'),('known_test','known_test')):
        before=old['reports']['original'][old_name];after=summary['reports'][new_name]
        mask=[i for i,row in enumerate(before['cases']) if row['passed']]
        eligible,reasons=quality_gate({'cases':[after['cases'][i] for i in mask]},
                                      {'cases':[before['cases'][i] for i in mask]})
        gates[new_name]={'eligible':eligible,'reasons':reasons,'checked_original_passes':len(mask)}
        retained &= eligible
    summary.update(status='complete',retention_gates=gates,
        eligible=bool(retained and summary['reports']['local_test']['passed']==6
            and summary['reports']['long_holds']['passed']==3
            and all(r['failures']==0 for r in summary['reports'].values())))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
