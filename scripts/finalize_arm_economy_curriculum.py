"""Frozen local choices: measure transfer without tuning on these scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from myo.arm_curriculum import conditions
from myo.arm_efficiency import quality_gate
from myo.arm_precision import TEST_CASES
from train_arm_relaxation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--continuous',type=Path)
    parser.add_argument('--precision-matched',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    study=json.loads((args.root/'summary.json').read_text())
    if study['status']!='complete' or not study.get('selection_frozen'):
        raise ValueError('Freeze local selections before looking at transfer')
    old_path=Path('runs/arm_weakening_10ms/checked/summary.json')
    old=json.loads(old_path.read_text())
    if old['protocol']['parent_sha256']!=study['protocol']['parent_sha256']:
        raise ValueError('Baseline parent identity mismatch')
    known_path=Path('runs/arm_relaxation_10ms/comparison/summary.json')
    known=json.loads(known_path.read_text())
    if known['protocol']['parent_sha256']!=study['protocol']['parent_sha256']:
        raise ValueError('Known-test baseline parent identity mismatch')
    precision=None
    if args.precision_matched:
        branch=next(b for b in study['branches'] if b['branch']=='weakening')
        rows=[h for h in branch['history'] if h['quality_eligible']
            and all(new['tail_rmse_deg']<=before['tail_rmse_deg']
                for new,before in zip(h['result']['cases'],study['original']['cases']))
            and all(h['result'][key]<study['original'][key]
                for key in ('activation_squared_mean','tail_activation_squared_mean'))]
        if not rows:raise ValueError('No locally precision-matched economy candidate')
        precision=min(rows,key=lambda h:h['result']['activation_squared_mean'])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    pilot=old['protocol']['pilot']
    long=[{**conditions(((75.,75.),))[0],'seconds':20.}]
    local_test=conditions(tuple((g+o,g) for g in (60.,90.) for o in (-10.,0.,10.)))
    result={'status':'running','selection_frozen':True,'parent_sha256':study['protocol']['parent_sha256'],
        'baseline_reference':str(old_path),'baseline_reference_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'known_test_reference':str(known_path),'known_test_reference_sha256':hashlib.sha256(known_path.read_bytes()).hexdigest(),
        'original':{'pilot':old['original'],'known_test':known['reports']['original']['known_test']},'branches':{}}
    if precision:
        result['precision_matched_selection']={'path':precision['path'],
            'rule':'exploratory secondary, selected after local training but before ANY transfer: '
                'fixed original quality gate, per-case local RMSE no worse than initial, lower full/tail a2; '
                'among those minimum full a2; primary selection and15% threshold unchanged',
            'local_result':precision['result']}

    def write():
        (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    result['original']['long_hold']=evaluate(Path(study['protocol']['parent']),long,args.out/'original_long_traces.npz')
    write()
    branches=[(branch['branch'],args.root/branch['branch']/'best.zip',branch) for branch in study['branches']]
    if precision:
        branches.append(('precision_matched',Path(precision['path']),{
            'reduction':1-precision['result']['activation_squared_mean']/study['original']['activation_squared_mean']}))
    if args.continuous:
        continuous=json.loads((args.continuous/'summary.json').read_text())
        if continuous['status']!='complete' or not continuous.get('selection_frozen'):
            raise ValueError('Continuous control must finish before transfer')
        branches.append(('continuous',args.continuous/'best.zip',continuous))
    seen={}
    for name,path,branch in branches:
        header=json.loads(Path(str(path)+'.metadata.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=header['checkpoint_sha256']:
            raise ValueError('Selected checkpoint hash mismatch')
        identity=(header['checkpoint_sha256'],json.dumps(header,sort_keys=True))
        state={'checkpoint':str(path),'checkpoint_sha256':header['checkpoint_sha256'],'reports':{}}
        result['branches'][name]=state
        if identity in seen:
            donor=seen[identity]
            state.update(reused_from=donor,reports=result['branches'][donor]['reports'],
                retention=result['branches'][donor]['retention'],
                eligible_for_full_range=False,full_range_status='skipped_identical_initial_control')
            write()
            continue
        for group,cases in [('pilot',pilot),('long_hold',long),('local_test',local_test),('known_test',conditions(TEST_CASES))]:
            r=evaluate(path,cases,args.out/f'{name}_{group}_traces.npz')
            state['reports'][group]=r
            print(json.dumps({'branch':name,'group':group,'passed':r['passed'],'total':r['total'],
                'rmse':r['score_deg'],'a2':r['activation_squared_mean'],'failures':r['failures']}),flush=True)
            write()
        state['retention']={}
        for group in ('pilot','long_hold','known_test'):
            before=result['original'][group];after=state['reports'][group]
            mask=[i for i,row in enumerate(before['cases']) if row['passed']]
            eligible,reasons=quality_gate({'cases':[after['cases'][i] for i in mask]},
                {'cases':[before['cases'][i] for i in mask]})
            state['retention'][group]={'eligible':eligible,'reasons':reasons}
        state['eligible_for_full_range']=(all(g['eligible'] for g in state['retention'].values())
            and state['reports']['local_test']['passed']==6 and branch['reduction']>=.15)
        state['full_range_status']='pending' if state['eligible_for_full_range'] else 'skipped_local_economy_or_transfer_gate'
        seen[identity]=name
        write()
    result.update(status='complete',accepted_default=False)
    write()


if __name__=='__main__':main()
