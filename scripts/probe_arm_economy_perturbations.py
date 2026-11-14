"""Frozen economy policies under the same native external force pulses as before."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import evaluate_arm_weakening_perturbations as pulse_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    transfer=json.loads(args.transfer.read_text())
    if transfer['status']!='complete' or not transfer['selection_frozen']:
        raise ValueError('Freeze selections and finish transfer before perturbations')
    baseline_path=Path('runs/arm_weakening_10ms/perturbations/summary.json')
    baseline=json.loads(baseline_path.read_text())['original']
    if baseline['checkpoint_sha256']!=transfer['parent_sha256']:
        raise ValueError('Pulse baseline parent mismatch')
    assert len(baseline['cases'])==8 and all(row['force_N']==1. for row in baseline['cases'])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'pulse_source.py').write_bytes(Path(pulse_module.__file__).read_bytes())
    summary={'status':'running','selection_frozen':True,
        'baseline_reference':str(baseline_path),
        'baseline_reference_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'original':baseline,'branches':{},
        'protocol':'reuse SHA-matched original8 pulse cases; freshly evaluate frozen primary and '
            'precision-matched candidates on55/75/95/125deg,+-1N tangential payload force100ms at2.5s,6s total; '
            'same per-case task/position/speed/recovery/deflection gate; no rescue interventions'}

    def write():
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

    write()
    for name in ('weakening','precision_matched'):
        path=Path(transfer['branches'][name]['checkpoint'])
        result=pulse_module.assess(path,1.,args.out/f'{name}_traces.npz');reasons=[]
        for i,(old,new) in enumerate(zip(baseline['cases'],result['cases'])):
            assert all(new[k]==old[k] for k in ('goal_deg','sign','force_N'))
            if new['failure'] or (old['passed'] and not new['passed']):reasons.append([i,'pass'])
            if new['tail_rmse_deg']>old['tail_rmse_deg']+.25:reasons.append([i,'position'])
            if new['tail_velocity_rms_deg_s']>max(5.,1.1*old['tail_velocity_rms_deg_s']):reasons.append([i,'speed'])
            if old['recovery_time_s'] is not None and (new['recovery_time_s'] is None or new['recovery_time_s']>old['recovery_time_s']+.2+1e-8):
                reasons.append([i,'recovery'])
            if new['max_after_pulse_deflection_from_pre_deg']>old['max_after_pulse_deflection_from_pre_deg']+1:
                reasons.append([i,'deflection'])
        summary['branches'][name]={'result':result,'eligible':not reasons,'reasons':reasons}
        print(json.dumps({'branch':name,'passed':result['passed'],'eligible':not reasons,'reasons':reasons}),flush=True)
        write()
    summary['status']='complete';write()


if __name__=='__main__':main()
