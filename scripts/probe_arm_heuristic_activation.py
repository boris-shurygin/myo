"""Measure transient and steady heuristic activation in known policy failures."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from myo.arm_workload import evaluate_workload
from myo.train_vertical_ppo import write_json


def job(conditions):
    torch.set_num_threads(1)
    return evaluate_workload('heuristic',conditions,samples=True)['cases']


def main():
    torch.set_num_threads(1)
    root=Path('runs/arm_reward_curriculum')
    sources=[(Path('runs/arm_workload')/name/'summary.json',('best_validation','best_test'),True)
             for name in ('hold_joint','hold_joint_corrective')]
    sources.append((root/'position_continuation/assessment.json',('train','validation','test'),False))
    cells={};provenance=[]
    for path,splits,heavy_only in sources:
        source=json.loads(path.read_text())
        provenance.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'splits':splits,'only_mass2kg':heavy_only})
        for split in splits:
            for row in source[split]['cases']:
                if row['passed'] or (heavy_only and row['payload_mass']!=2.):continue
                condition={k:row[k] for k in ('case','payload_mass','tilt_deg','capacity','seconds')}
                key=json.dumps(condition,sort_keys=True)
                cells.setdefault(key,{'condition':condition,'failed_sources':[]})['failed_sources'].append(str(path)+':'+split)
    conditions=[c['condition'] for c in cells.values()]
    report={'schema':'myo.arm_heuristic_activation_probe.v1','status':'running','sources':provenance,
        'scope':'all current local policy failures plus historical E8/E10 failures with2kg; native cold starts',
        'sampling':'20ms activation endpoints; peak is not excitation and may miss substep extrema',
        'muscles':['BIClong','BICshort','BRA','TRIlong','TRIlat','TRImed'],'cases':[]}
    print('conditions',len(conditions),flush=True)
    chunks=[conditions[i::2] for i in range(2)]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(job,chunk) for chunk in chunks]
        for future in futures:
            for row in future.result():
                trace=np.asarray(row['samples']);tail=trace[:,0]>row['seconds']-1.-1e-8
                a=trace[:,10:16]
                condition={k:row[k] for k in ('case','payload_mass','tilt_deg','capacity','seconds')}
                row.update(peak_activation_by_muscle=a.max(0).tolist(),
                    tail_max_activation_by_muscle=a[tail].max(0).tolist(),
                    tail_time_fraction_above90pct=(a[tail]>=.9).mean(0).tolist(),
                    failed_sources=cells[json.dumps(condition,sort_keys=True)]['failed_sources'])
                report['cases'].append(row)
            write_json(root/'heuristic_activation.json',report)
    report.update(status='complete',passed=sum(r['passed'] for r in report['cases']),
                  total=len(report['cases']),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'heuristic_activation.json',report)
    for field in ('peak_activation_by_muscle','tail_activation_by_muscle','tail_max_activation_by_muscle'):
        print(field,[float(max(r[field][i] for r in report['cases'])) for i in range(6)],flush=True)
    print('pass',report['passed'],report['total'],flush=True)


if __name__=='__main__':
    main()
