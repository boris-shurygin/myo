"""Finish reports after training; reuse only identical deterministic executions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from myo.arm_curriculum import conditions
from myo.arm_efficiency import evaluate_job, load_parent
from myo.arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES


def execution_identity(path):
    model,env,meta = load_parent(path)
    try:
        contract = meta['contract']
        # Reward/spec do not affect deterministic actions or physical reset;
        # every evaluation condition supplies all variable reset fields.
        physical = {key:contract[key] for key in ('native','timing','observation_names',
            'observation_scale','observation_offset','action_mapping','exploration')}
        with zipfile.ZipFile(path) as archive:
            weights = hashlib.sha256(archive.read('policy.pth')).hexdigest()
        return {'weights_sha256':weights,'execution_contract':physical,
            'policy_class':type(model.policy).__module__+'.'+type(model.policy).__qualname__}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.out
    summary = json.loads((root/'summary.json').read_text())
    if len(summary['stages']) != 3 or any(s['status'] != 'complete' for s in summary['stages']):
        raise ValueError('All three training stages must have finished before finalization')
    backup = root/'summary_before_finalization.json'
    if not backup.exists():
        backup.write_bytes((root/'summary.json').read_bytes())
    started = time.monotonic()
    paths = {'original':Path(summary['protocol']['parent']),
        **{s['name']:root/s['name']/'best.zip' for s in summary['stages']}}
    identities = {label:execution_identity(path) for label,path in paths.items()}
    reports = json.loads((root/'reports.json').read_text()) if (root/'reports.json').exists() else {}
    manifest = {'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'identities':identities,'reused':{},'evaluated':[],'initial_report_labels':list(reports)}
    manifest['equal_original_pilot_traces'] = {}
    with np.load(root/'original_pilot_traces.npz') as original:
        for label in ('stability','control','effort'):
            if identities[label] != identities['original']:
                continue
            with np.load(root/label/'selected_pilot_traces.npz') as other:
                equal = original.files == other.files and all(
                    np.array_equal(original[key],other[key]) for key in original.files)
            if not equal:
                raise ValueError('Equal policy/contracts produced different pilot physical traces')
            manifest['equal_original_pilot_traces'][label] = True
    (root/'finalizer_source.py').write_bytes(Path(__file__).read_bytes())
    pilot = summary['protocol']['train']+summary['protocol']['validation']+summary['protocol']['retention']
    groups = {'test':summary['protocol']['test'],
        'full_range':conditions(TRAIN_CASES)+conditions(VALIDATION_CASES),
        'long_holds':[{**row,'seconds':20.} for row in conditions(((55.,55.),(75.,75.),(95.,95.)))],
        'known_test':conditions(TEST_CASES)}
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        for label,path in paths.items():
            if label in reports:
                continue
            donor = next((other for other in reports if identities[other] == identities[label]),None)
            if donor is not None:
                reports[label] = reports[donor]
                manifest['reused'][label] = donor
            else:
                reports[label] = {name:pool.submit(evaluate_job,path,rows,root/f'{label}_{name}_traces.npz').result()
                    for name,rows in groups.items()}
                manifest['evaluated'].append(label)
            (root/'reports.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(json.dumps({'final':label,'reused':donor}),flush=True)
    baseline = summary['original_pilot']
    control,effort = summary['stages'][1]['selected'],summary['stages'][2]['selected']
    retention = {}
    for group in ('full_range','known_test'):
        old,new = reports['original'][group],reports['effort'][group]
        retention[group] = all(not previous['passed'] or (row['passed'] and not row['failure'])
            for previous,row in zip(old['cases'],new['cases'])) and new['failures'] == 0
    summary.update(status='complete',reports=reports,finalization=manifest,
        effort_reduction_vs_original=1-effort['activation_squared_mean']/baseline['activation_squared_mean'],
        effort_reduction_vs_control=1-effort['activation_squared_mean']/control['activation_squared_mean'],
        full_range_retained=retention['full_range'],known_test_retained=retention['known_test'])
    summary['success'] = bool(summary['stages'][2]['best_eligible'] and all(retention.values())
        and all(reports['effort'][name]['passed'] == reports['effort'][name]['total']
                and reports['effort'][name]['failures'] == 0 for name in ('test','long_holds'))
        and effort['activation_squared_mean'] <= .85*baseline['activation_squared_mean']
        and effort['activation_squared_mean'] <= .95*control['activation_squared_mean'])
    manifest['elapsed_s'] = time.monotonic()-started
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'finalization.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'complete','success':summary['success'],'retention':retention,
        'effort_reduction_vs_original':summary['effort_reduction_vs_original'],
        'effort_reduction_vs_control':summary['effort_reduction_vs_control']}),flush=True)


if __name__ == '__main__':
    main()
