"""Independent CPU evaluation workers; never touch PPO rollout state or RNG."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .arm_workload import load_workload, evaluate_workload
from .arm_reward_curriculum import load_reward_curriculum


def evaluation_job(path, conditions):
    torch.set_num_threads(1)
    meta=json.loads(Path(str(path)+'.metadata.json').read_text())
    loader=load_reward_curriculum if meta['contract']['schema']=='myo.arm_reward_curriculum.v1' else load_workload
    model,env,_=loader(path)
    env.close()
    return evaluate_workload(model,conditions)


def merge_results(results):
    rows=[row for result in results for row in result['cases']]
    groups={}
    for field in ('payload_mass','tilt_deg','capacity'):
        groups[field]={str(value):{'score_deg':float(np.mean([r['tail_rmse_deg'] for r in rows if r[field]==value])),
            'passed':sum(r['passed'] for r in rows if r[field]==value),
            'total':sum(r[field]==value for r in rows)} for value in sorted({r[field] for r in rows})}
    return {'cases':rows,'score_deg':float(np.mean([r['tail_rmse_deg'] for r in rows])),
        'passed':sum(r['passed'] for r in rows),'total':len(rows),
        'failures':sum(r['failure'] for r in rows),'groups':groups,
        'activation_squared_mean':float(np.mean([r['activation_squared_mean'] for r in rows])),
        'active_fraction_mean':float(np.mean([r['active_fraction_mean'] for r in rows]))}


def evaluate_parallel(pool,path,conditions,workers):
    chunks=[list(chunk) for chunk in np.array_split(np.arange(len(conditions)),min(workers,len(conditions)))]
    futures=[pool.submit(evaluation_job,path,[conditions[i] for i in chunk]) for chunk in chunks]
    return merge_results([future.result() for future in futures])
