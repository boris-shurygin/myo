"""Evaluate both predictors on fresh stochastic states of the new selected PPO."""
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_balanced import ROOT, OLD
from myo.arm_learning import load_arm, predictor_identity
from myo.arm_precision import TRAIN_CASES
from myo.train_arm_precision import forecast
from compare_arm_movement_coverage import statistics


def main():
    torch.set_num_threads(1)
    if json.loads((ROOT/'frozen/summary.json').read_text())['status'] != 'complete':
        raise ValueError('Fix controller selection and test first')
    path = ROOT/'selected_policy_probe.npz'
    if path.exists(): raise FileExistsError(path)
    model, env, identity = load_arm(ROOT/'frozen/best.zip')
    torch.manual_seed(20261110)
    data = {key: [] for key in ('observations','actions','next_observations','valid')}
    episodes = []
    for case in TRAIN_CASES:
        obs, _ = env.reset(seed=0, options={'case':case})
        start = len(data['valid'])
        for _ in range(200):
            action = model.predict(obs,deterministic=False)[0]
            following, _, failed, truncated, _ = env.step(action)
            for key,value in zip(data,(obs,action,following,not failed),strict=True): data[key].append(value)
            obs = following
            if failed or truncated: break
        episodes.append({'case':case,'split':'probe','start':start,'stop':len(data['valid']),'failure':bool(failed)})
    meta = {'contract':env.contract(),'checkpoint':identity,'episodes':episodes,
            'scope':'new stochastic rollout of selected policy, no fitting or checkpoint selection'}
    env.close(); data = {k:np.asarray(v) for k,v in data.items()}
    np.savez_compressed(path,**data)
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    report = {'statistics':statistics(data,meta,'probe'),'prediction':{}}
    for name,path in [('old',OLD/'body/best.pt'),('new',ROOT/'body/best.pt')]:
        body,_=predictor_identity(path)
        report['prediction'][name]={str(20*h):forecast(body,data,meta,'probe',h) for h in (1,3)}
    (ROOT/'selected_policy_probe_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['prediction']),flush=True)


if __name__ == '__main__': main()
