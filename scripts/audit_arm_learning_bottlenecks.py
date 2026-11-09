"""Read-only paired interventions, stochastic execution and goal-use diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.arm_learning import load_arm
from myo.arm_conditioning import load_conditioned_arm
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import load_policy
from myo.vertical_precision import VerticalPrecisionEnv
from myo.vertical_muscles import VerticalConfig


def policy_values(model, obs):
    with torch.no_grad():
        x = torch.as_tensor(obs[None], dtype=torch.float32)
        d = model.policy.get_distribution(x).distribution
        return d.mean[0].numpy().copy(), float(model.policy.predict_values(x)[0, 0])


def arm_episode(model, env, case, *, seed=0, stochastic=False, intervention='native', pulse=0,
                return_trace=False):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=0, options={'case': case})
    rows, parts, activations = [], [], []
    initial_value = policy_values(model, obs)[1]
    while True:
        action, value = policy_values(model, obs)
        if stochastic:
            action = model.predict(obs, deterministic=False)[0]
        index = len(rows)
        # A paired intervention after one second; no state restoration or physics edits.
        if 50 <= index < 50+pulse:
            if intervention in ('flex_plus', 'reciprocal'):
                action[:3] = np.minimum(action[:3]+.1, 1.)
            if intervention in ('ext_minus', 'reciprocal'):
                action[3:] = np.maximum(action[3:]-.05, 0.)
        if intervention == 'heuristic':
            action = env.unwrapped.baseline_action()
        obs, reward, failed, truncated, info = env.step(action)
        assert info['trigger'] == 1. and info['reward_components']['trigger_excess_cost'] == 0.
        activations.append(info['activation'])
        next_value = 0. if failed else policy_values(model, obs)[1]
        rows.append([float(np.rad2deg(info['q'])), float(np.rad2deg(info['qvel'])),
                     reward, value, next_value, *action])
        parts.append(info['reward_components'])
        if failed or truncated:
            break
    a = np.asarray(rows)
    tail = a[-50:]
    reward = a[:, 2]
    discount = model.gamma**np.arange(len(a))
    start = 50
    delta = a[start:, 2]+model.gamma*a[start:, 4]-a[start:, 3]
    w = (model.gamma*model.gae_lambda)**np.arange(len(delta))
    if len(delta):
        discounted_delta = model.gamma**np.arange(len(delta))@delta
        telescoping = (model.gamma**np.arange(len(delta))@a[start:, 2]
                       -a[start, 3]+model.gamma**len(delta)*a[-1, 4])
        assert np.isclose(discounted_delta, telescoping, atol=1e-10, rtol=1e-10)
    # This is a frozen-current-critic probe, not a replay of the historical training GAE.
    result = {'case': list(case), 'stochastic': stochastic, 'seed': seed,
        'intervention': intervention, 'pulse_steps': pulse, 'failure': bool(failed),
        'steps': len(a), 'initial_value': initial_value, 'return': float(reward.sum()),
        'discounted_return_no_bootstrap': float(discount@reward),
        'tail_rmse_deg': float(np.sqrt(np.mean((tail[:, 0]-case[1])**2))),
        'tail_mean_deg': float(tail[:, 0].mean()), 'max_deg': float(a[:, 0].max()),
        'passed': bool(not failed and len(a) == 200 and np.mean(
            (abs(tail[:, 0]-case[1]) <= 3.) & (abs(tail[:, 1]) <= 5.)) >= .9),
        'reward_components_sum_unscaled': {k: float(sum(p.get(k, 0.) for p in parts))
                                           for k in parts[0]},
        'mean_command': a[:, 5:].mean(0).tolist(),
        'paired_probe': {'state_at_1s': a[49, :2].tolist() if len(a) >= 50 else None,
            'discounted_return_from_1s_no_bootstrap': float(model.gamma**np.arange(len(delta))@a[start:, 2]),
            'gae_at_1s': float(w@delta),
            'first_200ms_reward_after_1s': float(a[start:start+10, 2].sum()),
            'angle_after_200ms_deg': float(a[min(59, len(a)-1), 0])}}
    if return_trace:
        result['trace'] = a.tolist()
        result['activation_trace'] = activations
    return result


def goal_grid(model, env):
    rows = []
    for initial in (20, 75, 125):
        for target in (15, 35, 55, 75, 95, 115, 135):
            obs, _ = env.reset(seed=0, options={'case': (initial, target)})
            action, value = policy_values(model, obs)
            rows.append({'initial_deg': initial, 'goal_deg': target,
                         'mean_command': action.tolist(), 'value': value})
    return rows


def vertical_initial(model, env, seed, stochastic):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=0, options={'initial_position_m': -.06, 'target_m': .08})
    positions, rewards = [], []
    while True:
        action = model.predict(obs, deterministic=not stochastic)[0]
        obs, reward, failed, truncated, info = env.step(action)
        positions.append(info['position_m']); rewards.append(reward)
        if failed or truncated:
            break
    return {'seed': seed, 'stochastic': stochastic, 'failure': bool(failed),
            'max_m': max(positions), 'last_m': positions[-1], 'return': sum(rewards)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/arm_bottlenecks/audit.json'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    specs = {'none_initial': ('runs/arm_preacher/none/initial.zip', load_arm),
             'none_best': ('runs/arm_preacher/none/best.zip', load_arm),
             'gain025_best': ('runs/arm_conditioning/gain_025/best.zip', load_conditioned_arm)}
    report = {'schema': 'myo.arm_bottlenecks.v1', 'models': {}, 'paired': [],
              'execution': [], 'goals': {}, 'vertical_initial': [],
              'scope': 'read-only diagnostics; no optimizer steps, no checkpoint selection'}
    for label, (path, loader) in specs.items():
        model, env, meta = loader(Path(path))
        before = parameter_hash(model)
        report['models'][label] = {'path': path, 'sha256': meta['checkpoint_sha256'],
                                  'contract': meta['contract']}
        report['goals'][label] = goal_grid(model, env)
        for case in ((20, 75), (75, 20), (25, 105)):
            for stochastic, seed in ((False, 0), *((True, s) for s in range(4))):
                row = arm_episode(model, env, case, seed=seed, stochastic=stochastic)
                report['execution'].append({'model': label, **row})
            print(json.dumps({'model': label, 'execution_case': case}), flush=True)
        for intervention, pulse in (('native', 0), ('heuristic', 0),
                *((name, n) for n in (1, 10) for name in ('flex_plus', 'ext_minus', 'reciprocal'))):
            row = arm_episode(model, env, (20, 75), intervention=intervention, pulse=pulse)
            report['paired'].append({'model': label, **row})
        paired = [r for r in report['paired'] if r['model'] == label and r['intervention'] != 'heuristic']
        assert all(r['paired_probe']['state_at_1s'] == paired[0]['paired_probe']['state_at_1s'] for r in paired)
        assert parameter_hash(model) == before
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == meta['checkpoint_sha256']
        env.close()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'completed': label, 'elapsed_s': time.monotonic()-started}), flush=True)
    path = Path('runs/vertical_precision/combined/initial.zip')
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = VerticalPrecisionEnv(VerticalConfig(**meta['contract']['physical']['config']))
    model, _ = load_policy(path, env)
    before = parameter_hash(model)
    for stochastic, seed in ((False, 0), *((True, s) for s in range(4))):
        report['vertical_initial'].append(vertical_initial(model, env, seed, stochastic))
    assert parameter_hash(model) == before
    env.close()
    report.update(status='complete', elapsed_s=time.monotonic()-started, unchanged_weights=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'elapsed_s': report['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
