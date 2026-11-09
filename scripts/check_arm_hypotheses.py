"""Isolated diagnostics of noise, gradient clipping and critic calibration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from myo.arm_learning import load_arm
from myo.arm_conditioning import load_conditioned_arm
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import load_policy
from myo.vertical_precision import VerticalPrecisionEnv
from myo.vertical_muscles import VerticalConfig


ARM_MODELS = {
    'none_best': ('runs/arm_preacher/none/best.zip', load_arm),
    'gain025_best': ('runs/arm_conditioning/gain_025/best.zip', load_conditioned_arm),
}


def moments(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'max': float(a.max())}


@torch.no_grad()
def distribution_value(model, obs):
    x = torch.as_tensor(obs[None], dtype=torch.float32)
    d = model.policy.get_distribution(x).distribution
    return d, float(model.policy.predict_values(x)[0, 0])


def load_vertical():
    path = Path('runs/vertical_precision/combined/best.zip')
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = VerticalPrecisionEnv(VerticalConfig(**meta['contract']['physical']['config']))
    model, meta = load_policy(path, env)
    return model, env, meta


def gradient_probe():
    specs = {'none_initial': ('runs/arm_preacher/none/initial.zip', load_arm),
             **ARM_MODELS,
             'raw_frozen_best': ('runs/arm_preacher_balanced/frozen/best.zip', load_arm),
             'vertical_best': (None, None)}
    results = {}
    for label, (path, loader) in specs.items():
        model, env, meta = load_vertical() if label == 'vertical_best' else loader(Path(path))
        source_hash = meta['checkpoint_sha256']
        before = parameter_hash(model)
        model.set_env(Monitor(env))
        model.set_random_seed(0)
        records = []
        named = dict(model.policy.named_parameters())
        original = torch.nn.utils.clip_grad_norm_

        def capture(parameters, max_norm, *args, **kwargs):
            squared = {'actor': 0., 'critic': 0., 'other': 0.}
            for name, parameter in named.items():
                if parameter.grad is None:
                    continue
                group = ('actor' if name.startswith(('action_net.', 'mlp_extractor.actor_private.',
                                                      'mlp_extractor.policy_net.'))
                         else 'critic' if name.startswith(('value_net.', 'mlp_extractor.critic_private.',
                                                          'mlp_extractor.value_net.')) else 'other')
                squared[group] += float(parameter.grad.double().square().sum())
            total = sum(squared.values())**.5
            assert squared['other'] == 0.
            row = {key+'_norm': val**.5 for key, val in squared.items()}
            row.update(total_norm=total, clip_factor=min(1., float(max_norm)/(total+1e-6)),
                       critic_fraction_squared=squared['critic']/max(total**2, 1e-30))
            records.append(row)
            return original(parameters, max_norm, *args, **kwargs)

        try:
            torch.nn.utils.clip_grad_norm_ = capture
            # One genuine on-policy rollout/update; only this in-memory copy changes.
            model.learn(1024, reset_num_timesteps=False)
        finally:
            torch.nn.utils.clip_grad_norm_ = original
        assert records and parameter_hash(model) != before
        if path:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == source_hash
        results[label] = {'checkpoint_sha256': source_hash, 'rollout_steps': 1024,
            'optimizer_steps': len(records), 'gradients': records,
            'summary': {k: moments([r[k] for r in records]) for k in records[0]},
            'metrics': {k: float(v) for k, v in model.logger.name_to_value.items()
                        if k.startswith('train/')},
            'in_memory_weights_changed': True, 'saved_checkpoint_unchanged': True}
        env.close()
        print(json.dumps({'probe': 'gradient', 'model': label,
                          'summary': results[label]['summary']}), flush=True)
    return results


def concentration_probe():
    results = []
    for label, (path, loader) in ARM_MODELS.items():
        model, env, meta = loader(Path(path))
        before = parameter_hash(model)
        for case in ((20, 75), (75, 20), (25, 105)):
            for concentration in (4., 16., 64.):
                for seed in (0, 1):
                    torch.manual_seed(seed)
                    obs, _ = env.reset(seed=0, options={'case': case})
                    trace, stds = [], []
                    while True:
                        d, _ = distribution_value(model, obs)
                        narrowed = torch.distributions.Beta(d.concentration1*concentration,
                                                           d.concentration0*concentration)
                        torch.testing.assert_close(d.mean, narrowed.mean, rtol=1e-6, atol=1e-7)
                        action = narrowed.sample()[0].numpy()
                        stds.append(narrowed.stddev[0].numpy().copy())
                        obs, reward, failed, truncated, info = env.step(action)
                        assert info['trigger'] == 1.
                        trace.append([float(np.rad2deg(info['q'])), float(np.rad2deg(info['qvel'])),
                                      reward, *action, *info['activation']])
                        if failed or truncated:
                            break
                    a = np.asarray(trace); tail = a[-50:]
                    row = {'model': label, 'case': list(case), 'concentration_multiplier': concentration,
                        'seed': seed, 'failure': bool(failed), 'checkpoint_sha256': meta['checkpoint_sha256'],
                        'tail_rmse_deg': float(np.sqrt(np.mean((tail[:, 0]-case[1])**2))),
                        'tail_mean_deg': float(tail[:, 0].mean()), 'return': float(a[:, 2].sum()),
                        'tail_mean_std': float(np.mean(np.asarray(stds)[-50:])),
                        'tail_mean_command': tail[:, 3:9].mean(0).tolist(),
                        'tail_mean_activation': tail[:, 9:].mean(0).tolist(),
                        'passed': bool(not failed and len(a) == 200 and np.mean(
                            (abs(tail[:, 0]-case[1]) <= 3.) & (abs(tail[:, 1]) <= 5.)) >= .9)}
                    results.append(row)
                print(json.dumps({'probe': 'concentration', 'model': label, 'case': case,
                                  'multiplier': concentration,
                                  'rmse': [r['tail_rmse_deg'] for r in results[-2:]]}), flush=True)
        assert parameter_hash(model) == before
        env.close()
    return results


def critic_probe():
    results = []
    horizon, samples = 600, 8  # 12 seconds, independent stochastic continuations.
    for label, (path, loader) in ARM_MODELS.items():
        model, env, meta = loader(Path(path))
        before = parameter_hash(model)
        for prefix in (0, 50):
            torch.manual_seed(3)
            obs, _ = env.reset(seed=0, options={'case': (20, 75)})
            for _ in range(prefix):
                action = model.predict(obs, deterministic=False)[0]
                obs, _, failed, truncated, _ = env.step(action)
                assert not failed and not truncated
            _, value = distribution_value(model, obs)
            # Verify the full deepcopy (MuJoCo, muscles, delay, command and history).
            left, right = copy.deepcopy(env), copy.deepcopy(env)
            for action in (np.full(6, .2), np.zeros(6), np.full(6, .1)):
                lo, lr, ld, lt, _ = left.step(action)
                ro, rr, rd, rt, _ = right.step(action)
                assert np.array_equal(lo, ro) and (lr, ld, lt) == (rr, rd, rt)
            left.close(); right.close()
            branches = []
            for seed in range(samples):
                branch = copy.deepcopy(env)
                state = branch.observation()
                assert np.array_equal(state, obs)
                torch.manual_seed(1000+seed)
                rewards, deltas = [], []
                for _ in range(horizon):
                    _, v = distribution_value(model, state)
                    action = model.predict(state, deterministic=False)[0]
                    state, reward, failed, truncated, info = branch.step(action)
                    _, nv = distribution_value(model, state)
                    if failed:
                        nv = 0.
                    assert reward <= .1+1e-12 and info['trigger'] == 1.
                    rewards.append(reward); deltas.append(reward+model.gamma*nv-v)
                    if failed:
                        break
                    if truncated:
                        # Same physical episode and fixed goal continue; no state rescue.
                        branch.unwrapped._episode_done = False
                n = len(rewards)
                g = float(model.gamma**np.arange(n)@np.asarray(rewards))
                gae = float((model.gamma*model.gae_lambda)**np.arange(n)@np.asarray(deltas))
                branches.append({'seed': 1000+seed, 'discounted_return_12s': g,
                                 'gae': gae, 'failure': bool(failed), 'steps': n,
                                 'discounted_terminal_value': float(model.gamma**n*nv)})
                branch.close()
            gs = np.array([r['discounted_return_12s'] for r in branches])
            av = np.array([r['gae'] for r in branches])
            row = {'model': label, 'checkpoint_sha256': meta['checkpoint_sha256'],
                   'case': [20, 75], 'stochastic_prefix_steps': prefix, 'value': value,
                   'angle_deg': float(np.rad2deg(env.unwrapped.data.qpos[env.unwrapped.qpos_index])),
                   'mc_12s_mean': float(gs.mean()), 'mc_12s_se': float(gs.std(ddof=1)/samples**.5),
                   'gae_mean': float(av.mean()), 'gae_se': float(av.std(ddof=1)/samples**.5),
                   'bias_vs_finite_return': float(value-gs.mean()),
                   'upper_bound_missing_return': float(model.gamma**horizon*.1/(1-model.gamma)),
                   'branches': branches}
            results.append(row)
            print(json.dumps({'probe': 'critic', **{k: v for k, v in row.items() if k != 'branches'}}), flush=True)
        assert parameter_hash(model) == before
        env.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('probe', choices=('gradient', 'concentration', 'critic'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or Path('runs/arm_hypotheses')/(args.probe+'.json')
    if output.exists():
        p.error('Output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    results = {'gradient': gradient_probe, 'concentration': concentration_probe, 'critic': critic_probe}[args.probe]()
    result = {'schema': 'myo.arm_hypotheses.v1', 'probe': args.probe, 'results': results,
              'status': 'complete', 'elapsed_s': time.monotonic()-started,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
