"""Measure exact policy KL after accepted optimizer steps, without rollback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from myo.arm_learning import load_arm
from myo.arm_conditioning import load_conditioned_arm
from myo.arm_precision import TRAIN_CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/arm_hypotheses/update_drift.json'))
    p.add_argument('--model', choices=('none_131072', 'none_139264', 'gain025_best', 'none_low_lr'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    torch.set_num_threads(1)
    specs = {
        'none_131072': ('runs/arm_preacher/none/best.zip', load_arm),
        'none_139264': ('runs/arm_hypotheses/snapshots/none_139264.zip', load_arm),
        'gain025_best': ('runs/arm_conditioning/gain_025/best.zip', load_conditioned_arm)}
    if args.model == 'none_low_lr':
        specs = {'none_low_lr': ('runs/arm_hypotheses/lr_1e4_none/initial.zip', load_arm)}
    elif args.model:
        specs = {args.model: specs[args.model]}
    results = {}
    for label, (path, loader) in specs.items():
        path = Path(path)
        model, env, meta = loader(path)
        model.set_env(Monitor(env)); model.set_random_seed(0); model.last_audit = {}
        grid = np.stack([env.reset(seed=0, options={'case': case})[0] for case in TRAIN_CASES])
        grid = torch.as_tensor(grid)
        records, baseline = [], {}
        original = torch.nn.utils.clip_grad_norm_

        @torch.no_grad()
        def frozen_distribution(observations):
            d = model.policy.get_distribution(observations).distribution
            return torch.distributions.Beta(d.concentration1.clone(), d.concentration0.clone())

        def clip_capture(parameters, *a, **kw):
            if not baseline:
                obs = model.rollout_buffer.observations
                x = torch.as_tensor(obs.reshape(-1, obs.shape[-1]))
                baseline.update(x=x, rollout=frozen_distribution(x), grid=frozen_distribution(grid))
                goal_index = env.observation_names.index('requested_target_centered_rad')
                baseline['observed_goals'] = np.unique(np.round(np.rad2deg(
                    obs.reshape(-1, obs.shape[-1])[:, goal_index]*np.pi/2+np.deg2rad(75)), 3)).tolist()
            return original(parameters, *a, **kw)

        @torch.no_grad()
        def after_step(*_):
            current = frozen_distribution(baseline['x'])
            query = frozen_distribution(grid)
            rollout_kl = torch.distributions.kl_divergence(baseline['rollout'], current).sum(-1)
            grid_kl = torch.distributions.kl_divergence(baseline['grid'], query).sum(-1)
            mean_changes = (query.mean-baseline['grid'].mean).abs()
            records.append({'step': len(records)+1,
                'rollout_mean_exact_kl': float(rollout_kl.mean().clamp_min(0)),
                'rollout_max_exact_kl': float(rollout_kl.max().clamp_min(0)),
                'grid_mean_exact_kl': float(grid_kl.mean().clamp_min(0)),
                'grid_max_command_change': float(mean_changes.max()),
                'grid_mean_command_change': float(mean_changes.mean())})

        hook = model.policy.optimizer.register_step_post_hook(after_step)
        try:
            torch.nn.utils.clip_grad_norm_ = clip_capture
            model.learn(1024, reset_num_timesteps=False)
        finally:
            torch.nn.utils.clip_grad_norm_ = original
            hook.remove()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['checkpoint_sha256']
        results[label] = {'checkpoint_sha256': meta['checkpoint_sha256'],
            'rollout_steps': 1024, 'observed_goals_deg': baseline['observed_goals'],
            'grid_cases': TRAIN_CASES, 'target_kl': model.target_kl,
            'stop_guard_threshold': 1.5*model.target_kl, 'steps': records,
            'first_step': records[0], 'last_step': records[-1],
            'max_accepted_rollout_kl': max(r['rollout_mean_exact_kl'] for r in records)}
        env.close()
        print(json.dumps({'model': label, 'observed_goals': baseline['observed_goals'],
                          'first': records[0], 'last': records[-1]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'complete', 'results': results,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
