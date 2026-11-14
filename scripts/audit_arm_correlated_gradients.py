"""Inspect gradient geometry on a real rollout without changing any weights."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from myo.arm_correlated_relaxation import CorrelatedDistribution, load_correlated
from myo.arm_learning import ArmPPO
from myo.arm_relaxation import relaxed_mean
from train_arm_correlated import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    model, env, meta = load_correlated(args.checkpoint)
    vector = DummyVecEnv([partial(worker, env.workload, env.exploration, env.timing,
        env.efficiency_reward, env.correlated, i) for i in range(4)])
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    try:
        # Initial checkpoint was built with one checking env; loading with an
        # explicit vector also rebuilds its rollout buffer for four workers.
        model = ArmPPO.load(args.checkpoint, env=vector, device='cpu')
        set_random_seed(0)
        vector.seed(0)
        model.set_logger(configure(str(args.out), []))
        _, callback = model._setup_learn(1024, reset_num_timesteps=False)
        model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=256)
        buffer = model.rollout_buffer
        observation = torch.as_tensor(buffer.observations.reshape(-1, 156))
        action = torch.as_tensor(buffer.actions.reshape(-1, 6))
        advantage = torch.as_tensor(buffer.advantages.reshape(-1))
        with torch.no_grad():
            d = model.policy.get_distribution(observation)
            initial_mean = d.nominal_mean.clone()
        # Analyze a reconstructed full-batch first PPO step (ratio1), not the
        # actual sequence of128-sample minibatch updates in the experiment.
        ell = torch.logit(initial_mean).requires_grad_(True)
        mean = ell.sigmoid()
        gains = observation[:, -6:]
        d = CorrelatedDistribution(6).configure(mean, 1-mean, gains, env.correlated)
        normalized = (advantage-advantage.mean())/(advantage.std()+1e-8)
        policy_loss = -(normalized*d.log_prob(action)).mean()
        gradient = torch.autograd.grad(policy_loss, ell, retain_graph=True)[0]
        entropy_gradient = torch.autograd.grad(-model.ent_coef*d.entropy().mean(), ell)[0]
        h = functional.normalize(gains*mean.detach()*(1-mean.detach()), dim=-1)
        direction = torch.logit(initial_mean)-torch.logit(relaxed_mean(initial_mean, gains, env.correlated.fraction))
        direction = functional.normalize(direction-(direction*h).sum(-1, keepdim=True)*h, dim=-1)

        def geometry(g):
            moment = (g*h).sum(-1)
            common = (g*direction).sum(-1)
            rest = g-moment[:, None]*h-common[:, None]*direction
            return {'total_norm': float(g.norm()), 'moment_projection_norm': float(moment.norm()),
                'common_projection_norm': float(common.norm()), 'other_projection_norm': float(rest.norm()),
                'moment_to_common_ratio': float(moment.norm()/common.norm().clamp_min(1e-12))}

        result = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': meta['checkpoint_sha256'],
            'distribution': asdict(env.correlated), 'seed': 0, 'transitions': 1024,
            'protocol': 'one native rollout, no optimizer steps; reconstructed globally normalized '
                'first PPO policy gradient at ratio1; projections in per-state logit-mean coordinates; '
                'not gradient norms of actual network weights or natural-gradient metric',
            'advantage_std': float(advantage.std()), 'advantage_mean': float(advantage.mean()),
            'return_minus_value_rms': float(np.sqrt(np.mean((buffer.returns-buffer.values)**2))),
            'policy_gradient': geometry(gradient), 'entropy_gradient_with_coefficient': geometry(entropy_gradient),
            'weights_unchanged': all(torch.equal(before[k], v) for k,v in model.policy.state_dict().items())}
        assert result['weights_unchanged']
        (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        np.savez_compressed(args.out/'rollout.npz', observation=observation.numpy(), action=action.numpy(),
            advantage=advantage.numpy(), returns=buffer.returns, values=buffer.values,
            policy_logit_gradient=gradient.numpy(), entropy_logit_gradient=entropy_gradient.numpy())
        print(json.dumps(result), flush=True)
    finally:
        env.close()
        vector.close()


if __name__ == '__main__':
    main()
