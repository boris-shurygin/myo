"""Frozen-policy rollouts: compare bootstrapped GAE and longer returns, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.compact_protocol import CompactProtocol
from myo.runtime import load_controller


def explained_variance(predicted, target):
    variance = float(np.var(target))
    return float(1 - np.var(target - predicted) / variance) if variance > 1e-12 else None


def correlation(a, b):
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def weights_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def actor_gradient(model, observations, actions, advantages):
    # Gradient only, never optimizer.step. At the collecting policy PPO ratio=1.
    params = [p for name, p in model.policy.named_parameters()
              if name.startswith(('mlp_extractor.policy_net', 'action_net'))]
    model.policy.zero_grad(set_to_none=True)
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for start in range(0, len(observations), 128):
        stop = min(start + 128, len(observations))
        obs = torch.as_tensor(observations[start:stop])
        action = torch.as_tensor(actions[start:stop])
        advantage = torch.as_tensor(normalized[start:stop], dtype=torch.float32)
        dist = model.policy.get_distribution(obs)
        loss = -(advantage * dist.log_prob(action)).sum() / len(observations)
        loss.backward()
    gradient = torch.cat([p.grad.detach().flatten() for p in params]).clone()
    model.policy.zero_grad(set_to_none=True)
    return gradient


def collect(checkpoint):
    physical, _, metadata = load_controller(str(checkpoint))
    env = CompactProtocol(physical)
    model = PPO.load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    before = weights_digest(model.policy)
    torch.manual_seed(0)
    env.next_scenario = 6  # Nominal sweep, raise, lower, no random task selection.
    reports, arrays = [], []
    try:
        for _ in range(3):
            obs, _ = env.reset(seed=0)
            rows = []
            while True:
                with torch.no_grad():
                    inputs = torch.as_tensor(obs[None])
                    dist = model.policy.get_distribution(inputs).distribution
                    action = dist.sample()[0].numpy()
                    mean = dist.mean[0].numpy()
                    std = dist.stddev[0].numpy()
                    value = float(model.policy.predict_values(inputs)[0, 0])
                following, reward, terminated, truncated, info = env.step(action)
                with torch.no_grad():
                    next_value = 0.0 if terminated else float(model.policy.predict_values(torch.as_tensor(following[None]))[0, 0])
                rows.append({'observation': obs.copy(), 'action': action, 'mean': mean, 'std': std,
                             'value': value, 'next_value': next_value, 'reward': reward * .01,
                             'q': info['q'], 'velocity': info['qvel'], 'target': info['requested_target_q'],
                             'terms': info['reward_components']})
                obs = following
                if terminated or truncated:
                    break
            value = np.asarray([r['value'] for r in rows])
            following = np.asarray([r['next_value'] for r in rows])
            reward = np.asarray([r['reward'] for r in rows])
            delta = reward + model.gamma * following - value
            gae, mc = np.zeros(len(rows)), np.zeros(len(rows))
            gae_tail, mc_tail = 0.0, following[-1]
            for index in reversed(range(len(rows))):
                gae_tail = delta[index] + model.gamma * model.gae_lambda * gae_tail
                mc_tail = reward[index] + model.gamma * mc_tail
                gae[index], mc[index] = gae_tail, mc_tail
            action = np.asarray([r['action'] for r in rows])
            mean = np.asarray([r['mean'] for r in rows])
            std = np.asarray([r['std'] for r in rows])
            innovation = (action - mean) / std
            q = np.rad2deg([r['q'] for r in rows])
            target = np.rad2deg([r['target'] for r in rows])
            velocity = np.rad2deg([r['velocity'] for r in rows])
            tail = slice(-min(125, len(rows)), None)
            report = {'scenario': env.scenario.name, 'steps': len(rows),
                'angle_min_deg': float(q.min()), 'angle_max_deg': float(q.max()), 'final_angle_deg': float(q[-1]),
                'angle_error_rmse_deg': float(np.sqrt(np.mean((q-target)**2))),
                'settled_sample_fraction': float(np.mean((np.abs(q-target)<=3)&(np.abs(velocity)<=5))),
                'tail_mean_action': action[tail].mean(0).tolist(), 'tail_mean_policy_mean': mean[tail].mean(0).tolist(),
                'tail_mean_reward_raw': float(reward[tail].mean() / .01),
                'mean_reward_terms_raw': {k:float(np.mean([r['terms'][k] for r in rows])) for k in rows[0]['terms']},
                'critic_ev_vs_gae_target': explained_variance(value, value+gae),
                'critic_ev_vs_bootstrapped_mc': explained_variance(value, mc),
                'critic_bias_vs_bootstrapped_mc_scaled': float((value-mc).mean()),
                'critic_rmse_vs_bootstrapped_mc_scaled': float(np.sqrt(np.mean((value-mc)**2))),
                'value_std_scaled': float(value.std()), 'gae_target_std_scaled': float((value+gae).std()),
                'mc_std_scaled': float(mc.std()), 'td_std_scaled': float(delta.std()),
                'gae_mc_advantage_correlation': correlation(gae, mc-value),
                'innovation_gae_correlation_by_muscle': [correlation(innovation[:,i], gae) for i in range(6)],
                'innovation_mc_correlation_by_muscle': [correlation(innovation[:,i], mc-value) for i in range(6)],
                'gae_std_scaled': float(gae.std()), 'mc_advantage_std_scaled': float((mc-value).std()),
                'median_action_std': float(np.median(std))}
            reports.append(report)
            arrays.append({'observations':np.asarray([r['observation'] for r in rows]), 'actions':action,
                           'values':value,'gae':gae,'mc':mc,'rewards':reward,'next_values':following,
                           'angles_deg':q,'targets_deg':target,'means':mean,'stds':std})
            print(json.dumps({'checkpoint':str(checkpoint),'scenario':env.scenario.name,
                              'ev_gae':report['critic_ev_vs_gae_target'],'ev_mc':report['critic_ev_vs_bootstrapped_mc'],
                              'max_angle':report['angle_max_deg']}),flush=True)
        merged={k:np.concatenate([r[k] for r in arrays]) for k in arrays[0]}
        g_gae=actor_gradient(model,merged['observations'],merged['actions'],merged['gae'])
        g_mc=actor_gradient(model,merged['observations'],merged['actions'],merged['mc']-merged['values'])
        cosine=float(torch.dot(g_gae,g_mc)/(g_gae.norm()*g_mc.norm()+1e-12))
        assert before == weights_digest(model.policy)
        result={'checkpoint':str(checkpoint),'trained_steps':model.num_timesteps,'weights_unchanged':True,
            'gamma':model.gamma,'gae_lambda':model.gae_lambda,'control_dt':physical.control_dt,
            'scenarios':reports,'combined':{
                'ev_gae':explained_variance(merged['values'],merged['values']+merged['gae']),
                'ev_bootstrapped_mc':explained_variance(merged['values'],merged['mc']),
                'gae_mc_advantage_correlation':correlation(merged['gae'],merged['mc']-merged['values']),
                'actor_gradient_cosine_gae_vs_mc':cosine,
                'actor_gradient_norm_gae':float(g_gae.norm()),'actor_gradient_norm_mc':float(g_mc.norm())}}
        return result,merged
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/d1_critic_signal_audit.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    output={'limitations':[
        'No training; only fresh stochastic rollouts of frozen initial/final policies, seed 0.',
        'GAE and n-step Monte Carlo use the same final critic bootstrap at time limits; MC is not an unbiased exact infinite-horizon value.',
        'Episode-wide returns omit the 1472-step rollout boundaries of the training runner.',
        'Actor gradient comparison uses one globally standardized batch, not the full sequence of PPO minibatches/Adam updates.',
        'One trajectory per nominal scenario gives noisy policy-gradient estimates.'], 'models':{}}
    for name in ('small','wide'):
        directory=Path('runs')/f'd1_architecture_{name}_tanh'
        for stage,file in [('initial','initial.zip'),('final','block_004.zip')]:
            result,arrays=collect(directory/file)
            label=f'{name}_{stage}'
            np.savez_compressed(args.output.with_name(args.output.stem+'_'+label+'.npz'),**arrays)
            output['models'][label]=result
            args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
