"""Read-only episode and PPO attribution records for arm continuation."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class ConditionMonitor(Monitor):
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.condition = {'case': list(self.env.case), 'tilt_deg': self.env.tilt_deg,
            'payload_mass': self.unwrapped.payload_mass,
            'capacity': float(np.mean(info['capacity'])),
            'seconds': self.unwrapped.config.episode_seconds}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['training_condition'] = self.condition
        return observation, reward, terminated, truncated, info


def episode_result(rows):
    condition = rows[0]['condition']
    error = np.array([r['q_deg']-r['target_deg'] for r in rows])
    velocity = np.array([r['velocity_deg_s'] for r in rows])
    tail = np.array([r['t'] > condition['seconds']-1.-1e-8 for r in rows])
    if not tail.any():
        tail[:] = True
    settled = (abs(error) <= 3.) & (abs(velocity) <= 5.)
    full = rows[-1]['done'] and not rows[-1]['failure'] and len(rows) == round(condition['seconds']/.02)
    run = longest = 0
    for good in settled:
        run = run+1 if good else 0
        longest = max(longest, run)
    return {'episode': rows[0]['episode'], 'worker': rows[0]['worker'], 'condition': condition,
        'start_transition': rows[0]['transition'], 'end_transition': rows[-1]['transition'],
        'complete': bool(rows[-1]['done']), 'failure': rows[-1]['failure'],
        'passed': bool(full and settled[tail].mean() >= .9),
        'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
        'settled_fraction': float(settled[tail].mean()), 'longest_settled_s': longest*.02,
        'reward_sum': float(sum(r['reward'] for r in rows)),
        'raw_advantage_mean': float(np.mean([r['raw_advantage'] for r in rows])),
        'action_logp_change_mean': float(np.mean([r['logp_after']-r['logp_before'] for r in rows]))}


class EpisodeAudit(BaseCallback):
    """Never samples actions or changes tensors, RNG, native state or gradients.

    Raw GAE is logged before minibatch normalization. Logp changes measure the
    net policy update at the recorded inputs, not attribution to one neuron.
    Episodes can span multiple rollout/update boundaries.
    """
    def __init__(self, root):
        super().__init__()
        root = Path(root)
        self.trace_file = gzip.open(root/'training_steps.jsonl.gz', 'wt', compresslevel=1)
        self.episode_file = (root/'training_episodes.jsonl').open('w')
        self.pending, self.completed, self.ids = {}, [], {}
        self.episode_count = self.passed_count = 0

    def _on_rollout_start(self):
        self.rows = []

    def _on_step(self):
        group = []
        for worker, info in enumerate(self.locals['infos']):
            done = bool(self.locals['dones'][worker])
            row = {'worker': worker, 'episode': self.ids.get(worker, 0),
                'condition': info['training_condition'], 'transition': self.num_timesteps,
                't': float(info['t']), 'q_deg': float(np.rad2deg(info['q'])),
                'target_deg': float(np.rad2deg(info['requested_target_q'])),
                'velocity_deg_s': float(np.rad2deg(info['qvel'])),
                'reward': float(self.locals['rewards'][worker]),
                'pressure_before': float(self.locals['obs_tensor'][worker, -1]),
                'pressure_after': float(info['exploration']['pressure']),
                'done': done, 'failure': done and not info.get('TimeLimit.truncated', False)}
            for name in ('excitation', 'delayed_excitation', 'activation', 'tendon_force_N',
                         'tendon_strain', 'capacity'):
                row[name] = list(info[name])
            group.append(row)
            self.pending.setdefault(worker, []).append(row)
            if done:
                self.completed.append(self.pending.pop(worker))
                self.ids[worker] = row['episode']+1
        self.rows.append(group)
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        # SB3 later flattens buffers in worker-major order. Preserve our own
        # time-major copies before that in-place reshape.
        self.observations = buffer.observations.copy().reshape(-1, buffer.observations.shape[-1])
        self.actions = buffer.actions.copy().reshape(-1, buffer.actions.shape[-1])
        with torch.no_grad():
            means = self.model.policy.get_distribution(torch.as_tensor(
                self.observations, device=self.model.device)).distribution.mean.cpu().numpy()
        for t, group in enumerate(self.rows):
            for worker, row in enumerate(group):
                row.update(raw_advantage=float(buffer.advantages[t, worker]),
                    value_before=float(buffer.values[t, worker]),
                    return_target=float(buffer.returns[t, worker]),
                    buffer_reward=float(buffer.rewards[t, worker]),
                    logp_before=float(buffer.log_probs[t, worker]),
                    mean_before=means[t*len(group)+worker].tolist())

    def _on_training_end(self):
        policy = self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        try:
            with torch.no_grad():
                observations = torch.as_tensor(self.observations, device=self.model.device)
                actions = torch.as_tensor(self.actions, device=self.model.device)
                _, logp, _ = policy.evaluate_actions(observations, actions)
                means = policy.get_distribution(observations).distribution.mean.cpu().numpy()
            for i, row in enumerate(r for group in self.rows for r in group):
                row.update(logp_after=float(logp[i]), mean_after=means[i].tolist(),
                           update_end_transition=self.num_timesteps)
                self.trace_file.write(json.dumps(row, separators=(',', ':'))+'\n')
            for rows in self.completed:
                result = episode_result(rows)
                self.episode_file.write(json.dumps(result)+'\n')
                self.episode_count += 1
                self.passed_count += result['passed']
            self.completed.clear()
            self.trace_file.flush()
            self.episode_file.flush()
        finally:
            policy.set_training_mode(was_training)

    def close(self):
        self.trace_file.close()
        self.episode_file.close()
