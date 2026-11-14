"""Reconstruct the first matched rollout and inspect PPO gradients without updates."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_exploration import load_exploration


def first_rollout(path):
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(next(stream)) for _ in range(1024)]
    assert len({row['update_end_transition'] for row in rows}) == 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path('runs/arm_effort_curriculum')
    control = first_rollout(root/'control/training_steps.jsonl.gz')
    effort = first_rollout(root/'activation_01/training_steps.jsonl.gz')
    assert [(r['worker'], r['transition']) for r in control] == [(r['worker'], r['transition']) for r in effort]
    assert all(a['excitation'] == b['excitation'] for a, b in zip(control, effort))
    model, env, meta = load_exploration(root/'control/initial.zip')
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    before_updates = model.optimizer_steps
    observations = [None]*len(control)
    max_replay_error = 0.
    current = None
    try:
        for i in sorted(range(len(control)), key=lambda i: (control[i]['worker'], control[i]['transition'])):
            row = control[i]
            identity = row['worker'], row['episode']
            if identity != current:
                obs, _ = env.reset(seed=0, options=row['condition'])
                current = identity
            observations[i] = obs.copy()
            obs, _, _, _, info = env.step(np.array(row['excitation']))
            for field in ('activation', 'delayed_excitation', 'tendon_force_N', 'tendon_strain', 'capacity'):
                max_replay_error = max(max_replay_error, float(np.max(np.abs(np.array(info[field])-row[field]))))
            max_replay_error = max(max_replay_error, abs(np.rad2deg(info['q'])-row['q_deg']))
        if max_replay_error > 1e-7:
            raise AssertionError(f'Physical rollout replay mismatch: {max_replay_error}')
        observation_tensor = torch.as_tensor(np.stack(observations))
        action_tensor = torch.tensor([r['excitation'] for r in control], dtype=torch.float32)
        values, logp, entropy = model.policy.evaluate_actions(observation_tensor, action_tensor)
        old_logp = torch.tensor([r['logp_before'] for r in control])
        logp_error = float((logp-old_logp).abs().max().detach())
        value_error = float((values.flatten()-torch.tensor([r['value_before'] for r in control])).abs().max().detach())
        # Batch1024 and rollout batch4 use different float32 matrix kernels;
        # high-concentration Beta log densities amplify their tiny roundoff.
        with torch.no_grad():
            rollout_logp = torch.cat([model.policy.evaluate_actions(
                observation_tensor[i:i+4], action_tensor[i:i+4])[1] for i in range(0,1024,4)])
        rollout_logp_error = float((rollout_logp-old_logp).abs().max())
        if max(rollout_logp_error, value_error) > 1e-5 or logp_error > 5e-4:
            raise AssertionError(f'Observation/policy replay mismatch: rollout logp={rollout_logp_error}, fullbatch logp={logp_error}, value={value_error}')
        c = torch.tensor([r['raw_advantage'] for r in control])
        e = torch.tensor([r['raw_advantage'] for r in effort])
        # Initial old/new policies are identical. Evaluate the denominator with
        # the same batch kernel so the mathematical initial ratio is exactly1.
        ratio = (logp-logp.detach()).exp()
        assert bool(((ratio > .8) & (ratio < 1.2)).all())
        delta = e-c
        losses = {
            'position_surrogate': -(((c-c.mean())/(c.std()+1e-8))*ratio).mean(),
            'position_effort_surrogate': -(((e-e.mean())/(e.std()+1e-8))*ratio).mean(),
            'effort_credit_fixed_control_scale': -(((delta-delta.mean())/(c.std()+1e-8))*ratio).mean(),
            'entropy_regularizer': -.001*entropy.mean(),
        }
        actor = [(name, parameter) for name, parameter in model.policy.named_parameters()
                 if parameter.requires_grad and name.startswith((
                     'mlp_extractor.actor_private.', 'mlp_extractor.policy_net.', 'action_net.'))]
        gradients = {}
        for name, loss in losses.items():
            gradients[name] = torch.cat([g.flatten() for g in torch.autograd.grad(
                loss, [parameter for _, parameter in actor], retain_graph=True)])
        gradients['actual_surrogate_change'] = gradients['position_effort_surrogate']-gradients['position_surrogate']
        norms = {name: float(value.norm()) for name, value in gradients.items()}

        def cosine(a, b):
            return float(torch.dot(gradients[a], gradients[b])/(gradients[a].norm()*gradients[b].norm()+1e-30))

        a2 = np.array([np.mean(np.square(row['activation'])) for row in control])
        reward_delta = np.array([b['reward']-a['reward'] for a, b in zip(control, effort)])
        unchanged = all(torch.equal(before[name], value) for name, value in model.policy.state_dict().items())
        assert unchanged and model.optimizer_steps == before_updates
        report = {'scope': 'first matched1024 transitions, full-batch initial PPO actor-gradient diagnostic; '
                    'actual training uses normalized advantages in128-sample minibatches and repeated updates; '
                    'no optimizer step, no claim about later gradients or sole causal explanation',
            'initial_checkpoint': str(root/'control/initial.zip'), 'checkpoint_sha256': meta['checkpoint_sha256'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_replay_max_error': max_replay_error, 'logp_replay_max_error': logp_error,
            'rollout_batch4_logp_replay_max_error': rollout_logp_error,
            'value_replay_max_error': value_error, 'weights_unchanged': unchanged,
            'mean_activation_squared': float(a2.mean()),
            'position_reward_mean': float(np.mean([r['reward'] for r in control])),
            'effort_reward_delta_mean': float(reward_delta.mean()),
            'reward_equation_max_error': float(np.max(np.abs(reward_delta+.01*a2))),
            'control_raw_advantage_mean': float(c.mean()), 'control_raw_advantage_std': float(c.std()),
            'effort_raw_advantage_mean': float(e.mean()), 'effort_raw_advantage_std': float(e.std()),
            'raw_advantage_delta_mean': float(delta.mean()), 'raw_advantage_delta_std': float(delta.std()),
            'gradient_l2': norms,
            'cosine_effort_entropy': cosine('effort_credit_fixed_control_scale', 'entropy_regularizer'),
            'cosine_position_effort_credit': cosine('position_surrogate', 'effort_credit_fixed_control_scale'),
            'cosine_control_effort_surrogate': cosine('position_surrogate', 'position_effort_surrogate'),
            'loss_values': {name: float(value.detach()) for name, value in losses.items()},
            'actor_parameter_names': [name for name, _ in actor]}
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'actor_parameter_names'}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
