"""Test whether feature scaling can be absorbed by existing readout weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from myo.arm_learning import load_arm, predictor_identity


ROOT = Path('runs/arm_conditioning')


def moments(x):
    mean = x.mean(0, keepdim=True)
    rms = float(x.square().mean().sqrt())
    constant = float(mean.square().mean().sqrt())
    variation = float((x-mean).square().mean().sqrt())
    np.testing.assert_allclose(rms*rms, constant*constant+variation*variation, rtol=1e-5, atol=1e-7)
    return {'rms': rms, 'constant_mean_rms': constant, 'centered_rms': variation,
            'mean_fraction_of_second_moment': constant*constant/(rms*rms) if rms else 0.}


def compensated_tail(tail, h, private, gain):
    first = tail[0]
    weight = first.weight.clone()
    weight[:, :128] /= gain
    return tail[1:](F.linear(torch.cat((h*gain, private), -1), weight, first.bias))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scale_equivalence.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    path = Path('runs/arm_preacher_balanced/selected_policy_probe.npz')
    with np.load(path) as data:
        obs = torch.as_tensor(data['observations'].copy())
    body, identity = predictor_identity(Path('runs/arm_preacher_balanced/body/best.pt'))
    indices = identity['contract']['predictor_indices']
    h = body.encoder(obs[:, indices])
    gain = .25
    original = body.head(h)
    adjusted = F.linear(h*gain, body.head.weight/gain, body.head.bias)
    torch.testing.assert_close(original, adjusted, rtol=0, atol=1e-6)
    report = {'scope': 'Functional equivalence only; no fitting, rollout or weight changes',
              'observations_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'states': len(obs),
              'rescale_factor': gain, 'predictor_max_abs_output_difference': float((original-adjusted).abs().max()),
              'policies': {}}
    for name, folder in [('raw', Path('runs/arm_preacher_balanced/frozen')),
                         ('gain025', ROOT/'gain_025')]:
        for label in ('initial', 'best'):
            model, env, meta = load_arm(folder/f'{label}.zip')
            env.close()
            policy, ex = model.policy, model.policy.mlp_extractor
            state = {k: v.clone() for k, v in policy.state_dict().items()}
            h = ex.features(obs)
            p, c = ex.actor_private(obs), ex.critic_private(obs)
            actor = ex.policy_net(torch.cat((h, p), -1))
            critic = ex.value_net(torch.cat((h, c), -1))
            compensated_actor = compensated_tail(ex.policy_net, h, p, gain)
            compensated_critic = compensated_tail(ex.value_net, h, c, gain)
            parameters = policy.action_net(actor)
            changed = policy.action_net(compensated_actor)
            value, changed_value = policy.value_net(critic), policy.value_net(compensated_critic)
            torch.testing.assert_close(parameters, changed, rtol=0, atol=1e-6)
            torch.testing.assert_close(value, changed_value, rtol=0, atol=1e-6)
            assert all(torch.equal(v, policy.state_dict()[k]) for k, v in state.items())
            report['policies'][f'{name}_{label}'] = {
                'checkpoint_sha256': meta['checkpoint_sha256'],
                'max_abs_action_parameter_difference': float((parameters-changed).abs().max()),
                'max_abs_value_difference': float((value-changed_value).abs().max()),
                'weights_unchanged': True,
                'actor_branch_moments': {'h': moments(h), 'private': moments(p),
                    'h_first_linear_contribution': moments(h@ex.policy_net[0].weight[:, :128].T),
                    'private_first_linear_contribution': moments(p@ex.policy_net[0].weight[:, 128:].T)}}
    report['audit'] = 'passed: h*0.25 with readout weights*4 preserves outputs'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
