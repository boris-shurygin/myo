"""Measure frozen predictor feature use without fitting or selecting weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence
from torch.nn import functional as F

from myo.arm_learning import load_arm
from myo.arm_precision import evaluate_arm


ROOT = Path('runs/arm_preacher_balanced')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(policy, latent):
    alpha, beta = (F.softplus(policy.action_net(latent)) + .05).chunk(2, dim=-1)
    return Beta(alpha.double(), beta.double())


def outputs(policy, observations, body_features):
    extractor = policy.mlp_extractor
    actor = extractor.policy_net(torch.cat((body_features, extractor.actor_private(observations)), -1))
    critic = extractor.value_net(torch.cat((body_features, extractor.critic_private(observations)), -1))
    return distribution(policy, actor), policy.value_net(critic).flatten().double()


def metrics(reference, changed, original_value, changed_value, muscle_names):
    difference = (changed.mean - reference.mean).abs()
    divergence = kl_divergence(reference, changed).sum(-1)
    value_difference = (changed_value - original_value).abs()
    return {
        'mean_abs_excitation_change': float(difference.mean()),
        'p95_max_excitation_change_per_state': float(torch.quantile(difference.amax(-1), .95)),
        'fraction_states_any_excitation_change_gt_001': float((difference.amax(-1) > .01).double().mean()),
        'per_muscle_mean_abs_excitation_change': dict(zip(muscle_names, difference.mean(0).tolist(), strict=True)),
        'mean_abs_std_change': float((changed.stddev - reference.stddev).abs().mean()),
        'mean_exact_joint_kl_native_to_changed': float(divergence.mean()),
        'p95_exact_joint_kl_native_to_changed': float(torch.quantile(divergence, .95)),
        'mean_abs_value_change': float(value_difference.mean()),
        'p95_abs_value_change': float(torch.quantile(value_difference, .95)),
    }


class FeatureIntervention:
    """Override only the actor's body vector; keep all raw observations intact."""

    def __init__(self, policy, mode, mean):
        self.policy, self.mode, self.mean = policy, mode, mean

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('Closed-loop diagnostic uses deterministic Beta means')
        obs, vectorized = self.policy.obs_to_tensor(observation)
        extractor = self.policy.mlp_extractor
        native = extractor.features(obs)
        features = native if self.mode == 'native' else (
            self.mean.expand_as(native) if self.mode == 'mean' else torch.zeros_like(native))
        actor = extractor.policy_net(torch.cat((features, extractor.actor_private(obs)), -1))
        # Match production float32 Beta mean before handing the command to physics.
        action = self.policy._get_action_dist_from_latent(actor).mode().cpu().numpy()
        return (action if vectorized else action[0]), None


@torch.no_grad()
def audit_checkpoint(path, observations, meta, closed_loop):
    model, env, identity = load_arm(path)
    try:
        if env.contract() != meta['contract']:
            raise ValueError('Diagnostic observation contract mismatch')
        names = env.unwrapped.muscle_names
        policy = model.policy
        policy.set_training_mode(False)
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        obs = torch.as_tensor(observations)
        features = policy.mlp_extractor.features(obs)
        mean = features.mean(0, keepdim=True)
        permutation = torch.randperm(len(obs), generator=torch.Generator().manual_seed(20261111))
        replacements = {'native': features, 'mean': mean.expand_as(features),
                        'shuffle': features[permutation], 'zero': torch.zeros_like(features)}
        reference, original_value = outputs(policy, obs, features)
        # Verify manual branch composition against the actual policy implementation.
        actual = policy.get_distribution(obs).distribution
        torch.testing.assert_close(reference.mean.float(), actual.mean, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(original_value.float(), policy.predict_values(obs).flatten())
        for mode in ('native', 'mean', 'zero'):
            wrapper = FeatureIntervention(policy, mode, mean)
            changed, _ = outputs(policy, obs, replacements[mode])
            torch.testing.assert_close(torch.from_numpy(wrapper.predict(observations)[0]),
                                       changed.mean.float(), rtol=2e-6, atol=2e-7)
        result = {'checkpoint': str(path), 'checkpoint_sha256': identity['checkpoint_sha256'],
                  'steps': identity['steps'], 'mode': policy.feature_mode,
                  'native_value_rms': float(original_value.square().mean().sqrt()),
                  'native_mean_excitation_per_muscle': dict(zip(names, reference.mean.mean(0).tolist(), strict=True)),
                  'feature_rms': float(features.square().mean().sqrt()),
                  'fixed_states': {}, 'validation': {}}
        for mode, replacement in replacements.items():
            changed, values = outputs(policy, obs, replacement)
            row = metrics(reference, changed, original_value, values, names)
            row['by_case'] = []
            for episode in meta['episodes']:
                a, b = episode['start'], episode['stop']
                row['by_case'].append({'case': episode['case'], **metrics(
                    Beta(reference.concentration1[a:b], reference.concentration0[a:b]),
                    Beta(changed.concentration1[a:b], changed.concentration0[a:b]),
                    original_value[a:b], values[a:b], names)})
            result['fixed_states'][mode] = row
        if policy.feature_mode == 'none':
            for row in result['fixed_states'].values():
                assert row['mean_abs_excitation_change'] == row['mean_abs_value_change'] == 0.
        if closed_loop:
            for mode in ('native', 'mean', 'zero'):
                validation = evaluate_arm(FeatureIntervention(policy, mode, mean))
                result['validation'][mode] = validation
                print(json.dumps({'checkpoint': str(path), 'intervention': mode,
                    'validation_rmse_deg': validation['score_deg'], 'passed': validation['passed'],
                    'failures': validation['failures']}), flush=True)
        assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
        assert digest(path) == identity['checkpoint_sha256']
        result['weights_and_buffers_unchanged'] = True
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'feature_use_audit.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    path = ROOT/'selected_policy_probe.npz'
    with np.load(path) as archive:
        observations = archive['observations'].copy()
    meta = json.loads(path.with_suffix('.json').read_text())
    report = {
        'scope': 'Post-training diagnostic only; no optimization, checkpoint selection or test evaluation',
        'observations': {'path': str(path), 'sha256': digest(path), 'count': len(observations),
                         'episodes': len(meta['episodes']), 'source': 'stochastic selected-policy TRAIN_CASES'},
        'interventions': {
            'mean': 'Same per-coordinate mean over the fixed diagnostic observations for every state',
            'shuffle': 'Fixed seed 20261111 permutation of entire feature vectors; marginal distribution preserved',
            'zero': 'Zero feature vector; potentially outside the training distribution',
            'closed_loop': 'Deterministic actor intervention on original nine validation cases; critic does not drive inference'},
        'models': {},
    }
    for name, checkpoint, closed_loop in (
        ('new_best', ROOT/'frozen/best.zip', True),
        ('new_last', ROOT/'frozen/last.zip', True),
        ('none_control', Path('runs/arm_preacher/none/best.zip'), False),
    ):
        report['models'][name] = audit_checkpoint(checkpoint, observations, meta, closed_loop)
        print(json.dumps({'model': name, 'fixed_states': {
            mode: {k: row[k] for k in ('mean_abs_excitation_change',
                'mean_exact_joint_kl_native_to_changed', 'mean_abs_value_change')}
            for mode, row in report['models'][name]['fixed_states'].items()}}), flush=True)
    summary = json.loads((ROOT/'frozen/summary.json').read_text())
    for name, expected in (('new_best', summary['best_validation']),
                           ('new_last', summary['history'][-1]['validation'])):
        actual = report['models'][name]['validation']['native']
        assert actual['passed'] == expected['passed']
        np.testing.assert_allclose(actual['score_deg'], expected['score_deg'], rtol=0, atol=1e-5)
    report['audit'] = 'passed: native outputs/validation reproduced; none is insensitive; weights unchanged'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
