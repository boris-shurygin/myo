"""Select by validation before test; audit separated feature trials with native physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.detached_vertical import load_detached, TRIALS, distribution_snapshot, distribution_kl
from myo.train_shared_vertical import prediction_metrics
from myo.train_vertical_ppo import compact, write_json
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import VerticalPrecisionEnv, PRECISION_VALIDATION, PRECISION_TEST, evaluate_precision
from audit_vertical_precision_motion import replay, metrics
from evaluate_shared_vertical import predictor_jacobian


def validation_rollouts(model):
    episodes = {key: [] for key in ('observations', 'actions', 'next_observations')}
    for initial, target in PRECISION_VALIDATION:
        env = VerticalPrecisionEnv(VerticalConfig(variant=4))
        obs, _ = env.reset(options={'initial_position_m': initial, 'target_m': target})
        rows = {key: [] for key in episodes}
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            following, _, terminated, _, _ = env.step(action)
            assert not terminated, 'Cannot compare full predictor episodes after physical failure'
            for key, value in zip(rows, (obs, action, following), strict=True):
                rows[key].append(value)
            obs = following
        env.close()
        for key in episodes:
            episodes[key].append(rows[key])
    return {key: np.asarray(value) for key, value in episodes.items()}


def persistence_metrics(dataset):
    result = {}
    for horizon in (1, 3):
        count = dataset['observations'].shape[1]-horizon+1
        error = (dataset['next_observations'][:, horizon-1:horizon-1+count, :2]
                 -dataset['observations'][:, :count, :2])
        rms = np.sqrt(np.mean(error**2, axis=(0, 1)))
        result[str(20*horizon)] = {'position_rmse_mm': float(rms[0]*100),
                                  'velocity_rmse_mm_s': float(rms[1]*500)}
    return result


def feature_use(policy, dataset):
    observations = torch.as_tensor(dataset['observations'].reshape(-1, 13))
    with torch.no_grad():
        features = policy.mlp_extractor.predictor_features(observations)
        before = distribution_snapshot(policy, observations)
        values = policy.predict_values(observations)
        average = features.mean(0, keepdim=True)
        handle = policy.mlp_extractor.predictor.register_forward_hook(lambda module, inputs, output: average.expand_as(output))
        try:
            after = distribution_snapshot(policy, observations)
            changed_values = policy.predict_values(observations)
        finally:
            handle.remove()
    return {'protocol': 'Replace h1 with its fixed-validation mean; own branches retain original inputs; diagnostic only',
        'policy_kl': distribution_kl(before, after),
        'mean_action_abs_change': float((before.mean-after.mean).abs().mean()),
        'value_rmse_change': float((values-changed_values).square().mean().sqrt()),
        'mean_feature_std': float(features.std(0, unbiased=False).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/detached_vertical'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summaries = {name: json.loads((args.root/name/'summary.json').read_text()) for name in TRIALS}
    assert all(s['status'] == 'training_complete' for s in summaries.values())
    assert len({s['steps'] for s in summaries.values()}) == 1
    for key in ('initial_parameter_sha256', 'first_rollout_sha256'):
        assert len({s[key] for s in summaries.values()}) == 1
    rank = lambda result: (result['failures'], -result['passed'], result['score_m'])
    for s in summaries.values():
        assert min(s['history'], key=lambda row: rank(row['validation']))['steps'] == s['best_steps']
        assert max(u['max_accepted_kl'] for u in s['updates']) <= .03
        assert len(s['updates']) == s['steps']//1024
        assert sum(u['accepted_steps'] for u in s['updates']) == s['accepted_optimizer_steps']
        assert sum(u['predictor_steps'] for u in s['updates']) == s['predictor_steps']
    write_json(args.root/'frozen_selection.json', {'basis': 'Completed validation only; test not used',
        'selections': {name: {'steps': s['best_steps'], 'sha256': json.loads((args.root/name/'best.zip.metadata.json').read_text())['checkpoint_sha256']}
                       for name, s in summaries.items()}})
    dataset_path = args.root/'prediction_validation.npz'
    dataset = dict(np.load(dataset_path))
    audit = {'status': 'running', 'strict_loads': 0, 'native_test_replays': 0,
             'prediction_validation_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest()}
    scales = []
    for name in TRIALS:
        folder = args.root/name
        models = {}
        for checkpoint in ('initial', 'best', 'last'):
            model, env, _ = load_detached(folder/f'{checkpoint}.zip')
            env.close(); models[checkpoint] = model
            audit['strict_loads'] += 1
        assert models['last'].accepted_steps == summaries[name]['accepted_optimizer_steps']
        assert models['last'].predictor_steps == summaries[name]['predictor_steps']
        selected = models['best']
        validation = evaluate_precision(selected, VerticalConfig(variant=4), PRECISION_VALIDATION)
        assert compact(validation) == summaries[name]['best_validation']
        motion = []
        for label in ('best', 'last'):
            result = evaluate_precision(models[label], VerticalConfig(variant=4), PRECISION_TEST)
            write_json(folder/f'{label}_test.json', result)
            summaries[name][label+'_test'] = compact(result)
            if label == 'best':
                for case in result['cases']:
                    trace = replay(case, VerticalConfig(variant=4))
                    audit['native_test_replays'] += 1
                    if case['initial_position_m'] != case['target_m']:
                        motion.append(metrics(trace, case['initial_position_m'], case['target_m']))
        write_json(folder/'motion.json', motion)
        prediction = prediction_metrics(selected.policy, dataset)
        features = feature_use(selected.policy, dataset)
        own_dataset = validation_rollouts(selected)
        np.savez_compressed(folder/'policy_prediction_validation.npz', **own_dataset)
        own_prediction = prediction_metrics(selected.policy, own_dataset)
        scales.append((selected.policy.delta_mean.detach().numpy(), selected.policy.delta_std.detach().numpy()))
        jacobian = predictor_jacobian(selected.policy, dataset) if name == 'online' else None
        updates = summaries[name]['updates']
        drift = {key: {'mean': float(np.mean([u[key] for u in updates])),
                       'max': float(np.max([u[key] for u in updates]))}
                 for key in ('predictor_phase_kl', 'total_phase_kl')}
        drift['total_phase_above_ppo_limit'] = sum(u['total_phase_kl'] > .03 for u in updates)
        drift['rollouts_no_ppo_update'] = sum(u['accepted_steps'] == 0 for u in updates)
        persistence = {'fixed': persistence_metrics(dataset), 'policy': persistence_metrics(own_dataset)}
        write_json(folder/'prediction_evaluation.json', {'forecast': prediction, 'policy_forecast': own_prediction,
            'persistence': persistence, 'jacobian': jacobian, 'drift': drift, 'feature_use': features})
        summaries[name].update(status='complete', motion=motion, prediction_evaluation=prediction,
            policy_prediction_evaluation=own_prediction, persistence=persistence, jacobian=jacobian, drift=drift,
            feature_use=features)
        write_json(folder/'summary.json', summaries[name])
        print(json.dumps({'trial': name, 'test_mm': summaries[name]['best_test']['score_m']*1000,
                          'passed': summaries[name]['best_test']['passed'], 'drift': drift}), flush=True)
    for left, right in zip(scales[0], scales[1], strict=True):
        np.testing.assert_array_equal(left, right)
    frozen_models = []
    for label in ('initial', 'last'):
        model, env, _ = load_detached(args.root/'frozen'/f'{label}.zip')
        env.close(); frozen_models.append(model)
    initial, frozen_last = frozen_models
    for key, value in initial.policy.mlp_extractor.predictor.state_dict().items():
        torch.testing.assert_close(value, frozen_last.policy.mlp_extractor.predictor.state_dict()[key], rtol=0, atol=0)
    audit.update(status='passed', initialization_and_first_rollout_identical=True,
                 frozen_delta_scales_identical=True, frozen_predictor_unchanged=True)
    write_json(args.root/'evaluation_audit.json', audit)


if __name__ == '__main__':
    main()
