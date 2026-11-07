"""Select by validation before test; audit separated feature trials with native physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.pretrained_vertical import load_pretrained, TRIALS
from myo.body_predictor import load_dataset, indices, forecast_metrics
from myo.train_shared_vertical import prediction_metrics
from myo.train_vertical_ppo import compact, write_json
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import VerticalPrecisionEnv, PRECISION_VALIDATION, PRECISION_TEST, evaluate_precision
from audit_vertical_precision_motion import replay, metrics
from evaluate_detached_vertical import validation_rollouts, persistence_metrics, feature_use


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/pretrained_vertical'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summaries = {name: json.loads((args.root/name/'summary.json').read_text()) for name in TRIALS}
    assert all(s['status'] == 'training_complete' for s in summaries.values())
    assert len({s['steps'] for s in summaries.values()}) == 1
    assert len({s['initial_private_sha256'] for s in summaries.values()}) == 1
    for key in ('initial_parameter_sha256', 'first_rollout_sha256'):
        assert summaries['pretrained_frozen'][key] == summaries['pretrained_online'][key]
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
    body_data, body_metadata = load_dataset(Path('runs/body_predictor'))
    scales = []
    for name in TRIALS:
        folder = args.root/name
        models = {}
        for checkpoint in ('initial', 'best', 'last'):
            model, env, _ = load_pretrained(folder/f'{checkpoint}.zip')
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
        body_forecasts = {label: {str(20*h): forecast_metrics(candidate.policy, body_data, indices(body_metadata, 'test', h), h)
                           for h in (1, 3)} for label, candidate in models.items()}
        write_json(folder/'body_forecasts.json', body_forecasts)
        updates = summaries[name]['updates']
        drift = {key: {'mean': float(np.mean([u[key] for u in updates])),
                       'max': float(np.max([u[key] for u in updates]))}
                 for key in ('predictor_phase_kl', 'total_phase_kl')}
        drift['total_phase_above_ppo_limit'] = sum(u['total_phase_kl'] > .03 for u in updates)
        drift['rollouts_no_ppo_update'] = sum(u['accepted_steps'] == 0 for u in updates)
        persistence = {'fixed': persistence_metrics(dataset), 'policy': persistence_metrics(own_dataset)}
        write_json(folder/'prediction_evaluation.json', {'forecast': prediction, 'policy_forecast': own_prediction,
            'persistence': persistence, 'body_forecasts': body_forecasts, 'drift': drift, 'feature_use': features})
        summaries[name].update(status='complete', motion=motion, prediction_evaluation=prediction,
            policy_prediction_evaluation=own_prediction, persistence=persistence, body_forecasts=body_forecasts, drift=drift,
            feature_use=features)
        write_json(folder/'summary.json', summaries[name])
        print(json.dumps({'trial': name, 'test_mm': summaries[name]['best_test']['score_m']*1000,
                          'passed': summaries[name]['best_test']['passed'], 'drift': drift}), flush=True)
    for candidate in scales[1:]:
        for left, right in zip(scales[0], candidate, strict=True):
            np.testing.assert_array_equal(left, right)
    for name in ('random_frozen', 'pretrained_frozen'):
        initial, env, _ = load_pretrained(args.root/name/'initial.zip'); env.close()
        last, env, _ = load_pretrained(args.root/name/'last.zip'); env.close()
        for key, value in initial.policy.state_dict().items():
            if key.startswith(('mlp_extractor.predictor.', 'prediction_net.', 'delta_')):
                torch.testing.assert_close(value, last.policy.state_dict()[key], rtol=0, atol=0)
    for name in TRIALS:
        initial, env, _ = load_pretrained(args.root/name/'initial.zip'); env.close()
        last, env, _ = load_pretrained(args.root/name/'last.zip'); env.close()
        for key in ('delta_mean', 'delta_std', 'delta_calibrated'):
            torch.testing.assert_close(getattr(initial.policy, key), getattr(last.policy, key), rtol=0, atol=0)
    audit.update(status='passed', private_initialization_identical=True,
                 pretrained_pair_initialization_and_first_rollout_identical=True,
                 frozen_delta_scales_identical=True, frozen_predictors_unchanged=True)
    write_json(args.root/'evaluation_audit.json', audit)


if __name__ == '__main__':
    main()
