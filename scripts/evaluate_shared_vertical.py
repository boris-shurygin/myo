"""Freeze all shared trial selections before test; compare physical behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.shared_vertical import load_shared, SharedPolicy
from myo.train_shared_vertical import prediction_metrics
from myo.train_vertical_ppo import compact, write_json
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import PRECISION_VALIDATION, PRECISION_TEST, evaluate_precision
from audit_vertical_precision_motion import replay, metrics


NAMES = ('independent', 'shared', 'shared_prediction')


def predictor_jacobian(policy, dataset):
    config = VerticalConfig(variant=4)
    from myo.vertical_precision import VerticalPrecisionEnv
    predictions, actual = [], []
    for episode in (0, 1, 4):
        initial, target = PRECISION_VALIDATION[episode]
        env = VerticalPrecisionEnv(config)
        env.reset(options={'initial_position_m': initial, 'target_m': target})
        reached = 0
        for index in (50, 100):
            for k in range(reached, index):
                env.step(dataset['actions'][episode, k])
            reached = index
            np.testing.assert_array_equal(env.observation(), dataset['observations'][episode, index])
            snapshot = env.physical.snapshot()
            observation = torch.as_tensor(env.observation()[None, :])
            commands = torch.as_tensor(dataset['actions'][episode, index:index+3].copy())
            first = commands[0].detach().clone().requires_grad_(True)
            current = observation
            for step in range(3):
                current = policy.next_observation(current, (first if step == 0 else commands[step])[None, :])
            predicted = torch.autograd.grad(current[0, 1]*.5, first)[0].detach().numpy()
            native = []
            for muscle in range(2):
                endpoints, values = [], []
                for sign in (-1, 1):
                    trial = commands[0].numpy().copy()
                    trial[muscle] = np.clip(trial[muscle]+sign*.005, 0, 1)
                    env.physical.restore(snapshot)
                    env.physical.step(trial)
                    for command in commands[1:].numpy():
                        env.physical.step(command)
                    endpoints.append(env.physical.velocity)
                    values.append(trial[muscle])
                native.append((endpoints[1]-endpoints[0])/(values[1]-values[0]))
            predictions.append(predicted); actual.append(native)
            env.physical.restore(snapshot)
        env.close()
    predicted, native = np.asarray(predictions), np.asarray(actual)
    return {'probes': len(native), 'horizon_s': .06, 'native': native.tolist(), 'predicted': predicted.tolist(),
        'relative_rmse_to_zero': float(np.sqrt(np.mean((predicted-native)**2)/np.mean(native**2))),
        'cosine': float(np.sum(predicted*native)/(np.linalg.norm(predicted)*np.linalg.norm(native)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/shared_vertical'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summaries = {name: json.loads((args.root/name/'summary.json').read_text()) for name in NAMES}
    assert all(s['status'] == 'training_complete' for s in summaries.values())
    assert len({s['steps'] for s in summaries.values()}) == 1
    assert summaries['shared']['initial_parameter_sha256'] == summaries['shared_prediction']['initial_parameter_sha256']
    assert summaries['shared']['first_rollout_sha256'] == summaries['shared_prediction']['first_rollout_sha256']
    for s in summaries.values():
        assert max(u['max_accepted_kl'] for u in s['updates']) <= .03
    rank = lambda result: (result['failures'], -result['passed'], result['score_m'])
    for s in summaries.values():
        assert min(s['history'], key=lambda row: rank(row['validation']))['steps'] == s['best_steps']
    write_json(args.root/'frozen_selection.json', {'basis': 'Completed tracking validation only; test not used for selection',
        'selections': {name: {'steps': s['best_steps'], 'sha256': json.loads((args.root/name/'best.zip.metadata.json').read_text())['checkpoint_sha256']}
                       for name, s in summaries.items()}})
    dataset = dict(np.load(args.root/'prediction_validation.npz'))
    audit = {'status': 'running', 'strict_loads': 0, 'native_test_replays': 0}
    scales = []
    for name in NAMES:
        folder = args.root/name
        models = {}
        for checkpoint in ('initial', 'best', 'last'):
            model, env, _ = load_shared(folder/f'{checkpoint}.zip')
            env.close(); models[checkpoint] = model
            audit['strict_loads'] += 1
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
        if isinstance(selected.policy, SharedPolicy):
            scales.append((selected.policy.delta_mean.detach().numpy(), selected.policy.delta_std.detach().numpy()))
        jacobian = predictor_jacobian(selected.policy, dataset) if name == 'shared_prediction' else None
        write_json(folder/'prediction_evaluation.json', {'forecast': prediction, 'jacobian': jacobian})
        summaries[name].update(status='complete', motion=motion, prediction_evaluation=prediction, jacobian=jacobian)
        write_json(folder/'summary.json', summaries[name])
        print(json.dumps({'trial': name, 'test_mm': summaries[name]['best_test']['score_m']*1000,
                          'passed': summaries[name]['best_test']['passed']}), flush=True)
    for left, right in zip(scales[0], scales[1], strict=True):
        np.testing.assert_array_equal(left, right)
    audit.update(status='passed', shared_initialization_and_first_rollout_identical=True,
                 frozen_training_delta_scales_identical=True)
    write_json(args.root/'evaluation_audit.json', audit)


if __name__ == '__main__':
    main()
