"""Train a standalone direct 60-ms predictor on whole native-arm episodes."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from .arm_anticipation import (AnticipationPredictor, identity, examples, physical_errors,
    predictions, save_predictor, load_predictor)
from .arm_balanced import sampling_weights
from .arm_learning import predictor_identity
from .train_arm_precision import load_data
from .train_vertical_ppo import write_json


ROOT = Path('runs/arm_anticipation60')
SOURCE = Path('runs/arm_preacher_balanced')


def prepare(data, meta, split):
    ids, x, u, actual = examples(data, meta, split)
    p, sampling = sampling_weights(data['observations'][ids], meta)
    return {'ids': ids, 'x': torch.as_tensor(x), 'u': torch.as_tensor(u),
            'delta': torch.as_tensor(actual-x[:, :20]), 'probability': p, 'sampling': sampling}


@torch.no_grad()
def metric(body, batch, maximum=None):
    ids = np.arange(len(batch['x']))
    if maximum is not None and len(ids) > maximum:
        ids = np.linspace(0, len(ids)-1, maximum, dtype=int)
    losses = []
    for part in np.array_split(ids, max(1, int(np.ceil(len(ids)/2048)))):
        target = (batch['delta'][part]-body.delta_mean)/body.delta_std
        losses.append((body.normalized_delta(batch['x'][part], batch['u'][part])-target).square().mean(1).numpy())
    losses = np.concatenate(losses)
    weights = batch['probability'][ids]; weights = weights/weights.sum()
    return {'samples': len(ids), 'uniform_mse': float(losses.mean()),
            'balanced_mse': float(losses@weights)}


def train(root, source, maximum, learning_rate):
    root.mkdir(parents=True, exist_ok=False)
    data, meta = load_data(source)
    spec = identity(meta['contract'])
    # Only physical state/FIFO must be continuous; goals can change between frames.
    checked = meta['contract']['physical_indices']+meta['contract']['pending_indices']
    for episode in meta['episodes']:
        a, b = episode['start'], episode['stop']
        np.testing.assert_array_equal(data['next_observations'][a:b-1][:, checked],
                                      data['observations'][a+1:b][:, checked])
    batches = {split: prepare(data, meta, split) for split in ('train', 'validation')}
    torch.manual_seed(0)
    body = AnticipationPredictor()
    delta = batches['train']['delta']
    body.delta_mean.copy_(delta.mean(0))
    body.delta_std.copy_(delta.std(0, unbiased=False).clamp_min(1e-5))
    body.calibrated.fill_(True)
    optimizer = torch.optim.RMSprop(body.parameters(), lr=learning_rate, eps=1e-5, alpha=.99)
    training = {'dataset': str(source/'body/dataset.npz'), 'dataset_sha256': meta['dataset_sha256'],
        'seed': 0, 'optimizer': 'RMSprop', 'learning_rate': learning_rate, 'batch_size': 128,
        'gradient_norm': .5, 'maximum_updates': maximum, 'check_updates': 1000,
        'objective': 'equal-channel train-normalized delta60 MSE, mixed stratified samples',
        'normalization': 'mean/std of all valid train delta60 only, fixed; floor1e-5',
        'sampling': batches['train']['sampling'], 'selection': 'minimum full balanced validation MSE',
        'stop': 'after20000, no accumulated0.5% validation improvement for20000 updates, or maximum',
        'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')},
        'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary = {'status': 'running', 'identity': spec, 'training': training, 'history': [],
        'samples': {s: len(b['ids']) for s, b in batches.items()},
        'parameter_count': sum(p.numel() for p in body.parameters())}
    save_predictor(root/'initial.pt', body, optimizer, 0, spec, training)
    target = (delta-body.delta_mean)/body.delta_std
    rng = np.random.default_rng(0)
    cdf = np.cumsum(batches['train']['probability']); cdf[-1] = 1.
    best, reference, progress = np.inf, np.inf, 0
    begun = time.monotonic()
    for update in range(maximum+1):
        if update % 1000 == 0 or update == maximum:
            validation = metric(body, batches['validation'])
            value = validation['balanced_mse']
            summary['history'].append({'update': update, 'validation': validation,
                'train': metric(body, batches['train'], 8192)})
            if value < best:
                best = value; summary['best_update'] = update
                save_predictor(root/'best.pt', body, optimizer, update, spec, training)
            if value < reference*.995:
                reference, progress = value, update
            summary.update(elapsed_s=time.monotonic()-begun, updates=update)
            write_json(root/'training.json', summary)
            print(json.dumps({'update': update, 'validation': value,
                'best_update': summary['best_update']}), flush=True)
            if update >= 20000 and update-progress >= 20000:
                summary['stop_reason'] = 'validation_plateau'; break
        if update == maximum:
            break
        indices = np.searchsorted(cdf, rng.random(128))
        optimizer.zero_grad(set_to_none=True)
        loss = (body.normalized_delta(batches['train']['x'][indices], batches['train']['u'][indices])-target[indices]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite direct anticipation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(body.parameters(), .5)
        optimizer.step()
    save_predictor(root/'last.pt', body, optimizer, update, spec, training)
    summary.update(status='training_complete', updates=update, elapsed_s=time.monotonic()-begun,
                   stop_reason=summary.get('stop_reason', 'maximum_updates'))
    write_json(root/'training.json', summary)
    selected, selected_meta = load_predictor(root/'best.pt')
    assert metric(selected, batches['validation']) == next(h['validation'] for h in summary['history']
                                                          if h['update'] == summary['best_update'])
    write_json(root/'frozen_selection.json', {'update': selected_meta['update'],
        'checkpoint_sha256': selected_meta['checkpoint_sha256'], 'selection_before_test': True})
    # The historical one-step predictor has never seen future actions in this comparison.
    old, old_meta = predictor_identity(source/'body/best.pt')
    result = {'direct_identity': selected_meta, 'iterative_identity': old_meta, 'groups': {}}
    groups = {'all': lambda e: True, **{s: (lambda e, s=s: e.get('family') == s)
              for s in ('hold', 'ramp', 'jitter', 'policy')},
              'open_loop': lambda e: e.get('family') in ('independent', 'reciprocal', 'common', 'multisine')}
    for group, match in groups.items():
        subset = {**meta, 'episodes': [e for e in meta['episodes'] if match(e)]}
        ids, x, u, actual = examples(data, subset, 'test')
        initial = x[:, :20]
        step = data['next_observations'][ids][:, meta['contract']['physical_indices']]
        values = {}
        for label, predicted in (
            ('direct', predictions(selected, x, u)),
            ('iterative', predictions(old, x, u, iterative=True)), ('persistence', initial)):
            values[label] = physical_errors(predicted, actual, initial, step, spec['output_names'])
        result['groups'][group] = values
    write_json(root/'test.json', result)
    summary.update(status='complete', test=result['groups']['all'], strict_reload=True)
    write_json(root/'training.json', summary)
    print(json.dumps({'status': 'complete', 'best_update': summary['best_update'],
        'test_q60_deg': summary['test']['direct']['angle_rmse_deg'],
        'test_v60_deg_s': summary['test']['direct']['velocity_rmse_deg_s']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--updates', type=int, default=150000)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    args = p.parse_args()
    if args.updates <= 0 or args.updates % 1000 or args.learning_rate <= 0:
        p.error('Use positive complete1000-update budgets and learning rate')
    torch.set_num_threads(1)
    train(args.root, args.source, args.updates, args.learning_rate)


if __name__ == '__main__':
    main()
