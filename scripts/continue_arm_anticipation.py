"""Continue the direct 60-ms predictor with its original optimizer and sample stream."""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.arm_anticipation import load_predictor, save_predictor, examples, predictions, physical_errors
from myo.arm_learning import predictor_identity
from myo.train_arm_anticipation import SOURCE, prepare, metric
from myo.train_arm_precision import load_data
from myo.train_vertical_ppo import write_json


ROOT = Path('runs/arm_anticipation60_extended')
PARENT = Path('runs/arm_anticipation60/last.pt')


def restore_training(path):
    """Reconstruct RNGs: this fixed model has no stochastic training layers."""
    metadata = json.loads(Path(str(path)+'.metadata.json').read_text())
    seed = metadata['training']['seed']
    # Original initialization was the only consumer of Torch randomness.
    torch.manual_seed(seed)
    body, metadata = load_predictor(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    optimizer = torch.optim.RMSprop(body.parameters(), lr=metadata['training']['learning_rate'],
                                    eps=1e-5, alpha=.99)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    rng = np.random.default_rng(seed)
    remaining = metadata['update']*metadata['training']['batch_size']
    # Replaying doubles avoids assuming the BitGenerator's per-draw advance semantics.
    while remaining:
        count = min(remaining, 131072)
        rng.random(count)
        remaining -= count
    return body.train(), optimizer, rng, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--parent', type=Path, default=PARENT)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--additional-updates', type=int, default=300000)
    args = p.parse_args()
    if args.additional_updates <= 0 or args.additional_updates % 1000:
        p.error('Use a positive complete1000-update budget')
    torch.set_num_threads(1)
    data, meta = load_data(args.source)
    body, optimizer, rng, parent = restore_training(args.parent)
    if parent['training']['dataset_sha256'] != meta['dataset_sha256']:
        raise ValueError('Continuation dataset differs from the parent')
    source_summary = json.loads((args.parent.parent/'training.json').read_text())
    start, maximum = parent['update'], parent['update']+args.additional_updates
    if start != source_summary['updates'] or start != source_summary['best_update']:
        raise ValueError('This experiment continues the last checkpoint that was also selected best')
    batches = {s: prepare(data, meta, s) for s in ('train', 'validation')}
    target = (batches['train']['delta']-body.delta_mean)/body.delta_std
    torch.testing.assert_close(body.delta_mean, batches['train']['delta'].mean(0), rtol=0, atol=0)
    torch.testing.assert_close(body.delta_std, batches['train']['delta'].std(0, unbiased=False).clamp_min(1e-5), rtol=0, atol=0)
    assert batches['train']['sampling'] == parent['training']['sampling']
    cdf = np.cumsum(batches['train']['probability']); cdf[-1] = 1.
    args.root.mkdir(parents=True, exist_ok=False)
    training = copy.deepcopy(parent['training'])
    training.update(maximum_updates=maximum,
        continuation={'parent': str(args.parent), 'parent_sha256': parent['checkpoint_sha256'],
            'parent_update': start, 'additional_budget': args.additional_updates,
            'numpy_stream': 'same PCG64 seed; replay update*batch_size previous doubles',
            'torch_stream': 'original seed and same constructor; no stochastic layers',
            'optimizer_moments_preserved': True, 'normalization_preserved': True},
        versions={k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')},
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    summary = {'status': 'running', 'identity': parent['identity'], 'training': training,
        'history': copy.deepcopy(source_summary['history']), 'source_update': start,
        'samples': source_summary['samples'], 'parameter_count': source_summary['parameter_count'],
        'elapsed_s_before_continuation': source_summary['elapsed_s']}
    save_predictor(args.root/'initial.pt', body, optimizer, start, parent['identity'], training)
    best, reference, progress = np.inf, np.inf, 0
    for h in source_summary['history']:
        value = h['validation']['balanced_mse']
        if value < reference*.995:
            reference, progress = value, h['update']
    begun = time.monotonic()

    def check(update):
        nonlocal best, reference, progress
        validation = metric(body, batches['validation'])
        value = validation['balanced_mse']
        if update == start:
            assert validation == source_summary['history'][-1]['validation']
        else:
            summary['history'].append({'update': update, 'validation': validation,
                'train': metric(body, batches['train'], 8192)})
        if value < best:
            best = value; summary['best_update'] = update
            save_predictor(args.root/'best.pt', body, optimizer, update, parent['identity'], training)
        if value < reference*.995:
            reference, progress = value, update
        summary.update(updates=update, added_updates=update-start,
            elapsed_s=time.monotonic()-begun, last_significant_update=progress)
        write_json(args.root/'training.json', summary)
        print(json.dumps({'update': update, 'added': update-start, 'validation': value,
            'best_update': summary['best_update'], 'last_significant_update': progress}), flush=True)

    check(start)
    for update in range(start+1, maximum+1):
        indices = np.searchsorted(cdf, rng.random(128))
        optimizer.zero_grad(set_to_none=True)
        loss = (body.normalized_delta(batches['train']['x'][indices], batches['train']['u'][indices])-target[indices]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite continued anticipation loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(body.parameters(), .5); optimizer.step()
        if update % 1000 == 0:
            check(update)
            if update-progress >= 20000:
                summary['stop_reason'] = 'validation_plateau'; break
    save_predictor(args.root/'last.pt', body, optimizer, update, parent['identity'], training)
    summary.update(status='training_complete', updates=update, added_updates=update-start,
        elapsed_s=time.monotonic()-begun, stop_reason=summary.get('stop_reason', 'maximum_updates'))
    write_json(args.root/'training.json', summary)
    selected, selected_meta = load_predictor(args.root/'best.pt')
    assert selected_meta['update'] == summary['best_update']
    assert metric(selected, batches['validation']) == next(h['validation'] for h in summary['history']
                                                        if h['update'] == summary['best_update'])
    write_json(args.root/'frozen_selection.json', {'update': selected_meta['update'],
        'checkpoint_sha256': selected_meta['checkpoint_sha256'], 'selection_before_test': True})
    old, old_meta = predictor_identity(args.source/'body/best.pt')
    original, original_meta = load_predictor(args.parent)
    result = {'direct_identity': selected_meta, 'iterative_identity': old_meta,
              'parent_direct_identity': original_meta, 'groups': {}}
    groups = {'all': lambda e: True, **{s: (lambda e, s=s: e.get('family') == s)
              for s in ('hold', 'ramp', 'jitter', 'policy')},
              'open_loop': lambda e: e.get('family') in ('independent', 'reciprocal', 'common', 'multisine')}
    for group, match in groups.items():
        subset = {**meta, 'episodes': [e for e in meta['episodes'] if match(e)]}
        ids, x, u, actual = examples(data, subset, 'test')
        initial = x[:, :20]
        step = data['next_observations'][ids][:, meta['contract']['physical_indices']]
        result['groups'][group] = {label: physical_errors(predicted, actual, initial, step, parent['identity']['output_names'])
            for label, predicted in (
                ('direct', predictions(selected, x, u)), ('parent150k', predictions(original, x, u)),
                ('iterative', predictions(old, x, u, iterative=True)), ('persistence', initial))}
    write_json(args.root/'test.json', result)
    assert hashlib.sha256(args.parent.read_bytes()).hexdigest() == parent['checkpoint_sha256']
    summary.update(status='complete', test=result['groups']['all'], strict_reload=True,
                   parent_unchanged=True)
    write_json(args.root/'training.json', summary)
    print(json.dumps({'status': 'complete', 'best_update': summary['best_update'],
        'added_updates': summary['added_updates'], 'stop_reason': summary['stop_reason'],
        'test_q60_deg': summary['test']['direct']['angle_rmse_deg'],
        'test_v60_deg_s': summary['test']['direct']['velocity_rmse_deg_s']}), flush=True)


if __name__ == '__main__':
    main()
