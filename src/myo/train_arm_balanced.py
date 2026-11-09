"""Run the broader-data predictor experiment, then the unchanged frozen-feature PPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .arm_balanced import ROOT, OLD, collect, train_predictor
from .arm_learning import load_arm
from .arm_precision import evaluate_arm, rank, TEST_CASES
from .train_arm_precision import train_policy
from .train_vertical_ppo import write_json


def evaluate(root):
    root = Path(root); folder = root/'frozen'
    summary = json.loads((folder/'summary.json').read_text())
    if summary['status'] != 'training_complete': raise ValueError('Finish PPO before test')
    references = {k: json.loads((OLD/k/'summary.json').read_text()) for k in ('none', 'frozen')}
    for previous in references.values():
        assert previous['initial_private_sha256'] == summary['initial_private_sha256']
        assert previous['initial_torch_rng_sha256'] == summary['initial_torch_rng_sha256']
        assert previous['training']['ppo'] == summary['training']['ppo']
        assert previous['training']['train_cases'] == summary['training']['train_cases']
        assert previous['contract'] == summary['contract']
    write_json(root/'frozen_selection.json', {'new_predictor_ppo': {'steps': summary['best_steps'],
        'checkpoint': json.loads((folder/'best.zip.metadata.json').read_text())},
        'reference_trials': {k: {'steps': s['best_steps'], 'status': s['status']} for k, s in references.items()}})
    models = {}
    for label in ('initial', 'best', 'last'):
        model, env, _ = load_arm(folder/f'{label}.zip'); env.close(); models[label] = model
    assert evaluate_arm(models['best']) == summary['best_validation']
    assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
    for name, value in models['initial'].policy.state_dict().items():
        if name.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_')):
            torch.testing.assert_close(value, models['last'].policy.state_dict()[name], rtol=0, atol=0)
    for label in ('best', 'last'):
        result = evaluate_arm(models[label], TEST_CASES, samples=True)
        write_json(folder/f'{label}_test.json', result)
        for row in result['cases']: row.pop('samples')
        summary[label+'_test'] = result
    summary['status'] = 'complete'
    write_json(folder/'summary.json', summary)
    write_json(root/'evaluation_audit.json', {'status': 'passed', 'strict_loads': 3,
        'unchanged_frozen_encoder_head_normalization': True, 'same_private_initialization_and_rng': True,
        'same_ppo_and_task_contract_as_original_trials': True})
    print(json.dumps({'test': summary['best_test'], 'last': summary['last_test']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('collect', 'predictor', 'train', 'evaluate'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--updates', type=int, default=100000)
    p.add_argument('--steps', type=int, default=131072)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(args.threads)
    if args.mode == 'collect': collect(args.root, args.workers)
    elif args.mode == 'predictor': train_predictor(args.root, args.updates)
    elif args.mode == 'evaluate': evaluate(args.root)
    else:
        if args.steps <= 0 or args.steps % 8192: p.error('Use complete8192-transition budgets')
        train_policy(args.root, 'frozen', args.steps)


if __name__ == '__main__': main()
