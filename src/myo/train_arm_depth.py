"""Compare four versus two post-concatenation layers at fixed gain and PPO settings."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor

from .arm_depth import make_deep_model, depth_identity, load_deep_arm
from .arm_conditioning import load_conditioned_arm
from .arm_learning import save_arm
from .arm_precision import ArmPrecisionEnv, TEST_CASES, evaluate_arm, rank
from .train_arm_conditioning import fit_policy, balanced_goals
from .train_vertical_ppo import write_json


ROOT = Path('runs/arm_depth')
REFERENCE = Path('runs/arm_conditioning/gain_025')


def train(root, budget):
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=False)
    reference = json.loads((REFERENCE/'summary.json').read_text())
    env = ArmPrecisionEnv()
    model = make_deep_model(Monitor(env))
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        original, checked, initial_meta = load_conditioned_arm(REFERENCE/'initial.zip')
        checked.close()
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    state = original.policy.state_dict()
    current = model.policy.state_dict()
    # Transfer only the frozen predictor/head/normalization; common private weights
    # already match the original seeded construction without copying trained ones.
    for name, value in state.items():
        if name.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_')):
            current[name].copy_(value)
    assert all(torch.equal(value, current[name]) for name, value in state.items())
    initial_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert initial_rng == reference['initial_torch_rng_sha256']
    assert env.contract() == reference['contract']
    training = {**reference['training'], 'maximum_transitions': budget, 'depth': depth_identity(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'common_initial_checkpoint_sha256': initial_meta['checkpoint_sha256'],
        'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')}}
    summary = {'status': 'running', 'training': training, 'contract': env.contract(),
        'common_initial_state_equal': True, 'initial_torch_rng_sha256': initial_rng,
        'parameter_count': sum(p.numel() for p in model.policy.parameters()),
        'ppo_parameter_count': sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
        'history': [], 'updates': [], 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    fit_policy(folder, env, model, training, summary, budget, 'post_depth4')


def evaluate(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    if summary['status'] != 'training_complete':
        raise ValueError('Complete depth training before test')
    selection = root/'frozen_selection.json'
    if selection.exists():
        raise FileExistsError(selection)
    write_json(selection, {'best_steps': summary['best_steps'],
        'sha256': json.loads((root/'best.zip.metadata.json').read_text())['checkpoint_sha256']})
    models = {}
    original, env, _ = load_conditioned_arm(REFERENCE/'initial.zip')
    env.close()
    for label in ('initial', 'best', 'last'):
        model, env, _ = load_deep_arm(root/f'{label}.zip')
        env.close()
        models[label] = model
        for name, value in original.policy.state_dict().items():
            if label == 'initial' or name.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_')):
                torch.testing.assert_close(value, model.policy.state_dict()[name], rtol=0, atol=0)
    assert evaluate_arm(models['best']) == summary['best_validation']
    assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
    for label in ('best', 'last'):
        result = evaluate_arm(models[label], TEST_CASES, samples=True)
        write_json(root/f'{label}_test.json', result)
        for row in result['cases']:
            row.pop('samples')
        summary[label+'_test'] = result
        summary[label+'_goal_balanced_test'] = balanced_goals(result)
    summary.update(status='complete', audit='passed: common initialization, frozen predictor, strict depth loads, selection')
    write_json(root/'summary.json', summary)
    print(json.dumps({'best_test': summary['best_test'], 'last_test': summary['last_test']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--steps', type=int, default=131072)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'evaluate':
        evaluate(args.root)
    else:
        if args.steps <= 0 or args.steps % 8192:
            parser.error('Use positive complete8192-transition budgets')
        train(args.root, args.steps)


if __name__ == '__main__':
    main()
