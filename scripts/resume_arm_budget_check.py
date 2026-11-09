"""Bounded unchanged-PPO continuation to test the insufficient-budget hypothesis."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3.common.monitor import Monitor

from myo.arm_learning import load_arm
from myo.arm_precision import evaluate_arm, rank, TEST_CASES
from myo.train_arm_conditioning import fit_policy
from myo.train_arm_precision import private_hash
from myo.train_vertical_ppo import write_json
from stable_baselines3.common.utils import FloatSchedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/arm_hypotheses/budget_none'))
    p.add_argument('--steps', type=int, default=32768)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--case-order', choices=('original', 'round_robin'), default='original')
    args = p.parse_args()
    if args.steps <= 0 or args.steps % 8192:
        p.error('Use complete8192-transition blocks')
    args.root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    parent = Path('runs/arm_preacher/none/best.zip')
    model, env, meta = load_arm(parent)
    parent_hash, parent_steps = meta['checkpoint_sha256'], model.num_timesteps
    initial_optimizer_steps = model.optimizer_steps
    initial_private = private_hash(model.policy)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            p.error('Learning rate must be positive')
        model.learning_rate = args.learning_rate
        model.lr_schedule = FloatSchedule(args.learning_rate)
        for group in model.policy.optimizer.param_groups:
            group['lr'] = args.learning_rate
    if args.case_order == 'round_robin':
        groups = {goal: [case for case in env.cases if case[1] == goal]
                  for goal in sorted({case[1] for case in env.cases})}
        original_cases = env.cases
        env.cases = tuple(cases[i] for i in range(max(map(len, groups.values())))
                          for cases in groups.values() if i < len(cases))
        assert len(env.cases) == len(original_cases) and set(env.cases) == set(original_cases)
    # Checkpoint has weights and optimizer, but not the old simulation/RNG stream.
    model.set_env(Monitor(env))
    model.set_random_seed(0)
    model.last_audit = {}
    continuation = {'parent': str(parent), 'parent_sha256': parent_hash,
        'parent_steps': parent_steps, 'additional_transitions': args.steps,
        'sampling_seed': 0, 'rng_restart': True, 'environment_restart': True,
        'optimizer_moments_preserved': True,
        'learning_rate_override': args.learning_rate, 'case_order': args.case_order,
        'scope': 'unchanged architecture, task and reward; optional single-factor LR or episode order'}
    training = {**meta['training'], 'maximum_transitions': parent_steps+args.steps,
        'ppo': {**meta['training']['ppo'], 'learning_rate': model.learning_rate},
        'train_cases': env.cases,
        'continuation': continuation,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary = {'status': 'running', 'experiment_trial': 'budget_none',
        'training': training, 'contract': env.contract(), 'history': [], 'updates': [],
        'initial_private_sha256': initial_private,
        'initial_optimizer_steps': initial_optimizer_steps}
    # fit_policy stores initial before advancing and keeps the original validation rank.
    summary = fit_policy(args.root, env, model, training, summary, args.steps, 'budget_none')
    restored, checked, initial_meta = load_arm(args.root/'initial.zip')
    assert private_hash(restored.policy) == initial_private
    assert restored.optimizer_steps == initial_optimizer_steps
    checked.close()
    selected = json.loads((args.root/'best.zip.metadata.json').read_text())
    write_json(args.root/'frozen_selection.json', {'steps': selected['steps'],
        'checkpoint_sha256': selected['checkpoint_sha256'], 'selection': 'original validation rank'})
    assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
    for label in ('best', 'last'):
        candidate, checked, _ = load_arm(args.root/f'{label}.zip')
        if label == 'best':
            assert evaluate_arm(candidate) == summary['best_validation']
        checked.close()
        result = evaluate_arm(candidate, TEST_CASES, samples=True)
        write_json(args.root/f'{label}_test.json', result)
        for row in result['cases']:
            row.pop('samples')
        summary[label+'_test'] = result
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_hash
    summary.update(status='complete', added_transitions=model.num_timesteps-parent_steps,
        added_optimizer_steps=model.optimizer_steps-initial_optimizer_steps,
        parent_unchanged=True, strict_reload=True)
    write_json(args.root/'summary.json', summary)
    print(json.dumps({'status': 'complete', 'added_transitions': summary['added_transitions'],
        'validation': summary['best_validation']['score_deg'],
        'passed': summary['best_validation']['passed'], 'test': summary['best_test']['score_deg'],
        'test_passed': summary['best_test']['passed']}), flush=True)


if __name__ == '__main__':
    main()
