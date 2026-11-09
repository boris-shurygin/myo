"""Matched-budget late learning-rate comparisons with and without frozen features."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule

from myo.arm_conditioning import conditioning_identity, load_conditioned_arm
from myo.arm_learning import load_arm, save_arm
from myo.arm_precision import evaluate_arm, rank, TEST_CASES
from myo.train_arm_conditioning import balanced_goals
from myo.train_arm_precision import private_hash
from myo.train_vertical_ppo import write_json


ROOT = Path('runs/arm_predictor_lr')
TRIALS = {
    'none_1e4': ('runs/arm_continuation/best.zip', 1e-4),
    'none_3e5': ('runs/arm_continuation/best.zip', 3e-5),
    'frozen_1e4': ('runs/arm_conditioning/gain_025/best.zip', 1e-4),
    'frozen_3e5': ('runs/arm_conditioning/gain_025/best.zip', 3e-5),
}


def load_checkpoint(path):
    header = json.loads(Path(str(path)+'.metadata.json').read_text())
    loader = load_conditioned_arm if 'conditioning' in header['training'] else load_arm
    return loader(path)


def override_learning_rate(model, training, learning_rate):
    """Preserve weights/moments while updating the schedule and declared identity."""
    training = copy.deepcopy(training)
    model.learning_rate = learning_rate
    model.lr_schedule = FloatSchedule(learning_rate)
    for group in model.policy.optimizer.param_groups:
        group['lr'] = learning_rate
    training['ppo']['learning_rate'] = learning_rate
    if 'conditioning' in training:
        training['conditioning'] = conditioning_identity(learning_rate, model.policy.feature_gain_spec)
    return training


def frozen_state(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()
            if k.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_'))
            or k == 'mlp_extractor.feature_gain'}


def train(root, trial, budget):
    folder = root/trial
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'checkpoints').mkdir()
    parent_name, lr = TRIALS[trial]
    parent = Path(parent_name)
    model, env, meta = load_checkpoint(parent)
    initial_private = private_hash(model.policy)
    initial_frozen = frozen_state(model)
    source_steps, source_optimizer = model.num_timesteps, model.optimizer_steps
    assert (model.n_steps, model.batch_size, model.n_epochs) == (1024, 128, 10)
    training = override_learning_rate(model, meta['training'], lr)
    training.update(previous_continuation=training.get('continuation'),
        continuation={'schema': 'myo.arm_predictor_lr.v1', 'parent': str(parent),
            'parent_sha256': meta['checkpoint_sha256'], 'parent_steps': source_steps,
            'additional_transitions': budget, 'learning_rate_override': lr,
            'environment_and_rng_restart': True, 'sampling_seed': 0,
            'optimizer_moments_preserved': True, 'selection': 'original validation failure/pass/RMSE rank'},
        maximum_transitions=source_steps+budget,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    model.set_env(Monitor(env))
    model.set_random_seed(0)
    model.last_audit = {}
    summary = {'status': 'running', 'trial': trial, 'training': training,
        'contract': env.contract(), 'source_steps': source_steps,
        'source_optimizer_steps': source_optimizer, 'initial_private_sha256': initial_private,
        'history': [], 'updates': []}
    save_arm(model, env, folder/'initial.zip', training)
    # Verify checkpoint identity, frozen feature gain and the overridden LR before training.
    with torch.random.fork_rng(devices=[]):
        restored, checked, _ = load_checkpoint(folder/'initial.zip')
    assert private_hash(restored.policy) == initial_private
    assert restored.optimizer_steps == source_optimizer
    assert restored.learning_rate == lr and restored.lr_schedule(1.) == lr
    assert all(group['lr'] == lr for group in restored.policy.optimizer.param_groups)
    checked.close()
    del restored
    started = time.monotonic()
    best = None

    def check():
        nonlocal best
        result = evaluate_arm(model)
        if best is None or rank(result) < best:
            best = rank(result)
            summary.update(best_steps=model.num_timesteps, best_validation=result)
            save_arm(model, env, folder/'best.zip', training)
        assert all(torch.equal(v, model.policy.state_dict()[k]) for k, v in initial_frozen.items())
        save_arm(model, env, folder/'checkpoints'/f'step_{model.num_timesteps}.zip', training)
        summary['history'].append({'steps': model.num_timesteps,
            'added_transitions': model.num_timesteps-source_steps,
            'validation': result, 'goal_balanced_validation': balanced_goals(result),
            'optimizer_steps': model.optimizer_steps})
        summary.update(steps=model.num_timesteps, elapsed_s=time.monotonic()-started)
        write_json(folder/'summary.json', summary)
        print(json.dumps({'trial': trial, 'added': model.num_timesteps-source_steps,
            'validation_deg': result['score_deg'], 'passed': result['passed'],
            'best_steps': summary['best_steps'],
            'sgd_added': model.optimizer_steps-source_optimizer}), flush=True)

    check()
    while model.num_timesteps-source_steps < budget:
        model.learn(1024, reset_num_timesteps=False)
        summary['updates'].append({'steps': model.num_timesteps, **model.last_audit})
        if (model.num_timesteps-source_steps) % 8192 == 0:
            check()
    save_arm(model, env, folder/'last.zip', training)
    summary.update(status='training_complete', stop_reason='fixed_equal_budget',
        added_transitions=model.num_timesteps-source_steps,
        added_optimizer_steps=model.optimizer_steps-source_optimizer,
        elapsed_s=time.monotonic()-started, frozen_predictor_unchanged=True)
    write_json(folder/'summary.json', summary)
    env.close()
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == meta['checkpoint_sha256']
    print(json.dumps({'trial': trial, 'status': summary['status'],
        'best_validation': summary['best_validation']['score_deg']}), flush=True)


def evaluate(root):
    summaries = {t: json.loads((root/t/'summary.json').read_text()) for t in TRIALS}
    if not all(s['status'] in ('training_complete', 'complete') for s in summaries.values()):
        raise ValueError('Complete all four branches before freezing selection and opening test')
    assert len({s['added_transitions'] for s in summaries.values()}) == 1
    # Freeze all choices jointly. Never choose checkpoints or extend training from test results.
    frozen = {t: {label: json.loads((root/t/(label+'.zip.metadata.json')).read_text())['checkpoint_sha256']
                  for label in ('initial', 'best', 'last')} for t in TRIALS}
    write_json(root/'frozen_selection.json', frozen)
    for trial, summary in summaries.items():
        folder = root/trial
        initial, checked, _ = load_checkpoint(folder/'initial.zip')
        checked.close()
        reference = frozen_state(initial)
        assert private_hash(initial.policy) == summary['initial_private_sha256']
        assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
        for label in ('best', 'last'):
            model, checked, meta = load_checkpoint(folder/(label+'.zip'))
            checked.close()
            assert meta['checkpoint_sha256'] == frozen[trial][label]
            assert all(torch.equal(v, model.policy.state_dict()[k]) for k, v in reference.items())
            expected = summary['best_validation'] if label == 'best' else summary['history'][-1]['validation']
            assert evaluate_arm(model) == expected
            result = evaluate_arm(model, TEST_CASES, samples=True)
            write_json(folder/(label+'_test.json'), result)
            for row in result['cases']:
                row.pop('samples')
            summary[label+'_test'] = result
        summary.update(status='complete', strict_reload=True, selection_frozen_before_test=True)
        write_json(folder/'summary.json', summary)
        print(json.dumps({'trial': trial, 'test_rmse': summary['best_test']['score_deg'],
            'test_passed': summary['best_test']['passed']}), flush=True)
    report(root)


def report(root):
    rows = {t: json.loads((root/t/'summary.json').read_text()) for t in TRIALS}
    write_json(root/'comparison.json', rows)
    lines = ['# Предиктор и меньший learning rate', '',
        '| Ветка | Старт | Best | Val RMSE, ° | Pass | Best test, ° | Pass | Last test, ° | SGD |',
        '| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |']
    for t, s in rows.items():
        test = s.get('best_test', {})
        last = s.get('last_test', {})
        lines.append(f"| {t} | {s['source_steps']} | {s['best_steps']} | "
            f"{s['best_validation']['score_deg']:.3f} | {s['best_validation']['passed']}/9 | "
            f"{test.get('score_deg', float('nan')):.3f} | {test.get('passed', '-')}/11 | "
            f"{last.get('score_deg', float('nan')):.3f} | {s.get('added_optimizer_steps', '-')} |")
    for t, s in rows.items():
        lines += ['', f'## {t}', '', '| Новых переходов | RMSE, ° | Equal-goal, ° | Pass |',
                  '| ---: | ---: | ---: | --- |']
        for h in s['history']:
            lines.append(f"| {h['added_transitions']} | {h['validation']['score_deg']:.3f} | "
                f"{h['goal_balanced_validation']['mean_tail_rmse_deg']:.3f} | {h['validation']['passed']}/9 |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'evaluate', 'report'))
    p.add_argument('--trial', choices=tuple(TRIALS))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--steps', type=int, default=65536)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'train':
        if args.trial is None or args.steps <= 0 or args.steps % 8192:
            p.error('Use --trial and a positive complete8192-transition budget')
        train(args.root, args.trial, args.steps)
    elif args.mode == 'evaluate':
        evaluate(args.root)
    else:
        report(args.root)


if __name__ == '__main__':
    main()
