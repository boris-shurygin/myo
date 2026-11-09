"""Matched common-parent PPO comparison of old/new/absent body features."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from myo.arm_feature_comparison import (KINDS, architecture_identity, frozen_digest,
    load_comparison, make_comparison)
from myo.arm_learning import load_arm, save_arm
from myo.arm_precision import ArmPrecisionEnv, evaluate_arm, rank, TEST_CASES
from myo.train_arm_conditioning import balanced_goals
from myo.train_vertical_ppo import write_json


ROOT = Path('runs/arm_anticipation_ppo')
PARENT = Path('runs/arm_continuation/best.zip')
PREDICTORS = {'none': None, 'old20': Path('runs/arm_preacher_balanced/body/best.pt'),
              'new60': Path('runs/arm_anticipation60_extended/best.pt')}


def train(root, kind, budget):
    folder = root/kind
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'checkpoints').mkdir()
    parent, checked, parent_meta = load_arm(PARENT)
    checked.close()
    env = ArmPrecisionEnv()
    model, predictor_meta = make_comparison(Monitor(env), parent, kind, PREDICTORS[kind])
    # The zero-readout insertion must preserve the complete starting controller.
    obs = []
    for case in env.cases:
        observation, _ = env.reset(seed=0, options={'case': case})
        obs.append(observation)
    obs = torch.as_tensor(np.asarray(obs))
    with torch.no_grad():
        assert torch.equal(parent.policy.predict_values(obs), model.policy.predict_values(obs))
        assert torch.equal(parent.policy.get_distribution(obs).distribution.concentration1,
                           model.policy.get_distribution(obs).distribution.concentration1)
        assert torch.equal(parent.policy.get_distribution(obs).distribution.concentration0,
                           model.policy.get_distribution(obs).distribution.concentration0)
    # Reset case order after the equivalence probe and restart the same RNG.
    env.next_case = 0
    model.set_random_seed(0)
    source_steps, source_sgd = model.num_timesteps, model.optimizer_steps
    ppo = {**parent_meta['training']['ppo'], 'learning_rate': model.learning_rate}
    training = {'trial': model.policy.feature_mode, 'seed': 0, 'ppo': ppo,
        'predictor': predictor_meta, 'feature_comparison': architecture_identity(model.policy),
        'common_parent': {'path': str(PARENT), 'sha256': parent_meta['checkpoint_sha256'],
            'steps': source_steps, 'optimizer_steps': source_sgd},
        'protocol': {'additional_transitions': budget, 'check_every': 8192,
            'selection': 'original failure/pass/RMSE validation rank, includes initial',
            'initial_function': 'parent unchanged; zero predictor readout columns in all branches',
            'optimizer': 'common private/tail moments preserved; feature columns cold',
            'rollout_rng': 'seed0 restarted once; continuous within branch',
            'predictor_training': 'none; encoder, head and normalization frozen',
            'test': 'all selections jointly frozen before historical test'},
        'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    initial_frozen = frozen_digest(model.policy)
    summary = {'status': 'running', 'kind': kind, 'training': training, 'contract': env.contract(),
        'source_steps': source_steps, 'source_optimizer_steps': source_sgd,
        'initial_function_equal_parent': True, 'history': [], 'updates': [],
        'ppo_parameters': sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)}
    save_arm(model, env, folder/'initial.zip', training)
    # Exercise strict identity before a long run; loading itself must not move rollout RNG.
    with torch.random.fork_rng(devices=[]):
        restored, restored_env, _ = load_comparison(folder/'initial.zip')
    restored_env.close()
    assert all(torch.equal(v, restored.policy.state_dict()[k]) for k, v in model.policy.state_dict().items())
    del restored, parent
    model.set_random_seed(0)
    started, best = time.monotonic(), None

    def check():
        nonlocal best
        result = evaluate_arm(model)
        if best is None or rank(result) < best:
            best = rank(result)
            summary.update(best_steps=model.num_timesteps, best_validation=result)
            save_arm(model, env, folder/'best.zip', training)
        assert frozen_digest(model.policy) == initial_frozen
        summary['history'].append({'steps': model.num_timesteps,
            'added_transitions': model.num_timesteps-source_steps, 'validation': result,
            'goal_balanced_validation': balanced_goals(result), 'optimizer_steps': model.optimizer_steps})
        save_arm(model, env, folder/'checkpoints'/f'step_{model.num_timesteps}.zip', training)
        summary.update(steps=model.num_timesteps, elapsed_s=time.monotonic()-started)
        write_json(folder/'summary.json', summary)
        print(json.dumps({'kind': kind, 'added': model.num_timesteps-source_steps,
            'validation_deg': result['score_deg'], 'passed': result['passed'],
            'best_steps': summary['best_steps'], 'sgd_added': model.optimizer_steps-source_sgd}), flush=True)

    check()
    while model.num_timesteps-source_steps < budget:
        model.learn(1024, reset_num_timesteps=False)
        summary['updates'].append({'steps': model.num_timesteps, **model.last_audit})
        if (model.num_timesteps-source_steps) % 8192 == 0:
            check()
    save_arm(model, env, folder/'last.zip', training)
    summary.update(status='training_complete', stop_reason='fixed_equal_budget',
        added_transitions=model.num_timesteps-source_steps,
        added_optimizer_steps=model.optimizer_steps-source_sgd,
        frozen_predictor_unchanged=True, elapsed_s=time.monotonic()-started)
    write_json(folder/'summary.json', summary)
    env.close()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == parent_meta['checkpoint_sha256']


def evaluate(root):
    summaries = {kind: json.loads((root/kind/'summary.json').read_text()) for kind in KINDS}
    assert all(s['status'] == 'training_complete' for s in summaries.values())
    assert len({s['added_transitions'] for s in summaries.values()}) == 1
    assert len({s['ppo_parameters'] for s in summaries.values()}) == 1
    assert len({s['training']['common_parent']['sha256'] for s in summaries.values()}) == 1
    selection = root/'frozen_selection.json'
    if selection.exists():
        raise FileExistsError(selection)
    write_json(selection, {kind: {label: json.loads((root/kind/(label+'.zip.metadata.json')).read_text())['checkpoint_sha256']
        for label in ('initial', 'best', 'last')} for kind in KINDS})
    for kind, summary in summaries.items():
        folder = root/kind
        assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
        for label in ('best', 'last'):
            model, env, _ = load_comparison(folder/(label+'.zip'))
            env.close()
            expected = summary['best_validation'] if label == 'best' else summary['history'][-1]['validation']
            assert evaluate_arm(model) == expected
            result = evaluate_arm(model, TEST_CASES, samples=True)
            write_json(folder/(label+'_test.json'), result)
            for case in result['cases']:
                case.pop('samples')
            summary[label+'_test'] = result
            summary[label+'_goal_balanced_test'] = balanced_goals(result)
        summary.update(status='complete', strict_reload=True, selection_frozen_before_test=True)
        write_json(folder/'summary.json', summary)
        print(json.dumps({'kind': kind, 'test_rmse': summary['best_test']['score_deg'],
            'passed': summary['best_test']['passed']}), flush=True)
    report(root)


def report(root):
    summaries = {kind: json.loads((root/kind/'summary.json').read_text()) for kind in KINDS}
    write_json(root/'comparison.json', summaries)
    lines = ['# PPO: общий старт, старый/новый предиктор и без предиктора', '',
        '| Ветка | Best шаг | Val, ° | Pass | Test, ° | Pass | Last test, ° | SGD |',
        '| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |']
    for kind, s in summaries.items():
        lines.append(f"| {kind} | {s['best_steps']} | {s['best_validation']['score_deg']:.3f} | "
            f"{s['best_validation']['passed']}/9 | {s['best_test']['score_deg']:.3f} | "
            f"{s['best_test']['passed']}/11 | {s['last_test']['score_deg']:.3f} | {s['added_optimizer_steps']} |")
    lines += ['', '| Старт → цель, ° | none | old20 | new60 |', '| --- | ---: | ---: | ---: |']
    for i, case in enumerate(TEST_CASES):
        values = []
        for s in summaries.values():
            row = s['best_test']['cases'][i]
            value = f"{row['tail_rmse_deg']:.3f}"
            values.append(value if row['passed'] else f'**{value}**')
        lines.append(f'| {case[0]}→{case[1]} | '+ ' | '.join(values)+' |')
    for kind, s in summaries.items():
        lines += ['', f'## {kind}', '', '| Добавлено переходов | Val, ° | Pass | SGD |',
                  '| ---: | ---: | --- | ---: |']
        for row in s['history']:
            lines.append(f"| {row['added_transitions']} | {row['validation']['score_deg']:.3f} | "
                f"{row['validation']['passed']}/9 | {row['optimizer_steps']-s['source_optimizer_steps']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'evaluate', 'report'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--kind', choices=KINDS)
    p.add_argument('--steps', type=int, default=65536)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'train':
        if args.kind is None or args.steps <= 0 or args.steps % 8192:
            p.error('Training requires --kind and positive complete8192-transition budget')
        train(args.root, args.kind, args.steps)
    else:
        (evaluate if args.mode == 'evaluate' else report)(args.root)


if __name__ == '__main__':
    main()
