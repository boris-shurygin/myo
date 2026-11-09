"""One-factor learning-rate and frozen-feature-scale comparisons on MyoArm."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .arm_conditioning import make_conditioned_model, conditioning_identity, load_conditioned_arm
from .arm_learning import predictor_identity, save_arm, load_arm
from .arm_precision import ArmPrecisionEnv, TRAIN_CASES, VALIDATION_CASES, TEST_CASES, evaluate_arm, rank
from .train_arm_precision import private_hash
from .train_vertical_ppo import write_json


ROOT = Path('runs/arm_conditioning')
REFERENCE = Path('runs/arm_preacher_balanced')
TRIALS = {'lr_1e4': (1e-4, 1.), 'lr_3e5': (3e-5, 1.), 'gain_025': (3e-4, .25)}


def balanced_goals(result):
    goals = sorted({r['target_deg'] for r in result['cases']})
    rows = [[r for r in result['cases'] if r['target_deg'] == goal] for goal in goals]
    return {'mean_tail_rmse_deg': float(np.mean([np.mean([r['tail_rmse_deg'] for r in group]) for group in rows])),
            'mean_pass_fraction': float(np.mean([np.mean([r['passed'] for r in group]) for group in rows])),
            'goals': goals, 'scope': 'equal weight per goal; diagnostic, original selection rank unchanged'}


def train(root, trial, budget):
    folder = Path(root)/trial
    folder.mkdir(parents=True, exist_ok=False)
    learning_rate, gain = TRIALS[trial]
    env = ArmPrecisionEnv()
    model = make_conditioned_model(Monitor(env), learning_rate, gain)
    with torch.random.fork_rng(devices=[]):
        body, identity = predictor_identity(REFERENCE/'body/best.pt')
    assert bool(body.delta_calibrated)
    body.copy_into_policy(model.policy)
    reference = json.loads((REFERENCE/'frozen/summary.json').read_text())
    initial_private = private_hash(model.policy)
    initial_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert initial_private == reference['initial_private_sha256']
    assert initial_rng == reference['initial_torch_rng_sha256']
    assert env.contract() == reference['contract']
    training = {**reference['training'], 'maximum_transitions': budget,
        'ppo': {**reference['training']['ppo'], 'learning_rate': learning_rate},
        'predictor': identity, 'conditioning': conditioning_identity(learning_rate, gain),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')}}
    summary = {'status': 'running', 'experiment_trial': trial, 'training': training, 'contract': env.contract(),
        'initial_private_sha256': initial_private, 'initial_torch_rng_sha256': initial_rng,
        'parameter_count': sum(p.numel() for p in model.policy.parameters()),
        'ppo_parameter_count': sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
        'history': [], 'updates': [], 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    return fit_policy(folder, env, model, training, summary, budget, trial)


def fit_policy(folder, env, model, training, summary, budget, trial):
    """The unchanged rollout/validation/selection loop shared by the depth trial."""
    save_arm(model, env, folder/'initial.zip', training)
    started = time.monotonic()
    best, progress_reference, progress_step = None, None, 0

    class Audit(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_start(self):
            if self.model.last_audit:
                summary['updates'].append({'steps': self.num_timesteps, **self.model.last_audit})

    def check():
        nonlocal best, progress_reference, progress_step
        result = evaluate_arm(model)
        current = rank(result)
        if best is None or current < best:
            best = current
            summary.update(best_steps=model.num_timesteps, best_validation=result)
            save_arm(model, env, folder/'best.zip', training)
        if (progress_reference is None or current[:2] < progress_reference[:2]
                or (current[:2] == progress_reference[:2] and current[2] < progress_reference[2]*.99)):
            progress_reference, progress_step = current, model.num_timesteps
        summary['history'].append({'steps': model.num_timesteps, 'validation': result,
            'goal_balanced_validation': balanced_goals(result), 'optimizer_steps': model.optimizer_steps})
        summary['elapsed_s'] = time.monotonic()-started
        write_json(folder/'summary.json', summary)
        print(json.dumps({'trial': trial, 'steps': model.num_timesteps, 'validation_deg': result['score_deg'],
            'passed': result['passed'], 'best_steps': summary['best_steps'],
            'optimizer_steps': model.optimizer_steps}), flush=True)

    check()
    for _ in range(budget//8192):
        model.learn(8192, reset_num_timesteps=False, callback=Audit())
        check()
        if model.num_timesteps >= 65536 and model.num_timesteps-progress_step >= 49152:
            summary['stop_reason'] = 'validation_plateau'
            break
    summary['updates'].append({'steps': model.num_timesteps, **model.last_audit})
    save_arm(model, env, folder/'last.zip', training)
    summary.update(status='training_complete', steps=model.num_timesteps, optimizer_steps=model.optimizer_steps,
        stop_reason=summary.get('stop_reason', 'maximum_transitions'), elapsed_s=time.monotonic()-started)
    write_json(folder/'summary.json', summary)
    env.close()
    return summary


def evaluate(root):
    root = Path(root)
    summaries = {trial: json.loads((root/trial/'summary.json').read_text()) for trial in TRIALS}
    if not all(s['status'] == 'training_complete' for s in summaries.values()):
        raise ValueError('Finish every declared arm conditioning run before test')
    write_json(root/'frozen_selection.json', {trial: {'steps': s['best_steps'],
        'sha256': json.loads((root/trial/'best.zip.metadata.json').read_text())['checkpoint_sha256']}
        for trial, s in summaries.items()})
    body, _ = predictor_identity(REFERENCE/'body/best.pt')
    reference_model, env, _ = load_arm(REFERENCE/'frozen/initial.zip')
    env.close()
    for trial, summary in summaries.items():
        folder = root/trial
        models = {}
        for label in ('initial', 'best', 'last'):
            model, env, _ = load_conditioned_arm(folder/f'{label}.zip')
            env.close()
            models[label] = model
            for a, b in zip(body.encoder.parameters(), model.policy.mlp_extractor.encoder.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(body.head.parameters(), model.policy.prediction_head.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ('delta_mean', 'delta_std', 'delta_calibrated'):
                torch.testing.assert_close(getattr(body, name), getattr(model.policy, name), rtol=0, atol=0)
        assert private_hash(models['initial'].policy) == private_hash(reference_model.policy)
        assert evaluate_arm(models['best']) == summary['best_validation']
        assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
        for label in ('best', 'last'):
            result = evaluate_arm(models[label], TEST_CASES, samples=True)
            write_json(folder/f'{label}_test.json', result)
            for row in result['cases']:
                row.pop('samples')
            summary[label+'_test'] = result
            summary[label+'_goal_balanced_test'] = balanced_goals(result)
        summary.update(status='complete', audit='passed: strict loads, frozen predictor, initialization and selection')
        write_json(folder/'summary.json', summary)
        print(json.dumps({'trial': trial, 'test': summary['best_test']['score_deg'],
            'passed': summary['best_test']['passed']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--trial', choices=tuple(TRIALS))
    parser.add_argument('--steps', type=int, default=131072)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'evaluate':
        evaluate(args.root)
    else:
        if args.trial is None or args.steps <= 0 or args.steps % 8192:
            parser.error('Training requires --trial and a positive complete8192-transition budget')
        train(args.root, args.trial, args.steps)


if __name__ == '__main__':
    main()
