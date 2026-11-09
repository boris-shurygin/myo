"""Continue the unchanged low-LR arm policy until validation stops improving."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from stable_baselines3.common.monitor import Monitor

from myo.arm_learning import load_arm, save_arm
from myo.arm_precision import evaluate_arm, rank, TEST_CASES
from myo.train_arm_conditioning import balanced_goals
from myo.train_arm_precision import private_hash
from myo.train_vertical_ppo import write_json


BLOCK_STEPS = 32768
CHECK_STEPS = 8192
PROBE_CASES = ((20, 75), (75, 20), (25, 105), (60, 60))


def progress_reference(result):
    return {'failures': result['failures'], 'passed': result['passed'],
            'balanced_rmse_deg': balanced_goals(result)['mean_tail_rmse_deg']}


def significant_progress(reference, candidate):
    if candidate['failures'] != reference['failures']:
        return candidate['failures'] < reference['failures']
    return (candidate['balanced_rmse_deg'] <= .99*reference['balanced_rmse_deg']
            or (candidate['passed'] > reference['passed']
                and candidate['balanced_rmse_deg'] <= reference['balanced_rmse_deg']))


def stochastic_probe(model, cases=PROBE_CASES, seeds=(0, 1)):
    """Evaluate in independent native envs without consuming the training RNG."""
    class Samples:
        def predict(self, obs, deterministic=True):
            return model.predict(obs, deterministic=False)

    rows = []
    with torch.random.fork_rng(devices=[]):
        for case in cases:
            for seed in seeds:
                torch.manual_seed(seed)
                result = evaluate_arm(Samples(), (case,))
                rows.append({'seed': seed, **result['cases'][0]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/arm_continuation'))
    p.add_argument('--parent', type=Path, default=Path('runs/arm_hypotheses/lr_1e4_none/best.zip'))
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root/'checkpoints').mkdir()
    torch.set_num_threads(1)
    model, env, parent_meta = load_arm(args.parent)
    assert model.policy.feature_mode == 'none' and model.learning_rate == 1e-4
    assert (model.n_steps, model.batch_size, model.n_epochs) == (1024, 128, 10)
    source_steps, source_optimizer_steps = model.num_timesteps, model.optimizer_steps
    source_weights = private_hash(model.policy)
    model.set_env(Monitor(env)); model.set_random_seed(0); model.last_audit = {}
    protocol = {'schema': 'myo.arm_continuation.v1', 'block_steps': BLOCK_STEPS,
        'check_steps': CHECK_STEPS, 'patience_blocks': 2,
        'progress': '>=1% cumulative balanced RMSE improvement, or more passes without balanced regression; fewer failures first',
        'selection': 'original failure/pass/RMSE rank; balanced_best is a separate diagnostic archive',
        'success': 'all nine native validation cases pass', 'learning_rate': 1e-4,
        'reset_between_blocks': False, 'rng_between_blocks': 'continuous; probes fork/restore Torch RNG',
        'initial_sampling_seed': 0, 'initial_environment_and_rng_restart': True,
        'probe_cases': PROBE_CASES, 'probe_seeds': (0, 1)}
    training = {**parent_meta['training'], 'maximum_transitions': None,
        'previous_continuation': parent_meta['training'].get('continuation'),
        'continuation': {'parent': str(args.parent), 'parent_sha256': parent_meta['checkpoint_sha256'],
                         'parent_steps': source_steps, 'protocol': protocol},
        'stop': protocol, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary = {'status': 'running', 'protocol': protocol, 'training': training,
        'contract': env.contract(), 'source_steps': source_steps,
        'source_optimizer_steps': source_optimizer_steps, 'initial_private_sha256': source_weights,
        'history': [], 'updates': [], 'blocks': []}
    save_arm(model, env, args.root/'initial.zip', training)
    best_rank, best_balanced, reference = None, None, None
    block_progress = False
    started = time.monotonic()

    def check():
        nonlocal best_rank, best_balanced, reference, block_progress
        result = evaluate_arm(model)
        candidate = progress_reference(result)
        if reference is None or significant_progress(reference, candidate):
            reference = candidate
            block_progress = True
        if best_rank is None or rank(result) < best_rank:
            best_rank = rank(result)
            summary.update(best_steps=model.num_timesteps, best_validation=result)
            save_arm(model, env, args.root/'best.zip', training)
        brank = (result['failures'], candidate['balanced_rmse_deg'])
        if best_balanced is None or brank < best_balanced:
            best_balanced = brank
            summary.update(balanced_best_steps=model.num_timesteps, balanced_best_validation=result)
            save_arm(model, env, args.root/'balanced_best.zip', training)
        save_arm(model, env, args.root/'checkpoints'/f'step_{model.num_timesteps}.zip', training)
        summary['history'].append({'steps': model.num_timesteps, 'validation': result,
            'goal_balanced_validation': balanced_goals(result), 'progress_reference': reference.copy(),
            'optimizer_steps': model.optimizer_steps})
        summary.update(steps=model.num_timesteps, elapsed_s=time.monotonic()-started)
        write_json(args.root/'summary.json', summary)
        print(json.dumps({'kind': 'validation', 'steps': model.num_timesteps,
            'rmse_deg': result['score_deg'], 'balanced_deg': candidate['balanced_rmse_deg'],
            'passed': result['passed'], 'best_steps': summary['best_steps'],
            'optimizer_steps': model.optimizer_steps}), flush=True)
        return result

    initial_result = check()
    initial_probe = stochastic_probe(model)
    summary['stochastic_initial'] = initial_probe
    write_json(args.root/'summary.json', summary)
    stagnant, block = 0, 0
    success = initial_result['passed'] == initial_result['total'] and not initial_result['failures']
    while not success and stagnant < 2:
        block += 1
        block_progress = False
        start_steps, start_optimizer = model.num_timesteps, model.optimizer_steps
        while model.num_timesteps-start_steps < BLOCK_STEPS:
            model.learn(1024, reset_num_timesteps=False)
            summary['updates'].append({'steps': model.num_timesteps, **model.last_audit})
            if (model.num_timesteps-start_steps) % 4096 == 0:
                print(json.dumps({'kind': 'training', 'block': block, 'steps': model.num_timesteps,
                    'block_optimizer_steps': model.optimizer_steps-start_optimizer}), flush=True)
            if (model.num_timesteps-start_steps) % CHECK_STEPS == 0:
                result = check()
                success = result['passed'] == result['total'] and not result['failures']
                if success:
                    break
        stagnant = 0 if block_progress else stagnant+1
        probe = stochastic_probe(model)
        summary['blocks'].append({'block': block, 'steps': model.num_timesteps,
            'added_transitions': model.num_timesteps-start_steps,
            'optimizer_steps': model.optimizer_steps-start_optimizer,
            'significant_progress': block_progress, 'stagnant_blocks': stagnant,
            'stochastic': probe, 'progress_reference': reference.copy()})
        summary.update(elapsed_s=time.monotonic()-started, stagnant_blocks=stagnant)
        save_arm(model, env, args.root/'last.zip', training)
        write_json(args.root/'summary.json', summary)
        print(json.dumps({'kind': 'block_end', 'block': block, 'steps': model.num_timesteps,
            'significant_progress': block_progress, 'stagnant_blocks': stagnant}), flush=True)
    stop_reason = 'validation_all_cases_passed' if success else 'two_blocks_without_significant_progress'
    save_arm(model, env, args.root/'last.zip', training)
    summary.update(status='training_complete', stop_reason=stop_reason,
        added_transitions=model.num_timesteps-source_steps,
        added_optimizer_steps=model.optimizer_steps-source_optimizer_steps,
        elapsed_s=time.monotonic()-started)
    write_json(args.root/'summary.json', summary)
    env.close()
    selection = {}
    for label in ('best', 'balanced_best', 'last'):
        meta = json.loads((args.root/(label+'.zip.metadata.json')).read_text())
        selection[label] = {'steps': meta['steps'], 'sha256': meta['checkpoint_sha256']}
    write_json(args.root/'frozen_selection.json', selection)
    # Test is not used during block stopping or checkpoint selection.
    for label in ('best', 'balanced_best', 'last'):
        candidate, checked, _ = load_arm(args.root/(label+'.zip'))
        expected = (summary['best_validation'] if label == 'best' else
                    summary['balanced_best_validation'] if label == 'balanced_best' else
                    summary['history'][-1]['validation'])
        assert evaluate_arm(candidate) == expected
        checked.close()
        result = evaluate_arm(candidate, TEST_CASES, samples=True)
        write_json(args.root/(label+'_test.json'), result)
        for row in result['cases']:
            row.pop('samples')
        summary[label+'_test'] = result
        summary[label+'_stochastic'] = stochastic_probe(candidate)
    initial, checked, _ = load_arm(args.root/'initial.zip')
    assert private_hash(initial.policy) == source_weights
    assert initial.optimizer_steps == source_optimizer_steps
    checked.close()
    assert hashlib.sha256(args.parent.read_bytes()).hexdigest() == parent_meta['checkpoint_sha256']
    assert rank(summary['best_validation']) == min(rank(h['validation']) for h in summary['history'])
    summary.update(status='complete', strict_reload=True, parent_unchanged=True)
    write_json(args.root/'summary.json', summary)
    print(json.dumps({'kind': 'complete', 'stop_reason': stop_reason, 'blocks': block,
        'added_transitions': summary['added_transitions'], 'best_steps': summary['best_steps'],
        'best_test_rmse': summary['best_test']['score_deg'], 'best_test_passed': summary['best_test']['passed']}), flush=True)


if __name__ == '__main__':
    main()
