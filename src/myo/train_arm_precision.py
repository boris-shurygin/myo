"""Body exploration, predictor pretraining and matched fixed-task PPO on MyoArm."""
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

from .arm_precision import ArmPrecisionEnv, TRAIN_CASES, VALIDATION_CASES, TEST_CASES, evaluate_arm, rank
from .arm_learning import ArmBodyPredictor, make_model, predictor_identity, save_arm, load_arm, TRIALS
from .body_predictor import smooth_knots
from .train_vertical_ppo import write_json


def commands(family, count, seed, limit):
    rng = np.random.default_rng(seed)
    times = np.arange(count)*.02
    if family == 'independent':
        u = np.stack([smooth_knots(rng, times, 0., limit) for _ in range(6)], -1)
    elif family == 'reciprocal':
        difference = smooth_knots(rng, times, -.45*limit, .45*limit)
        u = np.stack([.5*limit+difference*(1 if i < 3 else -1)
                      +smooth_knots(rng, times, -.05*limit, .05*limit) for i in range(6)], -1)
    elif family == 'common':
        common = smooth_knots(rng, times, 0., limit)
        u = np.stack([common+smooth_knots(rng, times, -.08*limit, .08*limit) for _ in range(6)], -1)
    elif family == 'multisine':
        frequency = rng.uniform(.1, 1.5, (6, 3)); phase = rng.uniform(0, 2*np.pi, (6, 3))
        u = .5*limit+limit/6*np.sin(2*np.pi*times[:, None, None]*frequency+phase).sum(-1)
    else:
        raise ValueError('Unknown motor family')
    ramp = np.clip(times/.2, 0, 1); ramp = ramp*ramp*(3-2*ramp)
    return ((1-ramp[:, None])*.03+ramp[:, None]*np.clip(u, 0, limit)).astype(np.float32)


def specifications():
    result = []
    for split, count, positions in [('train', 16, (20, 50, 80, 110, 130)),
                                     ('validation', 6, (25, 65, 105, 125)),
                                     ('test', 6, (30, 70, 100, 120))]:
        for family_index, family in enumerate(('independent', 'reciprocal', 'common', 'multisine')):
            for i in range(count):
                result.append({'split': split, 'family': family, 'position_deg': positions[(i+family_index) % len(positions)],
                    'capacity': (1., .8)[i % 2], 'limit': (.06, .12, .24, .4)[i % 4],
                    'seed': 20261108+len(result)})
    return result


def collect(root, specs=None, steps=200):
    folder = Path(root)/'body'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'dataset.npz').exists():
        raise FileExistsError(folder/'dataset.npz')
    data = {key: [] for key in ('observations', 'actions', 'next_observations', 'valid')}
    episodes = []
    env = ArmPrecisionEnv()
    start_time = time.monotonic()
    for spec in specifications() if specs is None else specs:
        obs, _ = env.reset(seed=spec['seed'], options={'case': (spec['position_deg'], spec['position_deg']),
                                                     'capacity': spec['capacity']})
        start = len(data['observations']); failed = False
        for u in commands(spec['family'], steps, spec['seed'], spec['limit']):
            following, _, failed, truncated, _ = env.step(u)
            for key, value in zip(data, (obs, u, following, not failed), strict=True):
                data[key].append(value)
            obs = following
            if failed or truncated:
                break
        episodes.append({**spec, 'start': start, 'stop': len(data['observations']), 'failure': bool(failed)})
    path = folder/'dataset.npz'
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in data.items()})
    metadata = {'contract': env.contract(), 'episodes': episodes, 'elapsed_s': time.monotonic()-start_time,
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'collection': 'smooth open-loop 6-channel commands; retain all episodes; final failed pair masked',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(folder/'dataset.json', metadata)
    env.close()
    print(json.dumps({'episodes': len(episodes), 'transitions': len(data['valid']),
                      'valid': sum(data['valid']), 'failures': sum(e['failure'] for e in episodes),
                      'elapsed_s': metadata['elapsed_s']}), flush=True)


def load_data(root):
    folder = Path(root)/'body'
    meta = json.loads((folder/'dataset.json').read_text())
    env = ArmPrecisionEnv()
    if meta['contract'] != env.contract():
        raise ValueError('Arm exploration data contract mismatch')
    env.close()
    if meta['dataset_sha256'] != hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest():
        raise ValueError('Arm exploration dataset hash mismatch')
    return dict(np.load(folder/'dataset.npz')), meta


def starts(data, meta, split, horizon=1):
    groups = []
    for episode in meta['episodes']:
        if episode['split'] != split:
            continue
        candidates = np.arange(episode['start'], episode['stop']-horizon+1, dtype=np.int64)
        if len(candidates):
            groups.extend(candidates[np.all(np.stack([data['valid'][candidates+k] for k in range(horizon)]), axis=0)])
    return np.asarray(groups, dtype=np.int64)


@torch.no_grad()
def forecast(body, data, meta, split, horizon=1):
    ids = starts(data, meta, split, horizon)
    if not len(ids):
        raise ValueError('No complete predictor windows in split')
    physical = meta['contract']['physical_indices']
    pending = meta['contract']['pending_indices']
    total = torch.zeros(20); persistence = torch.zeros(20)
    for block in np.array_split(ids, max(1, int(np.ceil(len(ids)/2048)))):
        obs = torch.as_tensor(data['observations'][block])
        initial = current = obs[:, physical]
        queue = obs[:, pending].reshape(-1, 2, 6)
        mass = obs[:, 2:3]
        for k in range(horizon):
            current = body.next_physical(torch.cat((current, queue[:, 0], mass), -1))
            issued = 2*torch.as_tensor(data['actions'][block+k])-1
            queue = torch.stack((queue[:, 1], issued), 1)
        actual = torch.as_tensor(data['next_observations'][block+horizon-1][:, physical])
        total += (current-actual).square().sum(0)
        persistence += (initial-actual).square().sum(0)
    rmse = (total/len(ids)).sqrt()
    return {'samples': len(ids), 'horizon_ms': horizon*20,
        'normalized_mse': float((total/len(ids)/body.delta_std.square()).mean()),
        'angle_rmse_deg': float(rmse[0]*90), 'velocity_rmse_deg_s': float(rmse[1]*4*180/np.pi),
        'persistence_angle_rmse_deg': float((persistence[0]/len(ids)).sqrt()*90),
        'persistence_velocity_rmse_deg_s': float((persistence[1]/len(ids)).sqrt()*4*180/np.pi),
        'physical_channel_rmse_normalized_observation': rmse.tolist()}


def save_body(path, body, optimizer, update, data_meta):
    torch.save({'state_dict': body.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'update': update}, path)
    write_json(Path(str(path)+'.metadata.json'), {'contract': data_meta['contract'], 'update': update,
        'dataset_sha256': data_meta['dataset_sha256'], 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'training': {'optimizer': 'RMSprop', 'lr': 3e-4, 'eps': 1e-5, 'alpha': .99,
                     'batch_size': 128, 'gradient_norm': .5, 'objective': 'normalized one-step physical delta MSE',
                     'normalization': 'all valid train transitions, frozen; STD floor1e-5'}})


def train_predictor(root, maximum=50000):
    folder = Path(root)/'body'
    if (folder/'training.json').exists():
        raise FileExistsError(folder/'training.json')
    data, meta = load_data(root)
    env = ArmPrecisionEnv(); seed_policy = make_model(env).policy
    body = ArmBodyPredictor()
    body.encoder.load_state_dict(seed_policy.mlp_extractor.encoder.state_dict())
    body.head.load_state_dict(seed_policy.prediction_head.state_dict()); env.close()
    train = starts(data, meta, 'train')
    x = torch.as_tensor(data['observations'][train][:, meta['contract']['predictor_indices']])
    delta = torch.as_tensor(data['next_observations'][train][:, meta['contract']['physical_indices']])-x[:, :20]
    body.delta_mean.copy_(delta.mean(0)); body.delta_std.copy_(delta.std(0, unbiased=False).clamp_min(1e-5))
    body.delta_calibrated.fill_(True)
    target = (delta-body.delta_mean)/body.delta_std
    optimizer = torch.optim.RMSprop(body.parameters(), lr=3e-4, eps=1e-5)
    rng = np.random.default_rng(0)
    summary = {'status': 'running', 'history': [], 'train_transitions': len(train), 'maximum_updates': maximum,
               'selection': 'minimum whole-validation normalized MSE20; test after selection',
               'plateau': 'after10000 updates: no accumulated0.5% improvement for6000 updates'}
    save_body(folder/'initial.pt', body, optimizer, 0, meta)
    best, significant, last_progress = float('inf'), float('inf'), 0
    started = time.monotonic()
    for update in range(maximum+1):
        if update % 500 == 0 or update == maximum:
            validation = forecast(body, data, meta, 'validation')
            value = validation['normalized_mse']
            summary['history'].append({'update': update, 'validation': validation,
                                      'train': forecast(body, data, meta, 'train')})
            if value < best:
                best = value; summary['best_update'] = update
                save_body(folder/'best.pt', body, optimizer, update, meta)
            if value < significant*.995:
                significant, last_progress = value, update
            summary['elapsed_s'] = time.monotonic()-started
            write_json(folder/'training.json', summary)
            print(json.dumps({'update': update, 'validation_mse': value,
                'q20_deg': validation['angle_rmse_deg'], 'v20_deg_s': validation['velocity_rmse_deg_s'],
                'best_update': summary['best_update']}), flush=True)
            if update >= 10000 and update-last_progress >= 6000:
                summary['stop_reason'] = 'validation_plateau'; break
        if update == maximum:
            break
        ids = rng.integers(0, len(x), size=128)
        optimizer.zero_grad(set_to_none=True)
        loss = (body.normalized_delta(x[ids])-target[ids]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite arm predictor loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(body.parameters(), .5); optimizer.step()
    save_body(folder/'last.pt', body, optimizer, update, meta)
    selected, _ = predictor_identity(folder/'best.pt')
    summary.update(status='complete', updates=update, stop_reason=summary.get('stop_reason', 'maximum_updates'),
        elapsed_s=time.monotonic()-started,
        test={str(20*h): forecast(selected, data, meta, 'test', h) for h in (1, 3)})
    write_json(folder/'training.json', summary)


def private_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode()); digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_policy(root, trial, budget):
    folder = Path(root)/trial; folder.mkdir(parents=True, exist_ok=False)
    env = ArmPrecisionEnv(); model = make_model(Monitor(env), trial)
    identity = None
    if trial == 'frozen':
        with torch.random.fork_rng(devices=[]):
            body, identity = predictor_identity(Path(root)/'body/best.pt')
        if not bool(body.delta_calibrated):
            raise ValueError('Cannot use uncalibrated arm predictor')
        body.copy_into_policy(model.policy)
    training = {'trial': trial, 'seed': 0, 'maximum_transitions': budget,
        'ppo': {'optimizer': 'RMSprop', 'learning_rate': 3e-4, 'rollout': 1024, 'batch': 128, 'epochs': 10,
                'gamma': .99, 'gae_lambda': .95, 'clip': .2, 'entropy': .001, 'value': .5,
                'grad_norm': .5, 'target_kl': .03, 'kl_control': 'standard SB3 approximate minibatch early stop; no rollback'},
        'predictor': identity, 'train_cases': TRAIN_CASES, 'validation_cases': VALIDATION_CASES, 'test_cases': TEST_CASES,
        'stop': 'maximum budget or no1% accumulated validation progress for49152 transitions after65536',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {k: version(k) for k in ('torch', 'stable-baselines3', 'mujoco', 'numpy')}}
    summary = {'status': 'running', 'training': training, 'history': [], 'updates': [], 'contract': env.contract(),
        'initial_private_sha256': private_hash(model.policy),
        'initial_torch_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'parameter_count': sum(p.numel() for p in model.policy.parameters()),
        'ppo_parameter_count': sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
        'cpu_affinity': sorted(os.sched_getaffinity(0))}
    save_arm(model, env, folder/'initial.zip', training)
    best, progress_reference, progress_step = None, None, 0
    started = time.monotonic()
    class Audit(BaseCallback):
        def _on_step(self): return True
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
                                  'optimizer_steps': model.optimizer_steps})
        summary['elapsed_s'] = time.monotonic()-started
        write_json(folder/'summary.json', summary)
        print(json.dumps({'trial': trial, 'steps': model.num_timesteps, 'validation_deg': result['score_deg'],
            'passed': result['passed'], 'failures': result['failures'], 'best_steps': summary['best_steps']}), flush=True)
    check()
    for _ in range(budget//8192):
        model.learn(8192, reset_num_timesteps=False, callback=Audit())
        check()
        if model.num_timesteps >= 65536 and model.num_timesteps-progress_step >= 49152:
            summary['stop_reason'] = 'validation_plateau'; break
    summary['updates'].append({'steps': model.num_timesteps, **model.last_audit})
    save_arm(model, env, folder/'last.zip', training)
    summary.update(status='training_complete', steps=model.num_timesteps, optimizer_steps=model.optimizer_steps,
        stop_reason=summary.get('stop_reason', 'maximum_transitions'), elapsed_s=time.monotonic()-started)
    write_json(folder/'summary.json', summary); env.close()


def evaluate(root):
    root = Path(root)
    summaries = {trial: json.loads((root/trial/'summary.json').read_text()) for trial in TRIALS}
    assert all(s['status'] == 'training_complete' for s in summaries.values())
    assert len({s['initial_private_sha256'] for s in summaries.values()}) == 1
    write_json(root/'frozen_selection.json', {trial: {'steps': s['best_steps'],
        'sha256': json.loads((root/trial/'best.zip.metadata.json').read_text())['checkpoint_sha256']}
        for trial, s in summaries.items()})
    audit = {'status': 'running', 'strict_loads': 0, 'frozen_predictors_unchanged': True}
    for trial, s in summaries.items():
        models = {}
        for label in ('initial', 'best', 'last'):
            models[label], env, _ = load_arm(root/trial/f'{label}.zip'); env.close()
            audit['strict_loads'] += 1
        assert evaluate_arm(models['best']) == s['best_validation']
        assert rank(s['best_validation']) == min(rank(row['validation']) for row in s['history'])
        for name, value in models['initial'].policy.state_dict().items():
            if name.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_')):
                torch.testing.assert_close(value, models['last'].policy.state_dict()[name], rtol=0, atol=0)
        for label in ('best', 'last'):
            result = evaluate_arm(models[label], TEST_CASES, samples=True)
            write_json(root/trial/f'{label}_test.json', result)
            for row in result['cases']: row.pop('samples')
            s[label+'_test'] = result
        s['status'] = 'complete'; write_json(root/trial/'summary.json', s)
        print(json.dumps({'trial': trial, 'test': s['best_test']}), flush=True)
    audit['status'] = 'passed'; write_json(root/'evaluation_audit.json', audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('collect', 'predictor', 'train', 'evaluate'))
    p.add_argument('--root', type=Path, default=Path('runs/arm_preacher'))
    p.add_argument('--trial', choices=TRIALS)
    p.add_argument('--steps', type=int, default=131072)
    p.add_argument('--updates', type=int, default=50000)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(args.threads)
    if args.mode == 'collect': collect(args.root)
    elif args.mode == 'predictor': train_predictor(args.root, args.updates)
    elif args.mode == 'evaluate': evaluate(args.root)
    else:
        if args.trial is None or args.steps <= 0 or args.steps % 8192:
            p.error('Choose trial and positive complete8192-step budget')
        train_policy(args.root, args.trial, args.steps)


if __name__ == '__main__': main()
