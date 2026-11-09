"""Broader native-arm motor probes and balanced physical-state regression."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from .arm_learning import load_arm, predictor_identity
from .arm_precision import ArmPrecisionEnv, VALIDATION_CASES, TEST_CASES
from .train_arm_precision import commands, load_data, starts
from .train_vertical_ppo import write_json

OLD = Path('runs/arm_preacher')
ROOT = Path('runs/arm_preacher_balanced')
POLICIES = {'initial': 'none/initial.zip', 'none': 'none/best.zip', 'frozen': 'frozen/best.zip'}
_policies = {}


def specifications():
    jobs = []
    for split, positions in [('train', (20, 35, 50, 65, 80, 95, 110, 125, 135)),
                             ('validation', (25, 55, 85, 115)), ('test', (30, 60, 90, 120))]:
        for position, capacity in itertools.product(positions, (1., .8)):
            for coactivation in (0., .025, .06):
                jobs.append(dict(split=split, family='hold', position_deg=position,
                                 capacity=capacity, coactivation=coactivation))
            for speed in ((15., 45., 90.) if split == 'train' else (25., 60., 100.)):
                jobs.append(dict(split=split, family='ramp', position_deg=position,
                                 capacity=capacity, speed_deg_s=speed))
            for concentration in (10., 30.):
                jobs.append(dict(split=split, family='jitter', position_deg=position,
                                 capacity=capacity, concentration=concentration))
        # Capacity and amplitude are a full Cartesian product, unlike the old data.
        for family, capacity, limit, repeat in itertools.product(
                ('independent', 'reciprocal', 'common', 'multisine'), (1., .8), (.12, .4, .8),
                range(2 if split == 'train' else 1)):
            position = positions[(len(jobs)+repeat) % len(positions)]
            jobs.append(dict(split=split, family=family, position_deg=position, capacity=capacity, limit=limit))
        if split != 'train':
            for policy, case in itertools.product(POLICIES, VALIDATION_CASES if split == 'validation' else TEST_CASES):
                jobs.append(dict(split=split, family='policy', policy=policy,
                                 position_deg=case[0], target_deg=case[1], capacity=1.))
    for i, job in enumerate(jobs):
        job['seed'] = 20261109+i
    return jobs


def ramp_target(position, speed, step):
    direction = 1. if position <= 75 else -1.
    unfolded = position-15+direction*speed*max(0., step*.02-.8)
    phase = unfolded % 240.
    return 15+min(phase, 240-phase)


def collect_episode(spec, count=200):
    torch.set_num_threads(1)
    rng = np.random.default_rng(spec['seed'])
    torch.manual_seed(spec['seed'])
    env = ArmPrecisionEnv()
    target = spec.get('target_deg', spec['position_deg'])
    obs, _ = env.reset(seed=spec['seed'], options={'case': (spec['position_deg'], target),
                                                 'capacity': spec['capacity']})
    family = spec['family']
    open_loop = None
    if family not in ('hold', 'ramp', 'jitter', 'policy'):
        open_loop = commands(family, count, spec['seed'], spec['limit'])
    if family == 'policy' and spec['policy'] not in _policies:
        model, check_env, _ = load_arm(OLD/POLICIES[spec['policy']])
        check_env.close(); _policies[spec['policy']] = model
        # Loading a checkpoint must not select the episode's exploration sequence.
        torch.manual_seed(spec['seed'])
    data = {key: [] for key in ('observations', 'actions', 'next_observations', 'valid')}
    for k in range(count):
        if family == 'ramp':
            env.unwrapped.set_command(float(np.deg2rad(ramp_target(spec['position_deg'], spec['speed_deg_s'], k))), 1.)
            obs = env.observation()
        if open_loop is not None:
            action = open_loop[k]
        elif family == 'policy':
            action = _policies[spec['policy']].predict(obs, deterministic=False)[0]
        else:
            action = env.unwrapped.baseline_action()
            if family == 'hold':
                action = np.clip(action+spec['coactivation'], 0, 1)
            elif family == 'jitter':
                mean = np.clip(action+.05, .005, .95)
                c = spec['concentration']
                action = rng.beta(mean*c, (1-mean)*c).astype(np.float32)
        following, _, failed, truncated, _ = env.step(action)
        for key, value in zip(data, (obs, action, following, not failed), strict=True):
            data[key].append(value)
        obs = following
        if failed or truncated:
            break
    env.close()
    return ({key: np.asarray(value) for key, value in data.items()}, {**spec, 'failure': bool(failed)})


def collect(root=ROOT, workers=3):
    import json
    folder = Path(root)/'body'; folder.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    old_data, old_meta = load_data(OLD)
    pieces = {key: [value] for key, value in old_data.items()}
    episodes = [{**ep, 'source': 'old_motor'} for ep in old_meta['episodes']]
    offset = len(old_data['valid'])
    sources = {'old_motor': old_meta['dataset_sha256']}
    # Earlier diagnostic training-task episodes are now explicitly training data.
    # They are not reused as held-out validation of the new predictor.
    for name, relative in POLICIES.items():
        path = OLD/'movement_coverage'/f'{name}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        model, env, identity = load_arm(OLD/relative); env.close()
        if identity != meta['checkpoint'] or meta['contract'] != old_meta['contract']:
            raise ValueError('Old policy diagnostic identity mismatch')
        del model
        data = dict(np.load(path))
        for key in pieces: pieces[key].append(data[key])
        episodes.extend({**ep, 'split': 'train', 'family': 'policy', 'policy': name,
                         'source': 'old_policy_diagnostic', 'start': offset+ep['start'],
                         'stop': offset+ep['stop']} for ep in meta['episodes'])
        offset += len(data['valid'])
        sources[name] = {'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checkpoint': identity}
    jobs = specifications()
    # Each worker owns a separate MuJoCo environment; ordered map keeps episode IDs stable.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        for i, (data, spec) in enumerate(executor.map(collect_episode, jobs, chunksize=1)):
            for key in pieces: pieces[key].append(data[key])
            episodes.append({**spec, 'source': 'new_probe', 'start': offset, 'stop': offset+len(data['valid'])})
            offset += len(data['valid'])
            if (i+1) % 40 == 0:
                print(json.dumps({'episodes_collected': i+1, 'new_episodes': len(jobs)}), flush=True)
    path = folder/'dataset.npz'
    np.savez_compressed(path, **{key: np.concatenate(value) for key, value in pieces.items()})
    meta = {'contract': old_meta['contract'], 'episodes': episodes, 'sources': sources,
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'collection': 'old motor splits retained; previous TRAIN_CASES diagnostics promoted to train; new feedback/open-loop/policy episodes split whole',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'workers': workers, 'elapsed_s': time.monotonic()-begun}
    write_json(folder/'dataset.json', meta)
    print(json.dumps({'transitions': offset, 'episodes': len(episodes), 'elapsed_s': meta['elapsed_s']}), flush=True)


def sampling_weights(observations, meta):
    obs = np.asarray(observations)
    q = obs[:, 0]*90+75
    velocity = obs[:, 1]*4*180/np.pi
    indices = meta['contract']['predictor_indices']
    # Excitation queued for the next interval versus current activation is causal.
    discrepancy = np.max(np.abs(obs[:, indices[20:26]]-obs[:, indices[8:14]])/2, axis=1)
    cells = np.digitize(q, (30, 60, 90, 120))*15+np.digitize(velocity, (-60, -5, 5, 60))*3
    cells += np.digitize(discrepancy, (.03, .1))
    unique, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    p = .5/len(obs)+.5/(len(unique)*counts[inverse])
    return p/p.sum(), {'cell_ids': unique.tolist(), 'counts': counts.tolist(),
        'mixture': 'half uniform transitions, half uniform occupied angle/signed-velocity/command-activation cells',
        'q_edges_deg': [30, 60, 90, 120], 'velocity_edges_deg_s': [-60, -5, 5, 60],
        'excitation_activation_gap_edges': [.03, .1]}


@torch.no_grad()
def regression_metric(body, data, meta, split, maximum=None):
    ids = starts(data, meta, split)
    if maximum is not None and len(ids) > maximum:
        ids = ids[np.linspace(0, len(ids)-1, maximum, dtype=int)]
    obs = torch.as_tensor(data['observations'][ids])
    physical = meta['contract']['physical_indices']
    target = torch.as_tensor(data['next_observations'][ids][:, physical])-obs[:, physical]
    x = obs[:, meta['contract']['predictor_indices']]
    weights, _ = sampling_weights(obs.numpy(), meta)
    error = (body.normalized_delta(x)-(target-body.delta_mean)/body.delta_std).square().mean(1)
    return {'samples': len(ids), 'balanced_mse': float(error@torch.as_tensor(weights, dtype=error.dtype)),
            'uniform_mse': float(error.mean())}


def save_body(path, body, optimizer, update, data_meta, initial_identity, sampling):
    torch.save({'state_dict': body.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'update': update}, path)
    write_json(Path(str(path)+'.metadata.json'), {'contract': data_meta['contract'], 'update': update,
        'dataset_sha256': data_meta['dataset_sha256'], 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'training': {'initial_predictor': initial_identity, 'optimizer': 'fresh RMSprop', 'lr': 3e-4,
            'eps': 1e-5, 'alpha': .99, 'batch_size': 128, 'gradient_norm': .5,
            'objective': 'normalized physical delta MSE with mixed stratified sampling', 'sampling': sampling,
            'normalization': 'frozen original train mean/STD; unchanged to compare errors and avoid moving targets',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})


def train_predictor(root=ROOT, maximum=100000):
    import json
    from .train_arm_precision import forecast
    root = Path(root); folder = root/'body'
    if (folder/'training.json').exists(): raise FileExistsError(folder/'training.json')
    data, meta = load_data(root)
    body, identity = predictor_identity(OLD/'body/best.pt')
    train = starts(data, meta, 'train')
    observations = data['observations'][train]
    x = torch.as_tensor(observations[:, meta['contract']['predictor_indices']])
    delta = torch.as_tensor(data['next_observations'][train][:, meta['contract']['physical_indices']])-x[:, :20]
    target = (delta-body.delta_mean)/body.delta_std
    probability, sampling = sampling_weights(observations, meta)
    cumulative = np.cumsum(probability); cumulative[-1] = 1.
    optimizer = torch.optim.RMSprop(body.parameters(), lr=3e-4, eps=1e-5)
    rng = np.random.default_rng(0)
    summary = {'status': 'running', 'history': [], 'train_transitions': len(train), 'maximum_updates': maximum,
        'initial_predictor': identity, 'selection': 'balanced whole-validation one-step MSE, original frozen units',
        'sampling': sampling, 'stop': 'after20000, no accumulated0.5% validation progress for12000 updates'}
    save_body(folder/'initial.pt', body, optimizer, 0, meta, identity, sampling)
    best, reference, progress = np.inf, np.inf, 0
    begun = time.monotonic()
    for update in range(maximum+1):
        if update % 1000 == 0 or update == maximum:
            result = regression_metric(body, data, meta, 'validation')
            summary['history'].append({'update': update, 'validation': result,
                'train': regression_metric(body, data, meta, 'train', maximum=8192)})
            if result['balanced_mse'] < best:
                best = result['balanced_mse']; summary['best_update'] = update
                save_body(folder/'best.pt', body, optimizer, update, meta, identity, sampling)
            if result['balanced_mse'] < reference*.995:
                reference, progress = result['balanced_mse'], update
            summary['elapsed_s'] = time.monotonic()-begun
            write_json(folder/'training.json', summary)
            print(json.dumps({'update': update, **result, 'best_update': summary['best_update']}), flush=True)
            if update >= 20000 and update-progress >= 12000:
                summary['stop_reason'] = 'validation_plateau'; break
        if update == maximum: break
        ids = np.searchsorted(cumulative, rng.random(128))
        optimizer.zero_grad(set_to_none=True)
        loss = (body.normalized_delta(x[ids])-target[ids]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite balanced predictor loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(body.parameters(), .5); optimizer.step()
    save_body(folder/'last.pt', body, optimizer, update, meta, identity, sampling)
    chosen, _ = predictor_identity(folder/'best.pt')
    original, _ = predictor_identity(OLD/'body/best.pt')
    # Both selections are fixed before looking at any new test outcomes.
    test = {name: {str(20*h): forecast(model, data, meta, 'test', h) for h in (1, 3)}
            for name, model in [('old', original), ('new', chosen)]}
    summary.update(status='complete', updates=update, test=test,
        stop_reason=summary.get('stop_reason', 'maximum_updates'), elapsed_s=time.monotonic()-begun)
    write_json(folder/'training.json', summary)
