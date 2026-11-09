"""Direct causal 60-ms arm forecasts with an action-independent body encoder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .arm_precision import ArmPrecisionEnv
from .shared_vertical import mlp
from .train_arm_precision import starts


HORIZON = 3


def identity(contract):
    names = contract['observation_names']
    indices = contract['physical_indices']+contract['pending_indices']+[2]
    return {'schema': 'myo.arm_anticipation.v1', 'native_contract': contract,
        'horizon_steps': HORIZON, 'horizon_s': .06, 'control_dt_s': .02,
        'encoder_input_names': [names[i] for i in indices],
        'encoder_indices': indices, 'encoder_layers': [128]*4,
        'latent_features': 128, 'activation': 'SiLU',
        'head_layers': [128, 20], 'head_condition': 'six current issued excitations, scaled 2*u-1; after encoder',
        'action_mapping': contract['action_mapping'],
        'output_names': [names[i] for i in contract['physical_indices']],
        'output': 'train-normalized 60-ms physical delta; residual added to current physical state',
        'future_commands': 'not input: commands issued at t+20ms arrive at boundary t+60ms',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class AnticipationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp(33, (128,)*4)
        self.head = nn.Sequential(nn.Linear(134, 128), nn.SiLU(), nn.Linear(128, 20))
        self.register_buffer('delta_mean', torch.zeros(20))
        self.register_buffer('delta_std', torch.ones(20))
        self.register_buffer('calibrated', torch.tensor(False))

    def normalized_delta(self, inputs, commands):
        latent = self.encoder(inputs)
        return self.head(torch.cat((latent, 2*commands-1), -1))

    def next_physical(self, inputs, commands):
        return inputs[:, :20]+self.delta_mean+self.delta_std*self.normalized_delta(inputs, commands)


def examples(data, meta, split):
    ids = starts(data, meta, split, HORIZON)
    if not len(ids):
        raise ValueError('No complete 60-ms windows in episode split')
    contract = meta['contract']
    observations = data['observations'][ids]
    indices = contract['physical_indices']+contract['pending_indices']+[2]
    inputs = observations[:, indices]
    commands = data['actions'][ids]
    actual = data['next_observations'][ids+HORIZON-1][:, contract['physical_indices']]
    return ids, inputs, commands, actual


@torch.no_grad()
def iterative_forecast(body, inputs, commands):
    """The previous one-step model repeated three times, with no future actions."""
    current = inputs[:, :20]
    queue, mass = inputs[:, 20:32].reshape(-1, 2, 6), inputs[:, 32:33]
    for k in range(HORIZON):
        current = body.next_physical(torch.cat((current, queue[:, 0], mass), -1))
        issued = 2*commands-1 if k == 0 else torch.zeros_like(commands)
        queue = torch.stack((queue[:, 1], issued), 1)
    return current


def physical_errors(predicted, actual, initial, one_step, names):
    error = np.sqrt(np.mean(np.square(np.asarray(predicted, dtype=np.float64)-actual), axis=0))
    change = np.sqrt(np.mean(np.square(np.asarray(actual, dtype=np.float64)-initial), axis=0))
    step = np.sqrt(np.mean(np.square(np.asarray(one_step, dtype=np.float64)-initial), axis=0))
    scales = np.array([90., 4*180/np.pi]+[.2]*6+[.5]*12)
    units = ['deg', 'deg/s']+['optimal_fiber_length_fraction']*6+['activation_fraction']*6+['capacity_fraction']*6
    channels = [{'name': name, 'rmse': float(e*s), 'unit': unit,
        'delta60_rms': float(d*s), 'delta20_rms': float(st*s),
        'error_over_delta60': float(e/d) if d > 1e-12 else None,
        'error_over_delta20': float(e/st) if st > 1e-12 else None}
        for name, e, d, st, s, unit in zip(names, error, change, step, scales, units, strict=True)]
    return {'samples': len(actual), 'angle_rmse_deg': channels[0]['rmse'],
        'velocity_rmse_deg_s': channels[1]['rmse'], 'channels': channels}


@torch.no_grad()
def predictions(body, inputs, commands, *, iterative=False):
    output = []
    for a in range(0, len(inputs), 2048):
        x = torch.as_tensor(inputs[a:a+2048])
        u = torch.as_tensor(commands[a:a+2048])
        output.append((iterative_forecast(body, x, u) if iterative else body.next_physical(x, u)).numpy())
    return np.concatenate(output)


def save_predictor(path, body, optimizer, update, spec, training):
    path = Path(path)
    payload = {'state_dict': body.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'update': update, 'identity': spec, 'training': training}
    torch.save(payload, path)
    metadata = {'identity': spec, 'training': training, 'update': update,
                'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+'.metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


def load_predictor(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = ArmPrecisionEnv()
    try:
        expected = identity(env.contract())
    finally:
        env.close()
    if meta['identity'] != expected:
        raise ValueError('Anticipation predictor horizon/input/physics identity mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('Anticipation predictor checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if any(payload[key] != meta[key] for key in ('identity', 'training', 'update')):
        raise ValueError('Anticipation predictor embedded identity mismatch')
    body = AnticipationPredictor()
    body.load_state_dict(payload['state_dict'])
    if not bool(body.calibrated) or not bool((body.delta_std > 0).all()):
        raise ValueError('Missing anticipation predictor normalization')
    return body.eval(), meta
