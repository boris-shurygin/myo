"""Neural acceleration policy above a frozen, delayed six-muscle policy."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .arm_hierarchy import ArmHierarchyEnv, HierarchySpec, Motion
from .arm_learning import ArmPPO


class NeuralUpperEnv(gym.Wrapper):
    def __init__(self, lower_policy, lower_contract, *, worker=0, workers=1):
        spec = lower_contract['native']['spec']
        inner = ArmHierarchyEnv(HierarchySpec(**lower_contract['hierarchy']),
            [Motion(**m) for m in lower_contract['motions']],
            masses=tuple(spec['masses']), tilts=tuple(spec['tilts']), worker=worker, workers=workers)
        if inner.contract() != lower_contract:
            inner.close()
            raise ValueError('Frozen lower physics/observation identity mismatch')
        super().__init__(inner)
        self.lower_policy, self.lower_contract = lower_policy, lower_contract
        self.lower_policy.set_training_mode(False)
        for parameter in self.lower_policy.parameters():
            parameter.requires_grad_(False)
        self.observation_names = inner.observation_names + ['upper/reference_position_error_rad',
            'upper/reference_velocity_error_rad_s', 'upper/reference_acceleration_scaled',
            'upper/last_measured_acceleration_scaled']
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)

    def observation(self):
        inner, base = self.env, self.unwrapped
        physical = inner.observation()
        # Parent computes a heuristic proposal; neither actor nor critic may see it.
        # It is overwritten before ANY lower action is computed.
        physical[-2] = 0.
        q, v, _ = inner.motion.sample(base.time+.05)
        aff = (inner.motion.sample(base.time+.06)[1]-inner.motion.sample(base.time+.04)[1])/.02
        return np.concatenate((physical, [q-float(base.data.qpos[base.qpos_index]),
            v-float(base.data.qvel[base.dof_index]), aff/5., inner.last_acceleration/5.])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        return self.observation(), info

    def heuristic_action(self):
        return np.array([self.env.tasks[-1]/20.], dtype=np.float32)

    def step(self, action):
        normalized = float(np.asarray(action).reshape(1)[0])
        if not np.isfinite(normalized) or abs(normalized) > 1.:
            raise ValueError('Upper acceleration action must be finite and in [-1,1]')
        inner = self.env
        issuing = inner.step_index < round(inner.motion.seconds/.02)
        inner.tasks[-1] = 20.*normalized if issuing else 0.
        excitation = self.lower_policy.predict(inner.observation(), deterministic=True)[0]
        _, _, failed, done, info = inner.step(excitation)
        position = abs(info['hierarchy']['position_error'])
        velocity = info['hierarchy']['velocity_error']
        c = (1-np.exp(-1))/np.log1p(90/3)
        terms = {'reference_position': .85*(1-c*np.log1p(position/np.deg2rad(3.))),
                 'reference_velocity': .15*np.exp(-(velocity/.5)**2),
                 'physical_failure': -float(failed)}
        info['upper'] = {'issued': issuing, 'acceleration_command_rad_s2': 20.*normalized if issuing else 0.,
                         'muscle_excitation': np.asarray(excitation).tolist()}
        info['reward_components'] = terms
        return self.observation(), .1*float(sum(terms.values())), failed, done, info

    def contract(self):
        return {'schema':'myo.arm_neural_upper.v1', 'lower_training_contract':self.lower_contract,
            'observation_names':self.observation_names,
            'action':'one normalized acceleration in [-1,1], times20 rad/s2, due40-60ms',
            'lower':'frozen deterministic Beta mean; same native delay/activation/compliance',
            'input':'lower physical/history/issued tasks with heuristic due40 proposal masked to0; '
                    'append q_ref(t+50)-q,v_ref(t+50)-v,mean_a_ref(40:60)/5,last_measured_a/5',
            'reward':'.1*(.85*anchored_log_position + .15*exp(-((v-v_ref)/.5)^2) - physical_failure)',
            'reference':'known task preview only, no future simulated state', 'predictor':None,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save_upper(model, env, path, training):
    path = Path(path)
    model.upper_contract = env.contract()
    model.upper_training = training
    model.frozen_lower = env.lower_policy
    model.save(path)
    meta = {'contract':env.contract(), 'training':training,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


def load_upper(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    if meta['contract']['schema'] != 'myo.arm_neural_upper.v1':
        raise ValueError('Not a neural-upper checkpoint')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Upper checkpoint bytes mismatch')
    model = ArmPPO.load(path,device='cpu')
    env = NeuralUpperEnv(model.frozen_lower, meta['contract']['lower_training_contract'])
    if (env.contract() != meta['contract'] or model.upper_contract != meta['contract']
            or model.upper_training != meta['training']):
        env.close()
        raise ValueError('Neural upper contract mismatch')
    return model, env, meta
