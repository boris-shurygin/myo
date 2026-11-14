"""Timing-only adapter for fresh direct arm PPO; pinned checkpoints stay intact."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from .arm_exploration import ArmExplorationEnv, ExplorationPolicy, ExplorationSpec, StagnationDetector
from .arm_learning import ArmPPO
from .arm_precision import ArmPrecisionEnv
from .arm_reward_curriculum import RewardStage
from .arm_workload import spec_from_dict
from .arm_workload_evaluation import merge_results
from .env import PreacherCurlEnv


@dataclass(frozen=True)
class ArmTiming:
    motor_delay_s: float = .01
    control_dt_s: float = .01


class ArmTimingEnv(ArmExplorationEnv):
    def __init__(self, spec=None, exploration=None, timing=None, *, worker=0, workers=1):
        # Use the existing observation calibration, reward, schedule and policy.
        # Replace only the native timing and its lossless command-queue encoding.
        super().__init__(spec, RewardStage('position_only'), exploration,
                         worker=worker, workers=workers)
        self.timing = timing or ArmTiming()
        previous = self.unwrapped
        frame_skip = round(self.timing.control_dt_s / previous.config.physics_dt)
        blocks = round(self.timing.motor_delay_s / self.timing.control_dt_s)
        if (frame_skip < 1 or blocks < 1
                or not np.isclose(frame_skip*previous.config.physics_dt, self.timing.control_dt_s)
                or not np.isclose(blocks*self.timing.control_dt_s, self.timing.motor_delay_s)):
            previous.close()
            raise ValueError('Timing requires positive whole physics/control steps')
        config = replace(previous.config, frame_skip=frame_skip, motor_delay=self.timing.motor_delay_s)
        previous.close()
        self.env = PreacherCurlEnv(config)
        self.native_config = config
        self.queue_steps = len(self.unwrapped.muscles[0].delay.pending)
        self.queue_blocks = blocks
        self.prefix = len(self.unwrapped.observation_names)-6*self.queue_steps
        self.observation_names = self.unwrapped.observation_names[:self.prefix]+[
            f'pending/{age}/{name}' for age in range(blocks) for name in self.unwrapped.muscle_names]
        # Calibration of all retained named channels is exactly the old one.
        self.scale = np.concatenate((self.scale[:self.prefix], np.full(6*blocks, 2.)))
        self.offset = np.concatenate((self.offset[:self.prefix], np.full(6*blocks, -1.)))
        self.pending_indices = [self.observation_names.index(f'pending/{age}/{name}')
                                for age in range(blocks) for name in self.unwrapped.muscle_names]
        self.predictor_indices = self.physical_indices+self.pending_indices[:6]+[2]
        self.native_contract = ArmPrecisionEnv.contract(self)
        self.native_contract['schema'] = 'myo.arm_precision_timing.v1'
        self.observation_names = self.observation_names+[
            'mount/sin_minus_sin30', 'mount/cos_minus_cos30', 'exploration/stagnation_pressure']
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(len(self.observation_names),), dtype=np.float32)
        self.detector = StagnationDetector(self.exploration, config.control_dt)

    def encode(self, observation):
        pending = observation[self.prefix:].reshape(6, self.queue_blocks, self.unwrapped.config.frame_skip)
        if not np.all(pending == pending[:, :, :1]):
            raise ValueError('Motor queue contains sub-command changes')
        compact = np.concatenate((observation[:self.prefix], pending[:, :, 0].T.reshape(-1)))
        angle = np.deg2rad(self.tilt_deg)
        return np.concatenate((compact*self.scale+self.offset,
            [np.sin(angle)-np.sin(np.deg2rad(30.)), np.cos(angle)-np.cos(np.deg2rad(30.)),
             self.detector.pressure])).astype(np.float32)

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_timing.v1', timing=asdict(self.timing),
            history='two preceding control frames; ages control_dt and 2*control_dt',
            timing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return contract


def make_fresh_model(env, contract, *, n_steps=256, batch_size=128, n_epochs=10):
    native = contract['native']
    return ArmPPO(ExplorationPolicy, env, seed=0, device='cpu', learning_rate=1e-5,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': native['predictor_indices'],
            'physical_indices': native['physical_indices'], 'feature_mode': 'none',
            'exploration': contract['exploration'], 'activation_fn': nn.SiLU,
            'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})


def load_timing(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    contract = meta['contract']
    if contract['schema'] != 'myo.arm_timing.v1':
        raise ValueError('Not an arm timing checkpoint')
    env = ArmTimingEnv(spec_from_dict(contract['spec']), ExplorationSpec(**contract['exploration']),
                       ArmTiming(**contract['timing']))
    try:
        if (env.contract() != contract
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']):
            raise ValueError('Arm timing physics/input/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != contract or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, ExplorationPolicy)
                or asdict(model.policy.exploration) != contract['exploration']
                or model.observation_space != env.observation_space or model.policy.feature_mode != 'none'):
            raise ValueError('Arm timing embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta


def evaluation_job(path, conditions):
    """Run in a separate process to preserve training RNG and physical episodes."""
    torch.set_num_threads(1)
    model, env, _ = load_timing(path)
    rows = []
    try:
        for condition in conditions:
            observation, _ = env.reset(seed=0, options=condition)
            trace = []
            while True:
                action = model.predict(observation, deterministic=True)[0]
                observation, _, failed, done, info = env.step(action)
                trace.append((env.unwrapped.time, np.rad2deg(info['q']), np.rad2deg(info['qvel']),
                    info['workload']['activation_squared_mean'], info['workload']['active_fraction_mean'],
                    info['activation'], info['exploration']['pressure']))
                if failed or done:
                    break
            times = np.array([s[0] for s in trace])
            error = np.array([s[1] for s in trace])-condition['case'][1]
            velocity = np.array([s[2] for s in trace])
            tail = times > condition['seconds']-1.-1e-8
            if not tail.any():
                tail = np.ones(len(trace), dtype=bool)
            full = not failed and len(trace) == round(condition['seconds']/env.unwrapped.control_dt)
            settled = (abs(error) <= 3.) & (abs(velocity) <= 5.)
            fraction = float(settled[tail].mean()) if full else 0.
            stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
            direction = np.sign(condition['case'][1]-condition['case'][0])
            rows.append({**condition, 'failure': bool(failed), 'passed': bool(full and fraction >= .9),
                'steps': len(trace), 'duration_s': float(times[-1]), 'settled_fraction': fraction,
                'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
                'tail_velocity_rms_deg_s': float(np.sqrt(np.mean(velocity[tail]**2))),
                'full_rmse_deg': float(np.sqrt(np.mean(error**2))),
                'overshoot_deg_10ms': float(max(0., np.max(direction*error))) if direction else None,
                'settling_time_s_10ms': float(times[stable[0]]) if len(stable) and full else None,
                'activation_squared_mean': float(np.mean([s[3] for s in trace])),
                'active_fraction_mean': float(np.mean([s[4] for s in trace])),
                'tail_activation_by_muscle': np.mean([s[5] for s, used in zip(trace, tail) if used], axis=0).tolist(),
                'pressure_fraction': float(np.mean([s[6] > .5 for s in trace]))})
    finally:
        env.close()
    return merge_results([{'cases': rows}])
