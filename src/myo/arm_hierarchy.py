"""Causal trajectory -> future acceleration -> delayed muscle control experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np

from .arm_precision import ArmPrecisionEnv
from .arm_workload import ArmWorkloadEnv, WorkloadSpec


@dataclass(frozen=True)
class Motion:
    name: str
    low_deg: float = 120.
    high_deg: float = 135.
    seconds: float = 6.
    rise_s: float = 1.6
    fall_s: float = 1.6
    kind: str = 'cycle'

    def sample(self, time):
        """Return analytic angle, velocity and acceleration, all in SI units."""
        low, high = np.deg2rad([self.low_deg, self.high_deg])
        t = max(0., time - .5)
        if self.kind == 'hold':
            return np.array([low, 0., 0.])
        if self.kind == 'oscillation':
            if time < .5:
                return np.array([low, 0., 0.])
            # Quintic half-waves: smooth starts, reversals and accelerations.
            duration = self.rise_s
            phase = int(t / duration)
            start, end = (low, high) if phase % 2 == 0 else (high, low)
            return self._segment(start, end, t % duration, duration)
        if t < self.rise_s:
            return self._segment(low, high, t, self.rise_s)
        if self.kind == 'reach' or t < self.rise_s + .5:
            return np.array([high, 0., 0.])
        return self._segment(high, low, t-self.rise_s-.5, self.fall_s)

    @staticmethod
    def _segment(start, end, time, duration):
        x = float(np.clip(time / duration, 0., 1.))
        delta = end-start
        return np.array([start+delta*(10*x**3-15*x**4+6*x**5),
                         delta*(30*x**2-60*x**3+30*x**4)/duration,
                         delta*(60*x-180*x**2+120*x**3)/duration**2])


TRAIN_MOTIONS = (
    Motion('hold120', kind='hold'), Motion('hold135', 135., 135., kind='hold'),
    Motion('raise', kind='reach'), Motion('lower', 135., 120., kind='reach'),
    Motion('oscillation', kind='oscillation'),
    Motion('fast_up_soft_down', rise_s=.8, fall_s=2.4),
    Motion('soft_up_fast_down', rise_s=2.4, fall_s=.8),
)
VALIDATION_MOTIONS = (
    Motion('val_raise', 122., 133., rise_s=1.3, kind='reach'),
    Motion('val_oscillation', 123., 132., rise_s=1.3, kind='oscillation'),
    Motion('val_asymmetric', 121., 134., rise_s=1., fall_s=2.1),
)
TEST_MOTIONS = (
    Motion('test_lower', 134., 121., rise_s=1.4, kind='reach'),
    Motion('test_oscillation', 122., 134., rise_s=1.7, kind='oscillation'),
    Motion('test_asymmetric', 122., 133., rise_s=2.2, fall_s=.9),
)


@dataclass(frozen=True)
class HierarchySpec:
    kp: float = 16.
    kd: float = 8.
    acceleration_limit: float = 20.
    acceleration_scale: float = 5.
    lead_steps: int = 2
    # Start without an extra exploration intervention. Compare it separately.
    broaden_on_stall: bool = False


def upper_command(motion, time, q, velocity, measured_acceleration, spec):
    """Causal constant-acceleration projection; no simulated future state."""
    lead, dt = spec.lead_steps*.02, .02
    horizon = lead + dt/2
    projected_q = q + horizon*velocity + .5*horizon**2*measured_acceleration
    projected_v = velocity + horizon*measured_acceleration
    target = motion.sample(time+horizon)
    feedforward = (motion.sample(time+lead+dt)[1]-motion.sample(time+lead)[1])/dt
    requested = feedforward + spec.kp*(target[0]-projected_q) + spec.kd*(target[1]-projected_v)
    return float(np.clip(requested, -spec.acceleration_limit, spec.acceleration_limit))


class ArmHierarchyEnv(ArmWorkloadEnv):
    """Lower policy sees physical history and issued acceleration deadlines only."""
    def __init__(self, spec=None, motions=None, *, masses=(1.,), tilts=(120.,), worker=0, workers=1):
        self.hierarchy = spec or HierarchySpec()
        if self.hierarchy.lead_steps != 2:
            raise ValueError('This protocol requires the native 40ms delay and 20ms interval')
        self.motions = tuple(motions or TRAIN_MOTIONS)
        self.tasks = np.zeros(3)
        self.pressure = 0.
        self.motion = self.motions[0]
        self.step_index = 0
        self.last_acceleration = 0.
        self.error_window = []
        super().__init__(WorkloadSpec(masses=masses, tilts=tilts, cases=((120., 120.),),
                                     seconds=self.motion.seconds), worker=worker, workers=workers)
        self.workload_contract = super().contract()
        self.workload_names = list(self.observation_names)
        # Delete all present and historical task-goal channels, retain motor history.
        self.keep = [i for i, n in enumerate(self.workload_names)
                     if n not in self.unwrapped._command_names and '/command/' not in n]
        self.observation_names = [self.workload_names[i] for i in self.keep] + [
            f'acceleration/due_in_{ms}ms_rad_s2_scaled' for ms in (0, 20, 40)] + ['exploration/pressure']
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                           shape=(len(self.observation_names),), dtype=np.float32)
        self.schedule = [(motion, mass, tilt) for motion in self.motions for mass in masses for tilt in tilts]
        order = np.random.default_rng(0).permutation(len(self.schedule))
        self.schedule = [self.schedule[i] for i in order]

    def encode(self, observation):
        native = super().encode(observation)
        return np.concatenate((native[self.keep], self.tasks/self.hierarchy.acceleration_scale,
                               [self.pressure])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        index = self.worker+self.workers*self.episode_index
        motion, mass, tilt = self.schedule[index % len(self.schedule)]
        self.motion = options.pop('motion', motion)
        self.tasks[:] = 0.
        self.pressure, self.last_acceleration, self.step_index = 0., 0., 0
        self.error_window = []
        # ArmWorkloadEnv's schedule has a different cell type; call its reset with
        # a temporary legal cell. Its actual physical reset/mount code is reused.
        schedule = self.schedule
        self.schedule = [(self.motion.low_deg, self.motion.low_deg, mass, tilt, 1.)]
        try:
            _, info = super().reset(seed=seed, options={'seconds': self.motion.seconds+.04,
                'payload_mass': options.pop('payload_mass', mass),
                'tilt_deg': options.pop('tilt_deg', tilt), **options})
        finally:
            self.schedule = schedule
        self.tasks[-1] = self.issue_command()
        return self.observation(), info

    def issue_command(self):
        base = self.unwrapped
        return upper_command(self.motion, base.time, float(base.data.qpos[base.qpos_index]),
                             float(base.data.qvel[base.dof_index]), self.last_acceleration, self.hierarchy)

    def step(self, action):
        base = self.unwrapped
        before = float(base.data.qvel[base.dof_index])
        desired = float(self.tasks[0])
        issue_step = self.step_index - 2
        _, _, terminated, truncated, info = super().step(action)
        actual = float((info['qvel']-before)/base.control_dt)
        self.last_acceleration = actual
        scored = issue_step >= 0
        error = actual-desired
        # Non-flat tails, no effort/position reward hidden in the lower task.
        tracking = -float(np.log1p((error/self.hierarchy.acceleration_scale)**2)) if scored else 0.
        reward = .1*(tracking-float(terminated))
        self.error_window.append(abs(error))
        self.error_window = self.error_window[-40:]
        stagnant = (scored and len(self.error_window) == 40 and np.mean(self.error_window[-20:]) > 2.
                    and np.mean(self.error_window[:20])-np.mean(self.error_window[-20:]) < .1)
        if self.hierarchy.broaden_on_stall:
            self.pressure = float(np.clip(self.pressure + (.1 if stagnant else -.2), 0., 1.))
        self.step_index += 1
        self.tasks[:-1] = self.tasks[1:]
        self.tasks[-1] = self.issue_command() if self.step_index < round(self.motion.seconds/.02) else 0.
        reference = self.motion.sample(base.time)
        info['hierarchy'] = {'issue_step': issue_step, 'scored': scored, 'desired_acceleration': desired,
            'actual_acceleration': actual, 'acceleration_error': error, 'reference_q': float(reference[0]),
            'reference_velocity': float(reference[1]), 'reference_acceleration': float(reference[2]),
            'position_error': float(info['q']-reference[0]),
            'velocity_error': float(info['qvel']-reference[1]), 'pressure': self.pressure}
        info['reward_components'] = {'acceleration_log_error': tracking, 'physical_failure': -float(terminated)}
        return self.observation(), reward, terminated, truncated, info

    def contract(self):
        return {'schema': 'myo.arm_hierarchy.v1', 'native': self.workload_contract,
            'hierarchy': asdict(self.hierarchy), 'motions': [asdict(m) for m in self.motions],
            'observation_names': self.observation_names, 'retained_native_indices': self.keep,
            'action_mapping': list(self.unwrapped.muscle_names), 'predictor': None,
            'timing': 'issued t; mean angular acceleration [t+40ms,t+60ms); two unscored warmups, two flush steps',
            'upper': 'analytic reference + PD on causal constant-acceleration projection at t+50ms',
            'reward': '-0.1*log1p((actual-desired)^2/scale^2), failure -0.1; no effort or position reward',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def heuristic_action(self):
        """Diagnostic inverse-force allocation, same delayed compliant muscle path."""
        base = self.unwrapped
        velocity = float(base.data.qvel[base.dof_index])
        requested = (float(base.model.dof_M0[base.dof_index])*self.tasks[-1]
                     +float(base.data.qfrc_bias[base.dof_index])
                     +float(base.model.dof_damping[base.dof_index])*velocity-base.passive_limit_torque())
        capacities = []
        for muscle in base.muscles:
            p = muscle.params
            length = muscle.fiber_length/p.optimal_fiber_length
            factor = max(0., 1.-((length-1.)/p.force_length_width)**2)
            capacities.append(p.max_isometric_force*p.active_force_scale*muscle.capacity*factor)
        torques = base.moment_arms*np.array(capacities)
        passive = float(base.moment_arms @ np.array([o.passive_force for o in base._muscle_outputs]))
        residual = requested-passive
        selected = torques > 0 if residual >= 0 else torques < 0
        action = np.zeros(6)
        available = float(np.abs(torques[selected]).sum())
        if available > 1e-8:
            action[selected] = abs(residual)/available
        return np.clip(action, 0., 1.).astype(np.float32)


def evaluate(model, env, motions, *, mass=1., tilt=120., traces=False):
    rows = []
    for motion in motions:
        observation, _ = env.reset(seed=0, options={'motion': motion, 'payload_mass': mass, 'tilt_deg': tilt})
        samples = []
        while True:
            action = env.heuristic_action() if model == 'heuristic' else model.predict(observation, deterministic=True)[0]
            observation, reward, failed, done, info = env.step(action)
            samples.append({'t': env.unwrapped.time, **info['hierarchy'], 'q': info['q'], 'v': info['qvel'],
                            'action': np.asarray(action).tolist(), 'activation': info['activation'], 'reward': reward})
            if failed or done:
                break
        selected = [s for s in samples if s['t'] >= .5 and s['scored']]
        def rmse(name):
            return float(np.sqrt(np.mean([s[name]**2 for s in selected]))) if selected else 1e6
        q_rmse, v_rmse, a_rmse = np.rad2deg(rmse('position_error')), np.rad2deg(rmse('velocity_error')), rmse('acceleration_error')
        row = {'motion': motion.name, 'mass': mass, 'tilt': tilt, 'failure': bool(failed),
            'position_rmse_deg': float(q_rmse), 'velocity_rmse_deg_s': float(v_rmse),
            'acceleration_rmse_rad_s2': a_rmse,
            'passed': bool(not failed and q_rmse <= 3. and v_rmse <= 10. and a_rmse <= 3.)}
        if traces:
            row['trace'] = samples
        rows.append(row)
    return {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
            'failures': sum(r['failure'] for r in rows), 'cases': rows,
            'mean_position_rmse_deg': float(np.mean([r['position_rmse_deg'] for r in rows])),
            'mean_acceleration_rmse_rad_s2': float(np.mean([r['acceleration_rmse_rad_s2'] for r in rows]))}


def save_policy(model, env, path, training):
    path = Path(path)
    model.hierarchy_contract = env.contract()
    model.hierarchy_training = training
    model.save(path)
    metadata = {'contract': env.contract(), 'training': training,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+'.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def load_policy(path):
    from .arm_learning import ArmPPO
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    contract = meta['contract']
    spec = contract['native']['spec']
    env = ArmHierarchyEnv(HierarchySpec(**contract['hierarchy']),
        [Motion(**m) for m in contract['motions']], masses=tuple(spec['masses']), tilts=tuple(spec['tilts']))
    try:
        if env.contract() != contract or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Hierarchy checkpoint contract or bytes mismatch')
        model = ArmPPO.load(path, device='cpu')
        if model.hierarchy_contract != contract or model.hierarchy_training != meta['training']:
            raise ValueError('Hierarchy checkpoint internal identity mismatch')
        return model, env, meta
    except BaseException:
        env.close()
        raise
