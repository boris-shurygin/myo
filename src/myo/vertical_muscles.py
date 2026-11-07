"""Minimal 1-DOF vertical load between an upper and a lower tensile muscle.

MuJoCo integrates the load. Variants add passive fiber elasticity, an
independent series tendon, transport delay, and activation dynamics. No arm,
contacts, fatigue, learned policy, or direct position/velocity correction.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .delay import FixedStepDelay
from .muscle import MuscleParameters, SeriesElasticMuscle


@dataclass(frozen=True)
class VerticalConfig:
    variant: int = 1
    physics_dt: float = .001
    frame_skip: int = 20
    mass_kg: float = 1.
    half_span_m: float = .6
    motor_delay_s: float = .04
    activation_tau_s: float = .015
    deactivation_tau_s: float = .05
    max_force_N: float = 80.
    fiber_optimal_m: float = .48
    tendon_slack_m: float = .12
    tendon_strain_at_max_force: float = .04
    passive_stiffness: float = 4.

    @property
    def dt(self):
        return self.physics_dt*self.frame_skip

    @property
    def elastic(self):
        return self.variant >= 2

    @property
    def delayed(self):
        return self.variant >= 3

    @property
    def activation_dynamic(self):
        return self.variant == 4

    @property
    def physical_size(self):
        return 2+2*self.elastic+2*self.activation_dynamic

    @property
    def queue_steps(self):
        return round(self.motor_delay_s/self.dt) if self.delayed else 0


class VerticalMuscles:
    def __init__(self, config=VerticalConfig()):
        self.config = config
        if config.variant not in (1, 2, 3, 4):
            raise ValueError("Choose variant 1, 2, 3 or 4")
        if config.delayed and not np.isclose(config.queue_steps*config.dt, config.motor_delay_s):
            raise ValueError("Motor delay must be an integer number of control intervals")
        h = config.half_span_m
        self.model = mujoco.MjModel.from_xml_string(f'''<mujoco model="vertical_two_muscles">
          <option timestep="{config.physics_dt}" gravity="0 0 -9.81" integrator="implicitfast"/>
          <visual><global azimuth="90" elevation="0"/></visual>
          <worldbody>
            <light pos="1 -1 2"/>
            <site name="top" pos="0 0 {2*h}" size=".018" rgba=".3 .3 .3 1"/>
            <site name="bottom" pos="0 0 0" size=".018" rgba=".3 .3 .3 1"/>
            <body pos="0 0 {h}">
              <joint name="height" type="slide" axis="0 0 1" damping="0"/>
              <geom type="sphere" size=".04" mass="{config.mass_kg}" contype="0" conaffinity="0" rgba=".2 .3 .7 1"/>
              <site name="load" size=".005"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="upper_path" width=".012" rgba=".7 .2 .2 1"><site site="top"/><site site="load"/></spatial>
            <spatial name="lower_path" width=".012" rgba=".2 .6 .2 1"><site site="load"/><site site="bottom"/></spatial>
          </tendon>
        </mujoco>''')
        self.data = mujoco.MjData(self.model)
        # Transport is external in every variant; never apply a second delay.
        self.parameters = MuscleParameters(max_isometric_force=config.max_force_N,
            optimal_fiber_length=config.fiber_optimal_m, tendon_slack_length=config.tendon_slack_m,
            tendon_strain_at_max_force=config.tendon_strain_at_max_force, motor_delay=0.,
            activation_time_constant=config.activation_tau_s, deactivation_time_constant=config.deactivation_tau_s,
            passive_stiffness=config.passive_stiffness if config.elastic else 0., fatigue_rate=0., recovery_rate=0.,
            force_length_width=.8)
        self.muscles = [SeriesElasticMuscle(self.parameters, config.physics_dt) for _ in range(2)]
        self.delays = [FixedStepDelay(config.physics_dt, config.motor_delay_s if config.delayed else 0.) for _ in range(2)]
        self.excitation = self.delivered = self.activation = np.zeros(2)
        self.applied_forces = np.zeros(2)

    @property
    def position(self):
        return float(self.data.qpos[0])

    @property
    def velocity(self):
        return float(self.data.qvel[0])

    @property
    def time(self):
        return float(self.data.time)

    def path_lengths(self):
        return self.config.half_span_m+np.array([-self.position, self.position])

    def fiber_lengths(self):
        return (np.array([m.fiber_length for m in self.muscles]) if self.config.elastic
                else self.path_lengths()-self.parameters.tendon_slack_length)

    def forces(self):
        if self.config.elastic:
            return np.array([m.sample(float(length)).tendon_force for m,length in zip(self.muscles, self.path_lengths())])
        return self.applied_forces.copy()

    def command_for_force(self, net_upward_N, common_N):
        lengths = self.fiber_lengths()
        passive = np.array([m._fiber_forces(float(length), 0., 1.)[1] for m,length in zip(self.muscles,lengths)])
        bottom = max(passive[1]+common_N, passive[0]+common_N-net_upward_N)
        forces = np.array([bottom+net_upward_N, bottom])
        coefficient = np.array([m._fiber_forces(float(length), 1., 1.)[0] for m,length in zip(self.muscles,lengths)])
        return np.clip((forces-passive)/np.maximum(coefficient, 1e-8), 0., 1.)

    def reset(self, position=0.):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = position
        mujoco.mj_forward(self.model, self.data)
        # Solve a small fixed point for a balanced preload; reset is not part
        # of any predicted transition. Passive preload differs with position.
        if self.config.elastic:
            for muscle,length in zip(self.muscles, self.path_lengths()):
                muscle.reset(float(length), activation=0.)
        command = self.command_for_force(self.config.mass_kg*9.81, 5.)
        if self.config.elastic:
            for _ in range(30):
                for muscle,length,a in zip(self.muscles,self.path_lengths(),command):
                    muscle.reset(float(length), activation=float(a))
                command = self.command_for_force(self.config.mass_kg*9.81, 5.)
            for muscle,length,a in zip(self.muscles,self.path_lengths(),command):
                muscle.reset(float(length), activation=float(a))
        for delay,a in zip(self.delays,command):
            delay.reset(float(a))
        self.excitation, self.delivered, self.activation = command.copy(), command.copy(), command.copy()
        self.applied_forces = (self.forces() if self.config.elastic else np.array([
            m._fiber_forces(float(length),float(a),1.)[0]
            for m,length,a in zip(self.muscles,self.fiber_lengths(),command)]))
        return self.state()

    def state(self):
        physical = [self.position,self.velocity]
        if self.config.elastic:
            physical.extend(self.fiber_lengths())
        if self.config.activation_dynamic:
            physical.extend(self.activation)
        if self.config.delayed:
            pending = np.stack([d.pending.reshape(self.config.queue_steps,self.config.frame_skip)[:,0] for d in self.delays],axis=-1)
            physical.extend(pending.ravel())
        return np.asarray(physical,dtype=np.float32)

    def step(self, command):
        command = np.asarray(command,dtype=float)
        if command.shape != (2,) or np.any((command < 0)|(command > 1)) or not np.isfinite(command).all():
            raise ValueError("Two finite excitation commands in [0,1] required")
        self.excitation = command.copy()
        for _ in range(self.config.frame_skip):
            self.delivered = np.array([d.step(float(u)) for d,u in zip(self.delays,command)])
            lengths = self.path_lengths()
            velocities = np.array([-self.velocity,self.velocity])
            if self.config.elastic:
                if not self.config.activation_dynamic:
                    # Exactly instantaneous activation, without a tiny tau or
                    # a second queue: a_next == delivered excitation.
                    for m,a in zip(self.muscles,self.delivered):
                        m.activation = float(a)
                outputs = [m.step(float(a),float(length),float(v))
                    for m,a,length,v in zip(self.muscles,self.delivered,lengths,velocities)]
                self.activation = np.array([o.activation for o in outputs])
                self.applied_forces = np.array([o.tendon_force for o in outputs])
            else:
                self.activation = self.delivered.copy()
                self.applied_forces = np.array([m._fiber_forces(float(length),float(a),1.,float(v))[0]
                    for m,length,a,v in zip(self.muscles,self.fiber_lengths(),self.activation,velocities)])
            self.data.qfrc_applied[0] = self.applied_forces[0]-self.applied_forces[1]
            mujoco.mj_step(self.model,self.data)
            if abs(self.position) > .28 or not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Vertical load left the benchmark workspace; no clamp or rescue applied")
        mujoco.mj_forward(self.model,self.data)
        return self.state()

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.model,spec))
        mujoco.mj_getState(self.model,self.data,integration,spec)
        return integration,copy.deepcopy((self.muscles,self.delays,self.excitation,self.delivered,self.activation,self.applied_forces))

    def restore(self, snapshot):
        mujoco.mj_setState(self.model,self.data,snapshot[0],mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model,self.data)
        self.muscles,self.delays,self.excitation,self.delivered,self.activation,self.applied_forces = copy.deepcopy(snapshot[1])

    def contract(self):
        physical = ["position_m","velocity_m_s"]
        if self.config.elastic:
            physical += ["fiber_length_m/upper","fiber_length_m/lower"]
        if self.config.activation_dynamic:
            physical += ["activation/upper","activation/lower"]
        pending = [f"pending/{age}/{name}" for age in range(self.config.queue_steps) for name in ("upper","lower")]
        return {"schema":"myo.vertical_predictor.v1","config":asdict(self.config),"mujoco_version":mujoco.__version__,"physical_names":physical,
            "state_names":physical+pending,"neural_input_names":physical+["delivered_excitation/upper","delivered_excitation/lower"],
            "action_mapping":["upper","lower"],"action_semantics":"excitation_0_1_held_for_control_dt_before_transport",
            "control_dt_s":self.config.dt,"physics_dt_s":self.config.physics_dt,
            "source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),Path(__file__).with_name("muscle.py"),Path(__file__).with_name("delay.py"),Path(__file__).with_name("fatigue.py"))}}


def reference(t, family="sine", version=0):
    phase = .31*version
    amplitude = .09+.008*(version%3)
    omega = 2*np.pi/(3.+.23*version)
    if family == "sine":
        return amplitude*np.sin(omega*t+phase),amplitude*omega*np.cos(omega*t+phase)
    if family == "multisine":
        return (.07*np.sin(omega*t+phase)+.025*np.sin(2.3*omega*t+.7),
                .07*omega*np.cos(omega*t+phase)+.025*2.3*omega*np.cos(2.3*omega*t+.7))
    # Smooth alternation between upper and lower targets, with dwell periods.
    period = 2.2+.1*version
    local = (t+phase)%period
    direction = 1 if int((t+phase)//period)%2 == 0 else -1
    u = min(1.,local/.8)
    smooth = u*u*(3-2*u)
    return direction*amplitude*(2*smooth-1),direction*amplitude*12*u*(1-u)/.8 if u<1 else 0.


def heuristic(env, family="sine", version=0):
    preview = (env.config.motor_delay_s if env.config.delayed else 0.) + (.025 if env.config.activation_dynamic else 0.)
    target,velocity = reference(env.time+preview,family,version)
    position = env.position+preview*env.velocity
    acceleration = 28.*(target-position)+9.*(velocity-env.velocity)
    common = 6.+5.*(1+np.sin(2*np.pi*env.time/(1.1+.09*version)+.4*version))
    return env.command_for_force(env.config.mass_kg*(9.81+acceleration),common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",type=int,choices=(1,2,3,4),default=4)
    parser.add_argument("--seconds",type=float,default=12.)
    parser.add_argument("--family",choices=("sine","steps","multisine"),default="steps")
    parser.add_argument("--view",action="store_true")
    args = parser.parse_args()
    env = VerticalMuscles(VerticalConfig(variant=args.variant))
    env.reset()
    viewer = None
    if args.view:
        from .vertical_viewer import VerticalViewer
        viewer = VerticalViewer(env)
    positions = []
    try:
        for _ in range(round(args.seconds/env.config.dt)):
            started = time.monotonic()
            env.step(heuristic(env,args.family))
            positions.append(env.position)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync(env)
                time.sleep(max(0.,env.config.dt-(time.monotonic()-started)))
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps({"variant":args.variant,"steps":len(positions),"min_m":min(positions),"max_m":max(positions)}))


if __name__ == "__main__":
    main()
