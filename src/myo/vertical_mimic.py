"""Fixed-base trajectory imitation on the native two-muscle vertical stand."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .vertical_muscles import VerticalConfig, VerticalMuscles


COMMIT = "47b653db2bce25de834360d2ee6b2dd88142bd78"
SOURCE_FILES = (
    "musclemimic/algorithms/common/networks.py",
    "musclemimic/algorithms/common/optimizer.py",
    "musclemimic/algorithms/ppo/loss.py",
    "musclemimic/rl_core/rollout_buffer.py",
    "bimanual/conf_bimanual.yaml",
    "musclemimic/core/reward/trajectory_based.py",
    "musclemimic/core/goals/bimanual.py",
    "musclemimic/algorithms/ppo/runner.py",
    "musclemimic/core/wrappers/mjx.py",
    "loco_mujoco/core/control_functions/default.py",
    "loco_mujoco/core/initial_state_handler/traj_init_state.py",
    "LICENSE",
)


def source_manifest(root):
    root = Path(root)
    if (root / "PINNED_COMMIT").read_text().strip() != COMMIT:
        raise ValueError("MuscleMimic source commit mismatch")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


@dataclass(frozen=True)
class Motion:
    name: str
    amplitude_m: float
    period_s: float
    offset_m: float = 0.
    phase_rad: float = 0.

    def sample(self, t):
        w = 2*np.pi/self.period_s
        phase = w*np.asarray(t)+self.phase_rad
        return self.offset_m+self.amplitude_m*np.sin(phase), self.amplitude_m*w*np.cos(phase)

    @property
    def duration_s(self):
        return 2*self.period_s


TRAIN_MOTIONS = (Motion("sine_60_3", .06, 3.), Motion("sine_100_4", .10, 4.),
                 Motion("sine_80_25", .08, 2.5), Motion("shift_up", .06, 3.5, .025),
                 Motion("shift_down", .06, 3.5, -.025), Motion("hold", 0., 3.))
VALIDATION_MOTIONS = (Motion("val_70", .07, 3.2), Motion("val_90", .09, 3.8),
                      Motion("val_shift", .055, 2.8, .015), Motion("val_hold", 0., 3., -.02))
TEST_MOTIONS = (Motion("test_75", .075, 3.1), Motion("test_95", .095, 3.7),
                Motion("test_up", .06, 2.9, .02), Motion("test_down", .06, 3.3, -.02),
                Motion("test_hold", 0., 3., .02))


class VerticalMimicEnv(gym.Env):
    """Known reference preview is a task, never a future physical outcome.

    Reward weights/temperatures and the default direct action adapter follow
    the pinned bimanual config. Constant orientation terms are retained.
    Independent fiber state and pending motor commands extend its observations
    because our native stand has compliant tendons and a transport queue.
    """
    metadata = {"render_modes": []}

    def __init__(self, config=VerticalConfig(frame_skip=10), motions=TRAIN_MOTIONS):
        super().__init__()
        if not np.isclose(config.dt, .01):
            raise ValueError("This protocol uses the source's 100 Hz control rate")
        self.physical = VerticalMuscles(config)
        self.motions = tuple(motions)
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        names = ["q_m", "v_m_s"]
        for quantity in ("mt_length_m", "mt_velocity_m_s", "tension_negative_N", "excitation", "activation"):
            names += [f"{quantity}/{side}" for side in ("upper", "lower")]
        names += ["delivered/upper", "delivered/lower"]
        if config.elastic:
            names += ["fiber_m/upper", "fiber_m/lower"]
        names += [f"pending/{k}/{side}" for k in range(config.queue_steps) for side in ("upper", "lower")]
        names += [f"site_position/{axis}" for axis in "xyz"]
        names += [f"site_rotation/{axis}" for axis in "xyz"]
        names += [f"site_velocity/{kind}/{axis}" for kind in ("angular", "linear") for axis in "xyz"]
        names += [f"reference/q/+{k*200}ms" for k in range(5)]
        names += [f"reference/v/+{k*200}ms" for k in range(5)]
        names += [f"reference/site/+{k*200}ms/{axis}" for k in range(5) for axis in "xyz"]
        names += ["motion_phase", "previous_raw_action/upper", "previous_raw_action/lower"]
        self.observation_names = names
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(names),), dtype=np.float32)
        self.done = True

    def reference(self, frame=None):
        frame = self.frame if frame is None else frame
        frame = np.clip(frame, 0, self.last_frame)
        return self.motion.sample(frame*self.physical.config.dt)

    def observation(self):
        p = self.physical
        values = [p.position, p.velocity, *p.path_lengths(), -p.velocity, p.velocity,
                  *(-p.forces()), *p.excitation, *p.activation, *p.delivered]
        if p.config.elastic:
            values.extend(p.fiber_lengths())
        if p.config.delayed:
            values.extend(p.state()[p.config.physical_size:])
        # The reference root is the fixed lower anchor; the moving load is one site.
        values.extend([0., 0., p.config.half_span_m+p.position, 0., 0., 0.,
                       0., 0., 0., 0., 0., p.velocity])
        q, v = self.reference(self.frame+np.arange(5)*20)
        values.extend(q)
        values.extend(v)
        for height in q:
            values.extend([0., 0., p.config.half_span_m+height])
        values.extend([self.frame/self.last_frame, *self.previous])
        return np.asarray(values, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        index = options.get("motion_index")
        index = int(self.np_random.integers(len(self.motions))) if index is None else index
        self.motion = self.motions[index]
        self.last_frame = round(self.motion.duration_s/self.physical.config.dt)
        self.frame = int(options.get("frame", self.np_random.integers(self.last_frame)))
        self.start_frame = self.frame
        q, v = self.reference()
        p = self.physical
        p.reset(float(q))
        # No inverse-dynamics teacher preload: source resets kinematics, with
        # template act/ctrl zero. Native elastic fibers start at zero-act equilibrium.
        for muscle, length in zip(p.muscles, p.path_lengths()):
            muscle.reset(float(length), activation=0.)
        for delay in p.delays:
            delay.reset(0.)
        p.excitation = np.zeros(2)
        p.delivered = np.zeros(2)
        p.activation = np.zeros(2)
        p.applied_forces = p.forces() if p.config.elastic else np.zeros(2)
        p.data.qvel[0] = v
        mujoco.mj_forward(p.model, p.data)
        self.previous = np.zeros(2)
        self.steps = 0
        self.done = False
        return self.observation(), {"frame": self.frame, "motion": self.motion.name}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset a completed imitation episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Two finite raw Gaussian actions required")
        excitation = np.clip((action+1)/2, 0., 1.)
        physical_failure = False
        try:
            self.physical.step(excitation)
        except RuntimeError as exc:
            if "left the benchmark workspace" not in str(exc):
                raise
            physical_failure = True
        self.steps += 1
        self.frame = min(self.frame+1, self.last_frame)
        q, v = self.reference()
        eq, ev = self.physical.position-q, self.physical.velocity-v
        # Single translational DOF/site, fixed orientation/angular velocity.
        # Pinned MimicReward averages all xyz entries, including the two
        # identically zero directions here, rather than summing a site's norm.
        site_position_error = np.mean(np.square([0., 0., eq]))
        site_velocity_error = np.mean(np.square([0., 0., ev]))
        components = {"qpos": .1*np.exp(-10*eq**2), "qvel": .1*np.exp(-2*ev**2),
                      "site_position": 2.*np.exp(-100*site_position_error), "site_orientation": .2,
                      "site_angular_velocity": .5, "site_linear_velocity": .5*np.exp(-.1*site_velocity_error),
                      "bounds_penalty": max(-1., -.01*float(np.mean(np.maximum(np.abs(action)-1, 0)**2)))}
        reward = max(0., float(sum(components.values())))
        terminated = physical_failure or abs(eq) > .25
        truncated = not terminated and (self.frame == self.last_frame or self.steps >= 1000)
        self.done = terminated or truncated
        self.previous = action.copy()
        info = {"q_m": self.physical.position, "v_m_s": self.physical.velocity,
                "reference_q_m": float(q), "reference_v_m_s": float(v),
                "error_q_m": float(eq), "error_v_m_s": float(ev), "reward_components": components,
                "physical_failure": bool(physical_failure), "tracking_failure": bool(abs(eq) > .25),
                "frame": self.frame, "excitation": excitation.tolist(),
                "delivered": self.physical.delivered.tolist(), "activation": self.physical.activation.tolist()}
        return self.observation(), reward, bool(terminated), bool(truncated), info

    def contract(self):
        return {"schema": "myo.vertical_musclemimic.v1", "physical": self.physical.contract(),
                "observations": self.observation_names, "observation_transform": "SI; source RunningMeanStd",
                "action": "raw independent Gaussian; clip((a+1)/2,0,1); previous raw action",
                "reference": "analytic sine/hold; five frames at 0/200/400/600/800ms; clamp at clip end",
                "reset": "reference q/v, zero excitation/activation/motor queue; native zero-act fiber equilibrium",
                "reward": "pinned conf_bimanual weights and MimicReward default temperatures; constant rotations retained",
                "termination": "site error > .25m or native physical failure; horizon1000 or reference end",
                "predictor_identity": None, "source_commit": COMMIT,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def evaluate_mimic(policy, config, motions, phases=(0.,), stochastic=False, seed=123):
    cases = []
    for index, motion in enumerate(motions):
        for phase in phases:
            env = VerticalMimicEnv(config, motions)
            frame = round(phase*round(motion.duration_s/config.dt))
            obs, _ = env.reset(seed=seed, options={"motion_index": index, "frame": frame})
            samples = []
            total_reward = 0.
            while True:
                action = policy(obs, stochastic=stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                samples.append([float(x) for x in (env.physical.time, info["q_m"], info["v_m_s"], info["reference_q_m"],
                                info["reference_v_m_s"], *action, *info["excitation"],
                                *info["delivered"], *info["activation"], reward)])
                if terminated or truncated:
                    break
            a = np.asarray(samples)
            qrmse = float(np.sqrt(np.mean((a[:, 1]-a[:, 3])**2)))
            vrmse = float(np.sqrt(np.mean((a[:, 2]-a[:, 4])**2)))
            complete = not terminated and env.frame == env.last_frame
            cases.append({"motion": asdict(motion), "start_phase": phase, "completed": complete,
                          "physical_failure": info["physical_failure"], "tracking_failure": info["tracking_failure"],
                          "q_rmse_m": qrmse, "v_rmse_m_s": vrmse,
                          "coverage": (env.frame-frame)/(env.last_frame-frame),
                          "passed": complete and qrmse <= .01 and vrmse <= .05,
                          "return": total_reward, "samples": samples})
    return {"completed": sum(c["completed"] for c in cases), "passed": sum(c["passed"] for c in cases),
            "total": len(cases), "mean_coverage": float(np.mean([c["coverage"] for c in cases])),
            "q_rmse_m": float(np.mean([c["q_rmse_m"] for c in cases])),
            "v_rmse_m_s": float(np.mean([c["v_rmse_m_s"] for c in cases])),
            "selection_score_m": float(np.mean([c["q_rmse_m"] if c["completed"] else .3 for c in cases])),
            "cases": cases}


def compact_mimic(result):
    return {**result, "cases": [{k: v for k, v in c.items() if k != "samples"} for c in result["cases"]]}
