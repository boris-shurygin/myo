"""Causal acceleration commands for a future interval of the native muscle rig."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .vertical_control import VerticalControlEnv, load_policy
from .vertical_muscles import VerticalConfig


@dataclass(frozen=True)
class AccelerationCase:
    name: str
    amplitude_m_s2: float
    hold_steps: int
    initial_position_m: float = 0.

    def commands(self):
        # Four balanced lobes return ideal velocity and displacement to zero.
        return np.repeat(np.tile([self.amplitude_m_s2, -self.amplitude_m_s2,
                                  -self.amplitude_m_s2, self.amplitude_m_s2, 0.], 2), self.hold_steps)


TRAIN_CASES = (AccelerationCase("up_short", .5, 5), AccelerationCase("down_short", -.5, 5),
               AccelerationCase("up_medium", 1., 8, -.03), AccelerationCase("down_medium", -1., 8, .03),
               AccelerationCase("up_long", .75, 10), AccelerationCase("down_long", -.75, 10),
               AccelerationCase("zero", 0., 8))
VALIDATION_CASES = (AccelerationCase("up_120", .6, 6, -.02), AccelerationCase("down_120", -.6, 6, .02),
                    AccelerationCase("up_180", .9, 9), AccelerationCase("down_180", -.9, 9),
                    AccelerationCase("zero_140", 0., 7, .02))
TEST_CASES = (AccelerationCase("up_140", .7, 7, .02), AccelerationCase("down_140", -.7, 7, -.02),
              AccelerationCase("up_200", .8, 10, -.015), AccelerationCase("down_200", -.8, 10, .015),
              AccelerationCase("zero_180", 0., 9, -.02))
REWARD = {"acceleration_weight": 1., "acceleration_scale_m_s2": .5,
          "excitation_squared_weight": .01, "excitation_change_weight": .002,
          "physical_failure": -1., "training_multiplier": .1}
ACCURACY = {"rmse_m_s2": .25, "absolute_error_m_s2": .35, "fraction": .8,
            "failure_score_m_s2": 3.}


class ScheduledAccelerationEnv(VerticalControlEnv):
    """Only issued tasks enter observations; the new task is due 40 ms later."""
    def __init__(self, config=VerticalConfig()):
        super().__init__(config)
        self.lead_steps = round(.04/config.dt)
        if not np.isclose(self.lead_steps*config.dt, .04):
            raise ValueError("40 ms task lead must contain whole control intervals")
        self.tasks = np.zeros(self.lead_steps+1)
        self.observation_names = self.observation_names[:-1]+[
            f"desired_acceleration_m_s2/due_in_{round(i*config.dt*1000)}ms" for i in range(self.lead_steps+1)]
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation_names),), dtype=np.float32)

    def observation(self):
        # Reuse the tested native-state/action transform, excluding position goal.
        return np.r_[super().observation()[:-1], self.tasks].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.case_index = 0
        if options and "case" in options:
            self.case = options["case"]
        else:
            self.case = TRAIN_CASES[self.case_index % len(TRAIN_CASES)]
            self.case_index += 1
        self.commands = self.case.commands()
        self.step_index = 0
        self.tasks[:] = 0.
        self.tasks[-1] = self.commands[0]
        self.episode_seconds = (len(self.commands)+self.lead_steps)*self.physical.config.dt
        self.physical.reset(self.case.initial_position_m)
        self.previous = self.physical.excitation.copy()
        self.done = False
        return self.observation(), {}

    def step(self, action):
        due, issued = float(self.tasks[0]), float(self.tasks[-1])
        velocity, started = self.physical.velocity, self.physical.time
        index = self.step_index
        _, _, terminated, truncated, info = super().step(action)
        actual = (self.physical.velocity-velocity)/(self.physical.time-started)
        error = actual-due
        parts = {"acceleration": REWARD["acceleration_weight"]*np.exp(-(error/REWARD["acceleration_scale_m_s2"])**2),
                 "effort": info["reward_components"]["effort"],
                 "command_change": info["reward_components"]["command_change"],
                 "physical_failure": info["reward_components"]["physical_failure"]}
        self.step_index += 1
        self.tasks[:-1] = self.tasks[1:]
        self.tasks[-1] = self.commands[self.step_index] if self.step_index < len(self.commands) else 0.
        for key in ("target_m", "error_m"):
            info.pop(key)
        info.update(reward_components=parts, raw_reward=float(sum(parts.values())),
                    desired_acceleration_m_s2=due, issued_acceleration_m_s2=issued,
                    actual_acceleration_m_s2=float(actual), acceleration_error_m_s2=float(error),
                    task_issue_step=index-self.lead_steps, task_due_step=index,
                    scored=self.lead_steps <= index < len(self.commands)+self.lead_steps)
        return self.observation(), REWARD["training_multiplier"]*info["raw_reward"], terminated, truncated, info

    def contract(self):
        return {**super().contract(), "schema": "myo.vertical_scheduled_ppo.v1", "reward": REWARD,
                "observation_transform": "native physical/action transform; three issued acceleration tasks in m/s2; no position goal",
                "task_lead_s": .04, "task_queue_size": self.lead_steps+1,
                "task_semantics": "issued at t, average acceleration due over [t+.04,t+.04+control_dt); reward at interval end",
                "task_reset": "earlier unissued intervals request zero; final issued tasks are flushed before truncation",
                "scheduled_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_scheduled_policy(path):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    env = ScheduledAccelerationEnv(VerticalConfig(**metadata["contract"]["physical"]["config"]))
    model, metadata = load_policy(path, env)
    return model, env, metadata


def evaluate_scheduled(model, config, cases):
    results = []
    for case in cases:
        env = ScheduledAccelerationEnv(config)
        obs, _ = env.reset(options={"case": case})
        samples, errors, raw, failed = [], [], [], False
        for _ in range(len(env.commands)+env.lead_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, failed, truncated, info = env.step(action)
            if info["scored"]:
                errors.append(info["acceleration_error_m_s2"])
            raw.append(info["raw_reward"])
            samples.append([float(v) for v in (env.physical.time, env.physical.position, env.physical.velocity,
                info["issued_acceleration_m_s2"], info["desired_acceleration_m_s2"],
                info["actual_acceleration_m_s2"], *action)])
            if failed or truncated:
                break
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        fraction = float(np.mean(np.abs(errors) <= ACCURACY["absolute_error_m_s2"])) if errors else 0.
        passed = not failed and rmse is not None and rmse <= ACCURACY["rmse_m_s2"] and fraction >= ACCURACY["fraction"]
        results.append({"case": asdict(case), "physical_failure": bool(failed), "duration_s": env.physical.time,
                        "scored_steps": len(errors), "acceleration_rmse_m_s2": rmse,
                        "within_tolerance_fraction": fraction, "passed": bool(passed), "raw_return": float(sum(raw)),
                        "samples": samples})
    score = float(np.mean([r["acceleration_rmse_m_s2"] if not r["physical_failure"] else ACCURACY["failure_score_m_s2"]
                           for r in results]))
    return {"score_m_s2": score, "passed": sum(c["passed"] for c in results), "total": len(results),
            "failures": sum(c["physical_failure"] for c in results), "ready": all(c["passed"] for c in results), "cases": results}


def compact_scheduled(result):
    return {**result, "cases": [{k: v for k, v in c.items() if k != "samples"} for c in result["cases"]]}
