"""Direct excitation control of the unchanged vertical muscle benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .vertical_muscles import VerticalConfig, VerticalMuscles


# Six movements and three holds; fixed order, without random physical resets.
TRAIN_CASES = ((-.06, .08), (0., -.08), (.06, 0.),
               (0., .08), (.06, -.08), (-.06, 0.),
               (0., 0.), (-.06, -.06), (.06, .06))
VALIDATION_CASES = ((-.04, .04), (.04, -.04), (-.04, 0.), (.04, 0.), (0., 0.))
TEST_CASES = ((-.03, .06), (.03, -.06), (-.03, -.03), (.03, .03), (0., 0.))
REWARD = {"position_weight": .85, "position_scale_m": .08,
          "velocity_weight": .15, "velocity_scale_m_s": .4,
          "excitation_squared_weight": .01, "excitation_change_weight": .002,
          "physical_failure": -1., "training_multiplier": .1}


class VerticalControlEnv(gym.Env):
    """Goal-conditioned Markov observations; no predictor or teacher labels."""
    metadata = {"render_modes": []}

    def __init__(self, config=VerticalConfig(), episode_seconds=4.):
        super().__init__()
        self.physical = VerticalMuscles(config)
        self.episode_seconds = episode_seconds
        self.action_space = spaces.Box(0., 1., (2,), dtype=np.float32)
        names = self.physical.contract()["state_names"]
        self.observation_names = names+["previous_excitation/upper", "previous_excitation/lower", "target_position_m"]
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation_names),), dtype=np.float32)
        self.case_index = 0
        self.target = 0.
        self.previous = np.zeros(2)
        self.done = True

    def observation(self):
        state = self.physical.state().copy()
        state[0] /= .1
        state[1] /= .5
        if self.physical.config.elastic:
            state[2:4] = (state[2:4]-.48)/.1
        start = 4 if self.physical.config.elastic else 2
        state[start:] = 2*state[start:]-1
        return np.r_[state, 2*self.previous-1, self.target/.1].astype(np.float32)

    def set_target(self, target):
        if not np.isfinite(target) or abs(target) > .15:
            raise ValueError("Target must be finite and within +/-0.15m")
        self.target = float(target)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.case_index = 0
        if not options:
            position, target = TRAIN_CASES[self.case_index % len(TRAIN_CASES)]
            self.case_index += 1
        else:
            position, target = options["initial_position_m"], options["target_m"]
        self.physical.reset(position)
        self.set_target(target)
        self.previous = self.physical.excitation.copy()
        self.done = False
        return self.observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset a completed episode before stepping")
        action = np.asarray(action, dtype=float)
        failed = False
        try:
            self.physical.step(action)
        except RuntimeError as exc:
            if "left the benchmark workspace" not in str(exc):
                raise
            failed = True
        error = self.physical.position-self.target
        parts = {
            "position": REWARD["position_weight"]*np.exp(-(error/REWARD["position_scale_m"])**2),
            "velocity": REWARD["velocity_weight"]*np.exp(-(self.physical.velocity/REWARD["velocity_scale_m_s"])**2),
            "effort": -REWARD["excitation_squared_weight"]*np.mean(action**2),
            "command_change": -REWARD["excitation_change_weight"]*np.mean((action-self.previous)**2),
            "physical_failure": REWARD["physical_failure"] if failed else 0.,
        }
        self.previous = action.copy()
        truncated = not failed and self.physical.time+1e-10 >= self.episode_seconds
        self.done = failed or truncated
        info = {"position_m": self.physical.position, "velocity_m_s": self.physical.velocity,
                "target_m": self.target, "error_m": error, "reward_components": parts,
                "raw_reward": float(sum(parts.values())), "physical_failure": failed,
                "excitation": self.physical.excitation.copy(), "delivered": self.physical.delivered.copy(),
                "activation": self.physical.activation.copy()}
        return self.observation(), REWARD["training_multiplier"]*info["raw_reward"], failed, truncated, info

    def contract(self):
        return {"schema": "myo.vertical_ppo.v1", "physical": self.physical.contract(),
                "observation_names": self.observation_names,
                "observation_transform": "q/.1,v/.5,(fiber-.48)/.1,2*activation_or_queue-1,2*previous-1,target/.1",
                "reward": REWARD, "predictor_identity": None,
                "action_distribution": "independent Beta; deterministic action is mean; direct excitations",
                "controller_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "policy_source_sha256": hashlib.sha256(Path(__file__).with_name("policy_beta.py").read_bytes()).hexdigest()}


def save_policy(model, env, path, training):
    path = Path(path)
    model.save(path)
    metadata = {"contract": env.contract(), "trained_steps": model.num_timesteps, "training": training,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


def load_policy(path, env):
    from stable_baselines3 import PPO
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["contract"] != env.contract():
        raise ValueError("Vertical policy contract mismatch: physics, observations, reward or policy source")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Vertical policy checkpoint hash mismatch")
    model = PPO.load(path, device="cpu")
    if model.num_timesteps != metadata["trained_steps"]:
        raise ValueError("Vertical policy step counter mismatch")
    return model, metadata


def evaluate_policy(model, config, cases, seconds=4., switches=False):
    """Evaluate means, physical failures and settling, separately from reward."""
    results = []
    for position, target in cases:
        env = VerticalControlEnv(config, episode_seconds=seconds)
        obs, _ = env.reset(options={"initial_position_m": position, "target_m": target})
        samples, reward, failed = [], 0., False
        for step in range(round(seconds/config.dt)):
            if switches:
                sequence = (target, -target, 0.)
                env.set_target(sequence[min(2, int(env.physical.time/2.))])
                obs = env.observation()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, failed, truncated, info = env.step(action)
            reward += info["raw_reward"]
            samples.append([float(x) for x in (env.physical.time, info["position_m"], info["velocity_m_s"], env.target, *action)])
            if failed or truncated:
                break
        a = np.asarray(samples)
        tail = a[:, 0] >= (seconds-1. if not switches else 1.)
        error = a[:, 1]-a[:, 3]
        settled = (np.abs(error) <= .005) & (np.abs(a[:, 2]) <= .03)
        passed = not failed and bool(np.mean(settled[tail]) >= .9) if tail.any() else False
        results.append({"initial_position_m": position, "target_m": target, "physical_failure": bool(failed),
                        "duration_s": float(a[-1, 0]), "raw_return": reward,
                        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
                        "last_second_rmse_m": float(np.sqrt(np.mean(error[tail]**2))) if tail.any() else None,
                        "last_second_settled_fraction": float(np.mean(settled[tail])) if tail.any() else 0.,
                        "passed": passed, "samples": samples})
    # Failures and steady errors both count; a short failed trajectory cannot win.
    score = float(np.mean([r["last_second_rmse_m"] if not r["physical_failure"] else .3 for r in results]))
    return {"score_m": score, "passed": sum(r["passed"] for r in results), "total": len(results),
            "failures": sum(r["physical_failure"] for r in results), "ready": all(r["passed"] for r in results),
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=24.)
    parser.add_argument("--targets", type=float, nargs="+", default=[.08, -.08, 0.])
    parser.add_argument("--hold-seconds", type=float, default=4.)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.controller)+".metadata.json").read_text())
    config = VerticalConfig(**metadata["contract"]["physical"]["config"])
    # Runtime continues through the training's four-second time limit.
    env = VerticalControlEnv(config, episode_seconds=args.seconds)
    model, _ = load_policy(args.controller, env)
    obs, _ = env.reset(options={"initial_position_m": 0., "target_m": args.targets[0]})
    viewer = None
    if args.view:
        from .vertical_viewer import VerticalViewer
        viewer = VerticalViewer(env.physical)
    started, errors, failed = time.monotonic(), [], False
    try:
        for _ in range(round(args.seconds/config.dt)):
            if viewer is not None and not viewer.is_running():
                break
            env.set_target(args.targets[int(env.physical.time/args.hold_seconds) % len(args.targets)])
            action, _ = model.predict(env.observation(), deterministic=True)
            obs, _, failed, truncated, info = env.step(action)
            errors.append(info["error_m"])
            if viewer is not None:
                import mujoco
                from .vertical_visualization import text_overlays
                viewer.sync(env.physical)
                viewer.handle.set_texts(text_overlays(env.physical)+[(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOP, f"PPO target {env.target*1000:+.0f} mm", "")])
                time.sleep(max(0., started+env.physical.time-time.monotonic()))
            if failed or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps({"variant": config.variant, "seconds": env.physical.time, "physical_failure": failed,
                      "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}))


if __name__ == "__main__":
    main()
