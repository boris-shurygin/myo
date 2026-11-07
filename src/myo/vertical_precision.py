"""Dense positioning curriculum and a near-goal-sensitive logarithmic reward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .vertical_control import (REWARD, TRAIN_CASES, VALIDATION_CASES, TEST_CASES,
                               VerticalControlEnv, load_policy)
from .vertical_muscles import VerticalConfig

# Keep the historical tasks; interleave holds and reaches rather than grouping holds.
DENSE_CASES = tuple(dict.fromkeys(
    case for target in (-.08, -.06, -.04, -.02, 0., .02, .04, .06, .08)
    for case in ((target, target), *((start, target) for start in (-.06, 0., .06) if start != target))
))
PRECISION_VALIDATION = VALIDATION_CASES + ((-.05, -.05), (.05, .05), (-.01, -.01), (.01, .01))
PRECISION_TEST = TEST_CASES + ((-.07, -.07), (.07, .07), (-.035, -.035), (.035, .035),
                              (-.025, .055), (.025, -.055))
PRECISION_REWARD = {**REWARD, "position_function": "anchored_log1p_abs",
                    "position_epsilon_m": .005,
                    "position_anchor_m": REWARD["position_scale_m"],
                    "position_anchor_drop": float(1-np.exp(-1.))}
PRECISION_REWARD.pop("position_scale_m")


def position_reward(error):
    r = PRECISION_REWARD
    drop = r["position_anchor_drop"]*np.log1p(np.abs(error)/r["position_epsilon_m"])
    drop /= np.log1p(r["position_anchor_m"]/r["position_epsilon_m"])
    return r["position_weight"]*(1-drop)


class VerticalPrecisionEnv(VerticalControlEnv):
    def reset(self, *, seed=None, options=None):
        if options is not None:
            return super().reset(seed=seed, options=options)
        index = 0 if seed is not None else self.case_index
        position, target = DENSE_CASES[index % len(DENSE_CASES)]
        result = super().reset(seed=seed, options={"initial_position_m": position, "target_m": target})
        self.case_index = index+1
        return result

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        info["reward_components"]["position"] = float(position_reward(info["error_m"]))
        info["raw_reward"] = float(sum(info["reward_components"].values()))
        return obs, REWARD["training_multiplier"]*info["raw_reward"], terminated, truncated, info

    def contract(self):
        return {**super().contract(), "reward": PRECISION_REWARD,
                "precision_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def evaluate_precision(model, config, cases, seconds=4., switches=False):
    """Same physical pass criteria as the baseline, with the actual new return."""
    results = []
    for position, target in cases:
        env = VerticalPrecisionEnv(config, episode_seconds=seconds)
        obs, _ = env.reset(options={"initial_position_m": position, "target_m": target})
        samples, reward, failed = [], 0., False
        for _ in range(round(seconds/config.dt)):
            if switches:
                env.set_target((target, -target, 0.)[min(2, int(env.physical.time/2.))])
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
        env.close()
    score = float(np.mean([r["last_second_rmse_m"] if not r["physical_failure"] else .3 for r in results]))
    return {"score_m": score, "passed": sum(r["passed"] for r in results), "total": len(results),
            "failures": sum(r["physical_failure"] for r in results), "ready": all(r["passed"] for r in results),
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=24.)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.controller)+".metadata.json").read_text())
    config = VerticalConfig(**metadata["contract"]["physical"]["config"])
    env = VerticalPrecisionEnv(config, args.seconds)
    model, _ = load_policy(args.controller, env)
    env.reset(options={"initial_position_m": 0., "target_m": .06})
    viewer = None
    if args.view:
        from .vertical_viewer import VerticalViewer
        viewer = VerticalViewer(env.physical)
    started, errors, failed = time.monotonic(), [], False
    try:
        for _ in range(round(args.seconds/config.dt)):
            if viewer is not None and not viewer.is_running():
                break
            env.set_target((.06, -.06, 0.)[int(env.physical.time/4.) % 3])
            action, _ = model.predict(env.observation(), deterministic=True)
            _, _, failed, truncated, info = env.step(action)
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
        env.close()
    print(json.dumps({"seconds": env.physical.time, "physical_failure": failed,
                      "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}))


if __name__ == "__main__":
    main()
