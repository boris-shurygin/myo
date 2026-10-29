"""Run the isolated muscle-arm prototype, headless or in MuJoCo's viewer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Sequence
import uuid

import numpy as np

from .runtime import load_controller
from .telemetry import LoopbackPublisher, encode_packet, make_packet


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rollout = commands.add_parser("rollout", help="Run fixed-step muscle physics and emit an evaluation summary.")
    rollout.add_argument("--seconds", type=float, default=10.0)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--controller", default="heuristic", help="heuristic, zero, or a myo.train PPO .zip checkpoint")
    rollout.add_argument("--config", type=Path)
    rollout.add_argument("--predictor", type=Path)
    rollout.add_argument("--output", type=Path, help="Optional JSONL state snapshots; use runs/.")
    rollout.add_argument("--udp-port", type=int, help="Send snapshots to a receiver on 127.0.0.1.")
    rollout.add_argument("--viewer", action="store_true", help="Open MuJoCo's debug viewer (requires a display).")
    rollout.add_argument("--realtime", action="store_true", help="Pace to wall-clock time; does not change physics dt.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    env, action, _ = load_controller(args.controller, args.config, args.predictor, "human" if args.viewer else None)
    base = env.unwrapped
    writer = None
    publisher = None
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = args.output.open("w", encoding="utf-8")
        if args.udp_port is not None:
            publisher = LoopbackPublisher(args.udp_port)
        observation, _ = env.reset(seed=args.seed)
        count = int(math.ceil(args.seconds / base.control_dt))
        errors, policy_times, step_times = [], [], []
        reward_sum = 0.0
        episode, sequence, failures = 0, 0, 0
        run_id = str(uuid.uuid4())
        start = time.perf_counter()

        def emit() -> None:
            nonlocal sequence
            prediction = None
            if hasattr(env, "predictor"):
                prediction = {
                    "horizon_s": env.predictor.horizon_s,
                    "error_angle_rad": float(observation[-2]),
                    "error_angular_velocity_rad_s": float(observation[-1]),
                    "predictor_sha256": env.predictor_sha256,
                }
            packet = make_packet(base, sequence, episode, run_id, prediction)
            if writer:
                writer.write(encode_packet(packet).decode() + "\n")
            if publisher:
                publisher.send(packet)
            sequence += 1

        emit()
        for step in range(count):
            before = time.perf_counter()
            control = action(observation)
            after_policy = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(control)
            after_step = time.perf_counter()
            policy_times.append(1000 * (after_policy - before))
            step_times.append(1000 * (after_step - after_policy))
            errors.append(info["tracking_error"])
            reward_sum += reward
            emit()
            if args.realtime:
                remaining = start + (step + 1) * base.control_dt - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            if terminated or truncated:
                failures += int(terminated)
                if step + 1 < count:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed + episode)
                    emit()
        wall = time.perf_counter() - start
        error_array = np.asarray(errors)
        summary = {
            "controller": args.controller,
            "simulated_seconds": count * base.control_dt,
            "wall_seconds": wall,
            "episodes": episode + 1,
            "physical_failures": failures,
            "angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error_array[:, 0] ** 2)))),
            "angular_velocity_rmse_rad_s": float(np.sqrt(np.mean(error_array[:, 1] ** 2))),
            "mean_reward": reward_sum / count,
            "policy_mean_ms": float(np.mean(policy_times)),
            "policy_p95_ms": float(np.percentile(policy_times, 95)),
            "env_step_mean_ms": float(np.mean(step_times)),
            "env_step_p95_ms": float(np.percentile(step_times, 95)),
            "timing_scope": "env.step includes predictor wrapper and viewer when enabled; excludes packet writing/pacing",
            "limitation": "Toy model. Headless timing is not an Unreal or full-body FPS benchmark.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        if writer:
            writer.close()
        if publisher:
            publisher.close()
        env.close()


if __name__ == "__main__":
    main()

