"""Run the muscle-arm prototype headless or as a compact interactive demo."""

from __future__ import annotations

import argparse
from collections import deque
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
    for name, help_text in (
        ("rollout", "Run fixed-step muscle physics and emit an evaluation summary."),
        ("demo", "Show bones, joint, and muscle activation until the window is closed."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--seconds", type=float, default=None if name == "demo" else 10.0,
                             help="Simulated duration; demo runs until closed by default.")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--controller", default="heuristic", help="heuristic, zero, or a myo.train PPO .zip checkpoint")
        command.add_argument("--config", type=Path)
        command.add_argument("--predictor", type=Path)
        command.add_argument("--output", type=Path, help="Optional JSONL state snapshots; use runs/.")
        command.add_argument("--udp-port", type=int, help="Send snapshots to a receiver on 127.0.0.1.")
        command.add_argument("--viewer", action="store_true", default=name == "demo",
                             help="Open the compact arm viewer at real-time speed (requires a display).")
        command.add_argument("--realtime", action="store_true", default=name == "demo",
                             help="Pace to wall-clock time; automatically enabled with --viewer.")
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    # Presentation stays outside env.py and its dynamics/checkpoint fingerprint.
    env, action, _ = load_controller(args.controller, args.config, args.predictor)
    base = env.unwrapped
    writer = None
    publisher = None
    viewer = None
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = args.output.open("w", encoding="utf-8")
        if args.udp_port is not None:
            publisher = LoopbackPublisher(args.udp_port)
        observation, _ = env.reset(seed=args.seed)
        if args.viewer:
            from .demo import DemoViewer

            viewer = DemoViewer(base)
        count = None if args.seconds is None else int(math.ceil(args.seconds / base.control_dt))
        # An interactive demo may stay open for hours. Keep bounded timing
        # samples while tracking overall angle/speed error with running sums.
        policy_times = deque(maxlen=5000) if args.command == "demo" else []
        step_times = deque(maxlen=5000) if args.command == "demo" else []
        viewer_times = deque(maxlen=5000) if args.command == "demo" else []
        error_square_sum = np.zeros(2, dtype=np.float64)
        reward_sum = 0.0
        episode, sequence, failures = 0, 0, 0
        run_id = str(uuid.uuid4())
        start = time.perf_counter()
        next_frame = start
        completed_steps = 0

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
        while count is None or completed_steps < count:
            if viewer:
                restart = viewer.poll()
                if not viewer.is_running():
                    break
                if restart:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed)
                    emit()
                if viewer.paused:
                    viewer.draw()
                    time.sleep(base.control_dt)
                    next_frame = time.perf_counter()
                    continue
            before = time.perf_counter()
            control = action(observation)
            after_policy = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(control)
            after_step = time.perf_counter()
            policy_times.append(1000 * (after_policy - before))
            step_times.append(1000 * (after_step - after_policy))
            error_square_sum += np.square(info["tracking_error"])
            reward_sum += reward
            completed_steps += 1
            emit()
            if viewer:
                before_viewer = time.perf_counter()
                viewer.draw()
                viewer_times.append(1000 * (time.perf_counter() - before_viewer))
            if args.realtime or viewer:
                next_frame += base.control_dt
                remaining = next_frame - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_frame = time.perf_counter()
            if terminated or truncated:
                failures += int(terminated)
                if count is None or completed_steps < count:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed + episode)
                    emit()
        wall = time.perf_counter() - start
        error_rmse = np.sqrt(error_square_sum / completed_steps) if completed_steps else None
        summary = {
            "controller": args.controller,
            "simulated_seconds": completed_steps * base.control_dt,
            "wall_seconds": wall,
            "episodes": episode + 1,
            "physical_failures": failures,
            "angle_rmse_deg": float(np.rad2deg(error_rmse[0])) if error_rmse is not None else None,
            "angular_velocity_rmse_rad_s": float(error_rmse[1]) if error_rmse is not None else None,
            "mean_reward": reward_sum / completed_steps if completed_steps else None,
            "policy_mean_ms": float(np.mean(policy_times)) if policy_times else None,
            "policy_p95_ms": float(np.percentile(policy_times, 95)) if policy_times else None,
            "env_step_mean_ms": float(np.mean(step_times)) if step_times else None,
            "env_step_p95_ms": float(np.percentile(step_times, 95)) if step_times else None,
            "viewer_mean_ms": float(np.mean(viewer_times)) if viewer_times else None,
            "timing_sample_steps": len(step_times),
            "timing_scope": "env.step includes predictor wrapper; viewer time is separate; excludes packet writing/pacing; demo retains last 5000 timing samples",
            "limitation": "Toy model. Headless timing is not an Unreal or full-body FPS benchmark.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        if viewer:
            viewer.close()
        if writer:
            writer.close()
        if publisher:
            publisher.close()
        env.close()


if __name__ == "__main__":
    main()

