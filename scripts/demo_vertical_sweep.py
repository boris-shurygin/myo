"""View a strictly checked PPO sweep branch with continuous native physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.vertical_sweep_env import load_sweep_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=24.)
    parser.add_argument("--targets", type=float, nargs="+", default=[.08, -.08, 0.])
    parser.add_argument("--hold-seconds", type=float, default=4.)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env, metadata = load_sweep_policy(args.controller, episode_seconds=args.seconds)
    env.reset(options={"initial_position_m": 0., "target_m": args.targets[0]})
    viewer = None
    if args.view:
        import mujoco
        from myo.vertical_viewer import VerticalViewer
        from myo.vertical_visualization import text_overlays
        viewer = VerticalViewer(env.physical)
    started, errors, failed = time.monotonic(), [], False
    try:
        for _ in range(round(args.seconds/env.physical.config.dt)):
            if viewer is not None and not viewer.is_running():
                break
            env.set_target(args.targets[int(env.physical.time/args.hold_seconds) % len(args.targets)])
            action, _ = model.predict(env.observation(), deterministic=True)
            _, _, failed, truncated, info = env.step(action)
            errors.append(info["error_m"])
            if viewer is not None:
                viewer.sync(env.physical)
                viewer.handle.set_texts(text_overlays(env.physical)+[(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOP, f"PPO target {env.target*1000:+.0f} mm", "")])
                time.sleep(max(0., started+env.physical.time-time.monotonic()))
            if failed or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps({"variant": env.physical.config.variant, "seconds": env.physical.time,
                      "changes": metadata["training"]["changes"], "physical_failure": failed,
                      "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}))


if __name__ == "__main__":
    main()
