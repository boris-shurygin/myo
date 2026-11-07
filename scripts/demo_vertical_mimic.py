"""One complete reference clip with the pinned-source MuscleMimic policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0, choices=range(5))
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    from myo.train_vertical_mimic import MimicPolicy, load_checkpoint
    from myo.vertical_mimic import TEST_MOTIONS, VerticalMimicEnv
    state, network, native_env, _ = load_checkpoint(args.controller)
    env = VerticalMimicEnv(native_env.physical.config, TEST_MOTIONS)
    obs, _ = env.reset(options={"motion_index": args.case, "frame": 0})
    policy = MimicPolicy(network, state)
    # Compile before opening the viewer or starting the real-time clock.
    initial_stats = policy.run_stats
    policy(obs)
    policy.run_stats = initial_stats
    viewer = None
    if args.view:
        from myo.vertical_viewer import VerticalViewer
        viewer = VerticalViewer(env.physical)
    start = time.monotonic()
    errors = []
    info = None
    try:
        while viewer is None or viewer.is_running():
            obs, _, terminated, truncated, info = env.step(policy(obs))
            errors.append(info["error_q_m"])
            if viewer is not None:
                import mujoco
                from myo.vertical_visualization import text_overlays
                viewer.sync(env.physical)
                viewer.handle.set_texts(text_overlays(env.physical)+[(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOP, f"MuscleMimic reference {info['reference_q_m']*1000:+.0f} mm", "")])
                time.sleep(max(0., start+env.physical.time-time.monotonic()))
            if terminated or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
    import numpy as np
    print(json.dumps({"variant": env.physical.config.variant, "motion": env.motion.name,
        "seconds": env.physical.time, "physical_failure": info["physical_failure"] if info else None,
        "q_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}))


if __name__ == "__main__":
    main()
