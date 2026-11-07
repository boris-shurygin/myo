"""Play one complete issued-acceleration scenario with its native motor delay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from myo.vertical_scheduled import TEST_CASES, load_scheduled_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--case", type=int, choices=range(5), default=0)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env, _ = load_scheduled_policy(args.controller)
    obs, _ = env.reset(options={"case": TEST_CASES[args.case]})
    viewer = None
    if args.view:
        import mujoco
        from myo.vertical_viewer import VerticalViewer
        from myo.vertical_visualization import text_overlays
        viewer = VerticalViewer(env.physical)
    started, failed, errors = time.monotonic(), False, []
    try:
        for _ in range(len(env.commands)+env.lead_steps):
            if viewer is not None and not viewer.is_running():
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, _, failed, truncated, info = env.step(action)
            if info["scored"]:
                errors.append(info["acceleration_error_m_s2"]**2)
            if viewer is not None:
                viewer.sync(env.physical)
                viewer.handle.set_texts(text_overlays(env.physical)+[(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOP,
                    f"Due {info['desired_acceleration_m_s2']:+.2f} | actual {info['actual_acceleration_m_s2']:+.2f} m/s2",
                    f"Issued for +40ms: {info['issued_acceleration_m_s2']:+.2f}")])
                time.sleep(max(0., started+env.physical.time-time.monotonic()))
            if failed or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps({"variant": env.physical.config.variant, "case": asdict(env.case), "seconds": env.physical.time,
                      "physical_failure": bool(failed), "acceleration_rmse_m_s2": (sum(errors)/len(errors))**.5 if errors else None}))


if __name__ == "__main__":
    main()
