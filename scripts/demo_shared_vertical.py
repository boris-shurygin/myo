"""Run a strictly loaded shared-encoder experiment policy on its native stand."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from myo.shared_vertical import load_shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, default=Path('runs/shared_vertical/shared_prediction/best.zip'))
    parser.add_argument('--seconds', type=float, default=24.)
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.controller)+'.metadata.json').read_text())
    if 'pretrained_design' in metadata['contract']:
        from myo.pretrained_vertical import load_pretrained
        model, env, _ = load_pretrained(args.controller, episode_seconds=args.seconds)
    elif 'detached_design' in metadata['contract']:
        from myo.detached_vertical import load_detached
        model, env, _ = load_detached(args.controller, episode_seconds=args.seconds)
    else:
        model, env, _ = load_shared(args.controller, episode_seconds=args.seconds)
    env.reset(options={'initial_position_m': 0., 'target_m': .06})
    viewer = None
    if args.view:
        from myo.vertical_viewer import VerticalViewer
        viewer = VerticalViewer(env.physical)
    started, errors, failed = time.monotonic(), [], False
    try:
        for _ in range(round(args.seconds/env.physical.config.dt)):
            if viewer is not None and not viewer.is_running():
                break
            env.set_target((.06, -.06, 0.)[int(env.physical.time/4.) % 3])
            action, _ = model.predict(env.observation(), deterministic=True)
            _, _, failed, truncated, info = env.step(action)
            errors.append(info['error_m'])
            if viewer is not None:
                import mujoco
                from myo.vertical_visualization import text_overlays
                viewer.sync(env.physical)
                viewer.handle.set_texts(text_overlays(env.physical)+[(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOP, f"PPO target {env.target*1000:+.0f} mm", '')])
                time.sleep(max(0., started+env.physical.time-time.monotonic()))
            if failed or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    print(json.dumps({'seconds': env.physical.time, 'physical_failure': failed,
                      'position_rmse_m': float(np.sqrt(np.mean(np.square(errors)))) if errors else None}))


if __name__ == '__main__':
    main()
