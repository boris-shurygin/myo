"""Control the learned preacher-curl arm with arrow keys, preserving native physics."""
import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from myo.arm_learning import load_arm
from myo.telemetry import make_packet, encode_packet


DEFAULT_CONTROLLER = Path(__file__).resolve().parents[1]/'runs/arm_continuation/best.zip'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', type=Path, default=DEFAULT_CONTROLLER)
    p.add_argument('--seconds', type=float)
    p.add_argument('--mass', type=float, help='Payload kg at reset; requires a workload policy')
    p.add_argument('--tilt', type=float, help='Fixed mount degrees at reset; requires a workload policy')
    p.add_argument('--view', action='store_true')
    p.add_argument('--scripted', action='store_true', help='Cycle targets 75/20/105/45 deg every 4 s without resetting state')
    p.add_argument('--output', type=Path, help='Optional native v2 telemetry JSONL in runs/')
    args = p.parse_args(argv); torch.set_num_threads(1)
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        p.error('--seconds must be finite and positive')
    duration = args.seconds if args.seconds is not None else (None if args.view else 16.)
    header = json.loads(Path(str(args.controller)+'.metadata.json').read_text())
    workload = header['contract'].get('schema') == 'myo.arm_workload.v1'
    if workload:
        from myo.arm_workload import load_workload
        model, env, metadata = load_workload(args.controller)
    elif 'feature_comparison' in header['training']:
        from myo.arm_feature_comparison import load_comparison
        model, env, metadata = load_comparison(args.controller)
    elif 'depth' in header['training']:
        from myo.arm_depth import load_deep_arm
        model, env, metadata = load_deep_arm(args.controller)
    elif 'conditioning' in header['training']:
        from myo.arm_conditioning import load_conditioned_arm
        model, env, metadata = load_conditioned_arm(args.controller)
    else:
        model, env, metadata = load_arm(args.controller)
    base = env.unwrapped
    reset_options = {'case': (20, 75)}
    if args.mass is not None or args.tilt is not None:
        if not workload:
            env.close()
            p.error('--mass/--tilt require a policy trained with the workload contract')
        if (args.mass is not None and (not np.isfinite(args.mass) or args.mass <= 0)
                or args.tilt is not None and not np.isfinite(args.tilt)):
            env.close()
            p.error('--mass must be positive and --tilt finite')
    if workload:
        reset_options.update(payload_mass=1. if args.mass is None else args.mass,
                             tilt_deg=30. if args.tilt is None else args.tilt, capacity=1.)
    env.reset(seed=0, options=reset_options)
    viewer = None
    writer = None
    run_id, episode, sequence = str(uuid.uuid4()), 0, 0
    errors, failed = [], False
    started = time.perf_counter()
    count = 0
    try:
        if args.view:
            from myo.demo import DemoViewer
            label = ('Learned PPO | '+metadata['training']['stage']+' | variable load/mount' if workload
                     else 'Learned PPO | positioning | tested goals 15-135 deg')
            viewer = DemoViewer(base, fixed_trigger=1., controller_label=label)
            print('Up/Down: target +/-5 deg; Space: pause; R: reset; F: fit camera; Esc: close.\n'
                  '1: tissues; 2: activation; 3: force; 4: strain; M: muscle; C: envelopes.\n'
                  f'Trigger fixed at 100%; payload {base.payload_mass:g} kg. '+str(args.controller), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = args.output.open('wb')
        while duration is None or count*base.control_dt < duration-1e-10:
            if viewer is not None:
                if not viewer.is_running(): break
                if viewer.poll():
                    env.reset(seed=0, options=reset_options)
                    viewer.stop_reason, viewer.paused = None, False
                    episode += 1
                if not viewer.is_running(): break
                if viewer.paused or viewer.stop_reason:
                    viewer.draw(); time.sleep(.02)
                    started = time.perf_counter()-count*base.control_dt
                    continue
            if args.scripted or viewer is None:
                target = (75, 20, 105, 45)[int(base.time/4.) % 4]
                base.set_command(float(np.deg2rad(target)), 1.)
            action = model.predict(env.observation(), deterministic=True)[0]
            _, _, failed, truncated, info = env.step(action)
            count += 1
            errors.append(float(np.rad2deg(info['q']-info['requested_target_q'])))
            if writer is not None:
                writer.write(encode_packet(make_packet(base, sequence, episode, run_id))+b'\n')
                sequence += 1
            if viewer is not None:
                viewer.draw()
                time.sleep(max(0., started+count*base.control_dt-time.perf_counter()))
            if failed:
                if viewer is None: break
                viewer.stop_reason = info['termination_reason'] or 'physical_failure'
                viewer.paused = True
            elif truncated:
                # Training horizon expires; time, fatigue, history and queues continue.
                base._episode_done = False
    finally:
        if writer is not None: writer.close()
        if viewer is not None: viewer.close()
        env.close()
    print(json.dumps({'seconds': count*base.control_dt, 'failure': bool(failed),
        'full_rmse_deg': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        'controller': str(args.controller), 'trial': metadata['training'].get('trial', model.policy.feature_mode),
        'stage': metadata['training'].get('stage'), 'payload_mass_kg': base.payload_mass,
        'tilt_deg': float(np.rad2deg(base.config.arm_mount_tilt)),
        'scope': 'phase-one positioning; low trigger/release/protection not trained'}))


if __name__ == '__main__': main()
