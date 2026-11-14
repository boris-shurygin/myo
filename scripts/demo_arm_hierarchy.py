"""Play a reference trajectory through the learned acceleration and muscle layers."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.arm_hierarchy import (ArmHierarchyEnv, Motion, TRAIN_MOTIONS, VALIDATION_MOTIONS,
                               TEST_MOTIONS, load_policy)


class TrajectoryDisplay:
    """Renderer-only reference overlay; never change native commands or physics."""
    def __init__(self, env):
        self.owner = env
        self.config = replace(env.unwrapped.config, task_mode='reference')

    def __getattr__(self, name):
        return getattr(self.owner.unwrapped, name)

    def reference_at(self, time):
        return self.owner.get_wrapper_attr('motion').sample(time)[:2]

    def telemetry(self):
        state = self.owner.unwrapped.telemetry()
        q, v = self.reference_at(self.owner.unwrapped.time)
        return state | {'target_q': float(q), 'requested_target_q': float(q), 'target_qvel': float(v),
                        'tracking_error': [state['q']-q, state['qvel']-v],
                        'command_source': 'hierarchical trajectory'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=Path, default=Path(__file__).resolve().parents[1]/
                        'runs/arm_hierarchy/neural_upper/best.zip')
    parser.add_argument('--motion', default='oscillation')
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    neural_upper = False
    if args.heuristic:
        model,env,metadata = 'heuristic',ArmHierarchyEnv(),{}
    else:
        header = json.loads(Path(str(args.controller)+'.metadata.json').read_text())
        neural_upper = header['contract']['schema'] == 'myo.arm_neural_upper.v1'
        if neural_upper:
            from myo.arm_neural_upper import load_upper
            model,env,metadata = load_upper(args.controller)
        else:
            model,env,metadata = load_policy(args.controller)
    task = env.env if neural_upper else env
    motions = {m.name:m for m in (*task.motions, *TRAIN_MOTIONS, *VALIDATION_MOTIONS, *TEST_MOTIONS)}
    for value in metadata.get('training', {}).get('validation_motions', []):
        motion = Motion(**value)
        motions[motion.name] = motion
    if args.motion not in motions:
        env.close()
        parser.error('Available motions: '+', '.join(motions))
    observation, _ = env.reset(seed=0, options={'motion': motions[args.motion]})
    viewer, finished, samples = None, False, []
    try:
        if args.view:
            from myo.demo import DemoViewer
            label = ('Neural upper + frozen lower' if neural_upper else
                     'Heuristic upper + heuristic muscles' if args.heuristic else 'Heuristic upper + inverse lower')
            viewer = DemoViewer(TrajectoryDisplay(env), controller_label=label+' | '+args.motion)
            print('Space: pause; R: restart; F: camera; Esc: close. End holds until R. '
                  '1–4: colors; M: muscle; C: contact envelopes.', flush=True)
        while viewer is None or viewer.is_running():
            started = time.monotonic()
            if viewer is not None:
                if viewer.poll():
                    observation, _ = env.reset(seed=0, options={'motion': motions[args.motion]})
                    finished, samples, viewer.paused, viewer.stop_reason = False, [], False, None
                if not viewer.is_running():
                    break
                if finished or viewer.paused:
                    viewer.draw()
                    time.sleep(.02)
                    continue
            action = env.heuristic_action() if args.heuristic else model.predict(observation, deterministic=True)[0]
            observation, _, failed, done, info = env.step(action)
            samples.append((env.unwrapped.time, info['hierarchy']['position_error']))
            if viewer is not None:
                viewer.draw()
                time.sleep(max(0., .02-(time.monotonic()-started)))
            if failed or done:
                selected = [e for t,e in samples if t >= .5]
                print(json.dumps({'motion':args.motion,'steps':len(samples),'failure':bool(failed),
                    'position_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(np.square(selected)))))}), flush=True)
                if viewer is None:
                    break
                finished, viewer.paused = True, True
                viewer.stop_reason = 'physical failure' if failed else 'trajectory complete; R restarts'
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == '__main__':
    main()
