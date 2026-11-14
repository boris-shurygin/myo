"""Separate using the command from succeeding at a closed-loop trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_hierarchy import load_policy, evaluate, VALIDATION_MOTIONS


class WithoutAcceleration:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, deterministic=True):
        observation = observation.copy()
        observation[-4:-1] = 0.
        return self.model.predict(observation, deterministic=deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env, _ = load_policy(args.checkpoint)
    try:
        normal = evaluate(model, env, VALIDATION_MOTIONS)
        ablated = evaluate(WithoutAcceleration(model), env, VALIDATION_MOTIONS)
        probes = []
        observation, _ = env.reset(seed=0, options={'motion': VALIDATION_MOTIONS[1]})
        for step in range(150):
            if step % 10 == 0:
                actions = []
                for command in (-10., 0., 10.):
                    changed = observation.copy()
                    changed[-2] = command/env.hierarchy.acceleration_scale
                    actions.append(model.predict(changed, deterministic=True)[0].tolist())
                probes.append({'step': step, 'commands_rad_s2': [-10., 0., 10.], 'actions': actions})
            action = model.predict(observation, deterministic=True)[0]
            observation, _, failed, done, info = env.step(action)
            if failed or done:
                break
        differences = [np.asarray(p['actions'][2])-np.asarray(p['actions'][0]) for p in probes]
        result = {'normal': normal, 'zero_command_input': ablated, 'probes': probes,
            'final_probe_pose': {'q_deg': float(np.rad2deg(info['q'])),
                'contact': env.unwrapped.tissue_contact_metrics(),
                'activation': info['activation'], 'delayed_excitation': info['delayed_excitation'],
                'capacity': info['capacity'], 'tendon_force_N': info['tendon_force_N']},
            'mean_action_change_for_minus10_to_plus10': np.mean(differences, axis=0).tolist(),
            'note': 'Input intervention at identical physical states, not a physical counterfactual rollout; '
                    'zeroing commands changes the distribution and cannot by itself prove useful representation.'}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'normal_q_rmse': normal['mean_position_rmse_deg'],
            'zero_q_rmse': ablated['mean_position_rmse_deg'],
            'action_change': result['mean_action_change_for_minus10_to_plus10']}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
