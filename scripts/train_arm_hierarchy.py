"""Run heuristic feasibility or PPO for the anticipated-acceleration arm layer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
import numpy as np
import gymnasium
import mujoco
import stable_baselines3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

from myo.arm_exploration import ExplorationPolicy, ExplorationSpec
from myo.arm_hierarchy import (ArmHierarchyEnv, HierarchySpec, TRAIN_MOTIONS,
    VALIDATION_MOTIONS, TEST_MOTIONS, evaluate, save_policy, load_policy)
from myo.arm_learning import ArmPPO


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--broaden', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initial-actor', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    spec = HierarchySpec(broaden_on_stall=args.broaden)
    env = ArmHierarchyEnv(spec)
    write(args.out/'manifest.json', {'contract': env.contract(), 'arguments': {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
            'mujoco': mujoco.__version__, 'gymnasium': gymnasium.__version__,
            'stable_baselines3': stable_baselines3.__version__}})
    if args.heuristic:
        result = evaluate('heuristic', env, TRAIN_MOTIONS+VALIDATION_MOTIONS+TEST_MOTIONS, traces=True)
        write(args.out/'heuristic.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)
        env.close()
        return
    factories = [lambda i=i: ArmHierarchyEnv(spec, worker=i, workers=args.workers) for i in range(args.workers)]
    vec = SubprocVecEnv(factories, start_method='forkserver') if args.workers > 1 else DummyVecEnv(factories)
    if args.resume:
        model, check_env, _ = load_policy(args.resume)
        if env.contract() != check_env.contract():
            raise ValueError('Resume protocol differs')
        check_env.close()
        model.set_env(vec)
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
    else:
        model = ArmPPO(ExplorationPolicy, vec, seed=0, device='cpu', learning_rate=args.lr,
            n_steps=256, batch_size=128, n_epochs=10, gamma=.99, gae_lambda=.95, clip_range=.2,
            ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
            policy_kwargs={'predictor_indices': [0, 1], 'physical_indices': [0, 1], 'feature_mode': 'none',
                'exploration': asdict(ExplorationSpec(mode='stall' if args.broaden else 'off')),
                'activation_fn': nn.SiLU, 'optimizer_class': torch.optim.RMSprop,
                'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})
    model.set_logger(configure(str(args.out), ['csv']))
    history, best = [], None
    start, initial_steps = time.monotonic(), model.num_timesteps
    training = {'algorithm': 'PPO', 'lr': args.lr, 'seed': 0, 'resume': str(args.resume), 'spec': asdict(spec)}
    if args.initial_actor:
        if args.resume:
            raise ValueError('Choose resume or initialization, not both')
        from myo.arm_exploration import load_exploration
        from myo.arm_hierarchy_transfer import transfer_fixed_goal_actor
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        parent, parent_env, _ = load_exploration(args.initial_actor)
        try:
            training['initialization'] = transfer_fixed_goal_actor(model, env, parent, parent_env)
            training['initialization']['path'] = str(args.initial_actor)
            training['initialization']['sha256'] = hashlib.sha256(args.initial_actor.read_bytes()).hexdigest()
            training['initialization']['source_sha256'] = hashlib.sha256(
                (Path(__file__).resolve().parents[1]/'src/myo/arm_hierarchy_transfer.py').read_bytes()).hexdigest()
        finally:
            parent_env.close()
            torch.set_rng_state(rng[0])
            np.random.set_state(rng[1])
            random.setstate(rng[2])
    try:
        budgets = [0]+list(range(16384, args.steps, 16384))+[args.steps]
        for budget in dict.fromkeys(budgets):
            if budget:
                model.learn(min(16384, args.steps-(model.num_timesteps-initial_steps)), reset_num_timesteps=False)
            validation = evaluate(model, env, VALIDATION_MOTIONS)
            row = {'steps': model.num_timesteps, 'elapsed_s': time.monotonic()-start,
                   'validation': validation, 'last_update': model.last_audit}
            rank = (validation['failures'], validation['mean_acceleration_rmse_rad_s2'])
            if best is None or rank < best:
                best = rank
                save_policy(model, env, args.out/'best.zip', training)
            history.append(row)
            write(args.out/'history.json', history)
            print(json.dumps({k: v for k, v in row.items() if k != 'validation'} |
                {'passed': validation['passed'], 'q_rmse_deg': validation['mean_position_rmse_deg'],
                 'a_rmse_rad_s2': validation['mean_acceleration_rmse_rad_s2']}), flush=True)
            if validation['passed'] == validation['total']:
                train = evaluate(model, env, TRAIN_MOTIONS)
                if train['passed'] == train['total']:
                    write(args.out/'train_gate.json', train)
                    break
            # Stop a demonstrably flat continuation, not the initial exploration.
            if len(history) >= 6:
                recent = [r['validation']['mean_acceleration_rmse_rad_s2'] for r in history[-4:]]
                previous = min(r['validation']['mean_acceleration_rmse_rad_s2'] for r in history[:-4])
                if min(recent) >= previous*.99:
                    training['stop'] = 'no >=1% validation acceleration improvement over four checks'
                    break
        save_policy(model, env, args.out/'last.zip', training)
        # Evaluate the selected checkpoint, not merely the last update.
        selected, selected_env, _ = load_policy(args.out/'best.zip')
        try:
            result = {name: evaluate(selected, selected_env, motions, traces=True)
                      for name, motions in [('train', TRAIN_MOTIONS), ('validation', VALIDATION_MOTIONS), ('test', TEST_MOTIONS)]}
            write(args.out/'evaluation.json', result)
            print(json.dumps({'selected': {k: {x: y for x, y in v.items() if x != 'cases'} for k, v in result.items()}}), flush=True)
        finally:
            selected_env.close()
    finally:
        vec.close()
        env.close()


if __name__ == '__main__':
    main()
