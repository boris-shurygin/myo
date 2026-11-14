"""Hindsight inverse dynamics: causal physical input + achieved acceleration goal -> excitation.

Future acceleration is used as a relabelled task, never as a physical sensor.
Only the currently issued action can first affect the 40--60ms interval.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from myo.arm_hierarchy import ArmHierarchyEnv, Motion, TRAIN_MOTIONS, VALIDATION_MOTIONS, TEST_MOTIONS, evaluate, save_policy
from myo.arm_learning import ArmPPO
from myo.arm_exploration import ExplorationPolicy, ExplorationSpec


def collect(env, motions):
    inputs, targets, episodes = [], [], []
    for episode, motion in enumerate(motions):
        observation, _ = env.reset(seed=0, options={'motion': motion})
        states, actions, accelerations = [], [], []
        while True:
            action = env.heuristic_action()
            states.append(observation.copy())
            actions.append(action)
            observation, _, failed, done, info = env.step(action)
            accelerations.append(info['hierarchy']['actual_acceleration'])
            if failed:
                raise RuntimeError('Do not build inverse labels across a physical failure')
            if done:
                break
        for i in range(len(states)-2):
            achieved = accelerations[i+2]
            if abs(achieved) > env.hierarchy.acceleration_limit:
                continue
            state = states[i].copy()
            state[-2] = achieved/env.hierarchy.acceleration_scale
            inputs.append(state)
            targets.append(actions[i])
            episodes.append(episode)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets)), episodes


def collect_policy_states(env, model, motions, *, scale=.8, stride=5, counterfactual=False):
    from myo.delayed_acceleration import NativeResponse, capture
    inputs, targets = [], []
    kernel = None
    try:
        for motion in motions:
            observation, _ = env.reset(seed=0, options={'motion': motion})
            if kernel is None:
                kernel = NativeResponse(env.unwrapped.config, horizon_steps=3)
            step = 0
            while True:
                if step % stride == 0:
                    teacher = env.heuristic_action()
                    snapshot = capture(env.unwrapped)
                    for factor in ((.7, 1., 1.3) if counterfactual else (1.,)):
                        candidate = np.clip(teacher*factor, 0., 1.)
                        response = kernel.response(snapshot, np.tile(teacher, (3,1)), candidate)
                        achieved = float(response[-1])
                        if abs(achieved) <= env.hierarchy.acceleration_limit:
                            state = observation.copy()
                            state[-2] = achieved/env.hierarchy.acceleration_scale
                            inputs.append(state)
                            targets.append(candidate)
                action = model.predict(observation, deterministic=True)[0]*scale
                observation, _, failed, done, _ = env.step(action)
                step += 1
                if failed or done:
                    break
    finally:
        if kernel is not None:
            kernel.close()
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def native_rank(report):
    return (report['failures'], -report['passed'], report['mean_position_rmse_deg'],
            report['mean_acceleration_rmse_rad_s2'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--updates', default=3000, type=int)
    parser.add_argument('--augment-from', type=Path)
    parser.add_argument('--native-selection', action='store_true')
    parser.add_argument('--cold-grid', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    train_motions = TRAIN_MOTIONS + (tuple(Motion(f'cold_hold{q}',float(q),float(q),kind='hold')
        for q in (123,126,129,132)) if args.cold_grid else ())
    env = ArmHierarchyEnv(motions=train_motions)
    try:
        x, y, episodes = collect(env, train_motions)
        validation_motions = VALIDATION_MOTIONS + ((Motion('val_lower133',133.,122.,rise_s=1.5,kind='reach'),
            Motion('val_hold131',131.,131.,kind='hold')) if args.native_selection else ())
        vx, vy, validation_episodes = collect(env, validation_motions)
        model = ArmPPO(ExplorationPolicy, env, seed=0, device='cpu', learning_rate=1e-5,
            n_steps=256, batch_size=128, n_epochs=10, gamma=.99, gae_lambda=.95,
            ent_coef=.001, target_kl=.03,
            policy_kwargs={'predictor_indices': [0,1], 'physical_indices': [0,1], 'feature_mode': 'none',
                'exploration': asdict(ExplorationSpec(mode='off')), 'activation_fn': nn.SiLU,
                'optimizer_class': torch.optim.RMSprop, 'optimizer_kwargs': {'eps':1e-5}})
        original_count = len(x)
        if args.augment_from:
            from myo.arm_hierarchy import load_policy
            model, replay, _ = load_policy(args.augment_from)
            replay.close()
            ax, ay = collect_policy_states(env, model, train_motions, counterfactual=args.native_selection,
                                           stride=10 if args.native_selection else 5)
            x, y = torch.cat((x,ax)), torch.cat((y,ay))
        parameters = [p for name, p in model.policy.named_parameters() if p.requires_grad and
                      (name.startswith('mlp_extractor.actor_private.') or
                       name.startswith('mlp_extractor.policy_net.') or name.startswith('action_net.'))]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        rng = torch.Generator().manual_seed(0)
        training = {'method': 'supervised inverse dynamics, mean excitation MSE', 'optimizer': 'Adam',
            'lr': 1e-3, 'samples': len(x), 'validation_samples': len(vx),
            'train_episodes': len(set(episodes)), 'validation_episodes': len(set(validation_episodes)),
            'goal_relabelling': 'only due40 command replaced by observed acceleration in interval step+2',
            'physical_future_in_sensors': False, 'PPO_updates': 0}
        training.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        augmented_samples=len(x)-original_count,
                        augmentation='own policy times0.8, native three-step branches, equal minibatch halves'
                            if args.augment_from else None,
                        branch_action_scales=([.7,1.,1.3] if args.native_selection else [1.]) if args.augment_from else [],
                        validation_motions=[asdict(m) for m in validation_motions],
                        selection='native failures,passes,position,acceleration' if args.native_selection else 'validation excitation MSE',
                        parent_sha256=hashlib.sha256(args.augment_from.read_bytes()).hexdigest() if args.augment_from else None)
        history, best, best_update = [], None, 0
        for update in range(args.updates+1):
            if update:
                index = (torch.cat((torch.randint(original_count, (128,), generator=rng),
                                     original_count+torch.randint(len(x)-original_count, (128,), generator=rng)))
                         if args.augment_from else torch.randint(len(x), (256,), generator=rng))
                action = model.policy.get_distribution(x[index]).get_actions(deterministic=True)
                loss = (action-y[index]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
            if update % 250 == 0 or update == args.updates:
                with torch.no_grad():
                    train_mse = float((model.policy.get_distribution(x).get_actions(deterministic=True)-y).square().mean())
                    val_mse = float((model.policy.get_distribution(vx).get_actions(deterministic=True)-vy).square().mean())
                row = {'update': update, 'train_mse': train_mse, 'validation_mse': val_mse}
                if args.native_selection and (update % 500 == 0 or update == args.updates):
                    report = evaluate(model, env, validation_motions)
                    row['native_validation'] = report
                    rank = native_rank(report)
                else:
                    rank = None if args.native_selection else (val_mse,)
                history.append(row)
                if rank is not None and (best is None or rank < best):
                    best, best_update = rank, update
                    # Save full supervised optimizer alongside the PPO-compatible policy.
                    model.inverse_optimizer_state = optimizer.state_dict()
                    save_policy(model, env, args.out/'best.zip', training | {'best_update':update})
                print(json.dumps({k:v for k,v in row.items() if k != 'native_validation'} |
                    ({'native_passed':report['passed'], 'native_q_rmse':report['mean_position_rmse_deg']}
                     if 'native_validation' in row else {})), flush=True)
                (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
                if update-best_update >= 1000:
                    break
        model.inverse_optimizer_state = optimizer.state_dict()
        save_policy(model, env, args.out/'last.zip', training | {'last_update':update})
        from myo.arm_hierarchy import load_policy
        selected, replay, _ = load_policy(args.out/'best.zip')
        try:
            result = {name:evaluate(selected, replay, motions, traces=True)
                      for name,motions in [('train',train_motions),('validation',VALIDATION_MOTIONS),('test',TEST_MOTIONS)]}
            (args.out/'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps({name:{k:v for k,v in r.items() if k != 'cases'} for name,r in result.items()}), flush=True)
        finally:
            replay.close()
    finally:
        env.close()


if __name__ == '__main__':
    main()
