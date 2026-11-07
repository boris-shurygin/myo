"""Compare complete PPO update cost on one identical native rollout, on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=Path("runs/vertical_mimic/variant_4/initial.msgpack"))
    parser.add_argument("--output", type=Path, default=Path("runs/vertical_mimic_adam/update_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from myo.train_vertical_mimic import (RewardNormalizer, load_checkpoint, make_learner, write_json)
    from myo.vertical_mimic import VerticalMimicEnv

    source, _, example, metadata = load_checkpoint(args.initial)
    training = metadata["training"]
    envs = [VerticalMimicEnv(example.physical.config) for _ in range(training["num_envs"])]
    obs = np.stack([env.reset(seed=training["seed"]+i)[0] for i, env in enumerate(envs)])
    learners = {name: make_learner(obs, training["hidden"], training["seed"], name)
                for name in ("muon", "adam")}
    for state, *_ in learners.values():
        for field in ("params", "run_stats"):
            for a, b in zip(jax.tree.leaves(getattr(state, field)), jax.tree.leaves(getattr(source, field))):
                np.testing.assert_array_equal(a, b)
    state, _, act, last_value, _ = learners["muon"]
    norm = RewardNormalizer(len(envs))
    key = jax.random.PRNGKey(training["seed"]+1)
    batches = []
    for _ in range(training["rollout_steps"]):
        key, action_key = jax.random.split(key)
        state, actions, values, lp = act(state, jnp.asarray(obs), action_key)
        actions_np = np.asarray(actions)
        next_obs, rewards, dones = [], [], []
        for env, action in zip(envs, actions_np):
            next_o, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs.append(env.reset()[0] if done else next_o)
            rewards.append(reward)
            dones.append(done)
        rewards, dones = np.asarray(rewards, np.float32), np.asarray(dones, bool)
        batches.append((obs, actions_np, np.asarray(values), norm.step(rewards, dones), dones, np.asarray(lp)))
        obs = np.stack(next_obs)
    arrays = [jnp.asarray(np.stack(x)) for x in zip(*batches)]
    key, update_key = jax.random.split(key)
    tail = last_value(state, jnp.asarray(obs))
    jax.block_until_ready((arrays, tail))
    prepared = {}
    report = {"description": "Same rollout and initial weights; full PPO loss/backprop/32 optimizer steps; excludes physics and validation",
              "initial_checkpoint_sha256": metadata["checkpoint_sha256"], "cpu_threads": args.cpu_threads,
              "transitions": len(envs)*training["rollout_steps"], "optimizers": {}}
    for name, (initial, _, _, _, update) in learners.items():
        initial = initial.replace(run_stats=state.run_stats)
        tick = time.perf_counter()
        jax.block_until_ready(update(initial, *arrays, tail, update_key))
        report["optimizers"][name] = {"compile_and_first_seconds": time.perf_counter()-tick,
                                      "warmed_seconds": []}
        prepared[name] = (initial, update)
    # Alternate order to reduce bias from temperature/frequency drift.
    for repeat in range(args.repeats):
        order = ("muon", "adam") if repeat % 2 == 0 else ("adam", "muon")
        for name in order:
            initial, update = prepared[name]
            tick = time.perf_counter()
            jax.block_until_ready(update(initial, *arrays, tail, update_key))
            report["optimizers"][name]["warmed_seconds"].append(time.perf_counter()-tick)
    for entry in report["optimizers"].values():
        entry["median_seconds"] = float(np.median(entry["warmed_seconds"]))
    report["muon_over_adam"] = (report["optimizers"]["muon"]["median_seconds"]
                                /report["optimizers"]["adam"]["median_seconds"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
