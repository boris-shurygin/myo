"""CPU bridge to pinned MuscleMimic network, optimizer, PPO loss and GAE."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .vertical_mimic import (TRAIN_MOTIONS, VALIDATION_MOTIONS, TEST_MOTIONS,
    VerticalMimicEnv, compact_mimic, evaluate_mimic, source_manifest)
from .vertical_muscles import VerticalConfig


SOURCE_ROOT = Path(__file__).resolve().parents[2]/"data/musclemimic_methodology"
EXPECTED_MANIFEST = Path(__file__).resolve().parents[2]/"configs/vertical_mimic_sources.json"
# Reviewed compatibility: this trainer used the same network, PPO and physics,
# but only supported Muon and did not record separate collection/update timings.
LEGACY_MUON_TRAINER_SHA256 = "e390982fd4bc76ce53593a05365f172e4f621fad1560c949da9069451b81f38f"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def upstream_modules(root=SOURCE_ROOT):
    actual = source_manifest(root)
    if actual != json.loads(EXPECTED_MANIFEST.read_text()):
        raise ValueError("Pinned MuscleMimic source hash mismatch; run preparation script")
    modules = []
    for name, relative in zip(("networks", "optimizer", "loss", "buffer"), tuple(actual)[:4]):
        spec = importlib.util.spec_from_file_location("myo_upstream_mimic_"+name, root/relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


class RewardNormalizer:
    """Literal NumPy bridge for NormalizeVecReward's done/reset convention."""
    def __init__(self, n):
        self.returns = np.zeros(n, dtype=np.float32)
        self.mean, self.var, self.count = 0., 1., 1e-4

    def step(self, reward, done):
        self.returns = self.returns*.99*(1-done)+reward
        mean, var, count = self.returns.mean(), self.returns.var(), len(reward)
        delta = mean-self.mean
        total = self.count+count
        self.var = (self.var*self.count+var*count+delta**2*self.count*count/total)/total
        self.mean += delta*count/total
        self.count = total
        return reward/np.sqrt(self.var+1e-8)


def make_learner(observations, hidden, seed, optimizer_type="muon"):
    import jax
    import jax.numpy as jnp
    from flax.training.train_state import TrainState
    from flax import struct
    networks, optimizer, loss, buffer = upstream_modules()

    class State(TrainState):
        run_stats: object = struct.field(pytree_node=True)

    network = networks.ActorCritic(action_dim=2, activation="silu", init_std=.2,
        hidden_layer_dims=tuple(hidden), use_layernorm=True, layernorm_eps=1e-5,
        use_residual=False, learnable_std=True)
    variables = network.init(jax.random.PRNGKey(seed), jnp.asarray(observations))
    # Upstream AdamW with zero decay is Adam; preserve its eps=1e-5.
    tx = optimizer.build_optimizer("adamw" if optimizer_type == "adam" else optimizer_type, 4e-4, 0., 1.,
        {"beta": .95, "nesterov": True, "ns_steps": 5, "adaptive": False})
    state = State.create(apply_fn=network.apply, params=variables["params"],
                         run_stats=variables["run_stats"], tx=tx)

    @jax.jit
    def act(state, obs, key):
        (pi, value), updates = network.apply({"params": state.params, "run_stats": state.run_stats},
                                             obs, mutable=["run_stats"])
        action = pi.sample(seed=key)
        return state.replace(run_stats=updates["run_stats"]), action, value, pi.log_prob(action)

    @jax.jit
    def last_value(state, obs):
        (_, value), _ = network.apply({"params": state.params, "run_stats": state.run_stats},
                                       obs, mutable=["run_stats"])
        return value

    @jax.jit
    def update(state, observations, actions, values, rewards, dones, log_probs, tail_value, key):
        # The source runner disables bootstrap at *all* done boundaries,
        # including clip end and horizon, not just physical termination.
        advantages, targets = buffer.compute_gae(rewards, values, dones, dones,
                                                 tail_value, .99, .95, bootstrap_mask=~dones)
        flat = tuple(x.reshape((-1,)+x.shape[2:]) for x in
                     (observations, actions, values, log_probs, advantages, targets))
        order = jax.random.permutation(key, len(flat[0])).reshape(32, -1)

        def minibatch(state, indices):
            obs, action, old_v, old_lp, advantage, target = (x[indices] for x in flat)

            def objective(params):
                # Source _update_network computes candidate running stats for
                # normalization but deliberately does not persist these updates.
                (pi, value), _ = network.apply({"params": params, "run_stats": state.run_stats},
                                                obs, mutable=["run_stats"])
                result = loss.ppo_loss(pi.log_prob(action), old_lp, value, old_v,
                    target, advantage, pi.entropy().mean(), .2, .5, 0., clip_eps_vf=.2)
                return result.total_loss, jnp.array([result.actor_loss, result.value_loss,
                                                     result.entropy, result.kl_mean])

            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), metrics

        state, metrics = jax.lax.scan(minibatch, state, order)
        ev = 1-jnp.var(targets-values)/(jnp.var(targets)+1e-8)
        return state, metrics.mean(axis=0), jnp.array([advantages.mean(), advantages.std(), ev])

    return state, network, act, last_value, update


class MimicPolicy:
    """Evaluation uses private running stats; never writes back to training."""
    def __init__(self, network, state, seed=123):
        import jax
        self.key = jax.random.PRNGKey(seed)
        self.params, self.run_stats = state.params, state.run_stats

        @jax.jit
        def forward(params, stats, obs, key):
            (pi, _), updates = network.apply({"params": params, "run_stats": stats},
                                             obs[None], mutable=["run_stats"])
            # Upstream RunningMeanStd squeezes a one-item observation batch.
            # Restore the explicit batch dimension before selecting its item.
            return pi.mean().reshape(-1, 2)[0], pi.sample(seed=key).reshape(-1, 2)[0], updates["run_stats"]
        self.forward = forward

    def __call__(self, observation, *, stochastic=False):
        import jax
        self.key, key = jax.random.split(self.key)
        mean, sample, self.run_stats = self.forward(self.params, self.run_stats, observation, key)
        return np.asarray(sample if stochastic else mean)


def save_checkpoint(path, state, env, training, steps, reward_normalizer):
    import flax.serialization
    path = Path(path)
    path.write_bytes(flax.serialization.to_bytes(state))
    write_json(str(path)+".metadata.json", {"contract": env.contract(), "training": training,
        "steps": steps, "optimizer_steps": int(state.step), "sources": source_manifest(SOURCE_ROOT),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reward_normalizer": {"returns": reward_normalizer.returns.tolist(), "mean": float(reward_normalizer.mean),
                              "var": float(reward_normalizer.var), "count": reward_normalizer.count}})


def load_checkpoint(path):
    import flax.serialization
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    env = VerticalMimicEnv(VerticalConfig(**metadata["contract"]["physical"]["config"]))
    if metadata["contract"] != env.contract() or metadata["sources"] != source_manifest(SOURCE_ROOT):
        raise ValueError("MuscleMimic checkpoint physics/observations/source mismatch")
    optimizer_type = metadata["training"].get("optimizer_type", "muon")
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    legacy = (metadata["trainer_sha256"] == LEGACY_MUON_TRAINER_SHA256
              and "optimizer_type" not in metadata["training"])
    if metadata["trainer_sha256"] != current_hash and not legacy:
        raise ValueError("MuscleMimic trainer source mismatch")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("MuscleMimic checkpoint hash mismatch")
    obs, _ = env.reset(seed=0)
    state, network, *_ = make_learner(np.stack([obs, obs]), metadata["training"]["hidden"], 0,
                                    optimizer_type=optimizer_type)
    state = flax.serialization.from_bytes(state, path.read_bytes())
    if int(state.step) != metadata["optimizer_steps"]:
        raise ValueError("MuscleMimic optimizer step mismatch")
    return state, network, env, metadata


def validation(state, network, config):
    return evaluate_mimic(MimicPolicy(network, state), config, VALIDATION_MOTIONS, phases=(0., .35))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024]*4)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--optimizer", choices=("muon", "adam"), default="muon")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.num_envs*20 % 32 or args.num_envs < 2:
        parser.error("num_envs*20 must divide into 32 nonempty minibatches")
    # Limit JAX's CPU thread pools before importing it, to keep the laptop usable.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:args.cpu_threads])
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = VerticalConfig(variant=args.variant, frame_skip=10)
    envs = [VerticalMimicEnv(config) for _ in range(args.num_envs)]
    obs = np.stack([e.reset(seed=args.seed+i)[0] for i, e in enumerate(envs)])
    state, network, act, last_value, update = make_learner(obs, args.hidden, args.seed, args.optimizer)
    key = jax.random.PRNGKey(args.seed+1)
    norm = RewardNormalizer(args.num_envs)
    training = {"method": "pinned MuscleMimic conf_bimanual.yaml CPU adaptation", "hidden": args.hidden,
        "seed": args.seed, "num_envs": args.num_envs, "rollout_steps": 20, "num_minibatches": 32,
        "update_epochs": 1, "lr": .0004, "anneal_lr": False, "gamma": .99, "gae_lambda": .95,
        "clip": .2, "clip_vf": .2, "vf_coef": .5, "ent_coef": 0., "max_grad_norm": 1.,
        "optimizer_type": args.optimizer,
        "optimizer": ("upstream optax Muon + Adam fallback; exact effective defaults" if args.optimizer == "muon"
                      else "upstream optax AdamW with zero weight decay (= Adam), eps=1e-5"),
        "weight_decay": 0., "cpu_threads": args.cpu_threads,
        "init_std": .2, "activation": "silu", "layernorm": True, "residual": False,
        "planned_updates": args.updates, "train_motions": [asdict(m) for m in TRAIN_MOTIONS],
        "validation_motions": [asdict(m) for m in VALIDATION_MOTIONS],
        "versions": {k: version(k) for k in ("jax", "jaxlib", "flax", "optax", "distrax", "numpy", "mujoco")}}
    summary = {"status": "running", "training": training, "contract": envs[0].contract(), "history": []}
    write_json(args.output_dir/"summary.json", summary)
    save_checkpoint(args.output_dir/"initial.msgpack", state, envs[0], training, 0, norm)
    best_score, best_step = float("inf"), 0
    if not args.benchmark:
        initial = validation(state, network, config)
        summary["initial_validation"] = compact_mimic(initial)
        best_score = initial["selection_score_m"]
        save_checkpoint(args.output_dir/"best.msgpack", state, envs[0], training, 0, norm)
    for iteration in range(1, args.updates+1):
        tick = time.monotonic()
        batches = []
        episode_endings = 0
        tracking_errors = []
        raw_rewards = []
        for _ in range(20):
            key, action_key = jax.random.split(key)
            state, actions, values, lp = act(state, jnp.asarray(obs), action_key)
            actions_np = np.asarray(actions)
            next_obs, rewards, dones = [], [], []
            for env, action in zip(envs, actions_np):
                next_o, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                tracking_errors.append(info["error_q_m"])
                episode_endings += int(done)
                next_obs.append(env.reset()[0] if done else next_o)
                rewards.append(reward)
                dones.append(done)
            rewards, dones = np.asarray(rewards, np.float32), np.asarray(dones, bool)
            raw_rewards.extend(rewards)
            batches.append((obs, actions_np, np.asarray(values), norm.step(rewards, dones), dones, np.asarray(lp)))
            obs = np.stack(next_obs)
        arrays = [jnp.asarray(np.stack(x)) for x in zip(*batches)]
        collection_seconds = time.monotonic()-tick
        update_started = time.monotonic()
        key, update_key = jax.random.split(key)
        state, metrics, audit = update(state, *arrays, last_value(state, jnp.asarray(obs)), update_key)
        metrics, audit = np.asarray(metrics), np.asarray(audit)
        record = {"update": iteration, "steps": iteration*args.num_envs*20,
            "seconds": time.monotonic()-tick, "raw_reward_mean": float(np.mean(raw_rewards)),
            "collection_seconds": collection_seconds, "update_seconds": time.monotonic()-update_started,
            "rollout_q_rmse_m": float(np.sqrt(np.mean(np.square(tracking_errors)))),
            "episode_endings": episode_endings, "actor_loss": float(metrics[0]), "value_loss": float(metrics[1]),
            "entropy": float(metrics[2]), "kl": float(metrics[3]), "advantage_mean": float(audit[0]),
            "advantage_std": float(audit[1]), "critic_explained_variance": float(audit[2]),
            "action_mean": np.mean(np.asarray(arrays[1]), axis=(0, 1)).tolist(),
            "action_clipped_fraction": float(np.mean(np.abs(np.asarray(arrays[1])) > 1)),
            "std": np.exp(np.asarray(state.params["log_std"])).tolist()}
        if not np.isfinite(np.array([*metrics, *audit])).all():
            raise RuntimeError("Nonfinite PPO update")
        if not args.benchmark and (iteration % args.eval_every == 0 or iteration == args.updates):
            result = validation(state, network, config)
            record["validation"] = compact_mimic(result)
            if result["selection_score_m"] < best_score:
                best_score, best_step = result["selection_score_m"], record["steps"]
                save_checkpoint(args.output_dir/"best.msgpack", state, envs[0], training, best_step, norm)
        summary["history"].append(record)
        summary.update({"steps": record["steps"], "best_step": best_step,
                        "wall_seconds": time.monotonic()-started})
        write_json(args.output_dir/"summary.json", summary)
        print(json.dumps({k: record[k] for k in ("update", "steps", "seconds", "rollout_q_rmse_m", "std")}), flush=True)
    save_checkpoint(args.output_dir/"last.msgpack", state, envs[0], training, summary["steps"], norm)
    summary["status"] = "benchmark_complete" if args.benchmark else "complete"
    summary["parameter_count"] = sum(x.size for x in jax.tree.leaves(state.params))
    # All training/validation selection is finished before the held-out motions.
    if not args.benchmark:
        best, network, _, _ = load_checkpoint(args.output_dir/"best.msgpack")
        for name, motions in (("train", TRAIN_MOTIONS), ("test", TEST_MOTIONS)):
            result = evaluate_mimic(MimicPolicy(network, best), config, motions)
            write_json(args.output_dir/f"{name}.json", result)
            summary[name] = compact_mimic(result)
    summary["wall_seconds"] = time.monotonic()-started
    write_json(args.output_dir/"summary.json", summary)
    print(json.dumps({"status": summary["status"], "wall_seconds": summary["wall_seconds"],
                      "parameter_count": summary["parameter_count"]}), flush=True)


if __name__ == "__main__":
    main()
