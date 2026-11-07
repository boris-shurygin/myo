"""One-factor PPO continuation experiments from identical per-mechanic parents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .train_vertical_ppo import compact, write_json
from .vertical_control import VALIDATION_CASES, TEST_CASES, VerticalControlEnv, evaluate_policy, load_policy, save_policy
from .vertical_muscles import VerticalConfig
from .vertical_sweep_env import VerticalSweepEnv, load_sweep_policy


def candidates():
    return [
        ("control", {}),
        ("lr_1e-4", {"learning_rate": .0001}), ("lr_3e-5", {"learning_rate": .00003}),
        ("rollout_2048", {"n_steps": 2048}), ("rollout_4096", {"n_steps": 4096}),
        ("entropy_1e-4", {"ent_coef": .0001}), ("entropy_0", {"ent_coef": 0.}),
        ("lambda_090", {"gae_lambda": .90}), ("lambda_098", {"gae_lambda": .98}),
        ("gamma_0995", {"gamma": .995}), ("batch_256", {"batch_size": 256}),
        ("epochs_5", {"n_epochs": 5}), ("clip_010", {"clip_range": .1}),
        ("kl_0015", {"target_kl": .015}),
        ("position_004", {"position_scale_m": .04}), ("position_002", {"position_scale_m": .02}),
    ]


def optimizer_steps(model):
    state = model.policy.optimizer.state[model.policy.action_net.weight]
    return int(state.get("step", 0))


def configure_model(model, env, changes):
    """Preserve weights/moments; update both PPO and its rollout-buffer settings."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import FloatSchedule
    model.set_env(Monitor(env), force_reset=True)
    for name, value in changes.items():
        if name != "position_scale_m":
            setattr(model, name, FloatSchedule(value) if name == "clip_range" else value)
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group["lr"] = model.lr_schedule(1.)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps, model.observation_space, model.action_space, device=model.device,
        gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs, **model.rollout_buffer_kwargs)
    model.set_random_seed(0)


def actual_parameters(model, env):
    return {"learning_rate": float(model.lr_schedule(1.)), "n_steps": model.n_steps,
            "ent_coef": model.ent_coef, "gae_lambda": model.gae_lambda, "gamma": model.gamma,
            "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "clip_range": float(model.clip_range(1.)), "target_kl": model.target_kl,
            "position_scale_m": env.position_scale_m}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_ppo"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_ppo_sweep"))
    parser.add_argument("--variants", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--candidates", nargs="+", choices=[name for name, _ in candidates()])
    parser.add_argument("--defer-test", action="store_true")
    args = parser.parse_args()
    if args.eval_every % 4096 or args.steps % args.eval_every:
        parser.error("Evaluation and budget must contain complete 4096-step candidate rollouts")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    started = time.monotonic()
    matrix = candidates()
    if args.candidates is not None:
        matrix = [(name, changes) for name, changes in matrix if name in args.candidates]
    summary = {"schema": "myo.vertical_ppo_sweep.v1", "status": "running",
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "matrix": matrix, "parents": {}, "trials": {},
               "scope": "One-factor continuations; same weights and Adam per mechanic; seed0, fresh episode/case cycle; no cumulative tuning",
               "selection": "Within trial: fewer failures, more passed validation cases, lower steady RMSE; includes update0",
               "evaluation_reward": "Common baseline reward on all validation/test, including reward-scale trials",
               "stopping": "Fixed additional physical-transition budget; no plateau/ready early stop"}

    def report():
        write_json(args.output_dir/"summary.json", summary)

    for variant in args.variants:
        parent_path = args.reference/f"variant_{variant}/best.zip"
        physical = VerticalControlEnv(VerticalConfig(variant=variant))
        parent, metadata = load_policy(parent_path, physical)
        summary["parents"][str(variant)] = {
            "path": str(parent_path), "checkpoint_sha256": metadata["checkpoint_sha256"],
            "trained_steps": parent.num_timesteps, "optimizer_steps": optimizer_steps(parent),
            "training": metadata["training"], "contract": metadata["contract"]}
    report()

    # Candidate order is fixed in advance; no parameter is chosen for the next family.
    for name, changes in matrix:
        for variant in args.variants:
            key = f"variant_{variant}/{name}"
            folder = args.output_dir/key
            folder.mkdir(parents=True)
            origin = summary["parents"][str(variant)]
            config = VerticalConfig(variant=variant)
            env = VerticalSweepEnv(config, position_scale_m=changes.get("position_scale_m", .08))
            model, source = load_policy(origin["path"], VerticalControlEnv(config))
            assert source["checkpoint_sha256"] == origin["checkpoint_sha256"]
            configure_model(model, env, changes)
            initial_steps, initial_optimizer = model.num_timesteps, optimizer_steps(model)
            params = actual_parameters(model, env)
            baseline = {k: source["training"].get(k, .08 if k == "position_scale_m" else None) for k in params}
            assert {k: v for k, v in params.items() if v != baseline[k]} == changes
            training = {"parent_checkpoint_sha256": origin["checkpoint_sha256"],
                        "parent_steps": initial_steps, "parent_optimizer_steps": initial_optimizer,
                        "parameters": params, "changes": changes, "seed": 0, "episode_seconds": 4.,
                        "additional_transition_budget": args.steps,
                        "runner_sha256": summary["runner_sha256"], "parent_training": source["training"]}
            record = {"variant": variant, "candidate": name, "changes": changes, "training": training,
                      "history": [], "audit": [], "status": "running"}
            summary["trials"][key] = record

            class Audit(BaseCallback):
                def _on_rollout_start(self):
                    self.raw_rewards = []

                def _on_step(self):
                    self.raw_rewards.append(self.locals["infos"][0]["raw_reward"])
                    return True

                def _on_rollout_end(self):
                    buffer = self.model.rollout_buffer
                    with torch.no_grad():
                        dist = self.model.policy.get_distribution(torch.tensor(buffer.observations[:, 0])).distribution
                    variance = float(np.var(buffer.returns))
                    record["audit"].append({
                        "additional_steps": self.num_timesteps-initial_steps,
                        "raw_reward_mean": float(np.mean(self.raw_rewards)),
                        "critic_ev": float(1-np.var(buffer.returns-buffer.values)/variance) if variance > 1e-12 else None,
                        "excitation_std": dist.stddev.mean(0).tolist(),
                        "out_of_bounds_actions": int(((buffer.actions < 0) | (buffer.actions > 1)).sum())})

            best_key = None
            trial_started = time.monotonic()

            def check():
                nonlocal best_key
                result = evaluate_policy(model, config, VALIDATION_CASES)
                rank = (result["failures"], -result["passed"], result["score_m"])
                step = model.num_timesteps-initial_steps
                if best_key is None or rank < best_key:
                    best_key = rank
                    record["best_additional_steps"] = step
                    save_policy(model, env, folder/"best.zip", training)
                    write_json(folder/"best_validation.json", result)
                record["history"].append({"additional_steps": step, "validation": compact(result),
                    "optimizer_updates": optimizer_steps(model)-initial_optimizer,
                    "ppo": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}
                           if step else {}})
                print(f"{key} +{step}/{args.steps} val={result['score_m']*1000:.3f}mm "
                      f"passed={result['passed']}/5 failures={result['failures']} best@{record['best_additional_steps']}", flush=True)
                report()

            check()
            audit = Audit()
            for _ in range(args.steps//args.eval_every):
                model.learn(args.eval_every, reset_num_timesteps=False, callback=audit)
                check()
            save_policy(model, env, folder/"last.zip", training)
            record.update(status="complete", elapsed_s=time.monotonic()-trial_started,
                          optimizer_updates=optimizer_steps(model)-initial_optimizer)
            best, _, _ = load_sweep_policy(folder/"best.zip")
            record["selected_validation"] = compact(evaluate_policy(best, config, VALIDATION_CASES))
            record["selection_frozen_before_test"] = {
                "best_additional_steps": record["best_additional_steps"],
                "checkpoint_sha256": json.loads((folder/"best.zip.metadata.json").read_text())["checkpoint_sha256"]}
            model.get_env().close()
            report()

    if args.defer_test:
        summary.update(status="training_complete", elapsed_s=time.monotonic()-started)
        report()
        return
    summary["all_selections_frozen_before_test"] = True
    report()
    for key, record in summary["trials"].items():
        folder = args.output_dir/key
        model, env, _ = load_sweep_policy(folder/"best.zip")
        test = evaluate_policy(model, env.physical.config, TEST_CASES)
        switches = evaluate_policy(model, env.physical.config, ((0., .06),), seconds=6., switches=True)
        record["test"], record["switches"] = compact(test), compact(switches)
        write_json(folder/"test.json", test)
        write_json(folder/"switches.json", switches)
        report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
