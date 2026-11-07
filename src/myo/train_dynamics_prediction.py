"""Headless comparison of one-step supervision and multi-step BPTT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .delayed_acceleration import NativeResponse, capture
from .dynamics_prediction import (TransitionModel, compact_state, contract, load_predictor,
                                  save_predictor, sequence_windows)
from .env import PreacherCurlEnv
from .runtime import read_config


def motor_probe(family, step, variant, muscles=6):
    """Three reproducible excitation families, with no heuristic or seed search."""
    t = step * .02
    phase = (0., .37, .19, .61)[variant]
    index = np.arange(muscles)
    if family == "pulses":
        widths = np.array([4, 8, 15, 25])
        width = int(widths[(step // 50 + variant) % len(widths)])
        block = (step + int(phase * 50)) // width
        command = np.full(muscles, .01)
        command[(block + variant) % muscles] = (.12, .25, .45)[block % 3]
        if block % 4 == 3:
            command[:] = 0.
    elif family == "smooth":
        command = .18 * (1 + np.sin(2 * np.pi * (t + phase) / (.45 + .13 * index) + index))
    else:
        block = (step + int(phase * 50)) // 12
        weights = .7 + .3 * np.sin(index + block)
        command = np.where((index < 3) == (block % 2 == 0), .38 * weights, .015)
        if block % 5 == 4:
            command[:] = .12  # Coactivation followed by another switch.
    return command.astype(np.float32)


def collect(config, output_dir, seconds):
    env = PreacherCurlEnv(config)
    episodes, probes, physical_steps = [], [], 0
    try:
        for split, variants in (("train", (0, 1)), ("validation", (2,)), ("test", (3,))):
            for variant in variants:
                for family in ("pulses", "smooth", "switching"):
                    for angle in (20., 75., 135.):
                        for capacity in (1., .8):
                            identifier = f"{split}/{family}/v{variant}/q{angle:g}/c{capacity:g}"
                            env.reset(seed=0, options={"q": float(np.deg2rad(angle)), "qvel": 0.,
                                "target_angle": float(np.deg2rad(angle)), "trigger": 1.,
                                "capacity": capacity, "payload_mass": config.payload_mass})
                            env.set_command(float(np.deg2rad(angle)), 1.)
                            states, actions, terminated = [compact_state(env)], [], False
                            snapshots = {}
                            for step in range(round(seconds / env.control_dt)):
                                if split == "test" and step in (15, 55):
                                    snapshots[step] = capture(env)
                                command = motor_probe(family, step, variant)
                                _, _, terminated, truncated, _ = env.step(command)
                                physical_steps += 1
                                actions.append(command)
                                states.append(compact_state(env))
                                if terminated:
                                    break
                                if truncated:
                                    env._episode_done = False  # Only time limit; no physical rescue.
                            episode = {"id": identifier, "split": split, "family": family,
                                       "initial_angle_deg": angle, "initial_capacity": capacity,
                                       "states": np.asarray(states), "actions": np.asarray(actions),
                                       "physical_failure": bool(terminated)}
                            episodes.append(episode)
                            for step, snapshot in snapshots.items():
                                if len(actions) >= step + 6:
                                    probes.append({"episode_id": identifier, "step": step,
                                        "state": snapshot, "compact": states[step],
                                        "actions": np.asarray(actions[step:step + 6])})
                            print(f"collected {len(episodes)}/72 {identifier}: {len(actions)} steps", flush=True)
        metadata = contract(env)
    finally:
        env.close()
    arrays = {f"e{i}_{key}": e[key] for i, e in enumerate(episodes) for key in ("states", "actions")}
    np.savez_compressed(output_dir / "episodes.npz", **arrays)
    (output_dir / "episodes.json").write_text(json.dumps([
        {k: v for k, v in e.items() if k not in ("states", "actions")} for e in episodes], indent=2) + "\n")
    with (output_dir / "native_probes.pkl").open("wb") as stream:
        pickle.dump(probes, stream)
    return episodes, probes, metadata, physical_steps


def tensors(data):
    return {k: torch.as_tensor(v) for k, v in data.items() if k != "episode_ids"}


@torch.no_grad()
def evaluate(model, episodes, horizons=(1, 2, 3, 6, 10, 20)):
    result = {}
    n = model.muscles
    for horizon in horizons:
        data = sequence_windows(episodes, horizon)
        x = tensors(data)
        predicted = torch.cat([model.rollout(s, a)[:, -1] for s, a in
            zip(x["initial"].split(512), x["actions"].split(512))]).numpy()
        actual = data["targets"][:, -1]
        error = predicted - actual
        initial = data["initial"]
        metrics = {
            "samples": len(actual), "angle_rmse_rad": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "velocity_rmse_rad_s": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "activation_rmse": float(np.sqrt(np.mean(error[:, 2:2 + n] ** 2))),
            "fiber_length_rmse_m": float(np.sqrt(np.mean(error[:, 2 + n:2 + 2*n] ** 2))),
            "capacity_proxy_rmse": float(np.sqrt(np.mean(error[:, 2 + 2*n:2 + 3*n] ** 2))),
            "persistence_velocity_rmse_rad_s": float(np.sqrt(np.mean((initial[:, 1] - actual[:, 1]) ** 2))),
            "constant_velocity_angle_rmse_rad": float(np.sqrt(np.mean(
                (initial[:, 0] + horizon * .02 * initial[:, 1] - actual[:, 0]) ** 2))),
            "nonfinite_fraction": float(np.mean(~np.isfinite(predicted))),
        }
        result[str(horizon)] = metrics
    return result


def validation_score(metrics):
    return float(np.mean([metrics[str(h)]["velocity_rmse_rad_s"] for h in (1, 3, 6)]))


def train_branch(initial, train_episodes, validation_episodes, metadata, args, mode):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Both branches use identical six-step windows and identical first-step batches.
    data = tensors(sequence_windows(train_episodes, 6))
    generator = torch.Generator().manual_seed(0)
    best, best_update, history = float("inf"), 0, []
    path = args.output_dir / f"{mode}.pt"
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        ids = torch.randint(len(data["initial"]), (args.batch_size,), generator=generator)
        horizon = 1 if mode == "one_step" else 6
        prediction = model.rollout(data["initial"][ids], data["actions"][ids, :horizon])
        residual = (prediction[..., :model.physical_size] -
                    data["targets"][ids, :horizon, :model.physical_size]) / model.delta_scale
        loss = residual.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in {mode}")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if update == 1 or update % args.check_every == 0 or update == args.updates:
            metrics = evaluate(model, validation_episodes, (1, 3, 6))
            score = validation_score(metrics)
            history.append({"update": update, "training_loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(norm), "score": score, "validation": metrics})
            if score < best:
                best, best_update = score, update
                save_predictor(path, model, {**metadata, "training_mode": mode, "update": update,
                    "seed": 0, "normalization": "train episodes only; frozen for both branches",
                    "loss": "physical state squared error / train one-step delta scale squared",
                    "training_horizon_steps": horizon})
            print(f"{mode}: update {update}, loss={float(loss.detach()):.5g}, val={score:.5g}, best={best:.5g}", flush=True)
    model, _ = load_predictor(path, metadata)
    return model, {"best_update": best_update, "updates": args.updates,
                   "elapsed_s": time.monotonic() - started, "history": history}


def native_audit(models, probes, config, output_dir):
    """Counterfactual changes and Jacobians, with all later commands held fixed."""
    kernel = NativeResponse(config, 6)
    references, predictions = [], {name: [] for name in models}
    try:
        for index, probe in enumerate(probes):
            commands = probe["actions"].astype(np.float64)
            _, jac_acc = kernel.linearize(probe["state"], commands, commands[0], epsilon=.005)
            jac_velocity = np.cumsum(jac_acc, axis=0) * config.control_dt
            references.append(jac_velocity)
            for name, model in models.items():
                state = torch.tensor(probe["compact"])
                actions = torch.tensor(commands, dtype=torch.float32, requires_grad=True)
                jac = torch.autograd.functional.jacobian(
                    lambda u: model.rollout(state, u)[:, 1], actions)[:, 0, :]
                predictions[name].append(jac.detach().numpy())
            print(f"native derivative audit {index + 1}/{len(probes)}", flush=True)
        actual = np.asarray(references)
        result = {}
        for name, rows in predictions.items():
            estimated = np.asarray(rows)
            meaningful = np.abs(actual[:, 2:]) > .05
            a, b = actual[:, 2:].ravel(), estimated[:, 2:].ravel()
            result[name] = {
                "post_delay_jacobian_rmse_rad_s_per_excitation": float(np.sqrt(np.mean((a - b) ** 2))),
                "post_delay_sign_agreement": float(np.mean(
                    np.sign(estimated[:, 2:][meaningful]) == np.sign(actual[:, 2:][meaningful]))),
                "meaningful_derivative_count": int(meaningful.sum()),
                "global_cosine": float(a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))),
                "pre_delay_max_abs": float(np.max(np.abs(estimated[:, :2]))),
            }
        np.savez_compressed(output_dir / "jacobians.npz",
                            native=actual, **{k: np.asarray(v) for k, v in predictions.items()})
        return {"probes": len(probes), "physical_control_steps": kernel.control_steps,
                "native_pre_delay_max_abs": float(np.max(np.abs(actual[:, :2]))), "models": result}
    finally:
        kernel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_dynamics_prediction"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=3.)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    config = read_config(args.config)
    episodes, probes, metadata, collection_steps = collect(config, args.output_dir, args.seconds)
    split = {name: [e for e in episodes if e["split"] == name] for name in ("train", "validation", "test")}
    initial = TransitionModel()
    initial.fit_normalization(split["train"])
    summary = {"schema": "myo.dynamics_experiment.v1", "status": "training", "contract": metadata,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "collection_physical_steps": collection_steps,
               "physical_failures": sum(e["physical_failure"] for e in episodes),
               "split_episodes": {name: [e["id"] for e in items] for name, items in split.items()}}
    def report():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report()
    models = {}
    for name in ("one_step", "bptt_6"):
        model, training = train_branch(initial, split["train"], split["validation"], metadata, args, name)
        models[name] = model
        summary[name] = {"training": training, "test": evaluate(model, split["test"])}
        report()
    summary["native_audit"] = native_audit(models, probes, config, args.output_dir)
    summary["status"] = "complete"
    summary["elapsed_s"] = time.monotonic() - started
    summary["limitations"] = ["Predictor experiment, no new motor policy or hierarchy trained",
        "Fixed 1 kg payload, fixed mount, compact simulator state; no noisy biological sensors",
        "Known transport queue is exact; physical muscle/body transition is learned",
        "Test future action sequences are recorded conditioning inputs, not future state observations",
        "Single initialization; no claim of architecture superiority or seed robustness"]
    report()
    print(json.dumps({name: summary[name]["test"] for name in models}, indent=2), flush=True)


if __name__ == "__main__":
    main()
