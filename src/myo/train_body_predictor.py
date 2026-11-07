"""Train the current predictor on fixed diverse motor episodes, independently of PPO."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from .body_predictor import (BodyPredictor, FAMILIES, collect, load_dataset, indices,
    forecast_metrics, save_predictor, load_predictor, contract)
from .detached_vertical import load_detached
from .train_vertical_ppo import write_json


def native_sensitivities(models, data, metadata):
    from .vertical_muscles import VerticalConfig
    from .vertical_precision import VerticalPrecisionEnv
    native, predictions, positions, velocities = [], {name: [] for name in models}, [], []
    for family in FAMILIES:
        episode = next(e for e in metadata['episodes'] if e['split'] == 'test' and e['family'] == family)
        env = VerticalPrecisionEnv(VerticalConfig(variant=4), episode['requested_duration_s'])
        env.reset(options={'initial_position_m': episode['position_m'], 'target_m': 0.})
        reached = 0
        for index in (50, 150, 350):
            if index+3 > episode['complete_transitions']:
                continue
            for k in range(reached, index):
                env.step(data['actions'][episode['start']+k])
            reached = index
            offset = episode['start']+index
            np.testing.assert_array_equal(env.observation(), data['observations'][offset])
            snapshot = env.physical.snapshot()
            positions.append(env.physical.position); velocities.append(env.physical.velocity)
            commands = data['actions'][offset:offset+3].copy()
            derivative = []
            for muscle in range(2):
                ends, issued = [], []
                for sign in (-1, 1):
                    action = commands[0].copy()
                    action[muscle] = np.clip(action[muscle]+sign*.005, 0, 1)
                    env.physical.restore(snapshot)
                    env.physical.step(action)
                    for command in commands[1:]:
                        env.physical.step(command)
                    ends.append(env.physical.velocity); issued.append(float(action[muscle]))
                derivative.append((ends[1]-ends[0])/(issued[1]-issued[0]))
            native.append(derivative)
            env.physical.restore(snapshot)
            for name, model in models.items():
                current = torch.as_tensor(data['observations'][offset:offset+1])
                first = torch.as_tensor(commands[0:1]).clone().requires_grad_(True)
                for k in range(3):
                    command = first if k == 0 else torch.as_tensor(commands[k:k+1])
                    current = model.next_observation(current, command)
                gradient = torch.autograd.grad(current[0, 1]*.5, first)[0]
                predictions[name].append(gradient.detach().numpy()[0].tolist())
        env.close()
    native = np.asarray(native)
    results = {}
    for name, values in predictions.items():
        values = np.asarray(values)
        results[name] = {'predicted': values.tolist(),
            'relative_rmse_to_zero': float(np.sqrt(np.mean((values-native)**2)/np.mean(native**2))),
            'cosine': float(np.sum(values*native)/max(1e-20, np.linalg.norm(values)*np.linalg.norm(native)))}
    return {'probes': len(native), 'horizon_ms': 60, 'native': native.tolist(),
            'positions_m': positions, 'velocities_m_s': velocities, 'models': results,
            'source_replay_exact': True, 'native_command_epsilon': .005}


def train(root, maximum, minimum, patience):
    if (root/"training.json").exists():
        raise FileExistsError(root/"training.json")
    data, metadata = load_dataset(root)
    train_ids, val_ids = indices(metadata, "train"), indices(metadata, "validation")
    model = BodyPredictor.from_current_initialization()
    assert sum(p.numel() for p in model.parameters()) == 51462
    x = torch.as_tensor(data["observations"][train_ids])
    labels = torch.as_tensor(data["next_observations"][train_ids])
    model.fit_normalization(x, labels)
    target = (labels[:, :6]-x[:, :6]-model.delta_mean)/model.delta_std
    optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4, eps=1e-5)
    generator = torch.Generator().manual_seed(0)
    training = {"optimizer": "RMSprop", "lr": 3e-4, "alpha": .99, "eps": 1e-5,
        "batch_size": 128, "gradient_norm_limit": .5, "seed": 0, "maximum_updates": maximum,
        "minimum_updates": minimum, "patience_updates": patience, "significant_fraction": .005,
        "check_every": 500, "select": "minimum fixed whole-validation one-step normalized physical MSE",
        "stop": "after minimum updates, no0.5% accumulated validation improvement for patience; else maximum",
        "normalization": "all complete train transitions only, frozen; original normalized input8 unchanged",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {k: version(k) for k in ("torch", "stable-baselines3", "mujoco", "numpy")}}
    summary = {"status": "running", "contract": contract(), "training": training,
        "dataset_sha256": metadata["dataset_sha256"], "parameters": 51462,
        "unique_train_transitions": len(train_ids), "history": []}
    started = time.monotonic()
    best, significant, significant_update = float("inf"), float("inf"), 0

    def check(update):
        nonlocal best, significant, significant_update
        train_metrics = forecast_metrics(model, data, train_ids)
        validation = forecast_metrics(model, data, val_ids)
        score = validation["normalized_mse"]
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation error")
        if score < best:
            best = score
            summary.update(best_update=update, best_validation=validation)
            save_predictor(root/"best.pt", model, optimizer, update, metadata, training)
        if score < significant*(1-.005):
            significant, significant_update = score, update
        row = {"update": update, "train": train_metrics, "validation": validation}
        if update % 2500 == 0:
            row["validation_60ms"] = forecast_metrics(model, data, indices(metadata, "validation", 3), 3)
        summary["history"].append(row)
        summary.update(elapsed_s=time.monotonic()-started, last_significant_update=significant_update)
        write_json(root/"training.json", summary)
        print(json.dumps({"update": update, "train_mse": train_metrics["normalized_mse"],
            "validation_mse": score, "q20_mm": validation["q_rmse_mm"],
            "v20_mm_s": validation["v_rmse_mm_s"], "best_update": summary["best_update"]}), flush=True)

    save_predictor(root/"initial.pt", model, optimizer, 0, metadata, training)
    check(0)
    reason = "maximum_updates"
    for update in range(1, maximum+1):
        batch = torch.randint(len(x), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = (model.normalized_delta(x[batch])-target[batch]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite training gradient")
        optimizer.step()
        if update % 500 == 0 or update == maximum:
            check(update)
            if update >= minimum and update-significant_update >= patience:
                reason = "fixed_validation_plateau"
                break
    save_predictor(root/"last.pt", model, optimizer, update, metadata, training)
    summary.update(status="training_complete", updates=update, stop_reason=reason,
        sampled_transitions=update*128, elapsed_s=time.monotonic()-started)
    write_json(root/"training.json", summary)


def evaluate(root):
    data, metadata = load_dataset(root)
    summary = json.loads((root/"training.json").read_text())
    assert summary["status"] == "training_complete"
    selected, _, selected_metadata = load_predictor(root/"best.pt")
    assert selected_metadata["update"] == summary["best_update"]
    assert forecast_metrics(selected, data, indices(metadata, "validation")) == summary["best_validation"]
    write_json(root/"frozen_selection.json", {"basis": "one-step validation only; no test selection",
        "best_update": summary["best_update"], "checkpoint_sha256": selected_metadata["checkpoint_sha256"]})
    models = {"separate_best": selected}
    last, _, _ = load_predictor(root/"last.pt")
    models["separate_last"] = last
    # Read-only references. No policy actions or old labels enter optimizer.
    for label, checkpoint in (("parallel_actor_best", "best"), ("parallel_last", "last")):
        policy, env, _ = load_detached(Path("runs/detached_vertical/online")/f"{checkpoint}.zip")
        env.close(); models[label] = policy.policy
    results = {}
    for label, model in models.items():
        results[label] = {split: {str(h*20): forecast_metrics(model, data, indices(metadata, split, h), h,
                        scales=selected.delta_std) for h in (1, 2, 3, 6)}
                        for split in ("train", "validation", "test")}
        results[label]["test_families"] = {family: {str(h*20): forecast_metrics(model, data,
            indices(metadata, "test", h, family), h, scales=selected.delta_std) for h in (1, 3)} for family in FAMILIES}
    # Same old fixed-policy data is a separate transfer audit, never train/selection.
    old = dict(np.load("runs/detached_vertical/prediction_validation.npz"))
    from .body_predictor import SCALES
    old_metrics = {}
    for label, model in models.items():
        row = {}
        for h in (1, 3):
            count = old["observations"].shape[1]-h+1
            current = torch.as_tensor(old["observations"][:, :count].reshape(-1, 13))
            with torch.no_grad():
                for step in range(h):
                    current = model.next_observation(current, torch.as_tensor(old["actions"][:, step:step+count].reshape(-1, 2)))
                actual = torch.as_tensor(old["next_observations"][:, h-1:h-1+count].reshape(-1, 13))
                rmse = (current[:, :6]-actual[:, :6]).square().mean(0).sqrt().numpy()*SCALES
            row[str(h*20)] = {"q_rmse_mm": float(rmse[0]*1000), "v_rmse_mm_s": float(rmse[1]*1000)}
        old_metrics[label] = row
    jacobian = native_sensitivities(models, data, metadata)
    result = {"status": "passed", "dataset_sha256": metadata["dataset_sha256"],
        "selection_reproduced": True, "metrics": results, "old_fixed_policy_transfer": old_metrics,
        "native_sensitivities": jacobian}
    write_json(root/"evaluation.json", result)
    summary["status"] = "complete"
    write_json(root/"training.json", summary)
    print(json.dumps({label: result["metrics"][label]["test"]["20"] for label in models}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "train", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("runs/body_predictor"))
    parser.add_argument("--maximum-updates", type=int, default=50000)
    parser.add_argument("--minimum-updates", type=int, default=10000)
    parser.add_argument("--patience-updates", type=int, default=6000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "collect":
        metadata = collect(args.root)
        print(json.dumps({"episodes": len(metadata["episodes"]),
            "transitions": sum(e["complete_transitions"] for e in metadata["episodes"]),
            "failures": sum(e["physical_failure"] for e in metadata["episodes"])}))
    elif args.mode == "train":
        train(args.root, args.maximum_updates, args.minimum_updates, args.patience_updates)
    else:
        evaluate(args.root)


if __name__ == "__main__":
    main()
