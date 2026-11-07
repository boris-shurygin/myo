"""Change the wide transition model's velocity loss to interval acceleration.

An acceleration label is (v_next - v_before) / control_dt, not endpoint qacc.
One-step normalized supervision is algebraically identical to the previous
delta-velocity loss. Only multi-step loss changes: each predicted interval is
compared with its native interval, rather than penalizing accumulated velocity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch

from .dynamics_prediction import contract, sequence_windows
from .env import PreacherCurlConfig, PreacherCurlEnv
from .train_dynamics_prediction import evaluate, native_audit, tensors, validation_score
from .train_dynamics_wide import (WideTransitionModel, load_wide_predictor,
                                 save_wide_predictor, wide_contract)


def acceleration_loss(initial, prediction, target, model, control_dt):
    residual = (prediction[..., :model.physical_size] - target[..., :model.physical_size]) / model.delta_scale
    predicted_before = torch.cat((initial[:, None, 1], prediction[:, :-1, 1]), dim=1)
    actual_before = torch.cat((initial[:, None, 1], target[:, :-1, 1]), dim=1)
    predicted_acceleration = (prediction[..., 1] - predicted_before) / control_dt
    actual_acceleration = (target[..., 1] - actual_before) / control_dt
    acceleration_residual = (predicted_acceleration - actual_acceleration) / (model.delta_scale[1] / control_dt)
    # Preserve all other 19 state channels and their weights exactly.
    residual = torch.cat((residual[..., :1], acceleration_residual.unsqueeze(-1), residual[..., 2:]), dim=-1)
    return residual.square().mean()


@torch.no_grad()
def acceleration_metrics(model, episodes, control_dt, horizons=(1, 2, 3, 6, 10, 20)):
    result = {}
    for horizon in horizons:
        data = tensors(sequence_windows(episodes, horizon))
        prediction = torch.cat([model.rollout(s, a) for s, a in
            zip(data["initial"].split(512), data["actions"].split(512))])
        before_predicted = data["initial"][:, 1] if horizon == 1 else prediction[:, -2, 1]
        before_actual = data["initial"][:, 1] if horizon == 1 else data["targets"][:, -2, 1]
        actual = (data["targets"][:, -1, 1] - before_actual) / control_dt
        predicted = (prediction[:, -1, 1] - before_predicted) / control_dt
        result[str(horizon)] = {
            "interval_acceleration_rmse_rad_s2": float((predicted - actual).square().mean().sqrt()),
            "zero_acceleration_rmse_rad_s2": float(actual.square().mean().sqrt()),
            "samples": len(actual),
        }
    return result


def acceleration_score(metrics):
    return float(np.mean([metrics[str(h)]["interval_acceleration_rmse_rad_s2"] for h in (1, 3, 6)]))


def train(initial, train_episodes, validation_episodes, metadata, args):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = tensors(sequence_windows(train_episodes, 6))
    generator = torch.Generator().manual_seed(0)
    best_acceleration, best_velocity = float("inf"), float("inf")
    history, selections = [], {}
    dt = metadata["control_dt_s"]
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        ids = torch.randint(len(data["initial"]), (args.batch_size,), generator=generator)
        prediction = model.rollout(data["initial"][ids], data["actions"][ids])
        loss = acceleration_loss(data["initial"][ids], prediction, data["targets"][ids], model, dt)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite acceleration loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if update == 1 or update % args.check_every == 0 or update == args.updates:
            velocity = evaluate(model, validation_episodes, (1, 3, 6))
            acceleration = acceleration_metrics(model, validation_episodes, dt, (1, 3, 6))
            score_a, score_v = acceleration_score(acceleration), validation_score(velocity)
            history.append({"update": update, "training_loss": float(loss.detach()),
                "gradient_norm_before_clip": float(norm), "score": score_v,
                "acceleration_score": score_a, "validation": velocity,
                "acceleration_validation": acceleration})
            for name, improved, criterion in (
                ("bptt_6", score_a < best_acceleration, "mean interval acceleration RMSE at 20/60/120 ms"),
                ("bptt_velocity_selection", score_v < best_velocity, "mean velocity RMSE at 20/60/120 ms"),
            ):
                if improved:
                    selections[name] = update
                    save_wide_predictor(args.output_dir / f"{name}.pt", model, {**metadata,
                        "training_mode": "bptt_6_interval_acceleration", "update": update,
                        "training_target": "mean acceleration of each 20 ms interval; other state channels unchanged",
                        "selection": criterion, "seed": 0, "training_horizon_steps": 6,
                        "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "normalization": "same frozen training-only delta scales as velocity reference",
                        "loss": "local interval acceleration residual replaces accumulated velocity residual"})
            best_acceleration, best_velocity = min(best_acceleration, score_a), min(best_velocity, score_v)
            print(f"update {update}: loss={float(loss.detach()):.5g}, acceleration={score_a:.5g}, "
                  f"velocity={score_v:.5g}, best_a={best_acceleration:.5g}, best_v={best_velocity:.5g}", flush=True)
    return {"updates": args.updates, "elapsed_s": time.monotonic() - started,
            "history": history, "selected_updates": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/d1_dynamics_prediction_wide"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_dynamics_acceleration"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--check-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    original = json.loads((args.reference / "summary.json").read_text())
    config = PreacherCurlConfig(**original["contract"]["config"])
    env = PreacherCurlEnv(config)
    try:
        env.reset(seed=0, options={"payload_mass": config.payload_mass})
        metadata = wide_contract(contract(env))
        if metadata != original["contract"]:
            raise ValueError("Reference dynamics/input/implementation contract changed")
    finally:
        env.close()
    records = json.loads((args.reference / "episodes.json").read_text())
    with np.load(args.reference / "episodes.npz", allow_pickle=False) as arrays:
        episodes = [{**record, "states": arrays[f"e{i}_states"], "actions": arrays[f"e{i}_actions"]}
                    for i, record in enumerate(records)]
    with (args.reference / "native_probes.pkl").open("rb") as stream:
        probes = pickle.load(stream)
    for filename in ("episodes.npz", "episodes.json", "native_probes.pkl", "one_step.pt", "one_step.pt.metadata.json"):
        shutil.copy2(args.reference / filename, args.output_dir / filename)
    split = {name: [e for e in episodes if e["split"] == name] for name in ("train", "validation", "test")}
    initial = WideTransitionModel(hidden_layers=original["arguments"]["hidden_layers"])
    initial.fit_normalization(split["train"])
    summary = {"schema": "myo.dynamics_acceleration_experiment.v1", "status": "training",
        "contract": metadata, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference": str(args.reference), "architecture": original["architecture"],
        "trainable_parameters": sum(p.numel() for p in initial.parameters()),
        "dataset_sha256": hashlib.sha256((args.output_dir / "episodes.npz").read_bytes()).hexdigest(),
        "collection_physical_steps": 0,
        "split_episodes": {name: [e["id"] for e in values] for name, values in split.items()},
        "evaluated_models": ["one_step", "bptt_6", "bptt_velocity_selection"],
        "one_step": {**original["one_step"], "trained_in_this_run": False,
                     "reason": "normalized one-step velocity and interval acceleration losses are equivalent"}}
    def report():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report()
    training = train(initial, split["train"], split["validation"], metadata, args)
    models = {}
    for name in summary["evaluated_models"]:
        models[name], _ = load_wide_predictor(args.output_dir / f"{name}.pt", metadata)
        if name != "one_step":
            summary[name] = {"training": {**training, "best_update": training["selected_updates"][name]},
                             "test": evaluate(models[name], split["test"])}
        summary[name]["acceleration_test"] = acceleration_metrics(models[name], split["test"], config.control_dt)
        report()
    summary["velocity_reference_acceleration_test"] = {}
    for name in ("one_step", "bptt_6"):
        reference, _ = load_wide_predictor(args.reference / f"{name}.pt", metadata)
        summary["velocity_reference_acceleration_test"][name] = acceleration_metrics(reference, split["test"], config.control_dt)
    summary["native_audit"] = native_audit(models, probes, config, args.output_dir)
    with np.load(args.reference / "jacobians.npz") as a, np.load(args.output_dir / "jacobians.npz") as b:
        summary["native_reference_identical"] = bool(np.array_equal(a["native"], b["native"]))
    summary["status"], summary["elapsed_s"] = "complete", time.monotonic() - started
    summary["limitations"] = ["One initialization and fixed 2000-update budget, not convergence",
        "Acceleration is an interval mean, not instantaneous MuJoCo qacc",
        "Other state losses remain; only the velocity channel is replaced",
        "Two checkpoint selections share one training trajectory, not independent runs",
        "No motor actor or hierarchy trained"]
    report()
    print(json.dumps({name: {"velocity": summary[name]["test"], "acceleration": summary[name]["acceleration_test"]}
                      for name in models}, indent=2), flush=True)


if __name__ == "__main__":
    main()
