"""Repeat the fixed dynamics experiment with a 1024x1024x512 MLP.

Reuse the original episodes and native probe states byte for byte. Keep the
original transition module untouched: its source hash is pinned by the small
predictors, which remain independently loadable for the comparison.
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
from torch import nn

from .dynamics_prediction import TransitionModel, contract, sequence_windows
from .env import PreacherCurlConfig, PreacherCurlEnv
from .train_dynamics_prediction import evaluate, native_audit, tensors, validation_score


class WideTransitionModel(TransitionModel):
    def __init__(self, muscles=6, queue_steps=2, hidden_layers=(1024, 1024, 512)):
        # Inherit exactly the same state, normalization and delay implementation.
        super().__init__(muscles=muscles, queue_steps=queue_steps, hidden=1)
        self.hidden_layers = tuple(hidden_layers)
        widths = (self.physical_size + muscles, *self.hidden_layers, self.physical_size)
        layers = []
        for i, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(before, after))
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


def wide_contract(base):
    return {**base, "schema": "myo.action_dynamics_wide.v1",
            "base_source_sha256": base["source_sha256"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save_wide_predictor(path, model, metadata):
    architecture = {"muscles": model.muscles, "queue_steps": model.queue_steps,
                    "hidden_layers": list(model.hidden_layers)}
    torch.save({"state_dict": model.state_dict(), "architecture": architecture}, path)
    sidecar = {**metadata, "architecture": architecture,
               "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path) + ".metadata.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")


def load_wide_predictor(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Predictor weights differ from metadata")
    for key in ("schema", "config_sha256", "dynamics_sha256", "geometry_sha256", "state_names",
                "action_mapping", "action_semantics", "physics_dt_s", "control_dt_s",
                "payload_mass_kg", "queue_steps", "source_sha256", "base_source_sha256"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"Predictor contract mismatch: {key}")
    if metadata["schema"] != "myo.action_dynamics_wide.v1":
        raise ValueError("Not a wide dynamics predictor")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Predictor architecture differs from metadata")
    model = WideTransitionModel(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


def train_branch(initial, train_episodes, validation_episodes, metadata, args, mode):
    """Same optimizer, minibatches, loss, clipping and selection as the small run."""
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
                save_wide_predictor(path, model, {**metadata, "training_mode": mode, "update": update,
                    "seed": 0, "normalization": "train episodes only; frozen for both branches",
                    "loss": "physical state squared error / train one-step delta scale squared",
                    "training_horizon_steps": horizon})
            print(f"{mode}: update {update}, loss={float(loss.detach()):.5g}, val={score:.5g}, best={best:.5g}", flush=True)
    model, _ = load_wide_predictor(path, metadata)
    return model, {"best_update": best_update, "updates": args.updates,
                   "elapsed_s": time.monotonic() - started, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/d1_dynamics_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_dynamics_prediction_wide"))
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[1024, 1024, 512])
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--check-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    original = json.loads((args.dataset / "summary.json").read_text())
    config = PreacherCurlConfig(**original["contract"]["config"])
    env = PreacherCurlEnv(config)
    try:
        env.reset(seed=0, options={"payload_mass": config.payload_mass})
        base = contract(env)
        if base != original["contract"]:
            raise ValueError("Original dataset physical/input contract differs from this installation")
        metadata = wide_contract(base)
    finally:
        env.close()
    records = json.loads((args.dataset / "episodes.json").read_text())
    with np.load(args.dataset / "episodes.npz", allow_pickle=False) as arrays:
        episodes = [{**record, "states": arrays[f"e{i}_states"], "actions": arrays[f"e{i}_actions"]}
                    for i, record in enumerate(records)]
    with (args.dataset / "native_probes.pkl").open("rb") as stream:
        probes = pickle.load(stream)  # Locally generated original experiment snapshots.
    for filename in ("episodes.npz", "episodes.json", "native_probes.pkl"):
        shutil.copy2(args.dataset / filename, args.output_dir / filename)
    split = {name: [e for e in episodes if e["split"] == name] for name in ("train", "validation", "test")}
    initial = WideTransitionModel(hidden_layers=args.hidden_layers)
    initial.fit_normalization(split["train"])
    summary = {"schema": "myo.dynamics_width_experiment.v1", "status": "training", "contract": metadata,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "shared_runner_sha256": hashlib.sha256(Path(__file__).with_name("train_dynamics_prediction.py").read_bytes()).hexdigest(),
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "architecture": [26, *args.hidden_layers, 20],
               "trainable_parameters": sum(p.numel() for p in initial.parameters()),
               "dataset": {"source": str(args.dataset), "collection_physical_steps": 0,
                   "original_collection_physical_steps": original["collection_physical_steps"],
                   "sha256": hashlib.sha256((args.dataset / "episodes.npz").read_bytes()).hexdigest(),
                   "summary_sha256": hashlib.sha256((args.dataset / "summary.json").read_bytes()).hexdigest()},
               "physical_failures": sum(e["physical_failure"] for e in episodes),
               "split_episodes": {name: [e["id"] for e in items] for name, items in split.items()}}
    def report():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report()
    print(f"Architecture {summary['architecture']}, parameters={summary['trainable_parameters']}; reused original data", flush=True)
    models = {}
    for name in ("one_step", "bptt_6"):
        model, training = train_branch(initial, split["train"], split["validation"], metadata, args, name)
        models[name] = model
        summary[name] = {"training": training, "test": evaluate(model, split["test"])}
        report()
    summary["native_audit"] = native_audit(models, probes, config, args.output_dir)
    with np.load(args.dataset / "jacobians.npz") as previous, np.load(args.output_dir / "jacobians.npz") as current:
        summary["native_reference_identical"] = bool(np.array_equal(previous["native"], current["native"]))
    summary["status"] = "complete"
    summary["elapsed_s"] = time.monotonic() - started
    summary["limitations"] = ["Same update budget, not equal CPU time or proof of convergence",
        "One fixed initialization, no seed or optimizer search",
        "Predictors only: no actor, hierarchy or feature transfer trained"]
    report()
    print(json.dumps({name: summary[name]["test"] for name in models}, indent=2), flush=True)


if __name__ == "__main__":
    main()
