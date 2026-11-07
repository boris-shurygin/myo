"""Repeat acceleration BPTT with causal fiber-velocity and tendon-force inputs."""
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

from .delayed_acceleration import NativeResponse
from .dynamics_prediction import contract
from .dynamics_sensors import SensorTransitionModel, augment_episodes, load_sensor_predictor, sensor_contract
from .env import PreacherCurlConfig, PreacherCurlEnv
from .train_dynamics_acceleration import acceleration_metrics, train
from .train_dynamics_prediction import evaluate, native_audit
from .train_dynamics_wide import load_wide_predictor, wide_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/d1_dynamics_acceleration"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_dynamics_sensors"))
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
    records = json.loads((args.reference / "episodes.json").read_text())
    with np.load(args.reference / "episodes.npz", allow_pickle=False) as arrays:
        source = [{**record, "states": arrays[f"e{i}_states"], "actions": arrays[f"e{i}_actions"]}
                  for i, record in enumerate(records)]
    episodes = augment_episodes(source)
    split = {name: [e for e in episodes if e["split"] == name] for name in ("train", "validation", "test")}
    initial = SensorTransitionModel()
    env = PreacherCurlEnv(config)
    try:
        env.reset(seed=0, options={"payload_mass": config.payload_mass})
        if wide_contract(contract(env)) != original["contract"]:
            raise ValueError("Reference physical/input contract changed")
        metadata = sensor_contract(env)
        initial.configure_geometry(env)
    finally:
        env.close()
    initial.fit_normalization(split["train"])
    with (args.reference / "native_probes.pkl").open("rb") as stream:
        probes = copy.deepcopy(pickle.load(stream))
    lookup = {e["id"]: e for e in episodes}
    for probe in probes:
        probe["compact"] = lookup[probe["episode_id"]]["states"][probe["step"]].copy()
    # Verify the derived force input against native endpoint force at the same
    # state; no future outcomes or integration steps are needed for this check.
    kernel = NativeResponse(config)
    force_errors = []
    try:
        for probe in probes:
            kernel.restore(probe["state"])
            force = np.array([o.tendon_force for o in kernel.env._state_outputs()])
            with torch.no_grad():
                reconstructed = initial.tendon_forces(torch.tensor(probe["compact"])).numpy()
            force_errors.extend(reconstructed - force)
    finally:
        kernel.close()
    force_audit = {"samples": len(force_errors), "rmse_N": float(np.sqrt(np.mean(np.square(force_errors)))),
                   "max_abs_N": float(np.max(np.abs(force_errors)))}
    print(f"Input width={initial.net[0].in_features}, parameters={sum(p.numel() for p in initial.parameters())}; force audit={force_audit}", flush=True)
    np.savez_compressed(args.output_dir / "episodes.npz", **{
        f"e{i}_{key}": e[key] for i, e in enumerate(episodes) for key in ("states", "actions")})
    shutil.copy2(args.reference / "episodes.json", args.output_dir / "episodes.json")
    with (args.output_dir / "native_probes.pkl").open("wb") as stream:
        pickle.dump(probes, stream)
    summary = {"schema": "myo.dynamics_sensor_experiment.v1", "status": "training", "contract": metadata,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_implementation_sha256": hashlib.sha256(Path(__file__).with_name("train_dynamics_acceleration.py").read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference": str(args.reference), "architecture": [38, 1024, 1024, 512, 20],
        "trainable_parameters": sum(p.numel() for p in initial.parameters()),
        "original_dataset_sha256": hashlib.sha256((args.reference / "episodes.npz").read_bytes()).hexdigest(),
        "collection_physical_steps": 0, "force_sensor_audit": force_audit,
        "split_episodes": {name: [e["id"] for e in values] for name, values in split.items()},
        "evaluated_models": ["bptt_6", "bptt_velocity_selection"]}
    def report():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report()
    training = train(initial, split["train"], split["validation"], metadata, args)
    models = {}
    for name in summary["evaluated_models"]:
        models[name], _ = load_sensor_predictor(args.output_dir / f"{name}.pt", metadata)
        summary[name] = {"training": {**training, "best_update": training["selected_updates"][name]},
                         "test": evaluate(models[name], split["test"]),
                         "acceleration_test": acceleration_metrics(models[name], split["test"], config.control_dt)}
        report()
    summary["native_audit"] = native_audit(models, probes, config, args.output_dir)
    with np.load(args.reference / "jacobians.npz") as a, np.load(args.output_dir / "jacobians.npz") as b:
        summary["native_reference_identical"] = bool(np.array_equal(a["native"], b["native"]))
    summary["reference_acceleration_test"] = {}
    reference_test = [e for e in source if e["split"] == "test"]
    for name in ("bptt_6", "bptt_velocity_selection"):
        model, _ = load_wide_predictor(args.reference / f"{name}.pt", original["contract"])
        summary["reference_acceleration_test"][name] = acceleration_metrics(model, reference_test, config.control_dt)
    summary["status"], summary["elapsed_s"] = "complete", time.monotonic() - started
    summary["limitations"] = ["Numerical sensory proxies, not a biological receptor model",
        "Force is a derived feature; fiber velocity adds one causal interval of history",
        "Known geometry/tendon law reconstructs force features during neural rollout, not native motion",
        "Same loss, update budget and physical targets; one initialization, not convergence",
        "No new actor or hierarchy trained"]
    report()
    print(json.dumps({name: summary[name]["test"] for name in models}, indent=2), flush=True)


if __name__ == "__main__":
    main()
