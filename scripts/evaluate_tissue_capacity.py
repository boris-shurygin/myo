"""Audit capacity/plateau artifacts and measure their native action Jacobians."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_tissue_quality import native_sensitivities
from myo.train_tissue_capacity import CapacityPredictor, capacity_contract, load_capacity
from myo.train_tissue_predictor import (
    TissuePredictor, evaluate_tissue, load_tissue, pooled_errors, tissue_contract, tissue_episodes,
)
from myo.train_vertical_history import read_source
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def audit_stop(stage, arguments):
    significant, last = float("inf"), 0
    first_plateau = None
    for row in stage["history"]:
        if row["validation_score"] < significant*(1-arguments["min_relative_improvement"]):
            significant, last = row["validation_score"], row["update"]
        if row["update"] >= 2000 and row["update"]-last >= arguments["plateau_patience"]:
            first_plateau = row["update"]
            break
    assert stage["last_significant_update"] == last
    if stage["stop_reason"] == "validation_plateau":
        assert first_plateau == stage["updates"]
    else:
        assert first_plateau is None and stage["updates"] == arguments["max_bptt_updates"]
    best = min(row["validation_score"] for row in stage["history"])
    near = next(row["update"] for row in stage["history"] if row["validation_score"] <= 1.05*best)
    selected_row = min(stage["history"], key=lambda row: row["validation_score"])
    return {"stop": stage["updates"], "reason": stage["stop_reason"],
            "best_update": stage["best_update"], "within_5pct_of_best_update": near,
            "sampled_windows_including_one_step": (arguments["one_step_updates"]+stage["updates"])*128,
            "predicted_transitions_including_one_step": arguments["one_step_updates"]*128+stage["updates"]*128*6,
            "selected_train_fixed": selected_row["train_fixed"], "selected_validation": selected_row["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_capacity"))
    parser.add_argument("--original", type=Path, default=Path("runs/tissue_predictor"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/"summary.json").read_text())
    original = json.loads((args.original/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for capacity training to complete")
    result = {"status": "running", "strict_checkpoints": 0, "variants": {}}
    for variant, record in summary["variants"].items():
        source = Path(record["source"])
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        raw = read_source(source, env)
        episodes = tissue_episodes(raw, env.config)
        assert record["data_sha256"] == original["variants"][variant]["data_sha256"]
        models, rows = {}, {}
        for name, branch in record["models"].items():
            prototype = CapacityPredictor(**branch["contract"]["architecture"])
            prototype.fit_normalization([e for e in episodes if e["split"] == "train"])
            expected = capacity_contract(env, prototype)
            for stage in ("one_step", "bptt", "budget_2000"):
                model, _ = load_capacity(args.run/f"variant_{variant}"/name/f"{stage}.pt", expected)
                for buffer in ("input_mean", "input_scale", "delta_mean", "delta_scale", "aux_scale"):
                    assert torch.equal(getattr(model, buffer), getattr(prototype, buffer))
                pooled = pooled_errors(model, episodes, 1.)
                test = evaluate_tissue(model, [e for e in episodes if e["split"] == "test"])
                if stage == "budget_2000":
                    budget = {"test": test, "pooled": pooled}
                    if name == "small":
                        old_branch = original["variants"][variant]["branches"]["derivative_loss"]
                        old_prototype = TissuePredictor(**old_branch["contract"]["architecture"])
                        old, _ = load_tissue(args.original/f"variant_{variant}/derivative_loss/bptt.pt",
                                            tissue_contract(env, old_prototype, 1.))
                        assert all(torch.equal(value, old.state_dict()[key]) for key, value in model.state_dict().items())
                else:
                    assert pooled == branch["stages"][stage]["pooled"]
                    assert test == branch["stages"][stage]["test"]
                    if stage == "bptt":
                        models[name] = model
                result["strict_checkpoints"] += 1
            rows[name] = {"plateau": audit_stop(branch["stages"]["bptt"], summary["arguments"]),
                          "budget_2000": budget}
        jacobians, arrays = native_sensitivities(env, raw, episodes, models)
        with np.load(args.original/f"quality_jacobians_{variant}.npz") as previous:
            np.testing.assert_array_equal(arrays["native"], previous["native"])
        np.savez_compressed(args.run/f"jacobians_{variant}.npz", **arrays)
        for name in rows:
            rows[name]["sensitivities"] = jacobians[name]
        result["variants"][variant] = rows
        (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(f"variant {variant}: checkpoints, original 2000-update control, stop criterion and Jacobians verified", flush=True)
    result["status"] = "passed"
    (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
