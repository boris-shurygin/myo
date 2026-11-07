"""Audit local-sensor predictors and compare the legacy model in the same units."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_tissue_predictor import (
    TissuePredictor, derivatives, evaluate_tissue, load_tissue, native_to_tissues,
    pooled_errors, residuals, tissue_contract, tissue_episodes,
)
from myo.train_vertical_history import history_contract, load_history, read_source, with_history
from myo.train_vertical_predictor import windows
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


@torch.no_grad()
def project_legacy(legacy, initial, actions, sensory_initial, config, reference):
    """Project only its own forecast; true intermediate sensations never enter."""
    converted = native_to_tissues(legacy.rollout(initial, actions), config)
    previous = sensory_initial
    result = []
    for t in range(converted.shape[1]):
        next_base = converted[:, t]
        aux = derivatives(next_base, previous,
                          previous[:, reference.base_size:reference.base_size+4], config.dt)
        previous = torch.cat((next_base, aux), dim=-1)
        result.append(previous)
    return torch.stack(result, dim=1)


@torch.no_grad()
def compare_legacy(folder, env, raw, sensory, reference):
    legacy, _ = load_history(folder/"sensory_history/bptt.pt", history_contract(env, 3))
    old = with_history(raw, env.config.physical_size, 3)
    result = {"pooled": {}}
    for split in ("train", "validation", "test"):
        initial, actions, _ = windows([e for e in old if e["split"] == split], 6)
        si, sa, targets = windows([e for e in sensory if e["split"] == split], 6)
        assert torch.equal(actions, sa)
        prediction = project_legacy(legacy, initial, actions, si, env.config, reference)
        base, aux = residuals(reference, prediction, targets)
        result["pooled"][split] = {"base_mse": float(base.double().square().mean()),
                                    "aux_mse": float(aux.double().square().mean())}
    initial, actions, _ = windows([e for e in old if e["split"] == "test"], 20, stride=5)
    si, sa, targets = windows([e for e in sensory if e["split"] == "test"], 20, stride=5)
    assert torch.equal(actions, sa)
    prediction = project_legacy(legacy, initial, actions, si, env.config, reference)
    base, aux = residuals(reference, prediction, targets)
    result["test"] = {}
    for h in (1, 3, 6, 20):
        error = reference.sensations(prediction[:, h-1])-reference.sensations(targets[:, h-1])
        result["test"][str(h*20)] = {"samples": len(initial),
            "channel_rmse": error.double().square().mean(0).sqrt().tolist(),
            "base_mse": float(base[:, h-1].double().square().mean()),
            "rate_rmse_normalized": float(aux[:, h-1, :4].double().square().mean().sqrt()),
            "acceleration_rmse_normalized": float(aux[:, h-1, 4:].double().square().mean().sqrt())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_predictor"))
    parser.add_argument("--legacy", type=Path, default=Path("runs/vertical_sensory_history_60ms"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for training to complete")
    legacy_summary = json.loads((args.legacy/"summary.json").read_text())
    torch.set_num_threads(2)
    result = {"status": "passed", "checked_checkpoints": 0, "variants": {},
              "units": "base strains dimensionless; strain rates 1/s; strain accelerations 1/s^2"}
    for variant, record in summary["variants"].items():
        source = Path(record["source"])
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        raw = read_source(source, env)
        sensory = tissue_episodes(raw, env.config)
        if record["data_sha256"] != legacy_summary["variants"][variant]["data_sha256"]:
            raise ValueError("Legacy comparison data differ")
        common_scale = None
        for name, branch in record["branches"].items():
            architecture = branch["contract"]["architecture"]
            model = TissuePredictor(**architecture)
            model.fit_normalization([e for e in sensory if e["split"] == "train"])
            expected = tissue_contract(env, model, branch["contract"]["aux_weight"])
            for stage in ("one_step", "bptt"):
                selected, _ = load_tissue(args.run/f"variant_{variant}"/name/f"{stage}.pt", expected)
                for buffer in ("input_mean", "input_scale", "delta_mean", "delta_scale", "aux_scale"):
                    assert torch.equal(getattr(model, buffer), getattr(selected, buffer)), buffer
                assert pooled_errors(selected, sensory, expected["aux_weight"]) == branch["stages"][stage]["pooled"]
                assert evaluate_tissue(selected, [e for e in sensory if e["split"] == "test"]) == branch["stages"][stage]["test"]
                result["checked_checkpoints"] += 1
            if common_scale is not None:
                assert torch.equal(model.delta_scale, common_scale[0])
                assert torch.equal(model.aux_scale, common_scale[1])
            common_scale = model.delta_scale, model.aux_scale
        result["variants"][variant] = {"legacy": compare_legacy(
            args.legacy/f"variant_{variant}", env, raw, sensory, model)}
        print(f"variant {variant}: checkpoints and errors verified; legacy projected into local sensations", flush=True)
    (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
