"""Independently verify frozen reservoirs, normalization and reported losses."""
import argparse
import json
from pathlib import Path

import torch

from myo.train_vertical_history import normalized_errors, read_source
from myo.train_vertical_reservoir import (
    ReservoirPredictor, load_reservoir, reservoir_contract, with_reservoir,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_reservoir"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for training to complete before auditing artifacts")
    torch.set_num_threads(2)
    checked = []
    for variant, record in summary["variants"].items():
        source = Path(record["source"])
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        episodes = read_source(source, env)
        for branch, result in record["branches"].items():
            architecture = result["contract"]["architecture"]
            regenerated = ReservoirPredictor(**architecture)
            regenerated.fit_normalization([e for e in episodes if e["split"] == "train"])
            augmented = with_reservoir(episodes, regenerated)
            for stage in ("one_step", "bptt"):
                path = args.run/f"variant_{variant}"/branch/f"{stage}.pt"
                model, _ = load_reservoir(path, reservoir_contract(env, architecture))
                for name in ("input_mean", "input_scale", "delta_mean", "delta_scale", "W_res", "W_in"):
                    if not torch.equal(getattr(model, name), getattr(regenerated, name)):
                        raise AssertionError((str(path), name))
                errors = normalized_errors(model, augmented)
                if errors != result["stages"][stage]["normalized"]:
                    raise AssertionError(f"Reported errors differ: {path}")
                checked.append(str(path))
        print(f"variant {variant}: all checkpoints, frozen matrices and errors verified", flush=True)
    report = {"status": "passed", "strict_checkpoints": len(checked),
        "train_only_normalization": True, "all_reservoir_matrices_equal_initialization": True,
        "all_normalized_errors_reproduced": True, "paths": checked}
    (args.run/"artifact_check.json").write_text(json.dumps(report, indent=2)+'\n')
    print({key: value for key, value in report.items() if key != "paths"})


if __name__ == "__main__":
    main()
