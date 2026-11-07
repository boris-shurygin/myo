"""Measure tissue-predictor capacity and validation plateaus on fixed episodes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .train_tissue_predictor import (
    TissuePredictor, evaluate_tissue, pooled_errors, residuals, save_tissue,
    tissue_contract, tissue_episodes,
)
from .train_vertical_history import read_source
from .train_vertical_predictor import windows
from .vertical_muscles import VerticalConfig, VerticalMuscles


class CapacityPredictor(TissuePredictor):
    def __init__(self, physical_size, queue_steps, dt=.02, history_steps=3,
                 derivative_inputs=True, hidden_sizes=(128, 128)):
        widths = list(hidden_sizes)
        super().__init__(physical_size, queue_steps, dt, history_steps, derivative_inputs, widths[0])
        if widths != [widths[0], widths[0]]:
            common = self.net
            layers, size = [], self.net[0].in_features
            for width in widths:
                layers.extend((nn.Linear(size, width), nn.Tanh()))
                size = width
            layers.append(nn.Linear(size, physical_size))
            self.net = nn.Sequential(*layers)
            # Match the first two hidden layers of the equally wide control.
            with torch.no_grad():
                for index in (0, 2):
                    if self.net[index].weight.shape == common[index].weight.shape:
                        self.net[index].weight.copy_(common[index].weight)
                        self.net[index].bias.copy_(common[index].bias)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.architecture.pop("hidden")
        self.architecture["hidden_sizes"] = widths


def capacity_contract(env, model):
    return {**tissue_contract(env, model, 1.), "schema": "myo.tissue_capacity.v1",
            "capacity_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_capacity(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Capacity weights differ from metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Capacity contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Capacity architecture differs from metadata")
    model = CapacityPredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


@torch.no_grad()
def short_metrics(model, data):
    initial, actions, target = data
    prediction = model.rollout(initial, actions[:, :6])
    base, aux = residuals(model, prediction, target[:, :6])
    return {"score_one_step": float(aux[:, 0, :4].double().square().mean().sqrt()),
            "score_bptt": float(np.mean([float(aux[:, h-1, :4].double().square().mean().sqrt())
                                         for h in (1, 3, 6)])),
            "base_mse": float(base.double().square().mean()),
            "aux_mse": float(aux.double().square().mean())}


def train(env, episodes, args, output, widths):
    torch.manual_seed(0)
    model = CapacityPredictor(6 if env.config.activation_dynamic else 4, env.config.queue_steps,
                              dt=env.config.dt, hidden_sizes=widths)
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    model.fit_normalization(split["train"])
    contract = capacity_contract(env, model)
    validation = windows(split["validation"], 20, stride=5)
    # A fixed full-trajectory-derived subset, independent of training minibatch RNG.
    train_full = windows(split["train"], 20, stride=5)
    generator = torch.Generator().manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    result = {"contract": contract, "parameters": sum(p.numel() for p in model.parameters()), "stages": {}}
    for stage, horizon, maximum in (("one_step", 1, args.one_step_updates), ("bptt", 6, args.max_bptt_updates)):
        initial, actions, targets = windows(split["train"], horizon)
        history, best, significant_best = [], float("inf"), float("inf")
        last_significant, reason = 0, "update_budget"
        started = time.monotonic()
        for update in range(1, maximum+1):
            ids = torch.randint(len(initial), (128,), generator=generator)
            prediction = model.rollout(initial[ids], actions[ids])
            base, aux = residuals(model, prediction, targets[ids])
            loss = base.square().mean()+aux.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite capacity loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if update == 1 or update % 200 == 0 or update == maximum:
                val, fitted = short_metrics(model, validation), short_metrics(model, train_full)
                score = val["score_one_step"] if stage == "one_step" else val["score_bptt"]
                if score < best:
                    best = score
                    save_tissue(output/f"{stage}.pt", model, contract, update, stage)
                if score < significant_best*(1-args.min_relative_improvement):
                    significant_best, last_significant = score, update
                history.append({"update": update, "sampled_windows": update*128,
                    "predicted_transitions": update*128*horizon, "minibatch_loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm), "train_fixed": fitted,
                    "validation": val, "validation_score": score, "best_score": best})
                print(f"v{env.config.variant} {widths} {stage} {update}/{maximum} val={score:.6g} "
                      f"train={fitted['base_mse']+fitted['aux_mse']:.6g}", flush=True)
                if stage == "bptt" and update == 2000:
                    budget_model, budget_metadata = load_capacity(output/"bptt.pt", contract)
                    save_tissue(output/"budget_2000.pt", budget_model, contract,
                                budget_metadata["update"], "bptt_budget_2000")
                if stage == "bptt" and update >= 2000 and update-last_significant >= args.plateau_patience:
                    reason = "validation_plateau"
                    break
        selected, metadata = load_capacity(output/f"{stage}.pt", contract)
        result["stages"][stage] = {"updates": update, "unique_windows": len(initial),
            "sampled_windows": update*128, "predicted_transitions": update*128*horizon,
            "stop_reason": reason, "last_significant_update": last_significant,
            "best_update": metadata["update"], "elapsed_s": time.monotonic()-started, "history": history,
            "validation": evaluate_tissue(selected, split["validation"]),
            "test": evaluate_tissue(selected, split["test"]), "pooled": pooled_errors(selected, episodes, 1.)}
        model = copy.deepcopy(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tissue_capacity"))
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--max-bptt-updates", type=int, default=6000)
    parser.add_argument("--plateau-patience", type=int, default=1600)
    parser.add_argument("--min-relative-improvement", type=float, default=.01)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema": "myo.tissue_capacity_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "scope": "derivative-input + derivative-loss branch, fixed local sensors/data; no actor",
        "architectures": {"small": [128, 128], "wide": [512, 512], "deep": [512, 512, 256]}, "variants": {}}
    started = time.monotonic()

    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    report()
    for variant in range(1, 5):
        source = args.reference/f"variant_{variant}"
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        episodes = tissue_episodes(read_source(source, env), env.config)
        record = {"source": str(source), "data_sha256": hashlib.sha256((source/"episodes.npz").read_bytes()).hexdigest(),
                  "models": {}}
        summary["variants"][str(variant)] = record
        for name, widths in summary["architectures"].items():
            output = args.output_dir/f"variant_{variant}"/name
            output.mkdir(parents=True)
            record["models"][name] = train(env, episodes, args, output, widths)
            report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
