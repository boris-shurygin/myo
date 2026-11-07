"""Compare vertical predictors with current sensations and a causal 200 ms window."""
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

from .train_vertical_predictor import (
    VerticalPredictor, evaluate, load_predictor, predictor_contract, windows,
)
from .vertical_muscles import VerticalConfig, VerticalMuscles


def with_history(episodes, physical_size, history_steps):
    """Append past physical frames, newest first; pad only with the episode reset."""
    result = []
    for episode in episodes:
        states = episode["states"]
        indices = np.arange(len(states))
        past = [states[np.maximum(indices-age, 0), :physical_size]
                for age in range(1, history_steps+1)]
        result.append({**episode, "states": np.concatenate([states, *past], axis=-1)})
    return result


class HistoryPredictor(VerticalPredictor):
    def __init__(self, physical_size, queue_steps, history_steps=10, hidden=128):
        super().__init__(physical_size, queue_steps, hidden)
        self.history_steps = history_steps
        self.base_size = physical_size+queue_steps*2
        if history_steps:
            # Preserve the baseline's exact initial function and shared weights.
            first = self.net[0]
            expanded = nn.Linear(first.in_features+physical_size*history_steps, hidden)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :first.in_features].copy_(first.weight)
                expanded.bias.copy_(first.bias)
            self.net[0] = expanded
            self.input_mean = torch.zeros(expanded.in_features)
            self.input_scale = torch.ones(expanded.in_features)

    def inputs(self, state, action):
        current = super().inputs(state, action)
        return torch.cat((current, state[..., self.base_size:]), dim=-1)

    def forward(self, state, action):
        if not self.history_steps:
            return super().forward(state, action)
        x = (self.inputs(state, action)-self.input_mean)/self.input_scale
        physical = state[..., :self.physical_size]+self.delta_mean+self.delta_scale*self.net(x)
        queue = state[..., self.physical_size:self.base_size].reshape(
            *state.shape[:-1], self.queue_steps, 2)
        if self.queue_steps:
            queue = torch.cat((queue[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        past = state[..., self.base_size:].reshape(
            *state.shape[:-1], self.history_steps, self.physical_size)
        past = torch.cat((state[..., :self.physical_size].unsqueeze(-2), past[..., :-1, :]), dim=-2)
        return torch.cat((physical, queue.flatten(-2), past.flatten(-2)), dim=-1)


def history_contract(env, history_steps):
    base = predictor_contract(env)
    past = [f"history/{age}/{name}" for age in range(1, history_steps+1)
            for name in base["physical_names"]]
    return {**base, "schema": "myo.vertical_sensory_history.v1",
            "history_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "history_steps": history_steps, "history_duration_s": history_steps*env.config.dt,
            "history_reset": "repeat initial physical frame within the same episode",
            "state_names": [*base["state_names"], *past],
            "neural_input_names": [*base["neural_input_names"], *past]}


def save_history(path, model, contract, update, stage):
    architecture = {"physical_size": model.physical_size, "queue_steps": model.queue_steps,
                    "history_steps": model.history_steps, "hidden": model.hidden}
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, path)
    metadata = {**contract, "architecture": architecture, "update": update, "stage": stage,
                "seed": 0, "normalization": "train episodes only",
                "target": "physical increments; exact transport and physical-history shifts",
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')


def load_history(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("History checkpoint weights differ from metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"History predictor contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("History architecture differs from metadata")
    model = HistoryPredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


@torch.no_grad()
def normalized_errors(model, episodes):
    result = {}
    for split in ("train", "validation", "test"):
        initial, actions, targets = windows([e for e in episodes if e["split"] == split], 6)
        prediction = model.rollout(initial, actions)[..., :model.physical_size]
        residual = (prediction-targets[..., :model.physical_size])/model.delta_scale
        mse = residual.double().square().mean((0, 1))
        result[split] = {"windows": len(initial), "loss_mse": float(mse.mean()),
                         "channel_normalized_rmse": mse.sqrt().tolist(),
                         "channel_20ms_normalized_rmse": residual[:, 0].double().square().mean(0).sqrt().tolist()}
    return result


def train_comparison_branch(env, episodes, args, output, history_steps):
    torch.manual_seed(0)
    model = HistoryPredictor(env.config.physical_size, env.config.queue_steps, history_steps)
    episodes = with_history(episodes, model.physical_size, history_steps)
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    model.fit_normalization(split["train"])
    metadata = history_contract(env, history_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(0)
    result = {"contract": metadata, "parameters": sum(p.numel() for p in model.parameters()),
              "input_size": model.net[0].in_features, "delta_scale": model.delta_scale.tolist(), "stages": {}}
    for stage, horizon, updates in (("one_step", 1, args.one_step_updates), ("bptt", 6, args.bptt_updates)):
        initial, actions, targets = windows(split["train"], horizon)
        history, best, best_update = [], float("inf"), 0
        started = time.monotonic()
        for update in range(1, updates+1):
            ids = torch.randint(len(initial), (128,), generator=generator)
            prediction = model.rollout(initial[ids], actions[ids])
            residual = (prediction[..., :model.physical_size]-targets[ids, :, :model.physical_size])/model.delta_scale
            loss = residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite sensory history loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if update == 1 or update % 200 == 0 or update == updates:
                metrics = evaluate(model, split["validation"])
                score = metrics["20"]["velocity_rmse_m_s"] if stage == "one_step" else float(np.mean([
                    metrics[str(h)]["velocity_rmse_m_s"] for h in (20, 60, 120)]))
                history.append({"update": update, "training_loss": float(loss.detach()),
                                "gradient_norm_before_clip": float(norm), "validation_score": score})
                if score < best:
                    best, best_update = score, update
                    save_history(output/f"{stage}.pt", model, metadata, update, stage)
                print(f"v{env.config.variant} history={history_steps} {stage} {update}/{updates} "
                      f"loss={float(loss.detach()):.5g} validation={score:.5g}", flush=True)
        selected, _ = load_history(output/f"{stage}.pt", metadata)
        result["stages"][stage] = {"updates": updates, "best_update": best_update,
            "elapsed_s": time.monotonic()-started, "history": history,
            "validation": evaluate(selected, split["validation"]),
            "test": evaluate(selected, split["test"]),
            "normalized": normalized_errors(selected, episodes)}
        model = copy.deepcopy(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    return result


def read_source(folder, env):
    # Reject stale physics/contracts without changing or recollecting source data.
    load_predictor(folder/"bptt.pt", predictor_contract(env))
    records = json.loads((folder/"episodes.json").read_text())
    with np.load(folder/"episodes.npz") as arrays:
        return [{**record, "states": arrays[f"e{i}_states"], "actions": arrays[f"e{i}_actions"]}
                for i, record in enumerate(records)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_sensory_history"))
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--bptt-updates", type=int, default=2000)
    args = parser.parse_args()
    if args.history_steps < 1 or min(args.one_step_updates, args.bptt_updates) < 1:
        parser.error("History length and update counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema": "myo.vertical_history_comparison.v1", "status": "running",
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "predictor only; unchanged physical features, data, loss and selection; no actor",
        "variants": {}}
    started = time.monotonic()

    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    report()
    for variant in range(1, 5):
        source = args.reference/f"variant_{variant}"
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        episodes = read_source(source, env)
        record = {"source": str(source), "data_sha256": hashlib.sha256((source/"episodes.npz").read_bytes()).hexdigest(),
                  "splits": {s: [e["id"] for e in episodes if e["split"] == s]
                             for s in ("train", "validation", "test")}, "branches": {}}
        for name, count in (("current", 0), ("sensory_history", args.history_steps)):
            output = args.output_dir/f"variant_{variant}"/name
            output.mkdir(parents=True)
            record["branches"][name] = train_comparison_branch(env, episodes, args, output, count)
        a, b = record["branches"].values()
        assert a["delta_scale"] == b["delta_scale"]
        control, _ = load_history(args.output_dir/f"variant_{variant}/current/bptt.pt", history_contract(env, 0))
        original, _ = load_predictor(source/"bptt.pt", predictor_contract(env))
        record["control_weights_equal_reference"] = all(torch.equal(value, original.state_dict()[key])
                                                        for key, value in control.state_dict().items())
        summary["variants"][str(variant)] = record
        report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
