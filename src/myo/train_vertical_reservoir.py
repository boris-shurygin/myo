"""Frozen recurrent sensory memory for the existing vertical prediction task."""
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

from .train_vertical_history import normalized_errors, read_source
from .train_vertical_predictor import VerticalPredictor, evaluate, predictor_contract, windows
from .vertical_muscles import VerticalConfig, VerticalMuscles


class ReservoirPredictor(VerticalPredictor):
    def __init__(self, physical_size, queue_steps, reservoir_size=256, hidden=128,
                 readout="mlp", seed=0, spectral_radius=.8, density=.1,
                 input_density=.1, warmup_steps=20, initialize=True):
        super().__init__(physical_size, queue_steps, hidden)
        self.base_size = physical_size+2*queue_steps
        self.reservoir_size = reservoir_size
        self.architecture = dict(physical_size=physical_size, queue_steps=queue_steps,
            reservoir_size=reservoir_size, hidden=hidden, readout=readout, seed=seed,
            spectral_radius=spectral_radius, density=density, input_density=input_density,
            warmup_steps=warmup_steps)
        self.register_buffer("W_res", torch.zeros(reservoir_size, reservoir_size))
        self.register_buffer("W_in", torch.zeros(reservoir_size, physical_size))
        if initialize:
            rng = np.random.default_rng(seed)
            recurrent = rng.normal(0., .5, (reservoir_size, reservoir_size))
            recurrent *= rng.random(recurrent.shape) < density
            radius = np.abs(np.linalg.eigvals(recurrent)).max()
            if radius == 0:
                raise ValueError("Degenerate reservoir; increase size or density")
            recurrent *= spectral_radius/radius
            incoming = rng.normal(0., .5, (reservoir_size, physical_size))
            incoming *= rng.random(incoming.shape) < input_density
            self.W_res.copy_(torch.tensor(recurrent, dtype=torch.float32))
            self.W_in.copy_(torch.tensor(incoming, dtype=torch.float32))
        input_size = physical_size+2+reservoir_size
        if readout == "mlp":
            first = self.net[0]
            expanded = nn.Linear(input_size, hidden)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :first.in_features].copy_(first.weight)
                expanded.bias.copy_(first.bias)
            self.net[0] = expanded
        elif readout == "linear":
            self.net = nn.Linear(input_size, physical_size)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            raise ValueError("Readout must be mlp or linear")

    def memory_step(self, memory, physical):
        sensory = (physical-self.input_mean[:self.physical_size])/self.input_scale[:self.physical_size]
        return torch.tanh(memory@self.W_res.T+sensory@self.W_in.T)

    def reset_memory(self, physical):
        memory = physical.new_zeros(*physical.shape[:-1], self.reservoir_size)
        for _ in range(self.architecture["warmup_steps"]):
            memory = self.memory_step(memory, physical)
        return memory

    def forward(self, state, action):
        current = (self.inputs(state, action)-self.input_mean)/self.input_scale
        memory = state[..., self.base_size:]
        physical = state[..., :self.physical_size]+self.delta_mean+self.delta_scale*self.net(
            torch.cat((current, memory), dim=-1))
        queue = state[..., self.physical_size:self.base_size].reshape(
            *state.shape[:-1], self.queue_steps, 2)
        if self.queue_steps:
            queue = torch.cat((queue[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        # The next memory receives the model's own forecast, never a future label.
        memory = self.memory_step(memory, physical)
        return torch.cat((physical, queue.flatten(-2), memory), dim=-1)


@torch.no_grad()
def with_reservoir(episodes, model):
    """Replay each episode causally once; no hidden-state carry across resets."""
    result = []
    for episode in episodes:
        states = torch.tensor(episode["states"], dtype=torch.float32)
        memory = model.reset_memory(states[0, :model.physical_size])
        memories = [memory]
        for physical in states[1:, :model.physical_size]:
            memory = model.memory_step(memory, physical)
            memories.append(memory)
        result.append({**episode, "states": torch.cat((states, torch.stack(memories)), dim=-1).numpy()})
    return result


def reservoir_contract(env, architecture):
    base = predictor_contract(env)
    memory = [f"reservoir/{i}" for i in range(architecture["reservoir_size"])]
    return {**base, "schema": "myo.vertical_reservoir.v1",
        "reservoir_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "architecture": architecture, "reservoir_input_names": base["physical_names"],
        "reservoir_update": "tanh(W_res @ memory + W_in @ normalized_current_physical)",
        "reservoir_state_timing": "memory includes sensory frame at current time",
        "reservoir_reset": "zero then warmup_steps repetitions of initial normalized physical frame",
        "reservoir_weights": "frozen buffers; gradients through inputs and recurrence retained",
        "state_names": [*base["state_names"], *memory],
        "neural_input_names": [*base["neural_input_names"], *memory],
        "sources": ["https://doi.org/10.1073/pnas.2522094123",
                    "https://github.com/GazzolaLab/SoftArmRC/blob/main/core/core_functions.py"]}


def save_reservoir(path, model, contract, update, stage):
    torch.save({"architecture": model.architecture, "state_dict": model.state_dict()}, path)
    metadata = {**contract, "update": update, "stage": stage, "seed": 0,
                "normalization": "train episodes only",
                "target": "physical state increments; no reservoir or queue loss",
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')


def load_reservoir(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Reservoir weights differ from metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Reservoir contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Reservoir architecture differs from metadata")
    model = ReservoirPredictor(**saved["architecture"], initialize=False)
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


def train_branch(env, episodes, args, output, readout):
    torch.manual_seed(0)
    model = ReservoirPredictor(env.config.physical_size, env.config.queue_steps,
                              reservoir_size=args.reservoir_size, readout=readout)
    model.fit_normalization([e for e in episodes if e["split"] == "train"])
    episodes = with_reservoir(episodes, model)
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    contract = reservoir_contract(env, model.architecture)
    frozen = {name: value.clone() for name, value in model.named_buffers() if name.startswith("W_")}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(0)
    result = {"contract": contract, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "frozen_reservoir_values": model.W_res.numel()+model.W_in.numel(),
              "delta_scale": model.delta_scale.tolist(), "stages": {}}
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
                raise RuntimeError("Non-finite reservoir loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if update == 1 or update % 200 == 0 or update == updates:
                metrics = evaluate(model, split["validation"])
                score = metrics["20"]["velocity_rmse_m_s"] if stage == "one_step" else float(np.mean([
                    metrics[str(h)]["velocity_rmse_m_s"] for h in (20, 60, 120)]))
                history.append({"update": update, "loss": float(loss.detach()),
                                "gradient_norm_before_clip": float(norm), "validation_score": score})
                if score < best:
                    best, best_update = score, update
                    save_reservoir(output/f"{stage}.pt", model, contract, update, stage)
                print(f"v{env.config.variant} reservoir={readout} {stage} {update}/{updates} "
                      f"loss={float(loss.detach()):.5g} validation={score:.5g}", flush=True)
        selected, _ = load_reservoir(output/f"{stage}.pt", contract)
        assert all(torch.equal(getattr(selected, name), value) for name, value in frozen.items())
        result["stages"][stage] = {"updates": updates, "best_update": best_update,
            "elapsed_s": time.monotonic()-started, "history": history,
            "validation": evaluate(selected, split["validation"]), "test": evaluate(selected, split["test"]),
            "normalized": normalized_errors(selected, episodes)}
        model = copy.deepcopy(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    result["reservoir_weights_unchanged"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--history-reference", type=Path, default=Path("runs/vertical_sensory_history"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_reservoir"))
    parser.add_argument("--reservoir-size", type=int, default=256)
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--bptt-updates", type=int, default=2000)
    args = parser.parse_args()
    if args.reservoir_size < 16 or min(args.one_step_updates, args.bptt_updates) < 1:
        parser.error("Reservoir size must be >=16; update counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    previous = json.loads((args.history_reference/"summary.json").read_text())
    if previous["status"] != "complete":
        raise ValueError("History reference is not complete")
    summary = {"schema": "myo.vertical_reservoir_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "scope": "fixed sensory reservoir, supervised predictor only; no actor or critic",
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
        digest = hashlib.sha256((source/"episodes.npz").read_bytes()).hexdigest()
        old = previous["variants"][str(variant)]
        if old["data_sha256"] != digest:
            raise ValueError("History comparison uses different episode data")
        # Strictly reload references instead of trusting detached summary metrics.
        from .train_vertical_history import history_contract, load_history
        for branch, result in old["branches"].items():
            load_history(args.history_reference/f"variant_{variant}"/branch/"bptt.pt",
                         history_contract(env, result["contract"]["history_steps"]))
        record = {"data_sha256": digest, "source": str(source), "references": old["branches"],
            "splits": {s: [e["id"] for e in episodes if e["split"] == s]
                       for s in ("train", "validation", "test")}, "branches": {}}
        for readout in ("mlp", "linear"):
            output = args.output_dir/f"variant_{variant}"/readout
            output.mkdir(parents=True)
            record["branches"][readout] = train_branch(env, episodes, args, output, readout)
            assert record["branches"][readout]["delta_scale"] == old["branches"]["current"]["delta_scale"]
        summary["variants"][str(variant)] = record
        report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
