"""Predict local fiber/tendon strains and their causal time derivatives."""
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

from .train_vertical_history import HistoryPredictor, read_source
from .train_vertical_predictor import predictor_contract, windows
from .vertical_muscles import VerticalConfig, VerticalMuscles


def sensor_names(dynamic_activation):
    strain = [f"{tissue}_strain/{muscle}" for tissue in ("fiber", "tendon")
              for muscle in ("upper", "lower")]
    base = strain+(["activation/upper", "activation/lower"] if dynamic_activation else [])
    return base, [f"rate_per_s/{name}" for name in strain]+[f"acceleration_per_s2/{name}" for name in strain]


def native_to_tissues(state, config):
    """Named sensor boundary: geometry only here; q/qvel never enter the predictor."""
    path = config.half_span_m+torch.stack((-state[..., 0], state[..., 0]), dim=-1)
    fiber = state[..., 2:4] if config.elastic else path-config.tendon_slack_m
    tendon = (path-fiber)/config.tendon_slack_m-1 if config.elastic else torch.zeros_like(fiber)
    base = [fiber/config.fiber_optimal_m-1, tendon]
    if config.activation_dynamic:
        base.append(state[..., 4:6])
    queue = state[..., config.physical_size:config.physical_size+2*config.queue_steps]
    return torch.cat([*base, queue], dim=-1)


def derivatives(following, current, previous_rates, dt):
    rate = (following[..., :4]-current[..., :4])/dt
    return torch.cat((rate, (rate-previous_rates)/dt), dim=-1)


def tissue_episodes(episodes, config, history_steps=3):
    base_size = 6 if config.activation_dynamic else 4
    result = []
    for episode in episodes:
        converted = native_to_tissues(torch.tensor(episode["states"]), config)
        aux = [torch.zeros(8)]
        for t in range(1, len(converted)):
            aux.append(derivatives(converted[t], converted[t-1], aux[-1][:4], config.dt))
        sensory = torch.cat((converted[:, :base_size], torch.stack(aux)), dim=-1)
        indices = torch.arange(len(converted))
        past = [sensory[(indices-age).clamp_min(0)] for age in range(1, history_steps+1)]
        states = torch.cat((converted, torch.stack(aux), *past), dim=-1)
        result.append({**episode, "states": states.numpy()})
    return result


class TissuePredictor(HistoryPredictor):
    def __init__(self, physical_size, queue_steps, dt=.02, history_steps=3,
                 derivative_inputs=True, hidden=128):
        super().__init__(physical_size, queue_steps, history_steps, hidden)
        self.dt, self.derivative_inputs = dt, derivative_inputs
        self.sensory_size = physical_size+8
        self.architecture = dict(physical_size=physical_size, queue_steps=queue_steps, dt=dt,
            history_steps=history_steps, derivative_inputs=derivative_inputs, hidden=hidden)
        if derivative_inputs:
            first = self.net[0]
            expanded = nn.Linear((physical_size+8)*(history_steps+1)+2, hidden)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :physical_size+2].copy_(first.weight[:, :physical_size+2])
                for age in range(history_steps):
                    old = physical_size+2+age*physical_size
                    new = physical_size+2+8+age*self.sensory_size
                    expanded.weight[:, new:new+physical_size].copy_(first.weight[:, old:old+physical_size])
                expanded.bias.copy_(first.bias)
            self.net[0] = expanded
            self.input_mean = torch.zeros(expanded.in_features)
            self.input_scale = torch.ones(expanded.in_features)
        self.register_buffer("aux_scale", torch.ones(8))

    def sensations(self, state):
        return torch.cat((state[..., :self.physical_size], state[..., self.base_size:self.base_size+8]), dim=-1)

    def inputs(self, state, action):
        delivered = state[..., self.physical_size:self.physical_size+2] if self.queue_steps else action
        past = state[..., self.base_size+8:].reshape(*state.shape[:-1], self.history_steps, self.sensory_size)
        if self.derivative_inputs:
            return torch.cat((state[..., :self.physical_size], delivered,
                              state[..., self.base_size:self.base_size+8], past.flatten(-2)), dim=-1)
        return torch.cat((state[..., :self.physical_size], delivered,
                          past[..., :self.physical_size].flatten(-2)), dim=-1)

    def fit_normalization(self, episodes):
        super().fit_normalization(episodes)
        aux = np.concatenate([e["states"][:-1, self.base_size:self.base_size+8] for e in episodes])
        self.aux_scale.copy_(torch.tensor(np.maximum(aux.std(0), 1e-6)))

    def forward(self, state, action):
        x = (self.inputs(state, action)-self.input_mean)/self.input_scale
        base = state[..., :self.physical_size]+self.delta_mean+self.delta_scale*self.net(x)
        aux = derivatives(base, state, state[..., self.base_size:self.base_size+4], self.dt)
        queue = state[..., self.physical_size:self.base_size].reshape(*state.shape[:-1], self.queue_steps, 2)
        if self.queue_steps:
            queue = torch.cat((queue[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        past = state[..., self.base_size+8:].reshape(*state.shape[:-1], self.history_steps, self.sensory_size)
        past = torch.cat((self.sensations(state).unsqueeze(-2), past[..., :-1, :]), dim=-2)
        return torch.cat((base, queue.flatten(-2), aux, past.flatten(-2)), dim=-1)


def tissue_contract(env, model, aux_weight):
    base, aux = sensor_names(env.config.activation_dynamic)
    frames = base+aux if model.derivative_inputs else base
    queue = [f"pending/{i}/{muscle}" for i in range(env.config.queue_steps) for muscle in ("upper", "lower")]
    history = [f"history/{age}/{name}" for age in range(1, model.history_steps+1) for name in base+aux]
    return {**predictor_contract(env), "schema": "myo.tissue_predictor.v1",
        "tissue_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "architecture": model.architecture, "physical_names": base, "auxiliary_names": aux,
        "state_names": base+queue+aux+history,
        "neural_input_names": base+["delivered_excitation/upper", "delivered_excitation/lower"]+
            (aux if model.derivative_inputs else [])+
            [f"history/{age}/{name}" for age in range(1, model.history_steps+1) for name in frames],
        "strain_definition": "fiber_length/optimal_length-1; tendon_path_length/slack_length-1, unclipped",
        "derivatives": "backward differences of local strains; reset rates/accelerations zero",
        "normalization": "train only; base loss uses delta STD; derivative loss uses derivative STD",
        "aux_weight": aux_weight, "target": "local strain and activation increments; derivatives computed consistently",
        "history_reset": "repeat initial local sensations; own predictions in BPTT",
        "sensor_boundary": "native geometry only in dataset adapter, never q/qvel in learned model"}


def save_tissue(path, model, contract, update, stage):
    torch.save({"architecture": model.architecture, "state_dict": model.state_dict()}, path)
    metadata = {**contract, "update": update, "stage": stage, "seed": 0,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')


def load_tissue(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Tissue checkpoint weights differ from metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Tissue predictor contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Tissue architecture differs from metadata")
    model = TissuePredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


def residuals(model, prediction, target):
    base = (prediction[..., :model.physical_size]-target[..., :model.physical_size])/model.delta_scale
    aux = (prediction[..., model.base_size:model.base_size+8]-target[..., model.base_size:model.base_size+8])/model.aux_scale
    return base, aux


@torch.no_grad()
def evaluate_tissue(model, episodes):
    initial, actions, targets = windows(episodes, 20, stride=5)
    prediction = model.rollout(initial, actions)
    base, aux = residuals(model, prediction, targets)
    result = {}
    for h in (1, 3, 6, 20):
        error = model.sensations(prediction[:, h-1])-model.sensations(targets[:, h-1])
        result[str(round(h*model.dt*1000))] = {
            "samples": len(initial), "channel_rmse": error.double().square().mean(0).sqrt().tolist(),
            "base_mse": float(base[:, h-1].double().square().mean()),
            "rate_rmse_normalized": float(aux[:, h-1, :4].double().square().mean().sqrt()),
            "acceleration_rmse_normalized": float(aux[:, h-1, 4:].double().square().mean().sqrt())}
    return result


@torch.no_grad()
def pooled_errors(model, episodes, aux_weight):
    result = {}
    for split in ("train", "validation", "test"):
        initial, actions, targets = windows([e for e in episodes if e["split"] == split], 6)
        prediction = model.rollout(initial, actions)
        base, aux = residuals(model, prediction, targets)
        a, b = float(base.double().square().mean()), float(aux.double().square().mean())
        result[split] = {"windows": len(initial), "base_mse": a, "aux_mse": b,
                         "training_objective": a+aux_weight*b,
                         "base_channel_rmse_normalized": base.double().square().mean((0, 1)).sqrt().tolist(),
                         "aux_channel_rmse_normalized": aux.double().square().mean((0, 1)).sqrt().tolist()}
    return result


def train_branch(env, episodes, args, output, derivative_inputs, aux_weight):
    torch.manual_seed(0)
    model = TissuePredictor(6 if env.config.activation_dynamic else 4, env.config.queue_steps,
                           dt=env.config.dt, derivative_inputs=derivative_inputs)
    model.fit_normalization([e for e in episodes if e["split"] == "train"])
    contract = tissue_contract(env, model, aux_weight)
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(0)
    result = {"contract": contract, "parameters": sum(p.numel() for p in model.parameters()),
              "input_size": model.net[0].in_features, "stages": {}}
    for stage, horizon, updates in (("one_step", 1, args.one_step_updates), ("bptt", 6, args.bptt_updates)):
        initial, actions, targets = windows(split["train"], horizon)
        history, best = [], float("inf")
        started = time.monotonic()
        for update in range(1, updates+1):
            ids = torch.randint(len(initial), (128,), generator=generator)
            prediction = model.rollout(initial[ids], actions[ids])
            base, aux = residuals(model, prediction, targets[ids])
            loss = base.square().mean()+aux_weight*aux.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite tissue predictor loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if update == 1 or update % 200 == 0 or update == updates:
                metrics = evaluate_tissue(model, split["validation"])
                times = (20,) if stage == "one_step" else (20, 60, 120)
                score = float(np.mean([metrics[str(t)]["rate_rmse_normalized"] for t in times]))
                history.append({"update": update, "loss": float(loss.detach()), "validation_score": score})
                if score < best:
                    best = score
                    save_tissue(output/f"{stage}.pt", model, contract, update, stage)
                print(f"v{env.config.variant} features={derivative_inputs} aux={aux_weight} {stage} "
                      f"{update}/{updates} loss={float(loss.detach()):.5g} val={score:.5g}", flush=True)
        selected, metadata = load_tissue(output/f"{stage}.pt", contract)
        result["stages"][stage] = {"best_update": metadata["update"], "elapsed_s": time.monotonic()-started,
            "history": history, "validation": evaluate_tissue(selected, split["validation"]),
            "test": evaluate_tissue(selected, split["test"]), "pooled": pooled_errors(selected, episodes, aux_weight)}
        model = copy.deepcopy(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tissue_predictor"))
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--bptt-updates", type=int, default=2000)
    args = parser.parse_args()
    if min(args.one_step_updates, args.bptt_updates) < 1:
        parser.error("Update counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema": "myo.tissue_predictor_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__}, "variants": {},
        "scope": "local sensory predictor only; no global position/velocity inputs or targets; no controller"}
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
            "splits": {s: [e["id"] for e in episodes if e["split"] == s]
                       for s in ("train", "validation", "test")}, "branches": {}}
        for name, features, weight in (("strain_history", False, 0.), ("derivative_inputs", True, 0.),
                                       ("derivative_loss", True, 1.)):
            output = args.output_dir/f"variant_{variant}"/name
            output.mkdir(parents=True)
            record["branches"][name] = train_branch(env, episodes, args, output, features, weight)
        summary["variants"][str(variant)] = record
        report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
