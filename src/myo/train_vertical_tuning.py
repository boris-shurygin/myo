"""Sequential learning-rate and activation ablations of the original predictor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .train_vertical_history import read_source
from .train_vertical_predictor import VerticalPredictor, predictor_contract, windows
from .vertical_muscles import VerticalConfig, VerticalMuscles


class IncrementOutput(nn.Module):
    def __init__(self, size, kind):
        super().__init__()
        if kind not in ("linear", "tanh_unit", "tanh_scaled"):
            raise ValueError("Unknown increment output activation")
        self.kind = kind
        self.register_buffer("bound", torch.ones(size))

    def forward(self, value):
        if self.kind == "linear":
            return value
        # Unit slope at zero, independent of the chosen output range.
        return self.bound*torch.tanh(value/self.bound)


class TuningPredictor(VerticalPredictor):
    def __init__(self, physical_size, queue_steps, hidden=128, hidden_activation="tanh", output_activation="linear"):
        super().__init__(physical_size, queue_steps, hidden)
        factories = {"tanh": nn.Tanh, "silu": nn.SiLU, "leaky_relu": lambda: nn.LeakyReLU(.2)}
        if hidden_activation not in factories:
            raise ValueError("Unknown hidden activation")
        # Reuse exactly the baseline's random linear weights across activations.
        self.net[1], self.net[3] = factories[hidden_activation](), factories[hidden_activation]()
        self.net.append(IncrementOutput(physical_size, output_activation))
        self.architecture = dict(physical_size=physical_size, queue_steps=queue_steps, hidden=hidden,
            hidden_activation=hidden_activation, output_activation=output_activation)

    def fit_normalization(self, episodes):
        super().fit_normalization(episodes)
        if self.architecture["output_activation"] == "tanh_scaled":
            delta = np.concatenate([e["states"][1:, :self.physical_size]-e["states"][:-1, :self.physical_size]
                                    for e in episodes])
            standardized = (torch.tensor(delta)-self.delta_mean)/self.delta_scale
            self.net[-1].bound.copy_((1.1*standardized.abs().amax(0)).clamp_min(1.))


def data_identity(folder):
    return hashlib.sha256((folder/"episodes.json").read_bytes()+(folder/"episodes.npz").read_bytes()).hexdigest()


def tuning_contract(env, model, identity):
    return {**predictor_contract(env), "schema": "myo.vertical_tuning.v1", "architecture": model.architecture,
        "tuning_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": identity, "history_steps": 0,
        "output_definition": "normalized state increment; current state + train delta mean + train delta STD * head",
        "output_bound_rule": "unit, or max(1,1.1*train maximum absolute standardized increment); unit derivative at zero",
        "normalization": "train only; identical physical state and increment statistics across candidates",
        "selection": "one_step: normalized velocity RMSE at 20ms; BPTT: mean at 20/60ms; equal mean over four mechanics",
        "optimizer": "Adam; betas=(0.9,0.999),eps=1e-8,weight_decay=0; fresh state per stage; no scheduler",
        "batch_size": 128, "gradient_clip_norm": 10., "seed": 0}


def save_model(path, model, contract, stage, lr, update, parent):
    torch.save({"architecture": model.architecture, "state_dict": model.state_dict()}, path)
    metadata = {**contract, "stage": stage, "learning_rate": lr, "update": update, "parent": parent,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')


def load_model(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise ValueError("Tuning checkpoint hash mismatch")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Tuning contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Tuning architecture mismatch")
    model = TuningPredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


@torch.no_grad()
def short_metrics(model, data, stage):
    initial, actions, target = data
    prediction = model.rollout(initial, actions)[..., :model.physical_size]
    error = (prediction-target[..., :model.physical_size]).double()
    normalized = error/model.delta_scale
    velocity = error[..., 1].square().mean(0).sqrt()
    score = velocity[0]/model.delta_scale[1] if stage == "one_step" else (velocity[0]+velocity[2])/(2*model.delta_scale[1])
    return {"score": float(score), "velocity_rmse_m_s": velocity.tolist(),
        "position_rmse_m": error[..., 0].square().mean(0).sqrt().tolist(),
        "normalized_mse": float(normalized.square().mean()),
        "channel_normalized_mse": normalized.square().mean((0, 1)).tolist()}


def train_stage(env, episodes, model, args, path, stage, lr, identity, parent=None):
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation")}
    horizon = 1 if stage == "one_step" else 3
    initial, actions, target = windows(split["train"], horizon)
    fixed = {s: windows(e, 3, stride=5) for s, e in split.items()}
    contract = tuning_contract(env, model, identity)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    generator = torch.Generator().manual_seed(0)
    maximum = args.one_step_updates if stage == "one_step" else args.bptt_updates
    history, best, clipped, maximum_norm = [], float("inf"), 0, 0.
    started = time.monotonic()

    def check(update, loss=None, norm=None):
        nonlocal best
        val = short_metrics(model, fixed["validation"], stage)
        fitted = short_metrics(model, fixed["train"], stage)
        if not np.isfinite(val["score"]):
            raise RuntimeError("Nonfinite short validation forecast")
        if val["score"] < best:
            best = val["score"]
            save_model(path, model, contract, stage, lr, update, parent)
        history.append({"update": update, "minibatch_loss": loss, "gradient_norm_before_clip": norm,
                        "train_fixed": fitted, "validation": val, "best_score": best})
        print(f"v{env.config.variant} {path.parent.name}/{path.stem} {stage} {update}/{maximum} "
              f"lr={lr:g} val={val['score']:.6g}", flush=True)

    # Retain the starting checkpoint: a bad first BPTT step must not erase it.
    check(0)
    for update in range(1, maximum+1):
        ids = torch.randint(len(initial), (128,), generator=generator)
        prediction = model.rollout(initial[ids], actions[ids])[..., :model.physical_size]
        loss = ((prediction-target[ids, :, :model.physical_size])/model.delta_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite tuning loss")
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
        clipped += int(norm > 10)
        maximum_norm = max(maximum_norm, float(norm))
        optimizer.step()
        if update == 1 or update % args.eval_every == 0 or update == maximum:
            check(update, float(loss.detach()), float(norm))
    selected, metadata = load_model(path, contract)
    return {"variant": env.config.variant, "stage": stage, "learning_rate": lr, "parent": parent,
        "contract": contract, "updates": maximum, "unique_windows": len(initial),
        "sampled_windows": maximum*128, "predicted_transitions": maximum*128*horizon,
        "best_update": metadata["update"], "best_score": best, "stop_reason": "fixed_update_budget",
        "clip_count_all_updates": clipped, "max_gradient_norm": maximum_norm,
        "parameters": sum(p.numel() for p in model.parameters()), "history": history,
        "train": short_metrics(selected, fixed["train"], stage),
        "validation": short_metrics(selected, fixed["validation"], stage), "elapsed_s": time.monotonic()-started}


@torch.no_grad()
def evaluate_test(model, episodes):
    initial, actions, target = windows([e for e in episodes if e["split"] == "test"], 20, stride=5)
    prediction = model.rollout(initial, actions)[..., :model.physical_size]
    result = {}
    for h in (1, 2, 3, 6, 20):
        error = (prediction[:, h-1]-target[:, h-1, :model.physical_size]).double()
        previous_p = initial[:, 1] if h == 1 else prediction[:, h-2, 1]
        previous_t = initial[:, 1] if h == 1 else target[:, h-2, 1]
        acceleration_error = ((prediction[:, h-1, 1]-previous_p)-(target[:, h-1, 1]-previous_t)).double()/.02
        result[str(h*20)] = {"samples": len(initial), "channel_rmse": error.square().mean(0).sqrt().tolist(),
            "channel_normalized_rmse": (error/model.delta_scale).square().mean(0).sqrt().tolist(),
            "normalized_mse": float((error/model.delta_scale).square().mean()),
            "velocity_rmse_m_s": float(error[:, 1].square().mean().sqrt()),
            "position_rmse_m": float(error[:, 0].square().mean().sqrt()),
            "interval_acceleration_rmse_m_s2": float(acceleration_error.square().mean().sqrt())}
    return result


@torch.no_grad()
def output_coverage(model, episodes):
    result = {}
    for split in ("train", "validation", "test"):
        delta = np.concatenate([e["states"][1:, :model.physical_size]-e["states"][:-1, :model.physical_size]
                                for e in episodes if e["split"] == split])
        z = (torch.tensor(delta)-model.delta_mean)/model.delta_scale
        excess = (z.abs()-model.net[-1].bound).clamp_min(0)
        result[split] = {"channel_fraction_outside_bound": (excess > 0).float().mean(0).tolist(),
            "irreducible_one_step_normalized_mse_if_bounded": float(excess.square().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_tuning"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .0003, .0001, .00003, .00001])
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--bptt-updates", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema": "myo.vertical_tuning_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "scope": "Original full physical state; no sensory history; 20ms transitions; 60ms BPTT; predictor only",
        "sources": {}, "trials": {}, "phases": {}, "selection": {}}
    contexts = {}
    started = time.monotonic()

    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    for variant in range(1, 5):
        source = args.reference/f"variant_{variant}"
        metadata = json.loads((source/"bptt.pt.metadata.json").read_text())
        env = VerticalMuscles(VerticalConfig(**metadata["config"]))
        episodes = read_source(source, env)
        contexts[str(variant)] = env, episodes, data_identity(source)
        summary["sources"][str(variant)] = {"path": str(source), "data_sha256": data_identity(source)}

    def new_model(variant, hidden="tanh", output="linear"):
        env, episodes, _ = contexts[variant]
        torch.manual_seed(0)
        model = TuningPredictor(env.config.physical_size, env.config.queue_steps,
                                hidden_activation=hidden, output_activation=output)
        model.fit_normalization([e for e in episodes if e["split"] == "train"])
        return model

    def read_trial(path):
        record = summary["trials"][path]
        env, _, identity = contexts[str(record["variant"])]
        prototype = TuningPredictor(**record["contract"]["architecture"])
        return load_model(args.output_dir/path, tuning_contract(env, prototype, identity))[0]

    def trial(variant, folder, filename, model, stage, lr, parent=None):
        env, episodes, identity = contexts[variant]
        relative = f"variant_{variant}/{folder}/{filename}.pt"
        path = args.output_dir/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        summary["trials"][relative] = train_stage(env, episodes, model, args, path, stage, lr, identity, parent)
        report()
        return relative

    def select(phase):
        candidates = summary["phases"][phase]
        for candidate in candidates.values():
            candidate["mean_validation_score"] = float(np.mean([summary["trials"][p]["validation"]["score"]
                                                                for p in candidate["models"].values()]))
        winner = min(candidates, key=lambda key: candidates[key]["mean_validation_score"])
        summary["selection"][phase] = winner
        print(f"SELECT {phase}: {winner}", flush=True)
        report()
        return candidates[winner]["models"]

    report()
    summary["phases"]["warmup_lr"] = {}
    for lr in args.learning_rates:
        key = f"{lr:.0e}"
        models = {}
        summary["phases"]["warmup_lr"][key] = {"learning_rate": lr, "models": models}
        for variant in contexts:
            models[variant] = trial(variant, "lr_warmup", key, new_model(variant), "one_step", lr)
    warm = select("warmup_lr")
    warm_lr = summary["phases"]["warmup_lr"][summary["selection"]["warmup_lr"]]["learning_rate"]

    summary["phases"]["bptt_lr"] = {}
    for lr in args.learning_rates:
        key = f"{lr:.0e}"
        models = {}
        summary["phases"]["bptt_lr"][key] = {"learning_rate": lr, "models": models}
        for variant in contexts:
            models[variant] = trial(variant, "lr_bptt", key, read_trial(warm[variant]), "bptt", lr, warm[variant])
    tuned = select("bptt_lr")
    bptt_lr = summary["phases"]["bptt_lr"][summary["selection"]["bptt_lr"]]["learning_rate"]

    # Retain a default/default control if the best one-step rate differs.
    # Selecting a warm-up rate in isolation need not select the best final pair.
    default_models = summary["phases"]["bptt_lr"].get("1e-03", {}).get("models") if warm_lr == .001 else None
    if default_models is None and "1e-03" in summary["phases"]["warmup_lr"]:
        default_models = {}
        for variant, parent in summary["phases"]["warmup_lr"]["1e-03"]["models"].items():
            default_models[variant] = trial(variant, "default_control", "bptt", read_trial(parent), "bptt", .001, parent)
    summary["phases"]["lr_pair"] = {"sequential": {"warmup_lr": warm_lr, "bptt_lr": bptt_lr, "models": tuned}}
    if default_models is not None:
        summary["phases"]["lr_pair"]["default"] = {"warmup_lr": .001, "bptt_lr": .001, "models": default_models}
    tuned = select("lr_pair")
    pair = summary["phases"]["lr_pair"][summary["selection"]["lr_pair"]]
    warm_lr, bptt_lr = pair["warmup_lr"], pair["bptt_lr"]

    def activation_candidate(hidden, output, folder):
        models = {}
        for variant in contexts:
            parent = trial(variant, folder, "one_step", new_model(variant, hidden, output), "one_step", warm_lr)
            models[variant] = trial(variant, folder, "bptt", read_trial(parent), "bptt", bptt_lr, parent)
        return {"hidden_activation": hidden, "output_activation": output, "models": models}

    summary["phases"]["hidden_activation"] = {"tanh": {"hidden_activation": "tanh", "output_activation": "linear", "models": tuned}}
    for hidden in ("silu", "leaky_relu"):
        summary["phases"]["hidden_activation"][hidden] = activation_candidate(hidden, "linear", f"hidden_{hidden}")
        report()
    hidden_models = select("hidden_activation")
    selected_hidden = summary["selection"]["hidden_activation"]

    summary["phases"]["output_activation"] = {"linear": {"hidden_activation": selected_hidden,
                                                         "output_activation": "linear", "models": hidden_models}}
    for output in ("tanh_unit", "tanh_scaled"):
        summary["phases"]["output_activation"][output] = activation_candidate(selected_hidden, output, f"output_{output}")
        report()
    final = select("output_activation")
    summary["selected_models"] = final
    # Selection is complete before the first test forecast is evaluated.
    summary["selection_frozen_before_test"] = json.loads(json.dumps(summary["selection"]))
    report()
    summary["test"] = {}
    final_candidates = {path for phase in ("bptt_lr", "lr_pair", "hidden_activation", "output_activation")
                        for candidate in summary["phases"][phase].values() for path in candidate["models"].values()}
    for path in sorted(final_candidates):
        record = summary["trials"][path]
        _, episodes, _ = contexts[str(record["variant"])]
        model = read_trial(path)
        summary["test"][path] = {"forecast": evaluate_test(model, episodes),
                                 "output_bound": model.net[-1].bound.tolist(), "coverage": output_coverage(model, episodes)}
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
