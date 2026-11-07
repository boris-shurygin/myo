"""Review saved predictor training settings without changing trained artifacts."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from myo.train_tissue_sampling import (
    SamplingPredictor, data_identity, load_model, metrics, model_contract, predict,
    read_episodes, sampling_windows, transition_windows,
)
from myo.vertical_muscles import VerticalConfig


def load_checked(path, record, config, identity):
    prototype = SamplingPredictor(**record["contract"]["architecture"])
    return load_model(path, model_contract(config, prototype, identity))[0]


def common_loss(model, prediction, target):
    stride = 20//model.transition_ms
    residual = (prediction[:, stride-1::stride]-target[:, stride-1::stride])/model.common_scale
    return residual.square().mean()


def logged_gradients(record):
    norms = [row["gradient_norm_before_clip"] for row in record["history"]]
    return {"logged_checks": len(norms), "checks_exceeding_clip_10": sum(value > 10 for value in norms),
            "min": min(norms), "max": max(norms),
            "scope": "Only saved checks, not every optimizer update"}


@torch.no_grad()
def forecast_diagnostics(model, data, autoregressive):
    hidden, handles = [], []
    for layer in model.net:
        if isinstance(layer, nn.Tanh):
            stats = {"count": 0, "abs_above_099": 0, "derivative_sum": 0.}
            hidden.append(stats)

            def capture(module, inputs, output, stats=stats):
                stats["count"] += output.numel()
                stats["abs_above_099"] += int((output.abs() > .99).sum())
                stats["derivative_sum"] += float((1-output.square()).double().sum())

            handles.append(layer.register_forward_hook(capture))
    try:
        forecast = predict(model, data, autoregressive)
    finally:
        for handle in handles:
            handle.remove()
    for stats in hidden:
        stats["saturated_fraction"] = stats["abs_above_099"]/stats["count"]
        stats["mean_tanh_derivative"] = stats.pop("derivative_sum")/stats["count"]
    if autoregressive:
        stride = 20//model.transition_ms
        forecast, target = forecast[:, stride-1::stride], data["target"][:, stride-1::stride]
    else:
        target = data["target"]
    residual = (forecast-target).double()/model.common_scale
    channel_mse = residual.square().mean((0, 1))
    total = channel_mse.sum()
    groups = {"fiber": slice(0, 2), "tendon": slice(2, 4)}
    if model.physical_size == 6:
        groups["activation"] = slice(4, 6)
    shares = {name: float(channel_mse[indices].sum()/total) for name, indices in groups.items()}
    return {"common_20ms_scaled_mse": float(channel_mse.mean()), "channel_mse": channel_mse.tolist(),
            "loss_shares": shares, "hidden_rollout_statistics": hidden}


@torch.no_grad()
def input_conditioning(model, data):
    delivered = model.delivered(data["pending"], data["actions"], model.transition_ms, 20)[:, 0]
    x = ((model.inputs(data["history"], delivered)-model.input_mean)/model.input_scale).double()
    x -= x.mean(0)
    eigenvalues = torch.linalg.eigvalsh(x.T@x).clamp_min(0).flip(0)
    cumulative = eigenvalues.cumsum(0)/eigenvalues.sum()
    components = {str(fraction): int(torch.searchsorted(cumulative, fraction))+1 for fraction in (.95, .99, .999)}
    return {"input_dimensions": x.shape[-1], "rows": len(x), "components_for_variance": components,
            "scope": "Correlated inputs; low-variance temporal differences can still be essential"}


@torch.no_grad()
def native_one_step(model, episodes, delayed):
    data = transition_windows(episodes, model.sensor_ms, model.transition_ms, delayed)
    errors = []
    for start in range(0, len(data["history"]), 512):
        batch = {key: value[start:start+512] for key, value in data.items()}
        errors.append((predict(model, batch, False)[:, 0]-batch["target"][:, 0]).double())
    error = torch.cat(errors)
    meters = torch.tensor([.48, .48, .12, .12])
    physical = error[:, :4]*meters
    return {"samples": len(error), "prediction_dt_ms": model.transition_ms,
            "mse_over_own_delta_std": float((error/model.delta_scale).square().mean()),
            "channel_rmse": error.square().mean(0).sqrt().tolist(),
            "fiber_rmse_mm": float(physical[:, :2].square().mean().sqrt()*1000),
            "tendon_rmse_mm": float(physical[:, 2:].square().mean().sqrt()*1000),
            "scope": "Actual past history and one native transition, all within-command phases"}


def first_bptt_step(model, train, validation, recorded, learning_rates):
    """Reproduce the first actual update; other rates are local diagnostics only."""
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(len(train["history"]), (128,), generator=generator)
    batch = {key: value[ids] for key, value in train.items()}
    with torch.no_grad():
        baseline_loss = float(common_loss(model, predict(model, train, True), train["target"]))
    result = {"before": {"validation": metrics(model, validation, True), "train_loss": baseline_loss}, "updates": {}}
    for lr in learning_rates:
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=lr)
        loss = common_loss(candidate, predict(candidate, batch, True), batch["target"])
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(candidate.parameters(), 10.)
        optimizer.step()
        with torch.no_grad():
            after_loss = float(common_loss(candidate, predict(candidate, train, True), train["target"]))
            delta = sum((new-old).double().square().sum() for new, old in zip(candidate.parameters(), model.parameters()))
            original = sum(p.double().square().sum() for p in model.parameters())
        evaluated = metrics(candidate, validation, True)
        result["updates"][str(lr)] = {"minibatch_loss_before": float(loss.detach()),
            "gradient_norm_before_clip": float(norm), "relative_parameter_update_norm": float((delta/original).sqrt()),
            "train_loss": after_loss, "validation": evaluated}
        if lr == .001:
            np.testing.assert_allclose(evaluated["score"], recorded["validation_score"], rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(float(norm), recorded["gradient_norm_before_clip"], rtol=1e-6, atol=1e-8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_sampling"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Review a completed run")
    result = {"status": "running", "scope": "read-only artifacts; single-step copies, no new trained checkpoint",
              "learning_rates_in_local_probe": [.001, .0003, .0001, .00003, .00001], "variants": {}}
    for variant, record in summary["variants"].items():
        folder = args.run/f"variant_{variant}"
        config, episodes = VerticalConfig(**record["config"]), read_episodes(folder)
        identity = data_identity(folder)
        assert identity == record["data_sha256"]
        split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
        item = {"direct": {}, "rollout": {}}
        result["variants"][variant] = item
        for mode in ("direct", "rollout"):
            for ms, branch in record[mode].items():
                stage = "bptt" if mode == "rollout" else "one_step"
                model = load_checked(folder/f"{mode}_{ms}ms/{stage}.pt", branch, config, identity)
                can_roll = mode == "rollout"
                data = {s: sampling_windows(split[s], model.sensor_ms, model.transition_ms,
                        60 if can_roll else 20, config.delayed, stride=5) for s in split}
                entry = {"gradient_checks": logged_gradients(branch),
                    "diagnostics": {s: forecast_diagnostics(model, data[s], can_roll) for s in split},
                    "input_conditioning": input_conditioning(model, data["train"])}
                item[mode][ms] = entry
                if mode == "rollout":
                    warm_record = branch["warmup"] or record["direct"]["20"]
                    warm_folder = folder/f"rollout_{ms}ms" if branch["warmup"] else folder/"direct_20ms"
                    warm = load_checked(warm_folder/"one_step.pt", warm_record, config, identity)
                    entry["native_one_step_validation"] = {
                        "warmup": native_one_step(warm, split["validation"], config.delayed),
                        "selected_bptt": native_one_step(model, split["validation"], config.delayed)}
                    entry["warmup_exposure"] = {"unique_windows": warm_record["unique_windows"],
                        "updates": warm_record["updates"], "best_update": warm_record["best_update"],
                        "equivalent_passes": warm_record["sampled_windows"]/warm_record["unique_windows"]}
                    entry["warmup_gradient_checks"] = logged_gradients(warm_record)
                    training = sampling_windows(split["train"], model.sensor_ms, model.transition_ms, 60, config.delayed)
                    entry["first_bptt_step"] = first_bptt_step(warm, training, data["validation"],
                                                             branch["history"][0], result["learning_rates_in_local_probe"])
                print(f"variant {variant} {mode} {ms}ms reviewed", flush=True)
        (args.run/"training_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result["status"] = "passed"
    (args.run/"training_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print("Training parameter audit passed; original checkpoints unchanged", flush=True)


if __name__ == "__main__":
    main()
