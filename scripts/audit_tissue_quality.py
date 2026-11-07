"""Forecast baselines, existing learning curves and native action sensitivities."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_tissue_predictor import (
    TissuePredictor, derivatives, load_tissue, residuals, tissue_contract, tissue_episodes,
)
from myo.train_vertical_history import read_source
from myo.train_vertical_predictor import windows
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


@torch.no_grad()
def extrapolate(model, initial, steps, constant_rate):
    previous = initial
    result = []
    for _ in range(steps):
        base = previous[..., :model.physical_size].clone()
        if constant_rate:
            base[..., :4] += model.dt*initial[..., model.base_size:model.base_size+4]
        aux = derivatives(base, previous, previous[..., model.base_size:model.base_size+4], model.dt)
        queue = previous[..., model.physical_size:model.base_size]
        previous = torch.cat((base, queue, aux), dim=-1)
        result.append(previous)
    return torch.stack(result, dim=1)


@torch.no_grad()
def forecast_metrics(model, prediction, target, config):
    base, aux = residuals(model, prediction, target)
    scale = np.array([config.fiber_optimal_m]*2+[config.tendon_slack_m]*2)
    result = {}
    for h in (1, 3, 6, 20):
        raw = model.sensations(prediction[:, h-1])-model.sensations(target[:, h-1])
        rmse = raw.double().square().mean(0).sqrt().numpy()
        b = model.physical_size
        result[str(h*20)] = {"base_mse": float(base[:, h-1].double().square().mean()),
            "rate_normalized_rmse": float(aux[:, h-1, :4].double().square().mean().sqrt()),
            "acceleration_normalized_rmse": float(aux[:, h-1, 4:].double().square().mean().sqrt()),
            "channel_rmse": rmse.tolist(),
            "length_rmse_mm": (rmse[:4]*scale*1000).tolist(),
            "rate_rmse_mm_s": (rmse[b:b+4]*scale*1000).tolist(),
            "acceleration_rmse_m_s2": (rmse[b+4:b+8]*scale).tolist(),
            "absolute_error_p95": torch.quantile(raw.abs().double(), .95, dim=0).tolist()}
    return result


def native_sensitivities(env, raw, sensory, models):
    actual, states, commands, replay_errors = [], [], [], []
    for episode, observed in zip(raw, sensory):
        if episode["split"] != "test":
            continue
        env.reset(episode["initial_position_m"])
        advanced = 0
        for start in (100, 250):
            while advanced < start:
                env.step(episode["actions"][advanced])
                advanced += 1
            replay_errors.append(float(np.max(np.abs(env.state()-episode["states"][start]))))
            snapshot = env.snapshot()
            u = episode["actions"][start:start+6].astype(float)

            def response(first):
                env.restore(snapshot)
                physical = [env.step(first if i == 0 else action) for i, action in enumerate(u)]
                from myo.train_tissue_predictor import native_to_tissues
                converted = native_to_tissues(torch.tensor(np.asarray(physical)), env.config)
                previous = torch.tensor(observed["states"][start])
                reference = next(iter(models.values()))
                result = []
                for current in converted:
                    aux = derivatives(current, previous, previous[reference.base_size:reference.base_size+4], env.config.dt)
                    result.append(aux[:4].numpy())
                    previous = torch.cat((current, aux))
                return np.asarray(result)

            jac = []
            for muscle in range(2):
                high, low = u[0].copy(), u[0].copy()
                high[muscle] = min(1., high[muscle]+.005)
                low[muscle] = max(0., low[muscle]-.005)
                jac.append((response(high)-response(low))/(high[muscle]-low[muscle]))
            actual.append(np.stack(jac, axis=-1))
            states.append(observed["states"][start])
            commands.append(u.astype(np.float32))
            env.restore(snapshot)
    native = np.asarray(actual)
    result, arrays = {}, {"native": native}
    for name, model in models.items():
        estimates = []
        for state, actions in zip(states, commands):
            def response(first):
                sequence = torch.cat((first.unsqueeze(0), torch.tensor(actions[1:])))
                return model.rollout(torch.tensor(state), sequence)[:, model.base_size:model.base_size+4]
            estimates.append(torch.autograd.functional.jacobian(response, torch.tensor(actions[0])).detach().numpy())
        estimate = np.asarray(estimates)
        arrays[name] = estimate
        scale = model.aux_scale[:4].numpy()[None, None, :, None]
        a, b = native[:, env.config.queue_steps:]/scale, estimate[:, env.config.queue_steps:]/scale
        norm = float(np.sqrt(np.mean(a*a)))
        threshold = max(1e-6, .01*norm)
        mask = np.abs(a) > threshold
        result[name] = {"probes": len(states), "post_delay_normalized_rmse": float(np.sqrt(np.mean((a-b)**2))),
            "zero_jacobian_rmse": norm,
            "relative_rmse_to_zero": float(np.sqrt(np.mean((a-b)**2))/max(norm, 1e-12)),
            "cosine": float(a.ravel()@b.ravel()/max(1e-12, np.linalg.norm(a)*np.linalg.norm(b))),
            "sign_agreement": float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))),
            "sign_elements": int(mask.sum()), "sign_threshold_normalized": threshold,
            "pre_delay_max_abs": float(np.abs(estimate[:, :env.config.queue_steps]).max()) if env.config.queue_steps else None,
            "max_native_replay_state_error": max(replay_errors)}
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_predictor"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/"summary.json").read_text())
    report = {"schema": "myo.tissue_quality_audit.v1", "status": "running", "variants": {},
              "sensitivity_probe_steps": [100, 250], "epsilon": .005}
    for variant, record in summary["variants"].items():
        source = Path(record["source"])
        config = json.loads((source/"bptt.pt.metadata.json").read_text())["config"]
        env = VerticalMuscles(VerticalConfig(**config))
        raw = read_source(source, env)
        sensory = tissue_episodes(raw, env.config)
        test = [e for e in sensory if e["split"] == "test"]
        initial, actions, targets = windows(test, 20, stride=5)
        models, rows, curves = {}, {}, {}
        for name, branch in record["branches"].items():
            prototype = TissuePredictor(**branch["contract"]["architecture"])
            model, _ = load_tissue(args.run/f"variant_{variant}"/name/"bptt.pt",
                                  tissue_contract(env, prototype, branch["contract"]["aux_weight"]))
            models[name] = model
            with torch.no_grad():
                rows[name] = forecast_metrics(model, model.rollout(initial, actions), targets, env.config)
            history = branch["stages"]["bptt"]["history"]
            best_before = min(h["validation_score"] for h in history if h["update"] <= 1200)
            best = min(h["validation_score"] for h in history)
            curves[name] = {"unique_train_transitions": sum(len(e["actions"]) for e in raw if e["split"] == "train"),
                "sampled_windows_one_step": 1000*128, "sampled_windows_bptt": 2000*128,
                "predicted_transitions_total": 1000*128+2000*128*6,
                "best_update": branch["stages"]["bptt"]["best_update"],
                "best_improvement_after_1200_fraction": (best_before-best)/best_before,
                "plateau_established": False, "stop_reason": "fixed_update_budget"}
        model = next(iter(models.values()))
        for name, constant in (("hold_lengths", False), ("constant_strain_rate", True)):
            rows[name] = forecast_metrics(model, extrapolate(model, initial, 20, constant), targets, env.config)
        sensitivities, arrays = native_sensitivities(env, raw, sensory, models)
        np.savez_compressed(args.run/f"quality_jacobians_{variant}.npz", **arrays)
        report["variants"][variant] = {"forecast": rows, "learning": curves, "sensitivities": sensitivities}
        (args.run/"quality_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f"variant {variant}: forecast baselines, curves and native sensitivities done", flush=True)
    report["status"] = "complete"
    (args.run/"quality_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
