"""Recheck sensory-rate artifacts and compare 60 ms action sensitivities."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_tissue_predictor import native_to_tissues
from myo.train_tissue_sampling import (
    SamplingPredictor, data_identity, load_model, metrics, model_contract, read_episodes, sampling_windows,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles, heuristic


def check_stage(folder, record, episodes, config, identity, autoregressive):
    prototype = SamplingPredictor(**record["contract"]["architecture"])
    prototype.fit_normalization([e for e in episodes if e["split"] == "train"], config.delayed)
    expected = model_contract(config, prototype, identity)
    model, metadata = load_model(folder, expected)
    for name, value in prototype.named_buffers():
        torch.testing.assert_close(value, dict(model.named_buffers())[name], rtol=0, atol=0)
    can_roll = model.sensor_ms == model.transition_ms
    for split in ("validation", "test"):
        data = sampling_windows([e for e in episodes if e["split"] == split], model.sensor_ms,
                                model.transition_ms, 60 if can_roll else 20, config.delayed, stride=5)
        assert metrics(model, data, can_roll) == record[split]
    best_row = min(record["history"], key=lambda row: row["validation_score"])
    assert metadata["update"] == record["best_update"] == best_row["update"]
    assert record["sampled_windows"] == record["updates"]*128
    assert record["predicted_transitions"] == record["updates"]*128*(60//model.transition_ms if autoregressive else 1)
    return model


def check_plateau(record, arguments):
    significant, last = float("inf"), 0
    first_stop = None
    for row in record["history"]:
        if row["validation_score"] < significant*.99:
            significant, last = row["validation_score"], row["update"]
        if row["update"] >= 1000 and row["update"]-last >= arguments["plateau_patience"]:
            first_stop = row["update"]
            break
    if record["stop_reason"] == "validation_plateau":
        assert first_stop == record["updates"]
    else:
        assert first_stop is None and record["updates"] == arguments["max_bptt_updates"]


def baselines(episodes, scale):
    current, old, targets = [], [], []
    fine_old = {5: [], 1: []}
    for e in episodes:
        for start in range(0, len(e["actions"])-2, 5):
            tick = start*20
            current.append(e["sensors"][tick, :4])
            old.append(e["sensors"][max(0, tick-20), :4])
            for ms in fine_old:
                fine_old[ms].append(e["sensors"][max(0, tick-ms), :4])
            targets.append(e["sensors"][tick+np.array([20, 40, 60]), :4])
    current, old, target = map(np.asarray, (current, old, targets))
    result = {}
    forecasts = {"persistence": np.repeat(current[:, None], 3, axis=1),
                 "constant_rate": current[:, None]+(current-old)[:, None]*np.arange(1, 4)[None, :, None]}
    for ms, earlier in fine_old.items():
        forecasts[f"constant_rate_{ms}ms"] = current[:, None]+(current-np.asarray(earlier))[:, None]*(20/ms)*np.arange(1, 4)[None, :, None]
    for name, predicted in forecasts.items():
        error = (predicted-target).astype(np.float64)
        result[name] = {}
        for index, ms in enumerate((20, 40, 60)):
            meters = np.array([.48, .48, .12, .12])
            rmse = np.sqrt(np.mean((error[:, index]*meters)**2, axis=0))
            result[name][str(ms)] = {"fiber_length_rmse_mm": float(np.sqrt(np.mean(rmse[:2]**2))*1000),
                "tendon_length_rmse_mm": float(np.sqrt(np.mean(rmse[2:]**2))*1000),
                "strain_rmse_over_20ms_delta_std": float(np.sqrt(np.mean((error[:, index]/scale[:4])**2)))}
    return result


def velocity_information(episodes):
    """Audit only: exact native velocity never enters the learned predictor."""
    result = {}
    for ms in (20, 5, 1):
        errors = []
        for e in episodes:
            ticks = np.arange(20, len(e["sensors"]), 20)
            delta = e["sensors"][ticks, :4].astype(float)-e["sensors"][ticks-ms, :4].astype(float)
            observed = (delta[:, :2]*.48+delta[:, 2:4]*.12)/(ms/1000)
            v = e["native_physical"][ticks, 1]
            errors.append(observed-np.stack((-v, v), axis=-1))
        result[str(ms)] = {"path_velocity_estimation_rmse_mm_s": float(np.sqrt(np.mean(np.concatenate(errors)**2))*1000)}
    return result


def native_jacobians(config, episodes, models):
    env = VerticalMuscles(config)
    actual, predicted, replay_errors = [], {name: [] for name in models}, []
    for episode in (e for e in episodes if e["split"] == "test"):
        env.reset(episode["initial_position_m"])
        for start in range(min(251, len(episode["actions"])-2)):
            if start in (100, 250):
                commands = episode["actions"][start:start+3].astype(float)
                snapshot = env.snapshot()
                native = env.state()
                replay_errors.append(float(np.max(np.abs(native[:config.physical_size]-episode["native_physical"][start*20]))))

                def response(first):
                    env.restore(snapshot)
                    sensors = []
                    for i, command in enumerate(commands):
                        env.step(first if i == 0 else command)
                        state = [env.position, env.velocity]
                        if config.elastic:
                            state.extend(env.fiber_lengths())
                        if config.activation_dynamic:
                            state.extend(env.activation)
                        sensors.append(native_to_tissues(torch.tensor(state, dtype=torch.float64), config)[:4].numpy())
                    return (sensors[2]-sensors[1])/.02

                jac = []
                for muscle in range(2):
                    high, low = commands[0].copy(), commands[0].copy()
                    high[muscle], low[muscle] = min(1., high[muscle]+.005), max(0., low[muscle]-.005)
                    jac.append((response(high)-response(low))/(high[muscle]-low[muscle]))
                actual.append(np.stack(jac, axis=-1))
                for name, model in models.items():
                    history = torch.tensor(episode["sensors"][np.maximum(0, start*20-np.arange(0, 61, model.sensor_ms))])
                    pending = torch.tensor(episode["actions"][start-2:start] if config.delayed else np.empty((0, 2)), dtype=torch.float32)
                    tail = torch.tensor(commands[1:], dtype=torch.float32)

                    def predicted_rate(first):
                        forecast = model.rollout(history, pending, torch.cat((first[None], tail)))
                        return (forecast[-1, :4]-forecast[40//model.transition_ms-1, :4])/.02

                    estimated = torch.autograd.functional.jacobian(predicted_rate, torch.tensor(commands[0], dtype=torch.float32))
                    predicted[name].append(estimated.detach().numpy())
                env.restore(snapshot)
            env.step(heuristic(env, episode["family"], episode["version"]))
    actual = np.asarray(actual)
    if max(replay_errors) > 1e-6:
        raise AssertionError("Dense sensors and native replay disagree at a command boundary")
    result, arrays = {}, {"native": actual}
    for name, model in models.items():
        estimate = np.asarray(predicted[name])
        scale = model.common_scale[:4].numpy()[None, :, None]/.02
        a, b = actual/scale, estimate/scale
        denominator = np.sqrt(np.mean(a*a))
        threshold = max(1e-6, .01*denominator)
        mask = np.abs(a) > threshold
        result[name] = {"probes": len(actual), "replay_max_abs": max(replay_errors),
            "relative_jacobian_rmse": float(np.sqrt(np.mean((a-b)**2))/denominator),
            "cosine": float(a.ravel()@b.ravel()/max(1e-12, np.linalg.norm(a)*np.linalg.norm(b))),
            "meaningful_sign_agreement": float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))}
        arrays[name] = estimate
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_sampling"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for sampling training to complete")
    result = {"status": "running", "strict_checkpoints": 0, "variants": {}}
    for variant, record in summary["variants"].items():
        folder = args.run/f"variant_{variant}"
        episodes = read_episodes(folder)
        identity = data_identity(folder)
        assert identity == record["data_sha256"]
        ids = {s: {e["id"] for e in episodes if e["split"] == s} for s in ("train", "validation", "test")}
        assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
        config = VerticalConfig(**record["config"])
        models = {}
        for sensor, branch in record["direct"].items():
            check_stage(folder/f"direct_{sensor}ms/one_step.pt", branch, episodes, config, identity, False)
            result["strict_checkpoints"] += 1
        for sensor, branch in record["rollout"].items():
            if branch["warmup"] is not None:
                check_stage(folder/f"rollout_{sensor}ms/one_step.pt", branch["warmup"], episodes, config, identity, False)
                result["strict_checkpoints"] += 1
            models[sensor] = check_stage(folder/f"rollout_{sensor}ms/bptt.pt", branch, episodes, config, identity, True)
            check_plateau(branch, summary["arguments"])
            result["strict_checkpoints"] += 1
        sensitivity, arrays = native_jacobians(config, episodes, models)
        np.savez_compressed(folder/"sensitivities.npz", **arrays)
        result["variants"][variant] = {"baselines": baselines([e for e in episodes if e["split"] == "test"], models["20"].common_scale.numpy()),
                                       "velocity_information": velocity_information([e for e in episodes if e["split"] == "test"]),
                                       "sensitivities": sensitivity}
        print(f"variant {variant}: checkpoints, train-only scales, metrics and native sensitivities verified", flush=True)
        (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result["status"] = "passed"
    (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
