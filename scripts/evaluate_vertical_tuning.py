"""Independently check tuning selections, test forecasts and native sensitivities."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_vertical_history import read_source
from myo.train_vertical_predictor import load_predictor, predictor_contract, windows
from myo.train_vertical_tuning import (
    TuningPredictor, data_identity, evaluate_test, load_model, output_coverage, short_metrics, tuning_contract,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles, heuristic


def probes_and_native(env, episodes):
    probes, native = [], []
    for episode in (e for e in episodes if e["split"] == "test"):
        env.reset(episode["initial_position_m"])
        for start in range(251):
            if start in (100, 250):
                np.testing.assert_array_equal(env.state(), episode["states"][start])
                snapshot = env.snapshot()
                commands = episode["actions"][start:start+6].astype(float)

                def response(first):
                    env.restore(snapshot)
                    return np.asarray([env.step(first if i == 0 else action)[1] for i, action in enumerate(commands)])

                derivatives = []
                for muscle in range(2):
                    high, low = commands[0].copy(), commands[0].copy()
                    high[muscle], low[muscle] = min(1., high[muscle]+.005), max(0., low[muscle]-.005)
                    derivatives.append((response(high).astype(float)-response(low))/(high[muscle]-low[muscle]))
                native.append(np.stack(derivatives, axis=-1))
                probes.append((torch.tensor(episode["states"][start]), torch.tensor(commands, dtype=torch.float32)))
                env.restore(snapshot)
            env.step(heuristic(env, episode["family"], episode["version"]))
    return probes, np.asarray(native)


def sensitivity(model, probes, native):
    predicted = []
    for state, commands in probes:
        jac = torch.autograd.functional.jacobian(
            lambda first: model.rollout(state, torch.cat((first[None], commands[1:])))[:, 1], commands[0])
        if model.queue_steps:
            assert torch.count_nonzero(jac[:model.queue_steps]) == 0
        predicted.append(jac.detach().numpy())
    predicted = np.asarray(predicted)

    def compare(a, b):
        norm = np.sqrt(np.mean(a*a))
        meaningful = np.abs(a) > max(1e-6, .01*norm)
        return {"relative_rmse_to_zero": float(np.sqrt(np.mean((a-b)**2))/norm),
            "cosine": float(a.ravel()@b.ravel()/max(1e-12, np.linalg.norm(a)*np.linalg.norm(b))),
            "meaningful_sign_agreement": float(np.mean(np.sign(a[meaningful]) == np.sign(b[meaningful])))}

    return {"probes": len(probes), "at_60ms": compare(native[:, 2], predicted[:, 2]),
            "after_transport_to_120ms": compare(native[:, model.queue_steps:], predicted[:, model.queue_steps:])}, predicted


@torch.no_grad()
def validation_breakdown(model, episodes):
    errors, starts, families = [], [], {}
    for episode in (e for e in episodes if e["split"] == "validation"):
        initial, actions, target = windows([episode], 3, stride=5)
        error = (model.rollout(initial, actions)[..., 1]-target[..., 1]).double()
        errors.append(error)
        starts.extend(np.arange(len(initial))*.1)
        families[episode["family"]] = {str((h+1)*20): float(error[:, h].square().mean().sqrt()) for h in (0, 2)}
    error = torch.cat(errors)
    early = torch.tensor(np.asarray(starts) < .2)
    return {"family_velocity_rmse_m_s": families, "initial_200ms_windows": int(early.sum()),
        "total_windows": len(error),
        "initial_200ms_fraction_of_squared_error": {str((h+1)*20): float(error[early, h].square().sum()/error[:, h].square().sum()) for h in (0, 2)},
        "velocity_rmse_after_initial_200ms_m_s": {str((h+1)*20): float(error[~early, h].square().mean().sqrt()) for h in (0, 2)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_tuning"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for tuning to finish")
    assert summary["selection"] == summary["selection_frozen_before_test"]
    contexts = {}
    for variant, source in summary["sources"].items():
        folder = Path(source["path"])
        metadata = json.loads((folder/"bptt.pt.metadata.json").read_text())
        env = VerticalMuscles(VerticalConfig(**metadata["config"]))
        episodes = read_source(folder, env)
        assert data_identity(folder) == source["data_sha256"]
        contexts[variant] = env, episodes, folder
    models = {}
    result = {"status": "running", "strict_checkpoints": 0, "selections": {}, "variants": {}}
    for path, record in summary["trials"].items():
        env, episodes, folder = contexts[str(record["variant"])]
        prototype = TuningPredictor(**record["contract"]["architecture"])
        prototype.fit_normalization([e for e in episodes if e["split"] == "train"])
        model, metadata = load_model(args.run/path, tuning_contract(env, prototype, data_identity(folder)))
        for name, value in prototype.named_buffers():
            torch.testing.assert_close(value, dict(model.named_buffers())[name], rtol=0, atol=0)
        for split in ("train", "validation"):
            data = windows([e for e in episodes if e["split"] == split], 3, stride=5)
            assert short_metrics(model, data, record["stage"]) == record[split]
        selected = min(record["history"], key=lambda row: row["validation"]["score"])
        assert selected["update"] == metadata["update"] == record["best_update"]
        assert selected["validation"]["score"] == record["best_score"]
        assert metadata["learning_rate"] == record["learning_rate"]
        if path in summary["test"]:
            assert evaluate_test(model, episodes) == summary["test"][path]["forecast"]
            assert output_coverage(model, episodes) == summary["test"][path]["coverage"]
            if model.architecture["output_activation"] == "tanh_scaled":
                assert not any(summary["test"][path]["coverage"]["train"]["channel_fraction_outside_bound"])
        models[path] = model
        result["strict_checkpoints"] += 1
    for phase, candidates in summary["phases"].items():
        scores = {}
        for key, candidate in candidates.items():
            assert set(candidate["models"]) == set(contexts)
            score = float(np.mean([summary["trials"][p]["validation"]["score"] for p in candidate["models"].values()]))
            assert score == candidate["mean_validation_score"]
            scores[key] = score
        assert min(scores, key=scores.get) == summary["selection"][phase]
        result["selections"][phase] = scores

    for variant, (env, episodes, folder) in contexts.items():
        probes, native = probes_and_native(env, episodes)
        arrays, measurements, breakdown = {"native": native}, {}, {}
        for index, path in enumerate(sorted(summary["test"])):
            if str(summary["trials"][path]["variant"]) != variant:
                continue
            measurements[path], arrays[f"predicted_{index}"] = sensitivity(models[path], probes, native)
            measurements[path]["array_key"] = f"predicted_{index}"
            breakdown[path] = validation_breakdown(models[path], episodes)
        historical, _ = load_predictor(folder/"bptt.pt", predictor_contract(env))
        historical_measurements, arrays["historical"] = sensitivity(historical, probes, native)
        result["variants"][variant] = {"sensitivities": measurements, "validation_breakdown": breakdown,
            "historical_120ms_training": {"test": evaluate_test(historical, episodes), "sensitivities": historical_measurements,
                                          "validation_breakdown": validation_breakdown(historical, episodes)}}
        np.savez_compressed(args.run/f"sensitivities_{variant}.npz", **arrays)
        print(f"variant {variant}: native replay and {len(measurements)} candidate Jacobians checked", flush=True)
        (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result["status"] = "passed"
    (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k: v for k, v in result.items() if k not in ("variants", "selections")})


if __name__ == "__main__":
    main()
