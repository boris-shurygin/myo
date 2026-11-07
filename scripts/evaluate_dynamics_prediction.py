"""Independent stratification of the completed action-dynamics experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.dynamics_prediction import contract, load_predictor, sequence_windows
from myo.env import PreacherCurlConfig, PreacherCurlEnv


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def acceleration_jacobian_metrics(native_velocity, predicted_velocity, dt):
    def differentiate(values):
        return np.diff(np.concatenate((np.zeros_like(values[:, :1]), values), axis=1), axis=1) / dt
    actual, estimated = differentiate(native_velocity), differentiate(predicted_velocity)
    result = {}
    for label, a, b in [("pooled_after_delay", actual[:, 2:], estimated[:, 2:]), *(
        (str(h), actual[:, h - 1], estimated[:, h - 1]) for h in (3, 4, 5, 6))]:
        mask = np.abs(a) > .05 / dt
        af, bf = a.ravel(), b.ravel()
        result[label] = {"rmse_rad_s2_per_excitation": rmse(a - b),
            "zero_gradient_baseline_rmse": rmse(a),
            "sign_agreement": float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))),
            "meaningful_elements": int(mask.sum()),
            "global_cosine": float(af @ bf / max(1e-12, np.linalg.norm(af) * np.linalg.norm(bf)))}
    return result


def load_run_models(run, summary):
    env = PreacherCurlEnv(PreacherCurlConfig(**summary["contract"]["config"]))
    try:
        env.reset(seed=0, options={"payload_mass": env.config.payload_mass})
        expected = contract(env)
        loader = load_predictor
        if summary["contract"]["schema"] == "myo.action_dynamics_wide.v1":
            from myo.train_dynamics_wide import load_wide_predictor, wide_contract
            expected, loader = wide_contract(expected), load_wide_predictor
        elif summary["contract"]["schema"] == "myo.action_dynamics_sensors.v1":
            from myo.dynamics_sensors import load_sensor_predictor, sensor_contract
            expected, loader = sensor_contract(env), load_sensor_predictor
        models = {name: loader(run / f"{name}.pt", expected)[0]
                  for name in summary.get("evaluated_models", ("one_step", "bptt_6"))}
        for model in models.values():
            model.input_state_size = len(expected["state_names"])
        return models
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d1_dynamics_prediction"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--reference-run", type=Path, help="Optional small-model reference for comparison plots")
    parser.add_argument("--reference-label", default="small")
    parser.add_argument("--run-label", default="wide")
    parser.add_argument("--comparison-title")
    parser.add_argument("--acceleration", action="store_true", help="Also plot interval accelerations")
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run / "summary.json").read_text())
    records = json.loads((args.run / "episodes.json").read_text())
    with np.load(args.run / "episodes.npz", allow_pickle=False) as arrays:
        episodes = [{**record, "states": arrays[f"e{i}_states"], "actions": arrays[f"e{i}_actions"]}
                    for i, record in enumerate(records)]
    test = [e for e in episodes if e["split"] == "test"]
    models = load_run_models(args.run, summary)
    partitions = [set(summary["split_episodes"][key]) for key in ("train", "validation", "test")]
    assert all(not (partitions[i] & partitions[j]) for i in range(3) for j in range(i))
    result = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "whole_episode_splits_disjoint": True, "horizons": {}, "jacobians": {},
              "acceleration_jacobians": {}}
    with torch.no_grad():
        for horizon in (1, 3, 6, 20):
            data = sequence_windows(test, horizon)
            actual, initial = data["targets"][:, -1], data["initial"]
            dt = summary["contract"]["control_dt_s"]
            actual_before = initial[:, 1] if horizon == 1 else data["targets"][:, -2, 1]
            actual_acceleration = (actual[:, 1] - actual_before) / dt
            masks = {
                "all": np.ones(len(initial), dtype=bool),
                "changing_velocity": np.abs(actual[:, 1] - initial[:, 1]) > .1,
                "enters_lower_region": (initial[:, 0] >= np.deg2rad(30.)) &
                    np.any(data["targets"][..., 0] < np.deg2rad(30.), axis=1),
                "stays_in_middle_region": (initial[:, 0] >= np.deg2rad(30.)) &
                    (initial[:, 0] < np.deg2rad(130.)) &
                    np.all((data["targets"][..., 0] >= np.deg2rad(30.)) &
                           (data["targets"][..., 0] < np.deg2rad(130.)), axis=1),
                "lower_support_region": initial[:, 0] < np.deg2rad(30.),
                "middle_region": (initial[:, 0] >= np.deg2rad(30.)) & (initial[:, 0] < np.deg2rad(130.)),
                "upper_region": initial[:, 0] >= np.deg2rad(130.),
                **{family: np.array([f"/{family}/" in e for e in data["episode_ids"]])
                   for family in ("pulses", "smooth", "switching")},
            }
            metrics = {key: {"samples": int(mask.sum()), "persistence_velocity_rmse_rad_s":
                       rmse((initial[:, 1] - actual[:, 1])[mask]), "zero_acceleration_rmse_rad_s2":
                       rmse(actual_acceleration[mask])} for key, mask in masks.items()}
            for name, model in models.items():
                sequence = torch.cat([model.rollout(torch.tensor(s), torch.tensor(a))
                    for s, a in zip(np.array_split(initial, 8), np.array_split(data["actions"], 8))]).numpy()
                predicted = sequence[:, -1]
                before = initial[:, 1] if horizon == 1 else sequence[:, -2, 1]
                predicted_acceleration = (predicted[:, 1] - before) / dt
                for key, mask in masks.items():
                    metrics[key][name] = {"velocity_rmse_rad_s": rmse((predicted[:, 1] - actual[:, 1])[mask]),
                        "angle_rmse_rad": rmse((predicted[:, 0] - actual[:, 0])[mask]),
                        "interval_acceleration_rmse_rad_s2": rmse((predicted_acceleration - actual_acceleration)[mask])}
            result["horizons"][str(horizon)] = metrics
    with np.load(args.run / "jacobians.npz") as arrays:
        actual = arrays["native"]
        for name in models:
            estimated = arrays[name]
            result["acceleration_jacobians"][name] = acceleration_jacobian_metrics(actual, estimated, dt)
            result["jacobians"][name] = {}
            for h in (3, 4, 5, 6):
                a, b = actual[:, h - 1], estimated[:, h - 1]
                mask = np.abs(a) > .05
                norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                supported = np.linalg.norm(a, axis=1) > .1
                cosines = (a * b).sum(axis=1)[supported] / np.maximum(norms[supported], 1e-12)
                result["jacobians"][name][str(h)] = {
                    "rmse": rmse(a - b), "zero_gradient_baseline_rmse": rmse(a),
                    "sign_agreement": float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))),
                    "supported_probes": int(supported.sum()),
                    "per_probe_cosine_median": float(np.median(cosines)),
                    "opposite_direction_probes": int(np.sum(cosines < 0)),
                }
        if args.reference_run:
            with np.load(args.reference_run / "jacobians.npz") as previous:
                if not np.array_equal(actual, previous["native"]):
                    raise ValueError("Different native references in derivative comparison")
                result["reference_acceleration_jacobians"] = {
                    name: acceleration_jacobian_metrics(actual, previous[name], dt)
                    for name in previous.files if name != "native"}
    (args.run / "independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_models = {name: (model, summary[name]["test"]) for name, model in models.items()}
        plot_training = {name: summary[name]["training"]["history"] for name in models
                         if name != "bptt_velocity_selection"}
        plot_accelerations = {name: summary[name].get("acceleration_test") for name in models}
        if "bptt_velocity_selection" in models and all(torch.equal(value,
                models["bptt_velocity_selection"].state_dict()[key])
                for key, value in models["bptt_6"].state_dict().items()):
            # Different selection criteria may select exactly the same weights.
            plot_models.pop("bptt_velocity_selection")
            plot_accelerations.pop("bptt_velocity_selection")
        if args.reference_run:
            reference = json.loads((args.reference_run / "summary.json").read_text())
            with np.load(args.run / "episodes.npz") as current, np.load(args.reference_run / "episodes.npz") as previous:
                if set(current.files) != set(previous.files):
                    raise ValueError("Different episodes in reference comparison")
                for key in current.files:
                    width = min(current[key].shape[-1], previous[key].shape[-1])
                    if not np.array_equal(current[key][..., :width], previous[key][..., :width]):
                        raise ValueError("Reference requires identical shared states/actions")
            reference_models = load_run_models(args.reference_run, reference)
            if "bptt_velocity_selection" in reference_models and all(torch.equal(value,
                    reference_models["bptt_velocity_selection"].state_dict()[key])
                    for key, value in reference_models["bptt_6"].state_dict().items()):
                reference_models.pop("bptt_velocity_selection")
            duplicates = {name for name in models if name in reference_models and
                (args.run / f"{name}.pt").read_bytes() == (args.reference_run / f"{name}.pt").read_bytes()}
            plot_models = {**{f"{args.reference_label}/{name}": (model, reference[name]["test"])
                              for name, model in reference_models.items()},
                           **{f"{args.run_label}/{name}": values for name, values in plot_models.items() if name not in duplicates}}
            plot_training = {**{f"{args.reference_label}/{name}": reference[name]["training"]["history"]
                                for name in reference_models},
                             **{f"{args.run_label}/{name}": values for name, values in plot_training.items() if name not in duplicates}}
            plot_accelerations = {**{f"{args.reference_label}/{name}":
                summary.get("reference_acceleration_test", summary.get("velocity_reference_acceleration_test", {})).get(
                    name, reference[name].get("acceleration_test"))
                for name in reference_models}, **{f"{args.run_label}/{name}": values
                for name, values in plot_accelerations.items() if name not in duplicates}}
        def display(name):
            return name.replace("bptt_velocity_selection", "BPTT, velocity selection").replace(
                "bptt_6", "BPTT").replace("one_step", "one step").replace("/", " — ")
        plot_models = {display(name): value for name, value in plot_models.items()}
        plot_training = {display(name): value for name, value in plot_training.items()}
        plot_accelerations = {display(name): value for name, value in plot_accelerations.items()}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        forecast_rows = []
        for ax, family in zip(axes.ravel()[:3], ("pulses", "smooth", "switching")):
            e = next(e for e in test if e["family"] == family and e["initial_angle_deg"] == 75.
                     and e["initial_capacity"] == 1.)
            start, horizon = 15, 20
            times = np.arange(horizon + 1) * .02
            actual = e["states"][start:start + horizon + 1, 1]
            forecasts = {}
            ax.plot(times, np.rad2deg(actual), color="black", lw=2, label="MuJoCo")
            for name, (model, _) in plot_models.items():
                with torch.no_grad():
                    p = model.rollout(torch.tensor(e["states"][start, :model.input_state_size]),
                                      torch.tensor(e["actions"][start:start + horizon]))[:, 1].numpy()
                ax.plot(times, np.rad2deg(np.r_[actual[0], p]), label=name)
                forecasts[name] = np.r_[actual[0], p]
            forecast_rows.append((family, times, actual, forecasts))
            ax.set(title=f"{family}: forecast from t=0.30 s", xlabel="Forecast time (s)", ylabel="Velocity (deg/s)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        ax = axes.ravel()[3]
        horizons = (1, 2, 3, 6, 10, 20)
        for name, (_, metrics) in plot_models.items():
            ax.plot(np.array(horizons) * 20, [metrics[str(h)]["velocity_rmse_rad_s"]
                                            for h in horizons], marker="o", label=name)
        baseline_key = "one_step" if "one_step" in summary else next(iter(models))
        ax.plot(np.array(horizons) * 20, [summary[baseline_key]["test"][str(h)]["persistence_velocity_rmse_rad_s"]
                                        for h in horizons], "--", label="Constant velocity")
        ax.set(title="All held-out test episodes", xlabel="Forecast horizon (ms)", ylabel="Velocity RMSE (rad/s)")
        ax.grid(alpha=.25)
        ax.legend()
        title = args.comparison_title or ("Action-conditioned dynamics: one step vs BPTT (no policy training)" +
                     (" — 128x128 vs 1024x1024x512" if args.reference_run else ""))
        fig.suptitle(title)
        fig.savefig(args.run / "comparison.png", dpi=160)
        fig.savefig(args.run / "comparison.svg")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for name, history in plot_training.items():
            updates = [row["update"] for row in history]
            axes[0].plot(updates, [row["training_loss"] for row in history], label=name)
            axes[1].plot(updates, [row["score"] for row in history], label=name)
        axes[0].set(title="Training objectives (not directly comparable)" if args.acceleration else
                         "Training loss (compare sizes within each method)",
                    xlabel="Optimizer updates", ylabel="Normalized state loss", yscale="log")
        axes[1].set(title="Velocity validation score",
                    xlabel="Optimizer updates", ylabel="Mean velocity RMSE at 20/60/120 ms (rad/s)")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(args.run / "training.png", dpi=160)
        fig.savefig(args.run / "training.svg")
        plt.close(fig)
        if args.acceleration:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for ax, (family, times, actual, forecasts) in zip(axes.ravel()[:3], forecast_rows):
                ax.plot(times[1:], np.diff(actual) / dt, color="black", lw=2, label="MuJoCo")
                for name, predicted in forecasts.items():
                    ax.plot(times[1:], np.diff(predicted) / dt, label=name)
                ax.set(title=family, xlabel="Interval end (s)", ylabel="Mean acceleration (rad/s²)")
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            ax = axes.ravel()[3]
            for name, metrics in plot_accelerations.items():
                ax.plot(np.array(horizons) * 20, [metrics[str(h)]["interval_acceleration_rmse_rad_s2"]
                        for h in horizons], marker="o", label=name)
            baseline = summary[baseline_key]["acceleration_test"]
            ax.plot(np.array(horizons) * 20, [baseline[str(h)]["zero_acceleration_rmse_rad_s2"]
                    for h in horizons], "--", label="Zero acceleration")
            ax.set(title="All held-out test episodes", xlabel="Interval end (ms)", ylabel="Acceleration RMSE (rad/s²)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.suptitle(title)
            fig.savefig(args.run / "acceleration.png", dpi=160)
            fig.savefig(args.run / "acceleration.svg")
            plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
