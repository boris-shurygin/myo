"""Audit excitation and sensor data in saved probes without physical rollouts.

Run after scripts/audit_exploration_discovery.py has saved its JSON and NPZ files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(1)
SOURCE = ROOT / "runs/d1_exploration_discovery_audit.json"
discovery = json.loads(SOURCE.read_text())
names = ("small_000", "small_003", "small_004", "wide_000", "wide_004")


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "quantiles": dict(zip(("p05", "p50", "p90", "p95", "p99"),
                                  np.quantile(values, [.05, .5, .9, .95, .99]).tolist()))}


def runs(mask, data):
    result = []
    start = None
    for i in range(len(mask) + 1):
        boundary = i == len(mask) or (i > 0 and (
            data["episodes"][i] != data["episodes"][i - 1]
            or not np.isclose(data["targets_deg"][i], data["targets_deg"][i - 1])))
        active = i < len(mask) and bool(mask[i])
        if start is not None and (boundary or not active):
            result.append({"first_index": int(start), "last_index": int(i - 1),
                           "steps": int(i - start), "duration_s": float((i - start) * .02),
                           "first_action_time_s": float(data["times_s"][start] - .02),
                           "end_action_time_s": float(data["times_s"][i - 1]),
                           "episode": int(data["episodes"][start]),
                           "target_deg": float(data["targets_deg"][start])})
            start = None
        if active and start is None:
            start = i
    return {"samples": int(mask.sum()), "series": len(result),
            "maximum_duration_s": max((r["duration_s"] for r in result), default=0.0),
            "series_length_histogram_steps": {
                str(n): sum(r["steps"] == n for r in result)
                for n in sorted({r["steps"] for r in result})},
            "longest_series": sorted(result, key=lambda r: r["steps"], reverse=True)[:8]}


def summarize_selection(data, selected):
    actions = data["actions"]
    flex, ext = actions[:, :3].mean(1), actions[:, 3:].mean(1)
    result = {"steps": int(selected.sum()),
              "sampled_flexor_average": summary(flex[selected]),
              "sampled_extensor_average": summary(ext[selected]),
              "sampled_flexor_minus_extensor_average": summary((flex - ext)[selected]),
              "mean_policy_flexor_average": float(data["means"][selected, :3].mean()),
              "mean_policy_extensor_average": float(data["means"][selected, 3:].mean()),
              "events": {}}
    for label, columns in (("activation", [3, 9, 15, 21, 27, 33]),
                           ("force_normalized", [4, 10, 16, 22, 28, 34])):
        values = data["observations"][:, columns]
        result["pre_action_" + label] = {
            "flexor_average": summary(values[selected, :3].mean(1)),
            "extensor_average": summary(values[selected, 3:].mean(1)),
            "per_muscle_mean": values[selected].mean(0).tolist()}
    for threshold in (.2, .3):
        all_event = (actions[:, :3] >= threshold).all(1) & (actions[:, 3:] <= .1).all(1)
        average_event = (flex >= threshold) & (ext <= .1)
        events = {"all_three_flexors_high_and_all_three_extensors_low": runs(selected & all_event, data),
                  "average_flexor_high_and_average_extensor_low": runs(selected & average_event, data)}
        result["events"][str(threshold)] = events
    return result


report = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "control_dt_s": .02, "models": {},
          "field_definitions": {
              "sampled_flexor_average": "Arithmetic mean of the three direct excitation commands, dimensionless 0..1.",
              "sampled_extensor_average": "Arithmetic mean of the three direct excitation commands, dimensionless 0..1.",
              "pre_action_activation": "Pre-action observation muscle activation, dimensionless 0..1; grouped arithmetic averages.",
              "pre_action_force_normalized": "Pre-action output.tendon_force / muscle.params.max_isometric_force (both in N), dimensionless; grouped arithmetic averages, not torque.",
              "force_observation_columns": [4, 10, 16, 22, 28, 34],
              "activation_observation_columns": [3, 9, 15, 21, 27, 33],
              "muscle_order": ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"],
              "goals_above_15_after_initial_half_second": "Post-step time >0.5 s and target >15.01 deg, all three scenarios.",
              "rising_more_than_5_deg_s": "Known post-step angular velocity >5 deg/s, target >=25 deg, post-step time >0.5 s.",
              "not_rising": "Known post-step angular velocity <=0 deg/s, target >=25 deg, post-step time >0.5 s."},
          "limitations": [
              "Saved frozen-policy probes are not the full original training history.",
              "Equal excitation averages are not torque averages: muscles have unequal forces and moment arms.",
              "Thresholds are descriptive, not established sufficient or necessary lift conditions.",
              "Series never cross reset or command changes; samples specify commands before transport delay.",
              "This report estimates no action-event or physical-success probabilities.",
              "Activations and normalized tendon forces are pre-action sensor values; angles are post-action states.",
              "Progress windows associate actions and movement but do not establish individual-action causal credit."]}

for name in names:
    info = discovery["models"][name]
    path = ROOT / info["probe"]
    data = dict(np.load(path))
    checkpoint = ROOT / info["checkpoint"]
    metadata = json.loads(Path(str(checkpoint) + ".metadata.json").read_text())
    assert metadata["contract"]["action_mapping"] == ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"]
    assert metadata["contract"]["control_dt_s"] == .02
    assert [metadata["observation_names"][i] for i in [3, 9, 15, 21, 27, 33]] == [
        name + "/activation" for name in metadata["contract"]["action_mapping"]]
    assert [metadata["observation_names"][i] for i in [4, 10, 16, 22, 28, 34]] == [
        name + "/force_normalized" for name in metadata["contract"]["action_mapping"]]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == info["checkpoint_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["probe_sha256"]
    model = PPO.load(checkpoint, device="cpu")
    model.policy.set_training_mode(False)
    with torch.no_grad():
        distribution = model.policy.get_distribution(torch.as_tensor(data["observations"])).distribution
        mean_difference = float(np.max(np.abs(distribution.mean.numpy() - data["means"])))
        assert mean_difference < 1e-5
    selection = (data["targets_deg"] > 15.01) & (data["times_s"] > .5 + 1e-10)
    item = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": info["checkpoint_sha256"],
            "probe_sha256": info["probe_sha256"], "mean_replay_max_error": mean_difference,
            "all_probes": summarize_selection(data, np.ones(len(selection), dtype=bool)),
            "goals_above_15_after_initial_half_second": summarize_selection(data, selection),
            "by_scenario": {}}
    for episode, scenario in enumerate(info["scenarios"]):
        selected = selection & (data["episodes"] == episode)
        if selected.any():
            item["by_scenario"][scenario["scenario"]] = summarize_selection(data, selected)
    high_target = (data["targets_deg"] >= 25) & (data["times_s"] > .5 + 1e-10)
    for label, condition in (("rising_more_than_5_deg_s", data["velocities_deg_s"] > 5),
                             ("not_rising", data["velocities_deg_s"] <= 0)):
        selected = high_target & data["velocity_known"] & condition
        if selected.any():
            item[label] = summarize_selection(data, selected)
    if name == "small_003":
        window = info["scenarios"][0]["best_fixed_goal_progress"]
        start, end = int(window["minimum_index"]), int(window["peak_index"])
        indices = np.arange(len(selection))
        fixed_goal = (data["episodes"] == 0) & np.isclose(data["targets_deg"], window["goal_deg"])
        item["example_rise"] = {"minimum_index": start, "peak_index": end,
            "minimum_angle_deg": float(data["angles_deg"][start]), "peak_angle_deg": float(data["angles_deg"][end]),
            "rise_only": summarize_selection(data, (indices >= start) & (indices <= end)),
            "including_up_to_0_2s_before_minimum_same_command": summarize_selection(
                data, (indices >= start - 10) & (indices <= end) & fixed_goal),
            "commands": [{"i": int(i), "time_s": float(data["times_s"][i]),
                          "q_deg": float(data["angles_deg"][i]),
                          "flexor_mean": float(data["actions"][i, :3].mean()),
                          "extensor_mean": float(data["actions"][i, 3:].mean()),
                          "actions": data["actions"][i].tolist()}
                         for i in range(start - 10, end + 1) if fixed_goal[i]]}
    report["models"][name] = item
    scoped = item["goals_above_15_after_initial_half_second"]
    print(name, "mean flex/ext", scoped["sampled_flexor_average"]["mean"],
          scoped["sampled_extensor_average"]["mean"],
          "avg event runs", [(t, e["average_flexor_high_and_average_extensor_low"]["samples"],
                              e["average_flexor_high_and_average_extensor_low"]["maximum_duration_s"])
                             for t, e in scoped["events"].items()], flush=True)

output = ROOT / "runs/d1_exploration_effort_audit.json"
output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(output)
