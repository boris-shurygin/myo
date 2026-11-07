"""Compare 20/5/1 ms learned transitions over the same 120 ms physical horizon.

Native physics stays at 1 ms. Recorded commands are held for 20 ms in every
dataset. The explicit delay queue, not the neural input width, grows at finer
control rates. All training batches use the same episode/time anchors.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .delayed_acceleration import NativeResponse, capture
from .dynamics_command_history import (HistoryTransitionModel, add_delivered_history,
                                       history_contract, load_history_predictor)
from .dynamics_prediction import compact_state
from .dynamics_sensors import SensorTransitionModel, augment_episodes, load_sensor_predictor, sensor_contract
from .env import PreacherCurlConfig, PreacherCurlEnv
from .train_dynamics_wide import save_wide_predictor


def resample_episode(episode, frame_skip, muscles=6):
    """Collapse only identical commands in each native-rate queue block."""
    physical = 2 + 3 * muscles
    states = episode["states"][::frame_skip]
    pending = states[:, physical:].reshape(len(states), -1, frame_skip, muscles)
    if not np.all(pending == pending[:, :, :1]):
        raise ValueError("Commands change within a resampled delay block")
    actions = episode["actions"].reshape(-1, frame_skip, muscles)
    if not np.all(actions == actions[:, :1]):
        raise ValueError("Commands change within a resampled action interval")
    result = {**episode, "states": np.concatenate((states[:, :physical], pending[:, :, 0].reshape(len(states), -1)), axis=1),
              "actions": actions[:, 0].copy()}
    return augment_episodes([result], muscles)[0]


def time_anchors(episodes, horizon_s, dt, stride_s=.02):
    horizon, stride = round(horizon_s / dt), round(stride_s / dt)
    return [(i, start) for i, e in enumerate(episodes)
            for start in range(0, len(e["actions"]) - horizon + 1, stride)]


def batch_windows(episodes, anchors, ids, horizon):
    rows = [anchors[int(i)] for i in ids]
    initial = np.stack([episodes[i]["states"][s] for i, s in rows])
    actions = np.stack([episodes[i]["actions"][s:s+horizon] for i, s in rows])
    # Do not materialize all overlapping windows or store future queue labels.
    targets = np.stack([episodes[i]["states"][s+1:s+horizon+1, :20] for i, s in rows])
    return tuple(torch.tensor(x, dtype=torch.float32) for x in (initial, actions, targets))


def interval_loss(initial, prediction, target, physical_scale, acceleration_scale, dt):
    residual = (prediction[..., :20] - target) / physical_scale
    before_p = torch.cat((initial[:, None, 1], prediction[:, :-1, 1]), dim=1)
    before_t = torch.cat((initial[:, None, 1], target[:, :-1, 1]), dim=1)
    acceleration = ((prediction[..., 1] - before_p) - (target[..., 1] - before_t)) / dt
    residual = torch.cat((residual[..., :1], (acceleration / acceleration_scale).unsqueeze(-1), residual[..., 2:]), dim=-1)
    return residual.square().mean()


@torch.no_grad()
def evaluate_time(model, episodes, dt, horizons_ms=(20, 60, 120), max_anchors=None):
    horizon = round(max(horizons_ms) / 1000 / dt)
    anchors = time_anchors(episodes, max(horizons_ms) / 1000, dt)
    ids = np.arange(len(anchors))
    if max_anchors is not None and len(ids) > max_anchors:
        ids = np.linspace(0, len(ids)-1, max_anchors, dtype=int)
    errors = {str(ms): [] for ms in horizons_ms}
    raw_acceleration_errors = {str(ms): [] for ms in horizons_ms}
    for chunk in np.array_split(ids, max(1, (len(ids)+127)//128)):
        initial, actions, target = batch_windows(episodes, anchors, chunk, horizon)
        prediction = model.rollout(initial, actions)
        for ms in horizons_ms:
            h, block = round(ms/1000/dt), round(.02/dt)
            before_p = initial[:, 1] if h == block else prediction[:, h-block-1, 1]
            before_t = initial[:, 1] if h == block else target[:, h-block-1, 1]
            acceleration_error = ((prediction[:, h-1, 1]-before_p) - (target[:, h-1, 1]-before_t))/.02
            actual_acceleration = (target[:, h-1, 1]-before_t)/.02
            error = prediction[:, h-1, :20] - target[:, h-1]
            persistence = initial[:, 1] - target[:, h-1, 1]
            errors[str(ms)].append(torch.stack((error[:, 0], error[:, 1], acceleration_error,
                                                actual_acceleration, persistence), dim=1))
            previous_p = initial[:, 1] if h == 1 else prediction[:, h-2, 1]
            previous_t = initial[:, 1] if h == 1 else target[:, h-2, 1]
            raw_acceleration_errors[str(ms)].append(((prediction[:, h-1, 1]-previous_p) -
                                                      (target[:, h-1, 1]-previous_t))/dt)
    result = {}
    for ms, values in errors.items():
        all_errors = torch.cat(values)
        rmse = all_errors.square().mean(0).sqrt().tolist()
        result[ms] = dict(zip(("angle_rmse_rad", "velocity_rmse_rad_s", "last_20ms_acceleration_rmse_rad_s2",
                              "zero_20ms_acceleration_rmse_rad_s2", "persistence_velocity_rmse_rad_s"), rmse))
        result[ms].update(samples=len(all_errors), native_interval_acceleration_rmse_rad_s2=float(
            torch.cat(raw_acceleration_errors[ms]).square().mean().sqrt()))
    return result


def collect_fine(reference, output):
    original = json.loads((reference / "summary.json").read_text())
    records = json.loads((reference / "episodes.json").read_text())
    reference_config = PreacherCurlConfig(**original["contract"]["config"])
    checker = PreacherCurlEnv(reference_config)
    try:
        checker.reset(seed=0, options={"payload_mass": reference_config.payload_mass})
        if sensor_contract(checker) != original["contract"]:
            raise ValueError("Reference predictor physics/sensor contract changed")
    finally:
        checker.close()
    config = replace(reference_config, frame_skip=1)
    if not np.isclose(config.physics_dt, .001):
        raise ValueError("This comparison requires unchanged native 1 ms physics")
    episodes, probes, max_error = [], [], 0.
    env = PreacherCurlEnv(config)
    started = time.monotonic()
    try:
        with np.load(reference / "episodes.npz") as arrays:
            for i, record in enumerate(records):
                angle = float(np.deg2rad(record["initial_angle_deg"]))
                env.reset(seed=0, options={"q": angle, "qvel": 0., "target_angle": angle,
                    "trigger": 1., "capacity": record["initial_capacity"], "payload_mass": config.payload_mass})
                env.set_command(angle, 1.)
                commands = np.repeat(arrays[f"e{i}_actions"], 20, axis=0)
                states = [compact_state(env)]
                for step, command in enumerate(commands):
                    if record["split"] == "test" and step in (300, 1100):
                        probes.append({"episode_index": i, "step_1ms": step, "state": capture(env)})
                    _, _, terminated, truncated, _ = env.step(command)
                    if terminated:
                        raise RuntimeError(f"Fine replay physically failed: {record['id']} at {step}")
                    if truncated:
                        env._episode_done = False
                    states.append(compact_state(env))
                states = np.asarray(states)
                max_error = max(max_error, float(np.max(np.abs(states[::20, :20] - arrays[f"e{i}_states"][:, :20]))))
                episodes.append({**record, "states": states, "actions": commands})
                if (i+1) % 6 == 0:
                    print(f"replay {i+1}/{len(records)}; shared physical state max error={max_error:.3g}", flush=True)
    finally:
        env.close()
    with (output / "native_probes_1ms.pkl").open("wb") as stream:
        pickle.dump(probes, stream)
    return episodes, probes, config, {"episodes": len(episodes), "native_physics_steps": sum(len(e["actions"]) for e in episodes),
        "max_shared_physical_state_difference": max_error, "elapsed_s": time.monotonic()-started}


def train_time(initial, episodes, metadata, physical_scale, args, output):
    model = copy.deepcopy(initial)
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    dt = metadata["control_dt_s"]
    horizon = round(.12/dt)
    anchors = time_anchors(split["train"], .12, dt)
    generator = torch.Generator().manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_a = best_v = float("inf")
    selections, history = {}, []
    started = time.monotonic()
    for update in range(1, args.updates+1):
        ids = torch.randint(len(anchors), (args.batch_size,), generator=generator)
        state, action, target = batch_windows(split["train"], anchors, ids, horizon)
        prediction = model.rollout(state, action)
        loss = interval_loss(state, prediction, target, physical_scale, physical_scale[1]/.02, dt)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if update == 1 or update % args.check_every == 0 or update == args.updates:
            metrics = evaluate_time(model, split["validation"], dt, max_anchors=360)
            score_a = float(np.mean([v["last_20ms_acceleration_rmse_rad_s2"] for v in metrics.values()]))
            score_v = float(np.mean([v["velocity_rmse_rad_s"] for v in metrics.values()]))
            history.append({"update": update, "training_loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                            "acceleration_score": score_a, "velocity_score": score_v, "validation": metrics})
            for name, improved, criterion in (("acceleration", score_a < best_a, "last 20 ms interval acceleration"),
                                               ("velocity", score_v < best_v, "endpoint velocity")):
                if improved:
                    selections[name] = update
                    save_wide_predictor(output / f"{name}.pt", model, {**metadata, "update": update,
                        "seed": 0, "training_horizon_steps": horizon, "training_horizon_s": .12,
                        "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "selection": f"mean {criterion} RMSE at 20/60/120 ms; 360 fixed validation anchors",
                        "loss": "local native-interval acceleration and 19 state errors; common 20ms train-only scales",
                        "loss_physical_scale": physical_scale.tolist()})
            best_a, best_v = min(best_a, score_a), min(best_v, score_v)
            print(f"dt={dt:g} update={update} loss={float(loss.detach()):.5g} val_a={score_a:.5g} val_v={score_v:.5g}", flush=True)
    return {"updates": args.updates, "batch_size": args.batch_size, "training_anchors": len(anchors),
            "elapsed_s": time.monotonic()-started, "selected_updates": selections, "history": history}


def native_reference(probes, fine_episodes, config, output):
    """Vary the first *20 ms* excitation, equally for every network timestep."""
    kernel = NativeResponse(config, 120)
    jacobians = []
    try:
        for i, probe in enumerate(probes):
            commands = fine_episodes[probe["episode_index"]]["actions"][probe["step_1ms"]:probe["step_1ms"]+120]
            def response(first):
                kernel.restore(probe["state"])
                velocities = []
                for j, command in enumerate(commands):
                    kernel.step(first if j < 20 else command)
                    if (j+1) % 20 == 0:
                        velocities.append(float(kernel.env.data.qvel[kernel.env.dof_index]))
                return np.asarray(velocities)
            jac = []
            for muscle in range(6):
                high, low = commands[0].astype(float).copy(), commands[0].astype(float).copy()
                high[muscle] = min(1., high[muscle]+.005)
                low[muscle] = max(0., low[muscle]-.005)
                jac.append((response(high)-response(low))/(high[muscle]-low[muscle]))
            jacobians.append(np.stack(jac, axis=-1))
            if (i+1) % 6 == 0:
                print(f"native shared derivative {i+1}/{len(probes)}", flush=True)
        result = np.asarray(jacobians)
        np.save(output / "native_jacobian.npy", result)
        return result, kernel.control_steps
    finally:
        kernel.close()


def derivative_metrics(model, probes, episodes, frame_skip, native, output):
    estimated = []
    dt = frame_skip*.001
    block, horizon = round(.02/dt), round(.12/dt)
    for probe in probes:
        episode = episodes[probe["episode_index"]]
        start = probe["step_1ms"]//frame_skip
        state = torch.tensor(episode["states"][start])
        continuation = torch.tensor(episode["actions"][start:start+horizon])
        def response(first):
            actions = torch.cat((first.expand(block, -1), continuation[block:]), dim=0)
            return model.rollout(state, actions)[block-1::block, 1]
        jac = torch.autograd.functional.jacobian(response, continuation[0])
        estimated.append(jac.detach().numpy())
    predicted = np.asarray(estimated)
    np.save(output, predicted)
    def compare(a, b):
        meaningful = np.abs(a) > .05
        return {"rmse": float(np.sqrt(np.mean((a-b)**2))),
                "global_cosine": float(a.ravel()@b.ravel()/max(1e-12, np.linalg.norm(a)*np.linalg.norm(b))),
                "meaningful_sign_agreement": float(np.mean(np.sign(a[meaningful]) == np.sign(b[meaningful]))),
                "meaningful_elements": int(meaningful.sum())}
    return {"velocity_60_to_120ms": compare(native[:, 2:], predicted[:, 2:]),
            "mean_20ms_acceleration_60_to_120ms": compare(np.diff(native, axis=1)[:, 1:]/.02,
                                                                       np.diff(predicted, axis=1)[:, 1:]/.02),
            "pre_delay_max_abs": float(np.max(np.abs(predicted[:, :2])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/d1_dynamics_sensors"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_dynamics_timestep"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--check-every", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    fine, probes, config, collection = collect_fine(args.reference, args.output_dir)
    native, native_steps = native_reference(probes, fine, config, args.output_dir)
    summary = {"schema": "myo.dynamics_timestep_experiment.v1", "status": "training",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "collection": collection, "native_audit_physics_steps": native_steps,
        "command_hold_s": .02, "anchor_stride_s": .02, "training_horizon_s": .12,
        "sensor_history": "one preceding fiber-length sample; mean velocity window equals model timestep",
        "derivative_command_duration_s": .02, "models": {}}
    def report():
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    report()
    physical_scale = None
    for frame_skip, history in ((20, False), (5, False), (1, False), (1, True)):
        name = f"{frame_skip}ms" + ("_history" if history else "")
        output = args.output_dir / name
        output.mkdir()
        episodes = [resample_episode(e, frame_skip) for e in fine]
        if history:
            episodes = [add_delivered_history(e, frame_skip*.001) for e in episodes]
        np.savez_compressed(output / "episodes.npz", **{f"e{i}_{key}": e[key]
            for i,e in enumerate(episodes) for key in ("states", "actions")})
        (output / "episodes.json").write_text(json.dumps([{k:v for k,v in e.items() if k not in ("states","actions")}
                                                         for e in episodes], indent=2)+'\n')
        env = PreacherCurlEnv(replace(config, frame_skip=frame_skip))
        try:
            env.reset(seed=0, options={"payload_mass": config.payload_mass})
            metadata = history_contract(env) if history else sensor_contract(env)
            torch.manual_seed(0)
            cls = HistoryTransitionModel if history else SensorTransitionModel
            model = cls(queue_steps=metadata["queue_steps"], control_dt=env.control_dt)
            model.configure_geometry(env)
        finally:
            env.close()
        model.fit_normalization([e for e in episodes if e["split"] == "train"])
        if physical_scale is None:
            physical_scale = model.delta_scale.clone()
        summary["models"][name] = {"contract": metadata, "state_size": len(metadata["state_names"]),
            "neural_input_size": model.net[0].in_features, "parameters": sum(p.numel() for p in model.parameters()),
            "training": train_time(model, episodes, metadata, physical_scale, args, output)}
        report()
        for selection in ("acceleration", "velocity"):
            loader = load_history_predictor if history else load_sensor_predictor
            trained, _ = loader(output / f"{selection}.pt", metadata)
            metrics = evaluate_time(trained, [e for e in episodes if e["split"] == "test"],
                                    frame_skip*.001, (20, 60, 120, 400))
            derivatives = derivative_metrics(trained, probes, episodes, frame_skip, native, output/f"jacobian_{selection}.npy")
            summary["models"][name][selection] = {"test": metrics, "derivatives": derivatives}
            report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    summary["limitations"] = ["Predictors only; demo and motor policy unchanged", "One seed and finite budget, not convergence",
        "Commands remain held 20 ms; no claim about arbitrary 1 ms switching",
        "Delivered history at 20/40/60ms tested only at 1ms; activation is already observed",
        "Native-interval acceleration labels and fiber-velocity windows change with timestep",
        f"Matched new control uses batch{args.batch_size}; earlier sensor run used batch128"]
    report()


if __name__ == "__main__":
    main()
