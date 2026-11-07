"""Compare sensory sampling rates without changing the 20 ms motor schedule."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .train_tissue_predictor import native_to_tissues, sensor_names
from .vertical_muscles import VerticalConfig, VerticalMuscles, heuristic


CONTROL_MS = 20
HISTORY_MS = 60


def collect_dense(variant, seconds=8.):
    """Observe each physical substep; recompute the heuristic only every 20 ms."""
    config = VerticalConfig(variant=variant)
    env = VerticalMuscles(replace(config, frame_skip=1))

    def physical_snapshot():
        values = [env.position, env.velocity]
        if config.elastic:
            values.extend(env.fiber_lengths())
        if config.activation_dynamic:
            values.extend(env.activation)
        return np.asarray(values)

    episodes = []
    for split, versions in (("train", (0, 1)), ("validation", (2,)), ("test", (3,))):
        for family in ("sine", "steps", "multisine"):
            for version in versions:
                position = (-.06, 0., .06, -.03)[version]
                env.reset(position)
                initial = env.excitation.copy()
                physical, actions = [physical_snapshot()], []
                for _ in range(round(seconds/config.dt)):
                    command = heuristic(env, family, version)
                    actions.append(command.astype(np.float32))
                    for _ in range(CONTROL_MS):
                        env.step(command)
                        physical.append(physical_snapshot())
                # Geometry is converted in float64 before rounding the sensor boundary.
                sensors = native_to_tissues(torch.tensor(np.asarray(physical), dtype=torch.float64), config)
                episodes.append({"id": f"{split}/{family}/v{version}", "split": split,
                    "family": family, "version": version, "initial_position_m": position,
                    "initial_excitation": initial.astype(np.float32),
                    "native_physical": np.asarray(physical),
                    "sensors": sensors.numpy().astype(np.float32),
                    "actions": np.asarray(actions)})
    return config, episodes


def save_episodes(path, episodes):
    metadata = [{k: v for k, v in e.items() if not isinstance(v, np.ndarray)} for e in episodes]
    arrays = {f"e{i}/{name}": e[name] for i, e in enumerate(episodes)
              for name in ("sensors", "actions", "initial_excitation", "native_physical")}
    np.savez_compressed(path/"episodes.npz", **arrays)
    (path/"episodes.json").write_text(json.dumps(metadata, indent=2)+'\n')


def read_episodes(path):
    metadata = json.loads((path/"episodes.json").read_text())
    with np.load(path/"episodes.npz") as data:
        return [{**e, **{name: data[f"e{i}/{name}"] for name in
                 ("sensors", "actions", "initial_excitation", "native_physical")}}
                for i, e in enumerate(metadata)]


def data_identity(path):
    return hashlib.sha256((path/"episodes.json").read_bytes()+(path/"episodes.npz").read_bytes()).hexdigest()


def sampling_windows(episodes, sensor_ms, transition_ms, horizon_ms, delayed, stride=1):
    """Starts always lie on the common command grid, regardless of sensor rate."""
    rows = {name: [] for name in ("history", "pending", "actions", "target")}
    command_count = (horizon_ms+CONTROL_MS-1)//CONTROL_MS
    ages = np.arange(0, HISTORY_MS+1, sensor_ms)
    times = np.arange(transition_ms, horizon_ms+1, transition_ms)
    for e in episodes:
        for start in range(0, len(e["actions"])-command_count+1, stride):
            tick = start*CONTROL_MS
            rows["history"].append(e["sensors"][np.maximum(0, tick-ages)])
            pending = [e["actions"][i] if i >= 0 else e["initial_excitation"]
                       for i in range(start-2, start)] if delayed else []
            rows["pending"].append(np.asarray(pending).reshape(-1, 2))
            rows["actions"].append(e["actions"][start:start+command_count])
            rows["target"].append(e["sensors"][tick+times])
    return {name: torch.tensor(np.asarray(values), dtype=torch.float32) for name, values in rows.items()}


def transition_windows(episodes, sensor_ms, transition_ms, delayed):
    """Warm-up includes every phase of a held command, not just command boundaries."""
    rows = {name: [] for name in ("history", "pending", "actions", "target")}
    ages = np.arange(0, HISTORY_MS+1, sensor_ms)
    for e in episodes:
        for tick in range(0, len(e["sensors"])-transition_ms, transition_ms):
            command = tick//CONTROL_MS
            rows["history"].append(e["sensors"][np.maximum(0, tick-ages)])
            pending = [e["actions"][i] if i >= 0 else e["initial_excitation"]
                       for i in range(command-2, command)] if delayed else []
            rows["pending"].append(np.asarray(pending).reshape(-1, 2))
            rows["actions"].append(e["actions"][command:command+1])
            rows["target"].append(e["sensors"][tick+transition_ms:tick+transition_ms+1])
    return {name: torch.tensor(np.asarray(values), dtype=torch.float32) for name, values in rows.items()}


class SamplingPredictor(nn.Module):
    def __init__(self, physical_size, sensor_ms, transition_ms, hidden=128):
        super().__init__()
        if sensor_ms not in (1, 5, 20) or transition_ms not in (1, 5, 20):
            raise ValueError("Supported sensor/transition intervals are 1, 5 and 20 ms")
        self.physical_size, self.sensor_ms, self.transition_ms = physical_size, sensor_ms, transition_ms
        self.architecture = dict(physical_size=physical_size, sensor_ms=sensor_ms,
                                 transition_ms=transition_ms, hidden=hidden)
        inputs = physical_size*(HISTORY_MS//sensor_ms+1)+2
        for name, size in (("input_mean", inputs), ("input_scale", inputs),
                           ("delta_mean", physical_size), ("delta_scale", physical_size),
                           ("common_scale", physical_size)):
            self.register_buffer(name, torch.ones(size) if "scale" in name else torch.zeros(size))
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, physical_size))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def delivered(pending, actions, transition_ms, horizon_ms):
        scheduled = torch.cat((pending, actions), dim=-2)
        return scheduled.repeat_interleave(CONTROL_MS//transition_ms, dim=-2)[..., :horizon_ms//transition_ms, :]

    def inputs(self, history, delivered):
        return torch.cat((history.flatten(-2), delivered), dim=-1)

    def fit_normalization(self, episodes, delayed):
        data = transition_windows(episodes, self.sensor_ms, self.transition_ms, delayed)
        delivered = self.delivered(data["pending"], data["actions"], self.transition_ms, self.transition_ms)[:, 0]
        x = self.inputs(data["history"], delivered)
        delta = data["target"][:, 0]-data["history"][:, 0]
        common = np.concatenate([e["sensors"][20::20]-e["sensors"][:-20:20] for e in episodes])
        for name, value in (("input_mean", x.mean(0)), ("input_scale", x.std(0, correction=0).clamp_min(1e-5)),
                            ("delta_mean", delta.mean(0)), ("delta_scale", delta.std(0, correction=0).clamp_min(1e-6)),
                            ("common_scale", torch.tensor(common.std(0)).clamp_min(1e-6))):
            getattr(self, name).copy_(value)

    def forward(self, history, delivered):
        x = (self.inputs(history, delivered)-self.input_mean)/self.input_scale
        return history[..., 0, :]+self.delta_mean+self.delta_scale*self.net(x)

    def rollout(self, history, pending, actions, horizon_ms=60):
        if self.transition_ms != self.sensor_ms:
            raise ValueError("Autoregression requires predictions at every sensory history tick")
        delivered = self.delivered(pending, actions, self.transition_ms, horizon_ms)
        result = []
        for step in range(delivered.shape[-2]):
            following = self(history, delivered[..., step, :])
            history = torch.cat((following.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            result.append(following)
        return torch.stack(result, dim=-2)


def model_contract(config, model, data_sha256):
    names, _ = sensor_names(config.activation_dynamic)
    sources = (Path(__file__), Path(__file__).with_name("vertical_muscles.py"),
               Path(__file__).with_name("muscle.py"), Path(__file__).with_name("delay.py"),
               Path(__file__).with_name("fatigue.py"), Path(__file__).with_name("train_tissue_predictor.py"))
    return {"schema": "myo.tissue_sampling.v1", "architecture": model.architecture,
        "config": asdict(config), "physics_dt_s": config.physics_dt, "command_dt_s": config.dt,
        "sensory_dt_s": model.sensor_ms/1000, "prediction_dt_s": model.transition_ms/1000,
        "history_duration_s": .06, "motor_delay_s": config.motor_delay_s if config.delayed else 0.,
        "physical_names": names, "neural_input_names":
            [f"history/{age}ms/{name}" for age in range(0, 61, model.sensor_ms) for name in names]+
            ["delivered_excitation/upper", "delivered_excitation/lower"],
        "action_mapping": ["upper excitation [0,1]", "lower excitation [0,1]"],
        "sensations": "local fiber/tendon relative lengths; activation only in variant 4; no derivatives or q/qvel",
        "history_reset": "repeat episode initial frame; newest first; own predictions during rollout",
        "transport": ("exact two-command FIFO at 20 ms; each delivered command held for 20 ms"
                      if config.delayed else "no transport delay; each command held for 20 ms"),
        "future_actions": "conditional recorded heuristic continuation, not future sensory observations",
        "normalization": "train only; transition delta STD; common 20 ms delta STD for BPTT and metrics",
        "warmup_sampling": "all transition-grid phases within a held command; BPTT starts on 20 ms grid",
        "data_sha256": data_sha256, "seed": 0,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def save_model(path, model, contract, update, stage):
    torch.save({"architecture": model.architecture, "state_dict": model.state_dict()}, path)
    metadata = {**contract, "update": update, "stage": stage,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')


def load_model(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Sampling checkpoint weights differ from metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Sampling checkpoint contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Sampling architecture differs from metadata")
    model = SamplingPredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata


def predict(model, data, autoregressive):
    if autoregressive:
        return model.rollout(data["history"], data["pending"], data["actions"],
                             data["target"].shape[1]*model.transition_ms)
    delivered = model.delivered(data["pending"], data["actions"], model.transition_ms, model.transition_ms)
    return model(data["history"], delivered[:, 0])[:, None]


@torch.no_grad()
def metrics(model, data, autoregressive):
    prediction = predict(model, data, autoregressive)
    times = (20, 40, 60) if autoregressive else (20,)
    result = {}
    for ms in times:
        index = ms//model.transition_ms-1
        error = (prediction[:, index]-data["target"][:, index]).double()
        previous = prediction[:, index-20//model.transition_ms] if ms > 20 else data["history"][:, 0]
        true_previous = data["target"][:, index-20//model.transition_ms] if ms > 20 else data["history"][:, 0]
        rate_error = ((prediction[:, index]-previous)-(data["target"][:, index]-true_previous)).double()/.02
        meters = torch.tensor([.48, .48, .12, .12], dtype=torch.float64)
        length_rmse = (error[:, :4]*meters).square().mean(0).sqrt()
        rate_rmse = (rate_error[:, :4]*meters).square().mean(0).sqrt()
        result[str(ms)] = {"samples": len(error),
            "channel_rmse": error.square().mean(0).sqrt().tolist(),
            "strain_rmse_over_20ms_delta_std": float((error[:, :4]/model.common_scale[:4]).square().mean().sqrt()),
            "fiber_length_rmse_mm": float(length_rmse[:2].square().mean().sqrt()*1000),
            "tendon_length_rmse_mm": float(length_rmse[2:].square().mean().sqrt()*1000),
            "fiber_rate_rmse_mm_s": float(rate_rmse[:2].square().mean().sqrt()*1000),
            "tendon_rate_rmse_mm_s": float(rate_rmse[2:].square().mean().sqrt()*1000)}
    result["score"] = float(np.mean([result[str(t)]["strain_rmse_over_20ms_delta_std"]
                                     for t in ((20, 60) if autoregressive else (20,))]))
    return result


def train_stage(config, episodes, model, args, output, stage, data_sha256):
    autoregressive = stage == "bptt"
    horizon = 60 if autoregressive else model.transition_ms
    split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
    train = (sampling_windows(split["train"], model.sensor_ms, model.transition_ms, horizon, config.delayed)
             if autoregressive else transition_windows(split["train"], model.sensor_ms, model.transition_ms, config.delayed))
    # Direct models evaluate only 20 ms; microstep warm-up evaluates its own rollout to 60 ms.
    can_roll = model.sensor_ms == model.transition_ms
    val_horizon = 60 if can_roll else 20
    evaluation = {s: sampling_windows(split[s], model.sensor_ms, model.transition_ms,
                  val_horizon, config.delayed, stride=5) for s in split}
    contract = model_contract(config, model, data_sha256)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(0)
    maximum = args.max_bptt_updates if autoregressive else args.one_step_updates
    best, significant, last_significant = float("inf"), float("inf"), 0
    history, reason = [], "update_budget"
    started = time.monotonic()
    for update in range(1, maximum+1):
        ids = torch.randint(len(train["history"]), (128,), generator=generator)
        batch = {k: v[ids] for k, v in train.items()}
        prediction = predict(model, batch, autoregressive)
        target = batch["target"]
        if autoregressive:
            prediction = prediction[:, 20//model.transition_ms-1::20//model.transition_ms]
            target = target[:, 20//model.transition_ms-1::20//model.transition_ms]
        scale = model.common_scale if autoregressive else model.delta_scale
        loss = ((prediction-target)/scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite sampling loss")
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if update == 1 or update % 200 == 0 or update == maximum:
            val = metrics(model, evaluation["validation"], can_roll)
            fitted = metrics(model, evaluation["train"], can_roll)
            score = val["score"] if autoregressive else val["20"]["strain_rmse_over_20ms_delta_std"]
            if score < best:
                best = score
                save_model(output/f"{stage}.pt", model, contract, update, stage)
            if score < significant*.99:
                significant, last_significant = score, update
            history.append({"update": update, "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(norm), "validation_score": score,
                "train_fixed": fitted, "validation": val})
            print(f"v{config.variant} sensor={model.sensor_ms} step={model.transition_ms} {stage} "
                  f"{update}/{maximum} val={score:.6g}", flush=True)
            if autoregressive and update >= 1000 and update-last_significant >= args.plateau_patience:
                reason = "validation_plateau"
                break
    selected, metadata = load_model(output/f"{stage}.pt", contract)
    return selected, {"contract": contract, "best_update": metadata["update"], "updates": update,
        "stop_reason": reason, "unique_windows": len(train["history"]), "sampled_windows": update*128,
        "predicted_transitions": update*128*(horizon//model.transition_ms),
        "parameters": sum(p.numel() for p in model.parameters()), "history": history,
        "elapsed_s": time.monotonic()-started,
        "validation": metrics(selected, evaluation["validation"], can_roll),
        "test": metrics(selected, evaluation["test"], can_roll)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tissue_sampling"))
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--one-step-updates", type=int, default=1000)
    parser.add_argument("--max-bptt-updates", type=int, default=2000)
    parser.add_argument("--plateau-patience", type=int, default=600)
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema": "myo.tissue_sampling_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "scope": "raw local sensory sampling only; fixed 20ms commands; no actor or critic", "variants": {}}
    started = time.monotonic()

    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    report()
    for variant in args.variants:
        folder = args.output_dir/f"variant_{variant}"
        folder.mkdir()
        config, episodes = collect_dense(variant, args.seconds)
        save_episodes(folder, episodes)
        data_sha256 = data_identity(folder)
        record = {"data_sha256": data_sha256, "config": asdict(config), "direct": {}, "rollout": {}}
        summary["variants"][str(variant)] = record
        for sensor_ms in (20, 5, 1):
            torch.manual_seed(0)
            model = SamplingPredictor(6 if config.activation_dynamic else 4, sensor_ms, 20)
            model.fit_normalization([e for e in episodes if e["split"] == "train"], config.delayed)
            output = folder/f"direct_{sensor_ms}ms"
            output.mkdir()
            selected, result = train_stage(config, episodes, model, args, output, "one_step", data_sha256)
            record["direct"][str(sensor_ms)] = result
            report()
            output = folder/f"rollout_{sensor_ms}ms"
            output.mkdir()
            warm = None
            if sensor_ms == 20:
                model = copy.deepcopy(selected)
            else:
                torch.manual_seed(0)
                model = SamplingPredictor(model.physical_size, sensor_ms, sensor_ms)
                model.fit_normalization([e for e in episodes if e["split"] == "train"], config.delayed)
                model, warm = train_stage(config, episodes, model, args, output, "one_step", data_sha256)
            _, result = train_stage(config, episodes, model, args, output, "bptt", data_sha256)
            record["rollout"][str(sensor_ms)] = {"warmup": warm, **result}
            report()
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
