"""Fixed-episode motor exploration and the unchanged detached predictor network."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .detached_vertical import DetachedVerticalEnv, make_model
from .shared_vertical import SharedPolicy, mlp
from .vertical_muscles import VerticalConfig
from .vertical_precision import VerticalPrecisionEnv


FAMILIES = ("independent", "reciprocal", "common", "multisine")
SCALES = np.array([.1, .5, .1, .1, .5, .5])
DESIGN = {"schema": "myo.body_predictor.v1", "architecture": "8->4x128 SiLU->linear6",
    "target": "normalized next physical delta at20ms; exact non-neural FIFO",
    "normalization": "all complete train transitions only; delta STD floor1e-5; frozen",
    "inputs": "original normalized physical6 and pending/0 excitation2; no new sensors/history",
    "data": "four smooth open-loop motor families; fixed whole-episode splits; no outcome filtering",
    "training": "RMSprop lr3e-4 alpha.99 eps1e-5; batch128; clip.5; one-step MSE all six channels"}


def contract():
    env = VerticalPrecisionEnv(VerticalConfig(variant=4))
    result = {"design": DESIGN, "environment": env.contract(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "detached_source_sha256": hashlib.sha256(Path(__file__).with_name("detached_vertical.py").read_bytes()).hexdigest(),
        "input_names": env.observation_names[:8], "output_names": env.observation_names[:6],
        "observation_order": env.observation_names, "action_mapping": "two excitations in[0,1]",
        "config": asdict(env.physical.config)}
    env.close()
    return result


class BodyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp(8, (128,)*4)
        self.head = nn.Linear(128, 6)
        self.register_buffer("delta_mean", torch.zeros(6))
        self.register_buffer("delta_std", torch.ones(6))
        self.register_buffer("delta_calibrated", torch.tensor(False))

    @classmethod
    def from_current_initialization(cls):
        # Reuse actual construction/initialization, not an approximately similar MLP.
        env = DetachedVerticalEnv()
        original = make_model(env, "online").policy
        result = cls()
        result.encoder.load_state_dict(original.mlp_extractor.predictor.state_dict())
        result.head.load_state_dict(original.prediction_net.state_dict())
        env.close()
        return result

    @torch.no_grad()
    def fit_normalization(self, observations, labels):
        if bool(self.delta_calibrated):
            raise RuntimeError("Predictor normalization is already frozen")
        delta = labels[:, :6]-observations[:, :6]
        self.delta_mean.copy_(delta.mean(0))
        self.delta_std.copy_(delta.std(0, unbiased=False).clamp_min(1e-5))
        self.delta_calibrated.fill_(True)

    def normalized_delta(self, observations):
        return self.head(self.encoder(observations[:, :8]))

    def next_observation(self, observations, actions):
        return SharedPolicy.next_observation(self, observations, actions)

    @torch.no_grad()
    def copy_into_policy(self, policy):
        """Copy into a freshly constructed policy; PPO behavior is not preserved."""
        policy.mlp_extractor.predictor.load_state_dict(self.encoder.state_dict())
        policy.prediction_net.load_state_dict(self.head.state_dict())
        policy.delta_mean.copy_(self.delta_mean)
        policy.delta_std.copy_(self.delta_std)
        policy.delta_calibrated.copy_(self.delta_calibrated)
        # Any future predictor training starts with a new optimizer history.
        policy.predictor_optimizer.state.clear()


def smooth_knots(rng, times, low, high):
    knots = [0.]
    values = [rng.uniform(low, high)]
    while knots[-1] < times[-1]+1.:
        knots.append(knots[-1]+rng.uniform(.2, .8))
        values.append(rng.uniform(low, high))
    knots, values = np.asarray(knots), np.asarray(values)
    index = np.searchsorted(knots, times, side="right")-1
    phase = np.clip((times-knots[index])/(knots[index+1]-knots[index]), 0, 1)
    phase = phase*phase*(3-2*phase)
    return values[index]+phase*(values[index+1]-values[index])


def motor_commands(family, steps, seed, limit, initial):
    rng = np.random.default_rng(seed)
    times = np.arange(steps)*.02
    if family == "independent":
        commands = np.stack([smooth_knots(rng, times, 0., limit) for _ in range(2)], axis=-1)
    elif family == "reciprocal":
        difference = smooth_knots(rng, times, -.5*limit, .5*limit)
        common = smooth_knots(rng, times, .35*limit, .65*limit)
        commands = np.stack((common+difference, common-difference), axis=-1)
    elif family == "common":
        common = smooth_knots(rng, times, 0., limit)
        difference = smooth_knots(rng, times, -.06*limit, .06*limit)
        commands = np.stack((common+difference, common-difference), axis=-1)
    elif family == "multisine":
        frequency = rng.uniform(.12, 1.5, (2, 3))
        phase = rng.uniform(0., 2*np.pi, (2, 3))
        commands = .5*limit+limit/6*np.sin(2*np.pi*times[:, None, None]*frequency+phase).sum(-1)
    else:
        raise ValueError("Unknown motor family")
    commands = np.clip(commands, 0, limit)
    ramp = np.clip(times/.2, 0, 1)
    ramp = (ramp*ramp*(3-2*ramp))[:, None]
    return ((1-ramp)*np.asarray(initial)+ramp*commands).astype(np.float32)


def specifications():
    result = []
    for split, count, positions, levels in (
        ("train", 8, (-.1, -.03, .03, .1), (.15, .3, .5, .8)),
        ("validation", 3, (-.08, 0., .08), (.2, .4, .7)),
        ("test", 3, (-.12, -.02, .12), (.25, .45, .75))):
        for family_index, family in enumerate(FAMILIES):
            for index in range(count):
                result.append({"id": f"{split}/{family}/{index}", "split": split, "family": family,
                    "position_m": positions[(index+family_index) % len(positions)],
                    "limit": levels[index % len(levels)], "seed": 20261107+len(result)})
    return result


def collect(root, *, seconds=12., specs=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"dataset.npz").exists():
        raise FileExistsError(root/"dataset.npz")
    arrays = {key: [] for key in ("observations", "actions", "next_observations", "forces_N", "tendon_strain")}
    episodes, offset = [], 0
    for spec in specifications() if specs is None else specs:
        env = VerticalPrecisionEnv(VerticalConfig(variant=4), seconds)
        obs, _ = env.reset(options={"initial_position_m": spec["position_m"], "target_m": 0.})
        commands = motor_commands(spec["family"], round(seconds/.02), spec["seed"], spec["limit"], env.previous)
        count, failed = 0, False
        for command in commands:
            forces = env.physical.applied_forces.copy()
            strain = (env.physical.path_lengths()-env.physical.fiber_lengths()-.12)/.12
            following, _, failed, truncated, _ = env.step(command)
            if failed:
                # A failure can stop within a20ms interval: do not label it as a complete step.
                break
            for key, value in zip(arrays, (obs, command, following, forces, strain), strict=True):
                arrays[key].append(value.copy())
            obs = following
            count += 1
            if truncated:
                break
        episodes.append({**spec, "start": offset, "stop": offset+count,
            "complete_transitions": count, "physical_failure": bool(failed),
            "physical_duration_s": env.physical.time, "requested_duration_s": seconds})
        offset += count
        env.close()
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    path = root/"dataset.npz"
    np.savez_compressed(path, **arrays)
    metadata = {"contract": contract(), "episodes": episodes,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection": "all valid complete prefixes retained; no rejection by outcome"}
    (root/"dataset.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def load_dataset(root):
    root = Path(root)
    metadata = json.loads((root/"dataset.json").read_text())
    if metadata["contract"] != contract():
        raise ValueError("Motor dataset contract mismatch")
    path = root/"dataset.npz"
    if metadata["dataset_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Motor dataset hash mismatch")
    arrays = dict(np.load(path))
    return arrays, metadata


def indices(metadata, split, horizon=1, family=None):
    return np.concatenate([np.arange(e["start"], e["stop"]-horizon+1, dtype=np.int64)
        for e in metadata["episodes"] if e["split"] == split and (family is None or e["family"] == family)])


@torch.no_grad()
def forecast_metrics(model, data, starts, horizon=1, *, scales=None):
    sums, persistence = torch.zeros(6), torch.zeros(6)
    for block in np.array_split(starts, max(1, int(np.ceil(len(starts)/2048)))):
        initial = torch.as_tensor(data["observations"][block])
        current = initial
        for step in range(horizon):
            current = model.next_observation(current, torch.as_tensor(data["actions"][block+step]))
        actual = torch.as_tensor(data["next_observations"][block+horizon-1, :6])
        sums += (current[:, :6]-actual).square().sum(0)
        persistence += (initial[:, :6]-actual).square().sum(0)
    mse, pmse = sums/len(starts), persistence/len(starts)
    delta_std = model.delta_std if scales is None else scales
    rms = mse.sqrt().numpy()*SCALES
    prms = pmse.sqrt().numpy()*SCALES
    return {"samples": len(starts), "horizon_ms": 20*horizon,
        "physical_channel_rmse": rms.tolist(), "persistence_physical_channel_rmse": prms.tolist(),
        "q_rmse_mm": float(rms[0]*1000), "v_rmse_mm_s": float(rms[1]*1000),
        "normalized_mse": float((mse/delta_std.square()).mean()),
        "persistence_normalized_mse": float((pmse/delta_std.square()).mean()),
        "relative_channel_rmse_to_persistence": (mse/pmse.clamp_min(1e-20)).sqrt().tolist()}


def save_predictor(path, model, optimizer, update, metadata, training):
    path = Path(path)
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "update": update}, path)
    description = {"contract": contract(), "dataset_sha256": metadata["dataset_sha256"],
        "training": training, "update": update,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(description, indent=2, allow_nan=False)+'\n')


def load_predictor(path):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["contract"] != contract():
        raise ValueError("Body predictor contract mismatch")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Body predictor checkpoint hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["update"] != metadata["update"]:
        raise ValueError("Body predictor update mismatch")
    model = BodyPredictor()
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved["optimizer_state_dict"], metadata
