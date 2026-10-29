"""Causal future-error prediction under a data-generating control policy.

The small NumPy network is deliberately separate from PPO. It predicts future
tracking error from the current base observation; future states are used only
as supervised labels. It is not an action-conditioned dynamics/world model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .runtime import config_digest, dynamics_digest, load_controller


OUTPUT_NAMES = ["predicted_tracking_error_rad", "predicted_tracking_error_rad_s"]


def horizon_steps(horizon_s: float, control_dt: float, motor_delay: float) -> int:
    if (not math.isfinite(control_dt) or control_dt <= 0
            or not math.isfinite(motor_delay) or motor_delay < 0):
        raise ValueError("Control dt must be positive and motor delay nonnegative.")
    if not math.isfinite(horizon_s) or horizon_s <= 0 or horizon_s < motor_delay:
        raise ValueError("Prediction horizon must be finite, positive and at least the motor delay.")
    ratio = horizon_s / control_dt
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Prediction horizon must be an integer number of control steps.")
    return steps


def make_episode_samples(observations: np.ndarray, errors: np.ndarray, steps: int):
    """Inputs at t and labels at t+H, formed inside exactly one episode."""
    observations = np.asarray(observations, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if (observations.ndim != 2 or errors.shape != (len(observations), 2)
            or steps < 1 or len(observations) <= steps):
        raise ValueError("Episode is too short or has incompatible observation/error shapes.")
    if not np.isfinite(observations).all() or not np.isfinite(errors).all():
        raise ValueError("Predictor data must be finite.")
    return observations[:-steps].copy(), errors[steps:].copy(), errors[:-steps].copy()


def split_episodes(episode_ids: np.ndarray, seed: int = 0):
    ids = np.asarray(episode_ids)
    if ids.ndim != 1:
        raise ValueError("Episode ids must be a vector.")
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("At least two whole episodes are required for train/validation separation.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    validation_count = max(1, int(math.ceil(len(unique) * 0.2)))
    validation = np.isin(ids, shuffled[:validation_count])
    return ~validation, validation


class ErrorPredictor:
    """Frozen one-hidden-layer tanh network with saved feature normalization."""

    def __init__(self, artifact: dict):
        if artifact.get("schema") != "myo.error_predictor.v1":
            raise ValueError("Unsupported predictor artifact schema.")
        self.artifact = artifact
        self.observation_names = list(artifact["observation_names"])
        self.horizon_s = float(artifact["horizon_s"])
        if not math.isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError("Invalid predictor horizon.")
        self.x_mean = np.asarray(artifact["x_mean"], dtype=np.float64)
        self.x_scale = np.asarray(artifact["x_scale"], dtype=np.float64)
        self.y_mean = np.asarray(artifact["y_mean"], dtype=np.float64)
        self.y_scale = np.asarray(artifact["y_scale"], dtype=np.float64)
        self.w1, self.b1, self.w2, self.b2 = [
            np.asarray(artifact[name], dtype=np.float64) for name in ("w1", "b1", "w2", "b2")
        ]
        n = len(self.observation_names)
        if (self.w1.ndim != 2 or self.w1.shape[0] != n
                or self.b1.shape != (self.w1.shape[1],)
                or self.w2.shape != (self.w1.shape[1], 2) or self.b2.shape != (2,)
                or self.x_mean.shape != (n,) or self.x_scale.shape != (n,)
                or self.y_mean.shape != (2,) or self.y_scale.shape != (2,)):
            raise ValueError("Predictor tensor dimensions do not match its observation contract.")
        for value in (self.x_mean, self.x_scale, self.y_mean, self.y_scale,
                      self.w1, self.b1, self.w2, self.b2):
            if not np.isfinite(value).all():
                raise ValueError("Predictor tensors must be finite.")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Normalization scales must be positive.")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.observation_names):
            raise ValueError("Predictor input has the wrong feature shape.")
        if not np.isfinite(values).all():
            raise ValueError("Predictor input must be finite.")
        hidden = np.tanh(((values - self.x_mean) / self.x_scale) @ self.w1 + self.b1)
        result = (hidden @ self.w2 + self.b2) * self.y_scale + self.y_mean
        if not np.isfinite(result).all():
            raise FloatingPointError("Predictor produced a non-finite output.")
        return result


class PredictorObservationWrapper(gym.ObservationWrapper):
    """Append a frozen prediction; inference never simulates or reads the future."""

    def __init__(self, env: gym.Env, predictor_path: Path | str):
        super().__init__(env)
        path = Path(predictor_path)
        self.predictor = ErrorPredictor.load(path)
        self.predictor_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        base = env.unwrapped
        if self.predictor.observation_names != list(base.observation_names):
            raise ValueError("Predictor observation order does not match this environment.")
        if self.predictor.artifact["config_sha256"] != config_digest(base.config):
            raise ValueError("Predictor was collected with a different environment configuration.")
        if self.predictor.artifact.get("dynamics_sha256") != dynamics_digest():
            raise ValueError("Predictor dynamics source/model fingerprint does not match this installation.")
        horizon_steps(self.predictor.horizon_s, base.control_dt, base.config.motor_delay)
        self.observation_names = list(base.observation_names) + OUTPUT_NAMES
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        prediction = self.predictor.predict(observation)
        with np.errstate(over="raise", invalid="raise"):
            result = np.concatenate((observation, prediction)).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Wrapped predictor observation is non-finite.")
        return result


def fit_predictor(
    x: np.ndarray, y: np.ndarray, episode_ids: np.ndarray, metadata: dict,
    epochs: int = 100, hidden: int = 32, seed: int = 0,
    current_error: np.ndarray | None = None,
) -> tuple[ErrorPredictor, dict]:
    """Fit using train-only normalization and whole-episode validation."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x), 2) or len(episode_ids) != len(x)
            or x.shape[1] != len(metadata["observation_names"])):
        raise ValueError("Incompatible predictor training arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or epochs < 1 or hidden < 1:
        raise ValueError("Training data must be finite and epochs/hidden positive.")
    train, validation = split_episodes(episode_ids, seed)
    xm, xs = x[train].mean(axis=0), x[train].std(axis=0)
    ym, ys = y[train].mean(axis=0), y[train].std(axis=0)
    xs[xs < 1e-6], ys[ys < 1e-6] = 1.0, 1.0
    xt, yt = (x[train] - xm) / xs, (y[train] - ym) / ys
    xv, yv = (x[validation] - xm) / xs, (y[validation] - ym) / ys
    rng = np.random.default_rng(seed)
    weights = [
        rng.normal(0, 1 / math.sqrt(x.shape[1]), (x.shape[1], hidden)),
        np.zeros(hidden), rng.normal(0, 1 / math.sqrt(hidden), (hidden, 2)), np.zeros(2),
    ]
    first, second = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    best_loss, best_weights, best_epoch, update = math.inf, None, 0, 0
    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        for begin in range(0, len(xt), 256):
            indices = order[begin:begin + 256]
            xb, yb = xt[indices], yt[indices]
            w1, b1, w2, b2 = weights
            h = np.tanh(xb @ w1 + b1)
            output = h @ w2 + b2
            derivative = 2 * (output - yb) / output.size
            dh = (derivative @ w2.T) * (1 - h * h)
            gradients = [xb.T @ dh, dh.sum(axis=0), h.T @ derivative, derivative.sum(axis=0)]
            update += 1
            for i, gradient in enumerate(gradients):
                first[i] = 0.9 * first[i] + 0.1 * gradient
                second[i] = 0.999 * second[i] + 0.001 * gradient * gradient
                weights[i] -= 0.002 * (first[i] / (1 - 0.9**update)) / (
                    np.sqrt(second[i] / (1 - 0.999**update)) + 1e-8
                )
        prediction = np.tanh(xv @ weights[0] + weights[1]) @ weights[2] + weights[3]
        loss = float(np.mean((prediction - yv) ** 2))
        if loss < best_loss:
            best_loss, best_weights, best_epoch = loss, [w.copy() for w in weights], epoch + 1
    if best_weights is None or not math.isfinite(best_loss):
        raise FloatingPointError("Predictor training diverged.")
    artifact = {
        "schema": "myo.error_predictor.v1",
        "model_type": "tanh_mlp",
        "observation_names": metadata["observation_names"],
        "output_names": OUTPUT_NAMES,
        "horizon_s": metadata["horizon_s"],
        "config_sha256": metadata["config_sha256"],
        "dynamics_sha256": metadata["dynamics_sha256"],
        "config": metadata["config"],
        "behavior_controller": metadata["behavior_controller"],
        "seed": seed,
        "x_mean": xm.tolist(), "x_scale": xs.tolist(),
        "y_mean": ym.tolist(), "y_scale": ys.tolist(),
        **{name: value.tolist() for name, value in zip(("w1", "b1", "w2", "b2"), best_weights)},
    }
    model = ErrorPredictor(artifact)
    predicted = model.predict(x[validation])
    report = {
        "train_samples": int(train.sum()), "validation_samples": int(validation.sum()),
        "train_episodes": np.unique(np.asarray(episode_ids)[train]).tolist(),
        "validation_episodes": np.unique(np.asarray(episode_ids)[validation]).tolist(),
        "best_epoch": best_epoch,
        "validation_rmse_rad_rad_s": np.sqrt(np.mean((predicted - y[validation]) ** 2, axis=0)).tolist(),
        "zero_prediction_rmse_rad_rad_s": np.sqrt(np.mean(y[validation] ** 2, axis=0)).tolist(),
        "warning": "Validation is under the data-generating policy. Re-evaluate after changing the controller; no independent test set here.",
    }
    if current_error is not None:
        current_error = np.asarray(current_error)
        if current_error.shape != y.shape or not np.isfinite(current_error).all():
            raise ValueError("Current-error baseline must match the label array.")
        report["persistent_error_rmse_rad_rad_s"] = np.sqrt(
            np.mean((current_error[validation] - y[validation]) ** 2, axis=0)
        ).tolist()
    artifact["validation"] = report
    return model, report


def collect_data(args) -> None:
    if args.episodes < 2:
        raise ValueError("Collect at least two episodes for a grouped validation split.")
    env, action, provenance = load_controller(args.controller, args.config, args.predictor)
    base = env.unwrapped
    try:
        steps = horizon_steps(args.horizon, base.control_dt, base.config.motor_delay)
        inputs, labels, current, ids = [], [], [], []
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_obs, episode_errors = [], []
            done = False
            while True:
                # Always collect base features even when the behavior policy
                # itself has a predictor. This avoids recursive predictor input.
                episode_obs.append(observation[:len(base.observation_names)].copy())
                episode_errors.append(base.telemetry()["tracking_error"])
                if done:
                    break
                observation, _, terminated, truncated, _ = env.step(action(observation))
                done = terminated or truncated
            x, y, e = make_episode_samples(np.asarray(episode_obs), np.asarray(episode_errors), steps)
            inputs.append(x)
            labels.append(y)
            current.append(e)
            ids.append(np.full(len(x), episode, dtype=np.int64))
        metadata = {
            "schema": "myo.predictor_dataset.v1",
            "observation_names": list(base.observation_names),
            "horizon_s": args.horizon,
            "config": asdict(base.config),
            "config_sha256": config_digest(base.config),
            "dynamics_sha256": dynamics_digest(),
            "behavior_controller": args.controller,
            "behavior_policy_metadata": provenance,
            "seed": args.seed,
        }
        if args.controller not in ("heuristic", "zero"):
            metadata["behavior_policy_sha256"] = hashlib.sha256(Path(args.controller).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Open explicitly so NumPy cannot silently append an extra .npz suffix.
        with args.output.open("wb") as handle:
            np.savez_compressed(handle, x=np.concatenate(inputs), y=np.concatenate(labels),
                                current_error=np.concatenate(current), episode_id=np.concatenate(ids),
                                metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        print(json.dumps({"dataset": str(args.output), "episodes": args.episodes,
                          "samples": sum(map(len, inputs)), "horizon_s": args.horizon}, indent=2))
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Collect causal supervised pairs from complete rollouts.")
    collect.add_argument("--controller", default="heuristic")
    collect.add_argument("--config", type=Path)
    collect.add_argument("--predictor", type=Path, help="Existing predictor required by the behavior controller, if any.")
    collect.add_argument("--episodes", type=int, default=12)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--horizon", type=float, default=0.1)
    collect.add_argument("--output", type=Path, default=Path("runs/predictor_data.npz"))
    train = commands.add_parser("train", help="Fit a small NumPy error predictor with whole-episode validation.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("runs/predictor.json"))
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect_data(args)
    else:
        with np.load(args.data, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("schema") != "myo.predictor_dataset.v1":
                raise ValueError("Unsupported predictor dataset schema.")
            model, report = fit_predictor(data["x"], data["y"], data["episode_id"], metadata,
                                          args.epochs, args.hidden, args.seed, data["current_error"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model.artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

