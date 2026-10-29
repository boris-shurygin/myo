"""Optional CPU PPO training for the analytic-reference muscle-arm baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Sequence

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.runtime import dynamics_digest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000, help="Total environment transitions, not a success guarantee.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_preacher_curl.zip"))
    parser.add_argument("--predictor", type=Path, default=None, help="Optional frozen error-predictor artifact.")
    parser.add_argument("--config", type=Path, default=None, help="JSON object matching PreacherCurlConfig fields.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; default avoids oversubscribing simulator workers.")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        import torch
    except ImportError as exc:
        raise SystemExit("PPO is optional. Install the project's train extra before using myo.train.") from exc
    torch.set_num_threads(args.threads)

    config = PreacherCurlConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else PreacherCurlConfig()
    base_env = PreacherCurlEnv(config=config)
    env = base_env
    predictor_bytes = None
    predictor_sha256 = None
    if args.predictor is not None:
        from myo.predictor import PredictorObservationWrapper

        env = PredictorObservationWrapper(env, args.predictor)
        # Keep the exact frozen artifact with the policy. Re-reading the
        # source after a long training run could capture a different model.
        predictor_bytes = args.predictor.read_bytes()
        predictor_sha256 = hashlib.sha256(predictor_bytes).hexdigest()
        if predictor_sha256 != env.predictor_sha256:
            env.close()
            raise ValueError("Predictor artifact changed while it was being loaded.")
    check_env(env, warn=True, skip_render_check=True)
    observation_names = list(env.observation_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitored = Monitor(env)
    # A compact MLP PPO baseline is normally a CPU workload. No experiment
    # tracker, network service, or automatic model download is involved.
    model = PPO(
        "MlpPolicy", monitored, device="cpu", seed=args.seed,
        policy_kwargs={"net_arch": [64, 64]}, n_steps=128, batch_size=64,
        learning_rate=3e-4, verbose=1,
    )
    try:
        model.learn(total_timesteps=args.steps)
        model.save(str(args.output))
        actual_output = args.output if args.output.suffix == ".zip" else Path(str(args.output) + ".zip")
        bundled_predictor = None
        if predictor_bytes is not None:
            bundled_predictor = Path(str(actual_output) + ".predictor.json")
            bundled_predictor.write_bytes(predictor_bytes)
        metadata = {
            "schema_version": 1,
            "task": "analytic_reference_preacher_curl",
            "algorithm": "stable_baselines3.PPO",
            "seed": args.seed,
            "torch_threads": args.threads,
            "requested_training_steps": args.steps,
            "trained_steps": int(model.num_timesteps),
            "config": asdict(base_env.config),
            "dynamics_sha256": dynamics_digest(),
            "observation_names": observation_names,
            "predictor_path": bundled_predictor.name if bundled_predictor else None,
            "predictor_sha256": predictor_sha256,
            "package_versions": {name: version(name) for name in ("mujoco", "gymnasium", "numpy", "stable-baselines3", "torch")},
            "source": "PPO analytic-reference pipeline smoke; not MuscleMimic reproduction",
            "limitations": "Toy 1-DOF tracking baseline; not MuscleMimic, a validated human model, or proof of convergence.",
        }
        Path(str(actual_output) + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        monitored.close()


if __name__ == "__main__":
    main()
