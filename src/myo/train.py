"""Local CPU muscle-control training with optional causal imitation initialization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.runtime import dynamics_digest, load_controller, policy_contract


def warm_start_actor(model, env, steps: int, seed: int) -> dict:
    """Fit the same actor to causal diagnostic demonstrations before PPO.

    Split by episode, never neighboring samples. The teacher executes the identical delayed
    actuators; neither future states nor extra forces enter observations.
    """
    import torch
    observation, _ = env.reset(seed=seed)
    observations, actions, episodes = [], [], []
    episode = 0
    for _ in range(steps):
        excitation = env.unwrapped.baseline_action()
        observations.append(observation.copy())
        actions.append(excitation.copy())
        episodes.append(episode)
        observation, _, terminated, truncated, _ = env.step(excitation)
        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=seed + episode)
    groups = np.asarray(episodes)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Warm start needs at least three episodes; increase --warm-start-steps.")
    validation = groups >= unique[-2]
    train_indices = np.flatnonzero(~validation)
    x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    random = np.random.default_rng(seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(100):
        for indices in np.array_split(random.permutation(train_indices), max(1, len(train_indices) // 256)):
            mean = model.policy.get_distribution(x[indices]).distribution.mean
            loss = torch.mean((mean - y[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            mean = model.policy.get_distribution(x[validation]).distribution.mean
            validation_loss = float(torch.mean((mean - y[validation]) ** 2))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    model.policy.load_state_dict(best_state)
    env.reset(seed=seed)
    return {"teacher": "diagnostic_heuristic", "demonstration_steps": steps, "seed": seed,
            "train_episodes": int(len(unique) - 2), "validation_episodes": 2,
            "validation_excitation_rmse": best_loss ** 0.5,
            "epochs": 100, "selected_epoch": best_epoch,
            "selection": "minimum held-out episode action MSE",
            "limitation": "Imitation of an engineering heuristic; no independent protection guarantee."}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000, help="Total environment transitions, not a success guarantee.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_preacher_curl.zip"))
    parser.add_argument("--predictor", type=Path, default=None, help="Optional frozen error-predictor artifact.")
    parser.add_argument("--config", type=Path, default=None, help="JSON object matching PreacherCurlConfig fields.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; default avoids oversubscribing simulator workers.")
    parser.add_argument("--protocol", choices=("default", "compact"), default="default",
                        help="compact: fixed trigger/capacity grid, sweep and single moves, result-based PPO reward.")
    parser.add_argument("--warm-start-steps", type=int, default=0,
                        help="Optional causal heuristic demonstrations to initialize this same actor.")
    parser.add_argument("--log-std-init", type=float, default=0.0,
                        help="Initial Gaussian log standard deviation of the PPO actor.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Optional PPO update early stopping at excessive approximate KL divergence.")
    parser.add_argument("--command-imitation", action="store_true",
                        help="Train normalized actor on holds/extremes/releases and corrective actor rollouts.")
    parser.add_argument("--imitation-rounds", type=int, default=3)
    parser.add_argument("--resume-imitation", type=Path,
                        help="Validated normalized checkpoint for focused endpoint/relaxation refinement.")
    parser.add_argument("--imitation-curriculum", choices=("endpoints", "endurance", "endurance_vertical", "endurance_reachable", "endurance_transitions", "endurance_contact"), default="endpoints",
                        help="Resume data: endpoints, or systematic holds/releases with asymmetric fatigue.")
    parser.add_argument("--imitation-vertical-noise", type=float, default=0.012,
                        help="Teacher excitation noise above140deg in endurance_vertical collection.")
    parser.add_argument("--imitation-inactive-flexor-margin", type=float, default=0.0,
                        help="Training-only negative target for inactive flexors while extensors are active.")
    parser.add_argument("--imitation-weak-weight", type=float, default=2.0,
                        help="Supervised weak-trigger sample weight in endurance_vertical refinement.")
    args = parser.parse_args(argv)
    if args.steps < 0 or (args.steps == 0 and args.warm_start_steps == 0):
        parser.error("--steps must be positive, or zero with --warm-start-steps for initialization-only evaluation")
    if args.protocol == "compact" and (args.warm_start_steps or args.command_imitation or args.resume_imitation):
        parser.error("Compact protocol trains on outcomes with PPO; do not mix teacher-label initialization into this experiment")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.warm_start_steps < 0 or not np.isfinite(args.log_std_init):
        parser.error("Warm-start steps must be nonnegative and log-std finite")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.target_kl is not None and (not np.isfinite(args.target_kl) or args.target_kl <= 0):
        parser.error("--target-kl must be finite and positive")
    if args.command_imitation and (args.imitation_rounds < 1 or args.warm_start_steps < 1000 * args.imitation_rounds):
        parser.error("Command imitation requires at least 1000 --warm-start-steps per round")
    if args.resume_imitation and (not args.command_imitation or args.steps != 0):
        parser.error("--resume-imitation requires --command-imitation and --steps 0")
    if args.imitation_curriculum != "endpoints" and not args.resume_imitation:
        parser.error("Endurance curriculum requires --resume-imitation")
    if (not np.isfinite(args.imitation_vertical_noise) or args.imitation_vertical_noise < 0
            or not np.isfinite(args.imitation_inactive_flexor_margin) or not 0 <= args.imitation_inactive_flexor_margin <= 0.05
            or not np.isfinite(args.imitation_weak_weight) or args.imitation_weak_weight <= 0):
        parser.error("Imitation noise/margin/weight must be finite; noise>=0, margin in[0,.05], weight>0")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        import torch
    except ImportError as exc:
        raise SystemExit("PPO is optional. Install the project's train extra before using myo.train.") from exc
    torch.set_num_threads(args.threads)

    parent_metadata = None
    if args.resume_imitation:
        checked, _, parent_metadata = load_controller(str(args.resume_imitation), args.config, args.predictor)
        checked.close()
        if not parent_metadata.get("features_sha256"):
            raise ValueError("Refinement requires a normalized command-imitation checkpoint.")
        if parent_metadata.get("predictor_sha256") and args.predictor is None:
            stored = Path(parent_metadata["predictor_path"])
            args.predictor = stored if stored.is_absolute() else args.resume_imitation.parent / stored
        config = PreacherCurlConfig(**parent_metadata["config"])
    else:
        config = PreacherCurlConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else PreacherCurlConfig()
    base_env = PreacherCurlEnv(config=config)
    source_fingerprint = dynamics_digest()
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
    protocol = None
    if args.protocol == "compact":
        from myo.compact_protocol import CompactProtocol, protocol_metadata

        env = CompactProtocol(env)
        protocol = protocol_metadata(base_env.config.payload_mass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitored = Monitor(env)
    # A compact MLP PPO baseline is normally a CPU workload. No experiment
    # tracker, network service, or automatic model download is involved.
    policy_kwargs = {"net_arch": [128, 128] if protocol else [64, 64], "log_std_init": args.log_std_init}
    features_sha256 = None
    if args.command_imitation:
        from myo.policy_features import NormalizedFeaturesExtractor

        policy_kwargs.update(net_arch=[128, 128], features_extractor_class=NormalizedFeaturesExtractor)
        features_sha256 = hashlib.sha256(Path(__file__).with_name("policy_features.py").read_bytes()).hexdigest()
    if args.resume_imitation:
        model = PPO.load(args.resume_imitation, env=monitored, device="cpu")
        model.set_random_seed(args.seed)
    else:
        model = PPO(
            "MlpPolicy", monitored, device="cpu", seed=args.seed,
            policy_kwargs=policy_kwargs, n_steps=128, batch_size=64,
            learning_rate=args.learning_rate, target_kl=args.target_kl, verbose=1,
        )
    try:
        if args.command_imitation:
            from myo.imitation import train_command_actor

            warm_start = train_command_actor(model, env, args.warm_start_steps, args.imitation_rounds,
                                            args.seed + 10_000, epochs=120 if args.resume_imitation else 180,
                                            resume=bool(args.resume_imitation),
                                            curriculum=args.imitation_curriculum,
                                            vertical_teacher_noise=args.imitation_vertical_noise,
                                            inactive_flexor_margin=args.imitation_inactive_flexor_margin,
                                            weak_trigger_weight=args.imitation_weak_weight)
        else:
            warm_start = warm_start_actor(model, env, args.warm_start_steps, args.seed + 10_000) if args.warm_start_steps else None
        if warm_start:
            print(json.dumps({"warm_start": warm_start}), flush=True)
        model.learn(total_timesteps=args.steps)
        model.save(str(args.output))
        actual_output = args.output if args.output.suffix == ".zip" else Path(str(args.output) + ".zip")
        bundled_predictor = None
        if predictor_bytes is not None:
            bundled_predictor = Path(str(actual_output) + ".predictor.json")
            bundled_predictor.write_bytes(predictor_bytes)
        metadata = {
            "schema_version": 2,
            "task": config.task_mode + "_preacher_curl",
            "algorithm": "stable_baselines3.PPO",
            "training_stage": "imitation_only" if args.steps == 0 else "imitation_then_ppo" if warm_start else "ppo",
            "training_protocol": protocol,
            "protocol_completed_episodes": env.completed_episodes if protocol else None,
            "seed": args.seed,
            "torch_threads": args.threads,
            "requested_training_steps": args.steps,
            "trained_steps": int(model.num_timesteps),
            "warm_start": warm_start,
            "parent_checkpoint_sha256": parent_metadata.get("checkpoint_sha256") if parent_metadata else None,
            "parent_warm_start": parent_metadata.get("warm_start") if parent_metadata else None,
            "log_std_init": parent_metadata.get("log_std_init", args.log_std_init) if parent_metadata else args.log_std_init,
            "learning_rate": parent_metadata.get("learning_rate", args.learning_rate) if parent_metadata else args.learning_rate,
            "target_kl": parent_metadata.get("target_kl", args.target_kl) if parent_metadata else args.target_kl,
            "config": asdict(base_env.config),
            "dynamics_sha256": source_fingerprint,
            "contract": policy_contract(base_env),
            "checkpoint_sha256": hashlib.sha256(actual_output.read_bytes()).hexdigest(),
            "features_sha256": features_sha256,
            "observation_names": observation_names,
            "predictor_path": bundled_predictor.name if bundled_predictor else None,
            "predictor_sha256": predictor_sha256,
            "package_versions": {name: version(name) for name in ("mujoco", "gymnasium", "numpy", "stable-baselines3", "torch")},
            "source": "Local CPU actor with independent delayed muscle excitations",
            "limitations": "Experimental 1-DOF controller; training budget is not proof of tracking or protection.",
        }
        Path(str(actual_output) + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        monitored.close()


if __name__ == "__main__":
    main()
