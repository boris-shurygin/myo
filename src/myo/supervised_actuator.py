"""Two supervised motor-control experiments on the explicit instant-rigid arm.

Both actors output muscle commands. The acceleration loss differentiates the
instantaneous mechanical response, never a critic or a teacher-action error.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from .actuator_ablation import contract, descriptor, load_ablation_controller, make_environment
from .env import PreacherCurlConfig
from .movement_curriculum import MovementProtocol, evaluate_movement
from .runtime import dynamics_digest
from .train_bounded import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_descriptor(config):
    details = descriptor("instant_rigid", config)
    details.pop("scope")  # This experiment deliberately uses heuristic supervision.
    return {"name": "supervised_instant_rigid_v1", "actuator": details,
            "supervised_source_sha256": sha(__file__)}


def physics_digest(details):
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()


class AccelerationKernel:
    """Exact 1-DOF MuJoCo acceleration at a fixed causal mechanical state."""

    def __init__(self, config):
        self.env = make_environment("instant_rigid", config)
        if (self.env.model.nq, self.env.model.nv, self.env.model.nu) != (1, 1, 0):
            raise ValueError("This diagnostic kernel is only for the unactuated 1-DOF MyoArm")
        self.data = mujoco.MjData(self.env.model)

    def acceleration(self, q, velocity, torque):
        d = self.data
        d.qpos[0], d.qvel[0] = q, velocity
        d.qfrc_applied[0] = torque
        d.qacc_warmstart[:] = 0.
        mujoco.mj_forward(self.env.model, d)
        return float(d.qacc[0])

    def capture(self, env):
        if env.payload_mass != self.env.config.payload_mass:
            raise ValueError("Acceleration kernel payload differs from the sampled state")
        q, v = float(env.data.qpos[0]), float(env.data.qvel[0])
        lengths = env.muscle_lengths()
        active, bias = [], []
        for m, length, r in zip(env.muscles, lengths, env.moment_arms):
            raw_fiber = length - m.params.tendon_slack_length
            taut = raw_fiber >= m.minimum_fiber_length
            fiber = max(m.minimum_fiber_length, raw_fiber)
            a, p, damping, *_ = m._fiber_forces(fiber, 1., m.capacity, -r * v if taut else 0.)
            active.append(a if taut else 0.)
            bias.append(p + damping if taut else 0.)
        passive = env.passive_limit_torque()
        offset = self.acceleration(q, v, passive)
        return {"q": q, "velocity": v, "force_active": active, "force_bias": bias,
                "moment_arms": env.moment_arms.copy(), "passive_torque": passive,
                "acceleration_offset": offset, "inverse_inertia": 1. / self.env.model.dof_M0[0],
                "free": self.data.nefc == 0}

    def response(self, state, command):
        forces = np.maximum(0., np.asarray(state["force_active"]) * command + state["force_bias"])
        torque = float(np.asarray(state["moment_arms"]) @ forces) + state["passive_torque"]
        return self.acceleration(state["q"], state["velocity"], torque)

    def close(self):
        self.env.close()


class ConstraintAcceleration(torch.autograd.Function):
    """Scalar torque derivative through MuJoCo's constrained solve, not time."""

    @staticmethod
    def forward(ctx, torque, q, velocity, passive, kernel):
        values, derivatives = [], []
        for t, angle, speed, p in zip(torque.detach().numpy(), q, velocity, passive):
            eps = 1e-4 * max(1., abs(float(t)))
            values.append(kernel.acceleration(angle, speed, float(t) + p))
            hi = kernel.acceleration(angle, speed, float(t) + p + eps)
            lo = kernel.acceleration(angle, speed, float(t) + p - eps)
            derivatives.append((hi - lo) / (2 * eps))
        ctx.save_for_backward(torch.as_tensor(derivatives, dtype=torque.dtype))
        return torch.as_tensor(values, dtype=torque.dtype)

    @staticmethod
    def backward(ctx, gradient):
        derivative, = ctx.saved_tensors
        return gradient * derivative, None, None, None, None


def acceleration_tensor(commands, data, indices, kernel):
    def tensor(name):
        return torch.as_tensor(data[name][indices], dtype=commands.dtype)
    forces = torch.relu(tensor("force_active") * commands + tensor("force_bias"))
    torque = (tensor("moment_arms") * forces).sum(1)
    acceleration = tensor("acceleration_offset") + tensor("inverse_inertia") * torque
    constrained = ~data["free"][indices].astype(bool)
    if constrained.any():
        selected = indices[constrained]
        result = ConstraintAcceleration.apply(torque[constrained], data["q"][selected],
            data["velocity"][selected], data["passive_torque"][selected], kernel)
        acceleration = acceleration.clone()
        acceleration[constrained] = result
    return acceleration


def collect(config, kernel, controller=None, initial_angle=20., episode_offset=0):
    env = MovementProtocol(make_environment("instant_rigid", config), 0)
    env.scenarios = tuple(replace(s, initial_angle_deg=initial_angle) for s in env.scenarios)
    rows, episodes, actual_steps, failures = [], [], 0, 0
    try:
        for episode in range(len(env.scenarios)):
            observation, _ = env.reset(seed=0)
            while True:
                base = env.unwrapped
                teacher = base.baseline_action()
                mechanics = kernel.capture(base)
                target = kernel.response(mechanics, teacher)
                rows.append({"observations": observation.copy(), "teacher_actions": teacher.copy(),
                             "target_acceleration": target, **mechanics})
                episodes.append(episode + episode_offset)
                action = teacher if controller is None else controller.predict(observation, deterministic=True)[0]
                observation, _, terminated, truncated, _ = env.step(action)
                actual_steps += 1
                if terminated or truncated:
                    failures += int(terminated)
                    break
    finally:
        env.close()
    data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    data["observations"] = data["observations"].astype(np.float32)
    data["teacher_actions"] = data["teacher_actions"].astype(np.float32)
    data["episode_ids"] = np.asarray(episodes)
    return data, {"physical_steps": actual_steps, "episodes": len(env.scenarios),
                  "physical_failures": failures, "initial_angle_deg": initial_angle,
                  "driver": "heuristic" if controller is None else "learned_actor"}


def mean_commands(model, observations):
    return model.policy.get_distribution(torch.as_tensor(observations, dtype=torch.float32)).distribution.mean


def load_supervised_controller(path, render_mode=None):
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    details = metadata["supervised_physics"]
    config = PreacherCurlConfig(**details["actuator"]["source_config"])
    if (metadata.get("schema_version") != 2 or details != physics_descriptor(config)
            or metadata["dynamics_sha256"] != physics_digest(details) or metadata["checkpoint_sha256"] != sha(path)):
        raise ValueError("Supervised experimental physics/checkpoint mismatch")
    for source, key in (("policy_beta.py", "beta_policy_sha256"), ("policy_features.py", "features_sha256")):
        if metadata[key] != sha(Path(__file__).with_name(source)):
            raise ValueError("Supervised actor implementation mismatch")
    env = make_environment("instant_rigid", config, render_mode)
    try:
        if (metadata["config"] != asdict(env.config) or metadata["contract"] != contract(env, "instant_rigid")
                or metadata["observation_names"] != env.observation_names):
            raise ValueError("Supervised observation/action contract mismatch")
        torch.set_num_threads(1)
        model = PPO.load(path, device="cpu")
        if model.num_timesteps != metadata["trained_steps"]:
            raise ValueError("Supervised counter mismatch")
        return env, lambda obs: model.predict(obs, deterministic=True)[0], metadata
    except Exception:
        env.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=Path("runs/d1_actuator_instant_rigid/initial.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if args.epochs < 20 or args.rounds < 1:
        parser.error("Use >=20 epochs and >=1 collection rounds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checked, _, parent = load_ablation_controller(args.initial)
    checked.close()
    if parent["trained_steps"] != 0 or parent["actuator_ablation"]["mode"] != "instant_rigid":
        raise ValueError("Use the common untrained instant-rigid initial checkpoint")
    config = PreacherCurlConfig(**parent["actuator_ablation"]["source_config"])
    kernel = AccelerationKernel(config)
    details = physics_descriptor(config)
    physical_before, started = dynamics_digest(), time.perf_counter()
    data, collection = collect(config, kernel)
    validation, validation_collection = collect(config, kernel, initial_angle=21., episode_offset=1_000_000)
    np.savez_compressed(args.output_dir / "training_round_1.npz", **data)
    np.savez_compressed(args.output_dir / "validation.npz", **validation)
    physical_steps = collection["physical_steps"] + validation_collection["physical_steps"]
    summary = {"status": "running", "source": str(args.initial), "source_sha256": sha(args.initial),
        "physics": details, "training": {"epochs_per_round": args.epochs, "max_rounds": args.rounds,
        "batch_size": 128, "learning_rate": 3e-4, "seed": 0, "acceleration_scale_rad_s2": 10.,
        "action_loss": "mean squared error of Beta means against heuristic muscle commands",
        "acceleration_loss": "mean squared instantaneous acceleration error / 10^2; no action-label loss or activation regularizer",
        "critic": "frozen and unused; no PPO/GAE", "normalization": "common initial buffers frozen",
        "aggregation": "identical union of states visited by both actors; labels queried at causal current states"},
        "collection": [collection], "validation_collection": validation_collection,
        "validation_split": "separate whole episodes, initial21deg; never optimizer samples",
        "scope": "first movement stage only; trained conditions; not generalization or full demo", "branches": {}}
    actors, evaluators, anchors = {}, {}, {}
    for name in ("actions", "accelerations"):
        model = PPO.load(args.initial, device="cpu")
        for p in model.policy.mlp_extractor.value_net.parameters():
            p.requires_grad_(False)
        for p in model.policy.value_net.parameters():
            p.requires_grad_(False)
        model.policy.set_training_mode(True)
        actors[name] = model
        evaluators[name] = MovementProtocol(make_environment("instant_rigid", config), 0)
        anchors[name] = {k: v.detach().clone() for k, v in model.policy.state_dict().items()
                         if k.startswith(("mlp_extractor.value_net", "value_net", "features_extractor"))}
        summary["branches"][name] = {"status": "training", "evaluations": [], "best_score": float(1e6),
                                    "best_ready": False, "gradient_updates": 0, "pass_confirmations": 0}
        (args.output_dir / name).mkdir()

    def save(name, filename):
        model = actors[name]
        path = args.output_dir / name / f"{filename}.zip"
        model.save(path)
        base = evaluators[name].unwrapped
        metadata = {**parent, "algorithm": "supervised_regression; PPO container only",
            "training_stage": f"instant_rigid_heuristic_{name}", "supervised_physics": details,
            "runner_sha256": sha(__file__),
            "dynamics_sha256": physics_digest(details), "config": asdict(base.config),
            "contract": contract(base, "instant_rigid"), "observation_names": base.observation_names,
            "trained_steps": model.num_timesteps, "counter_semantics": "supervised optimizer updates, not physical steps",
            "supervision": summary["training"], "loss_branch": name,
            "parent_checkpoint_sha256": sha(args.initial), "checkpoint_sha256": sha(path)}
        metadata.pop("actuator_ablation", None)
        write_json(str(path) + ".metadata.json", metadata)
        return path

    def measure(name, round_index, epoch):
        model, record = actors[name], summary["branches"][name]
        model.policy.set_training_mode(False)
        metrics = evaluate_movement(model, evaluators[name])
        with torch.no_grad():
            commands = mean_commands(model, validation["observations"])
            action_rmse = float(torch.mean((commands - torch.as_tensor(validation["teacher_actions"])) ** 2).sqrt())
            accelerations = acceleration_tensor(commands, validation, np.arange(len(commands)), kernel)
            acceleration_rmse = float(torch.mean((accelerations - torch.as_tensor(validation["target_acceleration"])) ** 2).sqrt())
        row = {"round": round_index, "epoch": epoch, "updates": model.num_timesteps,
               "validation_action_rmse": action_rmse, "validation_acceleration_rmse_rad_s2": acceleration_rmse,
               **{k: metrics[k] for k in ("score", "passed", "total", "ready")}}
        record["evaluations"].append(row)
        write_json(args.output_dir / name / f"round_{round_index}_epoch_{epoch}_evaluation.json", metrics)
        if (not metrics["ready"], metrics["score"]) < (not record["best_ready"], record["best_score"]):
            record.update(best_score=metrics["score"], best_ready=metrics["ready"],
                          best_passed=metrics["passed"], best_evaluation=row)
            save(name, "best")
        if epoch:
            record["pass_confirmations"] = record["pass_confirmations"] + 1 if metrics["ready"] else 0
        if record["pass_confirmations"] >= 2:
            record["status"] = "stage_ready"
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"branch": name, **row}), flush=True)
        model.policy.set_training_mode(True)

    try:
        for name in actors:
            save(name, "initial")
            measure(name, 0, 0)
        for round_index in range(1, args.rounds + 1):
            rng = np.random.default_rng(0)  # same ordering for both losses on the common data
            permutations = [rng.permutation(len(data["observations"])) for _ in range(args.epochs)]
            for name, model in actors.items():
                record = summary["branches"][name]
                if record["status"] == "stage_ready":
                    continue
                for epoch, permutation in enumerate(permutations, 1):
                    for start in range(0, len(permutation), 128):
                        indices = permutation[start:start + 128]
                        commands = mean_commands(model, data["observations"][indices])
                        if name == "actions":
                            target = torch.as_tensor(data["teacher_actions"][indices])
                            loss = torch.mean((commands - target) ** 2)
                        else:
                            acceleration = acceleration_tensor(commands, data, indices, kernel)
                            target = torch.as_tensor(data["target_acceleration"][indices], dtype=commands.dtype)
                            loss = torch.mean(((acceleration - target) / 10.) ** 2)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite supervised loss")
                        model.policy.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
                        model.policy.optimizer.step()
                        model.num_timesteps += 1
                        record["gradient_updates"] += 1
                    if epoch % 20 == 0 or epoch == args.epochs:
                        measure(name, round_index, epoch)
                        if record["status"] == "stage_ready":
                            break
                save(name, "last")
            if all(r["status"] == "stage_ready" for r in summary["branches"].values()):
                summary["status"] = "both_stage_ready"
                break
            if round_index < args.rounds:
                additions = []
                next_episode = int(data["episode_ids"].max()) + 1
                for name in actors:
                    collector = PPO.load(args.output_dir / name / "best.zip", device="cpu")
                    extra, info = collect(config, kernel, collector, episode_offset=next_episode)
                    next_episode += info["episodes"]
                    additions.append(extra)
                    info.update(branch=name, after_round=round_index)
                    summary["collection"].append(info)
                    physical_steps += info["physical_steps"]
                data = {k: np.concatenate([data[k], *(a[k] for a in additions)]) for k in data}
                np.savez_compressed(args.output_dir / f"training_round_{round_index + 1}.npz", **data)
        if summary["status"] == "running":
            summary["status"] = "supervised_budget_exhausted"
        for name, model in actors.items():
            current = model.policy.state_dict()
            assert all(torch.equal(current[k], v) for k, v in anchors[name].items())
            summary["branches"][name]["critic_and_normalization_unchanged"] = True
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(elapsed_seconds=time.perf_counter() - started,
            collection_physical_steps=physical_steps, training_samples=len(data["observations"]),
            validation_samples=len(validation["observations"]), base_dynamics_unchanged=dynamics_digest() == physical_before)
        write_json(args.output_dir / "summary.json", summary)
        for env in evaluators.values():
            env.close()
        kernel.close()


if __name__ == "__main__":
    main()
