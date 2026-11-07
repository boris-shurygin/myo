"""Train the original delayed/compliant arm by short-horizon physical error."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from .actuator_ablation import load_ablation_controller
from .delayed_acceleration import (NativeResponse, collect, exact_validation,
    linear_response, linearize_dataset, mean_commands, sha)
from .env import PreacherCurlConfig, PreacherCurlEnv
from .movement_curriculum import MovementProtocol, evaluate_movement
from .runtime import dynamics_digest, load_controller, policy_contract
from .train_bounded import write_json


def local_step(model, observations, tensors, start_index, trust_distance):
    """Accept only full-dataset descent inside the measured command neighborhood."""
    def objective():
        commands = mean_commands(model, observations)
        prediction = linear_response(commands, tensors["anchors"], tensors["accelerations"], tensors["jacobians"])
        errors = ((prediction[:, start_index:] - tensors["targets"][:, start_index:]) / 10.) ** 2
        return (errors.mean(1) * tensors["sample_weights"]).mean(), commands

    before, _ = objective()
    if not torch.isfinite(before):
        raise FloatingPointError("Non-finite native acceleration loss")
    model.policy.optimizer.zero_grad()
    before.backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    baseline = float(before.detach())
    for trial in range(12):
        model.policy.load_state_dict(weights)
        model.policy.optimizer.load_state_dict(optimizer)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 3e-4 * .5 ** trial
        model.policy.optimizer.step()
        with torch.no_grad():
            after, commands = objective()
            distance = float(torch.max(torch.abs(commands - tensors["anchors"])))
        if float(after) < baseline and distance <= trust_distance:
            return {"accepted": True, "loss_before": baseline, "loss_after": float(after),
                    "distance": distance, "backtracks": trial}
    model.policy.load_state_dict(weights)
    model.policy.optimizer.load_state_dict(optimizer)
    return {"accepted": False, "loss_before": baseline, "loss_after": baseline,
            "backtracks": 12}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=Path("runs/d1_actuator_full_check/initial.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trust-distance", type=float, default=.03,
                        help="Refresh native gradients once any sampled command shifts this far from its anchor")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.epochs < 50 or not 0 < args.trust_distance <= .1:
        parser.error("Use positive rounds and at least 50 epochs")
    initial_metadata = json.loads(Path(str(args.initial) + ".metadata.json").read_text())
    resumed = "native_acceleration_training" in initial_metadata
    checked, _, parent = (load_controller(str(args.initial)) if resumed
                          else load_ablation_controller(args.initial))
    checked.close()
    if not resumed and (parent["trained_steps"] != 0 or parent["actuator_ablation"]["mode"] != "full"):
        raise ValueError("Use the untrained full-physics initial actor")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    config = PreacherCurlConfig(**parent["config"])
    model = PPO.load(args.initial, device="cpu")
    # No old first-moment direction after changing the local mechanical model.
    model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
        lr=3e-4, betas=(0., .999), eps=1e-5)
    for module in (model.policy.mlp_extractor.value_net, model.policy.value_net):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    anchors = {key: value.clone() for key, value in model.policy.state_dict().items()
               if key.startswith(("mlp_extractor.value_net", "value_net", "features_extractor"))}
    kernel = NativeResponse(config)
    evaluation = MovementProtocol(PreacherCurlEnv(config), 0)
    start_index = round(config.motor_delay / config.control_dt)
    sample_steps = (0, 1, 2, 4, 6, 10, 20, 40, 60, 100, 150, 190)
    protocol = {"name": "native_delayed_acceleration_v3", "horizon_steps": kernel.horizon_steps,
        "horizon_s": kernel.horizon_steps * config.control_dt, "finite_difference_epsilon": .001,
        "action_variable": "first control command only; fixed identical teacher continuation in all branches",
        "target": "native realized average acceleration in each future20ms window from heuristic continuation",
        "loss": "MSE of locally linearized future accelerations after motor delay, scaled by 10 rad/s2",
        "linearization": "at current actor output; refresh all anchors/Jacobians between rounds",
        "refresh_trigger": "accepted full-batch descent; refresh at >=95% of command trust distance, rejected proposal, surrogate error, or epoch budget",
        "optimizer": "full-batch Adam beta1=0; backtrack learning rate until weighted loss decreases and command shift <=trust_distance",
        "sample_weighting": "Voronoi duration weights, sum represents whole4s episode despite dense early samples",
        "trust_distance": args.trust_distance,
        "snapshot": "mjSTATE_INTEGRATION plus muscle, queue, command and history state; derived caches recomputed",
        "critic": "frozen and unused; no PPO/GAE or action-label MSE",
        "actor": "Beta mean MLP128x128; common full-physics origin, optional saved native-acceleration warm start; frozen normalizer",
        "training_states": "teacher plus latest actor-visited states; fixed denser sampling during initial transient; six fixed stage0 conditions",
        "sample_steps_per_episode": sample_steps,
        "validation": "separate teacher episodes, start21deg, every20th sample; never optimizer data",
        "counter_semantics": "supervised optimizer updates, not physical control steps",
        "kernel_sha256": sha(Path(__file__).with_name("delayed_acceleration.py")),
        "runner_sha256": sha(__file__)}
    summary = {"status": "running", "dynamics_sha256": dynamics_digest(), "config": asdict(config),
        "initial": str(args.initial), "initial_sha256": sha(args.initial), "protocol": protocol,
        "evaluations": [], "collections": [], "best_ready": False, "best_score": 1e6,
        "best_passed": 0, "confirmations": 0, "gradient_updates": 0, "rounds": [],
        "initial_optimizer_updates": model.num_timesteps, "resumed": resumed,
        "scope": "first20->30deg stage only; 1kg; original physical model; one fixed seed0",
        "evaluation_physical_control_steps": 0}
    summary["accepted_updates"] = []
    write_json(args.output_dir / "summary.json", summary)

    def save(label):
        path = args.output_dir / f"{label}.zip"
        model.save(path)
        metadata = {**parent, "algorithm": "supervised_acceleration_with_native_finite_difference_jacobians",
            "training_stage": "native_delayed_acceleration_supervised", "config": asdict(config),
            "dynamics_sha256": dynamics_digest(), "contract": policy_contract(evaluation.unwrapped),
            "observation_names": list(evaluation.unwrapped.observation_names),
            "trained_steps": model.num_timesteps, "counter_semantics": protocol["counter_semantics"],
            "training_protocol": protocol, "native_acceleration_training": protocol,
            "parent_checkpoint_sha256": sha(args.initial), "checkpoint_sha256": sha(path),
            "runner_sha256": sha(__file__)}
        metadata.pop("actuator_ablation", None)
        write_json(str(path) + ".metadata.json", metadata)

    def measure(round_index, epoch, validation=None, validation_states=None, probe=None):
        model.policy.set_training_mode(False)
        metrics = evaluate_movement(model, evaluation)
        summary["evaluation_physical_control_steps"] += sum(s["sample_count"] for s in metrics["scenarios"])
        row = {"round": round_index, "epoch": epoch, "updates": model.num_timesteps,
               **{key: metrics[key] for key in ("score", "passed", "total", "ready")}}
        if validation is not None:
            row["validation"] = exact_validation(model, validation, validation_states, kernel)
        if probe is not None:
            row["training_probe"] = exact_validation(model, *probe[:2], kernel, linear=probe[2])
        summary["evaluations"].append(row)
        write_json(args.output_dir / f"round_{round_index}_epoch_{epoch}_evaluation.json", metrics)
        if (not metrics["ready"], metrics["score"]) < (not summary["best_ready"], summary["best_score"]):
            summary.update(best_ready=metrics["ready"], best_score=metrics["score"],
                           best_passed=metrics["passed"], best_evaluation=row)
            save("best")
        if epoch:
            summary["confirmations"] = summary["confirmations"] + 1 if metrics["ready"] else 0
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(row), flush=True)
        return row

    try:
        save("initial")
        measure(0, 0)
        data, states, info = collect(config, kernel, sample_steps=sample_steps)
        teacher_data, teacher_states = data, states.copy()
        summary["collections"].append(info)
        if resumed:
            extra, extra_states, info = collect(config, kernel, model,
                episode_offset=6, sample_steps=sample_steps)
            summary["collections"].append(info)
            data = {key: np.concatenate((value, extra[key])) for key, value in teacher_data.items()}
            states = teacher_states + extra_states
        validation, validation_states, info = collect(config, kernel, initial_angle=21., stride=20,
                                                     episode_offset=1_000_000)
        summary["validation_collection"] = info
        np.savez_compressed(args.output_dir / "validation.npz", **validation)
        with (args.output_dir / "validation_states.pkl").open("wb") as stream:
            pickle.dump(validation_states, stream)
        for round_index in range(1, args.rounds + 1):
            print(f"round {round_index}: {len(states)} native anchors", flush=True)
            linear = linearize_dataset(model, data, states, kernel)
            np.savez_compressed(args.output_dir / f"round_{round_index}_data.npz", **data, **linear)
            with (args.output_dir / f"round_{round_index}_states.pkl").open("wb") as stream:
                pickle.dump(states, stream)
            probe_indices = np.linspace(0, len(states) - 1, 12, dtype=int)
            probe = ({key: value[probe_indices] for key, value in data.items()},
                     [states[i] for i in probe_indices],
                     {key: value[probe_indices] for key, value in linear.items()})
            tensors = {key: torch.as_tensor(value, dtype=torch.float32)
                       for key, value in {**linear, "targets": data["targets"],
                                          "sample_weights": data["sample_weights"]}.items()}
            refresh = "epoch_budget"
            for epoch in range(1, args.epochs + 1):
                model.policy.set_training_mode(True)
                update = local_step(model, data["observations"], tensors, start_index, args.trust_distance)
                if update["accepted"]:
                    model.num_timesteps += 1
                    summary["gradient_updates"] += 1
                    summary["accepted_updates"].append({"round": round_index, "epoch": epoch, **update})
                needs_refresh = not update["accepted"] or update.get("distance", 0.) >= .95 * args.trust_distance
                if epoch % 50 == 0 or epoch == args.epochs or needs_refresh:
                    row = measure(round_index, epoch, validation, validation_states, probe)
                    if summary["confirmations"] >= 2:
                        summary["status"] = "stage_ready"
                        refresh = "stage_ready"
                        break
                    if row["training_probe"]["linearization_rmse_rad_s2"] > 1.:
                        refresh = "surrogate_error"
                        break
                    if needs_refresh:
                        refresh = "command_trust_distance"
                        break
            save("last")
            summary["rounds"].append({"round": round_index, "samples": len(states),
                                      "epochs": epoch, "reason": refresh})
            if summary["status"] == "stage_ready":
                break
            if round_index < args.rounds:
                extra, extra_states, info = collect(config, kernel, model,
                    episode_offset=int(data["episode_ids"].max()) + 1, sample_steps=sample_steps)
                info["after_round"] = round_index
                summary["collections"].append(info)
                data = {key: np.concatenate((value, extra[key])) for key, value in teacher_data.items()}
                states = teacher_states + extra_states
        if summary["status"] == "running":
            summary["status"] = "experiment_budget_exhausted"
        assert set(data["episode_ids"]).isdisjoint(validation["episode_ids"])
        current = model.policy.state_dict()
        assert all(torch.equal(current[key], value) for key, value in anchors.items())
        summary["critic_and_normalization_unchanged"] = True
        summary["validation_episodes_disjoint"] = True
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(elapsed_seconds=time.perf_counter() - started,
            native_response_physical_control_steps=kernel.control_steps,
            base_dynamics_unchanged=dynamics_digest() == summary["dynamics_sha256"])
        write_json(args.output_dir / "summary.json", summary)
        evaluation.close()
        kernel.close()


if __name__ == "__main__":
    main()
