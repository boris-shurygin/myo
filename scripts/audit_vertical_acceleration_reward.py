"""Rescore original and new real test trajectories with the same acceleration reward."""
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_vertical_ppo import write_json
from myo.vertical_acceleration import VerticalAccelerationEnv, load_acceleration_policy
from myo.vertical_control import TEST_CASES, VerticalControlEnv, load_policy
from myo.vertical_muscles import VerticalConfig


def main():
    root = Path("runs/vertical_acceleration_ppo")
    torch.set_num_threads(1)
    independent = json.loads((root/"independent_evaluation.json").read_text())
    assert independent["status"] == "passed"
    result = {"status": "running", "scope": "Deterministic native command replay and stochastic seed0 test rollouts, common acceleration reward", "variants": {}}
    for variant in (1, 2, 3, 4):
        row = {}
        for label in ("baseline", "acceleration"):
            cases = json.loads((root/f"variant_{variant}"/f"{label}_test_replay.json").read_text())["cases"]
            metrics = []
            for case in cases:
                env = VerticalAccelerationEnv(VerticalConfig(variant=variant))
                env.reset(options={"initial_position_m": case["initial_position_m"], "target_m": case["target_m"]})
                raw, errors, steady_errors = [], [], []
                for sample in case["samples"]:
                    _, _, terminated, _, info = env.step(np.asarray(sample[4:6]))
                    np.testing.assert_array_equal([env.physical.time, env.physical.position, env.physical.velocity], sample[:3])
                    raw.append(info["raw_reward"])
                    errors.append(info["acceleration_error_m_s2"])
                    if sample[0] >= 3.:
                        steady_errors.append(info["acceleration_error_m_s2"])
                    assert not terminated
                metrics.append({"target_m": case["target_m"], "mean_raw_acceleration_reward": float(np.mean(raw)),
                                "full_acceleration_rmse_m_s2": float(np.sqrt(np.mean(np.square(errors)))),
                                "steady_acceleration_rmse_m_s2": float(np.sqrt(np.mean(np.square(steady_errors)))),
                                "steady_position_rmse_m": case["last_second_rmse_m"]})
            row[label] = {"cases": metrics, **{key: float(np.mean([c[key] for c in metrics])) for key in (
                "mean_raw_acceleration_reward", "full_acceleration_rmse_m_s2", "steady_acceleration_rmse_m_s2", "steady_position_rmse_m")}}
            config = VerticalConfig(variant=variant)
            if label == "baseline":
                model, _ = load_policy(Path(f"runs/vertical_ppo/variant_{variant}/best.zip"), VerticalControlEnv(config))
            else:
                model, _, _ = load_acceleration_policy(root/f"variant_{variant}"/"best.zip")
            torch.manual_seed(0)
            stochastic = []
            for position, target in TEST_CASES:
                env = VerticalAccelerationEnv(config)
                obs, _ = env.reset(options={"initial_position_m": position, "target_m": target})
                rewards, errors, failed = [], [], False
                for _ in range(200):
                    action, _ = model.predict(obs, deterministic=False)
                    obs, _, failed, truncated, info = env.step(action)
                    rewards.append(info["raw_reward"])
                    errors.append(info["error_m"])
                    if failed or truncated:
                        break
                stochastic.append({"physical_failure": bool(failed), "steps": len(rewards),
                    "mean_raw_reward": float(np.mean(rewards)),
                    "discounted_raw_return": float(np.dot(np.asarray(rewards), model.gamma**np.arange(len(rewards)))),
                    "position_rmse_m": float(np.sqrt(np.mean(np.square(errors))))})
            row[label]["stochastic_seed0"] = {"cases": stochastic,
                "failures": sum(c["physical_failure"] for c in stochastic),
                "mean_raw_reward": float(np.mean([c["mean_raw_reward"] for c in stochastic])),
                "mean_discounted_raw_return": float(np.mean([c["discounted_raw_return"] for c in stochastic]))}
        result["variants"][str(variant)] = row
        print(variant, {label: {k: v for k, v in values.items() if k not in ("cases", "stochastic_seed0")} for label, values in row.items()}, flush=True)
        print('stochastic', variant, {label: {k: v for k, v in values['stochastic_seed0'].items() if k != 'cases'} for label, values in row.items()}, flush=True)
    result["status"] = "passed"
    write_json(root/"reward_audit.json", result)


if __name__ == "__main__":
    main()
