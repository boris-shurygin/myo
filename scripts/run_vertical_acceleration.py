"""Train four fresh acceleration-reward PPO policies locally, then open test."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from myo.train_vertical_ppo import write_json
from myo.vertical_acceleration import compact_acceleration, evaluate_acceleration_policy, load_acceleration_policy
from myo.vertical_control import TEST_CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_acceleration_ppo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"schema": "myo.vertical_acceleration_experiment.v1", "status": "running",
               "variants": {}, "baseline": "runs/vertical_ppo/summary.json",
               "baseline_sha256": hashlib.sha256(Path("runs/vertical_ppo/summary.json").read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256(Path("src/myo/train_vertical_acceleration.py").read_bytes()).hexdigest(),
               "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Fresh policies, same seed0 initial tensors, original PPO and task cases; position component replaced only",
               "control_preflight": "runs/vertical_acceleration_control_preflight/summary.json"}

    def report():
        summary["elapsed_s"] = time.monotonic()-started
        write_json(args.output_dir/"summary.json", summary)

    def worker(variant):
        folder = args.output_dir/f"variant_{variant}"
        with (args.output_dir/f"variant_{variant}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "myo.train_vertical_acceleration", "--variant", str(variant),
                            "--output-dir", str(folder)], stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((folder/"summary.json").read_text())
        assert result["status"] == "training_complete"
        assert result["training"]["runner_sha256"] == summary["runner_sha256"]
        return variant, result

    report()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(worker, variant) for variant in (1, 2, 3, 4)]):
            variant, result = future.result()
            summary["variants"][str(variant)] = result
            print(f"v{variant} training complete: {result['trained_steps']} steps, {result['stop_reason']}, "
                  f"best val {result['selected_validation']['passed']}/5 "
                  f"{result['selected_validation']['score_m']*1000:.3f}mm", flush=True)
            report()
    summary["all_selections_frozen_before_test"] = True
    report()
    import torch
    torch.set_num_threads(1)
    for variant, record in summary["variants"].items():
        folder = args.output_dir/f"variant_{variant}"
        model, env, _ = load_acceleration_policy(folder/"best.zip")
        for name, cases, options in (("test", TEST_CASES, {}),
                                     ("switches", ((0., .06),), {"seconds": 6., "switches": True})):
            result = evaluate_acceleration_policy(model, env.physical.config, cases, **options)
            record[name] = compact_acceleration(result)
            write_json(folder/f"{name}.json", result)
        report()
    summary["status"] = "complete"
    report()


if __name__ == "__main__":
    main()
