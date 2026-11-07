"""Local four-mechanic PPO experiment for time-stamped acceleration commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from myo.train_vertical_ppo import write_json
from myo.train_vertical_scheduled import validation_rank
from myo.vertical_scheduled import TEST_CASES, compact_scheduled, evaluate_scheduled, load_scheduled_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_scheduled_ppo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"schema": "myo.vertical_scheduled_experiment.v1", "status": "running", "variants": {},
               "runner_sha256": hashlib.sha256(Path("src/myo/train_vertical_scheduled.py").read_bytes()).hexdigest(),
               "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Four fresh policies; 40ms-issued acceleration goals; original native mechanics, direct Beta and PPO hyperparameters"}
    # Reuse only completed revision-preflight trials whose rankings/stopping
    # cannot change: every observed case count stayed zero, so the changed
    # partial-success priority was never exercised. Keep authentic metadata.
    archive = Path("runs/vertical_scheduled_threshold_selection")
    summary["reused_variants"] = {}
    if archive.exists():
        original = json.loads((archive/"summary.json").read_text())
        old_source = archive/"train_vertical_scheduled_original.py"
        assert original["runner_sha256"] == hashlib.sha256(old_source.read_bytes()).hexdigest()
        for variant in (1, 2, 3, 4):
            folder = archive/f"variant_{variant}"
            trial = json.loads((folder/"summary.json").read_text())
            checks = [(0, trial["initial_validation"])]+[(h["steps"], h["validation"]) for h in trial["history"]]
            if trial["status"] == "training_complete" and all(r["passed"] == 0 for _, r in checks):
                best = min(checks, key=lambda r: validation_rank(r[1]))
                assert best[0] == trial["best_steps"] and best[1] == trial["selected_validation"]
                assert trial["training"]["runner_sha256"] == original["runner_sha256"]
                shutil.copytree(folder, args.output_dir/f"variant_{variant}")
                summary["variants"][str(variant)] = trial
                summary["reused_variants"][str(variant)] = {"path": str(folder), "trainer_source": str(old_source),
                    "reason": "All validation passed counts zero; revised rank and stopping order identical"}

    def report():
        summary["elapsed_s"] = time.monotonic()-started
        write_json(args.output_dir/"summary.json", summary)

    def worker(variant):
        folder = args.output_dir/f"variant_{variant}"
        with (args.output_dir/f"variant_{variant}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "myo.train_vertical_scheduled", "--variant", str(variant),
                            "--output-dir", str(folder)], stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((folder/"summary.json").read_text())
        assert result["status"] == "training_complete"
        assert result["training"]["runner_sha256"] == summary["runner_sha256"]
        return variant, result

    report()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(worker, v) for v in (1, 2, 3, 4) if str(v) not in summary["variants"]]):
            variant, record = future.result()
            summary["variants"][str(variant)] = record
            print(f"v{variant}: {record['trained_steps']} steps, {record['stop_reason']}, "
                  f"best {record['selected_validation']['passed']}/5, "
                  f"{record['selected_validation']['score_m_s2']:.4f}m/s²", flush=True)
            report()
    summary["all_selections_frozen_before_test"] = True
    report()
    import torch
    torch.set_num_threads(1)
    for variant, record in summary["variants"].items():
        folder = args.output_dir/f"variant_{variant}"
        model, env, _ = load_scheduled_policy(folder/"best.zip")
        result = evaluate_scheduled(model, env.physical.config, TEST_CASES)
        record["test"] = compact_scheduled(result)
        write_json(folder/"test.json", result)
        report()
    summary["status"] = "complete"
    report()


if __name__ == "__main__":
    main()
