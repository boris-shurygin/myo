"""Run one parameter at a time, with independent local mechanic workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from myo.train_vertical_ppo import compact, write_json
from myo.train_vertical_sweep import candidates
from myo.vertical_control import TEST_CASES, evaluate_policy
from myo.vertical_sweep_env import load_sweep_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_ppo"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_ppo_sweep"))
    parser.add_argument("--variants", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--continue-run", action="store_true")
    args = parser.parse_args()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "continue_run"}
    runner = Path("src/myo/train_vertical_sweep.py")
    identities = {"runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
                  "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.continue_run:
        summary = json.loads((args.output_dir/"summary.json").read_text())
        assert summary["arguments"] == settings
        assert all(summary[k] == v for k, v in identities.items())
        if summary["status"] == "complete":
            print("Experiment already complete", flush=True)
            return
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        summary = {"schema": "myo.vertical_ppo_sweep.v1", "status": "running", "arguments": settings,
                   **identities, "matrix": candidates(), "parents": {}, "trials": {},
                   "scope": "One-factor continuations; independent mechanic processes, one torch thread each; fixed seed0 and parent Adam",
                   "selection": "fewer failures, more passed cases, smaller steady validation RMSE; includes update0",
                   "evaluation_reward": "Common baseline reward; test deferred until every selection is frozen",
                   "stopping": "Fixed additional physical-transition budget; no early stop"}
    started, earlier_elapsed = time.monotonic(), summary.get("elapsed_s", 0.)

    def report():
        summary["elapsed_s"] = earlier_elapsed+time.monotonic()-started
        write_json(args.output_dir/"summary.json", summary)

    def worker(name, variant):
        job = args.output_dir/"jobs"/name/f"mechanic_{variant}"
        job.parent.mkdir(parents=True, exist_ok=True)
        if job.exists():
            job.rename(job.with_name(job.name+f"_interrupted_{time.monotonic_ns()}"))
        log = job.with_suffix(".log")
        command = [sys.executable, "-m", "myo.train_vertical_sweep", "--reference", str(args.reference),
                   "--output-dir", str(job), "--variants", str(variant), "--candidates", name,
                   "--steps", str(args.steps), "--eval-every", str(args.eval_every), "--defer-test"]
        with log.open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((job/"summary.json").read_text())
        assert result["status"] == "training_complete"
        assert result["runner_sha256"] == identities["runner_sha256"]
        key = f"variant_{variant}/{name}"
        destination = args.output_dir/key
        shutil.copytree(job/key, destination, dirs_exist_ok=True)
        return variant, key, result

    report()
    for name, changes in candidates():
        pending = [v for v in args.variants if summary["trials"].get(f"variant_{v}/{name}", {}).get("status") != "complete"]
        print(f"FACTOR {name}: {changes}; pending mechanics {pending}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, name, variant) for variant in pending]
            for future in as_completed(futures):
                variant, key, child = future.result()
                origin = child["parents"][str(variant)]
                if str(variant) in summary["parents"]:
                    assert summary["parents"][str(variant)] == origin
                summary["parents"][str(variant)] = origin
                summary["trials"][key] = child["trials"][key]
                record = summary["trials"][key]
                val = record["selected_validation"]
                print(f"COMPLETE {key}: val={val['score_m']*1000:.3f}mm passed={val['passed']}/5 "
                      f"best+{record['best_additional_steps']}", flush=True)
                report()
    summary["all_selections_frozen_before_test"] = True
    report()
    import torch
    torch.set_num_threads(1)
    for key, record in summary["trials"].items():
        if "test" in record:
            continue
        folder = args.output_dir/key
        model, env, _ = load_sweep_policy(folder/"best.zip")
        test = evaluate_policy(model, env.physical.config, TEST_CASES)
        switches = evaluate_policy(model, env.physical.config, ((0., .06),), seconds=6., switches=True)
        record["test"], record["switches"] = compact(test), compact(switches)
        write_json(folder/"test.json", test)
        write_json(folder/"switches.json", switches)
        report()
    summary["status"] = "complete"
    report()


if __name__ == "__main__":
    main()
