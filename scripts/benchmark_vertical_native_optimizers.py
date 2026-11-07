"""CPU optimizer-step timings on identical native-controller gradients."""
import argparse
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/vertical_architecture/optimizer_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    import torch
    from myo.train_vertical_architecture import model_for_trial, parameter_hash
    from myo.train_vertical_ppo import write_json
    from myo.vertical_control import VerticalControlEnv
    from myo.vertical_muscles import VerticalConfig
    torch.set_num_threads(1)
    models = {name: model_for_trial(name, VerticalControlEnv(VerticalConfig(variant=4)))
              for name in ("control", "radam", "rmsprop", "muon")}
    assert len({parameter_hash(model) for model in models.values()}) == 1
    generator = torch.Generator().manual_seed(917)
    gradients = [torch.randn(p.shape, generator=generator)*.001 for p in models["control"].policy.parameters()]
    report = {"description": "Identical 2x128 Tanh parameters and fixed synthetic gradients; optimizer.step only, no backprop/clipping/physics",
              "torch": torch.__version__, "threads": 1,
              "parameter_count": sum(p.numel() for p in models["control"].policy.parameters()),
              "warmup_steps": 10, "repeats": args.repeats, "optimizers": {}}

    def prepare(model):
        model.policy.optimizer.zero_grad()
        for parameter, gradient in zip(model.policy.parameters(), gradients):
            parameter.grad = gradient.clone()

    for name, model in models.items():
        for _ in range(10):
            prepare(model)
            model.policy.optimizer.step()
        report["optimizers"][name] = {"seconds": []}
    names = list(models)
    for repeat in range(args.repeats):
        order = names[repeat % len(names):]+names[:repeat % len(names)]
        for name in order:
            model = models[name]
            prepare(model)
            tick = time.perf_counter()
            model.policy.optimizer.step()
            report["optimizers"][name]["seconds"].append(time.perf_counter()-tick)
    for entry in report["optimizers"].values():
        entry["median_seconds"] = float(np.median(entry["seconds"]))
        entry["p95_seconds"] = float(np.quantile(entry["seconds"], .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print({name: entry["median_seconds"] for name, entry in report["optimizers"].items()})
    from torch.utils.benchmark import Timer
    matrix = torch.randn(128, 128, generator=torch.Generator().manual_seed(917))
    precision = {"cpu_capability": torch.backends.cpu.get_cpu_capability(),
                 "matrix_shape": [128, 128], "operation": "a @ a.T", "threads": 1, "results": {}}
    for dtype in (torch.float32, torch.bfloat16):
        measurement = Timer("x @ x.T", globals={"x": matrix.to(dtype)}, num_threads=1).blocked_autorange(min_run_time=.5)
        precision["results"][str(dtype)] = {"median_seconds": measurement.median, "iqr_seconds": measurement.iqr}
    write_json(args.output.parent/"matmul_precision_benchmark.json", precision)


if __name__ == "__main__":
    main()
