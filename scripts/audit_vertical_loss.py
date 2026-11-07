"""Report channel-wise residuals using the exact frozen training loss scales."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_vertical_predictor import load_predictor, predictor_contract, windows
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


@torch.no_grad()
def audit(folder):
    records = json.loads((folder/"episodes.json").read_text())
    with np.load(folder/"episodes.npz") as arrays:
        episodes = [{**record,"states":arrays[f"e{i}_states"],"actions":arrays[f"e{i}_actions"]}
                    for i,record in enumerate(records)]
    metadata = json.loads((folder/"bptt.pt.metadata.json").read_text())
    env = VerticalMuscles(VerticalConfig(**metadata["config"]))
    model,metadata = load_predictor(folder/"bptt.pt",predictor_contract(env))
    result = {"checkpoint_sha256":metadata["checkpoint_sha256"],"selected_update":metadata["update"],
              "channels":metadata["physical_names"],"delta_scale":model.delta_scale.tolist(),"splits":{}}
    for split in ("train","validation","test"):
        selected = [episode for episode in episodes if episode["split"]==split]
        initial,actions,targets = windows(selected,6,stride=1)
        prediction = model.rollout(initial,actions)[...,:model.physical_size]
        error = prediction-targets[...,:model.physical_size]
        residual = error/model.delta_scale
        mse = residual.double().square().mean((0,1))
        loss = mse.mean()
        # Same residuals as training; aggregate in float64 for reporting only.
        assert abs(float(loss)-float(residual.square().mean())) < 1e-6*max(1.,float(loss))
        channels = []
        for i,name in enumerate(metadata["physical_names"]):
            channels.append({"name":name,"scale":float(model.delta_scale[i]),
                "pooled_raw_rmse":float(error[...,i].double().square().mean().sqrt()),
                "pooled_normalized_rmse":float(mse[i].sqrt()),"pooled_normalized_mse":float(mse[i]),
                "loss_share":float(mse[i]/mse.sum()),
                "at_20ms_normalized_rmse":float(residual[:,0,i].double().square().mean().sqrt()),
                "at_120ms_normalized_rmse":float(residual[:,-1,i].double().square().mean().sqrt())})
        result["splits"][split] = {"windows":len(initial),"horizon_steps":6,"stride":1,
            "loss_mse":float(loss),"sqrt_loss":float(loss.sqrt()),"per_channel":channels}
    # Preserve a comparison on exactly the published 231 test windows, whose
    # starts/available future differ from the full training-objective audit.
    initial,actions,targets = windows([e for e in episodes if e["split"]=="test"],20,stride=5)
    prediction = model.rollout(initial,actions)[...,:model.physical_size]
    result["published_windows"] = {"windows":len(initial),"stride":5,"horizons":{}}
    for horizon in (1,3,6,20):
        error = (prediction[:,horizon-1]-targets[:,horizon-1,:model.physical_size]).double()
        rmse = error.square().mean(0).sqrt()
        normalized = rmse/model.delta_scale.double()
        result["published_windows"]["horizons"][str(horizon*20)] = {
            "raw_rmse":rmse.tolist(),"normalized_rmse":normalized.tolist(),
            "normalized_loss_mse":float(normalized.square().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("runs/vertical_predictor"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {"definition":"r=(predicted_state-true_state)/training_one_step_delta_std; loss=mean(r^2)",
              "models":"selected bptt checkpoints","variants":{}}
    for variant in range(1,5):
        record = audit(args.run/f"variant_{variant}")
        result["variants"][str(variant)] = record
        print(variant,{s:round(r["loss_mse"],8) for s,r in record["splits"].items()})
    (args.run/"normalized_errors.json").write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
