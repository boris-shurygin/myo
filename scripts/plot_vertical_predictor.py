"""Native trajectories, muscle signals and held-out predictor comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from myo.train_vertical_predictor import load_predictor


def save(figure, root, name):
    figure.tight_layout()
    for extension in ("png","svg"):
        figure.savefig(root/f"{name}.{extension}",dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("runs/vertical_predictor"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    torch.set_num_threads(2)
    titles = ("1: active, rigid tendon", "2: passive fiber + elastic tendon",
              "3: elasticity + transport", "4: elasticity + transport + activation")
    figure,axes = plt.subplots(4,2,figsize=(13,12),sharex="col")
    for row,(key,result) in enumerate(summary["variants"].items()):
        folder = args.run/f"variant_{key}"
        records = json.loads((folder/"episodes.json").read_text())
        index = next(i for i,r in enumerate(records) if r["split"]=="test" and r["family"]=="steps")
        model,_ = load_predictor(folder/"bptt.pt",result["contract"])
        with np.load(folder/"episodes.npz") as data:
            telemetry = data[f"e{index}_telemetry"]
            state,actions = data[f"e{index}_states"],data[f"e{index}_actions"]
        t = telemetry[:,0]
        left,right = axes[row]
        left.plot(t,telemetry[:,1],"--",color=".6",label="heuristic target")
        left.plot(t,telemetry[:,2],color="black",label="native load")
        with torch.no_grad():
            for start in range(50,len(actions)-20,50):
                predicted = model.rollout(torch.tensor(state[start]),torch.tensor(actions[start:start+20])).numpy()
                left.plot(t[start:start+20],predicted[:,0],color="#c25c18",label="400 ms forecasts" if start==50 else None)
        left.set(title=titles[row],ylabel="Displacement, m")
        for muscle,color,label in ((0,"#b63e36","upper"),(1,"#2979a6","lower")):
            right.plot(t,telemetry[:,4+muscle],"--",color=color,alpha=.6,label=f"{label} excitation")
            right.plot(t,telemetry[:,8+muscle],color=color,label=f"{label} activation")
        right.set(ylabel="Excitation / activation",ylim=(-.02,1.02))
        left.grid(alpha=.2)
        right.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time, s")
    figure.suptitle("Native heuristic motion; independent 400 ms forecasts start from measured state")
    save(figure,args.run,"trajectories")
    figure,axes = plt.subplots(2,2,figsize=(12,8))
    colors = ("#637381","#d68400","#277da8","#a44b9c")
    for color,(key,result) in zip(colors,summary["variants"].items()):
        stage = result["stages"]["bptt"]
        horizons = (20,60,120,400)
        axes[0,0].plot(horizons,[1000*stage["test"][str(h)]["position_rmse_m"] for h in horizons],"o-",color=color,label=f"variant {key}")
        axes[0,1].plot(horizons,[stage["test"][str(h)]["velocity_rmse_m_s"] for h in horizons],"o-",color=color)
        axes[1,0].bar(key,stage["derivatives"]["common_60_to_120ms_cosine"],color=color)
        history = stage["history"]
        axes[1,1].plot([x["update"] for x in history],[x["validation_score"] for x in history],color=color)
    axes[0,0].set(title="Position prediction",xlabel="Horizon, ms",ylabel="RMSE, mm")
    axes[0,0].legend()
    axes[0,1].set(title="Velocity prediction",xlabel="Horizon, ms",ylabel="RMSE, m/s")
    axes[1,0].set(title="Velocity Jacobian vs native, 60–120 ms",xlabel="Variant",ylabel="Global cosine")
    axes[1,1].set(title="BPTT validation",xlabel="BPTT updates",ylabel="Mean velocity RMSE, m/s")
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    figure.suptitle("Vertical two-muscle predictor; 20 ms commands, 1 ms MuJoCo physics")
    save(figure,args.run,"comparison")


if __name__ == "__main__":
    main()
