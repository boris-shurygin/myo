"""Train only forward predictors on four minimal vertical-muscle variants."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .vertical_muscles import VerticalConfig, VerticalMuscles, heuristic, reference


class VerticalPredictor(nn.Module):
    def __init__(self, physical_size, queue_steps, hidden=128):
        super().__init__()
        self.physical_size,self.queue_steps,self.hidden = physical_size,queue_steps,hidden
        self.register_buffer("input_mean",torch.zeros(physical_size+2))
        self.register_buffer("input_scale",torch.ones(physical_size+2))
        self.register_buffer("delta_mean",torch.zeros(physical_size))
        self.register_buffer("delta_scale",torch.ones(physical_size))
        self.net = nn.Sequential(nn.Linear(physical_size+2,hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,physical_size))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def inputs(self, state, action):
        delivered = state[...,self.physical_size:self.physical_size+2] if self.queue_steps else action
        return torch.cat((state[...,:self.physical_size],delivered),dim=-1)

    def fit_normalization(self, episodes):
        states = np.concatenate([e["states"][:-1] for e in episodes])
        actions = np.concatenate([e["actions"] for e in episodes])
        following = np.concatenate([e["states"][1:] for e in episodes])
        x = self.inputs(torch.tensor(states),torch.tensor(actions)).numpy()
        delta = following[:,:self.physical_size]-states[:,:self.physical_size]
        for name,value in (("input_mean",x.mean(0)),("input_scale",np.maximum(x.std(0),1e-5)),
                           ("delta_mean",delta.mean(0)),("delta_scale",np.maximum(delta.std(0),1e-6))):
            getattr(self,name).copy_(torch.tensor(value,dtype=torch.float32))

    def forward(self, state, action):
        x = (self.inputs(state,action)-self.input_mean)/self.input_scale
        physical = state[...,:self.physical_size]+self.delta_mean+self.delta_scale*self.net(x)
        if not self.queue_steps:
            return physical
        queue = state[...,self.physical_size:].reshape(*state.shape[:-1],self.queue_steps,2)
        pending = torch.cat((queue[...,1:,:],action.unsqueeze(-2)),dim=-2)
        return torch.cat((physical,pending.flatten(-2)),dim=-1)

    def rollout(self, state, actions):
        result = []
        for step in range(actions.shape[-2]):
            state = self(state,actions[...,step,:])
            result.append(state)
        return torch.stack(result,dim=-2)


def predictor_contract(env):
    return {**env.contract(),"predictor_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save_predictor(path, model, contract, update, stage):
    architecture = {"physical_size":model.physical_size,"queue_steps":model.queue_steps,"hidden":model.hidden}
    torch.save({"architecture":architecture,"state_dict":model.state_dict()},path)
    metadata = {**contract,"architecture":architecture,"update":update,"stage":stage,
        "seed":0,"normalization":"train episodes only",
        "target":"physical state increments; exact non-neural transport queue",
        "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+".metadata.json").write_text(json.dumps(metadata,indent=2)+'\n')


def load_predictor(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Predictor weights differ from metadata")
    for key,value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Vertical predictor contract mismatch: {key}")
    saved = torch.load(path,map_location="cpu",weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Predictor architecture differs from metadata")
    model = VerticalPredictor(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(),metadata


def collect(config, seconds):
    env = VerticalMuscles(config)
    episodes,probes = [],[]
    for split,versions in (("train",(0,1)),("validation",(2,)),("test",(3,))):
        for family in ("sine","steps","multisine"):
            for version in versions:
                position = (-.06,0.,.06,-.03)[version]
                states,actions,telemetry = [env.reset(position)],[],[]
                episode_id = f"{split}/{family}/v{version}"
                episode_index = len(episodes)
                for step in range(round(seconds/config.dt)):
                    if split == "test" and step in (100,250):
                        probes.append({"episode_index":episode_index,"step":step,"snapshot":env.snapshot()})
                    action = heuristic(env,family,version)
                    state = env.step(action)
                    target,_ = reference(env.time,family,version)
                    actions.append(action.astype(np.float32))
                    states.append(state)
                    telemetry.append(np.r_[env.time,target,env.position,env.velocity,action,
                                            env.delivered,env.activation,env.fiber_lengths(),env.forces()])
                episodes.append({"id":episode_id,"split":split,"family":family,"version":version,
                    "initial_position_m":position,"states":np.asarray(states),"actions":np.asarray(actions),
                    "telemetry":np.asarray(telemetry)})
    return env,episodes,probes


def windows(episodes, horizon, stride=1):
    initial,actions,targets = [],[],[]
    for e in episodes:
        for start in range(0,len(e["actions"])-horizon+1,stride):
            initial.append(e["states"][start])
            actions.append(e["actions"][start:start+horizon])
            targets.append(e["states"][start+1:start+horizon+1])
    return tuple(torch.tensor(np.asarray(x),dtype=torch.float32) for x in (initial,actions,targets))


@torch.no_grad()
def evaluate(model, episodes):
    initial,actions,target = windows(episodes,20,stride=5)
    prediction = model.rollout(initial,actions)
    result = {}
    for h in (1,3,6,20):
        error = prediction[:,h-1]-target[:,h-1]
        before_p = initial[:,1] if h==1 else prediction[:,h-2,1]
        before_t = initial[:,1] if h==1 else target[:,h-2,1]
        result[str(h*20)] = {"position_rmse_m":float(error[:,0].square().mean().sqrt()),
            "velocity_rmse_m_s":float(error[:,1].square().mean().sqrt()),
            "interval_acceleration_rmse_m_s2":float((((prediction[:,h-1,1]-before_p)-(target[:,h-1,1]-before_t))/.02).square().mean().sqrt()),
            "persistence_velocity_rmse_m_s":float((initial[:,1]-target[:,h-1,1]).square().mean().sqrt()),
            "samples":len(initial),"nonfinite_fraction":float((~torch.isfinite(prediction[:,h-1])).float().mean())}
    return result


def train(env, episodes, args, output):
    torch.manual_seed(0)
    model = VerticalPredictor(env.config.physical_size,env.config.queue_steps)
    split = {s:[e for e in episodes if e["split"]==s] for s in ("train","validation","test")}
    model.fit_normalization(split["train"])
    metadata = predictor_contract(env)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    generator = torch.Generator().manual_seed(0)
    result = {"contract":metadata,"parameters":sum(p.numel() for p in model.parameters()),"stages":{}}
    for stage,horizon,updates in (("one_step",1,args.one_step_updates),("bptt",6,args.bptt_updates)):
        initial,actions,target = windows(split["train"],horizon)
        history,best,best_update = [],float("inf"),0
        started = time.monotonic()
        for update in range(1,updates+1):
            ids = torch.randint(len(initial),(128,),generator=generator)
            prediction = model.rollout(initial[ids],actions[ids])
            residual = (prediction[...,:model.physical_size]-target[ids,:,:model.physical_size])/model.delta_scale
            loss = residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite vertical predictor loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            if update==1 or update%200==0 or update==updates:
                metrics = evaluate(model,split["validation"])
                # One-step selects one-step error; BPTT selects a fixed
                # physical-time mean, without looking at test results.
                score = metrics["20"]["velocity_rmse_m_s"] if stage=="one_step" else float(np.mean([
                    metrics[str(h)]["velocity_rmse_m_s"] for h in (20,60,120)]))
                history.append({"update":update,"loss":float(loss.detach()),"gradient_norm_before_clip":float(norm),
                                "validation_score":score,"validation":metrics})
                if score < best:
                    best,best_update = score,update
                    save_predictor(output/f"{stage}.pt",model,metadata,update,stage)
                print(f"variant={env.config.variant} {stage} {update}/{updates}: loss={float(loss.detach()):.5g} val={score:.5g}",flush=True)
        selected,_ = load_predictor(output/f"{stage}.pt",metadata)
        result["stages"][stage] = {"updates":updates,"best_update":best_update,"elapsed_s":time.monotonic()-started,
            "history":history,"test":evaluate(selected,split["test"])}
        # Start BPTT from the selected one-step model, with a fresh optimizer.
        model = copy.deepcopy(selected)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    return result


def audit_derivatives(env, episodes, probes, model, output):
    native,predicted = [],[]
    for probe in probes:
        episode = episodes[probe["episode_index"]]
        step = probe["step"]
        commands = episode["actions"][step:step+6].astype(float)
        def response(first):
            env.restore(probe["snapshot"])
            return np.array([env.step(first if i==0 else u)[1] for i,u in enumerate(commands)])
        jac = []
        for i in range(2):
            high,low = commands[0].copy(),commands[0].copy()
            high[i],low[i] = min(1.,high[i]+.005),max(0.,low[i]-.005)
            jac.append((response(high)-response(low))/(high[i]-low[i]))
        native.append(np.stack(jac,axis=-1))
        state = torch.tensor(episode["states"][step])
        continuation = torch.tensor(commands[1:],dtype=torch.float32)
        estimated = torch.autograd.functional.jacobian(lambda u:model.rollout(state,torch.cat((u[None],continuation)))[:,1],
                                                       torch.tensor(commands[0],dtype=torch.float32))
        predicted.append(estimated.detach().numpy())
    actual,estimate = np.asarray(native),np.asarray(predicted)
    np.savez_compressed(output,native=actual,predicted=estimate)
    start = env.config.queue_steps
    a,b = actual[:,start:],estimate[:,start:]
    mask = np.abs(a)>.02
    common_a,common_b = actual[:,2:],estimate[:,2:]
    return {"probes":len(probes),"physical_substeps":len(probes)*4*6*env.config.frame_skip,
        "post_delay_velocity_jacobian_rmse":float(np.sqrt(np.mean((a-b)**2))),
        "zero_jacobian_baseline_rmse":float(np.sqrt(np.mean(a*a))),
        "global_cosine":float(a.ravel()@b.ravel()/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b))),
        "meaningful_sign_agreement":float(np.mean(np.sign(a[mask])==np.sign(b[mask]))),
        "common_60_to_120ms_rmse":float(np.sqrt(np.mean((common_a-common_b)**2))),
        "common_60_to_120ms_cosine":float(common_a.ravel()@common_b.ravel()/max(1e-12,np.linalg.norm(common_a)*np.linalg.norm(common_b))),
        "pre_delay_max_abs":float(np.max(np.abs(estimate[:,:start]))) if start else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("runs/vertical_predictor"))
    parser.add_argument("--seconds",type=float,default=8.)
    parser.add_argument("--one-step-updates",type=int,default=1000)
    parser.add_argument("--bptt-updates",type=int,default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    summary = {"schema":"myo.vertical_predictor_experiment.v1","status":"running",
        "versions":{"torch":torch.__version__,"numpy":np.__version__},
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "variants":{},"scope":"only predictor learning, heuristic commands, no actor or critic"}
    started = time.monotonic()
    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    report()
    for variant in (1,2,3,4):
        output = args.output_dir/f"variant_{variant}"
        output.mkdir()
        env,episodes,probes = collect(VerticalConfig(variant=variant),args.seconds)
        np.savez_compressed(output/"episodes.npz",**{f"e{i}_{k}":e[k] for i,e in enumerate(episodes)
                                                   for k in ("states","actions","telemetry")})
        (output/"episodes.json").write_text(json.dumps([{k:v for k,v in e.items() if k not in ("states","actions","telemetry")}
                                                      for e in episodes],indent=2)+'\n')
        result = train(env,episodes,args,output)
        for stage in result["stages"]:
            model,_ = load_predictor(output/f"{stage}.pt",predictor_contract(env))
            result["stages"][stage]["derivatives"] = audit_derivatives(env,episodes,probes,model,output/f"jacobian_{stage}.npz")
        result["collection"] = {"episodes":len(episodes),"split_episodes":{s:[e["id"] for e in episodes if e["split"]==s]
            for s in ("train","validation","test")},"native_substeps":sum(len(e["actions"]) for e in episodes)*env.config.frame_skip,
            "position_min_m":float(min(e["states"][:,0].min() for e in episodes)),
            "position_max_m":float(max(e["states"][:,0].max() for e in episodes))}
        summary["variants"][str(variant)] = result
        report()
    summary.update(status="complete",elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
