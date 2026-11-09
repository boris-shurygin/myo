"""Diagnose cold muscle starts versus static-equilibrium initial states, without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_workload import ArmWorkloadEnv,load_workload
from myo.muscle import SeriesElasticMuscle
from myo.train_vertical_ppo import write_json


def preload_at_reset(env):
    """Choose a valid initial muscle/FIFO/history state; never alter joint state."""
    base=env.unwrapped
    assert base.time==0.
    initial_q=base.data.qpos.copy();initial_v=base.data.qvel.copy()
    lengths=base.muscle_lengths()
    required=float(base.data.qfrc_bias[base.dof_index]-base.passive_limit_torque())
    arms=base.moment_arms.copy()
    capacity=[m.capacity for m in base.muscles]
    def torque(u):
        muscles=[SeriesElasticMuscle(m.params,base.config.physics_dt) for m in base.muscles]
        outputs=[m.reset(float(length),activation=float(a),capacity=c)
                 for m,length,a,c in zip(muscles,lengths,u,capacity,strict=True)]
        return float(arms@np.asarray([o.tendon_force for o in outputs]))
    passive=torque(np.zeros(6))
    selected=arms>0 if required>passive else arms<0
    lo,hi=0.,1.
    endpoints=[torque(selected.astype(float)*value) for value in (lo,hi)]
    assert min(endpoints)<=required<=max(endpoints)
    for _ in range(45):
        mid=(lo+hi)/2
        if (torque(selected.astype(float)*mid)<required)==bool(required>passive):lo=mid
        else:hi=mid
    excitation=selected.astype(float)*(lo+hi)/2
    base._muscle_outputs=[m.reset(float(length),activation=float(a),capacity=c)
        for m,length,a,c in zip(base.muscles,lengths,excitation,capacity,strict=True)]
    base._previous_excitation=excitation.copy()
    base._sensor_history.clear();base._action_history.clear();base._command_history.clear()
    for _ in range(base.config.history_steps):
        base._sensor_history.append(base._sensor_values())
        base._action_history.append(excitation.copy())
        base._command_history.append(base._command_values())
    np.testing.assert_array_equal(initial_q,base.data.qpos)
    np.testing.assert_array_equal(initial_v,base.data.qvel)
    assert base.time==0.
    return {'initial_excitation':excitation.tolist(),'required_torque_Nm':required,
            'equilibrium_torque_Nm':torque(excitation),'joint_state_unchanged':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--angle',type=float,default=125.)
    p.add_argument('--tilt',type=float,default=135.)
    p.add_argument('--mass',type=float,default=1.)
    args=p.parse_args();torch.set_num_threads(1)
    model,checked,meta=load_workload(args.controller);checked.close()
    result={'checkpoint_sha256':meta['checkpoint_sha256'],'cases':[]}
    for kind in ('cold','equilibrium'):
        for actor in ('learned','heuristic'):
            env=ArmWorkloadEnv()
            obs,_=env.reset(seed=0,options={'case':(args.angle,args.angle),'tilt_deg':args.tilt,'payload_mass':args.mass})
            initial=preload_at_reset(env) if kind=='equilibrium' else None
            obs=env.observation()
            trace=[];terms=[]
            while True:
                action=model.predict(obs,deterministic=True)[0] if actor=='learned' else env.unwrapped.baseline_action()
                obs,_,failed,truncated,info=env.step(action)
                trace.append([env.unwrapped.time,float(np.rad2deg(info['q'])),float(np.rad2deg(info['qvel'])),
                              *np.asarray(action).tolist(),*info['activation']])
                terms.append(info['reward_components'])
                if failed or truncated:break
            a=np.asarray(trace);tail=a[:,0]>3.-1e-8
            if not tail.any():tail=np.ones(len(a),dtype=bool)
            row={'initialization':kind,'controller':actor,'initial_state':initial,'failure':bool(failed),
                'tail_rmse_deg':float(np.sqrt(np.mean((a[tail,1]-args.angle)**2))),
                'last_angle_deg':float(a[-1,1]),'samples':trace,
                'tail_reward_components':{key:float(np.mean([r[key] for i,r in enumerate(terms) if tail[i]])) for key in terms[0]}}
            result['cases'].append(row)
            env.close()
            print(json.dumps({k:v for k,v in row.items() if k not in ('samples','tail_reward_components')}),flush=True)
    write_json(args.output,result)


if __name__=='__main__':main()
