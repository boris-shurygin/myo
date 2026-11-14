"""Deterministic native recovery from an external100-ms payload force pulse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_efficiency import load_parent
from myo.arm_weakening import load_weakening


def assess(path, force_N, out):
    schema=json.loads(Path(str(path)+'.metadata.json').read_text())['contract']['schema']
    if schema=='myo.arm_accumulated_economy.v1':
        from myo.arm_economy_curriculum import load_economy
        model,env,meta=load_economy(path)
    else:
        model,env,meta=load_weakening(path) if schema=='myo.arm_weakening.v1' else load_parent(path)
    rows,traces=[],{}
    try:
        for goal in (55.,75.,95.,125.):
            for sign in (-1.,1.):
                obs,_=env.reset(seed=0,options={'case':(goal,goal),'payload_mass':.5,
                    'tilt_deg':30.,'capacity':1.,'seconds':6.})
                base=env.unwrapped;direction=None;samples=[];moments=[]
                for step in range(600):
                    base.data.xfrc_applied[base.payload_id]=0.
                    if 250<=step<260:
                        radius=base.data.xipos[base.payload_id]-base.data.xanchor[base.elbow_id]
                        axis=base.data.xaxis[base.elbow_id]
                        if direction is None:
                            direction=np.cross(axis,radius)
                            direction/=np.linalg.norm(direction)
                        force=sign*force_N*direction
                        base.data.xfrc_applied[base.payload_id,:3]=force
                        moments.append(float(np.cross(radius,force)@axis))
                    obs,_,failed,done,info=env.step(model.predict(obs,deterministic=True)[0])
                    samples.append([info['t'],info['q'],info['qvel'],*info['activation']])
                    if failed or done:
                        break
                base.data.xfrc_applied[base.payload_id]=0.
                a=np.asarray(samples);error=np.rad2deg(a[:,1])-goal;speed=np.rad2deg(a[:,2])
                tail=a[:,0]>5.-1e-8;after=a[:,0]>2.6-1e-8;pre=(a[:,0]>2.3)&(a[:,0]<=2.5+1e-8)
                # Failed short episodes remain failures; finite diagnostic
                # values use their last available samples rather than NaN.
                if not tail.any():tail=np.arange(len(a))>=max(0,len(a)-100)
                if not after.any():after=np.ones(len(a),dtype=bool)
                if not pre.any():pre=np.arange(len(a))==0
                settled=(abs(error)<=3.)&(abs(speed)<=5.)
                stable=np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1]&after)
                row={'goal_deg':goal,'sign':sign,'force_N':force_N,'failure':bool(failed),
                    'passed':bool(not failed and len(a)==600 and settled[tail].mean()>=.9),
                    'tail_rmse_deg':float(np.sqrt(np.mean(error[tail]**2))),
                    'tail_velocity_rms_deg_s':float(np.sqrt(np.mean(speed[tail]**2))),
                    'max_after_pulse_error_deg':float(abs(error[after]).max()),
                    'max_after_pulse_deflection_from_pre_deg':float(abs(np.rad2deg(a[after,1]-a[pre,1].mean())).max()),
                    'recovery_time_s':float(max(0.,a[stable[0],0]-2.6)) if len(stable) else None,
                    'external_torque_mean_Nm':float(np.mean(moments)) if moments else None,
                    'actual_force_control_steps':len(moments),
                    'mean_activation_squared':float(np.mean(a[:,3:]**2))}
                if not failed:
                    assert len(moments)==10 and abs(row['external_torque_mean_Nm'])>0
                rows.append(row);traces[f'{goal}_{sign}']=a
                print(json.dumps({'checkpoint':str(path),**row}),flush=True)
        for goal in (55.,75.,95.,125.):
            negative,positive=traces[f'{goal}_-1.0'],traces[f'{goal}_1.0']
            if min(len(negative),len(positive))>=260:
                np.testing.assert_array_equal(negative[:250],positive[:250])
                assert abs(positive[259,2]-negative[259,2])>1e-4
    finally:
        env.close()
    np.savez_compressed(out,**traces)
    return {'checkpoint':str(path),'checkpoint_sha256':meta['checkpoint_sha256'],'cases':rows,
        'force_path_check':'opposite pulses have identical prehistory and different native velocities at2.6s',
        'passed':sum(r['passed'] for r in rows),'failures':sum(r['failure'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--force',type=float,default=1.)
    args=parser.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    summary={'protocol':'native MuJoCo Cartesian force on payload COM, tangential direction fixed at2.5s; '
        'sign+-1,1N default,100ms; no state reset during episodes, no instantaneous protective torque, '
        'native10ms command delay and15/50ms activation retained; no training; '
        'candidate must retain original passes, per-case tail RMSE<=old+.25deg, speed<=max(5,1.1old), '
        'recovery<=old+.2s and peak deflection<=old+1deg'}
    summary['original']=assess(args.original,args.force,args.out/'original_traces.npz')
    summary['candidate']=assess(args.candidate,args.force,args.out/'candidate_traces.npz')
    reasons=[]
    for i,(old,new) in enumerate(zip(summary['original']['cases'],summary['candidate']['cases'])):
        if new['failure'] or (old['passed'] and not new['passed']):reasons.append([i,'pass'])
        if new['tail_rmse_deg']>old['tail_rmse_deg']+.25:reasons.append([i,'position'])
        if new['tail_velocity_rms_deg_s']>max(5.,1.1*old['tail_velocity_rms_deg_s']):reasons.append([i,'speed'])
        if old['recovery_time_s'] is not None and (new['recovery_time_s'] is None or new['recovery_time_s']>old['recovery_time_s']+.2+1e-8):reasons.append([i,'recovery'])
        if new['max_after_pulse_deflection_from_pre_deg']>old['max_after_pulse_deflection_from_pre_deg']+1:reasons.append([i,'deflection'])
    summary.update(eligible=not reasons,reasons=reasons,status='complete')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
