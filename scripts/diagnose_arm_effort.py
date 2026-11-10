"""Replay fixed cases to distinguish position drift from speed failures."""
import json
from pathlib import Path
import numpy as np
import torch
from myo.arm_exploration import load_exploration

def main():
    torch.set_num_threads(1)
    root=Path('runs/arm_effort_curriculum')
    policies={'parent':Path('runs/arm_exploration/stall_audited_continuation/best.zip')}
    policies.update({n:root/n/'last.zip' for n in ('control','activation_01','control_lr3e6','activation_01_lr3e6')})
    cases=[{'case':[135.,135.],'payload_mass':2.,'tilt_deg':t,'capacity':1.,'seconds':4.} for t in (120.,150.)]
    cases.append({'case':[130.,130.],'payload_mass':2.,'tilt_deg':135.,'capacity':1.,'seconds':4.})
    out={'scope':'three prespecified failure probes; deterministic means, native mechanics, no policy selection','policies':{}}
    for name,path in policies.items():
        model,env,metadata=load_exploration(path);results=[]
        try:
            for case in cases:
                obs,info=env.reset(seed=0,options=case);trace=[]
                initial_capacity=np.array(info['capacity'])
                while True:
                    action=model.predict(obs,deterministic=True)[0]
                    obs,reward,failed,truncated,info=env.step(action)
                    trace.append({'t':info['t'],'q_deg':float(np.rad2deg(info['q'])),
                        'velocity_deg_s':float(np.rad2deg(info['qvel'])),'excitation':info['excitation'],
                        'delayed_excitation':info['delayed_excitation'],'activation':info['activation'],
                        'tendon_force_N':info['tendon_force_N'],'contact_force_N':info['tissue_contact']['normal_force_N'],
                        'capacity':info['capacity']})
                    if failed or truncated:break
                tail=[r for r in trace if r['t']>3.-1e-8]
                errors=np.array([r['q_deg']-case['case'][1] for r in tail]);vel=np.array([r['velocity_deg_s'] for r in tail]);a=np.array([r['activation'] for r in tail]);u=np.array([r['excitation'] for r in tail])
                position=abs(errors)<=3.;speed=abs(vel)<=5.
                result={'condition':case,'failed':bool(failed),'position_fraction':float(position.mean()),'speed_fraction':float(speed.mean()),'settled_fraction':float((position&speed).mean()),
                    'tail_rmse_deg':float(np.sqrt(np.mean(errors**2))),'tail_speed_rms_deg_s':float(np.sqrt(np.mean(vel**2))),
                    'tail_activation_squared_mean':float(np.mean(a*a)),'tail_activation_mean':a.mean(0).tolist(),'tail_excitation_std':u.std(0).tolist(),
                    'tail_contact_force_mean_N':float(np.mean([r['contact_force_N'] for r in tail])),
                    'capacity_loss':(initial_capacity-np.array(info['capacity'])).tolist(),'trace':trace}
                results.append(result)
                print(json.dumps({'policy':name,**{k:v for k,v in result.items() if k not in ('trace','tail_activation_mean','tail_excitation_std','capacity_loss')}}),flush=True)
        finally:env.close()
        out['policies'][name]={'checkpoint':str(path),'sha256':metadata['checkpoint_sha256'],'cases':results}
        (root/'failure_diagnostics.json').write_text(json.dumps(out,indent=2))
    out['status']='complete';(root/'failure_diagnostics.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
