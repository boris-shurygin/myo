"""Record actual reward tradeoffs and endpoint muscle torques in a failed arm pose."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from myo.arm_reward_curriculum import load_reward_curriculum
from myo.train_vertical_ppo import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controllers',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--angle',type=float,default=125.)
    parser.add_argument('--tilt',type=float,default=135.)
    parser.add_argument('--mass',type=float,default=2.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    report={'scope':'deterministic 4s rollouts; endpoint forces/arms, not control-interval averages; '
                    'finite-horizon returns are not independent critic calibration',
            'condition':{'case':[args.angle,args.angle],'tilt_deg':args.tilt,'payload_mass':args.mass,
                         'capacity':1.,'seconds':4.},'controllers':{}}
    for path in args.controllers:
        model,env,meta=load_reward_curriculum(path)
        try:
            obs,_=env.reset(seed=0,options=report['condition'])
            with torch.no_grad():
                initial_value=float(model.policy.predict_values(torch.as_tensor(obs[None])).item())
                samples=[];train_return=native_return=position_return=0.
                train_discounted=native_discounted=position_discounted=0.
                for index in range(200):
                    action=model.predict(obs,deterministic=True)[0]
                    obs,reward,terminated,truncated,info=env.step(action)
                    base=env.unwrapped
                    outputs=base._state_outputs()
                    forces=np.asarray([o.tendon_force for o in outputs])
                    torques=base.moment_arms*forces
                    native=.1*sum(info['native_reward_components'].values())
                    position=.1*info['native_reward_components']['position_log']
                    train_return+=reward;native_return+=native;position_return+=position
                    discount=model.gamma**index
                    train_discounted+=discount*reward;native_discounted+=discount*native
                    position_discounted+=discount*position
                    samples.append({'time_s':base.time,'q_deg':float(np.rad2deg(info['q'])),
                        'qvel_deg_s':float(np.rad2deg(info['qvel'])),
                        'excitation':np.asarray(action).tolist(),'activation':info['activation'],
                        'capacity':info['capacity'],'tendon_force_N':forces.tolist(),
                        'flexor_endpoint_torque_Nm':float(torques[:3].sum()),
                        'extensor_endpoint_torque_Nm':float(torques[3:].sum()),
                        'passive_limit_torque_Nm':float(base.passive_limit_torque()),
                        'bias_torque_Nm':float(base.data.qfrc_bias[base.dof_index]),
                        'constraint_torque_Nm':float(base.data.qfrc_constraint[base.dof_index]),
                        'peak_contact_force_N':info['substep_peaks']['tissue_contact_force_N'],
                        'reward':reward,'reward_terms':info['reward_components'],
                        'native_reward_terms':info['native_reward_components']})
                    if terminated or truncated:break
                bootstrap=0. if terminated else float(model.policy.predict_values(torch.as_tensor(obs[None])).item())
            tail=[s for s in samples if s['time_s']>=3.-1e-8]
            row={'checkpoint_sha256':meta['checkpoint_sha256'],'reward_stage':meta['contract']['reward_stage'],
                'initial_critic_value':initial_value,'training_return':train_return,'native_return':native_return,
                'position_return':position_return,'training_discounted_return':train_discounted,
                'native_discounted_return':native_discounted,'position_discounted_return':position_discounted,
                'training_bootstrapped_discounted_return':train_discounted+model.gamma**len(samples)*bootstrap,
                'failure':bool(terminated),'tail_rmse_deg':float(np.sqrt(np.mean([(s['q_deg']-args.angle)**2 for s in tail]))) if tail else None,
                'tail_endpoint_torques_Nm':{k:float(np.mean([s[k] for s in tail])) for k in
                    ('flexor_endpoint_torque_Nm','extensor_endpoint_torque_Nm','bias_torque_Nm','constraint_torque_Nm')} if tail else {},
                'samples':samples}
            report['controllers'][str(path)]=row
            write_json(args.output,report)
            print({k:v for k,v in row.items() if k!='samples'},flush=True)
        finally:
            env.close()
    report['status']='complete'
    report['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output,report)


if __name__=='__main__':
    main()
