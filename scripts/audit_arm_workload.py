"""Measure native segment masses, static torque reserve and antagonist ablation."""
from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_learning import load_arm
from myo.arm_precision import VALIDATION_CASES
from myo.arm_workload import ArmWorkloadEnv, evaluate_workload, scenarios
from myo.muscle import SeriesElasticMuscle
from myo.train_vertical_ppo import write_json


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effort-reference',action='store_true',
                        help='Only measure heuristic accuracy/effort on common validation and long holds')
    args=parser.parse_args()
    if args.effort_reference:
        root=Path('runs/arm_workload_reference');root.mkdir(exist_ok=False,parents=True)
        env=ArmWorkloadEnv();contract=env.contract();env.close()
        report={'contract':contract,'controller':'native heuristic; a reference, not an energy optimum'}
        conditions={'validation':scenarios(VALIDATION_CASES,masses=(.5,1.,2.)),
            'long_holds':scenarios(((60,60),(120,120)),masses=(.75,2.),capacities=(.85,),seconds=20.)}
        for name,cases in conditions.items():
            result=evaluate_workload('heuristic',cases)
            report[name]=result
            print(json.dumps({'set':name,'score_deg':result['score_deg'],'passed':result['passed'],
                'total':result['total'],'a2':result['activation_squared_mean'],
                'active_fraction':result['active_fraction_mean']}),flush=True)
        write_json(root/'reference.json',report)
        return
    root=Path('runs/arm_workload_audit');root.mkdir(exist_ok=False,parents=True)
    env=ArmWorkloadEnv()
    base=env.unwrapped
    env.reset(seed=0,options={'case':(75,75),'payload_mass':1.})
    model=base.model
    report={'moving_arm_mass_kg':float(model.body_subtreemass[model.body('ulna_r').id]-1.),
        'humerus_mass_kg':float(model.body_mass[model.body('humerus_r').id]),
        'method':'fixed-pose muscle-only equilibrium, fatigue rate0 for static capacity measurement; native force/delay/tendon law',
        'static':[]}
    for q in (20,30,60,90,120,125,135):
        env.reset(seed=0,options={'case':(q,q),'payload_mass':1.})
        lengths=base.muscle_lengths();arms=base.moment_arms.copy()
        reserves={}
        for capacity in (1.,.7):
            torques=[]
            for pattern in ((0,)*6,(1,1,1,0,0,0),(0,0,0,1,1,1)):
                muscles=[SeriesElasticMuscle(replace(m.params,fatigue_rate=0.,recovery_rate=0.),.001) for m in base.muscles]
                for m,length in zip(muscles,lengths,strict=True):
                    m.reset(float(length),activation=.03,capacity=capacity)
                for _ in range(1200):
                    outputs=[m.step(float(u),float(length),0.) for m,u,length in zip(muscles,pattern,lengths,strict=True)]
                torques.append(float(arms @ np.array([o.tendon_force for o in outputs])))
            reserves[capacity]=torques
        for tilt in range(0,360,30):
            for mass in (.5,1.,2.,2.5):
                env.reset(seed=0,options={'case':(q,q),'tilt_deg':tilt,'payload_mass':mass})
                required=float(base.data.qfrc_bias[base.dof_index]-base.passive_limit_torque())
                for capacity,torques in reserves.items():
                    report['static'].append({'q_deg':q,'tilt_deg':tilt,'mass_kg':mass,'capacity':capacity,
                        'required_muscle_torque_Nm':required,'relaxed_muscle_torque_Nm':torques[0],
                        'max_flexion_Nm':torques[1],'max_extension_Nm':torques[2],
                        'within_active_range':bool(torques[2]<=required<=torques[1])})
    env.close()
    actor,checked,_=load_arm(Path('runs/arm_continuation/best.zip'));checked.close()
    holds=scenarios(((q,q) for q in (30,60,75,90,105,120)),seconds=20.)
    report['baseline_holds']=evaluate_workload(actor,holds)
    report['zero_extensor_holds']=evaluate_workload(actor,holds,mask_extensors=True)
    report['mass_baseline']=evaluate_workload(actor,scenarios(((20,75),(75,20),(60,60),(120,120)),masses=(.5,1.,2.)))
    write_json(root/'audit.json',report)
    print(json.dumps({'static_outside':sum(not r['within_active_range'] for r in report['static']),
        'static_total':len(report['static']),'mass_baseline_deg':report['mass_baseline']['score_deg'],
        'native_holds_deg':report['baseline_holds']['score_deg'],'zero_extensor_deg':report['zero_extensor_holds']['score_deg']}),flush=True)


if __name__=='__main__':main()
