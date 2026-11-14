"""Protect explicit command learning and preserve PPO likelihood/physical state."""
import json

import numpy as np
import pytest
import torch

from myo.arm_efficiency import ArmEfficiencyEnv, EfficiencyReward
from myo.arm_timing import make_fresh_model
from myo.arm_weakening import (ArmWeakeningEnv, WeakeningSpec, load_weakening,
    make_weakening_model, weaken_actor, weakening_labels)
from myo.arm_workload import WorkloadSpec, save_workload
from myo.arm_weakening_objective import weaken_with_moment_constraint


def test_self_labels_are_gated_causal_torque_balanced_and_change_nominal_commands(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=((75.,75.),), seconds=.2)
    env = ArmWeakeningEnv(spec, weakening=WeakeningSpec(fraction=.1))
    original = ArmEfficiencyEnv(spec)
    try:
        parent = make_fresh_model(original, original.contract(), n_steps=16, batch_size=16, n_epochs=1)
        parent.learn(16)
        model = make_weakening_model(env, parent, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        obs, _ = env.reset(seed=0)
        old, _ = original.reset(seed=0)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],parent.predict(old,deterministic=True)[0])
        for action in (np.ones(6), np.zeros(6), np.full(6,.3)):
            obs,r,f,t,info = env.step(action)
            old,r0,f0,t0,actual = original.step(action)
            np.testing.assert_array_equal(obs[:-6],old)
            assert (r,f,t)==(r0,f0,t0)
            for k in ('q','qvel','activation','delayed_excitation','tendon_force_N','capacity'):
                np.testing.assert_array_equal(info[k],actual[k])
        # Two explicit sensory states: settled and moving. Gains are copied from
        # the actual observation; only the gate's sensory coordinates vary.
        x=np.repeat(obs[None],2,0)
        c=env.contract(); q=c['observation_names'].index('angle_centered_rad')
        goal=c['observation_names'].index('requested_target_centered_rad')
        v=c['observation_names'].index('angular_velocity_scaled')
        x[:,q]=x[:,goal];x[:,v]=[0.,1.]
        _,target,gate,alpha,beta=weakening_labels(model,x,c)
        old_mean=alpha/(alpha+beta)
        assert gate.tolist()==[True,False]
        torch.testing.assert_close(target[1],old_mean[1])
        assert bool((target[0]<old_mean[0]).any())
        gains=torch.as_tensor(x[:,-6:])
        torch.testing.assert_close((target*gains).sum(-1),(old_mean*gains).sum(-1),atol=1e-6,rtol=1e-5)
        critic={k:p.clone() for k,p in model.policy.state_dict().items() if 'critic' in k or 'value_' in k}
        before=model.predict(x[0],deterministic=True)[0]
        report=weaken_actor(model,x,c,updates=12,batch_size=16)
        after=model.predict(x[0],deterministic=True)[0]
        assert report['accepted_updates']>0 and after.mean()<before.mean()
        assert all(torch.equal(p,model.policy.state_dict()[k]) for k,p in critic.items())
        # This is actual nominal learning, not an external sampling correction.
        d=model.policy.get_distribution(torch.as_tensor(x))
        a=d.sample();torch.testing.assert_close(d.log_prob(a),d.distribution.log_prob(a).sum(-1))
        model.learn(32,reset_num_timesteps=False)
        path=tmp_path/'weakening.zip';save_workload(model,env,path,{'test':True})
        restored,check,_=load_weakening(path)
        try:
            np.testing.assert_array_equal(model.predict(x[0],deterministic=True)[0],restored.predict(x[0],deterministic=True)[0])
            assert restored.optimizer_steps==model.optimizer_steps
        finally:
            check.close()
        metadata=path.with_name(path.name+'.metadata.json');m=json.loads(metadata.read_text())
        m['contract']['weakening']['fraction']=.2;metadata.write_text(json.dumps(m))
        with pytest.raises(ValueError,match='identity mismatch'):
            load_weakening(path)
    finally:
        env.close();original.close()


def test_rejected_weakening_step_restores_both_weights_and_optimizer():
    torch.set_num_threads(1)
    env=ArmWeakeningEnv(WorkloadSpec(cases=((75.,75.),)),weakening=WeakeningSpec(mean_kl_limit=1e-12,maximum_kl_limit=1e-12))
    original=ArmEfficiencyEnv(env.workload)
    try:
        parent=make_fresh_model(original,original.contract(),n_steps=16,batch_size=16,n_epochs=1)
        parent.learn(16)
        model=make_weakening_model(env,parent,env.contract())
        x=np.repeat(env.reset(seed=0)[0][None],16,0)
        names=env.observation_names;x[:,0]=x[:,names.index('requested_target_centered_rad')];x[:,1]=0.
        weights={k:v.clone() for k,v in model.policy.state_dict().items()}
        moments={k:{f:v.clone() for f,v in row.items() if isinstance(v,torch.Tensor)} for k,row in model.policy.optimizer.state_dict()['state'].items()}
        report=weaken_actor(model,x,env.contract(),updates=1,batch_size=16)
        assert report['stop']=='kl_budget' and report['accepted_updates']==0
        assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in weights.items())
        current=model.policy.optimizer.state_dict()['state']
        assert all(torch.equal(v,current[k][f]) for k,row in moments.items() for f,v in row.items())
    finally:
        env.close();original.close()


def test_moment_objective_reduces_learned_torque_drift_while_weakening_commands():
    # Balanced targets alone did not keep the fitted network's moment balanced.
    # Compare real actor/RMSprop updates, not just the formula for the labels.
    torch.set_num_threads(1)
    env=ArmWeakeningEnv(WorkloadSpec(masses=(.5,),cases=((75.,75.),),seconds=.2))
    original=ArmEfficiencyEnv(env.workload)
    try:
        parent=make_fresh_model(original,original.contract(),n_steps=16,batch_size=16,n_epochs=1)
        parent.learn(16)
        x=np.repeat(env.reset(seed=0)[0][None],32,0)
        x[:,0]=x[:,env.observation_names.index('requested_target_centered_rad')];x[:,1]=0.
        obs=torch.as_tensor(x);drift=[]
        for update in (weaken_actor,weaken_with_moment_constraint):
            model=make_weakening_model(env,parent,env.contract(),learning_rate=3e-6)
            critic={k:p.clone() for k,p in model.policy.state_dict().items() if 'critic' in k or 'value_' in k}
            with torch.no_grad():before=model.policy.get_distribution(obs).distribution.mean.clone()
            report=update(model,x,env.contract(),updates=64,batch_size=32)
            with torch.no_grad():after=model.policy.get_distribution(obs).distribution.mean.clone()
            assert report['accepted_updates']>0 and after.mean()<before.mean()
            assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in critic.items())
            drift.append(float(((after-before)*obs[:,-6:]).sum(-1).square().mean().sqrt()))
        assert drift[1]<.75*drift[0]
    finally:
        env.close();original.close()
