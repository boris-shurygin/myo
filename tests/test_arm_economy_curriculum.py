"""Prevent cumulative precision drift when accepting multiple weakening steps."""
from copy import deepcopy

import json
import numpy as np
import pytest
import torch

from myo.arm_economy_curriculum import ArmEconomyEnv, economy_gate, load_economy
from myo.arm_efficiency import ArmEfficiencyEnv, EfficiencyReward
from myo.arm_timing import make_fresh_model
from myo.arm_weakening import make_weakening_model
from myo.arm_workload import WorkloadSpec, save_workload


def test_economy_reward_changes_cost_without_changing_native_dynamics_and_roundtrips(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(.5,), cases=((75.,75.),), seconds=.2)
    env = ArmEconomyEnv(spec, reward=EfficiencyReward(activation_squared_weight=.1))
    original = ArmEfficiencyEnv(spec)
    try:
        parent = make_fresh_model(original, original.contract(), n_steps=16, batch_size=16, n_epochs=1)
        parent.learn(16)
        model = make_weakening_model(env, parent, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        obs,_ = env.reset(seed=0); old,_ = original.reset(seed=0)
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],parent.predict(old,deterministic=True)[0])
        for action in (np.full(6,.4),np.zeros(6),np.ones(6)):
            obs,r,f,t,info = env.step(action)
            old,r0,f0,t0,old_info = original.step(action)
            np.testing.assert_array_equal(obs[:-6],old)
            assert (f,t)==(f0,t0)
            assert r==pytest.approx(r0-.01*info['workload']['activation_squared_mean'],abs=1e-12)
            for name in ('q','qvel','activation','delayed_excitation','tendon_force_N','capacity'):
                np.testing.assert_array_equal(info[name],old_info[name])
        model.learn(32,reset_num_timesteps=False)
        path=tmp_path/'candidate.zip'
        save_workload(model,env,path,{'method':'accumulated_economy'})
        restored,check,meta=load_economy(path)
        try:
            np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],restored.predict(obs,deterministic=True)[0])
            assert meta['contract']['efficiency_reward']['activation_squared_weight']==.1
            assert restored.optimizer_steps==model.optimizer_steps
            old_state=model.policy.optimizer.state_dict()['state'];new_state=restored.policy.optimizer.state_dict()['state']
            for key,row in old_state.items():
                for field,value in row.items():
                    if isinstance(value,torch.Tensor):assert torch.equal(value,new_state[key][field])
        finally:
            check.close()
        metadata=path.with_name(path.name+'.metadata.json');meta['contract']['efficiency_reward']['activation_squared_weight']=0.
        metadata.write_text(json.dumps(meta))
        with pytest.raises(ValueError,match='identity mismatch'):load_economy(path)
    finally:
        env.close();original.close()


def test_accumulated_economy_cannot_accumulate_accuracy_tolerance_or_hide_tail_cost():
    row = dict(case=[75.,75.], payload_mass=.5, tilt_deg=30., capacity=1., seconds=4.,
        failure=False, passed=True, tail_rmse_deg=.2, tail_velocity_rms_deg_s=.1,
        settling_time_s=.3, overshoot_deg=None)
    baseline = dict(cases=[row], activation_squared_mean=.04, tail_activation_squared_mean=.04)
    previous = deepcopy(baseline)
    previous['cases'][0]['tail_rmse_deg'] = .4
    previous.update(activation_squared_mean=.038, tail_activation_squared_mean=.038)
    candidate = deepcopy(previous)
    candidate['cases'][0]['tail_rmse_deg'] = .6
    candidate.update(activation_squared_mean=.036, tail_activation_squared_mean=.036)
    accepted,reasons = economy_gate(candidate, baseline, previous)
    assert not accepted and [0,'position'] in reasons
    candidate['cases'][0]['tail_rmse_deg'] = .35
    assert economy_gate(candidate, baseline, previous)[0]
    # A cheaper cold start cannot disguise more expensive actual holding.
    candidate['tail_activation_squared_mean'] = .039
    accepted,reasons = economy_gate(candidate, baseline, previous)
    assert not accepted and ['tail_activation_squared_mean','insufficient_reduction'] in reasons
