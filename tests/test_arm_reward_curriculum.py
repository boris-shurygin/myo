"""Reward ablation must preserve physics and reject another reward identity."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from myo.arm_workload import ArmWorkloadEnv, WorkloadSpec, make_workload_model, save_workload
from myo.arm_reward_curriculum import ArmRewardCurriculumEnv, RewardStage, load_reward_curriculum
from myo.arm_learning import make_model
from myo.arm_precision import ArmPrecisionEnv


def test_position_only_and_efficiency_change_reward_without_changing_native_dynamics():
    spec=WorkloadSpec(masses=(2.,),activation_squared_weight=.2)
    environments=[ArmWorkloadEnv(spec)]+[ArmRewardCurriculumEnv(spec,RewardStage(mode))
        for mode in ('native','position_only','position_efficiency')]
    try:
        for env in environments:
            env.reset(seed=0,options={'case':(125,125),'tilt_deg':135.})
        for _ in range(6):
            results=[env.step(np.linspace(.01,.8,6,dtype=np.float32)) for env in environments]
            for env,result in zip(environments[1:],results[1:]):
                np.testing.assert_array_equal(result[0],results[0][0])
                np.testing.assert_array_equal(env.unwrapped.data.qpos,environments[0].unwrapped.data.qpos)
                for a,b in zip(env.unwrapped.muscles,environments[0].unwrapped.muscles,strict=True):
                    np.testing.assert_array_equal(a.state_vector(),b.state_vector())
            assert results[1][1]==results[0][1]
            assert results[2][1]==pytest.approx(.1*results[0][4]['reward_components']['position_log'])
            assert results[3][1]==pytest.approx(results[2][1]+.1*results[0][4]['reward_components']['activation_squared_proxy_cost'])
        # Experimental-domain termination and its penalty survive the ablation.
        env=environments[2]
        env.reset(seed=0,options={'case':(165,165),'tilt_deg':135.})
        _,reward,terminated,_,info=env.step(np.zeros(6,dtype=np.float32))
        assert terminated and info['reward_components']['physical_failure']==-1.
        assert reward==pytest.approx(.1*sum(info['reward_components'].values()))
    finally:
        for env in environments:env.close()


def test_reward_checkpoint_reloads_and_rejects_another_reward_stage(tmp_path):
    torch.set_num_threads(1)
    old=ArmPrecisionEnv()
    parent=make_model(old,n_steps=32,batch_size=16,n_epochs=1)
    env=ArmRewardCurriculumEnv()
    try:
        model=make_workload_model(env,parent,env.contract(),n_steps=32,batch_size=16,n_epochs=1)
        model.learn(32)
        path=tmp_path/'reward.zip'
        save_workload(model,env,path,{'spec':{},'reward_stage':{'mode':'position_only'}})
        restored,checked,_=load_reward_curriculum(path)
        checked.close()
        assert all(torch.equal(v,restored.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        meta_path=tmp_path/'reward.zip.metadata.json'
        meta=json.loads(meta_path.read_text())
        meta['contract']['reward_stage']['mode']='native'
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError,match='embedded identity'):
            load_reward_curriculum(path)
    finally:
        env.close();old.close()
