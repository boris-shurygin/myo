"""New context, reset physics and reward must preserve delayed muscle dynamics."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from myo.arm_learning import make_model
from myo.arm_precision import ArmPrecisionEnv, arm_config
from myo.arm_workload import (ArmWorkloadEnv, WorkloadSpec, make_workload_model,
                              load_workload, save_workload)
from myo.compact_protocol import initialize_command_at_reset
from myo.env import PreacherCurlEnv


@pytest.mark.parametrize('tilt,mass', ((30., 1.), (210., 2.)))
def test_reset_mount_and_load_match_native_model_and_periodic_context(tilt, mass):
    env = ArmWorkloadEnv()
    native = PreacherCurlEnv(replace(arm_config(), arm_mount_tilt=float(np.deg2rad(tilt))))
    observation, _ = env.reset(seed=0, options={'case': (60, 90), 'tilt_deg': tilt, 'payload_mass': mass})
    native.reset(seed=0, options={'q': float(np.deg2rad(60)), 'target_angle': float(np.deg2rad(90)),
                                'trigger': 1., 'capacity': 1., 'payload_mass': mass})
    initialize_command_at_reset(native, 90, 1.)
    assert observation.shape == (155,)
    np.testing.assert_array_equal(env.unwrapped.data.qfrc_bias, native.data.qfrc_bias)
    for _ in range(4):
        action = np.linspace(.01, .15, 6, dtype=np.float32)
        native.step(action)
        env.step(action)
        np.testing.assert_array_equal(env.unwrapped.data.qpos, native.data.qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, native.data.qvel)
        for a,b in zip(env.unwrapped.muscles, native.muscles, strict=True):
            np.testing.assert_array_equal(a.state_vector(), b.state_vector())
    start,_ = env.reset(seed=0, options={'case': (60,90), 'tilt_deg': tilt})
    bias = env.unwrapped.data.qfrc_bias.copy()
    repeated,_ = env.reset(seed=0, options={'case': (60,90), 'tilt_deg': tilt+360.})
    np.testing.assert_array_equal(start,repeated)
    np.testing.assert_array_equal(bias,env.unwrapped.data.qfrc_bias)
    env.close();native.close()


def test_reward_proxy_changes_no_physics_or_motor_timing():
    baseline=ArmWorkloadEnv(WorkloadSpec(masses=(1.,)))
    efficient=ArmWorkloadEnv(WorkloadSpec(masses=(1.,),activation_squared_weight=.7,active_fraction_weight=.2))
    baseline.reset(seed=0,options={'case':(60,75)})
    efficient.reset(seed=0,options={'case':(60,75)})
    for _ in range(5):
        u=np.full(6,.2,np.float32)
        a,ra,_,_,_=baseline.step(u)
        b,rb,_,_,info=efficient.step(u)
        np.testing.assert_array_equal(a,b)
        for left,right in zip(baseline.unwrapped.muscles,efficient.unwrapped.muscles,strict=True):
            np.testing.assert_array_equal(left.state_vector(),right.state_vector())
        assert rb-ra==pytest.approx(-.1*(.7*info['workload']['activation_squared_mean']+
                                        .2*info['workload']['active_fraction_mean']))
    baseline.close();efficient.close()


def test_warm_context_transfer_preserves_old_function_and_optimizer_then_reloads(tmp_path):
    torch.set_num_threads(1)
    old=ArmPrecisionEnv()
    parent=make_model(old,n_steps=32,batch_size=16,n_epochs=1)
    parent.learn(32)
    env=ArmWorkloadEnv(WorkloadSpec(masses=(1.,),cases=((20,75),)))
    model=make_workload_model(env,parent,env.contract(),n_steps=32,batch_size=16,n_epochs=1)
    observation,_=env.reset(seed=0,options={'case':(20,75),'tilt_deg':30.})
    assert np.array_equal(observation[-2:],np.zeros(2))
    with torch.no_grad():
        a=torch.as_tensor(observation[None])
        assert torch.equal(parent.policy.predict_values(a[:,:153]),model.policy.predict_values(a))
        assert torch.equal(parent.policy.get_distribution(a[:,:153]).distribution.mean,
                           model.policy.get_distribution(a).distribution.mean)
    for name in ('actor_private','critic_private'):
        previous=getattr(parent.policy.mlp_extractor,name)[0].weight
        current=getattr(model.policy.mlp_extractor,name)[0].weight
        assert torch.count_nonzero(current[:,153:])==0
        for field,value in model.policy.optimizer.state[current].items():
            expected=parent.policy.optimizer.state[previous][field]
            if value.ndim==2:
                assert torch.equal(value[:,:153],expected) and torch.count_nonzero(value[:,153:])==0
            else:
                assert torch.equal(value,expected)
    model.learn(32,reset_num_timesteps=False)
    path=tmp_path/'workload.zip'
    save_workload(model,env,path,{'parent':'test'})
    restored,checked,_=load_workload(path)
    assert model.optimizer_steps==restored.optimizer_steps
    assert all(torch.equal(value,restored.policy.state_dict()[name])
               for name,value in model.policy.state_dict().items())
    checked.close();env.close();old.close()
