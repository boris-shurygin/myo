"""Upper actions must replace heuristic proposals, preserve native physics and freeze the lower."""
import numpy as np
import torch
import pytest

from myo.arm_hierarchy import ArmHierarchyEnv, Motion
from myo.arm_learning import ArmPPO
from myo.arm_neural_upper import NeuralUpperEnv, save_upper, load_upper
from myo.policy_beta import BetaExcitationPolicy


def test_upper_has_no_heuristic_answer_and_uses_same_delayed_muscle_path():
    torch.set_num_threads(1)
    physical = ArmHierarchyEnv(motions=(Motion('short',seconds=.2,kind='hold'),))
    lower = ArmPPO(BetaExcitationPolicy,physical,seed=0,n_steps=16,batch_size=16,
                   policy_kwargs={'net_arch':{'pi':[16],'vf':[16]}})
    env = NeuralUpperEnv(lower.policy,physical.contract())
    try:
        before,_ = env.reset(seed=0)
        physical.reset(seed=0)
        env.env.tasks[-1] = 19.
        np.testing.assert_array_equal(before,env.observation())
        for index in range(12):
            physical.tasks[-1] = 3. if index < 10 else 0.
            excitation = lower.policy.predict(physical.observation(),deterministic=True)[0]
            physical.step(excitation)
            _,_,_,_,info = env.step(np.array([.15]))
            np.testing.assert_allclose(env.unwrapped.data.qpos,physical.unwrapped.data.qpos,atol=1e-10)
            np.testing.assert_allclose(env.unwrapped.data.qvel,physical.unwrapped.data.qvel,atol=1e-10)
            np.testing.assert_allclose(info['upper']['muscle_excitation'],excitation,atol=1e-7)
            assert info['upper']['issued'] == (index < 10)
        assert len(env.observation()) == 151
    finally:
        env.close(); physical.close()


def test_upper_ppo_does_not_train_lower_and_bundle_reloads_without_lower_file(tmp_path):
    torch.set_num_threads(1)
    physical = ArmHierarchyEnv(motions=(Motion('short',seconds=.2,kind='hold'),))
    lower = ArmPPO(BetaExcitationPolicy,physical,seed=0,n_steps=16,batch_size=16,
                   policy_kwargs={'net_arch':{'pi':[16],'vf':[16]}})
    env = NeuralUpperEnv(lower.policy,physical.contract())
    physical.close()
    snapshot = {k:v.clone() for k,v in lower.policy.state_dict().items()}
    model = ArmPPO('MlpPolicy',env,seed=0,n_steps=16,batch_size=16,n_epochs=1,
                   policy_kwargs={'net_arch':{'pi':[16],'vf':[16]}})
    before = model.policy.action_net.weight.detach().clone()
    model.learn(32)
    assert not torch.equal(before,model.policy.action_net.weight)
    for k,v in snapshot.items():
        assert torch.equal(v,lower.policy.state_dict()[k])
    path = tmp_path/'upper.zip'
    save_upper(model,env,path,{'test':True})
    restored,replay,_ = load_upper(path)
    try:
        obs,_ = env.reset(seed=0)
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],restored.predict(obs,deterministic=True)[0])
        assert replay.contract() == env.contract()
        assert all(not p.requires_grad for p in replay.lower_policy.parameters())
        replay.reset(seed=0)
        from pathlib import Path
        import runpy
        display = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/demo_arm_hierarchy.py'))['TrajectoryDisplay']
        before_state = replay.unwrapped.observe().copy()
        assert np.isfinite(display(replay).telemetry()['target_q'])
        np.testing.assert_array_equal(before_state,replay.unwrapped.observe())
        with path.open('ab') as stream:
            stream.write(b'bad')
        with pytest.raises(ValueError,match='mismatch'):
            load_upper(path)
    finally:
        env.close(); replay.close()
