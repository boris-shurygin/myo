"""Physical distinctions and causal prediction in the minimal vertical stand."""
import numpy as np
import pytest

from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_no_elasticity_is_active_force_not_a_position_servo_and_passive_fiber_can_support_load():
    rigid = VerticalMuscles(VerticalConfig(variant=1))
    rigid.reset(-.08)
    initial_velocity = rigid.velocity
    rigid.step(np.zeros(2))
    assert rigid.velocity-initial_velocity == pytest.approx(-9.81*rigid.config.dt,abs=1e-12)
    assert rigid.position < -.08
    np.testing.assert_array_equal(rigid.activation,[0.,0.])
    elastic = VerticalMuscles(VerticalConfig(variant=2))
    elastic.reset(-.08)
    elastic.step(np.zeros(2))
    np.testing.assert_array_equal(elastic.activation,[0.,0.])
    assert elastic.muscles[0].last_output.passive_force > 0
    assert elastic.forces()[0] > 0
    output = elastic.muscles[0].last_output
    assert output.passive_energy > 0 and output.tendon_energy > 0
    assert output.fiber_length/elastic.parameters.optimal_fiber_length-1 > output.tendon_strain


def test_motor_transport_and_activation_are_independent_and_snapshot_restores_both():
    immediate = VerticalMuscles(VerticalConfig(variant=2,frame_skip=1))
    transport = VerticalMuscles(VerticalConfig(variant=3,frame_skip=1))
    gradual = VerticalMuscles(VerticalConfig(variant=4,frame_skip=1))
    for env in (immediate,transport,gradual):
        env.reset()
    initial = transport.activation.copy()
    command = np.array([.7,.3])
    immediate.step(command)
    np.testing.assert_array_equal(immediate.activation,command)
    for _ in range(40):
        transport.step(command)
        gradual.step(command)
    np.testing.assert_array_equal(transport.activation,initial)
    np.testing.assert_allclose(gradual.activation,initial,atol=1e-15,rtol=0)
    transport.step(command)
    gradual.step(command)
    np.testing.assert_array_equal(transport.activation,command)
    np.testing.assert_array_equal(gradual.delivered,command)
    assert np.all(gradual.activation > initial) and np.all(gradual.activation < command)
    snapshot = gradual.snapshot()
    predicted = gradual.step([.2,.1])
    gradual.restore(snapshot)
    np.testing.assert_array_equal(gradual.step([.2,.1]),predicted)


def test_predictor_current_command_and_delayed_command_paths_and_checkpoint_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from myo.train_vertical_predictor import VerticalPredictor,load_predictor,predictor_contract,save_predictor
    for variant in (1,4):
        env = VerticalMuscles(VerticalConfig(variant=variant))
        state = torch.tensor(env.reset())
        model = VerticalPredictor(env.config.physical_size,env.config.queue_steps,hidden=8)
        with torch.no_grad():
            for layer in model.net:
                if isinstance(layer,torch.nn.Linear):
                    layer.weight.fill_(.02)
                    layer.bias.zero_()
        actions = torch.full((6,2),.2)
        jac = torch.autograd.functional.jacobian(lambda u:model.rollout(state,u)[:,1],actions)
        if variant==1:
            assert jac[0,0].abs().sum() > 0
        else:
            assert torch.count_nonzero(jac[:2,0]) == 0
            assert jac[2,0].abs().sum() > 0
            path = tmp_path/"vertical.pt"
            expected = predictor_contract(env)
            save_predictor(path,model,expected,1,"test")
            restored,_ = load_predictor(path,expected)
            torch.testing.assert_close(restored.rollout(state,actions),model.rollout(state,actions),rtol=0,atol=0)
            with pytest.raises(ValueError,match="config"):
                load_predictor(path,predictor_contract(VerticalMuscles(VerticalConfig(variant=3))))
