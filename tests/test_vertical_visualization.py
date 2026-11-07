"""Independent material lengths, volume deformation and no physics mutation."""
import mujoco
import numpy as np
import pytest

from myo.vertical_muscles import VerticalConfig, VerticalMuscles
from myo.vertical_visualization import append_tissues, tissue_cylinders
from myo.visualization import activation_rgba, tendon_rgba


def test_tissues_follow_series_state_at_same_pose_preserve_volume_and_leave_physics_untouched():
    env = VerticalMuscles(VerticalConfig(variant=4))
    env.reset()
    lengths = []
    for activation in (.15,.6):
        # Two legitimate fiber/tendon equilibria at the same initial load pose.
        for muscle,path in zip(env.muscles,env.path_lengths()):
            muscle.reset(float(path),activation=activation)
        env.activation[:] = activation
        before = env.state().copy()
        cylinders = tissue_cylinders(env)
        assert len(cylinders) == 4
        scene = mujoco.MjvScene(env.model,maxgeom=16)
        assert append_tissues(scene,env) == 4
        np.testing.assert_array_equal(env.state(),before)
        for i in range(2):
            muscle,tendon = cylinders[i*2:i*2+2]
            assert muscle.length == pytest.approx(env.muscles[i].fiber_length)
            assert tendon.length == pytest.approx(env.path_lengths()[i]-env.muscles[i].fiber_length)
            np.testing.assert_allclose(muscle.end,tendon.start,atol=1e-14)
            np.testing.assert_allclose(tendon.end,[0,0,env.config.half_span_m+env.position],atol=1e-14)
            assert muscle.radius**2*muscle.length == pytest.approx(.024**2*env.parameters.optimal_fiber_length)
            assert tendon.radius**2*tendon.length == pytest.approx(.008**2*env.parameters.tendon_slack_length)
            np.testing.assert_array_equal(muscle.rgba,activation_rgba(activation))
            np.testing.assert_array_equal(tendon.rgba,tendon_rgba(env.forces()[i]/env.parameters.max_isometric_force))
        # MuJoCo cylinder sizes are radius, radius, half length.
        for geom,cylinder in zip(scene.geoms[:4],cylinders):
            assert geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            assert 2*geom.size[2] == pytest.approx(cylinder.length,rel=1e-6)
            assert geom.size[0] == pytest.approx(cylinder.radius,rel=1e-6)
        lengths.append(cylinders)
    assert env.position == 0.
    assert lengths[1][0].length < lengths[0][0].length
    assert lengths[1][0].radius > lengths[0][0].radius
    assert lengths[1][1].length > lengths[0][1].length
    assert lengths[1][1].radius < lengths[0][1].radius
    rigid = VerticalMuscles(VerticalConfig(variant=1))
    for position in (-.08,.08):
        rigid.reset(position)
        tendons = [c for c in tissue_cylinders(rigid) if c.tissue=="tendon"]
        assert all(t.length == pytest.approx(.12) and t.radius == pytest.approx(.008) for t in tendons)
