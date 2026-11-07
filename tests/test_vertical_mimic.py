"""Regressions for reference timing, direct Gaussian actions and PPO credit."""
import json
import numpy as np
import pytest

from myo.vertical_mimic import Motion, VerticalMimicEnv, evaluate_mimic
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_reference_preview_is_kinematic_reset_is_not_teacher_and_actions_keep_native_delay():
    env = VerticalMimicEnv(VerticalConfig(variant=4, frame_skip=10), (Motion("test", .08, 3.),))
    obs, _ = env.reset(seed=0, options={"motion_index": 0, "frame": 31})
    q, v = env.motion.sample(.31)
    assert env.physical.position == pytest.approx(q)
    assert env.physical.velocity == pytest.approx(v)
    np.testing.assert_array_equal(env.physical.activation, [0., 0.])
    np.testing.assert_array_equal(env.physical.excitation, [0., 0.])
    idx = env.observation_names.index("reference/q/+0ms")
    np.testing.assert_allclose(obs[idx:idx+5], env.motion.sample(.31+np.arange(5)*.2)[0], atol=1e-8)
    native = VerticalMuscles(env.physical.config)
    native.restore(env.physical.snapshot())
    for k in range(5):
        raw = np.array([1.4, -.6])
        obs, reward, terminated, truncated, info = env.step(raw)
        native.step([1., .2])
        np.testing.assert_array_equal(env.physical.state(), native.state())
        np.testing.assert_array_equal(obs[-2:], raw.astype(np.float32))
        assert not terminated and not truncated
        np.testing.assert_allclose(info["delivered"], [0., 0.] if k < 4 else [1., .2])
        eq, ev = info["error_q_m"], info["error_v_m_s"]
        expected = (.1*np.exp(-10*eq**2)+.1*np.exp(-2*ev**2)+2*np.exp(-100*np.mean(np.square([0, 0, eq])))
                    +.2+.5+.5*np.exp(-.1*np.mean(np.square([0, 0, ev])))-.01*.4**2/2)
        assert reward == pytest.approx(expected)
        json.dumps(info, allow_nan=False)


def test_reference_preview_clamps_at_clip_end_without_wrapping_or_future_physics():
    env = VerticalMimicEnv(motions=(Motion("test", .07, 3.),))
    obs, _ = env.reset(options={"motion_index": 0, "frame": 598})
    i = env.observation_names.index("reference/q/+0ms")
    assert obs[i+1] == pytest.approx(env.reference(600)[0], abs=1e-8)
    # All lookahead after the end is the last reference frame, not a new cycle.
    np.testing.assert_array_equal(obs[i+1:i+5], np.repeat(obs[i+1], 4))
    env.step([-1., -1.])
    _, _, terminated, truncated, _ = env.step([-1., -1.])
    assert not terminated and truncated and env.frame == 600
    report = evaluate_mimic(lambda obs, **kw: np.array([-1., -1.], dtype=np.float32),
        env.physical.config, env.motions, phases=(598/600,))
    json.dumps(report, allow_nan=False)


def test_reward_matches_executed_upstream_code_including_xyz_mean():
    """Use the actual pinned arithmetic, not a second copy of our formula."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    from myo.vertical_mimic import source_manifest
    root = Path(__file__).resolve().parents[1]/"data/musclemimic_methodology"
    if not root.exists():
        pytest.skip("Explicit public-source preparation required")
    expected = json.loads((Path(__file__).resolve().parents[1]/"configs/vertical_mimic_sources.json").read_text())
    assert source_manifest(root) == expected
    tree = ast.parse((root/"musclemimic/core/reward/trajectory_based.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MimicReward")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    start = next(i for i, n in enumerate(call.body)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "self._use_mean_exp_reward")
    end = next(i for i, n in enumerate(call.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "total_reward" and "nan_to_num" in ast.unparse(n.value))
    arithmetic = ast.Module(body=call.body[start:end+1], type_ignores=[])
    cost_tree = ast.parse((root/"loco_mujoco/core/reward/utils.py").read_text())
    cost_fn = next(n for n in cost_tree.body if isinstance(n, ast.FunctionDef))
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cost_fn], type_ignores=[])
    ast.fix_missing_locations(code)
    namespace = {"np": np}
    exec(compile(code, "upstream_bounds_cost", "exec"), namespace)

    env = VerticalMimicEnv(VerticalConfig(variant=1, frame_skip=10))
    env.reset(seed=0, options={"motion_index": 0, "frame": 0})
    # Change only the reference phase, creating a substantial position error.
    env.frame = 100
    raw = np.array([1.4, -.6])
    _, reward, _, _, info = env.step(raw)
    qr, vr = info["reference_q_m"], info["reference_v_m_s"]
    p = env.physical
    coefficients = dict(_use_mean_exp_reward=False, _quat_in_qpos=np.array([False]),
        _rel_site_ids=np.array([0, 1]), _free_joint_qpos_ind=None,
        _qpos_w_exp=10., _qvel_w_exp=2., _rpos_w_exp=100., _rquat_w_exp=10., _rvel_w_exp=.1,
        _qpos_w_sum=.1, _qvel_w_sum=.1, _rpos_w_sum=2., _rquat_w_sum=.2, _rvel_w_sum=.5,
        _root_pos_w_sum=0., _action_out_of_bounds_coeff=.01, _joint_acc_coeff=0.,
        _joint_torque_coeff=0., _action_rate_coeff=0., _activation_energy_coeff=0.)
    namespace.update(self=NS(**coefficients), backend=np, qpos=np.array([p.position]),
        qvel=np.array([p.velocity]), qpos_traj=np.array([qr]), qvel_traj=np.array([vr]),
        qpos_quat=np.array([]), qpos_quat_traj=np.array([]),
        site_rpos=np.array([[0., 0., .6+p.position]]), site_rpos_traj=np.array([[0., 0., .6+qr]]),
        site_rangles=np.zeros((1, 3)), site_rangles_traj=np.zeros((1, 3)),
        site_rvel=np.array([[0., 0., 0., 0., 0., p.velocity]]),
        site_rvel_traj=np.array([[0., 0., 0., 0., 0., vr]]),
        action=raw, env=NS(mdp_info=NS(action_space=env.action_space)),
        qvel_w_sum=.1, root_vel_w_sum=0.)
    exec(compile(arithmetic, "pinned_MimicReward_arithmetic", "exec"), namespace)
    assert reward == pytest.approx(namespace["total_reward"], abs=1e-14)


@pytest.mark.parametrize("optimizer_type", ["muon", "adam"])
def test_upstream_gae_ppo_update_and_checkpoint_roundtrip(tmp_path, optimizer_type):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from myo.train_vertical_mimic import (SOURCE_ROOT, upstream_modules, make_learner,
        save_checkpoint, load_checkpoint, RewardNormalizer, MimicPolicy)
    if not SOURCE_ROOT.exists():
        pytest.skip("Explicit public-source preparation required")
    # Keep CPU thread creation bounded during this optional training check.
    import jax
    import jax.numpy as jnp
    _, _, loss, buffer = upstream_modules()
    rewards = jnp.array([[0., 0.], [0., 2.], [1., 50.]], dtype=jnp.float32)
    values = jnp.zeros_like(rewards)
    dones = jnp.array([[False, False], [False, True], [True, False]])
    adv, _ = buffer.compute_gae(rewards, values, dones, dones, jnp.zeros(2), .99, .95,
                               bootstrap_mask=~dones)
    np.testing.assert_allclose(adv[:, 0], [.9405**2, .9405, 1.], rtol=1e-6)
    assert float(adv[1, 1]) == 2.  # Next episode's 50 must not cross the boundary.
    _, tangent = jax.value_and_grad(lambda lp: loss.ppo_actor_loss(lp, jnp.zeros(2), jnp.array([1., -1.]), .2))(
        jnp.zeros(2))
    assert tangent[0] < 0 < tangent[1]

    env = VerticalMimicEnv(VerticalConfig(variant=1, frame_skip=10))
    obs, _ = env.reset(seed=0)
    batch = np.stack([obs]*32)
    state, network, act, last_value, update = make_learner(batch, [16, 16], 0, optimizer_type)
    data = []
    for step in range(2):
        state, action, value, lp = act(state, batch, jax.random.PRNGKey(step))
        # Raw Gaussian values, not their clipped physical excitations, enter PPO.
        data.append((batch, np.asarray(action), np.asarray(value), np.ones(32), np.zeros(32, bool), np.asarray(lp)))
    before = state.params
    state, metrics, _ = update(state, *(jnp.asarray(np.stack(x)) for x in zip(*data)),
                                last_value(state, batch), jax.random.PRNGKey(3))
    assert int(state.step) == 32 and np.isfinite(metrics).all()
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(state.params)))
    path = tmp_path/"test.msgpack"
    save_checkpoint(path, state, env, {"hidden": [16, 16], "optimizer_type": optimizer_type}, 64, RewardNormalizer(32))
    restored, restored_network, _, _ = load_checkpoint(path)
    predicted = MimicPolicy(network, state)(obs)
    assert predicted.shape == (2,)
    np.testing.assert_array_equal(predicted, MimicPolicy(restored_network, restored)(obs))
    # Adding optimizer selection must keep the reviewed Muon checkpoints usable.
    if optimizer_type == "muon":
        from myo.train_vertical_mimic import LEGACY_MUON_TRAINER_SHA256
        metadata_path = tmp_path/"test.msgpack.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["training"].pop("optimizer_type")
        metadata["trainer_sha256"] = LEGACY_MUON_TRAINER_SHA256
        metadata_path.write_text(json.dumps(metadata))
        legacy, legacy_network, _, _ = load_checkpoint(path)
        np.testing.assert_array_equal(predicted, MimicPolicy(legacy_network, legacy)(obs))
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)
