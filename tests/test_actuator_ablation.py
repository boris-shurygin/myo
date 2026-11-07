"""The diagnostic shortcut must be real, isolated and impossible to load as full physics."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from myo.actuator_ablation import (
    InstantRigidMuscle, contract, descriptor, experiment_digest,
    load_ablation_controller, make_environment,
)
from myo.env import PreacherCurlConfig
from myo.muscle import MuscleParameters, SeriesElasticMuscle
from myo.runtime import load_controller


def test_instant_rigid_transmits_force_without_filter_or_tendon_extension():
    params = replace(MuscleParameters(), motor_delay=0., fatigue_rate=0., recovery_rate=0.)
    rigid = InstantRigidMuscle(params, .001)
    full = SeriesElasticMuscle(replace(params, motor_delay=.04), .001)
    length = params.optimal_fiber_length + params.tendon_slack_length
    rigid.reset(length)
    full.reset(length)
    force = rigid.step(.6, length)
    delayed = full.step(.6, length)
    assert force.excitation == force.delayed_excitation == force.activation == .6
    assert force.tendon_force == pytest.approx(.6 * params.max_isometric_force)
    assert force.fiber_length == pytest.approx(params.optimal_fiber_length)
    assert force.tendon_length == pytest.approx(params.tendon_slack_length)
    assert force.tendon_strain == force.tendon_energy == 0
    assert delayed.activation == delayed.tendon_force == 0
    # Preserve the force-velocity law instead of replacing a muscle by torque.
    assert rigid.sample(length, .1).tendon_force > force.tendon_force
    assert rigid.sample(length, -.1).tendon_force < force.tendon_force
    relaxed = rigid.step(0., length)
    assert relaxed.activation == 0
    assert relaxed.tendon_force == pytest.approx(0., abs=1e-8)


def arm_config():
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets for actuator diagnostics")
    return PreacherCurlConfig(**json.loads(Path("configs/d1_train.json").read_text()))


def test_diagnostic_preserves_input_size_and_separates_endpoint_tissue_signals():
    config = arm_config()
    full = make_environment("full", config)
    simple = make_environment("instant_rigid", config)
    try:
        options = {"q": np.deg2rad(20), "qvel": 0., "target_angle": np.deg2rad(30),
                   "trigger": 1., "capacity": 1., "payload_mass": 1.}
        full.reset(seed=0, options=options)
        obs, _ = simple.reset(seed=0, options=options)
        assert obs.shape == full.observation_space.shape == simple.observation_space.shape == (773,)
        assert simple.config.motor_delay == 0
        assert all(m.delay.delay_steps == 0 for m in simple.muscles)
        assert np.count_nonzero(obs[-simple.padding_count:]) == 0
        assert simple.observation_names[-simple.padding_count].startswith("inactive_delay_padding/")
        command = np.array([.2, .2, .2, 0., 0., 0.])
        _, _, terminated, _, info = simple.step(command)
        assert not terminated
        np.testing.assert_allclose(info["activation"], command)
        np.testing.assert_allclose(info["delayed_excitation"], command)
        np.testing.assert_allclose(info["tendon_strain"], 0)
        np.testing.assert_allclose(np.array(info["fiber_length_m"]) + info["tendon_length_m"],
                                   info["musculotendon_length_m"])
        assert info["muscle_torque_Nm"] == pytest.approx(
            np.array(info["moment_arm_m"]) @ info["tendon_force_N"])
        assert simple.model.nu == full.model.nu
    finally:
        simple.close()
        full.close()


def test_diagnostic_checkpoint_cannot_silently_run_with_full_physics(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from myo.policy_beta import BetaExcitationPolicy
    config = arm_config()
    env = make_environment("instant_rigid", config)
    model = PPO(BetaExcitationPolicy, env, n_steps=8, batch_size=4, seed=0,
                policy_kwargs={"net_arch": [16]}, device="cpu")
    path = tmp_path / "diagnostic.zip"
    model.save(path)
    details = descriptor("instant_rigid", config)
    root = Path(__file__).resolve().parents[1] / "src" / "myo"
    metadata = {"schema_version": 2, "actuator_ablation": details,
        "dynamics_sha256": experiment_digest(details), "config": asdict(env.config),
        "contract": contract(env, "instant_rigid"), "observation_names": env.observation_names,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trained_steps": 0,
        "features_sha256": hashlib.sha256((root / "policy_features.py").read_bytes()).hexdigest(),
        "beta_policy_sha256": hashlib.sha256((root / "policy_beta.py").read_bytes()).hexdigest()}
    Path(str(path) + ".metadata.json").write_text(json.dumps(metadata))
    env.close()
    with pytest.raises(ValueError, match="dynamics"):
        load_controller(str(path))
    loaded, act, _ = load_ablation_controller(path)
    try:
        obs, _ = loaded.reset(seed=0)
        action = act(obs)
        assert action.shape == (6,) and np.isfinite(action).all()
    finally:
        loaded.close()
