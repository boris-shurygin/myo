"""Protect continuation from evaluation RNG leakage and goal-specialist stopping."""
import numpy as np
import torch

from myo.arm_learning import make_model
from myo.arm_precision import ArmPrecisionEnv
from myo.train_arm_precision import private_hash
from scripts.continue_arm_precision import stochastic_probe, significant_progress


def test_stochastic_validation_preserves_training_stream_and_physical_episode():
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    model = make_model(env, n_steps=32, batch_size=16, n_epochs=1)
    model.learn(32)
    rng = torch.get_rng_state().clone()
    observation = env.observation().copy()
    time = env.unwrapped.time
    weights = private_hash(model.policy)
    steps = model.optimizer_steps
    rows = stochastic_probe(model, cases=((20, 75),), seeds=(0,))
    assert rows[0]['duration_s'] > 3.99
    assert torch.equal(torch.get_rng_state(), rng)
    assert np.array_equal(env.observation(), observation) and env.unwrapped.time == time
    assert private_hash(model.policy) == weights and model.optimizer_steps == steps
    env.close()


def test_more_passes_from_goal_specialization_do_not_reset_progress():
    reference = {'failures': 0, 'passed': 2, 'balanced_rmse_deg': 7.709}
    # The earlier 75-degree specialist: more passes, worse equal-goal error.
    specialist = {'failures': 0, 'passed': 4, 'balanced_rmse_deg': 8.261}
    assert not significant_progress(reference, specialist)
    # Small improvements can accumulate against the last significant reference.
    assert not significant_progress(reference, {**reference, 'balanced_rmse_deg': 7.66})
    assert significant_progress(reference, {**reference, 'balanced_rmse_deg': 7.61})
