"""Protect shared-component likelihoods and transfer of the actual arm policy."""
from dataclasses import replace

import numpy as np
import torch
from torch.distributions import Beta

from myo.arm_efficiency import ArmEfficiencyEnv, EfficiencyReward
from myo.arm_relaxation import (ArmRelaxationEnv, RelaxationDistribution, RelaxationSpec,
    load_relaxation, make_relaxation_model, relaxed_mean)
from myo.arm_timing import make_fresh_model
from myo.arm_workload import WorkloadSpec, save_workload


def test_joint_component_preserves_torque_and_uses_joint_likelihood_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    alpha = torch.tensor([[3., 4., 5., 4., 3., 2.]], dtype=torch.float64, requires_grad=True)
    beta = torch.full_like(alpha, 9., requires_grad=True)
    gains = torch.tensor([[5., 6., 7., -4., -5., -6.]], dtype=torch.float64)
    spec = RelaxationSpec(fraction=.4)
    distribution = RelaxationDistribution(6).configure(alpha, beta, gains, spec)
    mean, relaxed = distribution.nominal_mean, distribution.relaxed_mean
    torch.testing.assert_close((gains*mean).sum(-1), (gains*relaxed).sum(-1))
    assert bool((relaxed < mean).all())
    action = relaxed.detach().clone()
    k = (alpha+beta)*spec.concentration_multiplier
    component_logp = torch.stack([Beta(m*k, (1-m)*k).log_prob(action).sum(-1)
                                  for m in (mean, relaxed)], -1)
    expected = torch.logsumexp(component_logp+np.log(.5), -1)
    torch.testing.assert_close(distribution.log_prob(action), expected)
    independent = RelaxationDistribution(6).configure(alpha, beta, gains,
        replace(spec, mode='independent'))
    marginal_logp = torch.stack([Beta(m*k, (1-m)*k).log_prob(action)
                                for m in (mean, relaxed)], -1)
    torch.testing.assert_close(independent.log_prob(action),
                              torch.logsumexp(marginal_logp+np.log(.5), -1).sum(-1))
    (-distribution.log_prob(action).mean()).backward()
    assert bool(torch.isfinite(alpha.grad).all()) and alpha.grad.abs().sum() > 0
    # Fresh score-function entropy has the right value shape and finite gradient.
    distribution = RelaxationDistribution(6).configure(alpha, beta, gains, spec)
    entropy = distribution.entropy()
    assert entropy.shape == (1,)
    assert bool(torch.isfinite(torch.autograd.grad(entropy.sum(), alpha)[0]).all())
    # Same marginal probes, only the coordination changes.
    joint = distribution.distribution.sample((30000,))[:, 0]
    separate = RelaxationDistribution(6).configure(alpha, beta, gains,
        replace(spec, mode='independent')).distribution.sample((30000,))[:, 0]
    torch.testing.assert_close(joint.mean(0), separate.mean(0), atol=.002, rtol=0)
    assert torch.cov(joint.T)[0, 3] > .0005
    assert abs(torch.cov(separate.T)[0, 3]) < .0001


def test_arm_transfer_native_physics_ppo_and_strict_distribution_roundtrip(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=((75., 75.),), seconds=.2)
    reward = EfficiencyReward(activation_squared_weight=.1)
    parent_env = ArmEfficiencyEnv(spec, reward=reward)
    env = ArmRelaxationEnv(spec, reward=reward)
    try:
        parent = make_fresh_model(parent_env, parent_env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        parent.learn(16)
        model = make_relaxation_model(env, parent, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        obs, _ = env.reset(seed=0)
        original, _ = parent_env.reset(seed=0)
        np.testing.assert_array_equal(obs[:-6], original)
        expected = parent.predict(original, deterministic=True)[0]
        np.testing.assert_array_equal(expected, model.predict(obs, deterministic=True)[0])
        for command in (np.ones(6), np.zeros(6), np.full(6, .3)):
            obs, r, failed, done, info = env.step(command)
            other, r0, failed0, done0, native = parent_env.step(command)
            np.testing.assert_array_equal(obs[:-6], other)
            assert (r, failed, done) == (r0, failed0, done0)
            for key in ('q', 'qvel', 'activation', 'delayed_excitation', 'tendon_force_N', 'capacity'):
                np.testing.assert_array_equal(info[key], native[key])
        before = model.policy.action_net.weight.detach().clone()
        model.learn(32, reset_num_timesteps=False)
        assert not torch.equal(before, model.policy.action_net.weight)
        path = tmp_path/'joint.zip'
        save_workload(model, env, path, {'test': True})
        restored, check, _ = load_relaxation(path)
        try:
            x = torch.as_tensor(obs[None])
            first = model.policy.get_distribution(x)
            action = first.sample().detach()
            logp = first.log_prob(action).detach().clone()
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          restored.predict(obs, deterministic=True)[0])
            torch.testing.assert_close(logp, restored.policy.get_distribution(x).log_prob(action))
            assert restored.optimizer_steps == model.optimizer_steps
            assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
            assert check.contract() == env.contract()
        finally:
            check.close()
    finally:
        env.close()
        parent_env.close()
