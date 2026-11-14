"""Regression checks for bounded, genuinely correlated PPO exploration."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal

from myo.arm_correlated_relaxation import (ArmCorrelatedEnv, CorrelatedDistribution,
    CorrelatedSpec, load_correlated, make_correlated_model)
from myo.arm_efficiency import ArmEfficiencyEnv, EfficiencyReward
from myo.arm_timing import make_fresh_model
from myo.arm_workload import WorkloadSpec, save_workload


def test_full_joint_likelihood_preserves_marginals_and_suppresses_torque_noise():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    alpha = torch.tensor([[3., 4., 5., 4., 3., 2.]], dtype=torch.float64, requires_grad=True)
    beta = torch.full_like(alpha, 9., requires_grad=True)
    gains = torch.tensor([[5., 6., 7., -4., -5., -6.]], dtype=torch.float64)
    spec = CorrelatedSpec(common_std=.3, residual_std=.003)
    d = CorrelatedDistribution(6).configure(alpha, beta, gains, spec)
    independent = CorrelatedDistribution(6).configure(alpha, beta, gains, replace(spec, mode='diagonal'))
    torch.testing.assert_close(d.locations, independent.locations)
    torch.testing.assert_close(d.covariance.diagonal(dim1=-2, dim2=-1),
                               independent.covariance.diagonal(dim1=-2, dim2=-1))
    actions = d.sample().detach()
    latent = torch.logit(actions)
    component = MultivariateNormal(d.locations, covariance_matrix=d.covariance).log_prob(latent.unsqueeze(-2))
    jacobian = (torch.log(actions)+torch.log1p(-actions)).sum(-1)
    expected = torch.logsumexp(component+np.log(.5), -1)-jacobian
    torch.testing.assert_close(d.log_prob(actions), expected)
    h = gains.unsqueeze(-2)*d.locations.sigmoid()*(1-d.locations.sigmoid())
    variance = torch.einsum('bki,bkij,bkj->bk', h, d.covariance, h)
    torch.testing.assert_close(variance, spec.residual_std**2*h.square().sum(-1), atol=1e-12, rtol=1e-8)
    assert bool((torch.linalg.eigvalsh(d.covariance) > 0).all())
    loss = -d.log_prob(actions).mean()-.001*d.entropy().mean()
    grad = torch.autograd.grad(loss, alpha)[0]
    assert bool(torch.isfinite(grad).all()) and grad.abs().sum() > 0
    joint = d.distribution.sample((15000,))[:, 0]
    separate = independent.distribution.sample((15000,))[:, 0]
    torch.testing.assert_close(joint.mean(0), separate.mean(0), atol=.002, rtol=0)
    assert (joint@gains[0]).std() < .2*(separate@gains[0]).std()


def test_transferred_nominal_physics_ppo_update_and_strict_roundtrip(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=((75., 75.),), seconds=.2)
    reward = EfficiencyReward(activation_squared_weight=.1)
    native = ArmEfficiencyEnv(spec, reward=reward)
    env = ArmCorrelatedEnv(spec, reward=reward)
    try:
        parent = make_fresh_model(native, native.contract(), n_steps=16, batch_size=16, n_epochs=1)
        parent.learn(16)
        model = make_correlated_model(env, parent, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        obs, _ = env.reset(seed=0)
        original, _ = native.reset(seed=0)
        np.testing.assert_array_equal(obs[:-6], original)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], parent.predict(original, deterministic=True)[0])
        for action in (np.ones(6), np.zeros(6), np.full(6, .3)):
            obs, r, f, t, info = env.step(action)
            old, r0, f0, t0, info0 = native.step(action)
            np.testing.assert_array_equal(obs[:-6], old)
            assert (r, f, t) == (r0, f0, t0)
            for key in ('q', 'qvel', 'activation', 'delayed_excitation', 'tendon_force_N', 'capacity'):
                np.testing.assert_array_equal(info[key], info0[key])
        before = model.policy.action_net.weight.detach().clone()
        model.learn(32, reset_num_timesteps=False)
        assert not torch.equal(before, model.policy.action_net.weight)
        path = tmp_path/'correlated.zip'
        save_workload(model, env, path, {'test': True})
        restored, check, _ = load_correlated(path)
        try:
            x = torch.as_tensor(obs[None])
            d = model.policy.get_distribution(x)
            action = d.sample().detach()
            logp = d.log_prob(action).detach().clone()
            torch.testing.assert_close(logp, restored.policy.get_distribution(x).log_prob(action))
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
            assert restored.optimizer_steps == model.optimizer_steps
            assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
        finally:
            check.close()
        import json
        sidecar = path.with_name(path.name+'.metadata.json')
        metadata = json.loads(sidecar.read_text())
        metadata['contract']['correlated']['residual_std'] *= 2
        sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match='identity mismatch'):
            load_correlated(path)
    finally:
        env.close()
        native.close()
