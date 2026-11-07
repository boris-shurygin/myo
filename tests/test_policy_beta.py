"""Regressions for lost exploration and disagreement with physical actions."""
import numpy as np
import pytest


def test_beta_physical_mean_and_sampling_stay_inside_excitation_bounds():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from myo.policy_beta import BetaExcitationDistribution

    torch.manual_seed(0)
    dist = BetaExcitationDistribution(2)
    # Include a tail-heavy near-relaxation state; Gaussian means used to escape
    # the physical interval here, making useful commands practically absent.
    parameters = torch.tensor([[-4.0, 0.5, 9.0, 9.0]], requires_grad=True)
    dist.proba_distribution(parameters.expand(10000, -1))
    samples = dist.sample()
    assert bool(((samples > 0) & (samples < 1)).all())
    assert float((samples[:, 0] > 0.05).float().mean()) > 0.02
    torch.testing.assert_close(samples.mean(0), dist.mode()[0], atol=0.004, rtol=0)
    assert bool(torch.isfinite(dist.log_prob(samples)).all())
    (-dist.log_prob(samples).mean()).backward()
    assert bool(torch.isfinite(parameters.grad).all())


def test_beta_ppo_roundtrip_preserves_action_probabilities_and_updates(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from gymnasium import spaces
    import gymnasium as gym
    from myo.policy_beta import BetaExcitationPolicy

    class ExcitationBandit(gym.Env):
        observation_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        action_space = spaces.Box(0, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            assert np.all((action >= 0) & (action <= 1))
            return np.zeros(3, dtype=np.float32), -float(np.square(action - 0.4).sum()), False, True, {}

    torch.set_num_threads(1)
    model = sb3.PPO(BetaExcitationPolicy, ExcitationBandit(), n_steps=32,
                    batch_size=32, n_epochs=2, seed=0, device="cpu")
    before = model.policy.action_net.weight.detach().clone()
    model.learn(64)
    assert not torch.equal(before, model.policy.action_net.weight)
    obs = torch.zeros((1, 3))
    dist = model.policy.get_distribution(obs)
    action = dist.get_actions(deterministic=True).detach().clone()
    likelihood = dist.log_prob(action).detach().clone()
    model.save(tmp_path / "beta.zip")
    restored = sb3.PPO.load(tmp_path / "beta.zip", device="cpu")
    dist = restored.policy.get_distribution(obs)
    torch.testing.assert_close(dist.get_actions(deterministic=True), action)
    torch.testing.assert_close(dist.log_prob(action), likelihood)


def test_persistent_beta_uses_observed_previous_commands_for_collection_and_replay(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from gymnasium import spaces
    import gymnasium as gym
    from myo.policy_features import NormalizedFeaturesExtractor
    from myo.policy_persistent_beta import PersistentBetaPolicy

    class HistoryBandit(gym.Env):
        observation_space = spaces.Box(0, 1, (3,), dtype=np.float32)
        action_space = spaces.Box(0, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.array([0.2, 0.8, 0], dtype=np.float32), {}

        def step(self, action):
            return np.array([*action, 0], dtype=np.float32), -float(np.square(action - 0.4).sum()), False, False, {}

    torch.set_num_threads(1)
    model = sb3.PPO(PersistentBetaPolicy, HistoryBandit(), n_steps=32, batch_size=32,
        n_epochs=2, seed=0, device="cpu", policy_kwargs={"persistence": 0.9,
        "previous_action_indices": [0, 1], "features_extractor_class": NormalizedFeaturesExtractor})
    model.policy.features_extractor.set_normalization([0.3, 0.4, 0], [0.2, 0.5, 1])
    model.policy.synchronize_previous_action_normalization()
    obs = torch.tensor([[0.2, 0.8, 0], [0.7, 0.1, 0]])
    with torch.no_grad():
        action, _, collected_log_prob = model.policy(obs)
        replayed_log_prob = model.policy.evaluate_actions(obs.flip(0), action.flip(0))[1]
        torch.testing.assert_close(replayed_log_prob, collected_log_prob.flip(0))
        mean = model.policy.get_distribution(obs).distribution.mean
        torch.testing.assert_close(mean, 0.9 * obs[:, :2] + 0.01, atol=0.001, rtol=0)
    model.learn(64)
    model.save(tmp_path / "persistent.zip")
    restored = sb3.PPO.load(tmp_path / "persistent.zip", device="cpu")
    torch.testing.assert_close(restored.policy.get_distribution(obs).distribution.mean,
                               model.policy.get_distribution(obs).distribution.mean)
