"""Regressions in selecting a usable command actor after supervised fitting."""
import numpy as np
import pytest

from myo.env import PreacherCurlConfig, PreacherCurlEnv


def test_refinement_retains_previous_round_when_next_fit_controls_worse(monkeypatch):
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    import myo.imitation as imitation

    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        model = sb3.PPO("MlpPolicy", env, seed=18, device="cpu", n_steps=8,
                        batch_size=8, policy_kwargs={"net_arch": [8]})
        observation, _ = env.reset(seed=18)
        observations = np.repeat(observation[None, :], 32, axis=0)
        labels = np.full((32, env.action_space.shape[0]), 0.4, dtype=np.float32)

        def collect(*args, **kwargs):
            return (observations, labels, np.zeros(32, dtype=np.int64)), 1, {}

        candidates = []

        def validate(candidate, *args, **kwargs):
            candidates.append({key: value.detach().clone()
                               for key, value in candidate.policy.state_dict().items()})
            # The second real optimizer fit changes weights but performs worse
            # on the independent control scenarios, as happened in D1 training.
            return (2.0, 1.0, 3.0)[len(candidates) - 1], []

        monkeypatch.setattr(imitation, "_collect", collect)
        monkeypatch.setattr(imitation, "_closed_loop_validation", validate)
        report = imitation.train_command_actor(model, env, 2000, 2, 18,
                                                epochs=1, resume=True)
        assert len(candidates) == 3  # Parent, first fit, second fit.
        assert any(not torch.equal(candidates[1][key], value)
                   for key, value in candidates[2].items())
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, candidates[1][key])
        assert report["parent_closed_loop_validation"]["score"] == 2.0
        final = report["rounds"][-1]
        assert final["selected_from_round"] == 1
        assert final["selected_closed_loop_score"] == 1.0
        assert final["retained_previous_round"]
    finally:
        env.close()
