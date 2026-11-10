"""Protect causal context, true action likelihoods and strict continuation."""
import json
import gzip
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from myo.arm_exploration import (ArmExplorationEnv, ExplorationSpec, StagnationDetector,
                                load_exploration, make_exploration_model)
from myo.arm_learning import make_model
from myo.arm_precision import ArmPrecisionEnv
from myo.arm_reward_curriculum import ArmRewardCurriculumEnv
from myo.arm_workload import make_workload_model, save_workload
from myo.arm_training_audit import ConditionMonitor, EpisodeAudit
from stable_baselines3.common.vec_env import DummyVecEnv


def test_plateau_history_retains_accumulated_reference_across_phase_caps():
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/train_arm_exploration.py'))
    history = [{'added_transitions': i*16384,
                'validation': {'failures': 0, 'passed': 1, 'score_deg': score}}
               for i, score in enumerate((10.,9.,9.,9.01,9.,8.99,8.98))]
    reference, blocks = runner['reconstruct_stopping_state'](history)
    assert reference['score_deg'] == 9. and blocks == 2
    # A tiny new best does not erase the older accumulated-progress reference.
    assert reference['score_deg'] > history[-1]['validation']['score_deg']
    validation = {'failures': 0, 'passed': 25, 'total': 27, 'score_deg': 1.2, 'groups': {}}
    training = {'failures': 0, 'passed': 16, 'total': 18}
    assert not runner['full_stage_passed'](validation, training)
    # Cheap relaxation outside tolerance must not win the effort curriculum.
    initial = {'failures': 0, 'passed': 18, 'total': 18, 'score_deg': .8,
               'activation_squared_mean': .12, 'groups': {}}
    rank = runner['effort_rank']
    parent = rank(initial, initial, initial, initial)
    cheap_failure = {**initial, 'passed': 17, 'activation_squared_mean': .01}
    assert rank(cheap_failure, initial, initial, initial) > parent
    inaccurate = {**initial, 'score_deg': 1.1, 'activation_squared_mean': .01}
    assert rank(inaccurate, initial, initial, initial) > parent
    precise_saving = {**initial, 'score_deg': .9, 'activation_squared_mean': .10}
    assert rank(precise_saving, precise_saving, initial, initial) < parent
    training['passed'] = 18
    assert runner['full_stage_passed'](validation, training)
    validation['passed'] = 23
    assert not runner['full_stage_passed'](validation, training)


def test_episode_audit_preserves_training_and_cross_rollout_alignment(tmp_path):
    torch.set_num_threads(1)
    old, native = ArmPrecisionEnv(), ArmRewardCurriculumEnv()
    vectors = [DummyVecEnv([
        lambda worker=i: ConditionMonitor(ArmExplorationEnv(worker=worker, workers=2))
        for i in range(2)]) for _ in range(2)]
    audit = EpisodeAudit(tmp_path)
    try:
        seed = make_model(old, n_steps=32, batch_size=16, n_epochs=1)
        parent = make_workload_model(native, seed, native.contract(),
                                     n_steps=32, batch_size=16, n_epochs=1)
        contract = vectors[0].envs[0].env.contract()
        models = [make_exploration_model(vector, parent, contract,
                  n_steps=128, batch_size=64, n_epochs=1) for vector in vectors]
        for model, callback in zip(models, (None, audit)):
            model.set_random_seed(0)
            for _ in range(2):
                model.learn(256, reset_num_timesteps=False, callback=callback)
        assert all(torch.equal(v, models[1].policy.state_dict()[k])
                   for k, v in models[0].policy.state_dict().items())
    finally:
        audit.close()
        for vector in vectors:
            vector.close()
        old.close(); native.close()
    with gzip.open(tmp_path/'training_steps.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    episodes = [json.loads(line) for line in (tmp_path/'training_episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 512 and len(episodes) == 2
    assert {(r['worker'], r['episode']) for r in episodes} == {(0,0),(1,0)}
    for result in episodes:
        trace = [r for r in rows if r['worker'] == result['worker'] and r['episode'] == 0]
        assert len(trace) == 200 and trace[-1]['done'] and result['complete']
        assert [r['t'] for r in trace] == pytest.approx(np.arange(1,201)*.02)
        assert result['condition'] == trace[0]['condition'] == trace[-1]['condition']
        assert {r['update_end_transition'] for r in trace} == {256,512}
        assert all(np.isfinite(r['logp_after']) and np.isfinite(r['raw_advantage']) for r in trace)
        assert result['action_logp_change_mean'] == pytest.approx(
            np.mean([r['logp_after']-r['logp_before'] for r in trace]))
    # Audit rows remain time-major even though PPO changes buffer storage order.
    buffer = models[1].rollout_buffer
    for row in rows[256:]:
        index = row['worker']*128+(row['transition']-258)//2
        assert row['raw_advantage'] == float(buffer.advantages.reshape(-1)[index])


def test_stagnation_context_is_causal_and_does_not_change_reward_or_physics():
    detector = StagnationDetector(ExplorationSpec(), .02)
    for _ in range(35):
        detector.update(np.deg2rad(140), np.deg2rad(125))
    assert detector.pressure == pytest.approx(1.)
    for q in range(138, 126, -2):
        detector.update(np.deg2rad(q), np.deg2rad(125))
    assert detector.pressure == pytest.approx(0.)
    detector.update(np.deg2rad(140), np.deg2rad(100))
    assert detector.pressure == 0. and len(detector.history) == 1
    native, added = ArmRewardCurriculumEnv(), ArmExplorationEnv()
    try:
        for env in (native, added):
            env.reset(seed=0, options={'case': (125,125), 'tilt_deg': 135., 'payload_mass': 2.})
        pressures = []
        for _ in range(45):
            action = np.array([.001,.001,.001,.8,.4,.7], dtype=np.float32)
            a, b = native.step(action), added.step(action)
            np.testing.assert_array_equal(a[0], b[0][:-1])
            assert a[1:4] == b[1:4]
            np.testing.assert_array_equal(native.unwrapped.data.qpos, added.unwrapped.data.qpos)
            for x, y in zip(native.unwrapped.muscles, added.unwrapped.muscles):
                np.testing.assert_array_equal(x.state_vector(), y.state_vector())
            pressures.append(float(b[0][-1]))
        assert max(pressures) > .5
        obs, _ = added.reset(seed=0)
        assert obs[-1] == 0. and len(added.detector.history) == 1
    finally:
        native.close(); added.close()


def test_broadened_sampling_uses_same_likelihood_in_ppo_and_roundtrips(tmp_path):
    torch.set_num_threads(1)
    old, native, env = ArmPrecisionEnv(), ArmRewardCurriculumEnv(), ArmExplorationEnv()
    try:
        seed_model = make_model(old, n_steps=32, batch_size=16, n_epochs=1)
        parent = make_workload_model(native, seed_model, native.contract(),
                                    n_steps=32, batch_size=16, n_epochs=1)
        parent.learn(32)
        model = make_exploration_model(env, parent, env.contract(),
                                      n_steps=32, batch_size=16, n_epochs=1)
        obs, _ = env.reset(seed=0, options={'case': (125,125), 'tilt_deg': 135.})
        tensor = torch.as_tensor(obs[None].repeat(8, axis=0))
        # Zero new columns preserve both old functions and optimizer moments.
        before = parent.policy.get_distribution(tensor[:, :-1]).distribution.mean.detach().clone()
        original = model.policy.get_distribution(tensor).distribution
        torch.testing.assert_close(original.mean, before)
        torch.testing.assert_close(model.policy.predict_values(tensor),
                                   parent.policy.predict_values(tensor[:, :-1]))
        current_parameters = dict(model.policy.named_parameters())
        for name, source in parent.policy.named_parameters():
            current = current_parameters[name]
            if source.requires_grad and source in parent.policy.optimizer.state:
                for field, state in parent.policy.optimizer.state[source].items():
                    actual = model.policy.optimizer.state[current][field]
                    if state.shape == source.shape and source.shape != current.shape:
                        assert torch.equal(actual[:, :-1], state)
                        assert torch.count_nonzero(actual[:, -1]) == 0
                    else:
                        assert torch.equal(actual, state)
        # Both rollout forward and minibatch evaluate_actions must see pressure.
        tensor[:, -1] = 1.
        broad = model.policy.get_distribution(tensor).distribution
        torch.testing.assert_close(broad.mean, before)
        assert torch.all(broad.variance > original.variance)
        assert torch.all(broad.concentration0 >= .05-1e-7)
        actions, _, old_logp = model.policy(tensor)
        _, new_logp, entropy = model.policy.evaluate_actions(tensor, actions)
        torch.testing.assert_close(old_logp, new_logp)
        assert torch.isfinite(new_logp).all() and torch.isfinite(entropy).all()
        (-new_logp.mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.policy.parameters())
        model.learn(64, reset_num_timesteps=False)
        path = tmp_path/'exploration.zip'
        save_workload(model, env, path, {'mode': 'stall'})
        restored, checked, _ = load_exploration(path)
        checked.close()
        assert restored.optimizer_steps == model.optimizer_steps
        assert all(torch.equal(v, restored.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        resumed = make_exploration_model(env, restored, env.contract(),
                                        n_steps=32, batch_size=16, n_epochs=1)
        assert resumed.optimizer_steps == restored.optimizer_steps
        assert all(torch.equal(v, resumed.policy.state_dict()[k]) for k,v in restored.policy.state_dict().items())
        for current, loaded in zip(model.policy.parameters(), restored.policy.parameters()):
            for field, state in model.policy.optimizer.state.get(current, {}).items():
                actual = restored.policy.optimizer.state[loaded][field]
                if isinstance(state, torch.Tensor):
                    assert torch.equal(state, actual)
                else:
                    assert state == actual
        meta_path = Path(str(path)+'.metadata.json')
        meta = json.loads(meta_path.read_text())
        meta['contract']['exploration']['mode'] = 'off'
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match='embedded identity'):
            load_exploration(path)
    finally:
        env.close(); native.close(); old.close()
