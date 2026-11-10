"""Reward-only arm experiments, without changing pinned workload checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .arm_learning import ArmPPO
from .arm_workload import ArmWorkloadEnv, spec_from_dict


@dataclass(frozen=True)
class RewardStage:
    mode: str = 'position_only'

    def __post_init__(self):
        if self.mode not in ('native', 'position_only', 'position_efficiency'):
            raise ValueError('Unknown reward stage')


class ArmRewardCurriculumEnv(ArmWorkloadEnv):
    def __init__(self, spec=None, reward_stage=None, *, worker=0, workers=1):
        self.reward_stage = reward_stage or RewardStage()
        super().__init__(spec, worker=worker, workers=workers)

    def step(self, action):
        observation, native_reward, terminated, truncated, info = super().step(action)
        original = dict(info['reward_components'])
        info['native_reward_components'] = original
        if self.reward_stage.mode == 'native':
            return observation, native_reward, terminated, truncated, info
        selected = {'position_log': original['position_log']}
        if 'physical_failure' in original:
            selected['physical_failure'] = original['physical_failure']
        if self.reward_stage.mode == 'position_efficiency':
            for key in ('activation_squared_proxy_cost', 'active_fraction_fatigue_proxy_cost'):
                selected[key] = original[key]
        info['reward_components'] = selected
        return observation, .1*sum(selected.values()), terminated, truncated, info

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_reward_curriculum.v1', reward_stage=asdict(self.reward_stage),
            reward_stage_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reward_stage_rule='native unchanged; position_only keeps position and physical failure; '
                              'position_efficiency additionally keeps explicit workload activation costs')
        return contract


def load_reward_curriculum(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    if meta['contract']['schema'] != 'myo.arm_reward_curriculum.v1':
        raise ValueError('Not an arm reward curriculum checkpoint')
    env = ArmRewardCurriculumEnv(spec_from_dict(meta['contract']['spec']),
                                RewardStage(**meta['contract']['reward_stage']))
    try:
        if env.contract() != meta['contract']:
            raise ValueError('Arm reward/physics/input identity mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Arm reward checkpoint hash mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != meta['contract'] or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or model.policy.feature_mode != 'none'
                or model.observation_space.shape != (155,)):
            raise ValueError('Arm reward checkpoint embedded identity mismatch')
    except Exception:
        env.close()
        raise
    return model, env, meta
