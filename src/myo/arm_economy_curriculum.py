"""Acceptance of accumulated effort reduction against a fixed accuracy reference."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .arm_efficiency import EfficiencyReward, quality_gate
from .arm_exploration import ExplorationSpec
from .arm_learning import ArmPPO
from .arm_timing import ArmTiming
from .arm_weakening import ArmWeakeningEnv, WeakeningPolicy, WeakeningSpec
from .arm_workload import spec_from_dict


class ArmEconomyEnv(ArmWeakeningEnv):
    """Same native mechanics, explicit task-and-effort reward contract."""
    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_accumulated_economy.v1',
            economy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weakening_rule='single independent Beta with configured concentration and original stagnation; '
                'nominal deterministic output; self-labelled moment-aware auxiliary weakening; '
                'PPO task plus actual activation-squared cost from efficiency_reward; '
                'no post-sampling or inference command modification; causal gains excluded from learned features')
        return contract


def load_economy(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    c = meta['contract']
    if c['schema'] != 'myo.arm_accumulated_economy.v1':
        raise ValueError('Not an accumulated economy checkpoint')
    env = ArmEconomyEnv(spec_from_dict(c['spec']), ExplorationSpec(**c['exploration']),
        ArmTiming(**c['timing']), EfficiencyReward(**c['efficiency_reward']), WeakeningSpec(**c['weakening']))
    try:
        if env.contract() != c or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Economy physics/input/reward/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != c or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, WeakeningPolicy)
                or asdict(model.policy.weakening) != c['weakening']
                or asdict(model.policy.exploration) != c['exploration']
                or model.observation_space != env.observation_space
                or model.action_space != env.action_space or model.policy.feature_mode != 'none'):
            raise ValueError('Economy embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta


def economy_gate(result, baseline, previous, minimum_reduction=.001):
    """Keep original per-case precision and reduce full AND settled activation cost.

    Accuracy must never be compared only with the latest accepted parent: that
    would permit the .25-degree tolerance to accumulate on every cycle.
    """
    eligible, reasons = quality_gate(result, baseline)
    reasons = list(reasons)
    for metric in ('activation_squared_mean', 'tail_activation_squared_mean'):
        if result[metric] > (1.-minimum_reduction)*previous[metric]:
            reasons.append([metric, 'insufficient_reduction'])
    return eligible and not reasons, reasons
