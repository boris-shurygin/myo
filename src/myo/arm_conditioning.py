"""Fixed body-feature scaling for an isolated arm PPO conditioning experiment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .arm_learning import ArmExtractor, ArmPolicy, ArmPPO, load_arm


class ConditionedArmExtractor(ArmExtractor):
    def __init__(self, width, indices, mode, feature_gain):
        super().__init__(width, indices, mode)
        self.register_buffer('feature_gain', torch.tensor(float(feature_gain)))

    def features(self, observations):
        return super().features(observations)*self.feature_gain


class ConditionedArmPolicy(ArmPolicy):
    def __init__(self, *args, feature_gain, **kwargs):
        self.feature_gain_spec = float(feature_gain)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ConditionedArmExtractor(
            self.features_dim, self.predictor_indices, self.feature_mode,
            self.feature_gain_spec).to(self.device)


def make_conditioned_model(env, learning_rate, feature_gain, *, n_steps=1024, batch_size=128, n_epochs=10):
    return ArmPPO(ConditionedArmPolicy, env, seed=0, device='cpu', learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': env.get_wrapper_attr('predictor_indices'),
            'physical_indices': env.get_wrapper_attr('physical_indices'), 'feature_mode': 'frozen',
            'feature_gain': feature_gain, 'activation_fn': nn.SiLU,
            'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})


def conditioning_identity(learning_rate, feature_gain):
    return {'schema': 'myo.arm_conditioning.v1', 'learning_rate': float(learning_rate),
            'feature_gain': float(feature_gain), 'scope': 'actor and critic feature branches only; predictor unchanged',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_conditioned_arm(path):
    model, env, metadata = load_arm(path)
    try:
        spec = metadata['training']['conditioning']
        if (not isinstance(model.policy, ConditionedArmPolicy)
                or spec != conditioning_identity(model.learning_rate, model.policy.feature_gain_spec)
                or model.policy_kwargs['feature_gain'] != spec['feature_gain']
                or float(model.policy.mlp_extractor.feature_gain) != spec['feature_gain']):
            raise ValueError('Arm feature conditioning identity mismatch')
    except Exception:
        env.close()
        raise
    return model, env, metadata
