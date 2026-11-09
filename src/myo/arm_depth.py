"""Four post-concatenation layers with the original frozen body encoder."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .arm_conditioning import ConditionedArmPolicy, load_conditioned_arm
from .arm_learning import ArmPPO
from .shared_vertical import mlp


class DeepArmPolicy(ConditionedArmPolicy):
    def __init__(self, *args, post_depth=4, **kwargs):
        if post_depth != 4:
            raise ValueError('Deep arm experiment requires four post-concatenation layers')
        self.post_depth = post_depth
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Build the complete original policy first: common weights and RNG match.
        # Added SiLU layers change the initial function; this is not a residual net.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for tail in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
                extra = mlp(128, (128, 128)).to(self.device)
                for layer in extra:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                        nn.init.zeros_(layer.bias)
                for layer in extra:
                    tail.add_module(str(len(tail)), layer)
        # The original optimizer was created before the extra parameters existed.
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)


def make_deep_model(env, *, n_steps=1024, batch_size=128, n_epochs=10):
    return ArmPPO(DeepArmPolicy, env, seed=0, device='cpu', learning_rate=3e-4,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': env.get_wrapper_attr('predictor_indices'),
            'physical_indices': env.get_wrapper_attr('physical_indices'), 'feature_mode': 'frozen',
            'feature_gain': .25, 'post_depth': 4, 'activation_fn': nn.SiLU,
            'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})


def depth_identity():
    return {'schema': 'myo.arm_depth.v1', 'actor_post_layers': [128]*4,
            'critic_post_layers': [128]*4, 'private_layers': [128]*2,
            'activation': 'SiLU', 'additional_layer_seed': 0,
            'initialization': 'original common weights; extra orthogonal sqrt(2), zero biases; preserve Torch RNG',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_deep_arm(path):
    model, env, meta = load_conditioned_arm(path)
    try:
        if (not isinstance(model.policy, DeepArmPolicy)
                or meta['training']['depth'] != depth_identity()
                or model.policy.post_depth != 4 or model.policy_kwargs['post_depth'] != 4
                or len(model.policy.mlp_extractor.policy_net) != 8
                or len(model.policy.mlp_extractor.value_net) != 8):
            raise ValueError('Arm post-concatenation depth identity mismatch')
    except Exception:
        env.close()
        raise
    return model, env, meta
