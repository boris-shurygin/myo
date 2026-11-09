"""Frozen body features and native PPO for the six-muscle preacher-curl task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from stable_baselines3 import PPO

from .policy_beta import BetaExcitationPolicy
from .shared_vertical import mlp
from .arm_precision import ArmPrecisionEnv

TRIALS = ('none', 'frozen')


class ArmExtractor(nn.Module):
    def __init__(self, width, indices, mode):
        super().__init__()
        self.indices, self.mode = indices, mode
        self.encoder = mlp(len(indices), (128,)*4)
        self.actor_private = mlp(width, (128, 128))
        self.critic_private = mlp(width, (128, 128))
        self.policy_net = mlp(256, (128, 128))
        self.value_net = mlp(256, (128, 128))
        self.latent_dim_pi = self.latent_dim_vf = 128

    def features(self, observations):
        if self.mode == 'none':
            return observations.new_zeros((len(observations), 128))
        return self.encoder(observations[:, self.indices]).detach()

    def forward_actor(self, observations):
        return self.policy_net(torch.cat((self.features(observations), self.actor_private(observations)), -1))

    def forward_critic(self, observations):
        return self.value_net(torch.cat((self.features(observations), self.critic_private(observations)), -1))

    def forward(self, observations):
        return self.forward_actor(observations), self.forward_critic(observations)


class ArmPolicy(BetaExcitationPolicy):
    def __init__(self, *args, predictor_indices, physical_indices, feature_mode, **kwargs):
        self.predictor_indices, self.physical_indices = predictor_indices, physical_indices
        self.feature_mode = feature_mode
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ArmExtractor(self.features_dim, self.predictor_indices, self.feature_mode).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.prediction_head = nn.Linear(128, len(self.physical_indices)).to(self.device)
        nn.init.zeros_(self.prediction_head.weight); nn.init.zeros_(self.prediction_head.bias)
        self.register_buffer('delta_mean', torch.zeros(len(self.physical_indices)))
        self.register_buffer('delta_std', torch.ones(len(self.physical_indices)))
        self.register_buffer('delta_calibrated', torch.tensor(False))
        for p in self.mlp_extractor.encoder.parameters():
            p.requires_grad_(False)
        for p in self.prediction_head.parameters():
            p.requires_grad_(False)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)


class ArmPPO(PPO):
    def __init__(self, *args, **kwargs):
        self.optimizer_steps, self.last_audit = 0, {}
        super().__init__(*args, **kwargs)

    def train(self):
        before = self.optimizer_steps
        def count(*args):
            self.optimizer_steps += 1
        hook = self.policy.optimizer.register_step_post_hook(count)
        try:
            super().train()
        finally:
            hook.remove()
        self.last_audit = {'optimizer_steps': self.optimizer_steps-before,
            **{key: float(self.logger.name_to_value['train/'+key])
               for key in ('approx_kl', 'explained_variance', 'value_loss', 'policy_gradient_loss', 'entropy_loss')}}


def make_model(env, mode='none', *, n_steps=1024, batch_size=128, n_epochs=10):
    if mode not in TRIALS:
        raise ValueError('Unknown arm trial')
    # Monitor delegates custom contract fields through get_wrapper_attr.
    indices = env.get_wrapper_attr('predictor_indices')
    physical = env.get_wrapper_attr('physical_indices')
    return ArmPPO(ArmPolicy, env, seed=0, device='cpu', learning_rate=3e-4,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': indices, 'physical_indices': physical, 'feature_mode': mode,
            'activation_fn': nn.SiLU, 'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})


class ArmBodyPredictor(nn.Module):
    def __init__(self, input_size=27, output_size=20):
        super().__init__()
        self.encoder = mlp(input_size, (128,)*4)
        self.head = nn.Linear(128, output_size)
        self.register_buffer('delta_mean', torch.zeros(output_size))
        self.register_buffer('delta_std', torch.ones(output_size))
        self.register_buffer('delta_calibrated', torch.tensor(False))

    def normalized_delta(self, inputs):
        return self.head(self.encoder(inputs))

    def next_physical(self, inputs):
        return inputs[:, :20]+self.delta_mean+self.delta_std*self.normalized_delta(inputs)

    @torch.no_grad()
    def copy_into_policy(self, policy):
        policy.mlp_extractor.encoder.load_state_dict(self.encoder.state_dict())
        policy.prediction_head.load_state_dict(self.head.state_dict())
        for name in ('delta_mean', 'delta_std', 'delta_calibrated'):
            getattr(policy, name).copy_(getattr(self, name))


def predictor_identity(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = ArmPrecisionEnv()
    if meta['contract'] != env.contract():
        raise ValueError('Arm predictor contract mismatch')
    env.close()
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('Arm predictor hash mismatch')
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if payload['update'] != meta['update']:
        raise ValueError('Arm predictor update mismatch')
    model = ArmBodyPredictor()
    model.load_state_dict(payload['state_dict'])
    return model.eval(), meta


def save_arm(model, env, path, training):
    path = Path(path)
    model.arm_contract = env.contract()
    model.arm_training = json.loads(json.dumps(training, allow_nan=False))
    model.save(path)
    meta = {'contract': model.arm_contract, 'training': model.arm_training, 'steps': model.num_timesteps,
        'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')


def load_arm(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = ArmPrecisionEnv()
    if meta['contract'] != env.contract():
        raise ValueError('Arm policy contract mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('Arm policy checkpoint hash mismatch')
    model = ArmPPO.load(path, device='cpu')
    if model.arm_contract != meta['contract'] or model.arm_training != meta['training']:
        raise ValueError('Arm policy embedded identity mismatch')
    if model.num_timesteps != meta['steps'] or model.policy.feature_mode != meta['training']['trial']:
        raise ValueError('Arm policy step/mode mismatch')
    if meta['training']['trial'] == 'frozen' and not bool(model.policy.delta_calibrated):
        raise ValueError('Arm policy missing predictor normalization')
    return model, env, meta
