"""Common-parent PPO with absent, one-step or direct-60-ms frozen features."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from .arm_anticipation import AnticipationPredictor, identity as anticipation_identity, load_predictor
from .arm_learning import ArmExtractor, ArmPolicy, ArmPPO, load_arm, predictor_identity
from .shared_vertical import mlp


KINDS = ('none', 'old20', 'new60')


class ComparisonExtractor(ArmExtractor):
    def __init__(self, width, indices, mode, gain):
        super().__init__(width, indices, mode)
        self.register_buffer('feature_gain', torch.tensor(float(gain)))

    def features(self, observations):
        return super().features(observations)*self.feature_gain


class ComparisonPolicy(ArmPolicy):
    def __init__(self, *args, feature_kind, anticipation_indices, feature_gain=.25, **kwargs):
        if feature_kind not in KINDS:
            raise ValueError('Unknown comparison feature kind')
        self.feature_kind = feature_kind
        self.anticipation_indices = list(anticipation_indices)
        self.feature_gain_spec = float(feature_gain)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ComparisonExtractor(self.features_dim, self.predictor_indices,
            self.feature_mode, self.feature_gain_spec).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Construct common private/tail layers before the different-size encoder.
        # Predictors are subsequently loaded; their constructor must not move PPO RNG.
        if self.feature_kind == 'new60':
            with torch.random.fork_rng(devices=[]):
                self.mlp_extractor.encoder = mlp(33, (128,)*4).to(self.device)
                self.prediction_head = nn.Sequential(nn.Linear(134, 128), nn.SiLU(),
                                                    nn.Linear(128, 20)).to(self.device)
            self.mlp_extractor.indices = self.anticipation_indices
        for module in (self.mlp_extractor.encoder, self.prediction_head):
            module.requires_grad_(False)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)


def frozen_state(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()
        if k.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_'))
        or k == 'mlp_extractor.feature_gain'}


def frozen_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(frozen_state(policy).items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def architecture_identity(policy):
    return {'schema': 'myo.arm_feature_comparison.v1', 'kind': policy.feature_kind,
        'encoder_indices': list(policy.mlp_extractor.indices),
        'encoder_layers': [128]*4, 'actor_private_layers': [128]*2,
        'critic_private_layers': [128]*2, 'actor_post_layers': [128]*2,
        'critic_post_layers': [128]*2, 'feature_gain': policy.feature_gain_spec,
        'frozen_sha256': frozen_digest(policy),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def make_comparison(env, parent, kind, predictor_path=None, *, learning_rate=3e-5,
                    n_steps=1024, batch_size=128, n_epochs=10):
    contract = env.get_wrapper_attr('contract')()
    indices = contract['physical_indices']+contract['pending_indices']+[2]
    mode = 'none' if kind == 'none' else 'frozen'
    model = ArmPPO(ComparisonPolicy, env, seed=0, device='cpu', learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': contract['predictor_indices'],
            'physical_indices': contract['physical_indices'], 'feature_mode': mode,
            'feature_kind': kind, 'anticipation_indices': indices, 'feature_gain': .25,
            'activation_fn': nn.SiLU, 'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})
    policy = model.policy
    with torch.no_grad():
        common = dict(parent.policy.named_parameters())
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(common[name])
        # Parent never used these columns (zero features). Start all new paths
        # at zero contribution, retaining exactly the parent's mean/value function.
        for tail in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
            tail[0].weight[:, :128].zero_()
        if kind != 'none':
            with torch.random.fork_rng(devices=[]):
                body, predictor_meta = (load_predictor(predictor_path) if kind == 'new60'
                                        else predictor_identity(predictor_path))
            policy.mlp_extractor.encoder.load_state_dict(body.encoder.state_dict())
            policy.prediction_head.load_state_dict(body.head.state_dict())
            policy.delta_mean.copy_(body.delta_mean)
            policy.delta_std.copy_(body.delta_std)
            policy.delta_calibrated.fill_(True)
        else:
            predictor_meta = None
    # Carry the same private/readout RMSprop moments into all three branches.
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad and common[name] in parent.policy.optimizer.state:
            state = copy.deepcopy(parent.policy.optimizer.state[common[name]])
            if name in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'):
                for value in state.values():
                    if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                        value[:, :128].zero_()
            policy.optimizer.state[parameter] = state
    model.num_timesteps = parent.num_timesteps
    model.optimizer_steps = parent.optimizer_steps
    model._n_updates = parent._n_updates
    return model, predictor_meta


def load_comparison(path):
    model, env, meta = load_arm(path)
    try:
        spec = meta['training']['feature_comparison']
        policy = model.policy
        if not isinstance(policy, ComparisonPolicy) or spec != architecture_identity(policy):
            raise ValueError('Arm feature comparison identity mismatch')
        kind = policy.feature_kind
        expected_indices = (env.physical_indices+env.pending_indices+[2] if kind == 'new60'
                            else env.predictor_indices)
        if (policy.mlp_extractor.indices != expected_indices
                or policy.feature_mode != ('none' if kind == 'none' else 'frozen')
                or float(policy.mlp_extractor.feature_gain) != policy.feature_gain_spec
                or any(p.requires_grad for p in policy.mlp_extractor.encoder.parameters())
                or any(p.requires_grad for p in policy.prediction_head.parameters())):
            raise ValueError('Arm feature comparison input/frozen contract mismatch')
        predictor = meta['training']['predictor']
        if kind == 'new60' and predictor['identity'] != anticipation_identity(env.contract()):
            raise ValueError('Anticipation predictor identity mismatch')
        if kind == 'old20' and predictor['contract'] != env.contract():
            raise ValueError('One-step predictor identity mismatch')
        if kind == 'none' and predictor is not None:
            raise ValueError('Unexpected predictor in none branch')
    except Exception:
        env.close()
        raise
    return model, env, meta
