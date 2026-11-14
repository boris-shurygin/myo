"""Reuse muscle control without passing the old positional task to the lower layer."""
import torch


def transfer_fixed_goal_actor(model, env, parent, parent_env, *, fixed_goal_deg=127.5):
    """Fold constant old goal inputs into first-layer biases, zero new commands.

    This preserves the parent's excitation distribution at one fixed old goal.
    It does NOT turn the old position policy into an acceleration controller.
    PPO/optimizer counters and moments stay fresh for the changed objective.
    """
    import numpy as np
    old_names, names = parent_env.observation_names, env.observation_names
    fixed = {}
    for index, name in enumerate(old_names):
        if name in parent_env.unwrapped._command_names or '/command/' in name:
            field = name.split('/')[-1]
            if 'centered_rad' in field:
                raw = np.deg2rad(fixed_goal_deg)-parent_env.unwrapped.config.reference_center
            else:
                raw = 1. if field == 'trigger' else 0.
            fixed[index] = float(raw*parent_env.scale[index]+parent_env.offset[index])
    mapping = {i: old_names.index('exploration/stagnation_pressure' if name == 'exploration/pressure' else name)
               for i, name in enumerate(names)
               if name in old_names or name == 'exploration/pressure'}
    if len(mapping) != len(names)-3 or len(fixed) != 12:
        raise ValueError('Unexpected old/new physical or goal observation layout')
    source = dict(parent.policy.named_parameters())
    with torch.no_grad():
        for name, parameter in model.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == 'value_net.weight' or name == 'value_net.bias':
                parameter.zero_()  # New return units/sign; no old critic value bias.
            elif name in ('mlp_extractor.actor_private.0.weight', 'mlp_extractor.critic_private.0.weight'):
                parameter.zero_()
                for column, old_column in mapping.items():
                    parameter[:, column].copy_(source[name][:, old_column])
            elif name in ('mlp_extractor.actor_private.0.bias', 'mlp_extractor.critic_private.0.bias'):
                parameter.copy_(source[name])
                weights = source[name.removesuffix('bias')+'weight']
                for index, value in fixed.items():
                    parameter.add_(weights[:, index]*value)
            else:
                if parameter.shape != source[name].shape:
                    raise ValueError(f'Unexpected transfer shape: {name}')
                parameter.copy_(source[name])
    return {'type': 'fixed_goal_actor_transfer', 'fixed_goal_deg': fixed_goal_deg,
            'new_command_columns': 'zero', 'old_goal_columns': 'constant folded into biases',
            'value_output': 'zero', 'optimizer': 'fresh', 'old_goal_available_to_lower_policy': False}
