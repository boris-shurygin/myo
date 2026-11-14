"""Auxiliary command fitting with explicit retention of the active torque proxy.

Separate from the source-pinned initial weakening adapter so its checkpoints
remain loadable. This changes the training objective, not runtime commands.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

from .arm_weakening import weakening_labels


def weaken_with_moment_constraint(model, observations, contract, *, updates=64,
                                  batch_size=256, moment_scale_Nm=1.):
    if not np.isfinite(moment_scale_Nm) or moment_scale_Nm<=0:
        raise ValueError('Positive torque-proxy loss scale required')
    obs,target,gate,a,b=weakening_labels(model,observations,contract)
    if not gate.any():
        return {'attempted_updates':0,'accepted_updates':0,'gate_fraction':0.,'stop':'no_settled_states'}
    old_mean=a/(a+b)
    anchor=torch.linspace(0,len(obs)-1,min(2048,len(obs)),device=model.device).long()
    losses=[];commands=[];moments=[];count=0;rejected=False;mean_kl=maximum_kl=0.
    model.policy.set_training_mode(True)
    for index in range(updates):
        selected=(torch.arange(batch_size,device=model.device)+index*batch_size)%len(obs)
        weights=copy.deepcopy(model.policy.state_dict())
        optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
        mean=model.policy.get_distribution(obs[selected]).distribution.mean
        command_loss=.5*((mean-target[selected])/model.policy.weakening.loss_scale).square().mean()
        # g is an observed causal mechanical proxy. Differentiation is through
        # commands and the network, never through MuJoCo or a future outcome.
        delta_moment=((mean-target[selected])*obs[selected,-6:]).sum(-1)
        moment_loss=.5*(delta_moment/moment_scale_Nm).square().mean()
        loss=command_loss+moment_loss
        model.policy.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm)
        model.policy.optimizer.step();model.optimizer_steps+=1
        losses.append(float(loss.detach()));commands.append(float(command_loss.detach()));moments.append(float(moment_loss.detach()))
        with torch.no_grad():
            new=model.policy.get_distribution(obs[anchor]).distribution
            kl=kl_divergence(Beta(a[anchor].double(),b[anchor].double()),
                Beta(new.concentration1.double(),new.concentration0.double())).sum(-1).clamp_min(0.)
            trial_mean,trial_maximum=float(kl.mean()),float(kl.max())
        if trial_mean>model.policy.weakening.mean_kl_limit or trial_maximum>model.policy.weakening.maximum_kl_limit:
            model.policy.load_state_dict(weights);model.policy.optimizer.load_state_dict(optimizer)
            rejected=True;break
        count+=1;mean_kl,maximum_kl=trial_mean,trial_maximum
    with torch.no_grad():
        mean=model.policy.get_distribution(obs).distribution.mean
        delta=((mean-old_mean)*obs[:,-6:]).sum(-1)
    return {'attempted_updates':len(losses),'accepted_updates':count,'gate_fraction':float(gate.float().mean()),
        'mean_loss':float(np.mean(losses)),'command_loss':float(np.mean(commands)),
        'moment_loss':float(np.mean(moments)),'moment_scale_Nm':moment_scale_Nm,
        'mean_kl':mean_kl,'maximum_kl':maximum_kl,
        'mean_command_reduction':float((old_mean-mean).mean()),
        'settled_command_reduction':float((old_mean[gate]-mean[gate]).mean()),
        'settled_proxy_moment_delta_rms_Nm':float(delta[gate].square().mean().sqrt()),
        'nonsettled_command_change_rms':float((mean[~gate]-old_mean[~gate]).square().mean().sqrt()) if (~gate).any() else 0.,
        'stop':'kl_budget' if rejected else 'update_budget'}
