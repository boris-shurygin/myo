"""PyTorch Muon for matrices and Adam for vectors, with one SB3 state dict."""
from __future__ import annotations

import torch


class MuonAdam(torch.optim.Optimizer):
    """Delegate algorithms to torch; share parameter groups/state for LR and IO."""

    def __init__(self, params, lr=3e-4):
        params = list(params)
        self.muon = torch.optim.Muon([p for p in params if p.ndim == 2], lr=lr,
            weight_decay=0., momentum=.95, nesterov=True, ns_steps=5, adjust_lr_fn="original")
        self.adam = torch.optim.Adam([p for p in params if p.ndim != 2], lr=lr,
                                     eps=1e-5, weight_decay=0.)
        super().__init__(self.muon.param_groups+self.adam.param_groups, {"lr": lr})
        self._bind()

    def _bind(self):
        self.muon.param_groups = self.param_groups[:1]
        self.adam.param_groups = self.param_groups[1:]
        self.muon.state = self.adam.state = self.state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._bind()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adam.step()
        return loss
