"""arquitetura da rede neural do agente PPO discreto.

MLP feed-forward com body compartilhado entre policy e value heads:
    obs (341) → tanh(fc1, 64) → tanh(fc2, 64) → policy_head (18) + value_head (1)

decisões cravadas:
    - 2 hidden × 64 unidades. rede pequena ⇒ inferência JS <1ms e treino
      mais rápido em CPU que MPS para problemas low-dim discretos.
    - tanh em todas as hidden (default CleanRL).
    - init ortogonal: hidden std=√2, policy_head std=0.01 (evita explosão
      dos logits no início), value_head std=1.0.
    - body único compartilhado: ambas as heads recebem o mesmo gradiente.

tamanho total: ~27k parâmetros. esta arquitetura é a única do projeto —
mudar aqui = mudar tudo downstream (self_play, train, export_for_js).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.obs import OBS_SIZE


HIDDEN_SIZE: int = 64
NUM_HIDDEN_LAYERS: int = 2


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """init ortogonal canônico do CleanRL.

    convenção por tipo de camada:
        hidden:      std=sqrt(2)
        policy_head: std=0.01 (gradients pequenos no início)
        value_head:  std=1.0
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    """rede do AI Soccer 1v1 com body compartilhado entre policy e value."""

    def __init__(self):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(OBS_SIZE, HIDDEN_SIZE))
        self.fc2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.policy_head = layer_init(nn.Linear(HIDDEN_SIZE, ACTION_SPACE_SIZE), std=0.01)
        self.value_head = layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0)

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc2(torch.tanh(self.fc1(x))))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """forward só pra value head. usado em GAE bootstrap."""
        return self.value_head(self._body(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward completo.

        se action é fornecida (durante optimization), usa pra computar
        log_probs e entropia. se None (durante rollout), amostra da Categorical.

        retorna (action, log_prob, entropy, value).
        """
        body = self._body(x)
        logits = self.policy_head(body)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(body)

    @torch.no_grad()
    def select_action_greedy(self, obs: np.ndarray):
        """inferência argmax (deterministic). usado para opponent ou demo."""
        is_single = obs.ndim == 1
        x = torch.from_numpy(obs.astype(np.float32))
        if is_single:
            x = x.unsqueeze(0)
        logits = self.policy_head(self._body(x))
        actions = logits.argmax(dim=-1)
        if is_single:
            return int(actions.item())
        return actions.numpy()

    @torch.no_grad()
    def select_action_stochastic(self, obs: np.ndarray):
        """inferência amostrando da Categorical. mantém variabilidade mesmo em opponent congelado."""
        is_single = obs.ndim == 1
        x = torch.from_numpy(obs.astype(np.float32))
        if is_single:
            x = x.unsqueeze(0)
        logits = self.policy_head(self._body(x))
        actions = Categorical(logits=logits).sample()
        if is_single:
            return int(actions.item())
        return actions.numpy()
