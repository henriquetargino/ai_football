"""trackers para logging por componente.

sem isso, "reward total subiu mas não sei por quê" volta a ser o modo
padrão de debug. com isso, cada componente do reward tem média e cumulativo
separados em TensorBoard, e os parâmetros do currículo (goal_width,
spawn_distance, agent_initial_angle) são histogramados pra detectar
overfit a easy mode.

PPOTrainer mantém uma instância de cada tracker por vetor de envs.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


# chaves canônicas. tracker itera sobre BREAKDOWN_KEYS para se adaptar
# automaticamente a mudanças na composição do reward.
BREAKDOWN_KEYS: tuple[str, ...] = (
    "goal", "concede", "kick_on_goal",
)


class RewardBreakdownTracker:
    """acumula breakdown por env, retorna episódios completos quando terminam."""

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError(f"num_envs deve ser positivo, recebido {num_envs}")
        self._num_envs = num_envs
        self._acc: list[dict[str, float]] = [self._zero() for _ in range(num_envs)]
        self._length: list[int] = [0] * num_envs

    @staticmethod
    def _zero() -> dict[str, float]:
        return {k: 0.0 for k in BREAKDOWN_KEYS}

    def step(
        self,
        breakdowns: list[dict],
        dones: list[bool],
    ) -> list[dict]:
        """adiciona breakdowns ao acumulador e devolve episódios finalizados.

        cada dict retornado tem as chaves de BREAKDOWN_KEYS + "episode_length".
        reset interno do env i é feito depois de adicionar à lista de retornados.
        """
        if len(breakdowns) != self._num_envs or len(dones) != self._num_envs:
            raise ValueError(
                f"esperado {self._num_envs} entradas; "
                f"breakdowns={len(breakdowns)}, dones={len(dones)}"
            )

        finished = []
        for i, (b, done) in enumerate(zip(breakdowns, dones)):
            for k in BREAKDOWN_KEYS:
                self._acc[i][k] += float(b[k])
            self._length[i] += 1
            if done:
                ep = dict(self._acc[i])
                ep["episode_length"] = self._length[i]
                finished.append(ep)
                self._acc[i] = self._zero()
                self._length[i] = 0
        return finished


class EnvParamTracker:
    """histórico recente dos parâmetros de domain randomization.

    permite logar histograma + correlação com goal_for_rate para detectar
    overfit ao easy mode (ex: agente só marca em goal_width > 250).
    """

    def __init__(self, max_history: int = 1000):
        if max_history <= 0:
            raise ValueError(f"max_history deve ser positivo, recebido {max_history}")
        self.goal_widths: deque[float] = deque(maxlen=max_history)
        self.spawn_distances: deque[float] = deque(maxlen=max_history)
        self.agent_initial_angles: deque[float] = deque(maxlen=max_history)
        self.trained_teams: deque[str] = deque(maxlen=max_history)

    def add_episode(self, info: dict) -> None:
        """adiciona um episódio; info deve ter as keys vindas do env.reset()."""
        if "goal_width" in info:
            self.goal_widths.append(float(info["goal_width"]))
        if "spawn_distance" in info:
            self.spawn_distances.append(float(info["spawn_distance"]))
        if "agent_initial_angle" in info:
            self.agent_initial_angles.append(float(info["agent_initial_angle"]))
        if "trained_team" in info:
            self.trained_teams.append(str(info["trained_team"]))

    def to_summary(self) -> dict[str, float]:
        """estatísticas agregadas pro log scalar."""
        out: dict[str, float] = {"history_size": float(len(self.goal_widths))}
        if self.goal_widths:
            out["goal_width_mean"] = sum(self.goal_widths) / len(self.goal_widths)
            out["goal_width_min"] = min(self.goal_widths)
            out["goal_width_max"] = max(self.goal_widths)
        if self.spawn_distances:
            out["spawn_distance_mean"] = sum(self.spawn_distances) / len(self.spawn_distances)
        if self.agent_initial_angles:
            out["agent_initial_angle_mean"] = (
                sum(self.agent_initial_angles) / len(self.agent_initial_angles)
            )
        if self.trained_teams:
            n_red = sum(1 for t in self.trained_teams if t == "red")
            out["trained_team_red_ratio"] = n_red / len(self.trained_teams)
        return out
