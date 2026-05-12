"""função de recompensa do agente PPO discreto.

3 componentes ativos:
    goal           — terminal, +10.0 quando state.scoring_team == player.team
    concede        — terminal,  -5.0 quando outro time marcou
    kick_on_goal   — event-triggered, KICK_ON_GOAL_BASE × cos² × proximity²
                     (edge-only no is_kicking; direcional; proximity quadrática)

princípio de design: skill "ir até a bola e finalizar" é ensinada pelo
currículo (fases 1A/1B/1C com bola colada/próxima/com ângulo), não via
reward shaping denso. shaping denso historicamente produziu farming e
atratores de canto neste projeto.

invariantes preservados na fórmula KICK_ON_GOAL:
    - max(0, cos) zera chute na direção contrária ao gol.
    - edge-only no is_kicking impede que o reward dispare 6× por chute
      (o engine mantém is_kicking ativo por 6 frames).
    - proximity² faz chute longe valer ~0, evitando farming de chutes alinhados de longe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.physics.entities import GameState, Player


GOAL_REWARD: float = 10.0
CONCEDE_PENALTY: float = -5.0

# fórmula: KICK_ON_GOAL_BASE × cos² × proximity². range [0, 2.5].
# razão GOAL/kick_max = 4× — chute pago é minoria face ao gol.
KICK_ON_GOAL_BASE: float = 2.5
KICK_PROXIMITY_RANGE: float = 250.0


@dataclass(frozen=True)
class StateSnapshot:
    """mínimo necessário para calcular GOAL/CONCEDE/KICK_ON_GOAL."""
    player_x: float
    player_y: float
    player_angle: float
    player_is_kicking: bool
    player_team: str

    ball_x: float
    ball_y: float

    enemy_goal_center_x: float
    enemy_goal_center_y: float

    dist_player_to_ball: float
    dist_ball_to_enemy_goal: float
    is_in_ball_contact: bool


def make_snapshot(player: Player, state: GameState) -> StateSnapshot:
    """constrói um StateSnapshot a partir de Player + GameState vivos."""
    field_w = float(state.field.width)
    field_h = float(state.field.height)

    if player.team == "red":
        enemy_goal_x = field_w
    else:
        enemy_goal_x = 0.0
    enemy_goal_y = field_h / 2.0

    dx_pb = state.ball.x - player.x
    dy_pb = state.ball.y - player.y
    dist_pb = math.sqrt(dx_pb * dx_pb + dy_pb * dy_pb)

    dx_bg = enemy_goal_x - state.ball.x
    dy_bg = enemy_goal_y - state.ball.y
    dist_bg = math.sqrt(dx_bg * dx_bg + dy_bg * dy_bg)

    contact_threshold = player.radius + state.ball.radius

    return StateSnapshot(
        player_x=float(player.x),
        player_y=float(player.y),
        player_angle=float(player.angle),
        player_is_kicking=bool(player.is_kicking),
        player_team=str(player.team),
        ball_x=float(state.ball.x),
        ball_y=float(state.ball.y),
        enemy_goal_center_x=float(enemy_goal_x),
        enemy_goal_center_y=float(enemy_goal_y),
        dist_player_to_ball=float(dist_pb),
        dist_ball_to_enemy_goal=float(dist_bg),
        is_in_ball_contact=bool(dist_pb < contact_threshold),
    )


def compute_rewards(
    snap_prev: StateSnapshot,
    snap_curr: StateSnapshot,
    state_curr: GameState,
) -> tuple[float, dict[str, float]]:
    """calcula reward do step para o player do snapshot.

    retorna (total, breakdown). breakdown sempre contém as 3 chaves
    canônicas (goal, concede, kick_on_goal) — mesmo que zero — pra que
    o logger por componente nunca veja chave faltando.
    """
    if snap_prev.player_team != snap_curr.player_team:
        raise ValueError(
            f"snap_prev.player_team ({snap_prev.player_team!r}) != "
            f"snap_curr.player_team ({snap_curr.player_team!r}); snapshots trocados?"
        )

    breakdown: dict[str, float] = {
        "goal": 0.0,
        "concede": 0.0,
        "kick_on_goal": 0.0,
    }

    goal_scored = state_curr.goal_scored_this_step

    if goal_scored:
        if state_curr.scoring_team == snap_curr.player_team:
            breakdown["goal"] = GOAL_REWARD
        else:
            breakdown["concede"] = CONCEDE_PENALTY

    # edge-only: dispara apenas no frame em que is_kicking liga.
    if snap_curr.player_is_kicking and not snap_prev.player_is_kicking:
        heading_x = math.cos(snap_curr.player_angle)
        heading_y = math.sin(snap_curr.player_angle)
        vec_x = snap_curr.enemy_goal_center_x - snap_curr.player_x
        vec_y = snap_curr.enemy_goal_center_y - snap_curr.player_y
        norm = math.sqrt(vec_x * vec_x + vec_y * vec_y)
        if norm < 1e-6:
            cos_alignment = 0.0
        else:
            cos_alignment = (heading_x * vec_x + heading_y * vec_y) / norm
        cos_clamped = max(0.0, cos_alignment)
        proximity = max(
            0.0,
            1.0 - snap_curr.dist_ball_to_enemy_goal / KICK_PROXIMITY_RANGE,
        )
        breakdown["kick_on_goal"] = (
            KICK_ON_GOAL_BASE
            * (cos_clamped ** 2)
            * (proximity ** 2)
        )

    total = sum(breakdown.values())
    return float(total), breakdown
