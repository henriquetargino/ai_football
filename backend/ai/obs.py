"""vetor de observação 341-dim do Player.

traduz a saída ordinal de raycasting.cast_rays (96 floats: 48 raios ×
[distância normalizada, tipo ordinal]) em vetor com one-hot por tipo +
5 floats de estado extra:

    obs[  0 .. 335]  →  336 floats — 48 raios × 7 floats por raio
    obs[336 .. 340]  →    5 floats — speed, angularVel, canKick,
                                      ball_visible_anywhere, action_repeat

por raio (7 floats):
    [0]  distance normalizada ∈ [0, 1]   OU  -1.0 (sentinela "none")
    [1]  one_hot wall          ∈ {0, 1}
    [2]  one_hot ball          ∈ {0, 1}
    [3]  one_hot ally          ∈ {0, 1}
    [4]  one_hot enemy         ∈ {0, 1}
    [5]  one_hot goal_own      ∈ {0, 1}
    [6]  one_hot goal_enemy    ∈ {0, 1}

sentinela canônica de ausência:
    distance = -1.0  e  one_hot = (0, 0, 0, 0, 0, 0)   (none não tem bit).

state extra (índices 336..340):
    [336]  speed                  = ‖(vx, vy)‖ / MAX_SPEED_PLAYER ∈ [0, 1]
    [337]  angularVel             = player.rot                    ∈ [-1, 1]
    [338]  canKick                = 1.0 se player.can_kick, senão 0.0
    [339]  ball_visible_anywhere  = 1.0 se algum raio detectou bola
    [340]  action_repeat_idx      = idx / 4.0                     ∈ [0, 0.75]

contrato Python ↔ JS: cada decisão (sentinela, índices do one-hot,
normalização do action_repeat, dtype float32) é replicada bit-a-bit no frontend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from backend.config import MAX_SPEED_PLAYER, NUM_RAYS, TYPE_MAP
from backend.physics.raycasting import cast_rays

if TYPE_CHECKING:
    from backend.physics.entities import GameState, Player


NUM_RAYS_FROM_CONFIG: int = 48
RAY_FEATURE_SIZE: int = 7
STATE_EXTRA_SIZE: int = 5
OBS_SIZE: int = NUM_RAYS_FROM_CONFIG * RAY_FEATURE_SIZE + STATE_EXTRA_SIZE  # 341

assert NUM_RAYS == NUM_RAYS_FROM_CONFIG, (
    f"obs.py assume {NUM_RAYS_FROM_CONFIG} raios; config.NUM_RAYS={NUM_RAYS}"
)

# ordem fixa, replicada em JS. "none" não tem bit (sinalizado por distance=-1.0).
TYPE_TO_ONEHOT_IDX: dict[str, int] = {
    "wall":       0,
    "ball":       1,
    "ally":       2,
    "enemy":      3,
    "goal_own":   4,
    "goal_enemy": 5,
}

_FLOAT_TO_TYPE_STR: dict[float, str] = {v: k for k, v in TYPE_MAP.items()}

_STATE_EXTRA_BASE: int = NUM_RAYS_FROM_CONFIG * RAY_FEATURE_SIZE  # 336


def gather_obs(
    player: Player,
    state: GameState,
    action_repeat_idx: int = 0,
) -> np.ndarray:
    """constrói o vetor de observação de 341 floats float32 para o player.

    obs é construído da perspectiva do player. action_repeat_idx é o índice
    do frame atual no ciclo de action repeat (0..3).
    """
    if not 0 <= action_repeat_idx <= 3:
        raise ValueError(
            f"action_repeat_idx fora do range [0, 3]: {action_repeat_idx!r}"
        )

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    raw = cast_rays(player, state)

    ball_visible = False
    for i in range(NUM_RAYS_FROM_CONFIG):
        norm_dist = raw[2 * i]
        type_float = raw[2 * i + 1]
        type_str = _FLOAT_TO_TYPE_STR[type_float]
        base = i * RAY_FEATURE_SIZE

        if type_str == "none":
            obs[base] = -1.0
        else:
            obs[base] = norm_dist
            obs[base + 1 + TYPE_TO_ONEHOT_IDX[type_str]] = 1.0
            if type_str == "ball":
                ball_visible = True

    speed_norm = math.sqrt(player.vx * player.vx + player.vy * player.vy) / MAX_SPEED_PLAYER
    obs[_STATE_EXTRA_BASE + 0] = max(0.0, min(1.0, speed_norm))
    obs[_STATE_EXTRA_BASE + 1] = max(-1.0, min(1.0, float(player.rot)))
    obs[_STATE_EXTRA_BASE + 2] = 1.0 if player.can_kick else 0.0
    obs[_STATE_EXTRA_BASE + 3] = 1.0 if ball_visible else 0.0
    obs[_STATE_EXTRA_BASE + 4] = action_repeat_idx / 4.0

    return obs


def decode_ray_features(obs_vector: np.ndarray, ray_idx: int) -> dict:
    """helper de debug: decodifica o raio ray_idx em dict legível."""
    base = ray_idx * RAY_FEATURE_SIZE
    distance = float(obs_vector[base])
    raw_onehot = [float(x) for x in obs_vector[base + 1:base + RAY_FEATURE_SIZE]]

    if distance == -1.0:
        return {"distance": None, "detected_type": None, "raw_onehot": raw_onehot}

    detected_type = None
    for type_str, idx in TYPE_TO_ONEHOT_IDX.items():
        if raw_onehot[idx] == 1.0:
            detected_type = type_str
            break
    return {"distance": distance, "detected_type": detected_type, "raw_onehot": raw_onehot}
