"""adapter de ação discreta para o Player.

mapeia um índice em [0, 17] em três campos do Player:
    accel ∈ {-1.0, 0.0, +1.0}    (ré, parado, frente)
    rot   ∈ {-1.0, 0.0, +1.0}    (esquerda, parado, direita)
    kick_requested ∈ {False, True}

codificação canônica (replicada bit-a-bit em frontend/src/ai/actions.js):
    idx = (accel_i * 3 + rot_i) * 2 + kick
    onde accel_i = accel + 1 e rot_i = rot + 1.

inversa: kick = idx % 2; (accel_i, rot_i) = divmod(idx // 2, 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.physics.entities import Player


ACTION_SPACE_SIZE: int = 18

_ACCEL_VALUES = (-1, 0, 1)
_ROT_VALUES = (-1, 0, 1)
_KICK_VALUES = (0, 1)


def decode_action(action_idx: int) -> tuple[int, int, int]:
    """decodifica action_idx em (accel, rot, kick). inversa de encode_action."""
    if not 0 <= action_idx < ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_idx fora do range [0, {ACTION_SPACE_SIZE - 1}]: {action_idx!r}"
        )
    kick = action_idx % 2
    accel_i, rot_i = divmod(action_idx // 2, 3)
    return (_ACCEL_VALUES[accel_i], _ROT_VALUES[rot_i], _KICK_VALUES[kick])


def encode_action(accel: int, rot: int, kick: int) -> int:
    """codifica (accel, rot, kick) no índice canônico. inversa de decode_action."""
    if accel not in _ACCEL_VALUES:
        raise ValueError(f"accel deve estar em {_ACCEL_VALUES}, recebido {accel!r}")
    if rot not in _ROT_VALUES:
        raise ValueError(f"rot deve estar em {_ROT_VALUES}, recebido {rot!r}")
    if kick not in _KICK_VALUES:
        raise ValueError(f"kick deve estar em {_KICK_VALUES}, recebido {kick!r}")
    accel_i = accel + 1
    rot_i = rot + 1
    return (accel_i * 3 + rot_i) * 2 + kick


def apply_discrete_action(player: Player, action_idx: int) -> None:
    """decodifica action_idx e escreve nos três campos do Player consumidos por physics_step."""
    accel, rot, kick = decode_action(action_idx)
    player.accel = float(accel)
    player.rot = float(rot)
    player.kick_requested = bool(kick)
