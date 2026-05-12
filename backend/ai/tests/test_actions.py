"""testes unitários do adapter de ação discreta.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_actions.py -v
"""

import pytest

from backend.ai.actions import (
    ACTION_SPACE_SIZE,
    apply_discrete_action,
    decode_action,
    encode_action,
)
from backend.physics.entities import Player


def test_round_trip_encode_decode():
    """encode(decode(i)) == i para todos os 18 índices."""
    for i in range(ACTION_SPACE_SIZE):
        accel, rot, kick = decode_action(i)
        assert encode_action(accel, rot, kick) == i


def test_decode_covers_full_cartesian_product():
    """as 18 triplas decodificadas == produto cartesiano completo."""
    decoded = {decode_action(i) for i in range(ACTION_SPACE_SIZE)}
    expected = {
        (a, r, k)
        for a in (-1, 0, 1)
        for r in (-1, 0, 1)
        for k in (0, 1)
    }
    assert len(decoded) == ACTION_SPACE_SIZE
    assert decoded == expected


@pytest.mark.parametrize("bad_idx", [-1, 18, 100, -100])
def test_apply_invalid_idx_raises(bad_idx):
    """apply_discrete_action com idx inválido levanta ValueError."""
    p = Player(id="red_0", team="red")
    with pytest.raises(ValueError):
        apply_discrete_action(p, bad_idx)


@pytest.mark.parametrize(
    "accel, rot, kick",
    [
        (2, 0, 0),
        (-2, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, 0, 5),
        (0, 0, -1),
    ],
)
def test_encode_invalid_inputs_raise(accel, rot, kick):
    """encode_action com inputs fora do set permitido levanta ValueError."""
    with pytest.raises(ValueError):
        encode_action(accel, rot, kick)


def test_apply_sets_player_fields_correctly():
    """cinco índices escolhidos: campos do Player ficam com valores esperados."""
    p = Player(id="red_0", team="red")
    cases = {
        0:  (-1.0, -1.0, False),
        8:  ( 0.0,  0.0, False),
        9:  ( 0.0,  0.0, True),
        15: ( 1.0,  0.0, True),
        17: ( 1.0,  1.0, True),
    }
    for idx, (e_accel, e_rot, e_kick) in cases.items():
        apply_discrete_action(p, idx)
        assert p.accel == e_accel, f"idx={idx}: accel esperado {e_accel}, obtido {p.accel}"
        assert p.rot == e_rot, f"idx={idx}: rot esperado {e_rot}, obtido {p.rot}"
        assert p.kick_requested == e_kick, (
            f"idx={idx}: kick_requested esperado {e_kick}, obtido {p.kick_requested}"
        )


def test_apply_writes_correct_python_types():
    """accel/rot devem ser float; kick_requested deve ser bool (não int)."""
    p = Player(id="red_0", team="red")
    for idx in range(ACTION_SPACE_SIZE):
        apply_discrete_action(p, idx)
        assert type(p.accel) is float, f"idx={idx}: accel não é float ({type(p.accel)})"
        assert type(p.rot) is float, f"idx={idx}: rot não é float ({type(p.rot)})"
        assert type(p.kick_requested) is bool, (
            f"idx={idx}: kick_requested não é bool ({type(p.kick_requested)})"
        )
