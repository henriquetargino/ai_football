"""testes unitários do construtor de observação gather_obs.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_obs.py -v
"""

import math

import numpy as np
import pytest

from backend.ai.obs import (
    NUM_RAYS_FROM_CONFIG,
    OBS_SIZE,
    RAY_FEATURE_SIZE,
    STATE_EXTRA_SIZE,
    TYPE_TO_ONEHOT_IDX,
    decode_ray_features,
    gather_obs,
)
from backend.config import MAX_SPEED_PLAYER, TYPE_MAP
from backend.physics.entities import Ball, Field, GameState, Player


def _make_state_with_player():
    """state básico: red player no centro-esquerda, bola no centro do campo."""
    state = GameState()
    state.field = Field()
    state.init()
    state.ball = Ball(x=400.0, y=250.0)
    p = Player(id="red_0", team="red", x=200.0, y=250.0, angle=0.0)
    state.players = [p]
    return state, p


def _fake_raw(detections=None):
    """constrói 96 floats (48 raios × 2). detections = {ray_idx: (norm_dist, type_str)}.

    raios não listados ficam como "none" (norm_dist=1.0, type_id=-1.0).
    """
    detections = detections or {}
    raw = []
    for i in range(NUM_RAYS_FROM_CONFIG):
        if i in detections:
            d, t = detections[i]
            raw.extend([d, TYPE_MAP[t]])
        else:
            raw.extend([1.0, TYPE_MAP["none"]])
    return raw


def test_shape_and_dtype():
    state, p = _make_state_with_player()
    obs = gather_obs(p, state)
    assert obs.shape == (341,)
    assert obs.dtype == np.float32


def test_constants():
    assert OBS_SIZE == 341
    assert RAY_FEATURE_SIZE == 7
    assert STATE_EXTRA_SIZE == 5
    assert NUM_RAYS_FROM_CONFIG == 48
    assert len(TYPE_TO_ONEHOT_IDX) == 6
    assert "none" not in TYPE_TO_ONEHOT_IDX


def test_sentinel_when_no_detection(monkeypatch):
    state, p = _make_state_with_player()
    monkeypatch.setattr(
        "backend.ai.obs.cast_rays",
        lambda player, st: _fake_raw({}),
    )
    obs = gather_obs(p, state)
    for i in range(NUM_RAYS_FROM_CONFIG):
        base = i * RAY_FEATURE_SIZE
        assert obs[base] == -1.0, f"raio {i}: distance esperada -1.0, obtida {obs[base]}"
        for k in range(6):
            assert obs[base + 1 + k] == 0.0, (
                f"raio {i}: bit one-hot {k} deveria ser 0, obtido {obs[base + 1 + k]}"
            )


def test_onehot_mutually_exclusive_real_state():
    """em qualquer state real, soma do one-hot ∈ {0, 1}."""
    state, p = _make_state_with_player()
    obs = gather_obs(p, state)
    for i in range(NUM_RAYS_FROM_CONFIG):
        base = i * RAY_FEATURE_SIZE
        onehot_sum = float(obs[base + 1:base + RAY_FEATURE_SIZE].sum())
        assert onehot_sum in (0.0, 1.0), (
            f"raio {i}: soma do one-hot deve ser 0 ou 1, obtida {onehot_sum}"
        )
        if obs[base] == -1.0:
            assert onehot_sum == 0.0, f"raio {i}: distance=-1 mas one-hot soma {onehot_sum}"
        else:
            assert onehot_sum == 1.0, f"raio {i}: distance≠-1 mas one-hot soma {onehot_sum}"


@pytest.mark.parametrize("type_str, expected_idx", list(TYPE_TO_ONEHOT_IDX.items()))
def test_type_translation_to_onehot(monkeypatch, type_str, expected_idx):
    state, p = _make_state_with_player()
    monkeypatch.setattr(
        "backend.ai.obs.cast_rays",
        lambda player, st: _fake_raw({0: (0.5, type_str)}),
    )
    obs = gather_obs(p, state)

    assert obs[0] == pytest.approx(0.5)
    for k in range(6):
        expected_bit = 1.0 if k == expected_idx else 0.0
        assert obs[1 + k] == expected_bit, (
            f"tipo {type_str}: bit {k} esperado {expected_bit}, obtido {obs[1 + k]}"
        )

    for i in range(1, NUM_RAYS_FROM_CONFIG):
        base = i * RAY_FEATURE_SIZE
        assert obs[base] == -1.0
        assert obs[base + 1:base + RAY_FEATURE_SIZE].sum() == 0.0


def test_state_extra_speed_rot_kick():
    state, p = _make_state_with_player()
    p.vx, p.vy = 2.1, 0.0
    p.rot = 0.5
    p.can_kick = True
    obs = gather_obs(p, state)
    assert obs[336] == pytest.approx(2.1 / MAX_SPEED_PLAYER, rel=1e-5)
    assert obs[337] == pytest.approx(0.5)
    assert obs[338] == 1.0


def test_state_extra_can_kick_false():
    state, p = _make_state_with_player()
    p.can_kick = False
    obs = gather_obs(p, state)
    assert obs[338] == 0.0


def test_ball_visible_anywhere_false(monkeypatch):
    state, p = _make_state_with_player()
    monkeypatch.setattr(
        "backend.ai.obs.cast_rays",
        lambda player, st: _fake_raw({0: (0.5, "wall"), 5: (0.3, "enemy")}),
    )
    obs = gather_obs(p, state)
    assert obs[339] == 0.0


def test_ball_visible_anywhere_true_via_real_state():
    state, p = _make_state_with_player()
    obs = gather_obs(p, state)
    assert obs[339] == 1.0


def test_ball_visible_anywhere_true_via_monkeypatch(monkeypatch):
    state, p = _make_state_with_player()
    monkeypatch.setattr(
        "backend.ai.obs.cast_rays",
        lambda player, st: _fake_raw({7: (0.4, "ball")}),
    )
    obs = gather_obs(p, state)
    assert obs[339] == 1.0


@pytest.mark.parametrize("idx, expected", [(0, 0.0), (1, 0.25), (2, 0.5), (3, 0.75)])
def test_action_repeat_idx_normalization(idx, expected):
    state, p = _make_state_with_player()
    obs = gather_obs(p, state, action_repeat_idx=idx)
    assert obs[340] == pytest.approx(expected)


@pytest.mark.parametrize("bad_idx", [-1, 4, 100, -100])
def test_action_repeat_idx_invalid_raises(bad_idx):
    state, p = _make_state_with_player()
    with pytest.raises(ValueError):
        gather_obs(p, state, action_repeat_idx=bad_idx)


def test_deterministic_for_same_state():
    state, p = _make_state_with_player()
    obs1 = gather_obs(p, state)
    obs2 = gather_obs(p, state)
    assert np.array_equal(obs1, obs2)


def test_decode_ray_no_detection():
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = -1.0
    result = decode_ray_features(obs, 0)
    assert result == {
        "distance": None,
        "detected_type": None,
        "raw_onehot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_decode_ray_ball_detection():
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = 0.5
    obs[1 + TYPE_TO_ONEHOT_IDX["ball"]] = 1.0
    result = decode_ray_features(obs, 0)
    assert result == {
        "distance": 0.5,
        "detected_type": "ball",
        "raw_onehot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize("type_str, expected_idx", list(TYPE_TO_ONEHOT_IDX.items()))
def test_decode_ray_round_trip_all_types(monkeypatch, type_str, expected_idx):
    """round-trip: gather_obs → decode_ray_features deve recuperar o tipo."""
    state, p = _make_state_with_player()
    monkeypatch.setattr(
        "backend.ai.obs.cast_rays",
        lambda player, st: _fake_raw({3: (0.7, type_str)}),
    )
    obs = gather_obs(p, state)
    result = decode_ray_features(obs, 3)
    assert result["distance"] == pytest.approx(0.7)
    assert result["detected_type"] == type_str
    assert result["raw_onehot"][expected_idx] == 1.0
    assert sum(result["raw_onehot"]) == 1.0
