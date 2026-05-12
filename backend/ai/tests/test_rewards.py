"""testes unitários da função de recompensa.

cobre os 3 componentes (goal, concede, kick_on_goal).

rodar (a partir da raiz):
    pytest backend/ai/tests/test_rewards.py -v
"""

import math
import time
from types import SimpleNamespace

import pytest

from backend.ai.rewards import (
    CONCEDE_PENALTY,
    GOAL_REWARD,
    KICK_ON_GOAL_BASE,
    KICK_PROXIMITY_RANGE,
    StateSnapshot,
    compute_rewards,
    make_snapshot,
)
from backend.physics.entities import Ball, Field, GameState, Player


_BREAKDOWN_KEYS = {"goal", "concede", "kick_on_goal"}


def _state_for_compute(scored=False, scoring_team=None):
    """mock mínimo do GameState — compute_rewards só lê esses dois campos."""
    return SimpleNamespace(
        goal_scored_this_step=scored,
        scoring_team=scoring_team,
    )


def _snap(
    team="red",
    is_kicking=False,
    in_contact=False,
    dist_pb=200.0,
    dist_bg=400.0,
    player_x=200.0,
    player_y=250.0,
    player_angle=0.0,
    ball_x=400.0,
    ball_y=250.0,
    enemy_goal_x=None,
    enemy_goal_y=250.0,
):
    if enemy_goal_x is None:
        enemy_goal_x = 800.0 if team == "red" else 0.0
    return StateSnapshot(
        player_x=player_x,
        player_y=player_y,
        player_angle=player_angle,
        player_is_kicking=is_kicking,
        player_team=team,
        ball_x=ball_x,
        ball_y=ball_y,
        enemy_goal_center_x=enemy_goal_x,
        enemy_goal_center_y=enemy_goal_y,
        dist_player_to_ball=dist_pb,
        dist_ball_to_enemy_goal=dist_bg,
        is_in_ball_contact=in_contact,
    )


def _build_real_state(team="red", player_x=200.0, ball_x=400.0, ball_y=250.0):
    state = GameState()
    state.field = Field()
    state.init()
    state.ball = Ball(x=ball_x, y=ball_y)
    p = Player(id=f"{team}_0", team=team, x=player_x, y=250.0, angle=0.0)
    state.players = [p]
    return state, p


def test_make_snapshot_red_enemy_goal_is_right_side():
    state, p = _build_real_state(team="red")
    snap = make_snapshot(p, state)
    assert snap.enemy_goal_center_x == state.field.width
    assert snap.enemy_goal_center_y == state.field.height / 2.0
    assert snap.player_team == "red"


def test_make_snapshot_blue_enemy_goal_is_left_side():
    state, p = _build_real_state(team="blue")
    snap = make_snapshot(p, state)
    assert snap.enemy_goal_center_x == 0.0
    assert snap.enemy_goal_center_y == state.field.height / 2.0
    assert snap.player_team == "blue"


def test_make_snapshot_contact_flag_true_when_inside_threshold():
    state, p = _build_real_state(team="red")
    threshold = p.radius + state.ball.radius
    state.ball.x = p.x + (threshold - 1)
    state.ball.y = p.y
    snap = make_snapshot(p, state)
    assert snap.is_in_ball_contact is True
    assert snap.dist_player_to_ball == pytest.approx(threshold - 1)


def test_make_snapshot_contact_flag_false_when_outside_threshold():
    state, p = _build_real_state(team="red")
    threshold = p.radius + state.ball.radius
    state.ball.x = p.x + (threshold + 1)
    state.ball.y = p.y
    snap = make_snapshot(p, state)
    assert snap.is_in_ball_contact is False


def test_compute_rewards_returns_float_and_three_key_dict():
    snap = _snap()
    total, breakdown = compute_rewards(snap, snap, _state_for_compute())
    assert isinstance(total, float)
    assert isinstance(breakdown, dict)
    assert set(breakdown.keys()) == _BREAKDOWN_KEYS
    assert total == pytest.approx(sum(breakdown.values()))


def test_compute_rewards_raises_on_team_mismatch():
    snap_red = _snap(team="red")
    snap_blue = _snap(team="blue")
    with pytest.raises(ValueError):
        compute_rewards(snap_red, snap_blue, _state_for_compute())


def test_goal_for_player_team():
    snap = _snap(team="red")
    state = _state_for_compute(scored=True, scoring_team="red")
    _, b = compute_rewards(snap, snap, state)
    assert b["goal"] == GOAL_REWARD
    assert b["concede"] == 0.0


def test_concede_when_other_team_scores():
    snap = _snap(team="red")
    state = _state_for_compute(scored=True, scoring_team="blue")
    _, b = compute_rewards(snap, snap, state)
    assert b["goal"] == 0.0
    assert b["concede"] == CONCEDE_PENALTY


def test_no_goal_no_event():
    snap = _snap(team="red")
    state = _state_for_compute(scored=False, scoring_team=None)
    _, b = compute_rewards(snap, snap, state)
    assert b["goal"] == 0.0
    assert b["concede"] == 0.0


@pytest.mark.parametrize(
    "scored, scoring_team",
    [(False, None), (False, "red"), (True, "red"), (True, "blue")],
)
def test_goal_and_concede_mutually_exclusive(scored, scoring_team):
    snap = _snap(team="red")
    state = _state_for_compute(scored=scored, scoring_team=scoring_team)
    _, b = compute_rewards(snap, snap, state)
    assert not (b["goal"] != 0 and b["concede"] != 0), (
        f"goal={b['goal']} e concede={b['concede']} ambos ativos"
    )


def test_kick_on_goal_aligned_far_from_goal_pays_zero():
    # chute alinhado mas longe (proximity²=0) → kick=0.
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(0.0, abs=1e-9)


def test_kick_on_goal_perpendicular_pays_zero():
    snap_p = _snap(team="red", is_kicking=False, player_angle=math.pi / 2)
    snap_c = _snap(team="red", is_kicking=True, player_angle=math.pi / 2)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(0.0, abs=1e-9)


def test_kick_on_goal_backwards_clamped_to_zero():
    snap_p = _snap(team="red", is_kicking=False, player_angle=math.pi)
    snap_c = _snap(team="red", is_kicking=True, player_angle=math.pi)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == 0.0


def test_kick_on_goal_no_edge_during_cooldown():
    # is_kicking continua True por 6 frames após o chute — sem edge, não paga.
    snap_p = _snap(team="red", is_kicking=True, player_angle=0.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == 0.0


def test_kick_on_goal_no_kick_at_all():
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0)
    snap_c = _snap(team="red", is_kicking=False, player_angle=0.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == 0.0


def test_kick_on_goal_for_blue_player_far_pays_zero():
    # blue alinhado mas longe → kick=0.
    snap_p = _snap(team="blue", is_kicking=False, player_x=600.0, player_angle=math.pi)
    snap_c = _snap(team="blue", is_kicking=True, player_x=600.0, player_angle=math.pi)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(0.0, abs=1e-9)


def test_kick_on_goal_for_blue_player_close_pays_full():
    # blue alinhado e perto → kick=BASE.
    snap_p = _snap(team="blue", is_kicking=False, player_x=100.0,
                   player_angle=math.pi, dist_bg=0.0)
    snap_c = _snap(team="blue", is_kicking=True, player_x=100.0,
                   player_angle=math.pi, dist_bg=0.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(KICK_ON_GOAL_BASE)


def test_kick_on_goal_perfect_alignment_at_goal_mouth():
    """cos=1, dist=0 (na boca, proximity=1) → BASE × 1 × 1² = BASE."""
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0, dist_bg=0.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0, dist_bg=0.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(KICK_ON_GOAL_BASE, abs=1e-6)


def test_kick_on_goal_long_shot_aligned_pays_zero():
    """cos=1 mas dist=400 (proximity=0) → kick=0. anti-farming."""
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0, dist_bg=400.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0, dist_bg=400.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(0.0, abs=1e-9)


def test_kick_on_goal_medium_shot_medium_alignment():
    """cos=0.5, dist=180 → cos²=0.25, prox=(1-180/250)=0.28, prox²=0.0784."""
    snap_p = _snap(
        team="red", is_kicking=False, player_angle=math.pi / 3, dist_bg=180.0,
    )
    snap_c = _snap(
        team="red", is_kicking=True, player_angle=math.pi / 3, dist_bg=180.0,
    )
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    proximity = 1.0 - 180.0 / KICK_PROXIMITY_RANGE
    expected = KICK_ON_GOAL_BASE * (0.5 ** 2) * (proximity ** 2)
    assert b["kick_on_goal"] == pytest.approx(expected, abs=1e-5)


def test_kick_on_goal_proximity_clamps_at_zero_far_from_goal():
    """bola além de KICK_PROXIMITY_RANGE → proximity=0 → kick=0."""
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0, dist_bg=500.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0, dist_bg=500.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    assert b["kick_on_goal"] == pytest.approx(0.0, abs=1e-9)


def test_kick_on_goal_v5_quadratic_decay():
    """proximity² decai mais rápido que linear: em proximity=0.5, kick = BASE × 0.25."""
    # dist=125 → proximity = 1 - 125/250 = 0.5.
    snap_p = _snap(team="red", is_kicking=False, player_angle=0.0, dist_bg=125.0)
    snap_c = _snap(team="red", is_kicking=True, player_angle=0.0, dist_bg=125.0)
    _, b = compute_rewards(snap_p, snap_c, _state_for_compute())
    expected = KICK_ON_GOAL_BASE * (0.5 ** 2)
    assert b["kick_on_goal"] == pytest.approx(expected, abs=1e-6)


def test_total_equals_sum_composite_scenario():
    """cenário composto: gol + chute alinhado perto. total = sum(breakdown)."""
    snap_p = _snap(
        team="red", is_kicking=False, in_contact=False,
        dist_pb=20.0, dist_bg=10.0, player_angle=0.0,
    )
    snap_c = _snap(
        team="red", is_kicking=True, in_contact=True,
        dist_pb=20.0, dist_bg=10.0, player_angle=0.0,
    )
    state = _state_for_compute(scored=True, scoring_team="red")
    total, b = compute_rewards(snap_p, snap_c, state)
    assert total == pytest.approx(sum(b.values()), rel=1e-5)
    assert b["goal"] == GOAL_REWARD
    assert b["concede"] == 0.0
    proximity = 1.0 - 10.0 / KICK_PROXIMITY_RANGE
    expected_kick = KICK_ON_GOAL_BASE * 1.0 * (proximity ** 2)
    assert b["kick_on_goal"] == pytest.approx(expected_kick, rel=1e-5)


def test_compute_rewards_under_500us_per_call():
    snap_p = _snap(
        team="red", is_kicking=False, in_contact=False,
        dist_pb=200.0, dist_bg=400.0, player_angle=0.5,
    )
    snap_c = _snap(
        team="red", is_kicking=True, in_contact=True,
        dist_pb=180.0, dist_bg=380.0, player_angle=0.5,
    )
    state = _state_for_compute(scored=False, scoring_team=None)

    best_avg_us = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(1000):
            compute_rewards(snap_p, snap_c, state)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 1000) * 1e6
        best_avg_us = min(best_avg_us, avg_us)

    assert best_avg_us < 500.0, (
        f"compute_rewards muito lento: {best_avg_us:.1f}µs/call (limite 500µs)"
    )
