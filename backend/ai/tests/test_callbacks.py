"""testes unitários dos trackers de callbacks.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_callbacks.py -v
"""

import pytest

from backend.ai.callbacks import BREAKDOWN_KEYS, EnvParamTracker, RewardBreakdownTracker


def _zero_breakdown():
    return {k: 0.0 for k in BREAKDOWN_KEYS}


def _make_breakdown(**overrides):
    bd = _zero_breakdown()
    bd.update(overrides)
    return bd


def test_breakdown_tracker_finished_episode_has_six_keys_plus_length():
    tracker = RewardBreakdownTracker(num_envs=1)
    finished = tracker.step([_make_breakdown(goal=10.0, pbrs_approach=0.001)], [True])
    assert len(finished) == 1
    ep = finished[0]
    for k in BREAKDOWN_KEYS:
        assert k in ep
    assert ep["episode_length"] == 1
    assert ep["goal"] == 10.0


def test_breakdown_tracker_resets_after_done():
    tracker = RewardBreakdownTracker(num_envs=1)
    tracker.step([_make_breakdown(kick_on_goal=0.2)], [False])
    tracker.step([_make_breakdown(kick_on_goal=0.2)], [True])
    finished = tracker.step([_make_breakdown(kick_on_goal=0.2)], [False])
    assert finished == []
    finished2 = tracker.step([_make_breakdown(kick_on_goal=0.2)], [True])
    assert len(finished2) == 1
    # acumulado do segundo episódio = 2 × 0.2 (não 4 × 0.2).
    assert finished2[0]["kick_on_goal"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "scenario, dones_sequence, expected_finished_count",
    [
        ("gol_imediato", [True], 1),
        ("em_curso_5_steps", [False, False, False, False, False], 0),
        ("concede", [True], 1),
        ("timeout_apos_3_steps", [False, False, True], 1),
    ],
)
def test_breakdown_tracker_scenarios(scenario, dones_sequence, expected_finished_count):
    tracker = RewardBreakdownTracker(num_envs=1)
    total_finished = 0
    for done in dones_sequence:
        finished = tracker.step([_zero_breakdown()], [done])
        total_finished += len(finished)
    assert total_finished == expected_finished_count, scenario


def test_breakdown_tracker_multi_env_independent():
    tracker = RewardBreakdownTracker(num_envs=3)
    bds = [
        _make_breakdown(goal=10.0),
        _make_breakdown(concede=-5.0),
        _make_breakdown(touch_ball=0.2),
    ]
    finished = tracker.step(bds, [True, False, True])
    assert len(finished) == 2
    # env 0 e env 2 terminaram; env 1 em curso. ordem não é garantida.
    goals = sorted([ep["goal"] for ep in finished])
    assert goals == [0.0, 10.0]


def test_breakdown_tracker_raises_on_size_mismatch():
    tracker = RewardBreakdownTracker(num_envs=2)
    with pytest.raises(ValueError):
        tracker.step([_zero_breakdown()], [False, False])
    with pytest.raises(ValueError):
        tracker.step([_zero_breakdown()] * 2, [False])


def test_breakdown_tracker_zero_envs_raises():
    with pytest.raises(ValueError):
        RewardBreakdownTracker(num_envs=0)


def test_env_param_tracker_collects_basic_summary():
    tracker = EnvParamTracker()
    tracker.add_episode({
        "trained_team": "red",
        "goal_width": 200.0,
        "spawn_distance": 100.0,
        "agent_initial_angle": 0.5,
    })
    tracker.add_episode({
        "trained_team": "blue",
        "goal_width": 220.0,
        "spawn_distance": 150.0,
        "agent_initial_angle": -0.3,
    })
    summary = tracker.to_summary()
    assert summary["history_size"] == 2.0
    assert summary["goal_width_mean"] == pytest.approx(210.0)
    assert summary["goal_width_min"] == 200.0
    assert summary["goal_width_max"] == 220.0
    assert summary["spawn_distance_mean"] == pytest.approx(125.0)
    assert summary["trained_team_red_ratio"] == 0.5


def test_env_param_tracker_respects_max_history():
    tracker = EnvParamTracker(max_history=3)
    for w in [180, 200, 220, 240, 260]:
        tracker.add_episode({"goal_width": float(w)})
    summary = tracker.to_summary()
    assert summary["history_size"] == 3.0
    # apenas as 3 últimas (220, 240, 260) → mean=240.
    assert summary["goal_width_mean"] == pytest.approx(240.0)


def test_env_param_tracker_handles_partial_info():
    tracker = EnvParamTracker()
    tracker.add_episode({"goal_width": 200.0})
    summary = tracker.to_summary()
    assert "goal_width_mean" in summary
    assert "spawn_distance_mean" not in summary


def test_env_param_tracker_zero_history_raises():
    with pytest.raises(ValueError):
        EnvParamTracker(max_history=0)
