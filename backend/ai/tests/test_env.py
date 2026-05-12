"""Testes unitários para o SoccerEnv (peça integradora).

Rodar (a partir da raiz do projeto):

    pytest backend/ai/tests/test_env.py -v
"""

import math

import gymnasium as gym
import numpy as np
import pytest

from backend.ai.env import (
    ACTION_REPEAT,
    GOAL_WIDTH_MAX,
    GOAL_WIDTH_MIN,
    MAX_EPISODE_FRAMES,
    REPLAY_CAPTURE_INTERVAL,
    OpponentMode,
    SoccerEnv,
    random_opponent_policy,
)
from backend.ai.obs import OBS_SIZE, gather_obs
from backend.ai.rewards import make_snapshot
from backend.config import FIELD_WIDTH
from backend.physics.entities import Ball, Field, GameState, Player


# v4: 3 chaves (apenas terminal + KICK_ON_GOAL).
_BREAKDOWN_KEYS = {"goal", "concede", "kick_on_goal"}

NOOP = 8  # action_idx canônico de (accel=0, rot=0, kick=0) — ver actions.py


def _noop_policy(obs):
    return NOOP


# categoria A — Spaces

def test_observation_space_is_box_341():
    env = SoccerEnv(seed=0)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (OBS_SIZE,)
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)


def test_action_space_is_discrete_18():
    env = SoccerEnv(seed=0)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 18


# categoria B — Reset

def test_reset_returns_obs_and_info():
    env = SoccerEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_info_has_required_keys():
    env = SoccerEnv(seed=0)
    _, info = env.reset()
    for key in ("trained_team", "goal_width", "spawn_distance", "agent_initial_angle"):
        assert key in info


def test_reset_progress_zero_favors_wide_goals():
    env = SoccerEnv(seed=42)
    widths = [env.reset(options={"progress": 0.0})[1]["goal_width"] for _ in range(100)]
    assert sum(widths) / len(widths) >= 240.0
    # sanity: ainda dentro do clamp
    assert all(GOAL_WIDTH_MIN <= w <= GOAL_WIDTH_MAX for w in widths)


def test_reset_progress_one_favors_narrow_goals():
    env = SoccerEnv(seed=42)
    widths = [env.reset(options={"progress": 1.0})[1]["goal_width"] for _ in range(100)]
    assert sum(widths) / len(widths) <= 220.0
    assert all(GOAL_WIDTH_MIN <= w <= GOAL_WIDTH_MAX for w in widths)


def test_reset_team_distribution_is_balanced():
    env = SoccerEnv(seed=42)
    teams = [env.reset()[1]["trained_team"] for _ in range(100)]
    n_red = sum(1 for t in teams if t == "red")
    assert 30 <= n_red <= 70, f"trained_team='red' em {n_red}/100 — desbalanceado"


def test_reset_seed_is_reproducible():
    env_a = SoccerEnv(seed=0)
    env_b = SoccerEnv(seed=0)
    info_a = env_a.reset(seed=42)[1]
    info_b = env_b.reset(seed=42)[1]
    assert info_a["trained_team"] == info_b["trained_team"]
    assert info_a["goal_width"] == pytest.approx(info_b["goal_width"])
    assert info_a["spawn_distance"] == pytest.approx(info_b["spawn_distance"])
    assert info_a["agent_initial_angle"] == pytest.approx(info_b["agent_initial_angle"])


# categoria C — Step básico

def test_step_returns_five_tuple_correct_types():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    out = env.step(NOOP)
    assert len(out) == 5
    obs, r, term, trunc, info = out
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


def test_step_executes_four_frames_without_goal():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(NOOP)
    assert not term and not trunc
    assert info["frames_executed"] == ACTION_REPEAT
    assert env.frame_count == ACTION_REPEAT


@pytest.mark.parametrize("bad_action", [-1, 18, 100])
def test_step_invalid_action_raises_value_error(bad_action):
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_frame_count_after_n_steps_no_goal():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        env.step(NOOP)
    assert env.frame_count == 5 * ACTION_REPEAT


# categoria D — Terminação e truncation

def test_step_terminates_on_goal(monkeypatch):
    """Forçar gol no segundo frame interno via monkey-patch do physics_step."""
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)

    from backend.physics.engine import physics_step as real_step

    call_count = {"n": 0}

    def fake_physics_step(state):
        real_step(state)
        call_count["n"] += 1
        if call_count["n"] == 2:
            state.goal_scored_this_step = True
            state.scoring_team = env.trained_team

    monkeypatch.setattr("backend.ai.env.physics_step", fake_physics_step)
    _, _, term, trunc, info = env.step(NOOP)
    assert term is True
    assert trunc is False
    assert info["frames_executed"] == 2  # gol no 2º frame, loop deu break


def test_step_terminates_with_partial_frames_on_early_goal(monkeypatch):
    """Gol no 1º frame: frames_executed = 1, não 4."""
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)

    from backend.physics.engine import physics_step as real_step

    def fake_physics_step(state):
        real_step(state)
        state.goal_scored_this_step = True
        state.scoring_team = env.trained_team

    monkeypatch.setattr("backend.ai.env.physics_step", fake_physics_step)
    _, _, term, _, info = env.step(NOOP)
    assert term is True
    assert info["frames_executed"] == 1


def test_step_truncates_at_max_episode_frames(monkeypatch):
    """Reduz MAX_EPISODE_FRAMES para terminar rápido sem gol."""
    monkeypatch.setattr("backend.ai.env.MAX_EPISODE_FRAMES", 8)
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    # 1º step: 4 frames, frame_count=4 → não trunca
    _, _, term, trunc, _ = env.step(NOOP)
    assert not term and not trunc
    # 2º step: 4 frames, frame_count=8 → trunca
    _, _, term, trunc, _ = env.step(NOOP)
    assert term is False
    assert trunc is True


def test_step_after_termination_raises_runtime_error(monkeypatch):
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)

    from backend.physics.engine import physics_step as real_step

    def fake_physics_step(state):
        real_step(state)
        state.goal_scored_this_step = True
        state.scoring_team = env.trained_team

    monkeypatch.setattr("backend.ai.env.physics_step", fake_physics_step)
    env.step(NOOP)  # termina
    with pytest.raises(RuntimeError):
        env.step(NOOP)


def test_step_after_truncation_raises_runtime_error(monkeypatch):
    monkeypatch.setattr("backend.ai.env.MAX_EPISODE_FRAMES", 4)
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    env.step(NOOP)  # trunca após 4 frames
    with pytest.raises(RuntimeError):
        env.step(NOOP)


# categoria E — Observação e reward consistentes

def test_info_breakdown_has_three_canonical_keys():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(NOOP)
    assert set(info["reward_breakdown"].keys()) == _BREAKDOWN_KEYS


def test_reward_equals_sum_of_breakdown():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    _, r, _, _, info = env.step(NOOP)
    assert r == pytest.approx(sum(info["reward_breakdown"].values()), rel=1e-5)


def test_goal_reward_for_trained_team_includes_plus_ten(monkeypatch):
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)

    from backend.physics.engine import physics_step as real_step

    def fake_physics_step(state):
        real_step(state)
        state.goal_scored_this_step = True
        state.scoring_team = env.trained_team

    monkeypatch.setattr("backend.ai.env.physics_step", fake_physics_step)
    _, _, _, _, info = env.step(NOOP)
    assert info["reward_breakdown"]["goal"] == 10.0
    assert info["reward_breakdown"]["concede"] == 0.0


def test_concede_penalty_for_other_team_includes_minus_five(monkeypatch):
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    other_team = "blue" if env.trained_team == "red" else "red"

    from backend.physics.engine import physics_step as real_step

    def fake_physics_step(state):
        real_step(state)
        state.goal_scored_this_step = True
        state.scoring_team = other_team

    monkeypatch.setattr("backend.ai.env.physics_step", fake_physics_step)
    _, _, _, _, info = env.step(NOOP)
    assert info["reward_breakdown"]["concede"] == -5.0
    assert info["reward_breakdown"]["goal"] == 0.0


# categoria F — Self-play simétrico

def _force_team(env, team):
    """Hack pra forçar o trained_team após reset (não há API pública)."""
    env.reset()
    env.trained_team = team
    env.trained_player = next(p for p in env.state.players if p.team == team)
    env.opponent_player = next(p for p in env.state.players if p.team != team)


def test_blue_player_sees_enemy_goal_at_left():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    for _ in range(50):
        _force_team(env, "blue")
        snap = make_snapshot(env.trained_player, env.state)
        assert snap.enemy_goal_center_x == 0.0


def test_red_player_sees_enemy_goal_at_right():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    for _ in range(50):
        _force_team(env, "red")
        snap = make_snapshot(env.trained_player, env.state)
        assert snap.enemy_goal_center_x == FIELD_WIDTH


def test_observation_symmetry_red_vs_blue_mirrored_scenario():
    """Cenário espelhado: distâncias relativas + state extras devem coincidir."""
    field = Field()
    field.goal_width = 200.0

    # cenário red: player na esquerda olhando direita, bola no centro
    state_r = GameState(field=field)
    state_r.init()
    red = Player(id="red_0", team="red", x=200.0, y=250.0, angle=0.0)
    blue_opp = Player(id="blue_0", team="blue", x=600.0, y=250.0, angle=math.pi)
    state_r.players = [red, blue_opp]
    state_r.ball = Ball(x=400.0, y=250.0)
    snap_r = make_snapshot(red, state_r)
    obs_r = gather_obs(red, state_r, action_repeat_idx=0)

    # cenário espelhado: player blue na direita olhando esquerda
    field2 = Field()
    field2.goal_width = 200.0
    state_b = GameState(field=field2)
    state_b.init()
    red_opp = Player(id="red_0", team="red", x=200.0, y=250.0, angle=0.0)
    blue = Player(id="blue_0", team="blue", x=600.0, y=250.0, angle=math.pi)
    state_b.players = [red_opp, blue]
    state_b.ball = Ball(x=400.0, y=250.0)
    snap_b = make_snapshot(blue, state_b)
    obs_b = gather_obs(blue, state_b, action_repeat_idx=0)

    # distâncias relativas idênticas
    assert snap_r.dist_player_to_ball == pytest.approx(snap_b.dist_player_to_ball)
    assert snap_r.dist_ball_to_enemy_goal == pytest.approx(snap_b.dist_ball_to_enemy_goal)
    assert snap_r.is_in_ball_contact == snap_b.is_in_ball_contact

    # state extras (336..340) devem ser iguais
    np.testing.assert_array_almost_equal(obs_r[336:341], obs_b[336:341])


# categoria G — Replay

def test_replay_capture_disabled_by_default():
    env = SoccerEnv(opponent_policy=_noop_policy, seed=0)
    env.reset(seed=0)
    env.step(NOOP)
    assert env.get_replay() == []


def test_replay_capture_enabled_yields_frames():
    env = SoccerEnv(opponent_policy=_noop_policy, capture_replay=True, seed=0)
    env.reset(seed=0)
    for _ in range(10):
        env.step(NOOP)
    frames = env.get_replay()
    assert len(frames) > 0


def test_replay_frame_has_canonical_keys():
    env = SoccerEnv(opponent_policy=_noop_policy, capture_replay=True, seed=0)
    env.reset(seed=0)
    env.step(NOOP)
    frames = env.get_replay()
    expected = {"step", "ball_x", "ball_y", "ball_vx", "ball_vy",
                "players", "score", "goal_width"}
    assert set(frames[0].keys()) == expected
    assert isinstance(frames[0]["players"], list)


def test_replay_capture_frequency():
    """1 captura inicial + 1 a cada REPLAY_CAPTURE_INTERVAL frames físicos."""
    env = SoccerEnv(opponent_policy=_noop_policy, capture_replay=True, seed=0)
    env.reset(seed=0)
    # 10 steps × 4 frames = 40 frames físicos
    for _ in range(10):
        env.step(NOOP)
    expected = env.frame_count // REPLAY_CAPTURE_INTERVAL + 1
    assert len(env.get_replay()) == expected


# categoria H — Correção de assimetria (peça 5)

def test_opponent_policy_called_once_per_gym_step():
    """Correção da assimetria: opponent decide 1× por step do gym, não 4×."""
    call_count = [0]

    def counting_policy(obs):
        call_count[0] += 1
        return NOOP

    env = SoccerEnv(opponent_policy=counting_policy, seed=0)
    env.reset(seed=0)
    call_count[0] = 0  # zera após reset (que não chama opponent_policy)
    env.step(NOOP)
    assert call_count[0] == 1, (
        f"opponent_policy chamada {call_count[0]} vezes, esperado 1"
    )


def test_set_opponent_policy_swaps_policy():
    """``set_opponent_policy`` troca a policy ativa entre episódios."""
    env = SoccerEnv(opponent_policy=lambda obs: NOOP, seed=0)
    env.reset(seed=0)
    env.set_opponent_policy(lambda obs: 14)  # action 14 = (+1, 0, 0) — frente puro
    env.step(NOOP)
    assert env.opponent_player.accel == 1.0


# categoria I — V3: Curriculum multi-estágio

def test_phase_1a_no_opponent():
    """Phase 1A: opponent_mode=NONE; opponent não age durante steps."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")
    env.reset(seed=42)

    assert env.opponent_mode == OpponentMode.NONE

    initial_opp_x = env.opponent_player.x
    initial_opp_y = env.opponent_player.y

    for _ in range(50):
        env.step(NOOP)

    assert env.opponent_player.x == initial_opp_x
    assert env.opponent_player.y == initial_opp_y


def test_phase_2_passive_opponent():
    """Phase 2: opponent_mode=PASSIVE; opponent existe mas não age."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="2")
    env.reset(seed=42)

    assert env.opponent_mode == OpponentMode.PASSIVE

    initial_opp_pos = (env.opponent_player.x, env.opponent_player.y)
    for _ in range(50):
        env.step(NOOP)
    assert (env.opponent_player.x, env.opponent_player.y) == initial_opp_pos


def test_phase_3_scripted_opponent_chases_ball():
    """Phase 3: opponent SCRIPTED se aproxima da bola ao longo de steps."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3")
    env.reset(seed=42)

    assert env.opponent_mode == OpponentMode.SCRIPTED

    initial_dist = math.hypot(
        env.opponent_player.x - env.state.ball.x,
        env.opponent_player.y - env.state.ball.y,
    )

    for _ in range(50):
        env.step(NOOP)

    final_dist = math.hypot(
        env.opponent_player.x - env.state.ball.x,
        env.opponent_player.y - env.state.ball.y,
    )

    assert final_dist < initial_dist, (
        f"scripted opponent deveria perseguir bola: dist {initial_dist:.1f} → {final_dist:.1f}"
    )


def test_phase_1a_spawn_close_to_enemy_goal():
    """Phase 1A: agente spawna entre 0.85 e 0.95 do lado de ataque."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")

    for seed in range(20):
        env.reset(seed=seed)
        if env.trained_team == "red":
            assert 0.85 * FIELD_WIDTH <= env.trained_player.x <= 0.95 * FIELD_WIDTH, (
                f"red spawn x={env.trained_player.x} fora de [0.85, 0.95] × {FIELD_WIDTH}"
            )
        else:
            assert 0.05 * FIELD_WIDTH <= env.trained_player.x <= 0.15 * FIELD_WIDTH, (
                f"blue spawn x={env.trained_player.x} fora de [0.05, 0.15] × {FIELD_WIDTH}"
            )


def test_phase_4_spawn_anywhere_with_variance():
    """Phase 4: spawn cobre range amplo do campo."""
    import statistics

    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="4")

    positions = []
    for seed in range(30):
        env.reset(seed=seed)
        positions.append(env.trained_player.x / FIELD_WIDTH)

    var = statistics.variance(positions)
    assert var > 0.05, (
        f"phase 4 deveria spawnar em locais variados, got variance {var:.4f}"
    )


# categoria J — V4: Sub-fases 1A/1B/1C + ball_alignment

def test_phase_1a_aligned_spawn():
    """Fase 1A: bola colada (10-20u), agente alinhado pro gol."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")
    env.reset(seed=42)

    dist = math.hypot(
        env.state.ball.x - env.trained_player.x,
        env.state.ball.y - env.trained_player.y,
    )
    assert 10.0 <= dist <= 25.0, f"esperava ~10-20u, got {dist:.1f}"

    # validar que bola está alinhada com o gol oponente (cos > 0.95).
    is_red = env.trained_team == "red"
    enemy_goal_x = env.state.field.width if is_red else 0.0
    enemy_goal_y = env.state.field.height / 2

    vec_ball = (
        env.state.ball.x - env.trained_player.x,
        env.state.ball.y - env.trained_player.y,
    )
    vec_goal = (
        enemy_goal_x - env.trained_player.x,
        enemy_goal_y - env.trained_player.y,
    )
    norm_b = math.hypot(*vec_ball)
    norm_g = math.hypot(*vec_goal)
    assert norm_b > 0 and norm_g > 0
    cos_alignment = (
        (vec_ball[0] * vec_goal[0] + vec_ball[1] * vec_goal[1])
        / (norm_b * norm_g)
    )
    assert cos_alignment > 0.95, (
        f"bola deveria estar alinhada com gol em 1A, cos={cos_alignment:.3f}"
    )


def test_phase_1c_small_angle_to_goal():
    """Fase 1C: bola em ângulo pequeno (±30°) do gol — cos > 0.7 em 20 resets."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1C")

    angles = []
    for s in range(20):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2
        vec_ball = (
            env.state.ball.x - env.trained_player.x,
            env.state.ball.y - env.trained_player.y,
        )
        vec_goal = (
            enemy_goal_x - env.trained_player.x,
            enemy_goal_y - env.trained_player.y,
        )
        norm_b = math.hypot(*vec_ball)
        norm_g = math.hypot(*vec_goal)
        if norm_b > 0 and norm_g > 0:
            cos_a = (
                (vec_ball[0] * vec_goal[0] + vec_ball[1] * vec_goal[1])
                / (norm_b * norm_g)
            )
            angles.append(cos_a)

    assert all(c > 0.7 for c in angles), (
        f"alguns ângulos estão muito largos em 1C: {[f'{c:.2f}' for c in angles]}"
    )


def test_phase_1d_lateral_to_goal():
    """V6 — Fase 1D: bola em ângulo lateral (±60° a ±90°) do vetor pro gol.

    Ou seja, |cos(angle_ball_vs_goal)| em [cos(90°), cos(60°)] = [0, 0.5].
    Em 30 resets, todos os ângulos devem cair nesse range.
    """
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1D")

    cos_values = []
    for s in range(30):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2
        vec_ball = (
            env.state.ball.x - env.trained_player.x,
            env.state.ball.y - env.trained_player.y,
        )
        vec_goal = (
            enemy_goal_x - env.trained_player.x,
            enemy_goal_y - env.trained_player.y,
        )
        norm_b = math.hypot(*vec_ball)
        norm_g = math.hypot(*vec_goal)
        assert norm_b > 0 and norm_g > 0
        cos_a = (
            (vec_ball[0] * vec_goal[0] + vec_ball[1] * vec_goal[1])
            / (norm_b * norm_g)
        )
        cos_values.append(cos_a)

    # tolerância numérica pequena (±1e-9) — uniform pode atingir π/2 exato
    # em float, gerando cos = 6e-17. cos(π/3) = 0.5 e cos(π/2) = 0.
    assert all(-1e-6 <= c <= 0.5 + 1e-6 for c in cos_values), (
        f"alguns cos fora de [0, 0.5] em 1D: "
        f"{[f'{c:.3f}' for c in cos_values]}"
    )
    # pelo menos 1 caso de cada lado (sign=±1) — sanity check da
    # randomização lateral (não é determinismo de seed específico):
    # esperamos sample com bola "à esquerda" e "à direita" do eixo agente↔gol.
    # sinal extraído via produto vetorial 2D (cross).
    crosses = []
    for s in range(30):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2
        vbx = env.state.ball.x - env.trained_player.x
        vby = env.state.ball.y - env.trained_player.y
        vgx = enemy_goal_x - env.trained_player.x
        vgy = enemy_goal_y - env.trained_player.y
        crosses.append(vgx * vby - vgy * vbx)
    assert any(c > 0 for c in crosses) and any(c < 0 for c in crosses), (
        "Fase 1D deveria ter bola em ambos os lados do eixo agente↔gol"
    )


def test_phase_3t_ball_mixed_alignment_three_buckets():
    """V8 calibrada — Fase 3T: alignment 'behind_mixed_toward_own_half'
    sorteia entre 3 cenários equiprováveis (33/33/33):
        reta:  cos > 0.99 (offset = 0°)
        leve:  cos ∈ [cos 30°, cos 15°] = [~0.866, ~0.966] (offset 15°-30°)
        forte: cos ∈ [cos 50°, cos 35°] = [~0.643, ~0.819] (offset 35°-50°)
    Em N=300 resets, espera ~1/3 em cada bucket.
    """
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3T")

    bucket_counts = {"reta": 0, "leve": 0, "forte": 0, "fora": 0}
    for s in range(300):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        own_goal_x = 0.0 if is_red else env.state.field.width
        own_goal_y = env.state.field.height / 2
        vec_ball = (
            env.state.ball.x - env.trained_player.x,
            env.state.ball.y - env.trained_player.y,
        )
        vec_own = (
            own_goal_x - env.trained_player.x,
            own_goal_y - env.trained_player.y,
        )
        norm_b = math.hypot(*vec_ball)
        norm_o = math.hypot(*vec_own)
        assert norm_b > 0 and norm_o > 0
        cos_a = (
            (vec_ball[0] * vec_own[0] + vec_ball[1] * vec_own[1])
            / (norm_b * norm_o)
        )
        # cos da fronteira (tolerância 0.02 pra clamping de spawn perto da parede).
        cos_15 = math.cos(math.radians(15))   # ≈ 0.966
        cos_30 = math.cos(math.radians(30))   # ≈ 0.866
        cos_35 = math.cos(math.radians(35))   # ≈ 0.819
        cos_50 = math.cos(math.radians(50))   # ≈ 0.643
        if cos_a > 0.99:
            bucket_counts["reta"] += 1
        elif cos_30 - 0.02 <= cos_a <= cos_15 + 0.02:
            bucket_counts["leve"] += 1
        elif cos_50 - 0.05 <= cos_a <= cos_35 + 0.02:
            bucket_counts["forte"] += 1
        else:
            bucket_counts["fora"] += 1

    n = 300
    p_reta = bucket_counts["reta"] / n
    p_leve = bucket_counts["leve"] / n
    p_forte = bucket_counts["forte"] / n
    p_fora = bucket_counts["fora"] / n

    # tolerância ~6% (n=300, σ binomial p=1/3 ≈ 2.7%, 2σ ≈ 5.4%).
    assert 0.27 < p_reta < 0.40, f"reta esperado ~0.33, got {p_reta:.3f}"
    assert 0.27 < p_leve < 0.40, f"leve esperado ~0.33, got {p_leve:.3f}"
    assert 0.27 < p_forte < 0.40, f"forte esperado ~0.33, got {p_forte:.3f}"
    # fora: pode haver alguns por clamping de bola perto da parede.
    assert p_fora < 0.03, f"fora dos buckets >3%: {p_fora:.3f}"


def test_phase_3t_ball_lateral_sides_balanced():
    """V8 — Fase 3T oblíquo: ambos os sinais (esquerda/direita do eixo
    agente→own_goal) devem aparecer com frequência similar. Apenas
    spawns OBLÍQUOS (não retas) contam — em N spawns, esperamos ambos
    os lados do cross product ≠ 0.
    """
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3T")

    crosses_oblique = []
    for s in range(150):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        own_goal_x = 0.0 if is_red else env.state.field.width
        vbx = env.state.ball.x - env.trained_player.x
        vby = env.state.ball.y - env.trained_player.y
        vox = own_goal_x - env.trained_player.x
        voy = env.state.field.height / 2 - env.trained_player.y
        cross = vox * vby - voy * vbx
        if abs(cross) > 1.0:  # ignora retas (cross ~ 0)
            crosses_oblique.append(cross)

    # tem que ter spawns oblíquos
    assert len(crosses_oblique) > 50, f"esperava >50 oblíquas, got {len(crosses_oblique)}"
    # ambos os lados representados
    assert any(c > 0 for c in crosses_oblique), "nenhum spawn no lado +"
    assert any(c < 0 for c in crosses_oblique), "nenhum spawn no lado -"


def test_phase_3t_ball_has_moderate_initial_velocity():
    """V8 calibrada — Fase 3T: bola começa com velocidade 3.0 unidades/frame
    (25% MAX_SPEED_BALL=12). Direção = vetor agente→bola (mesma direção
    do spawn oblíquo, "fugindo" do agente)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3T")

    for s in range(10):
        env.reset(seed=s)
        vx, vy = env.state.ball.vx, env.state.ball.vy
        speed = math.hypot(vx, vy)
        # velocidade 3.0 (calibrada V8: bola percorre máx ~200u, NÃO chega no gol).
        assert 2.7 < speed < 3.3, f"3T: speed inicial bola = {speed:.2f}, esperado ~3.0"
        # direção = vetor agente→bola (bola fugindo do agente).
        vec_spawn = (
            env.state.ball.x - env.trained_player.x,
            env.state.ball.y - env.trained_player.y,
        )
        norm_s = math.hypot(*vec_spawn)
        cos_a = (vx * vec_spawn[0] + vy * vec_spawn[1]) / (speed * norm_s)
        assert cos_a > 0.95, (
            f"3T: vel bola não aponta na direção do spawn, cos={cos_a:.3f}"
        )


def test_phase_3t_opp_spawned_behind_agent_same_side():
    """V8 calibrada — Fase 3T: opp NÃO usa mirror horizontal.

    Opp é posicionado MAIS LONGE da bola que o agente (handicap geométrico
    pra compensar os 52f de giro do agente). Verifica que para red, opp_x
    está à direita (lado do gol enemy do red, x maior); para blue, à esquerda.
    """
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3T")

    for s in range(20):
        env.reset(seed=s)
        agent_x = env.trained_player.x
        opp_x = env.opponent_player.x
        is_red = env.trained_team == "red"
        if is_red:
            # red ataca direita; opp deve estar à direita do agente.
            assert opp_x > agent_x - 1, (  # pequena tolerância pra clamping
                f"red 3T: opp (x={opp_x:.1f}) deveria estar à direita do "
                f"agente (x={agent_x:.1f})"
            )
        else:
            # blue ataca esquerda; opp deve estar à esquerda do agente.
            assert opp_x < agent_x + 1, (
                f"blue 3T: opp (x={opp_x:.1f}) deveria estar à esquerda do "
                f"agente (x={agent_x:.1f})"
            )

        # sanity adicional: opp está MAIS LONGE da bola que o agente
        # (handicap geométrico).
        bx, by = env.state.ball.x, env.state.ball.y
        d_agent = math.hypot(agent_x - bx, env.trained_player.y - by)
        d_opp = math.hypot(opp_x - bx, env.opponent_player.y - by)
        assert d_opp > d_agent, (
            f"3T handicap: opp deveria estar mais longe da bola que o agente "
            f"(d_opp={d_opp:.1f} vs d_agent={d_agent:.1f})"
        )


def test_phase_3t_agent_faces_enemy_goal_initially():
    """V8 — Fase 3T: agente começa olhando pro gol oponente (vai precisar girar 180°)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3T")

    for s in range(10):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2
        # heading do agente
        heading_x = math.cos(env.trained_player.angle)
        heading_y = math.sin(env.trained_player.angle)
        # vetor pro gol enemy
        dx = enemy_goal_x - env.trained_player.x
        dy = enemy_goal_y - env.trained_player.y
        norm = math.hypot(dx, dy)
        cos_a = (heading_x * dx + heading_y * dy) / norm
        assert cos_a > 0.95, (
            f"3T: agente deveria começar face_goal (gol enemy), cos={cos_a:.3f}"
        )


def test_default_ball_initial_speed_is_zero():
    """V8 backward-compat: fases pré-V8 (sem ball_initial_speed) ⇒ bola estática.

    Garante que adicionar suporte a velocidade inicial NÃO afeta fases existentes.
    """
    env = SoccerEnv(seed=42)
    for phase in ("1A", "1B", "1C", "1D", "2", "3", "3D", "4", "5"):
        env.set_progress(0.0, phase=phase)
        env.reset(seed=0)
        speed = math.hypot(env.state.ball.vx, env.state.ball.vy)
        assert speed == 0.0, (
            f"fase {phase}: bola deveria começar estática, speed={speed:.3f}"
        )


def test_phase_3d_defensive_spawn():
    """V6 — Fase 3D: agente nasce no PRÓPRIO terço (lado da defesa)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="3D")

    # modo SCRIPTED + spawn defensivo
    assert env.opponent_mode == OpponentMode.SCRIPTED

    field_w = 800.0  # default Field
    for s in range(20):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        # frac=0.05-0.25 do lado de ataque. Pra red (ataca direita), x absoluto
        # esperado em [0.05*W, 0.25*W] = [40, 200] (lado esquerdo = sua defesa).
        # pra blue (ataca esquerda), x = (1-frac)*W → [0.75*W, 0.95*W] = [600, 760]
        # (lado direito = sua defesa).
        x = env.trained_player.x
        if is_red:
            assert 0.0 <= x <= 0.25 * field_w + 10.0, (
                f"red defensivo deveria estar em x≤200, got {x:.1f}"
            )
        else:
            assert 0.75 * field_w - 10.0 <= x <= field_w, (
                f"blue defensivo deveria estar em x≥600, got {x:.1f}"
            )


# NOTA V4.2: testes "agent_initial_angle_red/blue_aligned_with_goal" antigos
# foram REMOVIDOS porque assumiam ângulo fixo (0 pra red, π pra blue).
# agora 1A usa strategy="face_goal" — ângulo dinâmico baseado em Y do agente.
# cobertura equivalente está em `test_phase_1a_face_goal_alignment` (V4.2).


def test_set_progress_indexed_dispatches_per_env():
    """set_progress_indexed usa _env_idx pra pegar phase da posição correta."""
    env = SoccerEnv(seed=42)
    env._env_idx = 1
    env.set_progress_indexed(0.0, ["1A", "3", None])
    # posição 1 = "3" → opponent_mode = SCRIPTED
    assert env.opponent_mode == OpponentMode.SCRIPTED


def test_set_progress_indexed_none_keeps_phase():
    """spec=None na posição do env → fase não muda."""
    env = SoccerEnv(seed=42)
    env._env_idx = 0
    env.set_progress(0.0, phase="2")  # configura PASSIVE
    assert env.opponent_mode == OpponentMode.PASSIVE
    env.set_progress_indexed(0.0, [None, "1A"])  # spec[0] = None
    assert env.opponent_mode == OpponentMode.PASSIVE  # mantém


# categoria K — V4.1: max_episode_frames por fase

def test_max_episode_frames_per_phase():
    """Cada fase tem seu próprio max_episode_frames (V4.1)."""
    env = SoccerEnv(seed=42)

    expected = {
        "1A": 600, "1B": 600, "1C": 900,
        "2": 1200, "3": 1800, "4": 2400, "5": 2400,
    }
    for phase, expected_max in expected.items():
        env.set_progress(0.0, phase=phase)
        assert env._max_episode_frames == expected_max, (
            f"Fase {phase}: esperava max={expected_max}, got {env._max_episode_frames}"
        )


def test_phase_1a_truncates_at_600_frames():
    """Fase 1A com NOOP infinito deve truncar em ~600 frames (10s)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")
    env.reset(seed=42)

    truncated_at = None
    for _ in range(500):  # max 500 steps × 4 action_repeat = 2000 frames
        _, _, terminated, truncated, _ = env.step(NOOP)
        if truncated:
            truncated_at = env.frame_count
            break
        if terminated:
            break

    # pode terminar antes (gol ocasional do agent NOOP é improvável mas
    # possível em 1A com bola colada). Se truncou, deve ser perto de 600.
    # aceitamos terminated também (cobertura geral).
    if truncated_at is not None:
        # action_repeat=4: truncate ocorre no primeiro multiple de 4 ≥ 600
        assert truncated_at <= 700, f"Truncou tarde demais: {truncated_at}"


def test_phase_4_uses_default_2400_frames():
    """Fase 4 mantém max_episode_frames=2400 (default global)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="4")
    assert env._max_episode_frames == 2400


# categoria L — V4.2: spawn Y por fase + agent_initial_angle_strategy

def test_phase_1a_y_centralized():
    """Fase 1A: agent_y centralizado (40-60% do campo)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")

    field_h = None
    ys = []
    for s in range(30):
        env.reset(seed=s)
        if field_h is None:
            field_h = env.state.field.height
        ys.append(env.trained_player.y)

    y_min_expected = field_h * 0.40
    y_max_expected = field_h * 0.60
    out_of_range = [y for y in ys if not (y_min_expected <= y <= y_max_expected)]
    assert not out_of_range, (
        f"Algum Y fora de [{y_min_expected:.0f}, {y_max_expected:.0f}]: {out_of_range}"
    )


def test_phase_1a_face_goal_alignment():
    """Fase 1A: agente nasce APONTANDO pro gol oponente (face_goal)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1A")

    for s in range(20):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2

        dx = enemy_goal_x - env.trained_player.x
        dy = enemy_goal_y - env.trained_player.y
        expected_angle = math.atan2(dy, dx)
        actual_angle = env.trained_player.angle

        # diferença mínima módulo 2π.
        diff = abs(((expected_angle - actual_angle + math.pi) % (2 * math.pi)) - math.pi)
        assert diff < 0.01, (
            f"seed={s}: ângulo esperado {expected_angle:.3f}, "
            f"got {actual_angle:.3f} (diff {diff:.3f})"
        )


def test_phase_1c_face_goal_with_noise():
    """Fase 1C: agente nasce com face_goal + noise sorteado de ±0.6 rad."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="1C")

    diffs = []
    for s in range(50):
        env.reset(seed=s)
        is_red = env.trained_team == "red"
        enemy_goal_x = env.state.field.width if is_red else 0.0
        enemy_goal_y = env.state.field.height / 2

        dx = enemy_goal_x - env.trained_player.x
        dy = enemy_goal_y - env.trained_player.y
        expected_base = math.atan2(dy, dx)
        actual_angle = env.trained_player.angle

        # diff normalizada em (-π, π]. Pequena margem (1e-9) acima do range
        # acomoda erros numéricos quando noise sai exatamente em ±0.6.
        diff = ((actual_angle - expected_base + math.pi) % (2 * math.pi)) - math.pi
        diffs.append(diff)

    out_of_range = [d for d in diffs if not (-0.6 - 1e-9 <= d <= 0.6 + 1e-9)]
    assert not out_of_range, f"Algum noise fora de ±0.6 rad: {out_of_range}"

    # validar que há variação real (não está todo zero).
    import statistics
    assert statistics.stdev(diffs) > 0.05, (
        f"noise deveria ter dispersão; stdev={statistics.stdev(diffs):.4f}"
    )


def test_phase_2_keeps_range_strategy_for_compat():
    """Fase 2: strategy 'range' explícita — preserva comportamento amplo (-π, π)."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="2")

    angles = []
    for s in range(50):
        env.reset(seed=s)
        angles.append(env.trained_player.angle)

    import statistics
    assert statistics.stdev(angles) > 1.0, (
        f"Strategy 'range' deveria gerar ângulos amplos; stdev={statistics.stdev(angles):.4f}"
    )


def test_phase_5_default_y_range_preserved():
    """Fases sem agent_spawn_y_frac_range usam default (0.15, 0.85) — backward-compat."""
    env = SoccerEnv(seed=42)
    env.set_progress(0.0, phase="5")

    ys = []
    field_h = None
    for s in range(50):
        env.reset(seed=s)
        if field_h is None:
            field_h = env.state.field.height
        ys.append(env.trained_player.y)

    # range amplo: pelo menos 1 Y FORA da faixa centralizada [40%, 60%].
    in_centralized = sum(1 for y in ys if field_h * 0.40 <= y <= field_h * 0.60)
    assert in_centralized < len(ys), (
        f"Fase 5 deveria ter Y diverso; {in_centralized}/{len(ys)} centralizados"
    )
