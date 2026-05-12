"""loop de física principal."""

from __future__ import annotations
import math
import random
from typing import List, Optional

from backend.config import (
    FIELD_WIDTH, FIELD_HEIGHT,
    ACCEL_FORCE, ROTATION_SPEED, FRICTION_PLAYER, MAX_SPEED_PLAYER,
    FRICTION_BALL, MAX_SPEED_BALL,
    KICK_FORCE, KICK_COOLDOWN, KICK_REACH_TOLERANCE,
    BALL_RADIUS, PLAYER_RADIUS, PLAYER_MASS, BALL_MASS,
)
from backend.physics.entities import GameState, Player, Ball, Field
from backend.physics.collision import resolve_all_collisions


def create_game_state(mode: str = "1v1", goal_width: float = 150) -> GameState:
    """GameState completo pronto pra uma partida."""
    field = Field(width=FIELD_WIDTH, height=FIELD_HEIGHT, goal_width=goal_width)
    ball = Ball(
        x=FIELD_WIDTH / 2, y=FIELD_HEIGHT / 2,
        radius=BALL_RADIUS, mass=BALL_MASS, id="ball"
    )

    players = []
    if mode == "1v1":
        players = [
            Player(id="red_0",  team="red",  radius=PLAYER_RADIUS, mass=PLAYER_MASS),
            Player(id="blue_0", team="blue", radius=PLAYER_RADIUS, mass=PLAYER_MASS),
        ]
    elif mode == "2v2":
        players = [
            Player(id="red_0",  team="red",  radius=PLAYER_RADIUS, mass=PLAYER_MASS),
            Player(id="red_1",  team="red",  radius=PLAYER_RADIUS, mass=PLAYER_MASS),
            Player(id="blue_0", team="blue", radius=PLAYER_RADIUS, mass=PLAYER_MASS),
            Player(id="blue_1", team="blue", radius=PLAYER_RADIUS, mass=PLAYER_MASS),
        ]

    state = GameState(field=field, ball=ball, players=players)
    state.init()
    randomize_spawns(state)
    return state


def randomize_spawns(state: GameState):
    """spawn aleatório de jogadores e bola.

    usado em treino: a IA precisa aprender a lidar com qualquer situação
    inicial, então toda partida começa com posições/ângulos sorteados.
    """
    mid_x = state.field.width / 2
    margin = 50

    for player in state.players:
        if player.team == "red":
            player.x = random.uniform(margin, mid_x - margin)
        else:
            player.x = random.uniform(mid_x + margin, state.field.width - margin)
        player.y = random.uniform(margin, state.field.height - margin)
        player.angle = random.uniform(0, 2 * math.pi)
        player.vx = 0.0
        player.vy = 0.0

    state.ball.x = mid_x + random.uniform(-20, 20)
    state.ball.y = state.field.height / 2 + random.uniform(-20, 20)
    state.ball.vx = 0.0
    state.ball.vy = 0.0


def spawn_deterministic(state: GameState):
    """spawn fixo e simétrico, usado no modo live do frontend (demonstração).

    a IA já foi treinada em variedade; no replay/demo queremos ver o que
    ela aprendeu, não testá-la em edge cases. red a 1/4 do campo da esquerda,
    blue a 1/4 da direita, ambos no meio vertical olhando pro gol adversário.
    """
    mid_x = state.field.width / 2
    mid_y = state.field.height / 2
    quarter_x = state.field.width / 4

    for player in state.players:
        if player.team == "red":
            player.x = quarter_x
            player.angle = 0.0
        else:
            player.x = state.field.width - quarter_x
            player.angle = math.pi
        player.y = mid_y
        player.vx = 0.0
        player.vy = 0.0

    state.ball.x = mid_x
    state.ball.y = mid_y
    state.ball.vx = 0.0
    state.ball.vy = 0.0


def apply_controls(player: Player, outputs: List[float]):
    """aplica os 3 outputs da rede ao jogador."""
    player.accel = max(-1.0, min(1.0, outputs[0]))
    player.rot = max(-1.0, min(1.0, outputs[1]))
    player.kick_requested = outputs[2] > 0.5


def physics_step(state: GameState):
    """executa um step de física completo."""
    state.goal_scored_this_step = False
    state.scoring_team = None

    for player in state.players:
        player.vx += math.cos(player.angle) * player.accel * ACCEL_FORCE
        player.vy += math.sin(player.angle) * player.accel * ACCEL_FORCE
        player.angle += player.rot * ROTATION_SPEED
        player.angle = player.angle % (2 * math.pi)
        player.vx *= FRICTION_PLAYER
        player.vy *= FRICTION_PLAYER
        _clamp_speed(player, MAX_SPEED_PLAYER)

    state.ball.vx *= FRICTION_BALL
    state.ball.vy *= FRICTION_BALL
    _clamp_speed(state.ball, MAX_SPEED_BALL)

    for player in state.players:
        if player.kick_cooldown_timer > 0:
            player.kick_cooldown_timer -= 1
            if player.kick_cooldown_timer == 0:
                player.can_kick = True
            # is_kicking fica True por 6 frames (0.1s) após o chute.
            if player.kick_cooldown_timer <= KICK_COOLDOWN - 6:
                player.is_kicking = False
        player.kick_attempted = player.kick_requested and player.can_kick
        if player.kick_requested:
            _try_kick(player, state.ball, state)

    for player in state.players:
        player.x += player.vx
        player.y += player.vy
    state.ball.x += state.ball.vx
    state.ball.y += state.ball.vy

    resolve_all_collisions(state)

    _check_goal(state)

    state.step_count += 1


def _clamp_speed(entity, max_speed: float):
    speed = math.sqrt(entity.vx ** 2 + entity.vy ** 2)
    if speed > max_speed:
        factor = max_speed / speed
        entity.vx *= factor
        entity.vy *= factor


def _try_kick(player: Player, ball: Ball, state: GameState):
    if not player.can_kick:
        return
    dx = ball.x - player.x
    dy = ball.y - player.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > player.radius + ball.radius + KICK_REACH_TOLERANCE:
        return

    ball.vx += math.cos(player.angle) * KICK_FORCE
    ball.vy += math.sin(player.angle) * KICK_FORCE
    _clamp_speed(ball, MAX_SPEED_BALL)

    player.can_kick = False
    player.kick_cooldown_timer = KICK_COOLDOWN
    player.is_kicking = True
    state.last_kicker_team = player.team


def _check_goal(state: GameState):
    """verifica se a bola entrou em algum gol.

    regra: a bola precisa ter passado INTEIRA da linha do gol (regra real
    do futebol). pro gol da esquerda: a borda direita da bola precisa
    estar em x < 0. pro gol da direita: a borda esquerda da bola precisa
    estar em x > field.width.
    """
    bx, by = state.ball.x, state.ball.y

    for goal in state.goals:
        if goal.y_min <= by <= goal.y_max:
            if goal.side == "left" and bx + state.ball.radius < 0:
                state.score["blue"] += 1
                state.goal_scored_this_step = True
                state.scoring_team = "blue"
                _reset_after_goal(state)
                return
            elif goal.side == "right" and bx - state.ball.radius > state.field.width:
                state.score["red"] += 1
                state.goal_scored_this_step = True
                state.scoring_team = "red"
                _reset_after_goal(state)
                return


def _reset_after_goal(state: GameState):
    """reposiciona entidades após um gol."""
    randomize_spawns(state)
    for p in state.players:
        p.can_kick = True
        p.kick_cooldown_timer = 0
        p.is_kicking = False
        p.kick_attempted = False
    state.last_kicker_team = None
