"""sistema de colisão simétrica com transferência de momentum."""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

from backend.config import (
    FIELD_WIDTH, FIELD_HEIGHT, WALL_BOUNCE, ENTITY_BOUNCE, GOAL_DEPTH,
    POST_BOUNCE, CORNER_CUT,
)
from backend.physics.entities import GameState, Player, Ball


def resolve_all_collisions(state: GameState):
    """coleta todas as colisões, calcula respostas e aplica simultaneamente."""
    entities = _get_all_entities(state)
    corrections: Dict[str, List[float]] = {e.id: [0.0, 0.0, 0.0, 0.0] for e in entities}

    # fase 1: detectar e calcular.
    pairs = _get_colliding_pairs(state)
    for a, b in pairs:
        _resolve_circle_circle(a, b, corrections)
    for entity in entities:
        _resolve_wall_collisions(entity, state, corrections)
    # escanteios arredondados (arcos circulares) — depois das paredes axis-aligned.
    if CORNER_CUT > 0.0:
        for entity in entities:
            _resolve_corner_arc_collisions(entity, state, corrections)
    # postes depois das paredes — ordem importa pra paridade com JS.
    for entity in entities:
        _resolve_post_collisions(entity, state, corrections)

    # fase 2: aplicar simultaneamente.
    for entity in entities:
        dx, dy, dvx, dvy = corrections[entity.id]
        entity.x += dx
        entity.y += dy
        entity.vx += dvx
        entity.vy += dvy


def _get_all_entities(state: GameState) -> list:
    return [state.ball] + state.players


def _get_colliding_pairs(state: GameState) -> List[Tuple]:
    entities = _get_all_entities(state)
    pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            a, b = entities[i], entities[j]
            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < a.radius + b.radius and dist > 0:
                pairs.append((a, b))
    return pairs


def _resolve_circle_circle(a, b, corrections: Dict[str, List[float]]):
    """colisão elástica com transferência de momentum entre dois círculos."""
    dx = b.x - a.x
    dy = b.y - a.y
    dist = math.sqrt(dx * dx + dy * dy)
    min_dist = a.radius + b.radius

    if dist >= min_dist or dist == 0:
        return

    nx = dx / dist
    ny = dy / dist
    overlap = min_dist - dist
    total_mass = a.mass + b.mass
    a_ratio = b.mass / total_mass
    b_ratio = a.mass / total_mass

    corrections[a.id][0] -= nx * overlap * a_ratio
    corrections[a.id][1] -= ny * overlap * a_ratio
    corrections[b.id][0] += nx * overlap * b_ratio
    corrections[b.id][1] += ny * overlap * b_ratio

    dvx = a.vx - b.vx
    dvy = a.vy - b.vy
    dvn = dvx * nx + dvy * ny
    if dvn > 0:
        # objetos se separando mas ainda sobrepostos (contato sustentado).
        # impulso proporcional ao overlap evita efeito "imã".
        sep_force = overlap * 1.5
        corrections[a.id][2] -= nx * sep_force * a_ratio
        corrections[a.id][3] -= ny * sep_force * a_ratio
        corrections[b.id][2] += nx * sep_force * b_ratio
        corrections[b.id][3] += ny * sep_force * b_ratio
        return

    e = ENTITY_BOUNCE
    j = -(1 + e) * dvn / (1 / a.mass + 1 / b.mass)

    corrections[a.id][2] += (j / a.mass) * nx
    corrections[a.id][3] += (j / a.mass) * ny
    corrections[b.id][2] -= (j / b.mass) * nx
    corrections[b.id][3] -= (j / b.mass) * ny


def _is_in_goal_opening(y: float, side: str, state: GameState) -> bool:
    for goal in state.goals:
        if goal.side == side:
            return goal.y_min <= y <= goal.y_max
    return False


def _is_in_goal_opening_strict(entity, side: str, state: GameState) -> bool:
    """variante pra jogadores: exige que o jogador inteiro caiba na abertura."""
    for goal in state.goals:
        if goal.side == side:
            return (goal.y_min + entity.radius) <= entity.y <= (goal.y_max - entity.radius)
    return False


def _resolve_wall_collisions(entity, state: GameState, corrections: Dict[str, List[float]]):
    """colisão com as 4 paredes axis-aligned (com aberturas pros gols).

    cada parede só atua na sua FAIXA axis-aligned (entre os arcos dos cantos).
    as zonas dos cantos (x<CORNER_CUT, etc.) são tratadas apenas por
    _resolve_corner_arc_collisions. sem isso, a bola perto do canto seria
    empurrada duas vezes (axis + arco).
    """
    r = entity.radius
    eid = entity.id
    fw = state.field.width
    fh = state.field.height
    cut = CORNER_CUT

    if entity.x - r < 0:
        if entity.id == "ball":
            can_pass = _is_in_goal_opening(entity.y, "left", state)
        else:
            can_pass = _is_in_goal_opening_strict(entity, "left", state)
        inside_goal_l = entity.x < 0 and entity.x > -GOAL_DEPTH
        in_corner_zone = entity.y < cut or entity.y > fh - cut
        if not can_pass and not inside_goal_l and not in_corner_zone:
            corrections[eid][0] += r - entity.x
            if entity.vx < 0:
                corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE)

    if entity.x + r > fw:
        if entity.id == "ball":
            can_pass = _is_in_goal_opening(entity.y, "right", state)
        else:
            can_pass = _is_in_goal_opening_strict(entity, "right", state)
        inside_goal_r = entity.x > fw and entity.x < fw + GOAL_DEPTH
        in_corner_zone = entity.y < cut or entity.y > fh - cut
        if not can_pass and not inside_goal_r and not in_corner_zone:
            corrections[eid][0] -= (entity.x + r) - fw
            if entity.vx > 0:
                corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE)

    if entity.y - r < 0:
        in_corner_zone = entity.x < cut or entity.x > fw - cut
        if not in_corner_zone:
            corrections[eid][1] += r - entity.y
            if entity.vy < 0:
                corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE)

    if entity.y + r > fh:
        in_corner_zone = entity.x < cut or entity.x > fw - cut
        if not in_corner_zone:
            corrections[eid][1] -= (entity.y + r) - fh
            if entity.vy > 0:
                corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE)

    # paredes internas do gol (apenas pra jogadores, não pra bola).
    if entity.id != "ball":
        for goal in state.goals:
            gx = 0.0 if goal.side == "left" else fw
            behind_goal = (goal.side == "left" and entity.x < gx) or \
                          (goal.side == "right" and entity.x > gx)

            if behind_goal:
                if entity.y - r < goal.y_min:
                    corrections[eid][1] += goal.y_min + r - entity.y
                    if entity.vy < 0:
                        corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE)

                if entity.y + r > goal.y_max:
                    corrections[eid][1] -= entity.y + r - goal.y_max
                    if entity.vy > 0:
                        corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE)

                if goal.side == "left" and entity.x - r < gx - GOAL_DEPTH:
                    corrections[eid][0] += (gx - GOAL_DEPTH + r) - entity.x
                    if entity.vx < 0:
                        corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE)
                elif goal.side == "right" and entity.x + r > gx + GOAL_DEPTH:
                    corrections[eid][0] -= entity.x + r - (gx + GOAL_DEPTH)
                    if entity.vx > 0:
                        corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE)


def _resolve_post_collisions(entity, state: GameState, corrections: Dict[str, List[float]]):
    """colisão círculo-círculo contra os postes (colisores estáticos).

    o poste tem massa infinita — apenas a entidade é deslocada e o impulso
    normal é refletido com coeficiente POST_BOUNCE.
    """
    r = entity.radius
    eid = entity.id
    for post in state.posts:
        dx = entity.x - post.x
        dy = entity.y - post.y
        dist = math.sqrt(dx * dx + dy * dy)
        min_dist = r + post.radius
        if dist >= min_dist or dist == 0:
            continue
        nx = dx / dist
        ny = dy / dist
        overlap = min_dist - dist
        corrections[eid][0] += nx * overlap
        corrections[eid][1] += ny * overlap
        vn = entity.vx * nx + entity.vy * ny
        if vn < 0:
            corrections[eid][2] += -vn * (1 + POST_BOUNCE) * nx
            corrections[eid][3] += -vn * (1 + POST_BOUNCE) * ny


def _resolve_corner_arc_collisions(entity, state: GameState, corrections: Dict[str, List[float]]):
    """colisão com os 4 arcos arredondados nos cantos do campo.

    cada canto é um quadrante de círculo de raio CORNER_CUT centrado a
    CORNER_CUT unidades pra dentro do canto. a área jogável é o INTERIOR
    do círculo — entity dentro do campo quando dist(entity, centro) ≤ R.
    com entity de raio r, parede efetiva começa quando dist + r > R.
    """
    r = entity.radius
    eid = entity.id
    fw = state.field.width
    fh = state.field.height
    cut = CORNER_CUT

    if entity.x < cut and entity.y < cut:
        cx, cy = cut, cut
    elif entity.x > fw - cut and entity.y < cut:
        cx, cy = fw - cut, cut
    elif entity.x > fw - cut and entity.y > fh - cut:
        cx, cy = fw - cut, fh - cut
    elif entity.x < cut and entity.y > fh - cut:
        cx, cy = cut, fh - cut
    else:
        return

    # vetor radial: do centro pra entity (apontando radialmente pra fora).
    dx = entity.x - cx
    dy = entity.y - cy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist == 0:
        return

    overlap = dist + r - cut
    if overlap <= 0:
        return

    nx = dx / dist
    ny = dy / dist

    # empurra entity pra dentro (na direção -nx, -ny).
    corrections[eid][0] -= nx * overlap
    corrections[eid][1] -= ny * overlap

    # reflete velocidade radial (componente saindo do centro).
    vn = entity.vx * nx + entity.vy * ny
    if vn > 0:
        corrections[eid][2] -= vn * (1 + WALL_BOUNCE) * nx
        corrections[eid][3] -= vn * (1 + WALL_BOUNCE) * ny
