"""raycasting de percepção dos robôs."""

from __future__ import annotations
import math
from typing import List, Tuple

from backend.config import RAY_ANGLES, MAX_RAY_DISTANCE, TYPE_MAP
from backend.physics.entities import GameState, Player, Wall


def cast_rays(player: Player, state: GameState) -> List[float]:
    """lança N raios e retorna 2N floats (dist normalizada, type) por raio."""
    results = []
    px, py = player.x, player.y

    for ray_offset in RAY_ANGLES:
        ray_angle = player.angle + ray_offset
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        closest_dist = MAX_RAY_DISTANCE
        closest_type = "none"

        for wall in state.walls:
            d = _ray_segment_intersect(px, py, dx, dy, wall)
            if 0 < d < closest_dist:
                closest_dist = d
                closest_type = "wall"

        # postes são detectados como "wall" — não têm tipo próprio em TYPE_MAP.
        for post in state.posts:
            d = _ray_circle_intersect(px, py, dx, dy, post.x, post.y, post.radius)
            if 0 < d < closest_dist:
                closest_dist = d
                closest_type = "wall"

        for goal in state.goals:
            gx = 0.0 if goal.side == "left" else state.field.width
            d = _ray_segment_intersect(
                px, py, dx, dy,
                _make_wall(gx, goal.y_min, gx, goal.y_max)
            )
            if 0 < d < closest_dist:
                closest_dist = d
                if goal.team == player.team:
                    closest_type = "goal_own"
                else:
                    closest_type = "goal_enemy"

        d = _ray_circle_intersect(px, py, dx, dy,
                                   state.ball.x, state.ball.y, state.ball.radius)
        if 0 < d < closest_dist:
            closest_dist = d
            closest_type = "ball"

        for other in state.players:
            if other.id == player.id:
                continue
            d = _ray_circle_intersect(px, py, dx, dy,
                                       other.x, other.y, other.radius)
            if 0 < d < closest_dist:
                closest_dist = d
                if other.team == player.team:
                    closest_type = "ally"
                else:
                    closest_type = "enemy"

        norm_dist = closest_dist / MAX_RAY_DISTANCE
        type_id = TYPE_MAP[closest_type]
        results.append(norm_dist)
        results.append(type_id)

    return results


def _make_wall(x1, y1, x2, y2):
    return type('W', (), {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})()


def _ray_segment_intersect(ox: float, oy: float, dx: float, dy: float, wall) -> float:
    """intersecção raio-segmento. retorna distância ou inf."""
    # raio: P = O + t * D
    # segmento: Q = A + s * (B - A), s in [0, 1]
    ax, ay = wall.x1, wall.y1
    bx, by = wall.x2, wall.y2

    sx = bx - ax
    sy = by - ay

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return float('inf')

    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t > 0 and 0 <= s <= 1:
        return t
    return float('inf')


def _ray_circle_intersect(ox: float, oy: float, dx: float, dy: float,
                           cx: float, cy: float, r: float) -> float:
    """intersecção raio-círculo. retorna distância ou inf."""
    fx = ox - cx
    fy = oy - cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return float('inf')

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return float('inf')
