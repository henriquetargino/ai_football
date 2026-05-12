"""dataclasses das entidades do jogo."""

from __future__ import annotations
import math
from dataclasses import dataclass, field as datafield
from typing import List, Optional

from backend.config import POST_RADIUS, CORNER_CUT, CORNER_ARC_SEGMENTS


@dataclass
class Ball:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 12.0
    mass: float = 0.3
    id: str = "ball"

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


@dataclass
class Player:
    id: str = ""
    team: str = "red"
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0
    radius: float = 20.0
    mass: float = 2.0
    # controles (setados pela rede ou humano).
    accel: float = 0.0
    rot: float = 0.0
    kick_requested: bool = False
    # estado de chute.
    can_kick: bool = True
    kick_cooldown_timer: int = 0
    is_kicking: bool = False
    kick_attempted: bool = False
    # referência à rede (setada externamente).
    network: object = None

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


@dataclass
class Goal:
    """abertura na parede que conta como gol."""
    team: str = "red"          # time que DEFENDE este gol.
    side: str = "left"         # "left" (x=0) ou "right" (x=FIELD_WIDTH).
    y_min: float = 0.0
    y_max: float = 0.0


@dataclass
class Wall:
    """segmento de parede (linha reta)."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass
class Post:
    """trave — colisor estático circular nas bordas da abertura do gol.

    são quatro no total: dois por gol. colisão é tratada como círculo
    estático (massa infinita) — apenas a entidade que colide é deslocada
    e rebate.
    """
    x: float = 0.0
    y: float = 0.0
    radius: float = POST_RADIUS


@dataclass
class Field:
    width: float = 800.0
    height: float = 500.0
    goal_width: float = 150.0

    def build_goals(self) -> List[Goal]:
        mid_y = self.height / 2
        half_gw = self.goal_width / 2
        return [
            Goal(team="red",  side="left",  y_min=mid_y - half_gw, y_max=mid_y + half_gw),
            Goal(team="blue", side="right", y_min=mid_y - half_gw, y_max=mid_y + half_gw),
        ]

    def build_walls(self) -> List[Wall]:
        """segmentos de parede do campo (com aberturas de gol + arcos nos cantos).

        paredes axis-aligned ficam exatamente nas bordas do campo lógico.
        os 4 cantos são arredondados por arcos de quadrante de raio CORNER_CUT,
        aproximados no raycast por CORNER_ARC_SEGMENTS segmentos pequenos
        (a colisão usa fórmula analítica de círculo — ver collision.py).
        """
        mid_y = self.height / 2
        half_gw = self.goal_width / 2
        cut = CORNER_CUT
        walls = []
        walls.append(Wall(0, cut, 0, mid_y - half_gw))
        walls.append(Wall(0, mid_y + half_gw, 0, self.height - cut))
        walls.append(Wall(self.width, cut, self.width, mid_y - half_gw))
        walls.append(Wall(self.width, mid_y + half_gw, self.width, self.height - cut))
        walls.append(Wall(cut, 0, self.width - cut, 0))
        walls.append(Wall(cut, self.height, self.width - cut, self.height))
        # 4 arcos arredondados nos cantos — aproximados por N segmentos.
        # centros e ângulos (θ=0 é +X, θ cresce anti-horário):
        #   NW: (cut, cut)               θ ∈ [π, 3π/2]
        #   NE: (width-cut, cut)         θ ∈ [3π/2, 2π]
        #   SE: (width-cut, height-cut)  θ ∈ [0, π/2]
        #   SW: (cut, height-cut)        θ ∈ [π/2, π]
        N = CORNER_ARC_SEGMENTS
        arcs = [
            (cut,                cut,                 math.pi,           3 * math.pi / 2),
            (self.width - cut,   cut,                 3 * math.pi / 2,   2 * math.pi),
            (self.width - cut,   self.height - cut,   0.0,               math.pi / 2),
            (cut,                self.height - cut,   math.pi / 2,       math.pi),
        ]
        for cx, cy, t1, t2 in arcs:
            for i in range(N):
                a1 = t1 + (t2 - t1) * (i / N)
                a2 = t1 + (t2 - t1) * ((i + 1) / N)
                x1 = cx + cut * math.cos(a1)
                y1 = cy + cut * math.sin(a1)
                x2 = cx + cut * math.cos(a2)
                y2 = cy + cut * math.sin(a2)
                walls.append(Wall(x1, y1, x2, y2))
        return walls

    def build_posts(self) -> List[Post]:
        """4 postes circulares (2 por gol) nas bordas das aberturas.

        coordenadas exatas (0, y_min/y_max) e (width, y_min/y_max) coincidem
        com as pontas dos segmentos de parede adjacentes ao gol.
        """
        mid_y = self.height / 2
        half_gw = self.goal_width / 2
        return [
            Post(x=0.0,         y=mid_y - half_gw),
            Post(x=0.0,         y=mid_y + half_gw),
            Post(x=self.width,  y=mid_y - half_gw),
            Post(x=self.width,  y=mid_y + half_gw),
        ]


@dataclass
class GameState:
    """estado completo de uma partida."""
    field: Field = datafield(default_factory=Field)
    ball: Ball = datafield(default_factory=Ball)
    players: List[Player] = datafield(default_factory=list)
    goals: List[Goal] = datafield(default_factory=list)
    walls: List[Wall] = datafield(default_factory=list)
    posts: List[Post] = datafield(default_factory=list)
    score: dict = datafield(default_factory=lambda: {"red": 0, "blue": 0})
    step_count: int = 0
    goal_scored_this_step: bool = False
    scoring_team: Optional[str] = None
    last_kicker_team: Optional[str] = None

    def init(self):
        """inicializa gols, paredes e postes a partir do field."""
        self.goals = self.field.build_goals()
        self.walls = self.field.build_walls()
        self.posts = self.field.build_posts()
