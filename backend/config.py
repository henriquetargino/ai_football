"""constantes globais do AI Soccer.

física: paridade bit-a-bit obrigatória com `frontend/src/game/physics.js`.
qualquer mudança aqui exige mudança espelhada no JS.
"""

import math


# física

FIELD_WIDTH = 800
FIELD_HEIGHT = 500

PLAYER_RADIUS = 20
PLAYER_MASS = 2.0
MAX_SPEED_PLAYER = 3.5
ACCEL_FORCE = 0.3
ROTATION_SPEED = 0.06
FRICTION_PLAYER = 0.92

BALL_RADIUS = 12
BALL_MASS = 0.3
MAX_SPEED_BALL = 12.0
FRICTION_BALL = 0.985
KICK_FORCE = 7.0
KICK_COOLDOWN = 18
KICK_REACH_TOLERANCE = 1

WALL_BOUNCE = 0.5
ENTITY_BOUNCE = 0.7
GOAL_DEPTH = 40

# raio do arco circular nos cantos do campo. com R=30u, perda de área
# útil é ~0.18% — campo permanece essencialmente retangular, mas a bola
# não fica presa no escanteio.
CORNER_CUT = 30.0

# segmentos de reta que aproximam o arco do canto para o raycast.
# a colisão usa fórmula analítica exata; só o raycast precisa do polígono.
CORNER_ARC_SEGMENTS = 8

POST_RADIUS = 5
# trave é mais dura que parede (WALL_BOUNCE=0.5).
POST_BOUNCE = 0.85

PHYSICS_FPS = 60
REPLAY_FPS = 30


# raycasts: densidade frontal alta (zona de ação), traseira baixa.

NUM_RAYS = 48
MAX_RAY_DISTANCE = 1000

TYPE_MAP = {
    "none": -1.0,
    "wall": 0.0,
    "ball": 0.2,
    "ally": 0.4,
    "enemy": 0.6,
    "goal_own": 0.8,
    "goal_enemy": 1.0,
}


def _build_ray_angles():
    """gera offsets angulares relativos ao heading do robô."""
    angles = []
    for i in range(25):
        a = math.radians(-30 + (60 / 24) * i)
        angles.append(a)
    for i in range(8):
        a = math.radians(-90 + (60 / 8) * i)
        if a < math.radians(-30):
            angles.append(a)
    for i in range(8):
        a = math.radians(30 + (60 / 8) * i)
        if a > math.radians(30):
            angles.append(a)
    for i in range(4):
        a = math.radians(-170 + (80 / 4) * i)
        angles.append(a)
    for i in range(4):
        a = math.radians(90 + (80 / 4) * i)
        angles.append(a)
    angles.sort()
    return angles


RAY_ANGLES = _build_ray_angles()
