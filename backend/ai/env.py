"""Gymnasium env do AI Soccer 1v1.

junta as peças da camada de IA numa interface gym padrão:
    actions.apply_discrete_action  →  decodifica idx em (accel, rot, kick)
    obs.gather_obs                 →  vetor 341D do trained_player
    rewards.make_snapshot          →  snapshot imutável antes/depois do step
    rewards.compute_rewards        →  reward + breakdown por componente
    physics.physics_step           →  física do mundo

decisões cravadas:
    1 step do gym = ACTION_REPEAT (4) frames físicos.
    episódio: até MAX_EPISODE_FRAMES (2400 = 40s @ 60fps) ou gol.
    self-play: uma única rede; trained_team sorteado por reset (50/50).
    domain randomization: goal_width via TruncatedNormal com μ migrando
    pelo curriculum, spawn_distance e agent_initial_angle uniformes,
    aplicados no reset.

simetria — uma rede treina pros dois lados:
    a obs já é relativa ao heading do player; make_snapshot calcula o gol
    inimigo pelo team. opponent_player é controlado por opponent_policy
    (callable injetado), que NUNCA recebe gradients — policy congelada.

episódio termina ao marcar gol: o loop de action repeat dá break imediato
em goal_scored_this_step; o caller (PPO) recebe terminated=True e zera o
bootstrap V(next_obs).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import gymnasium as gym
import numpy as np

from backend.ai.actions import ACTION_SPACE_SIZE, apply_discrete_action, encode_action
from backend.ai.obs import OBS_SIZE, gather_obs
from backend.ai.rewards import compute_rewards, make_snapshot
from backend.physics.engine import physics_step
from backend.physics.entities import Ball, Field, GameState, Player


# spec serializável de oponente — strings/tuples atravessam pipes de
# multiprocessing; callables não. workers AsyncVectorEnv reconstroem a
# callable localmente a partir do spec + Policy carregada do disco.
OpponentSpec = Union[
    Literal["random"],
    Literal["latest"],
    tuple,
    None,
]


ACTION_REPEAT: int = 4
MAX_EPISODE_FRAMES: int = 2400

# 60/3 = 20fps gravado, interpolado linearmente pelo frontend pra 60fps.
# valores mais altos (10+) faziam manobras rápidas (giro+chute) parecerem
# discretizadas; ~300KB por episódio de 1500 frames.
REPLAY_CAPTURE_INTERVAL: int = 3
REPLAY_FPS: int = 60 // REPLAY_CAPTURE_INTERVAL

GOAL_WIDTH_MIN: float = 170.0
GOAL_WIDTH_MAX: float = 300.0
GOAL_WIDTH_MU_START: float = 280.0
GOAL_WIDTH_MU_END: float = 195.0
GOAL_WIDTH_SIGMA: float = 40.0

SPAWN_DISTANCE_MIN: float = 80.0
SPAWN_DISTANCE_MAX: float = 250.0


class OpponentMode(str, Enum):
    """modo de operação do oponente conforme a fase do curriculum."""
    NONE = "none"          # opp parado, mas existe.
    PASSIVE = "passive"    # idem, semanticamente "está aqui mas não age".
    SCRIPTED = "scripted"  # heurística básica vai-pra-bola.
    POLICY = "policy"      # callable injetado (random / self-play).


class CurriculumPhase(str, Enum):
    """identificadores das fases do curriculum.

    1A: bola colada e alinhada com o gol — chute trivial.
    1B: bola próxima alinhada — pratica "ir até a bola".
    1C: bola média com pequeno ângulo — pratica alinhar+chutar.
    1D: bola lateral/diagonal (±60° a ±90°) — pratica girar e finalizar
        fora do cone frontal.
    1E: diagonal curta (20-80u, ±30° a ±90°) — cobre tremor diagonal.
    2 : ângulos amplos com opp passivo — generaliza.
    3 : opp scripted — primeira competição.
    3D: spawn defensivo + opp scripted — pratica posicionar entre bola e
        gol próprio + recuperar/limpar.
    3DA: defesa ativa contra opp avançando com a bola.
    3GB: goal block — agente dentro/colado no gol, bola voando direto pra ele.
    3T: contra-ataque — bola passando pra própria meta com opp scripted.
    4 : spawn aleatório no campo — generaliza espacial.
    5 : self-play — refinamento contra pool.
    """
    P1A = "1A"
    P1B = "1B"
    P1C = "1C"
    P1D = "1D"
    P1E = "1E"
    P2  = "2"
    P3  = "3"
    P3D = "3D"
    P3DA = "3DA"
    P3GB = "3GB"
    P3T = "3T"
    P4  = "4"
    P5  = "5"


# convenções dos campos:
#   agent_spawn_x_frac é em referência ao lado de ATAQUE do player.
#     red ataca direita:  frac=0.85 → x = 0.85 * field_w
#     blue ataca esquerda: frac=0.85 → x = (1 - 0.85) * field_w
#   agent_initial_angle_range é em referência ao lado de ATAQUE
#     (0 = olhando pro gol oponente; pra blue, env soma π em runtime).
_PHASE_CONFIGS: dict[str, dict] = {
    CurriculumPhase.P1A.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.85, 0.95),
        "agent_spawn_y_frac_range": (0.40, 0.60),
        "spawn_distance_range": (10.0, 20.0),
        "agent_initial_angle_range": (0.0, 0.0),
        "agent_initial_angle_strategy": "face_goal",
        "ball_alignment": "aligned_to_goal",
        "fixed_goal_width": 280.0,
        "max_episode_frames": 600,
    },
    CurriculumPhase.P1B.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.80, 0.92),
        "agent_spawn_y_frac_range": (0.30, 0.70),
        "spawn_distance_range": (30.0, 50.0),
        "agent_initial_angle_range": (-0.3, 0.3),
        "agent_initial_angle_strategy": "face_goal",
        "ball_alignment": "aligned_to_goal",
        "fixed_goal_width": 280.0,
        "max_episode_frames": 600,
    },
    CurriculumPhase.P1C.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.75, 0.92),
        "agent_spawn_y_frac_range": (0.20, 0.80),
        "spawn_distance_range": (50.0, 100.0),
        "agent_initial_angle_range": (-0.6, 0.6),
        "agent_initial_angle_strategy": "face_goal_with_noise",
        "ball_alignment": "small_angle_to_goal",
        "fixed_goal_width": 280.0,
        "max_episode_frames": 900,
    },
    # bola lateral/diagonal força o agente a girar antes de finalizar.
    CurriculumPhase.P1D.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.55, 0.85),
        "agent_spawn_y_frac_range": (0.25, 0.75),
        "spawn_distance_range": (60.0, 120.0),
        "agent_initial_angle_range": (0.0, 0.0),
        "agent_initial_angle_strategy": "face_goal",
        "ball_alignment": "lateral_to_goal",
        "fixed_goal_width": 280.0,
        "max_episode_frames": 1500,
    },
    # diagonal curta (20-80u, 30-90°) — cobre tremor diagonal frontal.
    CurriculumPhase.P1E.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.55, 0.85),
        "agent_spawn_y_frac_range": (0.25, 0.75),
        "spawn_distance_range": (20.0, 80.0),
        "agent_initial_angle_range": (0.0, 0.0),
        "agent_initial_angle_strategy": "face_goal",
        "ball_alignment": "diagonal_lateral_to_goal",
        "fixed_goal_width": None,
        "max_episode_frames": 1500,
    },
    CurriculumPhase.P2.value: {
        "opponent_mode": OpponentMode.PASSIVE,
        "agent_spawn_x_frac_range": (0.30, 0.92),
        "agent_spawn_y_frac_range": (0.15, 0.85),
        "spawn_distance_range": (50.0, 150.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "agent_initial_angle_strategy": "range",
        "ball_alignment": "random",
        "fixed_goal_width": None,
        "max_episode_frames": 1200,
    },
    CurriculumPhase.P3.value: {
        "opponent_mode": OpponentMode.SCRIPTED,
        "agent_spawn_x_frac_range": (0.20, 0.80),
        "spawn_distance_range": (80.0, 200.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "random",
        "fixed_goal_width": None,
        "max_episode_frames": 1800,
    },
    # spawn defensivo (próprio terço) com opp scripted: pratica posicionar
    # entre bola e gol próprio e recuperar.
    CurriculumPhase.P3D.value: {
        "opponent_mode": OpponentMode.SCRIPTED,
        "agent_spawn_x_frac_range": (0.05, 0.25),
        "spawn_distance_range": (120.0, 250.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "random",
        "fixed_goal_width": None,
        "max_episode_frames": 1800,
    },
    # defesa ativa: agente em posição defensiva, opp atrás da bola simulando
    # ataque ativo. cenário força o agente a defender — concede=-5 se ele não age.
    CurriculumPhase.P3DA.value: {
        "opponent_mode": OpponentMode.SCRIPTED,
        "agent_spawn_x_frac_range": (0.06, 0.25),
        "agent_spawn_y_frac_range": (0.30, 0.70),
        "spawn_distance_range": (200.0, 400.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "aligned_to_goal",
        "ball_initial_speed": 0.0,
        "opponent_spawn_strategy": "behind_ball_in_attack",
        "fixed_goal_width": None,
        "max_episode_frames": 1800,
    },
    # goal block: agente já dentro/colado no gol, bola voando direto pra ele;
    # agente precisa reagir movendo pra cobrir o ângulo.
    CurriculumPhase.P3GB.value: {
        "opponent_mode": OpponentMode.NONE,
        "agent_spawn_x_frac_range": (0.00, 0.06),
        "agent_spawn_y_frac_range": (0.30, 0.70),
        "spawn_distance_range": (200.0, 350.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "incoming_shot",
        "ball_initial_speed": 6.0,
        "fixed_goal_width": None,
        "max_episode_frames": 600,
    },
    # contra-ataque calibrado: bola atrás do agente em ângulo oblíquo, velocidade
    # 3.0 (25% do max — bola não chega no gol sozinha em ≥200u), opp atrás do
    # agente lado do gol enemy (handicap geométrico que compensa os 52f de giro 180°).
    CurriculumPhase.P3T.value: {
        "opponent_mode": OpponentMode.SCRIPTED,
        "agent_spawn_x_frac_range": (0.60, 0.80),
        "agent_spawn_y_frac_range": (0.25, 0.75),
        "spawn_distance_range": (180.0, 250.0),
        "agent_initial_angle_range": (0.0, 0.0),
        "agent_initial_angle_strategy": "face_goal",
        "ball_alignment": "behind_mixed_toward_own_half",
        "ball_initial_speed": 3.0,
        "opponent_spawn_strategy": "behind_agent_same_side",
        # determinístico — scripted noisy degradou contra-ataque (oponente
        # caótico ensinava reações erradas).
        "opponent_noise_eps": 0.0,
        "fixed_goal_width": None,
        "max_episode_frames": 2400,
    },
    CurriculumPhase.P4.value: {
        "opponent_mode": OpponentMode.SCRIPTED,
        "agent_spawn_x_frac_range": (0.0, 1.0),
        "spawn_distance_range": (80.0, 250.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "random",
        "fixed_goal_width": None,
        "max_episode_frames": 2400,
    },
    CurriculumPhase.P5.value: {
        "opponent_mode": OpponentMode.POLICY,
        "agent_spawn_x_frac_range": (0.0, 1.0),
        "spawn_distance_range": (80.0, 250.0),
        "agent_initial_angle_range": (-math.pi, math.pi),
        "ball_alignment": "random",
        "fixed_goal_width": None,
        "max_episode_frames": 2400,
    },
}


_NOOP_ACTION_IDX = encode_action(0, 0, 0)
_FORWARD_NO_KICK_IDX = encode_action(1, 0, 0)


OpponentPolicy = Callable[[np.ndarray], int]


def random_opponent_policy(obs: np.ndarray) -> int:
    """policy default: ação uniforme aleatória ∈ [0, 17].

    usada em smoke tests e como fallback se nenhuma policy é injetada.
    """
    return random.randint(0, ACTION_SPACE_SIZE - 1)


@dataclass(frozen=True)
class ReplayFrame:
    """schema canônico do frame capturado a cada REPLAY_CAPTURE_INTERVAL.

    mudanças no schema exigem update do consumidor JS.
    """
    step: int
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    players: list[dict]
    score: dict
    goal_width: float


_BREAKDOWN_KEYS = (
    "goal", "concede", "kick_on_goal",
)


class SoccerEnv(gym.Env):
    """gymnasium env para AI Soccer 1v1 com PPO discreto."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        capture_replay: bool = False,
        seed: Optional[int] = None,
        async_sync_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.opponent_policy: OpponentPolicy = opponent_policy or random_opponent_policy
        self.capture_replay: bool = bool(capture_replay)

        # async self-play sync via disco. quando setado, o env aceita
        # set_opponent_spec e carrega a Policy correspondente do disco.
        # set_opponent_policy (callable direto) continua funcionando pra
        # caminhos Sync/non-self-play.
        self._async_sync_dir: Optional[Path] = (
            Path(async_sync_dir) if async_sync_dir is not None else None
        )
        self._cached_version: Optional[int] = None
        self._cached_latest: Optional["object"] = None
        self._cached_pool: list = []
        # setado por _make_env_fn no trainer (Async). default 0 pra compat
        # quando spec_indexed é chamado em Sync ou em testes single-env.
        self._env_idx: int = 0

        self._np_rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        self.state: Optional[GameState] = None
        self.trained_team: Optional[str] = None
        self.trained_player: Optional[Player] = None
        self.opponent_player: Optional[Player] = None
        self.frame_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False
        self._replay_frames: list[dict] = []
        # curriculum progress default — sobrescrito por reset(options) ou
        # via set_progress(). permite que workers AsyncVectorEnv respeitem
        # o currículo sem passar options pelo reset.
        self._last_progress: float = 0.5
        self._current_phase: str = CurriculumPhase.P5.value
        self.opponent_mode: OpponentMode = OpponentMode.POLICY
        self._agent_spawn_x_frac_range: tuple[float, float] = (0.0, 1.0)
        self._spawn_distance_range: tuple[float, float] = (
            SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX,
        )
        self._agent_initial_angle_range: tuple[float, float] = (-math.pi, math.pi)
        self._agent_spawn_y_frac_range: tuple[float, float] = (0.15, 0.85)
        # estratégia de cálculo do ângulo inicial:
        #   "range"                → uniform de _agent_initial_angle_range
        #   "face_goal"            → atan2(dy, dx) apontando pro gol oponente
        #   "face_goal_with_noise" → face_goal + uniform de range
        self._agent_initial_angle_strategy: str = "range"
        self._ball_alignment: str = "random"
        self._fixed_goal_width: Optional[float] = None
        self._max_episode_frames: int = MAX_EPISODE_FRAMES
        self._ball_initial_speed: float = 0.0
        self._opponent_spawn_strategy: str = "mirror"
        self._opponent_noise_eps: float = 0.0
        # vetor unitário do spawn da bola (do agente pra bola). usado pra
        # setar velocidade inicial coerente em alignments com ball_initial_speed.
        self._last_ball_spawn_unit_vec: tuple[float, float] = (1.0, 0.0)
        self._last_spawn_distance: float = 0.0
        self._last_agent_initial_angle: float = 0.0

    # gymnasium.Env exige observation_space/action_space como atributos;
    # expomos via property pra manter o construtor enxuto e evitar mutação acidental.
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space

    def set_opponent_policy(self, policy: Optional[OpponentPolicy]) -> None:
        """troca opponent_policy entre episódios.

        None reseta pra random_opponent_policy. não chamar dentro de um
        episódio — comportamento indefinido em meio do action repeat.
        """
        self.opponent_policy = policy or random_opponent_policy

    def set_progress(
        self,
        progress: float,
        phase: Optional[Union[int, str, CurriculumPhase]] = None,
    ) -> None:
        """atualiza progress + fase do curriculum.

        phase aceita string, enum CurriculumPhase ou int 1..5 (legacy).
        """
        self._last_progress = max(0.0, min(1.0, float(progress)))
        if phase is None:
            return

        if isinstance(phase, CurriculumPhase):
            phase_str = phase.value
        elif isinstance(phase, int):
            _legacy_map = {1: "1A", 2: "2", 3: "3", 4: "4", 5: "5"}
            phase_str = _legacy_map.get(int(phase), "5")
        else:
            phase_str = str(phase)

        if phase_str not in _PHASE_CONFIGS:
            raise ValueError(
                f"Fase inválida: {phase!r}. Válidas: {sorted(_PHASE_CONFIGS.keys())}"
            )
        self._current_phase = phase_str
        self._update_phase_settings()

    def set_progress_indexed(
        self,
        progress: float,
        phases: list,
    ) -> None:
        """variante async-friendly de set_progress que aceita lista por env.

        gymnasium.vector.call invoca em todos os workers com mesmos args.
        cada worker pega phases[self._env_idx]. spec None mantém a fase atual.
        """
        if not (0 <= self._env_idx < len(phases)):
            return
        phase = phases[self._env_idx]
        if phase is not None:
            self.set_progress(progress, phase=phase)

    def _update_phase_settings(self) -> None:
        """aplica configurações da fase atual ao ambiente."""
        cfg = _PHASE_CONFIGS[self._current_phase]
        self.opponent_mode = cfg["opponent_mode"]
        self._agent_spawn_x_frac_range = cfg["agent_spawn_x_frac_range"]
        self._spawn_distance_range = cfg["spawn_distance_range"]
        self._agent_initial_angle_range = cfg["agent_initial_angle_range"]
        self._ball_alignment = cfg["ball_alignment"]
        self._fixed_goal_width = cfg["fixed_goal_width"]
        self._max_episode_frames = int(
            cfg.get("max_episode_frames", MAX_EPISODE_FRAMES)
        )
        self._agent_spawn_y_frac_range = cfg.get(
            "agent_spawn_y_frac_range", (0.15, 0.85)
        )
        self._agent_initial_angle_strategy = cfg.get(
            "agent_initial_angle_strategy", "range"
        )
        self._ball_initial_speed = float(cfg.get("ball_initial_speed", 0.0))
        # estratégia de spawn do opponent:
        #   "mirror" (default): opp_x = field_w - agent_x.
        #   "behind_agent_same_side" (P3T): opp atrás do agente, lado do gol enemy
        #     — handicap geométrico que compensa os 52f de giro do agente.
        #   "behind_ball_in_attack" (P3DA): opp perto da bola, lado oposto ao
        #     gol do agente — simula opp recém-atacando.
        self._opponent_spawn_strategy = str(cfg.get("opponent_spawn_strategy", "mirror"))
        self._opponent_noise_eps = float(cfg.get("opponent_noise_eps", 0.0))

    def _refresh_async_pool(self) -> None:
        """recarrega Policy + pool do disco se manifest mudou.

        no-op se async_sync_dir é None ou se o manifest está na mesma
        versão cacheada. falhas de I/O mantêm o cache anterior.
        """
        if self._async_sync_dir is None:
            return
        # import lazy pra evitar ciclo (env.py é importado por self_play.py).
        from backend.ai.self_play import load_pool_from_disk

        result = load_pool_from_disk(self._async_sync_dir, self._cached_version)
        if result is not None:
            new_version, latest, pool = result
            self._cached_version = new_version
            self._cached_latest = latest
            self._cached_pool = pool

    def set_opponent_spec(self, spec: OpponentSpec) -> None:
        """versão async-friendly de set_opponent_policy.

        aceita SPEC serializável (string/tuple). o env constrói o callable
        internamente a partir da Policy local cacheada. specs:
            "random"      → opponent uniforme aleatório.
            "latest"      → policy_latest.pt cacheada.
            ("pool", idx) → pool[idx] cacheada.
            None          → mantém opponent atual.

        se cache não tem o item solicitado, fallback gracioso pra random.
        """
        if spec is None:
            return

        self._refresh_async_pool()

        from backend.ai.self_play import (
            make_opponent_callable_from_policy,
            random_opponent_callable,
        )

        if spec == "random":
            self.opponent_policy = random_opponent_callable()
        elif spec == "latest":
            if self._cached_latest is not None:
                self.opponent_policy = make_opponent_callable_from_policy(
                    self._cached_latest, deterministic=False,
                )
            else:
                self.opponent_policy = random_opponent_callable()
        elif isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "pool":
            idx = int(spec[1])
            if 0 <= idx < len(self._cached_pool):
                self.opponent_policy = make_opponent_callable_from_policy(
                    self._cached_pool[idx], deterministic=False,
                )
            else:
                self.opponent_policy = random_opponent_callable()
        else:
            raise ValueError(f"Unknown opponent spec: {spec!r}")

    def set_opponent_spec_indexed(self, specs: list) -> None:
        """helper pra envs.call: cada worker pega specs[self._env_idx]."""
        if 0 <= self._env_idx < len(specs):
            self.set_opponent_spec(specs[self._env_idx])

    def _sample_goal_width(self, progress: float) -> float:
        """TruncatedNormal[170, 300] com μ migrando 280→195 conforme progress.

        se a fase atual fixou _fixed_goal_width, retorna esse valor direto.
        """
        if self._fixed_goal_width is not None:
            return float(self._fixed_goal_width)
        progress = max(0.0, min(1.0, float(progress)))
        mu = GOAL_WIDTH_MU_START + (GOAL_WIDTH_MU_END - GOAL_WIDTH_MU_START) * progress
        sample = self._np_rng.normal(mu, GOAL_WIDTH_SIGMA)
        return float(np.clip(sample, GOAL_WIDTH_MIN, GOAL_WIDTH_MAX))

    def _custom_spawn(self) -> None:
        """posiciona bola, trained_player e opponent_player conforme a fase.

        convenções:
            agent_spawn_x_frac é em referência ao lado de ATAQUE:
                red ataca direita (frac=0.85 → x=680); blue ataca esquerda
                (frac=0.85 → x=120).
            agent_initial_angle_range é em referência ao lado de ATAQUE:
                0.0 = olhando pro gol oponente; pra blue, env soma π pra
                "olhando pra esquerda".
            ball_alignment controla onde a bola aparece em relação ao agente.
        """
        field_w = self.state.field.width
        field_h = self.state.field.height
        is_red = self.trained_team == "red"
        enemy_goal_x = field_w if is_red else 0.0
        enemy_goal_y = field_h / 2.0

        f_lo, f_hi = self._agent_spawn_x_frac_range
        frac = float(self._np_rng.uniform(f_lo, f_hi))
        agent_x = frac * field_w if is_red else (1.0 - frac) * field_w
        y_lo, y_hi = self._agent_spawn_y_frac_range
        agent_y = float(self._np_rng.uniform(field_h * y_lo, field_h * y_hi))

        sd_lo, sd_hi = self._spawn_distance_range
        dist_to_ball = float(self._np_rng.uniform(sd_lo, sd_hi))

        alignment = self._ball_alignment
        if alignment == "aligned_to_goal":
            dx = enemy_goal_x - agent_x
            dy = enemy_goal_y - agent_y
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-3:
                ball_x = agent_x + dist_to_ball * dx / norm
                ball_y = agent_y + dist_to_ball * dy / norm
            else:
                ball_x = agent_x + dist_to_ball
                ball_y = agent_y
        elif alignment == "small_angle_to_goal":
            # bola em ângulo pequeno (±30°) da direção do gol.
            dx = enemy_goal_x - agent_x
            dy = enemy_goal_y - agent_y
            base_angle = math.atan2(dy, dx)
            angle_offset = float(self._np_rng.uniform(-math.pi / 6, math.pi / 6))
            final_angle = base_angle + angle_offset
            ball_x = agent_x + dist_to_ball * math.cos(final_angle)
            ball_y = agent_y + dist_to_ball * math.sin(final_angle)
        elif alignment == "lateral_to_goal":
            # bola lateral (±60° a ±90°) do vetor pro gol. combinado com
            # face_goal, força o agente a sair do cone frontal e girar.
            dx = enemy_goal_x - agent_x
            dy = enemy_goal_y - agent_y
            base_angle = math.atan2(dy, dx)
            sign = 1.0 if self._np_rng.random() < 0.5 else -1.0
            offset_mag = float(self._np_rng.uniform(math.pi / 3, math.pi / 2))
            final_angle = base_angle + sign * offset_mag
            ball_x = agent_x + dist_to_ball * math.cos(final_angle)
            ball_y = agent_y + dist_to_ball * math.sin(final_angle)
        elif alignment == "diagonal_lateral_to_goal":
            # bola em diagonal ampla (±30° a ±90°) — cobre o gap entre
            # small_angle_to_goal (±30°) e lateral_to_goal (±60°-90°).
            dx = enemy_goal_x - agent_x
            dy = enemy_goal_y - agent_y
            base_angle = math.atan2(dy, dx)
            sign = 1.0 if self._np_rng.random() < 0.5 else -1.0
            offset_mag = float(self._np_rng.uniform(math.pi / 6, math.pi / 2))
            final_angle = base_angle + sign * offset_mag
            ball_x = agent_x + dist_to_ball * math.cos(final_angle)
            ball_y = agent_y + dist_to_ball * math.sin(final_angle)
        elif alignment == "behind_toward_own_goal":
            # bola atrás do agente em direção ao próprio gol em linha reta.
            own_goal_x = 0.0 if is_red else field_w
            own_goal_y = enemy_goal_y
            dx = own_goal_x - agent_x
            dy = own_goal_y - agent_y
            norm = math.sqrt(dx * dx + dy * dy)
            if norm < 1e-3:
                ball_x = agent_x - dist_to_ball
                ball_y = agent_y
            else:
                ball_x = agent_x + dist_to_ball * dx / norm
                ball_y = agent_y + dist_to_ball * dy / norm
            self._last_ball_spawn_unit_vec = (
                (dx / norm, dy / norm) if norm >= 1e-3 else (-1.0, 0.0)
            )
        elif alignment == "behind_mixed_toward_own_half":
            # contra-ataque com 3 cenários equiprováveis:
            #   1/3 reta: 0° offset.
            #   1/3 oblíqua leve: ±15°-30°.
            #   1/3 oblíqua forte: ±35°-50°.
            # com vel=3.0, dist=180-250u, bola pára antes do gol e o opp
            # precisa virar antes de chutar (≥2 toques).
            own_goal_x = 0.0 if is_red else field_w
            own_goal_y = enemy_goal_y
            dx = own_goal_x - agent_x
            dy = own_goal_y - agent_y
            base_angle = math.atan2(dy, dx)

            bucket = self._np_rng.random()
            if bucket < 1.0 / 3.0:
                offset_mag = 0.0
                sign = 1.0
            elif bucket < 2.0 / 3.0:
                sign = 1.0 if self._np_rng.random() < 0.5 else -1.0
                offset_mag = float(self._np_rng.uniform(
                    math.pi / 12,
                    math.pi / 6,
                ))
            else:
                sign = 1.0 if self._np_rng.random() < 0.5 else -1.0
                offset_mag = float(self._np_rng.uniform(
                    math.pi * 35.0 / 180.0,
                    math.pi * 50.0 / 180.0,
                ))
            final_angle = base_angle + sign * offset_mag
            ux = math.cos(final_angle)
            uy = math.sin(final_angle)
            ball_x = agent_x + dist_to_ball * ux
            ball_y = agent_y + dist_to_ball * uy
            self._last_ball_spawn_unit_vec = (ux, uy)
        elif alignment == "incoming_shot":
            # bola spawnada num arco em volta do gol AGENTE (não do agente!),
            # com velocidade direta pro gol. dist_to_ball é distância da bola
            # ao gol agente (agente está dentro/colado no gol).
            own_goal_x = 0.0 if is_red else field_w
            own_goal_y = field_h / 2.0
            # ângulo random em ±60° do eixo frontal do gol.
            angle = float(self._np_rng.uniform(-math.pi / 3, math.pi / 3))
            sign_x = 1.0 if is_red else -1.0
            ball_x = own_goal_x + sign_x * dist_to_ball * math.cos(angle)
            ball_y = own_goal_y + dist_to_ball * math.sin(angle)
            # vetor unitário: bola → own_goal (velocidade incoming).
            self._last_ball_spawn_unit_vec = (
                -sign_x * math.cos(angle), -math.sin(angle)
            )
        else:
            angle = float(self._np_rng.uniform(-math.pi, math.pi))
            ball_x = agent_x + dist_to_ball * math.cos(angle)
            ball_y = agent_y + dist_to_ball * math.sin(angle)

        ball_margin = self.state.ball.radius + 10.0
        ball_x = max(ball_margin, min(field_w - ball_margin, ball_x))
        ball_y = max(ball_margin, min(field_h - ball_margin, ball_y))

        margin_p = self.trained_player.radius + 5.0
        agent_x = max(margin_p, min(field_w - margin_p, agent_x))
        agent_y = max(margin_p, min(field_h - margin_p, agent_y))

        # ângulo inicial: 3 estratégias.
        #   face_goal: atan2 pro gol oponente (ignora range).
        #   face_goal_with_noise: face_goal + uniform de range.
        #   range (default): sorteia de range; +π pra blue.
        # em face_goal*, enemy_goal_x já leva is_red em conta — não somar π.
        strategy = self._agent_initial_angle_strategy
        if strategy == "face_goal":
            dx_g = enemy_goal_x - agent_x
            dy_g = enemy_goal_y - agent_y
            agent_initial_angle = math.atan2(dy_g, dx_g)
        elif strategy == "face_goal_with_noise":
            dx_g = enemy_goal_x - agent_x
            dy_g = enemy_goal_y - agent_y
            base = math.atan2(dy_g, dx_g)
            a_lo, a_hi = self._agent_initial_angle_range
            if a_lo == a_hi:
                noise = float(a_lo)
            else:
                noise = float(self._np_rng.uniform(a_lo, a_hi))
            agent_initial_angle = base + noise
        else:
            a_lo, a_hi = self._agent_initial_angle_range
            if not is_red:
                a_lo += math.pi
                a_hi += math.pi
            if a_lo == a_hi:
                agent_initial_angle = float(a_lo)
            else:
                agent_initial_angle = float(self._np_rng.uniform(a_lo, a_hi))

        self.state.ball.x = ball_x
        self.state.ball.y = ball_y
        self.state.ball.vx = 0.0
        self.state.ball.vy = 0.0

        # velocidade inicial da bola (apenas em alignments que setaram
        # _last_ball_spawn_unit_vec).
        if self._ball_initial_speed > 0.0 and alignment in (
            "behind_toward_own_goal", "behind_mixed_toward_own_half",
            "incoming_shot",
        ):
            ux, uy = self._last_ball_spawn_unit_vec
            self.state.ball.vx = self._ball_initial_speed * ux
            self.state.ball.vy = self._ball_initial_speed * uy

        self.trained_player.x = agent_x
        self.trained_player.y = agent_y
        self.trained_player.angle = agent_initial_angle
        self.trained_player.vx = 0.0
        self.trained_player.vy = 0.0

        if self._opponent_spawn_strategy == "behind_agent_same_side":
            # opp atrás do agente, lado do gol enemy. handicap geométrico:
            # opp começa mais longe da bola, compensando os 52 frames de
            # giro 180° do agente.
            sign = 1.0 if (enemy_goal_x > agent_x) else -1.0
            offset = float(self._np_rng.uniform(100.0, 200.0))
            opp_x = agent_x + sign * offset
            opp_y = agent_y
            opp_margin = self.opponent_player.radius + 5.0
            opp_x = max(opp_margin, min(field_w - opp_margin, opp_x))
        elif self._opponent_spawn_strategy == "behind_ball_in_attack":
            # opp perto da bola, lado oposto ao gol do agente, simulando
            # "opp acabou de chutar a bola pro gol do defensor".
            own_goal_x = 0.0 if is_red else field_w
            sign = 1.0 if (own_goal_x < ball_x) else -1.0
            offset = float(self._np_rng.uniform(40.0, 100.0))
            opp_x = ball_x + sign * offset
            opp_y = ball_y + float(self._np_rng.uniform(-40.0, 40.0))
            opp_margin = self.opponent_player.radius + 5.0
            opp_x = max(opp_margin, min(field_w - opp_margin, opp_x))
            opp_y = max(opp_margin, min(field_h - opp_margin, opp_y))
        else:
            opp_x = field_w - agent_x
            opp_y = agent_y
        self.opponent_player.x = opp_x
        self.opponent_player.y = opp_y
        self.opponent_player.angle = agent_initial_angle
        self.opponent_player.vx = 0.0
        self.opponent_player.vy = 0.0

        for p in self.state.players:
            p.can_kick = True
            p.kick_cooldown_timer = 0
            p.is_kicking = False
            p.kick_attempted = False
            p.accel = 0.0
            p.rot = 0.0
            p.kick_requested = False

        self._last_spawn_distance = dist_to_ball
        self._last_agent_initial_angle = agent_initial_angle

    def _scripted_opponent_action(self) -> int:
        """oponente das fases 3/3D/3T/4: heurística básica + ε-noise opcional.

        NÃO é uma policy treinada — é baseline determinístico fraco. se ficar
        bom demais, o agente não progride; manter simples é proposital.

        com _opponent_noise_eps > 0, com probabilidade ε retorna ação RANDOM
        no lugar da scripted (opp "human-like").

        lógica scripted:
            1. desalinhado com a bola (cos < 0.7): vira na direção dela.
            2. longe da bola (>120u) e alinhado: avança.
            3. perto da bola (<120u) e desalinhado com o gol: vira pro gol.
            4. alinhado com gol e perto (<80u): chuta.
            5. caso contrário: avança.
        """
        eps = getattr(self, "_opponent_noise_eps", 0.0)
        if eps > 0.0 and random.random() < eps:
            return random.randint(0, ACTION_SPACE_SIZE - 1)

        opp = self.opponent_player
        ball = self.state.ball
        field_w = self.state.field.width
        field_h = self.state.field.height

        dx_b = ball.x - opp.x
        dy_b = ball.y - opp.y
        dist_to_ball = math.sqrt(dx_b * dx_b + dy_b * dy_b)
        angle_to_ball = math.atan2(dy_b, dx_b)
        angle_diff_ball = (angle_to_ball - opp.angle + math.pi) % (2 * math.pi) - math.pi

        if dist_to_ball > 120.0:
            if abs(angle_diff_ball) > 0.4:
                return encode_action(0, 1 if angle_diff_ball > 0 else -1, 0)
            return _FORWARD_NO_KICK_IDX

        target_goal_x = field_w if opp.team == "red" else 0.0
        target_goal_y = field_h / 2.0

        dx_g = target_goal_x - opp.x
        dy_g = target_goal_y - opp.y
        angle_to_goal = math.atan2(dy_g, dx_g)
        angle_diff_goal = (angle_to_goal - opp.angle + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff_goal) > 0.3:
            return encode_action(0, 1 if angle_diff_goal > 0 else -1, 0)

        if dist_to_ball < 80.0:
            return encode_action(1, 0, 1)
        return encode_action(1, 0, 0)

    def _capture_replay_frame(self) -> None:
        frame = {
            "step": self.frame_count,
            "ball_x": float(self.state.ball.x),
            "ball_y": float(self.state.ball.y),
            "ball_vx": float(self.state.ball.vx),
            "ball_vy": float(self.state.ball.vy),
            "players": [
                {
                    "id": p.id,
                    "team": p.team,
                    "x": float(p.x),
                    "y": float(p.y),
                    "angle": float(p.angle),
                    "vx": float(p.vx),
                    "vy": float(p.vy),
                    "is_kicking": bool(p.is_kicking),
                }
                for p in self.state.players
            ],
            "score": dict(self.state.score),
            "goal_width": float(self.state.field.goal_width),
        }
        self._replay_frames.append(frame)

    def get_replay(self) -> list[dict]:
        """cópia da lista de frames capturados no último episódio."""
        return list(self._replay_frames)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
            random.seed(seed)

        progress = (options or {}).get("progress", self._last_progress)
        self._last_progress = float(progress)

        goal_width = self._sample_goal_width(progress)

        field = Field()
        field.goal_width = goal_width
        self.state = GameState(field=field)
        self.state.init()
        self.state.players = [
            Player(id="red_0", team="red"),
            Player(id="blue_0", team="blue"),
        ]
        self.state.ball = Ball()

        self.trained_team = "red" if self._np_rng.random() < 0.5 else "blue"
        self.trained_player = next(
            p for p in self.state.players if p.team == self.trained_team
        )
        self.opponent_player = next(
            p for p in self.state.players if p.team != self.trained_team
        )

        self._custom_spawn()
        spawn_distance = self._last_spawn_distance
        agent_initial_angle = self._last_agent_initial_angle

        self.frame_count = 0
        self.terminated = False
        self.truncated = False
        self._replay_frames = []

        if self.capture_replay:
            self._capture_replay_frame()

        initial_obs = gather_obs(self.trained_player, self.state, action_repeat_idx=0)
        info = {
            "trained_team": self.trained_team,
            "goal_width": goal_width,
            "spawn_distance": spawn_distance,
            "agent_initial_angle": agent_initial_angle,
        }
        return initial_obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError(
                "step() called on terminated/truncated env. Call reset() first."
            )

        action = int(action)

        # opponent decide UMA vez por step do gym, não 4× por frame interno.
        if self.opponent_mode in (OpponentMode.NONE, OpponentMode.PASSIVE):
            opp_action = _NOOP_ACTION_IDX
        elif self.opponent_mode == OpponentMode.SCRIPTED:
            opp_action = self._scripted_opponent_action()
        else:
            opp_obs = gather_obs(self.opponent_player, self.state, action_repeat_idx=0)
            opp_action = int(self.opponent_policy(opp_obs))

        cumulative_reward = 0.0
        cumulative_breakdown = {k: 0.0 for k in _BREAKDOWN_KEYS}
        frames_executed = 0
        terminated_this_step = False

        for repeat_idx in range(ACTION_REPEAT):
            snap_prev = make_snapshot(self.trained_player, self.state)

            apply_discrete_action(self.trained_player, action)
            apply_discrete_action(self.opponent_player, opp_action)

            physics_step(self.state)
            self.frame_count += 1
            frames_executed += 1

            snap_curr = make_snapshot(self.trained_player, self.state)

            r, breakdown = compute_rewards(snap_prev, snap_curr, self.state)
            cumulative_reward += r
            for k in _BREAKDOWN_KEYS:
                cumulative_breakdown[k] += breakdown[k]

            if self.capture_replay and self.frame_count % REPLAY_CAPTURE_INTERVAL == 0:
                self._capture_replay_frame()

            # gol marcado = terminal markoviano genuíno. engine já chamou
            # _reset_after_goal; o env precisa terminar antes que o caller
            # use o state pós-reset como continuação.
            if self.state.goal_scored_this_step:
                terminated_this_step = True
                break

        truncated = (
            self.frame_count >= self._max_episode_frames and not terminated_this_step
        )

        self.terminated = terminated_this_step
        self.truncated = truncated

        # captura 1 frame extra ao terminar/truncar se o último não foi
        # alinhado com o intervalo — sem isso, perde-se até INTERVAL-1 frames
        # finais. em caso de gol, o engine já reposicionou a bola pro centro
        # e o frontend detecta o teleporte (ver _detectGoalEvents em player.js).
        if (
            self.capture_replay
            and (terminated_this_step or truncated)
            and self.frame_count % REPLAY_CAPTURE_INTERVAL != 0
        ):
            self._capture_replay_frame()

        final_obs = gather_obs(
            self.trained_player,
            self.state,
            action_repeat_idx=frames_executed - 1,
        )

        info = {
            "reward_breakdown": cumulative_breakdown,
            "frames_executed": frames_executed,
            "trained_team": self.trained_team,
        }

        return final_obs, float(cumulative_reward), terminated_this_step, truncated, info
