"""PPO discreto adaptado pra self-play 1v1 com domain randomization.

estilo CleanRL ppo.py: rollout sequencial, GAE, K epochs com minibatch
shuffling, clip de policy + value, KL opcional (default None).

diferenças em relação ao CleanRL canônico:
    1. self-play via OpponentManager — sample por episódio.
    2. domain randomization via progress linear sobre total_timesteps,
       propagado aos envs via set_progress.
    3. reward breakdown logado em TensorBoard por componente.
    4. terminated e truncated tratados separadamente no GAE — apenas
       terminated zera bootstrap.
    5. RunningMeanStd próprio (Welford) — sem dependência do SB3.

uso CLI:
    python -m backend.ai.train --smoke-test 0
    python -m backend.ai.train --smoke-test 0 --total-timesteps 10000
    python -m backend.ai.train --name my_run --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium.vector
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.callbacks import BREAKDOWN_KEYS, EnvParamTracker, RewardBreakdownTracker
from backend.ai.env import SoccerEnv
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import Policy
from backend.ai.self_play import OpponentManager


@dataclass
class PPOConfig:
    """hyperparams do trainer. smoke tests sobrescrevem total_timesteps
    + flags de self-play/curriculum.
    """

    total_timesteps: int = 30_000_000
    num_envs: int = 8
    num_steps: int = 256

    learning_rate: float = 3e-4
    anneal_lr: bool = True

    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    # anneal linear de ent_coef ao longo do treino. None ⇒ constante.
    # ent_coef_atual = ent_coef + (ent_coef_end - ent_coef) × progress.
    # valor menor no fim permite convergência da policy; ent_coef=0.03
    # fixo trava entropy e a policy nunca decide uma ação.
    ent_coef_end: Optional[float] = None
    vf_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    norm_obs: bool = True
    norm_reward: bool = True

    enable_self_play: bool = True
    save_steps: int = 500_000
    fixed_random_opponent: bool = False
    # quando True E self-play ativo E não force_sync: workers Async carregam
    # Policy do oponente do disco (sync via manifest). speedup ~2-3× em
    # treinos longos. trade-off: workers podem usar Policy "ligeiramente
    # desatualizada" entre updates do pool (defasagem <0.2% em 500k steps).
    async_sync_pool_to_disk: bool = True

    enable_curriculum: bool = True
    # curriculum de fases (1A/1B/.../5) propagadas via set_progress(phase=).
    # cada phase_*_end_step define o LIMITE superior (exclusivo) da fase.
    # phase 5 vai até o fim. fases opcionais (1D, 1E, 3D, 3DA, 3GB, 3T) são
    # puladas quando o end_step correspondente é None.
    enable_curriculum_phases: bool = False
    phase_1a_end_step: int = 500_000
    phase_1b_end_step: int = 1_000_000
    phase_1c_end_step: int = 1_500_000
    phase_1d_end_step: Optional[int] = None
    phase_1e_end_step: Optional[int] = None
    phase_2_end_step:  int = 2_500_000
    phase_3_end_step:  int = 3_500_000
    phase_3d_end_step: Optional[int] = None
    phase_3da_end_step: Optional[int] = None
    phase_3gb_end_step: Optional[int] = None
    phase_3t_end_step: Optional[int] = None
    phase_4_end_step:  int = 4_500_000
    # rotação anti-esquecimento: probabilidade de um env em fase 2+ rodar
    # um episódio em fase de skill básica. weights opcionais por fase via
    # rotation_weights dict.
    anti_forgetting_rotation_prob: float = 0.20
    # pesos por fase no pool de rotação. None ⇒ uniforme. pesos para fases
    # ausentes do pool (ex: weight pra "1D" quando phase_1d_end_step is None)
    # são silenciosamente ignorados.
    rotation_weights: Optional[dict] = None

    log_frequency_updates: int = 1
    log_to_stdout_every_n_updates: int = 5

    seed: int = 42
    device: str = "cpu"

    # testes forçam SyncVectorEnv pra evitar custos de multiprocessing.
    force_sync_vec_env: bool = False

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)


def _smoke_test_configs() -> dict[int, PPOConfig]:
    """configs de smoke test pré-definidas, indexadas por número."""
    return {
        0: PPOConfig(
            total_timesteps=200_000,
            fixed_random_opponent=True,
            enable_self_play=False,
            enable_curriculum=False,
            save_steps=100_000,
        ),
        1: PPOConfig(
            total_timesteps=1_000_000,
            fixed_random_opponent=True,
            enable_self_play=False,
            enable_curriculum=True,
            save_steps=500_000,
        ),
        2: PPOConfig(
            total_timesteps=3_000_000,
            fixed_random_opponent=True,
            enable_self_play=False,
            enable_curriculum=True,
            save_steps=500_000,
        ),
        3: PPOConfig(
            total_timesteps=10_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            save_steps=500_000,
        ),
        4: PPOConfig(
            total_timesteps=13_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=  600_000,
            phase_1b_end_step=1_300_000,
            phase_1c_end_step=2_000_000,
            phase_2_end_step= 5_000_000,
            phase_3_end_step= 8_000_000,
            phase_4_end_step=10_500_000,
            save_steps=500_000,
            force_sync_vec_env=True,
        ),
        5: PPOConfig(
            total_timesteps=20_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=    800_000,
            phase_1b_end_step=  1_600_000,
            phase_1c_end_step=  2_400_000,
            phase_1d_end_step=  6_000_000,
            phase_2_end_step=   7_600_000,
            phase_3_end_step=   9_200_000,
            phase_3d_end_step= 11_600_000,
            phase_3t_end_step= 12_400_000,
            phase_4_end_step=  13_400_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={"1A": 0.20, "1D": 0.40, "3D": 0.20, "3T": 0.20},
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=500_000,
        ),
        7: PPOConfig(
            total_timesteps=300_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=  12_000_000,
            phase_1b_end_step=  24_000_000,
            phase_1c_end_step=  36_000_000,
            phase_1d_end_step=  90_000_000,
            phase_2_end_step=  114_000_000,
            phase_3_end_step=  138_000_000,
            phase_3d_end_step= 174_000_000,
            phase_3t_end_step= 186_000_000,
            phase_4_end_step=  201_000_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={"1A": 0.20, "1D": 0.40, "3D": 0.20, "3T": 0.20},
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=7_500_000,
        ),
        # V10.2 — config de produção (50M timesteps, ~6h em M4).
        10: PPOConfig(
            total_timesteps=50_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=    2_000_000,
            phase_1b_end_step=    4_000_000,
            phase_1c_end_step=    6_000_000,
            phase_1d_end_step=   13_500_000,
            phase_1e_end_step=   15_500_000,
            phase_2_end_step=    18_500_000,
            phase_3_end_step=    21_500_000,
            phase_3d_end_step=   26_000_000,
            phase_3da_end_step=  27_500_000,
            phase_3gb_end_step=  29_000_000,
            phase_3t_end_step=   None,
            phase_4_end_step=    33_000_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={
                "1A": 0.20, "1D": 0.45, "1E": 0.10,
                "3D": 0.10, "3DA": 0.05, "3GB": 0.10,
            },
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=1_250_000,
        ),
        9: PPOConfig(
            total_timesteps=50_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=    2_000_000,
            phase_1b_end_step=    4_000_000,
            phase_1c_end_step=    6_000_000,
            phase_1d_end_step=   13_000_000,
            phase_1e_end_step=   15_500_000,
            phase_2_end_step=    19_000_000,
            phase_3_end_step=    22_500_000,
            phase_3d_end_step=   27_500_000,
            phase_3da_end_step=  30_000_000,
            phase_3t_end_step=   None,
            phase_4_end_step=    33_500_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={"1A": 0.20, "1D": 0.40, "1E": 0.10, "3D": 0.20, "3DA": 0.10},
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=1_250_000,
        ),
        8: PPOConfig(
            total_timesteps=50_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=   2_000_000,
            phase_1b_end_step=   4_000_000,
            phase_1c_end_step=   6_000_000,
            phase_1d_end_step=  15_000_000,
            phase_2_end_step=   19_000_000,
            phase_3_end_step=   23_000_000,
            phase_3d_end_step=  30_000_000,
            phase_3t_end_step=  None,
            phase_4_end_step=   33_500_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={"1A": 0.25, "1D": 0.50, "3D": 0.25},
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=1_250_000,
        ),
        6: PPOConfig(
            total_timesteps=50_000_000,
            fixed_random_opponent=False,
            enable_self_play=True,
            enable_curriculum=True,
            enable_curriculum_phases=True,
            phase_1a_end_step=   2_000_000,
            phase_1b_end_step=   4_000_000,
            phase_1c_end_step=   6_000_000,
            phase_1d_end_step=  15_000_000,
            phase_2_end_step=   19_000_000,
            phase_3_end_step=   23_000_000,
            phase_3d_end_step=  28_000_000,
            phase_3t_end_step=  30_500_000,
            phase_4_end_step=   32_500_000,
            anti_forgetting_rotation_prob=0.30,
            rotation_weights={"1A": 0.25, "1D": 0.40, "3D": 0.25, "3T": 0.10},
            ent_coef=0.03,
            ent_coef_end=0.005,
            async_sync_pool_to_disk=True,
            save_steps=1_250_000,
        ),
    }


class RunningMeanStd:
    """Welford online stats (Chan, Golub, LeVeque 1979).

    suporta entrada single ou batch. usado pra normalizar obs e reward em
    treino. inferência usa fixos os mean/var aprendidos — exposto via
    state_dict / load_state_dict.
    """

    def __init__(self, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """update batch. x shape (B, *shape) ou (*shape,)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """normaliza com running stats. clamp em ±clip."""
        std = np.sqrt(self.var + 1e-8)
        normed = (x - self.mean) / std
        return np.clip(normed, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    next_terminated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """generalized advantage estimation com tratamento correto de truncation.

    terminated zera bootstrap (gol); truncated NÃO zera (envelope de tempo).
    distinção crítica: bootstrap em truncated preserva o valor estimado pro
    estado-final, enquanto em terminated descarta (futuro = 0).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_terminated
            next_v = next_value
        else:
            nonterminal = 1.0 - terminated[t]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """PPO discreto + self-play + domain randomization.

    saída em disco (em data/runs/{ts}_{run_name}/):
        checkpoints/checkpoint_{step}.pt — a cada save_steps
        checkpoints/final.pt — último
        replays/replay_{step}.json — episódio capturado a cada save
        tensorboard/ — SummaryWriter logs
        config.json — config usada (reprodutibilidade)
    """

    def __init__(self, config: PPOConfig, run_name: str = "default"):
        self.config = config
        self.run_name = run_name

        self._seed_all(config.seed)

        # RNG dedicado pra rotação anti-esquecimento. seed offset (+7919)
        # mantém determinismo independente de outros RNGs do trainer.
        self._curriculum_rng = random.Random(config.seed + 7919)

        # rastreia última fase com replay capturado. usado pra disparar
        # replay automático quando _current_phase() muda.
        self._last_captured_phase: Optional[str] = None

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path("data") / "runs" / f"{ts}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "replays").mkdir(exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        # basicConfig é no-op se root já tem handler (ex: pytest caplog);
        # em produção instala um handler stdout.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("PPOTrainer")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Run dir: {self.run_dir}")
        self.logger.info(
            f"Total timesteps: {config.total_timesteps:,} | "
            f"num_envs: {config.num_envs} | "
            f"updates totais: {config.num_updates}"
        )
        self.logger.info(
            f"Self-play: {config.enable_self_play} | "
            f"Curriculum: {config.enable_curriculum} | "
            f"Random opponent: {config.fixed_random_opponent}"
        )

        # async self-play sync via disco: dir setado ANTES de _make_envs
        # pra que workers recebam o caminho na construção.
        self.async_sync_dir: Path = self.run_dir / "_async_pool"

        self.envs = self._make_envs()

        self.policy = Policy().to(config.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

        if config.enable_self_play and not config.fixed_random_opponent:
            self.opponent_manager: Optional[OpponentManager] = OpponentManager(seed=config.seed)
            self.opponent_manager.add_checkpoint(self.policy)
        else:
            self.opponent_manager = None

        # async self-play: escreve pool inicial em disco. workers vão
        # carregar quando o primeiro set_opponent_spec for enviado.
        if self._should_use_async_with_self_play():
            from backend.ai.self_play import serialize_pool_to_disk
            serialize_pool_to_disk(
                self.async_sync_dir,
                self.policy,
                list(self.opponent_manager._pool),
            )
            self.logger.info(
                f"Async self-play habilitado. Sync dir: {self.async_sync_dir}"
            )

        self.breakdown_tracker = RewardBreakdownTracker(config.num_envs)
        self.env_param_tracker = EnvParamTracker()

        self.obs_rms: Optional[RunningMeanStd] = (
            RunningMeanStd((OBS_SIZE,)) if config.norm_obs else None
        )
        self.rew_rms: Optional[RunningMeanStd] = (
            RunningMeanStd(()) if config.norm_reward else None
        )

        self.global_step = 0

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_env_fn(self, seed: int, env_idx: int):
        # async_sync_dir só é passado se async self-play está ativo —
        # Sync ou async-com-random não precisam de sync via disco.
        sync_dir = (
            self.async_sync_dir
            if self._should_use_async_with_self_play()
            else None
        )

        def _make():
            env = SoccerEnv(seed=seed, async_sync_dir=sync_dir)
            env._env_idx = env_idx
            return env

        return _make

    def _should_use_async_with_self_play(self) -> bool:
        """async é viável com self-play quando sync via disco está habilitado."""
        return (
            self.config.enable_self_play
            and not self.config.fixed_random_opponent
            and not self.config.force_sync_vec_env
            and self.config.num_envs > 1
            and self.config.async_sync_pool_to_disk
        )

    def _make_envs(self):
        env_fns = [
            self._make_env_fn(self.config.seed + i, i)
            for i in range(self.config.num_envs)
        ]
        # async habilitado em dois casos:
        #   1. opponent fixo aleatório (callable trivialmente recriável).
        #   2. self-play com sync via disco (workers carregam Policy do disco).
        use_async = (
            not self.config.force_sync_vec_env
            and self.config.num_envs > 1
            and (
                self.config.fixed_random_opponent
                or self._should_use_async_with_self_play()
            )
        )
        if use_async:
            return gymnasium.vector.AsyncVectorEnv(env_fns)
        return gymnasium.vector.SyncVectorEnv(env_fns)

    def _sample_opponent_spec(self):
        """gera OpponentSpec serializável análogo ao sample_opponent_callable.

        replica a distribuição 60/20/20 (latest/pool/random) consumindo o
        mesmo RNG do OpponentManager — garante mesma sequência em sync vs
        async com mesmo seed.
        """
        if self.opponent_manager is None:
            return "random"

        rng = self.opponent_manager._rng
        from backend.ai.self_play import LATEST_RATIO, POOL_RATIO

        r = rng.random()
        if r < LATEST_RATIO:
            return "latest"
        if r < LATEST_RATIO + POOL_RATIO:
            n = self.opponent_manager.pool_size
            if n == 0:
                return "latest"
            return ("pool", rng.randrange(n))
        return "random"

    def _set_opponents_for_envs(self, mask) -> None:
        """re-injeta opponent nos envs marcados em mask.

        sync (envs.envs): usa callable diretamente via set_opponent_policy.
        async (sem .envs): gera OpponentSpec por env e envia em lote via
            envs.call("set_opponent_spec_indexed", specs); cada worker pega
            o spec do seu índice e carrega a Policy correspondente do disco.
        """
        if self.opponent_manager is None:
            return
        if not any(mask):
            return

        if hasattr(self.envs, "envs"):
            for i, do_set in enumerate(mask):
                if do_set:
                    opp = self.opponent_manager.sample_opponent_callable(self.policy)
                    self.envs.envs[i].set_opponent_policy(opp)
            return

        specs = [
            self._sample_opponent_spec() if do_set else None
            for do_set in mask
        ]
        self.envs.call("set_opponent_spec_indexed", specs)

    def _current_phase(self) -> str:
        """retorna identificador (string) da fase atual.

        ordem das fases:
            1A → 1B → 1C → [1D?] → [1E?] → 2 → 3 → [3D?] → [3DA?] →
            [3GB?] → [3T?] → 4 → 5

        fases opcionais são puladas quando o end_step correspondente é None.
        """
        if not self.config.enable_curriculum_phases:
            return "5"
        s = self.config.phase_1a_end_step
        if self.global_step < s: return "1A"
        s = self.config.phase_1b_end_step
        if self.global_step < s: return "1B"
        s = self.config.phase_1c_end_step
        if self.global_step < s: return "1C"
        if self.config.phase_1d_end_step is not None:
            if self.global_step < self.config.phase_1d_end_step: return "1D"
        if self.config.phase_1e_end_step is not None:
            if self.global_step < self.config.phase_1e_end_step: return "1E"
        if self.global_step < self.config.phase_2_end_step: return "2"
        if self.global_step < self.config.phase_3_end_step: return "3"
        if self.config.phase_3d_end_step is not None:
            if self.global_step < self.config.phase_3d_end_step: return "3D"
        if self.config.phase_3da_end_step is not None:
            if self.global_step < self.config.phase_3da_end_step: return "3DA"
        if self.config.phase_3gb_end_step is not None:
            if self.global_step < self.config.phase_3gb_end_step: return "3GB"
        if self.config.phase_3t_end_step is not None:
            if self.global_step < self.config.phase_3t_end_step: return "3T"
        if self.global_step < self.config.phase_4_end_step: return "4"
        return "5"

    def _rotation_pool(self) -> list[str]:
        """pool de fases base para a rotação anti-esquecimento.

        skills só entram no pool se a fase correspondente foi habilitada.
        """
        pool = ["1A"]
        if self.config.phase_1d_end_step is not None:
            pool.append("1D")
        if self.config.phase_1e_end_step is not None:
            pool.append("1E")
        if self.config.phase_3d_end_step is not None:
            pool.append("3D")
        if self.config.phase_3da_end_step is not None:
            pool.append("3DA")
        if self.config.phase_3gb_end_step is not None:
            pool.append("3GB")
        if self.config.phase_3t_end_step is not None:
            pool.append("3T")
        return pool

    def _rotation_weights_for(self, pool: list[str]) -> Optional[list[float]]:
        """resolve config.rotation_weights (dict) em lista alinhada com pool.

        fases ausentes do dict ganham peso 1.0. quando rotation_weights is
        None, retorna None (caller usa amostragem uniforme).
        """
        if self.config.rotation_weights is None:
            return None
        weights = [
            float(self.config.rotation_weights.get(phase, 1.0))
            for phase in pool
        ]
        if sum(weights) <= 0.0:
            return None
        return weights

    def _set_progress_for_envs(self, mask) -> None:
        """propaga progress (e phase, com rotação) aos envs marcados.

        rotação anti-esquecimento: quando a fase atual NÃO é uma das fases
        de aprendizado de skill base, com probabilidade
        anti_forgetting_rotation_prob o env é setado de volta pra uma fase
        do pool de rotação. reforça skills base evitando esquecimento.
        """
        if not self.config.enable_curriculum:
            return
        if not any(mask):
            return
        progress = min(1.0, self.global_step / max(1, self.config.total_timesteps))

        if not self.config.enable_curriculum_phases:
            self.envs.call("set_progress", progress)
            return

        base_phase = self._current_phase()
        # rotação só ativa fora das fases de aprendizado. durante uma fase
        # de aprendizado, não rotacionamos pra ela mesma.
        learning_phases = {"1A", "1B", "1C", "1D", "1E", "3D", "3DA", "3GB"}
        rotation_prob = (
            self.config.anti_forgetting_rotation_prob
            if base_phase not in learning_phases
            else 0.0
        )
        rotation_pool = self._rotation_pool()
        rotation_weights = self._rotation_weights_for(rotation_pool)

        phases: list = []
        for do_set in mask:
            if not do_set:
                phases.append(None)
                continue
            if rotation_prob > 0.0 and self._curriculum_rng.random() < rotation_prob:
                if rotation_weights is not None:
                    sampled = self._curriculum_rng.choices(
                        rotation_pool, weights=rotation_weights, k=1
                    )[0]
                else:
                    sampled = rotation_pool[
                        self._curriculum_rng.randrange(len(rotation_pool))
                    ]
                phases.append(sampled)
            else:
                phases.append(base_phase)

        if hasattr(self.envs, "envs"):
            for i, p in enumerate(phases):
                if p is not None:
                    self.envs.envs[i].set_progress(progress, phase=p)
        else:
            self.envs.call("set_progress_indexed", progress, phases)

    def _normalize_obs(self, obs_np: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs_np.astype(np.float32)
        self.obs_rms.update(obs_np)
        return self.obs_rms.normalize(obs_np)

    def _normalize_reward(self, reward_np: np.ndarray) -> np.ndarray:
        if self.rew_rms is None:
            return reward_np.astype(np.float32)
        self.rew_rms.update(reward_np)
        std = math.sqrt(float(self.rew_rms.var) + 1e-8)
        return np.clip(reward_np / std, -10.0, 10.0).astype(np.float32)

    def _extract_breakdowns(self, info: dict) -> list[dict]:
        out = []
        rb = info.get("reward_breakdown")
        for i in range(self.config.num_envs):
            bd = {}
            for k in BREAKDOWN_KEYS:
                if rb is not None and k in rb:
                    val = rb[k]
                    bd[k] = float(val[i]) if hasattr(val, "__len__") else float(val)
                else:
                    bd[k] = 0.0
            out.append(bd)
        return out

    def _extract_reset_info_per_env(self, info: dict) -> list[dict]:
        out = []
        for i in range(self.config.num_envs):
            sub = {}
            for k in ("trained_team", "goal_width", "spawn_distance", "agent_initial_angle"):
                v = info.get(k)
                if v is None:
                    continue
                if isinstance(v, np.ndarray) and v.shape:
                    sub[k] = v[i]
                else:
                    sub[k] = v
            out.append(sub)
        return out

    def _collect_rollout(self, next_obs: torch.Tensor, next_terminated: torch.Tensor):
        T, N = self.config.num_steps, self.config.num_envs

        obs_buf = torch.zeros((T, N, OBS_SIZE), dtype=torch.float32)
        action_buf = torch.zeros((T, N), dtype=torch.long)
        logprob_buf = torch.zeros((T, N), dtype=torch.float32)
        value_buf = torch.zeros((T, N), dtype=torch.float32)
        reward_buf = torch.zeros((T, N), dtype=torch.float32)
        terminated_buf = torch.zeros((T, N), dtype=torch.float32)

        for t in range(T):
            obs_buf[t] = next_obs

            with torch.no_grad():
                action, logprob, _, value = self.policy.get_action_and_value(next_obs)
            action_buf[t] = action
            logprob_buf[t] = logprob
            value_buf[t] = value.reshape(-1)

            actions_np = action.cpu().numpy()
            next_obs_np, reward_np, terminated, truncated, info = self.envs.step(actions_np)

            done_mask = np.logical_or(terminated, truncated)

            breakdowns = self._extract_breakdowns(info)
            finished = self.breakdown_tracker.step(breakdowns, [bool(d) for d in done_mask])
            for ep in finished:
                for k in BREAKDOWN_KEYS:
                    self.writer.add_scalar(f"reward/{k}_per_episode", ep[k], self.global_step)
                self.writer.add_scalar("episode/length", ep["episode_length"], self.global_step)

            # info do auto-reset (NEXT_STEP autoreset): gymnasium.vector
            # empacota como info[k] = np.array([..., val_i, ...]) com
            # máscara info["_k"] = np.array([..., True, ...]). sem usar
            # a máscara, 0.0s espúrios entram no tracker e enviesam.
            if "goal_width" in info:
                gw_arr = info["goal_width"]
                gw_mask = info.get("_goal_width")
                if hasattr(gw_arr, "__len__") and hasattr(gw_mask, "__len__"):
                    for i in range(self.config.num_envs):
                        if i >= len(gw_mask) or not bool(gw_mask[i]):
                            continue
                        gw_i = gw_arr[i]
                        sub = {"goal_width": float(gw_i)}
                        for k in ("spawn_distance", "agent_initial_angle", "trained_team"):
                            v = info.get(k)
                            mask = info.get("_" + k)
                            if (
                                v is None or mask is None
                                or not hasattr(v, "__len__")
                                or not hasattr(mask, "__len__")
                                or i >= len(v) or i >= len(mask)
                                or not bool(mask[i])
                            ):
                                continue
                            sub[k] = v[i]
                        self.env_param_tracker.add_episode(sub)

            self._set_progress_for_envs(done_mask)
            self._set_opponents_for_envs(done_mask)

            reward_norm = self._normalize_reward(reward_np.astype(np.float32))
            reward_buf[t] = torch.from_numpy(reward_norm)
            terminated_buf[t] = torch.from_numpy(terminated.astype(np.float32))

            next_obs_norm = self._normalize_obs(next_obs_np.astype(np.float32))
            next_obs = torch.from_numpy(next_obs_norm)

            self.global_step += N

        with torch.no_grad():
            next_value = self.policy.get_value(next_obs).reshape(-1)
        next_terminated_t = torch.from_numpy(terminated.astype(np.float32))

        return {
            "obs": obs_buf,
            "actions": action_buf,
            "logprobs": logprob_buf,
            "values": value_buf,
            "rewards": reward_buf,
            "terminated": terminated_buf,
            "next_value": next_value,
            "next_terminated": next_terminated_t,
            "last_obs": next_obs,
        }

    def _current_ent_coef(self) -> float:
        """retorna ent_coef atual (com anneal linear quando configurado)."""
        if self.config.ent_coef_end is None:
            return self.config.ent_coef
        progress = min(1.0, max(0.0, self.global_step / max(1, self.config.total_timesteps)))
        start = self.config.ent_coef
        end = self.config.ent_coef_end
        return float(start + (end - start) * progress)

    def _update_policy(self, rollout, advantages, returns):
        b_obs = rollout["obs"].reshape((-1, OBS_SIZE))
        b_actions = rollout["actions"].reshape(-1)
        b_logprobs = rollout["logprobs"].reshape(-1)
        b_values = rollout["values"].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_size = b_obs.shape[0]
        minibatch_size = max(1, batch_size // self.config.num_minibatches)
        indices = np.arange(batch_size)

        clipfracs = []
        last_pg_loss = last_v_loss = last_entropy = last_kl = 0.0
        # snapshot do ent_coef (uma vez por update) — mantém consistência
        # entre minibatches do mesmo update.
        ent_coef_current = self._current_ent_coef()

        for _epoch in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                    b_obs[mb_idx], action=b_actions[mb_idx]
                )
                log_ratio = new_logprob - b_logprobs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    last_kl = float(((ratio - 1) - log_ratio).mean().item())
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_idx]
                if mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value_flat = new_value.reshape(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_value_flat - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value_flat - b_values[mb_idx],
                        -self.config.clip_coef, self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value_flat - b_returns[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_current * entropy_loss + self.config.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())

            if self.config.target_kl is not None and last_kl > self.config.target_kl:
                break

        y_pred = b_values.detach().numpy()
        y_true = b_returns.detach().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        return {
            "policy_loss": last_pg_loss,
            "value_loss": last_v_loss,
            "entropy_loss": last_entropy,
            "approx_kl": last_kl,
            "clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": explained_var,
            "ent_coef": ent_coef_current,
        }

    def _capture_and_save_replay(
        self,
        suffix: str = "",
        phase: Optional[str] = None,
    ) -> None:
        """roda 1 episódio em eval_env e grava replay JSON.

        suffix opcional pra distinguir replays gerados em transição de fase
        dos regulares. phase opcional — quando setado, força o eval_env a
        usar essa fase em vez de _current_phase() (útil pra capturar
        estado final da fase recém-terminada).
        """
        eval_env = SoccerEnv(seed=self.config.seed ^ self.global_step, capture_replay=True)
        if self.opponent_manager is not None:
            eval_env.set_opponent_policy(self.opponent_manager.sample_opponent_callable(self.policy))

        progress = min(1.0, self.global_step / max(1, self.config.total_timesteps))
        if self.config.enable_curriculum_phases:
            eval_phase = phase if phase is not None else self._current_phase()
            eval_env.set_progress(progress, phase=eval_phase)
        else:
            eval_phase = None
            eval_env.set_progress(progress)

        obs, _ = eval_env.reset(options={"progress": progress})

        terminated = truncated = False
        while not (terminated or truncated):
            obs_norm = self.obs_rms.normalize(obs[None])[0] if self.obs_rms else obs
            # greedy (argmax) em vez de stochastic — frontend (JS) também
            # usa argmax, mantendo replay e Live consistentes.
            action = self.policy.select_action_greedy(obs_norm.astype(np.float32))
            obs, _, terminated, truncated, _ = eval_env.step(action)

        fname = f"replay_{self.global_step}"
        if suffix:
            fname += f"_{suffix}"
        fname += ".json"
        replay_path = self.run_dir / "replays" / fname
        from backend.ai.env import REPLAY_FPS

        payload = {
            "metadata": {
                "global_step": self.global_step,
                "trained_team": eval_env.trained_team,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "frame_count": eval_env.frame_count,
                # frontend lê esse valor de metadata pra reproduzir em real-time.
                "fps": REPLAY_FPS,
                "phase": eval_phase,
            },
            "frames": eval_env.get_replay(),
        }
        with open(replay_path, "w") as f:
            json.dump(payload, f)

    def save_checkpoint(self, path: Path) -> None:
        state = {
            "global_step": self.global_step,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "rew_rms": self.rew_rms.state_dict() if self.rew_rms else None,
            "config": asdict(self.config),
            "run_name": self.run_name,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, weights_only=False, map_location=self.config.device)
        self.global_step = state["global_step"]
        self.policy.load_state_dict(state["policy_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state["obs_rms"] is not None and self.obs_rms is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state["rew_rms"] is not None and self.rew_rms is not None:
            self.rew_rms.load_state_dict(state["rew_rms"])

    def train(self) -> None:
        self.logger.info(
            f"Iniciando treino. {self.config.num_updates} updates esperados."
        )

        next_obs_np, info = self.envs.reset(seed=self.config.seed)

        self._set_opponents_for_envs([True] * self.config.num_envs)
        self._set_progress_for_envs([True] * self.config.num_envs)
        for sub in self._extract_reset_info_per_env(info):
            self.env_param_tracker.add_episode(sub)

        next_obs = torch.from_numpy(self._normalize_obs(next_obs_np.astype(np.float32)))
        next_terminated = torch.zeros(self.config.num_envs, dtype=torch.float32)

        update_idx = 0
        last_save_step = 0

        try:
            while self.global_step < self.config.total_timesteps:
                if self.config.anneal_lr:
                    frac = max(0.0, 1.0 - update_idx / self.config.num_updates)
                    self.optimizer.param_groups[0]["lr"] = frac * self.config.learning_rate

                rollout = self._collect_rollout(next_obs, next_terminated)
                advantages, returns = compute_gae(
                    rollout["rewards"], rollout["values"], rollout["terminated"],
                    rollout["next_value"], rollout["next_terminated"],
                    self.config.gamma, self.config.gae_lambda,
                )
                metrics = self._update_policy(rollout, advantages, returns)

                next_obs = rollout["last_obs"]
                next_terminated = rollout["next_terminated"]
                update_idx += 1

                if update_idx % self.config.log_frequency_updates == 0:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"ppo/{k}", v, self.global_step)
                    self.writer.add_scalar(
                        "ppo/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )
                    for k, v in self.env_param_tracker.to_summary().items():
                        self.writer.add_scalar(f"env_params/{k}", v, self.global_step)

                if (
                    update_idx % self.config.log_to_stdout_every_n_updates == 0
                    or update_idx == self.config.num_updates
                ):
                    self.logger.info(
                        f"step={self.global_step:>10,} "
                        f"upd={update_idx:>4}/{self.config.num_updates} "
                        f"v_loss={metrics['value_loss']:7.3f} "
                        f"p_loss={metrics['policy_loss']:+.4f} "
                        f"H={metrics['entropy_loss']:.3f} "
                        f"kl={metrics['approx_kl']:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                # replay automático em transição de fase. captura imediatamente
                # quando _current_phase() muda, sem esperar o próximo save_steps.
                # garante visualização nos pontos críticos.
                # a captura usa a fase recém-terminada (estado polido), em vez
                # da fase nova (primeiro tropeço).
                if self.config.enable_curriculum_phases:
                    current_phase = self._current_phase()
                    if (
                        self._last_captured_phase is not None
                        and current_phase != self._last_captured_phase
                    ):
                        finished_phase = self._last_captured_phase
                        self.logger.info(
                            f"Transição de fase: "
                            f"{finished_phase} → {current_phase}. "
                            f"Capturando estado polido de {finished_phase} "
                            f"em step={self.global_step:,}."
                        )
                        self._capture_and_save_replay(
                            suffix=f"phase_polished_{finished_phase}",
                            phase=finished_phase,
                        )
                    self._last_captured_phase = current_phase

                if self.global_step - last_save_step >= self.config.save_steps:
                    ckpt_path = self.run_dir / "checkpoints" / f"checkpoint_{self.global_step}.pt"
                    self.logger.info(
                        f"Salvando checkpoint @ step={self.global_step:,}: {ckpt_path.name}"
                    )
                    self.save_checkpoint(ckpt_path)
                    self._capture_and_save_replay()
                    if self.opponent_manager is not None:
                        self.opponent_manager.add_checkpoint(self.policy)
                        # async self-play: rewrite pool em disco pra workers
                        # carregarem a versão mais recente da Policy + pool.
                        if self._should_use_async_with_self_play():
                            from backend.ai.self_play import serialize_pool_to_disk
                            serialize_pool_to_disk(
                                self.async_sync_dir,
                                self.policy,
                                list(self.opponent_manager._pool),
                            )
                    last_save_step = self.global_step
        finally:
            final_path = self.run_dir / "checkpoints" / "final.pt"
            self.save_checkpoint(final_path)
            self.writer.close()
            self.envs.close()
            self.logger.info(f"Treino concluído. final.pt em {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO trainer para AI Soccer")
    parser.add_argument(
        "--smoke-test", type=int, default=None,
        help="0, 1, 2 ou 3 — config predefinida.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override do total_timesteps. Aplicado APÓS smoke-test se ambos definidos.",
    )
    parser.add_argument(
        "--name", type=str, default="default",
        help="Nome do run (sufixado com timestamp).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config-json", type=Path, default=None,
        help="Carrega config de JSON (sobrescreve smoke-test).",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Não publicar automaticamente o run no frontend após o treino.",
    )
    parser.add_argument(
        "--frontend-data", type=Path, default=Path("frontend/data"),
        help="Pasta destino do auto-publish (default: frontend/data).",
    )
    args = parser.parse_args()

    if args.config_json is not None:
        with open(args.config_json) as f:
            config = PPOConfig(**json.load(f))
    elif args.smoke_test is not None:
        smokes = _smoke_test_configs()
        if args.smoke_test not in smokes:
            raise ValueError(f"smoke-test inválido: {args.smoke_test}; opções={list(smokes)}")
        config = smokes[args.smoke_test]
    else:
        config = PPOConfig()

    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    config.seed = args.seed

    trainer = PPOTrainer(config, run_name=args.name)
    trainer.train()

    # auto-publish pós-treino. roda apenas se train() retornou sem crash e
    # final.pt foi gravado. falhas no publish não derrubam o processo
    # (treino já está em disco). skipável via --no-publish.
    final_ckpt = trainer.run_dir / "checkpoints" / "final.pt"
    if not args.no_publish and final_ckpt.exists():
        from backend.ai.publish_run import publish_run
        try:
            args.frontend_data.mkdir(parents=True, exist_ok=True)
            publish_run(trainer.run_dir, args.frontend_data)
            print(
                f"\n🌐 Run publicado automaticamente. Pra visualizar:"
                f"\n   cd frontend && python -m http.server 8000"
                f"\n   Acesse: http://localhost:8000"
            )
        except Exception as e:
            print(
                f"\n⚠️  Auto-publish falhou ({e}). Treino concluído com sucesso. "
                f"Pra publicar manualmente:"
                f"\n   python -m backend.ai.publish_run --run-dir {trainer.run_dir}"
            )


if __name__ == "__main__":
    main()
