"""gerenciador de oponentes para self-play.

sample por episódio (não por step), com mistura 60/20/20:
    60% — latest policy (a que está treinando ativamente)
    20% — sample uniforme do pool histórico (FIFO 10 checkpoints)
    20% — policy uniforme aleatória

a fatia de 20% aleatória mitiga competitive overfitting silencioso de
self-play simétrico (Singh 2024, "Territory Paint Wars").

decisões cravadas:
    - pool size = 10 (FIFO).
    - trainer chama add_checkpoint a cada 500k steps.
    - trainer chama sample_opponent_callable a cada reset() do env.
      trocar opponent dentro do episódio rebaixa convergência.
    - snapshots no pool são deepcopy + requires_grad_(False) — gradients
      NUNCA fluem pra checkpoints históricos.
    - latest passa por referência (sem clone); torch.no_grad embutido no callable.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Optional

import numpy as np
import torch

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.policy import Policy


POOL_SIZE: int = 10
LATEST_RATIO: float = 0.60
POOL_RATIO: float = 0.20
RANDOM_RATIO: float = 0.20

ASYNC_SYNC_DIR_NAME: str = "_async_pool"
ASYNC_MANIFEST_NAME: str = "manifest.json"

OpponentCallable = Callable[[np.ndarray], int]


def random_opponent_callable() -> OpponentCallable:
    """cria callable que retorna ação uniforme aleatória ∈ [0, 17]."""
    def _policy(obs: np.ndarray) -> int:
        return random.randint(0, ACTION_SPACE_SIZE - 1)
    return _policy


def make_opponent_callable_from_policy(
    policy: Policy,
    deterministic: bool = False,
) -> OpponentCallable:
    """adapta uma Policy para o callable que o env consome.

    torch.no_grad é embutido aqui, então gradients nunca acumulam.
    deterministic=True usa argmax; False (default) amostra da Categorical
    para manter variabilidade mesmo em opponent congelado.
    """
    if deterministic:
        @torch.no_grad()
        def _policy(obs: np.ndarray) -> int:
            return policy.select_action_greedy(obs)
    else:
        @torch.no_grad()
        def _policy(obs: np.ndarray) -> int:
            return policy.select_action_stochastic(obs)
    return _policy


class OpponentManager:
    """gerencia o pool de policies congeladas para self-play."""

    def __init__(self, seed: Optional[int] = None):
        self._pool: Deque[Policy] = deque(maxlen=POOL_SIZE)
        self._rng = random.Random(seed)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def add_checkpoint(self, current_policy: Policy) -> None:
        """deepcopy de current_policy adicionado ao pool com gradientes desligados.

        pool cheio descarta o mais antigo. o parâmetro recebido NÃO é
        congelado — apenas o snapshot copiado tem requires_grad_(False).
        """
        snapshot = copy.deepcopy(current_policy)
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad_(False)
        self._pool.append(snapshot)

    def sample_opponent_callable(
        self,
        latest_policy: Policy,
        deterministic: bool = False,
    ) -> OpponentCallable:
        """sample de opponent para o próximo episódio.

        60% latest_policy, 20% pool aleatório (fallback latest se vazio),
        20% policy aleatória.
        """
        r = self._rng.random()
        if r < LATEST_RATIO:
            return make_opponent_callable_from_policy(latest_policy, deterministic)
        elif r < LATEST_RATIO + POOL_RATIO:
            if len(self._pool) == 0:
                return make_opponent_callable_from_policy(latest_policy, deterministic)
            sampled = self._rng.choice(list(self._pool))
            return make_opponent_callable_from_policy(sampled, deterministic)
        else:
            return random_opponent_callable()

    def get_pool_snapshot(self) -> list[Policy]:
        """cópias profundas do pool atual, seguras de mutar."""
        return [copy.deepcopy(p) for p in self._pool]


# sync via disco — workers AsyncVectorEnv carregam Policy de arquivo em vez
# de receber callables (que não atravessam pipes de multiprocessing).

def serialize_pool_to_disk(
    sync_dir: Path,
    latest_policy: Policy,
    pool: list[Policy],
) -> None:
    """escreve latest + pool em sync_dir atomicamente.

    ordem: latest → pool → limpa antigos → manifest por último (tmp+rename,
    atômico em POSIX). workers só consideram o pool válido se manifest existe.
    """
    sync_dir = Path(sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)

    latest_path = sync_dir / "policy_latest.pt"
    torch.save(latest_policy.state_dict(), latest_path)

    for i, pool_policy in enumerate(pool):
        torch.save(pool_policy.state_dict(), sync_dir / f"pool_{i}.pt")

    for f in sync_dir.glob("pool_*.pt"):
        try:
            idx = int(f.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        if idx >= len(pool):
            try:
                f.unlink()
            except OSError:
                pass

    manifest = {
        "version": int(time.time() * 1000),
        "pool_count": len(pool),
        "ts": time.time(),
    }
    manifest_tmp = sync_dir / "manifest.tmp.json"
    manifest_path = sync_dir / ASYNC_MANIFEST_NAME
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f)
    manifest_tmp.replace(manifest_path)


def load_pool_from_disk(
    sync_dir: Path,
    cached_version: Optional[int],
) -> Optional[tuple[int, Policy, list[Policy]]]:
    """lê pool do disco se manifest mudou desde cached_version.

    retorna None se manifest ausente, corrompido, versão não mudou ou
    falha de carregamento. em qualquer falha, worker mantém o cache antigo
    (graceful degradation).
    """
    sync_dir = Path(sync_dir)
    manifest_path = sync_dir / ASYNC_MANIFEST_NAME
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        return None

    new_version = manifest.get("version")
    if new_version is None:
        return None
    if cached_version is not None and new_version <= cached_version:
        return None

    latest_path = sync_dir / "policy_latest.pt"
    if not latest_path.exists():
        return None

    try:
        latest = Policy()
        state = torch.load(latest_path, map_location="cpu", weights_only=True)
        latest.load_state_dict(state)
        latest.eval()
        for p in latest.parameters():
            p.requires_grad_(False)
    except (RuntimeError, IOError, OSError, KeyError):
        return None

    pool_count = int(manifest.get("pool_count", 0))
    pool: list[Policy] = []
    for i in range(pool_count):
        pool_path = sync_dir / f"pool_{i}.pt"
        if not pool_path.exists():
            continue
        try:
            p = Policy()
            state = torch.load(pool_path, map_location="cpu", weights_only=True)
            p.load_state_dict(state)
            p.eval()
            for param in p.parameters():
                param.requires_grad_(False)
            pool.append(p)
        except (RuntimeError, IOError, OSError, KeyError):
            continue

    return (new_version, latest, pool)
