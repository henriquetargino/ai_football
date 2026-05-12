"""testes da sincronização async self-play via disco.

cobre:
    serialize_pool_to_disk + load_pool_from_disk (round-trip + cache).
    SoccerEnv.set_opponent_spec / _indexed.
    fallback gracioso quando disco está vazio/corrompido.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_async_self_play.py -v
"""

import json
import time

import numpy as np
import pytest
import torch

from backend.ai.env import SoccerEnv
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import Policy
from backend.ai.self_play import (
    ASYNC_MANIFEST_NAME,
    load_pool_from_disk,
    serialize_pool_to_disk,
)


def test_serialize_load_round_trip(tmp_path):
    """serializar pool e carregar deve dar Policies equivalentes."""
    sync_dir = tmp_path / "_async_pool"

    torch.manual_seed(0)
    latest = Policy()
    pool = [Policy(), Policy(), Policy()]

    for i, p in enumerate(pool):
        for param in p.parameters():
            param.data.fill_(float(i + 1))

    serialize_pool_to_disk(sync_dir, latest, pool)

    assert (sync_dir / ASYNC_MANIFEST_NAME).exists()
    assert (sync_dir / "policy_latest.pt").exists()
    for i in range(3):
        assert (sync_dir / f"pool_{i}.pt").exists()

    result = load_pool_from_disk(sync_dir, cached_version=None)
    assert result is not None
    new_version, loaded_latest, loaded_pool = result
    assert isinstance(new_version, int)
    assert len(loaded_pool) == 3

    for i, p in enumerate(loaded_pool):
        for param in p.parameters():
            assert torch.all(param.data == float(i + 1))


def test_load_returns_none_if_no_change(tmp_path):
    """se manifest version igual ao cached, load retorna None."""
    sync_dir = tmp_path / "_async_pool"
    serialize_pool_to_disk(sync_dir, Policy(), [])

    result1 = load_pool_from_disk(sync_dir, cached_version=None)
    assert result1 is not None
    cached_version, _, _ = result1

    result2 = load_pool_from_disk(sync_dir, cached_version=cached_version)
    assert result2 is None


def test_load_returns_new_after_rewrite(tmp_path):
    """rewrite muda version → load retorna nova versão."""
    sync_dir = tmp_path / "_async_pool"

    serialize_pool_to_disk(sync_dir, Policy(), [])
    result1 = load_pool_from_disk(sync_dir, cached_version=None)
    assert result1 is not None
    v1, _, _ = result1

    time.sleep(0.005)
    serialize_pool_to_disk(sync_dir, Policy(), [Policy()])

    result2 = load_pool_from_disk(sync_dir, cached_version=v1)
    assert result2 is not None
    v2, _, pool2 = result2
    assert v2 > v1
    assert len(pool2) == 1


def test_load_handles_missing_manifest(tmp_path):
    """sync dir sem manifest → load retorna None graciosamente."""
    sync_dir = tmp_path / "_async_pool"
    sync_dir.mkdir()
    result = load_pool_from_disk(sync_dir, cached_version=None)
    assert result is None


def test_load_handles_corrupted_manifest(tmp_path):
    """manifest com JSON inválido → load retorna None."""
    sync_dir = tmp_path / "_async_pool"
    sync_dir.mkdir()
    (sync_dir / ASYNC_MANIFEST_NAME).write_text("not valid json {{{")
    torch.save(Policy().state_dict(), sync_dir / "policy_latest.pt")

    result = load_pool_from_disk(sync_dir, cached_version=None)
    assert result is None


def test_serialize_cleans_old_pool_files(tmp_path):
    """pool encolheu → arquivos antigos são removidos."""
    sync_dir = tmp_path / "_async_pool"
    serialize_pool_to_disk(sync_dir, Policy(), [Policy() for _ in range(5)])
    assert (sync_dir / "pool_4.pt").exists()

    serialize_pool_to_disk(sync_dir, Policy(), [Policy() for _ in range(2)])
    assert (sync_dir / "pool_0.pt").exists()
    assert (sync_dir / "pool_1.pt").exists()
    assert not (sync_dir / "pool_2.pt").exists()
    assert not (sync_dir / "pool_4.pt").exists()


def _obs():
    return np.zeros(OBS_SIZE, dtype=np.float32)


def test_env_set_opponent_spec_random(tmp_path):
    env = SoccerEnv(seed=42, async_sync_dir=tmp_path)
    env.set_opponent_spec("random")
    a = env.opponent_policy(_obs())
    assert isinstance(a, int)
    assert 0 <= a < 18


def test_env_set_opponent_spec_latest_after_serialize(tmp_path):
    sync_dir = tmp_path / "_async_pool"
    serialize_pool_to_disk(sync_dir, Policy(), [])

    env = SoccerEnv(seed=42, async_sync_dir=sync_dir)
    env.set_opponent_spec("latest")

    a = env.opponent_policy(_obs())
    assert 0 <= int(a) < 18


def test_env_set_opponent_spec_latest_fallback_when_no_disk(tmp_path):
    """se disco vazio, spec='latest' faz fallback gracioso pra random."""
    env = SoccerEnv(seed=42, async_sync_dir=tmp_path / "nonexistent")
    env.set_opponent_spec("latest")
    a = env.opponent_policy(_obs())
    assert 0 <= int(a) < 18


def test_env_set_opponent_spec_pool(tmp_path):
    sync_dir = tmp_path / "_async_pool"
    pool = [Policy(), Policy()]
    serialize_pool_to_disk(sync_dir, Policy(), pool)

    env = SoccerEnv(seed=42, async_sync_dir=sync_dir)
    env.set_opponent_spec(("pool", 1))
    a = env.opponent_policy(_obs())
    assert 0 <= int(a) < 18


def test_env_set_opponent_spec_pool_out_of_range_falls_back(tmp_path):
    """spec=('pool', idx) com idx fora de range → random fallback."""
    sync_dir = tmp_path / "_async_pool"
    serialize_pool_to_disk(sync_dir, Policy(), [Policy()])

    env = SoccerEnv(seed=42, async_sync_dir=sync_dir)
    env.set_opponent_spec(("pool", 99))
    a = env.opponent_policy(_obs())
    assert 0 <= int(a) < 18


def test_env_caches_pool_avoids_reloading(tmp_path):
    """set spec 2× consecutivos: 2ª chamada não recarrega disco."""
    sync_dir = tmp_path / "_async_pool"
    serialize_pool_to_disk(sync_dir, Policy(), [Policy()])

    env = SoccerEnv(seed=42, async_sync_dir=sync_dir)
    env.set_opponent_spec("latest")
    v1 = env._cached_version

    env.set_opponent_spec("latest")
    v2 = env._cached_version

    assert v1 == v2


def test_env_idx_dispatch(tmp_path):
    """set_opponent_spec_indexed usa _env_idx pra selecionar."""
    env = SoccerEnv(seed=42)
    env._env_idx = 2
    env.set_opponent_spec_indexed(["random", "random", "random", None])
    a = env.opponent_policy(_obs())
    assert 0 <= int(a) < 18


def test_env_idx_dispatch_none_keeps_current(tmp_path):
    """spec=None na posição do env → opponent não muda."""
    env = SoccerEnv(seed=42)
    env._env_idx = 0
    initial_callable = env.opponent_policy
    env.set_opponent_spec_indexed([None, "random"])
    assert env.opponent_policy is initial_callable
