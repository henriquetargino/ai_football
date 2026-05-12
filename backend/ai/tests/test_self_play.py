"""testes unitários do OpponentManager + helpers de self-play.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_self_play.py -v
"""

import numpy as np
import pytest
import torch

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.env import SoccerEnv
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import Policy
from backend.ai.self_play import (
    LATEST_RATIO,
    POOL_RATIO,
    POOL_SIZE,
    RANDOM_RATIO,
    OpponentManager,
    make_opponent_callable_from_policy,
    random_opponent_callable,
)


def _classify_callable(cb, latest_policy: Policy) -> str:
    """inspeciona o __closure__ do callable para classificá-lo.

    random callable: closure vazio. latest callable: alguma cell é a
    latest_policy. pool callable: alguma cell é uma Policy distinta da latest.

    torch.no_grad() decorator envolve a função, então a closure com a
    Policy fica em cb.__wrapped__.__closure__. checamos os dois níveis.
    """
    candidates = [cb]
    inner = getattr(cb, "__wrapped__", None)
    if inner is not None:
        candidates.append(inner)

    for candidate in candidates:
        closure = getattr(candidate, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, Policy):
                return "latest" if val is latest_policy else "pool"
    return "random"


def test_new_manager_has_empty_pool():
    manager = OpponentManager()
    assert manager.pool_size == 0


def test_add_checkpoint_increments_pool_size():
    manager = OpponentManager()
    pol = Policy()
    manager.add_checkpoint(pol)
    assert manager.pool_size == 1
    manager.add_checkpoint(pol)
    assert manager.pool_size == 2


def test_pool_is_fifo_capped_at_pool_size():
    manager = OpponentManager()
    for _ in range(POOL_SIZE + 5):
        manager.add_checkpoint(Policy())
    assert manager.pool_size == POOL_SIZE


def test_add_checkpoint_does_deepcopy():
    """mutar a current_policy depois do snapshot não altera o checkpoint."""
    torch.manual_seed(0)
    pol = Policy()
    manager = OpponentManager()
    manager.add_checkpoint(pol)
    snapshot = manager.get_pool_snapshot()[0]

    saved_first_param = next(snapshot.parameters()).data.clone()
    with torch.no_grad():
        for p in pol.parameters():
            p.data.add_(1.0)

    new_first_param = next(snapshot.parameters()).data
    assert torch.allclose(new_first_param, saved_first_param), (
        "snapshot foi alterado quando current_policy mudou — deepcopy quebrado"
    )


def test_snapshot_has_requires_grad_false():
    pol = Policy()
    manager = OpponentManager()
    manager.add_checkpoint(pol)
    snapshot = manager.get_pool_snapshot()[0]
    for param in snapshot.parameters():
        assert not param.requires_grad


def test_sample_with_empty_pool_does_not_raise():
    pol = Policy()
    manager = OpponentManager(seed=0)
    cb = manager.sample_opponent_callable(pol)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    a = cb(obs)
    assert isinstance(a, int)
    assert 0 <= a < ACTION_SPACE_SIZE


def test_sample_distribution_approximates_60_20_20():
    """2000 samples com pool cheio devem se aproximar de 60/20/20 (±5%)."""
    torch.manual_seed(0)
    latest = Policy()
    manager = OpponentManager(seed=42)
    for _ in range(POOL_SIZE):
        manager.add_checkpoint(Policy())

    n = 2000
    counts = {"latest": 0, "pool": 0, "random": 0}
    for _ in range(n):
        cb = manager.sample_opponent_callable(latest)
        counts[_classify_callable(cb, latest)] += 1

    assert counts["latest"] / n == pytest.approx(LATEST_RATIO, abs=0.05)
    assert counts["pool"] / n == pytest.approx(POOL_RATIO, abs=0.05)
    assert counts["random"] / n == pytest.approx(RANDOM_RATIO, abs=0.05)


def test_sampled_callable_returns_valid_action():
    pol = Policy()
    manager = OpponentManager(seed=0)
    manager.add_checkpoint(pol)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    for _ in range(50):
        cb = manager.sample_opponent_callable(pol)
        a = cb(obs)
        assert isinstance(a, int)
        assert 0 <= a < ACTION_SPACE_SIZE


def test_managers_with_same_seed_produce_same_branch_sequence():
    """dois managers com seed igual sorteiam a mesma sequência de branches."""
    pol = Policy()
    m_a = OpponentManager(seed=42)
    m_b = OpponentManager(seed=42)
    for _ in range(POOL_SIZE):
        m_a.add_checkpoint(Policy())
        m_b.add_checkpoint(Policy())

    seq_a = [_classify_callable(m_a.sample_opponent_callable(pol), pol) for _ in range(50)]
    seq_b = [_classify_callable(m_b.sample_opponent_callable(pol), pol) for _ in range(50)]
    assert seq_a == seq_b


def test_random_opponent_callable_produces_valid_actions():
    cb = random_opponent_callable()
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    actions = [cb(obs) for _ in range(1000)]
    assert all(0 <= a < ACTION_SPACE_SIZE for a in actions)
    assert len(set(actions)) > 1


def test_callable_does_not_accumulate_gradients_on_eval_policy():
    pol = Policy()
    pol.eval()
    cb = make_opponent_callable_from_policy(pol)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    for _ in range(100):
        cb(obs)
    for param in pol.parameters():
        assert param.grad is None


def test_callable_does_not_accumulate_gradients_even_on_train_policy():
    pol = Policy()
    pol.train()
    cb = make_opponent_callable_from_policy(pol)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    for _ in range(100):
        cb(obs)
    for param in pol.parameters():
        assert param.grad is None


def test_deterministic_callable_returns_same_action_for_same_obs():
    torch.manual_seed(0)
    pol = Policy()
    cb = make_opponent_callable_from_policy(pol, deterministic=True)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    actions = {cb(obs) for _ in range(10)}
    assert len(actions) == 1, f"determinístico mas amostrou {actions}"


def test_stochastic_callable_yields_varied_actions_across_calls():
    torch.manual_seed(0)
    pol = Policy()
    cb = make_opponent_callable_from_policy(pol, deterministic=False)
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    actions = {cb(obs) for _ in range(100)}
    assert len(actions) >= 2, f"estocástico mas só amostrou {actions}"


def test_integration_manager_policy_env_one_step():
    """smoke: Policy + OpponentManager + SoccerEnv encadeados sem erro."""
    torch.manual_seed(0)
    train_policy = Policy()
    manager = OpponentManager(seed=0)
    manager.add_checkpoint(train_policy)

    env = SoccerEnv(seed=0)
    opp = manager.sample_opponent_callable(train_policy)
    env.set_opponent_policy(opp)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)

    obs, r, term, trunc, info = env.step(8)
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
