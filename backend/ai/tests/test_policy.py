"""testes unitários da Policy canônica.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_policy.py -v
"""

import numpy as np
import pytest
import torch

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import HIDDEN_SIZE, Policy


def test_forward_single_input_shapes():
    torch.manual_seed(0)
    policy = Policy()
    x = torch.randn(1, OBS_SIZE)
    action, log_prob, entropy, value = policy.get_action_and_value(x)
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1, 1)


def test_forward_batch_input_shapes():
    torch.manual_seed(0)
    policy = Policy()
    x = torch.randn(32, OBS_SIZE)
    action, log_prob, entropy, value = policy.get_action_and_value(x)
    assert action.shape == (32,)
    assert log_prob.shape == (32,)
    assert entropy.shape == (32,)
    assert value.shape == (32, 1)


def test_get_value_returns_b_by_one_shape():
    torch.manual_seed(0)
    policy = Policy()
    x = torch.randn(8, OBS_SIZE)
    v = policy.get_value(x)
    assert v.shape == (8, 1)


def test_get_action_and_value_no_action_returns_finite_quadruple():
    torch.manual_seed(0)
    policy = Policy()
    x = torch.randn(4, OBS_SIZE)
    action, log_prob, entropy, value = policy.get_action_and_value(x)

    assert action.dtype in (torch.int64, torch.long)
    assert torch.all(action >= 0) and torch.all(action < ACTION_SPACE_SIZE)

    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(torch.isfinite(entropy))
    assert torch.all(entropy > 0), "entropia categórica deve ser positiva"
    assert torch.all(torch.isfinite(value))


def test_get_action_and_value_with_given_action_returns_consistent_log_prob():
    """round-trip: sample uma action; passar a mesma deve retornar mesmo log_prob."""
    torch.manual_seed(0)
    policy = Policy()
    x = torch.randn(4, OBS_SIZE)
    action_a, log_prob_a, _, _ = policy.get_action_and_value(x)
    _, log_prob_b, _, _ = policy.get_action_and_value(x, action=action_a)
    assert torch.allclose(log_prob_a, log_prob_b)


def test_select_action_greedy_single_returns_python_int():
    torch.manual_seed(0)
    policy = Policy()
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    a = policy.select_action_greedy(obs)
    assert isinstance(a, int)
    assert 0 <= a < ACTION_SPACE_SIZE


def test_select_action_greedy_batch_returns_ndarray():
    torch.manual_seed(0)
    policy = Policy()
    obs = np.random.randn(8, OBS_SIZE).astype(np.float32)
    actions = policy.select_action_greedy(obs)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (8,)
    assert np.all(actions >= 0) and np.all(actions < ACTION_SPACE_SIZE)


def test_select_action_stochastic_can_yield_different_actions():
    """100 amostras com mesma obs devem produzir ≥2 actions distintas.

    init ortogonal + std=0.01 ⇒ distribuição ~uniforme nos logits.
    """
    torch.manual_seed(0)
    policy = Policy()
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    seen = {policy.select_action_stochastic(obs) for _ in range(100)}
    assert len(seen) >= 2, f"select_action_stochastic só amostrou {seen}"


def test_select_action_helpers_do_not_accumulate_gradients():
    torch.manual_seed(0)
    policy = Policy()
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    for _ in range(20):
        policy.select_action_greedy(obs)
        policy.select_action_stochastic(obs)
    for param in policy.parameters():
        assert param.grad is None


def test_two_policies_with_same_seed_produce_same_output():
    def _build():
        torch.manual_seed(42)
        return Policy()

    p1 = _build()
    p2 = _build()
    x = torch.randn(4, OBS_SIZE)
    v1 = p1.get_value(x)
    v2 = p2.get_value(x)
    assert torch.allclose(v1, v2)


def test_total_parameter_count_is_in_expected_range():
    """fc1 + fc2 + policy_head + value_head ≈ 27,283 — dentro de [20k, 30k]."""
    policy = Policy()
    n_params = sum(p.numel() for p in policy.parameters())
    assert 20_000 <= n_params <= 30_000, f"n_params = {n_params:,}"


def test_hidden_layers_have_correct_size():
    policy = Policy()
    assert policy.fc1.in_features == OBS_SIZE
    assert policy.fc1.out_features == HIDDEN_SIZE
    assert policy.fc2.in_features == HIDDEN_SIZE
    assert policy.fc2.out_features == HIDDEN_SIZE
    assert policy.policy_head.out_features == ACTION_SPACE_SIZE
    assert policy.value_head.out_features == 1
