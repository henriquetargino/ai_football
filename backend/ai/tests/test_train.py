"""Testes unitários para o PPO trainer.

Rodar (a partir da raiz do projeto):

    pytest backend/ai/tests/test_train.py -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ai.obs import OBS_SIZE
from backend.ai.train import (
    PPOConfig,
    PPOTrainer,
    RunningMeanStd,
    compute_gae,
)


# helpers

def _micro_config(tmp_path: Path, **overrides) -> PPOConfig:
    """Config mínimo viável pra rodar 1 update sem custar muito."""
    base = dict(
        total_timesteps=16,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        save_steps=1_000_000_000,  # gigante: nenhum save intermediário
        fixed_random_opponent=True,
        enable_self_play=False,
        enable_curriculum=False,
        force_sync_vec_env=True,
        log_frequency_updates=1,
        seed=0,
    )
    base.update(overrides)
    return PPOConfig(**base)


@pytest.fixture(autouse=True)
def _chdir_to_tmp(tmp_path, monkeypatch):
    """Cada teste roda num cwd isolado pra que ``data/runs/`` seja descartável."""
    monkeypatch.chdir(tmp_path)


# categoria A — Componentes

def test_running_mean_std_accepts_batch_input():
    rms = RunningMeanStd((OBS_SIZE,))
    rms.update(np.random.randn(32, OBS_SIZE).astype(np.float32))
    # não levanta; mean/var têm shape correto
    assert rms.mean.shape == (OBS_SIZE,)
    assert rms.var.shape == (OBS_SIZE,)


def test_running_mean_std_converges_to_data_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(10_000,)).astype(np.float64)
    rms = RunningMeanStd(())
    rms.update(data)
    # tolerância: 10k amostras → erro ~ scale/sqrt(N) = 0.02
    assert abs(rms.mean - 5.0) < 0.1
    assert abs(np.sqrt(rms.var) - 2.0) < 0.1


def test_running_mean_std_state_dict_round_trip():
    rms = RunningMeanStd((4,))
    rms.update(np.random.randn(100, 4).astype(np.float32))
    state = rms.state_dict()
    rms2 = RunningMeanStd((4,))
    rms2.load_state_dict(state)
    np.testing.assert_array_equal(rms.mean, rms2.mean)
    np.testing.assert_array_equal(rms.var, rms2.var)
    assert rms.count == rms2.count


def test_compute_gae_shape():
    T, N = 8, 4
    rewards = torch.zeros((T, N))
    values = torch.zeros((T, N))
    terminated = torch.zeros((T, N))
    next_value = torch.zeros(N)
    next_terminated = torch.zeros(N)
    advantages, returns = compute_gae(
        rewards, values, terminated, next_value, next_terminated, 0.99, 0.95
    )
    assert advantages.shape == (T, N)
    assert returns.shape == (T, N)


def test_compute_gae_zero_rewards_zero_values_yields_zero_advantages():
    T, N = 8, 2
    advantages, returns = compute_gae(
        torch.zeros((T, N)), torch.zeros((T, N)), torch.zeros((T, N)),
        torch.zeros(N), torch.zeros(N), 0.99, 0.95,
    )
    assert torch.allclose(advantages, torch.zeros((T, N)))
    assert torch.allclose(returns, torch.zeros((T, N)))


def test_compute_gae_terminated_zeros_bootstrap_truncated_does_not():
    """Cenário: 2 envs, mesma trajetória mas env 0 terminou (gol) e env 1 truncou.
    Bootstrap deve afetar SÓ env 0 (zerar) — env 1 mantém next_value normal."""
    T, N = 1, 2
    rewards = torch.tensor([[1.0, 1.0]])  # mesma reward
    values = torch.tensor([[0.0, 0.0]])
    terminated = torch.tensor([[1.0, 0.0]])  # env 0 terminou; env 1 não (truncou)
    next_value = torch.tensor([100.0, 100.0])  # bootstrap absurdo se mantido
    next_terminated = torch.tensor([1.0, 0.0])

    advantages, returns = compute_gae(
        rewards, values, terminated, next_value, next_terminated, 0.99, 0.95
    )
    # env 0 (terminated): advantage = reward + 0 - 0 = 1.0
    assert advantages[0, 0].item() == pytest.approx(1.0)
    # env 1 (truncated → bootstrap mantido): advantage = 1 + 0.99 * 100 - 0 = 100.0
    assert advantages[0, 1].item() == pytest.approx(1.0 + 0.99 * 100.0)


# categoria B — Pipeline integrada

def test_pipeline_runs_one_update_without_exception(tmp_path):
    config = _micro_config(tmp_path)
    trainer = PPOTrainer(config, run_name="micro_one_update")
    trainer.train()
    assert trainer.global_step >= config.batch_size
    assert (trainer.run_dir / "checkpoints" / "final.pt").exists()


def test_pipeline_writes_config_json(tmp_path):
    config = _micro_config(tmp_path)
    trainer = PPOTrainer(config, run_name="config_json")
    cfg_path = trainer.run_dir / "config.json"
    assert cfg_path.exists()
    with open(cfg_path) as f:
        loaded = json.load(f)
    assert loaded["total_timesteps"] == config.total_timesteps
    assert loaded["fixed_random_opponent"] == config.fixed_random_opponent


def test_save_load_checkpoint_round_trip(tmp_path):
    config = _micro_config(tmp_path)
    trainer = PPOTrainer(config, run_name="save_load")
    trainer.train()

    ckpt = trainer.run_dir / "checkpoints" / "final.pt"
    assert ckpt.exists()

    # criar trainer novo, carregar checkpoint
    config2 = _micro_config(tmp_path)
    trainer2 = PPOTrainer(config2, run_name="save_load_2")
    trainer2.load_checkpoint(ckpt)

    # state_dicts devem ser idênticos
    sd1 = trainer.policy.state_dict()
    sd2 = trainer2.policy.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"param {k} divergiu"
    assert trainer2.global_step == trainer.global_step


def test_self_play_flag_creates_opponent_manager(tmp_path):
    config = _micro_config(tmp_path,
        fixed_random_opponent=False,
        enable_self_play=True,
    )
    trainer = PPOTrainer(config, run_name="sp_on")
    assert trainer.opponent_manager is not None
    # opponent manager começa com 1 checkpoint (snapshot inicial)
    assert trainer.opponent_manager.pool_size == 1


def test_random_opponent_flag_skips_opponent_manager(tmp_path):
    config = _micro_config(tmp_path, fixed_random_opponent=True, enable_self_play=False)
    trainer = PPOTrainer(config, run_name="random_opp")
    assert trainer.opponent_manager is None


# categoria C — Smoke test 0 ABREVIADO

def test_smoke_0_abreviado_e2e(tmp_path):
    """Smoke 0 com total reduzido: pipeline + diretório + config + tensorboard."""
    config = _micro_config(
        tmp_path,
        total_timesteps=64,    # 2 updates com batch 32 (num_envs=2 × num_steps=16)
        num_envs=2,
        num_steps=16,
    )
    trainer = PPOTrainer(config, run_name="smoke_0_e2e")
    trainer.train()

    assert trainer.run_dir.exists()
    assert (trainer.run_dir / "config.json").exists()
    assert (trainer.run_dir / "tensorboard").exists()
    assert (trainer.run_dir / "checkpoints" / "final.pt").exists()
    # TensorBoard tem ao menos 1 arquivo de eventos
    tb_files = list((trainer.run_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert len(tb_files) >= 1


# categoria D — Determinismo

def test_two_trainers_same_seed_produce_same_initial_policy(tmp_path):
    config_a = _micro_config(tmp_path, seed=123)
    config_b = _micro_config(tmp_path, seed=123)
    trainer_a = PPOTrainer(config_a, run_name="det_a")
    trainer_b = PPOTrainer(config_b, run_name="det_b")
    sd_a = trainer_a.policy.state_dict()
    sd_b = trainer_b.policy.state_dict()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"param {k} divergiu antes do treino"


# categoria E — Logging UX

def test_trainer_logs_to_stdout(tmp_path, caplog):
    """Trainer emite mensagens informativas no logger 'PPOTrainer' — sem
    isso, ``--smoke-test`` parece travado pra quem está rodando."""
    import logging
    config = _micro_config(tmp_path, log_to_stdout_every_n_updates=1)
    with caplog.at_level(logging.INFO, logger="PPOTrainer"):
        trainer = PPOTrainer(config, run_name="caplog_test")
        trainer.train()

    messages = [r.message for r in caplog.records]
    assert any("Run dir" in m for m in messages)
    assert any("updates" in m.lower() for m in messages)
    assert any("conclu" in m.lower() for m in messages)
    # mensagem de update tem "step=" e "upd="
    assert any("step=" in m and "upd=" in m for m in messages)


# categoria F — Curriculum + EnvParamTracker fixes

def test_curriculum_progress_propagates_to_workers_sync(tmp_path):
    """Validar que set_progress chega aos workers em SyncVectorEnv."""
    config = _micro_config(
        tmp_path,
        total_timesteps=4096,
        num_envs=2,
        num_steps=256,
        enable_curriculum=True,
        force_sync_vec_env=True,
    )
    trainer = PPOTrainer(config, run_name="curriculum_sync")
    trainer.envs.reset(seed=0)

    progresses_before = trainer.envs.get_attr("_last_progress")
    assert all(abs(p - 0.5) < 1e-6 for p in progresses_before)

    trainer.global_step = trainer.config.total_timesteps  # progress = 1.0
    trainer._set_progress_for_envs([True, True])

    progresses_after = trainer.envs.get_attr("_last_progress")
    assert all(abs(p - 1.0) < 1e-6 for p in progresses_after), (
        f"set_progress não propagou em SyncVectorEnv: {progresses_after}"
    )

    trainer.envs.close()
    trainer.writer.close()


def test_curriculum_progress_propagates_to_workers_async(tmp_path):
    """Validar que set_progress chega aos workers em AsyncVectorEnv (smokes 0/1/2)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=4096,
        num_envs=2,
        num_steps=256,
        enable_curriculum=True,
        force_sync_vec_env=False,  # permite Async (default em fixed_random_opponent)
    )
    trainer = PPOTrainer(config, run_name="curriculum_async")
    trainer.envs.reset(seed=0)

    progresses_before = trainer.envs.get_attr("_last_progress")
    assert all(abs(p - 0.5) < 1e-6 for p in progresses_before), (
        f"_last_progress inicial inesperado: {progresses_before}"
    )

    trainer.global_step = trainer.config.total_timesteps
    trainer._set_progress_for_envs([True, True])

    progresses_after = trainer.envs.get_attr("_last_progress")
    assert all(abs(p - 1.0) < 1e-6 for p in progresses_after), (
        f"AsyncVectorEnv não recebeu set_progress: {progresses_after} "
        f"(BUG: smokes 0/1/2 não usariam curriculum!)"
    )

    trainer.envs.close()
    trainer.writer.close()


def test_env_param_tracker_grows_during_training(tmp_path):
    """Após o smoke, history_size do tracker deve ser maior que num_envs:
    o tracker registra params de auto-resets também, não só do reset inicial."""
    config = _micro_config(
        tmp_path,
        total_timesteps=8192,
        num_envs=4,
        num_steps=256,
        force_sync_vec_env=True,
    )
    trainer = PPOTrainer(config, run_name="param_tracker_grow")
    trainer.train()

    history_size = len(trainer.env_param_tracker.goal_widths)
    assert history_size > config.num_envs, (
        f"EnvParamTracker history_size={history_size} ≤ num_envs={config.num_envs}: "
        f"auto-resets durante o rollout não estão sendo capturados."
    )


# categoria G — V3: Curriculum multi-estágio (phase transitions)

def test_curriculum_phase_transitions_returns_strings(tmp_path):
    """V4: ``_current_phase`` retorna strings ("1A", "1B", "1C", "2", "3", "4", "5")."""
    config = _micro_config(
        tmp_path,
        total_timesteps=5_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=500_000,
        phase_1b_end_step=1_000_000,
        phase_1c_end_step=1_500_000,
        phase_2_end_step=2_500_000,
        phase_3_end_step=3_500_000,
        phase_4_end_step=4_500_000,
    )
    trainer = PPOTrainer(config, run_name="phase_str_test")

    trainer.global_step = 100_000
    assert trainer._current_phase() == "1A"
    trainer.global_step = 700_000
    assert trainer._current_phase() == "1B"
    trainer.global_step = 1_200_000
    assert trainer._current_phase() == "1C"
    trainer.global_step = 2_000_000
    assert trainer._current_phase() == "2"
    trainer.global_step = 3_000_000
    assert trainer._current_phase() == "3"
    trainer.global_step = 4_000_000
    assert trainer._current_phase() == "4"
    trainer.global_step = 4_800_000
    assert trainer._current_phase() == "5"

    trainer.envs.close()
    trainer.writer.close()


def test_curriculum_phase_transitions_disabled_returns_5(tmp_path):
    """V4: ``enable_curriculum_phases=False`` retorna "5" (default self-play)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=5_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=False,
    )
    trainer = PPOTrainer(config, run_name="phase_disabled")

    trainer.global_step = 0
    assert trainer._current_phase() == "5"
    trainer.global_step = 2_000_000
    assert trainer._current_phase() == "5"

    trainer.envs.close()
    trainer.writer.close()


def test_anti_forgetting_rotation_when_phase_2_plus(tmp_path):
    """V4: em fase 2+, anti_forgetting_rotation_prob faz alguns episódios voltarem
    pra "1A". Em fases 1A/1B/1C, rotação é DESLIGADA (já estamos na base)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=10_000_000,
        num_envs=4,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=500_000,
        phase_1b_end_step=1_000_000,
        phase_1c_end_step=1_500_000,
        phase_2_end_step=2_500_000,
        anti_forgetting_rotation_prob=0.5,  # alto pra teste estatístico
        force_sync_vec_env=True,
    )
    trainer = PPOTrainer(config, run_name="anti_forget_test")

    # fase 2 com prob 0.5: ~50% das amostras devem ser "1A"
    trainer.global_step = 2_000_000
    assert trainer._current_phase() == "2"

    rng = trainer._curriculum_rng
    n = 1000
    rotations = sum(1 for _ in range(n) if rng.random() < 0.5)
    ratio = rotations / n
    assert 0.40 < ratio < 0.60, f"ratio rotation deveria ser ~0.5, got {ratio}"

    trainer.envs.close()
    trainer.writer.close()


def test_v6_current_phase_includes_1d_3d_when_enabled(tmp_path):
    """V6: ``_current_phase`` retorna "1D" e "3D" quando os end_steps estão setados."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=    800_000,
        phase_1b_end_step=  1_600_000,
        phase_1c_end_step=  2_400_000,
        phase_1d_end_step=  4_400_000,   # V6
        phase_2_end_step=   6_400_000,
        phase_3_end_step=   8_400_000,
        phase_3d_end_step= 11_400_000,   # V6
        phase_4_end_step=  13_400_000,
    )
    trainer = PPOTrainer(config, run_name="v6_phase_test")

    trainer.global_step = 100_000;     assert trainer._current_phase() == "1A"
    trainer.global_step = 1_000_000;   assert trainer._current_phase() == "1B"
    trainer.global_step = 2_000_000;   assert trainer._current_phase() == "1C"
    trainer.global_step = 3_000_000;   assert trainer._current_phase() == "1D"
    trainer.global_step = 5_000_000;   assert trainer._current_phase() == "2"
    trainer.global_step = 7_000_000;   assert trainer._current_phase() == "3"
    trainer.global_step = 10_000_000;  assert trainer._current_phase() == "3D"
    trainer.global_step = 12_000_000;  assert trainer._current_phase() == "4"
    trainer.global_step = 15_000_000;  assert trainer._current_phase() == "5"

    trainer.envs.close()
    trainer.writer.close()


def test_v6_current_phase_skips_1d_3d_when_none(tmp_path):
    """V6 backward-compat: ``phase_1d/3d_end_step=None`` ⇒ fases puladas
    (sequência V4/V5: 1A→1B→1C→2→3→4→5)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=5_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=500_000,
        phase_1b_end_step=1_000_000,
        phase_1c_end_step=1_500_000,
        # phase_1d_end_step=None (default)
        phase_2_end_step=2_500_000,
        phase_3_end_step=3_500_000,
        # phase_3d_end_step=None (default)
        phase_4_end_step=4_500_000,
    )
    trainer = PPOTrainer(config, run_name="v6_skip_test")

    # step que SERIA "1D" em V6 → cai em "2" (phase_1c_end < step < phase_2_end).
    trainer.global_step = 2_000_000
    assert trainer._current_phase() == "2"
    # step que SERIA "3D" em V6 → cai em "4".
    trainer.global_step = 4_000_000
    assert trainer._current_phase() == "4"
    # "1D"/"3D" nunca devem aparecer
    for s in (0, 600_000, 1_200_000, 1_800_000, 2_800_000, 3_800_000, 4_800_000):
        trainer.global_step = s
        assert trainer._current_phase() not in ("1D", "3D")

    trainer.envs.close()
    trainer.writer.close()


def test_v6_rotation_pool_symmetric_when_1d_3d_enabled(tmp_path):
    """V6: pool de rotação inclui ["1A","1D","3D"] quando ambos end_steps setados."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=    800_000,
        phase_1b_end_step=  1_600_000,
        phase_1c_end_step=  2_400_000,
        phase_1d_end_step=  4_400_000,
        phase_2_end_step=   6_400_000,
        phase_3_end_step=   8_400_000,
        phase_3d_end_step= 11_400_000,
        phase_4_end_step=  13_400_000,
    )
    trainer = PPOTrainer(config, run_name="v6_rotation_pool")
    assert trainer._rotation_pool() == ["1A", "1D", "3D"]
    trainer.envs.close()
    trainer.writer.close()


def test_v6_rotation_pool_backward_compat_v4_only_1a(tmp_path):
    """V6 backward-compat: sem 1D/3D, pool de rotação fica ["1A"] (V4)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=5_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        # phase_1d_end_step=None, phase_3d_end_step=None (defaults)
    )
    trainer = PPOTrainer(config, run_name="v6_rotation_compat")
    assert trainer._rotation_pool() == ["1A"]
    trainer.envs.close()
    trainer.writer.close()


def test_v7_rotation_weights_respeita_distribuicao(tmp_path):
    """V7: rotation_weights={1D=0.50, 1A=0.25, 3D=0.25} ⇒ amostragem
    bate com a distribuição esperada em N samples (statistical test)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=    800_000,
        phase_1b_end_step=  1_600_000,
        phase_1c_end_step=  2_400_000,
        phase_1d_end_step=  6_000_000,
        phase_2_end_step=   7_600_000,
        phase_3_end_step=   9_200_000,
        phase_3d_end_step= 12_000_000,
        phase_4_end_step=  13_400_000,
        rotation_weights={"1A": 0.25, "1D": 0.50, "3D": 0.25},
    )
    trainer = PPOTrainer(config, run_name="v7_weighted")

    pool = trainer._rotation_pool()
    weights = trainer._rotation_weights_for(pool)
    assert pool == ["1A", "1D", "3D"]
    assert weights == [0.25, 0.50, 0.25]

    # sample N=3000 vezes, conta proporção.
    n = 3000
    counts = {"1A": 0, "1D": 0, "3D": 0}
    for _ in range(n):
        sampled = trainer._curriculum_rng.choices(pool, weights=weights, k=1)[0]
        counts[sampled] += 1
    p_1a = counts["1A"] / n
    p_1d = counts["1D"] / n
    p_3d = counts["3D"] / n
    # tolerância 4% (n=3000, 1 sigma binomial ~ 0.9% para p=0.5).
    assert 0.21 < p_1a < 0.29, f"1A esperado ~0.25, got {p_1a:.3f}"
    assert 0.46 < p_1d < 0.54, f"1D esperado ~0.50, got {p_1d:.3f}"
    assert 0.21 < p_3d < 0.29, f"3D esperado ~0.25, got {p_3d:.3f}"

    trainer.envs.close()
    trainer.writer.close()


def test_v7_rotation_weights_backward_compat_none_uniform(tmp_path):
    """V7 backward-compat: rotation_weights=None ⇒ _rotation_weights_for retorna None
    (caller usa amostragem uniforme V4/V5/V6)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1d_end_step= 6_000_000,
        phase_3d_end_step=12_000_000,
        # rotation_weights=None (default)
    )
    trainer = PPOTrainer(config, run_name="v7_no_weights")
    pool = trainer._rotation_pool()
    assert trainer._rotation_weights_for(pool) is None
    trainer.envs.close()
    trainer.writer.close()


def test_v8_current_phase_includes_3t_when_enabled(tmp_path):
    """V8: ``_current_phase`` retorna "3T" quando ``phase_3t_end_step`` setado.

    Ordem completa V8: 1A→1B→1C→1D→2→3→3D→3T→4→5.
    """
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1a_end_step=    800_000,
        phase_1b_end_step=  1_600_000,
        phase_1c_end_step=  2_400_000,
        phase_1d_end_step=  6_000_000,
        phase_2_end_step=   7_600_000,
        phase_3_end_step=   9_200_000,
        phase_3d_end_step= 11_600_000,
        phase_3t_end_step= 12_400_000,    # V8 NOVO
        phase_4_end_step=  13_400_000,
    )
    trainer = PPOTrainer(config, run_name="v8_phase_test")

    trainer.global_step = 100_000;     assert trainer._current_phase() == "1A"
    trainer.global_step = 1_000_000;   assert trainer._current_phase() == "1B"
    trainer.global_step = 2_000_000;   assert trainer._current_phase() == "1C"
    trainer.global_step = 3_000_000;   assert trainer._current_phase() == "1D"
    trainer.global_step = 7_000_000;   assert trainer._current_phase() == "2"
    trainer.global_step = 8_500_000;   assert trainer._current_phase() == "3"
    trainer.global_step = 10_000_000;  assert trainer._current_phase() == "3D"
    trainer.global_step = 12_000_000;  assert trainer._current_phase() == "3T"
    trainer.global_step = 13_000_000;  assert trainer._current_phase() == "4"
    trainer.global_step = 15_000_000;  assert trainer._current_phase() == "5"

    trainer.envs.close()
    trainer.writer.close()


def test_v8_rotation_pool_includes_3t_when_enabled(tmp_path):
    """V8: pool de rotação inclui "3T" quando phase_3t_end_step setado."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1d_end_step= 6_000_000,
        phase_3d_end_step=11_600_000,
        phase_3t_end_step=12_400_000,
    )
    trainer = PPOTrainer(config, run_name="v8_rotation_pool")
    assert trainer._rotation_pool() == ["1A", "1D", "3D", "3T"]
    trainer.envs.close()
    trainer.writer.close()


def test_v8_rotation_pool_skips_3t_when_disabled(tmp_path):
    """V8 backward-compat: phase_3t_end_step=None ⇒ 3T não entra no pool (V7 puro)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=20_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        phase_1d_end_step= 6_000_000,
        phase_3d_end_step=11_600_000,
        # phase_3t_end_step=None (default)
    )
    trainer = PPOTrainer(config, run_name="v8_no_3t")
    assert trainer._rotation_pool() == ["1A", "1D", "3D"]
    trainer.envs.close()
    trainer.writer.close()


def test_v7_rotation_weights_ignora_fase_ausente(tmp_path):
    """V7: weight para fase ausente do pool (ex: ``"1D": 0.50`` quando
    ``phase_1d_end_step is None``) é silenciosamente ignorado.

    Cenário: V4-style config (sem 1D/3D) recebe weights V7 errados.
    Em vez de crashar, weights são resolvidos só para fases presentes
    no pool — graceful degradation pra runs antigos relendo configs novas.
    """
    config = _micro_config(
        tmp_path,
        total_timesteps=5_000_000,
        num_envs=2,
        num_steps=128,
        enable_curriculum_phases=True,
        # 1D e 3D desabilitados (None default)
        rotation_weights={"1A": 0.25, "1D": 0.50, "3D": 0.25},
    )
    trainer = PPOTrainer(config, run_name="v7_partial")
    pool = trainer._rotation_pool()
    weights = trainer._rotation_weights_for(pool)
    # pool = ["1A"] apenas; weights = [0.25] (single element).
    assert pool == ["1A"]
    assert weights == [0.25]
    trainer.envs.close()
    trainer.writer.close()


# categoria H — Async self-play E2E

def test_async_self_play_e2e(tmp_path):
    """Treino curto self-play com Async deve completar e gravar sync_dir."""
    config = _micro_config(
        tmp_path,
        total_timesteps=4096,
        num_envs=2,
        num_steps=128,
        fixed_random_opponent=False,
        enable_self_play=True,
        enable_curriculum=False,
        enable_curriculum_phases=False,
        save_steps=2048,
        force_sync_vec_env=False,           # libera Async
        async_sync_pool_to_disk=True,
    )
    trainer = PPOTrainer(config, run_name="async_e2e")

    # Async foi escolhido?
    assert "AsyncVectorEnv" in type(trainer.envs).__name__, (
        f"esperado AsyncVectorEnv, got {type(trainer.envs).__name__}"
    )

    # __init__ já deveria ter escrito o pool inicial em disco.
    sync_dir = trainer.async_sync_dir
    assert (sync_dir / "manifest.json").exists()
    assert (sync_dir / "policy_latest.pt").exists()

    trainer.train()

    # após train, manifest continua presente (pode ter sido reescrito durante save).
    assert (sync_dir / "manifest.json").exists()
    assert (sync_dir / "policy_latest.pt").exists()


def test_replay_captured_at_phase_transition(tmp_path):
    """V4.1/V7: replay automático em transição de fase.

    Configura phase end_steps cedo (dentro do total_timesteps) pra forçar
    múltiplas transições durante o smoke. V7: arquivos têm sufixo
    ``phase_polished_X`` (X = fase ANTIGA recém-terminada), não mais
    ``phase_transition_Y`` (Y = fase nova).
    """
    config = _micro_config(
        tmp_path,
        total_timesteps=8192,
        num_envs=2,
        num_steps=128,
        enable_curriculum=True,
        enable_curriculum_phases=True,
        # phase transitions cedo: cada update (256 timesteps) muda de fase.
        phase_1a_end_step=2048,
        phase_1b_end_step=4096,
        phase_1c_end_step=6144,
        phase_2_end_step=7168,
        phase_3_end_step=7680,
        phase_4_end_step=8000,
        save_steps=999_999,  # save_steps gigante: nenhum replay regular
        force_sync_vec_env=True,
        fixed_random_opponent=True,
        enable_self_play=False,
    )
    trainer = PPOTrainer(config, run_name="phase_transition_replay")
    trainer.train()

    replays_dir = trainer.run_dir / "replays"
    polished_replays = list(replays_dir.glob("replay_*_phase_polished_*.json"))

    assert len(polished_replays) >= 2, (
        f"Esperava ≥2 replays polished, achei {len(polished_replays)}: "
        f"{[r.name for r in polished_replays]}"
    )

    # v7: cada replay polished_X deve ter sido capturado COM a fase X
    # (estado polido, não primeiro tropeço da fase nova). Verifica via
    # metadata.phase do payload.
    import json
    for r in polished_replays:
        with open(r) as f:
            payload = json.load(f)
        # sufixo do filename: replay_NNN_phase_polished_X.json
        # extrai X: rsplit("_", 3) → [head, "phase", "polished", X]
        stem = r.stem
        _, _, _, suffix_phase = stem.rsplit("_", 3)
        assert payload["metadata"]["phase"] == suffix_phase, (
            f"{r.name}: metadata.phase={payload['metadata']['phase']!r} "
            f"≠ suffix {suffix_phase!r} (V7: capturar com fase ANTIGA)"
        )


def test_ent_coef_anneal_decreases_linearly(tmp_path):
    """V4.3: anneal linear de ent_coef do start ao end conforme global_step."""
    config = _micro_config(
        tmp_path,
        total_timesteps=1000,
        num_envs=2,
        num_steps=64,
        ent_coef=0.03,
        ent_coef_end=0.005,
    )
    trainer = PPOTrainer(config, run_name="ent_anneal")

    # step 0 → ent_coef = start (0.03)
    trainer.global_step = 0
    assert trainer._current_ent_coef() == pytest.approx(0.03)

    # step 500 (50%) → ent_coef ≈ midpoint = 0.0175
    trainer.global_step = 500
    assert trainer._current_ent_coef() == pytest.approx(0.0175, abs=1e-6)

    # step 1000 (100%) → ent_coef = end (0.005)
    trainer.global_step = 1000
    assert trainer._current_ent_coef() == pytest.approx(0.005, abs=1e-6)

    # step além do total → clamp em end
    trainer.global_step = 2000
    assert trainer._current_ent_coef() == pytest.approx(0.005, abs=1e-6)

    trainer.envs.close()
    trainer.writer.close()


def test_ent_coef_constant_when_end_not_set(tmp_path):
    """V4.3: backward-compat — sem ent_coef_end, ent_coef permanece constante."""
    config = _micro_config(
        tmp_path,
        total_timesteps=1000,
        num_envs=2,
        num_steps=64,
        ent_coef=0.02,
        # ent_coef_end deixado como default (None)
    )
    trainer = PPOTrainer(config, run_name="ent_const")
    assert trainer.config.ent_coef_end is None

    for step in (0, 250, 500, 999, 5000):
        trainer.global_step = step
        assert trainer._current_ent_coef() == pytest.approx(0.02), (
            f"step={step}: ent_coef deveria ser constante 0.02, "
            f"got {trainer._current_ent_coef()}"
        )

    trainer.envs.close()
    trainer.writer.close()


def test_async_self_play_disabled_falls_back_to_sync(tmp_path):
    """async_sync_pool_to_disk=False com self-play → cai pra Sync (legacy)."""
    config = _micro_config(
        tmp_path,
        total_timesteps=2048,
        num_envs=2,
        num_steps=128,
        fixed_random_opponent=False,
        enable_self_play=True,
        force_sync_vec_env=False,
        async_sync_pool_to_disk=False,  # explicitamente desativado
    )
    trainer = PPOTrainer(config, run_name="async_disabled_e2e")

    assert "SyncVectorEnv" in type(trainer.envs).__name__, (
        f"esperado SyncVectorEnv quando async_sync_pool_to_disk=False, "
        f"got {type(trainer.envs).__name__}"
    )

    trainer.envs.close()
    trainer.writer.close()
