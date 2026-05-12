"""testes do módulo publish_run.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_publish_run.py -v
"""

import json
from pathlib import Path

import pytest
import torch

from backend.ai.policy import Policy
from backend.ai.publish_run import (
    _PHASE_LABELS,
    _format_step_label,
    _select_milestone_checkpoints,
    _select_phase_milestones,
    find_run_dir,
    publish_run,
)


def test_find_run_dir_single_match(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260101-120000_my_run").mkdir()
    (runs / "20260101-120000_other_run").mkdir()

    found = find_run_dir(runs, "my_run")
    assert found.name == "20260101-120000_my_run"


def test_find_run_dir_multiple_match_returns_latest(tmp_path):
    """múltiplos runs com mesmo nome → retorna mais recente (timestamp)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260101-120000_my_run").mkdir()
    (runs / "20260201-120000_my_run").mkdir()
    (runs / "20260115-120000_my_run").mkdir()

    found = find_run_dir(runs, "my_run")
    assert found.name == "20260201-120000_my_run"


def test_find_run_dir_no_match_lists_available(tmp_path):
    """sem match → erro lista runs disponíveis."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260101-120000_run_a").mkdir()
    (runs / "20260101-120000_run_b").mkdir()

    with pytest.raises(FileNotFoundError) as exc:
        find_run_dir(runs, "nonexistent")
    assert "run_a" in str(exc.value)
    assert "run_b" in str(exc.value)


def test_find_run_dir_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_run_dir(tmp_path / "does_not_exist", "any")


def test_format_step_label_round_M():
    """25M com total=50M → "25M" (não "25.0M")."""
    assert _format_step_label(25_000_000, 50_000_000) == "25M"


def test_format_step_label_decimal_M():
    assert _format_step_label(12_500_000, 50_000_000) == "12.5M"


def test_format_step_label_final_marker():
    """step ≈ total → adiciona " (final)"."""
    label = _format_step_label(50_000_000, 50_000_000)
    assert label == "50M (final)"


def test_format_step_label_k_range():
    """< 1M → "Nk"."""
    assert _format_step_label(500_000, 5_000_000) == "500k"


def test_select_milestones_uniform_distribution(tmp_path):
    """8 marcos uniformes de 40 checkpoints → 8 distintos espalhados."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    # 40 checkpoints a cada 1.25M.
    for i in range(1, 41):
        (ckpts_dir / f"checkpoint_{i * 1_250_000}.pt").touch()
    (ckpts_dir / "final.pt").touch()

    selected = _select_milestone_checkpoints(ckpts_dir, 50_000_000, n_milestones=8)

    assert len(selected) == 8
    assert selected[-1].name == "final.pt"
    intermediate_steps = [
        int(p.stem.removeprefix("checkpoint_"))
        for p in selected[:-1]
    ]
    assert intermediate_steps == sorted(intermediate_steps)
    # primeiro marco deve ser perto de 12.5% × 50M = 6.25M.
    assert abs(intermediate_steps[0] - 6_250_000) < 1_500_000


def test_select_milestones_short_run_dedups(tmp_path):
    """run curto (poucos ckpts) → seleciona o que tem sem duplicar."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    (ckpts_dir / "checkpoint_100000.pt").touch()
    (ckpts_dir / "checkpoint_200000.pt").touch()
    (ckpts_dir / "final.pt").touch()

    selected = _select_milestone_checkpoints(ckpts_dir, 200_000, n_milestones=8)

    assert len(selected) <= 3
    names = [p.name for p in selected]
    assert len(names) == len(set(names))
    assert "final.pt" in names


def test_select_milestones_only_final(tmp_path):
    """sem ckpts intermediários, só final.pt → retorna [final.pt]."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    (ckpts_dir / "final.pt").touch()
    selected = _select_milestone_checkpoints(ckpts_dir, 100_000, n_milestones=8)
    assert len(selected) == 1
    assert selected[0].name == "final.pt"


def test_phase_milestones_picks_last_ckpt_before_phase_end(tmp_path):
    """pra cada fase, pega o checkpoint mais próximo ANTES do end_step."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    for s in (400_000, 800_000, 1_200_000, 1_800_000, 2_400_000,
              2_800_000, 3_200_000, 3_600_000, 4_000_000):
        (ckpts_dir / f"checkpoint_{s}.pt").touch()
    (ckpts_dir / "final.pt").touch()

    config = {
        "enable_curriculum_phases": True,
        "phase_1a_end_step":   400_000,
        "phase_1b_end_step":   800_000,
        "phase_1c_end_step": 1_200_000,
        "phase_1d_end_step": 1_800_000,
        "phase_2_end_step":  2_400_000,
        "phase_3_end_step":  2_800_000,
        "phase_3d_end_step": 3_200_000,
        "phase_3t_end_step": 3_600_000,
        "phase_4_end_step":  4_000_000,
    }

    selected = _select_phase_milestones(ckpts_dir, config)

    assert len(selected) == 10
    phases = [p for p, _ in selected]
    assert phases == ["1A", "1B", "1C", "1D", "2", "3", "3D", "3T", "4", "5"]

    p1a, c1a = selected[0]
    assert c1a.name == "checkpoint_400000.pt"

    p1d, c1d = selected[3]
    assert p1d == "1D" and c1d.name == "checkpoint_1800000.pt"

    p3t, c3t = selected[7]
    assert p3t == "3T" and c3t.name == "checkpoint_3600000.pt"

    p5, c5 = selected[-1]
    assert p5 == "5" and c5.name == "final.pt"


def test_phase_milestones_skips_disabled_phases(tmp_path):
    """fases sem end_step (None) são puladas."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    for s in (500_000, 1_000_000, 1_500_000, 2_000_000,
              2_500_000, 3_000_000, 3_500_000, 4_000_000):
        (ckpts_dir / f"checkpoint_{s}.pt").touch()
    (ckpts_dir / "final.pt").touch()

    config = {
        "enable_curriculum_phases": True,
        "phase_1a_end_step":  500_000,
        "phase_1b_end_step": 1_000_000,
        "phase_1c_end_step": 1_500_000,
        "phase_2_end_step":  2_500_000,
        "phase_3_end_step":  3_500_000,
        "phase_4_end_step":  4_000_000,
    }

    selected = _select_phase_milestones(ckpts_dir, config)
    phases = [p for p, _ in selected]
    assert "1D" not in phases
    assert "3D" not in phases
    assert "3T" not in phases
    assert phases == ["1A", "1B", "1C", "2", "3", "4", "5"]


def test_phase_milestones_returns_empty_when_curriculum_disabled(tmp_path):
    """enable_curriculum_phases=False → []. caller usa fallback uniforme."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    (ckpts_dir / "checkpoint_500000.pt").touch()
    (ckpts_dir / "final.pt").touch()

    config = {"enable_curriculum_phases": False}
    selected = _select_phase_milestones(ckpts_dir, config)
    assert selected == []


def test_phase_labels_have_expected_format():
    """labels descritivos pra UI no formato 'F<X> · <skill>'."""
    assert _PHASE_LABELS["1A"].startswith("F1A · ")
    assert _PHASE_LABELS["1D"].startswith("F1D · ")
    assert _PHASE_LABELS["3T"].startswith("F3T · ")
    assert _PHASE_LABELS["5"].startswith("F5 · ")
    assert _PHASE_LABELS["0"].startswith("F0 · ")
    for phase, label in _PHASE_LABELS.items():
        assert "·" in label, f"label de F{phase} sem separador: {label!r}"


def test_phase_milestones_includes_all_f5_checkpoints(tmp_path):
    """fase 5 inclui TODOS os checkpoints intermediários nela + final.pt."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    for s in range(1_000_000, 10_000_001, 1_000_000):
        (ckpts_dir / f"checkpoint_{s}.pt").touch()
    (ckpts_dir / "final.pt").touch()

    config = {
        "enable_curriculum_phases": True,
        "phase_1a_end_step":  1_000_000,
        "phase_1b_end_step":  2_000_000,
        "phase_1c_end_step":  3_000_000,
        "phase_2_end_step":   4_000_000,
        "phase_3_end_step":   5_000_000,
        "phase_4_end_step":   6_000_000,
        "total_timesteps":   10_000_000,
    }

    selected = _select_phase_milestones(ckpts_dir, config)
    phases = [p for p, _ in selected]

    # pra fase 5 (step > 6M): ckpts 7M, 8M, 9M, 10M + final.pt = 5 marcos.
    f5_count = phases.count("5")
    assert f5_count == 5, (
        f"esperava 5 marcos F5 (4 ckpts + final), got {f5_count}"
    )

    f5_indices = [i for i, p in enumerate(phases) if p == "5"]
    last_f5_idx = f5_indices[-1]
    _, last_f5 = selected[last_f5_idx]
    assert last_f5.name == "final.pt"


def test_format_f5_step_label():
    """labels de F5 usam formato 'F5 · <step>' + (final) no último."""
    from backend.ai.publish_run import _format_f5_step_label

    assert _format_f5_step_label(50_000_000, 50_000_000, is_final=True) == "F5 · 50M (final)"
    assert _format_f5_step_label(35_000_000, 50_000_000, is_final=False) == "F5 · 35M"
    assert _format_f5_step_label(36_250_000, 50_000_000, is_final=False) == "F5 · 36.2M"


def test_random_replay_generation(tmp_path):
    """F0 random replay roda sem precisar de policy/checkpoint."""
    from backend.ai.publish_run import _generate_random_replay
    out = tmp_path / "f0_random.json"
    metadata = _generate_random_replay(out, seed=42)
    assert out.exists()
    assert metadata["phase"] == "0"
    assert metadata["global_step"] == 0
    assert metadata["frame_count"] > 0


def _make_minimal_run(tmp_path: Path, name: str, with_intermediate: bool = True) -> Path:
    """cria pasta de run sintética com final.pt + config.json válidos.

    config.json é necessário porque total_timesteps é lido pra calcular marcos.
    """
    run_dir = tmp_path / "data" / "runs" / f"20260101-120000_{name}"
    (run_dir / "checkpoints").mkdir(parents=True)

    config = {"total_timesteps": 1000, "num_envs": 8}
    (run_dir / "config.json").write_text(json.dumps(config))

    torch.manual_seed(0)
    base_state = {
        "global_step": 1000,
        "policy_state_dict": Policy().state_dict(),
        "optimizer_state_dict": {},
        "obs_rms": None,
        "rew_rms": None,
        "config": config,
        "run_name": name,
    }

    torch.save(base_state, run_dir / "checkpoints" / "final.pt")

    if with_intermediate:
        for step in (250, 500, 750):
            state_i = dict(base_state, global_step=step)
            torch.save(state_i, run_dir / "checkpoints" / f"checkpoint_{step}.pt")

    return run_dir


def test_publish_run_end_to_end_v8_schema(tmp_path):
    """manifest.schema_version=2, snapshots com step/label/policy_path."""
    run_dir = _make_minimal_run(tmp_path, "test_publish")
    frontend_data = tmp_path / "frontend_data"

    # generate_replays=False: pula simulação cara (testes de env/policy cobrem isso).
    manifest = publish_run(run_dir, frontend_data, n_milestones=4, generate_replays=False)

    assert (frontend_data / "manifest.json").exists()
    assert manifest["schema_version"] == 2
    assert manifest["run_label"] == "test_publish"
    assert manifest["total_timesteps"] == 1000
    assert "snapshots" in manifest

    snapshots = manifest["snapshots"]
    assert len(snapshots) >= 1
    last = snapshots[-1]
    assert "final" in last["label"]
    steps = [s["step"] for s in snapshots]
    assert steps == sorted(steps)
    for s in snapshots:
        assert s["policy_path"].startswith(f"data/runs/{run_dir.name}/")
        assert "replay_path" not in s


def test_publish_run_clears_old_run_artifacts(tmp_path):
    """re-publicar mesmo run não deixa artefatos antigos."""
    run_dir = _make_minimal_run(tmp_path, "test_clear")
    frontend_data = tmp_path / "frontend_data"

    publish_run(run_dir, frontend_data, n_milestones=2, generate_replays=False)

    spurious = frontend_data / "runs" / run_dir.name / "spurious_old_file.json"
    spurious.write_text("{}")
    assert spurious.exists()

    publish_run(run_dir, frontend_data, n_milestones=2, generate_replays=False)
    assert not spurious.exists(), "Artefato antigo deveria ter sido removido"


def test_publish_run_missing_final_pt_raises(tmp_path):
    run_dir = tmp_path / "data" / "runs" / "20260101-120000_broken"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"total_timesteps": 100}))
    with pytest.raises(FileNotFoundError, match="final.pt"):
        publish_run(run_dir, tmp_path / "out", generate_replays=False)


def test_publish_run_missing_config_raises(tmp_path):
    run_dir = tmp_path / "data" / "runs" / "20260101-120000_broken"
    (run_dir / "checkpoints").mkdir(parents=True)
    state = {
        "global_step": 0,
        "policy_state_dict": Policy().state_dict(),
        "optimizer_state_dict": {},
        "obs_rms": None,
        "rew_rms": None,
        "config": {},
        "run_name": "broken",
    }
    torch.save(state, run_dir / "checkpoints" / "final.pt")
    with pytest.raises(FileNotFoundError, match="config.json"):
        publish_run(run_dir, tmp_path / "out", generate_replays=False)
