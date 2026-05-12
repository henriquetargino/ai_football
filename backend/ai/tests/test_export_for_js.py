"""testes unitários do exportador Python → JavaScript.

rodar (a partir da raiz):
    pytest backend/ai/tests/test_export_for_js.py -v
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ai.actions import ACTION_SPACE_SIZE
from backend.ai.export_for_js import (
    POLICY_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    export_policy,
    export_replay,
)
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import HIDDEN_SIZE, Policy
from backend.ai.train import PPOConfig, PPOTrainer, RunningMeanStd


_EXPECTED_WEIGHT_KEYS = {
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
    "policy_head.weight", "policy_head.bias",
    "value_head.weight", "value_head.bias",
}


def _make_minimal_checkpoint(tmp_path: Path, with_obs_rms: bool = True) -> Path:
    """cria um checkpoint .pt sintético sem treinar — rápido pra testes."""
    torch.manual_seed(0)
    policy = Policy()
    state = {
        "global_step": 12345,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": {},
        "obs_rms": (
            RunningMeanStd((OBS_SIZE,)).state_dict() if with_obs_rms else None
        ),
        "rew_rms": None,
        "config": {"total_timesteps": 1000, "norm_obs": with_obs_rms},
        "run_name": "synthetic",
    }
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(state, ckpt_path)
    return ckpt_path


def _flat_replay_payload(n_frames: int = 3) -> dict:
    return {
        "metadata": {
            "global_step": 999,
            "trained_team": "red",
            "terminated": False,
            "truncated": True,
            "frame_count": n_frames * 30,
            "fps": 30,
        },
        "frames": [
            {
                "step": i * 30,
                "ball_x": 400.0 + i, "ball_y": 250.0,
                "ball_vx": 0.0, "ball_vy": 0.0,
                "players": [
                    {
                        "id": "red_0", "team": "red",
                        "x": 200.0 + i, "y": 250.0,
                        "angle": 0.5, "vx": 0.1, "vy": 0.0,
                        "is_kicking": False,
                    },
                    {
                        "id": "blue_0", "team": "blue",
                        "x": 600.0 - i, "y": 250.0,
                        "angle": math.pi, "vx": -0.1, "vy": 0.0,
                        "is_kicking": True,
                    },
                ],
                "score": {"red": 0, "blue": 0},
                "goal_width": 200.0,
            }
            for i in range(n_frames)
        ],
    }


@pytest.fixture(autouse=True)
def _chdir_to_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_policy_export_has_eight_weight_keys(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    assert set(data["weights"].keys()) == _EXPECTED_WEIGHT_KEYS
    assert data["schema_version"] == POLICY_SCHEMA_VERSION


def test_policy_export_weight_shapes(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    w = data["weights"]
    assert np.array(w["fc1.weight"]).shape == (HIDDEN_SIZE, OBS_SIZE)
    assert np.array(w["fc1.bias"]).shape == (HIDDEN_SIZE,)
    assert np.array(w["fc2.weight"]).shape == (HIDDEN_SIZE, HIDDEN_SIZE)
    assert np.array(w["fc2.bias"]).shape == (HIDDEN_SIZE,)
    assert np.array(w["policy_head.weight"]).shape == (ACTION_SPACE_SIZE, HIDDEN_SIZE)
    assert np.array(w["policy_head.bias"]).shape == (ACTION_SPACE_SIZE,)
    assert np.array(w["value_head.weight"]).shape == (1, HIDDEN_SIZE)
    assert np.array(w["value_head.bias"]).shape == (1,)


def test_policy_export_metadata(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    md = data["metadata"]
    assert md["global_step"] == 12345
    assert md["run_name"] == "synthetic"
    assert md["obs_size"] == OBS_SIZE
    assert md["hidden_size"] == HIDDEN_SIZE
    assert md["action_space_size"] == ACTION_SPACE_SIZE


def test_policy_export_uses_float32_precision(tmp_path):
    """fp64 round-trips perderiam precisão se exportados como Python float e
    relidos como fp32. valida que a perda é zero (originalmente já fp32).
    """
    ckpt = _make_minimal_checkpoint(tmp_path)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)

    state = torch.load(ckpt, weights_only=False, map_location="cpu")
    original_fp32 = state["policy_state_dict"]["fc1.weight"].numpy().astype(np.float32)

    with open(out) as f:
        data = json.load(f)
    reloaded = np.array(data["weights"]["fc1.weight"], dtype=np.float32)

    np.testing.assert_array_equal(original_fp32, reloaded)


def test_policy_export_with_obs_rms_present(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path, with_obs_rms=True)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    assert data["obs_rms"] is not None
    assert "mean" in data["obs_rms"]
    assert "var" in data["obs_rms"]
    assert len(data["obs_rms"]["mean"]) == OBS_SIZE
    assert len(data["obs_rms"]["var"]) == OBS_SIZE


def test_policy_export_with_obs_rms_absent(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path, with_obs_rms=False)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    assert data["obs_rms"] is None


def test_policy_export_raises_on_missing_keys(tmp_path):
    """validação cedo: se state_dict não tem todas as chaves esperadas, falha."""
    ckpt = tmp_path / "broken.pt"
    torch.save({
        "global_step": 0,
        "policy_state_dict": {"fc1.weight": torch.zeros(HIDDEN_SIZE, OBS_SIZE)},
        "optimizer_state_dict": {},
        "obs_rms": None,
        "rew_rms": None,
        "config": {},
        "run_name": "broken",
    }, ckpt)
    with pytest.raises(ValueError, match="chaves esperadas"):
        export_policy(ckpt, tmp_path / "out.json")


def test_replay_export_converts_flat_to_nested(tmp_path):
    flat = _flat_replay_payload(n_frames=2)
    in_path = tmp_path / "flat.json"
    out_path = tmp_path / "nested.json"
    with open(in_path, "w") as f:
        json.dump(flat, f)

    export_replay(in_path, out_path)
    with open(out_path) as f:
        data = json.load(f)

    assert data["schema_version"] == REPLAY_SCHEMA_VERSION
    frame = data["frames"][0]
    assert "ball" in frame and "ball_x" not in frame
    assert frame["ball"]["x"] == 400.0
    assert frame["ball"]["y"] == 250.0
    assert frame["ball"]["vx"] == 0.0
    assert frame["ball"]["vy"] == 0.0


def test_replay_export_preserves_metadata(tmp_path):
    flat = _flat_replay_payload()
    in_path = tmp_path / "flat.json"
    out_path = tmp_path / "nested.json"
    with open(in_path, "w") as f:
        json.dump(flat, f)
    export_replay(in_path, out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert data["metadata"] == flat["metadata"]


def test_replay_export_events_is_empty_list(tmp_path):
    flat = _flat_replay_payload(n_frames=3)
    in_path = tmp_path / "flat.json"
    out_path = tmp_path / "nested.json"
    with open(in_path, "w") as f:
        json.dump(flat, f)
    export_replay(in_path, out_path)
    with open(out_path) as f:
        data = json.load(f)
    for frame in data["frames"]:
        assert frame["events"] == []


def test_replay_export_can_kick_default_true(tmp_path):
    flat = _flat_replay_payload()
    in_path = tmp_path / "flat.json"
    out_path = tmp_path / "nested.json"
    with open(in_path, "w") as f:
        json.dump(flat, f)
    export_replay(in_path, out_path)
    with open(out_path) as f:
        data = json.load(f)
    for frame in data["frames"]:
        for player in frame["players"]:
            assert player["can_kick"] is True


def test_replay_export_preserves_frame_count(tmp_path):
    flat = _flat_replay_payload(n_frames=7)
    in_path = tmp_path / "flat.json"
    out_path = tmp_path / "nested.json"
    with open(in_path, "w") as f:
        json.dump(flat, f)
    export_replay(in_path, out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert len(data["frames"]) == 7


def test_cli_smoke(tmp_path):
    ckpt = _make_minimal_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"

    result = subprocess.run(
        [
            sys.executable, "-m", "backend.ai.export_for_js",
            "--checkpoint", str(ckpt),
            "--output", str(out_dir),
        ],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    out_file = out_dir / "final_policy.json"
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
    assert data["schema_version"] == POLICY_SCHEMA_VERSION


def _js_style_forward_argmax(weights: dict, obs: np.ndarray) -> int:
    """reconstrói o forward pass de JS puro: matmul + tanh + matmul + tanh +
    matmul + argmax. espelha o que inference.js faz.
    """
    fc1_w = np.array(weights["fc1.weight"], dtype=np.float32)
    fc1_b = np.array(weights["fc1.bias"], dtype=np.float32)
    fc2_w = np.array(weights["fc2.weight"], dtype=np.float32)
    fc2_b = np.array(weights["fc2.bias"], dtype=np.float32)
    ph_w = np.array(weights["policy_head.weight"], dtype=np.float32)
    ph_b = np.array(weights["policy_head.bias"], dtype=np.float32)

    h1 = np.tanh(fc1_w @ obs + fc1_b)
    h2 = np.tanh(fc2_w @ h1 + fc2_b)
    logits = ph_w @ h2 + ph_b
    return int(np.argmax(logits))


def test_forward_pass_parity_python_vs_js_style(tmp_path):
    """a prova mais importante: dado o JSON exportado, reconstruir o forward
    pass JS-style produz a MESMA action que select_action_greedy em Python.
    em 100 obs aleatórias, todas devem coincidir.
    """
    torch.manual_seed(0)
    policy = Policy()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({
        "global_step": 0,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": {},
        "obs_rms": None, "rew_rms": None,
        "config": {}, "run_name": "parity",
    }, ckpt)
    out = tmp_path / "policy.json"
    export_policy(ckpt, out)
    with open(out) as f:
        data = json.load(f)
    weights = data["weights"]

    rng = np.random.default_rng(0)
    for _ in range(100):
        obs = rng.standard_normal(OBS_SIZE).astype(np.float32)
        action_py = policy.select_action_greedy(obs)
        action_js = _js_style_forward_argmax(weights, obs)
        assert action_py == action_js, (
            f"divergência: py={action_py} js={action_js}"
        )
