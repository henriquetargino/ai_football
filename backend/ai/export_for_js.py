"""exportador Python → JavaScript.

carrega checkpoint salvo por train.py (PyTorch + RunningMeanStd) e gera
os JSONs consumidos pelo frontend:
    final_policy.json — pesos + normalização pra inferência em JS puro
    final_replay.json — replay no schema aninhado do frontend

JSON puro (não MessagePack/binary): simples, debuggable, browser-friendly.
np.float32 antes de listar preserva paridade bit-a-bit com Float32Array em JS.
rew_rms NÃO é exportado — afeta gradientes em treino, não inferência.

uso CLI:
    python -m backend.ai.export_for_js \\
        --checkpoint data/runs/{ts}_{name}/checkpoints/final.pt \\
        --output data/runs/{ts}_{name}/exports/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# bumpar se mudar o formato consumido pelo JS.
POLICY_SCHEMA_VERSION: int = 1
REPLAY_SCHEMA_VERSION: int = 1

# valida que todas as chaves estão presentes — falha cedo se arquitetura mudou.
_EXPECTED_WEIGHT_KEYS: tuple[str, ...] = (
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
    "policy_head.weight", "policy_head.bias",
    "value_head.weight", "value_head.bias",
)


def export_policy(checkpoint_path: Path, output_path: Path) -> dict:
    """carrega state_dict de um checkpoint e serializa pra JSON."""
    state = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu"
    )

    policy_sd = state["policy_state_dict"]
    missing = [k for k in _EXPECTED_WEIGHT_KEYS if k not in policy_sd]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} não tem chaves esperadas: {missing}. "
            f"arquitetura da Policy pode ter mudado."
        )

    weights: dict[str, list] = {}
    for key in _EXPECTED_WEIGHT_KEYS:
        arr = policy_sd[key].detach().cpu().numpy().astype(np.float32)
        weights[key] = arr.tolist()

    obs_rms_export: Optional[dict] = None
    if state.get("obs_rms") is not None:
        obs_rms_state = state["obs_rms"]
        obs_rms_export = {
            "mean": np.asarray(obs_rms_state["mean"], dtype=np.float32).tolist(),
            "var":  np.asarray(obs_rms_state["var"], dtype=np.float32).tolist(),
        }

    config = state.get("config", {})
    payload = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "metadata": {
            "global_step": int(state.get("global_step", 0)),
            "run_name": str(state.get("run_name", "")),
            "obs_size": int(np.array(weights["fc1.weight"]).shape[1]),
            "hidden_size": int(np.array(weights["fc1.weight"]).shape[0]),
            "action_space_size": int(np.array(weights["policy_head.weight"]).shape[0]),
        },
        "weights": weights,
        "obs_rms": obs_rms_export,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)

    return payload


def _adapt_frame(flat_frame: dict) -> dict:
    """converte frame do schema flat (backend) para o aninhado (frontend)."""
    return {
        "step": flat_frame["step"],
        "ball": {
            "x": flat_frame["ball_x"],
            "y": flat_frame["ball_y"],
            "vx": flat_frame["ball_vx"],
            "vy": flat_frame["ball_vy"],
        },
        "players": [
            {
                "id": p["id"],
                "team": p["team"],
                "x": p["x"], "y": p["y"],
                "angle": p["angle"],
                "vx": p["vx"], "vy": p["vy"],
                # backend não exporta can_kick; frontend usa pra LED.
                "can_kick": True,
                "is_kicking": p["is_kicking"],
            }
            for p in flat_frame["players"]
        ],
        "score": flat_frame["score"],
        "goal_width": flat_frame["goal_width"],
        # populável em versões futuras (ex: highlights de gols).
        "events": [],
    }


def export_replay(replay_path_in: Path, replay_path_out: Path) -> dict:
    """converte replay JSON do schema flat (backend) pro aninhado (frontend)."""
    with open(replay_path_in) as f:
        original = json.load(f)

    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "metadata": original.get("metadata", {}),
        "frames": [_adapt_frame(f) for f in original.get("frames", [])],
    }

    replay_path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(replay_path_out, "w") as f:
        json.dump(payload, f)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint → JS JSON")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Caminho pro .pt salvo por train.py.",
    )
    parser.add_argument(
        "--replay", type=Path, default=None,
        help="Caminho pro .json salvo por env.capture_replay. "
             "Se omitido, exporta apenas a policy.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Diretório de saída (cria se não existir).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    policy_out = args.output / "final_policy.json"
    export_policy(args.checkpoint, policy_out)
    print(f"Exported policy: {policy_out}")

    if args.replay is not None:
        replay_out = args.output / "final_replay.json"
        export_replay(args.replay, replay_out)
        print(f"Exported replay: {replay_out}")


if __name__ == "__main__":
    main()
