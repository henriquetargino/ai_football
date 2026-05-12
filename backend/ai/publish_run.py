"""automação de export + manifest pós-treino.

resolve o workflow manual que exigia:
    1. achar o timestamp da pasta do run
    2. rodar export_for_js 1x por replay
    3. renomear arquivos
    4. editar frontend/data/manifest.json na mão

uso (depois de qualquer python -m backend.ai.train ...):
    python -m backend.ai.publish_run --name v8_50M

com path explícito:
    python -m backend.ai.publish_run --run-dir data/runs/20260430-003750_v4_phase1a_only

listar runs disponíveis:
    python -m backend.ai.publish_run --list

estratégia:
    - exporta um marco por fase do curriculum (em vez de só final.pt),
      pra mostrar a evolução por skill no frontend.
    - cada marco roda 1 replay padronizado NA fase recém-aprendida.
    - schema do manifest é "snapshots: [...]" com step/label/policy_path/replay_path.

saída em frontend/data/:
    manifest.json
    runs/{run_name}/
        policy_{step}.json   (1 por marco)
        replay_{step}.json   (1 por marco)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from backend.ai.export_for_js import export_policy


def find_run_dir(runs_root: Path, name: str) -> Path:
    """encontra a pasta mais recente matching *_{name} em data/runs/.

    match feito pelo suffix após o primeiro underscore.
    """
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Pasta {runs_root} não existe")

    matches: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        parts = p.name.split("_", 1)
        if len(parts) != 2:
            continue
        _timestamp, run_name = parts
        if run_name == name:
            matches.append(p)

    if not matches:
        available = sorted({
            p.name.split("_", 1)[1]
            for p in runs_root.iterdir()
            if p.is_dir() and "_" in p.name
        })
        raise FileNotFoundError(
            f"Nenhum run com nome '{name}' encontrado em {runs_root}.\n"
            f"Runs disponíveis: {available}"
        )

    return sorted(matches, key=lambda p: p.name)[-1]


def _format_step_label(step: int, total: int) -> str:
    """label legível pra dropdown do frontend (ex: 12.5M, 50M (final))."""
    if step >= 1_000_000:
        m = step / 1e6
        if abs(m - round(m)) < 0.05:
            label = f"{int(round(m))}M"
        else:
            label = f"{m:.1f}M"
    elif step >= 1_000:
        label = f"{step // 1000}k"
    else:
        label = str(step)
    is_final = total > 0 and abs(step - total) < total * 0.01
    if is_final:
        label += " (final)"
    return label


def _select_milestone_checkpoints(
    ckpts_dir: Path,
    total_timesteps: int,
    n_milestones: int = 8,
) -> list[Path]:
    """seleciona até n_milestones checkpoints uniformemente espaçados.

    inclui sempre final.pt como último marco. demais (n-1) marcos vêm do
    closest match em checkpoint_*.pt aos targets i/n × total. deduplica
    pra runs muito curtos onde múltiplos targets cairiam no mesmo arquivo.
    """
    intermediate = sorted(
        ckpts_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.removeprefix("checkpoint_")),
    )
    final = ckpts_dir / "final.pt"

    selected: list[Path] = []
    seen: set[Path] = set()

    if intermediate and total_timesteps > 0:
        targets = [(i + 1) / n_milestones * total_timesteps for i in range(n_milestones - 1)]
        for tgt in targets:
            best = min(
                intermediate,
                key=lambda c: abs(int(c.stem.removeprefix("checkpoint_")) - tgt),
            )
            if best not in seen:
                selected.append(best)
                seen.add(best)

    if final.exists() and final not in seen:
        selected.append(final)

    return selected


def _ckpt_step(ckpt_path: Path, total_timesteps: int) -> int:
    """extrai o step de um checkpoint. final.pt lê do estado salvo."""
    if ckpt_path.name == "final.pt":
        import torch
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        return int(state.get("global_step", total_timesteps))
    return int(ckpt_path.stem.removeprefix("checkpoint_"))


_PHASE_LABELS: dict = {
    "0":  "F0 · Random (untrained)",
    "1A": "F1A · Touch the ball",
    "1B": "F1B · Walk to the ball",
    "1C": "F1C · Aim & shoot",
    "1D": "F1D · Side ball",
    "2":  "F2 · vs Passive",
    "3":  "F3 · vs Scripted",
    "3D": "F3D · Defense",
    "3T": "F3T · Counter-attack",
    "4":  "F4 · Random spawn",
    # fase 5 usa label dinâmico ("F5 · 35M") via _format_f5_step_label.
    "5":  "F5 · Self-play (final)",
}


def _format_f5_step_label(step: int, total: int, is_final: bool) -> str:
    """label de um snapshot dentro da fase 5: "F5 · 35M", "F5 · 50M (final)"."""
    if step >= 1_000_000:
        m = step / 1e6
        if abs(m - round(m)) < 0.05:
            num = f"{int(round(m))}M"
        else:
            num = f"{m:.1f}M"
    elif step >= 1_000:
        num = f"{step // 1000}k"
    else:
        num = str(step)
    label = f"F5 · {num}"
    if is_final:
        label += " (final)"
    return label


_PHASE_END_FIELDS: dict = {
    "1A": "phase_1a_end_step",
    "1B": "phase_1b_end_step",
    "1C": "phase_1c_end_step",
    "1D": "phase_1d_end_step",
    "2":  "phase_2_end_step",
    "3":  "phase_3_end_step",
    "3D": "phase_3d_end_step",
    "3T": "phase_3t_end_step",
    "4":  "phase_4_end_step",
    # 5 não tem end_step (vai até total_timesteps).
}

_PHASE_ORDER: list = ["1A", "1B", "1C", "1D", "2", "3", "3D", "3T", "4", "5"]


def _select_phase_milestones(
    ckpts_dir: Path, config: dict
) -> list[tuple]:
    """seleciona um checkpoint por fase ativa do curriculum.

    pra cada fase cujo end_step está setado, pega o checkpoint mais
    próximo ANTES do fim daquela fase. pra fase 5, inclui todos os
    checkpoints intermediários dentro dela + final.pt.
    """
    if not config.get("enable_curriculum_phases", False):
        return []

    intermediate = sorted(
        ckpts_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.removeprefix("checkpoint_")),
    )
    final = ckpts_dir / "final.pt"

    milestones: list[tuple] = []
    used: set = set()

    for phase in _PHASE_ORDER:
        if phase == "5":
            # fase 5: cada checkpoint intermediário dentro dela vira um marco
            # + final.pt sempre é o último.
            phase4_end = config.get("phase_4_end_step")
            if phase4_end is not None:
                in_p5 = [
                    c for c in intermediate
                    if int(c.stem.removeprefix("checkpoint_")) > phase4_end
                ]
                for c in in_p5:
                    if c not in used:
                        milestones.append(("5", c))
                        used.add(c)

            if final.exists() and final not in used:
                milestones.append(("5", final))
                used.add(final)
            continue

        end_field = _PHASE_END_FIELDS[phase]
        end_step = config.get(end_field)
        if end_step is None:
            continue

        if not intermediate:
            continue

        # prefere o último checkpoint cujo step <= end_step (policy ao
        # final da fase, ou ligeiramente antes).
        eligible = [
            c for c in intermediate
            if int(c.stem.removeprefix("checkpoint_")) <= end_step
        ]
        if eligible:
            best = eligible[-1]
        else:
            # fase tão curta que nenhum ckpt foi salvo dentro dela.
            best = intermediate[0]

        if best not in used:
            milestones.append((phase, best))
            used.add(best)

    return milestones


def _generate_standardized_replay(
    ckpt_path: Path,
    out_path: Path,
    seed: int = 42,
    phase: str = "5",
) -> dict:
    """roda 1 episódio com a Policy do ckpt_path na fase phase e grava o replay.

    em fase 5 (POLICY), o opponent é a mesma policy em greedy (self-play).
    nas outras fases, opponent_mode = NONE/PASSIVE/SCRIPTED (gerenciado
    pelo env via set_progress).

    concatena episódios curtos até atingir duração mínima (~22s).
    """
    # lazy imports pra evitar ciclo train → publish_run no auto-publish.
    import numpy as np
    import torch

    from backend.ai.env import OpponentMode, REPLAY_FPS, SoccerEnv
    from backend.ai.policy import Policy
    from backend.ai.train import RunningMeanStd

    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    global_step = int(state.get("global_step", 0))

    policy = Policy()
    policy.load_state_dict(state["policy_state_dict"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    obs_rms: Optional[RunningMeanStd] = None
    if state.get("obs_rms"):
        obs_rms = RunningMeanStd((341,))
        obs_rms.load_state_dict(state["obs_rms"])

    @torch.no_grad()
    def opp_callable(obs_raw: np.ndarray) -> int:
        if obs_rms is not None:
            obs_norm = obs_rms.normalize(obs_raw[None])[0]
        else:
            obs_norm = obs_raw
        return policy.select_action_greedy(obs_norm.astype(np.float32))

    env = SoccerEnv(seed=seed, capture_replay=True)
    # progress=1.0 → goal_width usa o "real" (distribuição final do DR).
    env.set_progress(1.0, phase=phase)
    if env.opponent_mode == OpponentMode.POLICY:
        env.set_opponent_policy(opp_callable)

    # concatena múltiplos episódios até atingir duração mínima. em fases
    # iniciais o agente faz gol em ~2s → episódio termina rápido → replay
    # curto demais. o frontend renderiza as transições como "gols" via
    # detecção de teleporte da bola.
    TARGET_MIN_PHYSICAL_FRAMES = 22 * 60
    MAX_EPISODES = 15

    all_frames: list = []
    total_physical_frames = 0
    last_terminated = False
    last_truncated = False

    for episode_idx in range(MAX_EPISODES):
        obs, _ = env.reset(options={"progress": 1.0})
        terminated = truncated = False

        while not (terminated or truncated):
            if obs_rms is not None:
                obs_norm = obs_rms.normalize(obs[None])[0]
            else:
                obs_norm = obs
            action = policy.select_action_greedy(obs_norm.astype(np.float32))
            obs, _, terminated, truncated, _ = env.step(action)

        base_step = (all_frames[-1]["step"] + 1) if all_frames else 0
        for f in env.get_replay():
            f_copy = dict(f)
            f_copy["step"] = f_copy["step"] + base_step
            all_frames.append(f_copy)

        total_physical_frames += env.frame_count
        last_terminated = terminated
        last_truncated = truncated

        if total_physical_frames >= TARGET_MIN_PHYSICAL_FRAMES:
            break

    payload = {
        "metadata": {
            "global_step": global_step,
            "trained_team": env.trained_team,
            "terminated": bool(last_terminated),
            "truncated": bool(last_truncated),
            "frame_count": total_physical_frames,
            "fps": REPLAY_FPS,
            "phase": phase,
            "episodes_concat": episode_idx + 1,
        },
        "frames": all_frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return payload["metadata"]


def _generate_random_replay(out_path: Path, seed: int = 42) -> dict:
    """F0: replay com policy 100% random, antes de qualquer treino.

    mostra o ponto de partida da IA. permite comparar F0 vs F5 final
    visualmente.
    """
    import random as pyrandom

    from backend.ai.actions import ACTION_SPACE_SIZE
    from backend.ai.env import OpponentMode, REPLAY_FPS, SoccerEnv

    pyrandom.seed(seed)

    def random_action(_obs):
        return pyrandom.randint(0, ACTION_SPACE_SIZE - 1)

    env = SoccerEnv(seed=seed, capture_replay=True)
    env.set_progress(1.0, phase="5")
    if env.opponent_mode == OpponentMode.POLICY:
        env.set_opponent_policy(random_action)

    obs, _ = env.reset(options={"progress": 1.0})
    terminated = truncated = False
    while not (terminated or truncated):
        action = random_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    payload = {
        "metadata": {
            "global_step": 0,
            "trained_team": env.trained_team,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "frame_count": env.frame_count,
            "fps": REPLAY_FPS,
            "phase": "0",
        },
        "frames": env.get_replay(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return payload["metadata"]


def _adapt_flat_to_nested(flat_replay: dict) -> dict:
    """converte replay do schema flat (env capture) pro aninhado (frontend)."""
    from backend.ai.export_for_js import _adapt_frame, REPLAY_SCHEMA_VERSION
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "metadata": flat_replay.get("metadata", {}),
        "frames": [_adapt_frame(f) for f in flat_replay.get("frames", [])],
    }


def publish_run(
    run_dir: Path,
    frontend_data: Path,
    n_milestones: int = 8,
    generate_replays: bool = True,
) -> dict:
    """exporta marcos do run + manifest novo.

    prioriza marcos por fase do curriculum (1 por fase ativa). fallback
    pra marcos uniformes em n_milestones quando o run não tem curriculum.
    cada marco roda 1 replay na fase recém-aprendida.

    generate_replays=False pula a geração de replays (rápido — só policies).
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir não existe: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json não encontrado em {run_dir}")
    config = json.loads(config_path.read_text())
    total_timesteps = int(config.get("total_timesteps", 0))

    ckpts_dir = run_dir / "checkpoints"
    final_ckpt = ckpts_dir / "final.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(
            f"final.pt não encontrado em {ckpts_dir}. Treino completou?"
        )

    run_name = run_dir.name
    out_dir = frontend_data / "runs" / run_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_milestones = _select_phase_milestones(ckpts_dir, config)

    if phase_milestones:
        print(f"📌 {len(phase_milestones)} marcos por FASE do curriculum:")
        for phase, ckpt in phase_milestones:
            step = _ckpt_step(ckpt, total_timesteps)
            print(f"   - {ckpt.name:30s} step={step:>10,}  phase={phase}")
        # F0 random no início (sentinela None pra ckpt; tratado abaixo).
        items = [("0", None)] + [(p, c) for p, c in phase_milestones]
    else:
        uniform = _select_milestone_checkpoints(
            ckpts_dir, total_timesteps, n_milestones
        )
        if not uniform:
            raise FileNotFoundError(f"Nenhum checkpoint encontrado em {ckpts_dir}")
        print(f"📌 {len(uniform)} marcos uniformes (fallback — sem curriculum):")
        items = [(None, c) for c in uniform]
        for c in uniform:
            step = _ckpt_step(c, total_timesteps)
            print(f"   - {c.name:30s} step={step:>10,}")

    snapshots: list[dict] = []
    for phase, ckpt_path in items:
        # F0: caso especial sem ckpt e sem policy.
        if phase == "0":
            label = _PHASE_LABELS["0"]
            replay_filename = "replay_0_F0_random.json"
            replay_path_str = None
            if generate_replays:
                replay_out = out_dir / replay_filename
                tmp_out = out_dir / ".tmp_replay_0.json"
                try:
                    _generate_random_replay(tmp_out, seed=42)
                    with open(tmp_out) as f:
                        flat = json.load(f)
                    nested = _adapt_flat_to_nested(flat)
                    with open(replay_out, "w") as f:
                        json.dump(nested, f)
                    tmp_out.unlink()
                    print(f"   📼 {replay_filename}  (F0 random, no policy)")
                    replay_path_str = f"data/runs/{run_name}/{replay_filename}"
                except Exception as e:
                    print(f"   ⚠️  F0 random replay falhou: {e}")
                    if tmp_out.exists():
                        tmp_out.unlink()
            snapshots.append({
                "step": 0,
                "label": label,
                "phase": "0",
                # F0 não tem policy_path — só replay.
                **({"replay_path": replay_path_str} if replay_path_str else {}),
            })
            continue

        step = _ckpt_step(ckpt_path, total_timesteps)
        if phase == "5":
            is_final = ckpt_path.name == "final.pt"
            label = _format_f5_step_label(step, total_timesteps, is_final)
            replay_phase = "5"
            replay_filename = f"replay_{step}_F5.json"
        elif phase is not None:
            label = _PHASE_LABELS.get(phase, f"F{phase} · {step // 1_000_000}M")
            replay_phase = phase
            replay_filename = f"replay_{step}_F{phase}.json"
        else:
            label = _format_step_label(step, total_timesteps)
            replay_phase = "5"
            replay_filename = f"replay_{step}.json"

        policy_filename = f"policy_{step}.json"
        policy_out = out_dir / policy_filename
        export_policy(ckpt_path, policy_out)
        print(f"   📦 {policy_filename}")

        replay_path_str = None
        if generate_replays:
            replay_out = out_dir / replay_filename
            tmp_out = out_dir / f".tmp_replay_{step}.json"
            try:
                _generate_standardized_replay(
                    ckpt_path, tmp_out,
                    seed=42 + step % 10_000,
                    phase=replay_phase,
                )
                with open(tmp_out) as f:
                    flat = json.load(f)
                nested = _adapt_flat_to_nested(flat)
                with open(replay_out, "w") as f:
                    json.dump(nested, f)
                tmp_out.unlink()
                print(
                    f"   📼 {replay_filename}  "
                    f"({nested['metadata']['frame_count']} frames, phase={replay_phase})"
                )
                replay_path_str = f"data/runs/{run_name}/{replay_filename}"
            except Exception as e:
                print(f"   ⚠️  replay {replay_filename} falhou: {e}")
                if tmp_out.exists():
                    tmp_out.unlink()

        snapshot = {
            "step": step,
            "label": label,
            "policy_path": f"data/runs/{run_name}/{policy_filename}",
        }
        if phase is not None:
            snapshot["phase"] = phase
        if replay_path_str is not None:
            snapshot["replay_path"] = replay_path_str
        snapshots.append(snapshot)

    snapshots.sort(key=lambda s: s["step"])

    short_name = run_name.split("_", 1)[1] if "_" in run_name else run_name
    manifest: dict = {
        "schema_version": 2,
        "run_name": run_name,
        "run_label": short_name,
        "total_timesteps": total_timesteps,
        "snapshots": snapshots,
    }

    manifest_path = frontend_data / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📝 Manifest atualizado: {manifest_path}")
    print(f"   {len(snapshots)} snapshots")

    return manifest


def _list_runs(runs_root: Path) -> None:
    if not runs_root.is_dir():
        print(f"Pasta {runs_root} não existe.")
        return
    print(f"Runs em {runs_root}:")
    entries = sorted(runs_root.iterdir())
    for p in entries:
        if not p.is_dir() or "_" not in p.name:
            continue
        ts, name = p.name.split("_", 1)
        ckpt = p / "checkpoints" / "final.pt"
        status = "✓" if ckpt.exists() else "✗ (incompleto)"
        print(f"  {status} {name:30s} ({ts})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta marcos do run + gera manifest do frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos:
  python -m backend.ai.publish_run --name v8_50M
  python -m backend.ai.publish_run --run-dir data/runs/20260430-003750_v4
  python -m backend.ai.publish_run --list
""",
    )
    parser.add_argument(
        "--name",
        help="Nome do run (sem timestamp). Usa o mais recente se múltiplos matches.",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="Path explícito pra pasta do run (override de --name).",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("data/runs"),
        help="Pasta raiz dos runs (default: data/runs).",
    )
    parser.add_argument(
        "--frontend-data", type=Path, default=Path("frontend/data"),
        help="Pasta de saída do frontend (default: frontend/data).",
    )
    parser.add_argument(
        "--n-milestones", type=int, default=8,
        help="Quantos marcos exportar (default: 8).",
    )
    parser.add_argument(
        "--no-replays", action="store_true",
        help="Não gerar replays padronizados (rápido — só policies).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lista runs disponíveis e sai.",
    )
    args = parser.parse_args()

    if args.list:
        _list_runs(args.runs_root)
        return

    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.name is not None:
        run_dir = find_run_dir(args.runs_root, args.name)
        print(f"📂 Encontrado: {run_dir}")
    else:
        parser.error("Forneça --name ou --run-dir (ou use --list)")

    args.frontend_data.mkdir(parents=True, exist_ok=True)
    publish_run(
        run_dir,
        args.frontend_data,
        n_milestones=args.n_milestones,
        generate_replays=not args.no_replays,
    )

    print(
        f"\n✅ Pronto! Rode: cd frontend && python -m http.server 8000"
        f"\n   Acesse: http://localhost:8000"
    )


if __name__ == "__main__":
    main()
