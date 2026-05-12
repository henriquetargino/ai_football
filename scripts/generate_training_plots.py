"""gera gráficos do treino (TensorBoard + replays) pra usar no vídeo do LinkedIn.

saída em docs/video_assets/<portuguese|english>/:
    return_per_episode.png    (retorno acumulado por episódio)
    reward_components.png     (3 componentes do reward separados)
    episode_length.png        (duração do episódio ao longo do treino)
    entropy_decay.png         (entropia da policy: explorando → decidida)
    heatmap_positions.png     (posições no campo: início vs fim do treino)

não modifica nada do treino existente — só lê e plota.

uso:
    python -m scripts.generate_training_plots
    python -m scripts.generate_training_plots --run-dir <path>
    python -m scripts.generate_training_plots --skip-heatmap
    python -m scripts.generate_training_plots --lang en
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tensorboard.backend.event_processing import event_accumulator


# cores das fases (esquema visual coerente com o vídeo).
PHASE_COLORS = {
    "1A":  "#a5d6a7",
    "1B":  "#81c784",
    "1C":  "#66bb6a",
    "1D":  "#43a047",
    "1E":  "#2e7d32",
    "2":   "#ffb74d",
    "3":   "#ff9800",
    "3D":  "#90caf9",
    "3DA": "#42a5f5",
    "3GB": "#1976d2",
    "4":   "#ce93d8",
    "5":   "#7b1fa2",
}

STRINGS = {
    "pt": {
        "saved": "✓ Salvo",
        "generating": "Gerando",
        "tag_missing": "⚠ tag faltando, pulando.",
        "rewards_missing": "⚠ tags de reward faltando, pulando.",
        "no_data": "⚠ Sem dados de replay suficientes.",
        "loading_tb": "📈 Carregando TensorBoard logs...",
        "loaded_tags": "tags carregados",
        "generating_plots": "🎨 Gerando plots...",
        "generating_heatmap": "🗺️  Gerando heatmap (lendo replays)...",
        "reading_early": "Lendo início:",
        "reading_late": "Lendo fim:",
        "replays_not_found": "⚠ Replays não encontrados em",
        "skipping_heatmap": ", pulando heatmap.",
        "plots_in": "✅ Gráficos em:",
        "return_title": "Retorno por episódio — a IA aprendeu a marcar mais do que sofrer",
        "return_xlabel": "Steps de treino (50M total)",
        "return_ylabel": "Retorno acumulado",
        "rc_suptitle": "Os 3 componentes do reward que ensinaram a IA",
        "rc_goal_title": "GOAL  (+10 quando IA marca)",
        "rc_concede_title": "CONCEDE  (-5 quando IA sofre, módulo)",
        "rc_kick_title": "KICK_ON_GOAL  (2.5 × cos² × proximity²) — ofensividade pura",
        "rc_goal_ylabel": "Reward de gol marcado",
        "rc_concede_ylabel": "Reward de gol sofrido",
        "rc_kick_ylabel": "Reward de chute no gol",
        "rc_xlabel": "Steps de treino (50M total)",
        "el_title": "Duração do episódio: cresce em fases defensivas complexas, estabiliza no self-play",
        "el_xlabel": "Steps de treino (50M total)",
        "el_ylabel": "Frames por episódio",
        "el_ylabel2": "Segundos (60 FPS)",
        "el_legend": "Duração do episódio (frames)",
        "ed_title": "De aleatória → decidida: a entropia da policy cai com o aprendizado",
        "ed_xlabel": "Steps de treino (50M total)",
        "ed_ylabel": "Entropia (alta = explorando, baixa = decidida)",
        "ed_legend": "Entropia da policy (incerteza)",
        "hm_suptitle": "Onde a IA fica no campo: início vs fim do treino",
        "hm_early_title": "Início do treino (0–5M steps, {} replays)",
        "hm_late_title": "Fim do treino (~45–50M steps, {} replays)",
        "hm_frames_suffix": "frames",
        "hm_stat_near_ball": "Perto da bola: {:.0f}%",
        "hm_stat_attacking": "No ataque: {:.0f}%",
    },
    "en": {
        "saved": "✓ Saved",
        "generating": "Generating",
        "tag_missing": "⚠ tag missing, skipping.",
        "rewards_missing": "⚠ reward tags missing, skipping.",
        "no_data": "⚠ Not enough replay data.",
        "loading_tb": "📈 Loading TensorBoard logs...",
        "loaded_tags": "tags loaded",
        "generating_plots": "🎨 Generating plots...",
        "generating_heatmap": "🗺️  Generating heatmap (reading replays)...",
        "reading_early": "Reading early:",
        "reading_late": "Reading late:",
        "replays_not_found": "⚠ Replays not found in",
        "skipping_heatmap": ", skipping heatmap.",
        "plots_in": "✅ Plots in:",
        "return_title": "Return per episode — the AI learned to score more than concede",
        "return_xlabel": "Training steps (50M total)",
        "return_ylabel": "Cumulative return",
        "rc_suptitle": "The 3 reward components that taught the AI",
        "rc_goal_title": "GOAL  (+10 when AI scores)",
        "rc_concede_title": "CONCEDE  (-5 when AI is scored on, |abs|)",
        "rc_kick_title": "KICK_ON_GOAL  (2.5 × cos² × proximity²) — pure offensive bias",
        "rc_goal_ylabel": "Reward from scoring",
        "rc_concede_ylabel": "Reward from being scored on",
        "rc_kick_ylabel": "Reward from kicking on goal",
        "rc_xlabel": "Training steps (50M total)",
        "el_title": "Episode length: grows in complex defensive phases, stabilizes during self-play",
        "el_xlabel": "Training steps (50M total)",
        "el_ylabel": "Frames per episode",
        "el_ylabel2": "Seconds (60 FPS)",
        "el_legend": "Episode length (frames)",
        "ed_title": "From random → decisive: policy entropy decreases with learning",
        "ed_xlabel": "Training steps (50M total)",
        "ed_ylabel": "Entropy (high = exploring, low = decisive)",
        "ed_legend": "Policy entropy (uncertainty)",
        "hm_suptitle": "Where the AI stays on the field: start vs end of training",
        "hm_early_title": "Start of training (0–5M steps, {} replays)",
        "hm_late_title": "End of training (~45–50M steps, {} replays)",
        "hm_frames_suffix": "frames",
        "hm_stat_near_ball": "Near ball: {:.0f}%",
        "hm_stat_attacking": "Attacking: {:.0f}%",
    },
}


# distribuição V10.2 (steps de cada fase).
V10_2_PHASES = [
    ("1A",   0,           2_000_000),
    ("1B",   2_000_000,   4_000_000),
    ("1C",   4_000_000,   6_000_000),
    ("1D",   6_000_000,  13_500_000),
    ("1E",  13_500_000,  15_500_000),
    ("2",   15_500_000,  18_500_000),
    ("3",   18_500_000,  21_500_000),
    ("3D",  21_500_000,  26_000_000),
    ("3DA", 26_000_000,  27_500_000),
    ("3GB", 27_500_000,  29_000_000),
    ("4",   29_000_000,  33_000_000),
    ("5",   33_000_000,  50_000_000),
]


def load_tb_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """carrega scalars do TensorBoard. retorna dict {tag: (steps, values)}."""
    tb_dir = run_dir / "tensorboard"
    print(f"  Lendo TensorBoard: {tb_dir}")
    ea = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 100_000}
    )
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        out[tag] = (steps, values)
    return out


def add_phase_bands(ax, phases: list, ymin: float, ymax: float, alpha: float = 0.15):
    """adiciona faixas verticais coloridas pra cada fase no gráfico."""
    for name, start, end in phases:
        color = PHASE_COLORS.get(name, "#cccccc")
        ax.axvspan(start, end, alpha=alpha, color=color, zorder=0)
        mid = (start + end) / 2
        ax.text(
            mid, ymax * 0.97, name,
            ha="center", va="top", fontsize=9, color=color,
            fontweight="bold", zorder=10
        )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """rolling mean simples (preenche NaNs no início)."""
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    rolled = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, rolled])


def plot_reward_components(scalars: dict, output_path: Path, t: dict) -> None:
    """plota os 3 componentes da função de reward em subplots empilhados.

    cada sub-gráfico tem escala própria (cada componente é diferente),
    mas compartilham eixo X. mostra quais componentes do reward "puxam"
    o aprendizado em cada fase.
    """
    print(f"  {t['generating']} {output_path.name}...")
    goal = scalars.get("reward/goal_per_episode")
    concede = scalars.get("reward/concede_per_episode")
    kick = scalars.get("reward/kick_on_goal_per_episode")
    if not all([goal, concede, kick]):
        print(f"  {t['rewards_missing']}")
        return

    g_steps, g_vals = goal
    c_steps, c_vals = concede
    k_steps, k_vals = kick
    window = max(10, len(g_vals) // 100)
    g_smooth = rolling_mean(g_vals, window)
    # módulo pra escala positiva.
    c_smooth = rolling_mean(np.abs(c_vals), window)
    k_smooth = rolling_mean(k_vals, window)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=120, sharex=True)
    fig.patch.set_facecolor("#0a0a14")

    components = [
        (axes[0], g_steps, g_smooth, "#4caf50",
         t["rc_goal_title"], t["rc_goal_ylabel"]),
        (axes[1], c_steps, c_smooth, "#f44336",
         t["rc_concede_title"], t["rc_concede_ylabel"]),
        (axes[2], k_steps, k_smooth, "#2196f3",
         t["rc_kick_title"], t["rc_kick_ylabel"]),
    ]

    for ax, steps, smooth, color, title, ylabel in components:
        ax.set_facecolor("#0f0f1f")
        ymax = np.nanmax(smooth) * 1.15 if not np.all(np.isnan(smooth)) else 1
        # evita ymax=0 pra concede.
        ymax = max(ymax, 0.1)
        add_phase_bands(ax, V10_2_PHASES, 0, ymax)
        ax.plot(steps, smooth, color=color, lw=2.0)
        ax.fill_between(steps, 0, smooth, alpha=0.20, color=color)

        ax.set_title(title, color="white", fontsize=11, loc="left", pad=8)
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="white")
        ax.set_xlim(0, 50_000_000)
        ax.set_ylim(0, ymax)

    axes[-1].set_xlabel(t["rc_xlabel"], color="white", fontsize=11)
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    fig.suptitle(t["rc_suptitle"], color="white", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  {t['saved']}: {output_path}")


def plot_return_per_episode(scalars: dict, output_path: Path, t: dict) -> None:
    """plota o return total (soma dos 3 componentes) por episódio.

    é a métrica "a IA está aprendendo?" do RL — mostra o agente passando
    de quase-aleatório (return ~0) pra estratégico (return positivo
    crescente). equivalente a rollout/ep_rew_mean do SB3, recomposto a
    partir dos 3 componentes.
    """
    print(f"  {t['generating']} {output_path.name}...")
    goal = scalars.get("reward/goal_per_episode")
    concede = scalars.get("reward/concede_per_episode")
    kick = scalars.get("reward/kick_on_goal_per_episode")
    if not all([goal, concede, kick]):
        print(f"  {t['rewards_missing']}")
        return

    g_steps, g_vals = goal
    _,       c_vals = concede
    _,       k_vals = kick

    # trunca pelo array mais curto (assume logs simultâneos).
    n = min(len(g_vals), len(c_vals), len(k_vals))
    steps = g_steps[:n]
    return_vals = g_vals[:n] + c_vals[:n] + k_vals[:n]

    window = max(10, n // 100)
    smooth = rolling_mean(return_vals, window)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0f0f1f")

    ymin = float(np.nanmin(smooth)) - 0.5 if not np.all(np.isnan(smooth)) else -1
    ymax = float(np.nanmax(smooth)) + 0.5 if not np.all(np.isnan(smooth)) else 5
    add_phase_bands(ax, V10_2_PHASES, ymin, ymax)

    # linha y=0 (referencial: agent aleatório fica ~0).
    ax.axhline(y=0, color="#666", linestyle="--", lw=0.9, alpha=0.6)

    ax.plot(steps, smooth, color="#ffd54f", lw=2.4)
    fill_base = 0 if ymin < 0 else ymin
    ax.fill_between(steps, fill_base, smooth, alpha=0.18, color="#ffd54f")

    ax.set_title(t["return_title"], color="white", fontsize=12, loc="left", pad=8)
    ax.set_xlabel(t["return_xlabel"], color="white", fontsize=11)
    ax.set_ylabel(t["return_ylabel"], color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlim(0, 50_000_000)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  {t['saved']}: {output_path}")


def plot_episode_length(scalars: dict, output_path: Path, t: dict) -> None:
    print(f"  {t['generating']} {output_path.name}...")
    ep_len = scalars.get("episode/length")
    if ep_len is None:
        print(f"  {t['tag_missing']}")
        return

    steps, vals = ep_len
    window = max(10, len(vals) // 100)
    smooth = rolling_mean(vals, window)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0f0f1f")

    ymax = np.nanmax(smooth) * 1.1 if not np.all(np.isnan(smooth)) else 2400
    add_phase_bands(ax, V10_2_PHASES, 0, ymax)

    ax.plot(steps, smooth, color="#03a9f4", lw=2.2, label=t["el_legend"])

    # conversão pra segundos no eixo Y secundário.
    ax2 = ax.twinx()
    ax2.set_ylim(0, ymax / 60)
    ax2.set_ylabel(t["el_ylabel2"], color="#03a9f4", fontsize=11)
    ax2.tick_params(colors="#03a9f4")
    ax2.spines["right"].set_color("#03a9f4")

    ax.set_xlabel(t["el_xlabel"], color="white", fontsize=11)
    ax.set_ylabel(t["el_ylabel"], color="white", fontsize=11)
    ax.set_title(t["el_title"], color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.legend(loc="upper right", facecolor="#0a0a14", edgecolor="#444",
              labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlim(0, 50_000_000)
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  {t['saved']}: {output_path}")


def plot_entropy_decay(scalars: dict, output_path: Path, t: dict) -> None:
    print(f"  {t['generating']} {output_path.name}...")
    ent = scalars.get("ppo/entropy_loss")
    if ent is None:
        print(f"  {t['tag_missing']}")
        return

    steps, vals = ent
    window = max(10, len(vals) // 100)
    smooth = rolling_mean(vals, window)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0f0f1f")

    ymax = np.nanmax(smooth) * 1.05 if not np.all(np.isnan(smooth)) else 2.9
    ymin = max(0, np.nanmin(smooth) * 0.95) if not np.all(np.isnan(smooth)) else 0.5
    add_phase_bands(ax, V10_2_PHASES, ymin, ymax)

    ax.plot(steps, smooth, color="#ffeb3b", lw=2.2, label=t["ed_legend"])
    ax.fill_between(steps, ymin, smooth, alpha=0.15, color="#ffeb3b")

    ax.set_xlabel(t["ed_xlabel"], color="white", fontsize=11)
    ax.set_ylabel(t["ed_ylabel"], color="white", fontsize=11)
    ax.set_title(t["ed_title"], color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", facecolor="#0a0a14", edgecolor="#444",
              labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlim(0, 50_000_000)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  {t['saved']}: {output_path}")


def collect_positions(replay_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """lê replays e extrai todas as posições (x, y) do trained_player.

    mantida pra compat — o heatmap usa collect_positions_and_meta que
    também retorna posição da bola e o trained_team.
    """
    xs, ys = [], []
    for path in replay_paths:
        try:
            data = json.loads(path.read_text())
            trained_team = data["metadata"].get("trained_team")
            for frame in data["frames"]:
                for p in frame["players"]:
                    if p["team"] == trained_team:
                        xs.append(p["x"])
                        ys.append(p["y"])
        except Exception as e:
            print(f"    ⚠ erro lendo {path.name}: {e}")
    return np.array(xs), np.array(ys)


def collect_positions_and_meta(
    replay_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """versão expandida: retorna posições do player + bola + trained_team."""
    xs, ys, bxs, bys = [], [], [], []
    trained_team = None
    for path in replay_paths:
        try:
            data = json.loads(path.read_text())
            tt = data["metadata"].get("trained_team")
            if tt and not trained_team:
                trained_team = tt
            for frame in data["frames"]:
                ball = frame.get("ball", {})
                bx = ball.get("x", 400.0)
                by = ball.get("y", 250.0)
                for p in frame["players"]:
                    if p["team"] == tt:
                        xs.append(p["x"])
                        ys.append(p["y"])
                        bxs.append(bx)
                        bys.append(by)
                        break
        except Exception as e:
            print(f"    ⚠ erro lendo {path.name}: {e}")
    return (
        np.array(xs), np.array(ys),
        np.array(bxs), np.array(bys),
        trained_team or "?",
    )


def compute_heatmap_stats(xs, ys, bxs, bys, trained_team):
    """estatísticas comportamentais pro overlay no heatmap.

    near_ball_pct: % frames com distância player↔bola < 150 — proxy de engajamento.
    attacking_pct: % frames no campo do adversário. red ataca direita (x>400),
                   blue ataca esquerda (x<400).
    """
    if len(xs) == 0:
        return {}
    dists = np.sqrt((xs - bxs) ** 2 + (ys - bys) ** 2)
    near_ball_pct = float((dists < 150).mean() * 100)
    if trained_team == "red":
        attacking_pct = float((xs > 400).mean() * 100)
    elif trained_team == "blue":
        attacking_pct = float((xs < 400).mean() * 100)
    else:
        attacking_pct = None
    return {"near_ball": near_ball_pct, "attacking": attacking_pct}


def plot_heatmap_comparison(
    early_replays: list[Path],
    late_replays: list[Path],
    output_path: Path,
    t: dict,
) -> None:
    """compara mapa de calor de posições do trained_player nos primeiros
    checkpoints vs os últimos. match exato no nº de replays (4 vs 4).
    """
    print(f"  {t['generating']} {output_path.name}...")
    print(f"    {t['reading_early']} {len(early_replays)} replay(s)")
    ex, ey, _, _, _ = collect_positions_and_meta(early_replays)
    print(f"    {t['reading_late']} {len(late_replays)} replay(s)")
    lx, ly, _, _, _ = collect_positions_and_meta(late_replays)

    if len(ex) == 0 or len(lx) == 0:
        print(f"  {t['no_data']}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)
    fig.patch.set_facecolor("#0a0a14")

    xbins = np.linspace(0, 800, 41)
    ybins = np.linspace(0, 500, 26)

    early_label = t["hm_early_title"].format(len(early_replays))
    late_label  = t["hm_late_title"].format(len(late_replays))
    frames_w = t["hm_frames_suffix"]

    for ax, (xs, ys, title) in zip(
        axes,
        [(ex, ey, f"{early_label}\n{len(ex):,} {frames_w}"),
         (lx, ly, f"{late_label}\n{len(lx):,} {frames_w}")],
    ):
        ax.set_facecolor("#0a0a14")
        H, xedges, yedges = np.histogram2d(xs, ys, bins=[xbins, ybins])
        # log scale pra realçar diferenças.
        H_log = np.log1p(H.T)
        im = ax.imshow(
            H_log, extent=[0, 800, 500, 0], aspect="auto",
            cmap="hot", origin="upper", interpolation="bilinear"
        )
        ax.add_patch(Rectangle((0, 0), 800, 500, fill=False, edgecolor="white", lw=1.5))
        ax.axvline(400, color="white", lw=1, alpha=0.6)
        from matplotlib.patches import Circle as _C
        ax.add_patch(_C((400, 250), 60, fill=False, edgecolor="white", lw=1, alpha=0.6))
        ax.add_patch(Rectangle((0, 70), 120, 360, fill=False, edgecolor="white", lw=1, alpha=0.6))
        ax.add_patch(Rectangle((680, 70), 120, 360, fill=False, edgecolor="white", lw=1, alpha=0.6))
        ax.plot([0, 0], [175, 325], color="yellow", lw=3)
        ax.plot([800, 800], [175, 325], color="yellow", lw=3)

        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(t["hm_suptitle"], color="white", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0a0a14")
    plt.close()
    print(f"  {t['saved']}: {output_path}")


def find_latest_run(pattern: str) -> Optional[Path]:
    """acha a run mais recente cujo nome bate com pattern."""
    runs = sorted(Path("data/runs").glob(f"*{pattern}*"))
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos do treino pra vídeo.")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run pra plotar (default: última v102_b2beta)"
    )
    parser.add_argument(
        "--lang", choices=["pt", "en"], default="pt",
        help="Idioma dos rótulos/títulos (pt=docs/video_assets/portuguese, "
             "en=docs/video_assets/english). Default: pt."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override do output dir (default: docs/video_assets/<portuguese|english>)"
    )
    parser.add_argument(
        "--skip-heatmap", action="store_true",
        help="Pula heatmap (lê replays, mais lento)"
    )
    args = parser.parse_args()

    t = STRINGS[args.lang]

    run_dir = args.run_dir or find_latest_run("v102_b2beta")
    if run_dir is None or not run_dir.exists():
        print(f"❌ Run não encontrada: {run_dir}")
        return 1

    output_dir = args.output_dir or Path(
        "docs/video_assets/" + ("portuguese" if args.lang == "pt" else "english")
    )

    print(f"📊 Gerando plots de: {run_dir} (lang={args.lang})")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(t["loading_tb"])
    scalars = load_tb_scalars(run_dir)
    print(f"   {len(scalars)} {t['loaded_tags']}")

    print("\n" + t["generating_plots"])
    plot_return_per_episode(scalars, output_dir / "return_per_episode.png", t)
    plot_reward_components(scalars, output_dir / "reward_components.png", t)
    plot_episode_length(scalars, output_dir / "episode_length.png", t)
    plot_entropy_decay(scalars, output_dir / "entropy_decay.png", t)

    if not args.skip_heatmap:
        print("\n" + t["generating_heatmap"])
        published_dir = Path("frontend/data/runs") / run_dir.name

        # heatmap início (F0+F1A+F1B+F1C) vs fim (4 últimos F5 — sample-matched
        # pra comparação simétrica, sem subsample dentro do plot).
        early_tags = ["F0", "F1A", "F1B", "F1C"]
        early_replays = []
        for tag in early_tags:
            early_replays.extend(published_dir.glob(f"replay_*_{tag}*.json"))

        late_replays = sorted(
            published_dir.glob("replay_*_F5.json"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
        )[-4:]

        if not early_replays or not late_replays:
            print(f"  {t['replays_not_found']} {published_dir}{t['skipping_heatmap']}")
        else:
            plot_heatmap_comparison(
                early_replays, late_replays,
                output_dir / "heatmap_positions.png",
                t,
            )

    print(f"\n{t['plots_in']} {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
