"""fine-tune cirúrgico de checkpoint existente.

carrega final.pt de uma run base e treina mais N timesteps com curriculum
customizado, sem mexer no checkpoint original (escreve em run nova).

garantias anti-regressão:
    - carrega checkpoint base (não começa do zero).
    - 25% anti-forgetting em fases base.
    - run de saída separada — original permanece intocado.
    - progress=1.0 fixo: goal_width sempre em range realista (μ=195).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from backend.ai.env import CurriculumPhase
from backend.ai.train import PPOConfig, PPOTrainer


# fração de cada fase no fine-tune (deve somar 1.0).
FT_PHASE_FRACTIONS: dict[str, float] = {
    "1A":  0.15,
    "1E":  0.20,
    "3DA": 0.25,
    "1D":  0.05,
    "3D":  0.05,
    "5":   0.30,
}

# pool de rotação anti-forgetting durante self-play (fase 5):
# peso maior em fases novas (1E, 3DA), algumas existentes pra não regredir.
FT_ROTATION_WEIGHTS: dict[str, float] = {
    "1A":  0.20,
    "1E":  0.25,
    "1D":  0.20,
    "3D":  0.15,
    "3DA": 0.20,
}


class FineTuneTrainer(PPOTrainer):
    """subclass de PPOTrainer com curriculum custom para fine-tune.

    overrides:
        _current_phase: usa FT_PHASE_FRACTIONS (sequência custom).
        _rotation_pool + _rotation_weights_for: usa fases novas.
        _set_progress_for_envs: força progress=1.0 fixo (gol pequeno).
    """

    def __init__(self, config: PPOConfig, run_name: str, checkpoint_path: Path):
        super().__init__(config, run_name=run_name)
        self.logger.info(f"Carregando checkpoint base: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        # load_checkpoint atualiza self.global_step pro step original.
        # zerar para que o curriculum custom funcione em [0, total_timesteps].
        ckpt_step = self.global_step
        self.global_step = 0
        self.logger.info(
            f"Checkpoint carregado (step original={ckpt_step:,}). "
            f"global_step zerado para fine-tune."
        )

    def _current_phase(self) -> str:
        """retorna a fase atual conforme FT_PHASE_FRACTIONS."""
        total = self.config.total_timesteps
        cumulative = 0.0
        # dict mantém ordem de inserção (Python 3.7+) — essa é a ordem do curriculum.
        for phase, frac in FT_PHASE_FRACTIONS.items():
            cumulative += frac
            end_step = int(cumulative * total)
            if self.global_step < end_step:
                return phase
        return list(FT_PHASE_FRACTIONS.keys())[-1]

    def _rotation_pool(self) -> list[str]:
        return list(FT_ROTATION_WEIGHTS.keys())

    def _rotation_weights_for(self, pool: list[str]) -> Optional[list[float]]:
        weights = [float(FT_ROTATION_WEIGHTS.get(phase, 0.0)) for phase in pool]
        if sum(weights) <= 0.0:
            return None
        return weights

    def _set_progress_for_envs(self, mask) -> None:
        """como o método pai, mas força progress=1.0 (gol pequeno realista).

        garante que goal_width sempre é amostrado com μ=195 (final do treino),
        evitando regressão de precisão pra μ=280 (gol grande).
        """
        if not self.config.enable_curriculum:
            return
        if not any(mask):
            return

        if not self.config.enable_curriculum_phases:
            self.envs.call("set_progress", 1.0)
            return

        base_phase = self._current_phase()
        # rotation só ativa em fase 5 (self-play); demais fases são "ensino"
        # e não rotacionam — comportamento conservador.
        learning_phases = {"1A", "1E", "3DA", "1D", "3D"}
        rotation_prob = (
            self.config.anti_forgetting_rotation_prob
            if base_phase not in learning_phases
            else 0.0
        )
        rotation_pool = self._rotation_pool()
        rotation_weights = self._rotation_weights_for(rotation_pool)

        phases: list = []
        for do_set in mask:
            if not do_set:
                phases.append(None)
                continue
            if rotation_prob > 0.0 and self._curriculum_rng.random() < rotation_prob:
                if rotation_weights is not None:
                    sampled = self._curriculum_rng.choices(
                        rotation_pool, weights=rotation_weights, k=1
                    )[0]
                else:
                    sampled = rotation_pool[
                        self._curriculum_rng.randrange(len(rotation_pool))
                    ]
                phases.append(sampled)
            else:
                phases.append(base_phase)

        if hasattr(self.envs, "envs"):
            for i, p in enumerate(phases):
                if p is not None:
                    self.envs.envs[i].set_progress(1.0, phase=p)
        else:
            self.envs.call("set_progress_indexed", 1.0, phases)


def _find_default_checkpoint() -> Optional[Path]:
    """procura o checkpoint v94_b2beta mais recente."""
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        return None
    matches = sorted(runs_dir.glob("*v94_b2beta*/checkpoints/final.pt"))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune V9.5 do checkpoint v94_b2beta")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path para final.pt. Default: última run v94_b2beta encontrada.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10_000_000,
        help="Total de timesteps no fine-tune (default 10M ≈ 50min).",
    )
    parser.add_argument(
        "--name", type=str, default="v94_b2beta_ft",
        help="Nome da run de fine-tune (sufixo após timestamp).",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Não publica auto no frontend após o FT.",
    )
    parser.add_argument(
        "--frontend-data", type=Path, default=Path("frontend/data"),
        help="Pasta destino do auto-publish.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or _find_default_checkpoint()
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint}. "
            f"Use --checkpoint /path/to/final.pt ou rode treino V9.4 primeiro."
        )

    total_frac = sum(FT_PHASE_FRACTIONS.values())
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"FT_PHASE_FRACTIONS deve somar 1.0, mas soma {total_frac}. "
            f"Corrija em finetune.py."
        )

    # config: herda hyperparams do treino base com tweaks:
    # learning_rate menor (1e-4): tunning fino, não exploração.
    # ent_coef pequeno: pouca exploração — refina policy existente.
    # anti_forgetting_rotation_prob 0.50: alto pra preservar habilidades.
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=8,
        num_steps=256,
        learning_rate=1e-4,
        anneal_lr=True,
        update_epochs=4,
        num_minibatches=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.005,
        ent_coef_end=0.001,
        vf_coef=0.5,
        clip_vloss=True,
        max_grad_norm=0.5,
        norm_obs=True,
        norm_reward=True,
        enable_self_play=True,
        save_steps=500_000,
        fixed_random_opponent=False,
        async_sync_pool_to_disk=True,
        enable_curriculum=True,
        enable_curriculum_phases=True,
        # phase_*_end_step são ignorados pelo override de _current_phase,
        # mas PPOConfig exige todos definidos — valores fictícios servem.
        phase_1a_end_step=1, phase_1b_end_step=1, phase_1c_end_step=1,
        phase_1d_end_step=1, phase_2_end_step=1, phase_3_end_step=1,
        phase_3d_end_step=1, phase_3t_end_step=1, phase_4_end_step=1,
        anti_forgetting_rotation_prob=0.50,
        rotation_weights=None,
        log_frequency_updates=1,
        log_to_stdout_every_n_updates=5,
        seed=args.seed,
        device="cpu",
        force_sync_vec_env=False,
    )

    print("=" * 70)
    print("V9.5 Fine-tune — curriculum cirúrgico")
    print("=" * 70)
    print(f"Checkpoint base: {checkpoint}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Curriculum (FT_PHASE_FRACTIONS):")
    cumulative = 0.0
    for phase, frac in FT_PHASE_FRACTIONS.items():
        cumulative += frac
        end = int(cumulative * config.total_timesteps)
        steps = int(frac * config.total_timesteps)
        print(f"  {phase:4s}: {steps:>10,} steps ({frac*100:.0f}%)  → end_step={end:>10,}")
    print(f"Anti-forgetting: {config.anti_forgetting_rotation_prob*100:.0f}% (rotation_weights={FT_ROTATION_WEIGHTS})")
    print(f"goal_width: progress=1.0 fixo → μ=195 (gol pequeno realista)")
    print("=" * 70)

    trainer = FineTuneTrainer(config, run_name=args.name, checkpoint_path=checkpoint)
    trainer.train()

    final_ckpt = trainer.run_dir / "checkpoints" / "final.pt"
    if not args.no_publish and final_ckpt.exists():
        from backend.ai.publish_run import publish_run
        try:
            args.frontend_data.mkdir(parents=True, exist_ok=True)
            publish_run(trainer.run_dir, args.frontend_data)
            print(
                f"\n🌐 Fine-tune publicado. Pra visualizar:"
                f"\n   cd frontend && python -m http.server 8000"
                f"\n   Acesse: http://localhost:8000"
            )
        except Exception as e:
            print(
                f"\n⚠️  Auto-publish falhou ({e}). Fine-tune concluído com sucesso. "
                f"Pra publicar manualmente:"
                f"\n   python -m backend.ai.publish_run --run-dir {trainer.run_dir}"
            )


if __name__ == "__main__":
    main()
