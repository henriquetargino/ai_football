# AI Soccer — 1v1 Reinforcement Learning

Simulação de futebol 1v1 onde robôs aprendem a jogar com PPO (reinforcement learning). Backend Python treina offline; frontend Three.js exibe replays, modo "jogar contra a IA" e visualização in-game da rede neural — tudo estático, hospedável no GitHub Pages.

> **Modelo de produção:** V10.2 (50M steps, treino fresh — não fine-tune).

---

## Estrutura

```
warehouse futebol/
├── backend/
│   ├── ai/              # PPO + env + policy export (Python, sb3)
│   ├── physics/         # Engine física (paridade 1:1 com frontend/src/game/physics.js)
│   ├── scripts/         # Utilitários (gera fixtures pra tests JS)
│   └── config.py        # Constantes físicas (KICK_FORCE, MAX_SPEED_BALL, etc.)
├── frontend/
│   ├── index.html       # Entry point — menu + 3 modos (replay, vs AI, AI vs AI)
│   ├── src/
│   │   ├── ai/          # Inferência JS (MlpNetwork, obs, actions) — paridade Python
│   │   ├── game/        # physics.js (port do backend pra JS)
│   │   ├── renderer/    # Three.js: campo, robôs, gols, partículas, neural viz
│   │   └── replay/      # ReplayPlayer + controles (timeline, speed)
│   └── data/            # Snapshots e replays publicados (JSON)
├── scripts/
│   └── generate_training_plots.py   # Gráficos pra vídeo (TB logs → PNGs)
├── docs/video_assets/   # PNGs gerados (gráficos de treino, en/pt)
└── data/runs/           # Checkpoints + tensorboard logs (raw, não commitado)
```

---

## Setup

**Python (backend):**
```bash
conda create -n neat-env python=3.11
conda activate neat-env
pip install stable-baselines3 torch tensorboard numpy
# (opcional pra plots) pip install matplotlib
```

**Node (frontend):**
```bash
cd frontend && npm install      # apenas pra rodar tests jest
```

Não há build step no frontend — é vanilla JS servido como assets estáticos.

---

## Treinar

```bash
# Treino completo (50M steps, ~12-18h em CPU moderna)
python -m backend.ai.train

# Continuar de checkpoint
python -m backend.ai.train --resume data/runs/<RUN_ID>/checkpoints/<STEP>.zip
```

Tensorboard:
```bash
tensorboard --logdir data/runs/<RUN_ID>/tensorboard
```

Curriculum (fases) e hiperparâmetros estão hardcoded em `backend/ai/train.py`. Para tweaks, editar lá.

---

## Publicar modelo pro frontend

```bash
# Exporta policy + replays pra frontend/data/runs/<RUN_ID>/
python -m backend.ai.publish_run data/runs/<RUN_ID>
```

Isso gera:
- `policy_<STEP>.json` — pesos da rede em formato MLP carregável pelo JS
- `replay_<STEP>_<PHASE>.json` — replays gravados durante o treino
- `manifest.json` — índice consumido pelo menu do frontend

---

## Rodar o frontend

```bash
python3 -m http.server 8080 --directory frontend
```

Acesse `http://localhost:8080`.

**Modos disponíveis:**
- **Watch the Replay** — tocar um replay gravado durante o treino
- **Take on the AI** — você joga (WASD + Espaço chuta) contra um snapshot da IA
- **AI vs AI** — dois snapshots se enfrentam ao vivo (inferência JS pura)

**Atalhos in-game:** `R` raycasts, `N` painel da rede neural, `ESC` volta ao menu.

---

## Tests

**Frontend (paridade Python ↔ JS):**
```bash
cd frontend && npm test
```

**Backend:**
```bash
pytest backend/ai/tests/ -v
```

---

## Gráficos do treino

```bash
python -m scripts.generate_training_plots
# Saída em docs/video_assets/*.png
```

Gera: return per episode, reward components, episode length, entropy decay, heatmap de posições.

---

## Licença

Projeto pessoal de portfólio. Sem licença explícita — copyright reservado ao autor (Henrique Targino).
