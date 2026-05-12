"""gera fixtures de paridade pra testes jest.

output em frontend/src/ai/tests/fixtures/:
    parity_policy.json       — policy mini exportada (mesmo formato do
                               final_policy.json de produção).
    parity_obs_fixtures.json — 200 obs aleatórias + actions + logits
                               esperados (computados em Python).

o JS lê esses JSONs e valida que reconstruir o forward em JS produz
mesma action e logits próximos.

uso:
    python -m backend.scripts.generate_js_test_fixtures
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from backend.ai.export_for_js import export_policy
from backend.ai.obs import OBS_SIZE
from backend.ai.policy import Policy
from backend.ai.train import RunningMeanStd


OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "frontend" / "src" / "ai" / "tests" / "fixtures"
)
N_FIXTURES = 200
RMS_PRIME_SAMPLES = 10_000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = Policy()
    policy.eval()

    # prime obs_rms com mean/var não-triviais — simula o efeito de treino real.
    obs_rms = RunningMeanStd((OBS_SIZE,))
    obs_rms.update(
        rng.standard_normal((RMS_PRIME_SAMPLES, OBS_SIZE)).astype(np.float32)
    )

    # salva checkpoint temporário e exporta via mesmo caminho da produção.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        torch.save(
            {
                "global_step": 0,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": {},
                "obs_rms": obs_rms.state_dict(),
                "rew_rms": None,
                "config": {},
                "run_name": "fixture",
            },
            ckpt_path,
        )
        export_policy(ckpt_path, OUT_DIR / "parity_policy.json")

    fixtures = []
    for _ in range(N_FIXTURES):
        obs_raw = rng.standard_normal(OBS_SIZE).astype(np.float32)

        # mesmo pipeline de MlpNetwork.activate em JS:
        #   1. normalize raw  2. forward  3. argmax
        obs_norm = obs_rms.normalize(obs_raw[None])[0]

        with torch.no_grad():
            x = torch.from_numpy(obs_norm).unsqueeze(0)
            body = torch.tanh(policy.fc2(torch.tanh(policy.fc1(x))))
            logits = policy.policy_head(body).squeeze(0).numpy()

        action_py = int(np.argmax(logits))

        fixtures.append({
            "obs_raw": obs_raw.tolist(),
            "expected_action": action_py,
            "expected_logits": logits.astype(np.float32).tolist(),
        })

    out_path = OUT_DIR / "parity_obs_fixtures.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "obs_size": OBS_SIZE,
                "fixtures": fixtures,
            },
            f,
        )

    print(f"Wrote {len(fixtures)} fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
