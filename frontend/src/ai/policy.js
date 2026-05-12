/**
 * MlpNetwork — espelho de backend/ai/policy.py::Policy em JS puro.
 *
 * carrega final_policy.json (gerado por export_for_js.py) e expõe:
 *   net.activate(player, state)  → [accel, rot, kick]      (produção)
 *   net.activateRaw(obsRaw)      → action_idx (0..17)      (testes)
 *   net.forwardLogits(obsRaw)    → Float32Array[18]        (debug)
 *
 * forward pass (espelho de Policy.select_action_greedy):
 *   1. obs raw (341) → normaliza com obs_rms (se presente)
 *   2. h1 = tanh(W_fc1 @ x + b_fc1)
 *   3. h2 = tanh(W_fc2 @ h1 + b_fc2)
 *   4. logits = W_policy @ h2 + b_policy
 *   5. action_idx = argmax(logits)
 *   6. [accel, rot, kick] = decodeAction(action_idx)
 *
 * normalização (espelho de RunningMeanStd.normalize):
 *   z = (x - mean) / sqrt(var + 1e-8); clamp em [-10, 10].
 *
 * convenção de pesos do export:
 *   weights[k] é Array<Array<number>> ou Array<number> (shape (out, in)).
 *   matmul: W @ x → (out, in) @ (in,) = (out,).
 */
(function () {
    "use strict";

    function MlpNetwork(modelData) {
        this._validateModelData(modelData);
        this.schemaVersion = modelData.schema_version;
        this.metadata = modelData.metadata || {};

        var w = modelData.weights;
        this.fc1W = this._toMatrix(w["fc1.weight"]);
        this.fc1B = new Float32Array(w["fc1.bias"]);
        this.fc2W = this._toMatrix(w["fc2.weight"]);
        this.fc2B = new Float32Array(w["fc2.bias"]);
        this.policyW = this._toMatrix(w["policy_head.weight"]);
        this.policyB = new Float32Array(w["policy_head.bias"]);

        // obs_rms é opcional no schema; se presente, usar é obrigatório
        // pra distribuição de inferência bater com a de treino.
        if (modelData.obs_rms) {
            this.obsMean = new Float32Array(modelData.obs_rms.mean);
            this.obsVar = new Float32Array(modelData.obs_rms.var);
        } else {
            this.obsMean = null;
            this.obsVar = null;
        }

        // buffers reutilizáveis — evita realloc por chamada (chamado a 60Hz).
        this._h1 = new Float32Array(this.fc1W.rows);
        this._h2 = new Float32Array(this.fc2W.rows);
        this._logits = new Float32Array(this.policyW.rows);
        this._normalizedObs = new Float32Array(this.fc1W.cols);
    }

    MlpNetwork.prototype._validateModelData = function (modelData) {
        if (!modelData) throw new Error("modelData é nulo");
        if (modelData.schema_version !== 1) {
            throw new Error(
                "schema_version " + modelData.schema_version + " não suportada"
            );
        }
        if (!modelData.weights) {
            throw new Error("modelData.weights ausente");
        }
        // value_head não é exigido — inferência usa só policy head + body.
        var requiredKeys = [
            "fc1.weight", "fc1.bias",
            "fc2.weight", "fc2.bias",
            "policy_head.weight", "policy_head.bias",
        ];
        for (var i = 0; i < requiredKeys.length; i++) {
            if (!modelData.weights[requiredKeys[i]]) {
                throw new Error("weights ausente: " + requiredKeys[i]);
            }
        }
    };

    /** converte Array<Array<number>> em Float32Array flat (cache-friendly). */
    MlpNetwork.prototype._toMatrix = function (nested) {
        var rows = nested.length;
        var cols = nested[0].length;
        var flat = new Float32Array(rows * cols);
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                flat[r * cols + c] = nested[r][c];
            }
        }
        return { data: flat, rows: rows, cols: cols };
    };

    /**
     * inferência completa: player + state → [accel, rot, kick].
     */
    MlpNetwork.prototype.activate = function (player, state) {
        var obsRaw = AISoccer.Obs.gatherObs(player, state, 0);
        var actionIdx = this.activateRaw(obsRaw);
        return AISoccer.Actions.decodeAction(actionIdx);
    };

    /** inferência sobre obs já construído (testes de paridade). */
    MlpNetwork.prototype.activateRaw = function (obsRaw) {
        var logits = this.forwardLogits(obsRaw);
        return this._argmax(logits);
    };

    /**
     * forward pass com activations completas — usado pela visualização.
     * retorna referências aos buffers internos (leitura only — não modificar).
     */
    MlpNetwork.prototype.forwardWithActivations = function (obsRaw) {
        var logits = this.forwardLogits(obsRaw);
        var inputAct = (this.obsMean !== null) ? this._normalizedObs : obsRaw;
        return {
            input: inputAct,
            h1: this._h1,
            h2: this._h2,
            logits: logits,
            actionIdx: this._argmax(logits),
        };
    };

    /** forward pass até logits (sem argmax). útil pra debug/testes. */
    MlpNetwork.prototype.forwardLogits = function (obsRaw) {
        var obs;
        if (this.obsMean !== null) {
            obs = this._normalize(obsRaw);
        } else {
            obs = obsRaw;
        }

        this._matmulAddTanh(this.fc1W, obs, this.fc1B, this._h1);
        this._matmulAddTanh(this.fc2W, this._h1, this.fc2B, this._h2);
        this._matmulAdd(this.policyW, this._h2, this.policyB, this._logits);

        return this._logits;
    };

    /**
     * (x - mean) / sqrt(var + 1e-8); clamp em [-10, 10].
     * espelho exato de RunningMeanStd.normalize do train.py.
     */
    MlpNetwork.prototype._normalize = function (obsRaw) {
        var n = obsRaw.length;
        for (var i = 0; i < n; i++) {
            var std = Math.sqrt(this.obsVar[i] + 1e-8);
            var z = (obsRaw[i] - this.obsMean[i]) / std;
            if (z > 10.0) z = 10.0;
            else if (z < -10.0) z = -10.0;
            this._normalizedObs[i] = z;
        }
        return this._normalizedObs;
    };

    /** out[i] = tanh(sum_j W[i,j] * x[j] + b[i]) */
    MlpNetwork.prototype._matmulAddTanh = function (W, x, b, out) {
        var rows = W.rows;
        var cols = W.cols;
        var data = W.data;
        for (var i = 0; i < rows; i++) {
            var sum = b[i];
            var rowOffset = i * cols;
            for (var j = 0; j < cols; j++) {
                sum += data[rowOffset + j] * x[j];
            }
            out[i] = Math.tanh(sum);
        }
    };

    /** out[i] = sum_j W[i,j] * x[j] + b[i]  (sem activation) */
    MlpNetwork.prototype._matmulAdd = function (W, x, b, out) {
        var rows = W.rows;
        var cols = W.cols;
        var data = W.data;
        for (var i = 0; i < rows; i++) {
            var sum = b[i];
            var rowOffset = i * cols;
            for (var j = 0; j < cols; j++) {
                sum += data[rowOffset + j] * x[j];
            }
            out[i] = sum;
        }
    };

    MlpNetwork.prototype._argmax = function (arr) {
        var best = 0;
        var bestVal = arr[0];
        for (var i = 1; i < arr.length; i++) {
            if (arr[i] > bestVal) {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    };

    if (typeof window !== "undefined") {
        window.AISoccer = window.AISoccer || {};
        window.AISoccer.MlpNetwork = MlpNetwork;
    }
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { MlpNetwork: MlpNetwork };
    }
})();
