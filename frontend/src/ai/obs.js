/**
 * build observation — espelho de backend/ai/obs.py.
 *
 * estrutura canônica (341 floats):
 *   [0..335]  336 floats raycasts (48 raios × 7: distance + 6 one-hot)
 *   [336]     speed normalizado [0, 1]
 *   [337]     angularVel ∈ [-1, 1]
 *   [338]     canKick ∈ {0, 1}
 *   [339]     ball_visible_anywhere ∈ {0, 1}
 *   [340]     action_repeat_idx / 4 ∈ {0, 0.25, 0.5, 0.75}
 *
 * por raio (7 floats):
 *   [0]  distance normalizada ∈ [0, 1]  OU  -1.0 (sentinela "none")
 *   [1]  one_hot wall          ∈ {0, 1}
 *   [2]  one_hot ball          ∈ {0, 1}
 *   [3]  one_hot ally          ∈ {0, 1}
 *   [4]  one_hot enemy         ∈ {0, 1}
 *   [5]  one_hot goal_own      ∈ {0, 1}
 *   [6]  one_hot goal_enemy    ∈ {0, 1}
 *
 * sentinela canônica: distance = -1.0, one_hot = [0,0,0,0,0,0] → "nada detectado".
 * mapeamento: wall=0, ball=1, ally=2, enemy=3, goal_own=4, goal_enemy=5.
 *
 * depende de AISoccer.Physics.castRays(player, state) retornar 96 floats
 * [dist, type_code, dist, type_code, ...] (ver frontend/src/game/physics.js).
 */
(function () {
    "use strict";

    var OBS_SIZE = 341;
    var NUM_RAYS = 48;
    var RAY_FEATURE_SIZE = 7;
    var STATE_EXTRA_SIZE = 5;
    var STATE_EXTRA_BASE = NUM_RAYS * RAY_FEATURE_SIZE; // 336

    // mapeamento canônico tipo → índice no one-hot. replicar exato em Python.
    var TYPE_TO_ONEHOT_IDX = {
        wall: 0,
        ball: 1,
        ally: 2,
        enemy: 3,
        goal_own: 4,
        goal_enemy: 5,
    };

    // bate com backend/config.py.
    var MAX_SPEED_PLAYER = 3.5;

    /**
     * decodifica o type_code (do TYPE_MAP de physics.js) num nome canônico.
     * espelho exato de physics.js::codeToType.
     */
    function _codeToTypeName(code) {
        if (code < -0.5) return "none";
        if (code < 0.1) return "wall";
        if (code < 0.3) return "ball";
        if (code < 0.5) return "ally";
        if (code < 0.7) return "enemy";
        if (code < 0.9) return "goal_own";
        return "goal_enemy";
    }

    /**
     * constrói o vetor de observação de 341 floats para o player.
     *
     * @param {Object} player Player vivo do GameState.
     * @param {Object} state  GameState completo.
     * @param {number} actionRepeatIdx frame atual no ciclo de action repeat (0..3).
     * @returns {Float32Array} length 341.
     * @throws {RangeError} se actionRepeatIdx fora de [0, 3] ou não-inteiro.
     */
    function gatherObs(player, state, actionRepeatIdx) {
        if (actionRepeatIdx === undefined) actionRepeatIdx = 0;
        if (
            !Number.isInteger(actionRepeatIdx) ||
            actionRepeatIdx < 0 ||
            actionRepeatIdx > 3
        ) {
            throw new RangeError(
                "actionRepeatIdx deve ser inteiro em [0, 3], recebido: " +
                actionRepeatIdx
            );
        }

        var obs = new Float32Array(OBS_SIZE);
        var rayData = AISoccer.Physics.castRays(player, state);

        var ballVisible = false;

        for (var i = 0; i < NUM_RAYS; i++) {
            var dist = rayData[i * 2];
            var typeCode = rayData[i * 2 + 1];
            var typeName = _codeToTypeName(typeCode);
            var base = i * RAY_FEATURE_SIZE;

            if (typeName === "none") {
                obs[base] = -1.0;
            } else {
                obs[base] = dist;
                var onehotIdx = TYPE_TO_ONEHOT_IDX[typeName];
                obs[base + 1 + onehotIdx] = 1.0;
                if (typeName === "ball") ballVisible = true;
            }
        }

        var speed =
            Math.sqrt(player.vx * player.vx + player.vy * player.vy) /
            MAX_SPEED_PLAYER;
        obs[STATE_EXTRA_BASE + 0] = Math.max(0.0, Math.min(1.0, speed));
        var rot = player.rot || 0.0;
        obs[STATE_EXTRA_BASE + 1] = Math.max(-1.0, Math.min(1.0, rot));
        obs[STATE_EXTRA_BASE + 2] = player.can_kick ? 1.0 : 0.0;
        obs[STATE_EXTRA_BASE + 3] = ballVisible ? 1.0 : 0.0;
        obs[STATE_EXTRA_BASE + 4] = actionRepeatIdx / 4.0;

        return obs;
    }

    var api = {
        OBS_SIZE: OBS_SIZE,
        NUM_RAYS: NUM_RAYS,
        RAY_FEATURE_SIZE: RAY_FEATURE_SIZE,
        STATE_EXTRA_SIZE: STATE_EXTRA_SIZE,
        TYPE_TO_ONEHOT_IDX: TYPE_TO_ONEHOT_IDX,
        gatherObs: gatherObs,
    };

    if (typeof window !== "undefined") {
        window.AISoccer = window.AISoccer || {};
        window.AISoccer.Obs = api;
    }
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})();
