/**
 * decode action — espelho de backend/ai/actions.py.
 *
 * encoding canônico (bate bit-a-bit com Python):
 *   accel_i ∈ {0=ré (-1), 1=zero (0), 2=frente (+1)}
 *   rot_i   ∈ {0=esq (-1), 1=zero (0), 2=dir (+1)}
 *   kick    ∈ {0=não, 1=sim}
 *   action_idx = (accel_i * 3 + rot_i) * 2 + kick
 *
 * total: 3 × 3 × 2 = 18 ações.
 */
(function () {
    "use strict";

    var ACTION_SPACE_SIZE = 18;

    /**
     * decodifica um índice de ação em [accel, rot, kick].
     *
     * @param {number} actionIdx inteiro em [0, 17].
     * @returns {Array<number>} [accel, rot, kick] em {-1,0,+1} × {-1,0,+1} × {0,1}.
     * @throws {RangeError} se actionIdx fora de [0, 17] ou não-inteiro.
     */
    function decodeAction(actionIdx) {
        if (
            !Number.isInteger(actionIdx) ||
            actionIdx < 0 ||
            actionIdx >= ACTION_SPACE_SIZE
        ) {
            throw new RangeError(
                "actionIdx deve ser inteiro em [0, " +
                (ACTION_SPACE_SIZE - 1) +
                "], recebido: " + actionIdx
            );
        }
        var kick = actionIdx % 2;
        var rest = (actionIdx - kick) / 2;
        var rotI = rest % 3;
        var accelI = (rest - rotI) / 3;
        return [accelI - 1, rotI - 1, kick];
    }

    var api = {
        ACTION_SPACE_SIZE: ACTION_SPACE_SIZE,
        decodeAction: decodeAction,
    };

    if (typeof window !== "undefined") {
        window.AISoccer = window.AISoccer || {};
        window.AISoccer.Actions = api;
    }
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})();
