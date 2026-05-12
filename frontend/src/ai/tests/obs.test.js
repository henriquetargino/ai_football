// o obs.js depende de AISoccer.Physics.castRays. Como rodamos em jest
// (Node) sem o physics.js completo, fornecemos um stub via global.AISoccer.
// (No browser, isso seria injetado pelo physics.js carregado antes.)

global.AISoccer = global.AISoccer || {};
global.AISoccer.Physics = {
    castRays: null, // setado por cada teste
};

// em Node, o IIFE do obs.js exporta via module.exports (window é undefined).
const obsModule = require("../obs");

function _player(overrides) {
    return Object.assign(
        { x: 0, y: 0, vx: 0, vy: 0, angle: 0, can_kick: true, rot: 0 },
        overrides || {},
    );
}

describe("gatherObs", () => {
    test("returns Float32Array of length 341", () => {
        global.AISoccer.Physics.castRays = () => new Float32Array(96).fill(0.5);
        const player = _player();
        const state = { players: [player], ball: { x: 100, y: 0 } };
        const obs = obsModule.gatherObs(player, state, 0);
        expect(obs).toBeInstanceOf(Float32Array);
        expect(obs.length).toBe(341);
    });

    test("sentinel: type=none → distance=-1.0 + one_hot zeros", () => {
        const fakeRays = new Float32Array(96);
        for (let i = 0; i < 48; i++) {
            fakeRays[i * 2] = 1.0;       // dist (irrelevante)
            fakeRays[i * 2 + 1] = -1.0;  // code = none
        }
        global.AISoccer.Physics.castRays = () => fakeRays;
        const player = _player({ can_kick: false });
        const state = { players: [player], ball: { x: 0, y: 0 } };
        const obs = obsModule.gatherObs(player, state, 0);

        for (let i = 0; i < 48; i++) {
            const base = i * 7;
            expect(obs[base]).toBe(-1.0);
            for (let bit = 1; bit < 7; bit++) {
                expect(obs[base + bit]).toBe(0.0);
            }
        }
        expect(obs[339]).toBe(0.0); // ball_visible_anywhere
    });

    test("each ray has at most ONE bit set in one-hot", () => {
        const fakeRays = new Float32Array(96);
        const codes = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        for (let i = 0; i < 48; i++) {
            fakeRays[i * 2] = 0.5;
            fakeRays[i * 2 + 1] = codes[i % codes.length];
        }
        global.AISoccer.Physics.castRays = () => fakeRays;
        const player = _player();
        const state = { players: [player], ball: { x: 0, y: 0 } };
        const obs = obsModule.gatherObs(player, state, 0);

        for (let i = 0; i < 48; i++) {
            let nbits = 0;
            for (let bit = 1; bit < 7; bit++) nbits += obs[i * 7 + bit];
            expect(nbits === 0 || nbits === 1).toBe(true);
        }
    });

    test("cada tipo aceso → bit correto no one-hot", () => {
        // constrói raio 0 com cada tipo conhecido e valida o slot.
        const expectedSlot = {
            "wall_0.0": 0,
            "ball_0.2": 1,
            "ally_0.4": 2,
            "enemy_0.6": 3,
            "goal_own_0.8": 4,
            "goal_enemy_1.0": 5,
        };
        const codes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const labels = ["wall_0.0", "ball_0.2", "ally_0.4", "enemy_0.6", "goal_own_0.8", "goal_enemy_1.0"];

        for (let k = 0; k < codes.length; k++) {
            const fakeRays = new Float32Array(96);
            fakeRays[0] = 0.7; // dist
            fakeRays[1] = codes[k];
            // raios 1..47 são "none"
            for (let i = 1; i < 48; i++) fakeRays[i * 2 + 1] = -1.0;

            global.AISoccer.Physics.castRays = () => fakeRays;
            const player = _player();
            const state = { players: [player], ball: { x: 0, y: 0 } };
            const obs = obsModule.gatherObs(player, state, 0);

            const slot = expectedSlot[labels[k]];
            // distância correta + bit certo aceso, demais zero.
            expect(Math.abs(obs[0] - 0.7) < 1e-6).toBe(true);
            for (let bit = 0; bit < 6; bit++) {
                const expected = bit === slot ? 1.0 : 0.0;
                expect(obs[1 + bit]).toBe(expected);
            }
        }
    });

    test("state_extra calculados corretamente", () => {
        global.AISoccer.Physics.castRays = () => new Float32Array(96);
        // vel = (2.1, 0) → speed = 2.1/3.5 = 0.6
        const player = _player({ vx: 2.1, vy: 0, rot: 0.5, can_kick: true });
        const state = { players: [player], ball: { x: 0, y: 0 } };
        const obs = obsModule.gatherObs(player, state, 2);

        expect(Math.abs(obs[336] - 0.6) < 1e-6).toBe(true); // speed
        expect(Math.abs(obs[337] - 0.5) < 1e-6).toBe(true); // angularVel
        expect(obs[338]).toBe(1.0);                          // can_kick
        expect(obs[340]).toBeCloseTo(0.5, 5);                // 2/4 = 0.5
    });

    test("ball_visible_anywhere=1 quando algum raio detecta ball", () => {
        const fakeRays = new Float32Array(96);
        for (let i = 0; i < 48; i++) {
            fakeRays[i * 2 + 1] = -1.0; // none default
        }
        // raio 5 vê ball
        fakeRays[5 * 2] = 0.3;
        fakeRays[5 * 2 + 1] = 0.2; // ball
        global.AISoccer.Physics.castRays = () => fakeRays;

        const player = _player();
        const state = { players: [player], ball: { x: 100, y: 0 } };
        const obs = obsModule.gatherObs(player, state, 0);
        expect(obs[339]).toBe(1.0);
    });

    test("invalid actionRepeatIdx throws RangeError", () => {
        global.AISoccer.Physics.castRays = () => new Float32Array(96);
        const player = _player();
        const state = { players: [player], ball: { x: 0, y: 0 } };
        expect(() => obsModule.gatherObs(player, state, 4)).toThrow(RangeError);
        expect(() => obsModule.gatherObs(player, state, -1)).toThrow(RangeError);
        expect(() => obsModule.gatherObs(player, state, 1.5)).toThrow(RangeError);
    });
});
