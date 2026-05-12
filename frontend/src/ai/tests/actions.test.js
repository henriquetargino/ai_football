const { decodeAction, ACTION_SPACE_SIZE } = require("../actions");

describe("decodeAction", () => {
    test("ACTION_SPACE_SIZE is 18", () => {
        expect(ACTION_SPACE_SIZE).toBe(18);
    });

    test("action 8 is NOOP (accel=0, rot=0, kick=0)", () => {
        // (1, 1, 0): (1*3+1)*2 + 0 = 8
        expect(decodeAction(8)).toEqual([0, 0, 0]);
    });

    test("action 15 is frente+chute (accel=+1, rot=0, kick=1)", () => {
        // (2, 1, 1): (2*3+1)*2 + 1 = 15
        expect(decodeAction(15)).toEqual([1, 0, 1]);
    });

    test("action 0 is ré-esquerda-sem chute (accel=-1, rot=-1, kick=0)", () => {
        // (0, 0, 0): (0*3+0)*2 + 0 = 0
        expect(decodeAction(0)).toEqual([-1, -1, 0]);
    });

    test("action 17 is frente-direita-chuta (accel=+1, rot=+1, kick=1)", () => {
        // (2, 2, 1): (2*3+2)*2 + 1 = 17
        expect(decodeAction(17)).toEqual([1, 1, 1]);
    });

    test("all 18 actions decode to valid triplets", () => {
        for (let i = 0; i < 18; i++) {
            const [a, r, k] = decodeAction(i);
            expect([-1, 0, 1]).toContain(a);
            expect([-1, 0, 1]).toContain(r);
            expect([0, 1]).toContain(k);
        }
    });

    test("invalid actionIdx throws RangeError", () => {
        expect(() => decodeAction(-1)).toThrow(RangeError);
        expect(() => decodeAction(18)).toThrow(RangeError);
        expect(() => decodeAction(1.5)).toThrow(RangeError);
        expect(() => decodeAction("3")).toThrow(RangeError);
    });

    test("encoding round-trip matches Python convention", () => {
        // (accel_i*3 + rot_i)*2 + kick = idx, e decode é inverso.
        for (let accel_i = 0; accel_i < 3; accel_i++) {
            for (let rot_i = 0; rot_i < 3; rot_i++) {
                for (let kick = 0; kick < 2; kick++) {
                    const idx = (accel_i * 3 + rot_i) * 2 + kick;
                    expect(decodeAction(idx)).toEqual([accel_i - 1, rot_i - 1, kick]);
                }
            }
        }
    });
});
