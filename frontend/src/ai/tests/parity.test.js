/**
 * paridade Python ↔ JS.
 *
 * lê fixtures geradas pelo Python (backend/scripts/generate_js_test_fixtures.py)
 * e valida que o forward em JS produz mesma action que o Python e logits
 * próximos dentro da precisão Float32.
 */
const fs = require("fs");
const path = require("path");
const { MlpNetwork } = require("../policy");

const FIXTURES_DIR = path.join(__dirname, "fixtures");

describe("Python ↔ JS parity", () => {
    let model, fixtures, net;

    beforeAll(() => {
        model = JSON.parse(
            fs.readFileSync(path.join(FIXTURES_DIR, "parity_policy.json"), "utf-8"),
        );
        fixtures = JSON.parse(
            fs.readFileSync(path.join(FIXTURES_DIR, "parity_obs_fixtures.json"), "utf-8"),
        );
        net = new MlpNetwork(model);
    });

    test("schema_version of fixtures is 1", () => {
        expect(fixtures.schema_version).toBe(1);
        expect(fixtures.obs_size).toBe(341);
    });

    test("fixtures has at least 100 entries", () => {
        expect(fixtures.fixtures.length).toBeGreaterThanOrEqual(100);
    });

    test("100% action match across all fixtures", () => {
        let mismatches = 0;
        const failureSamples = [];
        for (let i = 0; i < fixtures.fixtures.length; i++) {
            const f = fixtures.fixtures[i];
            const obsRaw = new Float32Array(f.obs_raw);
            const actionJs = net.activateRaw(obsRaw);
            if (actionJs !== f.expected_action) {
                mismatches++;
                if (failureSamples.length < 3) {
                    failureSamples.push({
                        idx: i,
                        py: f.expected_action,
                        js: actionJs,
                    });
                }
            }
        }
        expect({ mismatches, failureSamples }).toEqual({
            mismatches: 0,
            failureSamples: [],
        });
    });

    test("logits match within Float32 tolerance (atol=1e-4)", () => {
        // 1e-4 é folga generosa; Math.tanh do JS pode divergir levemente do
        // pyTorch em casos extremos, e fp32 round-trip via JSON contribui.
        const sample = fixtures.fixtures.slice(0, 20);
        for (const f of sample) {
            const obsRaw = new Float32Array(f.obs_raw);
            const logitsJs = net.forwardLogits(obsRaw);
            const logitsPy = f.expected_logits;
            for (let i = 0; i < logitsPy.length; i++) {
                const diff = Math.abs(logitsJs[i] - logitsPy[i]);
                expect(diff).toBeLessThan(1e-4);
            }
        }
    });
});
