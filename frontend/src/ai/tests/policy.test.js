const fs = require("fs");
const path = require("path");
const { MlpNetwork } = require("../policy");

const FIXTURES_DIR = path.join(__dirname, "fixtures");

function loadModel() {
    return JSON.parse(
        fs.readFileSync(path.join(FIXTURES_DIR, "parity_policy.json"), "utf-8"),
    );
}

describe("MlpNetwork", () => {
    test("constructor validates schema_version", () => {
        const bad = { schema_version: 999, weights: {}, obs_rms: null, metadata: {} };
        expect(() => new MlpNetwork(bad)).toThrow(/schema_version/);
    });

    test("constructor throws on null modelData", () => {
        expect(() => new MlpNetwork(null)).toThrow();
    });

    test("constructor validates required weight keys present", () => {
        const bad = {
            schema_version: 1,
            weights: { "fc1.weight": [[1]] },
            obs_rms: null,
            metadata: {},
        };
        expect(() => new MlpNetwork(bad)).toThrow(/ausente/);
    });

    test("loads valid model and exposes metadata", () => {
        const model = loadModel();
        const net = new MlpNetwork(model);
        expect(net.metadata.obs_size).toBe(341);
        expect(net.metadata.action_space_size).toBe(18);
        expect(net.metadata.hidden_size).toBe(64);
    });

    test("forwardLogits returns Float32Array of length 18", () => {
        const net = new MlpNetwork(loadModel());
        const obs = new Float32Array(341);
        const logits = net.forwardLogits(obs);
        expect(logits).toBeInstanceOf(Float32Array);
        expect(logits.length).toBe(18);
    });

    test("activateRaw returns int in [0, 17]", () => {
        const net = new MlpNetwork(loadModel());
        const obs = new Float32Array(341);
        for (let i = 0; i < 100; i++) {
            for (let j = 0; j < 341; j++) obs[j] = Math.random() * 4 - 2;
            const action = net.activateRaw(obs);
            expect(Number.isInteger(action)).toBe(true);
            expect(action).toBeGreaterThanOrEqual(0);
            expect(action).toBeLessThanOrEqual(17);
        }
    });

    test("deterministic — same obs gives same action 10×", () => {
        const net = new MlpNetwork(loadModel());
        const obs = new Float32Array(341);
        for (let j = 0; j < 341; j++) obs[j] = (j * 0.013) % 1.0;
        const first = net.activateRaw(obs);
        for (let i = 0; i < 9; i++) {
            expect(net.activateRaw(obs)).toBe(first);
        }
    });

    test("loaded model has obs_rms with mean+var of length 341", () => {
        const model = loadModel();
        expect(model.obs_rms).not.toBeNull();
        expect(model.obs_rms.mean.length).toBe(341);
        expect(model.obs_rms.var.length).toBe(341);
    });
});
