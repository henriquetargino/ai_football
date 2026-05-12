/**
 * smoke test pra API surface dos renderer modules adicionados em fase 1/3.
 *
 * os arquivos `goal_particles.js`, `robot_shatter.js`, `scoreboard.js`,
 * `neural_viz.js` são IIFE com globals (THREE, document) — incompatível
 * com `require()` direto em jest+node. Usamos `vm` pra rodar o código num
 * contexto isolado com mocks mínimos. Verificamos APENAS exposição de API
 * (não comportamento — esse é validado em runtime no browser).
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function makeContext() {
    // mock THREE com o subset usado pelos renderer files. Cada classe
    // retorna objeto com a interface mínima esperada (no-op stubs).
    const noop = () => {};
    const fakeMat = {
        color: { setHex: noop },
        opacity: 1, transparent: false,
        dispose: noop,
    };
    const fakeMesh = function () {
        this.position = { x: 0, y: 0, z: 0, set: noop };
        this.rotation = { x: 0, y: 0, z: 0, set: noop };
        this.scale    = { x: 1, y: 1, z: 1, set: noop };
        this.castShadow = false; this.receiveShadow = false;
        this.visible = true; this.frustumCulled = true;
        this.userData = {};
        this.parent = null;
        this.material = fakeMat;
        this.geometry = { dispose: noop };
        this.add = noop; this.remove = noop;
        this.traverse = (fn) => fn(this);
    };
    const fakeGeo = function () {
        this.setAttribute = noop;
        this.attributes = {
            position: { needsUpdate: false, array: new Float32Array(3) },
            aColor:   { needsUpdate: false },
            aSize:    { needsUpdate: false },
            aOpacity: { needsUpdate: false },
        };
        this.dispose = noop;
    };
    const THREE = {
        BufferGeometry: fakeGeo,
        BufferAttribute: function (arr, size) { this.array = arr; this.count = arr.length / size; },
        BoxGeometry: fakeGeo, CylinderGeometry: fakeGeo, PlaneGeometry: fakeGeo,
        SphereGeometry: fakeGeo, CircleGeometry: fakeGeo,
        MeshStandardMaterial: function () { Object.assign(this, fakeMat); },
        MeshBasicMaterial:    function () { Object.assign(this, fakeMat); },
        ShaderMaterial:       function () { Object.assign(this, fakeMat); },
        Points: fakeMesh, Mesh: fakeMesh,
        Group: function () { fakeMesh.call(this); this.children = []; this.add = (c) => this.children.push(c); },
        CanvasTexture: function () { this.dispose = noop; this.needsUpdate = false; },
        Color: function () { this.setHSL = noop; },
        AdditiveBlending: 1, NormalBlending: 0, LinearFilter: 1006,
    };

    // mock document — só pra scoreboard que cria <canvas>
    const fakeCanvas = {
        width: 256, height: 256,
        getContext: () => ({
            fillRect: noop, fillStyle: "", clearRect: noop,
            font: "", textAlign: "", textBaseline: "",
            fillText: noop, measureText: () => ({ width: 100 }),
            createLinearGradient: () => ({ addColorStop: noop }),
        }),
    };
    const document = {
        createElement: (tag) => tag === "canvas" ? fakeCanvas : {},
    };

    return { THREE, AISoccer: {}, document, performance: { now: () => 0 } };
}

function loadRenderer(filename) {
    const filepath = path.join(__dirname, "..", "..", "renderer", filename);
    const code = fs.readFileSync(filepath, "utf8");
    const ctx = makeContext();
    vm.createContext(ctx);
    vm.runInContext(code, ctx);
    return ctx.AISoccer;
}

describe("renderer API surface", () => {
    test("goal_particles.js expõe APIs esperadas", () => {
        const A = loadRenderer("goal_particles.js");
        expect(typeof A.createGoalCannons).toBe("function");
        expect(typeof A.createGoalParticles).toBe("function");
        expect(typeof A.celebrateGoal).toBe("function");
        expect(typeof A.updateGoalParticles).toBe("function");
        expect(typeof A.resetGoalParticles).toBe("function");
        expect(typeof A.disposeGoalParticles).toBe("function");
    });

    test("robot_shatter.js expõe APIs esperadas", () => {
        const A = loadRenderer("robot_shatter.js");
        expect(typeof A.createRobotShatter).toBe("function");
        expect(typeof A.shatterPlayer).toBe("function");
        expect(typeof A.updateShatter).toBe("function");
        expect(typeof A.resetRobotShatter).toBe("function");
        expect(typeof A.disposeRobotShatter).toBe("function");
    });

    test("neural_viz.js expõe APIs esperadas", () => {
        const A = loadRenderer("neural_viz.js");
        expect(typeof A.toggleNeuralViz).toBe("function");
        expect(typeof A.isNeuralVizVisible).toBe("function");
        expect(typeof A.updateNeuralViz).toBe("function");
        expect(typeof A.setNeuralVizTeam).toBe("function");
    });
});
