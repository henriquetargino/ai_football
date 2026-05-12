/**
 * setup da cena Three.js: câmera, luzes, fog, fundo.
 * escala: 1 unidade de jogo = 0.01 unidades de cena.
 * campo 800×500 → 8×5 no Three.js.
 */

var AISoccer = AISoccer || {};

AISoccer.SCALE = 0.01;
AISoccer.FIELD_WIDTH = 800;
AISoccer.FIELD_HEIGHT = 500;
AISoccer.FW = AISoccer.FIELD_WIDTH * AISoccer.SCALE;   // 8
AISoccer.FH = AISoccer.FIELD_HEIGHT * AISoccer.SCALE;   // 5

// largura ocupada por painéis fixos à direita (ex: View NN). scene.js
// subtrai do innerWidth no resize pra que o campo seja desenhado SOMENTE
// na área visível à esquerda, não atrás do painel.
AISoccer.activePanelWidth = 0;

AISoccer.createScene = function() {
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);
    scene.fog = new THREE.Fog(0x0a0a14, 18, 40);

    // helper: largura visível do canvas considerando painéis ativos
    // (ex: View NN ocupando à direita).
    function visibleWidth() {
        return Math.max(320, window.innerWidth - (AISoccer.activePanelWidth || 0));
    }

    // câmera
    var aspect = visibleWidth() / window.innerHeight;
    var camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    camera.position.set(AISoccer.FW / 2, 8, AISoccer.FH / 2 + 7);
    camera.lookAt(AISoccer.FW / 2, 0, AISoccer.FH / 2);

    // renderer
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(visibleWidth(), window.innerHeight);
    // PERF: pixel ratio cap em 1.25 (era 2). Em retina 3x reduz pixels
    // renderizados em ~5x. Diferença visual mínima — vale a pena trade
    // pra reduzir uso de GPU/CPU.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    var container = document.getElementById('canvas-container');
    container.appendChild(renderer.domElement);

    // ── Iluminação ──
    // ambient
    var ambient = new THREE.AmbientLight(0x1a1a2a, 0.5);
    scene.add(ambient);

    // directional fill (de cima)
    var dirLight = new THREE.DirectionalLight(0x4466aa, 0.25);
    dirLight.position.set(AISoccer.FW / 2, 10, AISoccer.FH / 2);
    scene.add(dirLight);

    // resize handler — considera painéis fixos (ex: View NN à direita).
    // quando o painel abre, AISoccer.activePanelWidth muda e o jogo é
    // re-renderizado na área visível, deixando o campo INTEIRO visível.
    window.addEventListener('resize', function() {
        var w = visibleWidth();
        var h = window.innerHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    return {
        scene: scene,
        camera: camera,
        renderer: renderer
    };
};
