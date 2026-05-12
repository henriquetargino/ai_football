/**
 * banner — cena Three.js com a representação IDÊNTICA ao cenário do jogo
 * (escuro, com floodlights, arquibancadas com público), pra a transição
 * index.html → modo de jogo ser zero (mesmo visual).
 *
 * diferenças mínimas vs scene.js do jogo:
 *   - Câmera mais distante e elevada (vista 3/4 do estádio inteiro).
 *   - Fog menor (precisamos enxergar o estádio até as arquibancadas).
 *   - Adiciona arquibancadas CURVAS nos cantos (4 torres) pra fechar
 *     o estádio em formato oval, conectando as 4 arquibancadas N/S/L/O.
 *
 * estática: o renderer renderiza a mesma cena sem animar nada.
 */

AISoccer.createBannerScene = function (canvasEl) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;

    // ─── Scene + background dark idênticos ao jogo ───
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);
    scene.fog = new THREE.Fog(0x0a0a14, 22, 50);

    var width = canvasEl.clientWidth || 1100;
    var height = canvasEl.clientHeight || 360;

    // câmera levemente mais alta que a do jogo, pra mostrar o estádio inteiro.
    // FOV menor = mais zoom (vista mais próxima do estádio).
    var camera = new THREE.PerspectiveCamera(32, width / height, 0.1, 100);
    camera.position.set(FW / 2, 5.5, FH + 6.0);
    camera.lookAt(FW / 2, 0.4, FH / 2);

    var renderer = new THREE.WebGLRenderer({
        canvas: canvasEl, antialias: true, alpha: false,
    });
    renderer.setSize(width, height, false);
    // pixel ratio cap em 1.25 (em vez de 2). Em retina 3x reduz a carga
    // da GPU em ~5x. Banner é estático — quality minor é imperceptível.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    // shadow map DESLIGADO no banner — é cena ESTÁTICA, sem objetos em
    // movimento, sem necessidade de atualização de sombras a cada frame.
    renderer.shadowMap.enabled = false;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // ─── Iluminação idêntica ao scene.js do jogo ───
    var ambient = new THREE.AmbientLight(0x1a1a2a, 0.5);
    scene.add(ambient);
    var dirLight = new THREE.DirectionalLight(0x4466aa, 0.25);
    dirLight.position.set(FW / 2, 10, FH / 2);
    scene.add(dirLight);

    // ─── Cenário do jogo COMPLETO ───
    AISoccer.createField(scene);
    if (AISoccer.createWalls) AISoccer.createWalls(scene, 220);
    AISoccer.createGoals(scene, 220);
    // banner usa stands MAIS ALTAS (10 fileiras vs 6 do jogo) pra ficarem
    // visualmente acima das traves. NÃO chamamos createStands aqui — usamos
    // versão custom abaixo.
    AISoccer._buildBannerSideStands(scene);
    if (AISoccer.createFloodlights) AISoccer.createFloodlights(scene);

    // 4 arquibancadas curvas nos cantos com a mesma altura das laterais
    // do banner (10 fileiras), pra estádio fechado/coerente.
    AISoccer._buildCornerStands(scene, { numRows: 10 });

    // ─── 2 robôs e bola posicionados perto do círculo central ───
    var robotRed = AISoccer.createRobot(scene, "red");
    var robotBlue = AISoccer.createRobot(scene, "blue");
    var ball = AISoccer.createBall(scene);

    var ballX = 400, ballY = 280;
    var redX = 280, redY = 320;
    var blueX = 520, blueY = 320;

    var redState = {
        id: "red_0", x: redX, y: redY,
        angle: Math.atan2(ballY - redY, ballX - redX),
        is_kicking: false, kick_attempted: false,
    };
    var blueState = {
        id: "blue_0", x: blueX, y: blueY,
        angle: Math.atan2(ballY - blueY, ballX - blueX),
        is_kicking: false, kick_attempted: false,
    };
    AISoccer.updateRobot(robotRed, redState, 0);
    AISoccer.updateRobot(robotBlue, blueState, 0);

    var S = AISoccer.SCALE;
    ball.position.set(ballX * S, 12 * S, ballY * S);
    if (ball.userData.shadow) {
        ball.userData.shadow.position.set(ballX * S, 0.004, ballY * S);
    }

    function update() {
        renderer.render(scene, camera);
    }

    function onResize() {
        var w = canvasEl.clientWidth;
        var h = canvasEl.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
        update();   // re-renderiza após resize (sem isso, fica em branco)
    }
    window.addEventListener('resize', onResize);

    // PERFORMANCE: render-on-demand. Renderiza UMA vez aqui, depois nunca
    // mais (a menos que onResize seja chamado). Antes era chamado em loop
    // 60fps via main.js animate() → consumia ~95% da CPU à toa.
    update();

    return {
        scene: scene, camera: camera, renderer: renderer,
        update: update, onResize: onResize,
    };
};


/**
 * versão ALTA (10 fileiras) das 4 arquibancadas N/S/L/O — usada APENAS
 * no banner do index.html, pra elas ficarem visualmente acima das traves.
 * espelha a lógica de createStands (stands.js) com numRows aumentado.
 *
 * nÃO afeta a visualização do treino (que continua usando createStands
 * com 6 fileiras).
 */
AISoccer._buildBannerSideStands = function (scene) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;

    var concreteMat = new THREE.MeshStandardMaterial({
        color: 0x444444, roughness: 0.9,
    });
    var numRows = 10;        // ← alterado vs stands.js (era 6)
    var rowDepth = 0.25;
    var rowHeight = 0.12;
    var seatSize = { w: 0.08, h: 0.07, d: 0.06 };
    var offset = 0.5;

    function createStand(length, rotY, posX, posZ) {
        var group = new THREE.Group();
        var seatsPerRow = Math.floor(length / (seatSize.w + 0.03));

        for (var row = 0; row < numRows; row++) {
            var stepGeo = new THREE.BoxGeometry(length, rowHeight, rowDepth);
            var step = new THREE.Mesh(stepGeo, concreteMat);
            step.position.set(0, row * rowHeight, row * rowDepth);
            step.receiveShadow = true;
            group.add(step);

            var startX = -length / 2 + seatSize.w / 2 + 0.02;
            for (var s = 0; s < seatsPerRow; s++) {
                var hue = 0.52 + Math.random() * 0.08;
                var seatMat = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(hue, 0.35, 0.2),
                });
                var seatGeo = new THREE.BoxGeometry(seatSize.w, seatSize.h, seatSize.d);
                var seat = new THREE.Mesh(seatGeo, seatMat);
                seat.position.set(
                    startX + s * (seatSize.w + 0.03),
                    row * rowHeight + rowHeight / 2 + seatSize.h / 2,
                    row * rowDepth
                );
                group.add(seat);

                if (Math.random() < 0.7) {
                    var headHue = Math.random();
                    var headMat = new THREE.MeshStandardMaterial({
                        color: new THREE.Color().setHSL(headHue, 0.4, 0.5),
                    });
                    var headGeo = new THREE.SphereGeometry(0.035, 6, 6);
                    var head = new THREE.Mesh(headGeo, headMat);
                    head.position.set(
                        seat.position.x,
                        seat.position.y + seatSize.h / 2 + 0.06,
                        seat.position.z
                    );
                    group.add(head);
                }
            }
        }

        group.rotation.y = rotY;
        group.position.set(posX, 0, posZ);
        scene.add(group);
    }

    // mesmas 4 posições do stands.js (S/N/O/L)
    createStand(FW, 0, FW / 2, FH + offset);
    createStand(FW, Math.PI, FW / 2, -offset);
    createStand(FH, -Math.PI / 2, -offset, FH / 2);
    createStand(FH, Math.PI / 2, FW + offset, FH / 2);
};


/**
 * 4 arquibancadas curvas conectando as 4 arquibancadas N/S/L/O do jogo,
 * criando um estádio fechado em formato oval/octagonal.
 *
 * cada canto é um arco de 90° formado por N segmentos pequenos seguindo
 * uma trajetória circular. Reproduz a mesma altura, número de fileiras
 * e espectadores das arquibancadas principais (`stands.js`) pra manter
 * coerência visual.
 *
 * @param {THREE.Scene} scene
 * @param {Object} [options]
 * @param {number} [options.numRows=6] número de fileiras (treino=6, banner=10)
 */
AISoccer._buildCornerStands = function (scene, options) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;
    options = options || {};

    // espelha as constantes de stands.js pra manter consistência visual.
    var concreteMat = new THREE.MeshStandardMaterial({
        color: 0x444444, roughness: 0.9,
    });
    var numRows = options.numRows || 6;
    var rowDepth = 0.25;
    var rowHeight = 0.12;
    var seatSize = { w: 0.08, h: 0.07, d: 0.06 };
    var offset = 0.5;

    // os 4 cantos. Cada um tem o "centro do arco" = canto do campo, e
    // o arco varre 90° conectando as 2 stands adjacentes.
    //   SE (FW, FH): da stand SUL até a LESTE  → ângulo varre 90°→0°
    //   NE (FW, 0):  da stand LESTE até a NORTE → 0°→-90°
    //   NW (0, 0):   da stand NORTE até a OESTE → 180°→90°
    //   SW (0, FH):  da stand OESTE até a SUL   → 90°→180°
    // em cada canto, varremos 90° em N segmentos.
    var corners = [
        { cx: FW, cz: FH, a0: Math.PI / 2, a1: 0 },
        { cx: FW, cz: 0, a0: 0, a1: -Math.PI / 2 },
        { cx: 0, cz: 0, a0: -Math.PI / 2, a1: -Math.PI },
        { cx: 0, cz: FH, a0: Math.PI, a1: Math.PI / 2 },
    ];

    // 12 segmentos suaviza a curva e elimina gaps visíveis. Cada degrau
    // tem largura calculada pelo raio EXTERNO da sua fileira (assim a
    // ponta externa fecha sem gap), com 8% de overlap pra cobrir
    // imperfeições.
    var nSegments = 12;
    var radiusBase = offset;
    var arcLen = Math.PI / 2;        // 90° por canto
    var arcStep = arcLen / nSegments;
    var arcStepHalf = arcStep / 2;
    var overlapFactor = 1.08;

    for (var c = 0; c < corners.length; c++) {
        var corner = corners[c];

        for (var s = 0; s < nSegments; s++) {
            var t0 = s / nSegments;
            var t1 = (s + 1) / nSegments;
            var aMid = corner.a0 + (corner.a1 - corner.a0) * (t0 + t1) / 2;

            var seg = new THREE.Group();
            for (var row = 0; row < numRows; row++) {
                var r = radiusBase + row * rowDepth;
                var rOuter = r + rowDepth / 2;

                // chord da PONTA EXTERNA da fileira (raio maior) — esse
                // é o ponto onde apareciam gaps. Multiplicar pelo overlap
                // garante boxes que se sobrepõem ligeiramente nas bordas.
                var chordLen = 2 * Math.sin(arcStepHalf) * rOuter * overlapFactor;
                chordLen = Math.max(chordLen, 0.10);

                var px = corner.cx + Math.cos(aMid) * r;
                var pz = corner.cz + Math.sin(aMid) * r;

                // degrau de concreto (alongado tangencialmente).
                // profundidade ligeiramente maior que rowDepth pra fechar
                // junções entre fileiras radiais.
                var stepGeo = new THREE.BoxGeometry(
                    chordLen, rowHeight, rowDepth * 1.02
                );
                var step = new THREE.Mesh(stepGeo, concreteMat);
                step.position.set(px, row * rowHeight, pz);
                step.rotation.y = -aMid + Math.PI / 2;
                step.receiveShadow = true;
                seg.add(step);

                // assentos em cima do degrau (2 por segmento — economiza
                // meshes; ainda dá densidade suficiente com 12 segmentos).
                var seatsPerSeg = 2;
                var seatSpacing = chordLen / (seatsPerSeg + 0.5);
                for (var k = 0; k < seatsPerSeg; k++) {
                    var localOffset = (k - (seatsPerSeg - 1) / 2) * seatSpacing;
                    var tangentX = -Math.sin(aMid) * localOffset;
                    var tangentZ = Math.cos(aMid) * localOffset;

                    var hue = 0.52 + Math.random() * 0.08;
                    var seatMat = new THREE.MeshStandardMaterial({
                        color: new THREE.Color().setHSL(hue, 0.35, 0.2),
                    });
                    var seatGeo = new THREE.BoxGeometry(seatSize.w, seatSize.h, seatSize.d);
                    var seat = new THREE.Mesh(seatGeo, seatMat);
                    seat.position.set(
                        px + tangentX,
                        row * rowHeight + rowHeight / 2 + seatSize.h / 2,
                        pz + tangentZ
                    );
                    seat.rotation.y = -aMid + Math.PI / 2;
                    seg.add(seat);

                    if (Math.random() < 0.45) {
                        var headHue = Math.random();
                        var headMat = new THREE.MeshStandardMaterial({
                            color: new THREE.Color().setHSL(headHue, 0.4, 0.5),
                        });
                        var headGeo = new THREE.SphereGeometry(0.035, 6, 6);
                        var head = new THREE.Mesh(headGeo, headMat);
                        head.position.set(
                            seat.position.x,
                            seat.position.y + seatSize.h / 2 + 0.06,
                            seat.position.z
                        );
                        seg.add(head);
                    }
                }
            }
            scene.add(seg);
        }
    }
};
