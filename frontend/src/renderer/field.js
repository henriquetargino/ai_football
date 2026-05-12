/**
 * gramado listrado (listras VERTICAIS no eixo X), linhas do campo, chão estendido.
 */

AISoccer.createField = function(scene) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;
    // v9.5 — raio do canto arredondado (paridade com walls.js e config.py).
    var CUT = 30 * AISoccer.SCALE;     // = 0.30 em 3D

    // ── Chão estendido ──
    var floorGeo = new THREE.PlaneGeometry(50, 50);
    var floorMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a });
    var floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(FW / 2, -0.01, FH / 2);
    floor.receiveShadow = true;
    scene.add(floor);

    // ── Bloco de borda escuro ──
    var borderGeo = new THREE.BoxGeometry(FW + 0.4, 0.05, FH + 0.4);
    var borderMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a });
    var border = new THREE.Mesh(borderGeo, borderMat);
    border.position.set(FW / 2, -0.025, FH / 2);
    border.receiveShadow = true;
    scene.add(border);

    // ── Gramado: TODAS as faixas como overlays para evitar artefatos de sombra ──
    // ambas as cores ficam em y=0.003, cobrindo completamente a superfície.
    var numStripes = 14;
    var stripeWidth = FW / numStripes;
    var darkGreen = new THREE.MeshStandardMaterial({ color: 0x2E7D32 });
    var lightGreen = new THREE.MeshStandardMaterial({ color: 0x388E3C });

    for (var i = 0; i < numStripes; i++) {
        var mat = (i % 2 === 0) ? lightGreen : darkGreen;
        var stripeGeo = new THREE.PlaneGeometry(stripeWidth + 0.002, FH + 0.002);
        var stripe = new THREE.Mesh(stripeGeo, mat);
        stripe.rotation.x = -Math.PI / 2;
        stripe.position.set((i + 0.5) * stripeWidth, 0.003, FH / 2);
        stripe.receiveShadow = true;
        scene.add(stripe);
    }

    // ── V9.5: máscaras nos 4 cantos pra "cortar" o gramado conforme o arco ──
    // cada máscara cobre a área {dentro do quadrado CUT × CUT do canto E
    // FORA do quadrante de círculo}. Cor = mesma da border (parece "fora
    // do campo"). Resultado: gramado segue a forma curva do escanteio.
    var maskMat = new THREE.MeshStandardMaterial({
        color: 0x2a2a2a, side: THREE.DoubleSide
    });

    function addCornerMask(corner) {
        var shape = new THREE.Shape();
        if (corner === 'NW') {
            shape.moveTo(0, 0);
            shape.lineTo(CUT, 0);
            shape.absarc(CUT, CUT, CUT, 3 * Math.PI / 2, Math.PI, true);
            shape.lineTo(0, 0);
        } else if (corner === 'NE') {
            shape.moveTo(FW, 0);
            shape.lineTo(FW, CUT);
            shape.absarc(FW - CUT, CUT, CUT, 2 * Math.PI, 3 * Math.PI / 2, true);
            shape.lineTo(FW, 0);
        } else if (corner === 'SE') {
            shape.moveTo(FW, FH);
            shape.lineTo(FW - CUT, FH);
            shape.absarc(FW - CUT, FH - CUT, CUT, Math.PI / 2, 0, true);
            shape.lineTo(FW, FH);
        } else if (corner === 'SW') {
            shape.moveTo(0, FH);
            shape.lineTo(0, FH - CUT);
            shape.absarc(CUT, FH - CUT, CUT, Math.PI, Math.PI / 2, true);
            shape.lineTo(0, FH);
        }
        var geo = new THREE.ShapeGeometry(shape);
        geo.rotateX(Math.PI / 2);
        var mesh = new THREE.Mesh(geo, maskMat);
        // y entre as listras (0.003) e as linhas brancas (0.006).
        mesh.position.y = 0.0045;
        scene.add(mesh);
    }
    addCornerMask('NW');
    addCornerMask('NE');
    addCornerMask('SE');
    addCornerMask('SW');

    // ── Linhas do campo ──
    var lineMat = new THREE.MeshStandardMaterial({
        color: 0xffffff, transparent: true, opacity: 0.9,
        side: THREE.DoubleSide
    });
    var lineY = 0.006;
    var lineW = 0.025;

    function addLineH(x, z, length) {
        var geo = new THREE.PlaneGeometry(length, lineW);
        var mesh = new THREE.Mesh(geo, lineMat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(x, lineY, z);
        scene.add(mesh);
    }

    function addLineV(x, z, length) {
        var geo = new THREE.PlaneGeometry(lineW, length);
        var mesh = new THREE.Mesh(geo, lineMat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(x, lineY, z);
        scene.add(mesh);
    }

    // bordas do campo — V9.5: encurtadas pra pararem nos arcos dos cantos.
    // linha topo/bottom: vai de x=CUT a x=FW-CUT (em vez de 0 a FW).
    // linha esq/dir: vai de y=CUT a y=FH-CUT.
    // os 4 arcos brancos completam o contorno (adicionados abaixo).
    addLineH(FW / 2, 0, FW - 2 * CUT);
    addLineH(FW / 2, FH, FW - 2 * CUT);
    addLineV(0, FH / 2, FH - 2 * CUT);
    addLineV(FW, FH / 2, FH - 2 * CUT);

    // v9.5 — 4 arcos brancos nos cantos completando o contorno do campo,
    // seguindo a mesma curva dos arcos das paredes/máscaras (raio=CUT).
    function addArcLine(cx, cz, thetaStart, thetaLength) {
        var arcGeo = new THREE.RingGeometry(
            CUT - lineW / 2, CUT + lineW / 2,
            16, 1,
            thetaStart, thetaLength
        );
        arcGeo.rotateX(Math.PI / 2);
        var arc = new THREE.Mesh(arcGeo, lineMat);
        arc.position.set(cx, lineY, cz);
        scene.add(arc);
    }
    addArcLine(CUT,        CUT,        Math.PI,           Math.PI / 2);  // NW
    addArcLine(FW - CUT,   CUT,        3 * Math.PI / 2,   Math.PI / 2);  // NE
    addArcLine(FW - CUT,   FH - CUT,   0,                 Math.PI / 2);  // SE
    addArcLine(CUT,        FH - CUT,   Math.PI / 2,       Math.PI / 2);  // SW

    // linha central
    addLineV(FW / 2, FH / 2, FH);

    // círculo central — RingGeometry para linha contínua perfeita
    var circleRadius = 0.6;
    var ringGeo = new THREE.RingGeometry(
        circleRadius - lineW / 2,
        circleRadius + lineW / 2,
        64
    );
    var ring = new THREE.Mesh(ringGeo, lineMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(FW / 2, lineY, FH / 2);
    scene.add(ring);

    // ponto central
    var dotGeo = new THREE.CircleGeometry(0.05, 16);
    var dot = new THREE.Mesh(dotGeo, lineMat);
    dot.rotation.x = -Math.PI / 2;
    dot.position.set(FW / 2, lineY, FH / 2);
    scene.add(dot);

    // ── Áreas de penalidade ──
    var penW = 1.2;
    var penH = 2.2;

    // Área esquerda
    addLineV(penW, FH / 2, penH);
    addLineH(penW / 2, FH / 2 - penH / 2, penW);
    addLineH(penW / 2, FH / 2 + penH / 2, penW);

    // Área direita
    addLineV(FW - penW, FH / 2, penH);
    addLineH(FW - penW / 2, FH / 2 - penH / 2, penW);
    addLineH(FW - penW / 2, FH / 2 + penH / 2, penW);
};
