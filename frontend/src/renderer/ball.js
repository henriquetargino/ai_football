/**
 * bola de futebol clássica — branca com pentágonos pretos.
 * usa projeção 3D real (Voronoi esférico) para textura perfeita.
 * a sombra é separada (não filha da bola) para não rotacionar junto.
 */

AISoccer.createBall = function(scene) {
    var S = AISoccer.SCALE;
    var ballRadius = 12 * S;  // 0.12

    // ── Textura procedural de bola de futebol ──
    var texture = AISoccer._createSoccerTexture();

    var geo = new THREE.SphereGeometry(ballRadius, 32, 32);
    var mat = new THREE.MeshStandardMaterial({
        map: texture,
        roughness: 0.35,
        metalness: 0.05,
    });
    var ball = new THREE.Mesh(geo, mat);
    ball.castShadow = true;
    ball.position.set(AISoccer.FW / 2, ballRadius, AISoccer.FH / 2);
    scene.add(ball);

    // ── Sombra no chão (mesh separado, não filho da bola) ──
    var shadowGeo = new THREE.CircleGeometry(ballRadius * 1.3, 16);
    var shadowMat = new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.25,
    });
    var shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(AISoccer.FW / 2, 0.004, AISoccer.FH / 2);
    scene.add(shadow);

    // guardar referência da sombra para atualizar posição no main loop
    ball.userData.shadow = shadow;

    return ball;
};

/**
 * cria uma CanvasTexture com padrão de bola de futebol usando projeção 3D real.
 * usa Voronoi esférico sobre os 32 centros de face do icosaedro truncado
 * (12 pentágonos + 20 hexágonos) para distribuição perfeita.
 */
AISoccer._createSoccerTexture = function() {
    var size = 512;
    var canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext('2d');

    // razão áurea
    var phi = (1 + Math.sqrt(5)) / 2;

    // 12 vértices do icosaedro = centros dos pentágonos
    var rawVerts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ];

    // 20 faces do icosaedro (índices dos vértices)
    var faces = [
        [0,1,8], [0,8,4], [0,4,5], [0,5,9], [0,9,1],
        [1,6,8], [8,6,10], [8,10,4], [4,10,2], [4,2,5],
        [5,2,11], [5,11,9], [9,11,7], [9,7,1], [1,7,6],
        [3,6,7], [3,7,11], [3,11,2], [3,2,10], [3,10,6]
    ];

    // todos os 32 centros normalizados na esfera unitária
    // primeiros 12 = pentágonos, próximos 20 = hexágonos
    var centers = [];
    var i, v, len, f, a, b, c, mx, my, mz;

    for (i = 0; i < rawVerts.length; i++) {
        v = rawVerts[i];
        len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        centers.push([v[0]/len, v[1]/len, v[2]/len]);
    }
    for (f = 0; f < faces.length; f++) {
        a = rawVerts[faces[f][0]];
        b = rawVerts[faces[f][1]];
        c = rawVerts[faces[f][2]];
        mx = (a[0]+b[0]+c[0]) / 3;
        my = (a[1]+b[1]+c[1]) / 3;
        mz = (a[2]+b[2]+c[2]) / 3;
        len = Math.sqrt(mx*mx + my*my + mz*mz);
        centers.push([mx/len, my/len, mz/len]);
    }

    // processar pixel a pixel
    var imageData = ctx.createImageData(size, size);
    var data = imageData.data;
    var seamThresh = 0.018;  // costuras finas como na bola real
    var pentShrink = 0.012;  // borda extra dos pentágonos vira costura (ficam menores)

    var py, px, theta, sinT, cosT, phiA, x3, y3, z3;
    var best, second, bestIdx, d, ci, idx;
    var isPentagon, isSeam, isPentBorder;

    for (py = 0; py < size; py++) {
        theta = (py + 0.5) / size * Math.PI;
        sinT = Math.sin(theta);
        cosT = Math.cos(theta);

        for (px = 0; px < size; px++) {
            phiA = (px + 0.5) / size * 2 * Math.PI;

            x3 = sinT * Math.cos(phiA);
            y3 = cosT;
            z3 = sinT * Math.sin(phiA);

            best = -2;
            second = -2;
            bestIdx = 0;

            for (ci = 0; ci < 32; ci++) {
                d = x3 * centers[ci][0] + y3 * centers[ci][1] + z3 * centers[ci][2];
                if (d > best) {
                    second = best;
                    bestIdx = ci;
                    best = d;
                } else if (d > second) {
                    second = d;
                }
            }

            idx = (py * size + px) * 4;
            isPentagon = (bestIdx < 12);
            isSeam = (best - second) < seamThresh;
            isPentBorder = isPentagon && !isSeam && (best - second) < (seamThresh + pentShrink);

            if (isSeam || isPentBorder) {
                // costura — cinza escuro
                data[idx]     = 55;
                data[idx + 1] = 55;
                data[idx + 2] = 55;
            } else if (isPentagon) {
                // pentágono — preto
                data[idx]     = 22;
                data[idx + 1] = 22;
                data[idx + 2] = 22;
            } else {
                // hexágono — branco/creme
                data[idx]     = 242;
                data[idx + 1] = 240;
                data[idx + 2] = 234;
            }
            data[idx + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);

    var tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
};
