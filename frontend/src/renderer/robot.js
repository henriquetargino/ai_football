/**
 * robô Cartoon Fofo — cubo arredondado com 4 rodinhas, olhos e boca
 * estéticos que combinam com o corpo, antena fica vermelha no chute.
 */

AISoccer.createRobot = function(scene, team) {
    var isOrange = (team === "red");

    // cores vivas
    var colors = {
        body:        isOrange ? 0xFF6B1A : 0x4A9BFF,
        bodyDark:    isOrange ? 0xCC5510 : 0x2A6FCC,
        wheelTire:   0x333333,
        wheelRim:    isOrange ? 0xE85A10 : 0x2A6FCC,
        wheelHub:    0x555555,
        eyeWhite:    0xFFFFFF,
        eyePupil:    0x1a1a1a,
        mouth:       isOrange ? 0xCC5510 : 0x2A6FCC,
        antennaPole: 0x888888,
        antennaBall: isOrange ? 0xFF8844 : 0x6BB3FF,
        antennaKick: isOrange ? 0xFF6B1A : 0x4A9BFF,
    };

    var group = new THREE.Group();

    // dimensões do corpo
    var SIZE = 0.36;
    var HEIGHT = 0.32;
    var bodyY = 0.16;

    // ══ CORPO — cubo arredondado ══
    var bodyMat = new THREE.MeshStandardMaterial({
        color: colors.body, roughness: 0.45, metalness: 0.05
    });

    var bodyGeo = new THREE.BoxGeometry(SIZE, HEIGHT, SIZE);
    var body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = bodyY;
    body.castShadow = true;
    group.add(body);

    // cantos arredondados (8 esferas)
    var cR = 0.04;
    var cornerGeo = new THREE.SphereGeometry(cR, 8, 8);
    var hx = SIZE / 2 - cR * 0.3;
    var hy = HEIGHT / 2 - cR * 0.3;
    var hz = SIZE / 2 - cR * 0.3;
    for (var cx = -1; cx <= 1; cx += 2) {
        for (var cy = -1; cy <= 1; cy += 2) {
            for (var cz = -1; cz <= 1; cz += 2) {
                var corner = new THREE.Mesh(cornerGeo, bodyMat);
                corner.position.set(cx * hx, bodyY + cy * hy, cz * hz);
                group.add(corner);
            }
        }
    }

    // arestas arredondadas (12 cilindros)
    var edgeR = 0.025;
    // verticais (4)
    var edgeVGeo = new THREE.CylinderGeometry(edgeR, edgeR, HEIGHT - cR, 8);
    for (var ex = -1; ex <= 1; ex += 2) {
        for (var ez = -1; ez <= 1; ez += 2) {
            var edgeV = new THREE.Mesh(edgeVGeo, bodyMat);
            edgeV.position.set(ex * hx, bodyY, ez * hz);
            group.add(edgeV);
        }
    }
    // horizontais X (4)
    var edgeHXGeo = new THREE.CylinderGeometry(edgeR, edgeR, SIZE - cR, 8);
    for (var ey = -1; ey <= 1; ey += 2) {
        for (var ez2 = -1; ez2 <= 1; ez2 += 2) {
            var edgeHX = new THREE.Mesh(edgeHXGeo, bodyMat);
            edgeHX.rotation.z = Math.PI / 2;
            edgeHX.position.set(0, bodyY + ey * hy, ez2 * hz);
            group.add(edgeHX);
        }
    }
    // horizontais Z (4)
    var edgeHZGeo = new THREE.CylinderGeometry(edgeR, edgeR, SIZE - cR, 8);
    for (var ey2 = -1; ey2 <= 1; ey2 += 2) {
        for (var ex2 = -1; ex2 <= 1; ex2 += 2) {
            var edgeHZ = new THREE.Mesh(edgeHZGeo, bodyMat);
            edgeHZ.rotation.x = Math.PI / 2;
            edgeHZ.position.set(ex2 * hx, bodyY + ey2 * hy, 0);
            group.add(edgeHZ);
        }
    }

    // ══ RODINHAS (4) ══
    var wheelR = 0.06;
    var wheelW = 0.04;
    var tireMat = new THREE.MeshStandardMaterial({ color: colors.wheelTire, roughness: 0.8 });
    var rimMat = new THREE.MeshStandardMaterial({ color: colors.wheelRim, roughness: 0.3, metalness: 0.3 });
    var hubMat = new THREE.MeshStandardMaterial({ color: colors.wheelHub, metalness: 0.5 });

    var wheelPositions = [
        { x: SIZE * 0.35, z: SIZE / 2 + wheelW / 2 + 0.005 },
        { x: SIZE * 0.35, z: -(SIZE / 2 + wheelW / 2 + 0.005) },
        { x: -SIZE * 0.35, z: SIZE / 2 + wheelW / 2 + 0.005 },
        { x: -SIZE * 0.35, z: -(SIZE / 2 + wheelW / 2 + 0.005) },
    ];

    for (var wi = 0; wi < wheelPositions.length; wi++) {
        var wp = wheelPositions[wi];
        var wheelGroup = new THREE.Group();

        var tireGeo = new THREE.CylinderGeometry(wheelR, wheelR, wheelW, 16);
        var tire = new THREE.Mesh(tireGeo, tireMat);
        tire.rotation.x = Math.PI / 2;
        wheelGroup.add(tire);

        var rimGeo = new THREE.CylinderGeometry(wheelR * 0.72, wheelR * 0.72, wheelW + 0.005, 16);
        var rim = new THREE.Mesh(rimGeo, rimMat);
        rim.rotation.x = Math.PI / 2;
        wheelGroup.add(rim);

        var hubGeo = new THREE.CylinderGeometry(wheelR * 0.2, wheelR * 0.2, wheelW + 0.01, 8);
        var hub = new THREE.Mesh(hubGeo, hubMat);
        hub.rotation.x = Math.PI / 2;
        wheelGroup.add(hub);

        wheelGroup.position.set(wp.x, wheelR * 0.7, wp.z);
        group.add(wheelGroup);
    }

    // ══ OLHOS — googly grandes 3D (estilo AI Warehouse) ══
    // esferas brancas grandes protuberantes com pupilas pretas
    var eyeSpacing = SIZE * 0.19;
    var eyeY = bodyY + HEIGHT * 0.1;
    var eyeX = SIZE / 2;
    var eyeRadius = 0.06;  // BEM grande em relação ao corpo

    var eyeWhiteMat = new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.15, metalness: 0.0
    });
    var pupilMat = new THREE.MeshStandardMaterial({
        color: 0x111111, roughness: 0.3
    });

    for (var ei = -1; ei <= 1; ei += 2) {
        // globo ocular branco (esfera grande, protuberante)
        var eyeGeo = new THREE.SphereGeometry(eyeRadius, 16, 16);
        var eye = new THREE.Mesh(eyeGeo, eyeWhiteMat);
        eye.position.set(eyeX + eyeRadius * 0.5, eyeY, ei * eyeSpacing);
        group.add(eye);

        // pupila preta (esfera menor, colada na superfície frontal)
        var pupilR = eyeRadius * 0.5;
        var pupilGeo = new THREE.SphereGeometry(pupilR, 12, 12);
        var pupil = new THREE.Mesh(pupilGeo, pupilMat);
        // posicionar na superfície do globo, levemente para baixo e para dentro
        pupil.position.set(
            eyeX + eyeRadius * 0.5 + eyeRadius * 0.7,
            eyeY - eyeRadius * 0.1,
            ei * eyeSpacing + ei * eyeRadius * 0.15
        );
        group.add(pupil);

        // brilho (pontinho branco para dar vida)
        var glintGeo = new THREE.SphereGeometry(pupilR * 0.35, 6, 6);
        var glintMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        var glint = new THREE.Mesh(glintGeo, glintMat);
        glint.position.set(
            eyeX + eyeRadius * 0.5 + eyeRadius * 0.85,
            eyeY + eyeRadius * 0.15,
            ei * eyeSpacing + ei * eyeRadius * -0.1
        );
        group.add(glint);
    }

    // ══ BOCA — sorriso simples 3D (torus cortado) ══
    // pequena e fofa, só um detalhe sutil
    var mouthY = bodyY - HEIGHT * 0.22;
    var mouthMat = new THREE.MeshStandardMaterial({
        color: 0x222222, roughness: 0.5
    });

    // sorriso feito de esferinhas em arco
    var smileW = SIZE * 0.18;
    var smileSegs = 6;
    for (var si = 0; si <= smileSegs; si++) {
        var st = si / smileSegs;  // 0 a 1
        var angle = Math.PI * 0.2 + st * Math.PI * 0.6;  // arco de sorriso
        var sz = -smileW / 2 + st * smileW;
        var sy = mouthY - Math.sin(angle) * 0.012;
        var dotGeo = new THREE.SphereGeometry(0.008, 6, 6);
        var dot = new THREE.Mesh(dotGeo, mouthMat);
        dot.position.set(eyeX + 0.008, sy, sz);
        group.add(dot);
    }

    // ══ ANTENA — feedback visual de chute ══
    var antennaMat = new THREE.MeshStandardMaterial({
        color: colors.antennaPole, metalness: 0.4, roughness: 0.4
    });
    var antennaGeo = new THREE.CylinderGeometry(0.014, 0.02, 0.15, 8);
    var antenna = new THREE.Mesh(antennaGeo, antennaMat);
    antenna.position.set(0, bodyY + HEIGHT / 2 + 0.075, 0);
    group.add(antenna);

    // bolinha da antena — neon forte da cor do time no chute
    var bulbMatIdle = new THREE.MeshStandardMaterial({
        color: colors.antennaBall, emissive: colors.antennaBall, emissiveIntensity: 0.3
    });
    var bulbMatKick = new THREE.MeshBasicMaterial({
        color: 0xffffff
    });
    var bulbGeo = new THREE.SphereGeometry(0.05, 12, 12);
    var bulb = new THREE.Mesh(bulbGeo, bulbMatIdle);
    var bulbY = bodyY + HEIGHT / 2 + 0.15 + 0.04;
    bulb.position.set(0, bulbY, 0);
    group.add(bulb);

    // halo de luz (esfera maior transparente, só aparece no chute)
    var haloMatKick = new THREE.MeshBasicMaterial({
        color: colors.body, transparent: true, opacity: 0.5
    });
    var haloMatOff = new THREE.MeshBasicMaterial({
        visible: false
    });
    var haloGeo = new THREE.SphereGeometry(0.09, 12, 12);
    var halo = new THREE.Mesh(haloGeo, haloMatOff);
    halo.position.set(0, bulbY, 0);
    group.add(halo);

    // ══ BOCHECHAS (esferas sutis) ══
    var cheekMat = new THREE.MeshStandardMaterial({
        color: isOrange ? 0xFF9955 : 0x77BBFF,
        transparent: true, opacity: 0.3, roughness: 0.6
    });
    var cheekGeo = new THREE.SphereGeometry(0.03, 8, 8);
    var cheekL = new THREE.Mesh(cheekGeo, cheekMat);
    cheekL.scale.set(0.4, 0.8, 1);
    cheekL.position.set(eyeX + 0.003, bodyY - HEIGHT * 0.05, -SIZE * 0.32);
    group.add(cheekL);
    var cheekR = new THREE.Mesh(cheekGeo, cheekMat);
    cheekR.scale.set(0.4, 0.8, 1);
    cheekR.position.set(eyeX + 0.003, bodyY - HEIGHT * 0.05, SIZE * 0.32);
    group.add(cheekR);

    // ── Referências para animação ──
    group.userData = {
        bulb: bulb,
        halo: halo,
        bulbMatIdle: bulbMatIdle,
        bulbMatKick: bulbMatKick,
        haloMatKick: haloMatKick,
        haloMatOff: haloMatOff,
        bulbBaseY: bulb.position.y,
        bodyMesh: body,
        baseBodyY: bodyY,
        kickTimer: 0,
        attemptFlash: 0,
        team: team,
    };

    // escala para caber na physics radius
    group.scale.set(0.72, 0.72, 0.72);
    group.position.y = 0;
    scene.add(group);

    return group;
};

/**
 * atualiza a posição/rotação do robô.
 */
AISoccer.updateRobot = function(robot, playerState, time) {
    var S = AISoccer.SCALE;

    // posição
    robot.position.x = playerState.x * S;
    robot.position.z = playerState.y * S;

    // rotação
    robot.rotation.y = -playerState.angle;

    var ud = robot.userData;

    // bob da antena
    ud.bulb.position.y = ud.bulbBaseY + Math.sin(time * 3 + (robot.id || 0)) * 0.01;

    // leve bounce do corpo (idle)
    ud.bodyMesh.position.y = ud.baseBodyY + Math.sin(time * 2.5 + (robot.id || 0)) * 0.004;

    // animação de chute: antena com dois níveis
    if (playerState.is_kicking) {
        ud.kickTimer = 0.3;
    }
    var attempted = playerState.kick_attempted || false;
    if (attempted && ud.kickTimer <= 0) {
        ud.attemptFlash = 0.12;
    }

    if (ud.kickTimer > 0) {
        ud.bulb.material = ud.bulbMatKick;
        ud.halo.material = ud.haloMatKick;
        var pulse = 1.0 + Math.sin(time * 25) * 0.3;
        ud.halo.scale.set(pulse, pulse, pulse);
        ud.kickTimer -= 1 / 60;
        ud.attemptFlash = 0;
    } else if (ud.attemptFlash > 0) {
        ud.bulb.material = ud.bulbMatKick;
        ud.halo.material = ud.haloMatOff;
        ud.halo.scale.set(1, 1, 1);
        ud.attemptFlash -= 1 / 60;
    } else {
        ud.bulb.material = ud.bulbMatIdle;
        ud.halo.material = ud.haloMatOff;
        ud.halo.scale.set(1, 1, 1);
    }
};
