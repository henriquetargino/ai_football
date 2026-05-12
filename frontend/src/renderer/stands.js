/**
 * arquibancadas com assentos coloridos e espectadores (esferas).
 * 4 arquibancadas (norte, sul, leste, oeste) posicionadas atrás das muretas.
 * CRÍTICO: rotation.y → norte=PI, sul=0, oeste=PI/2, leste=-PI/2
 * fileiras crescem no eixo Z local positivo (PARA FORA do campo).
 */

AISoccer.createStands = function(scene) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;

    var concreteMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.9 });
    var numRows = 6;
    var rowDepth = 0.25;
    var rowHeight = 0.12;
    var seatSize = { w: 0.08, h: 0.07, d: 0.06 };
    var offset = 0.5;  // distância da borda do campo

    function createStand(length, rotY, posX, posZ) {
        var group = new THREE.Group();

        var seatsPerRow = Math.floor(length / (seatSize.w + 0.03));

        for (var row = 0; row < numRows; row++) {
            // degrau de concreto
            var stepGeo = new THREE.BoxGeometry(length, rowHeight, rowDepth);
            var step = new THREE.Mesh(stepGeo, concreteMat);
            step.position.set(0, row * rowHeight, row * rowDepth);
            step.receiveShadow = true;
            group.add(step);

            // assentos
            var startX = -length / 2 + seatSize.w / 2 + 0.02;
            for (var s = 0; s < seatsPerRow; s++) {
                var hue = 0.52 + Math.random() * 0.08;
                var seatMat = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(hue, 0.35, 0.2)
                });
                var seatGeo = new THREE.BoxGeometry(seatSize.w, seatSize.h, seatSize.d);
                var seat = new THREE.Mesh(seatGeo, seatMat);
                seat.position.set(
                    startX + s * (seatSize.w + 0.03),
                    row * rowHeight + rowHeight / 2 + seatSize.h / 2,
                    row * rowDepth
                );
                group.add(seat);

                // espectador (~45% de ocupação — reduzido de 70% pra perf)
                if (Math.random() < 0.45) {
                    var headHue = Math.random();
                    var headMat = new THREE.MeshStandardMaterial({
                        color: new THREE.Color().setHSL(headHue, 0.4, 0.5)
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

    // sul (z = FH + offset) — rotation.y = 0, fileiras crescem em +Z (para fora)
    createStand(FW, 0, FW / 2, FH + offset);

    // norte (z = -offset) — rotation.y = PI, fileiras crescem em -Z (para fora)
    createStand(FW, Math.PI, FW / 2, -offset);

    // oeste (x = -offset) — rotation.y = -PI/2, fileiras crescem em -X (para fora)
    createStand(FH, -Math.PI / 2, -offset, FH / 2);

    // leste (x = FW + offset) — rotation.y = PI/2, fileiras crescem em +X (para fora)
    createStand(FH, Math.PI / 2, FW + offset, FH / 2);
};
