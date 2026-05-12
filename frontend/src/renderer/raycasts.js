/**
 * visualização de raycasts em tempo real (toggle com tecla R).
 * usa BufferGeometry + LineSegments para um único draw call.
 * calcula raycasts via AISoccer.Physics.castRays().
 */

var AISoccer = AISoccer || {};

(function () {

    var MAX_PLAYERS = 4;
    var NUM_RAYS = 48;  // RAY_ANGLES.length
    var VERTS_PER_RAY = 2;
    var TOTAL_VERTS = MAX_PLAYERS * NUM_RAYS * VERTS_PER_RAY;

    var visible = false;
    var linesMesh = null;
    var posAttr = null;
    var colAttr = null;

    // cores por tipo (r, g, b, opacity) — cores vivas para contrastar com campo verde
    var TYPE_COLORS = {
        "-1": [0.4, 0.4, 0.5, 0.35],     // none — cinza-azulado
        "0":  [0.9, 0.9, 1.0, 0.85],     // wall — branco-azulado
        "0.2": [1.0, 1.0, 1.0, 1.0],     // ball — branco puro
        "0.4": [0.1, 0.6, 1.0, 1.0],     // ally — azul brilhante
        "0.6": [1.0, 0.2, 0.2, 1.0],     // enemy — vermelho vivo
        "0.8": [1.0, 1.0, 0.0, 0.95],    // goal_own — amarelo
        "1":   [1.0, 0.5, 0.0, 0.95]     // goal_enemy — laranja
    };

    AISoccer.createRaycastVisuals = function (scene) {
        var positions = new Float32Array(TOTAL_VERTS * 3);
        var colors = new Float32Array(TOTAL_VERTS * 4);

        var geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));

        var material = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            depthTest: false
        });

        linesMesh = new THREE.LineSegments(geometry, material);
        linesMesh.visible = false;
        linesMesh.frustumCulled = false;
        scene.add(linesMesh);

        posAttr = geometry.getAttribute('position');
        colAttr = geometry.getAttribute('color');

        return linesMesh;
    };

    AISoccer.updateRaycasts = function (players, state) {
        if (!visible || !linesMesh || !AISoccer.Physics) return;

        var S = AISoccer.SCALE;
        var RAY_ANGLES = AISoccer.Physics.RAY_ANGLES;
        var MAX_DIST = AISoccer.Physics.MAX_RAY_DISTANCE;
        var vertIdx = 0;

        for (var pi = 0; pi < players.length && pi < MAX_PLAYERS; pi++) {
            var player = players[pi];
            var rayData = AISoccer.Physics.castRays(player, state);

            var ox = player.x * S;
            var oz = player.y * S;
            var oy = 0.3;  // elevado para melhor visibilidade

            for (var ri = 0; ri < RAY_ANGLES.length; ri++) {
                var normDist = rayData[ri * 2];
                var typeId = rayData[ri * 2 + 1];
                var rayAngle = player.angle + RAY_ANGLES[ri];
                var dist = normDist * MAX_DIST;

                // physics X → Three.js X, Physics Y → Three.js Z
                var ex = ox + Math.cos(rayAngle) * dist * S;
                var ez = oz + Math.sin(rayAngle) * dist * S;

                // origem
                posAttr.array[vertIdx * 3] = ox;
                posAttr.array[vertIdx * 3 + 1] = oy;
                posAttr.array[vertIdx * 3 + 2] = oz;

                // fim
                posAttr.array[(vertIdx + 1) * 3] = ex;
                posAttr.array[(vertIdx + 1) * 3 + 1] = oy;
                posAttr.array[(vertIdx + 1) * 3 + 2] = ez;

                // cor
                var typeKey = String(typeId);
                var col = TYPE_COLORS[typeKey] || TYPE_COLORS["-1"];

                for (var ci = 0; ci < 2; ci++) {
                    colAttr.array[(vertIdx + ci) * 4] = col[0];
                    colAttr.array[(vertIdx + ci) * 4 + 1] = col[1];
                    colAttr.array[(vertIdx + ci) * 4 + 2] = col[2];
                    colAttr.array[(vertIdx + ci) * 4 + 3] = col[3];
                }

                vertIdx += 2;
            }
        }

        // zerar vértices restantes (se menos de MAX_PLAYERS)
        while (vertIdx < TOTAL_VERTS) {
            posAttr.array[vertIdx * 3] = 0;
            posAttr.array[vertIdx * 3 + 1] = 0;
            posAttr.array[vertIdx * 3 + 2] = 0;
            colAttr.array[vertIdx * 4 + 3] = 0;
            vertIdx++;
        }

        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
    };

    AISoccer.toggleRaycasts = function () {
        visible = !visible;
        if (linesMesh) linesMesh.visible = visible;
        return visible;
    };

    AISoccer.isRaycastsVisible = function () {
        return visible;
    };

})();
