/**
 * robot shatter — quando um time leva gol, o robô do time que SOFREU
 * "explode" em vários cubinhos da cor do time, que caem no chão e
 * somem após GOAL_CELEBRATION_S. Inspirado em AI Warehouse.
 *
 *      ▢ robô feliz                 ▣▢▣▢ robô se desmonta
 *      ▢▢▢▢                         ▢▣▢▢▢ pra cima/lados, gravidade
 *                                    ▢▢▢▣▢▢▣▢ caem no chão
 *
 * aPIs:
 *   AISoccer.createRobotShatter(scene)
 *   AISoccer.shatterPlayer(team, x, y, z)
 *   AISoccer.updateShatter(dt)
 *   AISoccer.disposeRobotShatter()
 */
var AISoccer = AISoccer || {};

(function () {
    "use strict";

    var MAX_CUBES   = 220;          // pool total (suporta ~2 jogadores simultâneos)
    var BURST_SIZE  = 90;           // cubos por jogador "shattered"
    var GRAVITY     = -6.5;
    var GROUND_DAMP = 0.45;         // perda de energia ao bater no chão

    AISoccer._shatterPool = null;

    AISoccer.createRobotShatter = function (scene) {
        // shared geometry — antes cada cubo tinha seu próprio BoxGeometry
        // (220 alocações distintas com tamanhos aleatórios), gastando
        // ~200KB VRAM. Agora 1 geo unitária + scale individual via mesh.scale.
        var sharedGeo = new THREE.BoxGeometry(1, 1, 1);

        var cubes = [];
        for (var i = 0; i < MAX_CUBES; i++) {
            // tamanho variado por cubo via scale (não via geo)
            var size = 0.025 + Math.random() * 0.035;
            var mat = new THREE.MeshStandardMaterial({
                color: 0xffffff, roughness: 0.55, metalness: 0.05
            });
            var mesh = new THREE.Mesh(sharedGeo, mat);
            mesh.scale.set(size, size, size);
            mesh.castShadow = true;
            mesh.visible = false;
            mesh.userData.isShatterCube = true;
            scene.add(mesh);
            cubes.push({
                mesh: mesh, mat: mat,
                active: false,
                life: 0, maxLife: 0,
                vx: 0, vy: 0, vz: 0,
                avx: 0, avy: 0, avz: 0,
            });
        }
        AISoccer._shatterPool = { cubes: cubes, sharedGeo: sharedGeo };
    };

    AISoccer.shatterPlayer = function (team, x, y, z) {
        if (!AISoccer._shatterPool) return;
        // cor do time que LEVOU o gol — robô explodiu na cor dele mesmo,
        // criando contraste visual com os canhões (cor do que MARCOU).
        var color = team === "red" ? 0xFF6B1A : 0x4A9BFF;
        var spawned = 0;
        var cubes = AISoccer._shatterPool.cubes;
        for (var i = 0; i < cubes.length && spawned < BURST_SIZE; i++) {
            var c = cubes[i];
            if (c.active) continue;
            c.active = true;
            c.life = 0;
            c.maxLife = 2.5 + Math.random() * 0.5;     // dura quase os 3s
            c.mat.color.setHex(color);
            c.mat.opacity = 1;
            c.mat.transparent = false;
            c.mesh.visible = true;
            c.mesh.position.set(
                x + (Math.random() - 0.5) * 0.20,
                y + 0.10 + Math.random() * 0.30,
                z + (Math.random() - 0.5) * 0.20
            );
            c.mesh.rotation.set(
                Math.random() * Math.PI * 2,
                Math.random() * Math.PI * 2,
                Math.random() * Math.PI * 2
            );
            // explosão radial XZ + upward boost
            var angle = Math.random() * Math.PI * 2;
            var horiz = 0.6 + Math.random() * 1.6;
            c.vx = Math.cos(angle) * horiz;
            c.vy = 1.4 + Math.random() * 2.5;
            c.vz = Math.sin(angle) * horiz;
            // spin caótico
            c.avx = (Math.random() - 0.5) * 12;
            c.avy = (Math.random() - 0.5) * 12;
            c.avz = (Math.random() - 0.5) * 12;
            spawned++;
        }
    };

    AISoccer.updateShatter = function (dt) {
        if (!AISoccer._shatterPool) return;
        if (typeof dt !== "number" || !isFinite(dt) || dt <= 0) dt = 1 / 60;
        if (dt > 0.05) dt = 0.05;
        var cubes = AISoccer._shatterPool.cubes;

        for (var i = 0; i < cubes.length; i++) {
            var c = cubes[i];
            if (!c.active) continue;
            c.life += dt;
            if (c.life >= c.maxLife) {
                c.active = false;
                c.mesh.visible = false;
                continue;
            }
            // integração com gravidade
            c.vy += GRAVITY * dt;
            c.mesh.position.x += c.vx * dt;
            c.mesh.position.y += c.vy * dt;
            c.mesh.position.z += c.vz * dt;
            c.mesh.rotation.x += c.avx * dt;
            c.mesh.rotation.y += c.avy * dt;
            c.mesh.rotation.z += c.avz * dt;
            // quica no chão e perde energia
            if (c.mesh.position.y < 0.015) {
                c.mesh.position.y = 0.015;
                c.vy = -c.vy * GROUND_DAMP;
                c.vx *= GROUND_DAMP;
                c.vz *= GROUND_DAMP;
                c.avx *= GROUND_DAMP;
                c.avz *= GROUND_DAMP;
            }
            // fade nos últimos 0.6s
            var remain = c.maxLife - c.life;
            if (remain < 0.6) {
                c.mat.transparent = true;
                c.mat.opacity = remain / 0.6;
            }
        }
    };

    /** Soft reset — desativa cubos em vôo sem destruir o pool. Usado em
     *  replay seek-backward: cubos que representavam um gol futuro do
     *  playhead atual devem sumir imediatamente (não esperar fade). */
    AISoccer.resetRobotShatter = function () {
        if (!AISoccer._shatterPool || !AISoccer._shatterPool.cubes) return;
        AISoccer._shatterPool.cubes.forEach(function (c) {
            c.active = false;
            if (c.mesh) c.mesh.visible = false;
        });
    };

    AISoccer.disposeRobotShatter = function () {
        // self-contained: cada cubo tem material próprio (cor varia), mas
        // a geometry agora é COMPARTILHADA entre os 220 cubos — dispose
        // só uma vez via sharedGeo, e materials individualmente.
        if (AISoccer._shatterPool) {
            if (AISoccer._shatterPool.cubes) {
                AISoccer._shatterPool.cubes.forEach(function (c) {
                    if (c.mesh && c.mesh.parent) c.mesh.parent.remove(c.mesh);
                    if (c.mat) c.mat.dispose();
                });
            }
            if (AISoccer._shatterPool.sharedGeo) {
                AISoccer._shatterPool.sharedGeo.dispose();
            }
        }
        AISoccer._shatterPool = null;
    };

})();
