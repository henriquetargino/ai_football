/**
 * Goal celebration: canhões cilíndricos pretos atrás de cada gol +
 * jato VERTICAL de fogo quando alguém marca (cor do time que marcou).
 *
 * inspiração: canhões de fogo de estádio brasileiro (Botafogo).
 *   "Olha só, a cada gol é assim. Gigantes lança-chamas, olha"
 *
 *      🔥 vertical          🔥 vertical
 *      ║ jato alto         ║ jato alto
 *      █ canhão preto      █ canhão preto      ← sempre visível
 *      gol esquerdo        gol direito
 *
 * aPIs:
 *   AISoccer.createGoalCannons(scene)       // canhões (decoração permanente)
 *   AISoccer.createGoalParticles(scene)     // pool de partículas (chamas)
 *   AISoccer.celebrateGoal(scorer)          // dispara jato no gol que recebeu
 *   AISoccer.updateGoalParticles(dt)        // física: integra partículas
 *   AISoccer.disposeGoalParticles()         // reset entre partidas
 */
var AISoccer = AISoccer || {};

(function () {
    "use strict";

    // ── Configuração geométrica ────────────────────────────────────────
    // canhões na DIAGONAL da trave traseira — atrás do back post (depth
    // 0.35), ligeiramente além lateralmente da projeção da trave (GW/2 =
    // 0.80). Visualmente: cada canhão fica colado no canto traseiro do
    // gol, na diagonal de saída — como em estádios reais (Maracanã).
    var CANNON_RADIUS    = 0.08;
    var CANNON_HEIGHT    = 0.50;
    var CANNON_DEPTH_OFF = 0.55;     // 0.20 atrás do back post (em 0.35)
    var CANNON_LATERAL   = 0.92;     // 0.12 fora da trave (GW/2 = 0.80)

    // ── Configuração das partículas ────────────────────────────────────
    // calibrado pra parecer chama coerente (não blob/flash). Cada
    // partícula tem opacidade BAIXA — quando elas se sobrepõem em
    // additive blend, se somam suavemente sem saturar pra branco.
    var MAX_PER_CANNON = 140;        // pool por canhão
    var BURST_SIZE     = 90;         // partículas spawnadas por gol (chama enxuta)
    var SPAWN_DURATION = 1.2;        // segundos de jato sustentado

    var GRAVITY = -2.6;              // unidades three.js / s²
    var DAMPING = 0.78;              // resistência do ar (por segundo)

    AISoccer._goalCannons   = null;  // 4 mesh refs (debug/cleanup)
    AISoccer._goalParticles = null;  // 4 emitters

    // canhões (mesh estática, sempre visível)
    AISoccer.createGoalCannons = function (scene) {
        var FW   = AISoccer.FW;
        var midZ = AISoccer.FH / 2;

        // material: corpo preto MAS com leve metalness pra captar luz dos
        // floodlights (sem isso vira invisível no fundo escuro do estádio).
        var bodyMat  = new THREE.MeshStandardMaterial({
            color: 0x141418, roughness: 0.45, metalness: 0.55
        });
        var rimMat   = new THREE.MeshStandardMaterial({
            color: 0x2a2a32, roughness: 0.3, metalness: 0.75
        });

        function buildCannon(x, z) {
            var group = new THREE.Group();

            // base larga (cinza escuro), pra plantar visualmente o canhão
            var baseGeo = new THREE.CylinderGeometry(
                CANNON_RADIUS * 1.4, CANNON_RADIUS * 1.6, 0.05, 18);
            var base = new THREE.Mesh(baseGeo, bodyMat);
            base.position.y = 0.025;
            base.castShadow = true;
            base.receiveShadow = true;
            group.add(base);

            // corpo cilíndrico (preto)
            var bodyGeo = new THREE.CylinderGeometry(
                CANNON_RADIUS, CANNON_RADIUS, CANNON_HEIGHT, 18);
            var body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 0.05 + CANNON_HEIGHT / 2;
            body.castShadow = true;
            group.add(body);

            // anel de borda no topo (mais metálico, pra dar leitura visual)
            var rimGeo = new THREE.CylinderGeometry(
                CANNON_RADIUS * 1.08, CANNON_RADIUS * 1.08, 0.025, 18);
            var rim = new THREE.Mesh(rimGeo, rimMat);
            rim.position.y = 0.05 + CANNON_HEIGHT - 0.012;
            rim.castShadow = true;
            group.add(rim);

            group.position.set(x, 0, z);
            group.userData.isGoalCannon = true;
            scene.add(group);
            return group;
        }

        AISoccer._goalCannons = {
            // esquerda (atrás do gol esquerdo, x < 0)
            leftFront:  buildCannon(-CANNON_DEPTH_OFF, midZ - CANNON_LATERAL),
            leftBack:   buildCannon(-CANNON_DEPTH_OFF, midZ + CANNON_LATERAL),
            // direita (atrás do gol direito, x > FW)
            rightFront: buildCannon(FW + CANNON_DEPTH_OFF, midZ - CANNON_LATERAL),
            rightBack:  buildCannon(FW + CANNON_DEPTH_OFF, midZ + CANNON_LATERAL),
        };
    };

    // partículas (jato vertical)
    function buildEmitter(scene, originX, originZ) {
        var positions = new Float32Array(MAX_PER_CANNON * 3);
        var colors    = new Float32Array(MAX_PER_CANNON * 3);
        var sizes     = new Float32Array(MAX_PER_CANNON);
        var opacities = new Float32Array(MAX_PER_CANNON);
        var states    = [];
        for (var i = 0; i < MAX_PER_CANNON; i++) {
            states.push({ active: false });
            positions[i * 3 + 1] = -10;     // escondido até spawn
        }

        var geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setAttribute("aColor",   new THREE.BufferAttribute(colors,    3));
        geo.setAttribute("aSize",    new THREE.BufferAttribute(sizes,     1));
        geo.setAttribute("aOpacity", new THREE.BufferAttribute(opacities, 1));

        // soft sprite via shader. Centro brilhante, bordas suaves. Additive
        // mas sem boost de cor (evita virar branco quando partículas se
        // sobrepõem) — visualmente parece chama coerente.
        var mat = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            vertexShader: [
                "attribute vec3 aColor;",
                "attribute float aSize;",
                "attribute float aOpacity;",
                "varying vec3 vColor;",
                "varying float vOpacity;",
                "void main() {",
                "  vColor = aColor;",
                "  vOpacity = aOpacity;",
                "  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);",
                "  gl_PointSize = aSize * (300.0 / -mvPosition.z);",
                "  gl_Position = projectionMatrix * mvPosition;",
                "}"
            ].join("\n"),
            fragmentShader: [
                "varying vec3 vColor;",
                "varying float vOpacity;",
                "void main() {",
                "  vec2 c = gl_PointCoord - vec2(0.5);",
                "  float d = length(c);",
                "  if (d > 0.5) discard;",
                "  // Falloff radial agressivo (alpha alto SÓ no centro)",
                "  // pra que additive blend não sature em branco.",
                "  float alpha = pow(1.0 - smoothstep(0.0, 0.5, d), 2.4);",
                "  gl_FragColor = vec4(vColor, alpha * vOpacity);",
                "}"
            ].join("\n")
        });

        var points = new THREE.Points(geo, mat);
        points.frustumCulled = false;
        points.userData.isGoalParticles = true;
        scene.add(points);

        return {
            x: originX, z: originZ,
            geo: geo,
            positions: positions,
            colors: colors,
            sizes: sizes,
            opacities: opacities,
            states: states,
            points: points,
            // spawn-over-time: cada celebração marca um endTime e a cada
            // frame spawna proporcionalmente até atingir BURST_SIZE total.
            spawnEndTime: 0,
            spawnRemaining: 0,
            spawnColor: null,
        };
    }

    AISoccer.createGoalParticles = function (scene) {
        var FW   = AISoccer.FW;
        var midZ = AISoccer.FH / 2;
        var topY = 0.05 + CANNON_HEIGHT;     // saída da boca do canhão

        AISoccer._goalParticles = {
            leftFront:  buildEmitter(scene, -CANNON_DEPTH_OFF, midZ - CANNON_LATERAL),
            leftBack:   buildEmitter(scene, -CANNON_DEPTH_OFF, midZ + CANNON_LATERAL),
            rightFront: buildEmitter(scene, FW + CANNON_DEPTH_OFF, midZ - CANNON_LATERAL),
            rightBack:  buildEmitter(scene, FW + CANNON_DEPTH_OFF, midZ + CANNON_LATERAL),
            cannonTopY: topY,
        };
    };

    AISoccer.celebrateGoal = function (scorer) {
        if (!AISoccer._goalParticles) return;

        // decide quais 2 canhões disparam: os do gol que recebeu a bola.
        // RED scores → ball foi pro gol direito (BLUE defende) → canhões direitos
        // BLUE scores → ball foi pro gol esquerdo (RED defende) → canhões esquerdos
        var emitters = scorer === "red"
            ? [AISoccer._goalParticles.rightFront, AISoccer._goalParticles.rightBack]
            : [AISoccer._goalParticles.leftFront,  AISoccer._goalParticles.leftBack];

        // rate-limit: se ALGUM dos canhões selecionados ainda tem spawn
        // pendente (burst anterior em andamento), ignora a chamada.
        // defensivo contra duplo-trigger (ex: bug que dispare celebrate 2x
        // no mesmo frame). Pra um celebrate "fresh" depois do burst acabar,
        // emitters[k].spawnRemaining é 0 → segue normal.
        for (var i = 0; i < emitters.length; i++) {
            if (emitters[i].spawnRemaining > 0) return;
        }

        var color = scorer === "red"
            ? { r: 1.00, g: 0.50, b: 0.10 }     // laranja (red team)
            : { r: 0.20, g: 0.58, b: 1.00 };    // azul

        var now = (typeof performance !== "undefined")
            ? performance.now() / 1000 : Date.now() / 1000;

        for (var k = 0; k < emitters.length; k++) {
            var em = emitters[k];
            em.spawnEndTime = now + SPAWN_DURATION;
            em.spawnRemaining = BURST_SIZE;
            em.spawnColor = color;
        }
    };

    /** Spawna uma partícula nova num emitter (chama vertical). */
    function spawnParticle(em, color, topY) {
        for (var i = 0; i < em.states.length; i++) {
            var s = em.states[i];
            if (s.active) continue;
            s.active = true;
            s.life = 0;
            s.maxLife = 1.4 + Math.random() * 1.2;     // 1.4-2.6s

            // origem: boca do canhão, com microspread (~radius do canhão)
            var spreadXZ = 0.04;
            s.x = em.x + (Math.random() - 0.5) * spreadXZ;
            s.y = topY + Math.random() * 0.05;
            s.z = em.z + (Math.random() - 0.5) * spreadXZ;

            // velocidade: VERTICAL alta (jato direto pra cima).
            // spread lateral mínimo (chama coerente, não explosão).
            s.vx = (Math.random() - 0.5) * 0.3;
            s.vy = 6.5 + Math.random() * 3.5;          // 6.5-10 m/s pra cima
            s.vz = (Math.random() - 0.5) * 0.3;

            // tamanho REDUZIDO pra evitar blob sob additive blend
            s.size = 0.025 + Math.random() * 0.025;
            s.color = color;
            return true;
        }
        return false;       // pool cheio
    }

    AISoccer.updateGoalParticles = function (dt) {
        if (!AISoccer._goalParticles) return;
        if (typeof dt !== "number" || !isFinite(dt) || dt <= 0) dt = 1 / 60;
        if (dt > 0.05) dt = 0.05;

        var dampPow = Math.pow(DAMPING, dt);
        var topY = AISoccer._goalParticles.cannonTopY;
        var now = (typeof performance !== "undefined")
            ? performance.now() / 1000 : Date.now() / 1000;

        function step(em) {
            // spawn rate-limited: divide partículas restantes pela duração
            // restante (em frames) — produz jato sustentado, não explosão.
            if (em.spawnRemaining > 0 && em.spawnColor) {
                var timeLeft = em.spawnEndTime - now;
                if (timeLeft <= 0) {
                    // termina forçando spawn de tudo que sobrou (fim do jato)
                    while (em.spawnRemaining > 0) {
                        if (!spawnParticle(em, em.spawnColor, topY)) break;
                        em.spawnRemaining--;
                    }
                    em.spawnColor = null;
                } else {
                    var thisFrame = Math.ceil(em.spawnRemaining * (dt / timeLeft));
                    if (thisFrame > em.spawnRemaining) thisFrame = em.spawnRemaining;
                    for (var k = 0; k < thisFrame; k++) {
                        if (!spawnParticle(em, em.spawnColor, topY)) break;
                        em.spawnRemaining--;
                    }
                }
            }

            // integração + buffer write
            var pos = em.positions;
            var col = em.colors;
            var sz  = em.sizes;
            var op  = em.opacities;

            for (var i = 0; i < em.states.length; i++) {
                var s = em.states[i];
                if (!s.active) {
                    op[i] = 0;
                    continue;
                }
                s.life += dt;
                if (s.life >= s.maxLife) {
                    s.active = false;
                    op[i] = 0;
                    continue;
                }
                // damping vertical um pouco mais leve (pra subir mais alto)
                s.vx *= dampPow;
                s.vy = (s.vy + GRAVITY * dt) * Math.pow(0.92, dt);
                s.vz *= dampPow;
                s.x += s.vx * dt;
                s.y += s.vy * dt;
                s.z += s.vz * dt;

                pos[i * 3]     = s.x;
                pos[i * 3 + 1] = s.y;
                pos[i * 3 + 2] = s.z;
                col[i * 3]     = s.color.r;
                col[i * 3 + 1] = s.color.g;
                col[i * 3 + 2] = s.color.b;

                // fade: in rápido (50ms), out exponencial. Opacidade MAX
                // baixa (0.45) pra additive não saturar em branco.
                var t = s.life / s.maxLife;
                var fadeIn  = Math.min(1, s.life / 0.05);
                var fadeOut = Math.pow(1 - t, 1.4);
                op[i] = Math.max(0, fadeIn * fadeOut * 0.45);

                // tamanho cresce levemente subindo, depois encolhe.
                // multiplicador menor (size em pixels mais contido).
                var grow = t < 0.3 ? (1 + t * 0.8) : (1.24 - (t - 0.3) * 0.6);
                sz[i] = s.size * (40 + 30 * grow);
            }

            em.geo.attributes.position.needsUpdate = true;
            em.geo.attributes.aColor.needsUpdate   = true;
            em.geo.attributes.aSize.needsUpdate    = true;
            em.geo.attributes.aOpacity.needsUpdate = true;
        }

        var ems = AISoccer._goalParticles;
        step(ems.leftFront);
        step(ems.leftBack);
        step(ems.rightFront);
        step(ems.rightBack);
    };

    /** Soft reset — desativa partículas em vôo e zera spawn pendente sem
     *  destruir o pool. Usado quando o usuário arrasta a timeline do replay
     *  pra trás (não faz sentido manter chamas/cubos de um gol que vai
     *  acontecer "no futuro" do playhead). */
    AISoccer.resetGoalParticles = function () {
        if (!AISoccer._goalParticles) return;
        ["leftFront", "leftBack", "rightFront", "rightBack"].forEach(function (k) {
            var em = AISoccer._goalParticles[k];
            if (!em) return;
            em.spawnRemaining = 0;
            em.spawnColor = null;
            em.spawnEndTime = 0;
            for (var i = 0; i < em.states.length; i++) {
                em.states[i].active = false;
                em.opacities[i] = 0;
            }
            em.geo.attributes.aOpacity.needsUpdate = true;
        });
    };

    AISoccer.disposeGoalParticles = function () {
        // self-contained: remove cada Points/Group da cena E faz dispose
        // explícito de geo+material. Antes confiava só em clearStadium pra
        // disposer via traverse(), mas se o dispose é chamado fora de uma
        // cleanupCurrentMode sequence, ficavam órfãos na GPU (memory leak).
        if (AISoccer._goalParticles) {
            ["leftFront", "leftBack", "rightFront", "rightBack"].forEach(function (k) {
                var em = AISoccer._goalParticles[k];
                if (!em || !em.points) return;
                if (em.points.parent) em.points.parent.remove(em.points);
                if (em.points.geometry) em.points.geometry.dispose();
                if (em.points.material) em.points.material.dispose();
            });
        }
        if (AISoccer._goalCannons) {
            Object.keys(AISoccer._goalCannons).forEach(function (k) {
                var grp = AISoccer._goalCannons[k];
                if (!grp) return;
                grp.traverse(function (o) {
                    if (o.isMesh) {
                        if (o.geometry) o.geometry.dispose();
                        if (o.material) o.material.dispose();
                    }
                });
                if (grp.parent) grp.parent.remove(grp);
            });
        }
        AISoccer._goalParticles = null;
        AISoccer._goalCannons   = null;
    };

})();
