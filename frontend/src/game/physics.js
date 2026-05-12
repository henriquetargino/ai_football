/**
 * reimplementação 1:1 da física Python em JavaScript.
 * mesmas fórmulas, constantes, e ordem de operações.
 */

var AISoccer = AISoccer || {};

AISoccer.Physics = (function () {

    // ── Constantes (idênticas a config.py) ──
    var FIELD_WIDTH = 800;
    var FIELD_HEIGHT = 500;
    var PLAYER_RADIUS = 20;
    var PLAYER_MASS = 2.0;
    var MAX_SPEED_PLAYER = 3.5;
    var ACCEL_FORCE = 0.3;
    var ROTATION_SPEED = 0.06;
    var FRICTION_PLAYER = 0.92;
    var BALL_RADIUS = 12;
    var BALL_MASS = 0.3;
    var MAX_SPEED_BALL = 12.0;
    var FRICTION_BALL = 0.985;
    var KICK_FORCE = 7.0;
    var KICK_COOLDOWN = 18;
    var KICK_REACH_TOLERANCE = 1;
    var WALL_BOUNCE = 0.5;
    var CORNER_CUT = 30.0;             // V9.4 — RAIO do arco do canto (paridade backend)
    var CORNER_ARC_SEGMENTS = 8;       // V9.4 — paridade backend
    var SQRT2 = Math.sqrt(2);
    var INV_SQRT2 = 1 / SQRT2;
    var ENTITY_BOUNCE = 0.7;
    var POST_RADIUS = 5;
    var POST_BOUNCE = 0.85;
    var MAX_RAY_DISTANCE = 1000;
    var PHYSICS_FPS = 60;

    var TYPE_MAP = {
        none: -1.0, wall: 0.0, ball: 0.2,
        ally: 0.4, enemy: 0.6, goal_own: 0.8, goal_enemy: 1.0
    };

    // ── Ray angles (exatamente igual a config.py._build_ray_angles) ──
    var RAY_ANGLES = (function () {
        var angles = [];
        var i, a;
        var deg = Math.PI / 180;
        // zona frontal: -30 a +30, 25 raios
        for (i = 0; i < 25; i++) {
            a = (-30 + (60 / 24) * i) * deg;
            angles.push(a);
        }
        // zona lateral esquerda: -90 a -30, 8 raios
        for (i = 0; i < 8; i++) {
            a = (-90 + (60 / 8) * i) * deg;
            if (a < -30 * deg) angles.push(a);
        }
        // zona lateral direita: +30 a +90, 8 raios
        for (i = 0; i < 8; i++) {
            a = (30 + (60 / 8) * i) * deg;
            if (a > 30 * deg) angles.push(a);
        }
        // zona traseira esquerda: -170 a -90, 4 raios
        for (i = 0; i < 4; i++) {
            a = (-170 + (80 / 4) * i) * deg;
            angles.push(a);
        }
        // zona traseira direita: +90 a +170, 4 raios
        for (i = 0; i < 4; i++) {
            a = (90 + (80 / 4) * i) * deg;
            angles.push(a);
        }
        angles.sort(function (a, b) { return a - b; });
        return angles;
    })();

    // entities (objetos simples)
    function makeBall() {
        return {
            id: "ball", x: FIELD_WIDTH / 2, y: FIELD_HEIGHT / 2,
            vx: 0, vy: 0, radius: BALL_RADIUS, mass: BALL_MASS
        };
    }

    function makePlayer(id, team) {
        return {
            id: id, team: team, x: 0, y: 0, vx: 0, vy: 0,
            angle: 0, radius: PLAYER_RADIUS, mass: PLAYER_MASS,
            accel: 0, rot: 0, kick_requested: false,
            can_kick: true, kick_cooldown_timer: 0, is_kicking: false,
            kick_attempted: false
        };
    }

    function buildGoals(fw, fh, gw) {
        var midY = fh / 2;
        var half = gw / 2;
        return [
            { team: "red", side: "left", y_min: midY - half, y_max: midY + half },
            { team: "blue", side: "right", y_min: midY - half, y_max: midY + half }
        ];
    }

    function buildWalls(fw, fh, gw) {
        // v9.4 — paredes axis-aligned em 0/fw e 0/fh + 4 arcos arredondados
        // nos cantos. Arcos aproximados por CORNER_ARC_SEGMENTS=8 segmentos
        // (raycast) — colisão usa fórmula analítica de círculo.
        // paridade 1:1 com Field.build_walls em backend/physics/entities.py.
        var midY = fh / 2;
        var half = gw / 2;
        var cut = CORNER_CUT;
        var walls = [
            // esquerda (com abertura pro gol)
            { x1: 0, y1: cut, x2: 0, y2: midY - half },
            { x1: 0, y1: midY + half, x2: 0, y2: fh - cut },
            // direita (com abertura pro gol)
            { x1: fw, y1: cut, x2: fw, y2: midY - half },
            { x1: fw, y1: midY + half, x2: fw, y2: fh - cut },
            // superior / Inferior (entre arcos)
            { x1: cut, y1: 0, x2: fw - cut, y2: 0 },
            { x1: cut, y1: fh, x2: fw - cut, y2: fh }
        ];
        // 4 arcos arredondados (NW, NE, SE, SW) aproximados por N segmentos
        var N = CORNER_ARC_SEGMENTS;
        var arcs = [
            [cut,        cut,        Math.PI,           3 * Math.PI / 2],   // NW
            [fw - cut,   cut,        3 * Math.PI / 2,   2 * Math.PI],        // NE
            [fw - cut,   fh - cut,   0.0,               Math.PI / 2],        // SE
            [cut,        fh - cut,   Math.PI / 2,       Math.PI]             // SW
        ];
        for (var ai = 0; ai < arcs.length; ai++) {
            var cx = arcs[ai][0], cy = arcs[ai][1], t1 = arcs[ai][2], t2 = arcs[ai][3];
            for (var i = 0; i < N; i++) {
                var a1 = t1 + (t2 - t1) * (i / N);
                var a2 = t1 + (t2 - t1) * ((i + 1) / N);
                walls.push({
                    x1: cx + cut * Math.cos(a1),
                    y1: cy + cut * Math.sin(a1),
                    x2: cx + cut * Math.cos(a2),
                    y2: cy + cut * Math.sin(a2)
                });
            }
        }
        return walls;
    }

    function buildPosts(fw, fh, gw) {
        // 4 postes circulares estáticos — 2 por gol, nas pontas das aberturas.
        var midY = fh / 2;
        var half = gw / 2;
        return [
            { x: 0,  y: midY - half, radius: POST_RADIUS },
            { x: 0,  y: midY + half, radius: POST_RADIUS },
            { x: fw, y: midY - half, radius: POST_RADIUS },
            { x: fw, y: midY + half, radius: POST_RADIUS }
        ];
    }

    // engine
    function createGameState(mode, goalWidth) {
        goalWidth = goalWidth || 150;
        mode = mode || "1v1";
        var players = [];
        if (mode === "1v1") {
            players = [makePlayer("red_0", "red"), makePlayer("blue_0", "blue")];
        } else {
            players = [
                makePlayer("red_0", "red"), makePlayer("red_1", "red"),
                makePlayer("blue_0", "blue"), makePlayer("blue_1", "blue")
            ];
        }
        var state = {
            field: { width: FIELD_WIDTH, height: FIELD_HEIGHT, goal_width: goalWidth },
            ball: makeBall(),
            players: players,
            goals: buildGoals(FIELD_WIDTH, FIELD_HEIGHT, goalWidth),
            walls: buildWalls(FIELD_WIDTH, FIELD_HEIGHT, goalWidth),
            posts: buildPosts(FIELD_WIDTH, FIELD_HEIGHT, goalWidth),
            score: { red: 0, blue: 0 },
            step_count: 0,
            goal_scored_this_step: false,
            scoring_team: null
        };
        randomizeSpawns(state);
        return state;
    }

    function randomizeSpawns(state) {
        // posição sorteada — usado em TREINO (paridade com backend).
        var midX = state.field.width / 2;
        var margin = 50;
        for (var i = 0; i < state.players.length; i++) {
            var p = state.players[i];
            if (p.team === "red") {
                p.x = margin + Math.random() * (midX - 2 * margin);
            } else {
                p.x = midX + margin + Math.random() * (state.field.width - midX - 2 * margin);
            }
            p.y = margin + Math.random() * (state.field.height - 2 * margin);
            p.angle = Math.random() * 2 * Math.PI;
            p.vx = 0; p.vy = 0;
        }
        state.ball.x = midX + (Math.random() * 40 - 20);
        state.ball.y = state.field.height / 2 + (Math.random() * 40 - 20);
        state.ball.vx = 0; state.ball.vy = 0;
    }

    function spawnDeterministic(state) {
        // posição FIXA e simétrica — usado no MODO LIVE do frontend
        // (demonstração). Garante jogo justo: red e blue nascem à mesma
        // distância da bola, olhando um para o gol do outro.
        var midX = state.field.width / 2;
        var midY = state.field.height / 2;
        var quarterX = state.field.width / 4;
        for (var i = 0; i < state.players.length; i++) {
            var p = state.players[i];
            if (p.team === "red") {
                p.x = quarterX;          // 200 (com FIELD_WIDTH=800)
                p.angle = 0;             // olhando para o gol azul (direita)
            } else {
                p.x = state.field.width - quarterX;  // 600
                p.angle = Math.PI;       // olhando para o gol vermelho (esquerda)
            }
            p.y = midY;
            p.vx = 0; p.vy = 0;
        }
        state.ball.x = midX;
        state.ball.y = midY;
        state.ball.vx = 0; state.ball.vy = 0;
    }

    function applyControls(player, outputs) {
        player.accel = Math.max(-1, Math.min(1, outputs[0]));
        player.rot = Math.max(-1, Math.min(1, outputs[1]));
        player.kick_requested = outputs[2] > 0.5;
    }

    function clampSpeed(entity, maxSpeed) {
        var speed = Math.sqrt(entity.vx * entity.vx + entity.vy * entity.vy);
        if (speed > maxSpeed) {
            var f = maxSpeed / speed;
            entity.vx *= f;
            entity.vy *= f;
        }
    }

    function tryKick(player, ball, state) {
        if (!player.can_kick) return;
        var dx = ball.x - player.x;
        var dy = ball.y - player.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > player.radius + ball.radius + KICK_REACH_TOLERANCE) return;

        ball.vx += Math.cos(player.angle) * KICK_FORCE;
        ball.vy += Math.sin(player.angle) * KICK_FORCE;
        clampSpeed(ball, MAX_SPEED_BALL);

        player.can_kick = false;
        player.kick_cooldown_timer = KICK_COOLDOWN;
        player.is_kicking = true;
        if (state) state.last_kicker_team = player.team;
    }

    function isInGoalOpening(y, side, state) {
        for (var i = 0; i < state.goals.length; i++) {
            var g = state.goals[i];
            if (g.side === side) return g.y_min <= y && y <= g.y_max;
        }
        return false;
    }

    function isInGoalOpeningStrict(entity, side, state) {
        for (var i = 0; i < state.goals.length; i++) {
            var g = state.goals[i];
            if (g.side === side) {
                return (g.y_min + entity.radius) <= entity.y && entity.y <= (g.y_max - entity.radius);
            }
        }
        return false;
    }

    function checkGoal(state) {
        // regra: bola precisa ter passado INTEIRA da linha (igual futebol real).
        // esquerda  → borda DIREITA da bola (bx + radius) < 0
        // direita   → borda ESQUERDA da bola (bx - radius) > field.width
        // paridade 1:1 com _check_goal em backend/physics/engine.py.
        //
        // `state.celebration_active` (setado pelo main.js durante o delay
        // pós-gol) suprime detecção/respawn — bola fica off-field por uns
        // segundos pra dar tempo da celebração visual rodar antes do reset.
        if (state.celebration_active) return;

        var bx = state.ball.x, by = state.ball.y;
        // `state.deterministic_spawn` (setado pelo main.js no modo live) faz
        // o reset após gol usar spawnDeterministic em vez de randomizeSpawns.
        var respawn = state.deterministic_spawn ? spawnDeterministic : randomizeSpawns;
        for (var i = 0; i < state.goals.length; i++) {
            var goal = state.goals[i];
            if (by >= goal.y_min && by <= goal.y_max) {
                if (goal.side === "left" && bx + state.ball.radius < 0) {
                    state.score.blue += 1;
                    state.goal_scored_this_step = true;
                    state.scoring_team = "blue";
                    // nÃO chama respawn — main.js fará após delay de celebração
                    return;
                } else if (goal.side === "right" && bx - state.ball.radius > state.field.width) {
                    state.score.red += 1;
                    state.goal_scored_this_step = true;
                    state.scoring_team = "red";
                    return;
                }
            }
        }
    }

    // collision
    function resolveAllCollisions(state) {
        var entities = [state.ball].concat(state.players);
        var corrections = {};
        for (var i = 0; i < entities.length; i++) {
            corrections[entities[i].id] = [0, 0, 0, 0];
        }

        // detect circle-circle
        for (var i = 0; i < entities.length; i++) {
            for (var j = i + 1; j < entities.length; j++) {
                resolveCircleCircle(entities[i], entities[j], corrections);
            }
        }
        // detect wall
        for (var i = 0; i < entities.length; i++) {
            resolveWallCollisions(entities[i], state, corrections);
        }
        // v9.4: cantos arredondados (escanteios curvos) — após paredes axis-aligned.
        if (CORNER_CUT > 0) {
            for (var i = 0; i < entities.length; i++) {
                resolveCornerArcCollisions(entities[i], state, corrections);
            }
        }
        // postes DEPOIS das paredes — a ordem importa para paridade com Python.
        for (var i = 0; i < entities.length; i++) {
            resolvePostCollisions(entities[i], state, corrections);
        }
        // apply simultaneously
        for (var i = 0; i < entities.length; i++) {
            var e = entities[i];
            var c = corrections[e.id];
            e.x += c[0]; e.y += c[1]; e.vx += c[2]; e.vy += c[3];
        }
    }

    function resolveCircleCircle(a, b, corrections) {
        var dx = b.x - a.x;
        var dy = b.y - a.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        var minDist = a.radius + b.radius;
        if (dist >= minDist || dist === 0) return;

        var nx = dx / dist;
        var ny = dy / dist;
        var overlap = minDist - dist;
        var totalMass = a.mass + b.mass;
        var aRatio = b.mass / totalMass;
        var bRatio = a.mass / totalMass;

        // 1. Position correction (always)
        corrections[a.id][0] -= nx * overlap * aRatio;
        corrections[a.id][1] -= ny * overlap * aRatio;
        corrections[b.id][0] += nx * overlap * bRatio;
        corrections[b.id][1] += ny * overlap * bRatio;

        // 2. Velocity correction
        var dvx = a.vx - b.vx;
        var dvy = a.vy - b.vy;
        var dvn = dvx * nx + dvy * ny;

        if (dvn > 0) {
            // objects are "separating" but STILL overlapping (sustained contact).
            // apply overlap-proportional separation impulse to prevent magnet effect.
            // without this, the ball floats in front of the player like an elastic band.
            var sepForce = overlap * 1.5;
            corrections[a.id][2] -= nx * sepForce * aRatio;
            corrections[a.id][3] -= ny * sepForce * aRatio;
            corrections[b.id][2] += nx * sepForce * bRatio;
            corrections[b.id][3] += ny * sepForce * bRatio;
            return;
        }

        // normal elastic collision impulse (approaching objects)
        var imp = -(1 + ENTITY_BOUNCE) * dvn / (1 / a.mass + 1 / b.mass);
        corrections[a.id][2] += (imp / a.mass) * nx;
        corrections[a.id][3] += (imp / a.mass) * ny;
        corrections[b.id][2] -= (imp / b.mass) * nx;
        corrections[b.id][3] -= (imp / b.mass) * ny;
    }

    function resolveWallCollisions(entity, state, corrections) {
        // v9.4: paredes axis-aligned só atuam fora das zonas dos cantos
        // (arcos tratam essas zonas). Paridade 1:1 com Python.
        var r = entity.radius;
        var eid = entity.id;
        var fw = state.field.width;
        var fh = state.field.height;
        var GOAL_DEPTH = 40;
        var cut = CORNER_CUT;

        if (entity.x - r < 0) {
            var canPassL = entity.id === "ball"
                ? isInGoalOpening(entity.y, "left", state)
                : isInGoalOpeningStrict(entity, "left", state);
            var insideGoalL = entity.x < 0 && entity.x > -GOAL_DEPTH;
            var inCornerZoneL = entity.y < cut || entity.y > fh - cut;
            if (!canPassL && !insideGoalL && !inCornerZoneL) {
                corrections[eid][0] += r - entity.x;
                if (entity.vx < 0) corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE);
            }
        }
        if (entity.x + r > fw) {
            var canPassR = entity.id === "ball"
                ? isInGoalOpening(entity.y, "right", state)
                : isInGoalOpeningStrict(entity, "right", state);
            var insideGoalR = entity.x > fw && entity.x < fw + GOAL_DEPTH;
            var inCornerZoneR = entity.y < cut || entity.y > fh - cut;
            if (!canPassR && !insideGoalR && !inCornerZoneR) {
                corrections[eid][0] -= (entity.x + r) - fw;
                if (entity.vx > 0) corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE);
            }
        }
        if (entity.y - r < 0) {
            var inCornerZoneT = entity.x < cut || entity.x > fw - cut;
            if (!inCornerZoneT) {
                corrections[eid][1] += r - entity.y;
                if (entity.vy < 0) corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE);
            }
        }
        if (entity.y + r > fh) {
            var inCornerZoneB = entity.x < cut || entity.x > fw - cut;
            if (!inCornerZoneB) {
                corrections[eid][1] -= (entity.y + r) - fh;
                if (entity.vy > 0) corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE);
            }
        }

        // Goal interior walls (always for players; pra bola só DURANTE
        // celebration_active — antes do gol a bola precisa passar livre,
        // mas DEPOIS do gol ela ficaria voando pra fora; o back-wall a
        // contém pra ela ficar quicando dentro do gol nos 3s).
        var ballNeedsContainment = entity.id === "ball" && state.celebration_active;
        if (entity.id !== "ball" || ballNeedsContainment) {
            for (var gi = 0; gi < state.goals.length; gi++) {
                var goal = state.goals[gi];
                var gx = goal.side === "left" ? 0 : fw;
                var behindGoal = (goal.side === "left" && entity.x < gx) ||
                    (goal.side === "right" && entity.x > gx);

                if (behindGoal) {
                    // top post
                    if (entity.y - r < goal.y_min) {
                        corrections[eid][1] += goal.y_min + r - entity.y;
                        if (entity.vy < 0) corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE);
                    }
                    // bottom post
                    if (entity.y + r > goal.y_max) {
                        corrections[eid][1] -= entity.y + r - goal.y_max;
                        if (entity.vy > 0) corrections[eid][3] += -entity.vy * (1 + WALL_BOUNCE);
                    }
                    // back wall
                    if (goal.side === "left" && entity.x - r < gx - GOAL_DEPTH) {
                        corrections[eid][0] += (gx - GOAL_DEPTH + r) - entity.x;
                        if (entity.vx < 0) corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE);
                    } else if (goal.side === "right" && entity.x + r > gx + GOAL_DEPTH) {
                        corrections[eid][0] -= entity.x + r - (gx + GOAL_DEPTH);
                        if (entity.vx > 0) corrections[eid][2] += -entity.vx * (1 + WALL_BOUNCE);
                    }
                }
            }
        }
    }

    /**
     * v9.4 — Cantos arredondados (escanteios curvos, levemente).
     *
     * paridade 1:1 com backend/physics/collision.py::_resolve_corner_arc_collisions.
     * a área jogável na zona do canto é o INTERIOR de um quadrante de
     * círculo de raio CORNER_CUT centrado a CORNER_CUT do canto.
     */
    function resolveCornerArcCollisions(entity, state, corrections) {
        var r = entity.radius;
        var eid = entity.id;
        var fw = state.field.width;
        var fh = state.field.height;
        var cut = CORNER_CUT;

        // determinar qual canto (zonas mutuamente exclusivas)
        var cx, cy;
        if (entity.x < cut && entity.y < cut) {
            cx = cut;        cy = cut;        // NW
        } else if (entity.x > fw - cut && entity.y < cut) {
            cx = fw - cut;   cy = cut;        // NE
        } else if (entity.x > fw - cut && entity.y > fh - cut) {
            cx = fw - cut;   cy = fh - cut;   // SE
        } else if (entity.x < cut && entity.y > fh - cut) {
            cx = cut;        cy = fh - cut;   // SW
        } else {
            return;  // entity fora das zonas dos cantos
        }

        // vetor radial: do CENTRO PRA ENTITY (radialmente pra fora)
        var dx = entity.x - cx;
        var dy = entity.y - cy;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist === 0) return;  // caso degenerado

        // sem colisão se entity dentro do círculo: dist + r ≤ cut.
        // colisão se dist + r > cut → overlap = dist + r - cut.
        var overlap = dist + r - cut;
        if (overlap <= 0) return;

        // normal apontando RADIALMENTE PRA FORA (do centro pra entity)
        var nx = dx / dist;
        var ny = dy / dist;

        // push: empurrar entity PRA DENTRO (na direção -nx, -ny)
        corrections[eid][0] -= nx * overlap;
        corrections[eid][1] -= ny * overlap;

        // bounce: refletir velocidade radial
        var vn = entity.vx * nx + entity.vy * ny;
        if (vn > 0) {  // entity se afastando do centro = pra parede
            corrections[eid][2] -= vn * (1 + WALL_BOUNCE) * nx;
            corrections[eid][3] -= vn * (1 + WALL_BOUNCE) * ny;
        }
    }

    function resolvePostCollisions(entity, state, corrections) {
        // colisão círculo-círculo contra postes estáticos (massa infinita).
        // paridade 1:1 com backend/physics/collision.py::_resolve_post_collisions.
        var r = entity.radius;
        var eid = entity.id;
        var posts = state.posts || [];
        for (var i = 0; i < posts.length; i++) {
            var post = posts[i];
            var dx = entity.x - post.x;
            var dy = entity.y - post.y;
            var dist = Math.sqrt(dx * dx + dy * dy);
            var minDist = r + post.radius;
            if (dist >= minDist || dist === 0) continue;
            var nx = dx / dist;
            var ny = dy / dist;
            var overlap = minDist - dist;
            corrections[eid][0] += nx * overlap;
            corrections[eid][1] += ny * overlap;
            var vn = entity.vx * nx + entity.vy * ny;
            if (vn < 0) {
                corrections[eid][2] += -vn * (1 + POST_BOUNCE) * nx;
                corrections[eid][3] += -vn * (1 + POST_BOUNCE) * ny;
            }
        }
    }

    // physics Step
    function physicsStep(state) {
        state.goal_scored_this_step = false;
        state.scoring_team = null;
        var i, p;

        // 1. Velocidades dos jogadores
        for (i = 0; i < state.players.length; i++) {
            p = state.players[i];
            p.vx += Math.cos(p.angle) * p.accel * ACCEL_FORCE;
            p.vy += Math.sin(p.angle) * p.accel * ACCEL_FORCE;
            p.angle += p.rot * ROTATION_SPEED;
            p.angle = p.angle % (2 * Math.PI);
            if (p.angle < 0) p.angle += 2 * Math.PI;
            p.vx *= FRICTION_PLAYER;
            p.vy *= FRICTION_PLAYER;
            clampSpeed(p, MAX_SPEED_PLAYER);
        }

        // 2. Fricção da bola
        state.ball.vx *= FRICTION_BALL;
        state.ball.vy *= FRICTION_BALL;
        clampSpeed(state.ball, MAX_SPEED_BALL);

        // 3. Chutes
        for (i = 0; i < state.players.length; i++) {
            p = state.players[i];
            if (p.kick_cooldown_timer > 0) {
                p.kick_cooldown_timer -= 1;
                if (p.kick_cooldown_timer === 0) p.can_kick = true;
                if (p.kick_cooldown_timer <= KICK_COOLDOWN - 6) p.is_kicking = false;
            }
            p.kick_attempted = p.kick_requested && p.can_kick;
            if (p.kick_requested) tryKick(p, state.ball, state);
        }

        // 4. Posições
        for (i = 0; i < state.players.length; i++) {
            p = state.players[i];
            p.x += p.vx; p.y += p.vy;
        }
        state.ball.x += state.ball.vx;
        state.ball.y += state.ball.vy;

        // 5. Colisões
        resolveAllCollisions(state);

        // 6. Gol
        checkGoal(state);

        state.step_count += 1;
    }

    // raycasting
    function raySegmentIntersect(ox, oy, dx, dy, wall) {
        var ax = wall.x1, ay = wall.y1;
        var sx = wall.x2 - ax, sy = wall.y2 - ay;
        var denom = dx * sy - dy * sx;
        if (Math.abs(denom) < 1e-10) return Infinity;
        var t = ((ax - ox) * sy - (ay - oy) * sx) / denom;
        var s = ((ax - ox) * dy - (ay - oy) * dx) / denom;
        if (t > 0 && s >= 0 && s <= 1) return t;
        return Infinity;
    }

    function rayCircleIntersect(ox, oy, dx, dy, cx, cy, r) {
        var fx = ox - cx, fy = oy - cy;
        var a = dx * dx + dy * dy;
        var b = 2 * (fx * dx + fy * dy);
        var c = fx * fx + fy * fy - r * r;
        var disc = b * b - 4 * a * c;
        if (disc < 0) return Infinity;
        var sqrtDisc = Math.sqrt(disc);
        var t1 = (-b - sqrtDisc) / (2 * a);
        var t2 = (-b + sqrtDisc) / (2 * a);
        if (t1 > 0) return t1;
        if (t2 > 0) return t2;
        return Infinity;
    }

    function castRays(player, state) {
        var results = [];
        var px = player.x, py = player.y;

        for (var ri = 0; ri < RAY_ANGLES.length; ri++) {
            var rayAngle = player.angle + RAY_ANGLES[ri];
            var dx = Math.cos(rayAngle);
            var dy = Math.sin(rayAngle);

            var closestDist = MAX_RAY_DISTANCE;
            var closestType = "none";

            // paredes
            for (var wi = 0; wi < state.walls.length; wi++) {
                var d = raySegmentIntersect(px, py, dx, dy, state.walls[wi]);
                if (d > 0 && d < closestDist) { closestDist = d; closestType = "wall"; }
            }

            // postes (detectados como "wall" — TYPE_MAP não muda)
            var posts = state.posts || [];
            for (var pIdx = 0; pIdx < posts.length; pIdx++) {
                var post = posts[pIdx];
                var d = rayCircleIntersect(px, py, dx, dy, post.x, post.y, post.radius);
                if (d > 0 && d < closestDist) { closestDist = d; closestType = "wall"; }
            }

            // gols
            for (var gi = 0; gi < state.goals.length; gi++) {
                var goal = state.goals[gi];
                var gx = goal.side === "left" ? 0 : state.field.width;
                var gWall = { x1: gx, y1: goal.y_min, x2: gx, y2: goal.y_max };
                var d = raySegmentIntersect(px, py, dx, dy, gWall);
                if (d > 0 && d < closestDist) {
                    closestDist = d;
                    closestType = goal.team === player.team ? "goal_own" : "goal_enemy";
                }
            }

            // bola
            var d = rayCircleIntersect(px, py, dx, dy,
                state.ball.x, state.ball.y, state.ball.radius);
            if (d > 0 && d < closestDist) { closestDist = d; closestType = "ball"; }

            // outros jogadores
            for (var pi = 0; pi < state.players.length; pi++) {
                var other = state.players[pi];
                if (other.id === player.id) continue;
                var d = rayCircleIntersect(px, py, dx, dy,
                    other.x, other.y, other.radius);
                if (d > 0 && d < closestDist) {
                    closestDist = d;
                    closestType = other.team === player.team ? "ally" : "enemy";
                }
            }

            results.push(closestDist / MAX_RAY_DISTANCE);
            results.push(TYPE_MAP[closestType]);
        }
        return results;
    }

    // inputs (gather_inputs)
    function codeToType(code) {
        if (code < -0.5) return "none";
        if (code < 0.1) return "wall";
        if (code < 0.3) return "ball";
        if (code < 0.5) return "ally";
        if (code < 0.7) return "enemy";
        if (code < 0.9) return "goal_own";
        return "goal_enemy";
    }

    function gatherInputs(player, state) {
        var rayData = castRays(player, state);

        var best = {};
        var wallMinDist = 1.0;

        for (var i = 0; i < RAY_ANGLES.length; i++) {
            var dist = rayData[i * 2];
            var typeCode = rayData[i * 2 + 1];
            var angle = RAY_ANGLES[i];
            var typeName = codeToType(typeCode);

            if (typeName === "wall") {
                if (dist < wallMinDist) wallMinDist = dist;
            } else if (typeName !== "none") {
                if (!best[typeName] || dist < best[typeName][1]) {
                    best[typeName] = [angle / Math.PI, dist];
                }
            }
        }

        function getFeature(name) {
            // (0.0, -1.0) sinaliza "não detectado" (distingue de "detectado longe" = 1.0)
            // paridade com backend/ai/inputs.py e backend/ai/environment.py (Bug 9).
            return best[name] || [0.0, -1.0];
        }

        var ball = getFeature("ball");
        var goalEnemy = getFeature("goal_enemy");
        var goalOwn = getFeature("goal_own");
        var enemy = getFeature("enemy");

        var speed = Math.min(Math.sqrt(player.vx * player.vx + player.vy * player.vy) / MAX_SPEED_PLAYER, 1.0);
        var angularVel = player.rot || 0;
        var canKick = player.can_kick ? 1.0 : 0.0;

        var dxComp = 0, dyComp = 0;
        for (var pi = 0; pi < state.players.length; pi++) {
            var other = state.players[pi];
            if (other.id !== player.id && other.team === player.team) {
                dxComp = Math.max(-1, Math.min(1, (other.x - player.x) / FIELD_WIDTH));
                dyComp = Math.max(-1, Math.min(1, (other.y - player.y) / FIELD_HEIGHT));
                break;
            }
        }

        return [
            ball[0], ball[1],
            goalEnemy[0], goalEnemy[1],
            goalOwn[0], goalOwn[1],
            enemy[0], enemy[1],
            wallMinDist,
            speed, angularVel, canKick,
            dxComp, dyComp
        ];
    }

    // API pública
    return {
        FIELD_WIDTH: FIELD_WIDTH,
        FIELD_HEIGHT: FIELD_HEIGHT,
        PLAYER_RADIUS: PLAYER_RADIUS,
        BALL_RADIUS: BALL_RADIUS,
        MAX_RAY_DISTANCE: MAX_RAY_DISTANCE,
        KICK_COOLDOWN: KICK_COOLDOWN,
        PHYSICS_FPS: PHYSICS_FPS,
        RAY_ANGLES: RAY_ANGLES,
        TYPE_MAP: TYPE_MAP,
        createGameState: createGameState,
        randomizeSpawns: randomizeSpawns,
        spawnDeterministic: spawnDeterministic,
        applyControls: applyControls,
        physicsStep: physicsStep,
        castRays: castRays,
        gatherInputs: gatherInputs,
        clampSpeed: clampSpeed
    };

})();
