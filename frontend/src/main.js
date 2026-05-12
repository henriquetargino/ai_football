/**
 * entry point — menu, 3 modos de jogo, loop de renderização.
 *
 * v8 changes:
 *   - Manifest schema v2: lê `snapshots: [{step, label, policy_path, replay_path}]`
 *     em vez de `models` + `replays` separados. Frontend novo só lê v2 — runs
 *     antigos (v1) mostram mensagem "snapshot not available".
 *   - Goal width selectors removidos. Live games usam goal_width fixo (160).
 *   - AI vs AI Live: 2 dropdowns iguais (escolhe X timesteps vs Y timesteps).
 *   - Banner Three.js anima enquanto o menu está visível.
 */

(function() {
    var sceneData;
    var ballMesh;
    var robots = {};
    var replayPlayer;
    var clock;
    var currentMode = null; // "replay" | "human" | "aivai" | null (menu)
    var manifest = null;
    var bannerScene = null;
    var scoreboard = null;       // V9: placar 3D imersivo (substitui #hud)

    // Goal width fixo — selectors removidos no V8. 160 = "normal" do antigo.
    var FIXED_GOAL_WIDTH = 160;

    // live game state
    var liveState = null;
    var liveNets = {};
    var liveHumanTeam = null;
    var liveDuration = 0;
    var liveElapsed = 0;
    var liveRunning = false;
    var liveLastConfig = null;
    var liveTimeAccumulator = 0;
    var PHYSICS_DT = 1 / 60;

    // Goal detection state — rastreia mudança de placar pra disparar
    // partículas de comemoração apenas no momento do gol.
    var liveLastScore = { red: 0, blue: 0 };
    var replayLastGoalIdx = -1;

    // pós-gol: mantém celebration_active=true por GOAL_CELEBRATION_S antes
    // do respawn manual, dando tempo do efeito visual rodar.
    var GOAL_CELEBRATION_S = 3.0;
    var liveCelebrationEndAt = -1;       // liveElapsed quando termina (ou <0 se inativo)
    var replayLastCelebrateAt = -Infinity;   // performance.now/1000 do último celebrate
    // set de times explodidos (controles zerados + mesh invisível). Set
    // (não string única) pra cobrir casos extremos: replay fast-forward
    // que cruza múltiplos eventos de gol, ou modificações futuras que
    // permitam celebrações simultâneas. Reset em cleanup + fim de celebração.
    var concededTeams = {};              // {red: true, blue: true} subset

    /** True se o player do `team` está com mesh escondido + cubinhos
     *  espalhados (sofreu gol e a celebração ainda não acabou). Wrapper
     *  pra leitura clara nos call-sites. */
    function isShattered(team) {
        return liveState && liveState.celebration_active
            && concededTeams[team] === true;
    }

    // init
    function init() {
        sceneData = AISoccer.createScene();
        // exposição pra debug/preview headless (rAF não dispara em test runners,
        // permite forçar render manual: AISoccer._sceneData.renderer.render(...))
        AISoccer._sceneData = sceneData;
        clock = new THREE.Clock();

        // banner com cena Three.js separada (estática, render-on-demand).
        var bannerCanvas = document.getElementById('banner-canvas');
        if (bannerCanvas && AISoccer.createBannerScene) {
            try {
                bannerScene = AISoccer.createBannerScene(bannerCanvas);
            } catch (e) {
                console.warn('Banner scene failed:', e);
            }
        }

        loadManifest();
        setupMenuHandlers();
        setupGlobalKeys();

        // loop de animação ativo desde o início, mas faz no-op enquanto
        // currentMode === null (ver early return em animate()). Custo: ~60
        // chamadas/seg vazias — desprezível. Banner é render-on-demand
        // (renderiza 1x no createBannerScene, depois nada).
        animate();
    }

    // manifest (schema v2 = V8)
    function loadManifest() {
        fetch('data/manifest.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                manifest = data;
                populateDropdowns();
            })
            .catch(function(err) {
                console.warn('Manifest fetch failed:', err);
                manifest = { schema_version: 2, snapshots: [] };
                populateDropdowns();
            });
    }

    function getSnapshots() {
        if (!manifest) return [];
        if (manifest.schema_version === 2 && Array.isArray(manifest.snapshots)) {
            return manifest.snapshots;
        }
        // schema v1 antigo (V7) — não suportado. Retorna [] pra dar feedback no UI.
        return [];
    }

    function populateDropdowns() {
        var snapshots = getSnapshots();

        var selReplay = document.getElementById('sel-replay');
        var selHuman = document.getElementById('sel-human-snapshot');
        var selRed = document.getElementById('sel-ai-red');
        var selBlue = document.getElementById('sel-ai-blue');

        if (!snapshots.length) {
            var emptyMsg = (manifest && manifest.schema_version !== 2)
                ? 'Old manifest (re-publish needed)'
                : 'No snapshots available';
            [selReplay, selHuman, selRed, selBlue].forEach(function(s) {
                s.innerHTML = '<option value="">' + emptyMsg + '</option>';
            });
            return;
        }

        // ordem mais recente → mais antigo (final no topo).
        var ordered = snapshots.slice().sort(function(a, b) { return b.step - a.step; });

        // Replay dropdown — V9.2: divide em 2 selects. Primary lista 1
        // entry por fase (F0, F1A, ..., F4 + 1 entry "F5" gateway). Quando
        // f5 é selecionado, o sub-dropdown #sel-replay-f5 aparece com TODOS
        // os checkpoints intermediários da fase 5.
        var selReplayF5 = document.getElementById('sel-replay-f5');
        var byStepAsc = snapshots.slice().sort(function(a, b) { return a.step - b.step; });
        var withReplay = byStepAsc.filter(function(s) { return !!s.replay_path; });

        if (withReplay.length === 0) {
            selReplay.innerHTML = '<option value="">No replays in this run</option>';
        } else {
            // separa: snapshots em F5 (vão pro sub-dropdown) vs demais.
            var nonF5 = withReplay.filter(function(s) { return s.phase !== "5"; });
            var f5 = withReplay.filter(function(s) { return s.phase === "5"; });

            var primaryOpts = nonF5.map(function(s) {
                return '<option value="' + s.replay_path + '">' + s.label + '</option>';
            });
            // v9.2 — entry "gateway" pra F5 quando há ≥1 checkpoint dentro dela.
            // valor mágico "__F5__" sinaliza pro startReplay() ler o sub-dropdown.
            if (f5.length > 0) {
                var lastF5Label = f5[f5.length - 1].label;
                // "F5 · 50M (final)" → "F5 · choose timestep…"
                var f5Header = lastF5Label.split(" · ")[0] + " · choose timestep…";
                primaryOpts.push('<option value="__F5__">' + f5Header + '</option>');
            }
            selReplay.innerHTML = primaryOpts.join('');

            // sub-dropdown: opções apenas dos checkpoints F5 (em ordem
            // cronológica — last item é o " (final)").
            if (f5.length > 0) {
                selReplayF5.innerHTML = f5.map(function(s) {
                    return '<option value="' + s.replay_path + '">' + s.label + '</option>';
                }).join('');
                // default: último (= final)
                selReplayF5.selectedIndex = f5.length - 1;
            }

            // toggle do sub-dropdown baseado na escolha do primary.
            selReplay.addEventListener('change', function() {
                if (selReplay.value === '__F5__') {
                    selReplayF5.classList.remove('hidden');
                } else {
                    selReplayF5.classList.add('hidden');
                }
            });

            document.getElementById('btn-start-replay').disabled = false;
        }

        // human + AI dropdowns — só snapshots COM policy_path (F0 random
        // não tem — ele só existe pro Watch Replay).
        var withPolicy = ordered.filter(function(s) { return !!s.policy_path; });
        var policyOpts = withPolicy.map(function(s) {
            return '<option value="' + s.policy_path + '" data-step="' + s.step + '">' + s.label + '</option>';
        }).join('');

        selHuman.innerHTML = policyOpts;
        if (withPolicy.length > 0) {
            document.getElementById('btn-start-human').disabled = false;
        }

        // AI vs AI: red default = mais novo (final), blue default = mais antigo
        // (cria um "newer vs older" interessante por padrão).
        selRed.innerHTML = policyOpts;
        selBlue.innerHTML = policyOpts;
        if (withPolicy.length >= 2) {
            // selBlue padrão = mais antigo (último em withPolicy ordenado desc)
            selBlue.value = withPolicy[withPolicy.length - 1].policy_path;
        }
        if (withPolicy.length > 0) {
            document.getElementById('btn-start-aivai').disabled = false;
        }
    }

    // menu handlers
    function setupMenuHandlers() {
        document.getElementById('btn-start-replay').addEventListener('click', startReplay);
        document.getElementById('btn-start-human').addEventListener('click', startHumanVsAI);
        document.getElementById('btn-start-aivai').addEventListener('click', startAIvsAI);
        document.getElementById('btn-menu').addEventListener('click', returnToMenu);
        document.getElementById('btn-menu-live').addEventListener('click', returnToMenu);
        document.getElementById('btn-go-menu').addEventListener('click', returnToMenu);
        document.getElementById('btn-go-again').addEventListener('click', playAgain);
        document.getElementById('btn-rays-live').addEventListener('click', function() {
            var vis = AISoccer.toggleRaycasts();
            this.classList.toggle('active', vis);
        });
        // 🧠 Neural network viz toggle — apenas live (Human×AI e AI×AI).
        // no replay esse botão não existe (não faz sentido visualizar
        // inferência ao vivo num replay já gravado).
        var btnNNLive = document.getElementById('btn-nn-live');
        if (btnNNLive) btnNNLive.addEventListener('click', function() {
            var vis = AISoccer.toggleNeuralViz();
            this.classList.toggle('active', vis);
        });

        // pill toggle (Red / Blue) — Human vs AI side selector.
        // atualiza o hidden input #sel-human-team mantendo a API de
        // startHumanVsAI() inalterada (ainda lê .value desse input).
        var pillContainer = document.getElementById('pill-side');
        if (pillContainer) {
            var pills = pillContainer.querySelectorAll('button[data-side]');
            pills.forEach(function(p) {
                p.addEventListener('click', function() {
                    pills.forEach(function(b) { b.classList.remove('active'); });
                    p.classList.add('active');
                    document.getElementById('sel-human-team').value = p.dataset.side;
                });
            });
        }
    }

    function setupGlobalKeys() {
        window.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                // prioridade: se o painel da rede neural estiver aberto,
                // ESC fecha SÓ ele (não volta pro lobby). Segundo ESC volta.
                if (AISoccer.isNeuralVizVisible && AISoccer.isNeuralVizVisible()) {
                    AISoccer.toggleNeuralViz();
                    var btnLive = document.getElementById('btn-nn-live');
                    if (btnLive) btnLive.classList.remove('active');
                    return;
                }
                returnToMenu();
                return;
            }
            if ((e.key === 'r' || e.key === 'R') && currentMode) {
                var vis = AISoccer.toggleRaycasts();
                var btnId = currentMode === 'replay' ? 'btn-rays' : 'btn-rays-live';
                var btn = document.getElementById(btnId);
                if (btn) btn.classList.toggle('active', vis);
                return;
            }
            // atalho N: View NN — disponível apenas nos modos live (Human×AI
            // e AI×AI). No replay o painel da rede não faz sentido (a rede
            // do replay já está consolidada em frames; não há inferência ao
            // vivo pra visualizar). Pra evitar inconsistência com a UI, o
            // atalho também não funciona em replay.
            if ((e.key === 'n' || e.key === 'N') &&
                (currentMode === 'human' || currentMode === 'aivai')) {
                var nVis = AISoccer.toggleNeuralViz();
                var btnN = document.getElementById('btn-nn-live');
                if (btnN) btnN.classList.toggle('active', nVis);
                return;
            }
            if (e.key === ' ' && currentMode === 'replay') {
                e.preventDefault();
                var rp = AISoccer._activeReplayPlayer;
                if (rp) { if (rp.playing) rp.pause(); else rp.play(); }
            }
        });
    }

    function hideAll() {
        document.getElementById('menu').classList.add('hidden');
        document.getElementById('hud').classList.add('hidden');
        document.getElementById('controls').classList.add('hidden');
        document.getElementById('live-controls').classList.add('hidden');
        document.getElementById('game-over').classList.add('hidden');
    }

    function returnToMenu() {
        hideAll();
        cleanupCurrentMode();
        document.getElementById('menu').classList.remove('hidden');
        currentMode = null;
    }

    function cleanupCurrentMode() {
        if (replayPlayer) {
            replayPlayer = null;
            AISoccer._activeReplayPlayer = null;
        }
        if (liveRunning) {
            liveRunning = false;
            AISoccer.HumanInput.destroy();
        }
        // fecha o painel da rede neural se estiver aberto (e restaura
        // canvas Three.js pra largura total).
        if (AISoccer.isNeuralVizVisible && AISoccer.isNeuralVizVisible()) {
            AISoccer.toggleNeuralViz();
        }
        var btnNN = document.getElementById('btn-nn-live');
        if (btnNN) btnNN.classList.remove('active');
        // reset do state interno de raycasts ANTES de limpar o botão.
        // bug original: `visible` é variável de módulo em raycasts.js, fica
        // stale entre modos — entrar num replay depois de ter ligado os
        // raycasts num live deixava o linesMesh visível com o botão fresh
        // sem `.active`. Idempotente: se já estava off, no-op.
        if (AISoccer.isRaycastsVisible && AISoccer.isRaycastsVisible()) {
            AISoccer.toggleRaycasts();
        }
        var btnRaysReplay = document.getElementById('btn-rays');
        if (btnRaysReplay) btnRaysReplay.classList.remove('active');
        var btnRaysLive = document.getElementById('btn-rays-live');
        if (btnRaysLive) btnRaysLive.classList.remove('active');
        scoreboard = null;       // o group será removido em clearStadium()
        // reset detector de gol + libera partículas (objetos removidos em
        // clearStadium, mas a referência precisa ir a null pra recriar).
        liveLastScore = { red: 0, blue: 0 };
        replayLastGoalIdx = -1;
        liveCelebrationEndAt = -1;
        replayLastCelebrateAt = -Infinity;
        concededTeams = {};
        hideGoalText();
        if (AISoccer.disposeGoalParticles) AISoccer.disposeGoalParticles();
        if (AISoccer.disposeRobotShatter) AISoccer.disposeRobotShatter();
        clearStadium();
    }

    function clearStadium() {
        if (!sceneData) return;
        var scene = sceneData.scene;
        var toRemove = [];
        scene.traverse(function(obj) {
            if (obj.isMesh || obj.isLineSegments || obj.isGroup || obj.isPoints) {
                toRemove.push(obj);
            }
            if (obj.isLight && obj.userData && obj.userData.isFloodlight) {
                toRemove.push(obj);
            }
            if (!obj.isLight && !obj.isMesh && obj.userData && obj.userData.isFloodlight) {
                toRemove.push(obj);
            }
        });
        for (var i = 0; i < toRemove.length; i++) {
            var obj = toRemove[i];
            if (obj.parent) obj.parent.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (obj.material.map) obj.material.map.dispose();
                obj.material.dispose();
            }
            if (obj.shadow && obj.shadow.map) obj.shadow.map.dispose();
        }
        robots = {};
        ballMesh = null;
    }

    // mode A: Watch Replay
    function buildStadium(gw) {
        AISoccer.createField(sceneData.scene);
        AISoccer.createWalls(sceneData.scene, gw);
        AISoccer.createGoals(sceneData.scene, gw);
        AISoccer.createStands(sceneData.scene);
        // 4 arquibancadas curvas nos cantos — fecham o estádio. Mesma altura
        // (6 fileiras) das stands principais, sem alterar nada além disso.
        if (AISoccer._buildCornerStands) AISoccer._buildCornerStands(sceneData.scene);
        AISoccer.createFloodlights(sceneData.scene);
        ballMesh = AISoccer.createBall(sceneData.scene);
        AISoccer.createRaycastVisuals(sceneData.scene);
        // placar manual 3D (substitui o HUD HTML flutuante).
        if (AISoccer.createScoreboard) {
            scoreboard = AISoccer.createScoreboard(sceneData.scene);
            AISoccer._scoreboard = scoreboard;       // exposto pra debug/preview
        }
        // canhões pretos cilíndricos (decoração permanente, atrás de cada
        // gol — 4 no total). Disparam jato vertical de fogo no celebrateGoal.
        if (AISoccer.createGoalCannons) {
            AISoccer.createGoalCannons(sceneData.scene);
        }
        if (AISoccer.createGoalParticles) {
            AISoccer.createGoalParticles(sceneData.scene);
        }
        // pool de cubinhos pra "shatter" do robô que sofre o gol.
        if (AISoccer.createRobotShatter) {
            AISoccer.createRobotShatter(sceneData.scene);
        }
    }

    function startReplay() {
        var sel = document.getElementById('sel-replay');
        var path = sel.value;
        // v9.2: gateway "__F5__" → ler do sub-dropdown F5.
        if (path === '__F5__') {
            var selF5 = document.getElementById('sel-replay-f5');
            path = selF5.value;
        }
        if (!path) return;

        hideAll();
        currentMode = "replay";

        // reset do detector de gol pra esse replay
        replayLastGoalIdx = -1;

        replayPlayer = new AISoccer.ReplayPlayer();
        replayPlayer.load(path, function(data) {
            // v8: replays padronizados — goal_width herdado do treino (DR final ~160-200).
            // lê do metadata se disponível, senão usa o fixo.
            var gw = (data.metadata && data.metadata.goal_width) || FIXED_GOAL_WIDTH;
            // replays V8 não gravam goal_width na metadata (capture flat),
            // então `gw` usa frame-level se existir.
            if (data.frames && data.frames.length && data.frames[0].goal_width) {
                gw = data.frames[0].goal_width;
            }
            buildStadium(gw);

            var players = data.frames[0].players;
            for (var i = 0; i < players.length; i++) {
                var p = players[i];
                var team = p.id.indexOf('red') >= 0 ? 'red' : 'blue';
                robots[p.id] = AISoccer.createRobot(sceneData.scene, team);
            }

            // label do snapshot — pega do sub-dropdown F5 quando aplicável.
            var optLabel;
            if (sel.value === '__F5__') {
                var sf5 = document.getElementById('sel-replay-f5');
                optLabel = sf5.options[sf5.selectedIndex] ? sf5.options[sf5.selectedIndex].text : '';
            } else {
                optLabel = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : '';
            }
            if (scoreboard) scoreboard.updateSnapshot(optLabel);
            document.getElementById('controls').classList.remove('hidden');

            AISoccer.setupControls(replayPlayer);
            replayPlayer.play();
            startAnimationLoop();
        });
    }

    // mode B: Human vs AI
    function startHumanVsAI() {
        var modelPath = document.getElementById('sel-human-snapshot').value;
        if (!modelPath) return;

        liveHumanTeam = document.getElementById('sel-human-team').value;
        currentMode = "human";

        liveLastConfig = {
            modelPath: modelPath,
            humanTeam: liveHumanTeam,
        };

        hideAll();

        fetch(modelPath)
            .then(function(r) { return r.json(); })
            .then(function(modelData) {
                var aiTeam = liveHumanTeam === "red" ? "blue" : "red";
                liveNets = {};
                liveNets[aiTeam] = new AISoccer.MlpNetwork(modelData);

                startLiveGame();
                AISoccer.HumanInput.init();

                var sel = document.getElementById('sel-human-snapshot');
                var label = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : '?';
                if (scoreboard) scoreboard.updateSnapshot('You vs ' + label);
                startAnimationLoop();
            });
    }

    // mode C: AI vs AI Live
    function startAIvsAI() {
        var redPath = document.getElementById('sel-ai-red').value;
        var bluePath = document.getElementById('sel-ai-blue').value;
        if (!redPath || !bluePath) return;

        liveHumanTeam = null;
        currentMode = "aivai";

        liveLastConfig = {
            redPath: redPath,
            bluePath: bluePath,
        };

        hideAll();

        Promise.all([
            fetch(redPath).then(function(r) { return r.json(); }),
            fetch(bluePath).then(function(r) { return r.json(); })
        ]).then(function(results) {
            liveNets = {
                red: new AISoccer.MlpNetwork(results[0]),
                blue: new AISoccer.MlpNetwork(results[1])
            };
            startLiveGame();

            var redSel = document.getElementById('sel-ai-red');
            var blueSel = document.getElementById('sel-ai-blue');
            var redLbl = redSel.options[redSel.selectedIndex] ? redSel.options[redSel.selectedIndex].text : '?';
            var blueLbl = blueSel.options[blueSel.selectedIndex] ? blueSel.options[blueSel.selectedIndex].text : '?';
            if (scoreboard) scoreboard.updateSnapshot(redLbl + ' vs ' + blueLbl);
            startAnimationLoop();
        });
    }

    // live game (shared by B and C)
    /**
     * spawn FAIR aleatório pra AI vs AI Live.
     *
     * problema que resolve: como ambas as policies são iguais (ou muito
     * próximas) e roda greedy, posições simétricas produzem comportamento
     * ESPELHADO — toda partida é a mesma luta no centro, com o mesmo lado
     * sempre vencendo (artefato de assimetria do engine).
     *
     * solução: posições aleatórias MAS justas — bola no centro, cada robô
     * a uma distância IGUAL da bola, em ângulos aleatórios INDEPENDENTES
     * dentro do meio campo dele. Mesmas chances de chegar primeiro, mas
     * o ataque inicial nunca é simétrico → variabilidade real.
     */
    function spawnFairRandom(state) {
        var FW = AISoccer.FIELD_WIDTH;
        var FH = AISoccer.FIELD_HEIGHT;
        var midX = FW / 2;
        var midY = FH / 2;

        // bola no centro
        state.ball.x = midX;
        state.ball.y = midY;
        state.ball.vx = 0;
        state.ball.vy = 0;

        // distância igual da bola pros dois (justa). Aleatória 150-220u.
        var dist = 150 + Math.random() * 70;

        // cada robô em ângulo INDEPENDENTE dentro do MEIO CAMPO dele:
        //   red ataca direita: meio campo dele = lado esquerdo (x < midX)
        //     → ângulo no lado esquerdo da bola: π ± π/3 (60° de cone)
        //   blue ataca esquerda: meio campo dele = lado direito (x > midX)
        //     → ângulo: 0 ± π/3
        var redAngle  = Math.PI + (Math.random() - 0.5) * (Math.PI * 2 / 3);
        var blueAngle =           (Math.random() - 0.5) * (Math.PI * 2 / 3);

        for (var i = 0; i < state.players.length; i++) {
            var p = state.players[i];
            var ang = (p.team === "red") ? redAngle : blueAngle;
            p.x = midX + Math.cos(ang) * dist;
            p.y = midY + Math.sin(ang) * dist;
            // olhando pra bola
            p.angle = Math.atan2(midY - p.y, midX - p.x);
            p.vx = 0;
            p.vy = 0;
            if (p.kick_cooldown_timer !== undefined) p.kick_cooldown_timer = 0;
            if (p.is_kicking !== undefined) p.is_kicking = false;
        }
    }

    function startLiveGame() {
        buildStadium(FIXED_GOAL_WIDTH);
        liveState = AISoccer.Physics.createGameState("1v1", FIXED_GOAL_WIDTH);

        // spawn:
        //   - Human vs AI: SEMPRE deterministic (jogador precisa de posição
        //     consistente pra controlar).
        //   - AI vs AI Live: spawn FAIR aleatório — quebra simetria espelho
        //     (mesma policy nos 2 lados → comportamento idêntico em greedy)
        //     mas mantém mesma distância da bola pros 2 (justo).
        if (currentMode === "aivai") {
            spawnFairRandom(liveState);
        } else {
            liveState.deterministic_spawn = true;
            AISoccer.Physics.spawnDeterministic(liveState);
        }

        for (var i = 0; i < liveState.players.length; i++) {
            var p = liveState.players[i];
            robots[p.id] = AISoccer.createRobot(sceneData.scene, p.team);
        }

        // duração fixa: 60s (sem variar por goal width agora).
        liveDuration = 60;
        liveElapsed = 0;
        liveTimeAccumulator = 0;
        liveRunning = true;

        // reset do detector de gol pra essa partida (defensivo: se o
        // jogador saiu mid-celebração e voltou, garantir que a flag não
        // ficou "presa").
        liveLastScore = { red: 0, blue: 0 };
        liveCelebrationEndAt = -1;
        concededTeams = {};

        if (scoreboard) {
            scoreboard.updateScore(0, 0);
            scoreboard.updateTimer(0);
        }

        // cor do painel Neural Viz pela cor do time da IA.
        // human vs AI: IA é o oposto do humano. AI vs AI: usa RED por convenção.
        if (AISoccer.setNeuralVizTeam) {
            var aiVizTeam;
            if (currentMode === "human") {
                aiVizTeam = (liveHumanTeam === "red") ? "blue" : "red";
            } else {
                aiVizTeam = "red";
            }
            AISoccer.setNeuralVizTeam(aiVizTeam);
        }

        document.getElementById('live-controls').classList.remove('hidden');
    }

    function playAgain() {
        hideAll();
        cleanupCurrentMode();

        if (liveLastConfig) {
            if (currentMode === "human" || liveLastConfig.humanTeam) {
                liveHumanTeam = liveLastConfig.humanTeam;
                currentMode = "human";

                fetch(liveLastConfig.modelPath)
                    .then(function(r) { return r.json(); })
                    .then(function(modelData) {
                        var aiTeam = liveHumanTeam === "red" ? "blue" : "red";
                        liveNets = {};
                        liveNets[aiTeam] = new AISoccer.MlpNetwork(modelData);
                        startLiveGame();
                        AISoccer.HumanInput.init();
                        startAnimationLoop();
                    });
            } else {
                liveHumanTeam = null;
                currentMode = "aivai";

                Promise.all([
                    fetch(liveLastConfig.redPath).then(function(r) { return r.json(); }),
                    fetch(liveLastConfig.bluePath).then(function(r) { return r.json(); })
                ]).then(function(results) {
                    liveNets = {
                        red: new AISoccer.MlpNetwork(results[0]),
                        blue: new AISoccer.MlpNetwork(results[1])
                    };
                    startLiveGame();
                    startAnimationLoop();
                });
            }
        }
    }

    // animate
    function animate() {
        // rAF re-agendado SEMPRE no topo — garante que o loop NUNCA morre
        // (mesmo se algo lançar exceção no meio da função). O custo de um
        // rAF vazio é desprezível (~60 chamadas/seg no-op).
        requestAnimationFrame(animate);

        // PERFORMANCE: no menu, retorno aqui — sem update/render do estádio.
        // o banner já foi renderizado uma vez (render-on-demand) e mantém
        // sua imagem no canvas sem custo. Antes, o estádio do menu rodava
        // 3.5K meshes a 60fps no banner → 110% CPU.
        if (!currentMode || !sceneData) return;

        var dt = clock.getDelta();
        if (dt > 0.1) dt = 0.016;
        var time = performance.now() / 1000;

        if (currentMode === "replay" && replayPlayer) {
            updateReplayMode(dt, time);
        } else if ((currentMode === "human" || currentMode === "aivai") && liveRunning) {
            updateLiveMode(dt, time);
        }

        // avança partículas de gol + cubinhos do shatter independentemente
        // do modo — `dt` real garante animação suave mesmo em pause.
        if (AISoccer.updateGoalParticles) {
            AISoccer.updateGoalParticles(dt);
        }
        if (AISoccer.updateShatter) {
            AISoccer.updateShatter(dt);
        }
        // anima a "la ola" do GOAL no telão (no-op se não está em modo gol)
        if (scoreboard && scoreboard.tick) {
            scoreboard.tick();
        }

        sceneData.renderer.render(sceneData.scene, sceneData.camera);
    }

    // no-op (mantido pra compat com calls antigas — animate já está em loop).
    function startAnimationLoop() {}

    function updateReplayMode(dt, time) {
        replayPlayer.update(dt);
        var state = replayPlayer.getState();

        if (state) {
            var S = AISoccer.SCALE;

            ballMesh.position.x = state.ball.x * S;
            ballMesh.position.z = state.ball.y * S;

            if (ballMesh.userData.shadow) {
                ballMesh.userData.shadow.position.x = ballMesh.position.x;
                ballMesh.userData.shadow.position.z = ballMesh.position.z;
            }

            var ballSpeed = Math.sqrt(state.ball.vx * state.ball.vx + state.ball.vy * state.ball.vy);
            if (ballSpeed > 0.01) {
                var ballRadius3D = 12 * S;
                var distThisFrame = ballSpeed * S;
                var rollAngle = distThisFrame / ballRadius3D;
                var dirX = state.ball.vx / ballSpeed;
                var dirZ = state.ball.vy / ballSpeed;
                var q = new THREE.Quaternion();
                q.setFromAxisAngle(new THREE.Vector3(-dirZ, 0, dirX), rollAngle);
                ballMesh.quaternion.premultiply(q);
            }

            for (var i = 0; i < state.players.length; i++) {
                var p = state.players[i];
                if (robots[p.id]) {
                    AISoccer.updateRobot(robots[p.id], p, time);
                }
            }

            if (AISoccer.isRaycastsVisible() && AISoccer.Physics) {
                var physState = replayStateToPhysics(state);
                AISoccer.updateRaycasts(physState.players, physState);
            }

            updateHUD_replay(state);
        }

        if (AISoccer._updateControlsUI) AISoccer._updateControlsUI();
    }

    function replayStateToPhysics(state) {
        var gw = (replayPlayer && replayPlayer.metadata) ?
            (replayPlayer.metadata.goal_width || FIXED_GOAL_WIDTH) : FIXED_GOAL_WIDTH;
        var midY = 250;
        var half = gw / 2;
        return {
            field: { width: 800, height: 500, goal_width: gw },
            ball: { x: state.ball.x, y: state.ball.y, radius: 12 },
            players: state.players.map(function(p) {
                return {
                    id: p.id, team: p.id.indexOf('red') >= 0 ? 'red' : 'blue',
                    x: p.x, y: p.y, angle: p.angle, radius: 20
                };
            }),
            goals: [
                { team: "red", side: "left", y_min: midY - half, y_max: midY + half },
                { team: "blue", side: "right", y_min: midY - half, y_max: midY + half }
            ],
            walls: [
                {x1:0, y1:0, x2:0, y2:midY-half}, {x1:0, y1:midY+half, x2:0, y2:500},
                {x1:800, y1:0, x2:800, y2:midY-half}, {x1:800, y1:midY+half, x2:800, y2:500},
                {x1:0, y1:0, x2:800, y2:0}, {x1:0, y1:500, x2:800, y2:500}
            ]
        };
    }

    function updateLiveMode(dt, time) {
        if (!liveRunning) return;

        liveTimeAccumulator += dt;

        var stepsThisFrame = 0;
        while (liveTimeAccumulator >= PHYSICS_DT && stepsThisFrame < 4) {
            liveElapsed += PHYSICS_DT;

            if (liveElapsed >= liveDuration) {
                liveRunning = false;
                showGameOver();
                return;
            }

            // em human-vs-AI: capturar activations da IA (não do humano).
            // em AI vs AI: capturar do RED por convenção.
            var nnVizPlayer = null;
            if (currentMode === "human") {
                // a IA é o time oposto ao humano
                for (var pp = 0; pp < liveState.players.length; pp++) {
                    if (liveState.players[pp].team !== liveHumanTeam) {
                        nnVizPlayer = liveState.players[pp];
                        break;
                    }
                }
            } else {
                // AI vs AI: visualiza red
                for (var pp = 0; pp < liveState.players.length; pp++) {
                    if (liveState.players[pp].team === "red") {
                        nnVizPlayer = liveState.players[pp];
                        break;
                    }
                }
            }
            var nnVizActivations = null;

            for (var i = 0; i < liveState.players.length; i++) {
                var player = liveState.players[i];
                var outputs;
                // Player do time que SOFREU o gol fica "explodido" durante a
                // celebração — controles zerados (não chuta a bola "invisível",
                // não acelera, não rotaciona). Movimento natural por inércia
                // ainda continua até parar pelo damping.
                if (isShattered(player.team)) {
                    outputs = [0, 0, 0];
                } else if (currentMode === "human" && player.team === liveHumanTeam) {
                    outputs = AISoccer.HumanInput.getOutputs();
                } else {
                    var net = liveNets[player.team];
                    if (net) {
                        // se é o player escolhido pra viz E painel está visível,
                        // capturar activations completas (custo extra ~zero,
                        // mas só faz se ligado).
                        if (player === nnVizPlayer && AISoccer.isNeuralVizVisible &&
                            AISoccer.isNeuralVizVisible() && net.forwardWithActivations) {
                            var obsRaw = AISoccer.Obs.gatherObs(player, liveState, 0);
                            nnVizActivations = net.forwardWithActivations(obsRaw);
                            outputs = AISoccer.Actions.decodeAction(nnVizActivations.actionIdx);
                        } else {
                            outputs = net.activate(player, liveState);
                        }
                    } else {
                        outputs = [0, 0, 0];
                    }
                }
                AISoccer.Physics.applyControls(player, outputs);
            }

            // atualiza painel da rede neural (no-op se invisível)
            if (nnVizActivations && AISoccer.updateNeuralViz) {
                AISoccer.updateNeuralViz(nnVizActivations);
            }

            AISoccer.Physics.physicsStep(liveState);

            // detecta mudança de placar (gol acabou de acontecer). Quando
            // detecta: dispara canhões + texto GOAL + ativa celebration_active
            // pra suprimir re-detecção (bola fica off-field), e agenda
            // respawn manual depois de GOAL_CELEBRATION_S.
            var scorerJustNow = null;
            if (liveState.score.red > liveLastScore.red) {
                scorerJustNow = "red";
                liveLastScore.red = liveState.score.red;
            } else if (liveState.score.blue > liveLastScore.blue) {
                scorerJustNow = "blue";
                liveLastScore.blue = liveState.score.blue;
            }
            if (scorerJustNow) {
                if (AISoccer.celebrateGoal) AISoccer.celebrateGoal(scorerJustNow);
                showGoalText(scorerJustNow);
                // shatter do robô que SOFREU o gol (time oposto ao scorer).
                // esconde mesh e spawna cubinhos da mesma cor onde estava.
                var concedingTeam = scorerJustNow === "red" ? "blue" : "red";
                concededTeams[concedingTeam] = true;
                var S = AISoccer.SCALE;
                for (var pp = 0; pp < liveState.players.length; pp++) {
                    var pp_ = liveState.players[pp];
                    if (pp_.team === concedingTeam && robots[pp_.id]) {
                        if (AISoccer.shatterPlayer) {
                            AISoccer.shatterPlayer(concedingTeam, pp_.x * S, 0, pp_.y * S);
                        }
                        robots[pp_.id].visible = false;
                        // zera velocidade do player explodido (vira "fragmentos
                        // estáticos" — não desliza pelo campo invisível).
                        pp_.vx = 0; pp_.vy = 0;
                    }
                }
                liveState.celebration_active = true;
                liveCelebrationEndAt = liveElapsed + GOAL_CELEBRATION_S;
            }

            // fim da celebração: respawn deterministic/random (mesma escolha
            // que o checkGoal usaria), restore mesh dos robôs, libera
            // detecção de novos gols.
            if (liveCelebrationEndAt > 0 && liveElapsed >= liveCelebrationEndAt) {
                liveCelebrationEndAt = -1;
                liveState.celebration_active = false;
                if (liveState.deterministic_spawn) {
                    AISoccer.Physics.spawnDeterministic(liveState);
                } else {
                    AISoccer.Physics.randomizeSpawns(liveState);
                }
                // restaura visibilidade de TODOS os robôs (defensivo)
                for (var pr = 0; pr < liveState.players.length; pr++) {
                    var pr_ = liveState.players[pr];
                    if (robots[pr_.id]) robots[pr_.id].visible = true;
                }
                concededTeams = {};
                hideGoalText();
            }

            liveTimeAccumulator -= PHYSICS_DT;
            stepsThisFrame++;
        }

        var S = AISoccer.SCALE;

        ballMesh.position.x = liveState.ball.x * S;
        ballMesh.position.z = liveState.ball.y * S;
        if (ballMesh.userData.shadow) {
            ballMesh.userData.shadow.position.x = ballMesh.position.x;
            ballMesh.userData.shadow.position.z = ballMesh.position.z;
        }

        var bs = Math.sqrt(liveState.ball.vx * liveState.ball.vx +
                           liveState.ball.vy * liveState.ball.vy);
        if (bs > 0.01) {
            var ballRadius3D = 12 * S;
            var distThisFrame = bs * S;
            var ra = distThisFrame / ballRadius3D;
            var dirX = liveState.ball.vx / bs;
            var dirZ = liveState.ball.vy / bs;
            var q = new THREE.Quaternion();
            q.setFromAxisAngle(new THREE.Vector3(-dirZ, 0, dirX), ra);
            ballMesh.quaternion.premultiply(q);
        }

        for (var i = 0; i < liveState.players.length; i++) {
            var p = liveState.players[i];
            if (robots[p.id]) {
                AISoccer.updateRobot(robots[p.id], p, time);
            }
        }

        if (AISoccer.isRaycastsVisible()) {
            var rayPlayers = liveState.players;
            if (currentMode === "human" && liveHumanTeam) {
                rayPlayers = liveState.players.filter(function(p) {
                    return p.team !== liveHumanTeam;
                });
            }
            AISoccer.updateRaycasts(rayPlayers, liveState);
        }

        if (scoreboard) {
            scoreboard.updateScore(liveState.score.red, liveState.score.blue);
            scoreboard.updateTimer(Math.max(0, liveDuration - liveElapsed));
        }
    }

    // GOAL agora aparece NO TELÃO (scoreboard 3D) — não em full-screen.
    // estes wrappers preservam a API antiga (showGoalText/hideGoalText)
    // delegando pro scoreboard quando disponível. Mantido pra evitar
    // mudar 2 call sites (live + replay).
    function showGoalText(scorer) {
        if (scoreboard && scoreboard.showGoalMode) {
            scoreboard.showGoalMode(scorer);
        }
    }
    function hideGoalText() {
        if (scoreboard && scoreboard.hideGoalMode) {
            scoreboard.hideGoalMode();
        }
    }

    function showGameOver() {
        document.getElementById('go-red').textContent = liveState.score.red;
        document.getElementById('go-blue').textContent = liveState.score.blue;
        if (scoreboard) scoreboard.updateScore(liveState.score.red, liveState.score.blue);
        if (liveState.score.red > liveState.score.blue) {
            document.getElementById('go-title').textContent = 'Orange Wins!';
        } else if (liveState.score.blue > liveState.score.red) {
            document.getElementById('go-title').textContent = 'Blue Wins!';
        } else {
            document.getElementById('go-title').textContent = 'Draw!';
        }
        document.getElementById('game-over').classList.remove('hidden');
    }

    function updateHUD_replay(state) {
        if (!replayPlayer || !replayPlayer.metadata) return;

        // no replay, NÃO disparamos celebrações visuais (canhões, shatter,
        // texto GOAL, delay 3s) — esses efeitos são exclusivos dos modos
        // live (Human×AI e AI×AI). No replay queremos só ver a partida
        // gravada rodando sem pause artificial nem efeitos extras.
        // mantemos apenas a contagem de placar e atualização do timer.
        var goals = replayPlayer.goalEvents || [];
        var currentFrame = Math.floor(replayPlayer.currentTime * replayPlayer.replayFps);
        var scoreRed = 0, scoreBlue = 0;
        for (var i = 0; i < goals.length; i++) {
            if (goals[i].frameIdx > currentFrame) break;
            if (goals[i].scorer === 'red') scoreRed++;
            else scoreBlue++;
        }
        if (scoreboard) {
            scoreboard.updateScore(scoreRed, scoreBlue);
            scoreboard.updateTimer(replayPlayer.currentTime);
        }
    }

    init();
})();
