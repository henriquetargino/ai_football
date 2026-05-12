/**
 * placar manual 3D estilo "flip cards" — substitui o HUD HTML flutuante
 * por um scoreboard imersivo dentro da cena Three.js, montado na parede
 * NORTE do estádio em cima da linha de meio de campo.
 *
 * composição:
 *   ┌─────────────────────────────┐
 *   │       AI · SOCCER           │  ← header (texto branco)
 *   │   ┌────┐  X  ┌────┐         │
 *   │   │  0 │     │  0 │         │  ← flip cards (red / blue)
 *   │   └────┘     └────┘         │
 *   │     ┌──────────┐            │
 *   │     │  00:00   │            │  ← timer (digital amarelo)
 *   │     └──────────┘            │
 *   │       47.6M STEPS           │  ← footer (snapshot label)
 *   └─────────────────────────────┘
 *      ║                       ║
 *      ║  (2 polos de suporte) ║
 *      ╨                       ╨
 *
 * API:
 *   var sb = AISoccer.createScoreboard(scene);
 *   sb.updateScore(2, 1);
 *   sb.updateTimer(75);          // segundos → "01:15"
 *   sb.updateSnapshot('47.6M (final)');
 */

AISoccer.createScoreboard = function(scene) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;

    // ── Constantes de design ──
    var PANEL_W = 2.4;
    var PANEL_H = 1.10;     // espaço pra cards + timer + footer sem sobrepor
    var PANEL_D = 0.18;

    var COLOR_RED        = '#ff6b1a';
    var COLOR_BLUE       = '#2a8aff';
    var COLOR_TIMER      = '#ffd93d';
    var COLOR_TEXT_DIM   = '#aaaaaa';
    var COLOR_TEXT       = '#ffffff';
    var COLOR_PANEL_INNER = '#16161a';

    var group = new THREE.Group();

    // ─── Estrutura: caixa principal (corpo do scoreboard) ───
    var bodyMat = new THREE.MeshStandardMaterial({
        color: 0x0e0e12, roughness: 0.55, metalness: 0.30,
    });
    var body = new THREE.Mesh(
        new THREE.BoxGeometry(PANEL_W, PANEL_H, PANEL_D), bodyMat
    );
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    // moldura cinza-metálica (top + bottom) — feel de estádio antigo
    var frameMat = new THREE.MeshStandardMaterial({
        color: 0x2a2a30, roughness: 0.4, metalness: 0.55,
    });
    var frameTop = new THREE.Mesh(
        new THREE.BoxGeometry(PANEL_W + 0.10, 0.05, PANEL_D + 0.04), frameMat
    );
    frameTop.position.set(0, PANEL_H / 2 + 0.025, 0);
    group.add(frameTop);
    var frameBot = new THREE.Mesh(
        new THREE.BoxGeometry(PANEL_W + 0.10, 0.05, PANEL_D + 0.04), frameMat
    );
    frameBot.position.set(0, -PANEL_H / 2 - 0.025, 0);
    group.add(frameBot);

    // ─── Helper: criar plano com CanvasTexture redrawable ───
    function makeTexturedPlane(width, height, canvasW, canvasH) {
        var canvas = document.createElement('canvas');
        canvas.width = canvasW;
        canvas.height = canvasH;
        var tex = new THREE.CanvasTexture(canvas);
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        var mat = new THREE.MeshBasicMaterial({ map: tex });
        var mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
        return { mesh: mesh, canvas: canvas, tex: tex };
    }

    // ─── Flip Card (placar de cada time) ───
    function makeFlipCard(color) {
        var cw = 0.50, ch = 0.58;
        var cardGroup = new THREE.Group();

        // frame escuro (caixa)
        var fMat = new THREE.MeshStandardMaterial({
            color: 0x101012, roughness: 0.4, metalness: 0.4,
        });
        var f = new THREE.Mesh(new THREE.BoxGeometry(cw, ch, 0.06), fMat);
        cardGroup.add(f);

        // painel interno com o número
        var card = makeTexturedPlane(cw * 0.92, ch * 0.92, 256, 320);
        card.mesh.position.z = 0.031;
        cardGroup.add(card.mesh);

        function draw(text) {
            var ctx = card.canvas.getContext('2d');

            // fundo cartolina escura levemente sépia (estilo papel grosso
            // envelhecido, não plástico digital).
            ctx.fillStyle = '#1a1612';
            ctx.fillRect(0, 0, 256, 320);

            // highlight interno superior (luz batendo no papel)
            var grad = ctx.createLinearGradient(0, 0, 0, 156);
            grad.addColorStop(0, 'rgba(255,245,225,0.08)');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, 256, 156);

            // sombra inferior interna (sombra de "vinco" no flip)
            var grad2 = ctx.createLinearGradient(0, 164, 0, 320);
            grad2.addColorStop(0, 'rgba(0,0,0,0.55)');
            grad2.addColorStop(1, 'rgba(0,0,0,0.10)');
            ctx.fillStyle = grad2;
            ctx.fillRect(0, 164, 256, 156);

            // linha de "flip" no meio (preta com sombra grossa abaixo)
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 154, 256, 8);
            // sombra logo abaixo da dobra (escurece transição)
            var grad3 = ctx.createLinearGradient(0, 162, 0, 184);
            grad3.addColorStop(0, 'rgba(0,0,0,0.65)');
            grad3.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad3;
            ctx.fillRect(0, 162, 256, 22);

            // bordinha lateral interna (escurecimento das bordas, papel grosso)
            var sideShadow = ctx.createLinearGradient(0, 0, 256, 0);
            sideShadow.addColorStop(0,    'rgba(0,0,0,0.4)');
            sideShadow.addColorStop(0.08, 'rgba(0,0,0,0)');
            sideShadow.addColorStop(0.92, 'rgba(0,0,0,0)');
            sideShadow.addColorStop(1,    'rgba(0,0,0,0.4)');
            ctx.fillStyle = sideShadow;
            ctx.fillRect(0, 0, 256, 320);

            // número — cor sólida do time, SEM glow, fonte gorda (estilo
            // tinta impressa em papel grosso).
            ctx.fillStyle = color;
            ctx.font = 'bold 240px "Arial Black", "Helvetica", sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            // sombra discreta abaixo do número (relevo de impressão), SEM blur.
            ctx.fillText(text, 128, 162);

            card.tex.needsUpdate = true;
        }

        draw('0');
        cardGroup.userData.draw = draw;
        return cardGroup;
    }

    // cards na parte SUPERIOR do painel (sem header, eles sobem).
    // y=0.20 → cards de -0.09 a 0.49 (topo painel = 0.55, folga 0.06)
    var redCard = makeFlipCard(COLOR_RED);
    redCard.position.set(-0.46, 0.20, PANEL_D / 2 + 0.001);
    group.add(redCard);

    var blueCard = makeFlipCard(COLOR_BLUE);
    blueCard.position.set(0.46, 0.20, PANEL_D / 2 + 0.001);
    group.add(blueCard);

    // "X" entre os 2 cartões
    var x = makeTexturedPlane(0.20, 0.20, 128, 128);
    (function() {
        var ctx = x.canvas.getContext('2d');
        ctx.clearRect(0, 0, 128, 128);
        ctx.fillStyle = '#7a7a82';
        ctx.font = 'bold 84px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×', 64, 70);
        x.tex.needsUpdate = true;
    })();
    x.mesh.material.transparent = true;
    x.mesh.position.set(0, 0.20, PANEL_D / 2 + 0.005);
    group.add(x.mesh);

    // ─── Timer (estilo card manual também — sem glow LED) ───
    var timerFrameMat = new THREE.MeshStandardMaterial({
        color: 0x101012, roughness: 0.5, metalness: 0.3,
    });
    var timerFrame = new THREE.Mesh(
        new THREE.BoxGeometry(0.92, 0.20, 0.05), timerFrameMat
    );
    // y=-0.22 → timer ocupa -0.32 a -0.12 (cards param em -0.09, folga 0.03)
    timerFrame.position.set(0, -0.22, PANEL_D / 2 + 0.001);
    group.add(timerFrame);

    var tim = makeTexturedPlane(0.85, 0.16, 512, 128);
    tim.mesh.position.set(0, -0.22, PANEL_D / 2 + 0.027);
    group.add(tim.mesh);

    function drawTimer(text) {
        var ctx = tim.canvas.getContext('2d');
        // mesma cartolina escura sépia dos cards (estética unificada)
        ctx.fillStyle = '#1a1612';
        ctx.fillRect(0, 0, 512, 128);
        // highlight superior leve
        var grad = ctx.createLinearGradient(0, 0, 0, 64);
        grad.addColorStop(0, 'rgba(255,245,225,0.06)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 512, 64);
        // texto amarelo sólido (sem glow → manual)
        ctx.fillStyle = COLOR_TIMER;
        ctx.font = 'bold 76px "Arial Black", "Helvetica", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 256, 68);
        tim.tex.needsUpdate = true;
    }
    drawTimer('00:00');

    // ─── Goal overlay (telão muda pra "GOAL" durante celebração) ───
    // plano cobre o painel TODO (não deixa o footer "YOU VS F5..." aparecer
    // por baixo). Estilo HARMÔNICO com flip-cards: mesmo fundo sepia
    // escuro, gradientes idênticos aos cards, letra na cor do time
    // (não solid bg) — feel de "card analógico que mostra GOAL".
    var goalDisp = makeTexturedPlane(PANEL_W * 0.96, PANEL_H * 0.94, 1280, 600);
    goalDisp.mesh.material.transparent = true;
    goalDisp.mesh.position.set(0, 0.0, PANEL_D / 2 + 0.012);
    goalDisp.mesh.visible = false;
    group.add(goalDisp.mesh);

    // anima a palavra GOAL com efeito "la ola" — cada letra pulsa em y
    // num offset crescente. Mantém a estética dos flip cards: fundo
    // sepia escuro, gradientes (highlight top, sombra bottom), linha de
    // dobra horizontal, letra grossa na cor do time (impressa em papel).
    function drawGoalWaveFrame(scorer) {
        var ctx = goalDisp.canvas.getContext('2d');
        var W = 1280, H = 600;
        ctx.clearRect(0, 0, W, H);

        // fundo cartolina escura sépia (mesmo dos flip cards)
        ctx.fillStyle = '#1a1612';
        ctx.fillRect(0, 0, W, H);

        // highlight interno superior (luz batendo no papel)
        var grad = ctx.createLinearGradient(0, 0, 0, H * 0.49);
        grad.addColorStop(0, 'rgba(255,245,225,0.08)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H * 0.49);

        // sombra inferior interna (vinco do flip)
        var grad2 = ctx.createLinearGradient(0, H * 0.51, 0, H);
        grad2.addColorStop(0, 'rgba(0,0,0,0.55)');
        grad2.addColorStop(1, 'rgba(0,0,0,0.10)');
        ctx.fillStyle = grad2;
        ctx.fillRect(0, H * 0.51, W, H * 0.49);

        // linha de dobra no meio (preta + sombra abaixo)
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, H * 0.49 - 3, W, 6);
        var grad3 = ctx.createLinearGradient(0, H * 0.495, 0, H * 0.55);
        grad3.addColorStop(0, 'rgba(0,0,0,0.55)');
        grad3.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad3;
        ctx.fillRect(0, H * 0.495, W, H * 0.06);

        // bordas laterais escurecidas (papel grosso)
        var sideShadow = ctx.createLinearGradient(0, 0, W, 0);
        sideShadow.addColorStop(0,    'rgba(0,0,0,0.4)');
        sideShadow.addColorStop(0.06, 'rgba(0,0,0,0)');
        sideShadow.addColorStop(0.94, 'rgba(0,0,0,0)');
        sideShadow.addColorStop(1,    'rgba(0,0,0,0.4)');
        ctx.fillStyle = sideShadow;
        ctx.fillRect(0, 0, W, H);

        // texto GOAL — cor sólida do time (igual número impresso),
        // fonte gorda Arial Black, com onda "la ola".
        var letters = ['G', 'O', 'A', 'L'];
        var teamColor = scorer === 'red' ? COLOR_RED : COLOR_BLUE;
        var t = Date.now() / 1000;
        var fontSize = 320;
        ctx.font = 'bold ' + fontSize + 'px "Arial Black", "Helvetica", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        var spacing = 230;
        var startX = W / 2 - spacing * 1.5;
        var baseY = H * 0.49;       // alinhado com a linha de dobra (igual cards)
        for (var i = 0; i < letters.length; i++) {
            var phase = (t * 2.4 - i * 0.45) * Math.PI;
            var offsetY = Math.sin(phase) * 26;
            // letra principal cor do time
            ctx.fillStyle = teamColor;
            ctx.fillText(letters[i], startX + i * spacing, baseY + offsetY);
        }

        goalDisp.tex.needsUpdate = true;
    }

    // ─── Footer (snapshot label, ex: "YOU VS 47.6M (FINAL)") ───
    // tamanho ampliado (vs versão anterior 60px) + cor mais branca e
    // canvas resolução maior pra ficar legível à distância da câmera.
    var foot = makeTexturedPlane(PANEL_W * 0.92, 0.18, 1536, 200);
    foot.mesh.material.transparent = true;
    foot.mesh.position.set(0, -0.45, PANEL_D / 2 + 0.005);
    group.add(foot.mesh);

    function drawFooter(text) {
        var ctx = foot.canvas.getContext('2d');
        ctx.clearRect(0, 0, 1536, 200);
        ctx.fillStyle = '#ffffff';
        // fonte maior + auto-fit: se texto fica muito largo, encolhe
        // até caber em 92% da largura do canvas. Mantém legibilidade
        // mesmo com labels longas tipo "F5 · 50M (FINAL) VS F0 · RANDOM".
        var displayText = (text || '').toUpperCase();
        var maxW = 1536 * 0.92;
        var fontSize = 92;
        ctx.font = 'bold ' + fontSize + 'px sans-serif';
        while (ctx.measureText(displayText).width > maxW && fontSize > 40) {
            fontSize -= 4;
            ctx.font = 'bold ' + fontSize + 'px sans-serif';
        }
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(displayText, 768, 100);
        foot.tex.needsUpdate = true;
    }
    drawFooter('');

    // ─── Suporte (2 polos verticais descendo até a arquibancada) ───
    // poleH ajustado pra que o polo desça até ~y=0 absoluto (com group em y=1.30
    // e painel bottom em y=0.825 abs, polo de altura 0.85 cobre até y=-0.025).
    var poleMat = new THREE.MeshStandardMaterial({
        color: 0x3a3a44, metalness: 0.6, roughness: 0.45,
    });
    var poleH = 0.85;
    var poleGeo = new THREE.CylinderGeometry(0.045, 0.045, poleH, 10);
    var leftPole = new THREE.Mesh(poleGeo, poleMat);
    leftPole.position.set(-PANEL_W * 0.40, -PANEL_H / 2 - poleH / 2, 0);
    group.add(leftPole);
    var rightPole = new THREE.Mesh(poleGeo, poleMat);
    rightPole.position.set(PANEL_W * 0.40, -PANEL_H / 2 - poleH / 2, 0);
    group.add(rightPole);

    // ─── Posicionamento final no estádio ───
    // parede NORTE (z=0), centro do campo (x=FW/2), em altura média.
    // reduzida de 1.95 → 1.30 pra ficar visualmente mais próximo do campo
    // (sem header AI·SOCCER, o painel também é menor).
    group.position.set(FW / 2, 1.30, -0.48);

    scene.add(group);

    // estado do modo gol (telão muda pra "GOAL" durante celebração).
    var goalModeActive = false;
    var goalModeScorer = null;

    function setNormalElementsVisible(vis) {
        redCard.visible = vis;
        blueCard.visible = vis;
        x.mesh.visible = vis;
        timerFrame.visible = vis;
        tim.mesh.visible = vis;
        // também esconde o snapshot label embaixo — antes ficava visível
        // por baixo do plano GOAL pq o plano não cobria o painel todo.
        foot.mesh.visible = vis;
    }

    // ─── API pública ───
    return {
        group: group,

        updateScore: function(red, blue) {
            redCard.userData.draw(String(Math.max(0, red | 0)));
            blueCard.userData.draw(String(Math.max(0, blue | 0)));
        },

        updateTimer: function(elapsedSeconds) {
            var s = Math.max(0, Math.floor(elapsedSeconds));
            var mm = Math.floor(s / 60);
            var ss = s % 60;
            drawTimer(
                (mm < 10 ? '0' : '') + mm + ':' + (ss < 10 ? '0' : '') + ss
            );
        },

        updateSnapshot: function(label) {
            drawFooter(label || '');
        },

        // modo gol — telão troca placar/timer por "GOAL" animado.
        showGoalMode: function(scorer) {
            goalModeActive = true;
            goalModeScorer = scorer;
            setNormalElementsVisible(false);
            goalDisp.mesh.visible = true;
            drawGoalWaveFrame(scorer);
        },
        hideGoalMode: function() {
            goalModeActive = false;
            goalModeScorer = null;
            goalDisp.mesh.visible = false;
            setNormalElementsVisible(true);
        },
        // chamado a cada frame em main.js durante a animação ola.
        tick: function() {
            if (goalModeActive) drawGoalWaveFrame(goalModeScorer);
        },
    };
};
