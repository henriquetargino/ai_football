/**
 * neural Network Viz — visualização da policy em estilo "scientific paper"
 * (Distill.pub / 3Blue1Brown / TensorFlow Playground).
 *
 * estética minimalista: tipografia LIGHT (weight 300), nodes circulares com
 * outline fino, conexões em curvas bezier, SEM cards / heatmaps / glow.
 * o painel parece um diagrama editorial — não um dashboard "de IA".
 *
 * estrutura visual: rede neural com contagem REAL nos outputs
 *
 *   input (16 buckets)   h1 (12)   h2 (12)   output (18 individuais)
 *      ●                                              ↑·↺  ●  ← chosen
 *      ○                  ●           ●               ↑··  ○
 *      ●                  ●           ○               ↓··  ○
 *      ○                  ○           ●               ··✺  ○
 *      ●                  ●           ○               ...
 *      ...                ...         ...             (18)
 *
 * aPIs:
 *   AISoccer.toggleNeuralViz()
 *   AISoccer.isNeuralVizVisible()
 *   AISoccer.updateNeuralViz({input, h1, h2, logits, actionIdx})
 *   AISoccer.setNeuralVizTeam("red"|"blue")
 */
var AISoccer = AISoccer || {};

(function () {
    "use strict";

    // estado
    var visible = false;
    var canvas = null;
    var ctx = null;
    var container = null;
    var rafId = null;

    var targetAct = null;
    var displayed = null;
    var lastActionLockMs = 0;
    var displayedActionIdx = 0;

    var LERP_FACTOR = 0.05;             // ~1s pra estabilizar
    var ACTION_LOCK_MS = 600;
    var TOP_K = 5;

    // contagem REAL de nós por camada (input/hidden bucketizados, output integral)
    var LAYERS = {
        input:  { visual: 16, real: 341, type: "bucket" },
        h1:     { visual: 12, real: 64,  type: "bucket" },
        h2:     { visual: 12, real: 64,  type: "bucket" },
        output: { visual: 18, real: 18,  type: "full" },
    };

    // tokens — design editorial / scientific paper
    var TOKENS = {
        bg:       "#0a0a14",
        bgInner:  "#0f0f1c",
        line:     "rgba(255, 255, 255, 0.08)",
        lineHi:   "rgba(255, 255, 255, 0.16)",
        text:     "rgba(255, 255, 255, 0.92)",
        textSoft: "rgba(255, 255, 255, 0.56)",
        textDim:  "rgba(255, 255, 255, 0.32)",
        textVeryDim: "rgba(255, 255, 255, 0.18)",
        font:     "Poppins, Inter, -apple-system, system-ui, sans-serif",
        mono:     "'SF Mono', 'JetBrains Mono', Consolas, monospace",
    };

    var TEAM_COLORS = {
        red: {
            primary: "#ff6b1a",
            accent:  "rgba(255, 107, 26, ALPHA)",
        },
        blue: {
            primary: "#2a8aff",
            accent:  "rgba(42, 138, 255, ALPHA)",
        },
    };

    var teamColor = TEAM_COLORS.blue;

    AISoccer.setNeuralVizTeam = function (team) {
        if (TEAM_COLORS[team]) teamColor = TEAM_COLORS[team];
    };

    // setup
    function ensureContainer() {
        if (container) return;
        container = document.createElement("div");
        container.id = "neural-viz-panel";
        container.style.cssText = [
            "position: fixed",
            "top: 0",
            "right: 0",
            "width: 35vw",
            "height: 100vh",
            "background: " + TOKENS.bg,
            "border-left: 1px solid " + TOKENS.line,
            "z-index: 100",
            "display: none",
            "pointer-events: none",
        ].join(";");
        canvas = document.createElement("canvas");
        canvas.style.cssText = "width:100%;height:100%;display:block";
        container.appendChild(canvas);
        document.body.appendChild(container);
        resize();
        window.addEventListener("resize", resize);
    }

    function resize() {
        if (!canvas) return;
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var w = container.offsetWidth;
        var h = container.offsetHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
        ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // helpers
    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function colorWithAlpha(t, a) { return t.replace("ALPHA", a.toFixed(3)); }

    function bucketize(vec, nBuckets) {
        var out = new Float32Array(nBuckets);
        var n = vec.length;
        var per = n / nBuckets;
        for (var b = 0; b < nBuckets; b++) {
            var lo = Math.floor(b * per);
            var hi = Math.floor((b + 1) * per);
            var sum = 0, cnt = 0;
            for (var i = lo; i < hi && i < n; i++) {
                sum += vec[i];
                cnt++;
            }
            out[b] = cnt > 0 ? sum / cnt : 0;
        }
        return out;
    }

    function softmax(logits) {
        var max = -Infinity;
        for (var i = 0; i < logits.length; i++) {
            if (logits[i] > max) max = logits[i];
        }
        var exps = new Float32Array(logits.length);
        var sum = 0;
        for (var i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - max);
            sum += exps[i];
        }
        for (var i = 0; i < logits.length; i++) exps[i] /= sum;
        return exps;
    }

    function topKActions(probs, k) {
        var idx = [];
        for (var i = 0; i < probs.length; i++) idx.push(i);
        idx.sort(function (a, b) { return probs[b] - probs[a]; });
        return idx.slice(0, k);
    }

    function topNAbsIndices(arr, n) {
        var pairs = [];
        for (var i = 0; i < arr.length; i++) {
            pairs.push({ i: i, v: Math.abs(arr[i]) });
        }
        pairs.sort(function (a, b) { return b.v - a.v; });
        var out = [];
        for (var k = 0; k < Math.min(n, pairs.length); k++) {
            out.push(pairs[k].i);
        }
        return out;
    }

    function maxAbs(arr) {
        var m = 0;
        for (var i = 0; i < arr.length; i++) {
            var v = Math.abs(arr[i]);
            if (v > m) m = v;
        }
        return m || 1;
    }

    function decodeAction(idx) {
        var kick = Math.floor(idx / 9);
        var rest = idx - kick * 9;
        var rot = Math.floor(rest / 3) - 1;
        var accel = (rest % 3) - 1;
        return { accel: accel, rot: rot, kick: kick };
    }

    function actionShortLabel(idx) {
        var a = decodeAction(idx);
        return (a.accel > 0 ? "↑" : a.accel < 0 ? "↓" : "·")
            + "  " + (a.rot < 0 ? "↺" : a.rot > 0 ? "↻" : "·")
            + "  " + (a.kick ? "✺" : "·");
    }

    function actionLongLabel(idx) {
        var a = decodeAction(idx);
        var parts = [];
        if (a.accel > 0) parts.push("accelerate");
        else if (a.accel < 0) parts.push("reverse");
        else parts.push("idle");
        if (a.rot > 0) parts.push("right");
        else if (a.rot < 0) parts.push("left");
        if (a.kick) parts.push("kick");
        return parts.join(" · ");
    }

    // LERP suave
    function tickLerp() {
        if (!targetAct) return;
        if (!displayed) {
            displayed = {
                input: new Float32Array(targetAct.input.length),
                h1: new Float32Array(targetAct.h1.length),
                h2: new Float32Array(targetAct.h2.length),
                logits: new Float32Array(targetAct.logits.length),
            };
        }
        var a = LERP_FACTOR;
        var arrs = ["input", "h1", "h2", "logits"];
        for (var k = 0; k < arrs.length; k++) {
            var src = targetAct[arrs[k]];
            var dst = displayed[arrs[k]];
            for (var i = 0; i < src.length; i++) {
                dst[i] += (src[i] - dst[i]) * a;
            }
        }
        var now = (typeof performance !== "undefined")
            ? performance.now() : Date.now();
        if (now - lastActionLockMs > ACTION_LOCK_MS) {
            var best = 0;
            for (var i = 1; i < displayed.logits.length; i++) {
                if (displayed.logits[i] > displayed.logits[best]) best = i;
            }
            if (best !== displayedActionIdx) {
                displayedActionIdx = best;
                lastActionLockMs = now;
            }
        }
    }

    // render — minimalista, estilo "scientific paper"
    function buildNodePositions(x, topY, height, count) {
        var nodes = [];
        var spacing = height / count;
        for (var i = 0; i < count; i++) {
            nodes.push({ x: x, y: topY + spacing / 2 + i * spacing });
        }
        return nodes;
    }

    /** Node circular: outline fino + fill colorido. Sem glow. */
    function drawNode(ctx, x, y, radius, value, maxValue, isHighlight) {
        var intensity = clamp(Math.abs(value / (maxValue || 1)), 0, 1);

        // background sutil pra dar profundidade
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = TOKENS.bgInner;
        ctx.fill();

        // fill colorido se ativo
        if (intensity > 0.05) {
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = colorWithAlpha(teamColor.accent, intensity * 0.85);
            ctx.fill();
        }

        // outline fino
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = isHighlight
            ? teamColor.primary
            : intensity > 0.4
                ? colorWithAlpha(teamColor.accent, 0.7)
                : TOKENS.lineHi;
        ctx.lineWidth = isHighlight ? 1.4 : 0.8;
        ctx.stroke();

        // anel externo se highlight (chosen output)
        if (isHighlight) {
            ctx.beginPath();
            ctx.arc(x, y, radius + 3.5, 0, Math.PI * 2);
            ctx.strokeStyle = colorWithAlpha(teamColor.accent, 0.45);
            ctx.lineWidth = 0.8;
            ctx.stroke();
        }
    }

    /** Conexão entre 2 nodes — bezier curve sutil + alpha proporcional.
     *  threshold alto pra sparsificar (com mais nós, fica visualmente sufocado). */
    function drawConnection(ctx, x1, y1, x2, y2, magnitude) {
        if (magnitude < 0.26) return;
        var alpha = clamp(magnitude * 0.95, 0.10, 0.55);
        var midX = (x1 + x2) / 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(midX, y1, midX, y2, x2, y2);
        ctx.strokeStyle = colorWithAlpha(teamColor.accent, alpha);
        ctx.lineWidth = 0.4 + magnitude * 0.6;
        ctx.stroke();
    }

    function render() {
        if (!visible || !canvas || !ctx) return;
        var w = canvas.clientWidth;
        var h = canvas.clientHeight;
        ctx.clearRect(0, 0, w, h);

        var pad = 28;

        // ── Header ────────────────────────────────────
        ctx.font = "700 18px " + TOKENS.font;
        ctx.fillStyle = TOKENS.text;
        ctx.textAlign = "left";
        ctx.fillText("neural network", pad, 42);

        ctx.font = "300 11px " + TOKENS.mono;
        ctx.fillStyle = TOKENS.textDim;
        ctx.textAlign = "right";
        ctx.fillText("341 → 64 → 64 → 18", w - pad, 42);

        ctx.strokeStyle = TOKENS.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad, 56);
        ctx.lineTo(w - pad, 56);
        ctx.stroke();

        if (!displayed) {
            ctx.fillStyle = TOKENS.textDim;
            ctx.font = "300 13px " + TOKENS.font;
            ctx.textAlign = "center";
            ctx.fillText("waiting for inference", w / 2, h / 2);
            return;
        }

        // ── Network graph ─────────────────────────────
        var netTop = 78;
        var netBottom = h * 0.58;
        var netHeight = netBottom - netTop;

        var colXs = [
            pad + 22,       // input
            w * 0.36,       // h1
            w * 0.62,       // h2
            w - pad - 32,   // output
        ];

        var inputBuckets = bucketize(displayed.input, LAYERS.input.visual);
        var h1Buckets = bucketize(displayed.h1, LAYERS.h1.visual);
        var h2Buckets = bucketize(displayed.h2, LAYERS.h2.visual);
        var probs = softmax(displayed.logits);
        // output: usa todos os 18 (sem bucketize)
        var outValues = probs;

        var maxIn = maxAbs(inputBuckets);
        var maxH1 = maxAbs(h1Buckets);
        var maxH2 = maxAbs(h2Buckets);
        var maxOut = maxAbs(outValues);

        var inputNodes = buildNodePositions(colXs[0], netTop, netHeight, LAYERS.input.visual);
        var h1Nodes = buildNodePositions(colXs[1], netTop, netHeight, LAYERS.h1.visual);
        var h2Nodes = buildNodePositions(colXs[2], netTop, netHeight, LAYERS.h2.visual);
        var outNodes = buildNodePositions(colXs[3], netTop, netHeight, LAYERS.output.visual);

        // conexões — bezier
        function connectLayers(fromN, fromV, fromMax, toN, toV, toMax) {
            for (var i = 0; i < fromN.length; i++) {
                for (var j = 0; j < toN.length; j++) {
                    var fv = fromV[i] / fromMax;
                    var tv = toV[j] / toMax;
                    var mag = Math.abs(fv) * Math.abs(tv);
                    drawConnection(ctx,
                        fromN[i].x, fromN[i].y,
                        toN[j].x, toN[j].y,
                        mag);
                }
            }
        }
        connectLayers(inputNodes, inputBuckets, maxIn, h1Nodes, h1Buckets, maxH1);
        connectLayers(h1Nodes, h1Buckets, maxH1, h2Nodes, h2Buckets, maxH2);
        connectLayers(h2Nodes, h2Buckets, maxH2, outNodes, outValues, maxOut);

        // nodes — input
        for (var i = 0; i < LAYERS.input.visual; i++) {
            drawNode(ctx, inputNodes[i].x, inputNodes[i].y, 5,
                     inputBuckets[i], maxIn, false);
        }
        // nodes — h1
        for (var i = 0; i < LAYERS.h1.visual; i++) {
            drawNode(ctx, h1Nodes[i].x, h1Nodes[i].y, 6,
                     h1Buckets[i], maxH1, false);
        }
        // nodes — h2
        for (var i = 0; i < LAYERS.h2.visual; i++) {
            drawNode(ctx, h2Nodes[i].x, h2Nodes[i].y, 6,
                     h2Buckets[i], maxH2, false);
        }
        // nodes — output (18 individuais, chosen highlighted maior)
        var actionIdx = displayedActionIdx;
        for (var i = 0; i < LAYERS.output.visual; i++) {
            var isChosen = i === actionIdx;
            drawNode(ctx, outNodes[i].x, outNodes[i].y,
                     isChosen ? 6 : 4,
                     outValues[i], maxOut, isChosen);
        }

        // valor de ativação nos top-2 nodes mais ativos por camada hidden
        // (mostra "matemática real" — feel técnico, não cartoon)
        ctx.font = "300 8.5px " + TOKENS.mono;
        ctx.fillStyle = TOKENS.textVeryDim;
        ctx.textAlign = "right";

        function annotateTopActivations(values, nodes, max, count, alignX) {
            var topIdx = topNAbsIndices(values, count);
            for (var k = 0; k < topIdx.length; k++) {
                var idx = topIdx[k];
                var n = nodes[idx];
                var v = values[idx] / max;
                ctx.fillText(v.toFixed(2), alignX, n.y + 2);
            }
        }
        // discreto: anotações em h1 e h2 only (input bucketizado nem é tão rico)
        annotateTopActivations(h1Buckets, h1Nodes, maxH1, 2, colXs[1] - 12);
        annotateTopActivations(h2Buckets, h2Nodes, maxH2, 2, colXs[2] - 12);

        // símbolo da ação ao lado do output escolhido
        var chosenNode = outNodes[actionIdx];
        ctx.font = "400 11px " + TOKENS.font;
        ctx.fillStyle = teamColor.primary;
        ctx.textAlign = "left";
        ctx.fillText(actionShortLabel(actionIdx),
                     chosenNode.x + 12, chosenNode.y + 4);

        // labels embaixo dos layers
        ctx.font = "300 10px " + TOKENS.font;
        ctx.fillStyle = TOKENS.textSoft;
        ctx.textAlign = "center";
        var labelY = netBottom + 16;
        ctx.fillText("input", colXs[0], labelY);
        ctx.fillText("hidden", colXs[1], labelY);
        ctx.fillText("hidden", colXs[2], labelY);
        ctx.fillText("output", colXs[3], labelY);
        // contagem real (com asterisco nas camadas bucketizadas)
        ctx.font = "300 9px " + TOKENS.mono;
        ctx.fillStyle = TOKENS.textDim;
        ctx.fillText("341*", colXs[0], labelY + 14);
        ctx.fillText("64*",  colXs[1], labelY + 14);
        ctx.fillText("64*",  colXs[2], labelY + 14);
        ctx.fillText("18",   colXs[3], labelY + 14);

        // ── Decisão ───────────────────────────────────
        var decY = netBottom + 64;

        ctx.textAlign = "left";
        ctx.font = "300 10px " + TOKENS.font;
        ctx.fillStyle = TOKENS.textDim;
        ctx.fillText("decision", pad, decY);

        var topConfidence = probs[actionIdx];

        ctx.font = "400 18px " + TOKENS.font;
        ctx.fillStyle = TOKENS.text;
        ctx.fillText(actionLongLabel(actionIdx), pad, decY + 24);

        ctx.font = "400 13px " + TOKENS.font;
        ctx.fillStyle = teamColor.primary;
        ctx.fillText(actionShortLabel(actionIdx), pad, decY + 44);

        ctx.textAlign = "right";
        ctx.font = "300 11px " + TOKENS.mono;
        ctx.fillStyle = TOKENS.textSoft;
        ctx.fillText((topConfidence * 100).toFixed(0) + "% confidence",
                     w - pad, decY + 24);

        // ── Probabilities (top 5) ─────────────────────
        var probsY = decY + 78;

        ctx.textAlign = "left";
        ctx.font = "300 10px " + TOKENS.font;
        ctx.fillStyle = TOKENS.textDim;
        ctx.fillText("probabilities", pad, probsY);

        ctx.strokeStyle = TOKENS.line;
        ctx.beginPath();
        ctx.moveTo(pad, probsY + 8);
        ctx.lineTo(w - pad, probsY + 8);
        ctx.stroke();

        var top5 = topKActions(probs, TOP_K);
        var maxProb = probs[top5[0]];
        var rowY = probsY + 28;
        var rowGap = Math.min(28, (h - rowY - 40) / TOP_K);

        for (var i = 0; i < top5.length; i++) {
            var idx = top5[i];
            var p = probs[idx];
            var isChosen = (idx === actionIdx);

            ctx.font = "400 13px " + TOKENS.font;
            ctx.fillStyle = isChosen ? teamColor.primary : TOKENS.textSoft;
            ctx.textAlign = "left";
            ctx.fillText(actionShortLabel(idx), pad, rowY);

            var barX0 = pad + 70;
            var barX1 = w - pad - 50;
            var barY = rowY - 4;
            var barFullW = barX1 - barX0;
            var ratio = clamp(p / (maxProb || 1), 0, 1);
            var barFillW = Math.max(2, ratio * barFullW);

            ctx.strokeStyle = TOKENS.line;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(barX0, barY);
            ctx.lineTo(barX1, barY);
            ctx.stroke();

            ctx.strokeStyle = isChosen
                ? teamColor.primary
                : colorWithAlpha(teamColor.accent, 0.55);
            ctx.lineWidth = isChosen ? 2 : 1.5;
            ctx.beginPath();
            ctx.moveTo(barX0, barY);
            ctx.lineTo(barX0 + barFillW, barY);
            ctx.stroke();

            if (isChosen) {
                ctx.beginPath();
                ctx.arc(barX0 + barFillW, barY, 3, 0, Math.PI * 2);
                ctx.fillStyle = teamColor.primary;
                ctx.fill();
            }

            ctx.font = (isChosen ? "400" : "300") + " 12px " + TOKENS.mono;
            ctx.fillStyle = isChosen ? TOKENS.text : TOKENS.textSoft;
            ctx.textAlign = "right";
            ctx.fillText((p * 100).toFixed(0) + "%", w - pad, rowY);

            rowY += rowGap;
        }

        // footer: legenda + footnote do bucket
        ctx.font = "300 9px " + TOKENS.font;
        ctx.fillStyle = TOKENS.textDim;
        ctx.textAlign = "left";
        ctx.fillText("↑ accel   ↓ reverse   ↺ left   ↻ right   ✺ kick",
                     pad, h - 32);
        ctx.fillStyle = TOKENS.textVeryDim;
        ctx.fillText("* bucketed view (avg-pooled for display)",
                     pad, h - 16);
    }

    function loopTick() {
        if (!visible) return;
        tickLerp();
        render();
    }

    // API pública
    AISoccer.toggleNeuralViz = function () {
        ensureContainer();
        visible = !visible;
        container.style.display = visible ? "block" : "none";
        if (visible) {
            resize();
            if (rafId) clearInterval(rafId);
            rafId = setInterval(loopTick, 16);
            // notifica scene.js pra re-projetar o campo no espaço restante.
            // o renderer do jogo lê AISoccer.activePanelWidth na próxima
            // chamada de resize.
            AISoccer.activePanelWidth = container.offsetWidth;
        } else {
            if (rafId) clearInterval(rafId);
            rafId = null;
            AISoccer.activePanelWidth = 0;
        }
        // dispara o resize handler global (registrado em scene.js) pra
        // reposicionar o canvas Three.js fora da área do painel.
        window.dispatchEvent(new Event("resize"));
        return visible;
    };

    AISoccer.isNeuralVizVisible = function () {
        return visible;
    };

    AISoccer.updateNeuralViz = function (activations) {
        if (!visible) return;
        targetAct = activations;
    };

})();
