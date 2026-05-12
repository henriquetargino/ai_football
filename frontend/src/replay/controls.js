/**
 * controles de playback: play, pause, velocidade, timeline.
 */

// guarda a referência ao replayPlayer ativo para os atalhos globais
// (keyboard) em main.js consultarem sem recriar listeners.
AISoccer._activeReplayPlayer = null;

AISoccer.setupControls = function(replayPlayer) {
    var btnPlay = document.getElementById('btn-play');
    var btnPause = document.getElementById('btn-pause');
    var btnSpeedUp = document.getElementById('btn-speed-up');
    var btnSpeedDown = document.getElementById('btn-speed-down');
    var speedLabel = document.getElementById('speed-label');
    var timeline = document.getElementById('timeline');
    var btnRays = document.getElementById('btn-rays');

    AISoccer._activeReplayPlayer = replayPlayer;

    // reseta UI para o estado inicial do NOVO replay (antes a label ficava
    // presa no "4×" da sessão anterior).
    speedLabel.textContent = replayPlayer.speed + '×';
    timeline.value = 0;

    // `btn-menu` já é tratado por main.js → returnToMenu (limpeza completa).
    // evitado listener duplicado aqui; registrar um segundo handler causava
    // execução dupla e referência pendurada ao replayPlayer antigo.
    //
    // atalhos globais (R, Space) também foram movidos para setupGlobalKeys
    // em main.js — eles precisam de consciência de modo, e registrar no
    // `document` a cada replay novo vazava listeners ao longo da sessão.

    // handlers dos botões seguem escopo local — usam o replayPlayer por
    // closure, mas como os elementos do DOM persistem entre replays, esses
    // listeners se ACUMULAM também. Usamos cloneNode para garantir que cada
    // replay começa com o botão "limpo" (remove todos os listeners antigos).
    btnPlay = AISoccer._freshListener(btnPlay, 'click', function() { replayPlayer.play(); });
    btnPause = AISoccer._freshListener(btnPause, 'click', function() { replayPlayer.pause(); });
    btnSpeedUp = AISoccer._freshListener(btnSpeedUp, 'click', function() {
        replayPlayer.setSpeed(replayPlayer.speed * 2);
        speedLabel.textContent = replayPlayer.speed + '×';
    });
    btnSpeedDown = AISoccer._freshListener(btnSpeedDown, 'click', function() {
        replayPlayer.setSpeed(replayPlayer.speed / 2);
        speedLabel.textContent = replayPlayer.speed + '×';
    });
    timeline = AISoccer._freshListener(timeline, 'input', function() {
        var fraction = parseInt(timeline.value) / 1000;
        replayPlayer.seek(fraction);
    });
    btnRays = AISoccer._freshListener(btnRays, 'click', function() {
        var vis = AISoccer.toggleRaycasts();
        btnRays.classList.toggle('active', vis);
    });
    var btnNN = document.getElementById('btn-nn');
    if (btnNN) {
        btnNN = AISoccer._freshListener(btnNN, 'click', function() {
            var vis = AISoccer.toggleNeuralViz();
            btnNN.classList.toggle('active', vis);
        });
    }

    // atualizar timeline continuamente
    AISoccer._updateControlsUI = function() {
        if (replayPlayer.playing) {
            timeline.value = Math.floor(replayPlayer.getProgress() * 1000);
        }
    };
};

// remove todos os listeners de um elemento clonando-o; retorna o clone já
// com o novo listener anexado. Evita o leak clássico de "registrar handler
// de novo a cada entrada no modo replay".
AISoccer._freshListener = function(el, evt, handler) {
    var fresh = el.cloneNode(true);
    el.parentNode.replaceChild(fresh, el);
    fresh.addEventListener(evt, handler);
    return fresh;
};
