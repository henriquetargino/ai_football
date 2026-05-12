/**
 * leitor de replay JSON + interpolação 30fps → 60fps.
 */

AISoccer.ReplayPlayer = function() {
    this.data = null;
    this.frames = [];
    this.metadata = null;
    this.currentTime = 0;       // tempo em segundos
    this.duration = 0;
    this.playing = false;
    this.speed = 1.0;
    this.replayFps = 30;
    this.goalEvents = [];       // [{ frameIdx, scorer }] — preenchido em load()
};

AISoccer.ReplayPlayer.prototype.load = function(url, callback) {
    var self = this;
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            self.data = JSON.parse(xhr.responseText);
            self.frames = self.data.frames;
            self.metadata = self.data.metadata;
            self.replayFps = self.metadata.fps || 30;
            self.duration = self.frames.length / self.replayFps;
            self.currentTime = 0;
            self.goalEvents = AISoccer._detectGoalEvents(self.frames);
            if (callback) callback(self.data);
        }
    };
    xhr.send();
};

AISoccer.ReplayPlayer.prototype.play = function() {
    this.playing = true;
};

AISoccer.ReplayPlayer.prototype.pause = function() {
    this.playing = false;
};

AISoccer.ReplayPlayer.prototype.setSpeed = function(s) {
    this.speed = Math.max(0.25, Math.min(4.0, s));
};

AISoccer.ReplayPlayer.prototype.seek = function(fraction) {
    this.currentTime = fraction * this.duration;
};

AISoccer.ReplayPlayer.prototype.update = function(dt) {
    if (!this.playing || !this.frames.length) return;
    this.currentTime += dt * this.speed;
    if (this.currentTime >= this.duration) {
        this.currentTime = 0;  // loop
    }
};

AISoccer.ReplayPlayer.prototype.getState = function() {
    if (!this.frames.length) return null;

    var fps = this.replayFps;
    var frameIndex = Math.floor(this.currentTime * fps);
    var t = (this.currentTime * fps) - frameIndex;

    if (frameIndex >= this.frames.length - 1) {
        return this.frames[this.frames.length - 1];
    }

    var f0 = this.frames[frameIndex];
    var f1 = this.frames[frameIndex + 1];

    return {
        ball: {
            x: AISoccer._lerp(f0.ball.x, f1.ball.x, t),
            y: AISoccer._lerp(f0.ball.y, f1.ball.y, t),
            vx: AISoccer._lerp(f0.ball.vx, f1.ball.vx, t),
            vy: AISoccer._lerp(f0.ball.vy, f1.ball.vy, t),
        },
        players: f0.players.map(function(p, i) {
            var p1 = f1.players[i];
            return {
                id: p.id,
                x: AISoccer._lerp(p.x, p1.x, t),
                y: AISoccer._lerp(p.y, p1.y, t),
                angle: AISoccer._lerpAngle(p.angle, p1.angle, t),
                vx: AISoccer._lerp(p.vx, p1.vx, t),
                vy: AISoccer._lerp(p.vy, p1.vy, t),
                can_kick: p.can_kick,
                is_kicking: p.is_kicking || p1.is_kicking,
            };
        }),
        events: f0.events,
    };
};

AISoccer.ReplayPlayer.prototype.getProgress = function() {
    return this.duration > 0 ? this.currentTime / this.duration : 0;
};

// ── Funções de interpolação ──

AISoccer._lerp = function(a, b, t) {
    return a + (b - a) * t;
};

AISoccer._lerpAngle = function(a, b, t) {
    var diff = b - a;
    // normalizar para [-PI, PI]
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return a + diff * t;
};

// ── Detecção de gols via teleporte da bola ──
//
// o replay não grava um campo `events` populado — quando um gol acontece o
// engine chama randomize_spawns() e reposiciona a bola no centro no MESMO
// frame. O rastro disso no replay é um DELTA grande entre frames consecutivos
// da posição da bola.
//
// criterio do gol REAL: a bola estava PASSADA da linha do gol no frame
// anterior (b0.x < 0 → bola entrou no gol esquerdo / b0.x > FIELD_WIDTH →
// entrou no gol direito). Versão antiga usava só "lado do meio campo" pra
// inferir o time, mas isso confundia respawns de fase intermediária com
// gols (em F1A, por ex., depois do reset a bola é reposicionada perto do
// gol direito → 2º teleport detectado como "blue goal" falso). Restringir
// pra "bola passou a linha" elimina esses falsos positivos.
//
// threshold de teleporte: 100 unidades. MAX_SPEED_BALL=12 unidades/step,
// frame de replay = 2 steps = até ~24 unidades de deslocamento físico
// real. 100 dá folga confortável.
AISoccer._detectGoalEvents = function(frames) {
    var events = [];
    if (!frames || frames.length < 2) return events;
    var FIELD_WIDTH = 800;
    var TELEPORT_THRESHOLD = 100;
    for (var i = 1; i < frames.length; i++) {
        var b0 = frames[i - 1].ball;
        var b1 = frames[i].ball;
        var dx = b1.x - b0.x;
        var dy = b1.y - b0.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= TELEPORT_THRESHOLD) continue;
        // só conta se a bola REALMENTE estava no gol no frame anterior
        if (b0.x < 0) {
            events.push({ frameIdx: i, scorer: 'blue' });
        } else if (b0.x > FIELD_WIDTH) {
            events.push({ frameIdx: i, scorer: 'red' });
        }
        // caso contrário: teleport sem ter passado da linha é respawn de
        // currículo (não conta como gol).
    }
    return events;
};
