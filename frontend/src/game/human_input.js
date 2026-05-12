/**
 * input humano: WASD + espaço.
 * espaço registra kick apenas no frame do keydown (não-repetitivo).
 */

var AISoccer = AISoccer || {};

AISoccer.HumanInput = (function() {

    var keys = {};
    var kickQueued = false;
    var initialized = false;

    function onKeyDown(e) {
        var k = e.key.toLowerCase();
        if (k === " " || k === "w" || k === "a" || k === "s" || k === "d") {
            e.preventDefault();
        }
        if (k === " " && !keys[" "]) {
            kickQueued = true;
        }
        keys[k] = true;
    }

    function onKeyUp(e) {
        keys[e.key.toLowerCase()] = false;
    }

    function init() {
        if (initialized) return;
        keys = {};
        kickQueued = false;
        window.addEventListener("keydown", onKeyDown);
        window.addEventListener("keyup", onKeyUp);
        initialized = true;
    }

    function getOutputs() {
        var accel = 0, rot = 0, kick = 0;

        if (keys["w"]) accel += 1;
        if (keys["s"]) accel -= 1;
        if (keys["a"]) rot -= 1;
        if (keys["d"]) rot += 1;

        if (kickQueued) {
            kick = 1.0;
            kickQueued = false;
        }

        return [accel, rot, kick];
    }

    function destroy() {
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup", onKeyUp);
        keys = {};
        kickQueued = false;
        initialized = false;
    }

    return {
        init: init,
        getOutputs: getOutputs,
        destroy: destroy
    };

})();
