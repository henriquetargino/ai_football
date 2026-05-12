/**
 * 4 torres de refletores nos cantos do campo.
 * spotLights com sombras (apenas 2 primeiros).
 */

AISoccer.createFloodlights = function(scene) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;
    var towerOffset = 1.8;
    var poleHeight = 3.5;
    var poleRadius = 0.04;

    var poleMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.8, roughness: 0.3 });
    var bracketMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.6 });
    var lampMat = new THREE.MeshStandardMaterial({
        color: 0xffffdd, emissive: 0xffffcc, emissiveIntensity: 0.6
    });

    var positions = [
        { x: -towerOffset,        z: -towerOffset },
        { x: FW + towerOffset,    z: -towerOffset },
        { x: -towerOffset,        z: FH + towerOffset },
        { x: FW + towerOffset,    z: FH + towerOffset },
    ];

    for (var i = 0; i < positions.length; i++) {
        var pos = positions[i];

        // poste
        var poleGeo = new THREE.CylinderGeometry(poleRadius, poleRadius * 1.3, poleHeight, 8);
        var pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(pos.x, poleHeight / 2, pos.z);
        pole.castShadow = true;
        scene.add(pole);

        // bracket
        var bracketGeo = new THREE.BoxGeometry(0.15, 0.1, 0.15);
        var bracket = new THREE.Mesh(bracketGeo, bracketMat);
        bracket.position.set(pos.x, poleHeight, pos.z);
        scene.add(bracket);

        // lâmpada
        var lampGeo = new THREE.BoxGeometry(0.2, 0.06, 0.12);
        var lamp = new THREE.Mesh(lampGeo, lampMat);
        lamp.position.set(pos.x, poleHeight + 0.06, pos.z);
        scene.add(lamp);

        // spotLight
        var spotLight = new THREE.SpotLight(0xffeedd, 0.9, 18, Math.PI / 3.5, 0.4);
        spotLight.position.set(pos.x, poleHeight + 0.1, pos.z);
        spotLight.target.position.set(FW / 2, 0, FH / 2);
        spotLight.userData.isFloodlight = true;
        spotLight.target.userData.isFloodlight = true;
        scene.add(spotLight);
        scene.add(spotLight.target);

        // sombras apenas nos 2 primeiros spots. PERF: 1024 → 512 reduz
        // o custo do shadow pass em 4x (cada map é 1MB → 256KB).
        if (i < 2) {
            spotLight.castShadow = true;
            spotLight.shadow.mapSize.width = 512;
            spotLight.shadow.mapSize.height = 512;
            spotLight.shadow.camera.near = 1;
            spotLight.shadow.camera.far = 20;
        }
    }
};
