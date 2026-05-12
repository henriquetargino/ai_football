/**
 * gols 3D com traves, travessão, postes traseiros e redes.
 * tamanho reduzido para não sobrepor arquibancadas laterais.
 */

AISoccer.createGoals = function(scene, goalWidth) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;
    var GW = (goalWidth || 80) * AISoccer.SCALE;
    var goalDepth = 0.35;
    // POST_RADIUS físico = 5 unidades de jogo × SCALE (0.01) = 0.05 em three.js.
    // o visual espelha o colisor físico para evitar "bola passando pela trave".
    var postRadius = 5 * AISoccer.SCALE;
    var postHeight = 0.65;     // mais alto que os jogadores
    var midZ = FH / 2;

    var postMat = new THREE.MeshStandardMaterial({
        color: 0xeeeeee, metalness: 0.4, roughness: 0.3
    });
    var netMat = new THREE.MeshBasicMaterial({
        color: 0xffffff, wireframe: true, transparent: true, opacity: 0.15
    });

    function createGoal(x, direction) {
        var group = new THREE.Group();

        // postes frontais (2)
        var postGeo = new THREE.CylinderGeometry(postRadius, postRadius, postHeight, 8);
        var post1 = new THREE.Mesh(postGeo, postMat);
        post1.position.set(0, postHeight / 2, -GW / 2);
        post1.castShadow = true;
        group.add(post1);

        var post2 = new THREE.Mesh(postGeo, postMat);
        post2.position.set(0, postHeight / 2, GW / 2);
        post2.castShadow = true;
        group.add(post2);

        // travessão
        var barGeo = new THREE.CylinderGeometry(postRadius, postRadius, GW, 8);
        var bar = new THREE.Mesh(barGeo, postMat);
        bar.rotation.x = Math.PI / 2;
        bar.position.set(0, postHeight, 0);
        bar.castShadow = true;
        group.add(bar);

        // postes traseiros (2)
        var backPostGeo = new THREE.CylinderGeometry(postRadius * 0.7, postRadius * 0.7, postHeight, 6);
        var backPost1 = new THREE.Mesh(backPostGeo, postMat);
        backPost1.position.set(direction * goalDepth, postHeight / 2, -GW / 2);
        group.add(backPost1);

        var backPost2 = new THREE.Mesh(backPostGeo, postMat);
        backPost2.position.set(direction * goalDepth, postHeight / 2, GW / 2);
        group.add(backPost2);

        // barras superiores de profundidade (2 laterais + 1 traseira)
        var thinR = postRadius * 0.5;
        var depthBarGeo = new THREE.CylinderGeometry(thinR, thinR, goalDepth, 6);

        var depthBar1 = new THREE.Mesh(depthBarGeo, postMat);
        depthBar1.rotation.z = Math.PI / 2;
        depthBar1.position.set(direction * goalDepth / 2, postHeight, -GW / 2);
        group.add(depthBar1);

        var depthBar2 = new THREE.Mesh(depthBarGeo, postMat);
        depthBar2.rotation.z = Math.PI / 2;
        depthBar2.position.set(direction * goalDepth / 2, postHeight, GW / 2);
        group.add(depthBar2);

        // barra traseira superior
        var backBarGeo = new THREE.CylinderGeometry(thinR, thinR, GW, 6);
        var backBar = new THREE.Mesh(backBarGeo, postMat);
        backBar.rotation.x = Math.PI / 2;
        backBar.position.set(direction * goalDepth, postHeight, 0);
        group.add(backBar);

        // ── Redes ──
        // rede traseira
        var netBackGeo = new THREE.PlaneGeometry(postHeight, GW, 3, 6);
        var netBack = new THREE.Mesh(netBackGeo, netMat);
        netBack.rotation.y = Math.PI / 2;
        netBack.position.set(direction * goalDepth, postHeight / 2, 0);
        group.add(netBack);

        // rede superior
        var netTopGeo = new THREE.PlaneGeometry(goalDepth, GW, 3, 6);
        var netTop = new THREE.Mesh(netTopGeo, netMat);
        netTop.rotation.x = Math.PI / 2;
        netTop.position.set(direction * goalDepth / 2, postHeight, 0);
        group.add(netTop);

        // redes laterais
        var netSideGeo = new THREE.PlaneGeometry(goalDepth, postHeight, 3, 3);

        var netSide1 = new THREE.Mesh(netSideGeo, netMat);
        netSide1.position.set(direction * goalDepth / 2, postHeight / 2, -GW / 2);
        group.add(netSide1);

        var netSide2 = new THREE.Mesh(netSideGeo, netMat);
        netSide2.position.set(direction * goalDepth / 2, postHeight / 2, GW / 2);
        group.add(netSide2);

        group.position.set(x, 0, midZ);
        scene.add(group);
    }

    // gol esquerdo (abertura em x=0, rede para -X)
    createGoal(0, -1);
    // gol direito (abertura em x=FW, rede para +X)
    createGoal(FW, 1);
};
