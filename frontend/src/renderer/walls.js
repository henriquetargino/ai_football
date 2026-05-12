/**
 * moldura do campo (parede + escanteios arredondados) como UM ÚNICO
 * elemento contínuo — sem junções visíveis entre as paredes axis-aligned
 * e os arcos dos cantos.
 *
 * v9.5 — refatorado pra ExtrudeGeometry com Shape contínuo. Cada metade
 * (top/bottom) é uma única mesh em formato C, conectando uma abertura
 * de gol à outra passando pelos cantos. Aberturas para gols ficam onde
 * a mesh termina (sem material, deixando espaço pra bola entrar/sair).
 */

AISoccer.createWalls = function(scene, goalWidth) {
    var FW = AISoccer.FW;
    var FH = AISoccer.FH;
    var GW = (goalWidth || 80) * AISoccer.SCALE;
    var wallH = 0.35;
    var wallThick = 0.08;

    // v9.4 — paridade com backend/config.py.
    var CUT = 30 * AISoccer.SCALE;     // raio do arco = 0.30 em 3D
    var ARC_SEGMENTS = 12;             // mais segmentos = curva mais suave

    var concreteMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.8 });
    var capMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.7 });

    var midZ = FH / 2;
    var halfGW = GW / 2;

    /**
     * adiciona pontos de arco a um Shape (via lineTo). Pula o primeiro
     * ponto (assumindo que já foi adicionado pelo lineTo/moveTo anterior).
     */
    function addArcPoints(shape, cx, cy, R, t1, t2, segs) {
        for (var i = 1; i <= segs; i++) {
            var t = t1 + (t2 - t1) * (i / segs);
            shape.lineTo(cx + R * Math.cos(t), cy + R * Math.sin(t));
        }
    }

    /**
     * cria uma metade C-shape da moldura como ExtrudeGeometry.
     * caminho interno (face que toca o campo) → ponta direita externa →
     * caminho externo (face fora do campo) → ponta esquerda externa.
     *
     * plano 2D do Shape: (sx, sy) corresponde a (x_mundo, z_mundo).
     */
    function buildHalfMolding(isTop) {
        var shape = new THREE.Shape();

        if (isTop) {
            // top half: parede topo + cantos NW e NE.
            // entrada: y=midZ-halfGW (lado superior da abertura do gol).
            shape.moveTo(0, midZ - halfGW);                    // 1. parede esquerda interna (baixa)
            shape.lineTo(0, CUT);                               // 2. sobe interna
            addArcPoints(shape, CUT, CUT, CUT,                  // 3. arco NW interno (CCW)
                         Math.PI, 3 * Math.PI / 2, ARC_SEGMENTS);
            shape.lineTo(FW - CUT, 0);                          // 4. parede topo interna
            addArcPoints(shape, FW - CUT, CUT, CUT,             // 5. arco NE interno (CCW)
                         3 * Math.PI / 2, 2 * Math.PI, ARC_SEGMENTS);
            shape.lineTo(FW, midZ - halfGW);                    // 6. desce parede direita interna

            // volta pelo externo
            shape.lineTo(FW + wallThick, midZ - halfGW);        // 7. atravessa pra externo direito
            shape.lineTo(FW + wallThick, CUT);                  // 8. sobe parede direita externa
            addArcPoints(shape, FW - CUT, CUT, CUT + wallThick, // 9. arco NE externo (CW)
                         2 * Math.PI, 3 * Math.PI / 2, ARC_SEGMENTS);
            shape.lineTo(CUT, -wallThick);                      // 10. parede topo externa
            addArcPoints(shape, CUT, CUT, CUT + wallThick,      // 11. arco NW externo (CW)
                         3 * Math.PI / 2, Math.PI, ARC_SEGMENTS);
            shape.lineTo(-wallThick, CUT);                      // 12. parede esquerda externa
            shape.lineTo(-wallThick, midZ - halfGW);            // 13. desce parede esquerda externa
            shape.closePath();
        } else {
            // bottom half: parede inferior + cantos SW e SE.
            shape.moveTo(0, midZ + halfGW);                     // parede esquerda interna (alta)
            shape.lineTo(0, FH - CUT);                          // desce interna
            addArcPoints(shape, CUT, FH - CUT, CUT,             // arco SW interno (CCW)
                         Math.PI, Math.PI / 2, ARC_SEGMENTS);
            shape.lineTo(FW - CUT, FH);                         // parede inferior interna
            addArcPoints(shape, FW - CUT, FH - CUT, CUT,        // arco SE interno (CCW)
                         Math.PI / 2, 0, ARC_SEGMENTS);
            shape.lineTo(FW, midZ + halfGW);                    // sobe parede direita interna

            // volta pelo externo
            shape.lineTo(FW + wallThick, midZ + halfGW);
            shape.lineTo(FW + wallThick, FH - CUT);
            addArcPoints(shape, FW - CUT, FH - CUT, CUT + wallThick,  // arco SE externo (CW)
                         0, Math.PI / 2, ARC_SEGMENTS);
            shape.lineTo(CUT, FH + wallThick);
            addArcPoints(shape, CUT, FH - CUT, CUT + wallThick,       // arco SW externo (CW)
                         Math.PI / 2, Math.PI, ARC_SEGMENTS);
            shape.lineTo(-wallThick, FH - CUT);
            shape.lineTo(-wallThick, midZ + halfGW);
            shape.closePath();
        }

        // body (corpo da parede, extrudido na vertical).
        // extrudeGeometry extrude no +Z local. Após rotateX(+π/2), o eixo
        // y do shape vira Z do mundo (mantém o footprint), e o eixo Z do
        // shape (depth) vira -Y do mundo (geometria fica de Y=-wallH a 0).
        // translate(0, wallH, 0) move pra Y=0..wallH (acima do chão).
        var bodyGeo = new THREE.ExtrudeGeometry(shape, {
            depth: wallH,
            bevelEnabled: false,
            curveSegments: 1
        });
        bodyGeo.rotateX(Math.PI / 2);
        bodyGeo.translate(0, wallH, 0);
        bodyGeo.computeVertexNormals();
        var body = new THREE.Mesh(bodyGeo, concreteMat);
        body.castShadow = true;
        body.receiveShadow = true;
        scene.add(body);

        // cap (tampa em cima — mais escura, dá contraste estético)
        var capGeo = new THREE.ShapeGeometry(shape);
        capGeo.rotateX(Math.PI / 2);
        var cap = new THREE.Mesh(capGeo, capMat);
        cap.position.y = wallH + 0.005;
        scene.add(cap);
    }

    buildHalfMolding(true);   // top half (NW, NE, parede topo)
    buildHalfMolding(false);  // bottom half (SW, SE, parede inferior)
};
