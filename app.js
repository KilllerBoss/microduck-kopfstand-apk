/* Microduck Kopfstand - App (WebView/Chrome). Nutzt DuckCore (mj_core.js),
 * loadMujoco (mujoco_glue.js, WASM via data_wasm.js), DUCK_DATA (data_policy.js),
 * REF_DATA (data_ref.js: BEST_alpha_stand + roulade), THREE.
 *
 * Kopfstand-Ablauf (Policy-Kette):
 *   STEHEN   - BEST_alpha_stand.onnx balanciert im Stand (1.2 s)
 *   ROULADE  - roulade.onnx kippt zurueck; Uebergabe, sobald Kopf Boden
 *              beruehrt UND die Beine drueber sind (Apex-Catch)
 *   UEBERNAHME - kurze Ueberfuehrung in den trainierten Kopfstand-Init (0.4 s)
 *   KOPFSTAND- meine Policy (52D) balanciert auf dem Kopf; Kopfkontakt hat
 *              Rolling-Widerstand (condim 6) -> lange Haltezeit
 *   Sturz    - Auto-Reset, Ablauf startet von neu (Loop)
 */
'use strict';

/* ---------------- Hilfen ---------------- */
function $(id) { return document.getElementById(id); }
function b64ToUint8(b64) {
  var bin = atob(b64), n = bin.length, u = new Uint8Array(n);
  for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
  return u;
}
function setBar(id, v) { $(id).firstChild.style.width = (100 * Math.max(0, Math.min(1, v))).toFixed(1) + '%'; }

/* ---------------- Globale App-Zustände ---------------- */
var sim = null, policy = null, standPol = null, rollPol = null, mj = null;
var running = false, episodeOver = false, overAt = 0;
var chainMode = false, chainPhase = 'idle', chainSteps = 0, chainLowSteps = 0;
var blendData = null;
var holdCurSteps = 0, holdBestSteps = 0, sVal = 0, scores = null;
var pendingPush = null;
var rng = DuckCore.mulberry32(1234567);
var epCount = 0, epBestAllSteps = 0;
var bodyGroups = [], floorMesh = null;

var CHAIN_STAND_STEPS = 60;   // 1.2 s stehen
var CHAIN_ROLL_MAX = 150;     // 3 s Rouladen-Fenster (wie Referenz)
var CHAIN_BLEND_STEPS = 20;   // 0.4 s Uebernahme
var HEAD_ROLLING = 0.3;       // Rolling-Widerstand Kopfkontakt (Haltephase)

function radians(deg) { return deg * Math.PI / 180; }

/* ---------------- Boot ---------------- */
async function boot() {
  var msg = $('loadMsg');
  try {
    msg.textContent = 'Dekodiere MuJoCo WASM (10 MB)\u2026';
    await new Promise(function (r) { setTimeout(r, 30); });
    var bin = b64ToUint8(MUJOCO_WASM_B64);
    msg.textContent = 'Initialisiere MuJoCo WASM\u2026';
    await new Promise(function (r) { setTimeout(r, 30); });
    mj = await loadMujoco({ wasmBinary: bin });

    msg.textContent = 'Lade Microduck-Modell + 3 Policies\u2026';
    sim = new DuckCore.Sim(mj, DUCK_DATA.xml, DUCK_DATA.q2);
    policy = DuckCore.makePolicy(DUCK_DATA.weights);          // Kopfstand (52D)
    standPol = DuckCore.makeRefPolicy(REF_DATA.stand);        // Stehen (61D, Referenz)
    rollPol = DuckCore.makeRefPolicy(REF_DATA.roll);          // Roulade (61D, Referenz)

    msg.textContent = 'Baue 3D-Szene\u2026';
    initScene();
    initUI();

    $('load').style.opacity = '0';
    setTimeout(function () { $('load').style.display = 'none'; }, 420);
    $('app').style.visibility = 'visible';

    resetEpisode();

    if (location.search.indexOf('test=1') >= 0 || location.search.indexOf('test=2') >= 0) runTestHook();
    requestAnimationFrame(frame);
  } catch (e) {
    msg.innerHTML = 'Fehler: ' + (e && e.message ? e.message : e);
    throw e;
  }
}

/* ---------------- three.js Szene ---------------- */
var renderer, scene, camera, robotGroup;
var orbit = { theta: 0.7, phi: 0.95, r: 0.5, target: new THREE.Vector3(0, 0.08, 0) };

function initScene() {
  var view = $('view');
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(view.clientWidth, view.clientHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;          // wie der Original-Space-Renderer
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  view.insertBefore(renderer.domElement, view.firstChild);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0.0036, 0.0062, 0.0105); // kalibriert: ergibt ~0x0b0f16 nach ACES+sRGB
  camera = new THREE.PerspectiveCamera(42, view.clientWidth / view.clientHeight, 0.005, 30);

  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a3040, 0.75));
  var dl = new THREE.DirectionalLight(0xffffff, 0.7);
  dl.position.set(0.6, 1.2, 0.5);
  scene.add(dl);
  var dl2 = new THREE.DirectionalLight(0x88aaff, 0.3);
  dl2.position.set(-0.6, 0.4, -0.5);
  scene.add(dl2);

  // MuJoCo (z-up) -> three (y-up)
  robotGroup = new THREE.Group();
  robotGroup.rotation.x = -Math.PI / 2;
  scene.add(robotGroup);

  var m = sim.model;
  // Nur der Boden kommt aus der Physik-Szene; der Roboter wird als
  // Original-Microduck-GLB-Meshes (pollen-robotics) gerendert.
  for (var g = 0; g < sim.ngeom; g++) {
    if (m.geom_type[g] !== DuckCore.MJ_GEOM.PLANE) continue;
    var sx = m.geom_size[g * 3], sy = m.geom_size[g * 3 + 1];
    var geoF = new THREE.PlaneGeometry(Math.max(2 * (sx || 0.8), 1.4), Math.max(2 * (sy || 0.8), 1.4));
    var matF = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.0036, 0.0062, 0.0105), side: THREE.DoubleSide }); // kalibriert dunkel
    floorMesh = new THREE.Mesh(geoF, matF);
    robotGroup.add(floorMesh);
    break;
  }
  // dezentes Raster auf dem Boden
  var grid = new THREE.GridHelper(0.9, 18, 0x1c2530, 0x121821);
  grid.position.y = 0.0005;
  robotGroup.add(grid);

  buildRobotVisuals();

  updateCamera();
  window.addEventListener('resize', onResize);
  bindOrbit(view);
}

function onResize() {
  var view = $('view');
  if (!view.clientWidth) return;
  renderer.setSize(view.clientWidth, view.clientHeight);
  camera.aspect = view.clientWidth / view.clientHeight;
  camera.updateProjectionMatrix();
}

function updateCamera() {
  var sp = Math.sin(orbit.phi), cp = Math.cos(orbit.phi);
  camera.position.set(
    orbit.target.x + orbit.r * sp * Math.sin(orbit.theta),
    orbit.target.y + orbit.r * cp,
    orbit.target.z + orbit.r * sp * Math.cos(orbit.theta));
  camera.lookAt(orbit.target);
}

function bindOrbit(el) {
  var ptrs = {}, lastPinch = 0;
  el.addEventListener('pointerdown', function (e) { ptrs[e.pointerId] = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId); });
  el.addEventListener('pointermove', function (e) {
    if (!ptrs[e.pointerId]) return;
    var ids = Object.keys(ptrs);
    if (ids.length === 1) {
      var p = ptrs[e.pointerId];
      orbit.theta -= (e.clientX - p[0]) * 0.008;
      orbit.phi = Math.max(0.15, Math.min(1.45, orbit.phi - (e.clientY - p[1]) * 0.008));
      ptrs[e.pointerId] = [e.clientX, e.clientY];
      updateCamera();
    } else if (ids.length === 2) {
      ptrs[e.pointerId] = [e.clientX, e.clientY];
      var a = ptrs[ids[0]], b = ptrs[ids[1]];
      var d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (lastPinch > 0) {
        orbit.r = Math.max(0.18, Math.min(3.0, orbit.r * lastPinch / d));
        updateCamera();
      }
      lastPinch = d;
    }
  });
  var up = function (e) { delete ptrs[e.pointerId]; lastPinch = 0; };
  el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  el.addEventListener('wheel', function (e) {
    e.preventDefault();
    orbit.r = Math.max(0.18, Math.min(3.0, orbit.r * (e.deltaY > 0 ? 1.08 : 0.93)));
    updateCamera();
  }, { passive: false });
}

var _m4 = null;
function syncMeshes() {
  var d = sim.data;
  if (!_m4) _m4 = new THREE.Matrix4();
  for (var b = 0; b < bodyGroups.length; b++) {
    var grp = bodyGroups[b];
    if (!grp) continue;
    grp.position.set(d.xpos[b * 3], d.xpos[b * 3 + 1], d.xpos[b * 3 + 2]);
    var o = b * 9;
    _m4.set(
      d.xmat[o], d.xmat[o + 1], d.xmat[o + 2], 0,
      d.xmat[o + 3], d.xmat[o + 4], d.xmat[o + 5], 0,
      d.xmat[o + 6], d.xmat[o + 7], d.xmat[o + 8], 0,
      0, 0, 0, 1);
    grp.quaternion.setFromRotationMatrix(_m4);
  }
}

/* ---------------- Original-Microduck-Visuals (pollen-robotics) ---------------- */
function typedFromBin(u8, byteOff, count, compType) {
  var Ctor = compType === 5126 ? Float32Array : (compType === 5125 ? Uint32Array : Uint16Array);
  var esz = compType === 5121 ? 1 : (compType === 5123 ? 2 : 4);
  if (byteOff % esz === 0) return new Ctor(u8.buffer, u8.byteOffset + byteOff, count);
  var out = new Ctor(count), dv = new DataView(u8.buffer, u8.byteOffset);
  for (var i = 0; i < count; i++) {
    out[i] = compType === 5126 ? dv.getFloat32(byteOff + i * 4, true)
           : compType === 5125 ? dv.getUint32(byteOff + i * 4, true)
           : dv.getUint16(byteOff + i * 2, true);
  }
  return out;
}

function buildRobotVisuals() {
  var u8 = b64ToUint8(DUCK_VISUAL.bin);
  var geoms = {};
  var names = Object.keys(DUCK_VISUAL.meshes);
  for (var i = 0; i < names.length; i++) {
    var e = DUCK_VISUAL.meshes[names[i]];
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(typedFromBin(u8, e.p[0], e.p[1], e.p[2]), 3));
    if (e.n) g.setAttribute('normal', new THREE.BufferAttribute(typedFromBin(u8, e.n[0], e.n[1], e.n[2]), 3));
    g.setIndex(new THREE.BufferAttribute(typedFromBin(u8, e.i[0], e.i[1], e.i[2]), 1));
    if (!e.n) g.computeVertexNormals();
    geoms[names[i]] = g;
  }
  for (var bi = 0; bi < DUCK_VISUAL.bodies.length; bi++) {
    var B = DUCK_VISUAL.bodies[bi];
    var bid = sim.mj.mj_name2id(sim.model, 1, B.name); // 1 = mjOBJ_BODY
    if (bid < 0) continue;
    var grp = new THREE.Group();
    for (var gi = 0; gi < B.geoms.length; gi++) {
      var G = B.geoms[gi];
      var mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(G[8], G[9], G[10]).convertSRGBToLinear(),
        roughness: 0.55, metalness: 0.15
      });
      var mesh = new THREE.Mesh(geoms[G[0]], mat);
      mesh.position.set(G[1], G[2], G[3]);
      mesh.quaternion.set(G[5], G[6], G[7], G[4]); // MJCF (w,x,y,z) -> THREE (x,y,z,w)
      grp.add(mesh);
    }
    robotGroup.add(grp);
    bodyGroups[bid] = grp;
  }
}

/* ---------------- Episode / Steuerung ---------------- */
function resetEpisode() {
  chainMode = false; chainPhase = 'idle';
  sim.setHeadRolling(1.0, 0); // Rolling aus (nur Haltephase)
  $('tilt').disabled = false; $('cbRand').disabled = false;
  var tiltDeg = parseFloat($('tilt').value);
  var tiltRad = 0, axis = [1, 0, 0];
  if ($('cbRand').checked && tiltDeg > 0) {
    // Zufallsachse horizontal, Betrag wie Slider
    var ang = rng() * 2 * Math.PI;
    tiltRad = radians(tiltDeg);
    axis = [Math.cos(ang), Math.sin(ang), 0];
  } else if (tiltDeg > 0) {
    tiltRad = radians(tiltDeg);
  }
  sim.reset({ tiltRad: tiltRad, tiltAxis: axis, fall: 0.001, jointNoise: 0, velNoise: 0, rng: rng });
  holdCurSteps = 0; holdBestSteps = 0; episodeOver = false;
  pendingPush = null; sVal = 0; scores = sim.computeScores();
  epCount++;
  running = true;
  setStatus('mid', 'Direkt: Episode ' + epCount);
}

/* Policy-Kette: STEHEN -> ROULADE -> UEBERNAHME -> KOPFSTAND (-> Loop) */
function startChain() {
  chainMode = true; chainPhase = 'chainStand'; chainSteps = 0;
  sim.setHeadRolling(1.0, 0);
  $('tilt').disabled = true; $('cbRand').disabled = true;
  sim.resetStand();
  holdCurSteps = 0; holdBestSteps = 0; episodeOver = false;
  pendingPush = null; sVal = 0; scores = sim.computeScores();
  epCount++;
  running = true;
  setStatus('mid', 'STEHEN (Policy 1)');
}

function doControlStep() {
  if (episodeOver) return;
  if (chainMode) { doChainStep(); return; }
  var obs = sim.obs52();
  if ($('cbNoise').checked) {
    for (var n1 = 0; n1 < 3; n1++) obs[n1] += 0.03 * DuckCore.gaussPair(rng);
    for (var n2 = 0; n2 < 3; n2++) obs[3 + n2] += 0.08 * DuckCore.gaussPair(rng);
    for (var n3 = 0; n3 < 14; n3++) obs[6 + n3] += 0.01 * DuckCore.gaussPair(rng);
    for (var n4 = 0; n4 < 14; n4++) obs[20 + n4] += 0.15 * DuckCore.gaussPair(rng);
  }
  var a = policy.act(obs);
  var push = null;
  if (pendingPush) { push = pendingPush; pendingPush = null; }
  var r = sim.stepControl(a, push);
  sVal = r.s; scores = r.scores;
  if (r.s > 0.5) {
    holdCurSteps++;
    if (holdCurSteps > holdBestSteps) holdBestSteps = holdCurSteps;
    if (holdBestSteps > epBestAllSteps) epBestAllSteps = holdBestSteps;
  } else holdCurSteps = 0;
  if (r.terminated) {
    episodeOver = true; overAt = performance.now();
    setStatus('bad', 'Gekippt');
    flash('GEKIPPT', 'var(--bad)');
  } else if (sVal > 0.5) setStatus('ok', 'HALT \u2713');
  else if (sVal > 0.05) setStatus('mid', 'Instabil');
}

/* ---- Policy-Kette ---- */
function doChainStep() {
  var i;
  if (chainPhase === 'chainStand') {
    var rs = sim.stepRef(standPol.act(sim.obs61()));
    sVal = 0; scores = null;
    if (rs.terminated) { chainFail('STAND gekippt'); return; }
    if (++chainSteps >= CHAIN_STAND_STEPS) {
      chainPhase = 'chainRoll'; chainSteps = 0;
      setStatus('mid', 'ROULADE (Policy 2)');
    }
    return;
  }
  if (chainPhase === 'chainRoll') {
    var r = sim.stepRef(rollPol.act(sim.obs61()));
    var sc0 = sim.computeScores(); sVal = sc0.s; scores = sc0;
    var gyro = Math.hypot(sim.data.qvel[3], sim.data.qvel[4], sim.data.qvel[5]);
    var hz = sim.headZ(), fz = sim.feetZmin(), pgz = sim.projGravZ();
    // Uebergabe: Kopf beruehrt Boden UND Beine drueber (Apex)
    if (chainSteps > 8 && pgz > 0.65 && fz > hz + 0.03 && gyro < 5.0 && hz - sim.headR < 0.012) {
      // Blend-Daten einfrieren
      var q0 = [sim.data.qpos[3], sim.data.qpos[4], sim.data.qpos[5], sim.data.qpos[6]];
      var j0 = [];
      for (i = 0; i < sim.nu; i++) j0.push(sim.data.qpos[7 + i]);
      blendData = { p0: [sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2]], q0: q0, j0: j0 };
      chainPhase = 'chainBlend'; chainSteps = 0;
      setStatus('mid', '\u00dcBERNAHME');
      flash('KOPF AM BODEN \u2013 POLICY KOMMT', 'var(--accent)');
      return;
    }
    if (r.terminated || ++chainSteps >= CHAIN_ROLL_MAX) { chainFail('Roulade fehlgeschlagen'); return; }
    return;
  }
  if (chainPhase === 'chainBlend') {
    // Ueberfuehrung in den trainierten Kopfstand-Init (hsQ + q2)
    var hsQ = sim.hsQ, u = (chainSteps + 1) / CHAIN_BLEND_STEPS;
    var e = u * u * (3 - 2 * u);
    var p0 = blendData.p0, q0b = blendData.q0, j0 = blendData.j0;
    sim.data.qpos[0] = p0[0] + e * (hsQ[0] - p0[0]);
    sim.data.qpos[1] = p0[1] + e * (hsQ[1] - p0[1]);
    sim.data.qpos[2] = p0[2] + e * (hsQ[2] - p0[2]);
    var d = q0b[0] * hsQ[3] + q0b[1] * hsQ[4] + q0b[2] * hsQ[5] + q0b[3] * hsQ[6];
    d = Math.max(-1, Math.min(1, d));
    var th = Math.acos(d), sn = Math.sin(th) || 1e-9;
    var qa = Math.sin((1 - e) * th) / sn, qb = Math.sin(e * th) / sn;
    sim.data.qpos[3] = qa * q0b[0] + qb * hsQ[3];
    sim.data.qpos[4] = qa * q0b[1] + qb * hsQ[4];
    sim.data.qpos[5] = qa * q0b[2] + qb * hsQ[5];
    sim.data.qpos[6] = qa * q0b[3] + qb * hsQ[6];
    for (i = 0; i < sim.nu; i++) sim.data.qpos[7 + i] = j0[i] + e * (hsQ[7 + i] - j0[i]);
    sim.mj.mj_forward(sim.model, sim.data);
    var scb = sim.computeScores(); sVal = scb.s; scores = scb;
    if (++chainSteps >= CHAIN_BLEND_STEPS) {
      for (var v = 0; v < sim.data.qvel.length; v++) sim.data.qvel[v] = 0;
      sim.mj.mj_forward(sim.model, sim.data);
      sim._lastA = new Float64Array(sim.q2); sim._errPrev = null;
      sim.setHeadRolling(1.0, HEAD_ROLLING); // Rolling-Widerstand fuer lange Haltezeit
      chainPhase = 'chainHold'; chainSteps = 0; chainLowSteps = 0;
      holdCurSteps = 0;
      setStatus('ok', 'KOPFSTAND (meine Policy)');
      flash('POLICY \u00dcBERNIMMT', 'var(--ok)');
    }
    return;
  }
  if (chainPhase === 'chainHold') {
    var obs = sim.obs52();
    if ($('cbNoise').checked) {
      for (var m1 = 0; m1 < 3; m1++) obs[m1] += 0.03 * DuckCore.gaussPair(rng);
      for (var m2 = 0; m2 < 3; m2++) obs[3 + m2] += 0.08 * DuckCore.gaussPair(rng);
      for (var m3 = 0; m3 < 14; m3++) obs[6 + m3] += 0.01 * DuckCore.gaussPair(rng);
      for (var m4 = 0; m4 < 14; m4++) obs[20 + m4] += 0.15 * DuckCore.gaussPair(rng);
    }
    var a = policy.act(obs);
    var push = null;
    if (pendingPush) { push = pendingPush; pendingPush = null; }
    var rh = sim.stepControl(a, push);
    sVal = rh.s; scores = rh.scores;
    if (rh.s > 0.5) {
      holdCurSteps++; chainLowSteps = 0;
      if (holdCurSteps > holdBestSteps) holdBestSteps = holdCurSteps;
      if (holdBestSteps > epBestAllSteps) epBestAllSteps = holdBestSteps;
    } else { holdCurSteps = 0; chainLowSteps++; }
    if (rh.terminated || chainLowSteps >= 75) { // 1.5 s unter Schwelle = Sturz
      sim.setHeadRolling(1.0, 0);
      episodeOver = true; overAt = performance.now();
      setStatus('bad', 'Gekippt \u2013 Loop startet neu');
      flash('GEKIPPT', 'var(--bad)');
    } else if (sVal > 0.5) setStatus('ok', 'KOPFSTAND \u2713');
    else if (sVal > 0.05) setStatus('mid', 'KOPFSTAND: Instabil');
  }
}

function chainFail(reason) {
  sim.setHeadRolling(1.0, 0);
  episodeOver = true; overAt = performance.now();
  setStatus('bad', reason + ' \u2013 Loop startet neu');
  flash('FEHLGESCHLAGEN', 'var(--bad)');
}

function frame(now) {
  requestAnimationFrame(frame);
  var dt = Math.min((now - (frame._t || now)) / 1000, 0.1);
  frame._t = now;
  if (running) {
    var acc = (frame._acc || 0) + dt * ($('cbSlow').checked ? 0.25 : 1);
    var guard = 0;
    while (acc >= DuckCore.CONTROL_DT && guard++ < 40) {
      acc -= DuckCore.CONTROL_DT;
      doControlStep();
    }
    frame._acc = acc;
    if (episodeOver && $('cbAuto').checked && now - overAt > 1100) {
      if (chainMode) startChain(); else resetEpisode();
    }
  }
  syncMeshes();
  hud();
  renderer.render(scene, camera);
}

/* ---------------- HUD ---------------- */
function setStatus(cls, txt) {
  var el = $('status');
  el.className = cls; el.textContent = txt;
}
function flash(txt, color) {
  var el = $('flash');
  el.textContent = txt; el.style.color = color; el.style.opacity = '1';
  setTimeout(function () { el.style.opacity = '0'; }, 700);
}
function hud() {
  setBar('barS', sVal);
  $('sVal').textContent = sVal.toFixed(2);
  if (scores) {
    setBar('barInv', scores.inv); setBar('barHead', scores.head_down);
    setBar('barBody', scores.body_up); setBar('barFeet', scores.feet_up);
  }
  $('holdCur').textContent = (holdCurSteps * 0.02).toFixed(2);
  $('holdBest').textContent = (holdBestSteps * 0.02).toFixed(2);
  $('epT').textContent = (sim.t * 0.02).toFixed(1);
}

/* ---------------- UI ---------------- */
function initUI() {
  $('tilt').addEventListener('input', function () {
    $('tiltV').textContent = parseFloat(this.value).toFixed(1) + '\u00b0';
  });
  $('imp').addEventListener('input', function () {
    $('impV').textContent = parseFloat(this.value).toFixed(1) + ' Ns';
  });
  $('btnReset').addEventListener('click', resetEpisode);
  $('btnSeq').addEventListener('click', startChain);
  $('btnPush').addEventListener('click', function () {
    if (!running || episodeOver) return;
    var J = parseFloat($('imp').value);
    var ang = rng() * 2 * Math.PI;
    pendingPush = [J * Math.cos(ang), J * Math.sin(ang), 0];
    flash('SCHUBSER', 'var(--accent)');
  });
}

/* ---------------- Test-Hook (?test=1 Paritaet, ?test=2 Kette) ---------------- */
function runTestHook() {
  if (location.search.indexOf('test=2') >= 0) { runTestHookChain(); return; }
  var results = [];
  var scen = [
    { name: 'base_tilt0', tilt: 0, noise: false },
    { name: 'tilt2deg', tilt: 2.0, noise: false },
    { name: 'tilt0_noise', tilt: 0, noise: true }
  ];
  for (var i = 0; i < scen.length; i++) {
    var sc = scen[i];
    sim.reset({ tiltRad: radians(sc.tilt), tiltAxis: [1, 0, 0], fall: 0.001, rng: rng });
    var holdSteps = 0, holdBest = 0, cur = 0, sMax = 0, term = false, steps = 0;
    for (var t = 0; t < 500; t++) {
      var obs = sim.obs52();
      if (sc.noise) {
        for (var a1 = 0; a1 < 3; a1++) obs[a1] += 0.03 * DuckCore.gaussPair(rng);
        for (var a2 = 0; a2 < 3; a2++) obs[3 + a2] += 0.08 * DuckCore.gaussPair(rng);
        for (var a3 = 0; a3 < 14; a3++) obs[6 + a3] += 0.01 * DuckCore.gaussPair(rng);
        for (var a4 = 0; a4 < 14; a4++) obs[20 + a4] += 0.15 * DuckCore.gaussPair(rng);
      }
      var act = policy.act(obs);
      var r = sim.stepControl(act);
      steps++; sMax = Math.max(sMax, r.s);
      if (r.s > 0.5) { cur++; holdSteps++; if (cur > holdBest) holdBest = cur; } else cur = 0;
      if (r.terminated) { term = true; break; }
    }
    results.push({ name: sc.name, holdBest: +(holdBest * 0.02).toFixed(2), holdTotal: +(holdSteps * 0.02).toFixed(2), sMax: +sMax.toFixed(3), steps: steps, terminated: term });
  }
  window.__TEST = { done: true, results: results };
  console.log('TEST-RESULT', JSON.stringify(window.__TEST));
}

/* Ketten-Test (?test=2): STEHEN -> ROULADE -> UEBERNAHME -> KOPFSTAND headless. */
function runTestHookChain() {
  var i, t;
  var res = { standOK: false, standPGZ: 0, catchStep: -1, blendPGZ: 0, holdBest: 0, holdTotal: 0, terminated: false };
  // STAND
  sim.resetStand();
  for (t = 0; t < CHAIN_STAND_STEPS; t++) {
    var rs = sim.stepRef(standPol.act(sim.obs61()));
    if (rs.terminated) break;
  }
  res.standOK = sim.projGravZ() < -0.85 && sim.data.qpos[2] > 0.02;
  res.standPGZ = +sim.projGravZ().toFixed(3);
  // ROULADE + Catch
  for (t = 0; t < CHAIN_ROLL_MAX; t++) {
    var rr = sim.stepRef(rollPol.act(sim.obs61()));
    var gyro = Math.hypot(sim.data.qvel[3], sim.data.qvel[4], sim.data.qvel[5]);
    var hz = sim.headZ(), fz = sim.feetZmin(), pgz = sim.projGravZ();
    if (t > 8 && pgz > 0.65 && fz > hz + 0.03 && gyro < 5.0 && hz - sim.headR < 0.012) { res.catchStep = t; break; }
    if (rr.terminated) break;
  }
  // UEBERNAHME (Blend)
  if (res.catchStep >= 0) {
    var hsQ = sim.hsQ;
    var p0 = [sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2]];
    var q0 = [sim.data.qpos[3], sim.data.qpos[4], sim.data.qpos[5], sim.data.qpos[6]];
    var j0 = [];
    for (i = 0; i < sim.nu; i++) j0.push(sim.data.qpos[7 + i]);
    for (var s = 0; s < CHAIN_BLEND_STEPS; s++) {
      var u = (s + 1) / CHAIN_BLEND_STEPS, e = u * u * (3 - 2 * u);
      sim.data.qpos[0] = p0[0] + e * (hsQ[0] - p0[0]);
      sim.data.qpos[1] = p0[1] + e * (hsQ[1] - p0[1]);
      sim.data.qpos[2] = p0[2] + e * (hsQ[2] - p0[2]);
      var dq = q0[0] * hsQ[3] + q0[1] * hsQ[4] + q0[2] * hsQ[5] + q0[3] * hsQ[6];
      dq = Math.max(-1, Math.min(1, dq));
      var th = Math.acos(dq), sn = Math.sin(th) || 1e-9;
      var qa = Math.sin((1 - e) * th) / sn, qb = Math.sin(e * th) / sn;
      sim.data.qpos[3] = qa * q0[0] + qb * hsQ[3];
      sim.data.qpos[4] = qa * q0[1] + qb * hsQ[4];
      sim.data.qpos[5] = qa * q0[2] + qb * hsQ[5];
      sim.data.qpos[6] = qa * q0[3] + qb * hsQ[6];
      for (i = 0; i < sim.nu; i++) sim.data.qpos[7 + i] = j0[i] + e * (hsQ[7 + i] - j0[i]);
      sim.mj.mj_forward(sim.model, sim.data);
    }
    for (var v = 0; v < sim.data.qvel.length; v++) sim.data.qvel[v] = 0;
    sim.mj.mj_forward(sim.model, sim.data);
    res.blendPGZ = +sim.projGravZ().toFixed(3);
    // KOPFSTAND (10 s Fenster)
    sim.setHeadRolling(1.0, HEAD_ROLLING);
    sim._lastA = new Float64Array(sim.q2); sim._errPrev = null;
    var holdCur = 0;
    for (t = 0; t < 500; t++) {
      var rh = sim.stepControl(policy.act(sim.obs52()));
      if (rh.s > 0.5) { holdCur++; res.holdTotal++; if (holdCur > res.holdBest) res.holdBest = holdCur; } else holdCur = 0;
      if (rh.terminated) { res.terminated = true; break; }
    }
    sim.setHeadRolling(1.0, 0);
  }
  res.holdBest = +(res.holdBest * 0.02).toFixed(2);
  res.holdTotal = +(res.holdTotal * 0.02).toFixed(2);
  window.__TEST = { done: true, chain: res,
    go: res.standOK && res.catchStep >= 0 && res.blendPGZ > 0.95 && res.holdBest >= 5.0 };
  console.log('TEST-CHAIN', JSON.stringify(window.__TEST));
}

/* ---------------- Los ---------------- */
if (typeof DuckCore === 'undefined' || typeof loadMujoco === 'undefined' ||
    typeof MUJOCO_WASM_B64 === 'undefined' || typeof DUCK_DATA === 'undefined' ||
    typeof REF_DATA === 'undefined' || typeof DUCK_VISUAL === 'undefined') {
  document.getElementById('loadMsg').textContent = 'Fehler: Assets fehlen!';
} else {
  boot();
}
