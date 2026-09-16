/* Microduck Kopfstand - App (WebView/Chrome). Nutzt DuckCore (mj_core.js),
 * loadMujoco (mujoco_glue.js, WASM via data_wasm.js), DUCK_DATA (data_policy.js), THREE. */
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
var sim = null, policy = null, mj = null;
var running = false, episodeOver = false, overAt = 0;
var holdCurSteps = 0, holdBestSteps = 0, sVal = 0, scores = null;
var pendingPush = null;
var rng = DuckCore.mulberry32(1234567);
var epCount = 0, epBestAllSteps = 0;
var meshes = [];

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

    msg.textContent = 'Lade Microduck-Modell\u2026';
    sim = new DuckCore.Sim(mj, DUCK_DATA.xml, DUCK_DATA.q2);
    policy = DuckCore.makePolicy(DUCK_DATA.weights);

    msg.textContent = 'Baue 3D-Szene\u2026';
    initScene();
    initUI();

    $('load').style.opacity = '0';
    setTimeout(function () { $('load').style.display = 'none'; }, 420);
    $('app').style.visibility = 'visible';

    resetEpisode();

    if (location.search.indexOf('test=1') >= 0) runTestHook();
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
  view.insertBefore(renderer.domElement, view.firstChild);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1218);
  camera = new THREE.PerspectiveCamera(42, view.clientWidth / view.clientHeight, 0.005, 30);

  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a3040, 0.95));
  var dl = new THREE.DirectionalLight(0xffffff, 0.85);
  dl.position.set(0.6, 1.2, 0.5);
  scene.add(dl);
  var dl2 = new THREE.DirectionalLight(0x88aaff, 0.25);
  dl2.position.set(-0.6, 0.4, -0.5);
  scene.add(dl2);

  // MuJoCo (z-up) -> three (y-up)
  robotGroup = new THREE.Group();
  robotGroup.rotation.x = -Math.PI / 2;
  scene.add(robotGroup);

  var m = sim.model;
  for (var g = 0; g < sim.ngeom; g++) {
    var type = m.geom_type[g];
    var sx = m.geom_size[g * 3], sy = m.geom_size[g * 3 + 1], sz = m.geom_size[g * 3 + 2];
    var geo = null;
    if (type === DuckCore.MJ_GEOM.PLANE) {
      geo = new THREE.PlaneGeometry(Math.max(2 * (sx || 0.8), 1.4), Math.max(2 * (sy || 0.8), 1.4));
    } else if (type === DuckCore.MJ_GEOM.SPHERE) {
      geo = new THREE.SphereGeometry(sx, 28, 20);
    } else if (type === DuckCore.MJ_GEOM.CAPSULE) {
      geo = new THREE.CapsuleGeometry(sx, sz * 2, 6, 18);
      geo.rotateX(Math.PI / 2); // Kapsel-Achse Y -> Z (MuJoCo)
    } else if (type === DuckCore.MJ_GEOM.BOX) {
      geo = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    } else {
      geo = null;
    }
    if (!geo) { meshes.push(null); continue; }
    var rgba = [m.geom_rgba[g * 4], m.geom_rgba[g * 4 + 1], m.geom_rgba[g * 4 + 2], m.geom_rgba[g * 4 + 3]];
    var mat = new THREE.MeshLambertMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      transparent: rgba[3] < 0.999, opacity: rgba[3]
    });
    if (type === DuckCore.MJ_GEOM.PLANE) {
      mat.color.setHex(0x161c26); // dunkler Boden statt rgba
      mat.side = THREE.DoubleSide;
    }
    var mesh = new THREE.Mesh(geo, mat);
    robotGroup.add(mesh);
    meshes.push(mesh);
  }
  // dezentes Raster auf dem Boden
  var grid = new THREE.GridHelper(0.9, 18, 0x27303f, 0x1a212c);
  grid.position.y = 0.0005;
  robotGroup.add(grid);

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
  for (var g = 0; g < meshes.length; g++) {
    var mesh = meshes[g];
    if (!mesh) continue;
    mesh.position.set(d.geom_xpos[g * 3], d.geom_xpos[g * 3 + 1], d.geom_xpos[g * 3 + 2]);
    var o = g * 9;
    if (!_m4) _m4 = new THREE.Matrix4();
    _m4.set(
      d.geom_xmat[o], d.geom_xmat[o + 1], d.geom_xmat[o + 2], 0,
      d.geom_xmat[o + 3], d.geom_xmat[o + 4], d.geom_xmat[o + 5], 0,
      d.geom_xmat[o + 6], d.geom_xmat[o + 7], d.geom_xmat[o + 8], 0,
      0, 0, 0, 1);
    mesh.quaternion.setFromRotationMatrix(_m4);
  }
}

/* ---------------- Episode / Steuerung ---------------- */
function resetEpisode() {
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
  setStatus('mid', 'Episode ' + epCount);
}

function doControlStep() {
  if (episodeOver) return;
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
    if (episodeOver && $('cbAuto').checked && now - overAt > 1100) resetEpisode();
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
  $('btnPush').addEventListener('click', function () {
    if (!running || episodeOver) return;
    var J = parseFloat($('imp').value);
    var ang = rng() * 2 * Math.PI;
    pendingPush = [J * Math.cos(ang), J * Math.sin(ang), 0];
    flash('SCHUBSER', 'var(--accent)');
  });
}

/* ---------------- Test-Hook (?test=1) ---------------- */
function runTestHook() {
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

/* ---------------- Los ---------------- */
if (typeof DuckCore === 'undefined' || typeof loadMujoco === 'undefined' ||
    typeof MUJOCO_WASM_B64 === 'undefined' || typeof DUCK_DATA === 'undefined') {
  document.getElementById('loadMsg').textContent = 'Fehler: Assets fehlen!';
} else {
  boot();
}
