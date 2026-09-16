/* Microduck Kopfstand - gemeinsamer Core (Env + Policy) für Node-Test und WebView-APK.
 * Portiert aus duck/headstand_env.py + scripts/eval_onnx52.py (Raw52Env, eval_mode).
 * UMD: in Node require()-bar, im Browser global DuckCore.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DuckCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var CONTROL_DT = 0.02;          // 50 Hz
  var CONTROL_HZ = 50;
  var MJ_GEOM_PLANE = 0, MJ_GEOM_SPHERE = 2, MJ_GEOM_CAPSULE = 3, MJ_GEOM_BOX = 6;
  var MJ_OBJ_BODY = 1;

  // ---------------- kleine Mathe-Helfer (doppelte Genauigkeit) ----------------
  function rotVecByMat(R, v, out) { // R: row-major 3x3
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    return out;
  }
  function rotVecByMatT(R, v, out) { // R^T @ v
    out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
    return out;
  }
  function quatMul(q, r, out) { // (w,x,y,z)
    var qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    var rw = r[0], rx = r[1], ry = r[2], rz = r[3];
    out[0] = qw * rw - qx * rx - qy * ry - qz * rz;
    out[1] = qw * rx + qx * rw + qy * rz - qz * ry;
    out[2] = qw * ry - qx * rz + qy * rw + qz * rx;
    out[3] = qw * rz + qx * ry - qy * rx + qz * rw;
    return out;
  }
  function axisAngleQuat(axis, ang, out) {
    var n = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) || 1;
    var s = Math.sin(ang / 2);
    out[0] = Math.cos(ang / 2);
    out[1] = axis[0] / n * s; out[2] = axis[1] / n * s; out[3] = axis[2] / n * s;
    return out;
  }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // Deterministischer RNG (mulberry32)
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussPair(rng) { // Box-Muller
    var u = Math.max(rng(), 1e-12), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------------- Policy-MLP (52 -> 64 -> 64 -> 14, Tanh) ----------------
  // x = clip((obs - mean)/std, +-10); h=tanh; out = ((tanh(x W1'+b1)) W2'+b2) W3'+b3
  function makePolicy(W) {
    var n1 = W.W1.length, n2 = W.W2.length, n3 = W.W3.length; // 64,64,14 Zeilen
    var d = W.W1[0].length;                                   // 52
    var h1 = new Float64Array(n1), h2 = new Float64Array(n2), out = new Float64Array(n3);
    return {
      act: function (obs) {
        var i, j;
        var h1r = new Float64Array(d);
        for (i = 0; i < d; i++) {
          h1r[i] = clamp((obs[i] - W.obs_mean[i]) / W.obs_std[i], -10, 10);
        }
        for (i = 0; i < n1; i++) {
          var row = W.W1[i], acc = W.b1[i];
          for (j = 0; j < d; j++) acc += row[j] * h1r[j];
          h1[i] = Math.tanh(acc);
        }
        for (i = 0; i < n2; i++) {
          var row2 = W.W2[i], acc2 = W.b2[i];
          for (j = 0; j < n1; j++) acc2 += row2[j] * h1[j];
          h2[i] = Math.tanh(acc2);
        }
        for (i = 0; i < n3; i++) {
          var row3 = W.W3[i], acc3 = W.b3[i];
          for (j = 0; j < n2; j++) acc3 += row3[j] * h2[j];
          out[i] = acc3;
        }
        return out;
      }
    };
  }

  // ---------------- Simulation (Env-Port) ----------------
  function Sim(mj, xmlString, q2) {
    this.mj = mj;
    this.model = mj.MjModel.from_xml_string(xmlString);
    this.data = new mj.MjData(this.model);
    var m = this.model, d = this.data;

    this.nq = m.nq; this.nv = m.nv; this.nu = m.nu; this.ngeom = m.ngeom;
    this.timestep = m.opt.timestep;              // 0.005
    this.nSub = Math.round(CONTROL_DT / this.timestep); // 10

    // IDs
    this.trunkBid = mj.mj_name2id(m, MJ_OBJ_BODY, 'trunk_base');
    this.jawBid = mj.mj_name2id(m, MJ_OBJ_BODY, 'jaw_soft');
    this.ankleBids = [
      mj.mj_name2id(m, MJ_OBJ_BODY, 'ankle_left'),
      mj.mj_name2id(m, MJ_OBJ_BODY, 'ankle_right')
    ];

    // Kopf-Kugel (Sphere-Geom des jaw-Body)
    this.headG = -1;
    for (var g = 0; g < this.ngeom; g++) {
      if (m.geom_bodyid[g] === this.jawBid && m.geom_type[g] === MJ_GEOM_SPHERE) { this.headG = g; break; }
    }
    this.headR = m.geom_size[this.headG * 3];
    this.headC = [m.geom_pos[this.headG * 3], m.geom_pos[this.headG * 3 + 1], m.geom_pos[this.headG * 3 + 2]];

    // Sohlen-Geoms (Boxen der Ankle-Bodies)
    this.soleGids = [];
    for (var g2 = 0; g2 < this.ngeom; g2++) {
      if (m.geom_bodyid[g2] === this.ankleBids[0] || m.geom_bodyid[g2] === this.ankleBids[1]) {
        if (m.geom_type[g2] === MJ_GEOM_BOX) this.soleGids.push(g2);
      }
    }
    this.soleCorners = this.soleGids.map(function (g) {
      var sx = m.geom_size[g * 3], sy = m.geom_size[g * 3 + 1], sz = m.geom_size[g * 3 + 2];
      var cs = [];
      [-1, 1].forEach(function (a) { [-1, 1].forEach(function (b) { [-1, 1].forEach(function (c) {
        cs.push([a * sx, b * sy, c * sz]); }); }); });
      return cs;
    });

    // Gelenk-Limits ueber Aktuatoren (wie env.q_range)
    this.qLo = new Float64Array(this.nu); this.qHi = new Float64Array(this.nu);
    for (var a = 0; a < this.nu; a++) {
      var j = m.actuator_trnid[2 * a];
      this.qLo[a] = m.jnt_range[2 * j]; this.qHi[a] = m.jnt_range[2 * j + 1];
    }

    // Keyframes
    this.standQ = Array.from(m.key_qpos.slice(0, this.nq));
    this.hsQ = Array.from(m.key_qpos.slice(this.nq, 2 * this.nq));
    this.q2 = Array.from(q2);

    this._lastA = new Float64Array(this.nu);
    this._errPrev = null;
    this.t = 0;
    this._tmp3 = new Float64Array(3);
    this._tmp3b = new Float64Array(3);
    this._diff3 = new Float64Array(3);
  }

  Sim.prototype.reset = function (opts) {
    opts = opts || {};
    var d = this.data, q2 = this.q2, hs = this.hsQ;
    this.mj.mj_resetData(this.model, this.data);
    for (var i = 0; i < this.nq; i++) d.qpos[i] = hs[i];
    for (var j = 0; j < this.nu; j++) d.qpos[7 + j] = q2[j];
    for (var v = 0; v < this.nv; v++) d.qvel[v] = 0.0;
    var fall = opts.fall !== undefined ? opts.fall : 0.001;
    d.qpos[2] += fall;
    var tilt = opts.tiltRad || 0;
    if (tilt > 0) {
      var axis = opts.tiltAxis || [1, 0, 0];
      var qn = axisAngleQuat(axis, tilt, [0, 0, 0, 0]);
      var q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
      var qr = quatMul(qn, q, [0, 0, 0, 0]);
      var n = Math.sqrt(qr[0] * qr[0] + qr[1] * qr[1] + qr[2] * qr[2] + qr[3] * qr[3]) || 1;
      d.qpos[3] = qr[0] / n; d.qpos[4] = qr[1] / n; d.qpos[5] = qr[2] / n; d.qpos[6] = qr[3] / n;
    }
    if (opts.jointNoise && opts.rng) {
      for (var k = 0; k < this.nu; k++) d.qpos[7 + k] += opts.jointNoise * gaussPair(opts.rng);
    }
    if (opts.velNoise && opts.rng) {
      for (var k2 = 0; k2 < this.nu; k2++) d.qvel[6 + k2] += opts.velNoise * gaussPair(opts.rng);
    }
    this.mj.mj_forward(this.model, this.data);
    this._lastA = new Float64Array(this.nu);
    this._errPrev = null;
    this.t = 0;
  };

  // 52-dim RAW obs (Raw52Env._obs)
  Sim.prototype.obs52 = function () {
    var d = this.data, m = this.model;
    var obs = new Float64Array(52);
    var R = this._tmpR || (this._tmpR = new Float64Array(9));
    var tb = this.trunkBid * 9;
    for (var i = 0; i < 9; i++) R[i] = d.xmat[tb + i];
    // proj_grav = R^T @ [0,0,-1] = [-R6,-R7,-R8]
    obs[0] = -R[6]; obs[1] = -R[7]; obs[2] = -R[8];
    // gyro = qvel[3:6] (Body-Frame, Free Joint)
    obs[3] = d.qvel[3]; obs[4] = d.qvel[4]; obs[5] = d.qvel[5];
    for (var q = 0; q < this.nu; q++) obs[6 + q] = d.qpos[7 + q];
    for (var qd = 0; qd < this.nu; qd++) obs[20 + qd] = d.qvel[6 + qd];
    for (var la = 0; la < this.nu; la++) obs[34 + la] = this._lastA[la];
    // CoM-Features (Trunk-Frame)
    var tmp = this._tmp3, tmp2 = this._tmp3b, df = this._diff3;
    var tp = this.trunkBid * 3;
    var p = [d.xpos[tp], d.xpos[tp + 1], d.xpos[tp + 2]];
    var com = [d.subtree_com[0], d.subtree_com[1], d.subtree_com[2]];
    var hg = this.headG * 3;
    var hc = [d.geom_xpos[hg], d.geom_xpos[hg + 1], d.geom_xpos[hg + 2]];
    rotVecByMatT(R, [com[0] - p[0], com[1] - p[1], com[2] - p[2]], tmp);
    rotVecByMatT(R, [hc[0] - p[0], hc[1] - p[1], hc[2] - p[2]], tmp2);
    var ex = tmp[0] - tmp2[0], ey = tmp[1] - tmp2[1];
    if (this._errPrev === null) { df[0] = 0; df[1] = 0; }
    else {
      df[0] = (ex - this._errPrev[0]) / CONTROL_DT;
      df[1] = (ey - this._errPrev[1]) / CONTROL_DT;
    }
    this._errPrev = [ex, ey];
    obs[48] = ex / 0.05; obs[49] = ey / 0.05;
    obs[50] = df[0] / 0.1; obs[51] = df[1] / 0.1;
    return obs;
  };

  // Scores (identisch zu _scores())
  Sim.prototype.computeScores = function () {
    var d = this.data, m = this.model;
    var tb = this.trunkBid * 9;
    var R = this._tmpR2 || (this._tmpR2 = new Float64Array(9));
    for (var i = 0; i < 9; i++) R[i] = d.xmat[tb + i];
    // trunk z-Achse in Welt = R @ [0,0,1] = [R2,R5,R8]
    var dot = -R[8];
    var inv = clamp((dot - 0.5) / 0.45, 0, 1);
    // Kopf-Zentrum z
    var jb = this.jawBid * 9, jp = this.jawBid * 3, hg = this.headG * 3;
    var Rj = this._tmpR3 || (this._tmpR3 = new Float64Array(9));
    for (var i2 = 0; i2 < 9; i2++) Rj[i2] = d.xmat[jb + i2];
    var c = this.headC;
    var hz = d.xpos[jp + 2] + Rj[2] * c[0] + Rj[5] * c[1] + Rj[8] * c[2];
    var headDown = clamp((this.headR + 0.025 - hz) / 0.025, 0, 1);
    var trunkZ = d.qpos[2];
    var bodyUp = clamp((trunkZ - hz - 0.06) / 0.04, 0, 1);
    // Sohlen-Tiefpunkt
    var zmin = 1e9;
    for (var sIdx = 0; sIdx < this.soleGids.length; sIdx++) {
      var g = this.soleGids[sIdx];
      var gm = g * 9, gp = g * 3;
      var corners = this.soleCorners[sIdx];
      for (var ci = 0; ci < corners.length; ci++) {
        var cn = corners[ci];
        var wz = d.geom_xmat[gm + 6] * cn[0] + d.geom_xmat[gm + 7] * cn[1] + d.geom_xmat[gm + 8] * cn[2]
               + d.geom_xpos[gp + 2];
        if (wz < zmin) zmin = wz;
      }
    }
    var feetUp = clamp((zmin - trunkZ - 0.01) / 0.06, 0, 1);
    var s = inv * headDown * bodyUp * feetUp;
    var phi = 0.5 * inv + 0.2 * headDown + 0.15 * bodyUp + 0.15 * feetUp;
    return { s: s, phi: phi, inv: inv, head_down: headDown, body_up: bodyUp, feet_up: feetUp, head_z: hz, feet_min_z: zmin };
  };

  // Ein Control-Step (50 Hz): Push + 10 Substeps; wie env.step()
  Sim.prototype.stepControl = function (action, pushImpulse) {
    var d = this.data, m = this.model, a = new Float64Array(this.nu);
    for (var i = 0; i < this.nu; i++) a[i] = clamp(action[i], -1, 1);
    // Push (trunk, 3 Control-Steps lang)
    if (pushImpulse) {
      var f = [pushImpulse[0] / (3 * CONTROL_DT), pushImpulse[1] / (3 * CONTROL_DT), pushImpulse[2] / (3 * CONTROL_DT)];
      var off = this.trunkBid * 6;
      d.xfrc_applied[off] = f[0]; d.xfrc_applied[off + 1] = f[1]; d.xfrc_applied[off + 2] = f[2];
      this._pushLeft = 3; this._pushF = [0, 0, 0];
    } else if (this._pushLeft > 0) {
      this._pushLeft--;
      if (this._pushLeft === 0) {
        var off2 = this.trunkBid * 6;
        d.xfrc_applied[off2] = 0; d.xfrc_applied[off2 + 1] = 0; d.xfrc_applied[off2 + 2] = 0;
      }
    }
    // Zielwinkel clippen (q_range) -> ctrl
    for (var k = 0; k < this.nu; k++) d.ctrl[k] = clamp(a[k], this.qLo[k], this.qHi[k]);
    for (var st = 0; st < this.nSub; st++) this.mj.mj_step(this.model, this.data);
    this.t++;
    this._lastA = a;
    var sc = this.computeScores();
    var finite = isFinite(d.qpos[0]) && isFinite(d.qpos[1]) && isFinite(d.qpos[2]);
    var terminated = (!finite) || d.qpos[2] < -0.02;
    return { s: sc.s, scores: sc, terminated: terminated };
  };

  // ---------------- Episode-Runner (Test/Headless) ----------------
  function runEpisode(sim, policy, opts) {
    opts = opts || {};
    sim.reset({
      tiltRad: opts.tiltRad || 0, fall: opts.fall !== undefined ? opts.fall : 0.001,
      jointNoise: opts.jointNoise || 0, velNoise: opts.velNoise || 0, rng: opts.rng
    });
    var maxSteps = opts.maxSteps || 750;
    var holdSteps = 0, holdCur = 0, holdBest = 0, sMax = 0, sSum = 0, term = false;
    var log = opts.log ? [] : null;
    for (var t = 0; t < maxSteps; t++) {
      var obs = sim.obs52();
      if (opts.obsNoise && opts.rng) {
        // eval_noise: proj 0.03, gyro 0.08, q 0.01, qd 0.15
        for (var n1 = 0; n1 < 3; n1++) obs[n1] += 0.03 * gaussPair(opts.rng);
        for (var n2 = 0; n2 < 3; n2++) obs[3 + n2] += 0.08 * gaussPair(opts.rng);
        for (var n3 = 0; n3 < 14; n3++) obs[6 + n3] += 0.01 * gaussPair(opts.rng);
        for (var n4 = 0; n4 < 14; n4++) obs[20 + n4] += 0.15 * gaussPair(opts.rng);
      }
      if (log) log.push({ qpos: Array.from(sim.data.qpos), s: sim.computeScores().s, act: Array.from(sim._lastA) });
      var a = policy.act(obs);
      var r = sim.stepControl(a);
      sSum += r.s; sMax = Math.max(sMax, r.s);
      if (r.s > 0.5) { holdCur++; holdSteps++; holdBest = Math.max(holdBest, holdCur); }
      else holdCur = 0;
      if (r.terminated) { term = true; break; }
    }
    return { holdBest: holdBest * CONTROL_DT, holdTotal: holdSteps * CONTROL_DT, sMean: sSum / Math.max(1, sim.t),
             sMax: sMax, steps: sim.t, terminated: term, log: log };
  }

  return {
    CONTROL_DT: CONTROL_DT, CONTROL_HZ: CONTROL_HZ,
    Sim: Sim, makePolicy: makePolicy, runEpisode: runEpisode,
    mulberry32: mulberry32, gaussPair: gaussPair,
    MJ_GEOM: { PLANE: MJ_GEOM_PLANE, SPHERE: MJ_GEOM_SPHERE, CAPSULE: MJ_GEOM_CAPSULE, BOX: MJ_GEOM_BOX }
  };
});
