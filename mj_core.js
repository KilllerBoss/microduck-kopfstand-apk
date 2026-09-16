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

  // ---------------- Referenz-Policy (61 -> 512 -> 256 -> 128 -> 14, Elu) ----------------
  // pollen-robotics/microduck (BEST_alpha_stand, roulade). Graph:
  // x=(obs-mean)/std -> GEMM/Elu x3 -> GEMM -> act. ctrl = DEFAULT_POSE + act.
  // Gewichte: b64-encodierte Float32-Arrays, row-major [out,in] (transB=1).
  function b64ToF32(b64) {
    var bin = atob(b64), n = bin.length;
    var buf = new ArrayBuffer(n);
    var u8 = new Uint8Array(buf);
    for (var i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }
  function decodeRef(R) {
    var out = {};
    for (var k in R) {
      if (k === 'n1' || k === 'n2' || k === 'n3' || k === 'nOut' || k === 'nIn') out[k] = R[k];
      else out[k] = b64ToF32(R[k]);
    }
    return out;
  }
  function makeRefPolicy(RAW) {
    var W = decodeRef(RAW);
    var nIn = W.nIn, n1 = W.n1, n2 = W.n2, n3 = W.n3, nOut = W.nOut;
    var h1 = new Float64Array(n1), h2 = new Float64Array(n2), h3 = new Float64Array(n3);
    return {
      act: function (obs) {
        var i, j;
        var x = new Float64Array(nIn);
        for (i = 0; i < nIn; i++) {
          x[i] = clamp((obs[i] - W.mean[i]) / W.std[i], -1e4, 1e4);
        }
        for (i = 0; i < n1; i++) {
          var off = i * nIn, acc = W.b1[i];
          for (j = 0; j < nIn; j++) acc += W.W1[off + j] * x[j];
          h1[i] = acc > 0 ? acc : Math.expm1(acc); // Elu(alpha=1)
        }
        for (i = 0; i < n2; i++) {
          var off2 = i * n1, acc2 = W.b2[i];
          for (j = 0; j < n1; j++) acc2 += W.W2[off2 + j] * h1[j];
          h2[i] = acc2 > 0 ? acc2 : Math.expm1(acc2);
        }
        for (i = 0; i < n3; i++) {
          var off3 = i * n2, acc3 = W.b3[i];
          for (j = 0; j < n2; j++) acc3 += W.W3[off3 + j] * h2[j];
          h3[i] = acc3 > 0 ? acc3 : Math.expm1(acc3);
        }
        var out = new Float64Array(nOut);
        for (i = 0; i < nOut; i++) {
          var off4 = i * n3, acc4 = W.b4[i];
          for (j = 0; j < n3; j++) acc4 += W.W4[off4 + j] * h3[j];
          out[i] = acc4;
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
    this._lastAref = new Float64Array(this.nu);
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

  // ---------------- Referenz-Interface (61D obs + DEFAULT_POSE-Off-Konvention) ----------------
  var REF_CMD = null; // 13 Nullen, lazy
  Sim.prototype.obs61 = function () {
    var d = this.data;
    var obs = new Float64Array(61);
    var R = this._tmpR || (this._tmpR = new Float64Array(9));
    var tb = this.trunkBid * 9;
    for (var i = 0; i < 9; i++) R[i] = d.xmat[tb + i];
    // [0..2] base_ang_vel (gyro im Trunk-Frame == qvel[3:6], imu-Site hat Identitaetsquat)
    obs[0] = d.qvel[3]; obs[1] = d.qvel[4]; obs[2] = d.qvel[5];
    // [3..5] projected gravity
    obs[3] = -R[6]; obs[4] = -R[7]; obs[5] = -R[8];
    // [6..19] joint_pos - DEFAULT_POSE
    for (var q = 0; q < this.nu; q++) obs[6 + q] = d.qpos[7 + q] - SEQ_DEFAULT_POSE[q];
    // [20..33] joint_vel
    for (var qd = 0; qd < this.nu; qd++) obs[20 + qd] = d.qvel[6 + qd];
    // [34..47] last_action
    for (var la = 0; la < this.nu; la++) obs[34 + la] = this._lastAref[la];
    // [48..60] command: Nullen (kein Input, kein Head-Mode)
    for (var c = 0; c < 13; c++) obs[48 + c] = 0;
    return obs;
  };

  // Referenz-Step: ctrl = DEFAULT_POSE + act (KEIN q_range-Clip, wie Referenz), 10 Substeps
  Sim.prototype.stepRef = function (act) {
    var d = this.data, a = new Float64Array(this.nu);
    for (var i = 0; i < this.nu; i++) a[i] = act[i];
    for (var k = 0; k < this.nu; k++) d.ctrl[k] = SEQ_DEFAULT_POSE[k] + a[k];
    for (var st = 0; st < this.nSub; st++) this.mj.mj_step(this.model, this.data);
    this.t++;
    this._lastAref = a;
    return { terminated: (!isFinite(d.qpos[0]) || !isFinite(d.qpos[2])) };
  };

  // Physik-Experiment: Kopf-Kugel-Kontakt Rolling-Widerstand (condim 6).
  // friction[geom] = [slide, torsional, rolling]; kombiniert per Element-MAX im Kontakt.
  // roll == 0 -> condim 3 restauriert (exakt Originalphysik, keine Solver-Differenz).
  Sim.prototype.setHeadRolling = function (slide, roll) {
    var m = this.model;
    m.geom_condim[this.headG] = roll > 0 ? 6 : 3;
    m.geom_friction[this.headG * 3] = slide;
    m.geom_friction[this.headG * 3 + 1] = 0.005;
    m.geom_friction[this.headG * 3 + 2] = roll;
  };

  Sim.prototype.projGravZ = function () {
    var R = this._tmpR4 || (this._tmpR4 = new Float64Array(9));
    var tb = this.trunkBid * 9;
    for (var i = 0; i < 9; i++) R[i] = this.data.xmat[tb + i];
    return -R[8];
  };

  Sim.prototype.headZ = function () {
    var hg = this.headG * 3;
    return this.data.geom_xpos[hg + 2];
  };

  Sim.prototype.feetZmin = function () {
    var zmin = 1e9;
    for (var s2 = 0; s2 < this.soleGids.length; s2++) {
      var g = this.soleGids[s2], gm = g * 9, gp = g * 3;
      var corners = this.soleCorners[s2];
      for (var ci = 0; ci < corners.length; ci++) {
        var cn = corners[ci];
        var wz = this.data.geom_xmat[gm + 6] * cn[0] + this.data.geom_xmat[gm + 7] * cn[1] + this.data.geom_xmat[gm + 8] * cn[2]
               + this.data.geom_xpos[gp + 2];
        if (wz < zmin) zmin = wz;
      }
    }
    return zmin;
  };

  // ---------------- Sequenz: STAND -> Kippen -> Kopf-Kontakt -> Beine hoch -> Release ----------------
  // Portiert aus scripts/seq_test/seq_proto.py (Zweipivot-Flip, bewiesen).
  // Phasen: 'stand' (Physik, ctrl=DEFAULT_POSE) -> 'A' Kippen um Zehenspitzen bis
  // Kopfkontakt -> 'B' Rotation um Kopf-Kugelzentrum + Beine hochziehen ->
  // 'C' Blend auf exakten Eval-Init -> Release (App ruft sim.reset() und schaltet
  // auf Policy). Alles kinematisch (mj_forward), kein Bodendurchbruch (Ground-Clamp).

  var SEQ_DEFAULT_POSE = [0, -0.0872665, -0.457924, -0.00494, 0.452984,
                          0.349066, 0.349066, 0, 0, 0, 0.0872665, 0.457924, 0.00494, -0.452984];
  var SEQ_N_STAND = 25, SEQ_N_A = 55, SEQ_N_B = 75, SEQ_N_C = 20;
  var LEG_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13];
  var NECK_IDX = [5, 6, 7, 8];

  function seqSmooth(u) { return u * u * (3 - 2 * u); }
  function seqSlerp(q0, q1, t) {
    var dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    dot = Math.max(-1, Math.min(1, dot));
    var th = Math.acos(dot);
    if (th < 1e-6) return [(1 - t) * q0[0] + t * q1[0], (1 - t) * q0[1] + t * q1[1],
                           (1 - t) * q0[2] + t * q1[2], (1 - t) * q0[3] + t * q1[3]];
    var s = Math.sin(th);
    var a = Math.sin((1 - t) * th) / s, b = Math.sin(t * th) / s;
    return [a * q0[0] + b * q1[0], a * q0[1] + b * q1[1], a * q0[2] + b * q1[2], a * q0[3] + b * q1[3]];
  }

  function makeSequence(sim) {
    var mj = sim.mj, m = sim.model, d = sim.data;
    var nu = sim.nu;
    var st = {
      phase: 'idle', k: 0,
      thetaLandDeg: 0, headPen: 0, feetPen: 0,
      fp: null, axis: null, t: null
    };

    function setQposAll(pos, quat, joints) {
      d.qpos[0] = pos[0]; d.qpos[1] = pos[1]; d.qpos[2] = pos[2];
      d.qpos[3] = quat[0]; d.qpos[4] = quat[1]; d.qpos[5] = quat[2]; d.qpos[6] = quat[3];
      for (var j = 0; j < nu; j++) d.qpos[7 + j] = joints[j];
    }
    function headPos() {
      var hg = sim.headG * 3;
      return [d.geom_xpos[hg], d.geom_xpos[hg + 1], d.geom_xpos[hg + 2]];
    }
    function feetZmin() {
      var zmin = 1e9;
      for (var s2 = 0; s2 < sim.soleGids.length; s2++) {
        var g = sim.soleGids[s2], gm = g * 9, gp = g * 3;
        var corners = sim.soleCorners[s2];
        for (var ci = 0; ci < corners.length; ci++) {
          var cn = corners[ci];
          var wz = d.geom_xmat[gm + 6] * cn[0] + d.geom_xmat[gm + 7] * cn[1] + d.geom_xmat[gm + 8] * cn[2]
                 + d.geom_xpos[gp + 2];
          if (wz < zmin) zmin = wz;
        }
      }
      return zmin;
    }
    function groundClear() {
      var fz = feetZmin();
      if (fz < -0.0015) {
        d.qpos[2] += (-0.0015 - fz);
        mj.mj_forward(m, d);
      }
    }
    function axisAngleQuatW(axis, ang) { return axisAngleQuat(axis, ang, [0, 0, 0, 0]); }
    function rotAbout(pStart, pivot, axis, theta) {
      var n = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) || 1;
      var ax = [axis[0] / n, axis[1] / n, axis[2] / n];
      var c = Math.cos(theta), s = Math.sin(theta);
      // R = I c + s K + (1-c) K^2  (Rodrigues)
      var K = [0, -ax[2], ax[1], ax[2], 0, -ax[0], -ax[1], ax[0], 0];
      var K2 = [
        K[0] * K[0] + K[1] * K[3] + K[2] * K[6], K[0] * K[1] + K[1] * K[4] + K[2] * K[7], K[0] * K[2] + K[1] * K[5] + K[2] * K[8],
        K[3] * K[0] + K[4] * K[3] + K[5] * K[6], K[3] * K[1] + K[4] * K[4] + K[5] * K[7], K[3] * K[2] + K[4] * K[5] + K[5] * K[8],
        K[6] * K[0] + K[7] * K[3] + K[8] * K[6], K[6] * K[1] + K[7] * K[4] + K[8] * K[7], K[6] * K[2] + K[7] * K[5] + K[8] * K[8]];
      var R = [
        c + (1 - c) * K2[0], s * K[1] + (1 - c) * K2[1], s * K[2] + (1 - c) * K2[2],
        s * K[3] + (1 - c) * K2[3], c + (1 - c) * K2[4], s * K[5] + (1 - c) * K2[5],
        s * K[6] + (1 - c) * K2[6], s * K[7] + (1 - c) * K2[7], c + (1 - c) * K2[8]];
      var v = [pStart[0] - pivot[0], pStart[1] - pivot[1], pStart[2] - pivot[2]];
      return [pivot[0] + R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
              pivot[1] + R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
              pivot[2] + R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
    }

    var pStart, fp, axis, tdir, thetaLand, pLand, hcLand, posEnd, quatEnd;

    // begin(): nach dem Stand-Settle aufrufen (misst Pivot/Achse/Bisektion)
    function begin() {
      var hc0 = headPos();
      pStart = [d.qpos[0], d.qpos[1], d.qpos[2]];

      // Sohlen-Mitten und Tipprichtung senkrecht zur Fuss-Trennachse (beide
      // Fuesse bleiben auf der Rotationsachse), in Kopfrichtung orientiert
      var c0 = sim.soleGids.map(function (g) {
        var gp = g * 3; return [d.geom_xpos[gp], d.geom_xpos[gp + 1], d.geom_xpos[gp + 2]];
      });
      var mid = [(c0[0][0] + c0[1][0]) / 2, (c0[0][1] + c0[1][1]) / 2, 0];
      var dv = [hc0[0] - mid[0], hc0[1] - mid[1], hc0[2] - mid[2]];
      var sep = [c0[1][0] - c0[0][0], c0[1][1] - c0[0][1]];
      var nsep = Math.hypot(sep[0], sep[1]) || 1;
      sep = [sep[0] / nsep, sep[1] / nsep];
      tdir = [-sep[1], sep[0]];
      if (tdir[0] * dv[0] + tdir[1] * dv[1] < 0) tdir = [-tdir[0], -tdir[1]];
      axis = [tdir[1], -tdir[0], 0];

      // Pivot = Vorderkante der Sohlen in Tipprichtung (Zehenspitzen)
      var pts = [];
      for (var s3 = 0; s3 < sim.soleGids.length; s3++) {
        var g = sim.soleGids[s3], gm = g * 9, gp = g * 3;
        var cc = [d.geom_xpos[gp], d.geom_xpos[gp + 1], d.geom_xpos[gp + 2]];
        var best = -1e9;
        var corners = sim.soleCorners[s3];
        for (var ci = 0; ci < corners.length; ci++) {
          var cn = corners[ci];
          var wx = d.geom_xmat[gm] * cn[0] + d.geom_xmat[gm + 1] * cn[1] + d.geom_xmat[gm + 2] * cn[2];
          var wy = d.geom_xmat[gm + 3] * cn[0] + d.geom_xmat[gm + 4] * cn[1] + d.geom_xmat[gm + 5] * cn[2];
          var proj = (wx) * tdir[0] + (wy) * tdir[1];
          if (proj > best) best = proj;
        }
        pts.push([cc[0] + tdir[0] * best, cc[1] + tdir[1] * best]);
      }
      fp = [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2, 0];

      // Vorzeichencheck: Kopf muss sinken
      var pt = rotAbout(pStart, fp, axis, 0.1);
      setQposAll(pt, axisAngleQuatW(axis, 0.1), SEQ_DEFAULT_POSE);
      mj.mj_forward(m, d);
      if (headPos()[2] > hc0[2]) axis = [-axis[0], -axis[1], -axis[2]];

      // theta_land: Kopf-Kugel beruehrt Boden (Bisektion)
      function headZAt(theta) {
        var pos = rotAbout(pStart, fp, axis, theta);
        setQposAll(pos, axisAngleQuatW(axis, theta), SEQ_DEFAULT_POSE);
        mj.mj_forward(m, d);
        groundClear();
        return headPos()[2];
      }
      var lo = 0, hi = Math.PI / 2;
      for (var it = 0; it < 40; it++) {
        var mid2 = 0.5 * (lo + hi);
        if (headZAt(mid2) > sim.headR) lo = mid2; else hi = mid2;
      }
      thetaLand = 0.5 * (lo + hi);
      st.thetaLandDeg = thetaLand * 180 / Math.PI;

      // Endzustand (Eval-Init) fuer Kopf-Ziel/Blend
      mj.mj_resetData(m, d);
      for (var i = 0; i < sim.nq; i++) d.qpos[i] = sim.hsQ[i];
      for (var j2 = 0; j2 < nu; j2++) d.qpos[7 + j2] = sim.q2[j2];
      d.qpos[2] += 0.001;
      mj.mj_forward(m, d);
      posEnd = [d.qpos[0], d.qpos[1], d.qpos[2]];
      quatEnd = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];

      st.phase = 'A'; st.k = 0;
      st.headPen = 0; st.feetPen = 0;
      // Zurueck in den gemessenen Stand (App macht Physik-Settle separat)
    }

    function trackPen() {
      var hp = headPos(), fz = feetZmin();
      st.headPen = Math.min(st.headPen, hp[2] - sim.headR);
      st.feetPen = Math.min(st.feetPen, fz);
    }

    function step() {
      var k = st.k;
      if (st.phase === 'A') {
        var u = seqSmooth((k + 1) / SEQ_N_A);
        var th = thetaLand * u;
        var pos = rotAbout(pStart, fp, axis, th);
        var quat = axisAngleQuatW(axis, th);
        var joints = SEQ_DEFAULT_POSE.slice();
        for (var ni = 0; ni < NECK_IDX.length; ni++) {
          var i5 = NECK_IDX[ni];
          joints[i5] = (1 - u) * SEQ_DEFAULT_POSE[i5] + u * sim.q2[i5];
        }
        setQposAll(pos, quat, joints);
        mj.mj_forward(m, d);
        groundClear();
        pLand = [d.qpos[0], d.qpos[1], d.qpos[2]];
        hcLand = headPos();
        trackPen();
        if (k + 1 >= SEQ_N_A) { st.phase = 'B'; st.k = 0; return; }
      } else if (st.phase === 'B') {
        var b = seqSmooth((k + 1) / SEQ_N_B);
        var th2 = thetaLand + b * (Math.PI - thetaLand);
        var pos2 = rotAbout(pLand, hcLand, axis, th2 - thetaLand);
        var quat2 = axisAngleQuatW(axis, th2);
        var uLegs = 0.55 + 0.45 * seqSmooth(Math.min(b / 0.45, 1.0));
        var joints2 = SEQ_DEFAULT_POSE.slice();
        for (var li = 0; li < LEG_IDX.length; li++) {
          var i6 = LEG_IDX[li];
          joints2[i6] = (1 - uLegs) * SEQ_DEFAULT_POSE[i6] + uLegs * sim.q2[i6];
        }
        for (var ni2 = 0; ni2 < NECK_IDX.length; ni2++) joints2[NECK_IDX[ni2]] = sim.q2[NECK_IDX[ni2]];
        setQposAll(pos2, quat2, joints2);
        mj.mj_forward(m, d);
        groundClear();
        trackPen();
        if (k + 1 >= SEQ_N_B) { st.phase = 'C'; st.k = 0; return; }
      } else if (st.phase === 'C') {
        var b2 = seqSmooth((k + 1) / SEQ_N_C);
        var q0 = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
        var qm = seqSlerp(q0, quatEnd, b2);
        var p0 = [d.qpos[0], d.qpos[1], d.qpos[2]];
        var jm = [];
        for (var j3 = 0; j3 < nu; j3++) {
          jm.push((1 - b2) * d.qpos[7 + j3] + b2 * sim.q2[j3]);
        }
        setQposAll([p0[0] + b2 * (posEnd[0] - p0[0]), p0[1] + b2 * (posEnd[1] - p0[1]), p0[2] + b2 * (posEnd[2] - p0[2])], qm, jm);
        mj.mj_forward(m, d);
        trackPen();
        if (k + 1 >= SEQ_N_C) {
          st.phase = 'release'; st.k = 0;
          return;
        }
      }
      st.k++;
    }

    st.begin = begin;
    st.step = step;
    st.headPos = headPos;
    st.feetZmin = feetZmin;
    return st;
  }

  Sim.prototype.resetStand = function () {
    var d = this.data;
    this.mj.mj_resetData(this.model, this.data);
    for (var i = 0; i < this.nq; i++) d.qpos[i] = this.standQ[i];
    this.mj.mj_forward(this.model, this.data);
    this._lastA = new Float64Array(this.nu);
    this._lastAref = new Float64Array(this.nu);
    this._errPrev = null;
    this.t = 0;
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
    makeSequence: makeSequence, SEQ_DEFAULT_POSE: SEQ_DEFAULT_POSE, SEQ_N_STAND: SEQ_N_STAND,
    makeRefPolicy: makeRefPolicy,
    REF_DEFAULT_POSE: SEQ_DEFAULT_POSE,
    mulberry32: mulberry32, gaussPair: gaussPair,
    MJ_GEOM: { PLANE: MJ_GEOM_PLANE, SPHERE: MJ_GEOM_SPHERE, CAPSULE: MJ_GEOM_CAPSULE, BOX: MJ_GEOM_BOX }
  };
});
