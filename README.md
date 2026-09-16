# Microduck Kopfstand — Android-APK

**Dateien:**
| Datei | Beschreibung |
|---|---|
| `microduck-kopfstand.apk` | Android-App (minSdk 24 / Android 7.0+, arm64/arm/x86 — WebView-basiert, kein NDK nötig) |
| `microduck-kopfstand-web.zip` | Dieselbe App als reine Web-Variante (Ordner entpacken, `index.html` in Chrome öffnen) |

## Installation

APK auf das Android-Gerät kopieren und antippen. Da die APK selbst signiert ist:
„Installieren aus unbekannten Quellen" für den Dateimanager erlauben.
**Keine Internet-Berechtigung nötig** — Physik, Modell und Policy sind komplett offline eingebettet.

## Was die App macht

Der Microduck startet in der balancierten Kopfstand-Pose (q*), getragen von der
gelieferten Policy (`policy.onnx` als nativer JS-MLP, 52→64→64→14, Tanh):

- **Start / Reset** — neue Episode (Start-Tilt per Slider 0–4°, optional Zufallsachse)
- **Schubser** — Impuls 0.3–2.5 N·s in zufällige horizontale Richtung (wie im Training)
- **Obs-Rauschen** — realistisches IMU/Gelenk-Rauschen wie in der Eval (Standard: aus)
- **Zeitlupe ¼** — den Nadelhaarsitz-Pfad beobachten
- HUD: Score **s** (Produkt-Metrik aus dem Training), Teil-Scores (Invert/Kopf/Rumpf/Füße),
  Hold-Zeit aktuell/best, Kipp-Meldung + Auto-Reset

Kamera: 1 Finger = drehen, 2 Finger/Pinch = zoomen.

## Ehrliche Erwartung

Die Policy ist ein **Hold-Spezialist ab Kopfstandlage**: typisch **~1 s Halt** (beste Episoden
bis ~1.5 s), danach kippt der Kopf seitlich weg. Das ist kein Bug, sondern das in der
Entwicklung dokumentierte Physik-Ceiling: Der Kopf ist eine Kugel (r=41.7 mm) mit
rolling friction 1e-4 auf dem Boden — ein ungedämpftes Nadel-Gleichgewicht (λ≈7–14/s),
die Beine haben kaum seitliche CoM-Autorität, die Nackengelenke fast die gesamte.
Alle Reglerklassen (quasi-statisch, LQR, CMA-ES, BC, PPO) plateauten bei 0.7–1.1 s
(siehe `README.md` im selben Ordner).

## Architektur (100 % offline im Gerät)

```
Android Activity (WebView, fullscreen)
 └─ file:///android_asset/index.html
     ├─ three.js r147          → 3D-Renderer (Geoms generisch aus dem MJCF)
     ├─ @mujoco/mujoco 3.13.0  → MuJoCo-WASM (9.8 MB, base64 eingebettet, Apache-2.0)
     ├─ mj_core.js             → Env-Port: obs(52), Score s, 50 Hz Control-Loop,
     │                           Position-Actuatoren kp=0.55, ±0.96 Nm, 10 Substeps à 5 ms
     └─ data_policy.js         → Policy-Gewichte aus policy.onnx (num. verifiziert,
                                 max-Abweichung 6.6e-7 vs. ONNX-Referenz)
```

## Verifikationskette (vor dem APK-Bau bestanden)

1. **Gewichte:** NumPy-Forward vs. onnxruntime: max |Δ| = 6.6e-7 ✔
2. **Node-Parität** (WASM + JS-Port vs. echte Python-Env + ONNX):
   Hold 1.14 s vs. 1.14 s (0° Tilt), 0.66 s vs. 0.66 s (2° Tilt), obs exakt,
   Score-Profil |Δ| ≤ 2.4e-3, gleiche Termination ✔
   (Float-Chaos-Divergenz erst ab ~t=100 Schritten — erwartbar, nativ vs. wasm Rundung)
3. **Chromium-Browser-Test** (identischer Stack wie im WebView):
   Live-„HALT ✓" s=0.91, Best-Hold 1.14 s; Szenarien base/2°/Noise = 1.14/0.66/0.96 s ✔
4. **APK:** `apksigner verify` (v2+v3), Badging geprüft, Asset-CRCs byte-identisch ✔

## Repro-Build

```bash
apkbuild/build_assets.sh   # Assets: Glue-Patch, Policy-Daten, WASM-Base64
apkbuild/make_icon.py      # Launcher-Icons
apkbuild/build_apk.sh      # aapt2 → ECJ → d8 → zipalign → apksigner (kein Gradle)
```
Paritätstests: `apkbuild/core/test_parity.py` (Python-Referenz) +
`apkbuild/core/test_node.cjs` (JS/WASM-Gegenstück), Browser-Test-Hook: `index.html?test=1`.
