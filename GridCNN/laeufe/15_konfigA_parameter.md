# Konfiguration A — Modell, Daten, Parameter

40-Epochen-Lauf aus `15_konfigA_poc.txt`. Nicht gesetzte Flags sind die Vorgaben von `GridCNN/train.py`.

## Modell

- Architektur: `GridCNN`, Konfiguration A. Blackbox, nur `g_θ`. Der Physikterm `L(T) + Qsrc` ist aus (`--no-physics`).
- Update: `T_{t+1} = T_t + dt · g_θ`. Eine Rate je Gitterpunkt.
- Gitter: 3 × 11 × 11 = 363 Punkte.
- Eingang: 44 Kanäle. 9 Zustand (Temperatur plus zwei Lag-Raten), 17 statische Karten (Leitfähigkeit, `ρ·cp`, y/z-Koordinaten), 18 Treiber (7 Config + 11 Forcing).
- Rumpf: Breite 16, 3 Conv-Blöcke, 3×3, Reflect-Padding.
- Parameter: 11 427.
- Wand: aus. Adiabate Ablation. `--w-wall 0`.
- Koordinatenkarten: an.

## Daten

- Cache: `data_cache`, Schema v3, Subsample 10. Rohschritt 0.1 s, trainierter Schritt 1 s.
- Training, 11 OPs: OP01, OP02, OP03, OP04, OP05, OP07, OP08, OP10, OP11, OP12, OP14.
- Validierung, 2 OPs: OP06, OP09. Daran wird gerankt, nicht trainiert.
- Test, in diesem Lauf unbenutzt: OP13, OP15, OP16.
- Normierung: `T_μ = 32.99 °C`, `T_σ = 9.605 °C`. Labels sind `(T − T_μ) / T_σ`.
- OP06 nach Subsample: 1445 Schritte, 0.1 s bis 1444 s.

## Training

- Gerät: CUDA, Tesla T4. Seeds: 3 (0, 1, 2). Epochen: 40.
- Optimierer: Adam. Lernrate `1e-3`, Cosine-Abfall auf 5 % des Startwerts.
- Verlust: mittlerer quadratischer Fehler auf dem freien Rollout, gemittelt über das BPTT-Fenster. Kein Teacher Forcing.
- `--inner-steps 25`: 25 Updates je OP und Epoche, auf einer zu Epochenbeginn eingefrorenen Trajektorie.
- `--batch-t 32`: 32 Zeitindizes je Update.
- `--tbptt-start 4`, `--tbptt 16`: Fensterlänge k wächst von 4 auf 16 und bleibt ab der halben Laufzeit bei 16.
- `--clip-grad 1.0`.
- `--clamp 11`: der Rollout wird bei ±11 in normierten Einheiten festgehalten (etwa ±106 °C um `T_μ`).
- Lags: `lag1 = 1` Schritt (1 s), `lag2 = 4` Schritte (4 s).
- Gewichte: `--w-data 1`, `--w-phys 0`, `--w-wall 0`.
- `--val-every 2`: val-MAE alle 2 Epochen. Berichtet wird der Median über das letzte Drittel der Messpunkte, nicht die letzte Epoche.
- Berichtete val-MAE: OP06 6.57 ± 0.38 °C, OP09 5.81 ± 0.87 °C.
