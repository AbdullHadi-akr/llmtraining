# Konfiguration A, volle Auflösung — Modell, Daten, Parameter

60-Epochen-Lauf aus `16_konfigA_voll.txt`. Nicht gesetzte Flags sind die Vorgaben von `GridCNN/train.py`.

## ⚠ Was man über dieses Log wissen muss

- **Code-Stand vor PR #47.** Auf der Rechenmaschine war nicht gepullt. Der Trainingspfad ist identisch mit `main` nach PR #47: `rollout_loss`, `train_epoch`, `tbptt_bei`, Clipping und LR-Plan sind unverändert. Geprüft ist das per `git diff 52490c7 c9cffcd`. **Es fehlt nur die Messung aus PR #47**: kein `FEHLERPROFIL`, kein `drift`, kein Bias. Nachholen aus den Gewichten mit `GridCNN/tools/nachmessen.py`.
- **Zwei Prozesse, nicht einer** (Antwort der Maschine, 23.09., Frage 4a). Der erste Aufruf lief mit `--seeds 3`. Seed 0 und Seed 1 liefen darin vollständig durch. Während Seed 2 in **Epoche 35** war, wurde der Prozess von außen abgebrochen, ohne Traceback. Die Schlusstafel schreibt `main()` erst nach allen Seeds, deshalb fehlt sie. Seed 2 wurde gelöscht und von Epoche 1 an neu gestartet, als eigener Prozess: `--seed 2 --seeds 1 ... | tee -a 16_konfigA_voll.txt`. Die Zeilen des abgebrochenen Seed 2 stehen nicht im Log. Die Schlusstafel am Ende des Logs und `zusammenfassung.json` enthalten deshalb **nur Seed 2**. Die Tafel über alle drei Seeds steht im Bericht. Sie wurde mit `train.tafel_aus_metrics` aus den drei `metrics.json` nachgerechnet und stimmt auf vier Stellen mit den `BERICHTET`-Zeilen überein.
- **`seed0/model_best.pt` stammt nicht aus Lauf 16.** Die Datei ist von 08:10, Lauf 16 endete um 08:00. Nachgemessen ergibt sie OP06 6.5277 / OP09 8.6519 °C. Laut `metrics.json` ist das beste Modell von Seed 0 aber das aus Epoche 60, mit 5.5153 / 6.9282 °C, also identisch mit `model.pt`. Ein anderer Prozess hat die Datei überschrieben, bevor `A/` nach `A_16_voll/` gesichert wurde. **Das Profil von `seed0/model_best.pt` in [`16_konfigA_nachgemessen.txt`](16_konfigA_nachgemessen.txt) nicht verwenden.** Alle drei `model.pt` treffen die Zeile `letztes ep 60` exakt.
- **Materialdaten** (Frage 4b): echt. `PINNmodulusTwo/material_properties/constants.yaml` fasst das PDF „Material Properties Gridpoints" zusammen (Felix Schaefer, 18.06.2026). Die punktweisen Felder liegen unter `Cell Center/` und `JR1 Center/`. Das synthetische Fixture (`tools/make_synthetic_cache.py`) schreibt nur `data_cache`, nicht diese Tabellen. Die CFL-Warnung (110x) ist damit **echt** und kein Artefakt.

## Modell

Wie Lauf 15: `GridCNN`, Konfiguration A, Blackbox (`--no-physics`), 44 Eingangskanäle, Breite 16, 3 Blöcke, **11 427 Parameter**, adiabat (`--w-wall 0`), Koordinatenkarten an.

## Daten

- Cache: `data_cache`, **Subsample 2**. Rohschritt 0.1 s, trainierter Schritt **0.2 s**.
- Training: 11 OPs (OP01–05, 07, 08, 10–12, 14). Validierung: OP06, OP09. Test (OP13/15/16) unbenutzt.
- Normierung: `T_μ = 33 °C`, `T_σ = 9.602 °C`.
- Etwa 8040 Schritte je Trajektorie, fünfmal so viele wie in Lauf 15.

## Training

- Gerät: Tesla T4, torch 2.6.0+cu124. Seeds 0, 1, 2. Epochen: 60.
- Adam, `lr 1e-3`, Cosine auf 5 %. `--inner-steps 25`, `--batch-t 32`, `--clip-grad 1.0`.
- `--tbptt-start 4`, `--tbptt 16` **in Schritten**. Das sind **0.8 → 3.2 s**, in Lauf 15 waren es 4 → 16 s.
- Lags **in Sekunden** abgeleitet: `lag1 = 5` (1 s), `lag2 = 20` (4 s).
- ⚠ **`k = 16 < lag2 = 20`**: kein Update lief durch die Rückkopplung über lag2. Siehe Bericht, Abschnitt 3.
- `--clamp auto` = 11 (±106 °C um `T_μ`).
- `--val-every 2`. Berichtet wird der Median über die letzten 10 von 31 Messpunkten.

## Kommando

```bash
python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 4 --tbptt 16 \
    --val-every 2 --device cuda --cache data_cache \
    2>&1 | tee 16_konfigA_voll.txt
```

## Berichtete val-MAE (3 Seeds, aus den `BERICHTET`-Zeilen)

- OP06: **6.64 ± 1.52 °C**, Latte 10.8009, Güte **0.62x**, 3/3 Seeds unter der Latte.
- OP09: **8.60 ± 1.76 °C**, Latte 7.7625, Güte **1.11x**, 1/3 Seeds unter der Latte.
- Verdikt: **kein Ergebnis**. Die Seed-Streuung liegt über der Lesbarkeitsschwelle von ~1 °C.
