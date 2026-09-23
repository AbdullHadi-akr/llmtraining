# Konfiguration A, volle Auflösung, Fenster in Sekunden — Modell, Daten, Parameter

60-Epochen-Lauf aus `17_konfigA_voll_k_sekunden.txt`. Nicht gesetzte Flags sind die Vorgaben von `GridCNN/train.py`.

**Eine Änderung gegen Lauf 16:** `--tbptt-start 20 --tbptt 80` statt `4 → 16`. Das sind **4 → 16 s** wie im POC (Lauf 15), statt 0.8 → 3.2 s.

## Code-Stand

`main` nach PR #48 (`03d5050`), Modellversion **G4.1**. Der Kopf des Logs zeigt `[protokoll] k=20->80 (4->16 s)` und **kein** `!! [fenster]`, der Code war also gepullt. Ein Prozess mit `--seeds 3`, nicht aufgeteilt.

## Modell und Daten

Wie Lauf 16: Konfiguration A (`--no-physics`), 11 427 Parameter, adiabat, `--subsample 2` (dt = 0.2 s, ≈ 8040 Schritte), Lags 5 / 20 Schritte (1 s / 4 s), `--clamp auto` = 11, Haltemenge OP06 / OP09, Latten 10.8009 / 7.7625.

## Training

- Tesla T4, torch 2.6.0+cu124. Seeds 0, 1, 2. Epochen: 60.
- Adam, `lr 1e-3`, Cosine auf 5 %. `--inner-steps 25`, `--batch-t 32`, `--clip-grad 1.0`, `--val-every 2`.
- `k` wächst geometrisch von 20 auf 80 und steht ab Epoche 30.
- Epochenzeit: 14 s bei k = 20, **39 s bei k = 80** (geschätzt waren 6.2 s + 0.41 s × k = 39 s). Pro Seed ≈ 33 min, der ganze Lauf ≈ 1 h 42 min.

## Kommando

```bash
nohup python3 -u GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
    --val-every 2 --device cuda --cache data_cache \
    > 17_konfigA_voll_k_sekunden.txt 2>&1 &
```

## Berichtete val-MAE (Schlusstafel im Log, 3 Seeds)

| | Seed 0 | Seed 1 | Seed 2 | Mittel ± sd | Güte | unter Latte |
|---|---|---|---|---|---|---|
| OP06 | 5.050 | 6.343 | 8.539 | **6.64 ± 1.76** | **0.62x** | 3/3 |
| OP09 | 4.494 | 8.329 | 5.644 | **6.16 ± 1.97** | **0.79x** | 2/3 |

Verdikt der Tafel: **kein Ergebnis**, weil die Streuung über ~1 °C liegt.

## Diagnose je Seed (aus `history.json`, nicht im Repo)

| | Epochen > 1 % am Clamp | davon spät | max `\|g\|` | `data` am Ende |
|---|---|---|---|---|
| Seed 0 | 3 (ep 3, 6, 7) | — | 11.1 | 0.06 |
| Seed 1 | 5 (ep 4, 11, 12, 28, 29) | ep 28–29 (4.6 %) | 46.6 | 0.09 |
| Seed 2 | 10 (ep 2, 3, 8–13, 15, 35) | ep 35 (8.8 %) | **1.15e14** | **0.27** |

Lauf 16 zum Vergleich: 7 / 13 / 14 Epochen am Clamp, max `|g|` 91 / 18 / 86. Kein Update wurde verworfen (`uebersprungen` 0 in beiden Läufen).
