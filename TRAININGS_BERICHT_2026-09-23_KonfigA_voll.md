# Konfiguration A auf voller Auflösung — 23.09.2026

> **Wenn du nur eine Zeile liest:** Auf voller Auflösung hält **OP06**
> (0.62x, wie im POC), **OP09 fällt** (1.11x, nur 1/3 Seeds unter der
> Latte), und die Seed-Streuung ist mit 1.5–1.8 °C **wieder über der
> Lesbarkeitsschwelle**. Das ist **kein Ergebnis**. Der Lauf war aber auch
> **nicht der POC auf voller Auflösung**: Das Fenster `k` stand in Schritten,
> nicht in Sekunden. Es schrumpfte von 4→16 s auf 0.8→3.2 s und erreichte
> `lag2` nie.

> **Nachtrag, 23.09. mittags:** Abschnitt 4 und 5 sind beantwortet.
> Nachgemessen ist O13 hier ein **Pegelfehler** (anfangs zu warm, am Ende zu
> kalt, drift 1.61x / 1.76x). Lauf 17 holt OP09 auf 0.79x zurück, aber die
> Streuung bleibt, also ist er ebenfalls kein Ergebnis. Die Zeile „Arm B ist die
> Behandlung" in der Tabelle von Abschnitt 5 ist **überholt**: Der
> Diffusionskern sieht keinen Pegelfehler. →
> [`TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md`](TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md)

**Lauf:** [`GridCNN/laeufe/16_konfigA_voll.txt`](GridCNN/laeufe/16_konfigA_voll.txt) ·
**Parameter:** [`16_konfigA_parameter.md`](GridCNN/laeufe/16_konfigA_parameter.md) ·
**Vorgänger:** [`TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md`](TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md)
**Code-Stand auf der Maschine:** vor PR #47 (nicht gepullt). Der Trainingspfad
ist identisch, es fehlt nur das Fehlerprofil. Tesla T4, torch 2.6.0+cu124.

---

## 1. Was herauskam

| | Seed 0 | Seed 1 | Seed 2 | **Mittel ± sd** | Latte | Güte | unter Latte |
|---|---|---|---|---|---|---|---|
| **OP06** | 5.580 | 8.383 | 5.970 | **6.64 ± 1.52** | 10.8009 | **0.62x** | 3/3 |
| **OP09** | 7.391 | 10.618 | 7.798 | **8.60 ± 1.76** | 7.7625 | **1.11x** | **1/3** |

Je Seed der Median über das letzte Drittel (10 von 31 Messpunkten).
Gerechnet mit `train.zusammenfassung` aus den drei `BERICHTET`-Zeilen. Die
Schlusstafel **im Log** enthält nur Seed 2, siehe Parameterblatt.

**Verdikt der Tafel:** `⚠ KEIN ERGEBNIS` — Seed-Streuung 1.52 / 1.76 °C
gegen ~1 °C.

### Gegen den POC (Lauf 15, `--subsample 10`)

| | POC (1445 Schritte) | voll (≈8040 Schritte) |
|---|---|---|
| OP06 | 6.57 ± 0.38 (0.61x, 3/3) | 6.64 ± 1.52 (0.62x, 3/3) |
| OP09 | 5.81 ± 0.87 (0.75x, 3/3) | **8.60 ± 1.76 (1.11x, 1/3)** |

Die Latten stimmen auf drei Nachkommastellen mit dem POC überein (10.80 /
7.76). Die Haltemenge ist dieselbe.

---

## 2. Wie die Seeds dahin kamen

**Frühphase, alle drei Seeds instabil.** `[SATURATED]` bis 100 %, `|g|` bis 91.

| | wieder gefangen ab | k dort | Epochen verloren |
|---|---|---|---|
| Seed 0 | ep 8 | 6 | 7 |
| Seed 1 | ep 18 | 9 | 17 |
| Seed 2 | ep 19 | 9 | 18 |

Im POC fingen sich die Seeds spätestens bei ep 10–11. Hier verlieren zwei
von drei Seeds fast ein Drittel des Laufs. Der Cosine-Plan läuft dabei
weiter, die Lernrate ist also schon gesunken, wenn das eigentliche Lernen
beginnt.

**Seed 1 bleibt auf OP09 stehen.** Ab ep 28 liegt OP09 flach bei 1.29–1.41x
und bewegt sich nicht mehr. OP06 verbessert sich im selben Zeitraum von 1.09x
auf 0.69x. Das ist kein Rauschen, sondern ein Plateau.

**Seed 0 ist der Einzige, der OP09 knapp unterbietet** (0.95x). Er ist auch
der Einzige, der sich früh gefangen hat.

---

## 3. Der Befund: Das Protokoll war nicht dasselbe wie im POC

Der Befehl sollte „nur `--subsample` und `--epochs`" ändern. In Schritten
stimmt das, in Sekunden nicht:

| | POC (`--subsample 10`, dt = 1 s) | Lauf 16 (`--subsample 2`, dt = 0.2 s) |
|---|---|---|
| lag1 / lag2 | 1 / 4 Schritte = **1 s / 4 s** | 5 / 20 Schritte = **1 s / 4 s** ✓ |
| k (Fenster) | 4→16 Schritte = **4→16 s** | 4→16 Schritte = **0.8→3.2 s** ✗ |
| lag2 im Fenster? | ab `j = 4`, also fast immer | **nie** (`k ≤ 16 < 20`) |
| lag1 im Fenster? | immer | erst ab `k > 5` |
| Rollout / Fenster | 1445 / 16 ≈ 90 | 8040 / 16 ≈ **500** |

Die Lags wurden in PR #46 auf Sekunden umgestellt, genau damit so etwas nicht
passiert (`lags_aufloesen`). **`k` wurde nicht umgestellt.**

Warum das zählt: In `rollout_loss` kommt der lag2-Eingang erst ab
`j >= lag2` aus der eigenen Vorhersage, davor aus dem eingefrorenen Puffer.
Bei `k = 16 < 20` lief **kein einziges Update** durch die Rückkopplung über
lag2. Das Netz hat nie gelernt, was seine eigene Vorhersage von vor 4 s
mit ihm anrichtet, obwohl es sie im Rollout 8040-mal wieder frisst.

**Das ist eine Hypothese, kein Beweis.** Dass OP09 fällt und die Frühphase
länger instabil ist, passt dazu. Es kann aber auch schlicht der fünffache
Horizont sein. Der nächste Lauf trennt beides: **dasselbe Fenster in
Sekunden** (`--tbptt-start 20 --tbptt 80`), sonst nichts.

### Was im Code dafür geändert ist

* `train.fensterwarnung`: Meldet `!! [fenster]`, wenn `k ≤ lag2`, und nennt
  das Kommando für dieselben Sekunden wie im POC. Bei Lauf 16 hätte es gesagt:
  `--tbptt-start 20 --tbptt 80`.
* `protokollname`: nennt k jetzt auch in Sekunden, `k=4->16 (0.8->3.2 s)`.
  Zwei Logs mit derselben Zeile `k=4->16` waren bisher nicht zu unterscheiden.
* `train.profil_aus_checkpoint` und `GridCNN/tools/nachmessen.py`: holen das
  Fehlerprofil (MAE und Bias je Abschnitt, `drift`) aus gespeicherten
  Gewichten nach, ohne neu zu trainieren. Zur Probe muss die MAE von
  `model.pt` die Zeile `letztes ep 60` im Log treffen.
* Drei neue Tests, zusammen 133, alle grün.

---

## 4. Was fehlt, und woher es kommt

* **Das Fehlerprofil von Lauf 16.** Die Gewichte liegen auf der Maschine unter
  `GridCNN/artifacts/A/seed{0,1,2}/`. **Vor dem nächsten Lauf sichern**,
  sonst überschreibt Lauf 17 sie. Danach `nachmessen.py`. Die Frage, die es
  beantwortet: **Ist OP09 ein Spätfehler (O13, `drift > 1.5`) oder
  gleichmäßig daneben — und zu warm oder zu kalt?**
* **Warum der erste Prozess nach Seed 1 endete.** Er hat keine Schlusstafel.
* **Die Einordnung gegen den PINN** (6.270 / 3.585) bleibt beim POC-Stand.
  Dieser Lauf ist nicht rankfähig.

---

## 5. Das Nächste

1. Gewichte sichern, `nachmessen.py` auf Lauf 16 (Minuten).
2. **Lauf 17:** A, `--subsample 2`, **`--tbptt-start 20 --tbptt 80`**, sonst
   alles wie Lauf 16. Kosten grob geschätzt aus den Epochenzeiten von
   Lauf 16 (etwa 6.2 s + 0.41 s × k): **etwa 1.5–2 h für drei Seeds**.

```bash
python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
    --val-every 2 --device cuda --cache data_cache \
    2>&1 | tee 17_konfigA_voll_k_sekunden.txt
```

**Was Lauf 17 entscheidet:**

| Ausgang | Lesart | danach |
|---|---|---|
| OP09 < 1.0x, Streuung < 1 °C | Es war das Fenster. A hält auf voller Auflösung | Schritt 2 (CFL), dann Arm B |
| OP09 ≥ 1.0x, Frühphase wieder lang instabil | Das Fenster allein reicht nicht | `--tbptt-start` höher, dann EMA (eins nach dem anderen) |
| OP09 ≥ 1.0x, `drift > 1.5` | O13, der Pegel läuft weg | Arm B ist die Behandlung, nicht noch ein Protokoll-Hebel |
