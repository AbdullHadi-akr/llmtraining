# Modellstand und Experimente — chronologisch

> **Wozu diese Datei.** Die Fahrpläne erzählen, *warum* etwas gemacht wurde, und
> sortieren nach „zu tun" und „erledigt". Hier steht nur zweierlei, beides in
> zeitlicher Reihenfolge, **älteste Zeile zuerst**:
>
> 1. **welche Modellversion** es gab — was sich an Modell, Physik-Term oder
>    Training geändert hat, und ob sich dadurch das **Verhalten** geändert hat;
> 2. **welches Experiment auf welcher Version** lief, mit welcher Konfiguration,
>    auf welcher Maschine, mit welchem Ergebnis, und wo die Einzelheiten stehen.
>
> Wer eine Zahl zitiert, schlägt hier nach, auf welchem Modell sie entstanden
> ist. Wer das Modell ändert, trägt hier eine Zeile ein, **bevor** der nächste
> Lauf startet.

**Verweise:** [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) ·
[`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md) ·
[`GridCNN/BENCHMARK.md`](GridCNN/BENCHMARK.md) ·
Übergabe: [`README_NAECHSTE_SITZUNG.md`](README_NAECHSTE_SITZUNG.md)

---

## Die Regeln

* **Eine neue Modellversion** gibt es, sobald sich ändert, was ein Lauf mit
  derselben Konfiguration ausrechnet. Dann bekommt sie eine neue Nummer
  (`P3`, `G3`, …) und eine Zeile in Tabelle 1 — mit Commit und dem Test, der die
  Änderung festhält.
* **Code ohne Verhaltensänderung** (Diagnose, Logging, Geschwindigkeit, eine
  gesperrte Option) bekommt eine Unterversion (`P2.1`). Die Spalte „Verhalten"
  sagt dann **unverändert**, und der Test, der das belegt, steht daneben.
* **Jedes Experiment** bekommt eine Zeile in Tabelle 2: Datum, Modellversion,
  Konfiguration (nur was vom `config.yaml`-Default abweicht), Maschine,
  Ergebnis **mit Streuung**, Fundstelle.
* **Die Konfiguration ist nicht die Version.** Achse 1 lief mit `--ema-decay 0.5`
  auf demselben Modell P2 wie Achse 0 — das ist eine andere Konfiguration,
  kein anderes Modell.
* **Die Maschine gehört dazu.** Alles auf echten Daten nach dem 09.09. lief auf
  der **g4dn.2xlarge (Tesla T4, 15.6 GiB, 4 physische Kerne)** mit
  **CUDA-MPS** und parallelen Läufen über `sweep.py -j`. Ohne MPS ist dieselbe
  Arbeit 2.5× langsamer, ohne dass ein Log es sagt (FAHRPLAN, Kopf).

---

## Tabelle 1 — Modellversionen

### PINNmodulusTwo (MLP-Backbone + Rekurrenz, bleibt MLP)

| Version | ab | Commit | was sich geändert hat | Verhalten | festgehalten durch |
|---|---|---|---|---|---|
| **P1** | bis 31.08. | vor `bc5929a` | Stand der ersten Läufe auf echten Daten | — | — |
| **P2** | **01.09.** | `bc5929a` | **Quelle korrigiert**: `q_dot = jr1_w / V_JR1` statt `/ (V_JR1 · 121)` — die Gleichverteilung war doppelt gezählt, die Quelle 121× zu klein | **geändert** | Energiebilanz 147× → 0.5…0.9× (FAHRPLAN §11.1) |
| P2 | 09.09. | `ce08bf2` | Verluste auf dem Gerät akkumuliert (keine Syncs mehr), `sweep.py` | laut Commit unverändert | — |
| P2 | 14.09. | `496a1da` | Startbanner meldete δ = 1.0 s hartkodiert | nur Log | Test im Commit |
| P2 | 22.09. | `c795912` | `data._assert_shared_geometry` prüft, dass alle OPs dieselbe Punktreihenfolge haben | unverändert bei gültigem Cache | Test in beide Richtungen |
| **P2.1** | **23.09.** | dieser PR | **(1)** `heat_residual(..., return_parts=True)` gibt die Terme des Residuums mit heraus. **(2)** Schalter **`--phys-stencil {buffer,live}`** (`config.yaml: phys_stencil`), Default **`buffer`** = bisheriges Verhalten; `live` wertet die BDF-Lags mit dem lebenden Netz aus (O21). Wird im Checkpoint unter `run.phys_stencil` mitgeschrieben. **(3)** `train.py` **verweigert** `--time-deriv autograd` (O20). **(4)** neues Werkzeug `tools/residual_decomposition.py` | **unverändert** mit den Defaults, also für jeden bisherigen Lauf | `test_return_parts_leaves_the_default_call_bit_for_bit_unchanged`, `test_live_stencil_equals_the_buffer_stencil_on_a_fresh_rollout`, `test_train_refuses_the_autograd_time_derivative_before_reading_data` |
| *P3* | *nächste* | — | *P2.1 mit `--phys-stencil live` eingeschaltet* (Achse 5) — **oder** Ortsableitungen per Differenzenstern (`[BLIND]`, Übergabe §4 Schritt 3), je nachdem, was die Zerlegung sagt | geändert | — |

> **Die MLP-Struktur ist seit P2 unverändert** — `ModulusMLP` (4 × 128,
> lernbares Swish je Schicht, Weight-Norm), hybride Historie `[T(t−0.2 s),
> Rate 5 s, Rate 20 s]`, `residual_output: false`. Sie bleibt es auch mit P2.1;
> keine der Empfehlungen in FAHRPLAN §11.10 fasst sie an.

### GridCNN (Faltungsstapel in Δ-Form auf dem 3 × 11 × 11-Gitter)

| Version | ab | Commit | was sich geändert hat | Verhalten | festgehalten durch |
|---|---|---|---|---|---|
| **G0** | 02.09. / 09.09. | `f581465`, `fdb4e70`, `6f45f62` | nur Werkzeuge: `tools/spatial_rank.py` (Rangtest), `tools/balance_check.py` (Bilanz) | — | — |
| **G1** | **14.09.** | `7247691` | Unterbau ohne Netz: `grid.py`, `physics.py`, `solve.py` (expliziter Euler, **adiabat**), `benchmark.py` | neu | 45 Tests; Symmetrie und Randdifferenz **exakt** null |
| **G2** | **15.09.** | `7f1f5ec`, `b67ab08`, `cc9a83a` | `model.py` (16 × 3 = **11 427** Parameter, Kopf null-initialisiert → rechnet bei Init bitgleich den Löser), `train.py` (Ein-Schritt gegen eingefrorene freie Trajektorie), T4-Rollout gebatcht | neu | Tests in `test_model.py`, `test_train.py` |
| **G2.1** | **22.09.** | `477ffaa`, `c795912` | Ablationsarm **D** (`--no-coord-maps`, 11 139 Parameter); Geometrie gegen die echten Koordinaten geprüft; zwei Tests von Bitgleichheit auf relative Schranke 1e-12 | Default **unverändert**, D neu | sieben Tests für D, `test_layout_aus_den_echten_koordinaten` |
| G2.1 | 22.09. | `c5b16b4` | `balance_check.py`: jede CSV mit eigener Zeitachse (Abschnitt 4 war abgestürzt) | Werkzeug; Abschnitt 2 bitgleich | im Commit gegengeprüft |

> GridCNN hat **noch keinen einzigen Trainingslauf auf echten Daten**. Stufe 3
> und 4 sind gebaut, nicht gemessen (Ladepfad fehlt, Wandterm braucht Stufe 2).

---

## Tabelle 2 — Experimente

| Datum | Projekt | Experiment | Modell | Konfiguration (Abweichung vom Default) | Maschine | Ergebnis | Fundstelle |
|---|---|---|---|---|---|---|---|
| 31.08. | PINN | Schritt 5, erste Latte | P1 | `--epochs 2` | lokal | `LOSES TO` auf OP06/OP09 — **ungültig**, Quelle 121× zu klein | PINN-FAHRPLAN §9.3, Teil III |
| 01.09. | PINN | Schritt 5b-1 / 5b-2 | P2 | `--epochs 3`; 5b-2 zusätzlich `--w-phys 0 --w-bc 0` | lokal | mit Physik **beats** (10.540 / 7.494 °C), ohne **loses** (11.591 / 8.504) — 3 Epochen, **später als Untertrainiertheit erkannt** | §9.3, Stand-Tabelle |
| 01.09. | PINN | Schritt 6 | P2 | `--epochs 60` | CPU, ~2 h | val OP06 / OP09 **6.270 / 3.585 °C**, alle Simulations-OPs **beats**; OP19 10.334 °C **loses** (O11) | §11.3, §11.4 |
| 09.09. | GridCNN | Stufe 0, Rangtest | G0 | alle Cache-OPs | Rechenmaschine | **4 Moden bei 99.9 %** → Tor rot, am 15.09. überstimmt | GridCNN-FAHRPLAN Stand |
| 09.09. | GridCNN | Stufe 1, Bilanz, Lauf 1 | G0 | OP04, OP05, OP07, OP14 | Rechenmaschine | `jr2/jr1 = 1.000`, `U` ~1130 / ~50 W/m²K | GridCNN-FAHRPLAN 1a–1c |
| 10.09. | PINN | T4-Durchsatz, MPS | P2 | 4 × `--epochs 6`, 2 OPs | **T4** | `-j 4` ohne MPS 1.46×, **mit MPS 3.69×** | README_GPU_SERVER §6.4 |
| 10.09. | PINN | **Achse 0**: `w_phys` 0.1 / 0 | P2 | 3 Seeds, `--epochs 60 --w-bc 0`, `sweep.py -j 6` | **T4 + MPS**, 2 h 30 min | **5.248 ± 0.518** (ohne Physik) gegen 6.090 ± 0.882 °C — `[NOT SEPARATED]`. **Nullmessung** für alles Weitere. O17 (Fixpunkt 45–50 °C) gefunden | §11.8 |
| 14.09. | GridCNN | `benchmark.py` Stufe 0 + 2 | G1 | — | CI / lokal | 8 × ok, 4 × ok; Stern-Drift 3.6 % (R6) | BENCHMARK.md |
| 14.09. | PINN | Achse-1-Vorlauf | P2 | δ = 0.2, `--ema-decay 0.5` | **T4** | alle drei Signale grün; Banner-Bug gefunden | Stand-Tabelle „V" |
| 15.09. | PINN | **Achse 1**: δ 1.0 / 0.4 / 0.2 | P2 | 3 Seeds, `--epochs 60 --ema-decay 0.5`, Physik + BC an, `sweep.py -j 3` | **T4 + MPS**, 5 h 50 min | **5.428 ± 0.679 / 5.380 ± 1.240 / 4.868 ± 0.650** °C — `[NOT SEPARATED]`. `L_phys` 4.6e4 / 2.5e5 / 1.0e6 ∝ 1/δ² | §11.9 |
| 22.09. | GridCNN | Koordinaten und Materialverteilung am Cache | — (Daten) | alle 17 Cache-OPs | Rechenmaschine | Spannweite korrigiert, `xyz` bitgleich, λ-Band am Rand (R8, R9) | BENCHMARK.md, Lauf 22.09. |
| 22.09. | GridCNN | Stufe 1, Bilanz, Lauf 2 | G2.1 (Werkzeug vor `c5b16b4`) | 7 Konstant-Treiber-OPs, 3 Flusslevel | Rechenmaschine | Fluidbilanz 1.030…1.062 🟢; `Q_ht/tot` 0.70…0.77; `tot/jr1` 2.99…3.44 (dort „O17“, im PINN-Index **O19**); **Abschnitt 4 abgestürzt** | GridCNN-FAHRPLAN 1d |
| **23.09.** | PINN | **Residuenzerlegung, synthetisch** | **P2.1** (Training mit P2-Verhalten) | synthetischer Cache (400 s, OP01–03 Training, OP06 val), `--epochs 15 --inner-steps 50 --ema-decay 0.5`, δ = 1.0 / 0.4 / 0.2, Seed 0 | CPU (Cloud-Sitzung, keine GPU) | geloggtes `L_phys` **5.2× / 25.8×** (Achse 1 echt: 5.5× / 22×). Zerlegung: `[STALE]` in 2 von 3, `[JITTER]` und **`[BLIND]` in 3 von 3** (autograd sieht 0.01 … 9 % der Krümmung). **Synthetisch: Mechanismus belegt, Zahlen keine Ergebnisse** | PINN-FAHRPLAN §11.10 |
| **23.09.** | PINN | **`--phys-stencil live`, synthetisch** | **P2.1**, Schalter an (entspricht P3) | wie die Zeile darüber, zusätzlich `--phys-stencil live` | CPU (Cloud-Sitzung) | geloggtes `L_phys` 129 / 33 / 1.24e4 (statt 1.55e4 / 8.07e4 / 4.0e5); val OP06 **2.4 / 2.1 / 2.9 °C** gegen 10.4 / 7.6 / 6.3 °C mit `buffer` — **1 Seed, synthetisch, kein Ergebnis**; `[BLIND]` bleibt | PINN-FAHRPLAN §11.10 |
| *als Nächstes* | PINN | **Residuenzerlegung auf den 15 echten Checkpoints** (6 × Achse 0, 9 × Achse 1) | P2 (Checkpoints), Werkzeug P2.1 | `residual_decomposition.py … -j 4` | **T4 + MPS** | entscheidet §11.10: `[STALE]` / `[JITTER]` / `[BLIND]` auf echten Daten | Übergabe §3 |
| *danach* | PINN | **Achse 5**: `--phys-stencil live`, 3 Seeds | **P3** (= P2.1 + live) | `--epochs 60 --ema-decay 0.5 --delta-phys 0.2`, `sweep.py -j 3`; Vergleichsarm = Achse 1, δ = 0.2 | **T4 + MPS**, ~2 h | gegen 4.868 ± 0.650 und die Nullmessung 5.248 ± 0.518 | Übergabe §4 |
| *als Nächstes* | GridCNN | Stufe 1, Bilanz, Lauf 3 (Abschnitt 4) | G2.1 | wie Lauf 2 | Rechenmaschine, nur numpy | `U(V̇)` auf einer Kurve? | GridCNN-FAHRPLAN, Kopf |

---

## Das Update vom 23.09. — was vorher war, was jetzt ist

**Das Modell rechnet nichts anders.** Jeder Lauf mit `bdf1`/`bdf2` — also jeder,
der je gemacht wurde — gibt mit P2.1 bitgleich dasselbe wie mit P2. Geändert
hat sich, **was man über das Modell messen kann**, und das Messergebnis ändert
den Plan.

| | vorher (P2, bis 22.09.) | jetzt (P2.1, 23.09.) |
|---|---|---|
| `physics.heat_residual` | gibt nur das skalierte Residuum zurück | dasselbe; mit `return_parts=True` zusätzlich `T`, `T_1`, `T_2`, `dTdt`, `Txx/Tyy/Tzz`, die vier Leitungsanteile, `Qsrc`, den Divisor. Default **bitgleich** (Test) |
| `--time-deriv autograd` | lief — und trainierte über den Physik-Term ein **zweites** MLP (`mlp_with_time`), das der Rollout nie benutzt | **verweigert**, mit Begründung, bevor Daten gelesen werden (O20) |
| Diagnose des Physik-Terms | `L_phys` als eine Zahl je Epoche | `tools/residual_decomposition.py`: Sprung gegen den eingefrorenen Puffer, Rauheit des Rollouts, Sichtbarkeit der Krümmung für autograd, `L` je x-Ebene, Fit `A + B/δ²`; mehrere Checkpoints parallel (`-j`, MPS-Prüfung aus `sweep.py`) |
| BDF-Stencil des Physik-Terms | `T(t)` lebend, `T(t−δ)`, `T(t−2δ)` aus dem eingefrorenen Rollout — fest verdrahtet | dasselbe als Default `buffer`; Schalter `--phys-stencil live` wertet die Lags mit dem lebenden Netz aus. **Nicht eingeschaltet** |
| Tests PINNmodulusTwo | 135 passed, 1 skipped, 1 xfailed | **151** passed, 1 skipped, 1 xfailed (+16 in `test_residual_decomposition.py`) |
| Plan | „Das Nächste: O18 einbauen" | **O18 ausgesetzt**, erst die Zerlegung auf den echten Checkpoints (§11.10) |
| offene Punkte | O1 … O18 | O1 … **O22** (O19 Quelle unvollständig, O20 autograd-Zeit, O21 BDF-Zähler, O22 autograd blind) |

**Ist das Modell damit besser? Nein.** Es rechnet dasselbe. Besser ist, was man
über den Physik-Term weiß — und dass die erste Reparatur schaltfertig daliegt.
Ob sie etwas bringt, sagt erst Achse 5 auf der T4.

**GridCNN: nicht angefasst**, weder Code noch Dokumente (ausdrücklicher Wunsch).
Zwei Dinge, die es betreffen, stehen deshalb nur hier und in der Übergabe:
die unvollständige Quelle heißt in den GridCNN-Dokumenten „O17“, im PINN-Index
**O19** (PINN-O17 ist der Rollout-Fixpunkt); und Stufe 2 ist kleiner als dort
beschrieben — `total_w` und `mdot` liegen schon im Cache.
