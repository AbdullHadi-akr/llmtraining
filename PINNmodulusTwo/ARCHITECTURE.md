# PINNmodulusTwo — Kontrollfluss und Modell

Wie ein Trainingslauf tatsächlich abläuft, was das Modell ist, und wo man
ansetzt, um es zu erweitern. `README.md` sagt *was* das Projekt ist,
`README_GPU_SERVER.md` sagt *wie man es fährt* — dieses Dokument sagt *wie es
funktioniert*.

---

## 1. Die Kette auf einen Blick

```
generate_cache.py ──> data_cache/OP*.npz          (einmalig, offline)
                             │
                             ▼
 materials.py ──────> data.load_ops()   ──────────> NormBundle
 (material_properties/)      │                      ├─ Normierungskonstanten
                             │                      └─ ops: [OPData, ...]
                             ▼
                      train.fit(args)
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 model.RecurrentField   physics.heat_residual   physics.boundary_condition_loss
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                        _LossBalancer
                             │
                             ▼
                    artifacts/  (metrics, Plots, Checkpoints)
```

Darüber liegen die Benchmarks. Sie rufen alle dasselbe `train.fit()` auf und
unterscheiden sich nur in der Achse, die sie variieren:

| Skript | Achse | Wann |
|---|---|---|
| `selftest.py` | — (reine Arithmetik) | Sekunden, vor allem anderen |
| `smallBench.py` | `w_phys` (2 Werte) | 2–5 min Rauchtest |
| `benchmark_balance.py` | Loss-Skalierung, Eingangskanäle | **zuerst** — legt fest, was ein Gewicht bedeutet |
| `benchmark_wphys_wbc.py` | `w_phys` × `w_bc` | danach |
| `benchmark_arch.py` | Breite, Tiefe, Lags, `delta_grid` | danach |

Die gemeinsame Maschinerie (pro Seed trainieren, über Seeds mitteln, val/test
trennen, Rausch-Verdikt) liegt in `bench_common.py`. Ein neuer Benchmark
beschreibt nur seine eigene Achse.

---

## 2. Was das Modell ist

`model.RecurrentField` — ein MLP, das **einen Punkt zu einem Zeitpunkt**
vorhersagt. Die Zeitdimension entsteht nicht im Netz, sondern dadurch, dass das
Netz in einer Schleife auf seinen eigenen Ausgaben aufgerufen wird.

### Eingangsvektor

Alles wird zu einem flachen Vektor konkateniert (`field()`, `model.py:390`):

| Block | Breite | Woher | Bedeutung |
|---|---|---|---|
| `xn` | 3 | `data.py` | Ortskoordinaten, geteilt durch ein gemeinsames `L_ref` |
| `static` | 3 | `_static_features` | Temperaturleitfähigkeit (z-skaliert), JR1-Indikator, x-Ebene |
| `cfg` | 7 (+Raten) | `_normalise_config` | die Simulations-Configs zum Zeitpunkt `t` |
| `forcing` | 1 (+Energie) | `q_dot(t)` | Wärmequelle, z-skaliert |
| `hist` | `k_max` | `_history()` | **die Rekurrenz** — siehe unten |

Ausgang: ein Skalar, die z-skalierte Temperatur `Tn` an diesem Punkt.

Insgesamt `3 + n_static + n_config + n_forcing + k_max` = 17 in der
Standardkonfiguration (Hybrid-History mit zwei Rate-Lags).

### Der History-Block — der eigentliche Trick

Zwei Layouts, umschaltbar über `history_mode`:

**`raw`** (`_history_raw`, `model.py:372`) — `k_max` Temperaturwerte im Abstand
`δ`:

```
[T(t−δ), T(t−2δ), ..., T(t−k·δ)]
```

**`hybrid`** (`_history_hybrid`, `model.py:308`, Default) — ein Ankerwert plus
Raten über kumulative Segmente:

```
Anker : T(t−Δgrid)
Rate 1: [T(t−Δgrid)   − T(t−Δgrid−5)]  / 5
Rate 2: [T(t−Δgrid−5) − T(t−Δgrid−25)] / 20
```

`Δgrid` verschiebt, *wo* das Fenster sitzt; es ist kein Teil einer Spanne. Die
Endpunkte von Rate 1 liegen 5 s auseinander, egal wie weit der Anker zurückliegt.
Deshalb sind `--delta-grid` und `--subsample` unabhängige Regler — eine
Verwechslung, die früher teuer war.

> **Fallstrick, der schon einmal alles zerstört hat:** Die Rate wird durch die
> **nominale** Segmentlänge geteilt, nie durch die geklammerte tatsächlich
> verstrichene Spanne. Am Anfang des Rollouts ist die verstrichene Spanne ein
> einziger Gitterschritt (`dtn ≈ 1.4e-4` normiert bei Δt = 0.2 s); die Division
> dadurch verstärkt die Ein-Schritt-Differenz um ~7000×, füttert sie zurück ins
> Netz, und der Rollout läuft binnen weniger Schritte nach `inf → nan`.
> (`model.py:332`)

Lookups zwischen zwei Gitterpunkten werden linear interpoliert
(`interp_history`, `model.py:127`), also darf ein Lag zwischen Samples liegen.
Vor `t = 0` gilt `T(t) := T(0)` (`_padded_lookup`).

### Was gelernt wird und was nicht

**Gelernt:** die MLP-Gewichte, ein `β` pro Schicht im lernbaren Swish
`x·σ(βx)`, und die zwei Physik-Gains `src_gain` / `diff_gain`.

**Fest, nie trainiert** — als Buffer registriert, nicht als Parameter:
`δ`, `k_max`, `Δgrid`, `rate_lags`, und die Lag-Gates (`gates()` gibt konstant
Einsen zurück; die Methode existiert nur noch, damit Logs und Checkpoints eine
stabile Form behalten).

Das ist Absicht: die History-Anordnung wird einmal konfiguriert und mit
`benchmark_arch.py` gesweept, statt zu hoffen, dass das Netz sie findet.

> Eine frühere Version machte `rate_lags` lernbar und schickte sie durch
> `softplus`. Für die winzigen normierten Werte gilt `softplus(x) ≈ x` nicht —
> aus einem angeforderten 5-s-Lag wurden still 1024 s. (`model.py:207`)

### Woher Modulus kommt

`ModulusMLP` baut auf `modulus.models.layers.FCLayer` (Weight-Norm-Blöcke) und
`modulus.models.module.Module` (Save/Load, Device, Meta). Alles andere — der
lernbare Swish, die Rekurrenz, die differenzierbare Zeitinterpolation, das
Physik-Residuum — ist PyTorch. Das ist die ~50:50-Aufteilung aus Methode 2.

---

## 3. Der Rollout — warum es kein Teacher Forcing gibt

`rollout_train()` (`model.py:452`) und `rollout()` (`model.py:485`) sind
identisch bis auf die Gradienten:

```python
buf[0] = Tn_ic                       # die GEMESSENE Anfangsbedingung
for ti in range(1, n_t):
    hist    = model._history(buf[:ti], dtn, tn[ti], p_idx)   # eigene Vergangenheit
    buf[ti] = model.field(xn, static, cfg_seq[ti], forcing_seq[ti], hist)
```

Drei Konsequenzen, die man kennen muss:

1. **Der Datenverlust wird auf der eigenen Trajektorie genommen.** Nie auf
   Ground-Truth-History. Was im Training gemessen wird, ist genau das, was zur
   Inferenzzeit passiert — inklusive Fehlerakkumulation.
2. **Die Anfangsbedingung wird auferlegt, nicht gelernt.** `buf[0]` ist die
   Messung, und die History liest nur aus *strikt früheren*, bereits berechneten
   Zeiten. `t = 0` wird nie vorhergesagt.
3. **Truncated BPTT.** `buf[:ti].detach()` in `rollout_train` — jeder Schritt
   propagiert nur durch seine eigene Feldauswertung. Der Speicher bleibt
   beschränkt, aber es fließt kein Gradient entlang der Zeitachse.

**Das ist der Laufzeit-Engpass.** ~7000 sequentielle Schritte pro OP und Epoche,
die sich prinzipiell nicht parallelisieren lassen — jeder Schritt braucht den
vorherigen. Deshalb bringt eine GPU hier viel weniger als erwartet, und deshalb
spart das Überspringen der Nullgewichts-Terme gemessen nur ~3 %.

---

## 4. Das Physik-Residuum

`physics.heat_residual` (`physics.py:82`). Anisotrope Wärmeleitung, dimensionslos:

```
∂T/∂t  −  ∇·(Fo ∇T)  −  Q_src  =  0
```

- **Raum**: Autograd, zweimal. `grad1 = ∂T/∂x`, dann drei weitere
  `grad`-Aufrufe für die Hesse-Zeilen. Alle sechs unabhängigen Komponenten des
  anisotropen Tensors gehen ein.
- **Zeit**: Finite Differenz über die Rekurrenz — `bdf1`, `bdf2` (Default) oder
  `autograd`. Der BDF-Stencil liest über `history_at()` mit dem festen `δ`,
  **nicht** über das Hybrid-Layout: die Zeitableitung ist von der
  Feature-Anordnung entkoppelt.
- **Skalierung**: jeder der drei Terme wird durch seinen eigenen Trainings-RMS
  geteilt (`--residual-norm rms`), landet also bei Einheitsskala, *bevor* die
  lernbaren Gains ihn anfassen.

> Warum das wichtig ist: die `*_scale`-Konstanten in `data.py` sind bereits
> RMS-Werte. `x / scale` gibt `mean(res²) = 1`. Der ursprüngliche Code teilte
> durch `sqrt(scale)` und ließ `mean(res²) = scale` stehen — die drei Terme
> behielten also genau den Größenabstand, den die Skalen beseitigen sollten.
> `src_gain`/`diff_gain` konnten das teilweise auffangen, `dTdt` hat gar keinen
> Gain. `--residual-norm legacy` hält den alten Pfad offen.

**`--phys-norm` nach dem Fix:** Der Wert wirkt jetzt auf ein anders skaliertes
`L_phys` (in `rms`-Modus entfällt die äußere Division durch `phys_scale`). Ein
unter der alten Skalierung eingestellter fester Divisor ist nicht übertragbar.

`boundary_condition_loss` (`physics.py:36`) erzwingt `∂T/∂x = 0` in der
Symmetrieebene `x = 0`. Der Maßstab ist der **gemessene** RMS-Ortsgradient über
x-benachbarte Gitterpunkte (`data._measure_bc_scale`), nicht mehr das frühere
`1/L_ref`, in dem gar keine Temperatur vorkam.

---

## 5. Die Normierung — was `data.py` festlegt

Alles wird auf Trainingsdaten gefittet und in `NormBundle` abgelegt. Ein
gehaltenes OP läuft über `build_op()` durch **exakt dieselben** Konstanten —
nichts wird neu gefittet, sonst wäre es kein echter Out-of-Sample-Test.

| Konstante | Bedeutung |
|---|---|
| `T_mu`, `T_sigma` | gepoolter z-Score der Temperatur über alle Trainings-OPs und -Zeiten |
| `L_ref` | geometrisches Mittel der Achsenausdehnungen; ein gemeinsamer Ortsmaßstab |
| `T_span_ref` | längste Trajektorie; skaliert `t` auf `tn ∈ [0,1]` |
| `Fo` | `λ · T_span_ref / (ρ Cp L_ref²)` — der Fourier-Tensor pro Punkt |
| `dTdt_scale`, `aniso_scale`, `Qsrc_scale` | RMS je Residuenterm |
| `bc_scale` | gemessener RMS-Ortsgradient (siehe oben) |
| `q_mu`, `q_sigma` | Normierung der Wärmequelle |
| `config_mu`, `config_sigma` | z-Score der Config-Kanäle |

### Profile vs. Labels — der Punkt, an dem OPs sich unterscheiden

`load_ops()` teilt die Config-Kanäle in zwei Sorten (`data.py`, Abschnitt
*profile detection*):

- **Profil**: der Kanal ändert sich *innerhalb* eines OP über die Zeit.
- **Label**: der Kanal ist je OP konstant, unterscheidet sich aber zwischen OPs.

`train.py` gibt beides beim Start aus. Der Unterschied ist nicht kosmetisch:
Label-Kanäle sind Konstanten, die das Netz pro OP auswendig lernen kann. Bei
fünf Trainings-OPs können ein paar davon zusammen als OP-Kennung wirken, und ein
daran gelernter Offset überträgt sich nicht auf OP06/OP07.

Das ist auch die Begründung für die ganze Rekurrenz: zwei OPs können zum
Zeitpunkt `t` dieselbe momentane Config haben und trotzdem völlig verschiedene
Temperaturen, weil ihre *Vorgeschichte* verschieden war.

---

## 6. Der Trainingsschritt

`train.fit()`, innerste Schleife (eine Iteration pro OP, `train.py`):

```
für jede Epoche:
  für jedes OP:
    buf      = rollout_train(...)              # ~7000 sequentielle Schritte
    L_data   = mean((buf[1:] − Tn[1:])²)
    L_phys   = mean(heat_residual(...)²)       # batch_phys Stichproben
    L_bc     = mean(boundary_condition_loss(...)²)

    L_*_bal  = L_* / balance.divisor(...)      # je Term durch eigene Größe
    loss     = w_data·L_data_bal + w_phys·L_phys_bal + w_bc·L_bc_bal

    loss.backward();  clip_grad_norm_;  opt.step()
```

**Der Optimierer schrittet pro OP, nicht pro Epoche.** Bei fünf OPs sind das
fünf Schritte je Epoche, und das zweite OP sieht bereits aktualisierte Gewichte.

### Loss-Balancing (`_LossBalancer`)

Die drei Terme leben auf völlig verschiedenen Skalen; ein rohes Gewicht ist
darum kein Mischungsverhältnis. Jeder Term wird durch eine laufende Schätzung
seiner eigenen Größe geteilt:

| `--loss-balance` | geteilt werden | Folge |
|---|---|---|
| `ema` (Default) | alle drei | `w_data:w_phys:w_bc` ist ein echtes Verhältnis, in Epoche 1 wie in Epoche 60 |
| `legacy` | nur `L_phys`, `L_bc` | `L_data` bleibt roh und fällt um Größenordnungen → die Mischung wandert zur Physik, das beste `w_phys` hängt an `--epochs` |
| `fixed` | alle drei, nach Warm-up eingefroren | deterministisch, keine Rückkopplung |

Drei Details, die keine Kleinigkeiten sind:

- **Geteilt wird durch die Schätzung von *vorher*.** Fließt der aktuelle Wert
  zuerst in den Mittelwert ein, dämpft ein Ausschlag sich selbst — ein
  10×-Sprung meldet sich als ~5×.
- **`decay · nan = nan`**, deshalb aktualisiert nur ein endlicher Wert den
  Mittelwert. Sonst hinge der Divisor für den Rest des Laufs auf `nan`.
- **Der Horizont ist in Epochen definiert, nicht in Schritten.** Weil pro OP
  geschrittet wird, wäre ein reiner Schritt-Decay bei 5 OPs ~2 Epochen und bei
  einem OP ~10 — Läufe mit verschiedenen `--ops` wären nicht vergleichbar.
  `train.py` korrigiert das mit `decay^(1/n_ops)`.

Mitgeloggt wird `ratio_phys = w_phys·L_phys_bal / (w_data·L_data_bal)` je
Epoche: die Mischung, die der Optimierer tatsächlich gesehen hat. Genau darauf
sieht `benchmark_balance.py`.

### Terme mit Gewicht 0

Werden nicht berechnet (`--zero-weight-terms skip`) und als `NaN` protokolliert,
damit ein Plot eine Lücke zeigt statt einer flachen Linie, die es nie gab.
`0.0 · nan` ist `nan`, ein Nullgewicht neutralisiert also einen nicht-endlichen
Term **nicht** — deshalb wird der Term per `if` weggelassen und nicht
wegmultipliziert.

---

## 7. Erweitern — wo man ansetzt

| Vorhaben | Ort | Zu beachten |
|---|---|---|
| Neuer Eingangskanal | `data._assemble_op` + `NormBundle` | `n_config`/`n_forcing` mitziehen, Konstanten aus dem Bundle an `build_op` durchreichen, sonst ist der Held-out-Test verfälscht |
| Neues History-Layout | `model._history_*` + `history_mode` | `k_max` folgt dem Layout; `history_at()` für die Physik getrennt halten |
| Anderer Zeitableiter | `physics.heat_residual`, `time_deriv` | BDF-Stencils lesen über `δ`, nicht über `Δgrid` |
| Weiterer Loss-Term | `train.fit` + `_LossBalancer.KEYS` | einen Divisor spendieren, sonst ist sein Gewicht wieder skalenabhängig |
| Neue Sweep-Achse | neues `benchmark_*.py` | `bench_common.train_one_seed` benutzen; die Achse ist Daten, kein Code |
| Neues Trainings-Flag | `train.parse_args` **und** `bench_common.make_train_args` **und** `smallBench._make_args` | `fit()` liest per `getattr` — ein vergessenes Feld wirft nicht, es fällt still auf einen anderen Default zurück |

Die letzte Zeile ist die häufigste Falle in diesem Projekt. Sie hat schon einmal
zugeschlagen: `smallBench` reichte `delta_grid` nicht durch, der Rauchtest lief
still mit dem Datenschritt statt dem konfigurierten Anker — bei `subsample=2`
zufällig identisch, bei jedem anderen Wert nicht (`smallBench.py:114`).

**Nach jeder Änderung an der Skalierung:** `python3 PINNmodulusTwo/selftest.py`.
Sekunden, ohne Daten und ohne GPU. Die Eigenschaften, die er prüft, sind in
einem Trainingslog unsichtbar — die nächste Gelegenheit, sie zu bemerken, ist
ein mehrstündiger Benchmark.

---

## 8. Artefakte

Alles landet in `PINNmodulusTwo/artifacts/` (nicht in git):

| Datei | Von wem |
|---|---|
| `metrics.txt`, `training_curves.png`, `timeseries.png`, `pred_OP0*.npz` | `train.py` |
| `smallBench_results.txt`, `smallBench_convergence.png` | `smallBench.py` |
| `benchmark_balance.csv`, `_best.txt`, `.png`, `_ratio.png`, `balance_parts.json` | `benchmark_balance.py` |
| `benchmark_wphys_wbc.csv`, `_best.txt`, `_settings.txt`, `probe_parts.json` | `benchmark_wphys_wbc.py` |
| `benchmark_arch.csv`, `_best.txt`, `.png` | `benchmark_arch.py` |

Die `*_parts.json` tragen eine **Signatur** aller Einstellungen, die einen Lauf
prägen — Balancing eingeschlossen. Läuft ein zweiter Teil mit anderen
Einstellungen, wird der gespeicherte verworfen statt still dazugemischt.
