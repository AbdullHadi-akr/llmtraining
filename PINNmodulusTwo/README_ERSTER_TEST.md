# Erster Test — vollständige Beschreibung des Laufs

**Stand: 27.08.2026** · Projekt `PINNmodulusTwo` · gilt ab Commit „Ursache des
NaN-Abbruchs beheben"

Dieses Dokument beschreibt den ersten durchlaufenden Test vollständig: Modell,
Architektur, Trainingsparameter, Datenaufteilung, Verlustfunktion und
Ergebnisse. Es ist als Referenz gedacht — wer den Lauf reproduzieren oder eine
Zahl daraus zitieren will, findet hier, wovon sie stammt.

Alle Werte sind die Defaults aus [`config.yaml`](config.yaml), die identisch von
`train.py`, `smallBench.py` und den drei Benchmark-Skripten gelesen werden.

> **Vorher lief gar nichts.** Bis zu diesem Stand brach jeder Lauf in Epoche 1
> mit `L_data = nan` ab, bevor ein einziger Gradientenschritt passiert war.
> Kapitel 7 beschreibt die Ursache und was daran geändert wurde.

---

## Inhalt

1. [Modelltyp](#1-modelltyp)
2. [Architektur](#2-architektur)
3. [Training](#3-training)
4. [Daten](#4-daten)
5. [Verlustfunktion](#5-verlustfunktion)
6. [Ergebnisse](#6-ergebnisse)
7. [Was geändert wurde und warum](#7-was-geändert-wurde-und-warum)
8. [`hybrid` gegen `raw` — und die Segmentlänge](#8-hybrid-gegen-raw--und-die-segmentlänge)
8b. [Status — umgesetzt, fehlend, festgelegt](#8b-status--umgesetzt-fehlend-festgelegt)
9. [Einschränkungen](#9-einschränkungen)
10. [Reproduktion](#10-reproduktion)

---

## 1. Modelltyp

**Autoregressiver rekurrenter PINN-Surrogat für ein Temperaturfeld.**

Das Modell sagt die normierte Temperatur `Tn(t, x)` an einem Gitterpunkt zu
einem Zeitpunkt voraus. Es ist **rekurrent**: die Eingabe enthält die eigene
Vergangenheit des Modells. Und es ist **physikinformiert**: neben dem Datenterm
wird das Residuum der anisotropen Wärmeleitungsgleichung minimiert.

Die Aufteilung folgt der „~50:50 Modulus:PyTorch"-Methode:

| Teil | Herkunft |
|---|---|
| Funktionsapproximator (MLP-Blöcke, Weight Norm, `Module`-Basisklasse) | `modulus.models` |
| Rekurrenz, History-Kanäle, differenzierbare Zeit-Interpolation | PyTorch, `model.py` |
| Physik-Residuum (Autograd-Laplace + FD-Zeitableitung) | PyTorch, `physics.py` |

**Kein Teacher Forcing, nirgends.** Der Datenverlust wird auf genau der
Trajektorie genommen, die auch zur Inferenzzeit entsteht — free-running,
ausschließlich mit der gemessenen Anfangsbedingung als Startwert. Details in
[`ARCHITECTURE.md`](ARCHITECTURE.md) Kapitel 3.

Die Trajektorie entsteht so:

```python
buf[0] = Tn_ic                                   # die GEMESSENE Anfangsbedingung
for ti in range(1, n_t):                         # ~7000 Schritte je OP
    hist    = model.history_rollout(buf, ti, plan)    # eigene Vergangenheit
    buf[ti] = model.field(xn, static, cfg[ti], forcing[ti], hist)
```

`buf[0]` wird nie vorhergesagt, und die History liest nur aus *strikt früheren*
Zeiten. Die Anfangsbedingung ist damit auferlegt, nicht gelernt.

---

## 2. Architektur

### Netz

| Größe | Wert | Bemerkung |
|---|---|---|
| **Hidden Layers** | `num_layers: 4` | vier verdeckte Schichten |
| **Hidden Size** | `layer_size: 128` | konstante Breite je Schicht |
| **Aktivierung** | *Learnable Swish* `x · sigmoid(β·x)` | `β` ist **pro Schicht** ein gelernter Skalar, Init `β = 1.0` |
| **Ausgabeschicht** | linear, 1 Kanal, ohne Weight Norm | liefert `Tn` |
| **Weight Norm** | an, auf allen verdeckten Schichten | `nn.utils.parametrizations.weight_norm(dim=0)` |
| **Parameterzahl** | ~70 000 | bei 128/4 und den unten genannten Eingangskanälen |
| **Initialisierung** | `xavier_uniform_`, `bias = 0` | das ist die Init des echten `modulus.models.layers.FCLayer` |

Die Eingabe ist ein einziger flacher Vektor:

```
[ xn(3) , static(S) , config(C) , forcing(F) , history(k) ]
```

| Block | Breite | Inhalt |
|---|---|---|
| `xn` | 3 | normierte Koordinaten des Gitterpunkts |
| `static` | S | zeitunabhängige Material-/Geometriemerkmale (Temperaturleitfähigkeit, JR1-Indikator, x-Ebene) |
| `config` | C | Betriebspunktmerkmale (C-Rate, Fluid-Eintrittstemperatur, Volumenstrom …), z-normiert |
| `forcing` | F | momentane normierte Wärmequelle `q̇(t)` |
| `history` | k | die eigene Vergangenheit — siehe unten |

### Rekurrenz: `delta` und `k`

Hier gibt es **zwei verschiedene Verzögerungen**, die man nicht verwechseln
darf:

| Symbol | Config | Wert | Wofür |
|---|---|---|---|
| **`delta`** (`δ`) | `delta_seconds` (fest 1.0 s) | `1.0 s` → normiert `6.784e-4` | **nur** für die Finite-Differenzen-Zeitableitung im Physik-Residuum (`bdf2`) |
| **`delta_grid`** (`Δgrid`) | `delta_grid: 0.2` | `0.2 s` | Ankerverzögerung der Hybrid-History |

Beide sind **feste Puffer, keine Parameter** — sie werden nie trainiert.

**`k`** — die Zahl der History-Kanäle — folgt aus dem `history_mode`:

| `history_mode` | `k` | Kanäle |
|---|---|---|
| `raw` | `k_max = 2` | `[ T(t−δ), T(t−2δ) ]` |
| **`hybrid`** (Default) | `1 + len(rate_lags) = 3` | `[ T(t−Δgrid), rate₁, rate₂ ]` |

Im Hybrid-Modus ist `k_max` **nicht** frei wählbar: es ergibt sich aus der Zahl
der `rate_lags` und überschreibt ein übergebenes `k_max`.

Die Hybrid-Kanäle im Detail, mit den ausgelieferten `rate_lags: [5.0, 20.0]`:

```
Anker : T(t − 0.2 s)
Rate 1: ( T(t − 0.2)  − T(t − 5.2)  ) / (5 s  · rate_scale)
Rate 2: ( T(t − 5.2)  − T(t − 25.2) ) / (20 s · rate_scale)
```

Die Segmente sind **kumulativ und disjunkt**: jedes beginnt, wo das vorige
endete. `delta_grid` verschiebt nur, *wo* das Fenster liegt, und ist nicht Teil
einer Spanne. `rate_scale` ist `dTdt_scale`, der RMS von `dTn/dtn` — er bringt
den Kanal für ein echtes Signal auf O(1).

Weitere feste Entscheidungen:

* **`gates()` liefert immer Eins.** Es gibt kein weiches Gating der Lags; jeder
  History-Kanal ist voll aktiv. Die Methode existiert nur, damit Logging und
  Checkpoints eine stabile Form behalten.
* **`src_gain` / `diff_gain` stehen fest auf 1.0** (`learn_gains: false`). Sie
  existierten nur, um eine Normierung rückgängig zu machen, die `physics.py`
  nicht mehr vornimmt. Frei gelassen können sie gegen 0 laufen und `L_phys` mit
  einem konstanten Feld erfüllen — der Physikterm schaltet sich dann selbst ab.
* **Gelernt werden ausschließlich**: die MLP-Gewichte und die `β` je Schicht.

### Ausgabeparameterisierung

```yaml
residual_output: false
```

`field()` liefert die **absolute** normierte Temperatur, nicht die Abweichung
von einem mitgeführten Niveau. Warum das so sein muss, steht in Kapitel 7 — es
ist die zentrale Korrektur dieses Stands.

---

## 3. Training

### Der Ablauf

Pro Epoche und OP passiert genau das:

1. **Ein** free-running Rollout über die ganze Trajektorie unter `no_grad`
   (~7000 sequentielle Schritte) → eingefrorener Puffer `own_hist`.
2. **`inner_steps` Optimierer-Schritte** gegen diesen eingefrorenen Puffer, je
   auf einem Minibatch aus `(Zeit, Punkt)`-Paaren.

Das ist billiger als es klingt und dennoch exakt äquivalent: die Rekurrenz hat
die History zwischen den Schritten ohnehin detached (Truncated BPTT), der
Gradient bei `t` verließ also nie die eigene Feldauswertung dieses Schritts.
Der Gradient des alten Vollsequenz-`L_data` ist damit genau die Summe
unabhängiger Pro-`(t, Punkt)`-Gradienten gegen eine als konstant behandelte
Trajektorie — dieselbe Größe, die ein Minibatch erwartungstreu schätzt.

Der Unterschied in der Praxis: früher kostete **ein** Optimierer-Schritt einen
7000-Schritte-Rollout, ein 60-Epochen-Lauf auf 5 OPs kam also auf 300
Adam-Updates. Jetzt trägt derselbe Rollout 100.

### Hyperparameter

| Parameter | Config-Schlüssel | Wert |
|---|---|---|
| **Epochen** | `epochs` | `60` |
| **Optimierer-Schritte je OP und Epoche** | `inner_steps` | `100` |
| → Gesamtzahl Adam-Updates | | `60 × 5 OPs × 100 = 30 000` |
| **Seed** | `seed` | `0` |
| **Batch Size (Daten)** | `batch_data` | `2048` `(t, Punkt)`-Paare |
| **Batch Size (Physik)** | `batch_phys` | `256` |
| **Batch Size (Randbedingung)** | `batch_bc` | `128` |
| **Learning Rate** | `lr` | `2.0e-3` (Adam) |
| **Weight Decay** | `weight_decay` | `0.0` |
| **Gradient Clipping** | `grad_clip` | `1.0` (max. Gradientennorm; `0` schaltet ab) |
| **Early-Stopping-Geduld** | `early_stopping_patience` | `0` — **abgeschaltet** |
| **Präzision** | — | float32, **kein** AMP/fp16 |
| **TF32** | `tf32` | `false` |

Zu zwei Werten die Begründung, weil sie nicht selbsterklärend sind:

* **Kein AMP/fp16 und kein TF32.** Das Residuum differenziert das Netz
  *zweimal*; reduzierte Präzision verschlechtert diese zweiten Ableitungen.
  `--tf32` existiert als Opt-in, ist aber aus demselben Grund aus.
* **Early Stopping aus.** Der Trainingsverlust ist hier kein
  Validierungssignal — er wird auf demselben Abschnitt gemessen, auf dem
  gefittet wird. Ein Abbruch danach wäre willkürlich.

### Stabilitätsschalter

| Parameter | Wert | Zweck |
|---|---|---|
| `rollout_clamp` | `50.0` | sättigt `|Tn|` im Rollout-Puffer |
| `max_rate_amp` | `0.0` (aus) | Notnagel, deckelt `A` durch Umskalieren von `rate_scale` |

Beide sind in Kapitel 7 erklärt.

---

## 4. Daten

### Herkunft

Simulationsergebnisse einer Batteriezelle, je Betriebspunkt („OP") ein
Temperaturfeld über der Zeit auf einem festen Gitter, plus die Treibergrößen
des Betriebspunkts. Gelesen aus `data_cache/`, Materialdaten aus
`material_properties/`.

### Welche Betriebspunkte

Dieses Projekt nutzt ausschließlich die **konstanten** Betriebspunkte: eine
C-Rate, eine Fluid-Eintrittstemperatur und ein Volumenstrom, über den ganzen
Lauf gehalten. Alle sind Ladung (CH), `V_max` 4.35 V, SOC 10–90 %.

| OP | Art | C-Rate | T_start [°C] | T_fluid [°C] | Volumenstrom [l/min] | Rolle |
|---|---|---|---|---|---|---|
| OP01 | CC | 2.0 | 25 | 25 | 15 | Training |
| OP02 | CC | 2.0 | 15 | 15 | 15 | Training |
| OP03 | CC | 2.0 | 30 | 30 | 15 | Training |
| OP04 | CC | 2.0 | 25 | 25 | 30 | Training |
| OP05 | CC | 2.0 | 40 | 40 | 30 | Training |
| OP06 | CC | 2.0 | 25 | 25 | 0 | gehalten |
| OP07 | CC | 2.0 | 10 | 10 | 0 | **`test_op`** |

Ab OP08 werden die Treiber zu **Profilen** (Fluidtemperaturprofil, vorsimulierte
CC-CV-Ströme, Volumenstromprofil). Die gehören in
`PINNmodulusTwoExtProfiles`, nicht hierher. OP17–OP19 („Abgleich mit
Minimodul-Test": Entladung, Fast Charge, WLTP-Fahrzyklus) sind Messdaten und
nicht Teil des Trainings.

Der vollständige Plan ist in
[`../PINNmodulusTwoExtProfiles/op_registry.py`](../PINNmodulusTwoExtProfiles/op_registry.py)
transkribiert — inklusive der Einteilung in Tiers danach, was ein gehaltener OP
vom Modell verlangt, das das Training nie gezeigt hat.

Bemerkenswert für die Bewertung: **OP07 hat `Volumenstrom = 0`**, und im
Training kommt diese Betriebsart gar nicht vor (OP01–OP05 haben alle 15 oder
30). Der gehaltene OP ist damit kein reiner Interpolationstest.

### Aufteilung

Es gibt **zwei voneinander unabhängige Aufteilungen**, und beide werden
berichtet:

| | Was | Wert |
|---|---|---|
| **Training** | Betriebspunkte | `OP01, OP02, OP03, OP04, OP05` |
| | Zeitanteil je OP | die ersten **80 %** (`train_frac: 0.8`) |
| **Validierung (in-time)** | dieselben OPs, letzte **20 %** der Zeitachse | in `metrics.txt` die Spalte `MAE test` |
| **Test (gehaltener OP)** | ein Betriebspunkt, der **nie** im Training war | `test_op: OP07` |

Die in-time-Validierung beantwortet „kann das Modell die Trajektorie
fortschreiben, die es angefangen hat?". Der gehaltene OP beantwortet „kann es
einen Betriebspunkt, den es nie gesehen hat?". Das sind verschiedene Fragen und
die zweite ist die schwerere.

**Wichtig zur Sauberkeit der Aufteilung:** sämtliche Normierungskonstanten —
`T_mu`, `T_sigma`, jede Config- und Quellstatistik, `dTdt_scale`, `phys_scale`,
`bc_scale` — werden **ausschließlich über `[:split_t]` gepoolt**, also über den
Trainingsanteil. Der gehaltene Zeitabschnitt fließt in keine einzige Konstante
ein.

### Zeitliche Auflösung

| | Wert |
|---|---|
| Rohschritt der Simulation | `0.1 s` |
| `subsample_time` | `2` |
| → tatsächlicher Zeitschritt `Δt` | **`0.2 s`** |
| Schritte je OP | ~7000 |
| Referenzspanne `T_span_ref` | ~`1474 s` (längste Trajektorie) |
| normierter Schritt `dtn` | `1.357e-4` |
| `subsample_mode` | `stride` (jeden n-ten Wert; `mean` wäre grobes Anti-Aliasing und ändert die Daten) |

`train.py` gibt nach dem Laden eine **CFL-Prüfung** aus. Das Stabilitätslimit
liegt bei `Δt_max ≈ 0.241 s`; `Δt = 0.2 s` liegt darunter. Ein größeres
`subsample` überschreitet es — `subsample 40` ergäbe `Δt = 4.0 s`, also das
16-fache des Limits, und die Ausgabe warnt dann explizit.

### Normierung

| Größe | Normierung |
|---|---|
| Temperatur | z-Score mit `T_mu`, `T_sigma` (gepoolt über den Trainingsanteil) |
| Zeit | geteilt durch `T_span_ref`, also `tn ∈ [0, 1]` |
| Koordinaten | verschoben und durch `L_ref` geteilt |
| Wärmequelle | `q̇ · T_span_ref / (ρ Cp T_sigma)` |

Fourier-Tensor `Fo = λ · T_span_ref / (ρ Cp L_ref²)`.

---

## 5. Verlustfunktion

Drei Terme, jeder mit eigenem Gewicht:

```
L  =  w_data · L̂_data  +  w_phys · L̂_phys  +  w_bc · L̂_bc
```

| Term | Was | Gewicht |
|---|---|---|
| **`L_data`** | MSE zwischen Vorhersage und Label, in z-normierten Einheiten, auf einem Minibatch aus `(t, Punkt)` mit `t < split_t` | `w_data: 1.0` |
| **`L_phys`** | Residuum der anisotropen Wärmeleitungsgleichung: `∂T/∂t − ∇·(Fo ∇T) − Q` | `w_phys: 0.1` |
| **`L_bc`** | Randbedingung `∂T/∂x = 0` an der Zellmitte `x = 0` | `w_bc: 0.1` |

### Wie die Terme gebildet werden

* **Zeitableitung** im Physik-Residuum: `time_deriv: bdf2`, also
  Rückwärtsdifferenz zweiter Ordnung über die History-Lags `δ` und `2δ`.
  Alternativen: `bdf1`, `autograd`.
* **Laplace-Term**: per Autograd nach `xn`, also zweite Ableitungen des Netzes.
* **`residual_norm: rms`**: das zusammengesetzte Residuum wird durch **eine**
  Skala geteilt (`phys_scale`), sodass `mean(res²) = 1` bei Trainingsstatistik.
  Ausdrücklich **nicht** Term für Term durch je eigene Skalen — das würde die
  Gleichung ändern statt sie zu skalieren.
* **`L_data` ist auf `t < split_t` beschränkt.** Der Rollout deckt die ganze
  Trajektorie ab, weil die Rekurrenz sie braucht und die beiden unüberwachten
  Terme sie nutzen — aber auf Labels jenseits von `split_t` wird nie gefittet.

### Loss-Balancing

```yaml
loss_balance: ema
ema_decay: 0.9          # pro EPOCHE, intern korrigiert für len(ops) × inner_steps
data_floor: 1.0e-8
```

Jeder Term wird vor Anwendung seines Gewichts durch eine EMA-Schätzung seiner
eigenen Größenordnung geteilt. `w_data : w_phys : w_bc` ist damit ein Verhältnis
zwischen **Termen** und nicht zwischen ihren zufälligen Einheiten — und es
bedeutet in Epoche 1 dasselbe wie in Epoche 60.

`legacy` teilt nur `L_phys` und `L_bc`; `L_data` bleibt roh und fällt im Lauf um
Größenordnungen, die Mischung driftet also gegen Physik und das beste `w_phys`
wird zu einer Funktion von `epochs`. Ein unter `legacy` gemessenes `w_phys`
überträgt sich deshalb **nicht** auf `ema`.

`zero_weight_terms: skip` überspringt Terme mit Gewicht 0 vollständig — ein
solcher Term kostet sonst einen Vorwärtsdurchlauf und eine Autograd-Hessematrix.
Er wird dann als `NaN` geloggt, damit die Konvergenzkurve eine Lücke zeigt statt
einer flachen Linie, die es nie gab.

---

## 6. Ergebnisse

> **Alle Zahlen in diesem Kapitel stammen von einem synthetischen Bundle**, weil
> in der Entwicklungsumgebung kein `data_cache/` vorliegt. Sie sind belastbar
> für **Vergleiche zwischen Konfigurationen** und für die Aussage „läuft
> überhaupt". Sie sind **keine** Vorhersage der MAE auf den echten OPs.
> Kapitel 9 sagt genau, was sich überträgt und was nicht.

Aufbau: 2 OPs, **7000 Zeitschritte** (die echte Länge), Breite 128 / Tiefe 4,
`w_phys = 0.1`, `rollout_clamp = 50`, `residual_output: false`, 10 Epochen,
3 Seeds. MAE in physikalischen Grad Celsius, aus dem free-running Rollout.

Bei dieser Länge stimmt das Bundle mit den echten Daten überein:
`dTdt_scale = 2.467` gegen echte `2.479`, also dieselbe Verstärkung
`A = 119.5 / 29.9` für `[5, 20]` und dieselbe Zahl Gitterschritte im Fenster.

### Ist das Modell überhaupt von Nutzen?

Eine MAE-Zahl ohne Vergleichsmaßstab sagt nichts. Zwei triviale Vorhersager als
Untergrenze:

| Vorhersager | MAE train | MAE test |
|---|---|---|
| „Temperatur ändert sich nie", `T(t) = T(0)` | 5.36 °C | **11.96 °C** |
| „konstanter Mittelwert der Trainingslabels" | 2.69 °C | **6.60 °C** |
| **Modell (`hybrid [5,20]`, Default)** | **0.78 °C** | **1.21 °C** |

**Ja.** Das Modell liegt um Faktor 5–10 unter dem besseren der beiden trivialen
Vorhersager. Es lernt die Dynamik und gibt nicht bloß einen Mittelwert zurück.

### Die Segmentlänge

| `rate_lags` | `A` | MAE train | MAE test | pro Seed (test) |
|---|---|---|---|---|
| **`[5, 20]`** (Default) | 119 / 30 | 0.784 | **1.207** | 1.695 · 0.781 · 1.144 |
| `[50, 150]` | 12 / 4 | 0.594 | 2.102 | 2.846 · 1.579 · 1.880 |
| `[200, 600]` | 3 / 1 | 0.724 | 2.507 | 3.010 · 1.464 · 3.046 |
| `raw` | — | 0.824 | 2.601 | 2.677 · 1.790 · 3.336 |

`[5, 20]` gewinnt **auf allen drei Seeds**, und keiner der zwölf Läufe bricht ab.
Kapitel 8 erklärt, warum die langen Segmente verlieren.

### Wichtig: `L_data` ist **nicht** das Auswahlkriterium

`L_data` ist ein z-normierter Trainingsverlust auf dem Trainingsabschnitt. Auf
`L_data` lag `[200, 600]` zwei Größenordnungen vor allem anderen — auf MAE ist
es das zweitschlechteste. Wer `rate_lags` oder `history_mode` nach `L_data`
auswählt, wählt falsch. Dafür gibt es `benchmark_arch.py`.


## 7. Was geändert wurde und warum

Bis zu diesem Stand brach **jeder** Lauf in Epoche 1 mit `L_data = nan` ab.

### Warum ein Abbruch dort terminal ist

`train.py` rechnet **einen** Rollout je OP und Epoche unter `no_grad` und macht
dann `inner_steps` Updates gegen diesen eingefrorenen Puffer. Der Puffer ist
damit ein **Eingang** des ersten Gradientenschritts, kein Ergebnis davon. Steht
dort `inf`, ist jede Vorhersage `nan`, und es gibt keinen ersten
Gradientenschritt, aus dem heraus es besser werden könnte. **Aus einem NaN, in
dem man startet, kann man sich nicht heraustrainieren.**

### Ursache 1 (Haupttreiber): `residual_output`

`field()` lieferte `level(t) + net(...)`, wobei `level` das räumliche Mittel der
Ankerscheibe ist. Damit gilt

```
level(t) ≈ level(t − Δgrid) + mean(net)
```

Das ist ein **Integrator mit Verstärkung exakt 1 und ohne Leck**. Jeder
einseitige Anteil der Netzausgabe akkumuliert über die ~7000 Schritte
unbeschränkt, und nichts zieht ihn zurück. Wie *klein* dieser Anteil ist, spielt
keine Rolle — ein Integrator kennt nur das Vorzeichen. Bei zufälliger
Initialisierung gibt es eines: Swish ist nicht mittelwertfrei, also mittelt sich
`mean(net)` über eine Ziehung nicht weg.

Die Begründung im ursprünglichen Docstring lautete, das Mitführen des Niveaus
halte den Rollout vom Driften ab. Es tut das Gegenteil.

**Gemessen** (20 Epochen, 3 Seeds, ohne jedes Hilfsmittel):

| `residual_output` | History | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| **true** | hybrid `[5,20]` | ABORT | ABORT | ABORT |
| **true** | hybrid `[200,600]` | ABORT | ABORT | ABORT |
| **true** | raw | ABORT | ABORT | ABORT |
| false | hybrid `[5,20]` | 0.0148 | ABORT | 0.0061 |
| false | raw | 0.0074 | 0.0073 | 0.0069 |
| false | hybrid `[200,600]` | 3.3e-4 | 9.8e-5 | 7.2e-5 |

`residual_output: true` bricht **9/9 ab, in jeder History-Konfiguration** — auch
bei `raw`, wo es überhaupt keine Rate-Kanäle gibt. Genau das trennt die beiden
Treiber und identifiziert den Integrator als den wichtigeren.

→ **`residual_output: false`**

### Die Verstärkung des Rate-Kanals — real, aber nicht die Ursache

Der Rate-Kanal ist `(T_ende − T_start) / (lag_n · rate_scale)`. Für eine echte
Rate ist das die richtige Normierung. Die **Rauschverstärkung** derselben Formel
ist aber

```
A = 1 / (lag_n · rate_scale)
```

und bei `[5, 20] s` ist `A ≈ 119`, weil 5 s nur 0.34 % der ~1474 s
Referenzspanne sind.

**Keine andere Normierung entkommt dem.** Für ein glattes Signal gilt
`ΔT über lag ≈ (dT/dt)·lag`, also ist `lag_n · rate_scale` genau der RMS der
Differenz selbst. Nachgerechnet, auf drei Stellen:

| lag [s] | `lag_n · rate_scale` | `RMS(ΔT über lag)` | A |
|---|---|---|---|
| 5 | 0.00768 | 0.00776 | 130 |
| 20 | 0.03070 | 0.03068 | 33 |
| 200 | 0.30704 | 0.30035 | 3.3 |
| 600 | 0.92112 | 0.83285 | 1.1 |

Der Divisor **ist** die Größe, auf die man normiert. Eine echte
5-Sekunden-Änderung auf O(1) zu ziehen kostet zwangsläufig zwei
Größenordnungen Rauschverstärkung. Das ist kein Formelfehler, den man
umschreiben könnte — `A` senken geht nur über ein längeres Segment oder über
`--max-rate-amp`.

**Beides kostet mehr, als es bringt** (Kapitel 8): `A ≈ 119` erzeugt Sättigung,
die `rollout_clamp` abfängt, und dieser Preis ist kleiner als der Verlust an
Signal. Die richtige Antwort auf `A ≈ 119` ist also nicht, `A` zu senken.

→ **`rate_lags: [5.0, 20.0]` bleibt.**

`A` wird bei jedem Start ausgegeben und ab ~100 gewarnt — als Hinweis, nicht als
Handlungsaufforderung.

### Die Sättigungsgrenze: `rollout_clamp: 50.0`

Sättigt `|Tn|` im Rollout-Puffer. Eine plausible Trajektorie lebt innerhalb
weniger Einheiten, der Wert greift also bei einem funktionierenden Modell nie.

**Ohne Physik-Term ist das nur Diagnose** — ein gesättigter Rollout liefert eine
endliche Zahl und eine `[SATURATED]`-Zeile mit Zählstand, statt einer einzigen
`nan`-Zeile ohne Information.

**Mit `w_phys > 0` ist es tragend.** Der Physik-Gradient treibt die Gewichte
schneller aus dem stabilen Bereich:

| Konfiguration (`w_phys = 0.1`) | ohne Clamp | mit Clamp |
|---|---|---|
| `residual_output: false`, `[200,600]`, **128/4** | **ABORT** \| 0.0023 \| 0.0076 | **0.0156 \| 5.9e-4 \| 0.0135** |
| `residual_output: false`, **raw**, 64/3 | **ABORT \| ABORT** \| 0.0148 | **0.0134 \| 0.0095 \| 0.0148** |

`residual_output: false` ist also **notwendig, aber nicht hinreichend**.

Ein dauerhaft hoher `[SATURATED]`-Zählstand bleibt trotzdem ein Warnsignal:
fallend heißt, das Modell fängt sich; flach oder steigend heißt, es tut es
nicht.

### Was ausdrücklich **nicht** hilft

* **Eine bessere Initialisierung.** Setzt man die Ausgabeschicht auf 0, startet
  das Netz als reine Persistenz-Vorhersage und der Rollout bei Initialisierung
  ist perfekt stabil — 0/5 Divergenz über volle 7000 Schritte, `max|T| = 1.19`.
  Nach **20 Adam-Schritten** steht `|W_out|` bei 0.042 und der nächste Rollout
  erreicht 4.7e4. Der stabile Bereich im Gewichtsraum ist zu klein; gewöhnliches
  Training läuft in wenigen Schritten heraus. **Es ist ein Layout-Problem, kein
  Startpunkt-Problem.**
* **`max_rate_amp`.** Deckelt `A` durch Umskalieren von `rate_scale` — also
  durch Umskalieren eines Kanals statt durch Korrektur der Segmentlänge, die
  tatsächlich falsch war. Bleibt als Notnagel, Default `0.0`.
* **Der Clamp allein.** Mit `residual_output: true` verhindert er den Abbruch,
  stabilisiert aber nicht: über 3 Seeds lief einer am Ende wieder weg.

### Nebenbefund: das Testsubstitut hat die Messung verfälscht

`tests/conftest.py` ersetzte Modulus durch ein nacktes `nn.Linear`. Dessen
Default-Init (`kaiming_uniform`, `a=√5`) ist pro Schicht um `√(1/3)` weniger
expansiv als das `xavier_uniform` mit `bias = 0` des echten `FCLayer` — über
einen 4-Schicht-Stack rund **9×**. Eine Stabilitätsmessung gegen den alten Stub
meldete Konfigurationen als stabil, die auf jedem Seed nach `inf` laufen. Der
Stub kopiert die Init inzwischen.

### Nebenbefund: `O(n_t²)` in der Rollout-Schleife

`level(buf[:ti])` reduzierte in jedem Schritt über das **ganze Präfix**, obwohl
nur zwei Zeilen gelesen werden — `O(n_t² · P)` statt `O(n_t · P)`.
`level_rollout()` ersetzt das bitgleich (der räumliche Mittelwert wird vor der
Interpolation genommen, damit die Float-Additionen in derselben Reihenfolge
laufen). Gemessen 1.3× schneller bei `n_t = 7000, P = 363`; die
MLP-Auswertung dominiert.

---

## 8. `hybrid` gegen `raw` — und die Segmentlänge

### Was die Modi tun

| | `raw` | `hybrid` |
|---|---|---|
| Kanäle | `[ T(t−δ), T(t−2δ) ]` | `[ T(t−Δgrid), rate₁, rate₂ ]` |
| `k` | 2 | 3 |
| Was das Netz sieht | zwei vergangene **Temperaturen** | eine vergangene Temperatur plus zwei **Änderungsraten** |
| Normierung | keine (Temperaturen sind schon z-normiert) | Raten durch `lag_n · rate_scale` |
| Verstärkung `A` | — | `1/(lag_n · rate_scale)` |

Der Gedanke hinter `hybrid` ist gut: eine Änderungsrate ist für eine
Wärmeleitungsgleichung die physikalisch relevantere Größe, und ein Netz, das sie
direkt sieht, muss sie nicht aus zwei fast gleichen Temperaturen rekonstruieren.
Der Preis ist der Divisor — und damit `A`.

### Die Messung

`n_t = 7000` (die echte Länge, `A` stimmt also mit den echten Daten überein),
10 Epochen, 3 Seeds, 128/4, `w_phys = 0.1`, `rollout_clamp = 50`,
`residual_output: false`. MAE in °C.

| `rate_lags` | `A` | MAE train | MAE test | pro Seed (test) |
|---|---|---|---|---|
| **`[5, 20]`** | 119 / 30 | 0.784 | **1.207** | 1.695 · 0.781 · 1.144 |
| `[50, 150]` | 12 / 4 | **0.594** | 2.102 | 2.846 · 1.579 · 1.880 |
| `[200, 600]` | 3 / 1 | 0.724 | 2.507 | 3.010 · 1.464 · 3.046 |
| `raw` | — | 0.824 | 2.601 | 2.677 · 1.790 · 3.336 |

**`[5, 20]` gewinnt auf allen drei Seeds**, mit Faktor ~2 Abstand, und keiner
der zwölf Läufe bricht ab.

### Warum die langen Segmente verlieren

`[50, 150]` hat die **beste MAE train** (0.594) und die zweitschlechteste
MAE test (2.102). Das ist Überfitting, und der Mechanismus ist klar: um `A` zu
senken, muss das Fenster wachsen — aber ein 600-s-Fenster auf einer
1474-s-Trajektorie ist **keine Rate mehr, sondern ein Fortschrittsindikator**.
Es sagt dem Netz vor allem, *wo in der Trajektorie* es sich befindet. In-sample
ist das hochinformativ; jenseits von `split_t` reicht das Fenster in Bereiche,
auf die nie gefittet wurde.

`A ≈ 119` erzeugt dagegen Sättigung, die `rollout_clamp` abfängt — sichtbar an
den `[SATURATED]`-Zeilen im Log. Dieser Preis ist kleiner als der Nutzen einer
echten 5-Sekunden-Rate.

`--max-rate-amp` deckelt `A`, ohne das Fenster zu vergrößern. Es dämpft dafür
den Kanal, das Netz bekommt weniger Signal, und es wird ebenfalls schlechter:

| | MAE test |
|---|---|
| `[5, 20]` ohne Deckel | **0.718** |
| `[5, 20]`, `max_rate_amp = 3` | 1.082 |
| `[5, 20]`, `max_rate_amp = 1` | 1.573 |

### Was wir tun

**`history_mode: hybrid` mit `rate_lags: [5.0, 20.0]`** — also die
ursprünglichen Werte. `raw` ist mit 2.601 die schlechteste der vier Varianten.

Das ist eine **Korrektur gegenüber einem früheren Zwischenstand** dieses
Dokuments, der `[200, 600]` empfahl. Der Fehler und was daraus zu lernen ist:

1. **Die Messung lief auf einer verkürzten Trajektorie.** `n_t = 1200` bei einem
   `dtn` für 7000 Schritte heißt, dass nur 1/6 der Referenzspanne überspannt
   wurde. Dadurch war `dTdt_scale` 16.38 statt 2.5 und `[5, 20]` ergab `A = 18`
   statt 119 — und ein 5-s-Fenster waren 4 Gitterschritte statt 25. Es wurde
   also ein anderes Modell gemessen als das, das später läuft.
2. **`L_data` statt MAE als Kriterium.** Auf `L_data` liegt `[200, 600]` zwei
   Größenordnungen vorn. Auf der Lieferzahl ist es das zweitschlechteste.
3. **Eine Korrelation in einem kaputten System ist keine Ursache.** Solange
   `residual_output: true` war, divergierte *jede* Lag-Wahl. Daraus sah `A ≈ 119`
   nach einer zweiten, gleichrangigen Ursache aus. Erst nachdem der Integrator
   weg war, ließ sich `A` isoliert messen — und dann ist es tragbar.

### Der empfohlene Sweep auf echten Daten

`benchmark_arch.py` sweept jetzt die `A`-Achse statt beliebiger Sekunden:

| `rate_lags` [s] | `A` (erstes Segment) |
|---|---|
| `[5, 20]` | 119 — Default |
| `[2, 10]` | 297 — kürzer als der Default |
| `[10, 60]` | 59 |
| `[50, 150]` | 11.9 |
| `[200, 600]` | 3.0 — Fortschrittsindikator-Ende |
| `[5, 20, 60]` | drei Segmente |

```bash
python3 benchmark_arch.py                 # sweept das Gitter oben
python3 benchmark_arch.py --history-mode raw
```

Entscheidungskriterium: **MAE auf dem gehaltenen OP07**, nicht `L_data`.


## 8b. Status — umgesetzt, fehlend, festgelegt

### Die festgelegten Werte

| Schlüssel | Wert | Status |
|---|---|---|
| `residual_output` | **`false`** | **fest.** Der eigentliche Fix, mechanistisch begründet (Kapitel 7). Nicht anfassen ohne die Herleitung zu widerlegen. |
| `rollout_clamp` | **`50.0`** | **fest.** Tragend, sobald `w_phys > 0`. |
| `rate_lags` | **`[5.0, 20.0]`** | **vorläufig fest.** Gemessen am besten (Kapitel 8), aber auf synthetischen Labels. Offene Physikfrage dazu unten. |
| `max_rate_amp` | `0.0` (aus) | **fest.** Schadet gemessen monoton. |
| `history_mode` | `hybrid` | **vorläufig.** `raw` war die schlechteste der vier Varianten. |

### Umgesetzt

* `residual_output: false` als Config und CLI-Schalter, überall durchgereicht
* `rollout_clamp` neu, inklusive `[SATURATED]`-Meldung mit Zählstand je Epoche
* `A = 1/(lag_n · rate_scale)` wird bei jedem Start ausgegeben, Warnung ab ~100
  (`data.hybrid_rate_amplification`, aus `PINNmodulusTwoExtProfiles` portiert)
* Abbruchmeldung nennt `residual_output` als ersten Verdächtigen
* `bench_common.make_train_args`: `residual_output`-Default von `True` auf
  `False` — hätte sonst **jeden** Benchmark in Epoche 1 abbrechen lassen
* `benchmark_arch.DEFAULT_LAG_SETS` spannt jetzt die `A`-Achse mit `[5,20]` als
  Baseline; das alte Gitter lag mit `A` zwischen 20 und 297 vollständig im
  divergenten Bereich
* `level_rollout()`: `O(n_t²·P) → O(n_t·P)`, bitgleich, gemessen 1.3×
* `tests/conftest.py`: Modulus-Stub nutzt die Init des echten `FCLayer`
* `tests/test_rollout_stability.py`, `tools/rollout_divergence.py`, CI
* 92 Tests grün

### Fehlt

* **Ein einziger Lauf auf echten Daten.** Diese Session hatte keinen Zugriff auf
  `data_cache/` — alle Zahlen sind synthetisch.
* Echte MAE-Werte für Kapitel 6; die dortigen sind synthetisch und markiert.
* Eine Bewertung des Physik-Terms. Auf synthetischen Daten prinzipiell nicht
  möglich: die erfundene Trajektorie erfüllt die erfundene
  Wärmeleitungsgleichung nicht, der Physik-Term zieht dort gegen die Labels.

### Was als nächstes zu tun ist

1. **Offene Physikfrage zu den 5 Sekunden.** `A = 119` heißt: die echte
   Temperaturänderung über 5 s beträgt `1/119 = 0.0084` in z-normierten
   Einheiten — bei `T_sigma = 5 K` rund **42 Millikelvin**. Liegt das über der
   Auflösung der Simulation, oder misst der Kanal dort im Wesentlichen
   Diskretisierungsrauschen? Das entscheidet die Physik, nicht die Messung.
   Falls letzteres: längere erste Segmente prüfen (`[20, 60]` → `A = 30/10`),
   aber nicht so lang, dass der Kanal zum Fortschrittsindikator wird.
2. `python3 smallBench.py` — erster echter Lauf. Prüfen: die `A`-Startzeile
   (wirklich ~119/30?), `[CFL OK]`, kein `[ABORT] epoch 1`, und ob der
   `[SATURATED]`-Zählstand über die Epochen fällt.
3. `python3 tools/rollout_divergence.py` auf echten Daten — bestätigt oder
   widerlegt die Divergenztabelle aus Kapitel 7, kostet Sekunden.
4. `python3 benchmark_arch.py` — der Lag-Sweep, Kriterium MAE auf dem
   gehaltenen OP, **niemals** `L_data`.
5. `python3 benchmark_wphys_wbc.py` — was der Physik-Term bringt.

Die vollständige Übergabe steht in
[`../UEBERGABE_2026-08-27.txt`](../UEBERGABE_2026-08-27.txt).

---

## 9. Einschränkungen

Was dieses Dokument **nicht** belegt:

1. **Alle Trainings- und MAE-Zahlen stammen von einem synthetischen Bundle**, da
   in der Entwicklungsumgebung kein `data_cache/` vorhanden ist. Die *Richtung*
   ist robust (9/9 Abbruch gegen 0/9) und mechanistisch erklärt; die *Beträge*
   sind es nicht.

2. **Übertragbar ist `A`, nicht die Segmentlänge in Sekunden.** `A` hängt über
   `rate_scale = dTdt_scale` am Datensatz. Die Messungen in Kapitel 6 und 8 sind
   in der echten Geometrie gemacht (`n_t = 7000`, `dTdt_scale = 2.467` gegen
   echte `2.479`), gelten für OP01–05 also direkt. Für
   `PINNmodulusTwoExtProfiles` gelten sie **nicht**: das Pooling über OP01–OP16
   vergrößert `T_sigma`, verkleinert `dTdt_scale` und erhöht `A` über 119 hinaus.

   Frühere Messungen auf einer verkürzten Trajektorie (`n_t = 1200`) hatten
   `dTdt_scale = 16.38` und damit `A = 18` statt 119 — sie taugen für die
   Lag-Wahl nicht und sind in Kapitel 8 als Fehlerquelle dokumentiert.

3. **Die Trajektorien im Test sind 1200–4000 Schritte lang, echte haben ~7000.**
   Ein Integrator-Problem wächst mit `n_t`; bei `n_t = 4000` wurde
   nachgeprüft (alt 3/3 Abbruch, neu 3/3 bei ~1e-4), bei 7000 nicht.

4. **Die Überfitting-Aussage zu `hybrid` ist vermutlich überzeichnet**, siehe
   Kapitel 8.

### Offene Schritte auf echten Daten

1. `python3 tools/rollout_divergence.py` — Divergenz bei Initialisierung
   nachmessen. Braucht weder Modulus noch Daten, kostet Sekunden.
2. Ein voller `smallBench.py`-Lauf mit den neuen Defaults, um echte MAE-Zahlen
   für dieses Dokument zu bekommen.
3. Der `benchmark_arch.py`-Sweep aus Kapitel 8.

---

## 10. Reproduktion

```bash
# Rauchtest, wenige Minuten
python3 PINNmodulusTwo/smallBench.py

# Voller Lauf mit den Defaults aus config.yaml
python3 PINNmodulusTwo/train.py

# Divergenz bei Initialisierung nachmessen (ohne Modulus, ohne Daten)
python3 PINNmodulusTwo/tools/rollout_divergence.py

# Tests und Skalierungs-Selftest
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/selftest.py
```

Worauf im Log zu achten ist:

| Zeile | Bedeutung |
|---|---|
| `hybrid history amplification A = ...` | über ~100 sind die Rate-Kanäle der Treiber |
| `[CFL OK]` / `[CFL WARN]` | Zeitschritt gegen das Stabilitätslimit |
| `[SATURATED] epoch N: ... x/y steps` | der Clamp greift — fallend ist gut, flach oder steigend nicht |
| `[ABORT] epoch 1` | darf mit diesen Defaults nicht mehr vorkommen |

### Weiterführend

* [`ARCHITECTURE.md`](ARCHITECTURE.md) Kapitel 3.1 — Mechanismus im Detail
* [`README.md`](README.md) — Projektübersicht und alle CLI-Schalter
* [`../PINNmodulusTwoExtProfiles/README_UPDATE_2026-08-27.md`](../PINNmodulusTwoExtProfiles/README_UPDATE_2026-08-27.md)
  — derselbe Befund für die Profilerweiterung
