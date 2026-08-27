# Modellkritik — was gefixt ist, was offen ist, und woran man es sieht

Diese Datei beantwortet zwei Fragen:

1. **Was war am Modell kaputt, was davon ist repariert, und wie sicher ist das?**
2. **Was muss man in welchem Test sehen, um zu wissen, welcher Schritt als
   nächstes kommt?**

> ## Der wichtigste Satz zuerst
>
> **Alles hier ist bisher nur mathematisch verifiziert, nichts davon auf echten
> Daten gemessen.** Die Fixes sind gegen einen Modulus-Stub geprüft — Gradienten-
> gleichheit, Kausalität, Invarianz der Ableitungen, Form des Residuums, ein
> End-to-End-Lauf auf synthetischen OPs. Das beweist, dass der Code das tut, was
> er behauptet. Es beweist **nicht**, dass der MAE fällt.
>
> Ob die Diagnose stimmte, entscheidet [Schritt A](#schritt-a-smallbench--der-lauf-der-die-diagnose-prüft).
> Bis dahin ist jede Zahl in dieser Datei eine Erwartung, keine Messung.

Laufanleitung und Kommandos: **[README_GPU_SERVER.md](README_GPU_SERVER.md)**.
Diese Datei sagt, *worauf man dabei schaut*.

---

## 1. Der Befund

Der freilaufende Rollout-Fehler blieb groß. Vier Ursachen, in der Reihenfolge
ihrer Hebelwirkung.

### 1.1 Das Trainingsbudget — 300 Gradientenschritte

Die Schleife machte einen `opt.step()` **je OP je Epoche**: 5 OPs × 60 Epochen =
**300 Adam-Updates** für ein ~70k-Parameter-MLP. Und sie bezahlte für jedes
einzelne einen vollen, ~7000 Schritte langen sequentiellen Rollout. `batch_data`
wurde geparst und dann nie benutzt — der Datenterm lief Full-Batch.

**Fix:** Der Rollout wird einmal je OP je Epoche unter `no_grad` berechnet und
für `--inner-steps` Minibatch-Updates wiederverwendet. 30 000 statt 300 Updates.

**Warum das keine Näherung ist:** Die Rekurrenz hat ihre History ohnehin zwischen
den Schritten detached. Der alte Vollsequenz-Gradient war damit bereits eine
Summe unabhängiger Gradienten je `(t, Punkt)` gegen eine konstant gehaltene
Trajektorie — genau die Größe, die ein Minibatch erwartungstreu schätzt.

**Verifiziert:** Minibatch-Gradient gegen den alten Vollsequenz-Gradienten,
relative Abweichung **1e-15**. Das ist so nah an „identisch", wie float64 kommt.

### 1.2 Die Ausgabe-Parameterisierung

Das Netz sagte den absoluten z-gescorten Wert vorher und musste die Identität
`T(t) ≈ T(t−0.2 s)` in jedem der ~7000 Schritte neu erzeugen.

**Fix:** Es sagt die Abweichung vom **räumlich gemittelten** Temperaturniveau der
Ankerscheibe vorher (`residual_output`, Default an).

**Warum der räumliche Mittelwert und nicht der Anker je Punkt:** Der per-Punkt-
Anker kommt aus einem diskreten Buffer und ist für autograd unsichtbar. `∇²T`
hätte damit die Krümmung des Ankers verloren — also den größten Teil — und der
Physik-Term wäre still falsch geworden. Ein räumlich konstantes Niveau hat
Laplace-Operator exakt null und trägt genau den Anteil, der über einen langen
Rollout wegdriftet: das Gesamtniveau.

**Verifiziert:** Niveau ist über die Punkte konstant (Streuung 3e-17), `dT/dx`
und `∇²T` sind unverändert (Abweichung exakt 0).
**Nicht verifiziert:** dass die Drift dadurch tatsächlich sinkt. Siehe
[Schritt A](#schritt-a-smallbench--der-lauf-der-die-diagnose-prüft).

### 1.3 Das Physik-Residuum

`physics.py` teilte **jeden Term durch eine andere RMS**. Das skaliert die
Gleichung nicht, es ändert sie:

```
dTdt/√a − aniso/√b − Qsrc/√c   ist nur für a = b = c dieselbe Gleichung wie
dTdt − aniso − Qsrc
```

Die drei unterschieden sich um Größenordnungen. `aniso_scale` war nicht einmal
eine Termgröße, sondern die RMS des Fourier-Tensors — der Faktor `∇²T` fehlte
darin. Die lernbaren `src_gain`/`diff_gain` reparierten diesen Schaden; daher
brauchten sie die 25-fache Lernrate. Nebenbei spannen sie als freie Faktoren vor
zwei der drei Terme eine Richtung auf, in der `L_phys` durch ein *konstantes
Feld* minimiert wird.

**Fix:** Das Residuum wird in seinen eigenen Einheiten zusammengesetzt und
**einmal** skaliert. Die Gains stehen fest auf 1.0 (`--learn-gains` gibt sie
frei). `phys_scale` enthält nur noch echte Termgrößen.

**Verifiziert:** Das Residuum ist exakt `(dTdt − aniso − Qsrc) / phys_scale`
(Abweichung 0 gegen eine Handrechnung).

### 1.4 Das In-Time-Leck

Der Datenterm lief über die ganze Trajektorie, während `metrics.txt` und die
Spalte `MAE_in_C` der Benchmarks `[split_t:]` derselben Trainings-OPs als
ausgehaltenen In-Time-Check berichten — also **in-sample**. `data.py` behandelt
diesen Schwanz längst als ausgehalten: `T_mu`, `T_sigma` und jede Config- und
Quellenstatistik werden nur über `[:split_t]` gepoolt.

**Fix:** Der Datenterm liest Labels nur bis `split_t`. Physik- und BC-Term
bleiben auf der vollen Trajektorie — sie brauchen keine Labels.

**Kostet** 20 % Trainingssignal, **macht** eine bisher bedeutungslose Spalte echt.

---

## 2. Was offen ist

| # | Punkt | Warum es zählt | Aufwand |
|---|---|---|---|
| O1 | **BPTT-Länge 1** | Das Modell wird als Ein-Schritt-Prädiktor optimiert und über ~7000 Schritte bewertet. Nichts im Loss bestraft Fehlerakkumulation. Fenster von 20–50 Schritten wären die einzige Änderung, die Drift *direkt* adressiert | mittel, teuer zur Laufzeit |
| O2 | **Kein LR-Schedule** | Bei 300 Schritten egal, bei 30 000 nicht mehr. Cosine-Decay lohnt sich erst jetzt | klein |
| O3 | **Auswahl nach Trainings-Loss** | `early_stopping_patience` beobachtet die *Trainings*-Loss, und berichtet wird der Endzustand, nicht der beste. Bei 30 000 Schritten ist Overfitting erstmals möglich | klein |
| O4 | **Fünf Trainings-OPs** | Harte Grenze für Cross-OP-Generalisierung. Kein Hyperparameter repariert einen Config-Raum, den fünf Trajektorien nicht aufspannen | **nicht durch Code lösbar** |
| O5 | **Keine dokumentierte Zielgenauigkeit** | Ohne die Anforderung des Alterungsmodells kann kein Benchmark sagen, ob ein Ergebnis gut ist. 8 °C besteht den `smallBench`-Check und ist als Alterungs-Eingang vermutlich trotzdem zu grob | eine Zahl von der Fachseite |

O4 und O5 sind die wichtigsten — und beide keine Code-Fragen.

---

## 3. Der Entscheidungsbaum

Reihenfolge einhalten. Jeder Schritt entscheidet, ob der nächste überhaupt
sinnvoll ist.

### Schritt A: smallBench — der Lauf, der die Diagnose prüft

**Das ist der wichtigste Lauf im ganzen Dokument.** Zwei Läufe, ~5 min je:

```bash
# neu
python3 PINNmodulusTwo/smallBench.py

# alter Stand als Baseline -- die Konfiguration vor dem Umbau
python3 PINNmodulusTwo/smallBench.py \
    --inner-steps 1 --no-residual-output --learn-gains --loss-balance legacy
```

`--loss-balance legacy` gehoert dazu: im Default `ema` wird auch `L_data` durch
seine eigene laufende Groesse geteilt, vorher blieb es roh. Ohne das Flag misst
der Baseline-Lauf eine Mischung aus altem und neuem Stand, und der Vergleich
beantwortet nicht mehr die Frage, fuer die er da ist.

Beide schreiben nach `artifacts/smallBench_results.txt`. Verglichen wird
`Test MAE`.

| Was du siehst | Was es heißt | Nächster Schritt |
|---|---|---|
| Neu **deutlich** besser als Baseline | Die Diagnose stimmte, Unterversorgung war der Engpass | Schritt B |
| Neu ≈ Baseline | **Meine Hauptdiagnose war falsch.** Das Modell war nicht unterversorgt — dann liegt es an der Kapazität, an der History-Struktur oder an O4 | Erst 8.2 (Architektur), Gewichte-Sweep später |
| Neu **schlechter** | Vermutlich Overfitting durch 100× mehr Updates (O3), oder der Residual-Ausgang schadet auf diesen Daten | `--no-residual-output` einzeln testen, um die beiden Ursachen zu trennen |
| `✗ SOME CHECKS FAILED`, Loss NaN | CFL oder History-Rückkopplung | [Kapitel 10](README_GPU_SERVER.md#10-troubleshooting), nicht weitermachen |
| Test-MAE > 20 °C in **beiden** | Das Modell lernt grundsätzlich nicht | Daten prüfen, nicht Hyperparameter |

**Zwei Zahlen im Baseline-Lauf** — `src_gain(final)` und `diff_gain(final)` in
`artifacts/metrics.txt`:

- Nahe bei 1.0 → die freigegebenen Gains hatten nichts zu korrigieren.
- Weit weg von 1.0 (etwa > 5 oder < 0.2) → sie korrigieren etwas an der einen
  Skalierung, das der feste Wert 1.0 nicht trifft.
- Nahe bei **0** → der degenerierte Fall: `L_phys` wurde durch Abschalten der
  Physik minimiert. Das ist der Grund, warum die Gains ueberhaupt festgesetzt
  wurden.

> **Was dieser Lauf NICHT mehr beantwortet:** Diagnose 1.3 laesst sich damit
> nicht bestaetigen oder widerlegen. Die termweise Normierung ist aus
> `heat_residual` in *jedem* Modus verschwunden — auch der Baseline-Lauf rechnet
> also schon mit der Ein-Skalen-Assemblierung, und `--residual-norm legacy`
> stellt nur den alten Gesamtdivisor `sqrt(phys_scale)` her, nicht die drei
> verschiedenen Term-Divisoren. 1.3 haengt an der Handrechnung unter
> **Verifiziert** in Abschnitt 1.3, nicht an diesem Lauf.

**Und der Drift-Test** — die einzige Prüfung von 1.2, die überhaupt möglich ist,
weil ein 60-Schritt-Rollout auf synthetischen Daten Drift nicht erzeugen kann:

```bash
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP07.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
frueh, spaet = e[1:n//5].mean(), e[-(n//5):].mean()
print(f'frueh {frueh:.3f} C   spaet {spaet:.3f} C   Wachstum {spaet/frueh:.2f}x')
"
```

| Wachstumsfaktor | Deutung |
|---|---|
| nahe 1 | keine Drift — der Fehler ist Bias, keine Akkumulation. **O1 lohnt dann nicht** |
| 1,5–3 | moderate Drift, O1 ist ein sinnvoller nächster Hebel |
| > 3 | Drift dominiert. **O1 vorziehen**, vor jedem Gewichte-Sweep |

Ein Gewichte-Sweep bei starker Drift misst hauptsächlich, welches Gewicht die
Drift zufällig am wenigsten verstärkt — das ist die teure Art, nichts zu lernen.

---

### Schritt B: 6.3 + 6.4 — Laufzeit und VRAM

Erst wenn Schritt A grün ist. Siehe
[README_GPU_SERVER 6.3](README_GPU_SERVER.md#63-wie-lange-dauert-eine-epoche-10-min)
und [6.4](README_GPU_SERVER.md#64-batchgrößen--was-20-gb-vram-hergeben).

Abzulesen sind zwei Zeilen:

```
epoch 1 ... [118.7s/epoch = 112.4s rollout + 6.3s x100 inner, ...]
  peak VRAM 0.41 GB of 20.0 GB (batch_data=2048 batch_phys=256 batch_bc=128)
```

| Was du siehst | Nächster Schritt |
|---|---|
| `inner` klein gegen `rollout` (< ⅓) | Alles gut. `--epochs` aus der Tabelle in 6.3 wählen |
| `inner` groß (> ⅓) | `--inner-steps 50` — halbe Updates, nicht halber Nutzen |
| `peak VRAM` weit unter 20 GB | Batches hoch: `--batch-data 8192 --batch-phys 2048 --batch-bc 512`, dann 6.3 wiederholen |
| Batches hoch **und** `inner` kaum gewachsen | Sie waren gratis — behalten |
| `CUDA out of memory` | `--batch-phys` zuerst halbieren, das ist der teuerste Term |

Was hier gewählt wird, muss in **7.1 und 7.2 identisch** sein.

---

### Schritt C: 7.1–7.3 — Range-Probe der Gewichte

Neun Trainings, aufgeteilt in Blöcke ≤ 5 h. Die Aufteilung und das Budget stehen
in [Kapitel 7](README_GPU_SERVER.md#7-range-probe-der-loss-gewichte--in-drei-schritten).

Die entscheidende Ausgabe ist das Verdikt in
`artifacts/benchmark_wphys_wbc_best.txt`:

```
w_phys (at w_bc=0.1):
  0->9.12  0.01->8.71  0.1->8.20  1->8.44  3->9.90
  best w_phys=0.1 (val 8.200 °C), span over the decades = 1.700 °C
  vs w_phys=0 (9.120 °C): +0.920 °C at the best weight -> HELPS
```

**Die Zeile `vs w=0` ist die Antwort auf „bringt der Term überhaupt etwas".** Eine
Rangfolge von fünf Gewichten sagt das nicht — sie sagt nur, welches der fünf am
wenigsten schlecht ist. Deshalb ist `w = 0` als Kontrollpunkt fest in beiden
Achsen.

Die Gewichte sind **Mischanteile, keine rohen Faktoren**: `train.py` teilt
`L_phys` und `L_bc` vor der Gewichtung durch einen EMA ihrer eigenen Größe, also
heißt `w = 1` „zählt so viel wie der Datenterm". Darum liegt die Range
`[0, 0.01, 0.1, 1.0, 3.0]` um 1 herum, und darum ist die 3 dabei.

| Was im Verdikt steht | Nächster Schritt |
|---|---|
| `HELPS`, Minimum **im Inneren** der Range | 5×5-Gitter (8.3), zentriert auf diese Dekade |
| `NOT BRACKETED ... LARGEST weight` | Range nach oben erweitern: `--w-phys 0 1 3 10 30`. **Nicht** den Randwert als Optimum berichten |
| `no measurable gain` / `HURTS` | Dieser Term hilft hier nicht — Gitter dafür überspringen |
| `span is BELOW the seed spread` | Das Gewicht bewegt den Fehler nicht |
| **Beide** Achsen ohne Gewinn | Das Problem liegt nicht bei den Gewichten → Schritt D, dann 8.2 |
| `seed spread unknown` | Ein Seed. Vor jeder Schlussfolgerung Schritt D |

---

### Schritt D: 8.1 — Seed-Streuung

Ohne diese Zahl ist jede Rangfolge aus Schritt C unlesbar.

| Was du siehst | Was es heißt |
|---|---|
| Streuung ≪ den Unterschieden aus C | Die Rangfolgen sind echt, Gitter lohnen |
| Streuung ≈ den Unterschieden | Der Sieger ist Zufall. Alle folgenden Sweeps mit `--seeds 0 1 2`, Gitter klein halten |
| Streuung > den Unterschieden | Feineres Gitter bringt **nichts**. Erst O1/O3, oder O4 akzeptieren |

---

### Schritt E: 8.2 / 8.3 — Architektur und Gitter

Nur wenn D sagt, dass Unterschiede überhaupt messbar sind.

Die entscheidende Ausgabe von 8.2 ist **„Span per axis"**: liegt die Spannweite
einer Achse unter der Seed-Streuung, steht dort *„this knob does not matter
here"*.

**Wenn alle Achsen das sagen** — Breite, Tiefe, Lags, Ankerabstand — dann ist die
Aussage nicht „schlechter Benchmark", sondern: das Modell ist nicht durch seine
Hyperparameter begrenzt. Dann bleiben O1 und O4.

---

## 4. Woran man sieht, dass es an den Daten liegt (O4)

Diese Signale zeigen zusammen auf zu wenige Betriebspunkte, nicht auf
Hyperparameter:

- In-Time-MAE (`MAE_in_C`) **klein**, Val/Test-MAE **groß** — das Modell passt zu
  jeder Trajektorie, die es gesehen hat, und trifft keine neue. Das ist die
  klassische Signatur.
- Alle Architektur-Achsen in 8.2 flach.
- Beide Gewichte in Schritt C ohne Gewinn gegen `w = 0`.
- Boxplot in `_probe_boxplot.png`: der Median passt, aber einzelne Sensoren
  liegen dauerhaft weit daneben, mit langen oberen Whiskern.

Trifft das meiste davon zu, ist der nächste Schritt **mehr OPs simulieren**, nicht
mehr sweepen.

---

## 5. Verifikationsstand im Überblick

| Behauptung | Wie geprüft | Auf echten Daten? |
|---|---|---|
| Minibatch-Gradient = alter Vollsequenz-Gradient | numerisch, rel. 1e-15 | — |
| History liest nie den Vorhersagezeitpunkt | Sondenwert 1e6, Leck 0 | — |
| Niveau räumlich konstant, `dT/dx` und `∇²T` unverändert | numerisch, exakt 0 | — |
| Residuum = `(dTdt − aniso − Qsrc)/phys_scale` | gegen Handrechnung, exakt 0 | — |
| Gains fixiert / freigebbar | Parameterliste geprüft | — |
| Trainingsschleife läuft, Schrittzahl stimmt, Loss fällt | End-to-End auf synthetischen OPs | — |
| Kreuz-Aufteilungen decken alle 9 Punkte ab | alle vier Aufteilungen geprüft | — |
| Verdikt-Fälle (innen / Rand / hilft nicht) | drei synthetische Fälle | — |
| **Residual-Ausgang reduziert Drift** | **nicht geprüft** — 60-Schritt-Rollouts erzeugen keine Drift | **offen, Schritt A** |
| **MAE fällt** | **nicht geprüft** | **offen, Schritt A** |

Die Prüfungen liefen gegen einen Modulus-Stub (Torch war verfügbar, Modulus und
`data_cache/` nicht). Sie prüfen die Mathematik der geänderten Pfade, nicht das
Ergebnis.

**Nachtrag nach dem Zusammenführen der Parallel-PRs.** Die Zeile *"History liest
nie den Vorhersagezeitpunkt"* galt nach dem Merge nur noch für den allgemeinen
Pfad: der schnelle Rollout-Pfad (`rollout_plan`) kannte den kausalen Clamp nicht
und las bei `delta_grid < dtn` eine Zeile, die der allgemeine Pfad nie liest.
Beide laufen jetzt durch denselben Ausdruck. `tests/test_history_fastpath.py`
prüft die Gleichheit mit `torch.equal` statt `allclose` und deckt beide Pfade in
allen History-Layouts ab (84 passed, 1 skipped); die Datei brach nach dem Merge
schon beim Import ab, weil sie eine entfernte Funktion importierte.
