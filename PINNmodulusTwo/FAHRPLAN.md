# Fahrplan — OP01–OP07, von vorne

**Geltungsbereich:** nur der Basis-Datensatz `OP01–OP07` (Training OP01–05,
Validierung OP06, Test OP07). **Nicht** `PINNmodulusTwoExtProfiles/` — dort
poolt die Normierung über OP01–OP16, das verschiebt `T_sigma`, `dTdt_scale` und
`A`, und nichts aus diesem Plan überträgt sich dorthin.

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du
nur eine Datei liest, dann diese.

---

## 1. Wo wir stehen

| | Stand | Beleg |
|---|---|---|
| Datenpipeline CSV → NPZ → Training | ✅ läuft | Bericht 28.08. §1 |
| `A = 118.9 / 29.7`, `dTdt_scale = 2.479` | ✅ gegen Übergabe verifiziert | `UEBERGABE` Z. 29, 121–122 |
| Training läuft ohne Abbruch | ✅ | Bericht §4 |
| Rollout zahm | ❌ 342 saturierte Schritte auf einem **Trainings**-OP | `train.py:674` |
| Loss-Balance | ❌ beide Läufe `FAIL` | `smallBench.py:262` |
| Physik-Term | ❌ kollabiert (`L_phys_bal = 2.7e-06`) | Bericht §6 |
| **Schlägt das Modell „nichts tun"?** | ❓ **unbekannt** | kein Maßstab auf echten Daten |
| Schritt A (A/B gegen alten Stand) | ❌ nie gelaufen | `README_MODEL_CRITIQUE.md:159` |
| Drift-Test | ❌ nie gelaufen | ebd. Z. 206 |

Die vorletzte Zeile ist die wichtigste. **Alles andere zu optimieren, bevor sie
beantwortet ist, ist Blindflug.**

---

## 2. Das Problem mit dem Ist-Zustand

```
Doku:        5041 Zeilen in 9 Dateien   (README_GPU_SERVER allein: 1265)
Benchmarks:  2735 Zeilen in 4 Dateien   (benchmark_wphys_wbc allein: 1366)
Ergebnis:    ein Modell, das noch nie einen trivialen Vorhersager geschlagen hat
```

Die Benchmarks sind inhaltlich gut — das Problem ist, dass es vier Einstiege
gibt, deren Reihenfolge nirgends steht und die sich in den Docstrings sogar
**widersprechen**:

* `smallBench.py:320` → als Nächstes `benchmark_balance.py`
* `benchmark_balance.py` → „läuft VOR dem Gewichts-Sweep"
* `benchmark_arch.py` → „dieselben Stunden hier beantworten Fragen, die nie
  gestellt wurden" (also Architektur vor Gewichten)

Drei Dateien, drei Meinungen, kein Fahrplan. Abschnitt 6 räumt das auf.

---

## 3. Die eine Regel

> **Nichts, was Stunden kostet, bevor das Billige gelaufen ist, das es entwerten
> könnte.**

Jede Phase unten hat ein **Tor**. Ist es rot, geht es nicht weiter — dann wird
das Tor repariert, nicht die nächste Phase gestartet.

---

## 4. Die Phasen

### Phase 0 — Synthetisch (Minuten, keine Daten, kein GPU)

Prüft die Mathematik, nicht das Ergebnis. Läuft überall, auch ohne
`data_cache/`.

```bash
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/tools/rollout_divergence.py
```

**Tor G0:** alles grün. Rot → nicht weitermachen, `README_GPU_SERVER.md`
Kapitel 10.

> **Wozu synthetisch überhaupt gut ist:** für die **Rangfolge** zwischen
> Konfigurationen, nicht für Beträge. `README_ERSTER_TEST.md` Kapitel 9.1 sagt
> das ausdrücklich — „die *Richtung* ist robust, die *Beträge* sind es nicht".
> Wer synthetische MAE-Zahlen als Vorhersage für echte liest, macht genau den
> Fehler aus dem Bericht vom 28.08. §9.

### Phase 1 — Daten prüfen (Minuten, echte Daten, kein Training)

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07
python3 PINNmodulusTwo/tools/data_probe.py
python3 PINNmodulusTwo/tools/interface_probe.py
```

**Tor G1:**

| Prüfung | Muss |
|---|---|
| `A` für `[5, 20]` | ≈ 119 / 30 — sonst stimmt die Datenaufbereitung nicht |
| `dTdt_scale` | ≈ 2.479 |
| SNR | > 100, sonst misst der kurze Rate-Kanal Rauschen |
| Grenzflächenanteil | **notieren** — entscheidet später über `ARCHITECTURE.md` 4.1 Option A vs. B |

### Phase 2 — Der Maßstab (Minuten) ← **hier fehlt bisher alles**

```bash
python3 PINNmodulusTwo/smallBench.py --epochs 1
```

> Dieser Lauf meldet zwangsläufig `FAIL` — `converged` braucht mindestens zwei
> Epochen (`smallBench.py:253`). Das ist hier egal: die Latte hängt nicht am
> Modell, sondern nur an den Daten, und wird unter der Summary-Tabelle in jedem
> Fall gedruckt.

Interessiert an diesem Punkt **nicht** die MAE des Modells, sondern die drei
Zeilen, die `smallBench.py` inzwischen darunter druckt:

```
  vs. persistence T(t)=T(0):     ?? °C
  vs. constant mean of train:    ?? °C
  -> the bar to beat:            ?? °C
```

**Tor G2:** die Latte steht als Zahl fest, gemessen auf dem **echten** OP07.
Die synthetischen 11.96 / 6.60 aus `README_ERSTER_TEST.md` sind **nicht** diese
Zahl und dürfen sie nicht ersetzen.

Alles ab hier wird gegen diese eine Zahl gelesen.

### Phase 3 — Schritt A, das A/B (≈ 20 min CPU, Minuten GPU)

Der Lauf, den `README_MODEL_CRITIQUE.md:159` „den wichtigsten im ganzen
Dokument" nennt und der bis heute nicht gemacht wurde. **Zwei** Läufe:

```bash
python3 PINNmodulusTwo/smallBench.py                          # neuer Stand
python3 PINNmodulusTwo/smallBench.py \
    --inner-steps 1 --no-residual-output --learn-gains --loss-balance legacy
```

**Tor G3:** aus `README_MODEL_CRITIQUE.md:180-186`:

| Ergebnis | Heißt | Dann |
|---|---|---|
| neu deutlich besser | Unterversorgung war der Engpass | Phase 4 |
| neu ≈ Baseline | Hauptdiagnose war falsch | Architektur zuerst (Phase 5b) |
| neu schlechter | Overfitting durch 100× Updates, oder Residual-Ausgang | `--no-residual-output` einzeln testen |
| beide > 20 °C | lernt grundsätzlich nicht | **Daten prüfen, nicht Hyperparameter** |

Dazu der Drift-Test, der entscheidet, ob ein Sweep überhaupt etwas misst.

> ⚠️ **`pred_OP07.npz` schreibt `train.py`, nicht `smallBench.py`**
> (`train.py:844`). `README_MODEL_CRITIQUE.md:206` stellt den Drift-Test direkt
> hinter den smallBench-Abschnitt, was so gelesen ins Leere läuft — es braucht
> vorher einen `train.py`-Lauf.

```bash
python3 PINNmodulusTwo/train.py --epochs 10        # erzeugt artifacts/pred_OP07.npz
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP07.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
print('Wachstum', e[-(n//5):].mean() / e[1:n//5].mean())
"
```

Wachstum > 3 → Drift dominiert, O1 vor jedem Gewichts-Sweep.

### Phase 4 — Rollout zahm bekommen (Stunden)

**Tor G4: `[SATURATED]` muss in der letzten Epoche bei 0 stehen.**

Aktuell 342. `train.py:674` sagt dazu wörtlich: *„it is not a prediction, and a
run that only survives because of this is not trained."* Ein Sweep über
Konfigurationen, deren Rollout wegläuft, rankt Guard-Verhalten, nicht Physik.

Hebel in dieser Reihenfolge: mehr Epochen → `lr` runter → `rollout_clamp`
prüfen → `A` senken über längere `rate_lags`.

### Phase 5 — Erst jetzt Benchmarks

**5a — Balance (~4 h GPU).** `benchmark_balance.py`. Das gerissene Kriterium aus
Phase 3/4 **ist** die Balance, und `w_phys` bedeutet nichts, solange nicht
feststeht, wie die Terme skaliert werden.

**5b — Architektur (~1 Tag GPU bei `--epochs 20`).** `benchmark_arch.py`,
eine Achse nach der anderen. Nur wenn 5a die Balance geklärt hat.

**5c — Gewichte, und zwar `--probe` (9 Punkte), nicht das 10×10-Gitter.**
`benchmark_wphys_wbc.py --probe`. Das volle Gitter ist laut `smallBench.py:321`
„100 trainings (~6-8 days)" und misst Gewichte, bevor feststeht, was ein Gewicht
hier bedeutet.

**Tor G5, über allem:** sobald ein Lauf die Latte aus Phase 2 unterbietet, ist
das Modell zum ersten Mal mehr wert als „nichts tun". Vorher ist jede
Rangfolge zwischen Konfigurationen eine Rangfolge zwischen Verlierern.

---

## 5. Was du lokal machen musst

Kopiervorlage, in dieser Reihenfolge. Die ersten vier Blöcke brauchen **kein**
GPU.

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate

# Phase 0 -- Minuten, keine Daten
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/tools/rollout_divergence.py

# Phase 1 -- Minuten
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07
python3 PINNmodulusTwo/tools/data_probe.py
python3 PINNmodulusTwo/tools/interface_probe.py      | tee PINNmodulusTwo/artifacts/interface.txt

# Phase 2 -- Minuten. Nur die "bar to beat"-Zeile zaehlt.
python3 PINNmodulusTwo/smallBench.py --epochs 1      | tee PINNmodulusTwo/artifacts/latte.txt

# Phase 3 -- ~20 min CPU, das A/B
python3 PINNmodulusTwo/smallBench.py                 | tee PINNmodulusTwo/artifacts/A_neu.txt
python3 PINNmodulusTwo/smallBench.py --inner-steps 1 --no-residual-output \
        --learn-gains --loss-balance legacy          | tee PINNmodulusTwo/artifacts/A_alt.txt
```

> ⚠️ **Beide A/B-Läufe schreiben in dieselbe
> `artifacts/smallBench_results.txt`** — der zweite überschreibt den ersten.
> `README_MODEL_CRITIQUE.md:177` verweist genau auf diese Datei zum Vergleich.
> Das `tee` oben fängt stdout ab und genügt; wer die Datei selbst braucht,
> kopiert sie zwischen den beiden Läufen weg.

**Was du mir danach schicken kannst,** damit ich weiterrechne statt zu raten:
`artifacts/latte.txt`, `artifacts/A_neu.txt`, `artifacts/A_alt.txt`,
`artifacts/interface.txt`. Das sind vier kleine Textdateien.

Erst wenn G3 und G4 grün sind, lohnt sich der GPU-Server — vorher kostet er nur
Geld.

---

## 6. Aufräumen: 4 Benchmark-Einstiege → 1

**Vorschlag, noch nicht umgesetzt.** Die Benchmarks bleiben inhaltlich wie sie
sind; was sich ändert, ist die Zahl der Einstiege.

Was tatsächlich in den drei Benchmark-Dateien steckt:

| Anteil | Zeilen (geschätzt) | Wo es hingehört |
|---|---|---|
| Plot / Report / CSV (`draw_*_boxes`, `write_csv`, `run_report_only`) | ~350 | `bench_common.py` — dreimal derselbe Zweck |
| Resume / Teil-Checkpoints (`save_probe_part`, `merge_probe_parts`, `_probe_signature`) | ~150 | `bench_common.py` — generisch, nur in einer Datei vorhanden |
| **Die eigentliche Sweep-Achse** | **~40 je Benchmark** | bleibt |

Ziel:

```
bench.py --stage balance|arch|weights   # ein Einstieg, drei Gitterdefinitionen
bench_common.py                         # Plot, Report, Resume, Scoring
smallBench.py                           # bleibt: Rauchtest, eigener Zweck
```

Aus 2735 Zeilen in 4 Dateien werden grob 1200 in 2. **Wichtig:** das ist
Umsortieren, kein Neuschreiben — die gemessenen Achsen und die Scoring-Logik
bleiben identisch, sonst sind alte Läufe nicht mehr vergleichbar.

**Erst nach G3.** Ein Refactor der Benchmarks, bevor feststeht, ob die
Umbauten überhaupt geholfen haben, ist die zweite Variable in einem A/B.

## 7. Aufräumen: 9 Doku-Dateien

Nichts löschen — umetikettieren, damit klar ist, was Fahrplan und was Archiv
ist.

| Datei | Rolle |
|---|---|
| **`FAHRPLAN.md`** (diese) | **Einstieg. Hier anfangen.** |
| `ARCHITECTURE.md` | Nachschlagewerk: was das Modell ist, wie der Rollout läuft, offener Befund 4.1 |
| `README_MODEL_CRITIQUE.md` | Entscheidungstabellen für Schritt A |
| `TRAININGS_BERICHT_2026-08-28.md` | Messprotokoll, korrigiert |
| `README_GPU_SERVER.md` | Nur aufschlagen, wenn der GPU-Server dran ist (Phase 5) |
| `README_ERSTER_TEST.md` | Archiv. **Alle Zahlen darin sind synthetisch** |
| `README_LOKALER_LAUF.md` | geht in Abschnitt 5 auf → kann Verweis werden |
| `UEBERGABE_2026-08-27.txt` | Archiv, historischer Stand |

---

## 8. Was nicht zu tun ist

| Nicht | Warum |
|---|---|
| `w_phys` auf 1.0 / 10.0 erhöhen | `L_phys_bal = L_phys/EMA(L_phys)` ist selbstnormiert, `w_phys` kommt darin nicht vor (`train.py:600`) |
| Das 10×10-Gewichtsgitter | ~6–8 Tage, und misst Gewichte, bevor feststeht, was eines bedeutet |
| Gegen 11.96 °C vergleichen | Synthetisch, und es ist der Persistenz- nicht der Mittelwert-Vorhersager |
| „Full Grid 6358 Punkte" | Existiert nicht. 363 ist die native Sensorzahl |
| Am Physik-Term schrauben | Erst G3. Sonst zweite Variable im A/B |
| Nach `PINNmodulusTwoExtProfiles` schielen | Andere Normierung, andere `A`, nichts überträgt sich |

---

## 9. Abbruchkriterien

Ehrlichkeitsklausel — wann der Plan selbst falsch ist:

* **G3 sagt „beide > 20 °C"** → das Problem sind die Daten oder die
  Modellklasse, nicht die Hyperparameter. Dann zurück auf Phase 1 statt
  weiter nach unten.
* **G4 lässt sich nicht auf 0 bringen** → der Rollout ist instabil, nicht
  untertrainiert. Dann `ARCHITECTURE.md` 3.1, nicht mehr Epochen.
* **G5 bleibt nach Phase 5a+5b rot** → das Modell schlägt einen konstanten
  Mittelwert nicht. Dann ist die Frage nicht mehr, welches Gewicht gewinnt,
  sondern ob History-Struktur und Kapazität für diese Aufgabe reichen.

---

**Stand:** 2026-08-28
**Nicht ausgeführt:** in der Entwicklungsumgebung gibt es kein `torch`, kein
`data_cache/` und kein `material_properties/`. Alle Laufzeiten sind aus den
Projektdokumenten übernommen, nicht nachgemessen.
