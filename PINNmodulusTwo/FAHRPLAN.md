# Fahrplan — OP01–OP07, von vorne

**Geltungsbereich:** nur der Basis-Datensatz `OP01–OP07` (Training OP01–05,
Validierung OP06, Test OP07). **Nicht** `PINNmodulusTwoExtProfiles/` — dort
poolt die Normierung über OP01–OP16, das verschiebt `T_sigma`, `dTdt_scale` und
`A`, und nichts aus diesem Plan überträgt sich dorthin.

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du
nur eine Datei liest, dann diese.

---

## 0. Was sich am 31.08. geändert hat

**Die vier Benchmark-Skripte sind gelöscht** — `smallBench.py`,
`benchmark_balance.py`, `benchmark_arch.py`, `benchmark_wphys_wbc.py` und die
gemeinsame `bench_common.py`; in der Profil-Erweiterung ebenso `smokeBench.py`,
`profileBench.py` und `bench_profiles.py`. Zusammen 4735 Zeilen. Sie werden neu
aufgebaut, Schritt für Schritt, wenn feststeht, *was* gemessen werden soll.

Der Grund ist nicht, dass sie inhaltlich falsch waren, sondern dass **kein
einziges ihrer Ergebnisse auf echten Daten gemessen ist**. Jede Zahl in diesem
Repo kommt vom synthetischen Bündel. Ein Sweep, der 6–8 Tage GPU kostet und
eine Rangfolge über Konfigurationen aufstellt, die alle noch nie einen trivialen
Vorhersager geschlagen haben, ist eine Rangfolge zwischen Verlierern. Und vier
Einstiege, deren Reihenfolge sich in den eigenen Docstrings widersprach, waren
selbst ein Grund, warum der eine Lauf, auf den alles wartete (Schritt A), nie
gemacht wurde.

**Was an ihre Stelle getreten ist, ohne Sweep-Maschinerie:** `train.py` kann
jetzt selbst, was vorher nur `bench_common` konnte.

| vorher | jetzt |
|---|---|
| `smallBench.py` druckte „the bar to beat" | `train.py` druckt persistence + Trainings-Mittel neben **jeder** Held-out-MAE |
| `bench_common` baute Val-/Test-OPs über `data.build_op` | `train.py --val-ops / --test-ops` tut dasselbe, mit derselben Normierung |
| `smallBench` warnte vor synthetischen Daten | `train.py` druckt das Banner beim Start (`data.cache_is_synthetic`) |
| `bench_common.EMPTY_HIST` hielt die History-Serien synchron | `train.HISTORY_KEYS` + Assertion in `fit()` |

Ein einzelner `train.py`-Lauf beantwortet damit die Frage, an der alles hängt —
*schlägt das Modell „nichts tun"?* — ohne dass irgendein Benchmark existieren
muss. Genau das ist Phase 2 unten.

---

## 1. Wo wir stehen

| | Stand | Beleg |
|---|---|---|
| Datenpipeline CSV → NPZ → Training | ✅ läuft | Bericht 28.08. §1 |
| `A = 118.9 / 29.7`, `dTdt_scale = 2.479` | ✅ gegen Übergabe verifiziert | `UEBERGABE` Z. 29, 121–122 |
| Training läuft ohne Abbruch | ✅ | Bericht §4 |
| Echte Daten liegen lokal | ✅ **neu** | `data_cache/` auf der Arbeitsmaschine |
| Rollout zahm | ❌ 342 saturierte Schritte auf einem **Trainings**-OP | Bericht §5 |
| Physik-Term | ❌ kollabiert (`L_phys_bal = 2.7e-06`) | Bericht §6 |
| **Schlägt das Modell „nichts tun"?** | ❓ **unbekannt** | kein Maßstab auf echten Daten |
| Held-out-Auswertung im Trainingslauf | ✅ **neu** | `train.py --val-ops OP06 --test-ops OP07` |

Die vorletzte Zeile ist die wichtigste. **Alles andere zu optimieren, bevor sie
beantwortet ist, ist Blindflug.** Sie kostet jetzt einen einzigen Lauf.

---

## 2. Die eine Regel

> **Nichts, was Stunden kostet, bevor das Billige gelaufen ist, das es entwerten
> könnte.**

Jede Phase unten hat ein **Tor**. Ist es rot, geht es nicht weiter — dann wird
das Tor repariert, nicht die nächste Phase gestartet.

---

## 3. Die Phasen

### Phase 0 — Synthetisch (Minuten, keine Daten, kein GPU)

Prüft die Mathematik, nicht das Ergebnis. Läuft überall, auch ohne
`data_cache/`.

```bash
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/tools/rollout_divergence.py
```

**Tor G0:** alles grün.

> **Wozu synthetisch überhaupt gut ist:** für die **Rangfolge** zwischen
> Konfigurationen, nicht für Beträge. `README_ERSTER_TEST.md` Kapitel 9.1 sagt
> das ausdrücklich — „die *Richtung* ist robust, die *Beträge* sind es nicht".
> `train.py` druckt jetzt beim Start ein Banner, wenn der Cache synthetisch ist,
> damit eine solche Zahl nicht versehentlich als Messung zitiert wird.

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

Ein kurzer Lauf, allein wegen der Latte. Die MAE des Modells ist an dieser
Stelle egal:

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 \
        --val-ops OP06 --test-ops OP07
```

Interessant sind nur die Zeilen, die `evaluate()` unter den Trainings-OPs
druckt:

```
  [val ] OP06: MAE=?? C  (beats|LOSES TO the trivial baselines:
                          persistence=?? C, train-mean=?? C)
  [test] OP07: MAE=?? C  (...)
```

`persistence` ist „das Feld ändert sich nie", `train-mean` ist der konstante
Mittelwert der Trainings-Labels (`bundle.T_mu`, per Konstruktion dieselbe Größe).
Beide werden auf **genau dem OP** gerechnet, um das es geht — nicht zitiert.

**Tor G2:** die Latte steht als Zahl fest, gemessen auf dem **echten** OP06 und
OP07. Die synthetischen 11.96 / 6.60 aus `README_ERSTER_TEST.md` sind **nicht**
diese Zahl und dürfen sie nicht ersetzen.

Alles ab hier wird gegen diese eine Zahl gelesen.

### Phase 3 — Schritt A, das A/B (≈ 20 min CPU, Minuten GPU)

Der Lauf, den `README_MODEL_CRITIQUE.md` „den wichtigsten im ganzen Dokument"
nennt und der bis heute nicht gemacht wurde. **Zwei** Läufe, mit `--epochs 10`
und getrennten Ausgabeordnern, weil beide nach `artifacts/` schreiben:

```bash
# neuer Stand
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
    | tee artifacts_A_neu.txt

# der Stand vor den drei Umbauten (Trainingsbudget, Residual-Ausgang, Residuum)
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
    --inner-steps 1 --learn-gains --loss-balance legacy \
    | tee artifacts_A_alt.txt
```

**Tor G3:**

| Ergebnis | Heißt | Dann |
|---|---|---|
| neu deutlich besser | Unterversorgung war der Engpass | Phase 4 |
| neu ≈ Baseline | Hauptdiagnose war falsch | Architektur zuerst |
| neu schlechter | Overfitting durch 100× Updates | `--inner-steps` einzeln variieren |
| beide > 20 °C | lernt grundsätzlich nicht | **Daten prüfen, nicht Hyperparameter** |

Dazu der Drift-Test, der entscheidet, ob ein Sweep überhaupt etwas misst.
`train.py` schreibt `artifacts/pred_OP07.npz` jetzt selbst, weil OP07 als
`--test-ops` mitläuft:

```bash
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP07.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
print('Wachstum', e[-(n//5):].mean() / e[1:n//5].mean())
"
```

Wachstum > 3 → Drift dominiert; das ist dann das Thema, nicht die Gewichte.

### Phase 4 — Rollout zahm bekommen (Stunden)

**Tor G4: `[SATURATED]` muss in der letzten Epoche bei 0 stehen.**

Aktuell 342. `train.py` sagt dazu wörtlich: *„it is not a prediction, and a run
that only survives because of this is not trained."* Eine Rangfolge über
Konfigurationen, deren Rollout wegläuft, rankt Guard-Verhalten, nicht Physik.

Hebel in dieser Reihenfolge: mehr Epochen → `lr` runter → `rollout_clamp`
prüfen → `A` senken über längere `rate_lags`.

Ein zweites Signal dafür steht seit dem 31.08. in `metrics.txt`: `[DIVERGED]`.
Der Eval-Rollout läuft **ohne** Clamp, damit eine weggelaufene Trajektorie nicht
als bloß schlechte MAE erscheint — wird er nicht-endlich, sagt die Zeile ab
welchem Schritt.

### Phase 5 — Erst jetzt wieder messen, und zwar neu gebaut

Die Reihenfolge, in der die neuen Messungen entstehen sollen. **Jede einzeln,
nicht als Suite** — das war der Fehler beim letzten Mal.

**5a — Balance.** Was ein Gewicht überhaupt bedeutet, bevor eines gesucht wird.
`w_phys` multipliziert `L_phys/EMA(L_phys)`, also eine selbstnormierte Größe;
solange der Physik-Term kollabiert (`spread_space`/`spread_time` nahe 0), misst
jeder Gewichts-Sweep den Kollaps und nicht die Physik. Die beiden Diagnosen
dafür stehen schon in der History (`spread_*`, `div_*`) und im `[FLAT]`-Log.

**5b — Architektur.** Breite, Tiefe, `rate_lags`, `delta_grid` — eine Achse nach
der anderen, jeweils über mehrere Seeds. Nur wenn 5a geklärt ist.

**5c — Gewichte.** Zuletzt, klein (ein 3×3-Raster, kein 10×10).

**Tor G5, über allem:** sobald ein Lauf die Latte aus Phase 2 unterbietet, ist
das Modell zum ersten Mal mehr wert als „nichts tun".

---

## 4. Wie die Benchmarks neu aufgebaut werden

Der Punkt der Löschung war, dass beim Neuaufbau **eine** Sache pro Schritt
dazukommt. Reihenfolge:

1. **Erst ein Lauf, den man von Hand liest.** `train.py --val-ops --test-ops`
   kann das heute. Solange die Frage „ist eine Konfiguration besser als die
   andere?" mit zwei Läufen und zwei Zahlen beantwortbar ist, braucht es keine
   Maschinerie.
2. **Dann Seeds.** Der erste echte Bedarf: eine MAE-Differenz zwischen zwei
   Konfigurationen ist wertlos ohne die Streuung über Seeds daneben. Das ist
   eine Schleife über `--seed` plus Mittelwert/Std — mehr nicht, und es ist die
   einzige Ergänzung, die den bisherigen Ergebnissen wirklich gefehlt hat.
3. **Dann eine Achse.** Eine Liste von `fit()`-Overrides, eine CSV-Zeile pro
   Punkt. Kein Plot, kein Resume, kein Checkpoint-Merge.
4. **Plots und Resume ganz zuletzt**, und nur für die Achse, die tatsächlich
   Stunden läuft.

Was aus dem alten Code dabei bleiben soll, ist die *Bewertungslogik*, nicht die
Infrastruktur: Auswahl auf `--val-ops` und Bericht auf `--test-ops`, MAE als
Kriterium und niemals `L_data`, und das Seed-Rausch-Urteil, das sagt, ob eine
Rangfolge überhaupt verteidigt werden kann.

Was **nicht** wiederkommt: dass jedes Skript seine eigene Kopie der Defaults
mitbringt. `config.yaml` ist jetzt die einzige Quelle.

---

## 5. Was du lokal machen musst

Kopiervorlage, in dieser Reihenfolge. Die ersten drei Blöcke brauchen **kein**
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
python3 PINNmodulusTwo/tools/interface_probe.py  | tee interface.txt

# Phase 2 -- Minuten. Nur die [val]/[test]-Zeilen zaehlen.
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 \
        --val-ops OP06 --test-ops OP07        | tee latte.txt

# Phase 3 -- ~20 min CPU, das A/B
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
                                              | tee A_neu.txt
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
        --inner-steps 1 --learn-gains --loss-balance legacy \
                                              | tee A_alt.txt
```

**Was du mir danach schicken kannst,** damit ich weiterrechne statt zu raten:
`latte.txt`, `A_neu.txt`, `A_alt.txt`, `interface.txt`. Vier kleine Textdateien.

Erst wenn G3 und G4 grün sind, lohnt sich der GPU-Server — vorher kostet er nur
Geld.

---

## 6. Doku-Rollen

Nichts gelöscht außer den Benchmarks — umetikettiert, damit klar ist, was
Fahrplan und was Archiv ist.

| Datei | Rolle |
|---|---|
| **`FAHRPLAN.md`** (diese) | **Einstieg. Hier anfangen.** |
| `ARCHITECTURE.md` | Nachschlagewerk: was das Modell ist, wie der Rollout läuft, offener Befund 4.1 |
| `README.md` | Nachschlagewerk: Dateien, Flags, warum die Rekurrenz so aussieht |
| `README_MODEL_CRITIQUE.md` | Entscheidungstabellen für Schritt A |
| `TRAININGS_BERICHT_2026-08-28.md` | Messprotokoll, Archiv |
| `README_GPU_SERVER.md` | Nur aufschlagen, wenn der GPU-Server dran ist. **Die Benchmark-Kapitel 7/8 sind gegenstandslos** |
| `README_ERSTER_TEST.md` | Archiv. **Alle Zahlen darin sind synthetisch** |
| `README_LOKALER_LAUF.md` | Wohin die Daten gehören |
| `UEBERGABE_2026-08-27.txt` | Archiv, historischer Stand |

---

## 7. Was nicht zu tun ist

| Nicht | Warum |
|---|---|
| Die Benchmarks aus der Historie zurückholen | Sie zu haben war nie das Problem — sie ohne Maßstab zu fahren war es. Erst G2 |
| `w_phys` auf 1.0 / 10.0 erhöhen | `L_phys_bal = L_phys/EMA(L_phys)` ist selbstnormiert, `w_phys` kommt darin nicht vor |
| Gegen 11.96 °C vergleichen | Synthetisch, und es ist der Persistenz- nicht der Mittelwert-Vorhersager |
| „Full Grid 6358 Punkte" | Existiert nicht. 363 ist die native Sensorzahl |
| Am Physik-Term schrauben | Erst G3. Sonst zweite Variable im A/B |
| Nach `PINNmodulusTwoExtProfiles` schielen | Andere Normierung, andere `A`, nichts überträgt sich |

---

## 8. Abbruchkriterien

Ehrlichkeitsklausel — wann der Plan selbst falsch ist:

* **G3 sagt „beide > 20 °C"** → das Problem sind die Daten oder die
  Modellklasse, nicht die Hyperparameter. Dann zurück auf Phase 1 statt
  weiter nach unten.
* **G4 lässt sich nicht auf 0 bringen** → der Rollout ist instabil, nicht
  untertrainiert. Dann `ARCHITECTURE.md` 3.1, nicht mehr Epochen.
* **G5 bleibt rot, auch nach 5a und 5b** → das Modell schlägt einen konstanten
  Mittelwert nicht. Dann ist die Frage nicht mehr, welches Gewicht gewinnt,
  sondern ob History-Struktur und Kapazität für diese Aufgabe reichen.

---

**Stand:** 2026-08-31
**Ausgeführt:** Testsuite und ein Ende-zu-Ende-`train.py`-Lauf gegen das
synthetische Bündel (Banner, Held-out-Auswertung, Baselines). **Nicht
ausgeführt:** alles auf echten Daten — `data_cache/` und `material_properties/`
liegen nur auf der Arbeitsmaschine. Alle GPU-Laufzeiten sind aus den
Projektdokumenten übernommen, nicht nachgemessen.
