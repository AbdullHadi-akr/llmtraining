# Review des Trainingsberichts vom 2026-08-28

Gegenstand: [`TRAININGS_BERICHT_2026-08-28.md`](TRAININGS_BERICHT_2026-08-28.md)

**Kurzfassung:** Die Läufe selbst und die Datenvalidierung sind in Ordnung. Die
*Bewertung* der Läufe ist es überwiegend nicht. Der Bericht nennt den falschen
Grund für das `FAIL`, vergleicht gegen eine Baseline, die für diesen Vergleich
laut Projektdoku nicht zugelassen ist, begründet den Physik-Term mit einer
Größe, die von `w_phys` gar nicht abhängt, und leitet daraus Nächste-Schritte
ab, die dem widersprechen, was `smallBench.py` selbst als nächsten Schritt
ausgibt.

**Geltungsbereich:** Diese Review ist rein statisch. In dieser Umgebung gibt es
kein `torch`, kein `data_cache/` und kein `artifacts/` — es wurde nichts
nachgerechnet. Geprüft wurde ausschließlich, ob die *Interpretationen* des
Berichts zu Code, `config.yaml` und den Projektdokumenten passen. Die gemessenen
Rohzahlen (L_data, MAE, Saturation-Counts) sind als gegeben übernommen.

---

## 1. Was der Bericht richtig hat

Das sollte nicht untergehen, denn es ist der eigentliche Fortschritt:

* **Die Datenvalidierung ist echt und stimmt.** `A = 118.9 / 29.7` und
  `dTdt_scale = 2.479` decken sich exakt mit `UEBERGABE_2026-08-27.txt`
  (Zeilen 29, 121–122). Das ist genau die Prüfung, die
  `README_LOKALER_LAUF.md` als Punkt 1 von Schritt 2 verlangt, und sie ist
  bestanden.
* **Die Konfigurationsangaben stimmen** mit `config.yaml`: Breite 128, Tiefe 4,
  `subsample_time: 2`, `rate_lags: [5.0, 20.0]`, `ema_decay: 0.9`, CFL
  `0.2 s < 0.241 s`.
* **Der fallende Saturation-Count ist die richtige Größe zum Hinsehen** —
  `README_LOKALER_LAUF.md:171` nennt genau dieses Kriterium.
* **Die Symptome sind gesehen:** dass die MAE schlecht ist und dass mit
  `L_phys_bal` etwas nicht stimmt, steht korrekt im Bericht. Falsch ist erst,
  was daraus geschlossen wird.

---

## 2. Befunde

### B1 — Der Grund für das `FAIL` ist falsch (kritisch)

Der Bericht schreibt zweimal „Status: ❌ FAIL (MAE zu hoch)".

`smallBench.py:269` sagt:

```python
mae_ok = test_mae < 20.0
```

13.48 °C und 12.02 °C liegen **beide unter 20** — die MAE-Prüfung ist in beiden
Läufen **bestanden**. Durchgefallen ist Check 3 (`smallBench.py:262-263`):

```python
balanced = ((not phys_on or 0.01 < L_phys_bal < 100)
            and (not bc_on or 0.01 < L_bc_bal < 100))
```

Mit `L_phys_bal = 2.69e-06` fällt der `w_phys=0.1`-Lauf hier durch. Für
`w_phys=0.0` wird der Physik-Term übersprungen (`zero_weight_terms: skip`,
daher das `nan`), dort kann das `FAIL` **nur** von `L_bc_bal` kommen — und
`L_bc_bal` kommt im ganzen Bericht nicht vor.

**Konsequenz:** Beide Läufe sind an der Loss-Balance gescheitert, nicht an der
Genauigkeit. Der Bericht nennt für beide die falsche Ursache und lässt die
Variable weg, die eine davon tatsächlich verursacht hat.

### B2 — Der Baseline-Vergleich in §9 ist ungültig (kritisch)

Drei voneinander unabhängige Probleme, jedes für sich ausreichend:

**(a) Falsches Etikett.** Der Bericht nennt 11.96 °C den „naiven Predictor
(predict mean)". `README_ERSTER_TEST.md:387-389` führt zwei triviale
Vorhersager:

| Vorhersager | MAE test |
|---|---|
| „Temperatur ändert sich nie", `T(t) = T(0)` | **11.96 °C** |
| „konstanter Mittelwert der Trainingslabels" | **6.60 °C** |

11.96 ist die *Persistenz*, nicht der Mittelwert. Der Mittelwert-Vorhersager
liegt bei 6.60 °C, und die Doku verlangt ausdrücklich den Vergleich gegen „das
bessere der beiden trivialen Vorhersager". Gegen 6.60 °C ist 12.02 °C rund
**1,8× schlechter**.

**(b) Vorzeichenfehler.** Selbst gegen die falsch gewählten 11.96 °C gilt
12.02 > 11.96. Das ist 0,5 % **schlechter**. Der Bericht schreibt „leicht besser
als Baseline ✅ (+0.5%)".

**(c) Der Vergleich ist grundsätzlich nicht zulässig.** `README_ERSTER_TEST.md`
setzt über Kapitel 6 einen Kasten:

> **Alle Zahlen in diesem Kapitel stammen von einem synthetischen Bundle** […]
> Sie sind **keine** Vorhersage der MAE auf den echten OPs.

Kapitel 9.1 wiederholt es: „Die *Richtung* ist robust […]; die *Beträge* sind es
nicht." Die Baselines 11.96 und 6.60 sind selbst synthetisch. Eine auf echten
Daten gemessene MAE gegen sie zu halten, ist genau die Übertragung, die die Doku
verbietet.

**Das ist der wichtigste Punkt der ganzen Review:** Nach diesem Lauf ist
weiterhin **unbekannt**, ob das Modell auf echten Daten einen trivialen
Vorhersager schlägt — weil die trivialen Vorhersager auf den echten OPs nie
gerechnet wurden. Weder das ✅ noch das ❌ in §9 hat eine Grundlage.

### B3 — Die `L_phys_bal`-Diagnose ist mechanisch falsch (kritisch)

Der Bericht: „`w_phys=0.1` ist zu klein → Physik wird wegbalanciert", und daraus
die Empfehlung `w_phys = 1.0` oder `10.0`.

`train.py:600` in Verbindung mit `_LossBalancer.divisor` (`train.py:276-285`):

```python
L_phys_bal = L_phys / balance.divisor("phys", float(L_phys.detach()))
```

Der Divisor ist die laufende EMA von `L_phys` selbst. `L_phys_bal` ist also
`L_phys / EMA(L_phys)` — ein **selbstnormierter Quotient, in dem `w_phys` gar
nicht vorkommt.** `w_phys` skaliert erst danach den Gradientenbeitrag
(`train.py:613`). `w_phys` zu erhöhen kann `L_phys_bal` deshalb nicht in
Richtung O(1) bewegen. Die vorgeschlagene Maßnahme kann das genannte Problem
prinzipiell nicht beheben.

Was `2.69e-06` tatsächlich heißt: `L_phys` ist gegenüber seinem eigenen
laufenden Mittel um rund sechs Größenordnungen eingebrochen. Das ist ein
**kollabierender** Physik-Term — einer, der trivial erfüllt wird — nicht einer,
der zu schwach gewichtet ist. `README_MODEL_CRITIQUE.md:194-196` beschreibt
genau diesen degenerierten Fall als bekanntes Risiko. Mehr `w_phys` verstärkt
den Druck, der dorthin führt.

Nebenbei: der Bericht liest die Zahl als statisch, sie ist es nicht.
3.52e-08 (Epoche 5) → 2.69e-06 (Epoche 10) ist ein Anstieg um zwei
Größenordnungen. Das ist eine Zeitreihe mit Information darin.

### B4 — „Overfitting" ist die falsche Diagnose (hoch)

Der Bericht: „stark overfitted auf die Trainingsdaten", wegen Test − Train
= 12.02 − 7.65 = 4.37 °C.

Der Gap stimmt, die Deutung nicht. **Train MAE = 7.65 °C ist selbst schlecht.**
Ein Modell, das die eigenen Trainingsdaten nicht besser als auf 7,65 °C trifft,
ist unterangepasst. Overfitting setzt voraus, dass die Trainingsleistung *gut*
ist — sonst ist der Gap eine Verteilungsdifferenz zwischen OP01–05 und OP07 oder
schlicht ein Modell, das noch nichts kann. Die Unterscheidung ist nicht
akademisch: sie führt zu entgegengesetzten Maßnahmen (mehr Regularisierung vs.
mehr Kapazität/Budget).

### B5 — Die 6358 Gitterpunkte gibt es nicht (hoch)

„Nur 363 Gitterpunkte: Reduziertes Grid (Original hatte 6358)" und
„Empfohlene Next Action 4: Full Grid: 6358 statt 363 Punkte".

`6358` kommt im gesamten Repository **ausschließlich im Bericht selbst** vor.
363 ist die native Sensorzahl, durchgehend so bezeichnet — `README_GPU_SERVER.md`
Zeilen 406 („363 Sensoren"), 724 („einen Absolutfehler pro Sensor, also 363
Werte"), 869. Es gibt kein reduziertes Gitter und nichts zum Hochskalieren.
Auch der Zusatz „wahrscheinlich um CPU-Training zu beschleunigen" ist frei
ergänzt. Empfehlung 4 ist ersatzlos zu streichen.

### B6 — `T_sigma = 8.66 °C` ist unbelegt (mittel)

Kommt sonst nirgends im Repo vor. `README_ERSTER_TEST.md:699` rechnet mit
`T_sigma = 5 K`.

### B7 — Saturation wird als Erfolg verbucht (hoch)

Der Bericht: „✅ Stabilität: Keine NaN-Abstürze in Epoche 1 (wie früher)" und
„Epoche 10: Nur noch 342/7279 Steps saturiert bei OP03".

`train.py:674-679` gibt zu genau diesem Zähler aus:

> „the trajectory ran away and was held back — **it is not a prediction, and a
> run that only survives because of this is not trained.**"

`model.py:738`: „a clamp in the tens never binds on a model that is working."

Dass der Count *fällt*, ist richtig gelesen und ein gutes Zeichen. Dass er in
der letzten Epoche noch bei 342 liegt, heißt: der freilaufende Rollout läuft auf
einem **Trainings**-OP weiterhin weg und wird nur vom Guard gehalten. Das ist
kein ✅, und das „keine NaN-Abstürze mehr" ist wörtlich der Fall, vor dem der
Code warnt.

### B8 — Die Nächsten Schritte widersprechen dem Repo (hoch)

Der Bericht empfiehlt `benchmark_arch.py` (Lag-Sweep),
`benchmark_wphys_wbc.py` (Gewichts-Sweep) und `w_phys = 1.0/10.0`.

`smallBench.py:319-325` gibt bei Erfolg selbst den nächsten Schritt aus:

```python
print("  Next: python3 PINNmodulusTwo/benchmark_balance.py --part 1 ...")
# NOT the 10x10 grid: that is 100 trainings (~6-8 days) and it would
# sweep weights before anything has established what a weight means
# here. The balancing benchmark is ~4 h and settles that first.
```

Der Bericht empfiehlt exakt das, wovon dieser Kommentar abrät.
**`benchmark_balance.py` wird im Bericht nirgends erwähnt** — obwohl die Datei
im Repo liegt und obwohl das gescheiterte Kriterium (B1) genau die Balance ist,
die dieses Benchmark klärt.

Dazu kommt das Gate: `README_LOKALER_LAUF.md:193` — „Schritt 3 […] Erst wenn
Schritt 2 sauber durchläuft." Schritt 2 ist zweimal mit `FAIL` beendet worden.
Das Fazit des Berichts stellt es dennoch als gleichwertige Option hin („oder
[…] zu Schritt 3 übergehen").

### B9 — Schritt A ist nicht abgeschlossen (hoch)

`README_MODEL_CRITIQUE.md:159-186` nennt `smallBench.py` „**der wichtigste Lauf
im ganzen Dokument**" und definiert ihn als A/B mit **zwei** Läufen:

```bash
python3 PINNmodulusTwo/smallBench.py                       # neu
python3 PINNmodulusTwo/smallBench.py \                     # alter Stand
    --inner-steps 1 --no-residual-output --learn-gains --loss-balance legacy
```

Verglichen wird die Test-MAE beider. Der Bericht hat stattdessen einen
`w_phys`-Sweep `[0.0, 0.1]` gefahren. Das ist ein anderer Vergleich und
beantwortet die Frage von Schritt A nicht — ob nämlich die Umbauten (Budget,
Residual-Output, Physik-Residuum) überhaupt etwas gebracht haben. Solange das
offen ist, ist die Entscheidungstabelle in `README_MODEL_CRITIQUE.md:180-186`
nicht anwendbar, und alle „Nächste Schritte" des Berichts hängen in der Luft.

Ebenfalls nicht gelaufen: der **Drift-Test** auf `artifacts/pred_OP07.npz`
(`README_MODEL_CRITIQUE.md:206-223`). Der entscheidet, ob ein Gewichts-Sweep
überhaupt etwas misst — Zitat: „Ein Gewichte-Sweep bei starker Drift misst
hauptsächlich, welches Gewicht die Drift zufällig am wenigsten verstärkt — das
ist die teure Art, nichts zu lernen."

### B10 — Kleinigkeiten

* „`batch_size: 2048`" ist nur `batch_data`. `config.yaml:171-173` hat zusätzlich
  `batch_phys: 256` und `batch_bc: 128`.
* „Validation: OP06" ist in `smallBench.py` kein Konzept — einen `--val-op` gibt
  es nur in `benchmark_arch.py:119`. In diesem Lauf existierte kein
  Validierungssplit.
* Die `SATURATED`-Zeile steht in der Metrik-Tabelle neben `L_data` ohne Hinweis,
  dass „>6900 steps" **Rollout-Zeitschritte** zählt, nicht die 500
  Optimierer-Schritte pro Epoche aus derselben Berichtssektion. Zwei
  verschiedene „steps" in einer Tabelle.

---

## 3. Was tatsächlich als Nächstes zu tun ist

In dieser Reihenfolge. Punkt 1 ist billig und macht alle anderen Zahlen erst
lesbar.

1. **Die zwei trivialen Vorhersager auf dem echten OP07 rechnen** — Persistenz
   `T(t) = T(0)` und konstanter Mittelwert der Trainingslabels. Kein Training,
   Minuten. Erst damit bekommt „12.02 °C" eine Bedeutung; die synthetischen
   11.96/6.60 dürfen dafür nicht verwendet werden (B2c). Ohne diesen Schritt
   ist jede weitere Optimierung ein Blindflug.
2. **`L_bc_bal` aus `artifacts/metrics.txt` nachtragen.** Es ist die
   wahrscheinliche Fehlerursache des `w_phys=0.0`-Laufs und fehlt im Bericht
   vollständig (B1).
3. **Schritt A wirklich fahren** — den Baseline-Lauf mit
   `--inner-steps 1 --no-residual-output --learn-gains --loss-balance legacy`
   gegen den Default (B9). Vorher ist keine Aussage über die Umbauten möglich.
4. **Drift-Test auf `pred_OP07.npz`** (B9). Entscheidet, ob Sweeps überhaupt
   sinnvoll sind.
5. **Dann `benchmark_balance.py --part 1`** — nicht `benchmark_arch.py` und
   nicht `benchmark_wphys_wbc.py` (B8). Das gescheiterte Kriterium ist die
   Balance; das ist das Benchmark dafür.
6. **`w_phys` nicht als Mittel gegen `L_phys_bal` erhöhen** (B3). Wenn der
   Physik-Term kollabiert, ist der Hebel die Skalierung/Normierung des
   Residuums, nicht sein Gewicht.

Was aus den Empfehlungen des Berichts bestehen bleibt: **längeres Training** und
**GPU** sind plausibel (bei Train-MAE 7.65 °C ist Unteranpassung ein
realistischer Kandidat, B4). **Full Grid** entfällt ersatzlos (B5).

---

## 4. Einordnung

Der Bericht ist gut strukturiert und beschreibt korrekt, *was* gelaufen ist.
Sein Problem ist durchgehend dasselbe: Zahlen werden gegen Erwartungen gehalten,
ohne die Quelle der Erwartung zu prüfen. 11.96 wird zum Mittelwert-Vorhersager,
weil es plausibel klingt; das Gitter wird reduziert, weil 363 klein aussieht;
`w_phys` wird zu klein, weil `L_phys_bal` klein ist. Jedes Mal steht die
tatsächliche Antwort im Repo — in `README_ERSTER_TEST.md:387`, in
`README_GPU_SERVER.md:724`, in `train.py:600`.

Der eigentliche Fortschritt des Tages steht in §1 und ist echt: die Daten sind
generiert und ihre Kennzahlen gegen die Übergabe verifiziert. Die Läufe sind
gescheitert, aber sie sind an einem klar benannten Kriterium gescheitert, und
das Repo enthält das Werkzeug für genau dieses Kriterium.

---

**Erstellt:** 2026-08-28
**Grundlage:** statische Prüfung von `smallBench.py`, `train.py`, `model.py`,
`config.yaml`, `README_ERSTER_TEST.md`, `README_MODEL_CRITIQUE.md`,
`README_LOKALER_LAUF.md`, `README_GPU_SERVER.md`, `UEBERGABE_2026-08-27.txt`
**Nicht ausgeführt:** kein `torch`, kein `data_cache/`, kein `artifacts/` in
dieser Umgebung — es wurde nichts nachgerechnet
