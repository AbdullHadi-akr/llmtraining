# Fahrplan — OP01–OP16 trainiert, OP19 als Messvergleich

> ## ▶ Das Nächste: trägt der Physik-Term überhaupt?
>
> **Ab hier wieder Inhalt.** Die GPU-Seite ist abgeschlossen — `sweep.py` steht,
> die Karte ist vermessen, die Zahlen stehen in „Was am 09.09./10.09.
> dazugekommen ist". Nichts davon muss noch angefasst werden, um weiterzukommen.
>
> ### Der Lauf
>
> ```bash
> cd /home/student1/llmtraining
> git checkout main && git pull
> source modulus_env/bin/activate
>
> python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
>     --vary w-phys 0.1 0 -j 3 \
>     --out artifacts/achse0 --csv artifacts/achse0.csv \
>     -- --epochs 60 --w-bc 0
> ```
>
> **Über Nacht laufen lassen.** Sechs Läufe zu je ~1.9 h — das ist keine
> Schätzung mehr, sondern die gemessenen 116 s je Epoche bei elf OPs. Bei
> `ceil(6/3) = 2` Wellen und dem gemessenen Parallelitätsfaktor grob **8 h**,
> seriell wären es 12 h.
>
> **Das ist der Vergleichslauf zu Schritt 6**, mit genau einer Frage: trägt der
> Physik-Term, wenn beide Seiten auskonvergiert sind?
>
> **Warum das vor allem anderen kommt:** die Überschrift der letzten Sitzung
> lautet „der Physik-Term trägt". Diese Aussage stammt aus **drei** Epochen
> (Schritt 5b) — und **derselbe 5b-Lauf sagte auch, der `spread` kollabiere.
> Das hat Schritt 6 widerlegt** (§11.3). Bei 60 Epochen ist nur der Lauf **mit**
> Physik gemacht worden.
>
> | Ergebnis | heißt | dann |
> |---|---|---|
> | val-MAE deutlich schlechter als 6.270 / 3.585 C (etwa 11 / 8 C wie in 5b) | **der Physik-Term trägt.** Bestätigt bei Konvergenz | weiter mit den Sweep-Achsen δ → `w_bc` → `w_phys` |
> | val-MAE nahe 6.270 / 3.585 C | **er trägt nicht.** Die 5b-Differenz war Untertrainiertheit, wie schon beim `spread` | `w_phys` zurück auf 0 prüfen, Sweep-Achsen neu ordnen — O8 und O12 wären dann Nebensache |
>
> ### Warum als Sweep und nicht als ein Lauf
>
> Der Plan sagte „ein Lauf, kein Sweep". **Das war richtig, solange es kein
> `sweep.py` gab.** Jetzt widerspricht es dem eigenen Kernsatz dieser Datei: die
> Entscheidungsregel oben sagt „*nahe* 6.270" gegen „*deutlich* schlechter" — und
> „nahe" ist ohne Streuung nicht definiert. Genau daran ist §11.3 schon einmal
> gescheitert.
>
> Zwei Dinge fallen dabei gleichzeitig ab:
>
> 1. **Achse 0** — trägt der Physik-Term.
> 2. **Die Seed-Streuung, einmal gemessen.** Das ist der eigentliche Gewinn: ab
>    dann ist jede spätere Achse lesbar, statt geraten. Ohne sie ist „6.27 gegen
>    6.51" kein Ergebnis.
>
> `sweep.py` sagt am Ende selbst `[NOT SEPARATED]`, wenn der Abstand zwischen
> zwei Konfigurationen unter der Seed-Streuung liegt.
>
> **`--w-bc 0` steht bewusst in beiden Armen.** Damit ist es der Vergleich, den
> Achse 0 meint: Physik **und** BC gegen keins von beidem. Das 2×2-Gitter, das
> `w_bc` mit isolieren würde (Achse 2), wären zwölf Läufe und ~16 h — und die
> ursprüngliche Begründung für Achse 2 ist am 02.09. ohnehin weggefallen
> (§11.6). Erst nach Achse 0 entscheiden, ob sie noch jemanden interessiert.
>
> Wie gut Schritt 6 wirklich war — in-sample gegen ausgehalten, und warum der
> Volumenstrom die eigentliche Schwierigkeitsachse ist: **§11.5**.
---

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du nur
eine Datei liest, dann diese.

Sie ist von oben nach unten sortiert: **was zu tun ist, steht oben; was erledigt
ist, unten.**

| Teil | was drinsteht |
|---|---|
| **I — ZU TUN** | das nächste Kommando, die Signale darin, die offenen Punkte, was danach gebaut wird |
| **II — NACHSCHLAGEWERK** | Datensatz, Split, Dateien, Auswahl- und Abbruchregeln. Ändert sich selten |
| **III — ERLEDIGT** | geschlossene Punkte, abgehakte Schritte, gemessene Zahlen, Befunde. Archiv |

> ### Lebendes Dokument
>
> Diese Datei wird nach **jedem** Ergebnis fortgeschrieben. Die Regeln:
>
> * **Haken setzen, sobald ein Schritt durch ist**, und die gemessene Zahl in die
>   Tabelle „Stand" in Teil III eintragen. Ein Haken ohne Zahl ist wertlos:
>   „gelaufen" und „das Kriterium erfüllt" sind zwei Dinge.
> * **Gemessene Zahlen ersetzen Vermutungen im Text**, sie ergänzen sie nicht.
> * **Ein rotes Tor ändert den Plan, nicht nur den Haken.**
> * **Was erledigt ist, wandert nach Teil III.** Teil I enthält nur, was offen
>   ist — sonst sucht man das Nächste zwischen dem Erledigten.

---

# TEIL I — ZU TUN

# `sweep.py` — die Seed-Schleife

**Schritt 6 ist gelaufen und grün** (Zahlen in Teil III). Ab hier werden
Konfigurationen verglichen, und dafür fehlt das Werkzeug.

## Gebaut am 09.09. — `sweep.py` steht, gemessen ist noch nichts

```bash
python3 PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary delta-phys 1.0 0.4 0.2 -j 4 -- --epochs 20

  -> artifacts/sweep/<punkt>/  je Lauf ein eigener Ordner: history.csv,
                               op_metrics.csv, metrics.txt, model.pt, train.log
  -> artifacts/sweep.csv       eine Zeile je (Konfiguration, Seed)
  -> stdout                    Mittel und Std je Konfiguration über die val-OPs
```

Alles nach einem nackten `--` geht unverändert an `train.py` — der Sweep muss
dessen Flags nicht kennen und kann nicht aus dem Takt geraten. `--vary` ist
wiederholbar und ergibt dann ein Gitter.

Kein Plot, kein Resume, kein Checkpoint-Merge — die kommen zuletzt und nur für
die Achse, die wirklich Stunden läuft.

**Zwei Abweichungen vom ursprünglichen Entwurf**, beide bewusst:

* Es ruft **nicht** `train.fit()` in einer Schleife, sondern startet je Punkt
  einen eigenen `train.py`-Prozess. Das ist der Grund, warum `-j` überhaupt
  möglich ist (unten), und es kauft nebenbei Isolation: ein divergierter Punkt
  wird als `[FAIL]` verbucht, der Rest des Sweeps läuft weiter. Elf fertige
  Punkte an den zwölften zu verlieren wäre der teure Fehler.
* `train.py` hat dafür `--artifacts-dir` bekommen. Ohne das schrieben alle
  Läufe dieselbe `artifacts/model.pt` — die Kollision, vor der §Teil III
  ohnehin warnt.

Die Zeile, für die das Werkzeug existiert, steht am Ende der Ausgabe:

```
  [NOT SEPARATED] 'delta-phys=0.2' führt vor 'delta-phys=0.4' um 0.213 C,
                  aber die Seed-Streuung ist 0.121 C. Das ist keine Differenz.
```

> **Gelaufen, aber nur auf der Fixture.** Am 09.09. gegen den echten `train.py`
> auf dem synthetischen Cache: 4 Läufe (2 Punkte × 2 Seeds), `-j 4` auf 4 Kernen,
> 17.2 s Wanduhr gegen 66 s Summe der Läufe. `[NOT SEPARATED]` hat
> korrekt ausgelöst (0.624 C Abstand gegen 3.859 C Seed-Streuung bei zwei Seeds),
> und ein `[FAIL]`-Punkt bricht den Sweep nicht ab.
>
> **Daraus keinen Faktor ableiten.** Summe/Wanduhr ist eine Obergrenze, nicht
> eine Messung, und es war CPU auf vier Kernen statt der T4. Der Lauf zeigt nur,
> dass der Mechanismus greift. Die echte Zahl von der Instanz — **1.46×** —
> steht weiter unten. Absolute MAE aus der Fixture sagen ohnehin nichts.

## Warum zuerst das und nicht die nächste Achse

Eine MAE-Differenz zwischen zwei Läufen ist **nicht lesbar**, solange die
Streuung über Seeds daneben fehlt. 6.27 gegen 6.51 ist kein Ergebnis, wenn
derselbe Lauf mit anderem Seed zwischen 5.9 und 6.8 schwankt. Genau das hat allen
bisherigen Ergebnissen dieses Projekts gefehlt.

Schritt 6 hat dafür gerade das Lehrstück geliefert: nach drei Epochen stand
`spread` bei 0.201 und sah aus wie ein Artefakt; nach sechzig steht er bei 0.968.
**Drei Epochen sind kein Trend, und ein Seed ist keine Streuung.**

## Die Achsen danach, in dieser Reihenfolge

| # | Achse | warum in dieser Reihenfolge |
|---|---|---|
| **0** | **`--w-phys 0 --w-bc 0` über 60 Epochen** — ein Lauf, kein Sweep | **Die Überschrift dieser Sitzung steht auf drei Epochen.** „Der Physik-Term trägt" kommt aus 5b, und derselbe 5b-Lauf sagte auch „der `spread` kollabiert" — das war falsch (§11.3). Bis das A/B bei 60 Epochen wiederholt ist, ist die Kernaussage ungeprüft. Zwei Stunden, und sie entscheidet, ob die Achsen 1–3 überhaupt die richtigen sind |
| 1 | **δ (`--delta-phys`)**, O8 | sauber isoliert (speist im Hybrid-Modus nur `L_phys`), und der `[CFL WARN]` steht bei jedem Lauf: 1.0 s gegen Δt_max 0.24 s. **Drei Punkte: 1.0 (Default), 0.4, 0.2.** 0.2 ist das Datengitter und zugleich der Boden — darunter interpoliert `history_at` nur noch linear zwischen zwei Gitterzeilen, die Ableitung wird davon nicht genauer. **Der einzige bekannt falsche Parameter** |
| 2 | **`w_bc`**, ehemals O12 | **02.09.: die ursprüngliche Begründung ist weggefallen** — `ratio_bc` zeigt über 30 Epochen keinen Abfall (Median 0.0581, die 0.0178 waren eine Epoche), und `--batch-bc 121` rechnet dieselbe Zeile wie 128. Die Achse bleibt als bewusste Gegenprobe stehen, nicht mehr als Verdacht. Kann gestrichen werden, wenn Rechenzeit knapp wird. §11.6 |
| 3 | **`w_phys`**, O6 | erst wenn 1 und 2 stehen — sonst misst man δ mit dem Gewicht. **Und erst nach O15:** solange die Divisoren eingefroren sind, misst ein `w_phys`-Sweep den Anker des ersten Optimiererschritts mit |

> **Achse 0 braucht `sweep.py` nicht** und kann sofort laufen:
>
> ```bash
> python3 PINNmodulusTwo/train.py --epochs 60 --w-phys 0 --w-bc 0 2>&1 | tee 06b_ohne_physik.txt
> ```
>
> Kommt dort eine val-MAE nahe 6.270 / 3.585 C heraus, war der Physik-Term nie
> der Grund — und O8, O12 und der halbe Sweep zielen daneben. Kommt sie bei
> ~11.6 / 8.5 C heraus wie in 5b, ist die Aussage bestätigt und der Rest steht
> auf festem Grund.

## Was am 02.09. dazugekommen ist

Zwei Werkzeuge und ein Test, alle klein, alle sofort nutzbar:

```bash
# Nach JEDEM Lauf, statt die letzte Zeile abzulesen:
python3 PINNmodulusTwo/tools/analyse_history.py PINNmodulusTwo/artifacts/history.csv
```

| was | wo | wofür |
|---|---|---|
| `tools/analyse_history.py` | neu | Median und Streuung über die letzten *k* Epochen statt der letzten Zeile. Meldet von sich aus: `[LAST-LINE]` (der Fehler, aus dem O12 entstand), `[O15]` (Divisor im freien Fall), `[SPREAD]`, `[EXCURSION]`. Nur Standardbibliothek — läuft auch ohne die Trainingsumgebung |
| `late_bias`, `late_bias_frac`, `bias_early/mid/end` | `op_metrics.py` | der **signierte** Fehler. `late_bias_frac` nahe 1 heißt Drift, nahe 0 heißt Streuung — **das ist die Messung, die O13 entscheidet.** Steht ab dem nächsten Lauf in jeder OP-Zeile und in `metrics.txt` |
| `test_the_divisor_tracks_its_loss_within_one_run` | `tests/` | O15 als `xfail` festgehalten. **Schlägt der Test eines Tages fehl, weil er besteht**, ist O15 behoben — dann Marker entfernen und den Punkt schließen |

Noch nicht gebaut: `evaluate.py` (Checkpoint zurückladen). `sweep.py` ist am
09.09. dazugekommen, siehe ganz oben.

## Was am 09.09. dazugekommen ist — die Karte wird benutzt

Ausgangspunkt war eine Beobachtung auf der g4dn.2xlarge: **ein Lauf belegt
0.4 GB von 15 GB VRAM und lässt die GPU praktisch leer laufen.** Der Grund
steckt in der Form der Last, nicht in einer Einstellung:

* ~7000 **sequentielle** Rollout-Schritte je OP und Epoche, jeder ein
  363×128-Matmul. Das sind ~50 CUDA-Kernel zu ~5 µs Startlatenz gegen ~1.5 µs
  Rechenzeit — die Karte wartet darauf, dass Python ihr das nächste winzige
  Matmul reicht.
* Dazu kamen **~10 `cudaStreamSynchronize` je Optimiererschritt**: jedes
  `float(L_data.detach())` und das `if not torch.isfinite(loss)` hielten die CPU
  an, bis die GPU leergelaufen war. Bei elf OPs × 100 `inner_steps` sind das
  ~11.000 volle Pipeline-Stalls je Epoche, platziert direkt nach dem
  Vorwärtspass. CPU und GPU überlappten damit nie.

**Größere Batches sind nicht die Antwort.** `batch_data`, `batch_phys` und
`batch_bc` sind Experiment-Regler; sie hochzudrehen, um Speicher zu füllen,
ändert das Gradientenrauschen — genau das, was ein Sweep festhalten muss. Es
gibt in diesem Modell auch keinen Tensor, der 15 GB füllen könnte: das Netz hat
~70k Parameter, der Rollout-Puffer ist `7000 × 363 × 4 B` ≈ 10 MB.

Was die Karte füllt, ist **mehr unabhängige Arbeit gleichzeitig** — und genau
daraus besteht ein Sweep.

| was | wo | wofür |
|---|---|---|
| Syncs raus aus der inneren Schleife | `train.py` | `_LossBalancer` hält EMA, Divisoren und Akkumulatoren jetzt als 0-dim-Tensoren **auf dem Gerät**; die Finitheitsprüfung steht nach `backward()` statt davor. Ein Sync je Schritt statt zehn, und er liegt am *Ende* des Schritts, wenn Vor- und Rückwärtspass schon in der Warteschlange stehen |
| `bc_index` einmal statt je Schritt | `train.py`, `physics.py` | `boundary_condition_loss` machte je Schritt ein `torch.where(bc_mask)` und fragte dessen Länge ab — eine datenabhängige Form, also ein weiterer Sync. Die Maske ist über den ganzen Lauf konstant |
| `--artifacts-dir` | `train.py` | ein Ordner je Lauf. Ohne das kein paralleler Sweep |
| `op_metrics.csv` | `train.py` | dieselben Zahlen wie die Tabelle in `metrics.txt`, nur maschinenlesbar. `sweep.py` liest daraus die val-MAE |
| `sweep.py -j N` | neu | N Läufe gleichzeitig, je als eigener Prozess |

**Die Zahlen ändern sich dadurch nicht — nachgemessen, nicht behauptet.**
Derselbe Lauf (Seed 0, 3 Epochen, OP01/OP02, val OP06, echtes Modulus) auf dem
Stand vor und nach dem Umbau:

| verglichen | Ergebnis |
|---|---|
| `history.csv`, inkl. aller `div_*` auf 10 Stellen | **byte-identisch** |
| `metrics.txt` | **byte-identisch** |
| `pred_OP01/02/06.npz`, alle Arrays | **bitweise gleich** |
| 25 Gewichtstensoren im Checkpoint | **bitweise gleich** |
| Testsuite | 130 passed, 1 skipped, 1 xfailed — der O15-`xfail` bleibt `xfail` |

Dazu ist der branchlose EMA-Update separat gegen die alte Python-float-Fassung
durchgerechnet (NaN zuerst, NaN später, inf, 0, negativ, 400 Zufallsfolgen):
null Abweichung, auch im EMA-Zustand selbst. Die Akkumulatoren sind float64 wie
vorher.

> **Was nicht gilt: „bit-identisch mit einem Einzellauf".** Das stand hier
> zuerst und ist falsch. `sweep.py` setzt seinen Kindern `OMP_NUM_THREADS=1`,
> und **die Thread-Zahl verschiebt die letzten Stellen** — mehrfädige
> CPU-Reduktionen legen ihre Summationsreihenfolge nicht fest. Gegen einen
> Einzellauf mit der Voreinstellung wich es ab der 8. Stelle ab; mit gleicher
> Thread-Zahl auf beiden Seiten stimmte alles bitweise. Innerhalb eines Sweeps
> ist das folgenlos, weil alle Punkte dieselbe Umgebung bekommen und nur ihr
> Vergleich untereinander zählt. `--threads` setzt es, wenn eine von Hand
> erzeugte Zahl getroffen werden muss.

### Was das bringt — und was nicht

| | heute, 1 Lauf | mit `-j 4…6` |
|---|---|---|
| `nvidia-smi` GPU-Util | ~5–15 % | 60–100 % |
| VRAM | ~0.9 GB | ~4–6 GB |
| Durchsatz (Läufe/h) | 1× | **1.46×** (gemessen, siehe unten) |
| genutzte Rechenleistung der T4 | ~0.1 % | ~1 % |

**Für einen Einzellauf** — GPU gegen die CPU dieser Box — ist Faktor **2–3** zu
erwarten, nicht 10. Gerechnet aus einer Messung: 0.58 ms je Rollout-Schritt auf
der CPU (09.09., synthetische Fixture, 4 Kerne). Hochgerechnet auf 11 OPs ×
7000 Schritte × 100 inner_steps sind das ~45 s Rollout + ~33 s inner = ~78 s je
Epoche auf der CPU, gegen geschätzte ~27 s + ~5 s = ~32 s auf der T4. Der
Rollout gewinnt wenig, weil er nicht rechen-, sondern **Python-gebunden** ist —
~50 Ops je Schritt zu ~7 µs Dispatch, und Python läuft auf beiden Seiten gleich
schnell. Der Innenteil gewinnt viel: 2048×128-Matmuls plus doppeltes Autograd
sind echte Parallelarbeit.

Die Karte zahlt sich vor allem **in Kombination mit `-j`** aus: auf der CPU
konkurrieren vier parallele Läufe um dieselben FLOPs, auf der GPU nur um vier
Python-Kerne, während eine Karte die Mathematik für alle macht.

**Ein einzelner Lauf wird davon nicht schneller** — die 7000 Schritte bleiben
sequentiell. Schneller wird der *Sweep*. Die Grenze sind die 8 vCPU = **4
physische Kerne**, nicht die Karte: jeder Lauf ist eine Python-Schleife, die
Millionen Kernel-Starts absetzt, und will einen Kern für sich. `sweep.py` setzt
dafür `OMP_NUM_THREADS=1` in den Kindprozessen.

### 10.09. — der erste Lauf auf der Instanz

```bash
python3 PINNmodulusTwo/train.py --ops OP01 OP02 --epochs 3 --device cuda
```

**Exit 0 auf `cuda:0 Tesla T4`, torch 2.6.0+cu124.** ~2.7 min für den ganzen
Aufruf, 600 Optimizer-Steps (2 OPs × 100 `inner_steps` × 3 Epochen).

**Was das schließt:** der CUDA-Pfad des geräteseitigen `_LossBalancer` war bis
hierher ungetestet — in der Entwicklungsumgebung gibt es keine Karte, geprüft
war alles nur auf CPU. Er läuft.

**Was es misst: `peak VRAM 0.11 GB von 15.6 GB`.** Das ist keine Randnotiz,
sondern die Bestätigung der ganzen Prämisse: **Speicher ist hier nicht die
Ressource.** 0.11 GB bei 2 OPs; mit allen elf wären es grob 0.4–0.5 GB, weil je
OP nur `Tn`, `Qsrc` und `T_lab` dazukommen (~35 MB) und die Batchgrößen sich
nicht ändern. Von 15.6 GB. Es gibt nichts, was man an diesem Lauf drehen könnte,
um die Karte zu füllen, ohne das Experiment zu ändern.

**Was es NICHT misst — und das war der Zweck des Laufs:** die Zeile
`[Xs/epoch = Y rollout + Z inner]`. 2.7 min ist der ganze Aufruf inklusive
Datenladen, acht Auswertungs-Rollouts und Plots, also **nicht** durch 3 teilbar.
Ohne diese Zeile steht weiterhin kein Budget auf einer Messung. Nachzuholen:

```bash
grep "s/epoch" <logdatei>
```

**Die MAE aus diesem Lauf sind keine Ergebnisse** (3 Epochen, und trainiert
wurde nur auf OP01+OP02 statt auf allen elf — mit 5b/6 also nicht vergleichbar).
Der Vollständigkeit halber: train 6.73 / 5.43, val OP06/OP09 12.48 / 8.22, test
OP13/OP15/OP16 10.97 / 9.76 / 4.87 C.

Die Diagnostik dagegen ist lesbar, und sie verhält sich wie erwartet: Epoche 1
traf den `|Tn| <= 50`-Guard (`[SATURATED]`), Epochen 2–3 meldeten `[FLAT]` mit
`spread s/t = 0.055 / 0.111`. Beides ist bei drei Epochen der bekannte Verlauf —
Schritt 6 hat gezeigt, dass der `spread` von dort über 60 Epochen monoton auf
0.968 läuft (§11.3). **Aus drei Epochen darf daraus nichts geschlossen werden**;
genau dieser Fehlschluss steckt hinter O9.

---

### Was der Sweep dadurch spart — **1.46×, gemessen auf der T4**

**Am 10.09. auf der Instanz gemessen**, 4 Läufe (`--ops OP01 OP02 --epochs 6
--device cuda`, Auswertung auf die Trainings-OPs beschränkt):

| | `sweep_wall_s` | je Lauf | Summe `wall_s` |
|---|---|---|---|
| `-j 1` (seriell) | **596.2 s** (9.9 min) | 2.5 min | 596.2 s |
| `-j 4` (parallel) | **407.4 s** (6.8 min) | **6.8 min** | 1625.9 s |

**Der Faktor ist 596.2 / 407.4 = 1.46×**, also 37 % Effizienz auf vier Workern.
Gespart werden 3.1 von 9.9 Minuten.

> **Vorher stand hier 3.9×. Das war falsch, und der Fehler ist lehrreich.**
> Die 3.9× kamen aus `Summe der Laufzeiten / Wanduhr` — und genau das ist keine
> Messung, sondern eine **Obergrenze**: unter Konkurrenz wird jeder Lauf selbst
> langsamer, also wächst der Zähler mit. Hier auf 2.73× (596 → 1626 s), womit
> der Quotient 3.99 anzeigt, wo 1.46 stimmt. Nur eine echte `-j 1`-Baseline
> beantwortet die Frage. `sweep.py` schreibt die Warnung inzwischen selbst in
> die Ausgabe.

**Die eigentliche Zahl ist die dritte Spalte: 2.5 → 6.8 min je Lauf.** Und die
Epochenzeile sagt, *welcher* Teil das ist:

| | seriell | `-j 4` | |
|---|---|---|---|
| Rollout | 14.8 s | **49.6 s** | **3.35× langsamer** |
| Inner (×100) | 6.3 s | 7.9 s | 1.25× langsamer |

**Der Rollout serialisiert fast vollständig, der Innenteil kaum** — und das ist
die Diagnose, nicht mehr ein Verdacht. Der Rollout sind ~50 *winzige* Kernel je
Schritt (363×128), und winzige Kernel aus vier CUDA-Kontexten arbeitet der
Treiber zeitscheibenweise ab. Der Innenteil rechnet in größeren Batches (2048
bzw. 256 mit doppeltem Autograd), dort fällt ein Kontextwechsel kaum ins Gewicht.

**Damit ist MPS der passende Hebel**, nicht einer von zwei Verdächtigen: er lässt
Kernel verschiedener Prozesse nebeneinander auf den SMs laufen. Genau der Fall,
für den er gebaut wurde. Der Test (~7 min, noch nicht gelaufen):

```bash
nvidia-cuda-mps-control -d          # einmal je Boot
# denselben -j-4-Lauf wiederholen und sweep_wall_s vergleichen
```

Hilft MPS deutlich, war die Kontext-Umschaltung das Problem — und dann liegt
deutlich mehr als 1.46× drin, weil der Rollout 70 % der Epoche ausmacht. Hilft es
nicht, sind es doch die vier Kerne, und der Sweep gehört auf `--device cpu`, wo
es keine Kontext-Konkurrenz gibt.

**Mehr als 4 gleichzeitig lohnt nicht.** Bei vier ist die Effizienz schon auf
37 %; ein fünfter Prozess teilt dieselbe Karte und dieselben vier Kerne noch
feiner auf. Der Hebel ist MPS oder ein anderes Gerät, nicht mehr Prozesse.

**Was bleibt: 1.46× ist geschenkt und kostet nichts** — ein Sweep über eine
Achse (9 Läufe) fällt damit auf ~70 % seiner seriellen Zeit. Es ist nur nicht
die Größenordnung, mit der der Abschnitt darüber geplant hatte.

Die Wanduhr bleibt `ceil(Läufe / j)` Lauf-Dauern — eine letzte Welle mit weniger
Läufen als Arbeitern kostet eine ganze Dauer. **`-j` auf einen Teiler der
Laufzahl setzen**, 9 Läufe → `-j 3`. `sweep.py` rechnet die Wellen beim Start aus
und warnt bei krummem `-j`.

## Die Auswahlregeln stehen fest

Siehe §10 in Teil II. Kurz: Mittel über `--val-ops`, nie ein einzelner OP · nach
Tier getrennt berichten · Kriterium ist MAE, nie `L_data` · **`spread` als
Nebenbedingung in [0.7, 1.3]** · Spanne zwischen Konfigurationen < Spanne zwischen
Seeds → keine Rangfolge · **kein Befund aus der letzten Epoche** (neu 02.09.,
§11.6): Median über die letzten k Epochen, nie die letzte Zeile.

## Offene Punkte — nach Zuständigkeit

> **Was „O5", „O11" usw. bedeuten:** **O** steht für *offener Punkt*. Sie werden
> seit dem 31.08. fortlaufend durchnummeriert und **nie neu vergeben** — eine
> Nummer gehört für immer zu ihrer Sache, damit ein Verweis in einem Commit oder
> einer Nachricht nach Wochen noch stimmt. Offene stehen hier, geschlossene in
> Teil III unter „Geschlossene Punkte".

### Index: alle Punkte, O1 bis O16

| # | worum es geht | Zustand |
|---|---|---|
| O1 | Schritt-6-Ergebnisse bis Epoche 30 lagen unausgewertet | ✅ hinfällig (121er) |
| O2 | Schritt 5b war nie gelaufen | ✅ gelaufen, grün |
| O3 | OP15: `cell_current` fehlt im Bündel | ✅ nie exportiert |
| O4 | OP12: Profil endet bei 1440 s, Trajektorie bis 1605 s | ✅ Solver hielt den Wert |
| **O5** | **tote Eingangskanäle** (`soc_start`, Rate-Kanäle) | **offen — Envelope-Grenze**, kein MAE-Hebel (02.09. geklärt) |
| **O6** | **kein Gewicht auf Basis von Messungen gesetzt** | **offen — erstmals messbar**, aber blockiert durch O15 |
| O7 | Energiebilanz ging um ~147x nicht auf | ✅ der 121er, behoben |
| **O8** | **BDF-Stencil δ = 1.0 s gegen Δt_max 0.24 s** | **offen — entschieden: messen**, δ = 1.0 / 0.4 / 0.2 |
| O9 | dämpft der Physik-Term nur, statt Dynamik zu lernen? | ✅ **widerlegt** — `spread` → 0.968 |
| **O10** | **OP14s 0 °C sind geplant — nicht „reparieren"** | **stehende Warnung** |
| **O11** | **OP19 wird schlechter, je besser das Modell wird** | **dauerhafte Envelope-Grenze** (Datensatz ist fix) |
| O12 | BC-Term trägt fast nichts (`ratio` 0.0178) | ✅ **widerlegt** — 0.0178 war eine Epoche, Median 0.0581 |
| **O13** | **Fehler wächst zum Trajektorienende** | **offen** — und **kein** Verallgemeinerungsproblem |
| **O14** | **Volumenstrom ist die Schwierigkeitsachse** | **dauerhafte Envelope-Grenze** (Datensatz ist fix) |
| **O15** | **das Loss-Balancing greift nach 60 Epochen nicht** | **offen** — blockiert O6 |
| **O16** | **NEU: die Gehäusewand hat gar keine Randbedingung** | **offen** — Ziel ist gemessen, Umsetzung geplant |

Die offenen im Detail, nach Dringlichkeit:

| # | offen | wer / wann |
|---|---|---|
| **O15** | **NEU 02.09. Das Loss-Balancing hat nach 60 Epochen noch nicht eingesetzt.** Alle drei Divisoren fallen mit exakt 0.9000/Epoche — reiner geometrischer Zerfall, der aktuelle Loss trägt 0.009 % bei. `div_data/div_phys` ist damit über den ganzen Lauf **konstant**, das Balancing wirkt wie `fixed` mit Divisoren, die der erste Optimiererschritt gesetzt hat. Gemessen: `ratio_phys` = 0.586, wo `w_phys = 0.1` nominell 0.1 verspricht. Seit 02.09. als `xfail`-Test festgehalten und von `analyse_history.py` bei jedem Lauf gemeldet. §11.6 | **blockiert O6** |
| **O8** | Der BDF-Stencil nutzt δ = 1.0 s gegen Δt_max ≈ 0.24 s (`[CFL WARN]` bei jedem Lauf). **02.09. entschieden: wird gemessen, drei Punkte 1.0 / 0.4 / 0.2.** Nicht einfach richtiggestellt, weil es ein Zielkonflikt ist und kein Einheitenfehler — der Zähler `3T − 4T₋₁ + T₋₂` schrumpft mit δ, der Eigenfehler des Rollouts nicht. δ = 0.2 ist zugleich der Boden: darunter liest `history_at` nur noch die Gerade zwischen zwei Gitterpunkten. §11.2 | **Sweep-Achse 1** |
| **O13** | **Der Fehler sitzt am Ende der Trajektorie.** OP06: MAE 6.270 C, aber `late(held out)` = **13.248 C** — doppelt. Gleiches Muster auf OP03 (3.341 / 7.371) und OP16 (3.476 / 6.959). **02.09.: OP03 ist ein Trainings-OP und `--holdout-tail` ist aus — das Modell hat diese späten Zeitschritte mit Beschriftung gesehen und verdoppelt dort trotzdem seinen Fehler. O13 ist also kein Verallgemeinerungsproblem, sondern mechanisch.** §11.7 | **Messung gebaut 02.09.**, läuft beim nächsten Lauf mit |
| **O16** | **NEU 02.09. Die Gehäusewand hat keine Randbedingung.** `grep -rin "convect\|robin\|htc\|h_conv\|wall.*flux"` über `physics.py`, `model.py` und `train.py` findet **nichts**. `heat_residual` ist reine Leitung plus Quelle, **ohne Senke**, und der einzige BC-Term sitzt per `bc_mask = |x| < 1e-6` an der Symmetrieebene (`train.py:603`, „BC points (x=0): 121/363"). Der Wärmeaustritt an der Gehäusewand ist damit physikalisch **unbeschränkt** — das Netz lernt ihn allein aus dem Datenterm, obwohl dort laut Energiebilanz bis zur Hälfte der Quellenergie hinausgeht. **Neu behebbar**, weil der Wärmestrom `Q̇(t)` gemessen im Rohexport liegt (`*_Heat Transfer.csv`, Spalte `Heat Transfer: solid to fluid Monitor (W)`) — ein Robin-Term an `x = 0.0219` mit `U(V̇)` auf `A = 0.0206 m²` wäre also nicht geraten, sondern **gegen eine Messung beaufsichtigt**. ⚠ **Erklärt die val-Fehler NICHT:** §11.5 sagt, V̇ = 0 ist der *schwierigere* Fall (5.374 gegen 2.928 C), also bliebe es dafür bei O14. Herleitung und Einbauregel in [`GridCNN/README.md`](../GridCNN/README.md) §6 | **erst nach GridCNN-Stufe 1** (Bilanzprobe), sonst baut man eine falsche BC in ein laufendes Modell |
| **O6** | Kein Gewicht auf Basis von Messungen gesetzt. **02.09.: der Grund hat sich geändert.** Die fünf Gründe aus §10a (CFL-Verletzung, degenerierter Anker, saturierter Rollout, nie arbeitende Loss-Balance, 121er-Quelle) sind alle behoben — O6 ist nicht mehr „bewusst offen", sondern **erstmals messbar**. Nur nicht jetzt: solange O15 gilt, misst ein `w_phys`-Sweep den eingefrorenen Divisor mit | **nach O15** |
| **O5** | **Tote Eingangskanäle — 02.09. umgewidmet.** `soc_start` ist über alle OPs konstant 10 % (`DEAD -> forced to 0`), die Rate-Kanäle von `c_rate` und `fluid_mass_flow` sind im Training tot, auf OP15/OP16/OP19 aber lebendig. **Das ist kein MAE-Hebel:** eine Spalte ohne Varianz kann nichts erklären, und die SOC-Wirkung steckt über `q_dot` ohnehin schon im Modell. Der Punkt ist eine **Envelope-Grenze** und damit Vorbedingung für O11, nicht eine Code-Änderung. Kein Kanal wird gestrichen. §10a | dokumentiert, nichts zu tun |
| **O14** | **Volumenstrom ist die Schwierigkeitsachse, nicht der Tier.** V̇=0 im Mittel 5.374 C gegen 2.928 C mit Kühlung; alle drei No-Flow-OPs unter den letzten vier. Nur 2 von 11 Trainings-OPs haben V̇=0 (OP07 bei T0 = 10 °C, OP14 bei 0 °C), OP06 fährt 25 °C — das Regime „keine Kühlung bei mittlerer Starttemperatur" kommt im Training nicht vor. **02.09.: der Datensatz ist fix, es kommen keine OPs dazu.** Damit ist O14 keine Aufgabe mehr, sondern eine **dauerhafte Grenze**: nach V̇ getrennt berichten, nie als Modellfehler lesen. §11.5 | dauerhaft, nur berichten |
| **O11** | **OP19 wird schlechter, je besser das Modell wird.** 10.334 C sind 88 % schlechter als die 5.507 C nach drei Epochen, während jeder Simulations-OP sich verbessert hat. 16.7 σ unter dem trainierten `c_rate` (OP19 ist ein Fahrzyklus, OP01–OP16 sind ausnahmslos Ladungen), `soc_start` 77 % gegen 10 % über einen toten Kanal (O5). **02.09.: da der Datensatz fix ist und OP17/OP18 nie simuliert wurden, ist der Envelope nicht erweiterbar.** Jede Code-Änderung, die OP19 besser aussehen ließe, wäre eine Anpassung an die eine vorhandene Messung — wovor `op_registry.py:185` ausdrücklich warnt. §11.4 | dauerhaft, nie Auswahlkriterium |
| **O10** | **Warnung, kein Punkt.** OP14 startet über alle Punkte bei 0 °C. Das sieht nach Füllwert aus, ist aber die geplante Anfangsbedingung (`op_registry.py:123`: T0 = 0 °C, „coldest start in the set"). **Nicht maskieren, nicht ersetzen, OP14 nicht entfernen** — es ist einer von nur zwei OPs mit V̇ = 0, und die binden die Energiebilanz | nichts tun |

**Geschlossen am 01.09.:** O1, O2, O3, O4, O7, O9. **Am 02.09.:** O12. Siehe Teil III.

**Neu am 02.09.:** O16 — beim Entwurf von `GridCNN/` aufgefallen, siehe dort §6.

---

# TEIL II — NACHSCHLAGEWERK

Ändert sich selten. Hier steht, was der Datensatz ist, wie der Split begründet
wird, welche Datei was tut, und welche Regeln für Auswahl und Abbruch gelten.

## 1. Der Datensatz — und was OP17–OP19 wirklich sind

**Trainings-/Validierungs-/Testuniversum ist OP01–OP16.** Alle sechzehn sind
Ladevorgänge (CH), alle aus derselben Batemo+StarCCM+-Simulation. **OP19 kommt
als Messvergleich dazu** — nicht als siebzehnter Trainings-OP, sondern als
eigene Frage (siehe unten). Der Split
steht in `op_registry.py` und ist dort begründet:

| Rolle | OPs | Regel |
|---|---|---|
| `--ops` (Training) | OP01–05, 07, 08, 10, 11, 12, 14 | jeder Profil-**Typ**, den ein Selektions-OP braucht, kommt hier vor |
| `--val-ops` | OP06, OP09 | konstant + Profil, je einer. Darauf darf getunt werden |
| `--test-ops` | OP13, OP15, OP16 | Extrapolations-Tier. Einmal lesen, nie darauf auswählen |

**OP17–OP19 sind kein Teil davon.** Sie stehen im Plansheet unter einer eigenen
Überschrift — „Abgleich mit Minimodul-Test" — und vergleichen gegen **gemessene**
Minimodul-Daten statt gegen die Batemo/StarCCM+-Simulation. Jede Treiberspalte
liest dort `Test Data`, es gibt also keine Plansheet-Zeile zum Abschreiben wie
bei OP01–OP16. Was das Blatt nennt, ist die Art des Versuchs:

| | Art | Lade-/Entladerichtung | Besonderheit |
|---|---|---|---|
| **OP17** | `DCH, CC` | **Entladung**, 2C | die einzige Entladung überhaupt — OP01–OP16 sind alle CH |
| **OP18** | `Fast Charge Lotus` | Ladung | `V_max` 4.3 V statt 4.35 V |
| **OP19** | `Fahrzyklus TDD.3` | WLTP (synth.), gemischt | `V_max` 4.3 V |

> **OP19 existiert** und hat eine Zeile in
> `legacy/battery_surrogate_agenticWorkflow/op_matrix.yaml`, lässt sich also mit
> `generate_cache.py` bauen.
>
> **OP17 und OP18 sind schlicht noch nicht simuliert.** Deshalb haben sie weder
> eine Zeile noch einen Rohexport — es fehlt der Simulationslauf, nicht die
> Unterstützung. Sobald sie gerechnet sind, brauchen sie **keine Codeänderung**:
> nur ihre Id in `op_registry.MEASUREMENT_OPS_AVAILABLE`.
>
> „OP01 bis OP19" heißt heute also **siebzehn** verfügbare Betriebspunkte.

Sie werden über `--measurement-ops` ausgerollt und berichtet, aber **nie**
trainiert und **nie** ausgewählt. In `config.yaml` steht bereits
`measurement_ops: [OP19]`, er läuft also in jedem Lauf mit, sobald `OP19.npz`
gebaut ist. Fehlt das Bündel, gibt es eine `[SKIP]`-Zeile und der Lauf geht
normal weiter — anders als bei `ops`/`val_ops`/`test_ops`, die hart
fehlschlagen. Ein Bericht darf ein Training nicht blockieren.

Die Zahl ist anders zu lesen als jede andere in diesem Projekt: sie mischt
Modellfehler, Messfehler und die Lücke zwischen Simulation und Prüfstand, und
nichts trennt die drei. Und OP17 wie OP19 sind härter als jeder Test-OP, aus
einem Grund, den kein Coverage-Report formuliert: **das Modell hat Entladung nie
gesehen**, und einen Fahrzyklus auch nicht. Dass sie zunächst gegen die trivialen
Vorhersager verlieren, ist eine Aussage über den Trainings-Envelope — kein
Fehler.

Der Datenpfad ist davon unabhängig: `build_op` liest jedes Bündel, das da ist,
und misst am Bündel selbst, welche Kanäle Profile sind. Sobald ein `OP17.npz`
auftaucht, braucht es **keine Codeänderung** — nur einen Eintrag in
`op_registry.MEASUREMENT_OPS_AVAILABLE`.

---

## 2. Die eine Regel

> **Nichts, was Stunden kostet, bevor das Billige gelaufen ist, das es entwerten
> könnte.**

Jede Phase hat ein **Tor**. Ist es rot, geht es nicht weiter — dann wird das Tor
repariert, nicht die nächste Phase gestartet.

---

## 5. Die Dateien

Nach dem Aufräumen: **10 Python-Dateien, 4 Dokumente.**

| Datei | Rolle |
|---|---|
| `model.py` | `LearnableSwish`, `ModulusMLP`, `RecurrentField`, `rollout` |
| `physics.py` | anisotropes Wärme-Residuum + Neumann-BC |
| `data.py` | Laden, Normierung, Profil-Resampling, Treiber-Rate-Kanäle, Reports |
| `materials.py` | die Material-CSVs |
| `op_registry.py` | Plansheet OP01–OP16, Tiers, der Split |
| `op_metrics.py` | MAE/RMSE/peak/transient/quiescent/late je OP |
| `train.py` | Trainingsschleife + Auswertung + Checkpoint |
| `device_utils.py` | Device, Seed, TF32 |
| `generate_cache.py` | rohe CSVs → `OP*.npz` |
| `selftest.py` | Arithmetik-Checks, Sekunden |
| `tests/` | Rollout-Stabilität, History-Fastpath, Buchhaltung |
| `tools/` | Sonden: Daten, Grenzflächen, Rollout-Divergenz, synthetischer Cache |
| **`FAHRPLAN.md`** | **Einstieg. Hier anfangen.** |
| `README.md` | Nachschlagewerk: Dateien, Flags, warum die Rekurrenz so aussieht |
| `ARCHITECTURE.md` | wie es intern läuft, offener Befund 4.1 |
| `README_GPU_SERVER.md` | nur aufschlagen, wenn der Server dran ist |

---

## 6. Was nicht zu tun ist

| Nicht | Warum |
|---|---|
| Die Benchmarks aus der Historie zurückholen | Sie zu haben war nie das Problem — sie ohne Maßstab zu fahren war es. Erst G2 |
| Auf OP17/OP18 warten | Noch nicht simuliert. Sie blockieren nichts — OP01–OP16 sind vollständig |
| OP19 als Test-OP zählen | Fahrzyklus, gemischt geladen/entladen, gemessen statt simuliert. Verliert anfangs zu Recht |
| `w_phys` auf 1.0 / 10.0 erhöhen | `L_phys_bal = L_phys/EMA(L_phys)` ist selbstnormiert, `w_phys` kommt darin nicht vor |
| Zahlen aus dem alten OP01–OP05-Projekt übernehmen | Andere Normierung, anderes `A`, andere `phys_scale`. Nichts überträgt sich als Betrag |
| Am Physik-Term schrauben, bevor G3 grün ist | Sonst zweite Variable im Vergleich |
| `--holdout-tail` reflexartig einschalten | Bei den CC-CV-OPs **ist** das späte Fenster der CV-Auslauf; es abzuschneiden nimmt den schwersten Teil aus dem Training |

---

## 7. Abbruchkriterien

Ehrlichkeitsklausel — wann der Plan selbst falsch ist:

* **G2 zeigt `LOSES TO` auf beiden val-OPs, auch nach Phase 3** → das Problem
  sind die Daten oder die Modellklasse, nicht die Hyperparameter. Zurück auf
  Phase 1.
* **`[SATURATED]` lässt sich nicht auf 0 bringen** → der Rollout ist instabil,
  nicht untertrainiert. Dann `ARCHITECTURE.md` 3.1, nicht mehr Epochen.
* **Die Extrapolations-OPs bleiben katastrophal, während val gut ist** → das ist
  kein Fehler, das ist die Aussage. Elf Trainings-OPs sind eine harte Grenze;
  dann ist die Frage, ob der Envelope erweitert werden muss, nicht welches
  Gewicht gewinnt.

---

**Zuletzt fortgeschrieben:** 2026-09-01, Sitzungsende.

---

## 10. „Wird daraus am Ende ein Benchmark, oder füttere ich dich nur?"

Berechtigte Frage. Ehrliche Antwort: **im Moment ist es das Zweite, und das ist
nur für die ersten Schritte richtig.**

### Warum es gerade so läuft

Die ersten fünf Schritte beantworten Fragen, die **einmalig** sind und für die
sich keine Maschinerie lohnt: läuft der Code, stimmen die Bündel, wie groß ist
`A`, läuft der Rollout weg. Jede davon wird genau einmal gestellt. Ein
Sweep-Framework dafür zu bauen wäre wieder der Fehler, für den die acht
Benchmark-Skripte gelöscht wurden.

### Woran es kippt — und das ist ein hartes Kriterium

**Sobald zwei Konfigurationen verglichen werden sollen**, hört Hand-Auswerten
auf zu funktionieren, und zwar aus einem Grund, der nichts mit Bequemlichkeit zu
tun hat: eine MAE-Differenz zwischen zwei Läufen ist **nicht lesbar**, solange
die Streuung über Seeds daneben fehlt. Zwei Läufe mit 8.7 und 8.1 sind kein
Ergebnis, wenn derselbe Lauf mit einem anderen Seed zwischen 7.9 und 9.2
schwankt.

Das ist der Moment, an dem gebaut wird — nicht früher, nicht später.

### Was gebaut wird, konkret

Drei Dateien, in dieser Reihenfolge, jede einzeln nutzbar:

**1. `sweep.py` — die Seed-Schleife.** ~80 Zeilen.

```
python3 sweep.py --seeds 0 1 2 --epochs 20
  -> artifacts/sweep.csv: eine Zeile je (Konfiguration, Seed)
  -> stdout: Mittelwert und Std je Konfiguration über die val-OPs
```

Ruft `train.fit()` in einer Schleife, sonst nichts. Kein Plot, kein Resume, kein
Checkpoint-Merge. **Das allein hätte allen bisherigen Ergebnissen dieses Projekts
gefehlt.**

**2. Eine Achse.** Eine Liste von `fit()`-Overrides, eine CSV-Zeile je Punkt.
Erste Achse ist die Loss-Balance, nicht die Gewichte — solange der Physik-Term
kollabiert oder explodiert, misst ein Gewichts-Sweep das und nicht die Physik.

**3. Plots und Resume.** Zuletzt, und nur für die Achse, die wirklich Stunden
läuft.

### Die Bewertungsregeln stehen schon fest

Die müssen nicht erst gefunden werden — sie stammen aus dem gelöschten Code und
sind das Einzige daraus, was übernommen wird:

* Auswahl auf dem **Mittel über `--val-ops`**, nie über einen OP. Konstant-genau
  bleiben und einem bewegten Treiber folgen laufen gegeneinander; ein OP misst
  nur eines davon. Jede Einzel-MAE mitschleppen.
* **Nach Tier getrennt berichten.** Eine gemittelte Test-MAE über OP13/OP15/OP16
  mischt C-Raten-Extrapolation, ungesehenen Profiltyp und dreifachen Volumenstrom.
* **Kriterium ist MAE, nie `L_data`.** Die beiden ranken nachweislich
  verschieden — am 01.09. erneut belegt: der physikfreie 5b-Lauf hat die bessere
  `L_data` (0.298 gegen 0.831) und die schlechtere MAE.
* **`spread` ist eine Nebenbedingung, kein Ziel** (neu, 01.09.). Eine
  Konfiguration mit besserer MAE bei `spread = 0.2` hat nicht gewonnen, sie hat
  aufgehört vorherzusagen — `train.py` nennt das selbst die triviale Lösung.
  Auswahl also: **MAE minimieren unter `spread ∈ [0.7, 1.3]`**. Nicht als
  kombinierte Formel (die verschleiert, welche der beiden Zahlen entschieden
  hat), sondern als Tor davor. Siehe §11.3.
* **Seed-Rausch-Urteil.** Spanne zwischen Konfigurationen < Spanne zwischen Seeds
  → keine Rangfolge.
* **Kein Befund aus der letzten Epoche** (neu, 02.09.). Jede Größe, die je Epoche
  mitgeschrieben wird, ist über den Lauf verrauscht: `ratio_bc` streut mit 59 %
  relativ, `L_phys` springt in fünf von sechzig Epochen um ein bis zwei
  Größenordnungen. Ein Wert aus der letzten Zeile ist eine Stichprobe, kein
  Ergebnis — und genau so ist O12 entstanden. Berichtet wird der **Median über
  die letzten k Epochen**, mit der Streuung daneben. §11.6

### Was du dafür noch liefern musst: **einmal Schritt 5b oder 6 auswerten.**

Danach ist die Reihenfolge festgelegt und der Sweep wird gebaut. Das Füttern
endet an dieser Stelle, nicht irgendwann.

---

## 10a. Was aus den Messungen NICHT im Code gelandet ist — und warum

Ehrliche Bilanz zum 31.08. **An Gewichten, Vorverarbeitungs-Konstanten oder der
Loss-Balance ist nichts auf Basis der Messungen geändert worden.**

Geändert wurde nur, was unabhängig von den Zahlen falsch war: der
`residual_output`-Default, vier tote `config.yaml`-Schlüssel, `bc_scale` und
`phys_scale` (beim Merge regressiert), der `tier_of`-Absturz, dazu Diagnostik
(`env_check`, die `A`-Zeile, die MISMATCH-Aufschlüsselung) und Persistenz
(`history.csv`, periodischer Checkpoint).

**Warum nicht mehr:** die einzige Messung, die es gibt, kommt aus einem Lauf mit
CFL-Verletzung um Faktor 16.6, degeneriertem Anker, zu 99 % saturiertem Rollout
und einer Loss-Balance, die bei 2 Epochen und 10-Epochen-Horizont nie gearbeitet
hat (§9.3). Ein Gewicht auf dieser Grundlage zu setzen wäre geraten und würde
danach wie gemessen aussehen — genau die Sorte Zahl, wegen der die alten
Benchmarks gelöscht wurden.

**Am 01.09. kam ein fünfter Grund dazu, und er ist der härteste:** in diesem Lauf
war `Qsrc` um Faktor 121 zu klein (§11.1). Ein `w_phys`, das auf diesen Zahlen
gefunden worden wäre, hätte die Größe kompensiert, mit der der Fehler den
Physik-Term kleingerechnet hat — und wäre nach dem Fix um zwei Größenordnungen
falsch. Genau deshalb war „nichts an Gewichten ändern, bevor gemessen ist" hier
richtig und nicht bloß vorsichtig.

**Drei Dinge, die die Daten aber schon nahelegen**, festgehalten damit sie nicht
verlorengehen:

1. **`Qsrc_scale` ist ein einziger gepoolter Divisor über OPs, deren Qsrc-RMS um
   Faktor 2.3 auseinanderliegt** (OP05 0.0158 … OP11 0.0370). `w_phys` bedeutet
   damit auf OP05 etwas anderes als auf OP11. Ein per-OP-Divisor wäre denkbar —
   ändert aber die Gleichung pro OP und muss gemessen, nicht angenommen werden.
2. **Tote Kanäle (O5).** Seit 01.09. meldet `coverage_report` auch tote
   Skalar-Kanäle — vorher übersprang er sie, ausgerechnet den Fall, in dem der
   trainierte Umschlag ein einziger Punkt ist und `_normalise_config` den Wert
   unabhängig davon auf 0 zwingt. Die Entscheidung selbst bleibt offen.
   `soc_start` ist über alle sechzehn OPs konstant 10 %,
   trägt also null Information und kostet eine Eingangsdimension. Schlimmer: die
   Rate-Kanäle von `c_rate` und `fluid_mass_flow` sind im Training tot und auf
   OP15/OP16/OP19 lebendig — das Modell soll dort einen Kanal deuten, für den es
   nie ein Beispiel gesehen hat. Das ist **kein Hyperparameter, sondern eine
   Grenze des Trainings-Envelopes**, und der Coverage-Report sagt es bei jedem
   Lauf.
3. **`--batch-bc 128` gegen 121 BC-Punkte.** Es gibt nur 121 Punkte auf `x=0`,
   der BC-Gradient ist also rauschiger als das Gewicht suggeriert. Kleine Sache,
   aber sie gehört in den ersten Balance-Sweep.

---

## 3. Die Phasen

> **Hinweis:** Die Phasen sind die **ältere** Rahmung des Projekts. Operativ gilt
> die Schritt-Liste (1 … 6) aus Teil I und Teil III — sie ist feiner und
> aktueller. Die Phasen stehen hier, weil sie die Begründungen tragen, warum in
> dieser Reihenfolge vorgegangen wird. Zuordnung: Phase 0 ≙ Schritt 2, Phase 1 ≙
> Schritt 3–4, Phase 2 ≙ Schritt 5/5b, Phase 3 ≙ Schritt 6, Phase 4 ≙ `sweep.py`.

### Phase 0 — Rauchtest ohne Daten (Minuten, kein GPU)

Prüft die Mathematik, nicht das Ergebnis. Läuft auf einem frischen Checkout.

```bash
python3 PINNmodulusTwo/selftest.py            # Loss-Balance, Residuen-Skalierung
python3 -m pytest PINNmodulusTwo/tests -q     # Rollout, History-Fastpath, Checkpoint
python3 PINNmodulusTwo/op_registry.py         # der Split, ohne Daten
python3 PINNmodulusTwo/tools/rollout_divergence.py
```

**Tor G0:** alles grün.

Ohne `data_cache/` geht auch ein kompletter Trainingslauf, gegen ein
synthetisches Bündel:

```bash
python3 PINNmodulusTwo/tools/make_synthetic_cache.py
python3 PINNmodulusTwo/train.py --epochs 3 --subsample 40 \
        --ops OP01 OP02 --val-ops OP06 --test-ops
```

`train.py` druckt dabei ein Banner. Die Zahlen taugen zum Vergleich zweier
Läufe und zu sonst nichts — **nie als Ergebnis zitieren**.

### Phase 1 — Daten prüfen (Minuten, echte Daten, kein Training)

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
python3 PINNmodulusTwo/data.py            # Konstanten + profile_report + coverage
python3 PINNmodulusTwo/tools/data_probe.py
python3 PINNmodulusTwo/tools/interface_probe.py
```

**Tor G1:**

| Prüfung | Muss |
|---|---|
| `profile_report` | keine `MISMATCH`-Zeile. Das Plansheet ist eine Abschrift und kann falsch sein — glaub den Bündeln, nicht der Tabelle |
| `A` für `[5, 20]` | **notieren.** Das Pooling über OP01–OP16 verbreitert `T_sigma` und verkleinert damit `dTdt_scale`, also ist `A` hier **größer** als die 119/30 aus dem alten OP01–OP05-Projekt. Wie viel größer, ist ungemessen |
| `bc_scale` | aus x-Nachbarpaaren gemessen, **nicht** `[FALLBACK 1/L_ref]` |
| SNR | > 100, sonst misst der kurze Rate-Kanal Rauschen |
| Grenzflächenanteil | notieren — entscheidet über `ARCHITECTURE.md` 4.1 Option A vs. B |

### Phase 2 — Der Maßstab (Minuten) ← **am 01.09. gemessen, siehe Schritt 5b**

Ein kurzer Lauf, allein wegen der Latte. Die MAE des Modells ist hier egal:

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 | tee latte.txt
```

Interessant ist nur, was neben jeder OP-Zeile steht:

```
  OP06 [T1-interp  ] MAE=?? C  RMSE=?? C  ...
     baseline: beats|LOSES TO the trivial predictors
               (persistence=?? C, train-mean=?? C)
```

`persistence` ist „das Feld ändert sich nie", `train-mean` der konstante
Mittelwert der Trainings-Labels. Beide werden auf **genau dem OP** gerechnet, um
das es geht — nie zitiert, weil die Beträge zwischen OP-Sätzen nicht übertragbar
sind.

**Tor G2:** die Latte steht als Zahl fest, auf den **echten** OP06 und OP09.
Alles ab hier wird gegen diese Zahl gelesen.

### Phase 3 — Der erste ernsthafte Lauf

```bash
python3 PINNmodulusTwo/train.py --epochs 60 --device cuda | tee lauf1.txt
```

**Tor G3**, in dieser Reihenfolge zu prüfen:

| Signal im Log | Bedeutung | Reaktion |
|---|---|---|
| `[ABORT]` | Loss nicht-endlich | `--max-rate-amp 50`, dann `--history-mode raw`. Die A-Zeile aus G1 ansehen |
| `[SATURATED]` in der letzten Epoche | der Rollout ist weggelaufen und wurde festgehalten — **keine Vorhersage** | mehr Epochen → `lr` runter → längere `--rate-lags` |
| `[FLAT]` | `spread_space`/`spread_time` unter 0.2: das Feld ist konstant, und ein konstantes Feld erfüllt Residuum und Neumann-BC exakt. Ein fallendes `L_phys` ist dann die triviale Lösung | `--w-phys` / `--w-bc` senken |
| `[DIVERGED]` | der Eval-Rollout ist nicht-endlich (bewusst ungeclampt) | wie `[SATURATED]`, nur schlimmer |
| `LOSES TO` auf einem val-OP | das Modell ist schlechter als „nichts tun" | **das** ist das Problem, nicht die Gewichte |

Dazu der Drift-Test — `pred_OP13.npz` schreibt der Lauf selbst:

```bash
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP13.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
print('Wachstum', e[-(n//5):].mean() / e[1:n//5].mean())
"
```

Wachstum > 3 → Drift dominiert; das ist dann das Thema, nicht die Gewichte.

### Phase 4 — Erst jetzt wieder messen, und zwar neu gebaut

Reihenfolge des Neuaufbaus, **eine Sache pro Schritt** — das war der Fehler beim
letzten Mal:

1. **Zwei Läufe von Hand vergleichen.** Solange „ist A besser als B?" mit zwei
   `[val ]`-Zeilen beantwortbar ist, braucht es keine Maschinerie.
2. **Seeds.** Der erste echte Bedarf: eine MAE-Differenz ist wertlos ohne die
   Streuung über Seeds daneben. Eine Schleife über `--seed`, Mittelwert und Std.
   Das ist die einzige Ergänzung, die den bisherigen Ergebnissen wirklich
   gefehlt hat.
3. **Eine Achse.** Liste von `fit()`-Overrides, eine CSV-Zeile je Punkt. Kein
   Plot, kein Resume, kein Checkpoint-Merge.
4. **Plots und Resume ganz zuletzt**, nur für die Achse, die wirklich Stunden
   läuft.

Welche Achse zuerst: **Balance vor Gewichten vor Architektur.** `w_phys`
multipliziert `L_phys/EMA(L_phys)`, also eine selbstnormierte Größe — solange
der Physik-Term kollabiert, misst jeder Gewichts-Sweep den Kollaps.

Was aus dem alten Code übernommen wird, ist die **Bewertungslogik**, nicht die
Infrastruktur:

* Auswahl auf `--val-ops` als **Mittel über die Menge**, nie über einen OP.
  Konstant-genau bleiben und einem bewegten Treiber folgen sind zwei Ziele, die
  gegeneinander laufen; ein OP misst nur eines davon. Jede Einzel-MAE
  mitschleppen, sonst gewinnt eine Konfiguration den Mittelwert, indem sie eines
  der beiden ruiniert.
* **Nach Tier getrennt berichten.** Eine gemittelte Test-MAE über OP13, OP15 und
  OP16 mischt C-Raten-Extrapolation, einen ungesehenen Profiltyp und dreifachen
  Volumenstrom. Der Mittelwert dreier verschiedener Fragen beantwortet keine.
* **Kriterium ist MAE, nie `L_data`.** Die beiden ranken Konfigurationen
  nachweislich unterschiedlich.
* **Seed-Rausch-Urteil.** Ist die Spanne zwischen Konfigurationen kleiner als
  die zwischen Seeds einer Konfiguration, ist es keine Rangfolge.

Was **nicht** wiederkommt: dass jedes Skript seine eigene Kopie der Defaults
mitbringt. `config.yaml` ist die einzige Quelle.

---

## 4. Was du lokal machen musst

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate

# Phase 0 -- Minuten, keine Daten
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/op_registry.py

# Phase 1 -- Minuten. Der Cache braucht jetzt ALLE sechzehn.
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
python3 PINNmodulusTwo/data.py                | tee daten.txt
python3 PINNmodulusTwo/tools/interface_probe.py | tee interface.txt

# Phase 2 -- Minuten. Nur die baseline-Zeilen zaehlen.
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 | tee latte.txt

# Phase 3 -- der erste ernsthafte Lauf
python3 PINNmodulusTwo/train.py --epochs 60   | tee lauf1.txt
```

**Was du mir danach schicken kannst,** damit ich weiterrechne statt zu raten:
`daten.txt`, `latte.txt`, `interface.txt`, `lauf1.txt` und
`artifacts/metrics.txt`. Fünf kleine Textdateien.

Erst wenn G2 und G3 grün sind, lohnt sich der GPU-Server — vorher kostet er nur
Geld.

---

# TEIL III — ERLEDIGT

Archiv: abgehakte Schritte, gemessene Zahlen, geschlossene Befunde.

## Wo die Ergebnisse liegen

**Die Rohdateien liegen nur auf der Arbeitsmaschine** — `artifacts/` und die
`*.txt`-Logs sind gitignored, im Repo steht keine davon.

| Datei | was drin ist |
|---|---|
| `06_lauf.txt` | das volle Log von Schritt 6 (im Repo-Wurzelverzeichnis, dorthin `tee`t das Kommando) |
| `PINNmodulusTwo/artifacts/metrics.txt` | die MAE-Tabelle je OP mit Baselines und Coverage |
| `PINNmodulusTwo/artifacts/history.csv` | **eine Zeile je Epoche** — `L_data`, die balancierten Terme, `spread_space`, `spread_time`, `delta`, die Betas. Daraus kommt die `spread`-Kurve in §11.3 |
| `PINNmodulusTwo/artifacts/model.pt` | Checkpoint, alle 10 Epochen geschrieben. **Jeder neue Lauf überschreibt sie** — vor Achse 0 wegkopieren, wenn der Schritt-6-Stand erhalten bleiben soll. Es gibt bisher **keinen Weg, sie zurückzuladen** (§11.7) |
| `PINNmodulusTwo/artifacts/*.png` | die Plots |

**Im Fahrplan** stehen die Zahlen an vier Stellen, und die sind versioniert:

| wo | was |
|---|---|
| **Stand-Tabelle** (gleich unten) | alle gemessenen Werte je Schritt, mit Datum |
| **§11.3** | die `spread`-Kurve über 60 Epochen — O9 widerlegt |
| **§11.4** | OP19 wird schlechter — O11 |
| **§11.5** | die Einordnung: in-sample gegen ausgehalten, V̇ als Schwierigkeitsachse — O14 |
| **§11.6** | die volle Auswertung der `history.csv` — O12 widerlegt, **O15 neu** |
| **§11.7** | warum O13 mechanisch ist und nicht eine Frage der Verallgemeinerung |

## Geschlossene Punkte

| # | war | wie geschlossen | wann |
|---|---|---|---|
| **O1** | Schritt-6-Ergebnisse bis Epoche 30 lagen unausgewertet | hinfällig: der Lauf hatte die 121x zu kleine Quelle. 5b ersetzt ihn | 01.09. |
| **O2** | Schritt 5b war nie gelaufen | gelaufen und grün — `[SATURATED]` weg, beide val-OPs schlagen die trivialen Vorhersager | 01.09. |
| **O3** | OP15: `cell_current` fehlt im Bündel | Rohexport hat **keine** `CellCurrent(t).csv`. Nie exportiert, also stimmt das Plansheet für OP15 nicht. Blockiert nichts (Berichts-OP). §9a.1, Q2 | 01.09. |
| **O4** | OP12: Profil endet bei 1440 s, Trajektorie bis 1605 s | Kein Knick in `dT/dt` an der Nahtstelle → der Solver hat den letzten Wert gehalten, `np.interp` tut dasselbe. Keine Rückfrage nötig. §9a.2, Q1 | 01.09. |
| **O7** | Energiebilanz ging um ~147x nicht auf | Codefehler: `jr1_w` wurde zusätzlich durch die 121 JR1-Gitterpunkte geteilt. Behoben, mit Tests, und auf echten Daten bestätigt. §11.1 | 01.09. |
| **O9** | dämpft der Physik-Term nur, statt Dynamik zu lernen? | **Widerlegt.** `spread_time` steigt von 0.201 (Ep 3) auf 0.968 (Ep 60), während die MAE weiter fällt. Am 02.09. gegen die volle `history.csv` gehalten: Median über Ep31–60 = 0.922 bei Std 0.119 — die Aussage steht auf dreißig Epochen, nicht auf einer. §11.3, §11.6 | 01.09. |
| **O12** | BC-Term trägt fast nichts (`ratio bc` = 0.0178) | **Widerlegt, beide Hälften.** (a) Die 0.0178 sind der Wert **einer** Epoche einer Reihe mit 59 % relativer Streuung; Median Ep31–60 = 0.0581, Ep21–40 → Ep41–60 = 0.0540 → 0.0582, also kein Abfall. Der Term liegt bei 58 % seines nominellen `w_bc/w_data = 0.1` — unauffällig. (b) `--batch-bc 121` ändert nichts: `boundary_condition_loss` zieht `min(121, batch_bc)`, und `min(121,128) == min(121,121)`. §11.6 | 02.09. |

## Der Weg dahin (alles erledigt)

- [x] **1** Code holen — 31.08.
- [x] **2** Läuft der Code? (keine Daten nötig) — 31.08., grün
- [x] **3** Cache bauen, alle sechzehn — 31.08. (Bündel unberührt, gilt weiter)
- [x] **4** Stimmen die Daten? — **01.09. neu gelaufen, grün.** `[ENERGY]` weg,
  `Qsrc_scale` 2.916 / `phys_scale` 4.582 wie vorhergesagt, alles andere
  unverändert. O3 dabei entschieden
- [x] **5** Die Latte — 31.08. gelaufen, **nicht aussagekraeftig** (§9.3) und
  mit dem falschen `Qsrc` gerechnet. **Von 5b ersetzt, nicht zu wiederholen**
- [x] **5b** Kurzlauf bei der ECHTEN Konfiguration — **01.09., GRÜN.** `[SATURATED]` in Epoche 3 weg, und beide val-OPs schlagen zum ersten Mal die trivialen Vorhersager — **nur mit Physik-Term**
- [x] **6** Erster ernsthafter Lauf — **01.09., GRÜN auf allen fünf Signalen.** `spread` 0.201 → **0.968**, `[SATURATED]` ab Epoche 26 weg, beide val-OPs schlagen die trivialen Vorhersager deutlich

## Stand

Wird beim Abhaken ausgefüllt. Leer = noch nicht gemessen.

| Schritt | Kriterium | gemessen | Datum |
|---|---|---|---|
| 2 | `selftest.py` | **all checks passed** | 31.08. |
| 2 | `pytest` | **123 passed, 1 skipped** | 01.09. |
| 2 | `op_registry.py` | 11 train / 2 val / 3 test, keine Warnung | 31.08. |
| 3 | 16 OPs gebaut | ja (OP19 offen) | 31.08. |
| 4 | `MISMATCH`-Zeilen | **1 — OP15, `cell_current` fehlt.** Ursache am 01.09. entschieden: **nie markiert** (siehe O3) | 31.08. / 01.09. |
| 4 | `[ENERGY]`-Zeile | **weg.** „balance holds to within 0.9x on the binding OP" (war ~147x) | **01.09.** |
| 4 | Bilanz je OP | **0.5 .. 0.9x, und sie folgt dem Volumenstrom**: V̇=0 (OP07/OP14) -> 0.9x, V̇=0.0013 -> 0.6-0.7x, V̇=0.0026 -> 0.5x | **01.09.** |
| 4 | `q_dot` (physikalisch) | mu=**6.578e4** W/m^3, Bereich -5.238e4 .. 1.534e5 | **01.09.** |
| 4 | `bc_pairs` > 0 | **242** — gemessen, kein Fallback | 31.08. |
| 4 | **`A` je Lag** | **90.8 / 22.7** (bei dt = 4 s) | 31.08. |
| 4 | `dTdt_scale` | **3.534** | 31.08. |
| 4 | `T_sigma` / `T_span_ref` | 9.616 C / 1604 s | 31.08. |
| 4 | `phys_scale` / `Qsrc_scale` | **4.582 / 2.916** — gemessen. Vorhergesagt waren 4.582 / 2.916 (= 0.0241 x 121), auf vier Stellen getroffen. Alt: 3.535 / 0.0241 | **01.09.** |
| 5 | OP06 | `LOSES TO` (12.96 vs 10.82 C) — **nicht aussagekraeftig, §9.3** | 31.08. |
| 5 | OP09 | `LOSES TO` (8.71 vs 7.78 C) — dito | 31.08. |
| 5 | `[SATURATED]` | **ja, beide Epochen: 99 % / 94 % einer Trajektorie** | 31.08. |
| 5 | `spread s/t` | **4.9 / 4.2** — Rollout streut 5x so weit wie die Labels | 31.08. |
| — | alle Schritt-5-Zahlen | **mit `Qsrc` 121x zu klein gerechnet.** Der Physik-Term war praktisch abgeschaltet; was §9.3 zeigt, ist ein Rollout ohne Quelle | 01.09. |
| 5b | `[SATURATED]` bei subsample 2 | **verschwindet.** Ep1 OP05 7364/7368 (99.9 %), Ep2 OP04 6370/7225 (88 %), **Ep3 keine** | **01.09.** |
| 5b | `A` bei subsample 2 | **90.5 / 22.6** — praktisch identisch zu den 90.8/22.7 bei dt = 4 s. Die Warnung „A hängt am subsample" war für diesen Datensatz gegenstandslos | **01.09.** |
| 5b | `spread s/t` | **9.02/6.04 → 0.339/0.201** (mit Physik). Vom Explodieren ins Überdämpfte, an 1 vorbei | **01.09.** |
| 5b-1 | **val OP06** | **MAE 10.540 C — beats** (persistence 16.679, train-mean 10.801) | **01.09.** |
| 5b-1 | **val OP09** | **MAE 7.494 C — beats** (persistence 18.549, train-mean 7.762) | **01.09.** |
| 5b-2 | val OP06 / OP09 (ohne Physik) | **11.591 / 8.504 C — LOSES TO** auf beiden | **01.09.** |
| 5b-1 | test OP13 / OP15 / OP16 | 8.686 / 7.239 / 4.204 C — alle drei **beats** | **01.09.** |
| 5b-1 | OP19 (Messvergleich) | 5.507 C — **LOSES TO** (persistence 1.376), wie §9.4 vorhergesagt | **01.09.** |
| 6 | `[ABORT]` | **kam nicht** | 01.09. |
| 6 | `[SATURATED]` letzte Epoche | **weg.** Fällt monoton: Ep1 99.9 % (OP05) → Ep4 99.4 % → Ep14 47 % → Ep25 22 % → ab Ep26 keine Zeile | **01.09.** |
| 6 | **`spread s/t`** | **1.10 / 0.968** — von 9.02/6.04 in Ep1 monoton auf ~1 und dort geblieben. **O9 beantwortet** | **01.09.** |
| 6 | `[FLAT]` | **kam nie** | 01.09. |
| 6 | MAE OP06 / OP09 | **6.270 / 3.585 C — beide beats.** 42 % bzw. 54 % besser als `train-mean` (10.801 / 7.762), gegen `persistence` 16.679 / 18.549 | **01.09.** |
| 6 | MAE OP13 / OP15 / OP16 | **4.097 / 4.809 / 3.476 C — alle drei beats**, 57 / 40 / 24 % besser als `train-mean` | **01.09.** |
| 6 | MAE Trainings-OPs | 1.000 (OP01) … 5.656 (OP14), **alle beats** | 01.09. |
| 6 | MAE OP19 (Messvergleich) | **10.334 C — LOSES TO** (persistence 1.376). Und 88 % **schlechter** als in 5b. Siehe O11 | **01.09.** |
| 6 | `L_data` | 100.5 → **0.0515** | 01.09. |
| 6.3 | **läuft auf der Karte** | **Exit 0** auf `cuda:0 Tesla T4`, torch 2.6.0+cu124. 2 OPs, 3 Epochen, 600 Optimizer-Steps, ~2.7 min für den ganzen Aufruf (inkl. Auswertung und Plots) | **10.09.** |
| 6.3 | **peak VRAM** | **0.11 GB von 15.6 GB** — bei 2 OPs und den Default-Batches | **10.09.** |
| 6.3 | **Parallelität `-j 1` vs `-j 4`** | **1.46×** — `sweep_wall_s` 596.2 s seriell gegen 407.4 s parallel, 4 Läufe, `--epochs 6 --ops OP01 OP02 --device cuda`. Effizienz 37 %. Je Lauf 2.5 → 6.8 min, also **2.7× langsamer unter Konkurrenz** | **10.09.** |
| 6.3 | MPS-Gegentest | **offen** — `nvidia-cuda-mps-control -d`, dann `-j 4` wiederholen. Trennt „die eine Karte bremst" von „die vier Kerne bremsen" | — |
| 6.3 | **`s/epoch`** | **21.1 s = 14.8 rollout + 6.3 inner** (2 OPs, 100 inner_steps, seriell). Unter `-j 4`: 57.5 = 49.6 + 7.9 — **der Rollout wird 3.35× langsamer, der Innenteil nur 1.25×** | **10.09.** |
| 6.3 | **Planungszahl** | 10.55 s je OP und Epoche → ~116 s/Epoche bei elf OPs → **~1.9 h je 60-Epochen-Lauf** | **10.09.** |
| 6.3 | README-Beispiel `118.7s/epoch` | **veraltet, Faktor 5.6.** `inner` ist in beiden Zeilen identisch 6.3 s, also dieselbe Konfiguration; der Rollout fiel von 112.4 auf 14.8 s (**7.6×**, der `rollout_plan`-Fastpath). Budgettabellen in Kapitel 7/8 entsprechend zu pessimistisch | **10.09.** |
| 6 | Loss-Balance | arbeitet jetzt: `ratio phys/bc` 2.17/0.782 → **0.605/0.0178**, betas [0.91 …] → [0.98 2.64 3.9 3.99] | 01.09. |

Alles läuft aus dem Repo-Wurzelverzeichnis:

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate
```

---

### - [x] Schritt 1 — Code holen (1 min)

PR #20 ist gemergt, es reicht also `main`:

```bash
git checkout main
git pull
```

> `PINNmodulusTwoExtProfiles/` verschwindet dabei — das ist gewollt, der Ordner
> ist in `PINNmodulusTwo/` aufgegangen. Falls dort noch ein `data_cache/` liegt:
> **stehen lassen**, `data.py` sucht ihn weiterhin.

**Stopp wenn:** `git status` nach dem Pull nicht sauber ist.

---

### - [x] Schritt 2 — Läuft der Code überhaupt? (2 min, keine Daten nötig)

```bash
source modulus_env/bin/activate          # <- OHNE DAS schlaegt alles fehl
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/op_registry.py
```

**Gemessen am 31.08.:** `all checks passed`, **112 passed, 1 skipped** (83 s),
und die Tabelle mit 11 train / 2 val / 3 test OPs ohne Warnung. ✅

> **Die Aktivierungszeile ist der Stolperstein.** Ohne sie laeuft alles unter
> `/usr/bin/python3` und der Fehler lautet
> `ModuleNotFoundError: No module named 'pandas'` — vier Importe tief in
> `materials.py`, wo nichts kaputt ist. `op_registry.py` laeuft trotzdem durch,
> weil es reine Standardbibliothek ist, was den Eindruck verstaerkt, es fehle
> nur eine Bibliothek.
>
> Seit dem 31.08. faengt `env_check.py` das ab und sagt stattdessen, welcher
> Interpreter laeuft und dass das venv fehlt. **Nicht** `pip install pandas` ins
> System-Python — das macht den naechsten Fehler schwerer lesbar, nicht
> leichter.

**Stopp wenn:** irgendetwas davon rot ist. → Ausgabe schicken.

---

### - [x] Schritt 3 — Cache bauen, alle sechzehn (10–30 min)

Der Cache muss neu, weil bisher nur OP01–OP07 gebraucht wurden:

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16 2>&1 | tee 03_cache.txt
```

Und `OP19` — den gibt es, er ist der Messvergleich. Er gehört **nicht** ins
Training und wird separat gebaut:

```bash
python3 PINNmodulusTwo/generate_cache.py OP19 2>&1 | tee -a 03_cache.txt
```

`config.yaml` hat `measurement_ops: [OP19]`, er läuft ab dann in jedem
`train.py`-Lauf als Bericht mit. Fehlt das Bündel, gibt es eine `[SKIP]`-Zeile
und sonst nichts — ein Messvergleich darf einen Trainingslauf nie blockieren.

**Stopp wenn:** ein OP nicht baut. → `03_cache.txt` schicken.

---

### - [x] Schritt 4 — Stimmen die Daten? (2 min) ← **01.09. grün**

> **Am 01.09. neu gelaufen und grün.** Der 121er-Fehler aus §11.1 saß in
> `q_dot`, also in jeder Zahl, die aus der Quelle folgt. Gemessen: `Qsrc_scale`
> 0.0241 → **2.916**, `phys_scale` 3.535 → **4.582**, `[ENERGY]` **weg**
> („balance holds to within 0.9x"), und `dTdt_scale`, `T_sigma`, `A`, `bc_pairs`
> und die MISMATCH-Zeile exakt unverändert. Genau drei Zahlen haben sich bewegt,
> um exakt 121. Die Tabelle mit Vorhersage gegen Messung steht in §11.1.

```bash
python3 PINNmodulusTwo/data.py 2>&1 | tee 04_daten.txt
```

Vier Zeilen zählen, und zwar in dieser Reihenfolge:

| worauf schauen | gut | schlecht |
|---|---|---|
| `profile_report` | keine Zeile mit `MISMATCH` | jede `MISMATCH`-Zeile. Das Plansheet ist eine Abschrift — **glaub den Bündeln, nicht der Tabelle** |
| `bc_scale=… (from N x-neighbour pairs)` | `N > 0` | `[FALLBACK 1/L_ref]`. Dann ist `w_bc` bedeutungslos |
| `A = 1/(lag_n * rate_scale) per lag: …` | **notieren, egal welcher Wert** | — |
| `[ENERGY]`-Zeile | **fehlt**, oder „balance holds to within …x" mit einem kleinen x | jede `[ENERGY]`-Zeile. Dann ist die Quelle immer noch falsch und `w_phys` weiterhin nicht einstellbar (§11.1) |

**Gemessen am 31.08. — und die Vorhersage hier war falsch.** Es stand:
„`T_sigma` wird breiter, `dTdt_scale` kleiner, `A` damit **größer** als die
119/30 aus OP01–OP05." Gemessen ist `A` **kleiner**:

| | OP01–OP05 (alt) | OP01–OP16 (gemessen) |
|---|---|---|
| `T_span_ref` | 1474 s | **1604 s** |
| `dTdt_scale` | 2.479 | **3.534** |
| `A` bei 5 s / 20 s | 119 / 30 | **90.8 / 22.7** |

`T_sigma` ist tatsächlich breiter geworden (9.6 statt ~4.2), aber `dTdt_scale`
ist trotzdem **gestiegen**, nicht gefallen: die Profil-OPs bewegen sich schneller
als die konstanten, und das schlägt die Verbreiterung. Weniger Verstärkung heißt
weniger Abbruchrisiko in Epoche 1 — die gute Richtung.

> ⚠️ **`A` hängt am `--subsample`.** `dTdt_scale` ist die RMS einer zentralen
> Differenz **auf dem subgesampelten Gitter**: ein grobes Gitter glättet die
> Ableitung, ein feines nicht. Die 90.8/22.7 sind bei `--subsample 40`
> (dt = 4 s) gemessen, das Training läuft aber mit `subsample_time: 2`
> (dt = 0.2 s). Die maßgebliche Zahl druckt `train.py` beim Start — die aus
> Schritt 4 ist die Größenordnung, nicht der Wert.

**Stopp wenn:** ein `MISMATCH` auf einem **Trainings- oder val-OP**, oder
`FALLBACK` bei `bc_scale`. Ein `MISMATCH` auf einem **Test-OP** stoppt nicht —
er entwertet dessen Bericht, nicht das Training. → `04_daten.txt` schicken.

---

### - [x] Schritt 5 — Die Latte ← **erledigt, überholt, nicht wiederholen**

> **Zweimal überholt.** Erstens lief er bei dt = 4 s statt 0.2 s (§9.3),
> zweitens mit `Qsrc` 121x zu klein (§11.1). Nicht wiederholen — **5b ersetzt
> ihn**, und 5b läuft bei der echten Konfiguration.

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cpu \
        2>&1 | tee 05_latte.txt
```

Die MAE des Modells ist hier **egal** — zwei Epochen lernen nichts. Es geht um
die Zeile unter jedem OP:

```
  OP06 [T1-interp  ] MAE=?? C  ...
     baseline: beats|LOSES TO the trivial predictors
               (persistence=?? C, train-mean=?? C)
```

`persistence` = „das Feld ändert sich nie". `train-mean` = konstanter
Mittelwert. **Das ist die Zahl, auf die dieses Projekt seit Monaten wartet:**
schlägt das Modell „nichts tun"?

Nach zwei Epochen darf da noch `LOSES TO` stehen. Wichtig ist, dass die Zahlen
überhaupt da sind und der Lauf durchläuft.

> ⚠️ **Am 31.08. gelaufen — und die Zahlen taugen nicht als Latte.** `--subsample
> 40` bedeutet dt = 4 s und verletzt die CFL-Grenze (0.241 s) um Faktor 16.6,
> und `--delta-grid 0.2s` degeneriert unter einem 4-s-Gitter. Dieser Schritt
> zeigt also, **ob** der Lauf durchläuft, nicht **wie gut** er ist. Die echte
> Latte kommt aus 5b. Siehe §9.3.

**Stopp wenn:** `[ABORT]` — dann ist `A` zu groß, und Schritt 6 wäre
verschwendete Zeit. → `05_latte.txt` schicken, ich sage dir den Wert für
`--max-rate-amp`.

---

### - [x] Schritt 5b — 01.09. gelaufen, GRÜN → Schritt 6 ist frei

> ## Das Ergebnis, auf das dieses Projekt seit Monaten wartet
>
> | | 5b-1 (mit Physik) | 5b-2 (`--w-phys 0 --w-bc 0`) |
> |---|---|---|
> | **val OP06** | **10.540 C — beats** | 11.591 C — LOSES TO |
> | **val OP09** | **7.494 C — beats** | 8.504 C — LOSES TO |
> | `[SATURATED]` Ep3 | **weg** | weg |
> | `spread s/t` Ep3 | 0.339 / 0.201 | 2.15 / 1.03 |
>
> Zwei Aussagen, und die zweite ist die größere:
>
> 1. **`[SATURATED]` verschwindet.** Ep1 OP05 99.9 %, Ep2 OP04 88 %, Ep3 keine
>    Zeile. Der weglaufende Rollout aus §9.3 war dt, genau wie vermutet. Nach der
>    Entscheidungstabelle unten heißt das: **Schritt 6 starten.**
> 2. **Der Physik-Term trägt.** Ohne ihn verlieren beide val-OPs gegen die
>    trivialen Vorhersager, mit ihm schlagen sie beide. Das ist das erste Mal
>    überhaupt, und es ist erst seit dem 121er-Fix messbar — vorher waren beide
>    Läufe praktisch quellenfrei (§11.1).
>
> **Was diese Zahlen NICHT hergeben** (§10, unverändert gültig): ein Seed. Die
> MAE-Differenz 10.54 gegen 11.59 ist ohne Seed-Streuung daneben nicht lesbar,
> und der Vorsprung vor `train-mean` ist mit 2–4 % dünn (gegen `persistence` ist
> er komfortabel). 5b-2 schaltet außerdem `w_phys` **und** `w_bc` zusammen ab,
> trennt die beiden also nicht. Drei Epochen. Für eine Rangfolge zwischen
> Konfigurationen reicht das nicht — dafür wird jetzt die Seed-Schleife gebaut.

**Der ursprüngliche Plan, zur Nachvollziehbarkeit:**


**Der Schritt, der entscheidet, ob Schritt 6 seine Stunden wert ist.**

Schritt 5 lief bei dt = 4 s; das Training läuft bei dt = 0.2 s. Zwei der drei
Probleme aus §9.3 verschwinden dadurch von selbst — die CFL-Verletzung und der
degenerierte Anker. Das dritte, der **weglaufende Rollout**, verschwindet nicht
automatisch. Genau das wird hier gemessen, und zwar mit **einer** Variablen
zwischen den beiden Läufen.

> **Seit 01.09. misst 5b außerdem etwas anderes als geplant.** Der Physik-Term
> hatte bis dahin eine 121x zu kleine Quelle (§11.1), war also fast nur
> Diffusion. Der Vergleich unten — mit gegen ohne Physik — trennt zum ersten Mal
> zwei Läufe, deren Physik-Term wirklich die Gleichung enthält, der die Daten
> gehorchen. Und `phys_scale` ist gestiegen (Erwartung: 3.5 → ~4.6), das
> Residuum wird also **anders** normiert als in jedem bisherigen Lauf.

```bash
# 5b-1: die echte Konfiguration, nur kurz
python3 PINNmodulusTwo/train.py --epochs 3 2>&1 | tee 5b1_echt.txt

# 5b-2: dasselbe OHNE Physik- und BC-Term
python3 PINNmodulusTwo/train.py --epochs 3 --w-phys 0 --w-bc 0 \
        2>&1 | tee 5b2_ohne_physik.txt
```

Der zweite Lauf ist keine Spielerei: `README.md` sagt, der Clamp sei erst mit
`w_phys > 0` tragend — *„the physics gradient walks the weights out of the stable
region faster"*. Bei dt = 4 s war der Physik-Gradient nachweislich Rauschen; ob
er es bei dt = 0.2 s immer noch ist, trennen diese zwei Läufe.

**Zu notieren, aus beiden Läufen:**

| Zeile | warum |
|---|---|
| `A = 1/(lag_n * rate_scale) per lag: …` | bei dt = 0.2 s, das ist der **maßgebliche** Wert. Bei dt = 4 s waren es 90.8 / 22.7 |
| `[CFL …]` | muss jetzt `CFL OK` sagen |
| `[SATURATED]` je Epoche | **die Zahl, um die es geht** |
| `spread s/t` | bei 4.9/4.2 in Schritt 5; nahe 1 wäre gesund |
| `Qsrc_scale` / `phys_scale` | aus Schritt 4 wiederholen — sie sind seit dem 121er-Fix neue Zahlen |

**Die Entscheidung danach:**

| 5b-1 | 5b-2 | heißt | dann |
|---|---|---|---|
| kein `[SATURATED]` | — | dt war das Problem | **Schritt 6 starten** |
| saturiert | sauber | der Physik-Gradient treibt es | `--w-phys 0.01`, oder `phys_scale` prüfen — **nicht** 60 Epochen. Anders als am 31.08. ist das jetzt eine Aussage über die Physik und nicht über einen abgeschalteten Term |
| saturiert | saturiert | die Rekurrenz selbst | `--max-rate-amp 50`, dann `--history-mode raw` — **nicht** 60 Epochen |

**Stopp wenn:** beide saturieren. → beide Dateien schicken.

---


---

### - [x] Schritt 6 — 01.09. gelaufen, grün auf allen fünf Signalen

> **01.09. gelaufen, ~2 h auf CPU, grün auf allen fünf Signalen.** Kein
> `[ABORT]`, kein `[FLAT]`, `[SATURATED]` ab Epoche 26 verschwunden, `spread`
> auf 0.968, und alle sechzehn Simulations-OPs schlagen die trivialen
> Vorhersager. Die Zahlen stehen in der Stand-Tabelle, die Deutung in §11.3
> (O9 geschlossen) und §11.4 (O11 neu).

**Erst wenn 5b grün ist** — also `[SATURATED]` verschwunden. Sechzig Epochen auf
einem Rollout, der zu 99 % im Clamp hängt, ranken das Clamp-Verhalten und nicht
das Modell.

```bash
python3 PINNmodulusTwo/train.py --epochs 60 2>&1 | tee 06_lauf.txt
```

`--device` fragt jetzt nach und listet auf, was die Maschine hat:

```
Which device should this run use?
  [1] cpu      CPU  (32 threads visible)
  [2] cuda:0   NVIDIA …  24.0 GiB   <- default
Choice [1-2, Enter = cuda:0]:
```

Über `nohup` oder ohne Terminal fragt er nicht, sondern nimmt `auto` und sagt
das. Dauerhaft festlegen: `device: cuda` in `config.yaml`.

Vier Signale im Log, in dieser Rangfolge:

| Signal | heißt | Reaktion |
|---|---|---|
| `[ABORT]` | Loss nicht-endlich | zurück zu Schritt 5 |
| `[SATURATED]` in der **letzten** Epoche | Rollout weggelaufen und festgehalten — **keine Vorhersage** | mehr Epochen → `lr` runter → längere `--rate-lags` |
| `[FLAT]` | Feld konstant; ein fallendes `L_phys` ist dann die triviale Lösung, nicht Physik | `--w-phys` / `--w-bc` senken |
| **`spread` bleibt bei ~0.3** | **O9: die MAE-Verbesserung ist Varianzreduktion, keine Dynamik.** In 5b stand er bei 0.201, die `[FLAT]`-Schwelle ist 0.2 | `--w-phys 0.01` gegen `0.1` messen — **nicht** mehr Epochen. §11.3 |
| `LOSES TO` auf OP06/OP09 | schlechter als „nichts tun" | **das** ist das Problem, nicht die Gewichte |

#### Worauf geschaut wurde — und was herauskam

| # | Signal | gut | schlecht → was dann |
|---|---|---|---|
| 1 | `[ABORT]` | kommt nicht | Loss nicht-endlich → zurück zu 5b, `--max-rate-amp 50` |
| 2 | `[SATURATED]` in der **letzten** Epoche | keine Zeile | Rollout weggelaufen → mehr Epochen, dann `lr` runter |
| 3 | **`spread s/t`** über die Epochen | **steigt Richtung 1** | **bleibt bei ~0.3 oder fällt → O9 bestätigt**, siehe unten |
| 4 | `[FLAT]` | kommt nicht | dasselbe wie 3, nur lauter |
| 5 | `LOSES TO` auf OP06/OP09 | kommt nicht | schlechter als „nichts tun" → das Problem sind Daten/Modellklasse, nicht Gewichte |

**Signal 3 ist das neue und das wichtigste.** In 5b stand `spread_time` bei
**0.201** — die `[FLAT]`-Warnung löst bei 0.2 aus, es fehlten also 0.5 %. Ob das
Untertrainiertheit war oder ein Artefakt des Physik-Terms, entscheidet **dieser**
Lauf und sonst nichts:

* `spread` steigt Richtung 1 **und** MAE fällt → der Physik-Term trägt wirklich.
  Weiter mit der Seed-Schleife (§10).
* `spread` bleibt bei ~0.3 **und** MAE fällt trotzdem → **O9 bestätigt**: die
  MAE-Verbesserung ist Varianzreduktion, keine bessere Dynamik. Dann **nicht**
  mehr Epochen, sondern `--w-phys 0.01` gegen `0.1` messen.

#### Die Dateien dazu

```
06_lauf.txt   +   PINNmodulusTwo/artifacts/metrics.txt
              +   PINNmodulusTwo/artifacts/history.csv    <- wegen spread je Epoche
```

`history.csv` ist diesmal die wichtigere Datei: die `spread`-Spalten über 60
Epochen sind das, was 5b nicht beantworten konnte.

---

---

# Rohexport-Untersuchung Q1–Q3 — beantwortet am 01.09.

Der lokale Bot hat in die Rohdaten geschaut. **Zwei Fragen sind geschlossen,
eine war ein Fehlalarm — und dessen Empfehlung hätte Schaden angerichtet.**

## Q1 → **O4 GESCHLOSSEN**: der Solver hat gehalten, `data.py` tut dasselbe

`OP12_FluidInletTemperature(t).csv` hat **4 Stützpunkte** (0, 480, 960, 1440 s);
die Trajektorie läuft bis 1605.3 s. Die letzten 165 s (~10 %) werden mit dem
letzten Wert (35 °C) flach gehalten.

Drei Gründe, es so zu lassen:

1. **Die vorgeschlagene „Option B: Profil mit letztem Wert extrapolieren" ist
   der Ist-Zustand.** `np.interp` in `data.py` hält außerhalb des Bereichs
   genau den Randwert. Es gibt nichts umzustellen.
2. **Der Solver hat mit hoher Wahrscheinlichkeit dasselbe getan.** StarCCM+
   braucht nach 1440 s eine Randbedingung, und die Voreinstellung für eine
   Tabellen-RB ist das Halten des letzten Werts. Dann ist die Flachhaltung nicht
   eine Annahme über die Simulation, sondern eine Kopie davon.
3. **Empirisch tut es nicht weh: OP12 hat die BESTE MAE aller elf
   Trainings-OPs** (3.942 C). §9a.2 sagte, man erkenne das Problem daran, dass
   OP12 auffällig schlechter sei. Er ist auffällig besser.

„Trajektorie bei 1440 s abschneiden" wäre der schlechteste Weg: 10 % echter
Simulationsdaten wegwerfen, um eine Annahme zu vermeiden, die empirisch nicht
schadet. **Kein Cache-Neubau.**

**Und die Restfrage ist ebenfalls beantwortet — aus den Daten, nicht durch eine
Rückfrage.** Die Frage war: welchen Einlasstemperaturverlauf hat StarCCM+ nach
1440 s tatsächlich verwendet? Der Test dafür ist ein Knick in `dT/dt` an der
Nahtstelle — hätte der Solver dort etwas anderes benutzt als den letzten
Tabellenwert, müsste die Ableitung springen.

```
Sprung bei 1440 s: 0.000185 °C   (normal: ~0.000182 °C)
dT/dt: 2.3 -> 1.4 -> 0.7 mK/s
```

Kein Sprung, kein Knick: die Schrittdifferenz an der Nahtstelle liegt 1.6 % neben
der gewöhnlichen, und `dT/dt` fällt glatt weiter — eine Relaxation zum neuen
Gleichgewicht, wie sie sein muss, wenn der Treiber ab 1440 s konstant ist.

**StarCCM+ hat den letzten Wert (35 °C) gehalten, und `data.py` tut über
`np.interp` exakt dasselbe.** Die Flachhaltung ist damit keine Annahme über die
Simulation, sondern eine Kopie davon. **O4 ist ersatzlos erledigt.**

> Einschränkung der Methode, der Ehrlichkeit halber: der Test sieht eine Änderung
> nur, wenn sie groß genug ist, um durch die thermische Trägheit bis in `dT/dt`
> durchzuschlagen. Eine winzige Abweichung bliebe unsichtbar — sie wäre dann aber
> auch fürs Training belanglos.

**Nebenbefund, festhalten:** vier Stützpunkte sind eine grobe Rampe. „Profil"
suggeriert mehr Struktur, als `fluid_inlet_temp` auf OP12 hat.

---

## Q2 → **O3 geschlossen**: OP15 hat nie ein Stromprofil gehabt

Im Rohexport von OP15 gibt es **keine** `CellCurrent(t).csv` (OP12 hat eine).
`cell_current` wurde für OP15 nie exportiert — kein Assembly-Fehler, ein
fehlendes Eingangssignal.

Damit ist von den zwei Ursachen aus §9a.1 die zweite bestätigt: **das Plansheet
stimmt für OP15 nicht**, oder OP15 wurde bewusst ohne CC-CV-Auslauf simuliert.
Blockiert nichts — OP15 ist reiner Berichts-OP. Was er testet (ungesehener
Volumenstrom-Profiltyp), testet er weiterhin; was das Blatt zusätzlich verspricht
(CC-CV), ist nicht drin. **Eine Zeile fürs Plansheet, kein Codefix.**

## Q3 — KEIN Fehler. OP14 startet bei 0 °C, weil es so geplant ist

Befund war: OP14 startet über alle 363 Punkte bei ~0 °C und überschreitet erst
nach 123 s die 5 °C. Diagnose des Bots: „Füllwert, verzerrt Normierung",
Empfehlung: die ersten 124 s maskieren, die Initialwerte auf 10 °C setzen, oder
OP14 aus dem Training werfen.

**Das Plansheet sagt etwas anderes** (`op_registry.py:123`):

```python
OPSpec("OP14", "CC", "CH", 2.0, 0.0, 0.0, 0.0, tier=TIER_IN,
       note="coldest start in the set, no flow"),
#                 ^C-Rate ^T0  ^T_fluid ^V̇
```

`T0 = 0 °C`, `T_fluid = 0 °C`, `V̇ = 0`. OP14 **ist** der Kaltstart des
Datensatzes, die kalte Ecke des Envelopes. Die 0 °C sind die Anfangsbedingung,
kein verlorener Offset.

Drei Dinge bestätigen es:

* Ein Füllwert wäre exakt `0.0` oder `NaN`. Gemessen ist `-0.0011 → -0.0018 →
  +0.0156`: ein gelöstes Feld, das um null rauscht und dann steigt.
* 5 K in 123 s sind 0.04 K/s — die adiabate Rate der Quelle liegt bei ~0.03 K/s.
  Passt.
* `T_sigma` ist deshalb nicht „um 8.5 % aufgebläht", sondern **richtig**: der
  Trainingssatz spannt tatsächlich 0 °C bis 40 °C Starttemperatur.

> **Keine der drei Empfehlungen ausführen.** Maskieren, Ersetzen oder Entfernen
> würde einen legitimen Trainings-OP zerstören — und ausgerechnet einen der
> **zwei** mit V̇ = 0, also genau die, die die Energiebilanz binden (§11.1).
> Kein Cache-Neubau.

**Was OP14s hohe MAE stattdessen erklärt:** die zwei No-Flow-OPs sind die zwei
schwersten. OP14 10.866 C und OP07 9.293 C liegen auf Platz 10 und 9 von 11 —
kältester Start, keine Kühlung, größter Hub. Das ist eine kohärente physikalische
Geschichte, kein Datenfehler.

---

## 11. Was die Messung über das MODELL sagt (31.08.)

Zwei Befunde, die aus den Zahlen von Schritt 4/5 folgen und beide den
**Physik-Term** betreffen. Am 31.08. wurden beide nur sichtbar gemacht, keiner
blind repariert. Am 01.09. hat sich der erste (11.1) als **Codefehler mit
eindeutiger Ursache** herausgestellt und ist behoben; der zweite (11.2) bleibt
eine Achse, die gemessen und nicht geraten wird.

### 11.1 Die Energiebilanz ging um Faktor ~147 nicht auf — GEFUNDEN, 01.09.

```
dTdt_scale = 3.534        Qsrc_scale = 0.0241        Verhaeltnis 147x
```

In physikalischen Einheiten: das beheizte Gebiet steigt um **~34 K** über den
Lauf, die Quelle konnte davon **0.23 K** erklären.

Die nichtdimensionale Gleichung ist `dTn/dtn = Fo : ∇²Tn + Qsrc`. Über die Zelle
gemittelt integriert sich der Diffusionsterm zum Randfluss — bei **OP07 und
OP14, die beide Volumenstrom 0 haben**, kann also fast nichts abfließen, und
`<dTn/dtn> ≈ <Qsrc>` muss gelten. Es tat es um zwei Größenordnungen nicht.

**Es war eine doppelt gezählte 121, und sie stand in `data.py`:**

```python
q_dot_full = jr1_full / (V_JR1 * N_JR1_POINTS)     # <- der Fehler
q_dot_full = jr1_full / V_JR1                      # <- richtig, seit 01.09.
```

Die Vermutung vom 31.08. — „eine **fehlende** Volumendivision" — war falsch: die
Division stand da. Zu viel war die zusätzliche durch die **121 JR1-Gitterpunkte**,
übernommen aus `pinn/data/load_op01.py` des Basisprojekts, wo sie mit einer
„Gleichverteilung"-Lesart begründet ist. Diese Lesart zählt doppelt: die
Gesamtleistung gleichmäßig über die 121 Punkte zu verteilen ist **genau das, was
eine uniforme volumetrische Quelle über `V_JR1` schon tut**. Die Einheit von
`jr1_w` musste dafür niemand erfragen — sie steht im Bündelvertrag des
Vorgängerprojekts (`docs/opbundle_contract.md`: `q_source … | W |`), und der
README des Basis-PINN schreibt die richtige Formel sogar aus:
`q̇(t) = heatSourceJr1(t) / V_JR1`.

**Und das beheizte Gebiet ist keine Ausrede.** Am 01.09. von der
Simulationsseite bestätigt: `q_dot = 0` in Zelle und Gehäuse, geheizt wird
ausschließlich JR1. Die Bilanz ist damit **geschlossen** — `jr1_w` gleichmäßig
über `V_JR1`, sonst nirgends, macht genau `jr1_w` im Gebiet. Kein ungezähltes
Gebiet kann eine Abweichung aufnehmen, also ist jedes Verhältnis fern von 1 ein
Befund und keine Modellierungslücke. (Das Basis-README behauptet „JR1 + CC" —
für diesen Datensatz falsch; im Code steht jetzt eine Warnung davor.)

### Auf echten Daten bestätigt, 01.09.

Vorhergesagt war: genau drei Zahlen bewegen sich, und zwar um exakt 121.

| | 31.08. | vorhergesagt | **gemessen 01.09.** |
|---|---|---|---|
| `Qsrc_scale` | 0.0241 | 2.916 | **2.916** |
| `phys_scale` | 3.535 | 4.582 | **4.582** |
| `[ENERGY]` | ~147x | weg | **weg** |
| `dTdt_scale` | 3.534 | unverändert | **3.534** |
| `T_sigma` / `T_span_ref` | 9.616 / 1604 | unverändert | **9.616 / 1604** |
| `A` je Lag | 90.8 / 22.7 | unverändert | **90.8 / 22.7** |
| `bc_pairs` | 242 | unverändert | **242** |
| `MISMATCH` | 1 (OP15) | unverändert | **1 (OP15)** |

**Und ein Beleg, den niemand eingebaut hat.** Die Bilanz je OP folgt dem
Volumenstrom, monoton:

| V̇ | OPs | Verhältnis |
|---|---|---|
| 0 | OP07, OP14 | **0.9x** |
| 0.0013 | OP01–03, OP08, OP10–12 | 0.5–0.7x |
| 0.0026 | OP04, OP05 | 0.5x |

Ohne Kühlung bleiben 90 % der Quelle im Jelly Roll, der Rest leitet ins Gehäuse;
mit Kühlung wird mehr abgeführt, und mit doppeltem Volumenstrom noch mehr. Das
ist die Rangfolge, die eine Energiebilanz haben **muss**, und sie war vorher
nicht da — bei 147x war jedes Verhältnis gleich falsch. Für dieses Verhalten
wurde nichts angepasst; es fällt aus der korrigierten Konstante heraus.

`q_dot` liegt jetzt bei mu = 6.58e4 W/m³ (Bereich -5.24e4 .. 1.53e5). Die
negativen Werte sind kein Fehler: der entropische Anteil der Reaktionswärme ist
in Teilen des SOC-Bereichs endotherm.

**Die Zahl passt:** 147 gemessen gegen 121 aus der Formel. Der Rest ist genau
das, was übrig bleiben muss — was tatsächlich über den Rand abfließt. Nach dem
Fix erklärt die Quelle rund 28 K der ~34 K, also Bilanz auf ~1.2x statt ~147x.
Auf dem synthetischen Bündel geht die `[ENERGY]`-Zeile von 296x auf 2.4x, exakt
Faktor 121.

**Warum das keiner sehen konnte:** ein uniformer Faktor ist unsichtbar. Die
EMA-Balance teilt ihn direkt wieder heraus, `L_phys` landet trotzdem bei O(1),
und `phys_scale` wird aus denselben verfälschten Zahlen gebaut. Nur ein
Energieargument sieht ihn — und genau dafür wurde `energy_balance_report` am
31.08. gebaut. Es hat beim ersten Lauf funktioniert.

**Was das für die vorhandenen Messungen heißt:** das Residuum war
`dTdt = Fo : ∇²Tn` mit einer um 121 kleingerechneten Quelle, also praktisch
**ohne Quelle**. Der Physik-Term sagte dem Netz, die Zelle werde von nichts
geheizt und müsse ihre Temperatur allein durch Leitung erreichen. Das ist eine
andere PDE als die, der die Daten gehorchen, und kein `w_phys` konnte darauf
richtig sein. Deshalb sind `Qsrc_scale`, `phys_scale` und **jede Zahl aus
Schritt 5** — auch das `[SATURATED]` und der `spread` — mit dem falschen
Residuum entstanden und nicht weiterzuverwenden. Schritt 4 und 5b laufen neu.

> **O7 war vor jedem Gewichts-Sweep zu klären** — ein Balance-Sweep auf einem
> falschen Residuum misst, wie schnell man den Physik-Term abschaltet. Seit
> 01.09. ist das erledigt; die Sperre für O8 und den ersten Sweep ist damit weg,
> die Sperre „erst messen" (§10a) nicht.

**Abgesichert:** `test_q_dot_is_the_jr1_power_spread_over_the_jr1_volume` pinnt
die Umrechnung (`q_dot * V_JR1 == jr1_w`, exakt), und
`test_energy_balance_report_flags_a_shrunken_source` pinnt, dass der Bericht
einen uniformen Faktor überhaupt sehen kann — sonst bewacht der erste Test einen
Detektor, der nichts meldet. Vorher gab es zu beidem keinen Test; deshalb war ein
Vorzeichen dieser Größe still.

**Nächster Schritt:** `python3 PINNmodulusTwo/data.py` auf den echten Bündeln.
Die `[ENERGY]`-Zeile muss verschwinden oder nahe 1 stehen; `Qsrc_scale` und
`phys_scale` sind die neuen maßgeblichen Zahlen.

### 11.3 Der Physik-Term dämpft — **Untertrainiertheit. O9 geschlossen, 01.09.**

> ## Die Vermutung war falsch, und das ist das Ergebnis
>
> Nach 5b stand hier der Verdacht, die MAE-Verbesserung durch den Physik-Term sei
> **Varianzreduktion statt Dynamik** — ein Regularisierungs-Artefakt. Zwei
> Modelle sind unabhängig darauf gekommen, und der stärkste Beleg schien der
> `[FLAT]`-Detektor zu sein, der um 0.5 % nicht ausgelöst hatte.
>
> **Schritt 6 sagt: nein.** `spread_time` über 60 Epochen:
>
> | Ep | 1 | 5 | 10 | 20 | 30 | 40 | 50 | **60** |
> |---|---|---|---|---|---|---|---|---|
> | `spread_t` | 6.04 | 0.223 | 0.567 | 0.915 | 0.921 | 0.837 | 0.891 | **0.968** |
>
> Der Rollout kommt aus dem Explodieren (6.04), schießt ins Überdämpfte (0.223 in
> Epoche 5), und **steigt dann monoton auf 0.968** — also praktisch genau auf die
> gesunde 1. Räumlich dasselbe: 9.02 → 1.10. Die 0.201 nach drei Epochen war
> **ein Durchgangswert**, kein Fixpunkt.
>
> Gleichzeitig fällt die MAE weiter (OP06 10.540 → 6.270 C, OP09 7.494 → 3.585 C).
> Beides zusammen — `spread` → 1 **und** MAE fällt — ist genau der Fall, den die
> Signaltabelle als „der Physik-Term trägt wirklich" führt.
>
> **Was daraus zu lernen ist, über den Befund hinaus:** drei Epochen sind kein
> Trend. Der `spread` war in Epoche 5 bei 0.223 und in Epoche 60 bei 0.968 — wer
> nach drei Epochen extrapoliert, liest ein Vorzeichen als Ergebnis. §10a hat
> genau deshalb kein Gewicht auf 5b-Zahlen gesetzt, und das war richtig.
>
> Nicht gebaut wurden dabei: `L_spread`, Gradient-Balancing, adaptives `w_phys`.
> Alle drei wären Umbauten gegen ein Problem gewesen, das es nicht gibt.

**Die Ausgangslage, zur Nachvollziehbarkeit:**

Aus 5b, drei Epochen, ein Seed:

| | mit Physik | ohne (`--w-phys 0 --w-bc 0`) |
|---|---|---|
| val-MAE OP06 / OP09 | **10.54 / 7.49** beats | 11.59 / 8.50 LOSES TO |
| `spread` räumlich / zeitlich | 0.339 / **0.201** | 2.15 / **1.03** |
| `L_data` Epoche 3 | 0.831 | **0.298** |

Drei Dinge stehen da, und sie widersprechen sich.

**(a) Der `[FLAT]`-Detektor hat um 0.5 % nicht ausgelöst.** `train.py` warnt bei
`spread < 0.2`; gemessen sind 0.201. Der Code sagt an dieser Stelle selbst: *„ein
fallendes `L_phys` ist dann die triviale Lösung, nicht Physik."* Das ist der
stärkste Einzelbeleg für den Verdacht — stärker als die MAE-Differenz.

**(b) Die naheliegende Erklärung ist durch den 121er-Fix entfallen.** „Flach
vorhersagen löst `L_phys`" war **vorher** richtig und ist es nicht mehr. Ein
flaches Feld hat `dTdt = 0` und `∇²T = 0`, es bleibt `residual = −Qsrc`:

| | `Qsrc_scale` | `phys_scale` | `L_phys` bei flachem Feld |
|---|---|---|---|
| alt (`/121`) | 0.0241 | 3.535 | **0.00005** — flach war gratis |
| neu | 2.924 | 4.596 | **0.40** — flach kostet O(1) |

Das Overdamping tritt **trotzdem** auf. Der Mechanismus ist also offen, und die
bequeme Antwort ist ausgeschlossen.

**(c) `L_data` und MAE ranken wieder gegeneinander.** Der physikfreie Lauf hat
die **bessere** `L_data` (0.298 gegen 0.831) und die **schlechtere** MAE. §10s
Regel „Kriterium ist MAE, nie `L_data`" ist damit erneut belegt.

**Was NICHT gemacht wird**, obwohl es naheliegt: kein `L_spread`-Term (das ist
Anpassen an die Metrik, mit der man hinterher bewertet), kein Gradient-Balancing
statt EMA, kein adaptives `w_phys`. Drei Epochen und ein Seed tragen keinen
Umbau der Loss-Architektur — genau der Fehler, für den die acht Benchmark-Skripte
gelöscht wurden. Zwei Behauptungen, die dabei im Raum standen, halten für diese
Konfiguration ohnehin nicht: der Physik-Term dominiert nicht (`ratio phys/bc =
0.0402`, also ~4 % des Datenterms), und es gibt keinen Überhang an
Kollokationspunkten (`batch_data 2048 : batch_phys 256` = 8:1 zugunsten der
Daten).

**Entschieden wird es in Schritt 6, ohne eine Zeile Codeänderung:** steigt
`spread` über 60 Epochen Richtung 1, war 0.2 Untertrainiertheit. Bleibt er bei
~0.3 während die MAE fällt, ist es das Artefakt — dann `--w-phys 0.01` gegen
`0.1` messen statt mehr Epochen.

---

### 11.5 Wie gut ist das Ergebnis wirklich? (01.09.)

Die Frage ist berechtigt, weil „alle sechzehn OPs schlagen die trivialen
Vorhersager" mehr klingt, als es ist — und weil einzelne Zahlen wie OP01 mit
**1.000 C** besser klingen, als sie belegen. Beides aufgedröselt.

#### Was in-sample ist und was nicht

Elf der sechzehn sind `T0-in-time`: **Trainings-OPs, auf ihrer eigenen Zeitachse
bewertet.** OP01s 1.000 C und OP10s 1.373 C sagen, dass das Modell seine
Trainingsdaten kann — nicht, dass es verallgemeinert. Als Beleg zählen die
**fünf ausgehaltenen**:

| OP | MAE | Tier | V̇ |
|---|---|---|---|
| OP16 | **3.476 C** | T3-extrap (6.42 σ über dem trainierten Fluss) | 90 l/min |
| OP09 | **3.585 C** | T1/T2, val | 15 l/min |
| OP13 | **4.097 C** | T3-extrap (`c_rate` 4 gegen trainierte 2–3) | 15 l/min |
| OP15 | **4.809 C** | T3-extrap (ungesehener Profiltyp) | Profil |
| OP06 | **6.270 C** | T1-interp, val | **0 l/min** |

#### Der Befund darin: **Volumenstrom ist die Schwierigkeitsachse, nicht der Tier**

| | Mittel-MAE | n |
|---|---|---|
| **V̇ = 0** (OP06, OP07, OP14) | **5.374 C** | 3 |
| V̇ > 0 (alle übrigen) | **2.928 C** | 13 |

Alle drei No-Flow-OPs liegen unter den letzten vier. Und das
**Extrapolations-Tier schlägt den Interpolations-OP**: OP16 und OP13 sind besser
als OP06, obwohl OP06 der einfache Fall sein sollte. Der Grund ist nicht der
Tier, sondern dass OP06 **der einzige ausgehaltene OP ohne Kühlung** ist.

Das ist eine Envelope-Aussage, keine Modellaussage: nur **2 von 11**
Trainings-OPs haben V̇ = 0 (OP07, OP14), und beide sind Extremfälle (T0 = 10 °C
bzw. 0 °C). Das Regime „keine Kühlung bei mittlerer Starttemperatur" kommt im
Training schlicht nicht vor — OP06 ist genau das. Neu als **O14**.

> Nebenbei erklärt das OP14s vermeintlich schlechte 5.656 C, über O10 hinaus:
> kältester Start **und** kein Fluss, in einem Regime mit zwei Beispielen.

#### Die absolute Größenordnung

`T_sigma = 9.602 C`, der beheizte Bereich steigt über den Lauf um ~34 K. Ein
ausgehaltener Fehler von 3.5–4.8 C ist damit rund ein Drittel bis die Hälfte der
Datenstreuung, und er schlägt `persistence` um Faktor 2.5 bis 5.

**Für ein Surrogat, das eine Simulation ersetzen soll, ist das noch nicht genug**
— dort will man ~1 K. Als **erster untunierter Lauf** ist es viel: `w_phys`,
`w_bc`, `delta_phys`, Breite und Tiefe stehen alle auf ihren `config.yaml`-
Defaults, es ist nie ein Gewicht gesetzt worden (O6), und `delta_phys` ist
**nachweislich 4.1x zu grob** (O8, `[CFL WARN]` bei jedem Lauf). Es gibt also
Luft, die noch niemand angefasst hat, und mindestens eine davon ist bekannt
falsch.

#### Was die Zahl nicht hergeben kann

* **Ein Seed.** Und die 5b-Erfahrung sitzt: derselbe Lauf, der „der Physik-Term
  trägt" sagte, sagte auch „der `spread` kollabiert" — und das war falsch. Der
  `--w-phys 0`-Vergleich ist **nie über 60 Epochen** gelaufen. Bis er es ist,
  steht die Überschrift dieser Sitzung auf drei Epochen. Siehe die Achsen in
  Teil I.
* **O13:** der Fehler wächst zum Trajektorienende (OP06 6.270 C im Mittel,
  13.248 C spät). Bei einem Ladezyklus ist das Ende der Teil, der zählt.
* **O11:** der Messvergleich wird schlechter, nicht besser. §11.4.

#### Unterm Strich

Fünf ausgehaltene OPs, darunter drei im Extrapolations-Tier, schlagen beide
trivialen Vorhersager deutlich — nach **einem** Lauf ohne jede Abstimmung, in
einem Projekt, das bis zum 31.08. nie einen davon geschlagen hatte. Das ist ein
gutes Ergebnis. Es ist der Punkt, an dem das Modell auf dem Platz steht, nicht
der, an dem es gewinnt.

---

### 11.6 Was die `history.csv` sagt — O12 widerlegt, O15 neu (02.09.)

Bis zum 02.09. stand jede Aussage über den Lauf auf der **letzten** Zeile von
`artifacts/history.csv`. Die Datei hat 60 Zeilen. Ausgewertet wurden sie zum
ersten Mal am 02.09., und zwei der Aussagen halten das nicht aus.

> Damit das nicht wieder von der Sorgfalt eines einzelnen Nachmittags abhängt,
> macht `tools/analyse_history.py` es jetzt von selbst. Es liest dieselbe Datei
> und meldet die vier Muster, die unten hergeleitet werden.

#### O12 ist widerlegt: 0.0178 war eine Epoche, kein Trend

| `ratio_bc` | Wert |
|---|---|
| letzte Epoche (die Zahl, die bis 02.09. im Fahrplan stand) | 0.0178 |
| **Median Ep31–60** | **0.0581** |
| Streuung Ep31–60 | Std 0.0332 auf Mittel 0.0564 — **59 % relativ** |
| Median Ep21–40 → Ep41–60 | 0.0540 → 0.0582 — **kein Abfall** |
| Epochen unter dem Endwert | 5 von 30 |

Der BC-Term liegt stabil bei ~0.058 gegen den nominellen `w_bc/w_data = 0.1`,
also bei 58 % seines Sollwerts. Er ist nicht „auf 0.0178 gefallen"; die letzte
Epoche war zufällig eine der fünf niedrigsten von dreißig.

Die zweite Hälfte von O12 löst sich im Code auf: `boundary_condition_loss` zieht
`n_samples = min(len(bc_indices), len(tn_q))`, und `min(121, 128)` ist dasselbe
wie `min(121, 121)`. `--batch-bc 121` lässt die Warnung verschwinden und ändert
an der Mathematik nichts. **O12 ist damit geschlossen, ohne einen einzigen Lauf.**

> **Das ist dasselbe Muster ein zweites Mal.** 5b las einen Trend aus drei
> Epochen (`spread` = 0.201) — Schritt 6 hat ihn widerlegt. Hier wurde einer aus
> **einer** Epoche gelesen. Regel ab sofort: **kein Befund aus der letzten
> Zeile.** Median und Streuung über die letzten dreißig Epochen, oder gar nichts.

#### O15: das Balancing hat nach 60 Epochen noch nicht eingesetzt

Alle drei Divisoren fallen mit exakt derselben Rate:

| | Start | Ende | Faktor | pro Epoche |
|---|---|---|---|---|
| `div_data` | 2.94e4 | 58.7 | 500.1 | **0.9000** |
| `div_phys` | 7.34e9 | 1.48e7 | 495.9 | **0.9002** |
| `div_bc` | 4.80e-4 | 9.59e-7 | 500.5 | **0.9000** |

`0.9^59 = 1/501`. Das ist reiner geometrischer Zerfall: der aktuelle Loss trägt
zum Divisor nichts bei. Nachgerechnet an Epoche 60 — der EMA-Schritt ist
`div ← 0.9·div + 0.1·L`, also `0.9 × 65.27 = 58.74` gegen einen Beitrag von
`0.1 × 0.0515 = 0.005`, das sind **0.009 %**.

Der Grund steht in `train.py:462`: beim ersten Aufruf ist `prev is None`, also
`ema = value` — der Divisor wird auf den Loss des **allerersten
Optimiererschritts** verankert. Der lag bei ~3.3e4, das Epochenmittel von
Epoche 1 bei 100.5, also rund **325× darüber**. Bei einem Horizont von zehn
Epochen (`ema_decay: 0.9`) braucht allein der Abbau dieser Verankerung
`10·ln(325) ≈ 58` Epochen. Genau so lange lief der Lauf.

**Die Folge:** weil alle drei gleich schnell zerfallen, ist ihr *Verhältnis*
über den ganzen Lauf konstant. Das Balancing ist damit nicht adaptiv, sondern
faktisch `fixed` — mit Divisoren, die der Zufall des ersten Schritts gesetzt hat.
Gemessen kommt `ratio_phys` = **0.586** heraus (Median Ep31–60), wo `w_phys = 0.1`
nominell 0.1 verspricht: ein Faktor 6, den niemand gewählt hat.

Unter Adam (`train.py:591`) kürzt sich ein gemeinsamer Skalenfaktor weitgehend
heraus — deshalb hat Schritt 6 trotzdem sauber trainiert. Was **nicht**
weggeht, ist das eingefrorene Verhältnis, und genau das ist die Größe, die ein
Gewichts-Sweep messen soll. **Solange O15 gilt, misst ein `w_phys`-Sweep den
ersten Optimiererschritt mit.** Deshalb blockiert O15 den Punkt O6.

#### Warum kein Test das gefangen hat

`test_balanced_loss_is_the_raw_loss_over_the_recorded_divisor` prüft, dass
`div_phys` derselbe Divisor ist, der `L_phys_bal` erzeugt hat — die Buchhaltung
also stimmt. Sein Docstring nennt den Zweck ausdrücklich: *„which is the exact
question 'is the divisor stale or did the term really fall?' needs answered."*
Die Instrumentierung wurde also gebaut, damit die Frage beantwortbar ist.
**Gestellt hat sie danach niemand.**

Der fehlende Test ist nicht schwer: nach N Epochen eines Kurzlaufs muss `div_*`
innerhalb eines Faktors von `L_*` liegen. Wäre er da gewesen, hätte er O15 am
ersten Tag gemeldet statt nach sechzig Epochen. Er gehört zum ersten Schritt,
der O15 anfasst.

`--w-phys 0 --w-bc 0` ist davon nicht betroffen: bei `zero_weight_terms: skip`
werden beide Terme gar nicht berechnet (`train.py:708`). **Achse 0 bleibt
gültig und bleibt der erste Schritt.**

#### Was die Datei bestätigt

`spread_time`: Median Ep31–60 = **0.922**, Std 0.119, letzte Epoche 0.968.
`spread_space`: Median **1.118**. Die Überschrift „der `spread` ist gesund, O9
ist widerlegt" steht damit auf dreißig Epochen und nicht auf einer — sie hält.

Auch die Konsistenz stimmt: `delta` steht konstant bei 0.000622975 in normierter
Zeit, also `T_span_ref = 1605.2 s` — genau das Trajektorienende von OP12 aus O4.

#### Nebenbefund: das Training ist nicht glatt

In den Epochen 4, 14, 25, 36 und 38 springt `L_phys` um ein bis zwei
Größenordnungen (Ep 25: 1.26e7 gegen einen Median von 8.3e4), `spread_time`
schießt auf 6.5 bzw. 4.2 und fällt zurück. Für `sweep.py` heißt das: **die
Auswertung mittelt über die letzten k Epochen**, sie liest nicht die letzte. Ein
Lauf, der zufällig auf einer solchen Epoche endet, liefert sonst Unsinn — und
zwar Unsinn, der aussieht wie ein Ergebnis.

---

### 11.7 O13 ist mechanisch, nicht eine Frage der Verallgemeinerung (02.09.)

`split_t = 0.8 · n_t` (`config.yaml:72`), `late_mae` ist also schlicht der
mittlere Fehler über das letzte Fünftel der Trajektorie (`op_metrics.py:123`).

Der entscheidende Punkt stand am 01.09. schon in den Zahlen, ist aber nicht
ausgesprochen worden: **OP03 ist ein Trainings-OP, und `--holdout-tail` steht
auf `False`** (`train.py:307`). Der Datenverlust deckt die ganze Trajektorie ab —
das Modell hat diese späten Zeitschritte **mit Beschriftung gesehen** und
verdoppelt dort trotzdem seinen Fehler (3.341 → 7.371 C). Damit ist O13 kein
Verallgemeinerungsproblem. Das Modell scheitert an Punkten, die es auswendig
kennen könnte.

Drei Mechanismen kommen in Frage:

1. **Der Level-Integrator.** `field` sagt nicht absolut vorher, sondern
   `T(t) = level(t − delta_grid) + Netz(...)`, wobei `level` der räumliche
   Mittelwert der **eigenen vorherigen Vorhersage** ist (`model.py:450–484`).
   Über ~7000 Rollout-Schritte ist das ein Integrator mit Verstärkung 1: eine
   noch so kleine systematische Schieflage wird bei jedem Schritt aufaddiert, und
   es gibt keine rückstellende Kraft. Der Docstring nennt diese Konstruktion
   ausdrücklich das, was das Weglaufen verhindert — sie verhindert Weglaufen der
   *Form*, ist aber zugleich der perfekte Sammler für einen *Bias*.
2. **Es ist am Ende wirklich schwerer.** Ende des Ladezyklus heißt heißeste
   Phase, größte Gradienten, bei Profil-OPs der CV-Übergang.
3. **Die Normierung kennt das letzte Fünftel nicht.** `T_mu`, `T_sigma`,
   `q_mu/sigma`, `Qsrc_scale`, `phys_scale` werden über `[:split_t]` gepoolt
   (`data.py:674–679`, `753–756`). Das ist richtig so — Testdaten dürfen nicht in
   die Statistik lecken — bedeutet aber, dass die späten, heißesten Zustände am
   oberen Rand der kalibrierten Skala liegen.

**Eine Messung trennt 1 von 2 und 3, und sie braucht kein neues Training:** ist
der späte Fehler **vorzeichenbehaftet** (immer dieselbe Richtung → Drift,
Mechanismus 1) oder **ungerichtet** (Mechanismus 2/3)? Heute lässt sich das nicht
sagen, weil `op_metrics.py` bis zum 02.09. nur mit `np.abs` gerechnet hat — Drift
und Rauschen sahen identisch aus.

**Seit 02.09. ist die Messung gebaut.** `op_metrics` liefert zusätzlich
`late_bias` (signiertes Mittel über dasselbe Fenster wie `late_mae`),
`late_bias_frac = |late_bias| / late_mae` und die drei Drittel
`bias_early / bias_mid / bias_end`. Zu lesen ist es so:

| `late_bias_frac` | Vorzeichen der Drittel | heißt |
|---|---|---|
| nahe **1** | gleich, wachsend | **Mechanismus 1**: der Level-Integrator läuft weg. Dann ist der Hebel der Anker, nicht das Gewicht |
| nahe **0** | wechselnd | **Mechanismus 2 oder 3**: das späte Regime ist wirklich schwerer. Dann helfen Daten oder Normierung, kein Anker |
| dazwischen | — | beides, und die Aufteilung sagt zu welchen Teilen |

Es braucht dafür **keinen eigenen Lauf** — die Zahlen fallen bei Achse 0 mit ab.

> **Was dafür fehlt, ist weniger als es aussieht.** `save_checkpoint` schreibt
> `model.pt` samt `model_config` (`train.py:1140`), und
> `test_checkpoint_round_trips_without_config_yaml` lädt es bereits zurück:
> `RecurrentField(**ckpt["model_config"])` plus `load_state_dict(..., strict=True)`
> ist getestet und funktioniert. Was **fehlt, ist nur der Einstiegspunkt** — kein
> `--resume`, kein Auswertungs-Kommando. Ein `evaluate.py` (Checkpoint laden,
> Rollout rechnen, Metriken berichten) ist damit wenige Zeilen und macht aus zwei
> Stunden Minuten. Zu beachten: **jeder Lauf überschreibt `artifacts/model.pt`.**

---

### 11.4 OP19 wird schlechter, je besser das Modell wird (O11, 01.09.)

| | 5b (3 Epochen) | Schritt 6 (60 Epochen) |
|---|---|---|
| val OP06 / OP09 | 10.540 / 7.494 C | **6.270 / 3.585 C** |
| test OP13 / 15 / 16 | 8.686 / 7.239 / 4.204 C | **4.097 / 4.809 / 3.476 C** |
| **OP19 (Messung)** | 5.507 C | **10.334 C** ← 88 % schlechter |

Jeder Simulations-OP verbessert sich, der Messvergleich verschlechtert sich fast
um das Doppelte. Das ist kein Widerspruch, sondern die Definition von
Extrapolation: OP19 liegt 16.7 σ unter dem trainierten `c_rate`-Bereich (es ist
eine Entladung, OP01–OP16 sind alle Ladungen), 12.2 σ unter `cell_current`, fährt
`soc_start` = 77 % gegen trainierte 10 % — über einen **toten** Kanal, den das
Netz nicht sehen kann (O5) — und läuft mit `tn` bis 2.18 aus dem normierten
Zeitbereich heraus.

Ein untrainiertes Netz sagt dort etwas Beliebiges; ein gut trainiertes sagt
**mit Überzeugung** das, was in seinem Envelope richtig wäre. Die zweite Sorte
Fehler ist größer.

**Die Konsequenz für die Bewertung:** die Simulations-MAE sagt über den Prüfstand
nichts, und der Messvergleich darf nie als Auswahlkriterium dienen — er tut es
laut §1 auch nicht. Zu klären, bevor OP19 je als Erfolgsmaß gilt: erst den
Envelope erweitern (eine Entladung ins Training, `soc_start` beleben), dann
messen. `max = 96.760 C` auf OP19 zeigt außerdem, dass es dort Stellen gibt, an
denen die Vorhersage nicht nur ungenau, sondern unphysikalisch ist.

---

### 11.2 Der Physik-Stencil ist 4x zu grob — und wurde nie geprüft

`Fo` erreicht im **Aluminiumgehäuse** ~200 (Jelly Roll: ~0.1), die schnellste
Diffusions-Zeitskala liegt damit bei `Δt_max ≈ 0.24 s`. Der BDF2-Stencil in
`physics.py` nutzt aber `δ = 1.0 s` — bis zum 31.08. **hartverdrahtet**, kein
Flag, kein config-Schlüssel.

Schlimmer: die CFL-Prüfung bekam nur den **Datenschritt** übergeben, nicht `δ`.
Und `δ` folgt `--subsample` nicht — es bleibt bei seinem Wert, wie fein das
Gitter auch wird. Ein Lauf bei `subsample 2` druckte also `CFL OK` (dt = 0.2 s
gegen 0.24 s, allerdings nur 1.2x Reserve), während die Zeitableitung im
Residuum über eine 4x zu lange Spanne gebildet wurde.

Seit dem 31.08.: `--delta-phys` ist ein Knopf (Default unverändert 1.0, damit
sich nichts still ändert), und die CFL-Prüfung sieht beide Schritte an und warnt
getrennt.

Im **hybriden** History-Modus — dem Default — speist `δ` **ausschließlich**
`L_phys`; die History des Netzes hängt an `--delta-grid` und `--rate-lags`. Das
macht `δ` zu einer sauber isolierten Achse und damit zum **besten ersten
Sweep-Kandidaten**, sobald 11.1 geklärt ist.

**11.1 ist seit 01.09. geklärt, die Achse ist also frei** — und sie ist jetzt
erst sinnvoll: solange `Qsrc` 121x zu klein war, hätte ein δ-Sweep gemessen, wie
gut ein zu grober Stencil eine Gleichung ohne Quelle auflöst. Der Default bleibt
bei 1.0, weil er zu **messen** ist und nicht zu setzen (§10a); der erste Punkt
neben 1.0 ist 0.2, das Datengitter selbst.

---

## 9. Offene Befunde (31.08.)

### 9.3 Schritt 5 hat die Frage nicht beantwortet — 5b hat es getan (01.09.)

> **Beantwortet.** Der weglaufende Rollout war **dt**. Bei subsample 2 ist
> `[SATURATED]` in Epoche 3 weg (Ep1 99.9 % auf OP05, Ep2 88 % auf OP04, Ep3
> nichts). Grund (a) und (b) unten fallen bei dt = 0.2 s weg, wie erwartet;
> Grund (c) — der eigentliche Befund — löst sich mit ihnen auf.
>
> Zwei Nachträge, beide gemessen:
>
> * **`A` hängt hier NICHT nennenswert am subsample.** Bei dt = 0.2 s: 90.5/22.6
>   gegen 90.8/22.7 bei dt = 4 s. Die Warnung in Schritt 4 („die aus Schritt 4 ist
>   die Größenordnung, nicht der Wert") war für diesen Datensatz gegenstandslos.
> * **Der `spread` schießt jetzt über das andere Ziel hinaus:** 9.02/6.04 in
>   Epoche 1, 0.339/0.201 in Epoche 3. Er ist nicht bei 1 angekommen, sondern
>   daran vorbei ins Überdämpfte. Ohne Physik-Term bleibt er bei 2.15/1.03, also
>   näher an 1. Das ist eine offene Spannung: auf MAE gewinnt der Physik-Lauf,
>   auf `spread` der andere. Neuer Punkt **O9**.

Der Lauf sagt auf beiden val-OPs `LOSES TO`. **Diese Zahl zaehlt nicht**, aus

Der Lauf sagt auf beiden val-OPs `LOSES TO`. **Diese Zahl zaehlt nicht**, aus
drei Gruenden, und der dritte ist der eigentliche Befund.

**(a) `--subsample 40` verletzt die CFL-Grenze um Faktor 16.6.**

```
[CFL WARN] Δt=4.000s, Δt_max≈0.241s -> POTENTIALLY UNSTABLE
```

Bei dt = 4 s ist die Zeitableitung im Physik-Residuum Unsinn, also ist auch ihr
Gradient Rauschen. Das Training laeuft mit `subsample_time: 2`, dt = 0.2 s, und
damit innerhalb der Grenze. **Der Schritt-5-Lauf trainiert also eine andere
Konfiguration als der, den er freigeben soll.** Das ist ein Fehler in diesem
Fahrplan gewesen, nicht im Code.

**(b) `--delta-grid 0.2s` ist kleiner als der Datenschritt 4 s.**

```
[WARN] the anchor cannot resolve finer than the grid and will effectively act as 4s
```

Der Anker der hybriden History degeneriert. Faellt bei subsample 2 ebenfalls weg.

**(c) Der Rollout laeuft weg — und das faellt nicht automatisch weg.**

```
[SATURATED] epoch 1: OP05 365/369 steps   (98.9 %)
[SATURATED] epoch 2: OP12 376/402 steps   (93.5 %)
```

Fast die ganze Trajektorie haengt im Clamp. `train.py` sagt dazu selbst: *„it is
not a prediction, and a run that only survives because of this is not trained."*
Der Befund ist nicht ein schlechter OP, sondern das Modell: die Saettigung
**wandert** von OP05 nach OP12.

Bestaetigt von `spread s/t = 4.92/4.19`: der Rollout streut fuenfmal so weit wie
die Labels. Das ist das Gegenteil des `[FLAT]`-Falls — kein kollabiertes, sondern
ein explodierendes Feld.

**Nebenbefund:** bei 2 Epochen und einem EMA-Horizont von 10 Epochen sind die
Loss-Divisoren praktisch die aus Epoche 1 eingefrorenen —
`L_data/L_data_bal = 29696`, gesetzt vom saturierten Rollout der ersten Epoche.
Die Balance hat in diesem Lauf nie gearbeitet. Auch das verschwindet erst bei
laengeren Laeufen.

**Was der Lauf trotzdem gezeigt hat, und das ist viel:** die Pipeline laeuft von
Ende zu Ende auf echten Daten durch — 11 Trainings-OPs, 2 val, 3 test, OP19 als
Messvergleich, alle Metriken, alle Baselines, Checkpoint geschrieben. Und
`L_data` faellt von 94 auf 16, das Modell lernt also durchaus etwas.

> **Deshalb steht jetzt ein Schritt 5b vor Schritt 6.** Sechzig Epochen auf einem
> Rollout, der zu 99 % im Clamp haengt, ranken das Clamp-Verhalten und nicht das
> Modell — das sind verlorene Stunden.

### 9.4 OP19: die Latte ist dort fast unschlagbar

`persistence = 1.375 C`. Die Trajektorie bewegt sich kaum, „das Feld aendert sich
nie" ist also schon fast richtig. Dazu kommt: `tn` laeuft bis 2.18 (3496 s gegen
`T_span_ref` 1604 s), `c_rate` geht auf **-3.42** (Entladung, nie trainiert), und
`cell_current` wird negativ. `LOSES TO` ist dort erwartbar und sagt bis auf
Weiteres nichts ueber das Modell.

---

## 9a. Offene Befunde aus Schritt 4 (31.08.)

Beide kommen aus den Daten, nicht aus dem Code, und beide sind **nicht** durch
einen Codefix zu erledigen.

### 9.1 OP15: `cell_current` fehlt im Bündel

```
OP15 [held out] detected=fluid_inlet_temp,fluid_mass_flow
                sheet=cell_current,fluid_inlet_temp,fluid_mass_flow   <-- MISMATCH
```

Das Plansheet nennt OP15 „CC mit Fluidtemperaturprofil und Volumenstromprofil
**und CC-CV**". Im Bündel variiert `cell_current` nicht — der CC-CV-Auslauf ist
nicht drin.

**Blockiert nichts.** OP15 ist ein reiner Berichts-OP (`test_ops`), kein
Trainings- und kein Auswahl-OP. Was verloren geht, ist die Aussagekraft **dieses
einen** Berichts: OP15 sollte den ungesehenen Volumenstrom-Profiltyp testen, und
das tut er weiterhin — nur eben ohne den CC-CV-Anteil, den das Blatt verspricht.

> **GESCHLOSSEN 01.09.** Der Rohexport enthält keine `CellCurrent(t).csv` für
> OP15 (OP12 hat eine). Von den zwei Ursachen unten ist es die zweite: **nie
> markiert**, weil nie exportiert. Das Plansheet stimmt für OP15 nicht, oder
> OP15 wurde bewusst ohne CC-CV simuliert. Eine Zeile fürs Plansheet.

**Nächster Schritt:** `python3 PINNmodulusTwo/data.py` erneut laufen lassen. Seit
dem 31.08. druckt der Bericht bei einem MISMATCH zusätzlich, was die
Upstream-Assembly für dieses Bündel als Profil *markiert* hat, und das trennt die
beiden möglichen Ursachen:

* **markiert, aber konstant** → die Profildatei fehlte oder war leer, der Kanal
  ist still auf seinen Skalar zurückgefallen. Rohexport von OP15 prüfen, OP15 neu
  bauen.
* **nie markiert** → das Blatt stimmt für OP15 nicht, oder OP15 wurde ohne dieses
  Profil exportiert.

### 9.2 Profile enden vor der Trajektorie — auch auf einem **Trainings**-OP

```
OP12 [train   ] ! fluid_inlet_temp covers 0.0..1440.0 s but the OP runs 0.1..1604.1 s
OP15 [held out] ! fluid_inlet_temp und fluid_mass_flow, dasselbe
```

Die letzten ~164 s (rund **10 %**) werden mit dem letzten Profilwert flach
gehalten. Auf OP15 ist das ein Berichtsproblem; **auf OP12 ist es
Trainingsdaten**: das Modell lernt dort 10 % lang einen Treiber, der so nie
simuliert wurde, und die Temperatur, die es dazu sehen soll, gehört zu einem
Treiber, den es nicht sieht.

Zu klären ist, ob der Simulationslauf wirklich länger war als das Profil (dann
ist das Bündel richtig und die Flachhaltung die einzig mögliche Annahme), oder
ob der Profilexport abgeschnitten wurde (dann ist er nachzuliefern).

**Am 01.09. geschlossen, und die Antwort ist „so lassen".** Der Rohexport hat vier
Stützpunkte (0, 480, 960, 1440 s), das Profil endet dort wirklich. Die
Flachhaltung ist damit die einzig mögliche Annahme — und mit hoher
Wahrscheinlichkeit dieselbe, die StarCCM+ selbst getroffen hat, denn eine
Tabellen-Randbedingung hält voreingestellt den letzten Wert. **Am selben Tag aus
den Daten bestätigt:** an der Nahtstelle 1440 s gibt es keinen Knick in `dT/dt`
(Schrittdifferenz 0.000185 °C gegen gewöhnliche 0.000182 °C), die Ableitung fällt
glatt weiter (2.3 → 1.4 → 0.7 mK/s). Der Solver hat gehalten.

**Und der Test, der hier vorgeschlagen war, ist gelaufen:** oben stand, man
erkenne das Problem daran, dass OP12 später auffällig schlechter sei als die
anderen Trainings-OPs. In 5b-1 hat OP12 die **beste** MAE aller elf (3.942 C).
Der Befund zeigt sich nicht. Abschneiden der Trajektorie bei 1440 s würde 10 %
echter Simulationsdaten wegwerfen, um eine Annahme zu vermeiden, die messbar
nicht schadet — das wäre der schlechtere Tausch.

---

## 0b. Was sich am 02.09. geändert hat

**Kein Lauf, keine Code-Änderung.** Diese Sitzung hat die offenen Punkte
durchgesprochen und die vorhandenen Daten zum ersten Mal vollständig ausgewertet.
Das hat mehr geändert als erwartet.

| | |
|---|---|
| **O12 widerlegt** | Die `history.csv` hat 60 Zeilen; die Aussage stand auf der letzten. Median Ep31–60 = 0.0581, kein Abfall, 59 % relative Streuung. Dazu: `--batch-bc 121` rechnet dieselbe Zeile wie 128. Beide Hälften erledigt, §11.6 |
| **O15 neu** | Alle drei Loss-Divisoren fallen mit exakt 0.9000/Epoche — reiner Zerfall, der Loss trägt 0.009 % bei. Das Balancing wirkt wie `fixed` mit Divisoren aus dem ersten Optimiererschritt. **Blockiert O6**, §11.6 |
| **O13 geschärft** | OP03 ist ein Trainings-OP und `--holdout-tail` ist aus — der späte Fehler tritt auf Daten auf, die das Modell mit Beschriftung gesehen hat. O13 ist mechanisch, nicht eine Frage der Verallgemeinerung, §11.7 |
| **O8 entschieden** | Wird gemessen, drei Punkte 1.0 / 0.4 / 0.2. Nicht einfach auf 0.2 gesetzt, weil der Zähler mit δ schrumpft und der Rollout-Eigenfehler nicht — ein Zielkonflikt, kein Einheitenfehler |
| **O5 umgewidmet** | Kein MAE-Hebel: eine Spalte ohne Varianz kann nichts erklären, und die SOC-Wirkung steckt über `q_dot` schon im Modell. Bleibt als Envelope-Grenze stehen, kein Kanal wird gestrichen |
| **O11 und O14 sind dauerhaft** | Der Datensatz ist fix — es kommen keine OPs dazu, OP17/OP18 sind nie simuliert worden. Beide Punkte werden nicht mehr „gelöst", sondern berichtet |
| **O6 neu bewertet** | Die fünf Gründe aus §10a sind alle behoben. O6 ist nicht mehr bewusst offen, sondern erstmals messbar — nur nicht vor O15 |

### Die eine Lehre, und sie ist die zweite ihrer Art

5b las einen Trend aus drei Epochen, O12 aus **einer**. Beide Male sah eine Zahl
nach Befund aus, weil niemand die Streuung daneben gestellt hat. Beim `spread`
ist die Aussage nachträglich doch bestätigt worden (Median 0.922 über dreißig
Epochen), beim BC-Term nicht.

**Regel:** kein Befund aus der letzten Zeile. Median und Streuung über die
letzten dreißig Epochen, oder gar nichts. Für `sweep.py` heißt das, dass die
Auswertung über die letzten *k* Epochen mittelt — die Epochen 4, 14, 25, 36 und
38 zeigen Ausschläge von ein bis zwei Größenordnungen in `L_phys`, und ein Lauf,
der zufällig auf einer davon endet, liefert Unsinn, der aussieht wie ein Ergebnis.

---

## 0a. Was sich am 01.09. geändert hat

**Ein Fehler, gefunden und behoben; zwei Berichte, die ihn künftig sehen.**

| | |
|---|---|
| **Quelle war 121x zu klein** | `data._read_raw` teilte `jr1_w` durch `V_JR1 * N_JR1_POINTS`. Die 121 JR1-Gitterpunkte waren doppelt gezählt. Jetzt `jr1_w / V_JR1`. §11.1 |
| **Zwei Tests dazu** | die Umrechnung selbst (`q_dot * V_JR1 == jr1_w`, exakt) und die Empfindlichkeit des Energieberichts. Zu beidem gab es vorher keinen Test — deshalb konnte ein Faktor 121 still sein |
| **`q_dot = 0` außerhalb JR1 bestätigt** | von der Simulationsseite, 01.09. Zelle und Gehäuse bekommen nichts. Damit ist die Bilanz geschlossen: `jr1_w` gleichmäßig über `V_JR1`, sonst nirgends — genau `jr1_w` geht ins Gebiet. Im Code festgehalten, samt Warnung, es **nicht** am Basis-README auszurichten (das sagt „JR1 + CC" und ist für diesen Datensatz falsch) |
| **`coverage_report` meldet tote Kanäle** | er übersprang sie. Ein Kanal ohne Trainings-Varianz ist der Fall, in dem ein abweichender Wert am wenigsten interpolierbar und zugleich unsichtbar ist (`_normalise_config` zwingt ihn auf 0). O5 |

**Was das für die Zahlen vom 31.08. heißt:** `Qsrc_scale`, `phys_scale` und
sämtliche Schritt-5-Ergebnisse sind mit einem praktisch abgeschalteten
Physik-Term entstanden. Sie sind in der Stand-Tabelle als ungültig markiert und
nicht zu vergleichen. Schritt 3 (die Bündel) ist unberührt.

**Nicht geändert:** kein Gewicht, kein Default, keine Loss-Balance. `delta_phys`
steht weiter auf 1.0. Der 121er war ein Einheitenfehler mit einer eindeutigen
richtigen Antwort; alles andere in §11 bleibt eine Achse, die gemessen wird.

---

## 0. Was sich am 31.08. geändert hat

Zwei Dinge, beide Vereinfachungen.

**Die acht Benchmark-Skripte sind gelöscht** — `smallBench.py`,
`bench_common.py`, `benchmark_balance.py`, `benchmark_arch.py`,
`benchmark_wphys_wbc.py` und in der Erweiterung `smokeBench.py`,
`profileBench.py`, `bench_profiles.py`. Zusammen 4735 Zeilen. Der Grund ist
nicht, dass sie falsch waren, sondern dass **kein einziges ihrer Ergebnisse auf
echten Daten gemessen war**. Ein Sweep über Konfigurationen, die alle noch nie
einen trivialen Vorhersager geschlagen haben, ist eine Rangfolge zwischen
Verlierern.

**Und es gibt nur noch ein Projekt.** `PINNmodulusTwoExtProfiles/` ist in
`PINNmodulusTwo/` aufgegangen. Die Trennung „konstante Treiber hier, Profile
dort" war nie eine echte Grenze: die Profil-Pipeline ist eine **echte
Obermenge** — ein konstanter Treiber ist ein Profil, das sich nicht bewegt.
Trainiert wird ab jetzt auf dem ganzen Plansheet, OP01–OP16, konstante Treiber
und Profile gemeinsam. `--resample point --no-driver-history` stellt die alte
Vorverarbeitung exakt wieder her, falls ein Vergleich sie je braucht.

Was `train.py` dadurch selbst kann, ohne dass ein Benchmark existieren muss:

| vorher | jetzt |
|---|---|
| `smallBench.py` druckte „the bar to beat" | `train.py` druckt persistence + Trainings-Mittel neben **jeder** OP-Zeile |
| `bench_common` baute Val-/Test-OPs | `--val-ops` / `--test-ops`, gleiche Normierung (`data.build_op` re-fittet nichts) |
| `profileBench` berichtete je Tier | `op_metrics` + `op_registry.tier_of` in jeder Zeile |
| `smokeBench` prüfte Plansheet und Abdeckung | `profile_report()` und `coverage_report()` laufen in jedem Lauf mit |
| `smallBench` warnte vor synthetischen Daten | Banner beim Start (`data.cache_is_synthetic`) |
| nur `bench_common` konnte `torch.save` | `train.py` schreibt `artifacts/model.pt` |

---

## Sitzungsende 01.09. — Lagebericht

**Schritt 1 bis 6 sind durch. Fünf ausgehaltene OPs — darunter drei im
Extrapolations-Tier — schlagen beide trivialen Vorhersager.** Das ist die Zahl,
auf die dieses Projekt seit Monaten gewartet hat, und sie kam nach **einem** Lauf
ohne jede Abstimmung.

**Wie gut das wirklich ist, steht in §11.5** — inklusive der Unterscheidung
in-sample gegen ausgehalten (elf der sechzehn OPs sind Trainings-OPs auf ihrer
eigenen Zeitachse) und des Befundes, dass der **Volumenstrom** die
Schwierigkeitsachse ist und nicht der Tier.

## Der Weg dahin, an einem Tag

| | |
|---|---|
| **Der Fehler** | `q_dot` war um Faktor 121 zu klein — die „Gleichverteilung" wurde doppelt gezählt. Der Physik-Term war praktisch quellenfrei. §11.1 |
| **Schritt 4** | `Qsrc_scale` 0.0241 → **2.916**, `phys_scale` 3.535 → **4.582**, beide exakt wie vorhergesagt. `[ENERGY]` weg, Bilanz folgt monoton dem Volumenstrom |
| **Schritt 5b** | `[SATURATED]` weg, beide val-OPs schlagen erstmals — **nur mit** Physik-Term |
| **Schritt 6** | `spread` 0.201 → **0.968**, val-MAE 6.270 / 3.585 C (42 % / 54 % besser als `train-mean`), alle Test-OPs beats |

## Was widerlegt wurde, und das gehört dazu

Nach 5b stand hier die Vermutung, die MAE-Verbesserung durch den Physik-Term sei
**Varianzreduktion statt Dynamik** — ein Regularisierungs-Artefakt. Zwei Modelle
kamen unabhängig darauf. **Schritt 6 hat es widerlegt:** `spread` steigt monoton
von 0.223 (Epoche 5) auf 0.968 (Epoche 60), während die MAE weiter fällt.

Die 0.201 nach drei Epochen war ein Durchgangswert. **Drei Epochen sind kein
Trend** — und genau deshalb war es richtig, darauf kein Gewicht zu setzen und
weder `L_spread` noch Gradient-Balancing noch adaptives `w_phys` einzubauen. Alle
drei wären Umbauten gegen ein Problem gewesen, das es nicht gibt.

## Offene Punkte

> Diese Tabelle ist der Stand vom **01.09.** Der aktuelle Stand steht in Teil I
> („Index: alle Punkte, O1 bis O15") — am 02.09. sind O12 geschlossen, O15 neu,
> und O5/O11/O14 haben ihren Charakter geändert. Siehe **§0b**.

| # | | wer / wann |
|---|---|---|
| **O14** | Volumenstrom ist die Schwierigkeitsachse: V̇=0 im Mittel 5.374 C gegen 2.928 C. Nur 2 von 11 Trainings-OPs haben V̇=0, beide Temperatur-Extremfälle. §11.5 | Envelope-Frage |
| **O11** | OP19 wird schlechter, je besser das Modell wird (5.507 → 10.334 C). Extrapolation, kein Fehler — aber der Messvergleich taugt bis zur Envelope-Erweiterung nicht als Maß. §11.4 | vor jedem Messvergleich |
| **O12** | Der BC-Term ist auf `ratio 0.0178` gefallen. Tut `w_bc` überhaupt etwas? | Sweep-Achse 2 |
| **O13** | Der Fehler sitzt am Ende der Trajektorie: OP06 MAE 6.270 C, `late` 13.248 C | zu untersuchen |
| **—** | **Achse 0: `--w-phys 0 --w-bc 0` über 60 Epochen.** Die Kernaussage dieser Sitzung steht auf drei Epochen, und derselbe Lauf lag beim `spread` daneben. Ein Lauf, braucht `sweep.py` nicht | **zuerst** |
| **O8** | δ = 1.0 s gegen Δt_max 0.24 s — der einzige bekannt falsche Parameter | **Sweep-Achse 1** |
| **O6** | kein Gewicht auf Basis von Messungen gesetzt | Sweep-Achse 3 |
| **O5** | tote Kanäle — `soc_start` konstant, Rate-Kanäle tot | zu entscheiden |
| **O10** | Warnung: OP14s 0 °C sind geplant, nicht kaputt | nichts tun |

**Geschlossen am 01.09.:** O1, O2, O3, O4, O7, **O9**. Schritte 4, 5, 5b, 6.

## Die offene Kernfrage, im Klartext

**Trägt der Physik-Term, senkt er also die MAE — oder nicht?**

Gemessen ist bisher nur dies:

| | mit Physik | ohne (`--w-phys 0 --w-bc 0`) |
|---|---|---|
| **3 Epochen** (5b) | 10.540 / 7.494 C | 11.591 / 8.504 C |
| **60 Epochen** (Schritt 6) | **6.270 / 3.585 C** | **nie gelaufen** |

Bei drei Epochen war die Physik ~10 % besser. Aber derselbe Lauf sagte auch, der
`spread` kollabiere — und Schritt 6 hat das widerlegt (§11.3). Eine Aussage aus
diesem Lauf ist gefallen; die andere ist ungeprüft.

**Die Antwort braucht genau einen Lauf**, und er steht ganz oben in dieser Datei.
Bis er gelaufen ist, gilt: *bei drei Epochen ja, bei Konvergenz unbekannt.*

## Was der Code an diesem Tag gelernt hat

* `q_dot = jr1_w / V_JR1`, und `q_dot = 0` außerhalb JR1 — im Code gegen ein
  „Zurückreparieren" am Basis-README abgesichert.
* Drei neue Tests: die Umrechnung, die Empfindlichkeit des Energieberichts, tote
  Kanäle im Coverage-Report.
* `coverage_report` meldet tote Skalar-Kanäle — hat sofort OP19s `soc_start`
  gefunden.

**Kein Gewicht, kein Default, keine Loss-Balance geändert. Kein Cache-Neubau.**
Der 121er war ein Einheitenfehler mit einer eindeutigen richtigen Antwort; das
ist die einzige Sorte Änderung, die ohne Messung zulässig war.

**Ausgeführt:** `pytest` 123 passed / 1 skipped · `selftest.py` · `op_registry.py`
· Importcheck · Schritte 4, 5b und 6 auf echten Daten auf der Arbeitsmaschine.
