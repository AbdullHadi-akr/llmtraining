# Fahrplan — ein Projekt: OP01–OP16 trainiert, OP19 als Messvergleich

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du nur
eine Datei liest, dann diese.

> ### Lebendes Dokument
>
> Diese Datei wird nach **jedem** Ergebnis fortgeschrieben — sie ist kein Plan,
> den man einmal schreibt und dann abarbeitet. Die Regeln dafür:
>
> * **Haken setzen, sobald ein Schritt durch ist** (`- [ ]` → `- [x]`), und die
>   gemessene Zahl in die Tabelle „Stand" darunter eintragen. Ein Haken ohne
>   Zahl ist wertlos: „gelaufen" und „das Kriterium erfüllt" sind zwei Dinge.
> * **Gemessene Zahlen ersetzen Vermutungen im Text.** Wo hier „unbekannt" oder
>   „ungemessen" steht und eine Messung vorliegt, wird der Satz umgeschrieben,
>   nicht ergänzt.
> * **Ein rotes Tor ändert den Plan, nicht nur den Haken.** Wenn Schritt 5
>   `LOSES TO` sagt, ist die Reihenfolge darunter falsch und wird neu geschrieben
>   — nicht abgehakt und ignoriert.
> * **Nichts hier ist gemessen, solange kein Haken steht.** Jede Zahl in diesem
>   Repo stammt bis heute aus einem synthetischen Bündel.

---

# Offene Punkte für die nächste Session

Zuerst lesen. Was hier steht, ist das, was beim Abbruch der letzten Sitzung
offen war — nicht neu abzuleiten.

| # | offen | wer |
|---|---|---|
| **O1** | **Es liegen Schritt-6-Ergebnisse bis Epoche 30 vor, die noch niemand ausgewertet hat.** `artifacts/metrics.txt` und `artifacts/history.csv` einschicken. **Achtung: auch dieser Lauf hatte die 121x zu kleine Quelle (O7).** Er sagt weiter etwas über den Rollout und die Baselines, aber nichts über `w_phys` oder den Physik-Term | du schickst, ich werte |
| **O2** | **ERLEDIGT am 01.09.** — 5b gelaufen und grün, siehe Stand-Tabelle. ~~Schritt 5b ist nie gelaufen — und ist seit O7 **nicht mehr hinfällig zu machen**: 5b-2 (`--w-phys 0 --w-bc 0`) trennt jetzt zum ersten Mal zwei Läufe, die sich im Physik-Term wirklich unterscheiden. Vorher waren beide praktisch quellenfrei~~ | erledigt |
| **O3** | **GESCHLOSSEN am 01.09.** Der Rohexport von OP15 enthält **keine** `CellCurrent(t).csv` — das Signal wurde nie exportiert. Das Plansheet stimmt für OP15 nicht (oder OP15 wurde bewusst ohne CC-CV simuliert). Eine Zeile fürs Plansheet, kein Codefix, blockiert nichts | erledigt |
| **O4** | **ABGESTUFT am 01.09.** OP12s Profil hat 4 Stützpunkte bis 1440 s, Trajektorie bis 1605 s; die letzten 10 % werden flach gehalten — **das ist der Ist-Zustand und bleibt so**. OP12 hat die **beste** MAE aller Trainings-OPs (3.942 C), das Problem zeigt sich also nicht. Schmale Restfrage: was hat StarCCM+ nach 1440 s benutzt? Antwort „letzten Wert gehalten" erledigt es ersatzlos | Restfrage, blockiert nichts |
| **O10** | **KEIN offener Punkt, sondern eine Warnung.** OP14 startet über alle Punkte bei 0 °C. Das sieht nach Füllwert aus, ist aber die geplante Anfangsbedingung (`op_registry.py:123`, T0 = 0 °C, „coldest start in the set"). **Nicht maskieren, nicht ersetzen, OP14 nicht entfernen** — es ist einer von nur zwei OPs mit V̇ = 0, und die binden die Energiebilanz | nichts tun |
| **O5** | **Tote Eingangskanäle.** `soc_start` ist über alle OPs konstant 10 % (`DEAD -> forced to 0`), und die Rate-Kanäle von `c_rate` und `fluid_mass_flow` sind im Training tot — werden aber auf OP15/OP16/OP19 lebendig. Das Modell soll dort einen Kanal deuten, den es nie gesehen hat. **Am 01.09. sichtbar gemacht:** `coverage_report` übersprang tote Skalar-Kanäle bisher **komplett** — ausgerechnet den Fall, in dem der Umschlag ein einziger Punkt ist. Jetzt bekommt jeder tote Kanal, den ein Berichts-OP bewegt, eine eigene Zeile. Die **Entscheidung** (Kanal streichen? Envelope erweitern?) bleibt offen | zu entscheiden, siehe §10 |
| **O6** | Nichts an **Gewichten** ist auf Basis der Messungen geändert worden — bewusst, siehe §10a | offen bis 5b/6 |
| **O7** | **ERLEDIGT und auf ECHTEN DATEN bestätigt (01.09.).** `[ENERGY]` ist weg, Bilanz 0.9x auf den bindenden No-Flow-OPs. Die Umrechnung teilte `jr1_w` durch `V_JR1 * N_JR1_POINTS`; die 121 Gitterpunkte waren doppelt gezählt, jedes `Qsrc` war 121x zu klein. Behoben, mit Test. **Folge: `Qsrc_scale`, `phys_scale` und jede Zahl aus Schritt 5 sind ungültig — Schritt 4 und 5b müssen neu laufen.** §11.1 | erledigt |
| **O9** | **NEU 01.09. — die offene Frage, die Schritt 6 entscheidet.** Mit Physik-Term `spread s/t` = 0.339/**0.201**, ohne = 2.15/1.03. Die `[FLAT]`-Warnung löst bei 0.2 aus: es fehlten **0.5 %**. Verdacht: die MAE-Verbesserung ist Varianzreduktion, keine bessere Dynamik. §11.3 hat die Analyse | **Schritt 6 beantwortet es** |
| **O8** | Der BDF-Stencil in `L_phys` nutzt δ = 1.0 s gegen eine Diffusions-Zeitskala von ~0.24 s. Ein Knopf (`--delta-phys`) und mit `[CFL WARN]` versehen. **Seit O7 erledigt ist, ist diese Achse frei** — sie war ausdrücklich hinter 11.1 eingereiht. Default bleibt 1.0, bis gemessen ist. §11.2 | erste Sweep-Achse |

---

# Rohexport-Untersuchung Q1–Q3 — beantwortet am 01.09.

Der lokale Bot hat in die Rohdaten geschaut. **Zwei Fragen sind geschlossen,
eine war ein Fehlalarm — und dessen Empfehlung hätte Schaden angerichtet.**

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

## Q2 → **O3 geschlossen**: OP15 hat nie ein Stromprofil gehabt

Im Rohexport von OP15 gibt es **keine** `CellCurrent(t).csv` (OP12 hat eine).
`cell_current` wurde für OP15 nie exportiert — kein Assembly-Fehler, ein
fehlendes Eingangssignal.

Damit ist von den zwei Ursachen aus §9a.1 die zweite bestätigt: **das Plansheet
stimmt für OP15 nicht**, oder OP15 wurde bewusst ohne CC-CV-Auslauf simuliert.
Blockiert nichts — OP15 ist reiner Berichts-OP. Was er testet (ungesehener
Volumenstrom-Profiltyp), testet er weiterhin; was das Blatt zusätzlich verspricht
(CC-CV), ist nicht drin. **Eine Zeile fürs Plansheet, kein Codefix.**

## Q1 → **O4 abgestuft**: nichts tun ist hier die richtige Antwort

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

Was als schmale Rückfrage bleibt: *welchen Einlasstemperaturverlauf hat StarCCM+
nach 1440 s tatsächlich verwendet?* Lautet die Antwort „letzten Wert gehalten",
ist O4 ersatzlos erledigt.

**Nebenbefund, festhalten:** vier Stützpunkte sind eine grobe Rampe. „Profil"
suggeriert mehr Struktur, als `fluid_inlet_temp` auf OP12 hat.

---

# JETZT: genau eine Sache — Schritt 6

Schritt 1 bis 5b sind durch. **Es steht nur noch ein Kommando aus:**

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate
git pull origin claude/fahrplan-issues-nx1nqp     # falls noch nicht geschehen

python3 PINNmodulusTwo/train.py --epochs 60 2>&1 | tee 06_lauf.txt
```

CPU misst ~110 s/Epoche → ~2 h. Auf GPU deutlich weniger; `--device` fragt nach.

## Worauf du in `06_lauf.txt` schaust — in dieser Reihenfolge

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

## Was du danach schickst

```
06_lauf.txt   +   PINNmodulusTwo/artifacts/metrics.txt
              +   PINNmodulusTwo/artifacts/history.csv    <- wegen spread je Epoche
```

`history.csv` ist diesmal die wichtigere Datei: die `spread`-Spalten über 60
Epochen sind das, was 5b nicht beantworten konnte.

---

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
- [ ] **6** Erster ernsthafter Lauf (GPU) ← **freigegeben durch 5b**

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
| 6 | `[SATURATED]` letzte Epoche | | |
| 6 | MAE OP06 / OP09 | | |
| 6 | MAE OP13 / OP15 / OP16 | | |
| 6 | MAE OP19 (Messvergleich) | | |

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

### - [ ] Schritt 6 — Der erste ernsthafte Lauf (Stunden, jetzt GPU)

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

---

## Was du mir danach schickst

Vier Dateien, alle klein:

```
03_cache.txt      (nur falls Schritt 3 gehakt hat)
04_daten.txt      <- am wichtigsten: MISMATCH, bc_scale, A
05_latte.txt      <- die baseline-Zeilen
06_lauf.txt       + PINNmodulusTwo/artifacts/metrics.txt
```

Wenn du früher stoppen musst: die Datei des Schrittes, an dem es hakt, reicht.
Sag dazu, **bei welchem Schritt** du bist — dann weiß ich, wo wir stehen, ohne
zu raten.

Danach entscheidet sich, was zuerst gebaut wird: die Seed-Schleife (§3 Phase 4),
eine Achse, oder — falls `LOSES TO` bleibt — etwas ganz anderes als ein
Benchmark.

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

## 3. Die Phasen

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

### Phase 2 — Der Maßstab (Minuten) ← **hier fehlt bisher alles**

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

**Am 01.09. geklärt, und die Antwort ist „so lassen".** Der Rohexport hat vier
Stützpunkte (0, 480, 960, 1440 s), das Profil endet dort wirklich. Die
Flachhaltung ist damit die einzig mögliche Annahme — und mit hoher
Wahrscheinlichkeit dieselbe, die StarCCM+ selbst getroffen hat, denn eine
Tabellen-Randbedingung hält voreingestellt den letzten Wert.

**Und der Test, der hier vorgeschlagen war, ist gelaufen:** oben stand, man
erkenne das Problem daran, dass OP12 später auffällig schlechter sei als die
anderen Trainings-OPs. In 5b-1 hat OP12 die **beste** MAE aller elf (3.942 C).
Der Befund zeigt sich nicht. Abschneiden der Trajektorie bei 1440 s würde 10 %
echter Simulationsdaten wegwerfen, um eine Annahme zu vermeiden, die messbar
nicht schadet — das wäre der schlechtere Tausch.

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

### 11.3 Der Physik-Term dämpft — Artefakt oder Untertrainiertheit? (O9, 01.09.)

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

# Wo du stehst — in drei Sätzen

**Schritt 1–5b sind durch, Schritt 6 ist freigegeben und noch nicht gelaufen.**
Der Physik-Term hatte eine 121x zu kleine Quelle; das ist behoben und auf echten
Daten bestätigt. Seitdem schlagen beide val-OPs zum ersten Mal die trivialen
Vorhersager — aber nur mit Physik-Term, und der Rollout ist dabei verdächtig
stark gedämpft (O9).

**Dein nächstes Kommando steht ganz oben unter „JETZT".** Es ist eines.

## Was am 01.09. gemessen wurde

| | |
|---|---|
| **Schritt 4** | `[ENERGY]` weg (war ~147x). `Qsrc_scale` 0.0241 → **2.916**, `phys_scale` 3.535 → **4.582**, beide exakt wie vorhergesagt. Alles andere unverändert. Die Bilanz folgt monoton dem Volumenstrom: V̇=0 → 0.9x, V̇=0.0026 → 0.5x |
| **Schritt 5b** | `[SATURATED]` in Epoche 3 **weg** — der weglaufende Rollout aus §9.3 war `dt`. val OP06 **10.540 C beats**, OP09 **7.494 C beats**; ohne Physik-Term verlieren beide |
| **offen daraus** | `spread` = 0.339/**0.201** mit Physik gegen 2.15/1.03 ohne. `[FLAT]` löst bei 0.2 aus. §11.3, O9 |

## Offene Punkte, sortiert

| | | wer |
|---|---|---|
| **O9** | dämpft der Physik-Term nur, statt Dynamik zu lernen? | **Schritt 6 beantwortet es** |
| **O8** | δ = 1.0 s gegen Δt_max 0.24 s. Seit O7 frei, erste Sweep-Achse | nach Schritt 6 |
| **O5** | tote Kanäle. OP19 fährt `soc_start` 77 % gegen trainierte 10 %, unsichtbar fürs Netz | zu entscheiden |
| **O6** | kein Gewicht auf Basis von Messungen gesetzt — weiterhin bewusst | nach Schritt 6 |
| **O4** | OP12-Profil endet bei 1440 s — **so lassen**, nur schmale Restfrage an die Simulationsseite | blockiert nichts |
| **O10** | **Warnung, kein Punkt:** OP14s 0 °C sind geplant, nicht kaputt. Nicht „reparieren" | nichts tun |

**Erledigt in dieser Sitzung:** O1, O2, O3, O7 — Schritt 4, 5, 5b abgehakt —
und die Rohexport-Fragen Q1/Q2/Q3 beantwortet.

**Kein Cache-Neubau nötig.** Weder OP12 noch OP14 werden angefasst.

## Was danach gebaut wird

Die Bedingung aus §10 („einmal 5b oder 6 auswerten") ist erfüllt. Nach Schritt 6
kommt `sweep.py`, die Seed-Schleife, ~80 Zeilen. Vorher nicht: ein Seed und drei
Epochen tragen keine Rangfolge zwischen Konfigurationen, und genau deshalb ist in
dieser Sitzung **kein Gewicht, kein Default und keine Loss-Balance** geändert
worden. Der 121er war ein Einheitenfehler mit einer eindeutigen richtigen
Antwort — das ist die einzige Sorte Änderung, die ohne Messung zulässig war.

## Was der Code in dieser Sitzung gelernt hat

* `q_dot = jr1_w / V_JR1`, und `q_dot = 0` außerhalb JR1 (von der
  Simulationsseite bestätigt, im Code gegen ein „Zurückreparieren" am
  Basis-README abgesichert).
* Drei neue Tests: die Umrechnung selbst, die Empfindlichkeit des
  Energieberichts, tote Kanäle im Coverage-Report.
* `coverage_report` meldet jetzt tote Skalar-Kanäle — hat auf echten Daten
  sofort OP19s `soc_start` gefunden.

**Ausgeführt am 01.09.:** `pytest` 123 passed / 1 skipped · `selftest.py` all
checks passed · `op_registry.py` · Importcheck · Schritt 4 und 5b auf echten
Daten auf der Arbeitsmaschine.
