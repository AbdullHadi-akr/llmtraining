# GridCNN — Implementierungsplan

> **Lies zuerst [`README.md`](README.md).** Dort steht *warum*; hier steht *was,
> in welcher Reihenfolge, und woran es scheitert*.
>
> Heißt `FAHRPLAN.md` nach der Konvention des Repos — `PINNmodulusTwo` hat
> seinen eigenen, und die `.gitignore` führt genau diesen Namen auf ihrer
> Whitelist.

Der Plan ist eine **Leiter mit Toren**, keine gerade Linie. Grund steht in
README §11: die meisten Gewinne dieses Entwurfs sind *Gitter*-Gewinne, nicht
*CNN*-Gewinne. Also wird das Billigste und Sicherste zuerst gebaut, und jede
Stufe hat ein Tor, das die nächste absagen darf.

**Ein rotes Tor ändert den Plan, nicht nur den Haken.** Wenn Stufe 0 „fünf
Moden" sagt, wird das ROM gebaut und nicht der CNN — auch wenn der CNN im
Entwurf steht.

---

## ▶ Das Nächste

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
git checkout claude/cnn-problem-discussion-24ct9g && git pull
source modulus_env/bin/activate

python3 GridCNN/tools/spatial_rank.py   2>&1 | tee 07_rang.txt
python3 GridCNN/tools/balance_check.py  2>&1 | tee 08_bilanz.txt
```

Zusammen Sekunden bis Minuten, kein GPU, kein Torch, kein pandas. Danach wissen
wir, wie groß `f` sein muss — und ob der Wandterm überhaupt trägt.

> Beim ersten Lauf auf einer neuen Maschine zuerst
> `balance_check.py --ops OP04 --list-columns`: die Monitornamen stammen aus
> einem StarCCM+-Export und müssen nicht überall gleich heißen.

---

## Die Leiter

| Stufe | was | Dauer | Tor |
|---|---|---|---|
| **0** | Rangtest | Sekunden | wie groß muss `f` sein — oder reicht ein ROM? |
| **1** | Bilanz-Gegenprobe auf den Rohdaten ✔ gebaut | Minuten | geht die Wärmebilanz auf? |
| **2** | vier Größen in den Cache | 30 min | Reports weiter grün? |
| **3** | **Physik ohne Netz** — der Nullmodell-Lauf | Stunden Bauzeit | schlägt reine Physik die trivialen Vorhersager? |
| **4** | das Netz dazu, Ein-Schritt-Training | Tage | schlägt es Stufe 3? |
| **5** | truncated BPTT | Tage | fällt der Spätfehler (O13)? |
| **6** | Vergleich gegen PINNmodulusTwo | 1 Lauf | derselbe Split, dieselben Metriken |

---

# Stufe 0 — Der Rangtest

**Gebaut:** [`tools/spatial_rank.py`](tools/spatial_rank.py) ✔ (getestet gegen
ein nachgebautes Bündel, braucht nur numpy)

**Ausführen:** siehe „Das Nächste".

### Das Tor

Die **gepoolte Ortsstruktur bei 99.9 %** ist die Zahl:

| Ergebnis | Folge für den Plan |
|---|---|
| **≤ ~5 Moden** | 🔴 **Umbau.** Der Raum ist trivial. Statt Stufe 4–5 wird ein ROM gebaut: POD-Projektion auf ~5 Moden, GRU auf den Koeffizienten, dieselben Treiber. Stufen 1–3 bleiben **unverändert gültig** — sie sind Physik, nicht Architektur |
| ~6–20 Moden | 🟡 `f` klein halten: 16–24 Kanäle, 3 Blöcke |
| **~30+ Moden** | 🟢 wie geplant: 64 Kanäle, 4 Blöcke |

Zusätzlich abzulesen, ohne eigenes Tor: wächst das **Wandgefälle** mit `V̇`?
Wenn nein, ist die Annahme hinter `ghost_hi` falsch und Stufe 1 wird kritisch
statt bestätigend.

---

# Stufe 1 — Geht die Bilanz auf?

**Gebaut:** [`tools/balance_check.py`](tools/balance_check.py) ✔ (getestet gegen
einen nachgebauten Rohdatensatz). Liest die Roh-CSVs direkt — **kein
Cache-Umbau nötig** — mit `csv` und `cp1252`, derselben Konvention wie die
legacy-Assembly. Nur numpy.

Vier Prüfungen:

**1. Der Halbmodell-Faktor, aus den Daten statt aus einer Nachfrage.**
`*_Heat Source.csv` führt `jr1_w`, `jr2_w` **und** `total_w` nebeneinander. Ist
`total/jr1 ≈ 2` und `jr2/jr1 ≈ 1`, hat die Zelle zwei Wickel, das Halbmodell
(x = 0 … 0.0219) enthält genau einen, und `jr1_w` **ist** die
Halbmodell-Leistung. Damit ist die Frage beantwortet, ohne jemanden zu fragen.

**2. Die Fluidbilanz.**

```
ΔT_fluid = Q̇ / (ṁ · Cp_fluid)      gegen      Tmfavg_fluid_out − T_fluid_in
```

> ⚠ **Korrektur an Dokument 030.** Dort steht `∫Q̇dt / (ṁ·Cp)`. Das ist
> dimensionell **K·s, nicht K** — die Integralform gilt für ein *geschlossenes*
> Fluidvolumen, das sich aufheizt, nicht für einen Durchfluss. Ein Test gegen
> die falsche Formel würde fehlschlagen, ohne dass an der Physik etwas falsch
> wäre. Das Werkzeug rechnet die Durchflussform.

**3. Der Energieanteil über die Wand.** `∫Q̇dt` gegen `∫jr1_w dt` — das
Gegenstück zu den 0.5–0.9× aus `energy_balance_report`. Bei `ṁ = 0` (OP07,
OP14) muss der Anteil **nahe 0** liegen; tut er es nicht, fließt Wärme auf einem
Weg ab, den der Entwurf nicht kennt.

**4. `h_eff` gegen den Volumenstrom.** Braucht die Wandtemperatur, also den
`.npz`-Cache; ohne ihn wird der Teil übersprungen statt zu scheitern.

### Das Tor

| Ergebnis | Folge |
|---|---|
| Fluidbilanz-Verhältnis ≈ 1.0 | 🟢 `ghost_hi` steht |
| `h_eff` gegen `V̇` auf einer Kurve | 🟢 `h` wird **feste Funktion**, null freie Parameter |
| `h_eff` streut breit | 🟡 `h` wird gelernt, aber mit `L_wall` gegen `Q̇` beaufsichtigt |
| Verhältnis ≈ 2.0 oder ≈ 0.5 | 🔴 **halt.** Halbmodell-Faktor — Prüfung 1 sagt, auf welcher Seite. Kein Physikfehler, und nicht als solcher zu behandeln |
| bei `ṁ = 0` fließt Energie ab | 🔴 der Entwurf hat einen Pfad übersehen |

> Der Faktor-2-Fall ist kein hypothetisches Risiko: dieses Projekt hat schon
> einmal einen Faktor 121 an genau dieser Sorte Buchhaltung verloren
> (FAHRPLAN §11.1). Deshalb steht Prüfung 1 vor allen anderen.

**Nicht mehr blockiert:** `data_raw/` liegt auf der Maschine (02.09.).

# Stufe 2 — Die vier Größen in den Cache

**Zu ändern** (in `PINNmodulusTwo/`, nicht hier — geteilte Infrastruktur):

| Datei | was |
|---|---|
| `generate_cache.py` / die legacy-Assembly | vier Spalten mitschreiben: `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`, `mdot` |
| `docs/opbundle_contract.md` | Vertrag erweitern, `schema_version` hoch |
| `data.py` | die vier laden, alte Bündel ohne sie sauber ablehnen statt still auf 0 zu setzen |

Dann alle sechzehn OPs neu bauen (10–30 min).

### Das Tor

`profile_report`, `coverage_report` und `energy_balance_report` müssen **exakt
dieselben Zahlen** liefern wie vorher — die vier Größen kommen dazu, sie ändern
nichts. Eine Abweichung heißt, der Rebuild hat etwas anderes mitgeändert.

Zusätzlich neu: `energy_balance_report` kann den Fehlbetrag jetzt **beziffern**
statt ihn nur zu zeigen. Die 0.5–0.9× sollten durch `∫Q̇dt` erklärt sein.

**Entscheidung offen:** README §12.3 — Schema erweitern (empfohlen) oder
GridCNN-lokal lesen.

---

# Stufe 3 — Physik ohne Netz  ← die wichtigste Stufe

Der Kern des Plans, und der Teil, den ich beim Schreiben des Entwurfs
unterschätzt hatte.

Gebaut wird der **ganze Gitter-Unterbau ohne ein einziges gelerntes Gewicht**:

```
grid.py      Reshape 363 -> (3,11,11) abgeleitet aus xyz, plus die drei Paddings
physics.py   FD-Stencil (nicht-aequidistant in x, Kreuzterm fuer lam_XY),
             Quelle, Wandterm
solve.py     T_{t+1} = T_t + dt * (Fo : grad2 T + Qsrc)   -- explizites Euler
```

Das ist ein klassischer expliziter Löser auf dem 3×11×11-Gitter. Er wird über
alle sechzehn OPs gerollt und mit **denselben `op_metrics`** bewertet wie alles
andere.

### Warum das so viel wert ist

* **Es prüft Padding, Stencil und Wandterm unabhängig vom Lernen.** Ist der
  Kreuzterm falsch oder das Padding verdreht, sieht man es hier — und nicht
  drei Wochen später als „das Netz konvergiert schlecht".
* **Es liefert eine Physik-Latte.** Der Fahrplan vergleicht heute gegen
  `persistence` und `train-mean`. Beide sind trivial. „Reine Wärmeleitung mit
  kalibrierter Kühlwand" ist eine *ernsthafte* Latte, und ein Surrogat, das sie
  nicht schlägt, hat nichts gelernt, was die Physik nicht schon weiß.
* **Es beantwortet die CFL-Frage empirisch.** Läuft der Löser bei
  `subsample_time: 2` (dt = 0.2 s gegen Δt_max 0.241 s) stabil oder nicht? Das
  ist keine Schätzung mehr.
* **Es ist der Rest-Definitionspunkt.** Was der Löser *nicht* trifft, ist genau
  das, was `f` lernen muss. Damit ist die Aufgabe des Netzes definiert statt
  geraten.

### Das Tor

| Ergebnis | Folge |
|---|---|
| schlägt `train-mean` auf den ausgehaltenen OPs | 🟢 der Unterbau stimmt, weiter zu Stufe 4 |
| stabil, aber schlechter als `train-mean` | 🟡 normal — der Löser kennt die Materialdaten nur genähert. Weiter, aber der Rest ist groß |
| divergiert | 🔴 entweder CFL (dann `subsample_time: 1`) oder ein Vorzeichenfehler im Padding. **Nicht mit dem Netz übertünchen** |
| schlägt schon `PINNmodulusTwo` (6.270 / 3.585 C) | 🔴🟢 dann ist der interessante Befund, dass das Lernen bisher gegen fehlende Physik anlief — und der Plan wird ein anderer |

---

# Stufe 4 — Das Netz dazu

```
model.py     Conv-Stapel, Delta-Form:  T_{t+1} = T_t + dt * f(...)
             f = Loeser aus Stufe 3  +  Conv-Korrektur
train.py     Trainingsschleife, zunaechst Ein-Schritt wie PINNmodulusTwo
```

**`f` korrigiert den Löser, es ersetzt ihn nicht.** Damit startet das Modell
bei der Physik statt bei Rauschen, und der Rollout erbt die Dissipation des
Stencils — genau der Grund, warum die Δ-Form hier tragen sollte, wo
`residual_output` in `PINNmodulusTwo` wegläuft (README §4).

**Größe** nach dem Tor von Stufe 0. Verluste:

```
L = w_data * L_data  +  w_phys * L_phys  +  w_wall * L_wall
```

`L_bc` gibt es **nicht** — die Randbedingungen sind Padding (README §5). Das ist
eine Sweep-Achse weniger als im Fahrplan.

### Das Tor

Schlägt Stufe 3 auf den fünf ausgehaltenen OPs. Wenn nicht, hat das Netz
nichts beigetragen und die Frage ist, ob `L_data` überhaupt greift.

---

# Stufe 5 — Truncated BPTT

Der Hebel auf **O13** (Fehler wächst zum Trajektorienende: OP06 6.270 C im
Mittel, 13.248 C spät). Fenster von `k` Schritten, Gradient durch alle `k`,
`detach` am Fensterende.

`k` = 50 zuerst, dann 200. Aktivierungsspeicher ist grob 31k Floats je Schritt
(README §4), also ist 200 machbar.

### Das Tor

**`late_mae` fällt**, bei mindestens gleichem `mae`. Fällt nur `mae` und
`late_mae` nicht, hat BPTT nicht getan, wofür es da ist.

---

# Stufe 6 — Der Vergleich

Derselbe Split (train 11 / val OP06+OP09 / test OP13, OP15, OP16), dieselben
`op_metrics`, dieselben trivialen Vorhersager, **plus** die Physik-Latte aus
Stufe 3. Über mehrere Seeds — die Lehre aus Fahrplan Teil I gilt hier genauso:
*ein Seed ist keine Streuung.*

---

## Was übernommen wird — importiert, nicht kopiert

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PINNmodulusTwo"))
import data, op_registry, op_metrics
```

`data.py` trägt die teuer erkauften Korrekturen (die 121×-Quelle, die
Energiebilanz, die gepoolte Normierung, das Anti-Aliasing beim
Treiber-Resampling). Eine Kopie driftet weg — genau deshalb wurden am 31.08.
die beiden Vorgängerprojekte zusammengelegt. Der Fehler wird nicht wiederholt.

**Kein Modulus.** GridCNN braucht `FCLayer` nicht; reines PyTorch. Läuft
trotzdem in `modulus_env`, weil Torch dort schon liegt.

---

## Was NICHT gebaut wird

Damit der Umfang nicht wandert:

* **ConvGRU / ConvLSTM.** Erst wenn Stufe 5 steht. Ein unbeschränkter
  versteckter Zustand bei elf Trajektorien ist ein echtes Risiko.
* **FiLM-Konditionierung.** Die Treiber werden erst gebroadcastet. FiLM ist eine
  eigene Achse, keine Architekturentscheidung.
* **FNO / DeepONet / Neural ODE.** Bei 11×11 Overkill.
* **Plots, Resume, Checkpoint-Merge.** Zuletzt, und nur für Achsen, die wirklich
  Stunden laufen.

---

## Abbruchkriterien

* **Divergenz in Stufe 3 wird nicht mit dem Netz übertünchert.** Ein Löser, der
  wegläuft, hat einen Fehler im Padding oder in der CFL — beides ist zu finden,
  nicht zu überdecken.
* **Kein `L_bc` durch die Hintertür.** Braucht das Modell einen Strafterm für
  die Symmetrie, ist das Padding falsch.
* **`Q̇` und `T_fluid_out` gehen nie als Modelleingang rein** (README §6). Sie
  sind Aufsicht und Gegenprobe. Eine val-MAE, die mit ihnen als Eingang
  entsteht, ist wertlos.
* **Ein Seed ist keine Streuung.** Kein Vergleich zweier Konfigurationen ohne
  Seed-Schleife.

---

## Stand

Wird beim Abhaken ausgefüllt. Leer = noch nicht gemessen.

| Stufe | Kriterium | gemessen | Datum |
|---|---|---|---|
| 0 | Gitterprobe 3×11×11 | | |
| 0 | gepoolte Ortsstruktur @ 99.9 % | | |
| 0 | `uniform`-Anteil je OP | | |
| 0 | Wandgefälle wächst mit V̇ | | |
| 1 | Halbmodell-Faktor `total/jr1` | | |
| 1 | Fluidbilanz, Verhältnis | | |
| 1 | Wandanteil bei V̇ = 0 (muss ~0 sein) | | |
| 1 | `h_eff(V̇)` auf einer Kurve? | | |
| 2 | Reports unverändert | | |
| 3 | Löser stabil bei dt = 0.2 s | | |
| 3 | Physik-Latte, val OP06 / OP09 | | |
| 4 | Netz schlägt Physik-Latte | | |
| 5 | `late_mae` gefallen | | |
| 6 | val / test gegen PINNmodulusTwo | | |
