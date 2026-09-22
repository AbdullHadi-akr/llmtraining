# Konfiguration A, erster echter Lauf — 22.09.2026

> **Wenn du nur eine Zeile liest:** Der Lauf ist kein Ergebnis. Die
> Seed-Streuung ist **22 °C** bei einer Lesbarkeitsschwelle von **~1 °C**, und
> der beste Seed ist nicht besser als „immer den Trainingsmittelwert raten".
> Das ist **keine Latte** und darf nicht als eine zitiert werden.

**Lauf:** `14_konfigA.txt` · **Commit:** `c61514c` · Tesla T4, torch 2.6.0+cu124
**Kommando:**

```bash
python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --device cuda --cache data_cache
```

11 427 Parameter, 11 Trainings-OPs, 2 Halte-OPs, `--subsample 2` → dt = 0.2 s,
rund **8040 Zeitschritte** je OP.

---

## 1. Was herauskam

| | Seed 0 | Seed 1 | Seed 2 | Mittel ± σ |
|---|---|---|---|---|
| **val-MAE OP06** | 51.95 | 20.43 | **8.95** | **27.11 ± 22.26 °C** |
| **val-MAE OP09** | 51.45 | 15.48 | **8.03** | **24.99 ± 23.22 °C** |
| `data` bei ep 60 | 0.332 | 0.254 | 0.305 | — |
| Epochen mit `[SATURATED]` | **26/60** | 18/60 | 24/60 | |
| größter `data`-Ausschlag | 2500.7 | 2432.4 | 417.6 | |

### Die drei Befunde, die zählen

**1 — Die Streuung ist das Ergebnis, nicht der Mittelwert.**
Bester gegen schlechtester Seed: **Faktor 5.8** (OP06) bzw. **6.4** (OP09). Der
`FAHRPLAN` setzt die Schwelle bei ~1 °C: „ein Unterschied unter ~1 C ist mit
drei Seeds nicht lesbar". Hier sind es 22 °C. **Dieser Lauf kann nichts
ranken** — weder B gegen A noch sonst etwas.

**2 — Der beste Seed schlägt das triviale Modell nicht.**
Mit `T_sigma = 9.602 °C` liefert „sage überall den Trainingsmittelwert" eine
MAE von etwa `0.798 · σ ≈ **7.7 °C**` (Gauß-Näherung, die Halte-OPs haben ihre
eigene Verteilung — die Größenordnung stimmt trotzdem). Seed 2 kommt auf
**8.03 / 8.95 °C**. Das Netz hat also im besten Fall **so gut wie nichts**
gelernt, was über den Mittelwert hinausgeht; die anderen beiden Seeds sind
deutlich schlechter als raten.

**3 — Der Trainingsverlust sagt das Ergebnis nicht vorher.**
Seed 0 endet bei `data 0.332` → val-MAE **51.95 °C**.
Seed 2 endet bei `data 0.305` → val-MAE **8.95 °C**.
Praktisch derselbe Verlust, Faktor 5.8 im Ergebnis. Wer auf `data` schaut,
schaut auf die falsche Zahl.

---

## 2. Warum — die Diagnose

### Der strukturelle Grund: ein Schritt trainiert, achttausend gemessen

`train_epoch` macht zwei Dinge (siehe `README_DIAGNOSTIK.md`):

1. **einmal** frei ausrollen über ~8040 Schritte, einfrieren
2. je OP `inner_steps` **Ein-Schritt**-Updates gegen die Labels

**Nichts in der Schleife beschränkt das Verhalten über 8040 Schritte.** Ein
winziger systematischer Fehler je Schritt potenziert sich. Genau das zeigt die
Sättigung: bis zu **88 248** OP-Zeitschritte an der Schranke, gegen ein
Maximum von etwa `11 × 8039 ≈ 88 400` — also **≈ 99.8 %, jeder OP in praktisch
jedem Schritt**.

Bei `--clamp 50` und `T_sigma = 9.602 C` heißt das rund **±480 °C**. Der
Rollout ist dort nicht ungenau, er ist weg.

### Die Rückkopplung, die es schlimmer macht

Die Historie für Schritt 2 kommt aus dem eingefrorenen Rollout
(`history_at(traj, …)`), nicht aus den Labels — das ist der Grund, warum dies
**kein** Teacher Forcing ist. Die Kehrseite: ist `traj` festgenagelt, trainiert
Schritt 2 auf Eingaben, die nach einer Erholung **nie wieder vorkommen**. Eine
Sättigungsphase ist deshalb keine Delle, sondern eine Rückkopplung — und die
Läufe zeigen genau das Muster: Ausbruch, Sturm über 10–13 Epochen, Erholung,
neuer Ausbruch (Seed 0: ep 3–17, 26–28, **38–50**).

### Was es *nicht* ist

**Nicht CFL.** Die Warnung im Log (`110.3x UEBER der Schranke`) ist echt, aber
Arm A ist `--no-physics`: die Rekurrenz ist `T + dt·g_θ`, ohne Laplace-Term.
**Für B, C und D ist sie sehr wohl ein Blocker** — das ist vor dem ersten
B-Lauf zu klären und nicht danach.

**Nicht der Ladepfad.** Die Gitterabbildung ist punktweise geprüft (alle 363
Punkte, `tn_seq` und `fo`), die Normierung stimmt mit dem Basisprojekt überein,
und die Zahlen im Kopf (`T_mu=33`, `T_sigma=9.602`, `L_ref=0.0768039`,
`T_span_ref=1605.2`) decken sich mit `13_reports_NACHHER.txt`.

### Nebenbefund: wir haben die falsche Epoche abgelesen

Berichtet wird der Stand nach **Epoche 60**. Der `FAHRPLAN` verbietet das für
den PINN ausdrücklich: „Nie die letzte Zeile eines Laufs ablesen —
`analyse_history.py`, Median über die letzten Epochen." Für den CNN gab es
diese Möglichkeit bisher nicht: es wurde **keine val-MAE je Epoche**
mitgeschrieben. Epoche 60 ist damit eine Lotterie — Seed 0 hatte gerade eine
Erholung hinter sich und landete trotzdem bei 51.95 °C.

**Das ist eine Lücke in der Implementierung vom 22.09., nicht im Modell.**

---

## 3. Was daraus folgt — in dieser Reihenfolge

> **Grundsatz: erst messen können, dann ändern.** Solange die Seed-Streuung
> 22 °C beträgt, ist **keine** Modelländerung überprüfbar. Jede Verbesserung
> verschwindet im Rauschen.

### Stufe I — Messung reparieren (ändert das Experiment nicht)

| | was | warum |
|---|---|---|
| **I.1** | **val-MAE je Epoche** mitschreiben, bestes Modell sichern, Median über die letzten Epochen berichten | beendet die Epoche-60-Lotterie. Ohne das ist Stufe II nicht bewertbar |
| **I.2** | **Sättigung als Anteil** melden (`88248/88400 = 99.8 %`) statt als nackte Zahl | „88248" ist ohne Bezugsgröße unlesbar |
| **I.3** | **Triviale Latten mitberichten**: Mittelwert-Vorhersage und Persistenz `T(t) = T(0)` | ohne sie ist „8.95 °C" nicht einzuordnen. Sie hätten den Befund sofort gezeigt |

### Stufe II — Stabilität (ändert das Experiment, je als eigener Arm)

| | was | Erwartung |
|---|---|---|
| **II.1** | **Gradienten-Clipping** (`clip_grad_norm_`, z. B. 1.0) | Ausschläge auf 2500 bei 11 k Parametern sind explodierende Gradienten. Billigster Eingriff |
| **II.2** | **LR-Plan** statt konstant 1e-3 über 60 Epochen | die späten Stürme (Seed 0, ep 38–50) sehen aus wie ein Optimierer, der ein Becken verlässt |
| **II.3** | **EMA der Gewichte** | im Basisprojekt hat `--ema-decay 0.5` O15 repariert |

### Stufe III — die strukturelle Frage (braucht eine Entscheidung)

Der Horizont von 8040 Schritten ist das eigentliche Problem. Zwei Wege, und sie
schließen einander nicht aus:

1. **Rollout-Länge als Curriculum** — kurz anfangen, wachsen lassen. Das ist
   **kein** Teacher Forcing (die Historie bleibt die eigene), aber es ist eine
   Änderung am Experiment und gehört bewusst entschieden.
2. **Größeres `--subsample`** — weniger Schritte, größeres dt. Billiger, aber
   es ändert die Daten *und* verschärft CFL für B/C/D.

⚠ **Vor dem ersten B-Lauf ist die CFL-Frage zu klären** (110× bei subsample 2).
Sonst misst B eine Instabilität und nennt sie eine Architekturaussage.

---

## 4. Offene Fragen, die ich nicht selbst entscheiden kann

> **✅ Alle vier sind am 22.09., abends, entschieden.** Die Begründungen stehen
> als Tabelle im Kopf von [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md). Kurz:
> (1) das Budget war ein erster Schuss — konstant gehört `inner_steps × k`;
> (2) **ja**, das Experiment darf sich ändern — Clipping, LR-Plan und TBPTT
> sind *Protokoll* und gelten dann für A, B, C und D gleich; (3) `--clamp 50`
> war **geerbt**, die Vorgabe ist jetzt `auto` aus den Labels; (4) die
> CFL-Gabelung wird **nicht jetzt** entschieden, weil Arm A sie nicht braucht
> und der POC die Zahl erst liefert, mit der sie eine Rechnung wird.
>
> Die Fragen bleiben unverändert stehen — sie sind der Grund, warum die
> Entscheidungen so ausgefallen sind.

1. **Budget.** Sind 60 Epochen × 100 `inner_steps` gesetzt, oder war das ein
   erster Schuss? Davon hängt ab, ob Stufe II überhaupt Zeit hat zu wirken.
2. **Darf sich das Experiment ändern?** Clipping und LR-Plan sind Eingriffe.
   Als eigene Arme neben A sauber — aber dann ist A, wie es heute dasteht,
   die Latte, und die ist unbrauchbar.
3. **`--clamp 50`** — bewusst gewählt oder aus `PINNmodulusTwo` geerbt? Bei
   ±480 °C fängt er nichts ab, was noch zu retten wäre.
4. **CFL für B/C/D**: kleineres `subsample` (mehr Schritte → schlimmere
   Aufschaukelung) oder größeres (gröbere Daten)? Das ist eine echte Gabelung.

---

## 5. Was dieser Lauf trotzdem wert war

* **Der Ladepfad trägt.** Daten, Gitter, Normierung, Seeds, Artefakte — alles
  lief auf der echten Maschine durch, ohne Eingriff.
* **`[SATURATED]` hat funktioniert.** Genau wofür es gebaut wurde: ein stiller
  weglaufender Rollout hätte wie langsame Konvergenz ausgesehen.
* **Die Seed-Schleife hat sich sofort bezahlt gemacht.** Mit einem Seed hätten
  wir je nach Würfel „8.95 °C, brauchbar" oder „51.95 °C, kaputt" berichtet.
  Beides wäre falsch gewesen.
* **Die `[CFL]`-Zeile hat vorab gewarnt** — und die Diagnose, dass sie für A
  *nicht* gilt, hat eine halbe Stunde Irrweg gespart.
