# GridCNN — Fahrplan

> **Lies zuerst [`README.md`](README.md).** Dort steht *warum*; hier steht *was,
> in welcher Reihenfolge, und woran es scheitert*.
>
> Heißt `FAHRPLAN.md` nach der Konvention des Repos — `PINNmodulusTwo` hat
> seinen eigenen, und die `.gitignore` führt genau diesen Namen auf ihrer
> Whitelist.

> ## ⚠ 15.09. — Der CNN wird gebaut, gegen das rote Tor 0
>
> Tor 0 ist am 09.09. rot gefallen (**4 Moden bei 99.9 %**) und hätte ein ROM
> verlangt. Am 15.09. ist entschieden worden, den CNN trotzdem zu bauen; der
> ROM-Plan ist gestrichen.
>
> **Die vollständige Begründung steht im [`README.md`](README.md)** unter
> „Die Entscheidung, die man kennen muss" — hier nur die Folgen für den Plan:
>
> * Gebaut werden **16 × 3 = 11 427 Parameter**, nicht die 64 × 4 aus der
>   Präsentation (die **137 923** kosten, nicht „~100 k" wie §11.4 sagt).
>   Die Größe bleibt als Sweep-Achse erreichbar.
> * Stufe 3 ist wieder der explizite Löser, Stufe 4 der CNN. Stufe 3b entfällt.
> * Der Rangtest und sein Ergebnis bleiben in „Erledigt" und in der
>   Stand-Tabelle. **Eine Messung wird nicht dadurch ungültig, dass man sich
>   anders entscheidet** — sie ist die erste Stelle, an der man nachsieht,
>   wenn Stufe 4 nicht trägt.

**Sortierung:** von oben nach unten — was zu tun ist, steht oben; was erledigt
ist, wandert nach unten in „Erledigt".

Der Plan ist eine **Leiter mit Toren**, keine gerade Linie. **Ein rotes Tor
ändert den Plan, nicht nur den Haken.**

---

## ▶ Das Nächste: **A auf voller Auflösung bestätigen**

> ## 🟢 22.09., abends — der POC trägt: **der CNN ist rettbar**
>
> | | berichtet | Latte | Güte | Seed-Streuung |
> |---|---|---|---|---|
> | **OP06** | **6.571 °C** | 10.7995 | **0.61x** | 0.38 °C |
> | **OP09** | **5.813 °C** | 7.7663 | **0.75x** | 0.87 °C |
>
> **3/3 Seeds unter der Latte auf beiden OPs**, Streuung **unter** der
> Lesbarkeitsschwelle von ~1 °C. Von 27.11 ± 22.26 °C am Vormittag auf
> 6.57 ± 0.38 °C. Die Diagnose war richtig: es lag an der Schleife
> (`--tbptt`), nicht am Netz.
>
> Die trivialen Latten stimmen mit `PINNmodulusTwo/FAHRPLAN.md:1244` auf drei
> bis vier Nachkommastellen überein — der Ladepfad baut dieselbe Haltemenge,
> und der Vergleich gegen den PINN (6.270 / 3.585, **ein** Seed, mit
> Randterm) steht: **auf OP06 gleichauf, auf OP09 zurück.**
>
> ⚠ **Der Lauf lief bei `--subsample 10`** (1445 Schritte), nicht bei 2
> (~8040). Das ist die leichtere Aufgabe. Voller Bericht in
> **[`TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md`](../TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md)**,
> Übergabe in **[`UEBERGABE_2026-09-22_ABEND.md`](../UEBERGABE_2026-09-22_ABEND.md)**.

> ## 🔴 Und der Befund, der alles andere überlagert: **O13 ist zurück**
>
> Der Plot
> **[`15_konfigA_OP06_fehler_ueber_zeit.png`](laeufe/15_konfigA_OP06_fehler_ueber_zeit.png)**
> zeigt den Fehler über die Trajektorie statt als Mittelwert:
>
> | t [s] | 131 | 394 | 788 | 919 | 1181 | 1444 |
> |---|---|---|---|---|---|---|
> | **MAE [°C]** | 9.8 | **1.1** | 7.8 | 3.7 | 8.9 | **15.0** |
>
> **Bei 394 s ist das Modell auf dem ~1 K-Ziel. Am Ende liegt es bei 15 °C**,
> und der *kleinste* Fehler über alle 363 Punkte beträgt dort noch 11.5 °C —
> das ganze Feld liegt daneben. Ein **Pegelfehler**, keine Streuung.
>
> `model.py` hat genau das vorhergesagt: die Delta-Form trägt den Level
> **ohne Leck**, und der dissipative Diffusionskern sollte es liefern.
> **Arm A hat ihn nicht.** Damit ist Arm B nicht der nächste Listenpunkt,
> sondern die Behandlung für die beobachtete Krankheit.
>
> **Seit dem 22.09., abends, misst `train.py` das selbst**: sechs Abschnitte,
> MAE und vorzeichenbehafteter Bias je Abschnitt, plus
> `drift = MAE(letzter Abschnitt)/MAE(gesamt)`. Ab 1.5 meldet der Lauf O13.

### Schritt 1 — volle Auflösung, ~45 min

Nur `--subsample` und `--epochs` ändern sich. Die Lags leiten sich bei
`--subsample 2` automatisch wieder auf 5/20 ab.

```bash
python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 4 --tbptt 16 \
    --val-every 2 --device cuda --cache data_cache \
    2>&1 | tee 16_konfigA_voll.txt
```

Worauf zu schauen ist: hält die Güte < 1.0 bei fünffachem Horizont? Was sagt
`drift`? Und vor allem — **was sagt der Bias?** Negativ zum Ende heißt, das
Modell wird zu kalt; positiv, zu warm. Das ist neu und es entscheidet, wo man
ansetzt.

Geht es schief: Frühphase instabil → `--tbptt-start 8`; val-Kurve schwankt →
EMA. Beides begründet, aber **eins nach dem anderen**.

### Schritt 2 — CFL entscheiden (eine Rechnung, kein Lauf)

110× über der Schranke bei `--subsample 2`, 551× bei 10. Für A egal, für
B/C/D ein Blocker. Zuerst zu prüfen: sind die Materialdaten echt? Ein
synthetisches `material_properties/` macht das Problem viel steifer, als es
ist — die `[CFL]`-Zeile sagt das selbst.

### Schritt 3 — Arm B, der Physik-POC

Gated auf Schritt 2. Die Frage ist präzise: **holt der dissipative
Diffusionskern den Pegel zurück?** Messbar an `drift` und `bias`, nicht am
Mittelwert.

### Parallel, weil Code und kein Lauf: den Wandterm verdrahten

`_wall_ghost` wirft weiterhin — **der genannte Grund war abgestanden.**
Stufe 2 ist durch, `U(V̇)` kalibriert, `physics.UCurve` und
`physics.WallModel.ghost` gebaut und getestet. Was wirklich fehlt:

1. `op_tensoren` füllt `OPTensors.q_wall_meas` nie → `wall_loss` kann nicht
   laufen.
2. `WallModel.ghost` braucht `t_in` und `mdot` **je Zeitschritt**; beide
   liegen seit Stufe 2 im Bündel, sind aber nicht in `OPTensors` übernommen.

---

### Was vorher galt: **der POC — unterbietet der CNN die Latte überhaupt?**

> **Erledigt am 22.09., abends.** Er trägt, siehe oben.

### Die vier offenen Fragen aus Abschnitt 4 des Berichts — entschieden

| | Frage | Entscheidung |
|---|---|---|
| **1** | Budget: 60 × 100 gesetzt? | **Nein, erster Schuss.** Was konstant gehört, ist `inner_steps × k`, nicht `inner_steps`: mit einem Fenster von `k` kostet ein Update `k` Schritte. Der POC fährt 40 × 25 bei `k` bis 16 — viermal die Arbeit vom 22.09., bei sechzehnfachem Horizont |
| **2** | Darf sich das Experiment ändern? | **Ja.** Clipping, LR-Plan und TBPTT sind **Protokoll**, keine Experimentvariablen: sie beschreiben, *wie* trainiert wird, nicht *was* verglichen wird. Einmal festgelegt, dann identisch für A, B, C und D. Der Einwand „dann ist A die Latte, und die ist unbrauchbar" trägt nicht — A wie am 22.09. **ist** keine Latte und kann darum auch keine verlieren. Der alte Pfad bleibt exakt reproduzierbar: `--tbptt 1 --clip-grad 0 --lr-plan konstant --clamp 50 --lag1 5 --lag2 20` |
| **3** | `--clamp 50` — gewählt oder geerbt? | **Geerbt**, aus `PINNmodulusTwo`. Bei ±480 °C fängt er nichts ab, was noch zu retten wäre. Die Vorgabe ist jetzt `auto`: das Dreifache der größten Auslenkung in den Labels, aus den Daten gerechnet und in °C gedruckt |
| **4** | CFL für B/C/D — kleineres oder größeres `subsample`? | **Noch nicht zu entscheiden, und das ist die Entscheidung.** Arm A ist `--no-physics`, dort gibt es keine CFL-Schranke; der POC braucht die Antwort also nicht. Umgekehrt liefert er den Eingang für sie: erst wenn bekannt ist, wie viel Horizont das Netz überhaupt trägt, ist die Gabelung eine Rechnung statt eines Ratens. ⚠ **Festhalten:** `--subsample 10` *verschärft* CFL für B/C/D auf rund **550×**. Der POC-Befehl unten ist deshalb **kein** Vorgriff auf B |

### Was dafür gebaut wurde (22.09., abends)

| Stufe | | |
|---|---|---|
| **I — Messung** | val-MAE je `--val-every` Epochen, `model_best.pt`, Sättigung als **Anteil**, triviale Latten **vor** jedem Lauf und **in jeder val-Zeile** (`0.79x Latte`) | ändert das Experiment nicht |
| **I+ — Berichtsregel** | berichtet wird der **Median über das letzte Drittel** der Messpunkte. Nicht das Beste: das wäre auf der Haltemenge ausgewählt, und die Haltemenge ist hier die ganze Messung. Bestes und letztes stehen als Diagnose daneben | ändert das Experiment nicht |
| **II — Stabilität** | `--clip-grad 1.0`, `--lr-plan cosine`, `--clamp auto`, Verwerfen nicht-endlicher Updates, Gradientennorm im Log | Protokoll (Frage 2) |
| **5 — der Bruch selbst** | **`--tbptt`**: truncated BPTT. Das Fenster rollt **mit** Gradient und frisst sich selbst; ab Schritt 1 ist der Anker die eigene Vorhersage. Das schließt die Lücke „ein Schritt trainiert, ~8040 gemessen" | Protokoll (Frage 2) |
| **Tempo** | Lags aus **Sekunden** statt Schritten (5/20 bei `--subsample 2` unverändert), val-Rollout gebatcht, Wanduhr je Epoche | ändert das Experiment nicht |

### Der Befehl

```bash
{ python3 GridCNN/train.py --no-physics --seeds 1 --epochs 3 \
      --subsample 10 --inner-steps 4 --tbptt-start 2 --tbptt 8 \
      --val-every 1 --device cuda --cache data_cache \
      --artifacts-dir /tmp/poc_smoke \
  && python3 GridCNN/train.py --no-physics --seeds 3 --epochs 40 \
      --subsample 10 --inner-steps 25 --tbptt-start 4 --tbptt 16 \
      --val-every 2 --device cuda --cache data_cache ; } \
  2>&1 | tee 15_konfigA_poc.txt
```

Der erste Lauf ist ein Rauchtest von rund einer Minute: stürzt der Ladepfad
bei `--subsample 10` ab, fällt das auf, bevor eine halbe Stunde verbrannt ist.
Der zweite ist der POC. **`--subsample 10` braucht keinen Cache-Rebuild** —
die `.npz` halten die volle Rohauflösung, `subsample_time` wirkt erst beim
Laden.

### Was der POC beweist — und was nicht

| | |
|---|---|
| ✅ **Beweist** | ob der CNN in Arm A die trivialen Latten (Mittelwert **und** Persistenz) über einen freilaufenden Rollout von ~1608 Schritten unterbietet, und ob die Seed-Streuung unter die Lesbarkeitsschwelle von ~1 °C fällt. Beides liest das `[verdikt]` am Ende vor |
| ❌ **Beweist nicht** | die volle Auflösung (`--subsample 2`, ~8040 Schritte). Ein kürzerer Horizont ist **leichter**, ein Erfolg hier ist also die schwächere Aussage — ein **Misserfolg** hier ist dafür die starke |
| ❌ **Beweist nicht** | irgendetwas über B, C oder D. Arm A ist adiabat und ohne Physik: eine **Ablation, keine Latte**, und `--subsample 10` verschärft CFL (Frage 4) |

**Trägt der POC**, ist derselbe Befehl mit `--subsample 2 --epochs 60` die
Bestätigung auf voller Auflösung, und *danach* ist A die Latte für B.
**Trägt er nicht**, ist der nächste Hebel `--ema-decay` (hat im Basisprojekt
O15 repariert) und dann die Netzgröße — nicht ein weiterer Lauf desselben.

---

### Was vorher galt: **A wiederholen, nachdem die Messung repariert ist**

> ## 🔴 22.09., abends — A ist gelaufen und ist **kein Ergebnis**
>
> | | Seed 0 | Seed 1 | Seed 2 | Mittel ± σ |
> |---|---|---|---|---|
> | val-MAE OP06 | 51.95 | 20.43 | **8.95** | **27.11 ± 22.26 °C** |
> | val-MAE OP09 | 51.45 | 15.48 | **8.03** | **24.99 ± 23.22 °C** |
>
> **Die Streuung ist der Befund, nicht der Mittelwert.** 22 °C gegen eine
> Lesbarkeitsschwelle von ~1 °C — dieser Lauf kann **nichts ranken**. Und der
> beste Seed unterbietet die triviale Latte nicht: „sage überall den
> Trainingsmittelwert" liegt bei ~7.7 °C, Seed 2 bei 8.03 / 8.95.
>
> Ursache ist der Bruch zwischen Training und Messung: **ein Schritt trainiert,
> ~8040 gemessen.** Bis zu **99.8 %** aller OP-Zeitschritte lagen an `--clamp`.
> Volle Analyse in **[`TRAININGS_BERICHT_2026-09-22_KonfigA.md`](../TRAININGS_BERICHT_2026-09-22_KonfigA.md)**,
> die Epochenzeile erklärt **[`README_DIAGNOSTIK.md`](README_DIAGNOSTIK.md)**.
>
> **Stufe I ist am selben Abend gebaut** (Messung, ohne das Experiment
> anzufassen): val-MAE je `--val-every` Epochen mit `model_best.pt`, Sättigung
> als Anteil, und die trivialen Latten vor jedem Lauf.
>
> **Überholt noch am selben Abend.** Eine reine Wiederholung von A hätte die
> Messung repariert und die Ursache stehen lassen: der Bruch zwischen „ein
> Schritt trainiert" und „~8040 gemessen" liegt nicht in der Messung, sondern
> in der Schleife. Ein zweiter Lauf desselben Protokolls hätte dieselbe
> 22-°C-Streuung sauberer berichtet — und wäre wieder kein Ergebnis gewesen.
> Stattdessen läuft der POC oben, mit Stufe 5 und II im Protokoll.

### Was vorher galt: Konfiguration A fahren — Stufe 1 und 2 sind durch

> **Stand 22.09., abends.** Stufe 1 abgeschlossen (`U(V̇)` kalibriert), Stufe 2
> **durch mit grünem Tor** (17/17 OPs auf Schema v3, die drei Reports
> zeichengleich), und der **Ladepfad ist angeschlossen** — `train.py` trainiert.
> Was jetzt fehlt, ist kein Code, sondern eine **Messung**: Konfiguration A,
> `--seeds 3`, auf der Maschine mit echten Daten. Sie ist die Latte für B, C
> und D.
>
> ```bash
> python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
>     --device cuda --cache data_cache 2>&1 | tee 14_konfigA.txt
> ```
>
> Achte auf die `[CFL]`-Zeile am Anfang: liegt `dt_n` über der Schranke, ist
> ein weglaufender Rollout **damit** erklärt und nicht mit dem Netz. Arm A ist
> davon unberührt (keine Physik in der Architektur), B/C/D nicht.
>
> Der Lauf ist **adiabat** — `--w-wall` bleibt 0, bis `UCurve` an die vier
> neuen Cache-Größen angeschlossen ist. Das ist eine **Ablation, keine Latte**.

### Was Stufe 2 war — erledigt

**Drei Läufe am 22.09.** (`10`/`11`/`12_bilanz.txt`; sieben
Konstant-Treiber-Trainings-OPs, drei Flusslevel). **Stufe 1 ist damit
abgeschlossen** — `U(V̇)` ist kalibriert, gegen die Bezugstemperatur, die
`physics.UCurve` selbst vorschreibt:

| V̇ [l/min] | ṁ [kg/s] | **`U(T_mittel)`** [W/m²K] | Streuung |
|---|---|---|---|
| 0 | 0 | **50.4** | 2.10 % |
| 15 | 0.0013 | **421.0** | 1.38 % |
| 30 | 0.0026 | **501.7** | 1.09 % |

**Das sind genau die drei trainierten Flusslevel** — also genau die
Stützstellen, die `UCurve` braucht. Sie interpoliert linear dazwischen und
klemmt außerhalb. **Null freie Parameter, und keine Funktionsform nötig.**

> **Der Wandterm ist damit kalibrierbar.** Das war der Engpass in
> „Was fehlt"; er ist weg. Was noch fehlt, ist der Weg der vier Größen *in den
> Cache* — und das ist Stufe 2.

**Stufe 2** schreibt `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`, `mdot`
— und `total_w`, weil O17 sonst nie messbar wird — in `generate_cache.py` und
`opbundle_contract.md`, hebt `schema_version` und baut **alle siebzehn** OPs
neu (10–30 min). `generate_cache.py` ohne Argumente baut nur OP05–OP07; die
Liste gehört ausgeschrieben, **OP19 eingeschlossen**.

**Tor:** `profile_report`, `coverage_report` und `energy_balance_report`
liefern exakt dieselben Zahlen wie vorher.

> ### Was der Lauf vom 22.09. ergeben hat
>
> | Abschnitt | | |
> |---|---|---|
> | 2 — Fluidbilanz | **1.030 … 1.062** | 🟢 **`ghost_hi` steht.** Kein Halbmodell-Faktor auf dem Fluidpfad. `T_in` kam aus `Input Signale.csv`, wie gebaut |
> | 1 — `Q_ht/tot` | **0.700 … 0.772** mit Fluss | 🔴 **Die Hypothese aus 1a ist widerlegt** — der Monitor draint **nicht** die ganze Erzeugung |
> | 1 — `Q_ht/jr1` | **2.286 … 2.566** | 🟡 nahe „beide Platten", aber **nicht konstant** — siehe unten |
> | 1 — `tot/jr1` | **2.99 … 3.44** | die Zelle erzeugt **gut das Dreifache** einer Rolle |
> | 4 — `U(V̇)` | **50.4 / 421.0 / 501.7** W/m²K gegen `T_mittel` | 🟢 je Flusslevel auf **1.1–2.1 %** zusammen. Siehe 1e |
>
> **Der Befund, der den Plan ändert:** `Q_ht/jr1` **hängt vom Flusslevel ab** —
> 2.286 / 2.302 / 2.304 bei V̇ = 0.0013 gegen 2.488 / 2.566 bei 0.0026, und
> 0.334 / 0.336 ohne Fluss. **Ein Konventionsfaktor kann das nicht.** Eine
> Bezugsfläche ist Geometrie; sie weiß nicht, wie schnell das Kühlmittel
> fließt. Das Verhältnis ist also **keine Konstante, die man wegdividiert**,
> sondern zu einem Teil Physik: mehr Kühlung ⇒ mehr abgeführter Anteil.
>
> Dasselbe an `Q_ht/tot`: 0.11 ohne Fluss → 0.72 → 0.76. Es **sättigt gegen
> ~0.8, nicht gegen 1**. Im späten Fenster wird also immer noch gespeichert;
> die Zelle ist nicht quasistationär, und der Wandmonitor kann die Erzeugung
> gar nicht ausgleichen.
>
> **Was daraus für die Quelle folgt** — und das bleibt stehen, unabhängig vom
> Faktor: bei `jr2/jr1 = 1.000` und `tot/jr1 ≈ 3.2` liegen rund
> **ein Drittel bis 40 % der Gesamterzeugung außerhalb der beiden Wickel**
> (Ableiter, Stromschienen, Kontaktwiderstände). Die Modellquelle `q_dot` deckt
> nur JR1 ab. Das ist **O17 für `PINNmodulusTwo`** und es ist jetzt belegt,
> nicht vermutet. Damit `energy_balance_report` es je messen kann, muss
> `total_w` in **Stufe 2** mit in den Cache.
>
> ⚠ **Das Tor „Q halbieren oder A verdoppeln" greift so nicht mehr.** Es setzt
> voraus, dass `Q_ht/jr1` eine Konstante ist. Ist es nicht. Die Trennung
> zwischen Konvention und Physik hängt jetzt an Abschnitt 4: **liegt `U` über
> die drei Flusslevel auf einer Kurve, ist der flussabhängige Teil erklärt**,
> und was dann an konstantem Faktor übrig bleibt, ist die Konvention.

### Was am Werkzeug diesmal kaputt war

`np.interp(tw, r["t"], r["t_in"])` → `fp and xp are not of the same length`.
Das Skript liest fünf CSVs je OP und hat die Zeitachse von **zwei** mitgelesen.
`t_out` (aus `*_Temperaturen.csv`) und `t_in` (aus `*_Input Signale.csv`)
wurden behandelt, als lägen sie auf der Achse von `*_Heat Source.csv`. `t_in`
ist dort ein **Skalar** — die Datei führt Sollwerte und hat gar keine
Zeitspalte.

> **In Abschnitt 2 ist genau das nicht aufgefallen**, weil dort `t_out - t_in`
> steht und numpy Länge 1 still auf Länge N broadcastet. Die Zahl war richtig,
> aber aus dem falschen Grund — bei einer *nicht* konstanten Reihe wäre sie
> still falsch gewesen. **Dieselbe Klasse wie `xyz = raw[0]` in `data.py`:**
> eine Zusage, die stimmt, die aber niemand prüft.
>
> Repariert: `auf_achse()` verlangt, dass man die Achse einer Reihe nennt;
> ein Skalar trägt keine und gilt überall; eine Reihe **ohne** Achse wird
> abgelehnt statt geraten. `als_konstante()` prüft den Rückfallwert und bricht
> ab, wenn er sich doch ändert.

**Was am Werkzeug korrigiert ist:**

| | war | ist |
|---|---|---|
| Default-OPs | OP04, OP05, OP07, OP14 — **beide Fluss-OPs fahren V̇ = 30**, also nur zwei Level | alle sieben Konstant-Treiber-**Trainings**-OPs → Level 0 / 15 / 30, jedes mehrfach |
| `T_in` | ein einziger geratener Spaltenname → `nan` | Rückfallkette bis `Input Signale.csv`, und das Skript sagt, welche Quelle es benutzt hat |
| Abschnitt 1 | nur `Q_ht/jr1` | zusätzlich **`Q_ht/tot`** — die Zahl, die die 2.5 auflöst |
| Erwartung bei ṁ = 0 | „Anteil ≈ 0, sonst 🔴" | **falsch, korrigiert.** Begründung unter Stufe 1 |

> **OP16 fehlt in den Defaults absichtlich.** Es fährt V̇ = 90 und wäre der
> vierte Stützpunkt — aber es ist ein **Test**-OP. `U` daran zu kalibrieren wäre
> eine Auswahl auf dem Extrapolationstier. OP16 ist die Gegenprobe für `U(V̇)`,
> nie die Stütze.

---

## Die Leiter

| Stufe | was | Dauer | Tor |
|---|---|---|---|
| **0** | Rangtest | ✅ **erledigt — ROT, überstimmt** | 4 Moden. Folge: 16 × 3 statt 64 × 4 |
| **1** | Bilanz-Gegenprobe | ✅ **erledigt 22.09.** | Fluidbilanz 🟢, `U(V̇)` kalibriert, `Q_ht/jr1` als flussabhängig entlarvt |
| **2** | vier Größen in den Cache | 30 min | Reports weiter grün? |
| **3** | **Physik ohne Netz** — der explizite Löser | ✅ **gebaut**, läuft adiabat | schlägt reine Physik die trivialen Vorhersager? |
| **4** | **der CNN**, Ein-Schritt-Training | ✅ **gebaut**, Ladepfad offen | schlägt er Stufe 3? |
| **5** | truncated BPTT | Tage | fällt der Spätfehler (O13)? |
| **6** | Vergleich gegen PINNmodulusTwo | 1 Lauf | derselbe Split, dieselben Metriken |

> **Stufe 3 und 4 sind gebaut, aber nicht gemessen.** Beide hängen an
> derselben Sache: `solve.py` läuft adiabat und `train.py` hat keinen
> Ladepfad, weil `U(V̇)` nicht kalibrierbar ist und der `data_cache` die vier
> Größen aus Stufe 2 nicht enthält. **Stufe 2 ist der Engpass, nicht die
> Bauzeit.**

> **Was im Repo steht und was fehlt**, steht unten unter „Der Code"; das
> Protokoll der Läufe in [`BENCHMARK.md`](BENCHMARK.md).

---

# Stufe 1 — Geht die Bilanz auf? ✅ **abgeschlossen 22.09.**

**Gebaut:** [`tools/balance_check.py`](tools/balance_check.py). Erster Lauf am
09.09., Zahlen in der Stand-Tabelle. Drei Befunde:

## 1a. `Q_ht/JR1 ≈ 2.5` — weder 1 noch 2

Meine Verdikt-Bänder sagten „beide Platten" (1.6 … 2.6), aber 2.5 ist kein
Faktor 2. **Hypothese, die der zweite Lauf prüft:**

> `Q_ht ≈ total_w`, und `total_w / JR1 ≈ 2.5`.

Dokument 030 hat festgestellt, dass `Heat Source Monitor (total)` **nicht**
`JR1 + JR2` ist, sondern größer. Wenn der Monitor die *gesamte* Erzeugung
draint — was im quasistationären Spätfenster zu erwarten ist — dann ist 2.5
**kein Konventionsfehler**, sondern die Aussage:

**Die Zelle erzeugt mehr Wärme als 2 × JR1, und die Modellquelle `q_dot` deckt
nur JR1 ab.** Bei `jr2/jr1 = 1` (gemessen) und `tot/jr1 ≈ 2.5` wären das grob
**20 % der Gesamterzeugung**, die nirgends im Modell vorkommen — Ableiter,
Stromschienen, Kontaktwiderstände.

Das wäre ein neuer offener Punkt für `PINNmodulusTwo` (**O17**), kein
GridCNN-Thema. Aber erst, wenn `Q_ht/tot` es belegt — die Spalte gibt es jetzt.

## 1b. Bei ṁ = 0 bleiben ~0.27 — **mein Torkriterium war falsch**

Ich hatte geschrieben: *„bei ṁ = 0 fließt Energie ab → 🔴 der Entwurf hat einen
Pfad übersehen."* Das war falsch gedacht.

**`ṁ = 0` heißt kein *Fluss*, nicht kein *Fluid*.** Das Kühlmittel steht im
Kanal und nimmt Wärme in seine **eigene Wärmekapazität** auf. Ein
Solid-to-Fluid-Wärmestrom bei stehendem Fluid ist physikalisch richtig — die
Energie wird **gespeichert**, nicht abtransportiert.

Und die Zahlen bestätigen es:

| | `U` [W/m²K] | 22.09. korrigiert |
|---|---|---|
| mit Fluss (V̇ = 30) | ~~~1130~~ | **326** gegen `T_in`; gegen `T_mittel` höher, **der Lauf misst es** |
| ohne Fluss | ~50 | **50.3** — bestätigt |

~~Faktor **~23**.~~ ⚠ **Zurückgezogen am 22.09., siehe 1e.** Die 1130 war nicht
gemessen, sondern die Folge einer anderen Bezugstemperatur: `T_in` fehlte, das
Werkzeug fiel auf `T_out` zurück. Konsistent gerechnet ist der Faktor
**4.5× … 7.6×**, nicht 23.

**Die Aussage von 1b bleibt trotzdem stehen** — und darauf kam es an:
stehendes Flüssigkühlmittel bei O(50) und Zwangskonvektion in einer Kühlplatte
deutlich darüber sind beide lehrbuchplausibel. **Das ist eine Bestätigung der
Messkette, kein Fehler.** Was fällt, ist nur das „O(1000)": für einen
*Gesamtdurchgang* inklusive der 1.9 mm Festkörper sind O(200…500) das
Erwartbare, und die korrigierte Zahl ist damit die plausiblere.

> ⚠ Für V̇ = 15 ist `U(T_mittel) ≈ 381` belastbar — die Wandüberhöhung
> `dT_wand ≈ 8.92 K` lässt sich aus der 09.09.-Zahl zurückrechnen und trifft
> sie auf drei Stellen. Für V̇ = 30 gibt es **keinen solchen Anker**; dort
> steht keine Schätzung, sondern der Lauf.

### Was daraus folgt, ist ein Modellbefund

`ΔT_fluid = Q̇ / (ṁ · Cp)` **divergiert für ṁ → 0**. Die Form ist nur für einen
Durchfluss richtig. Richtig ist beides zusammen:

```
C_fluid · dT_fluid/dt  =  Q̇  −  ṁ · Cp · (T_fluid − T_in)
                            ^Quelle      ^Advektion
```

Bei ṁ = 0 bleibt reines Aufladen; bei großem ṁ fällt die alte stationäre Bilanz
heraus. **Eine Zustandsvariable mehr, und die Formel hat keine Singularität.**

> ### Und das trifft genau O14
>
> V̇ = 0 ist das Regime mit dem schlechtesten ausgehaltenen Wert (5.374 gegen
> 2.928 °C im Mittel, OP06 mit 6.270 °C). Bisher war die Erklärung **Abdeckung**
> (O14: nur zwei Trainings-OPs, beide Kälteextreme). Jetzt gibt es daneben einen
> **Mechanismus**: es ist das Regime, in dem die Fluidbeschreibung als reine
> Advektion zusammenbricht.
>
> **Die Abdeckungserklärung bleibt gültig** — die zwei Befunde ersetzen sich
> nicht. Sie sind aber **trennbar**, und das ist die interessante Messung:
> *verbessert ein Kapazitätsterm die V̇ = 0-OPs, ohne dass neue Daten dazukommen?*
> Wenn ja, war O14 nicht nur Abdeckung.

## 1c. `U(V̇)` hat zwei Punkte, nicht eine Kurve

Weil OP04 **und** OP05 beide V̇ = 30 fahren. Behoben, siehe „Das Nächste".

## Das Tor, neu formuliert

| Ergebnis | Folge | gemessen 22.09. |
|---|---|---|
| `Q_ht/tot ≈ 1` | 🟢 kein Konventionsfehler. Stattdessen **O17**: die Modellquelle ist unvollständig | **nein** — 0.700 … 0.772 |
| `Q_ht/jr1 ≈ 2`, `Q_ht/tot ≉ 1` | 🟡 beide Platten. **Entweder** `Q` halbieren **oder** `A` verdoppeln — nie beides | **2.286 … 2.566, flussabhängig** — siehe 1d |
| Fluidbilanz-Verhältnis ≈ 1.0 | 🟢 `ghost_hi` steht | ✅ **1.030 … 1.062** |
| `U` über drei Flusslevel auf einer Kurve | 🟢 `U(V̇)` wird feste Funktion, null freie Parameter | *offen* — Abschnitt 4 abgestürzt |
| ~~bei ṁ = 0 fließt Energie ab~~ | ~~🔴~~ **gestrichen** — war falsch, siehe 1b | — |

## 1d. `Q_ht/jr1` ist keine Konstante — und damit ist das Tor falsch gestellt

Das war am 09.09. nicht sichtbar, weil beide Fluss-OPs dasselbe Level fuhren.
Mit drei Leveln steht es da:

| V̇ | OPs | `Q_ht/jr1` | `Q_ht/tot` |
|---|---|---|---|
| 0 | OP07, OP14 | 0.334 / 0.336 | 0.109 / 0.112 |
| 0.0013 | OP01–OP03 | 2.286 / 2.302 / 2.304 | 0.709 / 0.739 / 0.700 |
| 0.0026 | OP04, OP05 | 2.488 / 2.566 | 0.772 / 0.746 |

**Eine Bezugsfläche weiß nicht, wie schnell das Kühlmittel fließt.** Ein
Konventionsfaktor ist Geometrie und damit konstant; dieses Verhältnis ist es
nicht. Also steckt Physik darin, und „Q halbieren oder A verdoppeln" —
beides *konstante* Eingriffe — kann nicht die ganze Antwort sein.

`Q_ht/tot` sättigt sichtbar gegen **~0.8, nicht gegen 1**: auch im späten
Fenster wird noch gespeichert. Das Fenster ist quasistationär genug für einen
Vergleich, aber die Zelle ist es nicht — der Wandmonitor *kann* die Erzeugung
dort gar nicht ausgleichen.

**Was trotzdem feststeht, unabhängig vom Faktor:** `jr2/jr1 = 1.000` und
`tot/jr1 = 2.99 … 3.44`. Rund **ein Drittel bis 40 % der Gesamterzeugung liegt
außerhalb der beiden Wickel** — und die Modellquelle `q_dot` deckt nur JR1 ab.
Das ist **O17** für `PINNmodulusTwo`, jetzt belegt statt vermutet. Messbar wird
es erst, wenn `total_w` in **Stufe 2** mit in den Cache geht.

**Die Trennung hängt jetzt an Abschnitt 4.** Liegt `U` über die drei
Flusslevel auf einer Kurve, ist der flussabhängige Teil erklärt; was dann an
konstantem Faktor übrig bleibt, ist die Konvention. Vorher ist jede Korrektur
an `Q` oder `A` geraten.

## 1e. `U` hing an der Bezugstemperatur — und die war falsch gewählt

Abschnitt 4 hat geliefert, wonach gefragt war:

| V̇ [l/min] | ṁ | OPs | `U(T_in)` [W/m²K] | Spanne |
|---|---|---|---|---|
| 0 | 0 | OP07, OP14 | 50.83 / 49.78 | **2.1 %** |
| 15 | 0.0013 | OP01–OP03 | 229.25 / 226.64 / 230.31 | **1.6 %** |
| 30 | 0.0026 | OP04, OP05 | 324.37 / 328.48 | **1.3 %** |

**Je Flusslevel liegen die OPs auf 1.3–2.1 % zusammen**, obwohl sie sich in
C-Rate, SOC und Starttemperatur unterscheiden. `U` hängt also **am Fluss und
an sonst nichts** — das ist der Teil des Tores, der hält.

### Aber der Absolutwert steht gegen die falsche Temperatur

`physics.UCurve` schreibt die Kalibrierung selbst vor:

```
U(t) = Q_dot(t) / (A * (T2_mittel(t) - T_fluid_mittel(t)))
```

**`T_fluid_mittel`** — und das passt zu `WallModel._advective`, das entlang +y
marschiert und dabei von `T_in` bis `T_out` läuft. Das Werkzeug rechnet gegen
**`T_in`**, also gegen den kältesten Punkt des Marsches. Das Fluid erwärmt sich
um `dT_fluid` = 7.11 K (V̇ = 15) bzw. 4.12 K (V̇ = 30) — bei einer
Wandüberhöhung von grob 9 K ist das **kein Detail, sondern der halbe Nenner**.

| Bezug (OP01) | `U` | |
|---|---|---|
| `T_in` | 229.25 | **untere** Schranke |
| `T_mittel` | **421.66** | **das, was `UCurve` will** — gemessen |
| `T_out` | **−8684** | **unbrauchbar**, siehe unten |

### Gemessen, alle drei nebeneinander

| V̇ | ṁ | `U(T_in)` | **`U(T_mittel)`** | `U(T_out)` | `dT_fluid` |
|---|---|---|---|---|---|
| 0 | 0 | 50.83 / 49.78 | **50.91 / 49.85** | 50.98 / 49.92 | 0.37 / 0.47 |
| 15 | 0.0013 | 229.25 / 226.64 / 230.31 | **421.66 / 417.71 / 423.52** | −8684 / +21933 / −12729 | 7.11 / 8.15 / 6.69 |
| 30 | 0.0026 | 324.37 / 328.48 | **498.92 / 504.40** | 1129.94 / 1132.60 | 4.12 / 3.47 |

**`U(T_mittel)` streut je Level um 1.09 / 1.38 / 2.10 %** — genauso eng wie
`U(T_in)`. Die Bezugstemperatur verschiebt den Wert, sie verrauscht ihn nicht.

> ### ⚠ `U(T_out)` ist keine obere Schranke, sondern unbrauchbar
>
> Bei V̇ = 15 kommen **−8684, +21933, −12729** heraus. `Tw` ist ein
> **Flächenmittel** der Wand, `T_out` das **heiße Ende** des Kanals — die
> Differenz wechselt das Vorzeichen, und der Mittelwert von `q/(A·dT)` läuft
> weg. Es werden die falschen Paare verglichen, nicht bloß die falsche Seite.
>
> **Bei V̇ = 30 wechselt sie nicht** und liefert brave 1129.94 / 1132.60.
> Die Default-OPs vom 09.09. waren **OP04, OP05, OP07, OP14** — also genau
> die zwei Fluss-OPs, bei denen der Fehler *plausibel aussieht*. Deshalb kam
> damals „~1130" heraus und nicht offensichtlicher Unsinn.

> ### Und damit ist die `~1130` vom 09.09. erklärt
>
> Am 09.09. fand das Werkzeug die `T_in`-Spalte nicht (`nan` in Abschnitt 2)
> und fiel in Abschnitt 4 auf **`T_out`** zurück. Das steht jetzt nicht mehr
> als Rekonstruktion da, sondern **als Zahl in derselben Tabelle**:
>
> | | gemessen `U(T_out)` | Stand-Tabelle 09.09. |
> |---|---|---|
> | OP04 / OP05 | **1129.94 / 1132.60** | „~1130" |
> | OP07 / OP14 | **50.98 / 49.92** | „~50" |
>
> **Die Zahl war kein Messergebnis, sondern die Bezugstemperatur.**

⚠ **Damit ist auch „Faktor ~23" aus 1b hinfällig.** Er verglich 1130
(`T_out`-Bezug, mit Fluss) gegen 50 (`T_in`-Bezug, ohne Fluss) — zwei
verschiedene Maßstäbe. Konsistent gerechnet sind es **4.5× gegen `T_in`** bzw.
grob **7.6× gegen `T_mittel`**. Die *Aussage* von 1b bleibt: stehendes Fluid
trägt deutlich weniger als strömendes, und beide Größenordnungen sind
plausibel. Was fällt, ist das „O(1000) in einer Kühlplatte" — als
**Gesamtdurchgang inklusive der 1.9 mm Festkörper** sind O(200…400) das
Erwartbare, und die korrigierte Zahl ist damit die plausiblere.

### Was das Tor jetzt wirklich sagt

| Teilfrage | Stand |
|---|---|
| hängt `U` nur von V̇ ab? | 🟢 **ja** — 1.1–2.1 % Streuung je Level über OPs mit verschiedener C-Rate, SOC, Starttemperatur |
| ist der Absolutwert belastbar? | 🟢 **ja** — gegen `T_mittel`, wie `UCurve` es vorschreibt |
| **reicht das für den Wandterm?** | 🟢 **ja** — die drei Level *sind* die Stützstellen. `UCurve` interpoliert und klemmt, **null freie Parameter** |
| liegt `U(V̇)` auf *einer Kurve*? | 🟡 **nicht entscheidbar** — siehe unten. **Blockiert aber nichts** |

⚠ **„Drei Flusslevel" sind für die Kurven*form* nur zwei.** V̇ = 0 ist nach 1b
ein **anderer Mechanismus** — stehendes Fluid lädt seine Wärmekapazität, Modus
`capacity`, nicht Konvektion. Für die Advektionsform bleiben **zwei**
Stützstellen, und durch zwei Punkte geht jedes Zweiparametergesetz exakt.

**Das ist aber kein Hindernis**, weil die Form gar nicht gebraucht wird: jeder
Trainings-OP fährt eines der drei Level, und dort steht der gemessene Wert.
Die Form wird erst außerhalb gebraucht — und außerhalb klemmt `UCurve`
ausdrücklich und zählt in `clamped_calls` mit.

### Was die zwei Punkte hergeben — und was nicht

Gemessen: `U(30)/U(15) = 501.7 / 421.0 = 1.1917`, also **`U ∝ ṁ^0.253`**.
Deutlich unter dem turbulenten 0.8 — passend zu einem **Serienwiderstand**,
denn die 1.9 mm Festkörper stecken laut `UCurve` ausdrücklich in `U`. Setzt
man `1/U = R_s + C·ṁ^{-0.8}` an, liefern die zwei Punkte
`U(ṁ→∞) ≈ 677 W/m²K`; `R_s` entspräche 1.9 mm bei **λ ≈ 1.29 W/mK** — also
kein Metall, sondern eher Kontakt oder Spalt.

> ⚠ **Zwei Punkte, zwei Parameter: das ist eine Interpolation, kein Test.**
> Der Exponent 0.8 ist angesetzt, nicht gemessen. `R_s` und die 677 sind
> Folgen dieser Annahme und dürfen nicht als Messwerte zitiert werden.

**Entschieden wird es an OP16** (V̇ = 90, ṁ ≈ 0.0078) — und der Test ist
scharf:

| Lesart | Vorhersage `U` bei OP16 |
|---|---|
| reines Potenzgesetz `ṁ^0.253` | **662** |
| Serienwiderstand mit `ṁ^{-0.8}` | **591** |

**12 % auseinander gegen 1.1–1.4 % Streuung je Level.** Die Gegenprobe
trennt die beiden also sauber. OP16 bleibt **Test**-OP: Gegenprobe, nie
Stützstelle.

### Nebenbefund: `A = 0.0206 m²` ist die Wandfläche des Gitters selbst

Nachgerechnet aus der am 22.09. gemessenen Geometrie:
`0.198094368 × 0.104431991 = 0.0206874 m²`. Die im Werkzeug benutzte `0.0206`
liegt **0.42 % darunter**. Also ist `A` keine geratene Größe, sondern genau
die y × z-Fläche, durch die das Modell kühlt — und damit zu `UCurve`
konsistent, weil Kalibrierung und Wandterm dieselbe Fläche benutzen.

> Die 0.42 % liegen **unter der Streuung je Flusslevel** (1.1–2.1 %), ein
> neuer Lauf lohnt dafür nicht. Aber die Zahl gehört nachgezogen, wenn
> `balance_check.py` das nächste Mal ohnehin läuft — es ist derselbe
> Konstantentyp, der am 22.09. schon bei der Spannweite falsch war.

**Blockiert nichts mehr:** `data_raw/` liegt vor.

---

# Stufe 2 — Die vier Größen in den Cache

Unverändert: `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`, `mdot` in
`generate_cache.py` und `opbundle_contract.md`, `schema_version` hoch, alle
siebzehn OPs neu bauen (10–30 min).

**Dazu neu:** `total_w` gehört mit hinein, falls 1a sich bestätigt — sonst ist
O17 nie messbar.

⚠ **Es sind siebzehn OPs, nicht sechzehn.** Im Cache liegen OP01–OP16 **plus
OP19** (der Report-only-OP aus O11). Wer sechzehn neu baut, lässt OP19 auf dem
alten `schema_version` zurück — und merkt es erst, wenn Stufe 6 ihn mitrollen
will. Gemessen am 22.09.

⚠ **Der Cache-Umbau macht die Gittergleichheit wieder ungeprüft.** Am 22.09. ist
gemessen, dass `xyz` über alle siebzehn OPs bitgleich ist; `load_ops` nimmt sie
aber weiterhin unbesehen aus `raw[0]`. Nach dem Rebuild gehört der Vergleich
als Prüfung in `load_ops` — drei Zeilen, und sie fangen genau den Fehler, den
`grid.derive_layout` nicht sehen kann.

**Tor:** `profile_report`, `coverage_report` und `energy_balance_report` müssen
**exakt dieselben Zahlen** liefern wie vorher.

---

# Stufe 3 — Physik ohne Netz  ✅ gebaut

**Gebaut:** [`solve.py`](solve.py), [`physics.py`](physics.py),
[`grid.py`](grid.py) — ein klassischer expliziter Euler auf dem 3 × 11 × 11:

```
T_{t+1} = T_t + dt * ( Fo : grad² T + Qsrc )
```

Kein gelerntes Gewicht. Der Wert liegt in vier Dingen, und alle vier fielen
weg, wenn man diese Stufe überspringt und gleich das Netz baut:

* **Es prüft Padding, Stencil und Wandterm unabhängig vom Lernen.** Ist der
  Kreuzterm falsch oder das Padding verdreht, sieht man es hier — und nicht
  drei Wochen später als „das Netz konvergiert schlecht".
* **Es liefert die Physik-Latte.** Verglichen wird sonst gegen `persistence`
  und `train-mean`; beide sind trivial. „Reine Wärmeleitung mit kalibrierter
  Kühlwand" ist eine *ernsthafte* Latte.
* **Es beantwortet die CFL-Frage empirisch.** Läuft der Löser bei
  `subsample_time: 2` (dt = 0.2 s gegen Δt_max 0.241 s) stabil?
* **Es ist der Rest-Definitionspunkt.** Was der Löser *nicht* trifft, ist genau
  das, was `g_θ` lernen muss. Damit ist die Aufgabe des Netzes definiert statt
  geraten — und weil `model.py` denselben Operator benutzt, ist es dieselbe
  Zahl und nicht eine ähnliche.

### ⚠ Er läuft heute adiabat, und das ist keine Latte

Der Wandterm ist gebaut und getestet, aber ohne `U(V̇)` nicht kalibrierbar.
`solve.rollout` trägt das als Notiz mit und `SolveResult.summary()` schreibt
„Wandterm AUS (adiabat)". **Das ist eine Ablation und darf nicht als Latte
zitiert werden.** Blockiert durch Stufe 2.

### Was der Rangtest an dieser Stufe geändert hat

Nichts. Sie ist Physik, nicht Architektur — das galt schon, als der ROM der
Plan war, und es gilt jetzt genauso.

### Das Tor

| Ergebnis | Folge |
|---|---|
| schlägt `train-mean` auf den ausgehaltenen OPs | 🟢 der Unterbau stimmt, weiter zu Stufe 4 |
| stabil, aber schlechter als `train-mean` | 🟡 normal — der Löser kennt die Materialdaten nur genähert. Weiter, aber der Rest ist groß |
| divergiert | 🔴 entweder CFL (dann `subsample_time: 1`) oder ein Vorzeichenfehler im Padding. **Nicht mit dem Netz übertünchen** |
| schlägt schon `PINNmodulusTwo` (6.270 / 3.585 °C) | 🔴🟢 dann lief das Lernen bisher gegen fehlende Physik, und der Plan wird ein anderer |

---

# Stufe 4 — Der CNN  ✅ gebaut, nicht gemessen

**Gebaut:** [`model.py`](model.py), [`train.py`](train.py), 46 Tests.

```
f  =  L(T_t)  +  Qsrc_t  +  g_θ(X_t)
      ^feste Physik         ^gelernte Korrektur, 11 427 Parameter
```

`L` **ist** `physics.anisotropic_laplacian` — derselbe Operator wie in Stufe 3,
keine zweite Fassung davon.

| | |
|---|---|
| Eingang `X_t` | **44 Kanäle** à 11 × 11: 9 Zustand/Historie + 17 statische Karten + 18 gebroadcastete Treiber |
| Ausgang | 3 × 11 × 11 — eine Änderungsrate je x-Ebene |
| Stapel | 44 → 16 → 16 → 16 → 3, alles 3×3, `reflect`-gepaddet |
| Verlust | `L = w_data·L_data + w_phys·L_phys + w_wall·L_wall` — **kein `L_bc`** |

### Die zwei Rollen der x-Achse

Der Punkt, an dem der Entwurf am leichtesten misszuverstehen ist:

* **im Physik-Term `L`** ist x eine **echte Achse** mit Geisterschichten —
  Symmetrie bei x = 0, Kühlwand bei x = 0.0219, nicht-äquidistanter
  Dreipunktstern dazwischen.
* **in der Korrektur `g_θ`** ist x in die **Kanäle** gefaltet. Drei Ebenen sind
  zu wenig, um darüber zu falten.

Kein Widerspruch: der Stencil braucht die Nachbarn in x, der Faltungskern nicht.

### Zwei Entscheidungen, die nicht kosmetisch sind

**`head` startet auf exakt null.** Damit ist `g_θ` beim ersten Schritt null und
das Modell rechnet **Bit für Bit** den Löser aus Stufe 3. Das löst den Satz
*„f korrigiert den Löser, es ersetzt ihn nicht"* ein, statt ihn zu behaupten —
`test_bei_init_rechnet_das_modell_exakt_die_physik` prüft es ohne Toleranz.

**Jede Faltung padded `reflect`, nicht `zeros`.** Mit dem Default sähe der Kern
am Rand eine erfundene Null, im z-Score also eine Temperatur von `T_mu` — eine
Randbedingung, die niemand beschlossen hat.

### Die Ablation steht als Erstes an

Drei Konfigurationen, die sich in genau einer Sache unterscheiden:

| | die Rate `f` | Eingang | `w_phys` | CLI | misst |
|---|---|---|---|---|---|
| **A** | `g_θ` allein | 44 Kanäle | 0 | `--no-physics` | reine Blackbox — die Latte |
| **B** | `L+Q+g_θ` | 44 Kanäle | 0 | (Vorgabe) | was die Physik **in der Architektur** bringt |
| **C** | `L+Q+g_θ` | 44 Kanäle | > 0 | `--w-phys 0.1` | was der Strafterm **obendrauf** bringt |
| **D** | `L+Q+g_θ` | **42** — ohne `y_map`/`z_map` | 0 | `--no-coord-maps` | ob der Ortsprior aus §2b **echt** ist |

A → B ändert die Architektur bei gleichem Verlust, B → C den Verlust bei
gleicher Architektur. Jede Differenz ist einem einzigen Eingriff zuzuordnen.

> ### ⚠ Parallel läuft hier **noch nichts** — und das ist eine Bremse, keine Fußnote
>
> Vier Arme × mindestens drei Seeds sind **zwölf Läufe**. Das ist genau die
> Form, für die `PINNmodulusTwo/sweep.py` gebaut wurde und dort **3.69× mit
> MPS** bringt. Für `GridCNN` steht davon nichts bereit, aus drei Gründen, die
> alle zuerst wegmüssen:
>
> 1. **`GridCNN` hat kein `sweep.py`**, und `PINNmodulusTwo/sweep.py` kann hier
>    nicht einspringen: `TRAIN_PY = THIS_DIR / "train.py"` zeigt fest auf den
>    PINN.
> 2. ~~Keine Artefakt-Trennung.~~ ✅ **22.09. erledigt** — jeder Seed
>    schreibt nach `<artifacts-dir>/<Arm>/seed<N>/`.
> 3. ~~`--seeds` dreht keine Schleife.~~ ✅ **22.09. erledigt** — `--seeds N`
>    fährt N Läufe und meldet Mittel ± `std(ddof=1)`. Vorher war es eine
>    Warnschwelle: wer `--seeds 3` tippte, bekam **einen** Lauf und glaubte,
>    er habe drei — dieselbe Klasse stiller Zusage wie `xyz = raw[0]`.
>
> **Es fehlt also nur noch der Treiber.** Der Ladepfad steht, die Trennung
> steht, die Seeds laufen. Ein `-j` über Arme × Seeds ist jetzt Fleißarbeit
> und kein Umbau mehr — und `sweep.py` bleibt trotzdem der falsche Ort dafür,
> weil `TRAIN_PY` dort fest auf den PINN zeigt.
>
> **MPS gehört trotzdem an** — er kostet einen Einzellauf nichts und ist dann
> schon da. Für den Cache-Rebuild ändert er allerdings **gar nichts**: der ist
> pandas auf CSVs, also CPU und Platte. Dort hilft `generate_cache.py -j`,
> nicht MPS.

> ### Arm D — neu am 22.09., ✅ gebaut
>
> **Die Frage:** README §2b führt als Vorteil des CNN an, dass er Position
> *nicht* auswendig lernen **kann** und räumliche Struktur deshalb über die
> Materialkarten begründen **muss**. [`model.py:193`](model.py) gibt ihm die
> Position trotzdem — zwei z-gescorte Koordinatenkarten, mit dem Kommentar,
> ohne sie wäre *„jede Randzelle von jeder Mittelzelle ununterscheidbar"*.
> Beide Texte gehen von derselben Prämisse aus und ziehen den entgegengesetzten
> Schluss. **A/B/C berühren das nicht — alle drei tragen die Karten.**
>
> **Was D ändert:** genau zwei Kanäle von 44. 11 139 Parameter statt 11 427
> (2 × 16 × 9 = 288 weniger in der ersten Faltung). Sonst identisch — ein Test
> hält fest, dass die fünfzehn Materialkarten **bitgleich** bleiben.
>
> | Ergebnis | Folge |
> |---|---|
> | **D ≈ B** | 🟢 der Ortsprior ist echt, §2b stimmt, und der CNN steht dort wirklich anders als das MLP |
> | **D deutlich schlechter** | 🔴 der CNN hat dieselbe Positionskrücke wie das MLP. §2b beschreibt dann einen Entwurf, den der Code nicht umsetzt — und §11.1 ist beantwortet, nur anders als erhofft |
>
> ⚠ **Und D ist ein enger Test.** Am 22.09. gemessen: `region`/`rho`/`Cp` sind
> je x-Ebene konstant, `lam` variiert nur in **zwei Zeilen am unteren y-Rand**
> (22 von 121 Punkten), und der Kontrast in der Ebene ist nach dem z-Score rund
> **zwölfmal schwächer** als der zwischen den Ebenen. D fragt also, ob dieses
> schmale Band reicht. Fällt D durch, ist damit **nicht** gezeigt, dass ein
> Ortsprior unmöglich wäre — nur, dass dieser Datensatz ihn nicht hergibt.

> **Beide starten auf etwas Sinnvollem.** Weil `head` null ist, startet **A**
> bei `persistence` (Rate null) und **B** bei der Physik. Keine der beiden
> startet bei Rauschen, und das hält den Vergleich fair.

⚠ Die Seed-Streuung liegt bei 0.518 / 0.882 °C, auf OP06 bis 1.63 °C. Ein
Unterschied unter ~1 °C ist mit drei Seeds **nicht lesbar** — jede
Konfiguration braucht eine Seed-Schleife. `train.py` meckert bei `--seeds < 3`.

### Die Größenachse

| | Parameter | |
|---|---|---|
| `--width 16 --blocks 3` | **11 427** | Vorgabe, nach Tor 0 |
| `--width 24 --blocks 3` | 20 595 | Gegenprobe |
| `--width 64 --blocks 4` | **137 923** | der Entwurf aus der Präsentation |

Die Achse prüft, ob größer überhaupt etwas bringt. Bringt sie nichts, ist das
kein Nullergebnis, sondern eine Bestätigung des Rangtests.

### Das Tor

Schlägt Stufe 3 auf den fünf ausgehaltenen OPs. Wenn nicht, hat das Netz nichts
beigetragen — und dann ist der Rangtest die erste Stelle, an der man nachsieht.

---

# Stufe 5 — Truncated BPTT

Der Hebel auf **O13** (OP06: 6.270 °C im Mittel, 13.248 °C spät). Fenster von
`k` Schritten, Gradient durch alle `k`, `detach` am Fensterende.

`k` = 50 zuerst, dann 200. Aktivierungsspeicher ist grob 31k Floats je Schritt
(README §4), also ist 200 machbar.

> ### ⚠ Warum das die *einzige* Änderung ist, die O13 erreicht
>
> `train.py` rollt je Epoche und OP einmal frei aus, friert die Trajektorie ein
> und trainiert darauf Ein-Schritt-Paare. Der Gradient überquert die Historie
> also **nie** — er sieht genau einen Schritt. Damit kann kein noch so langes
> Training den Spätfehler strukturell erreichen: er entsteht aus der
> Akkumulation über hunderte Schritte, und über die wird nicht optimiert.
>
> `test_der_gradient_ueberquert_die_historie_nicht` hält das fest, am Verhalten
> statt an der Absicht. Wer diese Schleife anfasst, muss ihn brechen sehen,
> bevor er Stufe 5 unmöglich macht.

**Tor:** `late_mae` fällt, bei mindestens gleichem `mae`. Fällt nur `mae` und
`late_mae` nicht, hat BPTT nicht getan, wofür es da ist.

---

# Stufe 6 — Der Vergleich

Derselbe Split (train 11 / val OP06+OP09 / test OP13, OP15, OP16), dieselben
`op_metrics`, dieselben trivialen Vorhersager, **plus** die Physik-Latte aus
Stufe 3. Über mehrere Seeds — *ein
Seed ist keine Streuung.*

OP19 wird mitgerollt: berichten, nie trainieren, nie selektieren (O11).

---

## Was übernommen wird — importiert, nicht kopiert

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PINNmodulusTwo"))
import data, op_registry, op_metrics
```

**Kein Modulus.** GridCNN braucht `FCLayer` nicht; reines PyTorch. Läuft
trotzdem in `modulus_env`, weil Torch dort schon liegt.

---

## Was NICHT gebaut wird

Damit der Umfang nicht wandert:

* **ConvGRU / ConvLSTM.** Erst wenn Stufe 5 steht. Ein unbeschränkter
  versteckter Zustand bei elf Trajektorien ist ein echtes Risiko.
* **FiLM-Konditionierung.** Die Treiber werden erst gebroadcastet. FiLM ist eine
  eigene Sweep-Achse, keine Architekturentscheidung.
* **FNO / DeepONet / Neural ODE.** Bei 11 × 11 Overkill.
* **Das ROM.** Am 15.09. gestrichen — der CNN ist die Entscheidung. Der
  Rangtest, der dafür gesprochen hätte, bleibt gemessen und in der
  Stand-Tabelle stehen; er ist die erste Stelle, an der man nachsieht, wenn
  Stufe 4 nicht trägt.
* **Plots, Resume, Checkpoint-Merge.** Zuletzt, und nur für Achsen, die
  wirklich Stunden laufen.

---

## Abbruchkriterien

* **Divergenz in Stufe 3 wird nicht mit dem Netz übertünchert.**
* **Kein `L_bc` durch die Hintertür.** Braucht das Modell einen Strafterm für
  die Symmetrie, ist die Basis falsch aufgebaut.
* **`Q̇` und `T_fluid_out` gehen nie als Modelleingang rein.** Aufsicht und
  Gegenprobe, nichts anderes.
* **`U` wird nie auf OP13/OP15/OP16 kalibriert.** Trainierte Flusslevel sind
  0 / 15 / 30; die 90 aus OP16 sind die Gegenprobe.
* **Ein Seed ist keine Streuung.**

---

# Erledigt

| Datum | was |
|---|---|
| 02.09. | Gitter verifiziert: 3 × 11 × 11, äquidistant, über alle OPs identisch, Randring **auf** der Flächenkante |
| 02.09. | Randbedingungen geklärt; Quellenseite als Halbmodell bestätigt (`V_JR1 = 4.394793e-04 m³`) |
| 02.09. | `A = 0.0206 m²`, Koeffizient als `U` (Gesamtdurchgang) etikettiert |
| 02.09. | `tools/spatial_rank.py` und `tools/balance_check.py` gebaut und getestet |
| 02.09. | `L_phys` bleibt — ein Seed ist keine Streuung (README §12.1) |
| 02.09. | F4–F8 entschieden (README §12.3) |
| 02.09. | Kritik am Versuchsplan: [`README_OPS.md`](README_OPS.md). Schwerster Befund: `T0 = T_fluid` in 11 von 11 |
| 02.09. | O16 im PINN-Fahrplan eingetragen; PR #31 gemergt |
| **09.09.** | **Stufe 0 gelaufen — ROT.** 4 Moden bei 99.9 %. Folge: der Stapel wird auf 16 × 3 verkleinert |
| **09.09.** | **Stufe 1, erster Lauf.** `jr2/jr1 = 1`, `U`-Verhältnis 1130 : 50 plausibel. Mein ṁ = 0-Torkriterium war falsch und ist gestrichen |
| **09.09.** | Werkzeug korrigiert: drei Flusslevel statt einem, `T_in`-Rückfallkette, `Q_ht/tot` |
| **14.09.** | **Der Gitter-Unterbau steht.** `grid.py`, `physics.py`, `solve.py`, `benchmark.py`, 45 Tests, CI grün |
| **14.09.** | Stencil-Ordnungen gemessen, beide Randzusagen **exakt** null, Route R6 beziffert |
| **15.09.** | **Entscheidung: CNN statt ROM**, 16 × 3 = 11 427 Parameter. Der ROM-Plan ist gestrichen |
| **15.09.** | `model.py` und `train.py` gebaut, 46 Tests. Bei Init rechnet das Modell **exakt** den Löser |
| **15.09.** | Zwei Funde aus Tests: 64 × 4 sind **137 923** statt „~100 k"; ein konstanter Materialkanal wurde zu ±1 statt 0 |
| **22.09.** | **Gittergleichheit gemessen statt angenommen.** `xyz` ist über alle **siebzehn** Cache-OPs bitgleich, gleiche Reihenfolge, `layer` ebenso — `data.py:653` nahm das bisher unbesehen aus `raw[0]` |
| **22.09.** | **Die dokumentierte Spannweite war falsch.** 0.198089 × 0.104441 an sieben Stellen; gemessen sind 0.198094368 × 0.104431991 m. Cache und Legacy-CSVs stimmen exakt überein |
| **22.09.** | **Der Geometrie-Test war zirkulär** und hätte gegen die echten Koordinaten in allen vier Vergleichen gefallen. `test_layout_aus_den_echten_koordinaten` schließt den Kreis |
| **22.09.** | **Materialverteilung gemessen:** `region`/`rho`/`Cp` je x-Ebene konstant; `lam` variiert, aber nur in zwei Zeilen am unteren y-Rand (22 von 121 Punkten) |
| **22.09.** | **§2b widerspricht `model.py:193`** — die zwei Koordinatenkarten geben dem Kern die Position, die §2b ihm abspricht. Offen, Route R8 |
| **22.09.** | **`main` war seit dem 17.09. rot** und niemand hat es bemerkt: `pytest (GridCNN)` meldete auf `163b21c` `2 failed, 97 passed`, und der Benchmark-Schritt dahinter wurde seither übersprungen. PR #40 erbt dieses Rot — seine vier Dateien liegen alle in `PINNmodulusTwo/`, dessen Suite grün ist |
| **22.09.** | **Ablationsarm D gebaut** — `--no-coord-maps`, 42 statt 44 Kanäle, 11 139 Parameter. Sieben Tests, darunter einer, der die fünfzehn Materialkarten bitgleich festhält |
| **22.09.** | **Die Gittergleichheit ist jetzt geprüft statt angenommen**: `data._assert_shared_geometry` fällt, wenn ein OP eine andere Punktreihenfolge hat. Test in beide Richtungen, mit reiner Zeilenvertauschung |
| **22.09.** | PR #40 (Achse 1, O18, MPS-Prüfung) in diesen Branch hereingeholt — ein PR statt zwei |
| **22.09.** | **Stufe 1, zweiter Lauf.** Fluidbilanz **1.030 … 1.062** → `ghost_hi` steht. `Q_ht/tot = 0.700 … 0.772`, **nicht ≈ 1** — die Hypothese aus 1a ist widerlegt. `tot/jr1 = 2.99 … 3.44` belegt **O17** |
| **22.09.** | **`Q_ht/jr1` ist keine Konstante** — 2.29 bei V̇ = 0.0013 gegen 2.53 bei 0.0026. Eine Bezugsfläche kann nicht flussabhängig sein, also ist das Tor „Q halbieren oder A verdoppeln" falsch gestellt. Siehe 1d |
| **22.09.** | **`balance_check.py` Abschnitt 4 abgestürzt und repariert:** das Skript las die Zeitachse von zwei der fünf CSVs und legte den Rest stillschweigend darauf. In Abschnitt 2 hat numpy das still gebroadcastet — richtige Zahl, falscher Grund |
| **22.09.** | **`U` je Flusslevel gemessen:** 50.3 / 228.7 / 326.4 W/m²K, je Level **1.3–2.1 %** Streuung über OPs mit verschiedener C-Rate, SOC und Starttemperatur. `U` hängt am Fluss und an sonst nichts |
| **22.09.** | **Die `~1130` vom 09.09. ist zurückgezogen.** Sie war der `T_out`-Bezug, in den das Werkzeug fiel, weil es `T_in` nicht fand — nachgerechnet ergibt er 1128.5 aus `U(T_in) = 229.25` und `dT_fluid = 7.108`. Damit fällt auch „Faktor ~23" aus 1b. Siehe 1e |
| **22.09.** | **Die Bezugstemperatur ist der größere Hebel als die Halbmodellfrage:** `U` gegen `T_in` / `T_mittel` / `T_out` steht wie 1 : 1.7 : 4.9. `UCurve` verlangt `T_mittel`, das Werkzeug rechnete gegen `T_in` — beide Schranken stehen jetzt nebeneinander in der Tabelle |
| **22.09.** | **Stufe 1 abgeschlossen. `U(T_mittel)` = 50.4 / 421.0 / 501.7 W/m²K** für V̇ = 0 / 15 / 30, Streuung 2.10 / 1.38 / 1.09 %. Das sind genau die Stützstellen, die `UCurve` braucht — **der Wandterm ist kalibrierbar, null freie Parameter** |
| **22.09.** | **`U(T_out)` ist unbrauchbar, nicht nur eine obere Schranke:** −8684 / +21933 / −12729 bei V̇ = 15. `Tw` ist ein Flächenmittel, `T_out` das heiße Kanalende — die Differenz wechselt das Vorzeichen. Bei V̇ = 30 wechselt sie *nicht*, und die Default-OPs vom 09.09. waren genau diese zwei. Deshalb sah der Fehler damals wie eine Messung aus |
| **22.09.** | **Die Kurvenform bleibt offen und blockiert nichts.** Zwei Konvektionspunkte, `U ∝ ṁ^0.253`. OP16 trennt Potenzgesetz (662) von Serienwiderstand (591) — 12 % gegen 1.1–1.4 % Streuung, also eine scharfe Gegenprobe |
| **22.09.** | **`A = 0.0206 m²` ist die y × z-Fläche des Gitters selbst** — gemessen 0.0206874, benutzt 0.0206, 0.42 % darunter. Unter der Streuung, aber nachzuziehen |
| **22.09.** | Beide fallenden Tests repariert: die float32-Schranke im gepaddeten Rollout (1.2e-6 gegen 1e-6, hielt zufällig) und **`torch.equal` im float64-Vergleich** — Bitgleichheit zwischen zwei Batchgrößen ist durch nichts garantiert und war nicht portabel (lokal grün, auf dem Runner rot). Jetzt beide gegen eine relative Schranke `1e-12` |
| **22.09.** | **Stufe 2 ist durch — Tor grün.** `generate_cache.py --all` hat **17/17** OPs auf Schema v3 gebaut; `profile_report`, `coverage_report` und `energy_balance_report` liefern danach **zeichengleich** dieselben Zahlen (`diff` leer). Der Wandpfad liegt im Cache, ohne dass sich etwas Bestehendes bewegt hat |
| **22.09.** | **Der Ladepfad ist angeschlossen.** `train.py` trainiert — `lade_datensatz` zieht die Bündel über `PINNmodulusTwo/data.py`, `layout_aus_bundle` leitet den Reshape aus `bundle.xn` (float64) ab, `statics_aus_bundle` holt `lam` aus `materials`, nicht aus `Fo` zurückgerechnet. Alle vier Arme A/B/C/D laufen durch; **11 427 Parameter** wie geplant |
| **22.09.** | **Die Gitterabbildung ist punktweise geprüft**, nicht angenommen: alle 363 Punkte, für `tn_seq` *und* für den 3×3-Tensor `fo` — der trägt seine Matrixachsen hinten und stünde bei naivem `to_field` auf dem Kopf |
| **22.09.** | **`--seeds` dreht jetzt eine Schleife.** Vorher war es eine Warnschwelle: wer `--seeds 3` tippte, bekam **einen** Lauf. Jede Wiederholung schreibt in ihr eigenes `--artifacts-dir`, und die Zusammenfassung nennt Mittel **und** `std(ddof=1)` |
| **22.09.** | **CFL wird vor dem Lauf ausgerechnet, nicht danach gerätselt.** Auf dem synthetischen Fixture liegt der Schritt selbst bei `subsample 1` **53×** über der Schranke (1.9 ms gegen die echten 0.241 s) — die Ersatz-Materialdaten machen das Problem viel steifer, als es ist. Arm A ist davon unberührt, B/C/D laufen dort weg. **Auf der Rechenmaschine mit echten Materialdaten ist das eine andere Zahl** |
| **22.09.** | 🔴 **Konfiguration A gelaufen — und es ist kein Ergebnis.** val-MAE **27.11 ± 22.26 °C** (OP06) über drei Seeds, bester gegen schlechtesten **Faktor 5.8**. Die Schwelle liegt bei ~1 °C. Der beste Seed unterbietet die Mittelwert-Vorhersage (~7.7 °C) **nicht** |
| **22.09.** | **Die Ursache ist der Horizont, nicht das Netz:** ein Schritt trainiert, ~8040 frei laufend gemessen. Bis **99.8 %** aller OP-Zeitschritte an `--clamp` (= ±480 °C bei `T_sigma` 9.602). Eine Sättigungsphase vergiftet die nächste Epoche, weil die Historie aus dem eingefrorenen Rollout kommt |
| **22.09.** | **`data` sagt das Ergebnis nicht vorher:** Seed 0 endet bei 0.332 → 51.95 °C, Seed 2 bei 0.305 → 8.95 °C. Derselbe Verlust, Faktor 5.8 |
| **22.09.** | **Stufe I gebaut — Messung repariert, Experiment unberührt:** val-MAE je `--val-every` Epochen statt nur am Schluss (der FAHRPLAN verbot „die letzte Zeile ablesen" schon für den PINN), `model_best.pt`, Sättigung als **Anteil** (`88248/88400 = 99.8 %`), und **triviale Latten** vor jedem Lauf |

## Stand

| Stufe | Kriterium | gemessen | Datum |
|---|---|---|---|
| 0 | Gitterprobe 3×11×11 | **bestätigt** | 09.09. |
| 0 | **gepoolte Ortsstruktur** (90/99/99.9/99.99 %) | **1 / 2 / 4 / 6 Moden** | **09.09.** |
| 0 | Tor | 🔴 **≤ 5 Moden** — überstimmt, Folge ist 16 × 3 | 09.09. |
| 1 | `jr2/jr1` | **1.000** — zwei gleiche Wickel | 09.09. |
| 1 | `Q_ht/jr1` spät, mit Fluss | **2.286 … 2.566** — und **flussabhängig**, also kein Konventionsfaktor. Siehe 1d | **22.09.** |
| 1 | `Q_ht/tot` | **0.700 … 0.772** mit Fluss, 0.11 ohne — **nicht ≈ 1**. Sättigt gegen ~0.8: es wird noch gespeichert | **22.09.** |
| 1 | `tot/jr1` | **2.99 … 3.44** — ⅓ bis 40 % der Erzeugung liegt außerhalb der Wickel. **O17 belegt** | **22.09.** |
| 1 | Fluidbilanz-Verhältnis | ✅ **1.030 … 1.062** — 🟢 `ghost_hi` steht. `T_in` aus `Input Signale.csv` | **22.09.** |
| 1 | Wandanteil bei ṁ = 0 | **≈ 0.27** — physikalisch richtig (Fluidkapazität), Kriterium war falsch | 09.09. |
| 1 | ~~`U` mit Fluss / ohne, Faktor ~23~~ | ~~**~1130 / ~50**~~ — **zurückgezogen 22.09.**: die 1130 war der `T_out`-Bezug, nachgerechnet 1128.5. Siehe 1e | 09.09. |
| 1 | `U(T_in)` je Flusslevel | **50.3 / 228.7 / 326.4 W/m²K** — je Level **1.3–2.1 %** Streuung über verschiedene Treiber | **22.09.** |
| 1 | `U` hängt nur von V̇ ab? | 🟢 **ja** — das ist der Teil des Tores, der hält | **22.09.** |
| 1 | **`U(T_mittel)`, wie `UCurve` es will** | ✅ **50.4 / 421.0 / 501.7 W/m²K** für V̇ = 0 / 15 / 30, Streuung 2.10 / 1.38 / 1.09 % | **22.09.** |
| 1 | `U(T_out)` | **unbrauchbar** — Vorzeichenwechsel, −8684 … +21933 bei V̇ = 15. Bei V̇ = 30 *nicht*, daher die brave „~1130" vom 09.09. | **22.09.** |
| 1 | **Wandterm kalibrierbar?** | 🟢 **ja** — die drei trainierten Level *sind* die Stützstellen von `UCurve`, null freie Parameter | **22.09.** |
| — | `A` gegen die gemessene Geometrie | `0.198094368 × 0.104431991 = 0.0206874 m²`; benutzt wird 0.0206, **0.42 % darunter** | **22.09.** |
| 1 | `U(V̇)`-Kurven*form* | 🟡 **nicht entscheidbar, blockiert aber nichts** — zwei Konvektionspunkte, `U ∝ ṁ^0.253`. OP16 trennt Potenzgesetz (662) von Serienwiderstand (591): **12 %** gegen 1.1–1.4 % Streuung | **22.09.** |
| 2 | Reports unverändert | | |
| — | Reshape aus Koordinaten ableitbar und umkehrbar | **ja**, 0.198094368 × 0.104431991 m | 14.09. / korrigiert 22.09. |
| — | `xyz` über alle siebzehn Cache-OPs identisch | **ja, bitgleich**, gleiche Reihenfolge | 22.09. |
| — | Cache-Koordinaten = Legacy-CSVs | **ja, exakt** | 22.09. |
| — | Äquidistanz in y/z im Cache | **bis float32**: 4e-7 relativ gegen `_uniform`-Toleranz 1e-6, Reserve 3.7× | 22.09. |
| — | `region`/`rho`/`Cp` in der Ebene | **konstant** je x-Ebene | 22.09. |
| — | `lam` in der Ebene | **variabel**, aber nur zwei Zeilen am unteren y-Rand; 99/121 Punkte ein Wert | 22.09. |
| — | Kontrast in der Ebene gegen zwischen den Ebenen (nach z-Score) | **Faktor ~12 schwächer** | 22.09. |
| — | Kanäle, die in der Ebene variieren | **17 von 44** (9 Zustand + 6 `lam` + 2 Koordinaten) | 22.09. |
| — | davon strukturell flache Gewichte in Schicht 1 | **3 456 von 11 427 = 30 %**, effektiv ≈ 7 971 | 22.09. |
| — | `dT/dx` an der Symmetrieebene | **exakt `0.0`**, ohne Toleranz | 14.09. |
| — | zentrale Differenz am y/z-Rand | **exakt `0.0`** | 14.09. |
| — | Stencil-Ordnung d²/dy², d²/dz² | **1.98 / 1.98** | 14.09. |
| — | Stencil-Ordnung d²/dx² (gestreckt) | **1.28** — richtig für den nicht-äquidistanten Dreipunktstern | 14.09. |
| — | Bilanztreue des Sterns | **3.6 % Drift, sättigt** (R6) | 14.09. |
| 3 | Löser stabil bei dt = 0.2 s | | |
| 3 | Physik-Latte, val OP06 / OP09 | | |
| 4 | CNN schlägt Physik-Latte | | |
| 5 | `late_mae` gefallen | | |
| 6 | val / test gegen PINNmodulusTwo | | |

---

# Der Code — Stand 15.09.

> Dieser Abschnitt beschreibt, **was im Repo liegt**. Der Plan oben sagt, was
> gemessen werden soll; hier steht, womit. Das fortlaufende Protokoll der Läufe
> steht in [`BENCHMARK.md`](BENCHMARK.md), zusammen mit den offenen Routen.

## Was steht

| Datei | was sie tut | geprüft durch |
|---|---|---|
| [`grid.py`](grid.py) | Reshape 363 → (3,11,11), **aus den Koordinaten abgeleitet**, plus die drei Paddings | `tests/test_grid.py`, 12 Tests |
| [`physics.py`](physics.py) | nicht-äquidistanter x-Stern, Kreuzterm `λ_xy`, Quelle, Wandterm, `U(V̇)` | `tests/test_physics.py`, 19 Tests |
| [`solve.py`](solve.py) | explizites Euler, entdimensioniert wie `data.py` | `tests/test_solve.py`, 14 Tests |
| [`model.py`](model.py) | **der Faltungsstapel in Δ-Form**, 11 427 Parameter | `tests/test_model.py`, 23 Tests |
| [`train.py`](train.py) | **die Trainingsschleife**, Ein-Schritt gegen eine eingefrorene Trajektorie | `tests/test_train.py`, 23 Tests |
| [`benchmark.py`](benchmark.py) | Stufenläufer, schreibt ins lebende Dokument | CI, Stufe 0 und 2 |

**91 Tests, Sekunden, ohne Modulus und ohne `data_cache`.** Die CI fährt
`GridCNN` und `PINNmodulusTwo` in **getrennten** Invokationen — beide Projekte
haben ein Modul namens `physics`, und in einem gemeinsamen Lauf gewänne das
erste. Das ist nachgeprüft und nicht nur vermutet: ein gemeinsamer Lauf bricht
beim Import ab.

### Die vier Zusagen, die bewiesen sind statt behauptet

1. Der Reshape wird **abgeleitet**, nicht geraten, und trifft die **gemessene**
   Geometrie: Spannweite 0.198094368 × 0.104431991 m, `dx = 10.785542 / 11.114458 mm`.

   > ⚠ **Diese Zusage war bis zum 22.09. keine.** Der Test, der sie prüfte, las
   > dieselben Konstanten, aus denen `conftest.py` sein Testgitter baut — er
   > konnte nicht fallen. Und die Konstanten waren falsch: gegen die echten
   > Koordinaten wären alle vier Vergleiche durchgefallen (`dx` um 4.6e-7, die
   > Spannweiten um 5.4e-6 bzw. 9.0e-6). Seit dem 22.09. leitet
   > `test_layout_aus_den_echten_koordinaten` das Gitter direkt aus den drei
   > CSVs in `legacy/.../coordinates/` ab — den Dateien, die im Repo liegen und
   > mit dem Cache aller siebzehn OPs bitgleich sind. **Erst dieser Test macht
   > die Zusage falsifizierbar.**
2. `dT/dx` an der Symmetrieebene ist **exakt `0.0`** — ohne Toleranz geprüft,
   über Feldskalen von 1e-8 bis 1e8. Im Zähler steht `T₁ − T₁`.
3. Die zentrale Differenz am y/z-Rand ist **exakt `0.0`**. `reflect`, nicht
   `replicate`.
4. **Bei Initialisierung rechnet `model.py` Bit für Bit `solve.py`.** `head` ist
   null initialisiert, also ist `g_θ` null. Das ist die Einlösung von
   *„f korrigiert den Löser, es ersetzt ihn nicht"*.

### Drei Befunde, die beim Bauen abgefallen sind

**Der Stern ist nicht bilanztreu** (Route R6). Rollt man adiabat aus, driftet
das Mittel um **3.6 % der Feldstreuung** und **sättigt** dann; das Feld läuft
sauber gegen eine Konstante (`std` → 4e-5). Ursache sind zwei bewusste
Entscheidungen: die nicht-konservative Form `Fo : ∇²T` und knotenzentriertes
`reflect` ohne Halbzellgewichte. Dass die Drift sättigt, ist der Beleg, dass
kein Rand *echte* Energie leckt — sie gehört aber **neben** die Physik-Latte
geschrieben, nicht darunter versteckt.

**64 × 4 sind 137 923 Parameter, nicht „~100 k."** Siehe den Kasten ganz oben.

**Ein konstanter Materialkanal wurde zu ±1 statt zu 0.** In float32 ist
`ρ·Cp = 2.25e6` überall gleich, der Mittelwert trägt aber einen Rundungsfehler
von O(0.25) — und dieser Rest geteilt durch eine genauso winzige Streuung ergibt
Rauschen mit Standardabweichung 1. Das Netz hätte es nicht von echter Struktur
unterscheiden können. Jetzt wird in float64 gescort, der konstante Fall
abgefangen, und tote Kanäle werden **gemeldet** statt still zu bleiben.

## Was fehlt

| | was | blockiert durch |
|---|---|---|
| ~~**Ladepfad**~~ | ~~`train.py` an `PINNmodulusTwo/data.py` anschließen~~ | ✅ **erledigt 22.09.** — `lade_datensatz`, `op_tensoren`, `val_mae`, Seed-Schleife. Gegen das synthetische Fixture rauchgetestet, alle vier Arme laufen. `op_metrics` ist noch nicht angeschlossen: die val-MAE kommt heute aus dem freilaufenden Rollout gegen `tn_seq × T_sigma` |

> **Was der Ladepfad schon vorfindet** (22.09. gebaut, mit Tests):
> `train.modell_kwargs(args)` übersetzt die Ablationsflags an **einer** Stelle
> in die Argumente von `build_static_maps` und `GridCNN` — er muss sie nur noch
> aufrufen. Und `derive_layout` gehört mit `bundle.xn` (float64) × `L_ref`
> gefüttert, **nicht** mit `op.xn`: der ist float32 und halbiert die Reserve der
> Äquidistanzprüfung (4e-7 gemessen gegen `rtol` 1e-6).
| ~~**Wandterm benutzbar**~~ | ~~`U(V̇)` kalibrieren~~ | ✅ **erledigt 22.09.** — `U(T_mittel)` = 50.4 / 421.0 / 501.7 W/m²K für V̇ = 0 / 15 / 30, die drei trainierten Level. Siehe 1e |
| **Wandterm im Training** | die vier Größen aus dem Bündel ziehen | **Stufe 2**: `q_solid_to_fluid`, `mdot`, `cp_fluid`, `fluid_out_temp` fehlen im Cache — `U` ist kalibriert, der *Weg dorthin* fehlt |
| **Physik-Latte** | Stufe 3 mit echter Wand | Wandterm |
| **`L_wall`** | der einzige Verlustterm mit gemessenem Ziel | Wandterm |
| **`C_fluid`** | Wärmekapazität des Kühlmittels im Kanal, für den Kapazitätsmodus | liegt nicht vor — **nicht raten** |
| ~~**Artefakt-Trennung**~~ | ~~`--artifacts-dir` je Lauf~~ | ✅ **erledigt 22.09.** — jeder Seed bekommt `<artifacts-dir>/<Arm>/seed<N>/` mit `model.pt`, `history.json`, `metrics.json` |
| ~~**Seed-Schleife**~~ | ~~`--seeds` dreht keine~~ | ✅ **erledigt 22.09.** — `--seeds N` fährt N Läufe mit `seed, seed+1, …` und meldet Mittel ± `std(ddof=1)` |
| **`op_metrics` anschließen** | die Kennzahlen des Basisprojekts (`peak_pred`, `late_bias_frac`) auch für den CNN | nichts — Fleißarbeit. Ohne sie ist **O17 am CNN nicht messbar** |
| **Ein Sweep-Treiber** | `-j N` über Arme × Seeds, wie `PINNmodulusTwo/sweep.py` | nichts mehr. Die Artefakt-Trennung stand im Weg, sie steht jetzt |

⚠ **Der Löser und das Training laufen heute adiabat.** Der Wandterm ist gebaut
und getestet, aber ohne die vier Cache-Größen nicht kalibrierbar.
`solve.rollout` trägt das als Notiz mit, `SolveResult.summary()` schreibt
„Wandterm AUS (adiabat)", und `train._wall_ghost` **wirft**, statt ein `U` zu
raten. Das ist eine **Ablation, keine Latte**, und darf auch nicht als eine
zitiert werden.

## Die Trainingsschleife — und was man an ihr nicht kaputtmachen darf

⚠ **Kein Teacher Forcing, und das wird oft falsch erzählt.**
`train.py` rollt **einmal je Epoche je OP** die *eigene* Trajektorie unter
`torch.no_grad()` aus, gesät nur von der gemessenen Anfangsbedingung, und friert
sie ein. Erst darauf laufen `inner_steps` Ein-Schritt-Updates gegen die Labels.

Der Unterschied zwischen Training und Auswertung ist also **nicht** der
Eingangszustand — beide rollen frei. Er ist:

| | Training | Auswertung |
|---|---|---|
| Trajektorie | eingefroren, je Epoche erneuert | live |
| Labels | ja, als Ziel | nein |
| Gradient | nur **ein** Schritt (Historie detached) | keiner |

**Daraus folgt der ganze Hebel von Stufe 5** — und drei Tests halten es am
*Verhalten* fest statt an der Absicht: eine Störung der Trajektorie an einem
Zeitpunkt, der weder Anker noch Lag ist, darf den Verlust **nicht** ändern;
Anker und beide Lags müssen sehr wohl durchschlagen; und eine Störung der
**Labels** vor dem Ziel darf nichts tun — genau dort entstünde Teacher Forcing.

### Was mitläuft, weil Stille hier teuer wäre

* **`[SATURATED]`** — ein festgehaltener Rollout wird gezählt und gemeldet.
  Ohne den Zähler sähe ein weglaufendes Modell aus wie langsame Konvergenz.
* **Streuungsverhältnis in Ort und Zeit.** Beide Residuen verschwinden identisch
  auf einem flachen Feld. Ein fallender Physik-Verlust ist also nur so lange ein
  Beleg für Physik, wie diese Verhältnisse nahe 1 bleiben. Fällt es mit, hat der
  Optimierer die triviale Lösung gefunden — und keine Verlustkurve zeigte das.
* **Ein abgeschalteter Term läuft als `NaN` mit, nicht als `0.0`** — damit die
  Kurve eine Lücke zeigt statt einer flachen Linie, die es nie gab.
* **Labels nur bis `split_t`.** Sonst wird aus einer ausgehaltenen Zahl eine
  Trainingszahl. Ein Test prüft es.
* **`--seeds < 3` wird angemeckert.** Ein Seed ist keine Streuung.

## Die nächsten drei Schritte

1. **`balance_check.py`, Abschnitt 4 nachholen** — Minuten, nur numpy. Der Lauf
   vom 22.09. hat `Q_ht/tot` geklärt (0.700 … 0.772, **nicht** ≈ 1) und die
   Fluidbilanz bestätigt (1.030 … 1.062); `U(V̇)` fehlt noch, weil Abschnitt 4
   an den Zeitachsen abgestürzt ist. Werkzeug repariert, Lauf steht aus.
2. **Stufe 2, der Cache-Umbau** — danach ist der Wandterm kalibrierbar, die
   Physik-Latte messbar und `L_wall` anschließbar.
3. **Den Ladepfad in `train.py` anschließen** und Konfiguration **A** als ersten
   Lauf fahren, weil sie die Latte für B und C ist.

> Schritt 1 und 2 brauchen die Rechenmaschine mit `data_raw/` und `data_cache/`.
> Schritt 3 braucht den Cache, aber **nicht** die vier neuen Größen — er kann
> also parallel zu Schritt 2 laufen, solange man die Ergebnisse als das liest,
> was sie dann sind: adiabat.