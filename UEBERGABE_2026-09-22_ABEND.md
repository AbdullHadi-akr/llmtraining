# Übergabe — Sitzung vom 22.09.2026, abends

> **Wenn du nur eine Sache liest:** Der GridCNN funktioniert jetzt. Alle drei
> Seeds unterbieten die trivialen Latten, die Seed-Streuung ist lesbar. Der
> nächste Schritt ist **ein Kommando** auf der Maschine mit `data_raw/`, und
> es steht unten unter „Das Nächste".

**Branch:** `claude/nifty-bohr-5vy1pz` · **PRs:** [#46](https://github.com/AbdullHadi-akr/llmtraining/pull/46) (gemerged), dieser · **Basis:** `main` nach PR #46

---

## 0. Für einen Kaltstart — worum es überhaupt geht

Ein CNN soll das Temperaturfeld einer Batteriezelle über die Zeit
vorhersagen. Gitter **3 × 11 × 11 = 363 Punkte**, 11 Trainings-Betriebspunkte
(OPs), 2 Halte-OPs (OP06, OP09), 3 Test-OPs (OP13/15/16, nie benutzt).

Das Modell macht **einen** Zeitschritt:

```
T_{t+1} = T_t + dt · g_θ( T_t, T_{t-lag1}, T_{t-lag2}, Treiber_t )
```

Eine ganze Trajektorie entsteht, indem es sich selbst wieder füttert —
1445 Mal bei `--subsample 10`, ~8040 Mal bei `--subsample 2`. Es ist also ein
**rekurrentes Netz**, dessen Zustand das Temperaturfeld selbst ist.

Vier Arme werden verglichen (`konfigurationsname()` in `train.py`):

| | |
|---|---|
| **A** | Blackbox. `rate = g_θ`, keine Physik in der Architektur (`--no-physics`) |
| **B** | Physik in der Architektur: `rate = Laplace(T) + Qsrc + g_θ` |
| **C** | B plus Physik-**Strafterm** (`--w-phys > 0`) |
| **D** | B ohne die zwei Koordinatenkarten (`--no-coord-maps`) |

**Bisher ist nur A gemessen.** Der Plan steht in
[`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md), das *Warum* in
[`GridCNN/README.md`](GridCNN/README.md), die Epochenzeile in
[`GridCNN/README_DIAGNOSTIK.md`](GridCNN/README_DIAGNOSTIK.md).

---

## 1. Was am Anfang dieser Sitzung galt

Konfiguration A war am Vormittag gelaufen und war **kein Ergebnis**:
27.11 ± 22.26 °C Seed-Streuung gegen eine Lesbarkeitsschwelle von ~1 °C. Der
Bericht dazu ist
[`TRAININGS_BERICHT_2026-09-22_KonfigA.md`](TRAININGS_BERICHT_2026-09-22_KonfigA.md)
und seine Diagnose war: **ein Schritt trainiert, ~8040 gemessen.** Nichts in
der Schleife beschränkte das Verhalten über diesen Horizont; bis 99.8 % aller
OP-Zeitschritte lagen am `--clamp`.

Er endete mit vier offenen Fragen, die ich nicht selbst entscheiden sollte.

---

## 2. Die vier Fragen — entschieden

| | Frage | Entscheidung |
|---|---|---|
| **1** | Budget: 60 Epochen × 100 `inner_steps` gesetzt? | **Nein, erster Schuss.** Konstant gehört `inner_steps × k` |
| **2** | Darf sich das Experiment ändern? | **Ja.** Clipping, LR-Plan und TBPTT sind **Protokoll**, keine Experimentvariablen — einmal festgelegt, dann identisch für A/B/C/D. A wie am Vormittag *ist* keine Latte und kann keine verlieren. Der alte Pfad bleibt exakt reproduzierbar |
| **3** | `--clamp 50` gewählt oder geerbt? | **Geerbt** aus `PINNmodulusTwo`. ±480 °C fängt nichts ab. Vorgabe ist jetzt `auto` aus den Labels |
| **4** | CFL für B/C/D: kleineres oder größeres `subsample`? | **Noch nicht zu entscheiden, und das ist die Entscheidung.** Arm A braucht die Antwort nicht (kein Laplace-Term). Siehe „Das Nächste" |

---

## 3. Was gebaut wurde — PR #46 (gemerged)

**Der Kern: `--tbptt k`** (Stufe 5, truncated BPTT). Das Fenster rollt **mit**
Gradient über `k` Zeitschritte und frisst sich dabei selbst: ab Schritt 1 ist
der Anker die eigene Vorhersage.

```
k = 1 (alt):  T_t ──[Netz]──> T̂_{t+1}  gegen Label     Gradient endet hier
k = 4 (neu):  T_t ──> T̂_{t+1} ──> T̂_{t+2} ──> T̂_{t+3} ──> T̂_{t+4}
                        │           │           │           │
                      Label       Label       Label       Label
              └──────── Gradient läuft durch alle vier ────────┘
```

Der Startzustand kommt aus dem eingefrorenen, frei gelaufenen Rollout — nicht
aus einem Label. **Es ist kein Teacher Forcing.** `k = 1` ruft weiterhin
`data_loss` und reproduziert den alten Pfad Zeichen für Zeichen.

Dazu: `--clip-grad 1.0`, `--lr-plan cosine`, `--clamp auto` (3 × größte
Auslenkung in den Labels), Lags aus **Sekunden** statt Schritten (bei
`--subsample 2` unverändert 5/20), nicht-endliche Updates werden verworfen und
gezählt, Gradientennorm und Wanduhr im Log, berichtet wird der **Median über
das letzte Drittel** der Messpunkte.

**Diese Sitzung, zweiter PR:** das **Fehlerprofil** (Abschnitt 5 unten) und
die Korrektur am Wandterm.

---

## 4. Was gemessen wurde

[`GridCNN/laeufe/15_konfigA_poc.txt`](GridCNN/laeufe/15_konfigA_poc.txt),
`--subsample 10`, 40 Epochen, 3 Seeds, rund 15 Minuten.

| | berichtet | Latte | Güte | Seed-Streuung |
|---|---|---|---|---|
| **OP06** | **6.571 °C** | 10.7995 | **0.61x** | 0.38 °C |
| **OP09** | **5.813 °C** | 7.7663 | **0.75x** | 0.87 °C |

**3/3 Seeds unter der Latte auf beiden OPs. Streuung unter der
Lesbarkeitsschwelle von ~1 °C.** Zum ersten Mal seit dem 09.09. ist ein Lauf
überhaupt rankfähig.

**Der Anschluss ans Basisprojekt steht:** die trivialen Latten stimmen mit
`PINNmodulusTwo/FAHRPLAN.md:1244` auf drei bis vier Nachkommastellen überein
(10.801 / 7.762 und 16.679 / 18.549). Damit ist der Vergleich möglich:

| | OP06 | OP09 |
|---|---|---|
| **PINN** (01.09., ein Seed, mit Randterm) | 6.270 | 3.585 |
| **GridCNN A** (3-Seed-Median, adiabat, Blackbox) | 6.571 ± 0.38 | 5.813 ± 0.87 |

Auf OP06 gleichauf, auf OP09 zurück. **Das Ziel bleibt ~1 K** — beide Modelle
sind Faktor 4–6 davon entfernt.

Voller Bericht:
[`TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md`](TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md).

---

## 5. Der Befund, der alles andere überlagert

Der Plot
[`15_konfigA_OP06_fehler_ueber_zeit.png`](GridCNN/laeufe/15_konfigA_OP06_fehler_ueber_zeit.png)
zeigt den Fehler über die Trajektorie statt als Mittelwert:

| t [s] | 131 | 394 | 788 | 919 | 1181 | 1444 |
|---|---|---|---|---|---|---|
| **MAE [°C]** | 9.8 | **1.1** | 7.8 | 3.7 | 8.9 | **15.0** |

**Bei 394 s ist das Modell auf dem ~1 K-Ziel. Am Ende liegt es bei 15 °C** —
und der *kleinste* Fehler über alle 363 Punkte beträgt dort noch 11.5 °C.
Also liegt das ganze Feld daneben: ein **Pegelfehler**, keine Streuung. Der
Mittelwert 6.57 °C verdeckt beides.

Das ist **O13**, der Spätfehler. Und `GridCNN/model.py` hat genau das
vorhergesagt: die Delta-Form `T + dt·g` trägt den Level **ohne Leck**, und der
dissipative Diffusionskern sollte das Leck liefern — **den hat Arm A nicht**
(`--no-physics` heißt `rate = g_θ`, ohne Laplace-Term).

> Damit ist Arm B nicht mehr „der nächste Punkt auf der Liste", sondern die
> **Behandlung für die beobachtete Krankheit**.

**Deshalb misst `train.py` das ab jetzt selbst:** sechs gleich lange
Abschnitte, MAE **und vorzeichenbehafteter Bias** je Abschnitt, plus
`drift = MAE(letzter Abschnitt) / MAE(gesamt)`. Ab 1.5 meldet der Lauf O13
von sich aus. Der Bias entscheidet die offene Frage „zu warm oder zu kalt",
die ein Betragsplot nicht beantworten kann.

---

## 6. Das Nächste — in dieser Reihenfolge

### Schritt 1: Volle Auflösung bestätigen (~45 min)

**Nur `--subsample` und `--epochs` ändern sich**, damit ein Fehlschlag
zuordenbar bleibt. Die Lags leiten sich bei `--subsample 2` automatisch wieder
auf 5/20 ab — identisch zum Vormittagslauf.

```bash
python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 4 --tbptt 16 \
    --val-every 2 --device cuda --cache data_cache \
    2>&1 | tee 16_konfigA_voll.txt
```

Worauf zu schauen ist:

* **Hält die Güte < 1.0?** 8040 Schritte statt 1445 — fünffacher Horizont.
* **Was sagt `drift`?** Bei 2.0x ist der Spätfehler bestätigt.
* **Was sagt der Bias?** Negativ zum Ende = das Modell wird zu kalt, positiv =
  zu warm. Das ist neu und es entscheidet, wo man ansetzt.
* **Die Latten bei `--subsample 2`** — sie machen die Einordnung gegen den
  PINN exakt.

Geht es schief: Frühphase instabil → `--tbptt-start 8`. Val-Kurve schwankt →
EMA. Beides ist begründet, aber **eins nach dem anderen**.

### Schritt 2: CFL entscheiden (eine Rechnung, kein Lauf)

`dt_n` liegt **110× über der expliziten Schranke** bei `--subsample 2`, 551×
bei 10. Für A egal, für B/C/D ein Blocker. Zwei Wege, beide mit Preis:

* **kleineres `subsample`** → mehr Schritte → schlimmere Aufschaukelung
* **größeres `subsample`** → gröbere Daten → CFL noch schlechter

Zu prüfen ist zuerst, ob die Materialdaten echt sind: ein synthetisches
`material_properties/` macht das Problem viel steifer, als es ist (die
`[CFL]`-Zeile sagt das selbst).

### Schritt 3: Arm B — der Physik-POC

Gated auf Schritt 2. Die Frage ist präzise: **holt der dissipative
Diffusionskern den Pegel zurück?** Messbar an `drift` und `bias` aus
Schritt 1, nicht am Mittelwert.

### Parallel, weil Code und kein Lauf: den Wandterm verdrahten

`_wall_ghost` in `train.py` wirft weiterhin. **Der genannte Grund war
abgestanden** — Stufe 2 ist durch, `U(V̇)` ist kalibriert, `physics.UCurve`
und `physics.WallModel.ghost` sind gebaut und getestet. Was wirklich fehlt:

1. `op_tensoren` füllt `OPTensors.q_wall_meas` nie — bleibt `None`, also kann
   `wall_loss` nicht laufen.
2. `WallModel.ghost` braucht `t_in` und `mdot` **je Zeitschritt**. Beide
   liegen seit Stufe 2 im Bündel, sind aber nicht in `OPTensors` übernommen.

Der heutige Lauf ist deshalb **adiabat** — eine Ablation, keine Latte.

---

## 7. Was sonst offen liegt

* **Die Haltemenge sind zwei OPs.** Die ganze Messung hängt an zwei Zahlen.
  Es gibt drei Test-OPs (OP13, OP15, OP16), auf denen nie ausgewählt wurde —
  sie sind bisher **nie berichtet** worden. Nach Schritt 1 wäre das fällig.
* **O14, die Envelope-Grenze.** `GridCNN/README.md:903` ordnet OP06s ~6.3 °C
  als Datenproblem ein: „keine Kühlung bei mittlerer Starttemperatur" kommt im
  Training nicht vor. Wenn das stimmt, ist der Rest auf OP06 **weder durch
  Physik noch durch ein größeres Netz** zu holen.
* **Netzgröße.** 11 427 Parameter. Die Präsentation sah 64 × 4 vor
  (137 923). Als Sweep-Achse erreichbar, bisher unberührt.

---

## 8. Wie man hier arbeitet — Konventionen

* **Läufe** landen als Rohtext in `GridCNN/laeufe/`, Parameterblatt und Plot
  daneben. Ein Bericht ohne den Lauf, auf den er sich beruft, ist eine
  Behauptung.
* **Nie die letzte Zeile eines Laufs ablesen.** Berichtet wird der Median über
  das letzte Drittel. Das Beste wäre auf der Haltemenge ausgewählt.
* **Drei Seeds, nie einer.** Am Vormittag lagen bester und schlechtester Seed
  um Faktor 5.8 auseinander.
* **Eine Änderung pro Lauf.** Sonst ist ein Ergebnis nicht zuordenbar.
* **Latten immer daneben.** `0.61x Latte` ist lesbar, `6.57 °C` nicht.
* **Tests:** `python3 -m pytest GridCNN/tests -q` — 130 Stück, ~8 s, brauchen
  weder GPU noch Cache. CI fährt sie bei jedem PR.
