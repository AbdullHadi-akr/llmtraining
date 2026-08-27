# Update 27.08.2026 — Rollout-Divergenz und was sich hier ändert

Dieses Dokument fasst einen Befund aus dem Basisprojekt zusammen und beschreibt,
was daraus für **diese Erweiterung** folgt. Die ausführliche Herleitung mit allen
Messungen steht in
[`../PINNmodulusTwo/README_ERSTER_TEST.md`](../PINNmodulusTwo/README_ERSTER_TEST.md)
und [`../PINNmodulusTwo/ARCHITECTURE.md`](../PINNmodulusTwo/ARCHITECTURE.md)
Kapitel 3.1.

**Kurzfassung:** Der free-running Rollout eines untrainierten Netzes divergierte,
bevor ein einziger Gradientenschritt passiert war. `L_data` war `nan`, jeder Lauf
brach in Epoche 1 ab. Zwei Ursachen, beide Layout-Entscheidungen, beide hier
genauso wirksam wie im Basisprojekt — eine davon **schlimmer**.

---

## Warum das diese Erweiterung besonders betrifft

Diese Erweiterung ist von beiden Ursachen betroffen, und beim Rate-Kanal ist sie
es **stärker als das Basisprojekt**.

### Der Integrator lief hier ungebremst und ohne Schalter

`train.py` hat `residual_output` **nie an `RecurrentField` übergeben**. Damit lief
diese Erweiterung still mit dem alten Modell-Default `True` — also mit dem
Integrator an — und hatte keine Möglichkeit, ihn abzuschalten. Das ist behoben:
`residual_output` ist jetzt ein Config-Schlüssel und ein CLI-Schalter, Default
`false`.

### `A` ist hier größer als im Basisprojekt

Der Rate-Kanal teilt eine Temperaturdifferenz durch `lag_n · rate_scale`, mit

```
A = 1 / (lag_n · rate_scale)
```

`rate_scale` ist `dTdt_scale`, der RMS von `dTn/dtn` auf z-normierter
Temperatur. **Das Pooling über OP01–OP16 vergrößert `T_sigma`** — OP14 startet
bei 0 °C, OP05 bei 40 °C, ein großer Teil der Streuung ist also Versatz
*zwischen* OPs und trägt zur Rate *innerhalb* eines OPs nichts bei. Ein
größeres `T_sigma` schrumpft `Tn`, schrumpft dessen Zeitableitung, schrumpft
`dTdt_scale` — und **erhöht damit `A`**.

Die Niveausprünge schrumpfen nicht mit `T_sigma`; sie sind das, woraus `T_sigma`
besteht. Der Befund des Basisprojekts (`A ≈ 119` bei `[5, 20] s`) ist hier also
die **Untergrenze**. `data.hybrid_rate_amplification` hat genau das schon
beschrieben — was fehlte, war die Erkenntnis, dass `residual_output` der
größere Treiber ist.

---

## Was geändert wurde

| Datei | Änderung |
|---|---|
| `config.yaml` | `residual_output: false` (neu), `rate_lags: [200.0, 600.0]` (war `[5.0, 20.0]`), `rollout_clamp: 50.0` (neu) |
| `train.py` | `--residual-output` / `--no-residual-output` und `--rollout-clamp` als CLI-Schalter; `residual_output` wird jetzt an `RecurrentField` übergeben; `rollout(...)` bekommt `clamp` |
| `smokeBench.py`, `profileBench.py` | Default für `--rate-lags` auf `[200.0, 600.0]` |

`model.py` kommt unverändert aus dem Basisprojekt (siehe `_paths.py`) — die
Sättigungsgrenze und der `level_rollout`-Fix sind also automatisch geteilt.
`--driver-rate-lags` ist **nicht** betroffen: das sind exogene Treiberkanäle,
keine Rückkopplung, und `[5.0, 20.0]` bleibt dort richtig.

---

## Die Begründung in Kurzform

### Ursache 1 (Haupttreiber): `residual_output`

`field()` lieferte `level(t) + net(...)`, also

```
level(t) ≈ level(t − Δgrid) + mean(net)
```

Ein **Integrator mit Verstärkung exakt 1 und ohne Leck**. Jeder einseitige
Anteil der Netzausgabe akkumuliert über die ~7000 Schritte unbeschränkt, und
nichts zieht ihn zurück. Wie klein er ist, spielt keine Rolle — ein Integrator
kennt nur das Vorzeichen, und Swish ist nicht mittelwertfrei.

Gemessen im Basisprojekt, 20 Epochen, 3 Seeds, ohne jedes Hilfsmittel:
`residual_output: true` bricht **9/9 ab, in jeder History-Konfiguration** —
auch bei `raw`, wo es gar keine Rate-Kanäle gibt. Genau das identifiziert den
Integrator und nicht die Verstärkung als Haupttreiber.

### Ursache 2 (Nebentreiber): zu kurze `rate_lags`

Für ein glattes Signal ist `lag_n · rate_scale` auf drei Stellen genau der RMS
der Differenz selbst. Der Divisor **ist** die Größe, auf die normiert wird — eine
echte 5-Sekunden-Änderung auf O(1) zu ziehen kostet zwangsläufig zwei
Größenordnungen Rauschverstärkung. Keine andere Normierung entkommt dem, nur ein
längeres Segment.

### `rollout_clamp: 50.0`

Sättigt `|Tn|` im Puffer. Ohne Physik-Term nur Diagnose; mit `w_phys > 0`
tragend — im Basisprojekt machte er aus einem 1-von-3-Abbruch bei Breite 128 drei
konvergierende Läufe. `residual_output: false` ist notwendig, aber **nicht
hinreichend**.

### Was ausdrücklich nicht hilft

* **Eine bessere Initialisierung.** Mit genullter Ausgabeschicht ist der Rollout
  bei Initialisierung perfekt stabil (0/5 über 7000 Schritte). Nach 20
  Adam-Schritten erreicht der nächste Rollout 4.7e4. Layout-Problem, kein
  Startpunkt-Problem.
* **`max_rate_amp` allein.** Skaliert einen Kanal um, statt die Segmentlänge zu
  korrigieren. Bleibt als Notnagel, Default `0.0`. Der Hinweis in
  `data.effective_rate_scale`, dies sei „das erste, was man probiert", ist damit
  überholt — das erste ist `--no-residual-output`.

---

## Wichtig für diese Erweiterung: `A`, nicht Sekunden

`rate_lags` in Sekunden übertragen sich **nicht** zwischen Bundles, `A`
überträgt sich. Weil das Pooling über OP01–OP16 `dTdt_scale` gegenüber OP01–05
verändert, sind `[200.0, 600.0]` hier **nicht automatisch dieselbe Verstärkung**
wie im Basisprojekt.

`train.py` gibt `A` bei jedem Start aus. **Prüfe die Zeile beim ersten Lauf:**

```
hybrid history amplification A = 1/(lag_n * rate_scale) per lag: ...
```

Die Regel lautet **`A` auf O(1) bringen**. Liegt `A` mit `[200, 600]` hier
deutlich über ~10, gehören die Segmente länger — nicht `max_rate_amp` gesetzt.

---

## `hybrid` gegen `raw` — offen, auch hier

Im Basisprojekt gemessen (5 Seeds, MAE in °C, synthetisches Bundle):

| | MAE train | MAE test | Streuung test | Generalisierungslücke |
|---|---|---|---|---|
| `hybrid [200,600]` | **0.308** | 1.150 | 6.6× | 3.7× |
| `raw` | 0.436 | **0.780** | **3.5×** | **1.8×** |

`hybrid` passt in-sample besser und generalisiert schlechter — die Signatur von
Überfitting. Der Mechanismus: um `A` auf O(1) zu bringen, mussten die Segmente
von 5/20 s auf 200/600 s wachsen, und ein 600-s-Fenster auf einer 1474-s-
Trajektorie ist keine Rate mehr, sondern ein **Fortschrittsindikator**.

Daraus die Spannung, die vorher niemand sehen konnte, weil vorher nichts
durchlief:

```
zu kurz  →  A groß  →  Rollout divergiert
zu lang  →  Fortschrittsindikator  →  überfittet
```

`hybrid` bleibt vorerst Default, weil die Messung synthetisch ist und ein
Wechsel des `history_mode` die Vergleichbarkeit aller bisherigen Ergebnisse
entwertet. Entschieden wird das auf echten Daten mit `profileBench.py` /
`bench_profiles.py`, Kriterium ist **MAE auf gehaltenen OPs**, nicht `L_data`.

> **`L_data` ist nicht das Auswahlkriterium.** Auf `L_data` lag
> `hybrid [200,600]` 100× vor `raw`; auf MAE liegt `raw` vorn. Die beiden ordnen
> die Konfigurationen unterschiedlich.

**Für diese Erweiterung ist die Frage besonders offen**, weil die
Profil-Betriebspunkte ab OP08 zeitveränderliche Treiber haben. Ein
Fortschrittsindikator, der bei konstanten Treibern (OP01–05) gut funktioniert,
kann bei einem CC-CV-Profil etwas ganz anderes bedeuten.

---

## Nächste Schritte hier

1. `A` beim ersten Lauf prüfen (Startzeile), Segmente anpassen falls ≫ 10.
2. `python3 smokeBench.py` mit den neuen Defaults — läuft Epoche 1 durch?
3. `profileBench.py` über `history_mode` und `rate_lags`, Kriterium MAE auf
   gehaltenen OPs.

Zum Nachmessen der Divergenz bei Initialisierung, ohne Modulus und ohne Daten:

```bash
python3 ../PINNmodulusTwo/tools/rollout_divergence.py
```

---

## Einschränkungen

* Alle Zahlen stammen aus dem **Basisprojekt** und von einem **synthetischen
  Bundle**. Die Richtung ist mechanistisch erklärt und robust (9/9 gegen 0/9);
  die Beträge sind es nicht.
* **In dieser Erweiterung wurde noch kein Lauf gemacht.** Die Änderungen sind
  aus dem Befund des Basisprojekts abgeleitet und nicht auf OP01–OP16
  nachgemessen. Der `smokeBench.py`-Lauf oben ist der erste Prüfstein.
* Die Überfitting-Aussage zu `hybrid` ist auf einem glatten synthetischen
  Verlauf gemessen und dort vermutlich überzeichnet.
