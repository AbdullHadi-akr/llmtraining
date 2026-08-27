# Update 27.08.2026 — Rollout-Divergenz und was sich hier ändert

Dieses Dokument fasst einen Befund aus dem Basisprojekt zusammen und beschreibt,
was daraus für **diese Erweiterung** folgt. Die ausführliche Herleitung mit allen
Messungen steht in
[`../PINNmodulusTwo/README_ERSTER_TEST.md`](../PINNmodulusTwo/README_ERSTER_TEST.md)
und [`../PINNmodulusTwo/ARCHITECTURE.md`](../PINNmodulusTwo/ARCHITECTURE.md)
Kapitel 3.1.

**Kurzfassung:** Der free-running Rollout eines untrainierten Netzes divergierte,
bevor ein einziger Gradientenschritt passiert war. `L_data` war `nan`, jeder Lauf
brach in Epoche 1 ab. Die Ursache ist **`residual_output`** — und diese
Erweiterung lief still damit, ohne Schalter dagegen.

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
| `config.yaml` | `residual_output: false` (neu), `rollout_clamp: 50.0` (neu). `rate_lags` bleiben bei `[5.0, 20.0]` |
| `train.py` | `--residual-output` / `--no-residual-output` und `--rollout-clamp` als CLI-Schalter; `residual_output` wird jetzt an `RecurrentField` übergeben; `rollout(...)` bekommt `clamp` |


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

### Die Verstärkung `A` — real, aber **nicht** die Ursache

Für ein glattes Signal ist `lag_n · rate_scale` auf drei Stellen genau der RMS
der Differenz selbst. Der Divisor **ist** die Größe, auf die normiert wird — eine
echte 5-Sekunden-Änderung auf O(1) zu ziehen kostet zwangsläufig zwei
Größenordnungen Rauschverstärkung, `A ≈ 119` im Basisprojekt.

**Das ist trotzdem nicht der Grund für die Abbrüche.** Im Basisprojekt wurde in
der echten Geometrie (`n_t = 7000`) gemessen: ist `residual_output` aus und
`rollout_clamp` an, gewinnt `[5, 20]` die MAE auf allen drei Seeds, vor
`[50,150]`, `[200,600]` und `raw`, ohne einen einzigen Abbruch. Längere Segmente
senken `A`, machen den Kanal aber zu einem Fortschrittsindikator und
generalisieren schlechter. `--max-rate-amp` dämpft den Kanal und schadet
ebenfalls.

**Deshalb bleiben die `rate_lags` hier bei `[5.0, 20.0]`.**

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
* **`max_rate_amp`.** Dämpft den Rate-Kanal; gemessen wird die MAE monoton
  schlechter, je härter gedeckelt wird. Bleibt aus (`0.0`). Der Hinweis in
  `data.effective_rate_scale`, dies sei „das erste, was man probiert", ist damit
  überholt — das erste ist `--no-residual-output`.
* **Längere `rate_lags`.** Senken `A`, verschlechtern die MAE. `[5, 20]` bleibt.

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

`A` ist hier **höher als die 119 des Basisprojekts**, und in diesem Bereich hat
niemand gemessen. Falls ein Lauf trotz `residual_output: false` und
`rollout_clamp: 50` abbricht, ist das der erste Verdächtige — dann sind längere
Segmente die Gegenprobe. Sie sind aber kein Reflex: im Basisprojekt haben sie
geschadet.

---

## `hybrid` gegen `raw` — offen, auch hier

Im Basisprojekt gemessen, in der **echten Geometrie** (`n_t = 7000`, 3 Seeds,
MAE in °C auf dem gehaltenen Abschnitt):

| `rate_lags` | `A` | MAE train | MAE test |
|---|---|---|---|
| **`[5, 20]`** | 119 / 30 | 0.784 | **1.207** — bester Seed-für-Seed |
| `[50, 150]` | 12 / 4 | **0.594** | 2.102 |
| `[200, 600]` | 3 / 1 | 0.724 | 2.507 |
| `raw` | — | 0.824 | 2.601 |

`[50, 150]` hat die beste MAE train und die zweitschlechteste MAE test — das ist
Überfitting. Um `A` zu senken, muss das Fenster wachsen, und ein 600-s-Fenster
auf 1474 s ist **keine Rate mehr, sondern ein Fortschrittsindikator**: es sagt
dem Netz, *wo in der Trajektorie* es ist. In-sample hilft das, ausserhalb nicht.

`raw` ist die schlechteste der vier Varianten. `hybrid [5, 20]` bleibt Default.

> **`L_data` ist nicht das Auswahlkriterium.** Auf `L_data` lag `[200, 600]` zwei
> Größenordnungen vorn — auf MAE ist es das zweitschlechteste. Die beiden ordnen
> die Konfigurationen unterschiedlich.

**Für diese Erweiterung ist die Frage besonders offen**, weil die
Profil-Betriebspunkte ab OP08 zeitveränderliche Treiber haben. Ein
Fortschrittsindikator, der bei konstanten Treibern (OP01–05) gut funktioniert,
kann bei einem CC-CV-Profil etwas ganz anderes bedeuten.

---

## Status — umgesetzt, fehlend, festgelegt

### Die festgelegten Werte

| Schlüssel | Wert | Status |
|---|---|---|
| `residual_output` | **`false`** | **fest.** Der Fix. Vorher gab es hier nicht einmal einen Schalter dafür. |
| `rollout_clamp` | **`50.0`** | **fest.** Tragend, sobald `w_phys > 0`. |
| `rate_lags` | **`[5.0, 20.0]`** | **vorläufig.** Aus dem Basisprojekt übernommen; `A` ist hier höher und ungetestet. |
| `driver_rate_lags` | `[5.0, 20.0]` | **unverändert.** Exogene Treiberkanäle ohne Rückkopplung — davon war nie etwas betroffen. |
| `max_rate_amp` | `0.0` (aus) | **fest.** Schadet im Basisprojekt gemessen monoton. |
| `history_mode` | `hybrid` | **vorläufig.** |

### Umgesetzt

* `residual_output` als Config-Schlüssel und CLI-Schalter (`--residual-output` /
  `--no-residual-output`), Default `false`, **und tatsächlich an
  `RecurrentField` übergeben** — vorher wurde es dort schlicht nicht gesetzt
* `rollout_clamp` als Config und CLI, am `rollout(...)`-Aufruf durchgereicht
* `model.py` kommt unverändert aus dem Basisprojekt (siehe `_paths.py`), die
  Sättigungsgrenze und der `level_rollout`-Fix sind also automatisch geteilt

### Fehlt

* **Hier wurde noch gar nichts ausgeführt.** Alle Änderungen sind aus dem
  Befund des Basisprojekts abgeleitet und auf OP01–OP16 nicht nachgemessen.
* Ob `[5, 20]` hier ebenfalls gewinnt. `A` ist durch das Pooling höher als die
  119 des Basisprojekts, und in diesem Bereich hat niemand gemessen.
* Ob das Ergebnis für **Profil**-Betriebspunkte trägt. Der Lag-Vergleich im
  Basisprojekt lief auf konstanten Treibern (OP01–OP07). Ein Rate-Kanal, der bei
  konstanter C-Rate gut funktioniert, kann bei einem CC-CV-Profil oder einem
  Fluidtemperaturprofil etwas ganz anderes bedeuten.

### Was als nächstes zu tun ist

1. `python3 smokeBench.py` — erster Prüfstein. **Zuerst die `A`-Startzeile
   lesen:** `hybrid history amplification A = ...`. Sie ist hier höher als 119,
   und wie viel höher, weiß niemand.
2. Läuft Epoche 1 durch? Falls nicht, ist `--no-residual-output` bereits an —
   dann `--rollout-clamp` prüfen und danach längere Segmente als Gegenprobe.
   Längere Segmente sind aber **kein Reflex**: im Basisprojekt haben sie
   geschadet.
3. `profileBench.py` / `bench_profiles.py` über `history_mode` und `rate_lags`.
   Kriterium: **MAE auf gehaltenen OPs**, niemals `L_data`.
4. `python3 ../PINNmodulusTwo/tools/rollout_divergence.py` — Divergenz bei
   Initialisierung, braucht weder Modulus noch Daten.

Die vollständige Übergabe steht in
[`../UEBERGABE_2026-08-27.txt`](../UEBERGABE_2026-08-27.txt).

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
* **Die Lag-Messung gilt für OP01–05, nicht für OP01–OP16.** Sie ist bei
  `A = 119/30` gemacht; das Pooling hier hebt `A` darüber hinaus.
* **In dieser Erweiterung wurde noch kein Lauf gemacht.** Die Änderungen sind
  aus dem Befund des Basisprojekts abgeleitet und nicht auf OP01–OP16
  nachgemessen. Der `smokeBench.py`-Lauf oben ist der erste Prüfstein.
* Die Überfitting-Aussage zu `hybrid` ist auf einem glatten synthetischen
  Verlauf gemessen und dort vermutlich überzeichnet.
