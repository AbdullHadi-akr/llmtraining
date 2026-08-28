# Lokaler Lauf — Anleitung für die Claude-Code-Sitzung auf deinem Rechner

**Diese Datei ist das Übergabeblatt an den lokalen Assistenten.** Öffne Claude
Code in diesem Repo-Ordner auf dem Rechner, auf dem die Simulationsdaten liegen,
und sag:

> Lies `PINNmodulusTwo/README_LOKALER_LAUF.md` und arbeite sie von oben nach
> unten ab.

Hintergrund, warum es diese Datei gibt: die Cloud-Sitzung, die den NaN-Abbruch
in Epoche 1 behoben hat (PR #14), hatte **nie Zugriff auf die echten Daten** —
`data_cache/` und `material_properties/` sind gitignored und liegen nur lokal.
Alle bisherigen MAE-Zahlen sind synthetisch. Was noch offen ist, steht in
`../UEBERGABE_2026-08-27.txt`, Kapitel 7 und 8. Diese Anleitung arbeitet genau
diese TODOs ab, in der Reihenfolge, in der sie sich gegenseitig bedingen.

---

## Schritt 0 — Wo die Daten hinmüssen

**An den lokalen Assistenten: fang hiermit an.** Prüfe, ob die beiden
Datenordner existieren, und **nenne dem Nutzer den vollständigen absoluten Pfad**,
an den er sie legen soll — er muss sie noch selbst einbetten und weiß sonst nicht,
wohin. Prüfe danach, dass sie wirklich da sind, bevor du irgendetwas startest.

```bash
ls PINNmodulusTwo/data_cache/*.npz 2>/dev/null | head
ls PINNmodulusTwo/material_properties/ 2>/dev/null
```

### Die zwei Ordner

**1. `PINNmodulusTwo/data_cache/`** — die Simulationsbündel, eine `.npz` je
Betriebspunkt:

```
PINNmodulusTwo/data_cache/OP01.npz
PINNmodulusTwo/data_cache/OP02.npz
...
PINNmodulusTwo/data_cache/OP07.npz
```

Gebraucht werden für das Basisprojekt **OP01–OP05** (Training, `config.yaml:
ops`) und **OP07** (gehaltener Testpunkt, `config.yaml: test_op`). OP06 wird
ebenfalls gehalten. Jedes Bündel enthält mindestens `t_fast`, `T`, `q_source`,
`xyz`, `layer`, `sim_config_scalar` und `sim_config_scalar_names_json`
(siehe `data.py:_read_raw`). Wenn die Bündel noch nicht existieren, baut
`generate_cache.py` sie aus den Roh-CSVs.

**2. `PINNmodulusTwo/material_properties/`** — die Stoffwerte, die
`materials.py` liest:

```
PINNmodulusTwo/material_properties/constants.yaml
PINNmodulusTwo/material_properties/Cell Center/*.csv
PINNmodulusTwo/material_properties/JR1 Center/*.csv
```

Die Property-CSVs haben eine Kopfzeile und **eine** Datenzeile, eine Spalte je
Gitterpunkt (`materials.py:_load_row_csv`).

### Alternative Orte

`data.py` sucht den Cache in dieser Reihenfolge und nimmt den ersten Treffer:

```
PINNmodulusTwo/data_cache/                                        <- projektlokal
<repo>/data_cache/                                                <- bevorzugt, oben
<repo>/legacy/battery_surrogate_agenticWorkflow/data_cache/
<repo>/battery_surrogate_agenticWorkflow/data_cache/
```

`material_properties/` dagegen ist **fest** auf `PINNmodulusTwo/` verdrahtet
(`materials.py: MAT_DIR`) — dieser Ordner muss genau dort liegen.

Auf einem GPU-Server gehen beide per `rsync` hin, siehe `README_GPU_SERVER.md`:

```bash
rsync -avz <lokal>/data_cache          <server>:<repo>/PINNmodulusTwo/
rsync -avz <lokal>/material_properties <server>:<repo>/PINNmodulusTwo/
```

### Die Daten gehören nicht ins Repo

Beide Ordner sind in `.gitignore` **hart ausgeschlossen** — die Regeln stehen
bewusst als letzte in der Datei, damit die `!*.py` / `!README*`-Freigabeliste
darüber nichts wieder herausziehen kann. Ein ignorierter Ordner wird von git
gar nicht erst betreten, es kann also auch keine Datei darin versehentlich
getrackt werden.

**An den lokalen Assistenten: verifiziere das, bevor du committest**, und zwar
mit einer Datei, die es beim Nutzer wirklich gibt:

```bash
git check-ignore -v PINNmodulusTwo/data_cache/OP01.npz
git check-ignore -v PINNmodulusTwo/material_properties/constants.yaml
git status --porcelain          # darf keine Datendatei zeigen
```

Beide Aufrufe müssen die Regel ausgeben, die greift (`.gitignore:…:data_cache/`
bzw. `material_properties/`). Tun sie das nicht, **nichts committen** und den
Nutzer informieren. `git add -f` auf eine Datendatei ist unter keinen Umständen
richtig.

---

## Schritt 1 — Der kleine Test zuerst (TODO-3 und TODO-1)

Bevor irgendein Training läuft. Kostet Sekunden, braucht **nur numpy** — kein
Modulus, kein torch, keine GPU, kein `material_properties/`:

```bash
python3 PINNmodulusTwo/tools/data_probe.py
```

Die Sonde liest aus den Bündeln nur `t_fast` und `T` und beantwortet die zwei
Fragen, an denen alles Weitere hängt:

**(a) Stimmt A?** Sie rechnet `T_span_ref`, `T_mu`/`T_sigma` und `dTdt_scale`
genau so, wie `data.py` sie poolt, und gibt `A = 1/(lag_n · rate_scale)` je
`rate_lag` aus. Die Übergabe nimmt `dTdt_scale ≈ 2.479` und damit `A ≈ 119/30`
an — das stammt aus einem synthetischen Bündel. Weicht der echte Wert deutlich
ab, **warnt die Sonde**, und dann trägt Kapitel 4 der Übergabe (die Lag-Wahl)
nicht mehr und muss gegen die echten Zahlen neu gerechnet werden.

**(b) Ändert sich in 5 Sekunden überhaupt etwas?** Das ist TODO-1, die einzige
offene Frage, bei der die Physik entscheidet und nicht die Messung. `A = 119`
heißt: die echte Temperaturänderung über 5 s beträgt 1/119 der typischen Rate,
bei `T_sigma = 5 K` also rund **42 Millikelvin**. Die Sonde stellt die
tatsächliche Änderung über jedes Rate-Fenster gegen eine Rauschabschätzung aus
der zweiten Differenz und gegen die Quantisierung, mit der die Werte gespeichert
sind. `sigma_n` ist absichtlich eine **obere** Schranke (Krümmung zählt hinein),
die SNR-Spalte also eine **untere** — was dort als Signal ausgewiesen wird, ist
Signal.

Wie die Verdict-Zeile zu lesen ist:

| SNR | Bedeutung | Konsequenz |
|---|---|---|
| ≥ 10 | echtes Signal | `rate_lags: [5.0, 20.0]` bleiben, weiter zu Schritt 2 |
| 3–10 | dünn | trotzdem weiter, aber in TODO-4 `[20, 60]` mit in den Sweep |
| < 3 | überwiegend Diskretisierungsrauschen | erste Segmente verlängern, z. B. `[20, 60]` (A = 30/10) — **aber nicht so lang, dass der Kanal zum Fortschrittsindikator wird**, siehe Übergabe Kapitel 4 |

**An den lokalen Assistenten: gib dem Nutzer die vollständige Ausgabe der Sonde
und deine Einordnung dazu.** Die Ausgabe ist aggregiert, keine
Trajektoriendaten — sie kann bedenkenlos geteilt werden.

Dann noch der datenfreie Teil, der ohnehin vor jedem langen Lauf gehört
(braucht torch):

```bash
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/tools/rollout_divergence.py
```

---

## Schritt 2 — Der erste echte Lauf (TODO-2)

Das ist die eigentliche Aufgabe. Vorher `README_GPU_SERVER.md` für die
Umgebung (torch aus dem CUDA-Wheel-Index, dann `requirements-gpu.txt`).

```bash
python3 PINNmodulusTwo/smallBench.py
```

**Worauf zu achten ist, in dieser Reihenfolge:**

1. **Startzeile** `hybrid history amplification A = ...` — ist A wirklich
   ~119/30? Muss mit dem übereinstimmen, was die Sonde in Schritt 1 gesagt hat.
   Tut es das nicht, stimmt etwas an der Datenaufbereitung nicht.
2. **`[CFL OK]` oder `[CFL WARN]`** — Zeitschritt gegen das Stabilitätslimit
   (`subsample_time: 2` → dt = 0.2 s, Grenze ~0.241 s).
3. **Kein `[ABORT] epoch 1` mehr.** Falls doch: erster Verdächtiger ist
   `--no-residual-output`, dann `--rollout-clamp 50`.
4. **`[SATURATED] epoch N: ... x/y steps`** — der Zählstand soll **fallen**.
   Flach oder steigend heißt, das Modell fängt sich nicht.
5. **Die MAE-Zahlen aus `metrics.txt`** in `README_ERSTER_TEST.md` Kapitel 6
   eintragen und die dort stehenden synthetischen Zahlen ersetzen. Dabei den
   Hinweis, dass es synthetische Zahlen sind, mit entfernen.

**Wichtig zur Einordnung:** vorher gab es *überhaupt keine* MAE, weil jeder Lauf
abbrach. Die 11.96 °C aus den alten Dokumenten sind eine **Baseline** (der
Fehler eines Vorhersagers, der nichts tut), keine frühere Messung. Es gibt
keine Verbesserung „von 11 auf 0.5". Diese Zahl ist der Maßstab, gegen den die
erste echte MAE zu lesen ist.

---

## Schritt 3 — Die Sweeps (TODO-4 und TODO-5)

Erst wenn Schritt 2 sauber durchläuft. Beide dauern lange, also auf der GPU.

```bash
python3 PINNmodulusTwo/benchmark_arch.py      # Lag-Sweep
python3 PINNmodulusTwo/benchmark_wphys_wbc.py # was bringt der Physik-Term?
```

Das Gitter von `benchmark_arch.py` ist bereits auf die A-Achse umgestellt:
`[5,20]` (Default) · `[2,10]` · `[10,60]` · `[50,150]` · `[200,600]` ·
`[5,20,60]`. Sagt Schritt 1 SNR < 10, `[20,60]` ergänzen.

**Zwei Fallstricke, die schon einmal eine Empfehlung gekippt haben:**

* **Kriterium ist MAE auf dem gehaltenen OP, NIEMALS `L_data`.** Die beiden
  ordnen die Konfigurationen unterschiedlich: auf `L_data` liegt `[200,600]`
  zwei Größenordnungen vorn, auf der Lieferzahl MAE ist es das
  zweitschlechteste.
* **Die Geometrie muss stimmen.** Eine verkürzte Trajektorie verschiebt
  `dTdt_scale` und damit A — bei `n_t=1200` statt 7000 um Faktor 6.6. Wer
  Lag-Wahl oder A misst, braucht die volle Länge. Genau daran ist die
  zwischenzeitliche Empfehlung `[200,600]` gescheitert.
* **Mindestens drei Seeds, besser fünf.** Auf Seed 0 allein sieht die Rangfolge
  mehrfach anders aus als über drei.

---

## Schritt 4 — Die Profilerweiterung (TODO-6)

In `PINNmodulusTwoExtProfiles` wurde **überhaupt nichts ausgeführt**; alle
Änderungen dort sind aus dem Basisbefund abgeleitet. Erster Prüfstein:

```bash
python3 PINNmodulusTwoExtProfiles/smokeBench.py
```

Dort ist **A größer** als im Basisprojekt, weil das Pooling über OP01–OP16
`T_sigma` mit dem Zwischen-OP-Versatz aufbläht und `dTdt_scale` schrumpft. Die
A-Startzeile dort ist besonders zu beachten. Die Sonde aus Schritt 1 läuft auch
auf diesen Bündeln:

```bash
python3 PINNmodulusTwo/tools/data_probe.py \
    --data-cache PINNmodulusTwoExtProfiles/data_cache \
    --ops OP01 OP02 OP03 OP04 OP05 OP08 OP09
```

---

## Was am Ende zurückkommen soll

**An den lokalen Assistenten:** fasse für den Nutzer zusammen und committe die
Doku-Änderungen (nicht die Daten):

1. Die vollständige Ausgabe von `data_probe.py` und ob A den angenommenen
   ~119/30 entspricht.
2. Die Antwort auf TODO-1: Signal oder Diskretisierungsrauschen über 5 s.
3. Die **echten** MAE-Zahlen aus `metrics.txt`, eingetragen in
   `README_ERSTER_TEST.md` Kapitel 6 anstelle der synthetischen.
4. Ob `[ABORT] epoch 1` weg ist und wie sich der `[SATURATED]`-Zählstand über
   die Epochen entwickelt hat.
5. Falls A deutlich abweicht: welche Aussagen aus `UEBERGABE_2026-08-27.txt`
   Kapitel 4 dadurch hinfällig sind.

---

## Weiterführende Dokumente

| Datei | Inhalt |
|---|---|
| `../UEBERGABE_2026-08-27.txt` | Der vollständige Stand: Ursache, was A ist, TODOs, OP-Plan, Fallstricke |
| `README_ERSTER_TEST.md` | Der Lauf im Detail — **das Hauptdokument** |
| `ARCHITECTURE.md` Kap. 3/3.1 | Mechanismus, alle Messtabellen, die Sackgassen |
| `README_GPU_SERVER.md` | Server-Setup, Treiber, Wheels, `rsync` der Daten |
| `README.md` | Projektübersicht, alle CLI-Schalter, Stabilitätsabschnitt |
| `README_MODEL_CRITIQUE.md` | Historisch: `residual_output` war dort als *Fix* geführt, jetzt als zurückgenommen markiert |
