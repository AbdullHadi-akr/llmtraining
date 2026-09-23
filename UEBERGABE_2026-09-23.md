# Übergabe — Sitzung vom 23.09.2026

> **Wenn du nur eine Sache liest:** Konfiguration A lief auf voller Auflösung
> (Lauf 16). **OP06 hält (0.62x), OP09 fällt (1.11x)**, und die Streuung ist
> wieder unlesbar. Der Lauf war aber **nicht** der POC auf voller Auflösung:
> Das TBPTT-Fenster `k` stand in Schritten und schrumpfte dadurch von 4→16 s
> auf 0.8→3.2 s. Mit `k = 16 < lag2 = 20` bekam die lag2-Rückkopplung **nie**
> einen Gradienten. Der nächste Lauf korrigiert genau das und sonst nichts.

**Branch:** `claude/gracious-wozniak-g9xomm` · **Basis:** `main` nach PR #47
Für einen Kaltstart zuerst [`UEBERGABE_2026-09-22_ABEND.md`](UEBERGABE_2026-09-22_ABEND.md),
Abschnitt 0. Diese Seite setzt dort auf.

---

## 1. Was gemessen wurde — Lauf 16

[`GridCNN/laeufe/16_konfigA_voll.txt`](GridCNN/laeufe/16_konfigA_voll.txt),
`--subsample 2`, 60 Epochen, 3 Seeds, `k = 4→16` Schritte.

| | berichtet | Latte | Güte | unter Latte | Streuung |
|---|---|---|---|---|---|
| **OP06** | 6.64 °C | 10.8009 | **0.62x** | 3/3 | 1.52 °C |
| **OP09** | 8.60 °C | 7.7625 | **1.11x** | **1/3** | 1.76 °C |

Verdikt: **kein Ergebnis** (Streuung > ~1 °C). Voller Bericht:
[`TRAININGS_BERICHT_2026-09-23_KonfigA_voll.md`](TRAININGS_BERICHT_2026-09-23_KonfigA_voll.md).

**Zwei Eigenheiten des Logs** (Details im
[Parameterblatt](GridCNN/laeufe/16_konfigA_parameter.md)):

* Auf der Maschine lief der **Code vor PR #47**, es war nicht gepullt. Der
  Trainingspfad ist identisch, geprüft per Diff. Es **fehlt das
  Fehlerprofil** (`drift`, Bias).
* Seeds 0–1 und Seed 2 liefen in **zwei Prozessen**. Die Schlusstafel im Log
  enthält nur Seed 2. Die Tafel über alle drei steht im Bericht.

## 2. Der Befund

| | POC (Lauf 15) | Lauf 16 |
|---|---|---|
| dt | 1 s | 0.2 s |
| Lags | 1 s / 4 s | 1 s / 4 s ✓ |
| Fenster k | **4→16 s** | **0.8→3.2 s** ✗ |
| lag2 im Fenster | ja | **nie** |

In `rollout_loss` kommt `t2` erst ab `j >= lag2` aus der eigenen Vorhersage.
Bei `k ≤ lag2` lernt das Netz nie, was seine Vorhersage von vor 4 s mit ihm
anrichtet. **Das ist eine Hypothese**: Sie passt zu OP09 und zur langen
Frühphase, trennt sich aber erst durch Lauf 17 vom „fünffachen Horizont".

## 3. Was gebaut wurde — dieser PR

* **`train.fensterwarnung`**: `!! [fenster]`, sobald `k ≤ lag2` (oder das
  Startfenster `lag1` nicht erreicht). Nennt die Flags für dieselben Sekunden
  wie im POC.
* **`protokollname`**: k auch in Sekunden, z. B. `k=4->16 (0.8->3.2 s)`.
* **`train.profil_aus_checkpoint`** und **`GridCNN/tools/nachmessen.py`**:
  Fehlerprofil aus `model.pt` / `model_best.pt` nachholen, ohne Training.
* 3 neue Tests, **133 grün** (`python3 -m pytest GridCNN/tests -q`).
* Lauf 16, Parameterblatt, Bericht, Kopf des Fahrplans.

## 4. Das Nächste — in dieser Reihenfolge

1. **Auf der Maschine:** `git pull`, Gewichte von Lauf 16 sichern,
   `nachmessen.py` (Minuten). Der Prompt dafür steht unten in Abschnitt 6.
2. **Lauf 17:** `--tbptt-start 20 --tbptt 80`, sonst alles wie Lauf 16,
   geschätzt etwa 1.5–2 h.
3. Danach wie im [Fahrplan](GridCNN/FAHRPLAN.md): Schritt 2 (CFL), Schritt 3
   (Arm B). Unverändert offen: Wandterm verdrahten, Test-OPs nie berichtet,
   Datenlücken OP06 (O14) und OP09 (T0/T_fluid).

## 5. Offene Fragen an die Maschine

1. Warum endete der erste Prozess nach Seed 1 ohne Schlusstafel? Mit welchem
   Kommando lief Seed 2 (`--seed 2 --seeds 1`, `tee -a`)?
2. Sind die Checkpoints von Lauf 16 alle da (`seed{0,1,2}/model.pt`,
   `model_best.pt`)? Seed 2 liegt dort aus dem zweiten Prozess.
3. Trifft `nachmessen.py` die Zeilen `letztes ep 60` im Log? (Probe)
4. Sind die Materialdaten echt? Das ist die Vorfrage zu Schritt 2 (CFL): ein
   synthetisches `material_properties/` macht das Problem viel steifer.

---

## 6. Prompts

### 6a. Für den lokalen Bot (die Maschine mit `data_raw/`)

```text
Repo AbdullHadi-akr/llmtraining. Bitte in dieser Reihenfolge, und nach jedem
Schritt die Ausgabe vollständig in eine Datei schreiben:

1. git fetch && git checkout main && git pull
   Falls der PR "Lauf 16 ..." noch nicht gemerged ist:
   git checkout claude/gracious-wozniak-g9xomm && git pull
   Dann: git log --oneline -1 und python3 -m pytest GridCNN/tests -q
   (erwartet: 133 passed)

2. Die Gewichte von Lauf 16 sichern, BEVOR irgendetwas trainiert:
   cp -r GridCNN/artifacts/A GridCNN/artifacts/A_16_voll
   ls -la GridCNN/artifacts/A_16_voll/seed*/

3. Lauf 16 nachmessen (nur Rollout, kein Training, Minuten):
   python3 GridCNN/tools/nachmessen.py --no-physics --subsample 2 \
       --device cuda --cache data_cache --laeufe GridCNN/artifacts/A_16_voll \
       2>&1 | tee 16_konfigA_nachgemessen.txt
   Probe: die MAE je seedN/model.pt muss die Zeile "letztes ep 60" aus
   16_konfigA_voll.txt treffen (Seed 0: OP06 5.5153 / OP09 6.9282,
   Seed 1: 7.4229 / 10.0384, Seed 2: 5.7848 / 7.8347).

4. Beantworte schriftlich:
   a) Warum endete der erste Lauf-16-Prozess nach Seed 1 ohne Schlusstafel,
      und mit welchem Kommando lief Seed 2?
   b) Sind die Materialdaten unter material_properties/ echte Messdaten
      oder synthetisch? Woher stammen sie?

5. Erst danach Lauf 17 (etwa 1.5-2 h):
   python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
       --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
       --val-every 2 --device cuda --cache data_cache \
       2>&1 | tee 17_konfigA_voll_k_sekunden.txt
   Im Kopf des Logs darf KEIN "!! [fenster]" stehen, und die Zeile
   [protokoll] muss "k=20->80 (4->16 s)" zeigen. Steht dort etwas anderes,
   abbrechen: dann ist der Code nicht gepullt.
   Den Lauf in EINEM Prozess mit --seeds 3 fahren, nicht aufteilen.

Liefere: 16_konfigA_nachgemessen.txt, 17_konfigA_voll_k_sekunden.txt
(vollständig, erst wenn die Schlusstafel "3 Seed(s)" da ist) und die
Antworten zu 4a/4b.
```

### 6b. Für die nächste Session hier

```text
Repo AbdullHadi-akr/llmtraining, Branch main (nach dem PR "Lauf 16 ...").
Lies zuerst UEBERGABE_2026-09-23.md, dann den Kopf von GridCNN/FAHRPLAN.md
und TRAININGS_BERICHT_2026-09-23_KonfigA_voll.md.

Stand: Konfiguration A auf voller Auflösung (Lauf 16, --subsample 2):
OP06 0.62x, OP09 1.11x (1/3 Seeds), Streuung 1.5-1.8 C -> kein Ergebnis.
Befund: k stand in Schritten (0.8->3.2 s statt 4->16 s wie im POC),
k=16 < lag2=20, die lag2-Rückkopplung bekam nie einen Gradienten.

Ich lade dir hoch: 16_konfigA_nachgemessen.txt (Fehlerprofil von Lauf 16
aus den Gewichten), 17_konfigA_voll_k_sekunden.txt (Lauf 17, k=20->80 =
4->16 s) und die Antworten der Maschine zu den Materialdaten und zum
Prozessabbruch von Lauf 16.

Werte aus: (1) Trifft die Probe in nachmessen (MAE = "letztes ep 60")?
(2) drift und Bias von Lauf 16 auf OP09. (3) Lauf 17 gegen die
Entscheidungstabelle in Abschnitt 5 des Berichts. Dann Bericht, Fahrplan,
Übergabe, PR. Sag mir immer nur eine Sache: was der nächste Schritt ist.
```
