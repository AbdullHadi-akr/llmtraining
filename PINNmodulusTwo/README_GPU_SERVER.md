# PINNmodulusTwo auf einem NVIDIA-GPU-Server

Schritt-für-Schritt vom frischen Linux-Server (Ubuntu 22.04/24.04, SSH-Zugang mit
`sudo`) bis zum laufenden Training. Der Code wählt das Device seit dem
GPU-Umbau automatisch: `--device auto` (Default) nimmt die GPU, wenn eine da
ist, sonst die CPU.

> **Kurzfassung für Ungeduldige**
> ```bash
> nvidia-smi                                   # Treiber da?
> python3 -m venv .venv && source .venv/bin/activate
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -r PINNmodulusTwo/requirements-gpu.txt
> rsync -avz <lokal>/data_cache <server>:<repo>/PINNmodulusTwo/          # Daten!
> rsync -avz <lokal>/material_properties <server>:<repo>/PINNmodulusTwo/ # Daten!
> cd PINNmodulusTwo && python3 train.py --epochs 2 --subsample 40 --device cuda
> ```

---

## 1. Treiber prüfen und ggf. installieren

```bash
nvidia-smi
```

Zeigt das eine Tabelle mit GPU-Namen und Treiberversion → weiter zu Schritt 2.

Falls `nvidia-smi: command not found`:

```bash
ubuntu-drivers devices                 # empfohlenen Treiber anzeigen
sudo apt update
sudo apt install -y nvidia-driver-550  # oder die als "recommended" markierte Version
sudo reboot
```

Nach dem Reboot noch einmal `nvidia-smi` — jetzt muss die GPU auftauchen.

**Das CUDA-Toolkit muss nicht separat installiert werden.** Die PyTorch-Wheels
bringen die komplette CUDA-Runtime mit; gebraucht wird nur der Kernel-Treiber.
`apt install nvidia-cuda-toolkit` ist hier nicht nötig und zieht oft eine
veraltete CUDA-Version nach.

Welcher Wheel-Index zu welchem Treiber passt (dank CUDA *minor version
compatibility* ist ein Treiber ≥ 525 für alle CUDA-12.x-Wheels ausreichend):

| Treiberversion (`nvidia-smi`) | empfohlener Index |
|---|---|
| ≥ 525 | `cu121` |
| ≥ 550 | `cu124` |
| ≥ 560 | `cu126` |
| ≥ 570 (Blackwell / RTX 50xx, sm_120) | `cu128` |

Im Zweifel: den neuesten Treiber installieren und `cu124` nehmen.

---

## 2. Repo und virtuelle Umgebung

```bash
git clone <repo-url> llmtraining
cd llmtraining
git checkout claude/nvidia-gpu-server-setup-phnzof

sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 3. PyTorch mit CUDA installieren

**Reihenfolge ist wichtig:** erst `torch` vom CUDA-Index, danach der Rest.
Sonst zieht pip die CPU-Variante von PyPI.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r PINNmodulusTwo/requirements-gpu.txt
```

Verifikation:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Erwartet z. B. `2.5.1+cu121 12.1 True NVIDIA A100-SXM4-40GB`.
Steht dort `False` → siehe Troubleshooting unten.

> Die `requirements.txt` im Repo-Root **nicht** verwenden: das ist ein
> UTF-16-kodierter Windows-`pip freeze` (inklusive `pywinpty`) und installiert
> die CPU-Wheels.

---

## 4. NVIDIA Modulus

```bash
pip install nvidia-modulus
```

Gebraucht werden nur `modulus.models.layers.FCLayer`,
`modulus.models.meta.ModelMetaData` und `modulus.models.module.Module`
(siehe `model.py`) — **nicht** `modulus.sym` / `physicsnemo.sym`. Der Import
in `model.py` gibt bei fehlendem Paket eine Meldung mit genau diesem Hinweis aus.

**Fallback, falls die Installation an Torch-Versionskonflikten scheitert:** das
NGC-Container-Image benutzen, dort ist alles vorinstalliert:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/modulus/modulus:24.09 bash
```

(Voraussetzung: `nvidia-container-toolkit` installiert, dann testet
`docker run --rm --gpus all ubuntu nvidia-smi` das Setup.)

---

## 5. Daten übertragen — der eigentliche Stolperstein

**Die Eingangsdaten liegen nicht im Git.** Die `.gitignore` behält nur
`*.py`, `*.ipynb`, `README*` sowie `config.yaml`/`requirements-gpu.txt`.
Nach dem `git clone` fehlen also zwei Ordner:

| Ordner | wird gebraucht von | Inhalt |
|---|---|---|
| `PINNmodulusTwo/data_cache/` | `data.py` (`_resolve_data_cache`) | `OP01.npz`, `OP02.npz`, … |
| `PINNmodulusTwo/material_properties/` | `materials.py` | `constants.yaml`, `Cell Center/*.csv`, `JR1 Center/*.csv` |

Vom lokalen Rechner aus (Pfade anpassen):

```bash
rsync -avz --progress \
  ~/batterysurrogatemodell/PINNmodulusTwo/data_cache/ \
  user@gpu-server:~/llmtraining/PINNmodulusTwo/data_cache/

rsync -avz --progress \
  ~/batterysurrogatemodell/PINNmodulusTwo/material_properties/ \
  user@gpu-server:~/llmtraining/PINNmodulusTwo/material_properties/
```

Prüfen (auf dem Server):

```bash
ls PINNmodulusTwo/data_cache/            # OP01.npz OP02.npz OP03.npz ... + --test-op
ls PINNmodulusTwo/material_properties/   # constants.yaml, "Cell Center", "JR1 Center"
```

`data.py` fällt alternativ auf `battery_surrogate_agenticWorkflow/data_cache/`
zurück, falls der lokale Ordner fehlt.

*Alternative:* die Roh-CSVs mitschicken und den Cache auf dem Server erzeugen
lassen — `python3 PINNmodulusTwo/generate_cache.py` (nutzt
`battery_surrogate_agenticWorkflow/src/battery_surrogate/data/assemble.py`).

---

## 6. Smoke-Test

Zwei Stufen: erst prüfen, ob überhaupt etwas auf der GPU rechnet, dann ob das
Training konvergiert. Beides zusammen dauert wenige Minuten — deutlich billiger
als nach 20 Stunden Benchmark festzustellen, dass die Verluste explodiert sind.

### 6.1 GPU-Kurzcheck (< 1 Minute)

```bash
cd ~/llmtraining/PINNmodulusTwo
python3 train.py --epochs 2 --subsample 40 --device cuda
```

Bewusst `--device cuda` statt `auto`: so bricht ein falsch aufgesetzter Server
mit einer klaren Fehlermeldung ab, statt still auf der CPU zu rechnen.

Erwartete Ausgabe (Auszug):

```
[device] cuda:0 NVIDIA A100-SXM4-40GB  39.6 GiB  sm_80  torch=2.5.1+cu121 cuda=12.1
[CFL OK] Δt=4.000s, Δt_max≈...
OPs=['OP01', 'OP02', 'OP03'] n_config=7 ...
model params=...
  epoch   1  L_data=...  L_phys_bal=...
```

Parallel in einer zweiten Session `nvidia-smi` — der Python-Prozess muss dort
mit belegtem Speicher auftauchen.

### 6.2 Konvergenz-Smoke-Test

Prüft, ob das Training tatsächlich konvergiert, **bevor** der große Benchmark startet.

```bash
cd ~/llmtraining
source .venv/bin/activate

python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
```

Was dabei passiert:

- trainiert zwei Modelle: `w_phys=0.0` (data-only) und `w_phys=0.1` (mit Physik)
- nur 5 Epochen, nur 2 OPs (OP01, OP02)
- Test auf OP07 (held-out; im Full Benchmark ist OP07 der Val-OP)
- Laufzeit: ~5 min auf CPU, auf der GPU entsprechend weniger

Erwartete Outputs:

```
PINNmodulusTwo/artifacts/smallBench_results.txt      -> PASS/FAIL-Bericht
PINNmodulusTwo/artifacts/smallBench_convergence.png  -> Loss-Kurven über die Epochen
```

Auswerten:

```bash
cat PINNmodulusTwo/artifacts/smallBench_results.txt
ls -lh PINNmodulusTwo/artifacts/smallBench_convergence.png
```

Am Ende des Berichts steht entweder

```
✓ ALL CHECKS PASSED - Ready for full benchmark!
```

oder

```
✗ SOME CHECKS FAILED - Review issues above before full benchmark
```

### Entscheidung

- **ALL CHECKS PASSED** → weiter zu [Kapitel 7](#7-erste-testreihe--läuft-in-7-stunden-durch).
- **SOME CHECKS FAILED** → **stopp**, erst die Ursache klären:
  - Loss explodiert (inf/NaN) → CFL-Problem, `--subsample` verkleinern (z. B. `--subsample 2`)
  - Test-MAE > 20 °C → das Modell lernt nicht richtig
  - noch nicht konvergiert → mit mehr Epochen gegenprüfen

---

## 7. Erste Testreihe — läuft in ~7 Stunden durch

Alles, was über Nacht fertig wird und danach entscheidet, ob und wie sich die
großen Läufe aus [Kapitel 8](#8-große-benchmarks--tage) lohnen. Die Reihenfolge
ist bindend: jeder Schritt ist ein Tor für den nächsten.

| # | Schritt | Laufzeit | Weiter nur wenn |
|---|---|---|---|
| 1 | Smoke-Test | ~8 min | `ALL CHECKS PASSED` |
| 2 | Zeit pro Epoche messen | ~10 min | — |
| 3 | Range-Probe der Gewichte (`--epochs 20`) | ~6 h | — |
| | **gesamt** | **~6,5 h** | |

Der Trick, mit dem das in eine Nacht passt, ist `--epochs 20` statt 60. Für die
Frage „in welcher Dekade wirkt ein Gewicht überhaupt" reicht das; für den
Endwert eines Modells nicht. Kapitel 8 fährt dann mit 60 Epochen.

> **Warum nicht mehr in 7 h?** Bei `--subsample 2` ist der Rollout ~7000
> sequentielle Zeitschritte pro OP und Epoche. Eine Epoche über 5 Trainings-OPs
> kostet grob **1,5–2,5 min**, ein Training mit 20 Epochen also ~40 min. Sieben
> Stunden sind damit rund **zehn Trainings** — mehr ist in einer Nacht nicht
> drin. Schritt 2 misst die echte Zahl; wenn sie deutlich abweicht, rechne das
> Budget neu, bevor du Schritt 3 startest.

**Datenaufteilung:** Training `OP01–OP05`, Validierung `OP06` (entscheidet die
Auswahl), Test `OP07` (wird nur berichtet und fließt in keine Auswahl ein).

---
### 7.0 Was variabel ist — und was nicht

Damit klar ist, woran man drehen kann und woran nicht.

> ⚠️ **`config.yaml` und die Benchmarks haben unterschiedliche Defaults.** Wer
> `train.py` ohne Flags startet, bekommt ein *anderes Modell* als die Benchmarks
> — anderes Zeitraster, anderer History-Modus. Die Spalte „Bench" unten ist die,
> die für Kapitel 7 und 8 gilt.

**Struktur — konfigurierbar, aber nicht trainiert:**

| Flag | `config.yaml` | Bench | Bedeutung |
|---|---|---|---|
| `--subsample` | **40** | **2** | Datenraster: `dt = 0.1 s × subsample` → 0.2 s. Bestimmt die Rollout-Länge und damit die Laufzeit. CFL-Grenze ~0.241 s |
| `--delta-grid` | `0.2` s | `0.2` s | **Anker** der History: der Block ist `[T(t−Δgrid), rate₁, …]`. Unabhängig von `--subsample` |
| `--rate-lags` | **`5 25`** | **`5 20`** | kumulative Segmentlängen in Sekunden; jedes Segment beginnt, wo das vorige endete |
| `--history-mode` | **`raw`** | **`hybrid`** | `hybrid` = Anker + Raten, `raw` = reiner Lag-Stapel |
| `--k-max` | **4** | **2** | nur im `raw`-Modus wirksam; im `hybrid`-Modus folgt `k` aus der Zahl der `rate_lags` und das Flag wird ignoriert |
| `--width` / `--depth` | `128` / `4` | `128` / `4` | MLP-Geometrie |
| `--time-deriv` | `bdf2` | `bdf2` | Zeitableitung im Physik-Residuum |

**Lernparameter** (die einzigen Dinge, die der Gradient anfasst): MLP-Gewichte,
das per-Layer `β` des Swish, und die beiden Physik-Gains `src_gain`/`diff_gain`.
Die History-Struktur wird **nicht** gelernt — kein lernbares `δ`, keine
Lag-Gates. Wer sie optimieren will, sweept sie mit `benchmark_arch.py` (8.2).

**Optimierung und Loss:**

| Flag | `config.yaml` | Bench | Bedeutung |
|---|---|---|---|
| `--epochs` | `60` | `60` | in Kapitel 7 bewusst `20` |
| `--lr` | `2e-3` | `2e-3` | Basis-Lernrate |
| `--gain-lr-mult` | `25.0` | `25.0` | `src_gain`/`diff_gain` lernen 25× schneller, sonst bleiben sie bei 1.0 |
| `--grad-clip` | **0** (aus) | **1.0** | maximale Gradientennorm |
| `--w-phys` / `--w-bc` | `0.1` / `0.1` | gesweept | Gewichte von Physik- und BC-Term |
| `--phys-norm` | `0` | `0` | `0` = adaptiver EMA, `>0` = fester Divisor |
| `--batch-phys` / `--batch-bc` | `256` / `128` | `256` / `128` | Kollokationspunkte — hier liegt der GPU-Hebel |
| `--seeds` | — | `0` | ein Trainingslauf je Seed, Bewertung über den Mittelwert (nur Benchmarks) |

**Daten:** `--ops` (Training), `--val-op` (Auswahl), `--test-op` (nur Bericht).

`config.yaml` gilt für `train.py` und `smallBench.py`; die Benchmarks setzen ihre
eigenen Defaults im Skript. Die CLI überschreibt beides pro Lauf. Wenn du
`train.py` von Hand mit den Benchmark-Einstellungen laufen lassen willst:

```bash
python3 PINNmodulusTwo/train.py --subsample 2 --history-mode hybrid \
  --rate-lags 5 20 --delta-grid 0.2 --grad-clip 1.0 \
  --ops OP01 OP02 OP03 OP04 OP05 --device cuda
```

---

### 7.1 Schritt 1 — Smoke-Test (~8 min)

Vollständig in **[Kapitel 6](#6-smoke-test)** beschrieben. Kurzfassung:

```bash
cd ~/llmtraining
source .venv/bin/activate

python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cuda
python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
cat PINNmodulusTwo/artifacts/smallBench_results.txt
```

Erwartung: `✓ ALL CHECKS PASSED - Ready for full benchmark!`

Bei `[ABORT] ... loss exploded` **nicht** weitermachen — die Meldung nennt den
zuerst betroffenen Loss-Term, siehe [Kapitel 10](#10-troubleshooting).

---

### 7.2 Schritt 2 — Zeit pro Epoche messen (~10 min)

Bevor irgendetwas Langes startet: einen einzelnen kurzen Trainingslauf machen und
die gemessene Zeit pro Epoche ablesen.

```bash
python3 PINNmodulusTwo/train.py --ops OP01 OP02 OP03 OP04 OP05 \
  --subsample 2 --epochs 5 --history-mode hybrid --rate-lags 5 20 \
  --grad-clip 1.0 --device cuda
```

Das Log meldet ab Epoche 1:

```
  epoch   1  L_data=...  [112.4s/epoch, this run ~7 min left]
```

Damit rechnest du selbst:

```
Sekunden/Epoche × 60 Epochen ÷ 3600        = Stunden pro Gitterpunkt
Stunden/Punkt × Punkte × Seeds ÷ 24        = Tage für den Sweep
```

Bei 112 s/Epoche wären das 1,9 h pro Punkt — 100 Punkte also ~7,8 Tage. Wenn das
zu viel ist, sind die Hebel in dieser Reihenfolge wirksam: weniger `--epochs`,
kleineres Gitter, größeres `--subsample`.

---

#### Nebenbei: wie viel bringt die GPU hier eigentlich?

Derselbe Lauf mit `--device cpu` beantwortet das in denselben zehn Minuten:

```bash
python3 PINNmodulusTwo/train.py --ops OP01 OP02 --subsample 2 --epochs 3 \
  --history-mode hybrid --rate-lags 5 20 --grad-clip 1.0 --device cpu
```

Die Sekunden pro Epoche aus beiden Läufen ins Verhältnis setzen — **erwarte
keinen großen Faktor.** Der Grund steckt in der Struktur der Last:

| Anteil | parallelisierbar? | GPU-Gewinn |
|---|---|---|
| Rollout: ~7000 sequentielle Schritte je OP und Epoche | nein, jeder Schritt braucht den vorherigen | gering |
| Python-Schleife und Kernel-Starts, ~37.000 Iterationen je Epoche | nein, identisch auf beiden | keiner |
| MLP-Matmuls je Schritt (363 Punkte × 128 breit) | ja, aber winzig | klein |
| Physik-Residuum (`--batch-phys`, doppeltes Autograd) | ja, echte Parallelarbeit | groß |

Die GPU gewinnt dort, wo tatsächlich große Batches anfallen — beim
Physik-Residuum. Der Rollout, der die Laufzeit dominiert, ist latenzgebunden und
läuft auf beiden ähnlich schnell. Deshalb steht in der Troubleshooting-Tabelle
auch „läuft auf der GPU kaum schneller" als *erwartbar*, nicht als Fehler.

**Praktische Konsequenz:** wenn `nvidia-smi` viel freien Speicher zeigt, ist
`--batch-phys 2048 --batch-bc 1024` der Hebel, der die GPU wirklich nutzt — nicht
ein breiteres Netz.

---

### 7.3 Schritt 3 — Range-Probe: in welcher Dekade wirken die Gewichte? (~6 h)

Bevor ein Gitter Stunden investiert, um Unterschiede *innerhalb* eines Bereichs
aufzulösen, klärt die Probe, ob dieser Bereich überhaupt der richtige ist — und
ob die Gewichte den Fehler überhaupt bewegen.

Statt eines Gitters läuft ein **Kreuz** durch einen gemeinsamen Mittelpunkt:
jedes Gewicht wird über die Dekaden `[0, 0.001, 0.01, 0.1, 1.0]` gefahren,
während das andere im Zentrum steht. **9 Punkte statt 25.**

```bash
# 9 Punkte x 1 Seed x 20 Epochen = 9 Trainings ~ 6 h
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --epochs 20 --device cuda
```

> **Ein Seed, mit Absicht.** `--seeds 0 1 2` wären 27 Trainings und damit ~18 h —
> das sprengt die Nacht. Der Preis: das Verdikt kann die gefundenen Unterschiede
> nicht vom Init-Rauschen trennen und sagt das auch (`seed spread unknown`). Die
> Streuung klärt Schritt 8.1 separat. Für „welche Dekade überhaupt" reicht ein
> Seed, solange die Spannweite deutlich ist.

`benchmark_wphys_wbc_best.txt` endet dann mit einem Verdikt pro Achse:

```
RANGE PROBE - per-axis verdict:
  w_phys (at w_bc=0.1):
    0->9.12  0.001->8.94  0.01->8.71  0.1->8.83  1->10.40
    best w_phys=0.01 (val 8.710 °C), span over the decades = 1.690 °C
    span exceeds the seed spread (0.210 °C) - worth a grid, centred on the best decade.
```

Liegt die Spannweite **unter** der Seed-Streuung, bewegt dieses Gewicht den
Fehler nicht — dann spart man sich das Gitter dafür.

---

### 7.4 Auswertung — was als Nächstes?

```bash
cat PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
```

Am Ende steht das Verdikt je Achse. Daraus folgt direkt, was aus Kapitel 8 sich
lohnt:

| Befund in der Probe | Konsequenz |
|---|---|
| Spannweite über die Dekaden **groß**, klares Minimum | 5×5-Gitter (8.3), zentriert auf diese Dekade |
| Spannweite **klein / flach** | Gitter für dieses Gewicht überspringen — es bewegt den Fehler nicht |
| beide Gewichte flach | zuerst Architektur-Benchmark (8.2): das Problem liegt woanders |
| Läufe divergiert (`[SKIP]`) | nicht weitermachen, [Kapitel 10](#10-troubleshooting) |

Die Probe lief mit **einem** Seed, das Verdikt sagt deshalb „seed spread
unknown". Ob die gefundenen Unterschiede echt sind, klärt Schritt 8.1.

---

### 7.5 Komplette Session zum Kopieren

```bash
# ---------------------------------------------------------------------------
# SCHRITT 1: SMOKE-TEST  (~8 min)
# ---------------------------------------------------------------------------
cd ~/llmtraining
source .venv/bin/activate

python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
cat PINNmodulusTwo/artifacts/smallBench_results.txt
# Erwartung: "✓ ALL CHECKS PASSED - Ready for full benchmark!"
# NUR bei PASSED weitermachen.

# ---------------------------------------------------------------------------
# SCHRITT 2: ZEIT PRO EPOCHE MESSEN  (~10 min)  <- bestimmt das Budget
# ---------------------------------------------------------------------------
python3 PINNmodulusTwo/train.py --ops OP01 OP02 OP03 OP04 OP05 \
  --subsample 2 --epochs 5 --history-mode hybrid --rate-lags 5 20 \
  --grad-clip 1.0 --device cuda
# Log: "[112.4s/epoch, this run ~7 min left]"
# 9 Punkte x 20 Epochen x (Sekunden/Epoche) / 3600 = Stunden fuer Schritt 3.
# Deutlich ueber 7 h? Dann --epochs 10 nehmen.

# ---------------------------------------------------------------------------
# SCHRITT 3: RANGE-PROBE  (9 Punkte, ~6 h)  -> ueber Nacht
# ---------------------------------------------------------------------------
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --epochs 20 \
  --device cuda > probe.log 2>&1 &
echo $! > probe.pid

echo "PID:  $(cat probe.pid)"
echo "Live: tail -f probe.log"
echo "Stop: kill \$(cat probe.pid)"
# Terminal kann jetzt geschlossen werden.

# ---------------------------------------------------------------------------
# AM NAECHSTEN MORGEN
# ---------------------------------------------------------------------------
grep -A20 'RANGE PROBE' PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
```

---

### 7.6 Checkliste

Vor dem Start:

- [ ] venv aktiviert (`source .venv/bin/activate`)
- [ ] `ls data_cache/*.npz` zeigt OP01–OP07
- [ ] Smoke-Test gelaufen und PASSED
- [ ] Zeit pro Epoche gemessen und das 7-h-Budget nachgerechnet
- [ ] `nohup` oder `tmux` benutzt, damit der Lauf die SSH-Session überlebt

Danach:

- [ ] Verdikt je Achse gelesen — welche Dekade, und bewegt sich überhaupt etwas?
- [ ] entschieden, welcher Lauf aus Kapitel 8 als Nächstes kommt

---

## 8. Große Benchmarks — Tage

Alles hier läuft mit **60 Epochen** und dauert Tage, nicht Stunden. Erst starten,
wenn Kapitel 7 durch ist und die Probe gesagt hat, wo es sich lohnt.

| Lauf | Trainings | Laufzeit |
|---|---|---|
| 8.1 Seed-Streuung an einem Punkt | 3–5 | ~6–10 h |
| 8.2 Architektur-Benchmark (16 Konfigs) | 16 | ~1–1,5 Tage |
| 8.3 5×5-Gitter der Gewichte | 25 | ~1,5–2 Tage |
| 8.3 mit `--extended-grid` (10×10) | 100 | ~6–8 Tage |

Mit Seeds multipliziert sich das entsprechend. `--epochs 20` drittelt alles und
ist für Vergleiche zwischen Konfigurationen meist ausreichend.

---
### 8.1 Seed-Streuung an einem Gitterpunkt

**Der erste Lauf dieses Kapitels**, weil er entscheidet, wie die anderen zu
lesen sind. Er beantwortet: bewegt sich der Fehler zwischen zwei Konfigurationen
mehr, als er sich zwischen zwei Zufalls-Initialisierungen *derselben*
Konfiguration bewegt? Ist die Antwort nein, sind die Rangfolgen aller folgenden
Läufe Rauschen — und ein feineres Gitter macht es nicht besser.

```bash
# 5 Seeds an einem Punkt, ~10 h. Mit --epochs 20 sind es ~3 h.
python3 PINNmodulusTwo/benchmark_wphys_wbc.py \
  --w-phys 0.05 --w-bc 0.1 --seeds 0 1 2 3 4 --epochs 60 --device cuda
```

Dann in `benchmark_wphys_wbc_best.txt` die Spalte `+/-` ansehen — das ist die
Standardabweichung über die fünf Seeds.

- **Streuung klein** (deutlich unter den Unterschieden, die du in einer Heatmap
  vergleichen willst): die langen Sweeps liefern lesbare Rangfolgen.
- **Streuung groß**: ein feineres Gitter bringt nichts, weil der Sieger ohnehin
  vom Zufall bestimmt wird. Dann in 8.2 und 8.3 **mit `--seeds 0 1 2`**
  arbeiten und die Gitter kleiner halten.

---

### 8.2 Architektur-Benchmark

Misst Breite, Tiefe und History-Lags — Werte, die im Gewichte-Sweep ungemessen
festliegen. Läuft **eine Achse nach der anderen** gegen eine gemeinsame
Baseline, statt ein Produktgitter aufzuspannen: die Frage ist, welcher Regler den
Fehler überhaupt bewegt. `width × depth × lags` wären mehrere hundert Trainings,
achsenweise sind es zwölf.

```bash
nohup python3 PINNmodulusTwo/benchmark_arch.py --device cuda --seeds 0 1 2 \
  > benchmark_arch.log 2>&1 &

# oder nur eine Achse, z. B. die Lags:
python3 PINNmodulusTwo/benchmark_arch.py --axes lags --seeds 0 1 2 --device cuda
```

| Achse | Werte | Konfigurationen |
|---|---|---|
| `width` | 64, 128, 256 | 3 |
| `depth` | 2, 3, 4, 6 | 4 |
| `lags` | 5+20, 2+10, 10+60, 5+20+60, 30 | 5 |
| `dgrid` | 0.2, 0.5, 1.0, 2.0 s | 4 |
| | **gesamt** | **16** |

`dgrid` ist der **Ankerabstand** der Hybrid-History: der Block ist
`[T(t−Δgrid), rate₁, rate₂]`, und die Ratensegmente laufen von diesem Anker aus
rückwärts. Bisher war er fest an den Datenraster-Schritt gekoppelt — jetzt ist er
ein eigener Regler mit Default 0.2 s (also unverändertes Verhalten).

Die Baseline (`--width 128 --depth 4 --rate-lags 5 20`) taucht in jeder Achse
einmal auf. Das ist Absicht: jede Achse braucht ihren eigenen Bezugspunkt, und
dieselbe Konfiguration mehrfach zu trainieren zeigt nebenbei die reine
Seed-Streuung.

**Outputs**

```
PINNmodulusTwo/artifacts/benchmark_arch.csv       -> eine Zeile pro Konfiguration
PINNmodulusTwo/artifacts/benchmark_arch.png       -> ein Panel je Achse, Fehlerbalken = Seed-Streuung
PINNmodulusTwo/artifacts/benchmark_arch_best.txt  -> Ranking, Baseline-Vergleich, Rausch-Verdikt
```

Der entscheidende Abschnitt in `benchmark_arch_best.txt` heißt **"Span per
axis"**: die Spannweite der Validierungs-MAE über jede Achse. Liegt sie unter der
Seed-Streuung, steht dort *"this knob does not matter here"* — dann lohnt es
nicht, diesen Regler weiter zu drehen.

**Laufzeit**

| Umfang | Trainings | Laufzeit bei 2 h/Training |
|---|---|---|
| nur `width`, 1 Seed | 3 | ~6 h |
| nur `lags`, 1 Seed | 5 | ~10 h |
| alle Achsen, 1 Seed | 12 | **~1 Tag** |
| alle Achsen, 3 Seeds | 36 | ~3 Tage |

> Setze die in 7.2 gemessene Zeit pro Training ein, statt diese Tabelle zu
> übernehmen. Die Werte gelten außerdem für `width=128`: breitere Netze kosten
> mehr pro Schritt, aber der Rollout ist latenzgebunden (~7000 sequentielle
> Zeitschritte), nicht rechengebunden — `width=256` skaliert daher schwächer als
> die vierfache FLOP-Zahl vermuten lässt.
>
> Mit `--epochs 20` statt 60 drittelt sich alles. Für die Frage „welche Achse
> bewegt überhaupt etwas" reicht das meist.

---

### 8.3 5×5-Gitter der Loss-Gewichte

Sucht das optimale Paar `(w_phys, w_bc)` auf dem 5×5-Standardgitter — zentriert
auf die Dekade, die die Range-Probe aus 7.3 als wirksam ausgewiesen hat.

```bash
cd ~/llmtraining
source .venv/bin/activate

# Werte an die Dekade aus der Probe anpassen:
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --device cuda \
  --w-phys 0.003 0.01 0.03 0.1 0.3 --w-bc 0.03 0.1 0.3 0.7 1.0 \
  > benchmark_grid.log 2>&1 &
echo $! > benchmark.pid

echo "Gitter gestartet - PID $(cat benchmark.pid), Log: benchmark_grid.log"
```

Durch `nohup` läuft der Benchmark weiter, wenn das Terminal geschlossen wird.
Wer lieber interaktiv mitliest, nimmt stattdessen `tmux new -s bench` und startet
das Kommando ohne `nohup`/`&` (Detach mit `Ctrl-b`, dann `d`).

| Parameter | Wert |
|---|---|
| Grid | 5×5 = 25 Punkte (Default) |
| `w_phys` / `w_bc` | über `--w-phys` / `--w-bc` auf die gefundene Dekade legen |
| Trainings-OPs | OP01–OP05 |
| Val-OP (Auswahl) | OP06 |
| Test-OP (nur Bericht) | OP07 |
| Seeds | 1 (`--seeds`, siehe 8.4) |
| Epochen | 60 pro Gitterpunkt |
| Laufzeit | 1,5–2,5 h/Punkt → **~1,5–2 Tage** |

Mit `--epochs 20` sind es ~14 h. Wenn 8.1 eine große Seed-Streuung gezeigt hat,
ist ein kleines Gitter mit drei Seeds aussagekräftiger als ein großes mit einem.

**`--extended-grid`** schaltet auf 10×10 = 100 Punkte um, also **~6–8 Tage bei
einem Seed**. Das lohnt praktisch nie: die Probe hat die Dekade bereits
eingegrenzt, und 100 Punkte auf einem Seed liefern vor allem das Minimum aus 100
Ziehungen — siehe das Rausch-Verdikt in 8.4.

---

### 8.4 Seeds — wie viele Läufe pro Gitterpunkt

Standard ist **ein** Seed pro Punkt. Damit ist nicht entscheidbar, ob der
Unterschied zwischen zwei Zellen der Heatmap echt ist oder nur unterschiedliche
Initialisierung: der Sweep berichtet das Minimum aus 100 Ziehungen, und ein Teil
davon ist schlicht der glücklichste Startwert.

```bash
# drei Seeds pro Punkt: jeder Punkt wird über den Mittelwert bewertet
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --seeds 0 1 2 --device cuda
```

Jeder Punkt wird dann über den **Mittelwert** seiner Seeds bewertet und trägt die
Standardabweichung mit (Spalten `MAE_val_std_C` / `MAE_test_std_C` in der CSV,
Spalten `+/-` in der Tabelle). `benchmark_wphys_wbc_best.txt` vergleicht am Ende
den Abstand des Siegers zum Zweitplatzierten mit der Seed-Streuung und sagt
explizit, ob die Rangfolge belastbar ist oder im Rauschen liegt.

> **Laufzeit skaliert linear.** 25 Punkte × 3 Seeds = 75 Trainings, also
> ~4,5–6 Tage bei 60 Epochen — mit `--epochs 20` rund 1,7 Tage.

---

### 8.5 Erwartete Outputs

```
PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv               -> alle 100 Punkte
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_heatmap.png       -> 2D-MAE-Heatmap (Val)
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_convergence.png   -> Loss-Kurven (Ecken + Best)
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt          -> beste Kombination + Tabelle
PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/*.pt            -> 100 Modelle (mehrere GB!)
```

---

### 8.6 Monitoring

Gilt für jeden der Läufe oben — Log- und PID-Datei entsprechend ersetzen (`probe.log`/`probe.pid`, `benchmark_arch.log`, `benchmark_grid.log`).

```bash
# Live mitlesen (Ctrl-C beendet nur tail, nicht den Benchmark)
tail -f benchmark_grid.log

# Welcher Punkt läuft gerade?
grep "Training w_phys" benchmark_grid.log | tail -3
#   [23/100] Training w_phys=0.05, w_bc=0.1

# Restlaufzeit
grep "ETA" benchmark_grid.log | tail -3
#   Train time: 118.3 min | ETA: 11712.0 min

# Wie viele Ergebnisse stehen schon in der CSV? (steigt bis 101 = Header + 100)
wc -l PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv

# Läuft der Prozess noch?
ps -p $(cat benchmark.pid)

# GPU-Auslastung
watch -n 2 nvidia-smi
```

**Batchgrößen.** Der Geschwindigkeitsgewinn kommt weniger aus dem Netz selbst
(kleines MLP) als daraus, dass auf der GPU deutlich größere Physik-Batches
bezahlbar sind. Wenn `nvidia-smi` viel freien Speicher zeigt, lohnt ein Neustart
mit z. B. `--batch-phys 2048 --batch-bc 1024`.

**Mehrere GPUs.** Der Code nutzt genau eine GPU pro Prozess (kein DDP). Für
parallele Läufe je einen Prozess pro Karte starten — `--device cuda:0`,
`--device cuda:1`, alternativ `CUDA_VISIBLE_DEVICES=1 ... --device cuda`. Achtung:
alle Läufe schreiben nach `PINNmodulusTwo/artifacts/` und überschreiben sich
gegenseitig, also Artefakte pro Lauf wegsichern oder `--model-dir` setzen.

---

### 8.7 Auswertung nach Abschluss

```bash
cat PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
```

Erwartete Ausgabe (Ende der Datei):

```
Selection ran on OP06 (MAE_val); OP07 (MAE_test) was never used to choose anything.
BEST (by MAE_val): w_phys=0.1, w_bc=0.3
  -> val 8.220°C, test 8.641°C, in-time 6.432°C
  Report the test number. MAE_val is optimistic: it is the minimum over 100 grid points.
  NOTE: one seed per point - the ranking cannot be separated from init noise.
  Re-run with --seeds 0 1 2 to find out.
Total runtime: 4.85 hours (291.2 min)
Checkpoints dir: /home/user/llmtraining/PINNmodulusTwo/artifacts/checkpoints_wphys_wbc
```

**Berichtet wird die Test-Zahl, nicht die Val-Zahl.** Die Val-MAE ist das Minimum
über alle Gitterpunkte und damit zu optimistisch.

```bash
# Plots
ls -lh PINNmodulusTwo/artifacts/benchmark_wphys_wbc*.png

# Top 10 nach Val-MAE - das ist die Spalte, auf der ausgewaehlt wurde.
# Spalten: 6=MAE_in_C  7=MAE_val_C  8=MAE_val_std_C  9=MAE_test_C  10=MAE_test_std_C
head -1 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv && \
  tail -n +2 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv | sort -t',' -k7 -n | head -10

# Dieselben Punkte mit ihrer Test-MAE (Spalte 9) - die Zahl, die berichtet wird.
# Wenn die Reihenfolge hier stark von der obigen abweicht, ist die Auswahl instabil.
tail -n +2 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv | sort -t',' -k7 -n | \
  head -10 | cut -d',' -f1,2,7,8,9,10

# Checkpoints: Anzahl und Platzbedarf
ls PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/ | wc -l   # sollte 100 sein
du -sh PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/
```

---

### 8.8 Abbrechen und neu starten

```bash
kill $(cat benchmark.pid)
# falls das nicht greift:
pkill -f benchmark_wphys_wbc
ps -p $(cat benchmark.pid)   # "No such process" = gestoppt
```

**Es gibt keine Resume-Funktion** — ein Neustart beginnt wieder bei Punkt 1.

---

### 8.9 Weiter mit den besten Gewichten

```bash
# Werte aus benchmark_wphys_wbc_best.txt einsetzen
python3 PINNmodulusTwo/train.py \
    --epochs 100 --w-phys 0.1 --w-bc 0.3 --subsample 2 \
    --ops OP01 OP02 OP03 OP04 OP05 --test-op OP07 --device cuda
```

Bestes Checkpoint laden (Dateiname folgt dem Schema `model_p<w_phys>_b<w_bc>.pt`,
Punkte werden zu `p`, also `w_phys=0.1, w_bc=0.3` → `model_p0p1_b0p3.pt`):

```python
# aus dem Ordner PINNmodulusTwo/ heraus ausführen
import torch
from model import RecurrentField

ckpt = torch.load("artifacts/checkpoints_wphys_wbc/model_p0p1_b0p3.pt")
model = RecurrentField(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# ckpt["bundle_stats"] enthält T_mu / T_sigma / T_span_ref zum Rückskalieren
```

---

## 9. Ergebnisse zurückholen

```bash
rsync -avz user@gpu-server:~/llmtraining/PINNmodulusTwo/artifacts/ ./artifacts/
```

Enthält je nach gelaufenem Schritt:

- `train.py`: `metrics.txt`, `training_curves.png`, `timeseries.png`, `pred_OP*.npz`
- `smallBench.py`: `smallBench_results.txt`, `smallBench_convergence.png`
- `benchmark_wphys_wbc.py`: `benchmark_wphys_wbc.csv`, `*_heatmap.png`,
  `*_convergence.png`, `*_best.txt` und `checkpoints_wphys_wbc/` (mehrere GB —
  ggf. gezielt nur das beste Checkpoint holen)

Das Benchmark-Log liegt dort, wo der Lauf gestartet wurde:

```bash
rsync -avz user@gpu-server:~/llmtraining/benchmark_grid.log ./
```

Die Plots werden mit dem `Agg`-Backend erzeugt, es wird also kein X-Server auf
dem Server gebraucht.

---

## 10. Troubleshooting

| Symptom | Ursache / Abhilfe |
|---|---|
| `torch.cuda.is_available()` → `False` | CPU-Wheel installiert (`torch.version.cuda` ist `None`) → torch neu vom `cu12x`-Index installieren. Oder Treiber fehlt/zu alt → Schritt 1. |
| `--device cuda ... torch.cuda.is_available() is False` | Genau derselbe Fall — die Fehlermeldung nennt Torch- und CUDA-Version zum Abgleich. |
| `CUDA error: no kernel image is available` | GPU zu neu für das Wheel (z. B. RTX 50xx mit `cu121`) → `cu128`-Index nehmen. |
| `CUDA out of memory` | `--batch-phys` / `--batch-data` / `--batch-bc` senken oder `--subsample` erhöhen. Belegt ein Zombie-Prozess die Karte? → `nvidia-smi`, ggf. `kill`. |
| `ModuleNotFoundError: modulus` | venv nicht aktiviert oder `pip install nvidia-modulus` fehlt → Schritt 4. |
| `FileNotFoundError: .../data_cache/OP01.npz` | Daten nicht übertragen → Schritt 5. |
| `FileNotFoundError: .../material_properties/constants.yaml` | dito → Schritt 5. |
| `[CFL WARN]` beim Start | Zeitschritt zu groß für die Diffusion — `--subsample` verkleinern (z. B. `--subsample 2`) oder `--grad-clip 1.0` setzen. Unabhängig von der GPU. |
| Training bricht mit `[ABORT] ... loss exploded` ab | Gleiche Ursache wie oben; die Meldung nennt die empfohlenen Gegenmaßnahmen. |
| `No space left on device` während des Benchmarks | Die 100 Checkpoints brauchen mehrere GB → mit `--no-save-models` (gar keine) oder `--save-best-only` (nur das beste Modell) neu starten. |
| Benchmark startet nicht, "läuft schon" | Alten Prozess finden und beenden: `ps aux \| grep benchmark_wphys_wbc`, dann `kill <PID>`. |
| Loss explodiert mitten im Benchmark | Lauf stoppen (`kill $(cat benchmark.pid)`) und mit stabileren Einstellungen neu starten: `--grad-clip 2.0 --lr 0.001`. |
| Läuft auf der GPU kaum schneller | Erwartbar: pro Epoche gibt es ~7000 *sequentielle* Rollout-Schritte je OP, die sich nicht parallelisieren lassen. Batchgrößen erhöhen hilft nur dem Physik-Term (8.6). |

---

## Was der GPU-Umbau am Code geändert hat

- `device_utils.py` (neu): `resolve_device()`, `seed_everything()`, `enable_tf32()`.
- `train.py`, `smallBench.py`, `benchmark_wphys_wbc.py`:
  `--device` steht jetzt auf `auto` statt `cpu`; ein explizites `cuda` schlägt
  hart fehl, wenn keine GPU da ist, statt still auf die CPU zu wechseln.
- `train.py`: zusätzliches `--tf32`-Flag, `torch.cuda.manual_seed_all()` beim Seeding.
- `config.yaml` (neu): die Datei, die `train.py` schon immer gesucht hat — mit
  `device`/`tf32` und allen bisherigen Defaults.
- `requirements-gpu.txt` (neu): schlanke, UTF-8-kodierte Abhängigkeitsliste.
