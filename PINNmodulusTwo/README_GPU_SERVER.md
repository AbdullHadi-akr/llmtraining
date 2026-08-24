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

- **ALL CHECKS PASSED** → weiter zu Schritt 7.
- **SOME CHECKS FAILED** → **stopp**, erst die Ursache klären:
  - Loss explodiert (inf/NaN) → CFL-Problem, `--subsample` verkleinern (z. B. `--subsample 2`)
  - Test-MAE > 20 °C → das Modell lernt nicht richtig
  - noch nicht konvergiert → mit mehr Epochen gegenprüfen

---

## 7. Full Benchmark (Extended Grid)

Sucht das optimale Paar `(w_phys, w_bc)` auf einem 10×10-Gitter. **Nur starten,
wenn der Smoke-Test aus Schritt 6 PASSED ist.**

### 7.1 Start

```bash
cd ~/llmtraining
source .venv/bin/activate

nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --extended-grid --device cuda \
  > benchmark_extended.log 2>&1 &
echo $! > benchmark.pid

echo "Full Benchmark gestartet - PID $(cat benchmark.pid), Log: benchmark_extended.log"
```

Durch `nohup` läuft der Benchmark weiter, wenn das Terminal geschlossen wird.
Wer lieber interaktiv mitliest, nimmt stattdessen `tmux new -s bench` und startet
das Kommando ohne `nohup`/`&` (Detach mit `Ctrl-b`, dann `d`).

### 7.2 Was läuft

| Parameter | Wert |
|---|---|
| Grid | 10×10 = 100 Punkte (quasi-logarithmisch) |
| `w_phys` | 0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0 |
| `w_bc` | 0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1.0 |
| Trainings-OPs | OP01–OP06 |
| Val-OP (Auswahl) | OP07 |
| Test-OP (nur Bericht) | OP08 |
| Seeds | 1 (`--seeds`, siehe 7.3) |
| Epochen | 60 pro Gitterpunkt |
| Laufzeit | ~17 min/Punkt → **~28 h** (CPU-Referenzwert) |

**Val und Test sind getrennt.** Die Auswahl des besten `(w_phys, w_bc)` läuft auf
dem Val-OP, der Test-OP fließt in keine Auswahl ein und liefert deshalb die Zahl,
die man berichtet. Vorher rankte der Sweep auf demselben OP, den er als Ergebnis
meldete — das Minimum über 100 Kandidaten ist dabei systematisch zu optimistisch.

Die ~28 h stammen aus CPU-Läufen. Die echte GPU-Laufzeit steht im Log: nach jedem
Punkt wird `Train time: ... min | ETA: ... min` aus den tatsächlich gemessenen
Zeiten ausgegeben. Nach zwei, drei Punkten weißt du, woran du bist.

Ohne `--extended-grid` läuft das kleinere 5×5-Standardgitter (25 Punkte, ~7 h auf CPU).

### 7.3 Seeds — wie viele Läufe pro Gitterpunkt

Standard ist **ein** Seed pro Punkt. Damit ist nicht entscheidbar, ob der
Unterschied zwischen zwei Zellen der Heatmap echt ist oder nur unterschiedliche
Initialisierung: der Sweep berichtet das Minimum aus 100 Ziehungen, und ein Teil
davon ist schlicht der glücklichste Startwert.

```bash
# drei Seeds pro Punkt: jeder Punkt wird über den Mittelwert bewertet
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --extended-grid --seeds 0 1 2 --device cuda
```

Jeder Punkt wird dann über den **Mittelwert** seiner Seeds bewertet und trägt die
Standardabweichung mit (Spalten `MAE_val_std_C` / `MAE_test_std_C` in der CSV,
Spalten `+/-` in der Tabelle). `benchmark_wphys_wbc_best.txt` vergleicht am Ende
den Abstand des Siegers zum Zweitplatzierten mit der Seed-Streuung und sagt
explizit, ob die Rangfolge belastbar ist oder im Rauschen liegt.

> **Laufzeit skaliert linear.** 100 Punkte × 3 Seeds = 300 Trainings. Bei ~3
> min/Training sind das ~15 h statt ~5 h. Wer das nicht investieren will, macht
> zuerst den billigen Test: **einen einzelnen** Gitterpunkt mit mehreren Seeds
> laufen lassen und die Streuung mit den Unterschieden in der bestehenden Heatmap
> vergleichen.
>
> ```bash
> # ~15 Minuten: ist die Seed-Streuung so groß wie die Heatmap-Unterschiede,
> # dann ist die Heatmap Rauschen und ein feineres Gitter bringt nichts.
> python3 PINNmodulusTwo/benchmark_wphys_wbc.py \
>   --w-phys 0.05 --w-bc 0.1 --seeds 0 1 2 3 4 --epochs 60 --device cuda
> ```

### 7.4 Erwartete Outputs

```
PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv               -> alle 100 Punkte
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_heatmap.png       -> 2D-MAE-Heatmap
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_convergence.png   -> Loss-Kurven (Ecken + Best)
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt          -> beste Kombination + Tabelle
PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/*.pt            -> 100 Modelle (mehrere GB!)
```

### 7.5 Monitoring

```bash
# Live mitlesen (Ctrl-C beendet nur tail, nicht den Benchmark)
tail -f benchmark_extended.log

# Welcher Punkt läuft gerade?
grep "Training w_phys" benchmark_extended.log | tail -3
#   [23/100] Training w_phys=0.05, w_bc=0.1

# Restlaufzeit
grep "ETA" benchmark_extended.log | tail -3
#   Train time: 16.8 min | ETA: 1155.0 min

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

### 7.6 Auswertung nach Abschluss

```bash
cat PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
```

Erwartete Ausgabe (Ende der Datei):

```
Selection ran on OP07 (MAE_val); OP08 (MAE_test) was never used to choose anything.
BEST (by MAE_val): w_phys=0.1, w_bc=0.3
  -> val 8.220°C, test 8.641°C, in-time 6.432°C
  Report the test number. MAE_val is optimistic: it is the minimum over 100 grid points.
  NOTE: one seed per point - the ranking cannot be separated from init noise.
  Re-run with --seeds 0 1 2 to find out.
Total runtime: 27.85 hours (1671.2 min)
Checkpoints dir: /home/user/llmtraining/PINNmodulusTwo/artifacts/checkpoints_wphys_wbc
```

Die besten Werte sind immer Gitterpunkte aus 7.2.

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

### 7.7 Abbrechen und neu starten

```bash
kill $(cat benchmark.pid)
# falls das nicht greift:
pkill -f benchmark_wphys_wbc
ps -p $(cat benchmark.pid)   # "No such process" = gestoppt
```

**Es gibt keine Resume-Funktion** — ein Neustart beginnt wieder bei Punkt 1.

### 7.8 Weiter mit den besten Gewichten

```bash
# Werte aus benchmark_wphys_wbc_best.txt einsetzen
python3 PINNmodulusTwo/train.py \
    --epochs 100 --w-phys 0.1 --w-bc 0.3 --subsample 2 \
    --ops OP01 OP02 OP03 OP04 OP05 OP06 --test-op OP08 --device cuda
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

### 7.9 Komplette Session zum Kopieren

```bash
# ---------------------------------------------------------------------------
# TEIL 1: SMOKE TEST
# ---------------------------------------------------------------------------
cd ~/llmtraining
source .venv/bin/activate

python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cuda
python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
cat PINNmodulusTwo/artifacts/smallBench_results.txt
# Erwartung: "✓ ALL CHECKS PASSED - Ready for full benchmark!"

# ---------------------------------------------------------------------------
# TEIL 2: FULL BENCHMARK  -  nur bei PASSED starten!
# ---------------------------------------------------------------------------
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --extended-grid --device cuda \
  > benchmark_extended.log 2>&1 &
echo $! > benchmark.pid

echo "PID:  $(cat benchmark.pid)"
echo "Log:  benchmark_extended.log"
echo "Live: tail -f benchmark_extended.log"
echo "Stop: kill \$(cat benchmark.pid)"
# Terminal kann jetzt geschlossen werden.

tail -f benchmark_extended.log   # optional, Ctrl-C beendet nur das tail
```

### 7.10 Checkliste

Vor dem Start:

- [ ] venv aktiviert (`source .venv/bin/activate`)
- [ ] Smoke-Test gelaufen und PASSED
- [ ] genug Plattenplatz (~10–20 GB für die Checkpoints, `df -h`)
- [ ] `nohup` oder `tmux` benutzt, damit der Lauf die SSH-Session überlebt

Während des Laufs:

- [ ] alle paar Stunden ins Log schauen (`tail benchmark_extended.log`)
- [ ] Prozess läuft noch (`ps -p $(cat benchmark.pid)`)
- [ ] GPU wird ausgelastet (`nvidia-smi`)

Danach:

- [ ] beste Kombination aus `benchmark_wphys_wbc_best.txt` notieren
- [ ] Heatmap und Convergence-Plot ansehen
- [ ] Production-Training mit den optimalen Gewichten starten

---

## 8. Architektur-Benchmark (Breite, Tiefe, Lags)

Der Gewichte-Sweep aus Schritt 7 kostet Stunden für zwei Zahlen, während Breite,
Tiefe und die History-Lags auf ungemessenen Werten festliegen. Dieser Benchmark
misst genau die.

Er läuft **eine Achse nach der anderen** gegen eine gemeinsame Baseline, statt
ein volles Produktgitter aufzuspannen: die Frage ist, welcher Regler den Fehler
überhaupt bewegt, nicht deren gemeinsames Optimum. `width x depth x lags` wären
mehrere hundert Trainings, achsenweise sind es zwölf.

```bash
cd ~/llmtraining
source .venv/bin/activate

# alle drei Achsen, drei Seeds
nohup python3 PINNmodulusTwo/benchmark_arch.py --device cuda --seeds 0 1 2 \
  > benchmark_arch.log 2>&1 &

# nur die Lags, bei den Gewichten aus Schritt 7
python3 PINNmodulusTwo/benchmark_arch.py --axes lags --w-phys 0.05 --w-bc 0.1 \
  --seeds 0 1 2 --device cuda
```

| Achse | Werte | Konfigurationen |
|---|---|---|
| `width` | 64, 128, 256 | 3 |
| `depth` | 2, 3, 4, 6 | 4 |
| `lags` | 5+20, 2+10, 10+60, 5+20+60, 30 | 5 |
| | **gesamt** | **12** |

Die Baseline (`--width 128 --depth 4 --rate-lags 5 20`) taucht in jeder Achse
einmal auf. Das ist Absicht: jede Achse braucht ihren eigenen Bezugspunkt, und
dieselbe Konfiguration dreimal zu trainieren zeigt nebenbei, wie weit die Seeds
allein streuen.

### 8.1 Outputs

```
PINNmodulusTwo/artifacts/benchmark_arch.csv       -> eine Zeile pro Konfiguration
PINNmodulusTwo/artifacts/benchmark_arch.png       -> ein Panel je Achse, Fehlerbalken = Seed-Streuung
PINNmodulusTwo/artifacts/benchmark_arch_best.txt  -> Ranking, Baseline-Vergleich, Rausch-Verdikt
```

`benchmark_arch_best.txt` enthält den entscheidenden Abschnitt **"Span per
axis"**: die Spannweite der Validierungs-MAE über jede Achse. Liegt sie unter der
Seed-Streuung, steht dort *"this knob does not matter here"* — dann lohnt es
nicht, diesen Regler weiter zu drehen.

### 8.2 Laufzeit

Referenz: ~2,9 min pro Training (60 Epochen, 6 Trainings-OPs, `--subsample 2`,
RTX 5090 Laptop) aus dem Log von Schritt 7.

| Umfang | Trainings | Laufzeit |
|---|---|---|
| alle Achsen, 1 Seed | 12 | ~35 min |
| alle Achsen, 3 Seeds | 36 | **~1,8 h** |
| nur `lags`, 3 Seeds | 15 | ~45 min |
| nur `width`, 3 Seeds | 9 | ~26 min |

> Die Werte gelten für `width=128`. Breitere Netze kosten mehr pro Schritt, aber
> der Rollout ist latenzgebunden (~7000 sequentielle Zeitschritte), nicht
> rechengebunden — `width=256` skaliert daher deutlich schwächer als die
> vierfache FLOP-Zahl vermuten lässt. Die echte Zahl steht nach zwei
> Konfigurationen in der `ETA`-Zeile des Logs.

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
rsync -avz user@gpu-server:~/llmtraining/benchmark_extended.log ./
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
| Läuft auf der GPU kaum schneller | Erwartbar bei kleinen Batches: pro Epoche gibt es viele *sequentielle* Rollout-Schritte, die sich nicht parallelisieren lassen. Batchgrößen erhöhen (Schritt 7). |

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
