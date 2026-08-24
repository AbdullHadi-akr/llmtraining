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

```bash
cd PINNmodulusTwo
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

---

## 7. Längere Läufe

Ein Training überlebt keine getrennte SSH-Verbindung — also `tmux` benutzen:

```bash
tmux new -s pinn
source ~/llmtraining/.venv/bin/activate
cd ~/llmtraining/PINNmodulusTwo
mkdir -p artifacts
python3 train.py --epochs 200 --subsample 40 2>&1 | tee artifacts/train.log
# Detach: Ctrl-b, dann d      Reattach: tmux attach -t pinn
```

Monitoring:

```bash
watch -n 2 nvidia-smi
tail -f ~/llmtraining/PINNmodulusTwo/artifacts/train.log
```

**Mehrere GPUs.** Der Code nutzt genau eine GPU pro Prozess (kein DDP). Für
parallele Hyperparameter-Läufe je einen Prozess pro Karte starten:

```bash
python3 train.py --device cuda:0 --w-phys 0.1 &
python3 train.py --device cuda:1 --w-phys 0.5 &
# alternativ:  CUDA_VISIBLE_DEVICES=1 python3 train.py --device cuda
```

Achtung: alle Läufe schreiben nach `PINNmodulusTwo/artifacts/` und
überschreiben sich gegenseitig — Artefakte pro Lauf wegsichern.

**Batchgrößen.** Der Geschwindigkeitsgewinn kommt hier weniger aus dem Netz
selbst (kleines MLP) als daraus, dass auf der GPU deutlich größere
Physik-Batches bezahlbar sind. Nach dem Smoke-Test lohnt sich:

```bash
python3 train.py --batch-phys 2048 --batch-bc 1024 --epochs 200
```

und dann per `nvidia-smi` schauen, wie viel Speicher noch frei ist.

**TF32** (`--tf32`) ist per Default aus. Es beschleunigt Matmuls auf Ampere+,
verschlechtert aber die zweiten Ableitungen im Physik-Residual. Nur einschalten,
wenn ein Vergleichslauf zeigt, dass die MAE-Werte gleich bleiben.
AMP/fp16/bf16 wird aus demselben Grund gar nicht angeboten.

---

## 8. Ergebnisse zurückholen

```bash
rsync -avz user@gpu-server:~/llmtraining/PINNmodulusTwo/artifacts/ ./artifacts/
```

Enthält `metrics.txt`, `training_curves.png`, `timeseries.png` und
`pred_OP*.npz`. Die Plots werden mit dem `Agg`-Backend erzeugt, es wird also
kein X-Server auf dem Server gebraucht.

---

## 9. Troubleshooting

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
| Läuft auf der GPU kaum schneller | Erwartbar bei kleinen Batches: pro Epoche gibt es viele *sequentielle* Rollout-Schritte, die sich nicht parallelisieren lassen. Batchgrößen erhöhen (Schritt 7). |

---

## Was der GPU-Umbau am Code geändert hat

- `device_utils.py` (neu): `resolve_device()`, `seed_everything()`, `enable_tf32()`.
- `train.py`, `smallBench.py`, `benchmark_wphys.py`, `benchmark_wphys_wbc.py`:
  `--device` steht jetzt auf `auto` statt `cpu`; ein explizites `cuda` schlägt
  hart fehl, wenn keine GPU da ist, statt still auf die CPU zu wechseln.
- `train.py`: zusätzliches `--tf32`-Flag, `torch.cuda.manual_seed_all()` beim Seeding.
- `config.yaml` (neu): die Datei, die `train.py` schon immer gesucht hat — mit
  `device`/`tf32` und allen bisherigen Defaults.
- `requirements-gpu.txt` (neu): schlanke, UTF-8-kodierte Abhängigkeitsliste.
