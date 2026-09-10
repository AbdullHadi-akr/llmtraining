# PINNmodulusTwo auf einem NVIDIA-GPU-Server

> **Update 31.08.2026.** Die Kapitel 7-10 (Benchmark-Sweeps, ~800 Zeilen) sind
> entfernt: die acht Skripte, die sie aufriefen, sind gelöscht und werden Schritt
> für Schritt neu aufgebaut. Kapitel 1-6 sind unverändert gültig und das, was ein
> frischer Server braucht. Der Einstieg ist [`FAHRPLAN.md`](FAHRPLAN.md).
>
> Ebenfalls neu: es gibt nur noch **ein** Projekt. `PINNmodulusTwoExtProfiles/`
> ist in `PINNmodulusTwo/` aufgegangen, trainiert wird auf dem ganzen Plansheet
> OP01–OP16 statt auf OP01–OP05.

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

> **Bevor du einen langen Sweep startest:
> [FAHRPLAN.md](FAHRPLAN.md).**
> Diese Datei hier sagt, *wie* man die Läufe startet. Die Kritik sagt, *worauf man
> in den Ergebnissen schaut* und **welcher Schritt danach überhaupt sinnvoll
> ist** — inklusive der Fälle, in denen ein Gewichte-Sweep die falsche nächste
> Maßnahme wäre. Sie hält außerdem fest, was am Modell repariert wurde und was
> davon bisher nur mathematisch verifiziert und noch nicht auf echten Daten
> gemessen ist.

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
git checkout main

sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> **Das Repo liegt schon auf dem Server?** Dann zuerst den Stand holen — die
> Flags in den Befehlen dieser Anleitung gibt es nur auf einem aktuellen `main`:
>
> ```bash
> cd ~/llmtraining && git checkout main && git pull origin main
> ```
>
> Bricht ein Aufruf später mit `error: unrecognized arguments: ...` ab, ist genau
> das die Ursache — der Code auf dem Server ist älter als diese Anleitung.
**Immer `main`, nie ein `claude/...`-Branch.** Die Feature-Branches sind
Momentaufnahmen und bleiben stehen, wo sie gemergt wurden — der Stand aus dem
GPU-Setup kennt die Flags aus Kapitel 6 und 7 noch nicht. **Schon geklont?**
Dann zuerst nachziehen:

```bash
cd ~/llmtraining
git checkout main
git pull origin main
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
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Erwartet z. B. `2.5.1+cu121 12.1 True NVIDIA A100-SXM4-40GB`.
Steht dort `False` → siehe Troubleshooting unten.

> Es gibt nur noch `PINNmodulusTwo/requirements-gpu.txt`. Die alte
> `requirements.txt` im Repo-Root ist gelöscht: ein UTF-16-kodierter
> Windows-`pip freeze` (inklusive `pywinpty`), der die CPU-Wheels installierte.
> Wer sie noch braucht, holt sie aus der Git-Historie.

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
docker run --gpus all -it --rm -v "$PWD":/workspace -w /workspace nvcr.io/nvidia/modulus/modulus:24.09 bash
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
rsync -avz --progress ~/batterysurrogatemodell/PINNmodulusTwo/data_cache/ user@gpu-server:~/llmtraining/PINNmodulusTwo/data_cache/

rsync -avz --progress ~/batterysurrogatemodell/PINNmodulusTwo/material_properties/ user@gpu-server:~/llmtraining/PINNmodulusTwo/material_properties/
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

## 6. Smoke-Test und Vorprüfung (~20 min)

Drei kurze Läufe, bevor irgendetwas Langes startet: rechnet die GPU, konvergiert
das Training, und wie lange dauert eine Epoche wirklich.

**Alles am Stück zum Kopieren:**

```bash
cd ~/llmtraining
git checkout main && git pull origin main   # 6.3 braucht den aktuellen Stand
source .venv/bin/activate

# 6.1  rechnet die GPU?  (< 1 min)
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cuda

# 6.2  konvergiert das Training?  (~5 min)
python3 PINNmodulusTwo/selftest.py          # Skalierungs-Checks, wenige Sekunden
python3 -m pytest PINNmodulusTwo/tests -q   # Rollout, History-Fastpath, Buchhaltung
python3 PINNmodulusTwo/train.py --epochs 5 --subsample 40 --device cuda

# 6.3  wie lange dauert eine Epoche?  (~10 min)
# Volle Konfiguration aus config.yaml, nur wenige Epochen. Die Zahl, die zaehlt,
# steht am Ende jeder Epochenzeile: "[Xs/epoch = Y rollout + Z inner]".
python3 PINNmodulusTwo/train.py --epochs 5 --device cuda
```

**Weiter nur, wenn:**

- 6.1 zeigt `[device] cuda:0 …` und der Prozess taucht in `nvidia-smi` auf
- 6.2 laeuft ohne `[ABORT]`, `[SATURATED]`, `[FLAT]` oder `[DIVERGED]` durch
- 6.3 hat eine Zahl für `s/epoch` geliefert und das Budget ist nachgerechnet

**Hyperparameter dieses Kapitels** — bewusst *nicht* die des Benchmarks: 6.1
läuft absichtlich grob und schnell, 6.2 kurz, und erst 6.3 nutzt die echten
Einstellungen aus `config.yaml`.

| | 6.1 | 6.2 | 6.3 |
|---|---|---|---|
| `--subsample` (→ `dt`) | `40` (4 s) | `2` (0.2 s) | `2` (0.2 s) |
| `--epochs` | `2` | `5` | `5` |
| `--ops` | alle | `OP01 OP02` | `OP01`–`OP05` |
| `--w-phys` / `--w-bc` | `0.1` / `0.1` | `0.0 0.1` / `0.1` | `0.1` / `0.1` |
| `--history-mode` | `hybrid` | `hybrid` | `hybrid` |
| `--delta-grid` | `0.2` s | `0.2` s | `0.2` s |
| `--rate-lags` | `5 20` | `5 20` | `5 20` |

---

### 6.1 Rechnet die GPU? (< 1 min)

Bewusst `--device cuda` statt `auto`: so bricht ein falsch aufgesetzter Server
mit einer klaren Fehlermeldung ab, statt still auf der CPU zu rechnen.

```
[device] cuda:0 NVIDIA A100-SXM4-40GB  39.6 GiB  sm_80  torch=2.5.1+cu121 cuda=12.1
[CFL WARN] Δt=4.000s, Δt_max≈0.241s -> POTENTIALLY UNSTABLE
  [WARN] --delta-grid 0.2s is below the data step 4s; ...
  epoch   1  L_data=...  [12.4s/epoch, this run ~0 min left]
```

**Die beiden Warnungen gehören hier dazu.** Beide kommen aus `--subsample 40`,
das absichtlich grob ist, damit der Check unter einer Minute bleibt: Δt = 4 s
liegt weit über der CFL-Grenze von ~0.241 s, und ein Anker 0.2 s zurück lässt
sich auf einem 4-s-Raster nicht auflösen. Ab 6.2 läuft alles mit `--subsample 2`,
dann sind Δt und Δgrid beide 0.2 s und beide Warnungen verschwinden.

Parallel in einer zweiten Session `nvidia-smi` — der Python-Prozess muss dort mit
belegtem Speicher auftauchen.

<details>
<summary>Was Δt und Δgrid unterscheidet (die zwei Warnungen im Detail)</summary>

Beides sind Zeiten in Sekunden, aber sie zeigen in verschiedene Richtungen:

- **Δt** ist die Schrittweite des Rollouts — wie weit das Modell pro Schritt nach
  **vorne** geht. `subsample 40` × 0.1 s Rohtakt = 4 s. Darauf bezieht sich die
  CFL-Prüfung.
- **`delta_grid`** ist der Anker der History — wie weit `T(t−Δgrid)` nach
  **hinten** nachgeschlagen wird, und von dort laufen die Ratensegmente
  (`--rate-lags 5 20`) weiter rückwärts.

Auf einem 4-s-Raster gibt es keinen Punkt 0.2 s zurück; der Nachschlag klemmt auf
die letzte vorhandene Stufe und wirkt effektiv wie 4 s. Kein Widerspruch, sondern
eine Folge des groben Rasters — bei `--subsample 2` fallen Δt und Δgrid auf
0.2 s zusammen und passen exakt.

</details>

---

### 6.2 Konvergiert das Training? (~5 min)

Trainiert zwei Modelle — `w_phys=0.0` (data-only) und `w_phys=0.1` (mit Physik) —
über 5 Epochen auf OP01/OP02 und testet auf OP07.

Outputs:

```
PINNmodulusTwo/artifacts/metrics.txt                 -> alle OPs, alle Metriken
PINNmodulusTwo/artifacts/training_curves.png         -> Loss-Kurven
PINNmodulusTwo/artifacts/model.pt                    -> Gewichte + Konfiguration
```

Am Ende des Berichts steht `✓ ALL CHECKS PASSED` oder `✗ SOME CHECKS FAILED`.

Bei **FAILED** nicht weitermachen, sondern die Ursache klären:

| Befund | Ursache |
|---|---|
| Loss explodiert (inf/NaN) | CFL-Problem — `--subsample` verkleinern |
| Test-MAE > 20 °C | das Modell lernt nicht richtig |
| noch nicht konvergiert | mit mehr Epochen gegenprüfen |

Bei `[ABORT] ... loss exploded` nennt die Meldung den zuerst betroffenen
Loss-Term, siehe [Kapitel 10](#11-troubleshooting).

---

### 6.3 Wie lange dauert eine Epoche? (~10 min)

Ein kurzer Lauf mit den **echten** Einstellungen. Alle Zeitangaben in Kapitel 8
**Zuerst den Stand prüfen.** Dies ist der erste Aufruf, der `--delta-grid`
explizit setzt — das Flag gibt es erst seit Commit `beaf673`. Auf einem älteren
Arbeitsbaum bricht der Lauf sofort ab, noch bevor Torch überhaupt startet:

```
train.py: error: unrecognized arguments: --delta-grid 0.2
```

Das ist kein Tippfehler im Aufruf, sondern ein veralteter Checkout — typisch,
wenn Kapitel 2 auf einem `claude/...`-Branch stehen geblieben ist. Zwei weitere
Merkmale desselben Stands: `--gain-lr-mult` fehlt, und das längst entfernte
`--delta-init-steps` ist noch da. Prüfen und nachziehen:

```bash
python3 PINNmodulusTwo/train.py --help | grep -e --delta-grid   # muss etwas ausgeben
git checkout main && git pull origin main                       # falls nicht
```

Ein kurzer Lauf mit den **echten** Einstellungen. Alle Zeitangaben in Kapitel 7
und 8 hängen an dieser einen Zahl. Die Epochenzeile schlüsselt sie auf:

```
  epoch 1  L_data=...  [118.7s/epoch = 112.4s rollout + 6.3s x100 inner, ...]
```

> **Diese Beispielzahl ist veraltet. Am 10.09. auf der T4 gemessen:**
>
> ```
>   epoch 1  L_data=...  [21.1s/epoch = 14.8s rollout + 6.3s x100 inner, ...]
> ```
>
> **Der `inner`-Teil ist in beiden Zeilen identisch 6.3 s** — es ist also
> dieselbe Konfiguration (2 OPs, 100 `inner_steps`), und der Vergleich ist
> zulässig. Der Rollout ist von **112.4 s auf 14.8 s** gefallen, Faktor **7.6**.
> Das ist der `rollout_plan`-Fastpath: `level_rollout` reduzierte davor je
> Schritt den ganzen Puffer-Präfix, also O(n_t²·P), und „dominierte die Epoche"
> (so der Docstring in `model.py`).
>
> **Damit sind alle Budgettabellen in Kapitel 7 und 8 um Faktor 5.6 zu
> pessimistisch** (118.7 / 21.1). Die Rechenvorschrift darunter stimmt weiter —
> nur `S` muss aus einem heutigen Lauf kommen.
>
> **Die Planungszahl für die echte Konfiguration:** 21.1 s bei 2 OPs sind
> 10.55 s je OP und Epoche, also ~116 s/Epoche bei elf OPs und **~1.9 h für
> einen 60-Epochen-Lauf.**

Zwei Hälften, die sich völlig verschieden verhalten:

- **rollout** — ~7000 sequentielle Schritte je OP. Latenzgebunden, hängt nur an
  `--subsample` und der Zahl der OPs. Das ist der Löwenanteil.
- **inner** — die `--inner-steps` Minibatch-Updates gegen diesen Rollout. Skaliert
  linear mit `--inner-steps` und ist der einzige Regler, der die Epoche teurer
  macht, ohne dass Punkte oder Epochen wachsen.

Ist der Innenteil unerwartet groß (sagen wir über einem Drittel), senke
`--inner-steps` auf 50 — das kostet die Hälfte der Updates, nicht die Hälfte des
Nutzens. `--subsample` **nicht** erhöhen: 0.2 s liegt schon knapp unter der
CFL-Grenze von ~0.241 s.

**Daraus das Budget rechnen.** Mit `S` = Sekunden pro Epoche und `P` = Punkte im
Block:

```
Stunden pro Punkt  = S × Epochen ÷ 3600
Stunden pro Block  = Stunden pro Punkt × P
Für 5 h:  Epochen_max = 18000 ÷ (P × S)
```

Der bindende Block ist 7.1 mit **5 Punkten**. Was dort in 5 h passt:

| gemessene s/Epoche | `--epochs` für 5 Punkte in ≤ 5 h | Block 7.1 | Block 7.2 (4 Punkte) |
|---|---|---|---|
| 60 | 60 | 5,0 h | 4,0 h |
| 120 | 30 | 5,0 h | 4,0 h |
| 180 | 20 | 5,0 h | 4,0 h |
| 240 | 15 | 5,0 h | 4,0 h |
| 360 | 10 | 5,0 h | 4,0 h |
| > 500 | < 7 — stattdessen **den Arm halbieren**, siehe 7.1 | | |

`--epochs` unter ~10 zu drücken lohnt nicht: dann misst der Sweep hauptsächlich,
welche Konfiguration schneller startet, nicht welche besser wird. Ab dort ist
`--probe-part 1a` / `1b` der richtige Hebel — gleiche Epochen, weniger Punkte
pro Sitzung.

Der Benchmark rechnet das ab dem ersten fertigen Punkt selbst mit und meldet
sich, wenn der Block über `--max-hours` (Default 5) läuft:

```
  Train time: 62.1 min | ETA: 248.4 min | block total ~5.17 h
  [BUDGET] this block projects to 5.17 h, over the 5.0 h limit (--max-hours).
           --epochs 19 would fit (20 is running now); or split the arm and run
           fewer points per session.
           Ctrl-C now costs one point. Finishing costs 0.17 h over.
```

Das stoppt nichts — ein Abbruch mitten im Arm hinterließe einen Part, den der
Report-Schritt zu Recht ablehnt. Es sagt dir nach dem ersten Punkt, was der
ganze Block kostet, damit ein Neustart einen Punkt kostet und nicht einen Abend.

Für Kapitel 9 dieselbe Rechnung mit 60 Epochen:

```
Sekunden/Epoche × 60 Epochen ÷ 3600  = Stunden pro Gitterpunkt
Stunden/Punkt × Punkte × Seeds ÷ 24  = Tage für den Sweep
```

<details>
<summary>Wie viel bringt die GPU hier eigentlich? (optional, weitere 10 min)</summary>

Derselbe Lauf mit `--device cpu` beantwortet das:

```bash
python3 PINNmodulusTwo/train.py --ops OP01 OP02 --subsample 2 --epochs 3 --history-mode hybrid --rate-lags 5 20 --grad-clip 1.0 --device cpu
```

Die Sekunden pro Epoche aus beiden Läufen ins Verhältnis setzen — **erwarte
keinen großen Faktor.** Der Grund steckt in der Struktur der Last:

| Anteil | parallelisierbar? | GPU-Gewinn |
|---|---|---|
| Rollout: ~7000 sequentielle Schritte je OP und Epoche | nein, jeder Schritt braucht den vorherigen | gering |
| Python-Schleife und Kernel-Starts, ~37.000 Iterationen je Epoche | nein, identisch auf beiden | keiner |
| MLP-Matmuls je Schritt (363 Sensoren × 128 breit) | ja, aber winzig | klein |
| Physik-Residuum (`--batch-phys`, doppeltes Autograd) | ja, echte Parallelarbeit | groß |

Die GPU gewinnt dort, wo tatsächlich große Batches anfallen — beim
Physik-Residuum. Der Rollout, der die Laufzeit dominiert, ist latenzgebunden und
läuft auf beiden ähnlich schnell. Deshalb steht in der Troubleshooting-Tabelle
auch „läuft auf der GPU kaum schneller" als *erwartbar*, nicht als Fehler.

**Praktische Konsequenz:** wenn `nvidia-smi` viel freien Speicher zeigt, ist
`--batch-phys 2048 --batch-bc 1024` der Hebel, der die GPU wirklich nutzt — nicht
ein breiteres Netz. Wenn du das machst, dann **in 7.1 und 7.2 gleich**: die
beiden Schritte müssen in jedem Hyperparameter übereinstimmen.

</details>

---

### 6.4 Mehrere Läufe gleichzeitig — der Weg, der das Experiment nicht anfasst

Ein einzelner Lauf lastet die Karte nicht aus und kann es nicht: ~7000
**sequentielle** Rollout-Schritte je OP, jeder ein 363×128-Matmul. Das sind ~50
Kernel zu ~5 µs Startlatenz gegen ~1.5 µs Rechenzeit — die GPU wartet auf Python.
Kein Regler ändert daran etwas, denn die Schritte hängen voneinander ab.

Was die Karte füllt, ist **mehr unabhängige Arbeit gleichzeitig**, und ein Sweep
besteht aus nichts anderem — die Punkte und die Seeds sind getrennte Experimente,
die sich gerade *nicht* beeinflussen dürfen:

```bash
nvidia-cuda-mps-control -d          # einmal je Boot, sonst time-sliced der Treiber

python3 PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary delta-phys 1.0 0.4 0.2 -j 4 -- --epochs 20
```

Jeder Punkt ist ein eigener `train.py`-Prozess mit eigenem `--artifacts-dir`.
Prozesse teilen weder RNG noch Allokator noch Optimiererzustand — genau der
Grund, warum es Prozesse sind und nicht eine Schleife über `train.fit()`.

> **Eine Einschränkung, gemessen statt vermutet.** `sweep.py` setzt seinen
> Kindern `OMP_NUM_THREADS=1`, und **die Thread-Zahl verschiebt die letzten
> Stellen**: mehrfädige CPU-Reduktionen legen ihre Summationsreihenfolge nicht
> fest. Auf der synthetischen Fixture geprüft — mit `OMP_NUM_THREADS=1` auf
> beiden Seiten stimmten alle 25 Gewichtstensoren und alle Vorhersage-Arrays
> bitweise überein; gegen einen Lauf mit der Voreinstellung wich es ab der
> 8. Stelle ab. **Innerhalb eines Sweeps ist das folgenlos** — alle Punkte
> bekommen dieselbe Umgebung, und nur ihr Vergleich untereinander zählt. Wer
> gegen eine von Hand erzeugte Zahl vergleichen will, setzt `--threads` passend.

Die Grenze ist die **CPU, nicht der Speicher**: jeder Lauf ist eine
Python-Schleife, die Millionen Kernel-Starts absetzt, und will einen Kern für
sich. Auf einer g4dn.2xlarge (8 vCPU, 4 physisch) sind 4–6 gleichzeitige Läufe
der nutzbare Bereich; das landet bei ~4–6 GB VRAM. `sweep.py` setzt
`OMP_NUM_THREADS=1` in den Kindprozessen, sonst öffnet jedes Kind einen
Thread-Pool in Maschinenbreite und die Läufe verdrängen sich gegenseitig.

`nvidia-smi` wird danach fast ausgelastet aussehen. Das täuscht: es zeigt, dass
*irgendein* Kernel läuft, nicht dass die 40 SMs voll sind. Ein 363-Zeilen-Matmul
belegt eine Handvoll Thread-Blocks, egal wie viele Prozesse gleichzeitig eins
schicken. Der Gewinn kommt daher, dass sich die CPU-Arbeit der einen Läufe mit
der GPU-Arbeit der anderen überlappt — Durchsatz, nicht Rechenleistung.

#### Was MPS überhaupt ist

**Multi-Process Service, von NVIDIA, Teil des CUDA-Treibers.** Es hat nichts mit
AWS, mit dieser Instanz oder mit diesem Projekt zu tun — auf einer Workstation im
Keller ist es dasselbe. Seit CUDA 9 dabei, auf der T4 voll unterstützt.

**Das Problem:** jeder Prozess bekommt normalerweise seinen eigenen
CUDA-Kontext, und die Karte kann immer nur **einen** Kontext gleichzeitig
bedienen. Der Treiber schaltet also im Zeitscheibenverfahren um — A, dann B,
dann C, dann D. Vier Prozesse teilen sich die Karte **nacheinander**, nicht
nebeneinander.

Bei großen Kerneln fällt das kaum auf: ein Kernel, der 10 ms rechnet, überdeckt
den Umschaltvorgang. **Bei winzigen Kerneln ist es fatal** — und genau das ist
der Rollout hier: ~50 Kernel je Schritt auf einer 363×128-Matrix. Da wird
überwiegend umgeschaltet und kaum gerechnet. Deshalb wurde unter `-j 4` der
Rollout 3.35× langsamer und der Innenteil mit seinen größeren Batches nur 1.25×.

**Was MPS macht:** ein Server-Prozess hält **einen** Kontext, alle Clients
schicken ihre Kernel durch ihn hindurch. Die Karte sieht keine vier
konkurrierenden Kontexte mehr, sondern einen — und Kernel aus verschiedenen
Prozessen laufen **gleichzeitig auf verschiedenen SMs**.

Drei Dinge, die man dazu wissen sollte:

* Es ist ein **Hintergrunddienst**, kein Teil des Programms. Der Code merkt
  nichts davon, CUDA-Anwendungen verbinden sich automatisch.
* Er **stirbt beim Reboot.** Nach jedem Neustart der Instanz einmal starten.
* **Kein Speicherschutz zwischen den Clients.** Stürzt ein Prozess hart ab, kann
  er die anderen mitreißen. Für einen Sweep, in dem jeder Punkt ohnehin einzeln
  protokolliert wird und ein `[FAIL]` den Rest nicht aufhält, ist das ein
  akzeptabler Tausch.

#### Der Arbeitsablauf für einen Sweep, komplett

```bash
# 1. Stand und Umgebung
cd /home/student1/llmtraining
git checkout main && git pull
source modulus_env/bin/activate        # python, NICHT python3

# 2. MPS -- nach jedem Neustart der Instanz einmal
nvidia-cuda-mps-control -d
#    "An instance of this daemon is already running" = laeuft schon, alles gut.
#    Die Meldung ueber /var/log/nvidia-mps ist folgenlos (nur Logs).

# 3. Sweep starten, abgekoppelt von der SSH-Sitzung
nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary <flag> <wert> <wert> -j <teiler der laufzahl> \
    --out artifacts/<name> --csv artifacts/<name>.csv \
    -- --epochs 60 > <name>.log 2>&1 &

# 4. Auswerten
tail -30 <name>.log                    # Mittel und Std je Konfiguration
cat artifacts/<name>.csv               # eine Zeile je (Konfiguration, Seed)
python PINNmodulusTwo/tools/analyse_history.py \
    artifacts/<name>/<punkt>/history.csv
```

Vier Dinge, die dabei schiefgehen können, und was davor schützt:

| | |
|---|---|
| `python3` statt `python` | `sweep.py` prüft den Interpreter vorab und bricht mit **einer** Zeile ab, statt jeden Punkt an `import torch` sterben zu lassen |
| MPS vergessen | `sweep.py` warnt bei jedem parallelen CUDA-Sweep und nennt die gemessenen Zahlen. **Kein Abbruch** — ein fehlender Daemon ist kein Fehler |
| krummes `-j` | die Wanduhr ist `ceil(Läufe / j)` Lauf-**Dauern**; bei 9 Läufen ist `-j 4` exakt so schnell wie `-j 3`. `sweep.py` rechnet es aus und weist darauf hin |
| Artefakte überschreiben sich | jeder Punkt bekommt `--artifacts-dir` **und** `PINN_ART_DIR`. Ohne das schrieben alle Läufe dieselbe `model.pt` |

Und die Regel, die nichts mit der GPU zu tun hat: **nie die letzte Zeile eines
Laufs ablesen.** `analyse_history.py` gibt Median und Streuung über die letzten
Epochen — genau aus dem Ablesen der letzten Zeile entstand O12.

#### Wie viel das spart — **3.69×, und MPS ist die halbe Miete**

Am 10.09. auf der Instanz, 4 Läufe (`--ops OP01 OP02 --epochs 6 --device cuda`):

| | `sweep_wall_s` | je Lauf | Faktor |
|---|---|---|---|
| `-j 1` (seriell) | 596.2 s (9.9 min) | 2.5 min | — |
| `-j 4` **ohne** MPS | 407.4 s (6.8 min) | **6.8 min** | 1.46× |
| `-j 4` **mit** MPS | **161.7 s (2.7 min)** | **2.7 min** | **3.69×** |

**92 % Effizienz.** Die Konkurrenz je Lauf fällt von 2.72× auf **1.08×** — vier
gleichzeitige Läufe kosten fast so viel wie einer. **MPS allein bringt 2.52×.**

> **`nvidia-cuda-mps-control -d` ist Pflicht, nicht Kür.** Einmal je Boot, vor
> jedem Sweep. Ohne den Daemon verschenkt man Faktor 2.5 — lautlos, nichts im Log
> sagt, dass er fehlt.

**Warum, stand vorher schon in der Epochenzeile.** Ohne MPS wurde der Rollout
3.35× langsamer (14.8 → 49.6 s), der Innenteil nur 1.25× (6.3 → 7.9 s). Der
Rollout sind ~50 *winzige* Kernel je Schritt (363×128), und winzige Kernel aus
vier CUDA-Kontexten arbeitet der Treiber zeitscheibenweise ab; der Innenteil
rechnet in größeren Batches, dort fällt ein Kontextwechsel kaum ins Gewicht.
**MPS behebt genau das** — Kernel verschiedener Prozesse laufen nebeneinander auf
den SMs statt abzuwechseln.

**Nicht die `ratio`-Zeile der Ausgabe als Faktor nehmen.** Sie rechnet `Summe der
Laufzeiten / Wanduhr`, und unter Konkurrenz wird jeder Lauf selbst langsamer —
das ist eine Obergrenze. Nur `-j 1` gegen `-j N` beantwortet es.

**Wie weit `-j` gehen darf, ist offen.** Bei vier Workern sind 92 % erreicht, die
vier physischen Kerne also nicht ausgereizt. Mit MPS ist nicht mehr die Karte die
Grenze, sondern die CPU — jeder Lauf ist eine Python-Schleife und will einen
Kern. `-j 6` auf 8 vCPU ist der nächste sinnvolle Versuch, gemessen ist er nicht.

Die Verhältnisse oben sind exakt; absolut hängen sie an der Lauf-Dauer, und die
ist auf der T4 nicht gemessen (Kapitel 6.3).

---

### 6.5 Batchgrößen — was 20 GB VRAM hergeben

Der erste Epochen-Log auf einer GPU nennt den gemessenen Spitzenverbrauch:

```
  peak VRAM 0.41 GB of 20.0 GB (batch_data=2048 batch_phys=256 batch_bc=128)
```

> **10.09. auf der g4dn.2xlarge nachgemessen: `0.11 GB von 15.6 GB`** (Tesla T4,
> torch 2.6.0+cu124, `--ops OP01 OP02 --epochs 3`, Default-Batches). Mit allen
> elf OPs wären es grob 0.4–0.5 GB — je OP kommen nur `Tn`, `Qsrc` und `T_lab`
> dazu (~35 MB), die Batchgrößen ändern sich nicht. **Der Absatz unten stimmt
> also nicht nur ungefähr, sondern mit Faktor 100 Luft.**

**Speicher ist hier nicht die Grenze.** Das Netz ist mit 128×4 winzig (~70k
Parameter), der Rollout-Buffer ist `7000 × 363 × 4 B` ≈ 10 MB, und selbst der
doppelte Autograd des Physik-Residuums hängt nur linear an `--batch-phys`. Bei
den Defaults liegt der Verbrauch im Bereich weniger hundert MB — von 20 GB.

Das heißt: die Defaults verschenken die Karte. Größere Batches kosten fast
keinen Speicher und, solange die Tensoren klein bleiben, auch kaum Zeit — die
Läufe sind startlatenz-gebunden, nicht rechengebunden. Ein größerer Batch füllt
denselben Kernel-Start besser aus und macht den Gradienten je Schritt leiser.

Vorschlag als Startpunkt, ausgehend von den 20 GB:

```bash
--batch-data 8192 --batch-phys 2048 --batch-bc 512
```

Danach **6.3 noch einmal fahren** und zwei Zahlen ablesen:

| Beobachtung | Konsequenz |
|---|---|
| `peak VRAM` weit unter 20 GB **und** der `inner`-Anteil kaum gewachsen | die größeren Batches waren gratis — behalten |
| `inner`-Anteil deutlich gewachsen | die GPU ist jetzt rechengebunden; entweder Batches wieder kleiner oder `--inner-steps` senken |
| `CUDA out of memory` | `--batch-phys` zuerst halbieren, das ist der teuerste Term |

Zwei Dinge, die **nicht** helfen: ein breiteres Netz (der Rollout dominiert, und
der wird davon langsamer, nicht besser) und `--subsample` erhöhen (die CFL-Grenze
liegt bei ~0.241 s, `dt = 0.2 s` ist schon nah dran).

> **Nachtrag 09.09. — größere Batches sind nicht gratis.** Der Absatz oben sagt
> „kostet fast keinen Speicher und kaum Zeit", und das stimmt. Er verschweigt,
> was es sonst kostet: `--batch-data` ist die Zahl der (t, Punkt)-Paare je
> Adam-Schritt, ein größerer Batch macht den Gradienten leiser, und **das ist
> eine andere Optimierung** — nicht dieselbe Rechnung, nur schneller. Innerhalb
> eines Sweeps ist das in Ordnung, solange **alle** Punkte dieselben Batchgrößen
> benutzen; gegen die Zahlen aus Schritt 6 vergleichbar bleibt es nicht.
>
> Wer die Karte auslasten will, ohne am Experiment zu drehen, nimmt **6.4**.

Was du hier wählst, muss über alle Läufe, die du miteinander vergleichst,
identisch sein — sonst mischt der Vergleich zwei Experimente.

---

---

## 7. Danach

Was hier stand -- die Range-Probe der Loss-Gewichte, der Architektur-Sweep
und das 10x10-Gitter, zusammen rund 800 Zeilen -- rief die vier
Benchmark-Skripte auf, die am 31.08.2026 geloescht wurden. Es ist entfernt
statt umgeschrieben: der Neuaufbau soll nicht die alte Reihenfolge erben.

Was als Naechstes zu tun ist, steht in [`FAHRPLAN.md`](FAHRPLAN.md).
Kurz: ein `train.py`-Lauf beantwortet inzwischen selbst, ob das Modell die
trivialen Vorhersager schlaegt, und bis das feststeht misst ein Sweep
nichts. Alles bis Kapitel 6 hier bleibt gueltig -- Treiber, CUDA-PyTorch,
Modulus, Datenuebertragung und der Vorlauf sind unveraendert das, was ein
frischer Server braucht.
