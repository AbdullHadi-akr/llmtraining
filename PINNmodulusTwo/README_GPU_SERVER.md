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
source .venv/bin/activate

# 6.1  rechnet die GPU?  (< 1 min)
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cuda

# 6.2  konvergiert das Training?  (~5 min)
python3 PINNmodulusTwo/selftest.py          # Skalierungs-Checks, wenige Sekunden
python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
cat PINNmodulusTwo/artifacts/smallBench_results.txt

# 6.3  wie lange dauert eine Epoche?  (~10 min)
python3 PINNmodulusTwo/train.py --ops OP01 OP02 OP03 OP04 OP05 --subsample 2 --epochs 5 --history-mode hybrid --rate-lags 5 20 --delta-grid 0.2 --grad-clip 1.0 --device cuda
```

**Weiter zu [Kapitel 7](#7-balancing-benchmark--was-die-gewichte-überhaupt-bedeuten-7-h-zwei-sessions) nur, wenn:**

- 6.1 zeigt `[device] cuda:0 …` und der Prozess taucht in `nvidia-smi` auf
- 6.2 endet mit `✓ ALL CHECKS PASSED - Ready for full benchmark!`
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
PINNmodulusTwo/artifacts/smallBench_results.txt      -> PASS/FAIL-Bericht
PINNmodulusTwo/artifacts/smallBench_convergence.png  -> Loss-Kurven
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
und 8 hängen an dieser einen Zahl:

```
  epoch   1  L_data=...  [112.4s/epoch, this run ~7 min left]
```

Damit rechnest du die Kapitel-7-Schritte durch:

```
Sekunden/Epoche × 20 Epochen ÷ 3600  = Stunden pro Punkt
Stunden/Punkt × 5                    = Schritt 8.1  (w_phys-Arm)
Stunden/Punkt × 4                    = Schritt 8.2  (w_bc-Arm)
```

Bei 112 s/Epoche sind das 0,62 h pro Punkt — 7.1 also ~3,1 h und 7.2 ~2,5 h.
Kommt deutlich mehr heraus, ist der Hebel `--epochs 10` (halbiert beide
Schritte). `--subsample` **nicht** erhöhen: 0.2 s liegt schon knapp unter der
CFL-Grenze von ~0.241 s.

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

## 7. Balancing-Benchmark — was die Gewichte überhaupt bedeuten (~7 h, zwei Sessions)

Was intern passiert — Kontrollfluss, Modell, wo die Skalierung sitzt —
steht in [ARCHITECTURE.md](ARCHITECTURE.md).

**Vor Kapitel 8.** Eine Gewichte-Probe misst nur dann Gewichte, wenn die
Skalierung darunter feststeht. Genau das klärt dieses Kapitel.

### 7.0 Das Problem in einem Satz

`L_phys` und `L_bc` werden durch ihren eigenen laufenden Mittelwert geteilt,
`L_data` bisher nicht. Die beiden normierten Terme liegen damit über den ganzen
Lauf bei ~1, während `L_data` um Größenordnungen fällt:

| | Start (`L_data ≈ 1`) | konvergiert (`L_data ≈ 5e-3`) |
|---|---|---|
| `w_phys = 0.001` | 0,1 % des Losses | ~20 % |
| `w_phys = 0.1` | 10 % | ~20× `L_data` |
| `w_phys = 1.0` | ~50 % | ~200× `L_data` |

Die Mischung, die der Optimierer sieht, wandert also während des Trainings zur
Physik. Zwei Folgen: das beste `w_phys` hängt an `--epochs` — ein in der
20-Epochen-Probe gefundener Wert ist für den 60-Epochen-Lauf zu groß — und die
bisherige Probe-Range liegt größtenteils im physikdominierten Bereich.

`--loss-balance ema` (neuer Default) teilt **alle drei** Terme durch ihren
eigenen Mittelwert. Dann ist `w_data:w_phys:w_bc` ein echtes Verhältnis, das in
Epoche 1 dasselbe bedeutet wie in Epoche 60.

### 7.1 Teil 1 — Balancing und Residuen-Skalierung (5 Trainings, ~3,5 h)

```bash
cd ~/llmtraining
source .venv/bin/activate

nohup python3 PINNmodulusTwo/benchmark_balance.py --part 1 --epochs 20 --device cuda > balance_part1.log 2>&1 &
echo $! > balance.pid
echo "Live: tail -f balance_part1.log"
```

Fünf Konfigurationen an **einem festen Gewichtspunkt** (`w_phys=0.1`,
`w_bc=0.1`): die drei `--loss-balance`-Modi und die zwei `--residual-norm`-Modi.
Gesweept wird die Skalierung, nicht das Gewicht.

Entscheidend ist die Spalte `drift` im Report:

```
    axis    value |  MAE_in  MAE_val    +/-  MAE_test |  ratio_1  ratio_N   drift
 balance      ema |    ...      ...    ...       ... |    0.100    0.098    0.98
 balance   legacy |    ...      ...    ...       ... |     12.4     198.    16.0
```

- `ratio` = `w_phys*L_phys_bal / (w_data*L_data_bal)`, also die tatsächliche
  Mischung. Unter `ema` startet sie per Konstruktion bei `w_phys/w_data`.
- `drift` = `ratio_N / ratio_1`. Nahe 1 heißt: ein in einer kurzen Probe
  gefundenes Gewicht überträgt sich auf den langen Lauf. Weit weg von 1 heißt:
  es überträgt sich nicht.

Zusätzlich entsteht `benchmark_balance_ratio.png` — die Mischung je Epoche, eine
Kurve pro Konfiguration. Flach = stabil.

### 7.2 Teil 2 — die optionalen Eingangskanäle (4 Trainings, ~2,8 h)

Erst laufen lassen, wenn 7.1 einen Modus gewählt hat, und diesen mitgeben:

```bash
nohup python3 PINNmodulusTwo/benchmark_balance.py --part 2 --epochs 20 --device cuda --loss-balance ema > balance_part2.log 2>&1 &
echo $! > balance.pid
```

Vier Varianten: `none`, `energy`, `rates`, `both`.

- **`energy`** hängt die kumulierte eingebrachte Wärme als zweiten
  Forcing-Kanal an, ausgedrückt als adiabate Temperaturerhöhung in Sigma. Ein
  thermisches System *integriert* Leistung; das momentane `q_dot`, das das Netz
  bisher bekommt, sagt nur, wie stark gerade geheizt wird — nie, wie viel Wärme
  schon in der Zelle steckt.
- **`rates`** hängt `d(config)/dt` für die Config-Kanäle an, die echte
  Zeitprofile sind. Dasselbe Argument wie bei der Temperatur-Rekurrenz, auf die
  Configs angewandt.

Beide sind per Default **aus**. Sie verbreitern den Netzeingang, also werden sie
gemessen und nicht angenommen.

> **Profile vs. Labels.** `train.py` meldet beim Start, welche Config-Kanäle
> sich innerhalb eines OP über die Zeit ändern (*Profile*) und welche je OP
> konstant sind (*Labels*). Label-Kanäle sind Konstanten, die das Netz pro OP
> auswendig lernen kann — bei fünf Trainings-OPs können ein paar davon als
> OP-Kennung wirken, die sich nicht auf OP06/OP07 überträgt. Steht im Log
> direkt unter der `OPs=`-Zeile.

### 7.3 Was danach feststeht

`benchmark_balance_best.txt` nennt den Gewinner und den Aufruf für Kapitel 8.
Ab hier gilt: **jede Gewichte-Probe muss mit denselben Balancing-Flags laufen.**
Sie stecken in der Signatur von `benchmark_wphys_wbc.py`; ein zweiter Arm mit
anderem `--loss-balance` verwirft den ersten, statt ihn stillschweigend
dazuzumischen.

**Weiter zu Kapitel 8, wenn:**

- 7.1 durchgelaufen ist und einen `--loss-balance`-Modus gewählt hat
- die `drift`-Spalte gelesen wurde — sie sagt, ob die Probe in Kapitel 8
  überhaupt auf den 60-Epochen-Lauf übertragbar ist
- 7.2 gesagt hat, ob die Zusatzkanäle etwas bringen (sonst bleiben sie aus)

---

## 8. Range-Probe der Loss-Gewichte — in drei Schritten

Dieselben neun Trainings wie bisher, mit denselben Parametern (`dt = 0.2 s`),
nur in **drei getrennte Schritte** zerlegt: zwei Trainingsblöcke und eine
Auswertung. Kein Block blockiert die Maschine länger als ~3,5 h, und jeder
Schritt speichert seine Ergebnisse sofort.

| Schritt | Was läuft | Trainings | Laufzeit | Was entsteht |
|---|---|---|---|---|
| **7.1** | `w_phys`-Arm | 5 | **~3,5 h** | CSV + Settings + Rohzeilen |
| **7.2** | `w_bc`-Arm | 4 | **~2,5 h** | CSV (jetzt 9 Zeilen) + Settings |
| **7.3** | Auswertung, kein Training | 0 | **~1 min, ohne GPU** | Verdikt + Plots |

**Heute reichen [Kapitel 6](#6-smoke-test-und-vorprüfung-20-min) und 8.1.** 8.2
und 8.3 können beliebig später laufen — Schritt 8.1 legt seine Ergebnisse in
`artifacts/probe_parts.json` und `artifacts/benchmark_wphys_wbc.csv` ab und
wartet dort.

> **Warum 5 + 4 und nicht 6 + 3?** Weil die Probe ein **Kreuz** ist, kein
> Gitter, und 5 + 4 genau die beiden Arme dieses Kreuzes sind (siehe 7.1). Jeder
> andere Schnitt zerlegt einen Arm und macht beide Hälften einzeln
> uninterpretierbar. Bei den gemessenen ~40 min pro Punkt ist das ~3,5 h + ~2,5 h
> — so nah an „4 h + 3 h", wie neun Punkte es zulassen. Wer den ersten Block
> kürzer braucht, nimmt `--epochs 10` — dann aber **in beiden** Schritten.

> **Es entsteht bewusst kein Verdikt vor 7.3.** Ergebnisse werden immer
> gespeichert, aber ausgewertet wird erst am Schluss: das Verdikt vergleicht
> jeden Arm gegen den **gemeinsamen Mittelpunkt**, und der steckt in 7.1 — 7.2
> enthält ihn nicht. Ein halbes Kreuz auszuwerten hieße, eine Achse gegen sich
> selbst zu vergleichen, und es käme trotzdem eine plausibel aussehende Zahl
> heraus.

**Datenaufteilung:** Training `OP01–OP05`, Validierung `OP06` (entscheidet die
Auswahl), Test `OP07` (wird nur berichtet und fließt in keine Auswahl ein).

---
### 8.0 Hyperparameter — für 8.1 und 8.2 identisch

**Das ist die verbindliche Liste.** 7.1 und 7.2 müssen in *jedem* dieser Werte
übereinstimmen; weicht 7.2 ab, verwirft der Lauf den gespeicherten Teil aus 7.1,
statt zwei verschiedene Experimente zu mischen. Der Lauf schreibt denselben
Block nach `artifacts/benchmark_wphys_wbc_settings.txt` — dort steht im Zweifel,
womit 7.1 gelaufen ist.

**Daten und Zeitraster:**

| Flag | Wert | Bedeutung |
|---|---|---|
| `--ops` | `OP01 … OP05` | Trainings-OPs |
| `--val-op` | `OP06` | entscheidet die Auswahl |
| `--test-op` | `OP07` | nur Bericht, fließt in keine Auswahl ein |
| `--subsample` | `2` | `dt = 0.1 s × subsample` → **0.2 s**. **Nach oben gesperrt:** die CFL-Grenze liegt bei ~0.241 s, `subsample 3` wären schon 0.3 s. Als Laufzeit-Hebel steht dieses Flag hier also *nicht* zur Verfügung |

**Architektur — konfigurierbar, aber nicht trainiert:**

| Flag | Wert | Bedeutung |
|---|---|---|
| `--history-mode` | `hybrid` | `hybrid` = Anker + Raten, `raw` = reiner Lag-Stapel |
| `--delta-grid` | `0.2` s | **Anker** der History: `[T(t−Δgrid), rate₁, …]`, die Ratensegmente laufen von dort rückwärts. Unabhängig von `--subsample` — aber sinnlos kleiner als das Datenraster: liegt Δgrid unter `dt`, kann der Nachschlag nicht feiner auflösen und wirkt effektiv wie `dt` (`train.py` warnt dann). Hier gilt Δgrid = `dt` = 0.2 s. Nur im `hybrid`-Modus wirksam |
| `--rate-lags` | `5 20` | kumulative Segmentlängen in Sekunden; jedes Segment beginnt, wo das vorige endete. Nenner der Rate ist die **eigene Segmentlänge** (5 bzw. 20 s) |
| `--k-max` | `2` | nur im `raw`-Modus wirksam; im `hybrid`-Modus folgt `k` aus der Zahl der `rate_lags` |
| `--width` / `--depth` | `128` / `4` | MLP-Geometrie |
| `--time-deriv` | `bdf2` | Zeitableitung im Physik-Residuum |
| `--use-static` / `--use-forcing` | an / an | statische bzw. Forcing-Features |

**Optimierung:**

| Flag | Wert | Bedeutung |
|---|---|---|
| `--epochs` | **`20`** | Kapitel 8 fährt mit 60. Für „in welcher Dekade wirkt das Gewicht" reichen 20; für den Endwert eines Modells nicht |
| `--lr` | `2e-3` | Basis-Lernrate |
| `--weight-decay` | `0.0` | |
| `--gain-lr-mult` | `25.0` | `src_gain`/`diff_gain` lernen 25× schneller, sonst bleiben sie bei 1.0 |
| `--grad-clip` | `1.0` | maximale Gradientennorm |
| `--early-stopping-patience` | `0` | aus |
| `--batch-data` | `2048` | |
| `--batch-phys` / `--batch-bc` | `256` / `128` | Kollokationspunkte — hier liegt der GPU-Hebel |
| `--phys-norm` | `0` | `0` = adaptiver EMA, `>0` = fester Divisor |
| `--seeds` | `0` | ein Seed, mit Absicht (siehe 7.1) |

**Gesweept** werden nur `w_phys` und `w_bc`, jeweils über
`[0, 0.001, 0.01, 0.1, 1.0]`. `w_data` bleibt fest bei `1.0`.

**Lernparameter** (die einzigen Dinge, die der Gradient anfasst): MLP-Gewichte,
das per-Layer `β` des Swish, und die beiden Physik-Gains `src_gain`/`diff_gain`.
Die History-Struktur wird **nicht** gelernt — kein lernbares `δ`, keine
Lag-Gates. Wer sie optimieren will, sweept sie mit `benchmark_arch.py` (9.2).

Die Defaults stehen in `PINNmodulusTwo/config.yaml`; die CLI überschreibt sie
pro Lauf.

---

### 8.1 Schritt 1 — der `w_phys`-Arm (5 Punkte, ~3,5 h)

Bevor ein Gitter Stunden investiert, um Unterschiede *innerhalb* eines Bereichs
aufzulösen, klärt die Probe, ob dieser Bereich überhaupt der richtige ist — und
ob die Gewichte den Fehler überhaupt bewegen.

Statt eines Gitters läuft ein **Kreuz** durch einen gemeinsamen Mittelpunkt:
jedes Gewicht wird über die Dekaden `[0, 0.001, 0.01, 0.1, 1.0]` gefahren,
während das andere im Zentrum steht. **9 Punkte statt 25** — und diese 9
zerfallen sauber in die beiden Arme:

| Schritt | Arm | Punkte | Was variiert |
|---|---|---|---|
| **7.1** | `w_phys` | **5** | `w_phys ∈ [0, 0.001, 0.01, 0.1, 1.0]` bei festem `w_bc = 0.1` |
| 7.2 | `w_bc` | 4 | `w_bc ∈ [0, 0.001, 0.01, 1.0]` bei festem `w_phys = 0.01` |

Der Mittelpunkt `(w_phys=0.01, w_bc=0.1)` gehört zu 7.1; 7.2 überspringt ihn,
deshalb 5 + 4 = 9 und nicht 10. Genau deshalb geht 8.1 zuerst: **8.2 allein wäre
nicht auswertbar**, weil der Bezugspunkt fehlt.

```bash
cd ~/llmtraining
source .venv/bin/activate

# 5 Punkte x 1 Seed x 20 Epochen = 5 Trainings ~ 3,5 h
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --probe-part 1 --epochs 20 --device cuda > probe_part1.log 2>&1 &
echo $! > probe.pid
```

Der Lauf schreibt zuerst den vollständigen Hyperparameter-Block ins Log
(`subsample=2 -> dt=0.2s`, `delta_grid=0.2s`, `epochs=20`, alle Batches …) und
legt ihn zusätzlich als Datei ab. Am Ende steht:

```
  [probe] part 1/2 (w_phys arm) stored in .../artifacts/probe_parts.json
Part 1/2 (w_phys arm) done - 5 of 9 points trained.
  results saved: .../artifacts/benchmark_wphys_wbc.csv
  settings saved: .../artifacts/benchmark_wphys_wbc_settings.txt
  still to train: (w_phys=0.01, w_bc=0), (w_phys=0.01, w_bc=0.001), ...

Next - the other part, with these same flags:
  python3 benchmark_wphys_wbc.py --probe --probe-part 2 --epochs 20 --device cuda
```

Das ist der erwartete Abschluss, kein Fehler. **Die fünf Ergebnisse sind
gespeichert** — CSV, Settings und Rohzeilen. Nur ausgewertet wird noch nichts.

> **Ein Seed, mit Absicht.** `--seeds 0 1 2` wären 27 Trainings und damit ~18 h.
> Der Preis: das Verdikt kann die gefundenen Unterschiede nicht vom Init-Rauschen
> trennen und sagt das auch (`seed spread unknown`). Die Streuung klärt Schritt
> 9.1 separat. Für „welche Dekade überhaupt" reicht ein Seed, solange die
> Spannweite deutlich ist.

**Zwischendurch reinschauen:**

```bash
tail -f probe_part1.log                                   # live
grep -c '^[0-9]' PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv   # fertige Punkte
nvidia-smi                                                # läuft die GPU?
```

---

### 8.2 Schritt 2 — der `w_bc`-Arm (4 Punkte, ~2,5 h)

**Exakt dieselben Flags wie 7.1**, nur `--probe-part 2`:

```bash
cd ~/llmtraining
source .venv/bin/activate

# 4 Punkte x 1 Seed x 20 Epochen = 4 Trainings ~ 2,5 h
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --probe-part 2 --epochs 20 --device cuda > probe_part2.log 2>&1 &
echo $! > probe.pid
```

Weicht hier ein Hyperparameter von 7.1 ab, meldet der Lauf

```
  [probe] stored part(s) ran with different settings - discarding them;
  the other part has to be re-run to match this one.
```

und der Teil aus 7.1 ist verloren. Vorher also vergleichen:

```bash
cat PINNmodulusTwo/artifacts/benchmark_wphys_wbc_settings.txt
```

Am Ende:

```
Part 2/2 (w_bc arm) done - 9 of 9 points trained.
  results saved: .../artifacts/benchmark_wphys_wbc.csv

The cross is complete. Plots and verdict:
  python3 benchmark_wphys_wbc.py --probe --report-only --epochs 20 --device cuda
```

---

### 8.3 Schritt 3 — Auswertung und Plots (~1 min, ohne GPU)

Trainiert nichts. Liest die gespeicherten Ergebnisse beider Arme, fügt sie zum
vollständigen Kreuz zusammen und erzeugt **erst jetzt** Verdikt und Plots.

```bash
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --report-only --epochs 20 --device cpu
```

`--device cpu` ist hier in Ordnung — es wird nicht gerechnet, das Flag geht nur
in den Vergleich der Einstellungen ein. Ist das Kreuz unvollständig, bricht der
Schritt ab und nennt die fehlenden Punkte, statt ein halbes Ergebnis zu
berichten.

**Was entsteht:**

| Datei | Inhalt |
|---|---|
| `artifacts/benchmark_wphys_wbc_probe_boxplot.png` | **Boxplot:** ein Panel je Konfiguration, darin eine Box je Zeitpunkt |
| `artifacts/benchmark_wphys_wbc_probe.png` | je ein Panel pro Achse, MAE über die Dekaden, bestes Val-Ergebnis markiert |
| `artifacts/benchmark_wphys_wbc_probe_convergence.png` | Loss-Kurven (`L_data`, `L_phys`, `L_bc`) aller 9 Punkte über die Epochen |
| `artifacts/benchmark_wphys_wbc_best.txt` | Zusammenfassung, Hyperparameter-Block, **Verdikt je Achse** |
| `artifacts/benchmark_wphys_wbc.csv` | alle 9 Zeilen |
| `artifacts/benchmark_wphys_wbc_settings.txt` | die Einstellungen, die das erzeugt haben |

**Was im Boxplot steckt.** Alles darin kommt aus dem einen Test-OP `OP07`, der
nie an einer Auswahl beteiligt war. Daraus werden **10 zufällige Zeitpunkte**
gezogen — zufällig, weil ein gleichmäßiges Raster mit den Lastwechseln des OP in
Takt geraten kann, und mit festem Seed, damit alle 9 Konfigurationen an
*denselben* Momenten gemessen werden. Zu jedem dieser Zeitpunkte liefert der
Rollout einen Absolutfehler **pro Sensor**, also 363 Werte:

| Element der Box | Bedeutung |
|---|---|
| Box | mittlere 50 % der Sensoren — 25 % liegen darüber, 25 % darunter |
| rote Linie | der Median-Sensor |
| Whisker | bis 1,5 × Interquartilsabstand |
| Punkte darüber | einzelne Sensoren jenseits davon — die, die das Modell nicht trifft |

Genau das ist der Punkt gegenüber einer einzelnen MAE-Zahl: ein Mittelwert kann
gut aussehen, während eine Handvoll Sensoren weit daneben liegt. Im Boxplot ist
das ein langer oberer Whisker, im Mittelwert nicht zu sehen. Und weil die x-Achse
die Zeit ist, sieht man zusätzlich, ob der Fehler entlang des Rollouts wächst
oder gleich bleibt.

Die Panels folgen den Armen des Kreuzes: erst der `w_phys`-Arm in Dekaden-
Reihenfolge, dann der `w_bc`-Arm. Der auf `OP06` ausgewählte Punkt ist rot
umrandet.

Der Achsen-Plot (`_probe.png`) benutzt bewusst **keine Log-Achse**: die Dekaden
enthalten die 0, und die ließe sich logarithmisch nicht darstellen. Die Punkte
sitzen deshalb gleichmäßig verteilt, die echten Werte stehen als
Achsenbeschriftung.

Die Zahlen sind identisch mit einem Lauf in einem Stück: die Punkte werden vor
der Auswertung in dieselbe Kreuz-Reihenfolge sortiert, und die Trainingszeit
reist mit jeder Zeile mit, damit die Gesamtzeit die Zerlegung überlebt.

Anschauen:

```bash
grep -A20 'RANGE PROBE' PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
```

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

### 8.4 Was folgt daraus?

| Befund in der Probe | Konsequenz |
|---|---|
| Spannweite über die Dekaden **groß**, klares Minimum | 5×5-Gitter (8.3), zentriert auf diese Dekade |
| Spannweite **klein / flach** | Gitter für dieses Gewicht überspringen — es bewegt den Fehler nicht |
| beide Gewichte flach | zuerst Architektur-Benchmark (8.2): das Problem liegt woanders |
| Läufe divergiert (`[SKIP]`) | nicht weitermachen, [Kapitel 10](#11-troubleshooting) |

Die Probe lief mit **einem** Seed, das Verdikt sagt deshalb „seed spread
unknown". Ob die gefundenen Unterschiede echt sind, klärt Schritt 9.1.

---

### 8.5 Alles zum Kopieren

**Heute — Kapitel 6 und Schritt 7.1.** Kapitel 7 entscheidet, was die Gewichte in Kapitel 8 überhaupt bedeuten:

```bash
cd ~/llmtraining
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 6.1 + 6.2  RECHNET DIE GPU, KONVERGIERT DAS TRAINING?  (~6 min)
# ---------------------------------------------------------------------------
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cuda
python3 PINNmodulusTwo/smallBench.py --epochs 5 --w-phys 0.0 0.1 --w-bc 0.1 --device cuda
cat PINNmodulusTwo/artifacts/smallBench_results.txt
# Erwartung: "✓ ALL CHECKS PASSED - Ready for full benchmark!"
# NUR bei PASSED weitermachen.

# ---------------------------------------------------------------------------
# 6.3  WIE LANGE DAUERT EINE EPOCHE?  (~10 min)  <- bestimmt das Budget
# ---------------------------------------------------------------------------
python3 PINNmodulusTwo/train.py --ops OP01 OP02 OP03 OP04 OP05 --subsample 2 --epochs 5 --history-mode hybrid --rate-lags 5 20 --delta-grid 0.2 --grad-clip 1.0 --device cuda
# Log: "[112.4s/epoch, this run ~7 min left]"
# Sekunden/Epoche x 20 x 5 / 3600 = Stunden fuer 7.1
# Sekunden/Epoche x 20 x 4 / 3600 = Stunden fuer 7.2
# Deutlich zu viel? Dann --epochs 10 -- aber in BEIDEN Schritten.

# ---------------------------------------------------------------------------
# 7.1  BALANCING-BENCHMARK, Teil 1  (5 Trainings, ~3,5 h)
# ---------------------------------------------------------------------------
nohup python3 PINNmodulusTwo/benchmark_balance.py --part 1 --epochs 20 --device cuda > balance_part1.log 2>&1 &
echo $! > balance.pid
# Spalte "drift" lesen: ~1 = Gewicht aus einer kurzen Probe uebertraegt sich.
# Danach 7.2 (--part 2) und erst dann die Gewichte-Probe unten, IMMER mit
# denselben --loss-balance-Flags.

# ---------------------------------------------------------------------------
# 8.1  RANGE-PROBE, w_phys-ARM  (5 Punkte, ~3,5 h)
# ---------------------------------------------------------------------------
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --probe-part 1 --epochs 20 --device cuda > probe_part1.log 2>&1 &
echo $! > probe.pid

echo "PID:  $(cat probe.pid)"
echo "Live: tail -f probe_part1.log"
echo "Stop: kill \$(cat probe.pid)"
# Terminal kann jetzt geschlossen werden.
# Erwarteter Abschluss: "Part 1/2 (w_phys arm) done - 5 of 9 points trained."
# Das ist KEIN Fehler: Ergebnisse sind gespeichert, ausgewertet wird in 7.3.
```

**Später — Schritte 8.2 und 8.3:**

```bash
cd ~/llmtraining
source .venv/bin/activate

# Erst die Einstellungen aus 8.1 nachlesen und die Flags danach richten:
cat PINNmodulusTwo/artifacts/benchmark_wphys_wbc_settings.txt

# ---------------------------------------------------------------------------
# 8.2  RANGE-PROBE, w_bc-ARM  (4 Punkte, ~2,5 h)
# ---------------------------------------------------------------------------
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --probe-part 2 --epochs 20 --device cuda > probe_part2.log 2>&1 &
echo $! > probe.pid
echo "Live: tail -f probe_part2.log"

# ---------------------------------------------------------------------------
# 8.3  AUSWERTUNG UND PLOTS  (~1 min, kein Training)
# ---------------------------------------------------------------------------
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --report-only --epochs 20 --device cpu

grep -A20 'RANGE PROBE' PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt
ls -lh PINNmodulusTwo/artifacts/benchmark_wphys_wbc_probe*.png
# _probe_boxplot.png  -> Fehlerverteilung ueber die 363 Sensoren je Zeitpunkt
# _probe.png          -> MAE je Achse ueber die Dekaden
# _probe_convergence  -> Loss-Kurven
```

---

### 8.6 Checkliste

Vor 7.1:

- [ ] venv aktiviert (`source .venv/bin/activate`)
- [ ] `ls data_cache/*.npz` zeigt OP01–OP07
- [ ] Kapitel 6 gelaufen, Smoke-Test PASSED
- [ ] Zeit pro Epoche gemessen, 7.1 und 7.2 einzeln nachgerechnet
- [ ] `nohup` oder `tmux` benutzt, damit der Lauf die SSH-Session überlebt

Nach 7.1:

- [ ] Log endet mit `Part 1/2 (w_phys arm) done - 5 of 9 points trained`
- [ ] `benchmark_wphys_wbc.csv` hat 5 Datenzeilen
- [ ] `benchmark_wphys_wbc_settings.txt` und `probe_parts.json` existieren

Vor 7.2:

- [ ] Flags gegen `benchmark_wphys_wbc_settings.txt` verglichen (Epochen, OPs, Seeds, Batches, lr, subsample)

Nach 7.3:

- [ ] `benchmark_wphys_wbc.csv` hat 9 Datenzeilen
- [ ] beide `*_probe*.png` existieren
- [ ] Verdikt je Achse gelesen — welche Dekade, und bewegt sich überhaupt etwas?
- [ ] entschieden, welcher Lauf aus Kapitel 9 als Nächstes kommt

---

## 9. Große Benchmarks — Tage

Alles hier läuft mit **60 Epochen** und dauert Tage, nicht Stunden. Erst starten,
wenn Kapitel 8 durch ist und die Probe gesagt hat, wo es sich lohnt.

| Lauf | Trainings | Laufzeit |
|---|---|---|
| 8.1 Seed-Streuung an einem Punkt | 3–5 | ~6–10 h |
| 8.2 Architektur-Benchmark (16 Konfigs) | 16 | ~1–1,5 Tage |
| 8.3 5×5-Gitter der Gewichte | 25 | ~1,5–2 Tage |
| 8.3 mit `--extended-grid` (10×10) | 100 | ~6–8 Tage |

Mit Seeds multipliziert sich das entsprechend. `--epochs 20` drittelt alles und
ist für Vergleiche zwischen Konfigurationen meist ausreichend.

---
### 9.1 Seed-Streuung an einem Gitterpunkt

**Der erste Lauf dieses Kapitels**, weil er entscheidet, wie die anderen zu
lesen sind. Er beantwortet: bewegt sich der Fehler zwischen zwei Konfigurationen
mehr, als er sich zwischen zwei Zufalls-Initialisierungen *derselben*
Konfiguration bewegt? Ist die Antwort nein, sind die Rangfolgen aller folgenden
Läufe Rauschen — und ein feineres Gitter macht es nicht besser.

```bash
# 5 Seeds an einem Punkt, ~10 h. Mit --epochs 20 sind es ~3 h.
python3 PINNmodulusTwo/benchmark_wphys_wbc.py --w-phys 0.05 --w-bc 0.1 --seeds 0 1 2 3 4 --epochs 60 --device cuda
```

Dann in `benchmark_wphys_wbc_best.txt` die Spalte `+/-` ansehen — das ist die
Standardabweichung über die fünf Seeds.

- **Streuung klein** (deutlich unter den Unterschieden, die du in einer Heatmap
  vergleichen willst): die langen Sweeps liefern lesbare Rangfolgen.
- **Streuung groß**: ein feineres Gitter bringt nichts, weil der Sieger ohnehin
  vom Zufall bestimmt wird. Dann in 9.2 und 9.3 **mit `--seeds 0 1 2`**
  arbeiten und die Gitter kleiner halten.

---

### 9.2 Architektur-Benchmark

Misst Breite, Tiefe und History-Lags — Werte, die im Gewichte-Sweep ungemessen
festliegen. Läuft **eine Achse nach der anderen** gegen eine gemeinsame
Baseline, statt ein Produktgitter aufzuspannen: die Frage ist, welcher Regler den
Fehler überhaupt bewegt. `width × depth × lags` wären mehrere hundert Trainings,
achsenweise sind es zwölf.

```bash
nohup python3 PINNmodulusTwo/benchmark_arch.py --device cuda --seeds 0 1 2 > benchmark_arch.log 2>&1 &

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

> Setze die in 6.3 gemessene Zeit pro Epoche ein, statt diese Tabelle zu
> übernehmen. Die Werte gelten außerdem für `width=128`: breitere Netze kosten
> mehr pro Schritt, aber der Rollout ist latenzgebunden (~7000 sequentielle
> Zeitschritte), nicht rechengebunden — `width=256` skaliert daher schwächer als
> die vierfache FLOP-Zahl vermuten lässt.
>
> Mit `--epochs 20` statt 60 drittelt sich alles. Für die Frage „welche Achse
> bewegt überhaupt etwas" reicht das meist.

---

### 9.3 5×5-Gitter der Loss-Gewichte

Sucht das optimale Paar `(w_phys, w_bc)` auf dem 5×5-Standardgitter — zentriert
auf die Dekade, die das Verdikt aus [Schritt 8.3](#83-schritt-3--auswertung-und-plots-1-min-ohne-gpu)
als wirksam ausgewiesen hat.

```bash
cd ~/llmtraining
source .venv/bin/activate

# Werte an die Dekade aus der Probe anpassen:
nohup python3 PINNmodulusTwo/benchmark_wphys_wbc.py --device cuda --w-phys 0.003 0.01 0.03 0.1 0.3 --w-bc 0.03 0.1 0.3 0.7 1.0 > benchmark_grid.log 2>&1 &
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

Mit `--epochs 20` sind es ~14 h. Wenn 9.1 eine große Seed-Streuung gezeigt hat,
ist ein kleines Gitter mit drei Seeds aussagekräftiger als ein großes mit einem.

**`--extended-grid`** schaltet auf 10×10 = 100 Punkte um, also **~6–8 Tage bei
einem Seed**. Das lohnt praktisch nie: die Probe hat die Dekade bereits
eingegrenzt, und 100 Punkte auf einem Seed liefern vor allem das Minimum aus 100
Ziehungen — siehe das Rausch-Verdikt in 9.4.

---

### 9.4 Seeds — wie viele Läufe pro Gitterpunkt

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

### 9.5 Erwartete Outputs

```
PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv               -> alle 100 Punkte
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_settings.txt      -> Hyperparameter des Laufs
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_heatmap.png       -> 2D-MAE-Heatmap (Val)
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_boxplot.png       -> Sensor-Fehler je Konfiguration
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_convergence.png   -> Loss-Kurven (Ecken + Best)
PINNmodulusTwo/artifacts/benchmark_wphys_wbc_best.txt          -> beste Kombination + Tabelle
PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/*.pt            -> 100 Modelle (mehrere GB!)
```

---

### 9.6 Monitoring

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

### 9.7 Auswertung nach Abschluss

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
head -1 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv && tail -n +2 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv | sort -t',' -k7 -n | head -10

# Dieselben Punkte mit ihrer Test-MAE (Spalte 9) - die Zahl, die berichtet wird.
# Wenn die Reihenfolge hier stark von der obigen abweicht, ist die Auswahl instabil.
tail -n +2 PINNmodulusTwo/artifacts/benchmark_wphys_wbc.csv | sort -t',' -k7 -n | head -10 | cut -d',' -f1,2,7,8,9,10

# Checkpoints: Anzahl und Platzbedarf
ls PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/ | wc -l   # sollte 100 sein
du -sh PINNmodulusTwo/artifacts/checkpoints_wphys_wbc/
```

---

### 9.8 Abbrechen und neu starten

```bash
kill $(cat benchmark.pid)
# falls das nicht greift:
pkill -f benchmark_wphys_wbc
ps -p $(cat benchmark.pid)   # "No such process" = gestoppt
```

**Es gibt keine Resume-Funktion** — ein Neustart beginnt wieder bei Punkt 1.

---

### 9.9 Weiter mit den besten Gewichten

```bash
# Werte aus benchmark_wphys_wbc_best.txt einsetzen
python3 PINNmodulusTwo/train.py --epochs 100 --w-phys 0.1 --w-bc 0.3 --subsample 2 --ops OP01 OP02 OP03 OP04 OP05 --test-op OP07 --device cuda
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

## 10. Ergebnisse zurückholen

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

## 11. Troubleshooting

| Symptom | Ursache / Abhilfe |
|---|---|
| `train.py: error: unrecognized arguments: --delta-grid 0.2` (oder `--gain-lr-mult`, `--test-op`, …) | Der Code auf dem Server ist älter als diese Anleitung — das Flag gibt es dort noch nicht. `git checkout main && git pull origin main`, dann `python3 PINNmodulusTwo/train.py --help` gegenprüfen. Der Aufruf in `usage:` listet immer genau die Flags, die dein Checkout kennt. |
| `torch.cuda.is_available()` → `False` | CPU-Wheel installiert (`torch.version.cuda` ist `None`) → torch neu vom `cu12x`-Index installieren. Oder Treiber fehlt/zu alt → Schritt 1. |
| `--device cuda ... torch.cuda.is_available() is False` | Genau derselbe Fall — die Fehlermeldung nennt Torch- und CUDA-Version zum Abgleich. |
| `CUDA error: no kernel image is available` | GPU zu neu für das Wheel (z. B. RTX 50xx mit `cu121`) → `cu128`-Index nehmen. |
| `CUDA out of memory` | `--batch-phys` / `--batch-data` / `--batch-bc` senken oder `--subsample` erhöhen. Belegt ein Zombie-Prozess die Karte? → `nvidia-smi`, ggf. `kill`. |
| `ModuleNotFoundError: modulus` | venv nicht aktiviert oder `pip install nvidia-modulus` fehlt → Schritt 4. |
| `FileNotFoundError: .../data_cache/OP01.npz` | Daten nicht übertragen → Schritt 5. |
| `FileNotFoundError: .../material_properties/constants.yaml` | dito → Schritt 5. |
| `L_phys_bal=nan` / leere Kurve im Plot | Kein Fehler: bei `w_phys 0` (bzw. `w_bc 0`) wird der Term gar nicht mehr berechnet — er kostet einen Autograd-Hessian und trägt nichts bei. Als NaN protokolliert, damit der Plot eine Lücke zeigt statt einer flachen Linie, die es nie gab. Mit `--zero-weight-terms compute` wird er zum Mitloggen weiter gerechnet. |
| Gewichte-Probe verwirft den anderen Arm | Die Balancing-Flags stecken in der Signatur. Teil 2 braucht dieselben wie Teil 1 — den fertigen Aufruf druckt Teil 1 am Ende aus. |
| `[CFL WARN]` beim Start | Zeitschritt zu groß für die Diffusion — `--subsample` verkleinern (z. B. `--subsample 2`) oder `--grad-clip 1.0` setzen. Unabhängig von der GPU. |
| Training bricht mit `[ABORT] ... loss exploded` ab | Gleiche Ursache wie oben; die Meldung nennt die empfohlenen Gegenmaßnahmen. |
| `No space left on device` während des Benchmarks | Die 100 Checkpoints brauchen mehrere GB → mit `--no-save-models` (gar keine) oder `--save-best-only` (nur das beste Modell) neu starten. |
| Benchmark startet nicht, "läuft schon" | Alten Prozess finden und beenden: `ps aux \| grep benchmark_wphys_wbc`, dann `kill <PID>`. |
| Loss explodiert mitten im Benchmark | Lauf stoppen (`kill $(cat benchmark.pid)`) und mit stabileren Einstellungen neu starten: `--grad-clip 2.0 --lr 0.001`. |
| Läuft auf der GPU kaum schneller | Erwartbar: pro Epoche gibt es ~7000 *sequentielle* Rollout-Schritte je OP, die sich nicht parallelisieren lassen. Batchgrößen erhöhen hilft nur dem Physik-Term (9.6). |

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
