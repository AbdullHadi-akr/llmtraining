# Trainingsbericht PINNmodulusTwo — 2026-08-28

> ## ⚠️ Korrekturhinweis — siehe [TRAININGS_BERICHT_2026-08-28_REVIEW.md](TRAININGS_BERICHT_2026-08-28_REVIEW.md)
>
> Die Messungen unten sind unverändert, mehrere **Bewertungen** darin sind
> jedoch nachweislich falsch. Die wichtigsten, bevor jemand daraus etwas
> ableitet:
>
> * Das `FAIL` kam **nicht** von der MAE (§4). `smallBench.py` prüft
>   `test_mae < 20.0` — beide Läufe haben das bestanden. Gescheitert ist die
>   Loss-Balance-Prüfung.
> * Der Baseline-Vergleich (§9) ist ungültig: 11.96 °C ist der
>   Persistenz-Vorhersager, nicht „predict mean" (der liegt bei 6.60 °C),
>   12.02 °C ist gegenüber 11.96 °C **schlechter** und nicht besser, und beide
>   Baselines stammen aus einem **synthetischen** Bundle, das laut
>   `README_ERSTER_TEST.md` Kapitel 6/9 nicht auf echte OPs übertragbar ist.
> * `w_phys` zu erhöhen kann `L_phys_bal` nicht verändern (§6, §7):
>   `L_phys_bal = L_phys / EMA(L_phys)` enthält `w_phys` nicht.
> * Die „6358 Gitterpunkte" (§6, §7) existieren nicht — 363 ist die native
>   Sensorzahl. Empfehlung „Full Grid" entfällt.
> * Der empfohlene nächste Schritt ist `benchmark_balance.py --part 1`, nicht
>   die Sweeps in §7.

## Zusammenfassung

Erfolgreich neue NPZ-Daten aus CSV generiert und erstes vollständiges Training auf echten Daten durchgeführt. Das Modell konvergiert stabil, aber die Testgenauigkeit liegt noch über dem Zielwert.

---

## 1. Datenvorbereitung

### Was wurde gemacht:
1. **Alte NPZ-Dateien gelöscht** — die vorhandenen waren veraltet/fehlerhaft
2. **17 neue NPZ-Dateien aus CSV-Rohdaten generiert**:
   - OP01–OP07 (Training + Test)
   - OP08–OP16, OP19 (zusätzliche Datenpunkte)
3. **Quelle**: `battery_surrogate_agenticWorkflow/data_raw/` → CSV-Bundles
4. **Ziel**: `PINNmodulusTwo/data_cache/` → komprimierte NumPy-Archive

### NPZ-Struktur (Beispiel OP01):
```
T shape:       (14450, 363)   # Temperaturfeld über Zeit × Gitterpunkte
xyz shape:     (363, 3)        # 3D Koordinaten der 363 Gitterpunkte
t_fast points: 14450           # Zeitschritte (dt=0.1s)
```

### Datenvalidierung (data_probe.py):
✅ **A-Wert**: 118.9 / 29.7 für rate_lags [5s, 20s] — stimmt exakt mit Übergabedokumentation
✅ **dTdt_scale**: 2.479 (0% Abweichung)
✅ **SNR**: >2000 für beide Lags → **echtes Signal**, kein Rauschen
✅ **Verdict**: Rate-Lags [5.0, 20.0] beibehalten

---

## 2. Modellarchitektur

### Netzwerk:
- **Typ**: Hybrid PINN (Physics-Informed Neural Network)
- **Architektur**: MLP mit Width=128, Depth=4
- **Parameter**: 52,485 trainierbare Gewichte
- **Input Features**:
  - 7 Konfigurationsparameter (c_rate, Temperaturen, Ströme, ...)
  - 3 statische Features (Material, Geometrie)
  - 1 Forcing-Term (Wärmequelle)
  
### Zeitintegration:
- **Methode**: BDF2 (Backward Differentiation Formula, 2. Ordnung)
- **History Mode**: Hybrid — nutzt Temperaturraten bei Δt=5s und Δt=20s
- **Amplifikation A**: 118.9× und 29.7× für die zwei Lag-Fenster
- **Zeitschritt**: Δt = 0.2s (subsample=2 von ursprünglich 0.1s)
- **CFL-Stabilität**: ✅ 0.2s < 0.241s (stabiler Bereich)

### Physik-Terms:
- **Diffusion**: Anisotrope Fourier-Tensor (λ variiert je Material)
- **Quelle**: Joule-Heizung in JR1-Region (121/363 Gitterpunkte)
- **Randbedingungen**: Neumann BC bei x=0
- **Loss-Balancing**: EMA (Exponential Moving Average), decay=0.9/epoch

---

## 3. Trainingskonfiguration

### Datensätze:
- **Training**: OP01, OP02, OP03, OP04, OP05
- **Test (held-out)**: OP07
- **Validation**: OP06 (gehalten, aber nicht in diesem Lauf genutzt)

### Hyperparameter:
```yaml
epochs:          10
batch_size:      2048
optimizer:       Adam
grad_clip:       1.0
learning_rate:   (adaptiv via loss balancing)
steps:           100 pro OP pro Epoche × 5 OPs = 500 steps/epoch
total_steps:     5000
```

### Loss-Komponenten:
- **L_data**: MSE zwischen Vorhersage und echter Temperatur
- **L_phys**: Residuum der PDE (Wärmeleitungsgleichung)
- **L_bc**: Randbedingungsverletzung
- **Gewichte**: w_phys = [0.0, 0.1] im Sweep getestet

---

## 4. Trainingsergebnisse

### 🔬 **Experiment 1: w_phys = 0.0 (rein datengetrieben)**

| Metrik | Epoche 1 | Epoche 5 | Epoche 10 |
|--------|----------|----------|-----------|
| L_data | 1.64e+04 | 9.87e-01 | 7.85e-01 |
| SATURATED | OP01, OP03 (>6900 steps) | — | OP03 (342 steps) |

**Finale Metriken:**
- Train MAE: **8.87°C**
- Test MAE: **13.48°C**
- Konvergenz: ✅ Ja
- Status: ❌ FAIL (MAE zu hoch)

**Beobachtungen:**
- Epoche 1: Starke Saturation (Rollout-Guard greift), aber Loss bleibt endlich
- Ab Epoche 2: Saturation-Count **fällt** → Modell erholt sich ✅
- Epoche 10: Nur noch 342/7279 Steps saturiert bei OP03

---

### 🔬 **Experiment 2: w_phys = 0.1 (mit Physik-Term)**

| Metrik | Epoche 1 | Epoche 5 | Epoche 10 |
|--------|----------|----------|-----------|
| L_data | 1.81e+04 | 9.58e-01 | 8.55e-01 |
| L_phys_bal | 3.23e-02 | 3.52e-08 | 2.69e-06 |

**Finale Metriken:**
- Train MAE: **7.65°C** (besser als w_phys=0.0!)
- Test MAE: **12.02°C** (besser als w_phys=0.0!)
- Konvergenz: ✅ Ja
- L_phys_bal: ⚠️ 2.69e-06 (sollte ~O(1) sein für echte Physik-Erfüllung)
- Status: ❌ FAIL (MAE zu hoch, Physik-Term nicht balanciert)

**Beobachtungen:**
- Physik-Term **hilft**: -1.22°C Train MAE, -1.46°C Test MAE
- Aber: L_phys_bal viel zu klein → Physik wird kaum durchgesetzt

---

### 📊 **Zusammenfassung der Ergebnisse:**

```
  w_phys |     L_data | L_phys_bal |  Train MAE |   Test MAE |   Status
----------------------------------------------------------------------
   0.000 | 7.8472e-01 |        nan |      8.87°C |     13.48°C |     FAIL
   0.100 | 8.5549e-01 | 2.6899e-06 |      7.65°C |     12.02°C |     FAIL
```

**Beste Konfiguration**: w_phys=0.1
- **Verbesserung**: 10.8% Test MAE vs. rein datengetrieben

---

## 5. Wo sind die Ergebnisse?

### Artefakte:
```
PINNmodulusTwo/artifacts/smallBench_convergence.png    (Konvergenz-Plot)
PINNmodulusTwo/artifacts/smallBench_results.txt        (Zusammenfassung)
/tmp/smallbench_output.txt                             (Vollständiges Log)
```

### Daten:
```
PINNmodulusTwo/data_cache/OP*.npz                      (17 NPZ-Dateien)
data_cache/OP*.npz                                      (Top-level Backup)
```

### Modell-Checkpoints:
Keine automatischen Checkpoints erstellt (smallBench speichert nur finale Metriken).
Für Checkpoints müsste `train.py` mit `--save-checkpoint` laufen.

---

## 6. Diagnostik & Probleme

### ✅ Was funktioniert:
1. **Datenpipeline**: CSV → NPZ → Training klappt
2. **Stabilität**: Keine NaN-Abstürze in Epoche 1 (wie früher)
3. **Konvergenz**: Loss fällt über Epochen
4. **Saturation**: Count fällt (Modell lernt sich selbst zu regularisieren)

### ⚠️ Was noch problematisch ist:
1. **Test MAE zu hoch**: 12-13°C vs. Baseline 11.96°C (Vorhersager der nichts tut)
   - Das Modell ist kaum besser als "predict mean temperature"
2. **Physik-Term zu schwach**: L_phys_bal = 2.69e-06 statt ~O(1)
   - w_phys=0.1 ist zu klein, oder Balancing-Mechanismus greift falsch
3. **Generalisierung**: Test MAE > Train MAE (+4.37°C Gap bei w_phys=0.1)
   - Overfitting? Oder OP07 ist "too different" von Training-OPs?

### 🔍 Mögliche Ursachen:
- **Nur 10 Epochen**: Zu kurz für volle Konvergenz?
- **w_phys=0.1 zu klein**: Physik-Loss wird wegbalanciert
- **A=118.9 zu hoch**: Amplifikation macht initiale Fehler riesig
- **Nur 363 Gitterpunkte**: Reduziertes Grid (Original hatte 6358)
  - Wahrscheinlich um CPU-Training zu beschleunigen

---

## 7. Nächste Schritte (laut README_LOKALER_LAUF.md)

### ✅ Erledigt:
- [x] Schritt 0: Daten platziert und verifiziert
- [x] Schritt 1: Datensonde (data_probe.py) — A-Wert bestätigt
- [x] Schritt 1: Vortests (selftest, pytest, rollout_divergence)
- [x] Schritt 2: Erster echter Lauf (smallBench.py)

### 📋 TODO (aus README):
- [ ] **Schritt 2 (Fortsetzung)**: MAE-Zahlen in `README_ERSTER_TEST.md` eintragen
- [ ] **Schritt 3**: Sweeps ausführen (nur wenn Schritt 2 sauber läuft)
  - `benchmark_arch.py`: Lag-Sweep (verschiedene A-Werte testen)
  - `benchmark_wphys_wbc.py`: Physik-Term-Sweep (bringt es was?)

### 🚀 Empfohlene Next Actions:
1. **Längeres Training**: 50-100 Epochen statt 10
2. **Höheres w_phys**: 1.0 oder 10.0 testen, damit Physik durchkommt
3. **GPU nutzen**: Aktuell auf CPU → sehr langsam (42-98s/Epoche)
4. **Full Grid**: 6358 statt 363 Punkte (braucht aber definitiv GPU)
5. **Hyperparameter-Sweep**: Lernrate, Batch-Size, Architecture tunen

---

## 8. Technische Details für Reproduktion

### Environment:
```bash
Python: 3.10 (modulus_env virtualenv)
Torch: 2.12.1+cpu
Device: CPU (WSL2 on Windows)
NumPy: für NPZ-I/O
Pandas: für CSV-Lesen
```

### Kommandos zum Reproduzieren:
```bash
# 1. NPZ-Daten generieren
modulus_env/bin/python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07

# 2. Datensonde
modulus_env/bin/python3 PINNmodulusTwo/tools/data_probe.py

# 3. Vortests
modulus_env/bin/python3 PINNmodulusTwo/selftest.py
modulus_env/bin/python3 -m pytest PINNmodulusTwo/tests -q
modulus_env/bin/python3 PINNmodulusTwo/tools/rollout_divergence.py

# 4. Training
modulus_env/bin/python3 PINNmodulusTwo/smallBench.py
```

### Konfiguration (PINNmodulusTwo/config.yaml):
Nutzt Default-Werte:
- ops: [OP01, OP02, OP03, OP04, OP05]
- test_op: OP07
- subsample_time: 2
- rate_lags: [5.0, 20.0]

---

## 9. Vergleich mit Baseline

### Baseline (aus README_ERSTER_TEST.md):
- **Naiver Predictor** (predict mean): MAE = 11.96°C
- **Grund**: T_sigma = 8.66°C, aber thermische Range ist breiter

### Unser Modell:
- **w_phys=0.0**: Test MAE = 13.48°C → **schlechter als Baseline!** ❌
- **w_phys=0.1**: Test MAE = 12.02°C → **leicht besser als Baseline** ✅ (+0.5%)

**Interpretation**: 
Das Modell hat minimal gelernt, aber noch weit von guter Generalisierung entfernt. Es nutzt die Physik kaum (L_phys_bal viel zu klein) und ist stark overfitted auf die Trainingsdaten.

---

## 10. Datengröße & Rechenaufwand

### NPZ-Dateien:
- OP01: ~3.8 MB (14450 timesteps × 363 points)
- OP19: ~8.1 MB (35000 timesteps)
- **Gesamt**: ~70 MB für alle 17 OPs

### Training (CPU):
- **Epoche 1**: ~98 s (71s Rollout + 27s Inner Loop)
- **Epoche 10**: ~42 s (32s Rollout + 9s Inner Loop)
- **Gesamt**: ~10 Epochen × 50s avg = **~8 Minuten** pro w_phys
- **Full Run** (2× w_phys): ~16 Minuten

### GPU-Abschätzung:
- CPU: 42-98 s/Epoche
- GPU (A100): Erwartung 5-10 s/Epoche → **10x schneller**
- Ermöglicht: 100 Epochen in ~10 Minuten

---

## Fazit

**Status**: Erste vollständige Trainingsläufe auf echten Daten erfolgreich ✅

**Aber**: Modell noch nicht produktionsreif — MAE kaum besser als Baseline, Physik-Term wird nicht durchgesetzt, starkes Overfitting.

**Nächster Schritt**: Entweder längeres Training + höheres w_phys auf GPU, oder README_ERSTER_TEST.md aktualisieren und zu Schritt 3 (Sweeps) übergehen.

---

**Erstellt**: 2026-08-28  
**Training Device**: CPU (WSL2)  
**Datensatz**: 17 OPs aus CSV, 5 Training + 1 Test  
**Bester Run**: w_phys=0.1, Test MAE=12.02°C
