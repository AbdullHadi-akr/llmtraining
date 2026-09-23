# Nächste Sitzung — Stand 23.09.2026

> **Der eine Einstieg nach einer Pause.** Wo du stehst, wie es läuft, womit du
> anfängst — und die Antworten auf die Fragen der letzten Sitzung, damit keine
> offen bleibt. Die Einzelheiten stehen in den lebenden Dokumenten:
>
> * [`README_MODELLSTAND.md`](README_MODELLSTAND.md) — **neu:** jede Modellversion und jedes Experiment, chronologisch, mit der Version, auf der es lief
> * [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) — Kopf-Kasten und **§11.10** sind neu
> * GridCNN: [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md) und [`UEBERGABE_2026-09-23.md`](UEBERGABE_2026-09-23.md) — **in dieser Sitzung nicht angefasst**
>
> Die `UEBERGABE_*.md` im Wurzelverzeichnis sind die Protokolle der
> GridCNN-Sitzungen vom 22./23.09. und bleiben stehen. **Diese Datei ist der
> Einstieg für beide Projekte.**

---

## 0 · Die Antworten zuerst

| Frage | Antwort |
|---|---|
| **Ist das PINN jetzt besser?** | **Nein — das eingesetzte Modell rechnet bitgleich wie vorher** (P2.1 mit Defaults = P2, per Test belegt). Besser ist die **Diagnose**: der Plan „O18 einbauen" hätte an der falschen Stelle repariert. **Mit dem neuen Schalter** `--phys-stencil live` war die val-MAE auf dem **synthetischen** Cache in allen drei Paaren niedriger (2.1–2.9 statt 6.3–10.4 °C) — ein Seed, synthetisch: ein Grund, Achse 5 zu fahren, kein Beleg |
| **Fahren wir mit diesem Modell fort?** | **Ja.** Das MLP bleibt, wie es ist (4 × 128, lernbares Swish, hybride Historie). Geändert wird als Nächstes nur, **wo der Physik-Term das MLP auswertet** — und erst, nachdem die Messung auf den echten Checkpoints sagt, welche Reparatur passt |
| **Wurde die Physik angepasst? Sie ist doch kaputt.** | **Die Rechnung nicht — mit Absicht.** Kaputt ist sie auf dem synthetischen Cache nachgewiesen, auf echten Daten noch nicht. **Gebaut ist die erste Reparatur als Schalter:** `--phys-stencil live` (Default `buffer` = wie bisher). Die zweite (`[BLIND]`) ist entworfen, alle Entscheidungen vorbelegt (§4 Schritt 3), gebaut wird sie, wenn die Messung sie verlangt. Ohne Messung einzuschalten hieße, den O18-Fehler ein zweites Mal zu machen |
| **Was ist mit GridCNN?** | **Nicht angefasst**, weder Code noch Doku. Andere Sitzungen haben es am 22./23.09. weit gebracht (§2); sein nächster Schritt steht in §4 |
| **Wo stand ich in den Benchmarks?** | §2 |
| **Wie läuft es, 1–10?** | **6/10** — §3 |

---

## 1 · Die Maschine — vor JEDEM Kommando in dieser Datei

Alles unten läuft auf **deiner g4dn.2xlarge**: Tesla **T4** (15.6 GiB, sm_75),
8 vCPU, davon **4 physische Kerne**. Sie trägt mehrere Läufe gleichzeitig —
aber nur mit **MPS**. Gemessen (10.09.): `-j 4` **ohne** MPS 1.46×, **mit** MPS
**3.69×**. Ohne MPS verschenkst du Faktor 2.5, und kein Log sagt es.

```bash
cd ~/llmtraining                        # der Linux-Rechner, nicht /mnt/c/...
git checkout main && git pull
source modulus_env/bin/activate         # danach `python`, nicht `python3` (PINN-FAHRPLAN §11.8)
systemctl is-active nvidia-mps          # "active" -- MPS als Unit seit 22.09. (README_GPU_SERVER §6.4)
pgrep -x nvidia-cuda-mps || nvidia-cuda-mps-control -d   # Notnagel, falls die Unit fehlt
pgrep -af "sweep.py|train.py|residual_decomposition" || echo "Karte frei"
```

* **Die Unit einmalig einrichten**, falls `systemctl` nicht `active` sagt:
  `sudo cp PINNmodulusTwo/deploy/nvidia-mps.service /etc/systemd/system/` und
  README_GPU_SERVER §6.4. Danach überlebt MPS jeden Reboot.
* **Parallel heißt `-j`** — `sweep.py -j 4` und seit heute
  `residual_decomposition.py -j 4`. Ein einzelner `train.py` ist **ein** Prozess.
* **Das Budget sind 4 Prozesse** (4 physische Kerne; mit MPS ist die CPU die
  Grenze, nicht die Karte). Zwei Jobs teilen es sich: z. B. PINN-Achse 5 mit
  `-j 3` **plus** GridCNN-Lauf 17 als ein Prozess = 4.
* **Planungszahl PINN:** ~2 h je 60-Epochen-Lauf auf 11 OPs; Wanduhr ≈
  `ceil(Läufe/j)` × 2 h.

---

## 2 · Wo du stehst

### Die Versionen (Einzelheiten und Commits: `README_MODELLSTAND.md`)

| | Version | seit | Stand |
|---|---|---|---|
| PINN | **P2** | 01.09. | Quelle korrigiert — **alle** echten Läufe (Schritt 5b, 6, Achse 0, 1) |
| PINN | **P2.1** | **23.09.** | + Zerlegungswerkzeug, `--phys-stencil live` (aus), `--time-deriv autograd` gesperrt — Verhalten **unverändert** |
| GridCNN | **G4.1** | 22./23.09. | Ladepfad angeschlossen, Stufe 5 (TBPTT) gebaut, Fehlerprofil + `nachmessen.py` |
| Cache | **Schema v3** | 22.09. | Wandpfad dazu; `T` und `q_source` gleich gebaut |

### GridCNN — die Leiter hat sich bewegt

| Stufe | Stand |
|---|---|
| 0 Rangtest | ✅ rot, überstimmt (16 × 3) |
| 1 Bilanz | ✅ **22.09.** — Fluidbilanz 🟢, `U(V̇)` kalibriert, null freie Parameter |
| 2 Cache | ✅ **22.09.** — Schema v3 (`8d76084`) |
| 3 Löser | gebaut, adiabat, nicht als Latte gemessen |
| 4 CNN | **trainiert**: Lauf 14 (kein Ergebnis) → Lauf 15 POC, dt = 1 s: **OP06 6.571 ± 0.383 (0.61×), OP09 5.813 ± 0.867 (0.75×)** — erstmals lesbar |
| 5 TBPTT | ✅ gebaut 22.09.; Lauf 16 (volle Auflösung): 0.62× / 1.11×, Streuung 1.5–1.8 °C → **kein Ergebnis**, weil `k` in Schritten stand und `lag2` nie erreichte |
| 6 Vergleich mit dem PINN | offen |

> Die Leitertabelle *in* `GridCNN/FAHRPLAN.md` ist an Stufe 2, 4, 5 nicht
> nachgezogen — hier steht der Stand aus den Commits und Berichten.
>
> `benchmark.py` (die andere Leiter, ⚠ auch „Stufe" genannt): letzter Lauf
> 14.09., Stufe 0 und 2 🟢, Stufe 1 und 3 nie gelaufen. Heute im Trockenlauf
> bestätigt.

### PINN-Achsen

| Achse | Stand |
|---|---|
| 0 `w_phys` 0.1 / 0 | ✅ 10.09., P2 — `[NOT SEPARATED]`. **Nullmessung 5.248 ± 0.518 °C** (Mittel OP06 + OP09) |
| 1 δ 1.0 / 0.4 / 0.2 | ✅ 15.09., P2 — `[NOT SEPARATED]`, bestes 4.868 ± 0.650. **Seit 23.09. anders gedeutet** (§3) |
| 2 O16 Gehäusewand | ⏸ hängt an der Zerlegung — per autograd wäre ein Robin-Term blind |
| 3 `w_phys`/`w_bc`-Gitter | ⏸ hängt an der Zerlegung |
| 4 O17 Fixpunkt (`evaluate.py`) | ⬜ nicht angefangen, braucht keine GPU |
| **5 `--phys-stencil live`** | ⬜ **neu, startbereit** — §4 Schritt 2 |

> **PINN und GridCNN sind noch nicht vergleichbar.** Die GridCNN-POC-Zahlen
> sind bei dt = 1 s, die PINN-Zahlen bei dt = 0.2 s, und GridCNN berichtet je
> OP, das PINN das Mittel. Das ist Stufe 6 — derselbe Split, dieselben Metriken.

---

## 3 · Was diese Sitzung gefunden hat — und wie es läuft

### Vier Befunde am PINN (Einzelheiten: PINN-FAHRPLAN §11.10)

1. **O18 erklärt die Achse-1-Signatur nicht.** `L_phys ∝ 1/δ²` kann nur aus dem
   BDF-Zähler kommen; ein räumlicher Term wird addiert, nie durch δ geteilt. Der
   Fit `A + B/δ²` an die echten Achse-1-Zahlen trifft auf 5 %, der 1/δ²-Teil ist
   **91 / 98 / 99.6 %**. Für O18 bleiben höchstens 9 %. Und „`∇λ` analytisch" ist
   an den 363 Knoten, an denen das Residuum ausgewertet wird, **null**.
2. **O21 — der Zähler trägt einen Sprung.** `T(t)` ist das lebende Netz,
   `T(t−δ)`, `T(t−2δ)` kommen aus dem zu Epochenbeginn eingefrorenen Rollout.
   Jeder Datenschritt zieht das Netz zum Label, der Puffer bleibt — der Sprung
   wird durch δ geteilt. Aus den echten Zahlen: **3.65 °C RMS**.
3. **O22 — der autograd-Laplace ist blind.** Das Netz trägt seine Ortsstruktur
   über den Historien-Anker `T(t − 0.2 s)`, und den behandelt autograd als
   Konstante. Synthetisch sah autograd **0.01 … 9 %** der Krümmung. Dann ist der
   Leitungsterm fast null, `L_phys ≈ dT/dt − Qsrc`, und **keine** Änderung *im*
   Leitungsterm (O18, ein Robin-Term für O16) kann wirken.
4. **O20 — `--time-deriv autograd` trainierte ein zweites Netz**, das der Rollout
   nie benutzt. Jetzt gesperrt.

Dazu **O19**: die unvollständige Quelle (`tot/jr1 ≈ 3.2`). In den
GridCNN-Dokumenten heißt sie „O17" — im PINN-Index ist O17 aber der
Rollout-Fixpunkt. `total_w` liegt im Cache (`q_source[:, 2]`,
`data.py:421` liest nur Spalte 0), O19 ist also messbar.

**Synthetisch nachgestellt** (CPU, drei Läufe, nur δ verschieden): geloggtes
`L_phys` **5.2× / 25.8×** bei δ = 0.4 / 0.2 (echt: 5.5× / 22×). Zerlegung:
`[STALE]` 2 von 3, `[BLIND]` 3 von 3. **Mit `--phys-stencil live`**, sonst
gleich: `L_phys` **32 … 2 400× kleiner**, zwischen δ = 1.0 und 0.4 keine
δ-Skalierung mehr (bei δ = 0.2, einer Datenzeile, bleibt der raue Rollout
sichtbar); val OP06 **2.4 / 2.1 / 2.9 °C** statt 10.4 / 7.6 / 6.3 °C. `[BLIND]`
bleibt — der Schalter behebt O21, nicht O22. **Ein Seed, synthetisch:
Mechanismus belegt, keine Ergebnisse.**

### Wie es läuft: **6 von 10**

| | Note | warum |
|---|---|---|
| Methodik | 9 | Streuung vor Vergleich, widerlegte Hypothesen bleiben stehen. Diesmal hat sie einen eigenen Schluss (O18) eingefangen, bevor gebaut wurde |
| Werkzeug & Tests | 9 | PINN **163** Tests (+16 aus diesem PR, +12 für Schema v3 aus `main`); die Aussagen gegen O18 sind in der CI festgenagelt, nicht nur aufgeschrieben |
| PINN-Ergebnis | 3 | keine neue Zahl auf echten Daten; der Physik-Term trägt weiter nichts |
| GridCNN-Ergebnis | **4** | der POC ist lesbar und unter der Latte (0.61× / 0.75×); auf voller Auflösung noch kein Ergebnis |
| Vorankommen | 5 | ein falscher Plan ist gestoppt, der nächste PINN-Schritt kostet 40 min statt eines Tages Bauen, die erste Reparatur ist schaltfertig. Aber das PINN hatte wieder eine Sitzung ohne echte Messung |
| Dokumentation | 8 | mit `README_MODELLSTAND.md` ist jede Zahl einer Version zugeordnet |

**Die eine Sache, die sich ändern muss:** die nächste Sitzung **an der T4**
macht §4 Schritt 1 und 2 in einem Zug und schreibt erst danach auf.

---

## 4 · Was du als Nächstes tust — in dieser Reihenfolge

> Vor jedem Block: **§1** (venv, MPS aktiv, Karte frei).

### Schritt 1 — die 15 PINN-Checkpoints zerlegen · T4 + MPS, `-j 4`, ~40 min (geschätzt)

```bash
ls artifacts/*/*/model.pt               # 6 aus Achse 0, 9 aus Achse 1 -- Pfade pruefen
nohup python PINNmodulusTwo/tools/residual_decomposition.py \
    artifacts/achse0/*/model.pt artifacts/achse1/*/model.pt \
    -j 4 --device cuda > zerlegung.log 2>&1 &
tail -f zerlegung.log
```

Am Ende steht **eine Tabelle, eine Zeile je Checkpoint**, mit `[STALE]`,
`[JITTER]`, `[BLIND]`, `[NO 1/delta^2]`. Je Checkpoint liegen
`residual_decomposition.txt` (voller Bericht) und `.json` daneben.

> Der Cache ist seit 22.09. auf Schema v3. `T` und `q_source` sind gleich
> gebaut, die Normierung sollte also passen. **Verweigert** das Werkzeug mit
> „normalisation mismatch", ist das selbst ein Befund: dann hat v3 die Daten
> geändert, und das gehört in `README_MODELLSTAND.md` Tabelle 1c.

### Schritt 2 — die Tabelle entscheidet

| Befund (Mehrheit der 15) | was du tust |
|---|---|
| **`[STALE]`** | **Achse 5 sofort starten** (unten), ~2 h |
| **`[BLIND]`** | Schritt 3 — Ortsableitungen per Differenzenstern. Eine Codesitzung |
| beides | Achse 5 laufen lassen, in der Zeit Schritt 3 bauen |
| `[JITTER]` allein | ein Rollout-Problem → Achse 4 (O17) vorziehen |
| nichts davon | O18 ist wieder offen; Ausgangspunkt ist `A` und die Aufteilung je x-Ebene im Bericht |

**Achse 5 — `--phys-stencil live`** (Modellversion **P3**, sobald eingeschaltet):

```bash
nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary phys-stencil live -j 3 \
    --out artifacts/achse5 --csv artifacts/achse5.csv \
    -- --epochs 60 --ema-decay 0.5 --delta-phys 0.2 --device cuda > achse5.log 2>&1 &
```

* **Der Vergleichsarm ist schon gerechnet:** Achse 1, δ = 0.2 (**4.868 ± 0.650**),
  dieselbe Konfiguration mit `buffer`. P2.1 ist mit `buffer` bitgleich zu P2, die
  drei Läufe gelten weiter. Dazu die Nullmessung **5.248 ± 0.518**.
* **Lesen:** zuerst `L_phys` gegen Achse 1 — bleibt es bei ~1e6, wirkt der
  Schalter nicht; fällt es um Größenordnungen, war der Sprung der Treiber. Dann
  die val-MAE mit Streuung: unter ~1 °C Abstand ist es **keine** Differenz.
  `analyse_history.py` statt letzter Zeile.
* **Vor dem Start** die Zeile „P3 = P2.1 + `--phys-stencil live`" in
  `README_MODELLSTAND.md` Tabelle 1a; **nach** dem Lauf die Experimentzeile in
  Tabelle 2 und die Stand-Zeile im PINN-FAHRPLAN.
* `-j 3` lässt einen Kern frei — für GridCNN-Lauf 17 (unten).

### Schritt 3 — nur bei `[BLIND]`: Ortsableitungen per Differenzenstern (Codesitzung, keine GPU)

Die Entscheidungen sind vorbelegt, damit die Sitzung baut statt diskutiert:

| Frage | Entscheidung | warum |
|---|---|---|
| Was ändert sich? | Nur **wo** der Physik-Term das MLP auswertet: für jeden gezogenen Zeitpunkt das MLP an **allen 363 Punkten** (jeder mit seiner Historie), dann `∇·(λ∇T)` per Stern auf dem 3 × 11 × 11-Gitter | das MLP bleibt, wie es ist |
| Wo? | neue Funktion in `PINNmodulusTwo/physics.py`, Schalter `--phys-space {autograd,fd}`, Default `autograd` | nichts ändert sich ungefragt |
| Operator | wie `GridCNN/physics.py`: 3-Punkt in y/z, nicht-äquidistant in x, Kreuzterm `λ_xy` auf JR1 — **nachgebaut, nicht importiert** | GridCNN bleibt unangetastet; beide Projekte haben ein Modul `physics`, ein Import kollidiert |
| Ränder | x = 0 gespiegelt (Symmetrie exakt), y/z-Ränder gespiegelt, Gehäusewand adiabat bis O16 | dieselben Annahmen wie GridCNN — sonst misst Stufe 6 Randbedingungen statt Architekturen |
| `L_bc` | bei `fd` überflüssig: die Spiegelung macht `dT/dx = 0` exakt. `w_bc` bleibt, wirkt aber nicht | so hält es GridCNN auch |
| Kosten | `B_t × 363` Vorwärtsläufe, **keine** Hesse-Matrix — eher billiger als heute | |
| Tests | Stern exakt auf Quadratik; Symmetrie exakt null; `fd` sieht, was autograd nicht sieht (`_ThroughAnchor`-Test umkehren) | wie `test_residual_decomposition.py` |
| Messen | Achse 6, 3 Seeds, gegen Nullmessung und Achse 5 | Latte ~1 °C |

Danach sind **O16** (Robin-Term) und **O18** (Leitwerte zwischen den Knoten,
GridCNN-Route R4) keine autograd-Fragen mehr, sondern Terme dieses Operators.

### Parallel, ohne GPU: PINN-Achse 4 / O17

`evaluate.py` gibt es nicht. Die Messung: **Teacher Forcing gegen freien
Rollout auf OP06**, auf den 15 Checkpoints. Trennt „kann den Einzelschritt
nicht" von „akkumuliert über 7000 Schritte".

### GridCNN — sein nächster Schritt (aus `UEBERGABE_2026-09-23.md`, hier nicht geändert)

```bash
cp -r GridCNN/artifacts/A GridCNN/artifacts/A_16_voll        # Gewichte von Lauf 16 sichern
python3 GridCNN/tools/nachmessen.py --no-physics --subsample 2 \
    --device cuda --cache data_cache --laeufe GridCNN/artifacts/A_16_voll \
    2>&1 | tee 16_konfigA_nachgemessen.txt                    # Minuten, kein Training
nohup python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
    --val-every 2 --device cuda --cache data_cache \
    > 17_konfigA_voll_k_sekunden.txt 2>&1 &                   # Lauf 17, ~1.5-2 h
```

Im Kopf des Logs darf **kein** `!! [fenster]` stehen, und `[protokoll]` muss
`k=20->80 (4->16 s)` zeigen. Offen dort (§5): warum der erste Lauf-16-Prozess
nach Seed 1 endete, und ob `material_properties/` echte Daten sind.

**Reihenfolge an einem Stück:** Schritt 1 (PINN, `-j 4`, ~40 min) → dann
**gleichzeitig** Achse 5 (`-j 3`) und GridCNN-Lauf 17 (1 Prozess) = 4
Prozesse, ~2 h, MPS an.

---

## 5 · So wird dokumentiert — damit der nächste Schritt immer klar ist

1. **Modell ändern** → zuerst eine Zeile in `README_MODELLSTAND.md` Tabelle 1
   (neue Version, Commit, „Verhalten geändert ja/nein", Test). Erst dann laufen
   lassen.
2. **Experiment** → eine Zeile in Tabelle 2: Datum, **Version**, nur die
   Abweichungen vom Default, Maschine (T4, MPS, `-j`), Ergebnis **mit Streuung**,
   Fundstelle.
3. **FAHRPLAN** → Stand-Tabelle, und der Kopf-Kasten sagt, was jetzt dran ist.
4. **Chronologisch**, älteste Zeile zuerst. Nichts löschen, was überholt ist —
   als überholt markieren (wie der O18-Kasten vom 15.09. im PINN-FAHRPLAN).
5. **Diese Datei** am Ende jeder Sitzung neu schreiben.

---

## 6 · Fallen

| | |
|---|---|
| **MPS aus** | Faktor 2.5, lautlos. §1 |
| **mehr als 4 Prozesse** | 4 physische Kerne — zwei Jobs teilen sich `-j 4` |
| **`--phys-stencil live` mit δ zwischen zwei Datenzeilen** | wird verweigert; δ muss ein Vielfaches von 0.2 s sein |
| **eine Zahl ohne Streuung** | Latte ~1 °C (0.518 / 0.882, auf OP06 bis 1.63) |
| **letzte Epoche ablesen** | Median über die letzten k Epochen |
| **synthetische Zahlen zitieren** | nur Mechanismen, nie Ergebnisse |
| **„O17" verwechseln** | PINN-O17 = Rollout-Fixpunkt; GridCNN-„O17" = PINN-**O19** (Quelle) |
| **PINN- und GridCNN-Zahlen nebeneinanderstellen** | verschiedene dt und Mittelung — erst Stufe 6 |
| **OP14 bei 0 °C „reparieren"** | geplant so (O10) |
| **OP19 als Auswahlkriterium** | nie (O11) |

---

## 7 · Was geprüft ist und was nicht

**In dieser Sitzung ausgeführt (Cloud, CPU, nach dem Merge von `main`):**
PINN-Tests **163 passed**, 1 skipped, 1 xfailed · GridCNN-Tests **133 passed** ·
`selftest.py` alle Checks · `benchmark.py --stage 0 2 --dry-run` 12 × OK ·
sechs synthetische Trainingsläufe (drei `buffer`, drei `live`) und die Zerlegung
auf allen sechs.

**Nicht geprüft:** alles auf echten Daten — hier liegen weder `data_cache` noch
die Checkpoints, und es gibt keine GPU. Die Zahlen aus Achse 0/1 und aus den
GridCNN-Läufen sind aus den Dokumenten übernommen. Die Laufzeit der Zerlegung
auf der T4 ist geschätzt, nicht gemessen. Dass Schema v3 `T` und `q_source`
unverändert lässt, ist am Code gelesen, nicht am Cache geprüft.
