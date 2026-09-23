# Nächste Sitzung — Stand 23.09.2026

> **Der eine Einstieg nach einer Pause, für beide Projekte.** Wo du stehst, was
> neu ist, womit du anfängst — und die Antworten auf die offenen Fragen.
>
> * 🆕 [`PINNmodulusTwo/README_MODELL_P3_POC.md`](PINNmodulusTwo/README_MODELL_P3_POC.md) — **das neue Modell P3 und sein POC: Priorität 1**
> * [`README_MODELLSTAND.md`](README_MODELLSTAND.md) — jede Modellversion und jedes Experiment, chronologisch, mit der Version, auf der es lief
> * [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) — ganz oben der POC, darunter die Zerlegung, **§11.10** die Herleitung
> * GridCNN: [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md) und [`UEBERGABE_2026-09-23.md`](UEBERGABE_2026-09-23.md) — **in dieser Sitzung nicht angefasst**

---

## 0 · Die Antworten zuerst

| Frage | Antwort |
|---|---|
| **Gibt es ein neues Modell?** | **Ja: P3.** P3 = das bisherige MLP (unverändert) mit **`--phys-stencil live`**: der Physik-Term nimmt `T(t−δ)`, `T(t−2δ)` aus dem **lebenden** Netz statt aus dem zu Epochenbeginn eingefrorenen Rollout. Es existiert als Schalter (PR #49), ist **nicht** Default und ist auf echten Daten **noch nie gelaufen**. Beschreibung, vorher/nachher: `README_MODELL_P3_POC.md` |
| **Ist das Modell jetzt besser?** | **Das eingesetzte nicht** — mit den Defaults rechnet der Code bitgleich wie vorher (P2.1 = P2, per Test). **P3** war auf dem synthetischen Cache besser, zuletzt mit genau dem POC-Kommando: val OP06 **3.330 ± 0.264** gegen **6.585 ± 0.179 °C**, 2 Seeds. Das belegt den Mechanismus, **nicht** das Modell. Auf echten Daten entscheidet es der **POC** |
| **Fahren wir mit diesem Modell fort?** | **Ja.** Das MLP bleibt (4 × 128, lernbares Swish, hybride Historie). Geändert wird nur, **wo der Physik-Term das MLP auswertet**: P3 (live-Stencil) jetzt im POC, P4 (Ortsableitungen per Differenzenstern) erst, wenn die Zerlegung `[BLIND]` auf echten Daten zeigt |
| **Wurde die Physik angepasst?** | **Ja, als Schalter — eingeschaltet wird sie im POC.** Der Default bleibt `buffer`, damit jede alte Zahl gültig bleibt und der POC einen sauberen Vergleichsarm hat. P3 wird erst Default, wenn POC **und** Achse 5 es tragen |
| **Was ist mit GridCNN?** | **Nicht angefasst.** Andere Sitzungen haben es am 22./23.09. weit gebracht (§2). Sein Lauf 17 läuft **parallel** zum PINN-POC (§4) |
| **Wo stand ich in den Benchmarks?** | §2 |
| **Wie läuft es, 1–10?** | **6/10** — §3 |

---

## 1 · Die Maschine — vor JEDEM Kommando in dieser Datei

Alles läuft auf **deiner g4dn.2xlarge**: Tesla **T4** (15.6 GiB, sm_75), 8 vCPU,
davon **4 physische Kerne**. Mehrere Läufe gleichzeitig trägt sie nur mit **MPS**.
Gemessen (10.09.): `-j 4` **ohne** MPS 1.46×, **mit** MPS **3.69×**. Ohne MPS
verschenkst du Faktor 2.5, und kein Log sagt es.

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
* **Das Budget sind 4 Prozesse** (4 physische Kerne; mit MPS ist die CPU die
  Grenze, nicht die Karte). **Parallel heißt `-j`** (`sweep.py`,
  `residual_decomposition.py`); ein einzelner `train.py` ist **ein** Prozess.
* **Zwei Jobs teilen sich die 4:** der PINN-POC mit `-j 3` **plus**
  GridCNN-Lauf 17 als ein Prozess. 6 POC-Läufe sind bei `-j 3` wie bei `-j 4`
  zwei Wellen — der freie Kern kostet nichts.
* **Planungszahl PINN:** ~2 h je 60-Epochen-Lauf auf 11 OPs bei dt 0.2 s;
  Wanduhr ≈ `ceil(Läufe/j)` × Laufdauer.

---

## 2 · Wo du stehst

### Die Versionen (Einzelheiten und Commits: `README_MODELLSTAND.md`)

| | Version | seit | Stand |
|---|---|---|---|
| PINN | **P2** | 01.09. | Quelle korrigiert — **alle** echten Läufe (Schritt 5b, 6, Achse 0, 1) |
| PINN | **P2.1** | **23.09.** | im Code: Zerlegungswerkzeug, Schalter `--phys-stencil`, `--time-deriv autograd` gesperrt — Default **bitgleich** zu P2 |
| PINN | 🆕 **P3** | **23.09.** | P2.1 + `--phys-stencil live` — **wartet auf seinen POC** |
| PINN | *P4* | — | Ortsableitungen per Differenzenstern — nur falls `[BLIND]` |
| GridCNN | **G4.1** | 22./23.09. | Ladepfad angeschlossen, Stufe 5 (TBPTT) gebaut, Fehlerprofil + `nachmessen.py` |
| Cache | **Schema v3** | 22.09. | Wandpfad dazu; `T` und `q_source` gleich gebaut |

### PINN-Achsen

| Achse | Stand |
|---|---|
| 0 `w_phys` 0.1 / 0 | ✅ 10.09., P2 — `[NOT SEPARATED]`. **Nullmessung 5.248 ± 0.518 °C** (Mittel OP06 + OP09) |
| 1 δ 1.0 / 0.4 / 0.2 | ✅ 15.09., P2 — `[NOT SEPARATED]`, bestes 4.868 ± 0.650. **Seit 23.09. anders gedeutet** (§3) |
| 🆕 **POC P3** | ⬜ **Priorität 1** — `buffer` gegen `live`, dt 1 s, 3 Seeds (§4 Schritt 1) |
| 5 P3 auf voller Auflösung | ⬜ nur bei 🟢 im POC |
| 2 O16 Gehäusewand | ⏸ hängt an der Zerlegung — per autograd wäre ein Robin-Term blind |
| 3 `w_phys`/`w_bc`-Gitter | ⏸ hängt an der Zerlegung |
| 4 O17 Fixpunkt (`evaluate.py`) | ⬜ nicht angefangen, braucht keine GPU |

### GridCNN — die Leiter hat sich bewegt (nur berichtet, nicht geändert)

| Stufe | Stand |
|---|---|
| 0 Rangtest | ✅ rot, überstimmt (16 × 3) |
| 1 Bilanz | ✅ **22.09.** — Fluidbilanz 🟢, `U(V̇)` kalibriert, null freie Parameter |
| 2 Cache | ✅ **22.09.** — Schema v3 (`8d76084`) |
| 3 Löser | gebaut, adiabat, nicht als Latte gemessen |
| 4 CNN | **trainiert**: Lauf 14 (kein Ergebnis) → **Lauf 15 POC**, dt 1 s: **OP06 6.571 ± 0.383 (0.61×), OP09 5.813 ± 0.867 (0.75×)** — erstmals lesbar |
| 5 TBPTT | ✅ gebaut 22.09.; Lauf 16 (volle Auflösung): 0.62× / 1.11×, Streuung 1.5–1.8 °C → **kein Ergebnis**, `k` stand in Schritten und erreichte `lag2` nie |
| 6 Vergleich mit dem PINN | offen |

> Die Leitertabelle *in* `GridCNN/FAHRPLAN.md` ist an Stufe 2, 4, 5 nicht
> nachgezogen — hier steht der Stand aus Commits und Berichten.
> `benchmark.py` (die andere Leiter, ⚠ auch „Stufe"): letzter Lauf 14.09.,
> Stufe 0 und 2 🟢, Stufe 1 und 3 nie gelaufen.
>
> **PINN und GridCNN sind noch nicht vergleichbar** — verschiedene dt, GridCNN
> je OP, PINN gemittelt. Das ist Stufe 6.

---

## 3 · Was diese Sitzung gefunden hat — und wie es läuft

### Vier Befunde am PINN (Herleitung: PINN-FAHRPLAN §11.10)

1. **O18 erklärt die Achse-1-Signatur nicht.** `L_phys ∝ 1/δ²` kann nur aus dem
   BDF-Zähler kommen; ein räumlicher Term wird addiert, nie durch δ geteilt. Der
   Fit `A + B/δ²` an die echten Zahlen trifft auf 5 %, der 1/δ²-Teil ist
   **91 / 98 / 99.6 %**. Für O18 bleiben höchstens 9 %. „`∇λ` analytisch" ist an
   den 363 Knoten **null**. **O18 ist ausgesetzt.**
2. **O21 — der Zähler trägt einen Sprung** zwischen lebendem Netz und
   eingefrorenem Rollout, aus den echten Zahlen **3.65 °C RMS**. → **P3 behebt
   genau das.**
3. **O22 — der autograd-Laplace ist blind** für die Ortsstruktur, die über den
   Historien-Anker kommt (synthetisch: **0.01 … 9 %** der Krümmung gesehen).
   Dann ist `L_phys ≈ dT/dt − Qsrc`. → P3 behebt das **nicht**; dafür wäre P4.
4. **O20 — `--time-deriv autograd`** trainierte ein zweites Netz, das der
   Rollout nie benutzt. Jetzt gesperrt.

Dazu **O19**: die unvollständige Quelle (`tot/jr1 ≈ 3.2`) — in den
GridCNN-Dokumenten „O17", im PINN-Index O19 (PINN-O17 ist der Rollout-Fixpunkt).

### P3 auf dem synthetischen Cache — was schon belegt ist

| | `buffer` (P2) | `live` (P3) |
|---|---|---|
| dt 0.2 s, 1 Seed, δ = 1.0 / 0.4 / 0.2 — geloggtes `L_phys` | 1.55e4 / 8.07e4 / 4.0e5 | **129 / 33 / 1.24e4** |
| dieselben Läufe — val OP06 | 10.4 / 7.6 / 6.3 °C | **2.4 / 2.1 / 2.9 °C** |
| **das POC-Kommando**, dt 1 s, 2 Seeds, 12 Epochen — val OP06 | 6.585 ± 0.179 °C | **3.330 ± 0.264 °C** |
| dasselbe — `L_phys` | 1.2e4 … 3.0e4 | **~400** |
| Zerlegung | `[BLIND]` überall | `[BLIND]` bleibt |

**Synthetisch: Mechanismus belegt, keine Ergebnisse.** Dazu 16 Tests, u. a.: der
Schalter ändert auf einem frischen Rollout nichts, und er hebt einen
Gewichtsversatz exakt auf.

### Wie es läuft: **6 von 10**

| | Note | warum |
|---|---|---|
| Methodik | 9 | Streuung vor Vergleich, widerlegte Hypothesen bleiben stehen; diesmal hat sie einen eigenen Schluss (O18) eingefangen, bevor gebaut wurde |
| Werkzeug & Tests | 9 | PINN **163** Tests (+16 aus PR #49, +12 für Schema v3 aus `main`); die Aussagen gegen O18 sind in der CI festgenagelt |
| PINN-Ergebnis | 3 | noch keine neue Zahl auf echten Daten — der POC ist der erste Schritt dahin |
| GridCNN-Ergebnis | 4 | der POC ist lesbar und unter der Latte (0.61× / 0.75×); auf voller Auflösung noch kein Ergebnis |
| Vorankommen | 5 | ein falscher Plan ist gestoppt, das neue Modell P3 ist gebaut, getestet und hat einen fertigen POC. Aber das PINN hatte wieder eine Sitzung ohne echte Messung |
| Dokumentation | 8 | jede Zahl ist einer Version zugeordnet; das neue Modell hat eine eigene Beschreibung |

**Die eine Sache, die sich ändern muss:** die nächste Sitzung **an der T4**
startet den POC, bevor sie irgendetwas anderes tut.

---

## 4 · Was du als Nächstes tust — in dieser Reihenfolge

> Vor jedem Block: **§1** (venv, MPS aktiv, Karte frei).

### Schritt 1 — 🆕 POC P3 · T4 + MPS, `-j 3`, ~1–1.5 h (geschätzt) · **Priorität 1**

Vollständig beschrieben in
[`PINNmodulusTwo/README_MODELL_P3_POC.md`](PINNmodulusTwo/README_MODELL_P3_POC.md).

```bash
nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary phys-stencil buffer live -j 3 \
    --out artifacts/poc_p3 --csv artifacts/poc_p3.csv \
    -- --subsample 10 --delta-grid 1.0 --delta-phys 1.0 \
       --epochs 40 --ema-decay 0.5 --device cuda > poc_p3.log 2>&1 &
```

* Wie Achse 1 (Physik und BC an, `--ema-decay 0.5`), nur dt 1 s und 40 Epochen;
  zwei Arme, nur der Schalter verschieden.
* Jedes `train.log` des `live`-Arms muss `physics stencil: LIVE -- ... lag = 1
  rows (O21)` zeigen, sonst ist der Code nicht gepullt.
* **Gleichzeitig** auf dem vierten Kern: GridCNN-Lauf 17 (unten).

**Urteil** (Einzelheiten `README_MODELL_P3_POC.md` §3):

| | Folge |
|---|---|
| 🟢 `L_phys` von `live` ≥ 10× kleiner **und** `live` vorn um ≥ ~1 °C über der Seed-Streuung | Achse 5: P3 auf voller Auflösung |
| 🟡 `[NOT SEPARATED]` | P3 nicht Default; weiter mit Schritt 2 und dem `[BLIND]`-Weg |
| 🔴 `buffer` vorn, oder der Rollout läuft weg (`[SATURATED]`) | P3 verworfen, mit Begründung stehen lassen |

**Danach schreiben:** `TRAININGS_BERICHT_<datum>_PINN_P3_POC.md`, die Zeilen
P3 und „POC P3" in `README_MODELLSTAND.md`, die Stand-Tabelle im PINN-FAHRPLAN.

### Schritt 2 — die Checkpoints zerlegen · T4 + MPS, `-j 4`, keine Trainingszeit

```bash
ls artifacts/*/*/model.pt               # poc_p3 (6), achse0 (6), achse1 (9) -- Pfade pruefen
nohup python PINNmodulusTwo/tools/residual_decomposition.py \
    artifacts/poc_p3/*/model.pt artifacts/achse0/*/model.pt artifacts/achse1/*/model.pt \
    -j 4 --device cuda > zerlegung.log 2>&1 &
```

Am Ende eine Tafel, eine Zeile je Checkpoint: `[BLIND]`, `[STALE]`, `[JITTER]`,
`[NO 1/delta^2]`. Die POC-Checkpoints gehen schnell (dt 1 s), die 15 aus Achse
0/1 geschätzt ~40 min. ⚠ Auch `live`-Checkpoints zeigen `[STALE]` — das
Werkzeug misst, was der **alte** Stencil loggen würde. Verweigert es mit
„normalisation mismatch", hat Schema v3 die Daten geändert: Befund, gehört in
`README_MODELLSTAND.md` Tabelle 1c.

### Schritt 3 — was folgt

| Lage nach Schritt 1 + 2 | was du tust |
|---|---|
| POC 🟢 | **Achse 5** (unten), ~2 h |
| `[BLIND]` auf den echten Checkpoints | **Schritt 4** — P4 bauen. Eine Codesitzung |
| POC 🟢 **und** `[BLIND]` | Achse 5 laufen lassen, in der Zeit P4 bauen |
| POC 🟡/🔴, kein `[BLIND]` | O18 wieder offen; Ausgangspunkt `A` und die x-Ebenen im Zerlegungsbericht |

**Achse 5 — P3 auf voller Auflösung:**

```bash
nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary phys-stencil live -j 3 \
    --out artifacts/achse5 --csv artifacts/achse5.csv \
    -- --epochs 60 --ema-decay 0.5 --delta-phys 0.2 --device cuda > achse5.log 2>&1 &
```

Vergleichsarm schon gerechnet: **Achse 1, δ = 0.2, 4.868 ± 0.650** (P2.1 mit
`buffer` = P2 bitgleich), dazu die Nullmessung **5.248 ± 0.518**. Erst wenn
Achse 5 trägt, wird `phys_stencil: live` Default — mit eigener Zeile im
Modellprotokoll.

### Schritt 4 — nur bei `[BLIND]`: P4, Ortsableitungen per Differenzenstern (Codesitzung, keine GPU)

Die Entscheidungen sind vorbelegt, damit die Sitzung baut statt diskutiert:

| Frage | Entscheidung | warum |
|---|---|---|
| Was ändert sich? | Nur **wo** der Physik-Term das MLP auswertet: je gezogenem Zeitpunkt das MLP an **allen 363 Punkten** (jeder mit seiner Historie), dann `∇·(λ∇T)` per Stern auf dem 3 × 11 × 11-Gitter | das MLP bleibt, wie es ist |
| Wo? | neue Funktion in `PINNmodulusTwo/physics.py`, Schalter `--phys-space {autograd,fd}`, Default `autograd` | nichts ändert sich ungefragt |
| Operator | wie `GridCNN/physics.py`: 3-Punkt in y/z, nicht-äquidistant in x, Kreuzterm `λ_xy` auf JR1 — **nachgebaut, nicht importiert** | GridCNN bleibt unangetastet; beide Projekte haben ein Modul `physics` |
| Ränder | x = 0 gespiegelt, y/z-Ränder gespiegelt, Gehäusewand adiabat bis O16 | dieselben Annahmen wie GridCNN — sonst misst Stufe 6 Randbedingungen statt Architekturen |
| `L_bc` | bei `fd` überflüssig: die Spiegelung macht `dT/dx = 0` exakt | so hält es GridCNN auch |
| Kosten | `B_t × 363` Vorwärtsläufe, **keine** Hesse-Matrix | eher billiger als heute |
| Tests | Stern exakt auf Quadratik; Symmetrie exakt null; `fd` sieht, was autograd nicht sieht (`_ThroughAnchor`-Test umkehren) | wie `test_residual_decomposition.py` |
| Messen | Achse 6, 3 Seeds, gegen Nullmessung und Achse 5 — vorher ein POC wie für P3 | Latte ~1 °C |

Danach sind O16 (Robin-Term) und O18 (Leitwerte zwischen den Knoten,
GridCNN-Route R4) Terme dieses Operators statt autograd-Fragen.

### Parallel, ohne GPU: PINN-Achse 4 / O17

`evaluate.py` gibt es nicht. Die Messung: **Teacher Forcing gegen freien
Rollout auf OP06**, auf den vorhandenen Checkpoints.

### GridCNN — sein nächster Schritt (aus `UEBERGABE_2026-09-23.md`, hier nicht geändert)

```bash
cp -r GridCNN/artifacts/A GridCNN/artifacts/A_16_voll        # Gewichte von Lauf 16 sichern
python3 GridCNN/tools/nachmessen.py --no-physics --subsample 2 \
    --device cuda --cache data_cache --laeufe GridCNN/artifacts/A_16_voll \
    2>&1 | tee 16_konfigA_nachgemessen.txt                    # Minuten, kein Training
nohup python3 GridCNN/train.py --no-physics --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
    --val-every 2 --device cuda --cache data_cache \
    > 17_konfigA_voll_k_sekunden.txt 2>&1 &                   # Lauf 17, ~1.5-2 h, der 4. Prozess
```

Im Log-Kopf **kein** `!! [fenster]`, und `[protokoll]` zeigt
`k=20->80 (4->16 s)`. Offen dort (§5): warum der erste Lauf-16-Prozess nach
Seed 1 endete, und ob `material_properties/` echte Daten sind.

### Die Sitzung an einem Stück

1. §1, dann GridCNN `nachmessen.py` (Minuten).
2. **POC P3 (`-j 3`) und GridCNN-Lauf 17 gleichzeitig starten** — 4 Prozesse, MPS an, ~1.5–2 h.
3. Zerlegung (`-j 4`), sobald beide fertig sind.
4. Auswerten, Berichte schreiben, `README_MODELLSTAND.md`, Fahrpläne, diese Datei neu.

---

## 5 · So wird dokumentiert — damit der nächste Schritt immer klar ist

1. **Modell ändern** → zuerst eine Zeile in `README_MODELLSTAND.md` Tabelle 1
   (Version, Commit, „Verhalten geändert ja/nein", Test). Erst dann laufen lassen.
   Ein neues Modell bekommt zusätzlich eine eigene Beschreibung wie
   `README_MODELL_P3_POC.md`, und der Fahrplan verweist ganz oben darauf.
2. **Experiment** → eine Zeile in Tabelle 2: Datum, **Version**, nur die
   Abweichungen vom Default, Maschine (T4, MPS, `-j`), Ergebnis **mit Streuung**,
   Fundstelle. Ein Lauf mit Urteil bekommt einen `TRAININGS_BERICHT_*.md`.
3. **Neues Modell → erst ein POC** (grob, kurz, 3 Seeds, gegen den Vorgänger),
   dann der volle Lauf.
4. **FAHRPLAN** → Stand-Tabelle, und der Kopf-Kasten sagt, was jetzt dran ist.
5. **Chronologisch**, älteste Zeile zuerst. Nichts löschen, was überholt ist —
   als überholt markieren (wie der O18-Kasten vom 15.09. im PINN-FAHRPLAN).
6. **Diese Datei** am Ende jeder Sitzung neu schreiben.

---

## 6 · Fallen

| | |
|---|---|
| **MPS aus** | Faktor 2.5, lautlos. §1 |
| **mehr als 4 Prozesse** | 4 physische Kerne — POC `-j 3` + Lauf 17 = 4 |
| **POC ohne `git pull`** | dann fehlt `--phys-stencil`; der `live`-Arm bricht ab oder die LIVE-Zeile fehlt |
| **`--phys-stencil live` mit δ zwischen zwei Datenzeilen** | wird verweigert; δ muss ein Vielfaches von dt sein (POC: 1 s, voll: 0.2 s) |
| **`--delta-grid 0.2` bei `--subsample 10`** | der Anker kann nicht feiner als dt; im POC steht deshalb 1.0 |
| **POC-Zahlen gegen Achse 1 stellen** | verschiedene dt — der POC vergleicht nur `buffer` gegen `live` in sich |
| **eine Zahl ohne Streuung** | Latte ~1 °C (0.518 / 0.882, auf OP06 bis 1.63) |
| **letzte Epoche ablesen** | Median über die letzten k Epochen (`analyse_history.py`) |
| **synthetische Zahlen zitieren** | nur Mechanismen, nie Ergebnisse |
| **„O17" verwechseln** | PINN-O17 = Rollout-Fixpunkt; GridCNN-„O17" = PINN-**O19** (Quelle) |
| **PINN- und GridCNN-Zahlen nebeneinanderstellen** | verschiedene dt und Mittelung — erst Stufe 6 |
| **OP14 bei 0 °C „reparieren"** / **OP19 als Auswahlkriterium** | nie (O10, O11) |

---

## 7 · Was geprüft ist und was nicht

**In dieser Sitzung ausgeführt (Cloud, CPU, nach dem Merge von `main`):**
PINN-Tests **163 passed**, 1 skipped, 1 xfailed · GridCNN-Tests **133 passed** ·
`selftest.py` alle Checks · `benchmark.py --stage 0 2 --dry-run` 12 × OK ·
sechs synthetische Trainingsläufe bei dt 0.2 s (drei `buffer`, drei `live`) ·
**das POC-Kommando selbst** über `sweep.py` auf dem synthetischen Cache
(4 Läufe, `-j 4`, alle `[ok]`) · die Zerlegung auf allen zehn Checkpoints,
auch über den Sweep-Pfad `artifacts/poc_p3/*/model.pt`.

**Nicht geprüft:** alles auf echten Daten — hier gibt es weder `data_cache`
noch Checkpoints noch GPU. Die Laufzeiten auf der T4 sind geschätzt. Dass
Schema v3 `T` und `q_source` unverändert lässt, ist am Code gelesen, nicht am
Cache geprüft.
