# Nächste Sitzung — Stand 23.09.2026

> **Der Einstieg nach einer Pause.** Eine Seite: wo du stehst, wie es läuft,
> womit du anfängst — und die Antworten auf die Fragen der letzten Sitzung, damit
> keine offen bleibt. Die Wahrheit steht weiter in den lebenden Dokumenten; diese
> Seite verweist dorthin:
>
> * [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) — Kopf-Kasten und **§11.10** sind neu
> * [`README_MODELLSTAND.md`](README_MODELLSTAND.md) — **neu:** Modellversionen und alle Experimente, chronologisch
> * [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md), [`GridCNN/BENCHMARK.md`](GridCNN/BENCHMARK.md) — **unverändert** in dieser Sitzung

---

## 0 · Die Antworten zuerst

| Frage | Antwort |
|---|---|
| **Ist das Modell jetzt besser?** | **Nein — das eingesetzte Modell rechnet bitgleich wie vorher** (P2.1 mit Defaults = P2, per Test belegt). Besser ist die **Diagnose**: der Plan „O18 einbauen" hätte an der falschen Stelle repariert. **Mit dem neuen Schalter** `--phys-stencil live` war die val-MAE auf dem **synthetischen** Cache in allen drei Paaren niedriger (2.1–2.9 statt 6.3–10.4 °C) — ein Seed, synthetisch: ein Grund, Achse 5 zu fahren, kein Beleg. Ob es auf echten Daten besser ist, sagt erst Achse 5 |
| **Fahren wir mit diesem Modell fort?** | **Ja.** Das MLP bleibt, wie es ist (4 × 128, lernbares Swish, hybride Historie). Geändert wird als Nächstes **wo der Physik-Term das MLP auswertet**, nicht das MLP — und erst, nachdem die Messung auf den echten Checkpoints sagt, welche Reparatur passt |
| **Wurde die Physik angepasst? Sie ist doch kaputt.** | **Die Rechnung nicht — mit Absicht.** Kaputt ist sie auf dem synthetischen Cache nachgewiesen, auf echten Daten noch nicht. **Gebaut ist die erste Reparatur als Schalter:** `--phys-stencil live` (Default `buffer` = wie bisher). Die zweite (`[BLIND]`) ist entworfen, mit allen Entscheidungen vorbelegt (§4, Schritt 3), und wird gebaut, wenn die Messung sie verlangt. Ohne Messung einzuschalten hieße, denselben Fehler wie bei O18 ein zweites Mal zu machen |
| **Was ist mit GridCNN?** | **Nicht angefasst**, weder Code noch Doku. Eine Abhängigkeit gibt es: der Cache-Umbau aus GridCNN-Stufe 2 muss **nach** der PINN-Messung aus §4 Schritt 1 kommen |
| **Wo stand ich in den Benchmarks?** | §2 |
| **Wie läuft es, 1–10?** | **6/10** — §3 |

---

## 1 · Die Maschine — vor JEDEM Kommando in dieser Datei

Alles unten läuft auf **deiner g4dn.2xlarge**: Tesla **T4** (15.6 GiB, sm_75),
8 vCPU, davon **4 physische Kerne**. Sie trägt mehrere Läufe gleichzeitig —
aber nur mit **MPS**. Gemessen (10.09.): 4 Läufe `-j 4` **ohne** MPS 1.46×,
**mit** MPS **3.69×**. Ohne MPS verschenkst du Faktor 2.5, und kein Log sagt es.

```bash
cd ~/llmtraining                        # der Linux-Rechner, nicht /mnt/c/...
git checkout main && git pull
source modulus_env/bin/activate         # danach `python`, nicht `python3` (FAHRPLAN §11.8)
nvidia-cuda-mps-control -d              # nach JEDEM Neustart der Maschine
pgrep -x nvidia-cuda-mps && echo "MPS laeuft"      # nicht "...-control": comm hat 15 Zeichen
pgrep -af "sweep.py|PINNmodulusTwo/train.py" || echo "Karte frei"
```

* **Parallel heißt `-j`.** `sweep.py -j 4` und seit heute auch
  `residual_decomposition.py -j 4`. Ein einzelner `train.py` ist **ein** Prozess —
  die ~7000 Rollout-Schritte hängen voneinander ab.
* **`-j 4`** ist die Vorgabe (CPU ist die Grenze, nicht die Karte). Bei 9 Läufen
  ist `-j 3` gleich schnell, bei 3 Läufen `-j 3`.
* **Planungszahl:** ~2 h je 60-Epochen-Lauf auf 11 OPs; Wanduhr ≈ `ceil(Läufe/j)` × 2 h.
* Beide Werkzeuge warnen selbst, wenn MPS bei `-j > 1` fehlt, und laufen trotzdem
  weiter.

---

## 2 · Wo du stehst

### Die Modelle (chronologisch; Details in `README_MODELLSTAND.md`)

| Version | seit | was | Verhalten |
|---|---|---|---|
| PINN **P1** | bis 31.08. | Quelle 121× zu klein | — |
| PINN **P2** | 01.09. | Quelle korrigiert — **alle** echten Läufe (Schritt 5b, 6, Achse 0, Achse 1) | geändert |
| PINN **P2.1** | **23.09.** | `return_parts`, `--phys-stencil live` (aus), `--time-deriv autograd` gesperrt, Zerlegungswerkzeug | **unverändert** |
| GridCNN **G2.1** | 22.09. | CNN 16 × 3, Arm D gebaut — **nie auf echten Daten trainiert** | — |

### Die Benchmarks — ⚠ zwei Leitern heißen „Stufe"

**GridCNN `benchmark.py` (Protokoll in `BENCHMARK.md`)** — letzter echter Lauf **14.09.**

| Stufe | Stand |
|---|---|
| 0 `workflow` | 🟢 8 × ok (heute im Trockenlauf bestätigt) |
| 1 `grid` | ⏸ **nie gelaufen** — braucht `data_cache` |
| 2 `stencil` | 🟢 4 × ok (heute bestätigt) |
| 3 `solver` | ⏸ **nie gelaufen** — braucht `data_cache` |

**GridCNN-Projektleiter (`GridCNN/FAHRPLAN.md`)** — du stehst auf **Stufe 1**:
zweiter Bilanzlauf am 22.09., Abschnitt 4 (`U(V̇)`) abgestürzt, Werkzeug
repariert, **Wiederholungslauf offen**. Stufe 2 (Cache-Umbau) nicht angefangen.
Stufe 3/4 gebaut, nie gemessen.

**PINN-Achsen (`PINNmodulusTwo/FAHRPLAN.md`)**

| Achse | Stand |
|---|---|
| 0 `w_phys` 0.1 / 0 | ✅ 10.09., P2 — `[NOT SEPARATED]`. **Nullmessung 5.248 ± 0.518 °C** |
| 1 δ 1.0 / 0.4 / 0.2 | ✅ 15.09., P2 — `[NOT SEPARATED]`, bestes 4.868 ± 0.650. **Seit 23.09. anders gedeutet** (§3) |
| 2 O16 Gehäusewand | ⏸ hängt an der Zerlegung (per autograd wäre ein Robin-Term blind) |
| 3 `w_phys`/`w_bc`-Gitter | ⏸ hängt an der Zerlegung |
| 4 O17 Fixpunkt (`evaluate.py`) | ⬜ nicht angefangen, braucht keine GPU |
| **5 `--phys-stencil live`** | ⬜ **neu, startbereit** — §4 Schritt 2 |

---

## 3 · Was diese Sitzung gefunden hat — und wie es läuft

### Vier Befunde am PINN (Einzelheiten: FAHRPLAN §11.10)

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

Dazu **O19**: die unvollständige Quelle (`tot/jr1 ≈ 3.2`) — im GridCNN-Fahrplan
heißt sie „O17", die Nummer ist im PINN-Index aber der Rollout-Fixpunkt. Und
`total_w` liegt **schon** im Cache (`q_source[:, 2]`, `data.py:421` liest nur
Spalte 0) — O19 ist heute messbar, ohne Rebuild.

**Synthetisch nachgestellt** (CPU, drei Läufe, nur δ verschieden): geloggtes
`L_phys` **5.2× / 25.8×** bei δ = 0.4 / 0.2 (echt: 5.5× / 22×). Zerlegung:
`[STALE]` 2 von 3, `[BLIND]` 3 von 3.

Mit `--phys-stencil live` (dieselben drei Läufe, nur der Schalter an): geloggtes `L_phys` **129 / 33 / 1.24e4** statt 1.55e4 / 8.07e4 / 4.0e5 — **32 … 2 400× kleiner**, und zwischen δ = 1.0 und 0.4 skaliert nichts mehr; bei δ = 0.2 (eine Datenzeile) bleibt der raue Rollout (`[JITTER]`) sichtbar. val OP06: **2.4 / 2.1 / 2.9 °C** mit `live` gegen 10.4 / 7.6 / 6.3 °C mit `buffer` — in allen drei Paaren besser, aber **ein Seed, synthetisch: ein Hinweis, der Achse 5 rechtfertigt, kein Ergebnis**. `[BLIND]` bleibt (Sichtbarkeit 0.02 … 0.03) — der Schalter behebt O21, nicht O22.

Mechanismus belegt, **Zahlen keine Ergebnisse**.

### Wie es läuft: **6 von 10** — unverändert, aber aus anderem Grund

| | 22.09. | 23.09. | warum |
|---|---|---|---|
| Methodik | 9 | 9 | Streuung vor Vergleich, widerlegte Hypothesen bleiben stehen. Diesmal hat sie eine eigene Schlussfolgerung (O18) eingefangen, bevor gebaut wurde |
| Werkzeug & Tests | 8 | **9** | PINN 135 → **151** Tests; die Aussagen gegen O18 sind jetzt in der CI festgenagelt, nicht nur aufgeschrieben |
| PINN-Ergebnis | 3 | 3 | keine neue Zahl auf echten Daten; der Physik-Term trägt weiter nichts |
| GridCNN-Ergebnis | 2 | 2 | unverändert, nie auf echten Daten |
| Vorankommen | 4 | **5** | ein falscher Plan ist gestoppt, der nächste Schritt kostet 40 min statt eines Tages Bauen, und die erste Reparatur ist schaltfertig. Aber: **wieder eine Sitzung ohne echte Messung** |
| Dokumentation | 7 | **8** | mit `README_MODELLSTAND.md` ist jede Zahl einer Modellversion zugeordnet |

**Die eine Sache, die sich ändern muss:** die nächste Sitzung **an der T4**
macht §4 Schritt 1 und 2 in einem Zug und schreibt erst danach auf.

---

## 4 · Was du als Nächstes tust — in dieser Reihenfolge

> Vor jedem Block: **§1** (venv, MPS, Karte frei).

### Schritt 1 — die 15 Checkpoints zerlegen · T4 + MPS, ~40 min (geschätzt), keine Trainingszeit

**Vor** dem GridCNN-Cache-Umbau — das Werkzeug rechnet die Normierung aus dem
Cache nach und verweigert, wenn er sich geändert hat.

```bash
ls artifacts/*/*/model.pt               # 6 aus Achse 0, 9 aus Achse 1 -- Pfade pruefen
nohup python PINNmodulusTwo/tools/residual_decomposition.py \
    artifacts/achse0/*/model.pt artifacts/achse1/*/model.pt \
    -j 4 --device cuda > zerlegung.log 2>&1 &
tail -f zerlegung.log
```

Am Ende steht **eine Tabelle, eine Zeile je Checkpoint**, mit den Befunden
`[STALE]`, `[JITTER]`, `[BLIND]`, `[NO 1/delta^2]`. Je Checkpoint liegen
`residual_decomposition.txt` (voller Bericht) und `.json` daneben.

### Schritt 2 — die Tabelle entscheidet

| Befund (Mehrheit der 15) | was du tust |
|---|---|
| **`[STALE]`** | **Achse 5 sofort starten** (unten). ~2 h |
| **`[BLIND]`** | Schritt 3 — der Physik-Term bekommt Ortsableitungen per Differenzenstern. Eine Codesitzung |
| beides | erst Achse 5 laufen lassen (läuft von allein), in der Zeit Schritt 3 bauen |
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
* **Vor dem Start** eine Zeile „P3 = P2.1 + `--phys-stencil live`" in
  `README_MODELLSTAND.md` Tabelle 1; **nach** dem Lauf die Experimentzeile in
  Tabelle 2 und die Stand-Zeile im FAHRPLAN.

### Schritt 3 — nur wenn `[BLIND]`: Ortsableitungen per Differenzenstern (Codesitzung, keine GPU)

Die Entscheidungen sind vorbelegt, damit die Sitzung baut statt diskutiert:

| Frage | Entscheidung | warum |
|---|---|---|
| Was ändert sich? | Nur **wo** der Physik-Term das MLP auswertet: für jeden gezogenen Zeitpunkt das MLP an **allen 363 Punkten** (jeder mit seiner Historie), dann `∇·(λ∇T)` per Stern auf dem 3 × 11 × 11-Gitter | das MLP bleibt, wie es ist |
| Wo? | neue Funktion in `PINNmodulusTwo/physics.py`, Schalter `--phys-space {autograd,fd}`, Default `autograd` | nichts ändert sich ungefragt |
| Operator | wie `GridCNN/physics.py`: 3-Punkt in y/z, nicht-äquidistant in x, Kreuzterm `λ_xy` auf JR1 — **nachgebaut, nicht importiert** | GridCNN bleibt unangetastet; beide Projekte haben ein Modul `physics`, ein Import kollidiert |
| Ränder | x = 0 gespiegelt (Symmetrie exakt), y/z-Ränder gespiegelt, Gehäusewand adiabat bis O16 | dieselben Annahmen wie GridCNN heute — sonst misst Stufe 6 Randbedingungen statt Architekturen |
| `L_bc` | bei `fd` überflüssig: die Spiegelung macht `dT/dx = 0` exakt. `w_bc` bleibt, wirkt aber nicht | so hält es GridCNN auch |
| Kosten | `B_t × 363` Vorwärtsläufe, **keine** Hesse-Matrix — eher billiger als heute | |
| Tests | Stern exakt auf Quadratik; Symmetrie exakt null; `fd` sieht, was autograd nicht sieht (`_ThroughAnchor`-Test umkehren) | wie `test_residual_decomposition.py` |
| Messen | Achse 6, 3 Seeds, gegen Nullmessung und Achse 5 | Latte ~1 °C |

Danach sind **O16** (Robin-Term) und **O18** (Leitwerte zwischen den Knoten,
GridCNN-Route R4) keine autograd-Fragen mehr, sondern Terme dieses Operators.

### Parallel, ohne GPU: Achse 4 / O17

`evaluate.py` gibt es nicht. Die Messung: **Teacher Forcing gegen freien
Rollout auf OP06**, auf den 15 Checkpoints. Trennt „kann den Einzelschritt nicht"
von „akkumuliert über 7000 Schritte".

### GridCNN — unverändert, nur die Reihenfolge

Sein eigener Plan gilt (`GridCNN/FAHRPLAN.md`, Kopf): `balance_check.py`
Abschnitt 4 nachholen (Minuten, nur numpy), dann Stufe 2. **Stufe 2 erst nach
Schritt 1.** Zwei Hinweise aus dieser Sitzung, dort nicht eingetragen:
`total_w` und `mdot` (`fluid_mass_flow`, kg/s) liegen schon im Cache — real
fehlen `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`; und `schema_version`
steht in `legacy/battery_surrogate_agenticWorkflow/build.yaml` (heute 2).

---

## 5 · So wird dokumentiert — damit der nächste Schritt immer klar ist

1. **Modell ändern** → zuerst eine Zeile in `README_MODELLSTAND.md` Tabelle 1
   (neue Version, Commit, „Verhalten geändert ja/nein", Test). Erst dann laufen
   lassen.
2. **Experiment** → eine Zeile in Tabelle 2: Datum, **Modellversion**, nur die
   Abweichungen vom Default, Maschine (T4, MPS, `-j`), Ergebnis **mit Streuung**,
   Fundstelle.
3. **FAHRPLAN** → Stand-Tabelle in Teil III, und der Kopf-Kasten sagt, was jetzt
   dran ist.
4. **Chronologisch**, älteste Zeile zuerst. Nichts löschen, was überholt ist —
   als überholt markieren (wie der O18-Kasten vom 15.09.).

---

## 6 · Fallen

| | |
|---|---|
| **MPS vergessen** | Faktor 2.5, lautlos. §1 |
| **Stufe 2 vor Schritt 1** | die Zerlegung verweigert danach die Checkpoints |
| **`--phys-stencil live` mit δ zwischen zwei Datenzeilen** | wird verweigert; δ muss ein Vielfaches von 0.2 s sein |
| **eine Zahl ohne Streuung** | Latte ~1 °C (0.518 / 0.882, auf OP06 bis 1.63) |
| **letzte Epoche ablesen** | Median über die letzten k Epochen |
| **synthetische Zahlen zitieren** | nur Mechanismen, nie Ergebnisse |
| **„O17" verwechseln** | PINN-O17 = Rollout-Fixpunkt; GridCNN-„O17" = PINN-**O19** (Quelle) |
| **OP14 bei 0 °C „reparieren"** | geplant so (O10) |
| **OP19 als Auswahlkriterium** | nie (O11) |

---

## 7 · Was geprüft ist und was nicht

**In dieser Sitzung ausgeführt (Cloud, CPU):** PINN-Tests **151 passed**,
1 skipped, 1 xfailed · GridCNN-Tests **107 passed** · `selftest.py` alle Checks ·
`benchmark.py --stage 0 2 --dry-run` 12 × OK · sechs synthetische Trainingsläufe
(drei `buffer`, drei `live`) und die Zerlegung darauf.

**Nicht geprüft:** alles auf echten Daten — hier liegen weder `data_cache` noch
die Checkpoints, und es gibt keine GPU. Die Zahlen aus Achse 0/1 und Stufe 1
sind aus den Dokumenten übernommen. Die Laufzeit der Zerlegung auf der T4 ist
geschätzt, nicht gemessen.
