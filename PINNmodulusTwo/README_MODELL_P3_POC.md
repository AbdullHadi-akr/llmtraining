# Modell P3 und sein POC — Stand 23.09.2026

> **Priorität 1 für die nächste Sitzung an der T4.** Diese Datei beschreibt das
> neue PINN-Modell **P3**, was sich gegenüber P2 ändert, und den **POC** — den
> kurzen Lauf auf echten Daten, der zeigt, ob die Änderung trägt, bevor ein
> voller Lauf Stunden kostet. Vorbild ist der POC von GridCNN (Lauf 15,
> `TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md`): kleinere Zeitauflösung,
> weniger Epochen, drei Seeds, und ein klares Ja/Nein.
>
> Verwiesen von: [`FAHRPLAN.md`](FAHRPLAN.md) (ganz oben) ·
> [`README_NAECHSTE_SITZUNG.md`](../README_NAECHSTE_SITZUNG.md) §4 ·
> [`README_MODELLSTAND.md`](../README_MODELLSTAND.md) Tabelle 1a

---

## 1 · Das neue Modell in einem Satz

**P3 = P2.1 mit `--phys-stencil live`.** Das MLP ist **dasselbe** (4 × 128,
lernbares Swish, Weight-Norm, hybride Historie `[T(t−0.2 s), Rate 5 s, Rate 20 s]`,
`residual_output: false`). Anders ist nur, **woher der Physik-Term die beiden
zurückliegenden Zeitpunkte seiner Zeitableitung nimmt.**

| | P2 (alle Läufe bis 23.09.) | P2.1 (Code seit 23.09., PR #49) | **P3** (der POC) |
|---|---|---|---|
| MLP, Historie, Datenterm | — | unverändert | **unverändert** |
| `T(t)` im Physik-Term | lebendes Netz | lebendes Netz | lebendes Netz |
| `T(t−δ)`, `T(t−2δ)` im Physik-Term | aus dem Rollout, der zu Epochenbeginn **eingefroren** wurde | wie P2 (Default `buffer`) | aus dem **lebenden Netz**, jeweils mit seiner Historie aus dem Puffer |
| Schalter | — | `--phys-stencil buffer` | **`--phys-stencil live`** |
| Verhalten gegenüber P2 | — | **bitgleich** (Test) | **geändert** |

### Warum das die Änderung ist, die man zuerst probiert

Achse 1 hat `L_phys ∝ 1/δ²` gemessen. Das kann nur aus dem Zähler der
BDF-Ableitung `(3T − 4T₁ + T₂)/(2δ)` kommen — ein Fehler in der Leitungsgleichung
(O18) wird addiert und nie durch δ geteilt. Der Fit an die echten Zahlen sagt:
**≥ 91 %** von `L_phys` ist ein δ-unabhängiger Sprung im Zähler, rund **3.65 °C
RMS**. Die naheliegende Quelle: `T` ist das lebende Netz, `T₁`/`T₂` sind
eingefroren, und jeder Datenschritt zieht das Netz zum Label, während der Puffer
bleibt (**O21**). Mit `live` bewegen sich alle drei Punkte gemeinsam; ein
gemeinsamer Versatz hebt sich auf (3 − 4 + 1 = 0).

Einzelheiten und alle Zahlen: [`FAHRPLAN.md` §11.10](FAHRPLAN.md).

### Was P3 **nicht** behebt

**O22 — der autograd-Laplace ist blind** für die Ortsstruktur, die über den
Historien-Anker kommt (synthetisch: 0.01 … 9 % der Krümmung gesehen). Dafür ist
P3 nicht gebaut. Das ist der zweite Kandidat (Ortsableitungen per
Differenzenstern, [`README_NAECHSTE_SITZUNG.md`](../README_NAECHSTE_SITZUNG.md)
§4 Schritt 3) und wird erst nach der Zerlegung auf echten Checkpoints gebaut.

---

## 2 · Was schon belegt ist — und was nicht

| | Stand |
|---|---|
| Der Schalter ändert mit Default nichts | ✅ Test, bitgleich |
| Auf einem frischen Rollout ist `live` = `buffer` | ✅ `test_live_stencil_equals_the_buffer_stencil_on_a_fresh_rollout` |
| Nach verschobenen Gewichten hebt `live` den Sprung exakt auf | ✅ `test_live_stencil_cancels_the_drift_the_buffer_stencil_amplifies` |
| δ zwischen zwei Datenzeilen wird verweigert | ✅ `test_live_stencil_refuses_a_lag_between_rows` |
| Synthetisch, dt = 0.2 s, 1 Seed, 15 Epochen | `L_phys` 32 … 2 400× kleiner; val OP06 2.4 / 2.1 / 2.9 °C gegen 10.4 / 7.6 / 6.3 °C (`buffer`) |
| **Synthetisch, das POC-Kommando unten** (dort nur 3 Trainings-OPs + OP06 und 12 Epochen), dt = 1 s, 2 Seeds, CPU | `L_phys` ~400 gegen 1.2e4 … 3.0e4; val OP06 **3.330 ± 0.264** (`live`) gegen **6.585 ± 0.179 °C** (`buffer`); Sweep, Checkpoints und Zerlegung laufen durch |
| **Auf echten Daten** | ❌ **nie gelaufen — das ist der POC** |

Die synthetischen Zahlen belegen den Mechanismus und dass die Kommandos
funktionieren. Sie sind **kein Ergebnis**: der synthetische Cache ist kein
Wärmeleitungsfeld, und seine Labels sind viel glatter als echte.

---

## 3 · Der POC

### Was er beantwortet

> **Senkt `--phys-stencil live` auf echten Daten `L_phys` um Größenordnungen —
> und wird die val-MAE dabei besser, gleich oder schlechter?**

### Die Konfiguration

Wie Achse 1 (Physik und BC an, `--ema-decay 0.5`, elf Trainings-OPs, val OP06 +
OP09), nur **gröber und kürzer**, damit er in rund einer Stunde steht:

| | Achse 1 (15.09.) | **POC P3** | warum |
|---|---|---|---|
| `--subsample` | 2 (dt 0.2 s) | **10 (dt 1 s)** | rund 7 000–8 000 → rund 1 500 Rollout-Schritte je OP, wie der GridCNN-POC |
| `--delta-grid` | 0.2 | **1.0** | der Anker kann nicht feiner als dt auflösen (`train.py` warnt sonst) |
| `--delta-phys` | 1.0 / 0.4 / 0.2 | **1.0** | = eine Datenzeile; `live` verlangt ein Vielfaches von dt |
| `--epochs` | 60 | **40** | wie der GridCNN-POC |
| Arme | δ | **`buffer` gegen `live`** | derselbe Lauf, nur der Schalter |
| Seeds | 3 | **3** | ein Seed ist keine Streuung |

Die `[CFL WARN]`-Zeile für δ = 1 s kommt wie in Achse 0 — bekannt, für einen
Vergleich zweier Arme unter gleichen Bedingungen unerheblich.

### Das Kommando — T4, venv, MPS, parallel

```bash
cd ~/llmtraining && git checkout main && git pull
source modulus_env/bin/activate                          # danach `python`, nicht `python3`
systemctl is-active nvidia-mps                           # "active"; sonst README_GPU_SERVER §6.4
pgrep -x nvidia-cuda-mps || nvidia-cuda-mps-control -d   # Notnagel ohne Unit
pgrep -af "sweep.py|train.py|residual_decomposition" || echo "Karte frei"

nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary phys-stencil buffer live -j 3 \
    --out artifacts/poc_p3 --csv artifacts/poc_p3.csv \
    -- --subsample 10 --delta-grid 1.0 --delta-phys 1.0 \
       --epochs 40 --ema-decay 0.5 --device cuda > poc_p3.log 2>&1 &
tail -f poc_p3.log
```

* **6 Läufe, `-j 3`, MPS an** → zwei Wellen. Dauer **geschätzt, nicht gemessen:**
  ~30–35 min je Lauf (aus 10.55 s je OP und Epoche bei dt 0.2 s, Rollout-Anteil
  auf ein Fünftel), also **~1–1.5 h Wanduhr**.
* **Warum `-j 3`:** 6 Läufe sind bei `-j 3` wie bei `-j 4` zwei Wellen, also
  gleich schnell — aber der vierte Kern bleibt frei. Auf ihm läuft
  **gleichzeitig** GridCNN-Lauf 17 (Kommando in
  [`README_NAECHSTE_SITZUNG.md`](../README_NAECHSTE_SITZUNG.md) §4). Zusammen
  vier Prozesse, genau das, was die vier physischen Kerne mit MPS tragen.
* In jedem `train.log` des `live`-Arms muss stehen:
  `physics stencil: LIVE -- ... lag = 1 rows (O21)`. Fehlt die Zeile, ist der
  Code nicht gepullt — abbrechen.
* `sweep.py` meldet selbst, wenn MPS fehlt, und schreibt am Ende die Tafel
  „val-MAE per configuration, pooled over seeds".

### Direkt danach: die sechs POC-Checkpoints zerlegen (Minuten)

```bash
python PINNmodulusTwo/tools/residual_decomposition.py \
    artifacts/poc_p3/*/model.pt -j 4 --device cuda 2>&1 | tee poc_p3_zerlegung.txt
```

Die Tafel am Ende zeigt je Lauf `[BLIND]` / `[STALE]` / `[JITTER]` auf **echten**
Daten. ⚠ Auch die `live`-Läufe zeigen dort `[STALE]` — das Werkzeug misst immer,
was der **alte** Stencil auf diesen Gewichten loggen würde. Maßgeblich für `live`
ist das `L_phys` in seiner `history.csv`.

### Wie er gelesen wird

Drei Signale, in dieser Reihenfolge:

1. **`L_phys` (Mechanismus).**
   `python PINNmodulusTwo/tools/analyse_history.py artifacts/poc_p3/<lauf>/history.csv`
   je Lauf, Median über die letzten Epochen. `live` **≥ 10× kleiner** als
   `buffer` → der Sprung war auf echten Daten der Treiber von `L_phys` (O21
   bestätigt). Ähnlich groß → O21 ist auf echten Daten nicht der Treiber.
2. **val-MAE mit Streuung (Wirkung).** Die Schlusstafel von `sweep.py`:

   | Tafel | Urteil | Folge |
   |---|---|---|
   | `live` vorn um **≥ ~1 °C und** mehr als die Seed-Streuung | 🟢 **POC trägt** | Achse 5: P3 auf voller Auflösung (unten) |
   | `[NOT SEPARATED]` | 🟡 **neutral** — schadet nicht, hilft nicht messbar | P3 wird nicht Default. Weiter mit der Zerlegung und dem `[BLIND]`-Weg |
   | `buffer` vorn um mehr als die Streuung | 🔴 **POC trägt nicht** | P3 verworfen, mit Begründung stehen lassen. `[BLIND]`-Weg |

3. **Stabilität.** `[SATURATED]`-Zeilen und `spread s/t` beider Arme
   vergleichen. `live` darf den Rollout nicht weglaufen lassen; tut er es, ist
   das ein 🔴, egal was die MAE sagt.

### Was danach geschrieben wird — in dieser Reihenfolge

1. **`TRAININGS_BERICHT_<datum>_PINN_P3_POC.md`** im Wurzelverzeichnis, wie die
   GridCNN-Berichte: Kommando, Commit, Maschine (T4, MPS, `-j 4`), Tafel je Seed,
   `L_phys` je Arm, Zerlegungstafel, Urteil.
2. **`README_MODELLSTAND.md`**: Tabelle 1a die Zeile **P3** mit Datum und Commit,
   Tabelle 2 die Experimentzeile „POC P3".
3. **`FAHRPLAN.md`**: Stand-Tabelle (Teil III) und der Kopf-Kasten auf den
   nächsten Schritt.
4. **`README_NAECHSTE_SITZUNG.md`** neu.

---

## 4 · Wenn der POC trägt: Achse 5 auf voller Auflösung

```bash
nohup python PINNmodulusTwo/sweep.py --seeds 0 1 2 \
    --vary phys-stencil live -j 3 \
    --out artifacts/achse5 --csv artifacts/achse5.csv \
    -- --epochs 60 --ema-decay 0.5 --delta-phys 0.2 --device cuda > achse5.log 2>&1 &
```

Der `buffer`-Vergleichsarm ist schon gerechnet: **Achse 1, δ = 0.2,
4.868 ± 0.650 °C** — P2.1 ist mit `buffer` bitgleich zu P2. Dazu die
Nullmessung **5.248 ± 0.518 °C**. ~2 h mit MPS; `-j 3` lässt einen Kern für
einen GridCNN-Lauf frei.

**Erst wenn Achse 5 trägt, wird `phys_stencil: live` Default in
`config.yaml`** — mit eigener Zeile in `README_MODELLSTAND.md`.
