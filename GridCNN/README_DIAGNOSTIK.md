# Die Epochenzeile lesen — was jede Zahl bedeutet

> Entstanden am 22.09. nach dem ersten echten Lauf von Konfiguration A, weil
> `[SATURATED] 88248` dastand und niemand sagen konnte, ob das viel ist.
> Es ist viel. Es ist praktisch alles.

Eine Zeile aus `GridCNN/train.py` sieht seit dem 22.09., abends, so aus:

```
[seed 0] ep    9 | k  16 | data 0.305 | phys   --   | wall   --   | Streuung Ort 1.023 Zeit 0.981 | |g| 0.47 | 12.4s
[seed 0]      val-MAE OP06 6.1200 (0.79x Latte)  OP09 5.8400 (0.76x Latte)  (Mittel 5.9800 C)  <- bestes bisher
```

Vorher sah sie so aus, und **genau daran ist der Lauf vom 22.09. fast
vorbeigegangen**:

```
[seed 0] ep    9 | data 2500.7 | phys   --   | wall   --   | Streuung Ort 196.123 Zeit 3.689 | [SATURATED] 88188
```

Die erste Zeile wird von `EpochStats.line()` gedruckt und fasst **eine** Epoche
über **alle** Trainings-OPs zusammen. Die zweite steht alle `--val-every`
Epochen darunter.

### Was seit dem 22.09. dazugekommen ist

| Feld | | |
|---|---|---|
| `k` | Fensterlänge des Gradienten (`--tbptt`) | **das wichtigste neue Feld** — siehe unten |
| `\|g\|` | mittlere Gradientennorm **vor** dem Clipping | trennt „der Verlust ist groß" von „der Gradient ist explodiert" |
| `12.4s` | Wanduhr dieser Epoche | damit sich ein Budget aus dem Log planen lässt |
| `[UEBERSPRUNGEN] n` | Updates mit nicht-endlichem Verlust, verworfen | ein NaN vergiftet sonst still den ganzen Lauf |
| `(0.79x Latte)` | val-MAE geteilt durch die beste triviale Latte | **< 1 heißt gelernt** — die Zahl und ihr Maßstab in einer Zeile |
| `[SATURATED] n/max = x %` | Sättigung als **Anteil** | „88248" ohne Bezugsgröße ist unlesbar |

---

## `k` — wie weit der Gradient läuft

`--tbptt k` ist die Fensterlänge je Update. Sie ist die Antwort auf den
Befund vom 22.09.: **ein Schritt trainierte, rund 8040 wurden gemessen.**

| `k` | was der Optimierer sieht |
|---|---|
| **1** | genau einen Schritt. Der Pfad vom 22.09.; ein winziger systematischer Fehler je Schritt ist hier unsichtbar und über den Horizont tödlich |
| **> 1** | das Fenster rollt **mit** Gradient und frisst sich dabei selbst: ab Schritt 1 ist der Anker die eigene Vorhersage, nicht der Puffer. Der aufsummierte Fehler ist damit erreichbar |

Das Fenster startet immer im eingefrorenen Puffer, nie im Label — es bleibt
**kein Teacher Forcing**. Zwei Tests halten beide Hälften fest:
`test_der_gradient_ueberquert_die_historie_nicht` für `k = 1` und
`test_der_gradient_ueberquert_die_historie_bei_k_groesser_eins_sehr_wohl` für
den anderen Fall.

Der Preis ist linear: ein Update kostet `k` Vorwärts- und Rückwärtsschritte.
**Was konstant gehört, ist `inner_steps × k`**, nicht `inner_steps`.

`--tbptt-start` lässt `k` geometrisch wachsen und ab der halben Laufzeit
stehen. Ein langes Fenster auf einem untrainierten Netz summiert nur Rauschen;
und weil der berichtete Median aus dem letzten Drittel kommt, darf `k` dort
nicht mehr wandern.

---

## `[SATURATED] N` — der wichtigste und der am leichtesten misszuverstehende

### Was gezählt wird

`train.rollout_batched` rollt die Trajektorie **frei laufend** aus, gesät nur
von der gemessenen Anfangsbedingung. Läuft sie weg, hielte nichts den Verlust
endlich — deshalb gibt es `--clamp` (Vorgabe **`auto`**, siehe unten).
Und weil ein stilles Festhalten wie langsame Konvergenz *aussieht*, wird es
gezählt:

```python
saturated += int((nxt.abs() >= clamp).flatten(1).any(dim=1).sum())
nxt = nxt.clamp(-clamp, clamp)
```

Zeile für Zeile:

| | |
|---|---|
| `nxt` | `(B, nx, ny, nz)` — der nächste Zeitschritt für **alle B OPs gleichzeitig** |
| `.flatten(1)` | `(B, 363)` — die 363 Gitterpunkte je OP |
| `.any(dim=1)` | `(B,)` — hat **irgendein** Punkt dieses OPs geklemmt? |
| `.sum()` | wie viele der B OPs in **diesem** Schritt geklemmt haben, also 0 … B |
| `+=` | aufsummiert über alle Zeitschritte des Rollouts |

**Die Einheit ist also „OP-Zeitschritte", nicht „Gitterpunkte".** Ein OP, der
an einem einzigen seiner 363 Punkte klemmt, zählt genauso wie einer, der
überall klemmt. Das ist Absicht — sonst hinge die Zahl an der Batchgröße
statt am Verhalten — aber es heißt auch: **die Zahl sagt, wie oft, nicht wie
schlimm.**

### Was „viel" ist — die Bezugsgröße, die bisher fehlte

Der Zähler hat eine Obergrenze:

```
max = B × (n_max − 1)
```

`B` ist die Zahl der Trainings-OPs, `n_max` die Länge des längsten davon.

**Rechenbeispiel aus dem Lauf vom 22.09.** (Konfiguration A, `--subsample 2`):
elf Trainings-OPs, der längste rund 8040 Zeitschritte.

```
max ≈ 11 × 8039 ≈ 88 400
```

Beobachtet wurde bis zu **88 248**. Das sind **≈ 99.8 %** — also
**jeder OP in praktisch jedem Zeitschritt geklemmt.** Der freilaufende Rollout
lag über die ganze Trajektorie an der Schranke.

> ⚠ **`--clamp 50` war keine harmlose Zahl.** Normiert heißt 50 bei
> `T_sigma = 9.602 C` und `T_mu = 33 C` rund **±480 °C** Abstand vom Mittel.
> Ein Rollout, der dort anliegt, ist nicht „etwas zu warm", er ist weg — und
> der Zähler meldet dann etwas, das längst nichts mehr rettet.

### `--clamp auto` — die Schranke kommt aus den Daten

Die 50 waren aus `PINNmodulusTwo` **geerbt**, nicht gewählt. Seit dem
22.09., abends, ist die Vorgabe `auto`:

```
clamp = ceil(--clamp-faktor × max|tn| über alle Labels)      (Faktor 3)
```

Über dem Dreifachen der größten Auslenkung, die in den Labels überhaupt
vorkommt, ist kein Zustand mehr physikalisch. Die Schranke greift damit früh
genug, dass `[SATURATED]` noch eine Aussage ist statt ein Nachruf. Der Lauf
druckt die gewählte Zahl in °C:

```
[clamp] auto = ceil(3 x max|tn|=9.271) = 28  (±269 C um T_mu; die Labels reichen bis ±89.0 C)
```

`--clamp 50` von Hand und `--clamp aus` gibt es weiter.

### Warum eine Sättigungsphase die *nächste* Epoche vergiftet

Das ist der Teil, der beim Lesen der Zeile leicht untergeht. `train_epoch`
macht zwei Dinge nacheinander:

1. **einmal** frei ausrollen, unter `no_grad`, Ergebnis einfrieren → `traj`
2. je OP `inner_steps` Updates über ein Fenster von `k` Schritten, und der
   **Startzustand dafür kommt aus `traj`** (`history_at`), nicht aus den
   Labels

Schritt 2 ist genau der Grund, warum das **kein** Teacher Forcing ist. Aber er
hat eine Kehrseite: ist `traj` in Epoche *n* an der Schranke festgenagelt, dann
trainiert Schritt 2 das Netz auf Eingaben, die es nach einer Erholung **nie
wieder sieht**. Eine Sättigungsphase ist deshalb keine vorübergehende Delle,
sondern eine Rückkopplung.

Genau deshalb ist die Schranke aus den Daten wichtiger geworden, nicht
unwichtiger: je später sie greift, desto länger füttert der Puffer das
Training mit Zuständen, die es nie wieder gibt.

### Was der Zähler **nicht** sagt

* **Nicht**, wie viele Gitterpunkte betroffen sind (`any`, nicht `sum`).
* **Nicht**, ob es früh oder spät in der Trajektorie passiert.
* **Nicht**, ob die Haltemenge betroffen ist — gezählt wird nur der
  Trainings-Rollout. `val_mae` rollt getrennt aus.
* **Nichts über die Qualität:** `0` heißt nur „nicht an der Schranke". Am
  22.09. endete Seed 0 mit `[SATURATED]` **leer** bei `data 0.332` — und einer
  val-MAE von **51.95 °C**.

---

## `data` — der mittlere Fensterverlust

Der Mittelwert über die **wirklich gezählten** Updates (übersprungene zählen
nicht mit, sonst sähe eine halb verworfene Epoche wie eine halbierte
Verlustkurve aus). Je Update der Mittelwert über die `k` Schritte des
Fensters. Labels nur bis `split_t` — dahinter liegt das Fenster, das
`op_metrics` als in-time ausgehalten berichtet, und mit `k > 1` endet `idx`
um `k` früher statt um 1, damit das Fenster nicht darüber greift.

> ⚠ **`data` sagt auch jetzt wenig über die freilaufende Güte.** Am 22.09.
> endeten Seed 0 bei `0.332` (val-MAE 51.95 °C) und Seed 2 bei `0.305`
> (val-MAE 8.95 °C) — praktisch derselbe Verlust, ein Faktor **5.8** im
> Ergebnis. Mit `k = 16` ist der Abstand kleiner als mit `k = 1`, aber er ist
> nicht weg: optimiert werden sechzehn Schritte, gemessen werden Tausende.
> **Die Zahl, auf die es ankommt, steht in der val-Zeile darunter**, und dort
> steht seit dem 22.09. die Latte daneben.

## `phys` / `wall` — `--` heißt „nicht gerechnet", nicht „null"

`phys` erscheint nur bei `--w-phys > 0`, `wall` nur mit kalibriertem Wandterm.
Ein `--` ist eine **Abwesenheit**, keine Messung. In Konfiguration A stehen
beide auf `--`, und der Lauf ist adiabat — eine **Ablation, keine Latte**.

## `Streuung Ort` / `Streuung Zeit` — trägt das Modell noch Struktur?

Aus `spread_ratios`: die Standardabweichung der **eigenen** Trajektorie geteilt
durch die der Labels, einmal über den Ort, einmal über die Zeit.

| Wert | Lesart |
|---|---|
| **≈ 1** | das Modell trägt so viel Struktur wie die Wirklichkeit |
| **→ 0** | flach gelaufen — die triviale Lösung, `[FLAT]` im Basisprojekt |
| **≫ 1** | über-strukturiert, meist gemeinsam mit einer Sättigungsphase |

Beide Residuenterme verschwinden identisch auf einem Feld, das in Ort und Zeit
konstant ist — deshalb steht die Zahl überhaupt da.

> ⚠ Auch sie trennt gut von schlecht **nicht** zuverlässig: am 22.09. endeten
> Seed 0 bei `Ort 2.395` und Seed 2 bei `Ort 2.633` — der schlechtere Seed hatte
> den *näher an 1* liegenden Wert.

---

## `!! [CFL]` — steht vor dem Lauf, nicht in der Epochenzeile

`dt_max = 1 / (2 · Σ_i Fo_ii / h_i²)` am schlechtesten Punkt, die übliche
Schranke für den expliziten Diffusionsstern (`physics.cfl_limit`). Der
Kreuzterm steckt **nicht** darin, es ist also eine Richtgröße — deshalb eine
Warnung und kein Abbruch.

**Sie betrifft Konfiguration A nicht.** Arm A ist `--no-physics`: die Rekurrenz
ist `T + dt·g_θ`, ohne Laplace-Term. Läuft A weg, ist CFL **nicht** die
Erklärung.

**Für B, C und D betrifft sie sehr wohl.** Am 22.09. lag `--subsample 2`
(dt = 0.2 s) **110.3×** über der Schranke (dt_max = 1.814 ms). Das ist vor dem
ersten B-Lauf zu klären.

---

## Kurzreferenz

| Zeichen | Bedeutung | Gute Werte |
|---|---|---|
| `k` | Fensterlänge des Gradienten | > 1. Bei 1 trainiert ein Schritt, was über Tausende gemessen wird |
| `data` | mittlerer Fensterverlust | fallend — sagt aber wenig über frei laufend |
| `\|g\|` | Gradientennorm vor dem Clipping | stabil. Ausschläge auf Hunderte sind explodierende Gradienten |
| `(x.xx Latte)` | val-MAE / beste triviale Latte | **< 1.00.** Darüber hat das Netz nichts gelernt, was über Raten hinausgeht |
| `[UEBERSPRUNGEN]` | verworfene nicht-endliche Updates | **0** |
| `phys` | Physik-Residuum | nur bei `--w-phys > 0`; `--` = nicht gerechnet |
| `wall` | Wandterm | nur mit kalibriertem `U`; `--` = adiabat |
| `Streuung Ort/Zeit` | Struktur gegen die Labels | nahe 1; 0 = flach, ≫1 = weggelaufen |
| `[SATURATED] N` | OP-Zeitschritte an `--clamp` | **0**. Ins Verhältnis zu `B × (n_max−1)` setzen |
