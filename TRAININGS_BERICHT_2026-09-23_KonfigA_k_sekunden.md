# Konfiguration A, Fenster in Sekunden (Lauf 17) — und das Profil von Lauf 16 — 23.09.2026

> **Wenn du nur eine Zeile liest:** Das Fenster in Sekunden holt **OP09 zurück**
> (1.11x → **0.79x**, 2/3 Seeds unter der Latte) und halbiert die Frühphase.
> **Seed 0 ist der beste Einzellauf bisher** (0.47x / 0.58x). Die Streuung
> liegt aber bei 1.8–2.0 °C, also ist das **kein Ergebnis**. Und das
> Fehlerprofil zeigt in beiden Läufen, allen Seeds und auf beiden OPs dasselbe:
> **am Anfang zu warm, am Ende zu kalt**. Das ist ein Pegelfehler, und den kann
> der Diffusionskern von Arm B **nicht sehen**. Einzige Ausnahme ist das Ende
> von OP09 bei Seed 0 in Lauf 17, das dort −0.1 °C trifft.

**Lauf 17:** [`GridCNN/laeufe/17_konfigA_voll_k_sekunden.txt`](GridCNN/laeufe/17_konfigA_voll_k_sekunden.txt) ·
**Parameter:** [`17_konfigA_parameter.md`](GridCNN/laeufe/17_konfigA_parameter.md) ·
**Lauf 16 nachgemessen:** [`16_konfigA_nachgemessen.txt`](GridCNN/laeufe/16_konfigA_nachgemessen.txt) ·
**Vorgänger:** [`TRAININGS_BERICHT_2026-09-23_KonfigA_voll.md`](TRAININGS_BERICHT_2026-09-23_KonfigA_voll.md)
**Code:** `main` nach PR #48 (G4.1) · Tesla T4 · ≈ 1 h 42 min

---

## 1. Die Kopfzahlen

| | Lauf 15 (POC) | Lauf 16 | **Lauf 17** |
|---|---|---|---|
| dt / Schritte | 1 s / 1445 | 0.2 s / ≈ 8040 | 0.2 s / ≈ 8040 |
| Fenster k | 4→16 s | 0.8→3.2 s | **4→16 s** |
| **OP06** | 6.57 ± 0.38 · 0.61x · 3/3 | 6.64 ± 1.52 · 0.62x · 3/3 | **6.64 ± 1.76 · 0.62x · 3/3** |
| **OP09** | 5.81 ± 0.87 · 0.75x · 3/3 | 8.60 ± 1.76 · 1.11x · 1/3 | **6.16 ± 1.97 · 0.79x · 2/3** |
| Verdikt | lesbar | kein Ergebnis | **kein Ergebnis** |

Je Seed ist der Median über das letzte Drittel angegeben (10 von 31 Punkten), dazu Mittel ± sd über drei Seeds, die Güte gegen die Latte und die Zahl der Seeds unter der Latte.

| Lauf 17 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| OP06 | **5.050 (0.47x)** | 6.343 (0.59x) | 8.539 (0.79x) |
| OP09 | **4.494 (0.58x)** | 8.329 (1.07x) ⚠ | 5.644 (0.73x) |

**Hypothese aus dem Vorbericht bestätigt, soweit ein Lauf das kann:** Mit dem
Fenster in Sekunden fällt OP09 von 1.11x auf 0.79x. OP06 bleibt, wo es war.
Seed 0 unterbietet den POC auf beiden OPs, obwohl sein Horizont fünfmal so
lang ist.

---

## 2. Woher die Streuung kommt — zwei benannte Fehlerbilder

Die Streuung ist **kein Rauschen**. Jeder der zwei schwächeren Seeds scheitert
auf eine eigene, benennbare Art:

### Seed 2: explodierende Gradienten ab k ≈ 57

| ep | 22 | 23 | 24 | 28 | 34 | 50 | 59 |
|---|---|---|---|---|---|---|---|
| k | 55 | 57 | 60 | 73 | 80 | 80 | 80 |
| `\|g\|` | 1.6 | 26 | 2.3e5 | 1.7e6 | 2.3e8 | **1.2e14** | 1.5e12 |

Seed 0 und Seed 1 bleiben bei k = 80 meist bei 0.15–0.5. Seed 1 hat zwei Spitzen bis 26 (ep 33, 35) und fängt sich danach wieder. Weil `--clip-grad 1.0`
greift, läuft das Training weiter, ohne `nan` und ohne verworfenes Update.
Aber jedes Update zeigt in die explodierende Richtung. Der `data`-Verlust
bleibt deshalb ab ep 24 bei **0.25–0.28** stehen, gegen 0.06 bei Seed 0, und
OP06 friert bei 0.8x ein. Das ist die bekannte Gefahr von BPTT über lange
Fenster: Es werden 80 Schritt-Jacobis multipliziert, im POC waren es 16.
**„Gleich viele Sekunden" heißt nicht „gleich viele Faktoren"**, und dieser
Seed hat gelernt, was in 80 Faktoren aufschaukelt. Im Vorwärtslauf ist davon
nichts zu sehen: kein `[SATURATED]` zwischen ep 16 und 60, außer ep 35.

### Seed 1: OP09 läuft in der zweiten Hälfte weg

| ep | 24 | 30 | 40 | 48 | 60 |
|---|---|---|---|---|---|
| OP09 | **0.75x** | 0.77x | 0.93x | 1.02x | 1.11x |
| OP06 | 0.68x | 0.62x | 0.61x | 0.56x | 0.59x |
| `data` | 0.162 | 0.139 | 0.091 | 0.101 | 0.091 |

Der Trainingsverlust fällt, OP06 bleibt gut, **nur OP09 wird schlechter**. Das
ist Überanpassung in eine Richtung, die OP09 nicht teilt. Es passt zu seiner
Datenlücke: In allen elf Trainings-OPs gilt `T0 = T_fluid`, OP09 ist der Fall,
der beide trennt ([Steckbrief](GridCNN/README_OPS_STECKBRIEF.md)).

### Frühphase: halbiert, nicht beseitigt

| Epochen > 1 % am Clamp | Seed 0 | Seed 1 | Seed 2 | Summe |
|---|---|---|---|---|
| Lauf 16 | 7 (bis ep 8) | 13 (bis ep 17) | 14 (bis ep 18) | 34 |
| **Lauf 17** | **3** (bis ep 7) | **5** (bis ep 12, dazu ep 28–29) | **10** (bis ep 15, dazu ep 35) | **18** |

Gezählt mit `train.fruehphase`. Ab dieser Version schreibt `train.py` die
Zahl selbst ins Log.

---

## 3. Das Fehlerprofil — Lauf 16 nachgemessen, Lauf 17 im Log

Die Tabelle zeigt den Bias je Sechstel der Trajektorie in °C als Median über
die Seeds. Positiv heißt, das Modell ist zu warm.

| | 1 | 2 | 3 | 4 | 5 | 6 | drift |
|---|---|---|---|---|---|---|---|
| **OP06**, Lauf 16 | **+12.5** | −0.8 | −3.3 | −4.4 | −3.4 | **−9.3** | 1.61x |
| **OP06**, Lauf 17 | **+9.1** | +2.8 | −2.9 | −5.3 | −4.4 | **−10.0** | 1.73x |
| **OP09**, Lauf 16 | **+6.2** | −0.9 | −5.7 | −8.5 | −12.2 | **−15.2** | 1.76x |
| **OP09**, Lauf 17 | **+4.2** | −1.6 | −5.8 | −8.2 | −7.8 | **−9.7** | 1.66x |

Lauf 16 ist aus den `model.pt` nachgemessen (Stand ep 60), Lauf 17 ist der
Median über das letzte Drittel. Die Probe hält: Alle drei `model.pt` von
Lauf 16 treffen die Zeile `letztes ep 60` im Log auf vier Stellen.
`seed0/model_best.pt` ist von einem fremden Prozess überschrieben und fällt
heraus, siehe [Parameterblatt 16](GridCNN/laeufe/16_konfigA_parameter.md).

**Das Muster ist systematisch.** In **12 von 12** Fällen (2 Läufe × 3 Seeds ×
2 OPs) ist das erste Sechstel zu warm. In 11 von 12 ist das letzte zu kalt,
der zwölfte Fall (Lauf 17, Seed 0, OP09) liegt bei −0.1 °C. Das Modell heizt
**anfangs zu schnell und später zu langsam**. Es flacht ab, wie ein gekühlter
OP abflacht, und liegt am Ende feldweit zu kalt. Das ist **O13 mit Vorzeichen**,
und das längere Fenster hat es am Ende von OP09 gedämpft (−15.2 → −9.7), aber
nicht beseitigt.

---

## 4. Was daraus für Arm B folgt — eine Korrektur

Am 22.09. hieß es, Arm B sei „die Behandlung": Der dissipative
Diffusionskern liefere das Leck, das der Δ-Form fehlt (README Sec. 4,
`model.py`). **Für einen Pegelfehler stimmt das nicht.** Ein gleichmäßiger
Versatz ändert den Laplace nicht, `L(T + c) = L(T)`. Das gilt auch mit
wechselndem Fo, mit Kreuzterm und am adiabaten Rand, dessen Geisterschicht
den Versatz mitspiegelt. Neuer Test:
`test_ein_gleichmaessiger_versatz_ist_fuer_den_laplace_unsichtbar`.

Der Kern dämpft **räumliche** Abweichungen, nicht den Pegel. Im Physikteil
von B wirken auf den Pegel nur zwei Terme:

* **`Qsrc`** — die bekannte Wärmequelle. Sie liefert die richtige
  Heizleistung und kann damit das „später zu langsam" treffen, vor allem auf
  OP06 (V̇ = 0, die Wärme bleibt in der Zelle).
* **der Wandterm** — der **einzige** Term, der auf den Pegel *reagiert*
  (`U · (T_Wand − T_fluid)`). Er ist weiterhin nicht verdrahtet.

Adiabat gerechnet hat Arm B also eine Quelle, aber keine Senke, die auf den
Pegel antwortet. Das verschiebt die Gewichte im Fahrplan: **Der Wandterm ist
für die beobachtete Krankheit wichtiger als der Diffusionskern.**

---

## 5. Und CFL: Die Frage ist entschieden

Die Materialdaten sind **echt**. Das hat die Maschine beantwortet:
`constants.yaml` stammt aus dem PDF „Material Properties Gridpoints",
Schaefer, 18.06.2026. Damit ist die Steifigkeit echt: 110x über der expliziten
Schranke bei `--subsample 2`, **55x selbst bei `--subsample 1`** (dt = 0.1 s,
Rohdaten). Ein kleineres `--subsample` kann das Problem nicht lösen. Die
`[CFL]`-Zeile sagt das jetzt selbst, statt es zu empfehlen (`train.cfl_text`).

**Für B/C/D braucht der Physikterm einen eigenen Integrator.** Zwei Wege:

| | wie | Kosten | Haken |
|---|---|---|---|
| **Unterschritte** | `L(T) + Qsrc` mit n ≈ 111 expliziten Unterschritten je Datenschritt, `g_θ` einmal (Operator-Splitting) | ≈ 111 Stencil-Aufrufe je Schritt. Im val-Rollout (8040 Schritte) grob geschätzt Minuten statt Sekunden | Referenz existiert (`solve.rollout` mit kleinem dt) |
| **Exponentieller Integrator** | `L` ist linear in T: einmal als 363×363-Matrix bauen, `exp(dt·L)` vorab rechnen, dann **ein** Matrix-Vektor-Produkt je Schritt | billig, **unbedingt stabil** | mit Wandterm wird die Randbedingung zeitabhängig (V̇-Profil bei OP15), dann ändert sich die Matrix |

Das ist Code ohne Lauf und ohne Daten, prüfbar gegen `solve.rollout`.

---

## 6. Das Nächste

**Schritt 2 als Code: der Integrator für den Physikterm.** Ohne ihn kann
Arm B nicht laufen. Vorschlag: zuerst den exponentiellen Integrator für den
adiabaten Fall, mit einem Test gegen `solve.rollout` bei kleinem dt. Die
Maschine wird dafür nicht gebraucht.

**Was ausdrücklich warten kann, und warum:** An den zwei Fehlerbildern der
Streuung (Seed 2 explodiert, Seed 1 läuft auf OP09 weg) ließe sich am
Protokoll drehen: `k` kürzer, eine Gradientenschranke je Fenster, EMA. Jede
Protokolländerung gilt aber für A/B/C/D gleich, und A muss mit B unter
**demselben** Protokoll neu laufen. Das Protokoll jetzt nur an A zu tunen,
hieße es zweimal zu tunen.

Offen bleiben: der Wandterm (jetzt vorrangig, siehe Abschnitt 4), die drei
Test-OPs (nie berichtet), die Datenlücken von OP06 (O14) und OP09
(T0/T_fluid).
