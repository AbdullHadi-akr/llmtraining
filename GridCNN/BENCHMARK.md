# GridCNN — Benchmark

> **Lebendes Dokument.** Die Läufe unten schreibt [`benchmark.py`](benchmark.py)
> selbst, oben hinein, neueste zuerst. Der Abschnitt „Offene Routen" wird von
> Hand fortgeschrieben — dort steht, was aus den Messungen folgt.
>
> Die Arbeitsteilung ist Absicht: **was gemessen wurde, gehört der Maschine;
> was daraus folgt, gehört uns beiden.** Eine Route, die niemand geschrieben
> hat, gibt es nicht.

```bash
python3 GridCNN/benchmark.py                # alle lauffähigen Stufen
python3 GridCNN/benchmark.py --stage 0      # nur der Workflow-Test
python3 GridCNN/benchmark.py --dry-run      # ausprobieren, nichts eintragen
```

> **⚠ Was hier gemessen wird, ist der Unterbau — nicht die Architektur.**
> Die vier Stufen unten prüfen Reshape, Padding, Stencil und Löser. Eine Stufe,
> die rot wird, sagt etwas über die **Physik** — nicht darüber, ob der CNN die
> richtige Wahl war. Diese Frage beantwortet Stufe 4 des Fahrplans über die
> Ablation A/B/C, und keine Zahl auf dieser Seite nimmt sie vorweg.

---

## Die Leiter

| Stufe | Name | braucht | fragt |
|---|---|---|---|
| **0** | `workflow` | nichts | Trägt die Werkzeugkette überhaupt? |
| **1** | `grid` | `data_cache` | Ist das Gitter 3 × 11 × 11, über alle OPs gleich? |
| **2** | `stencil` | nichts | Haben die Differenzensterne die richtige Ordnung? |
| **3** | `solver` | `data_cache` | Ist der Löser stabil, und schlägt er die trivialen Vorhersager? |

**Stufe 0 prüft absichtlich noch keine Physik.** Sie prüft die vier Zusagen,
die *exakt* gelten müssen und auf denen alles Weitere steht:

1. der Reshape 363 → (3, 11, 11) lässt sich aus den Koordinaten ableiten und ist umkehrbar,
2. `dT/dx` an der Symmetrieebene ist **exakt** null — nicht klein, null,
3. die zentrale Differenz am y/z-Rand ist **exakt** null,
4. ein Euler-Schritt läuft durch und die Diffusion glättet.

Fällt eine davon, ist jede Zahl aus den späteren Stufen wertlos. Deshalb steht
sie vorn und nicht am Ende.

---

## Der Stand

| | Stufe 0 `workflow` | Stufe 1 `grid` | Stufe 2 `stencil` | Stufe 3 `solver` |
|---|---|---|---|---|
| zuletzt | 🟢 8 × ok | ⏸ kein Cache | 🟢 4 × ok | ⏸ kein Cache |
| Datum | 14.09. | — | 14.09. | — |

**Kurzfassung des ersten Laufs.** Die Werkzeugkette trägt: der Reshape wird aus
den Koordinaten abgeleitet und trifft die echte Geometrie (0.198094368 × 0.104431991 m,
dx 10.786 / 11.114 mm), beide Randzusagen sind **exakt** null, und die
Differenzensterne konvergieren (d²/dy², d²/dz² mit Ordnung 1.98; d²/dx² mit
1.28 auf gestrecktem Gitter, was für den nicht-äquidistanten Dreipunktstern
richtig ist).

Ein Befund ist dabei abgefallen und steht als **R6**: der Stern ist nicht
bilanztreu, die Drift beträgt 3.6 % der Feldstreuung und sättigt.

Stufe 1 und 3 warten auf einen `data_cache` — der liegt nicht im Repo.

---

## Offene Routen

Hier wird von Hand geschrieben. Jede Route bekommt einen Zustand: **offen**,
**gewählt**, **verworfen** — und verworfene bleiben mit Begründung stehen,
damit sie nicht in drei Wochen als neue Idee zurückkommen.

### R1 · Stufe 2 zuerst, dann die echte Physik-Latte — **offen**

Der Wandterm ist gebaut, aber nicht kalibrierbar: `U(V̇)` braucht
`q_solid_to_fluid`, `mdot`, `cp_fluid` und `fluid_out_temp` im Bündel, und die
liegen erst nach Stufe 2 im Cache. Bis dahin läuft der Löser **adiabat** — das
ist eine Ablation, keine Latte, und der Benchmark weist es auch so aus.

*Kosten:* 30 min Cache-Umbau plus Rebuild aller siebzehn OPs.
*Tor:* `profile_report`, `coverage_report` und `energy_balance_report` müssen
exakt dieselben Zahlen liefern wie vorher.

### R2 · Adiabat vorrollen, um Vorzeichenfehler billig zu finden — **offen**

Den Löser ohne Wandterm über alle OPs rollen. Das misst keine Latte, findet
aber Stabilitäts- und Vorzeichenfehler, solange sie billig zu finden sind.
Kann parallel zu R1 laufen und braucht nur den heutigen Cache.

### R3 · Der Kapazitätsterm im Fluidpfad — **offen**

`WallModel` kennt zwei Modi. `advective` ist die Form aus README §6 und hat bei
ṁ = 0 eine Polstelle; `capacity` ist die Korrektur vom 09.09.:

```
C_fluid · dT_fluid/dt = Q̇ − ṁ · Cp · (T_fluid − T_in)
```

Die interessante Messung steht im Fahrplan Stufe 3: *verbessert ein
Kapazitätsterm die V̇ = 0-Betriebspunkte, ohne dass neue Daten dazukommen?*
Wenn ja, war O14 nicht nur eine Abdeckungsgrenze.

⚠ `capacity` braucht `C_fluid` in J/K — die Wärmekapazität des Kühlmittels im
Kanal. Die liegt **nicht** vor. Ein geratener Wert wäre ein freier Parameter,
und bei elf Trajektorien zählt jeder, den man nicht braucht. Deshalb wirft der
Modus heute, statt zu raten.

### R4 · Die konservative Form `div(λ ∇T)` — **verworfen, vorerst**

Gerechnet wird `Fo : ∇²T`, also die nicht-konservative Form mit ortskonstantem
λ. Streng richtig wäre die konservative, die sich dort unterscheidet, wo λ
springt — an den Grenzen Cell Center / JR1 / Gehäuse.

**Bewusst nicht geändert:** `PINNmodulusTwo/physics.py:189` rechnet dieselbe
Form. Würde GridCNN die andere nehmen, wäre ein Vergleich der beiden Modelle
keine Architekturaussage mehr, sondern eine Diskretisierungsaussage. Wenn
geändert, dann in beiden Projekten gleichzeitig und als eigene Achse.

### R5 · Die Größe des Faltungsstapels — **offen, gebaut**

Der Rangtest hat vier Moden bei 99.9 % gemessen, und der Entwurf sah 64 × 4 vor
— auf elf Trajektorien. Gebaut ist deshalb **16 × 3 = 11 427 Parameter**; die
Präsentationsgröße bleibt als Sweep-Achse erreichbar.

| `--width` / `--blocks` | Parameter | |
|---|---|---|
| 16 / 3 | **11 427** | Vorgabe |
| 24 / 3 | 20 595 | Gegenprobe |
| 64 / 4 | **137 923** | der Entwurf |

⚠ **137 923, nicht „~100 k".** README §11.4 und frühere Fassungen des Fahrplans
nennen für 64 × 4 rund 100 k Parameter. Nachgerechnet sind es 38 % mehr; die
Schätzung hatte den Sprung 44 → 64 in der ersten Schicht und die vierte
64×64-Faltung zu klein angesetzt. `tests/test_model.py` nagelt beide Zahlen fest.

Die Achse prüft, ob größer überhaupt etwas bringt. Bringt sie nichts, ist das
kein Nullergebnis, sondern eine Bestätigung des Rangtests.
### R6 · Der Stern ist nicht bilanztreu — **offen, beziffert**

Gemessen am 14.09. im Workflow-Test: rollt man ein zufälliges Feld adiabat aus,
driftet das Mittel um **~3.6 % der Feldstreuung** und **sättigt** dann. Das Feld
läuft sauber gegen eine Konstante (std fällt auf 4e-5) — der Löser ist also
stabil und korrekt, aber die Konstante ist nicht exakt das Anfangsmittel.

Ursache, zwei bewusste Entscheidungen:

1. die nicht-konservative Form `Fo : ∇²T` mit ortskonstantem Fo (siehe R4),
2. knotenzentriertes `reflect`-Padding **ohne Halbzellgewichte** am Rand — die
   Spaltensummen des diskreten Operators sind dort ungleich null.

Dass die Drift sättigt, ist der Beleg, dass kein Rand *echte* Energie leckt:
sie läuft, solange das Feld ungleichmäßig ist, und hört auf, sobald es flach
ist. Ein Test nagelt beides fest (`test_der_stern_ist_nicht_konservativ_und_die_drift_saettigt`).

**Warum das trotzdem zählt:** die Physik-Latte aus Stufe 3 bekommt damit einen
kleinen systematischen Bias. Solange er unter der Seed-Streuung (~1 °C) bleibt,
ist er egal; er gehört aber neben die Latte geschrieben, nicht darunter
versteckt. Möglicher Umbau, falls er zu groß wird: Halbzellgewichte an den
y/z-Rändern — das wäre eine eigene Achse und keine stille Korrektur.


### R7 · Die POD-Basis als eigene Stufe — **verworfen, 15.09.**

Der Rangtest hat vier Moden gemessen; eine Stufe, die den **Projektionsrest**
`‖T − ΦΦᵀT‖ / ‖T‖` je ausgehaltenem OP misst, wäre die einzige Messung im
Projekt, die eine Obergrenze **vor** dem Training nennt: liegt der Rest auf
OP06 bei 2 %, kommt kein `g_θ` darunter.

Verworfen, weil der ROM gestrichen ist (Fahrplan, 15.09.). Bleibt mit
Begründung stehen, wie es das Dokument für verworfene Routen vorschreibt — und
weil sie die Messung ist, die man nachholt, wenn Stufe 4 den CNN nicht trägt.

### R8 · Die zwei Koordinatenkarten — **offen, neu am 22.09.**

README §2b begründet den CNN damit, dass er Position **nicht** auswendig lernen
*kann* und räumliche Struktur deshalb über die Materialkarten begründen *muss*.
[`model.py:193`](model.py) gibt ihm die Position trotzdem: zwei z-gescorte
Koordinatenkarten (`y_map`, `z_map`) unter den 17 statischen Kanälen, mit dem
Kommentar, ohne sie wäre *„jede Randzelle von jeder Mittelzelle
ununterscheidbar"*. Beide Texte gehen von derselben Prämisse aus und ziehen den
entgegengesetzten Schluss.

Am 22.09. am Cache gemessen, was die Materialkarten stattdessen hergeben:

| | |
|---|---|
| `region`, `rho`, `Cp` | je x-Ebene **konstant** — in der Ebene strukturlos |
| `lam_xx/yy/zz` | variabel, aber **99 von 121 Punkten tragen einen Wert**; die Variation sitzt in zwei Zeilen am unteren y-Rand |
| innerhalb dieses Bandes | **keine räumliche Ordnung** — `lam_xx` springt zwischen Nachbarn 5.18 … 6.31 |
| Kontrast in der Ebene ÷ zwischen den Ebenen (nach z-Score) | **~1 : 12** |
| Kanäle, die in der Ebene variieren | **17 von 44** — 9 Zustand, 6 `lam`, 2 Koordinaten |

**Die Messung entscheidet die Frage nicht, sie schärft sie.** Was die
Materialkarten in der Ebene hergeben, ist im Wesentlichen ein Randband — also
fast dieselbe Information wie eine geschwellte y-Karte. Ob das Netz ohne die
Koordinatenkarten auskommt, ist damit eine offene, messbare Frage und kein
Entwurfsargument mehr.

**Die Route:** ein vierter Ablationsarm **D** = B ohne `y_map`/`z_map`
(`--no-coord-maps`), zwei Kanäle von 44, sonst identisch. A/B/C berühren die
Frage nicht — alle drei tragen die Karten.

* **D ≈ B** → der Ortsprior aus §2b ist echt, und der Absatz stimmt.
* **D deutlich schlechter** → der CNN hat dieselbe Positionskrücke wie das MLP,
  und §2b beschreibt einen Entwurf, den der Code nicht umsetzt.

*Kosten:* ein Lauf je Seed, kein neuer Code außer dem Flag. Blockiert durch den
Ladepfad wie A/B/C.

> **Nebenbefund, klein aber gegen die eigene Doktrin:** `rho*Cp` ist auf Ebene 0
> und 1 bitgleich, die beiden Kanäle sind nach dem z-Score identische
> Konstanten. Der `dead`-Melder in `build_static_maps` prüft nur auf *global*
> konstante Größen und sieht Duplikate nicht — obwohl der Modulkopf sagt: *„Ein
> redundanter Kanal ist bei elf Trajektorien kein harmloser Kanal."*

### R9 · 30 % der Gewichte sind in Schicht 1 flach — **offen, beziffert 22.09.**

27 der 44 Eingangskanäle sind in der Ebene konstant (9 Materialkarten, 18
gebroadcastete Treiber). Für einen solchen Kanal ist nach `reflect`-Padding nur
die **Summe** der neun Kernelgewichte identifizierbar; die übrigen acht
Richtungen sind flach im Verlust.

| | |
|---|---|
| Schicht 1 (44 → 16, 3×3) | 6 352 Parameter |
| davon auf konstanten Kanälen | 27 × 16 × 9 = 3 888 |
| davon identifizierbar | 27 × 16 = 432 |
| **flach** | **3 456 = 30.2 % aller 11 427** |
| effektive Parameterzahl | **≈ 7 971** |

Kein Fehler, aber es verschiebt Tor 0: die 11 427 überzeichnen die Kapazität um
knapp ein Drittel. Gegen *„vier Moden reichen"* steht ein kleineres Modell, als
die Zahl behauptet. Ob man daraus etwas macht — FiLM statt Broadcast für die
Treiber wäre der naheliegende Weg —, ist eine eigene Sweep-Achse und
ausdrücklich **nicht** Teil von Stufe 4.

---

## Was hier nicht hineingehört

* **`Q̇(t)` und `Tmfavg_fluid_out(t)` als Modelleingang.** Beide sind
  Simulationsergebnisse und zur Laufzeit nicht verfügbar. Aufsicht (`L_wall`)
  und Gegenprobe ja — Eingang nie. Eine val-MAE, die mit ihnen als Eingang
  entsteht, ist wertlos.
* **`U` auf OP13/OP15/OP16 kalibrieren.** Trainierte Flusslevel sind 0/15/30
  l/min. OP16 fährt 90 und ist die *Gegenprobe* für `U(V̇)`, nie die Stütze.
  `UCurve` klemmt deshalb außerhalb der Stützstellen und zählt mit, wie oft.
* **Eine Zahl ohne Streuung daneben.** Die Seed-Streuung ist gemessen: 0.518 °C
  ohne Physik, 0.882 °C mit, auf OP06 allein bis 1.63 °C. Eine Achse, die
  weniger als ~1 °C bewegt, ist mit drei Seeds nicht lesbar.

---

## Lauf-Protokoll

<!-- LAUF-PROTOKOLL -->

### 2026-09-22 — Koordinaten und Materialverteilung, am Cache gemessen

Kein `benchmark.py`-Lauf: die Zahlen kommen von der Maschine mit `data_cache/`
(siebzehn OPs) und aus den drei Koordinaten-CSVs, die im Repo liegen.

- `OK` **`xyz` über alle siebzehn OPs bitgleich**, gleiche Zeilenreihenfolge,
  `layer` ebenso. `data.py:653` nahm die Geometrie bisher unbesehen aus
  `raw[0]` — der Kommentar *„grid identical across OPs"* ist jetzt gemessen.
  **Gilt nach dem Cache-Umbau von Stufe 2 wieder als ungeprüft.**
- `OK` **Cache = Legacy-CSVs, exakt.** Spannweite 0.198094368 × 0.104431991 m,
  `dy = 19.809437 mm`, `dz = 10.443199 mm`, x-Ebenen 0 / 10.785542 / 21.9 mm.
- `FUND` **Die dokumentierte Spannweite war falsch**: 0.198089 × 0.104441 stand
  an sieben Stellen und stammt aus keiner der beiden Quellen. Korrigiert.
  Auf drei Stellen stimmte sie, die Zusage „Kante auf Kante" bleibt gültig.
- `FUND` **Der Geometrie-Test war zirkulär.** `test_layout_trifft_die_echte_geometrie`
  prüfte die Konstanten, aus denen `conftest.py` sein Gitter baut. Gegen die
  echten Koordinaten wären alle vier Vergleiche gefallen (`dx` um 4.6e-7, die
  Spannweiten um 5.4e-6 / 9.0e-6). Neu:
  `test_layout_aus_den_echten_koordinaten` liest die CSVs direkt.
- `WARN` **Äquidistanz hält nur bis float32.** Der Cache speichert `xyz` als
  float32; die Abstände streuen dadurch um 4e-7 relativ. `grid._uniform` prüft
  mit `rtol=1e-6`, Reserve **3.7×**. Über `OPData.xn` (nochmals float32, siehe
  `data.py:604`) sinkt sie auf 2.5×, über die unskalierten `xn` auf 1.8×.
  → Der Ladepfad soll `derive_layout` mit `bundle.xn` (float64) × `L_ref`
  füttern, nicht mit `op.xn`.
- `FUND` **Materialverteilung**: `region`/`rho`/`Cp` je x-Ebene konstant; `lam`
  variabel, aber nur in zwei Zeilen am unteren y-Rand. Siehe **R8** und **R9**.
- `OK` `lam_xz`/`lam_yz` sind global null — genau die zwei, die
  `build_static_maps` von vornherein weglässt. `lam_xy` ist ungleich null
  **genau auf JR1**, wie der Kommentar dort behauptet. Beides bestätigt.
- `FUND` Eine float32-Schranke im gepaddeten Rollout-Test (`< 1e-6`) hielt
  zufällig; der Schwestertest nennt ~8e-6 als normale float32-Abweichung.
  Jetzt float64 **ohne Toleranz** — dort ist die Differenz exakt null, auch für
  den gepaddeten OP.
- **100 Tests grün** (99 + der neue Koordinatentest), 5.3 s, ohne GPU und ohne
  `data_cache`.

### 2026-09-14 11:25 UTC — teilweise

**Stufe 0 · `workflow`** — 8x ok

- `OK` Importe (torch, grid, physics, solve)
  - torch 2.14.0+cu130, numpy 2.4.6
- `OK` Reshape aus Koordinaten abgeleitet
  - 3 x 11 x 11 = 363 Punkte
- `OK` to_field / to_flat ist exakt umkehrbar
- `OK` dT/dx an der Zellmitte ist exakt null `0`
  - Der Zaehler ist T1 - T1; alles ausser 0.0 heisst, das Padding ist falsch.
- `OK` zentrale Differenz am y-Rand ist exakt null `0`
  - reflect spiegelt um den Randknoten -- sonst waere es replicate.
- `OK` fuenf Euler-Schritte laufen durch `2.52669`
  - stabil | max|Tn| 2.527 | dt 0.4923 gegen dt_max 2.462 (0.20 x) | Wandterm AUS (adiabat)
- `OK` Diffusion glaettet (Varianz faellt) `0.192756`
  - Varianz 0.9392 -> 0.181. Steigt sie, hat der Stern ein falsches Vorzeichen.
- `OK` Drift des Mittels saettigt (Stern ist nicht bilanztreu) `0.036058`
  - Drift 0.03455 -> 0.03499 bei 400 -> 1600 Schritten, 3.6 % von std. Bekannte Naeherung, siehe Route R6.

**Stufe 1 · `grid`** — 1x skip

- `--` data_cache gefunden
  - Kein Cache. Auf der Rechenmaschine liegt er unter data_cache/; mit --cache einen Pfad angeben.

**Stufe 2 · `stencil`** — 4x ok

- `OK` Konvergenzordnung d2/dx2 (nicht aequidistant) `1.27513`
  - Fehler 0.044 -> 0.0182 bei 17 -> 33 Knoten
- `OK` Konvergenzordnung d2/dy2 `1.98078`
  - Fehler 0.00217 -> 0.00055 bei 17 -> 33 Knoten
- `OK` Konvergenzordnung d2/dz2 `1.98111`
  - Fehler 3.47e-06 -> 8.8e-07 bei 17 -> 33 Knoten
- `OK` Kreuzterm d2/dxdy `0.000826103`
  - lambda_xy ist auf JR1 ungleich null -- dieser Term darf nicht fehlen.

**Stufe 3 · `solver`** — 1x skip

- `--` data_cache gefunden
  - siehe Stufe 1

**Vorgeschlagene Routen**

1. Workflow traegt. Naechste sinnvolle Stufe ist 2 (Stencil-Ordnung) -- sie braucht ebenfalls keine Daten.
2. Fuer Stufe 1 und 3 wird ein data_cache gebraucht. Der liegt nicht im Repo; auf der Rechenmaschine liegt er unter data_cache/.
3. Stufe 1 und 3 auf der Maschine mit den Daten laufen lassen. Sie sind schnell und brauchen kein GPU.
4. Die Sterne stimmen. Damit ist physics.py so weit belastbar, wie es ohne echte Materialdaten geht.
5. Offen bleibt die konservative Form div(lambda grad T): sie wird bewusst NICHT gerechnet, damit der Vergleich mit PINNmodulusTwo eine Architekturaussage bleibt. Wenn geaendert, dann in beiden Projekten gleichzeitig und als eigene Achse.

---
