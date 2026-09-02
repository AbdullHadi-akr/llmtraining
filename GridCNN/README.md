# GridCNN — das Feld auf einmal, statt Punkt für Punkt

> **Status: Entwurf, kein Code.** Diese Datei ist die Design-Diskussion. Sie
> wird erst zu Code, wenn der Input-Block und die Rekurrenz unten abgenickt
> sind. Offene Fragen stehen am Ende und sind als solche markiert — nichts
> davon ist stillschweigend entschieden.

Ein zweiter, **unabhängiger** Modellansatz neben
[`PINNmodulusTwo/`](../PINNmodulusTwo/). Nicht dessen Ersatz: dasselbe Datum,
derselbe Split, dieselben Metriken, damit die beiden vergleichbar sind — aber
eigenes Modell, eigenes Training, eigener Verlust.

## Warum der Name

`Grid`, weil das die tragende Eigenschaft ist (siehe unten: das Gitter ist
strukturiert und über alle OPs identisch — ohne das ginge kein CNN), `CNN` für
die Architektur. Ein `git mv` kostet nichts, falls dir etwas Besseres einfällt.

---

## 1. Die Voraussetzung: das Gitter ist strukturiert

Ausgezählt aus `legacy/battery_surrogate_agenticWorkflow/coordinates/`:

| Layer | x | Punkte |
|---|---|---|
| Cell Center | 0.0 (Symmetrieebene) | 121 |
| JR1 Center | 0.010786 m | 121 |
| Gehäusewand | 0.021900 m | 121 |

Alle drei Layer haben **exakt dieselben 11 y- und 11 z-Werte**, äquidistant:

* Δy = 19.809 mm (−98.91 mm … +99.18 mm)
* Δz = 10.444 mm (−52.26 mm … +52.18 mm)
* in x: 10.786 mm und 11.114 mm — 3 % Unterschied, also fast äquidistant

Also ein **3 × 11 × 11 Tensorgitter, identisch über alle sechzehn OPs**.
`data.Tn` ist `(n_t, 363)` und wird mit *einem* `reshape` zu `(n_t, 3, 11, 11)`.
Kein Vernetzen, kein Interpolieren.

**Layout:** x kommt in die **Kanäle**, gefaltet wird über (y, z), also
`Conv2d` auf `(B, C, 11, 11)`. Drei Ebenen sind keine Faltungsrichtung — man
kann darauf nichts stapeln, und die x-Kopplung ist als 1×1-Kanalmischung
ohnehin ausdrucksstärker als ein 3-Punkt-Stencil.

---

## 2. Was tatsächlich für den CNN spricht

Das Tempo-Argument trägt **nicht** — `rollout()` in `PINNmodulusTwo/model.py`
batcht alle 363 Punkte schon in einen MLP-Forward pro Zeitschritt. Pro Schritt
sind beide grob gleich teuer. Was bleibt:

**a) Die Diffusion wird Architektur statt Strafterm.** Im punktweisen
Koordinaten-MLP gibt es zwischen zwei Nachbarpunkten *keinerlei* Kopplung. Dass
das Feld zusammenhängt, kommt allein aus `L_phys`. Ein 3×3-Kernel auf einem
äquidistanten Gitter **ist** ein FD-Stencil — der Nachbar ist dann eingebaut,
nicht anerzogen.

**b) Statische Karten werden Pflicht, nicht Kür — und das ist gut.** Weil der
Kernel über (y, z) geteilt wird, *kann* der CNN Position nicht auswendig
lernen. Das MLP bekommt (x, y, z) und darf; genau daher kommt ein Teil seiner
guten In-Sample-Zahlen (OP01: 1.000 C) bei mäßiger Verallgemeinerung. Der CNN
muss räumliche Struktur über die Materialkarten begründen. Das ist der
schärfere Prior.

**c) Randbedingungen werden Padding statt Gewicht.** `dT/dx = 0` ist eine
Spiegelung, exakt, ohne `w_bc`. Achse 2 des Fahrplans löst sich auf.

**d) Der Physik-Stencil wird ein fester Kernel.** Kein Autograd-Hessian im
Raum mehr, damit auch `tf32: false` hinfällig, und O8 / der `[CFL WARN]` sind
direkt adressierbar statt über einen Hyperparameter `delta_phys`.

**e) Truncated BPTT wird bezahlbar.** Das ist das eigentliche Speicher-Argument
und der einzige Punkt, an dem der CNN in einer harten Ressource gewinnt —
Begründung in §4.

---

## 3. Der Input

Layout je Zeitschritt: `(C, 11, 11)`, x in den Kanälen.

### 3a. Zustand und Historie — 6 bis 9 Kanäle

| Block | Kanäle | was |
|---|---|---|
| `T_t` | 3 | das normierte Feld jetzt (die 3 x-Ebenen) |
| `T_t − T_{t−Δ1}` | 3 | Rate über kurzem Lag |
| `T_t − T_{t−Δ2}` | 3 | Rate über langem Lag *(optional)* |

Das ist bewusst die **Hybrid-Historie aus `model.py`** (1 Anker + eine Rate je
Lag), nur feldweise statt punktweise. Damit ist der Vergleich zum bestehenden
Modell eine Architekturfrage und keine Featurefrage.

### 3b. Zeitabhängige Feldgrößen — 3 Kanäle

| Block | Kanäle | was |
|---|---|---|
| `Qsrc_t` | 3 | die volumetrische Quelle, `(n_t, 363)` in `data.py` |

Der wichtigste Input nach `T` selbst: die Quelle ist **räumlich strukturiert**
(sie sitzt auf der JR1-Maske), sie ist der Forcing-Term der PDE, und sie liegt
schon punktweise vor.

### 3c. Statische Karten — konstant, ~14 Kanäle

Konstant über Zeit *und* über alle OPs, also einmal berechnet und
durchgereicht. Alles schon in `data._grid_arrays` / `_static_features` da:

| Block | Kanäle | Anmerkung |
|---|---|---|
| α (Temperaturleitfähigkeit) | 3 | z-scored, wie heute |
| λ_xx, λ_yy, λ_zz | 9 | die Anisotropie, je x-Ebene |
| ρ·Cp | 3 | |
| y-, z-Koordinatenkarte | 2 | gegen die Translationsäquivarianz am Rand |
| `region` (CC/JR1/Housing) | **0** | **redundant** — das ist der Kanalindex |

`region` fällt weg, weil x in den Kanälen steckt. Ein Beispiel dafür, dass das
Layout Features spart statt welche zu kosten.

### 3d. Globale Treiber — 18 Skalare

Die `n_config = 7` (`CONFIG_ORDER`) plus `n_forcing = 11` (`q_dot_z` + fünf
Treiber × zwei kausale Rate-Lags) aus `data.py`, unverändert.

**Zwei Wege, sie einzuspeisen:**

1. **Broadcasten** als konstante 11×11-Kanäle. Fünf Zeilen, und bei 121 Pixeln
   kostet es nichts. **Für die erste Version.**
2. **FiLM-Konditionierung**: ein kleines MLP bildet die 18 Skalare auf (γ, β)
   je Feature-Kanal ab, die nach jedem Conv skalieren und verschieben.
   Ausdrucksstärker, kaum mehr Code. **Als erste dokumentierte Erweiterung**,
   damit sie eine eigene Sweep-Achse ist und nicht mit der Architektur vermischt.

**Nicht drin: die absolute Zeit.** Bewusst. Ein `t / T_span`-Kanal lädt das Netz
ein, die Uhr der Trajektorie auswendig zu lernen statt Dynamik — bei elf
Trajektorien wäre das der billigste Weg zu guten In-Sample-Zahlen und die
teuerste Art, OP06 zu verfehlen.

### Summe

| | Kanäle |
|---|---|
| Zustand + Historie | 6–9 |
| Quelle | 3 |
| Statik | 14 |
| Treiber (broadcast) | 18 |
| **Input gesamt** | **41–44** |

Bei 11×11 ist das nichts. Mit FiLM statt Broadcast: 23–26.

---

## 4. Die Rekurrenz

**Ja, geht — und der interessante Teil ist, dass das heutige Training gar keine
echte Rekurrenz trainiert.**

### Was PINNmodulusTwo heute tut

`train.py:772` rollt **unter `torch.no_grad()`** einmal je OP je Epoche aus,
friert die Trajektorie ein, und macht darauf `inner_steps` Minibatch-Updates auf
(t, Punkt)-Paaren. Der Kommentar dort sagt es selbst: die Rekurrenz „detached
its history between steps (truncated BPTT)". Es ist also **Ein-Schritt-Training
gegen die eigene eingefrorene Trajektorie** — DAgger-artig, kein BPTT.

Das ist eine gute Entscheidung gewesen (sie hat die Update-Zahl von 300 auf
`inner_steps × Epochen` gehoben). Aber sie hat eine strukturelle Grenze:

> **Ein Verfahren, das nur Ein-Schritt-Fehler sieht, kann O13 nicht beheben.**
> O13 ist „der Fehler wächst zum Trajektorienende" (OP06: 6.270 C im Mittel,
> 13.248 C spät). Das ist per Definition ein Mehr-Schritt-Fehler. Der Gradient
> sieht ihn nie.

### Warum der CNN das ändern kann

Aktivierungsspeicher je Rollout-Schritt, grob:

* MLP: 4 Layer × 128 Breite × 363 Punkte ≈ **186 k** Floats
* CNN: 4 Layer × 64 Kanäle × 121 Pixel ≈ **31 k** Floats

Rund **6-fach leichter**, plus der Graph der differenzierbaren
Historien-Interpolation (`interp_history`) fällt weg. Damit wird **truncated
BPTT über 50–200 Schritte** bezahlbar, und das ist der direkte Hebel auf O13.
Das ist der einzige Punkt, an dem der CNN in einer harten Ressource gewinnt —
und er zielt auf genau den offenen Punkt, den die heutige Trainingsschleife
nicht erreichen kann.

### Die drei Stufen, in der Reihenfolge

| Stufe | Form | wann |
|---|---|---|
| **1 — Markov-Schritt** | `T_{t+1} = T_t + Δt · f(T_t, Raten, Qsrc_t, Treiber_t, Statik)` | **zuerst.** Exakt parallel zum heutigen Modell, damit der Vergleich eine Architekturaussage ist |
| 2 — ConvGRU | gelernter versteckter Zustand `h_t` statt expliziter Lags | erst wenn 1 steht. Der Zustand ist unbeschränkt und kann driften; bei elf Trajektorien ist das ein echtes Risiko |
| 3 — Neural ODE / Mehrschritt-Integrator | | nicht absehbar nötig |

### Die Δ-Form und warum sie hier gehen sollte

`residual_output` ist in PINNmodulusTwo **aus**, weil es „den Level durch einen
Integrator mit Gain exakt 1 ohne Leck trägt und jeden Rollout weglaufen lässt".
Stufe 1 oben ist genau diese Δ-Form. Der Unterschied: ein Diffusions-Kernel ist
**dissipativ** — er liefert das fehlende Leck von selbst, was ein punktweises
MLP nicht kann.

**Das ist eine Hypothese, keine Messung.** Sie ist mechanistisch begründet und
sie ist billig zu widerlegen: läuft der Rollout weg, war sie falsch. Der
`[SATURATED]`-Zähler aus `train.py` wird mitgenommen, damit man es sieht statt
es zu erraten.

### CFL

Die Δ-Form mit explizitem Stencil ist explizites Euler, also CFL-gebunden.
`config.yaml` steht auf `subsample_time: 2` → dt = 0.2 s bei einem
Δt_max ≈ 0.241 s. Das ist **knapp innerhalb**, mit 20 % Luft. Zwei Anmerkungen:

* `f` ist gelernt, nicht der wahre Stencil — das Netz ist nicht streng an die
  CFL des echten Diffusionsoperators gebunden.
* Für den **Physik-Residuen-Term** gilt sie aber schon. Wenn der Term bei
  dt = 0.2 s getragen werden soll, ist `subsample_time: 1` der sichere Wert.

---

## 5. Randbedingungen — hier steht eine echte Unklarheit

Beim CNN ist die BC ein **Padding**, also eine Zeile Konfiguration statt eines
Gewichts. Deswegen kostet die Unklarheit unten nichts: beide Lesarten sind ein
Flag, und wir können sie gegeneinander messen.

**Was der Code heute tut:** `physics.py:36` heißt
`boundary_condition_loss(...)` und sein Docstring sagt
*„Boundary condition: dT/dx = 0 at cell center (x=0)"*, mit `bc_mask` als
*„boolean mask for boundary points (x=0)"*. Der Strafterm sitzt also **an der
Zellmitte**.

**Was du sagst:** die BC sitzt **an der Gehäusewand**, und an der Zellmitte
wird auf den Rest der Batterie gespiegelt.

Beides ist miteinander verträglich, wenn man sie als zwei verschiedene Dinge
liest, und dann ist das Bild sauber:

| x | was | im CNN |
|---|---|---|
| 0.0 — Cell Center | **Symmetrieebene.** `dT/dx = 0` gilt dort *automatisch*, weil gespiegelt wird — es ist keine Bedingung, die man erzwingt, sondern eine Eigenschaft der Geometrie | **Mirror-Padding.** Exakt, kein Verlustterm |
| 0.0219 — Gehäusewand | **die eigentliche BC**, und du sagst dort gilt ebenfalls `dT/dx = 0` (adiabat nach außen) | **Mirror-Padding** — dieselbe eine Zeile |

Wenn das so stimmt, ist der Fall in x vollständig erledigt: an beiden Enden
gespiegelt, `w_bc` gestrichen, kein Strafterm.

> **⚠ Offene Frage 1 — die physikalische Folge.** Sind *beide* x-Flächen
> nullgradient, kann in x keine Wärme abfließen. Die Kühlung muss dann komplett
> über y/z gehen. Das passt nicht zum `energy_balance_report`: der findet
> **0.5–0.9× und folgt dem Volumenstrom** (V̇ = 0 → 0.9×, V̇ = 0.0026 → 0.5×).
> Bei hohem Fluss verschwindet also die Hälfte der Quellenergie in einer Senke,
> die die PDE in dieser Form gar nicht hat. Siehe §6.

> **⚠ Offene Frage 2 — der y/z-Rand.** Dazu weiß ich nichts, und beim CNN sind
> das **40 von 121 Pixeln** je Ebene — falsches Padding dort ist nicht
> kosmetisch. Vorschlag bis auf Widerruf: `replicate` (nullgradient, die
> neutralste Annahme). Sitzt die Kühlung dort, ist es falsch und wir müssen es
> wissen.

---

## 6. Der fehlende Senkenterm

Fällt beim Aufschreiben des Inputs auf und gehört hierher, weil es die
Architekturfrage berührt:

Die Fluid-Treiber (`fluid_inlet_temp`, `fluid_mass_flow`) gehen heute als
**globale Skalare** ins Netz. Die Kühlung wirkt aber an einer *Fläche*. Und
`energy_balance_report` sagt, dass bei V̇ = 0.0026 rund **die Hälfte** der
Quellenergie nicht in den Temperaturanstieg geht — sie fließt in eine Senke,
die im Residuum (Leitung + Quelle, sonst nichts) nicht vorkommt. Der Datenterm
muss das ausgleichen.

Dazu passt §11.5 des Fahrplans: **der Volumenstrom ist die Schwierigkeitsachse,
nicht der Tier** — V̇ = 0 im Mittel 5.374 C, V̇ > 0 im Mittel 2.928 C, und der
schlechteste ausgehaltene OP ist OP06, der einzige ohne Kühlung.

Auf dem Gitter ist der Term hinschreibbar:

```
q_conv = h(V̇) · (T − T_fluid(t)) · mask_gekühlte_Fläche
```

Das setzt die Fluid-Treiber **dorthin, wo sie physikalisch wirken**, statt sie
gleichverteilt einzustreuen. Ohne Gitter und ohne Flächenmaske ist das mühsam;
mit beidem sind es ein paar Zeilen.

> **⚠ Offene Frage 3:** welche Fläche ist gekühlt? Hängt an Frage 2. `h(V̇)`
> wäre entweder eine kleine gelernte Funktion oder eine Korrelation.

---

## 7. Was von PINNmodulusTwo übernommen wird

Unabhängiges **Modell**, geteilter **Datenpfad**. Das ist Absicht:

`data.py` trägt die teuer erkauften Korrekturen — die 121×-Quelle (§11.1), die
Energiebilanz, die gepoolte Normierung, das anti-aliasing beim
Treiber-Resampling, die vier Reports. Eine Kopie davon driftet weg, und genau
deshalb sind am 31.08. die beiden Vorgängerprojekte zusammengelegt worden. Der
Fehler wird nicht wiederholt.

| übernommen (importiert, nicht kopiert) | neu in `GridCNN/` |
|---|---|
| `data.py` — Laden, Normierung, Reports | `model.py` — der Conv-Kern |
| `op_registry.py` — Split und Tiers | `physics.py` — FD-Stencil statt Autograd |
| `op_metrics.py` — MAE/RMSE/peak/late je OP | `train.py` — Rollout mit truncated BPTT |
| die trivialen Vorhersager | |
| die Abbruch- und Auswahlregeln | |

Der Split bleibt **identisch** (train 11 / val OP06+OP09 / test OP13, OP15,
OP16), sonst ist der Vergleich wertlos.

---

## 8. Vor dem ersten Code: der Rangtest

Steht weiter aus und ist in Minuten erledigt: **POD/SVD auf `Tn`.** 121 Punkte
je Ebene, ein diffusives Feld, getrieben von glatten globalen Skalaren.

* Kommen ~4 Moden auf 99.9 %, ist der Raum trivial und der ganze Fehler sitzt
  in der Zeit-Abbildung. Dann ist ein **5-Moden-ROM** die ehrlichere Antwort als
  ein CNN, und es wäre um Größenordnungen billiger.
* Braucht es 30+ Moden, hat der CNN echte Struktur zu holen.

Das entscheidet nicht, *ob* wir das hier bauen — die Argumente (a)–(e) in §2
hängen nicht daran. Es entscheidet, **wie groß** `f` sein muss, und das ist bei
elf Trajektorien keine Nebenfrage.

---

## 9. Was der CNN nicht repariert

Damit es geschrieben steht und nicht später als Überraschung auftaucht:

* **Elf Trajektorien.** OP06s 6.270 C ist laut §11.5 ein Envelope-Problem
  (O14): „keine Kühlung bei mittlerer Starttemperatur" kommt im Training nicht
  vor. Keine Architektur erfindet Daten.
* **O11 / OP19.** Der Sim-vs-Messung-Abstand wird davon nicht kleiner.
* **Das ~1 K-Ziel.** Wenn der Engpass die Abdeckung ist und nicht die
  Architektur, verschiebt der CNN die Zahl, aber nicht die Ursache.
