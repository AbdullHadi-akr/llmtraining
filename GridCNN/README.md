# GridCNN — das Feld auf einmal, statt Punkt für Punkt

> **Status: Entwurf, kein Code.** Diese Datei ist die Design-Diskussion.
>
> **02.09. — die Geometriefragen sind beantwortet** (Auskunft aus der Doku, plus
> Nachprüfung im Code). Damit stehen die Randbedingungen und das x-Layout
> (§5), und der Wandterm ist als *die* fehlende Bilanzhälfte bestätigt (§6).
> Offen ist nur noch, wie groß `f` sein muss — dafür der Rangtest in §8.

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

**Und der äußere Ring ist die Domänengrenze.** Der `legacy`-PINN-README sagt
*„Face spans `dy=0.198 m`, `dz=0.104 m`"*; das Raster spannt gemessen
0.198089 m × 0.104441 m — beides exakt die README-Werte auf drei Stellen. Das
Raster liegt also **Kante auf Kante auf der Zellfläche**, nicht irgendwo innen
drin. Damit gilt am Randring eine echte Randbedingung, und Padding ist das
richtige Mittel (§5).

Also ein **3 × 11 × 11 Tensorgitter, identisch über alle sechzehn OPs**.
`data.Tn` ist `(n_t, 363)` und wird mit *einem* `reshape` zu `(n_t, 3, 11, 11)`.
Kein Vernetzen, kein Interpolieren.

**Layout:** gefaltet wird über (y, z) — dort liegen die 11×11 äquidistanten
Punkte. Wie x behandelt wird, ist die eine offene Layout-Frage: als Kanäle
(einfach, aber dann sind die Randbedingungen nicht strukturell) oder als kurze
Achse mit Geisterschichten (beide BCs exakt). **Siehe §5** — die Antwort hängt
an den Randbedingungen und nicht am Geschmack.

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

**c) Die Symmetrie wird Struktur statt Gewicht.** `dT/dx = 0` an der Zellmitte
ist eine Spiegelung -- exakt, ohne `w_bc`, Achse 2 des Fahrplans löst sich auf.
Das gilt aber nur mit der Geisterschicht-Variante aus §5, nicht wenn x einfach
in den Kanälen liegt. Die Gehäusewand ist **kein** Nullgradient und braucht
einen eigenen Term (§6).

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

### 3b. Die Quelle — 0 zeitabhängige Kanäle

`data.py:594` baut sie so:

```python
Qsrc = r["q_dot"][:, None] * q_mask[None, :] * T_span_ref / (rho * Cp * T_sigma)
```

`q_dot(t)` ist **ein Skalar je Zeitschritt** (JR1-Gesamtleistung / V_JR1),
`q_mask` ist die JR1-Ebene, `ρCp` ist statisch. Also:

> **`Qsrc` = Skalar(t) × feste Ortskarte.** Das räumliche Muster ist über die
> ganze Trajektorie konstant, nur die Amplitude bewegt sich.

Damit fällt der Block auf **eine statische Karte** (`q_mask/(ρCp)`) plus den
Skalar `q_dot_z(t)`, der ohnehin schon in `n_forcing` steckt. Keine
zeitabhängigen Feldkanäle.

**Die Folge ist größer als die eingesparten drei Kanäle:** der *einzige*
räumlich strukturierte und gleichzeitig zeitabhängige Input ist `T` selbst.
Alles andere ist statische Karte × globaler Skalar. Das macht den Rangtest in
§8 noch schärfer — wenn nichts Zeitabhängiges räumliche Struktur einträgt,
kann das Feld kaum hochrangig sein.

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

**Zwei davon gehen zusaetzlich woanders hin.** `fluid_inlet_temp` und
`fluid_mass_flow` sind nicht nur globale Skalare -- sie sind die
Gehaeusewand-Randbedingung (§5, §6). Sie werden also *auch* broadcastet, aber
ihre eigentliche Wirkung hat der Wandterm.

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

## 5. Randbedingungen — geklaert (02.09.)

Beantwortet aus der Doku (`legacy/battery_surrogate_agenticWorkflow_PINN/README.md`)
plus Nachpruefung im Code. Alle drei Flaechen sind jetzt entschieden.

| Flaeche | was gilt | im CNN |
|---|---|---|
| x = 0, Zellmitte | **Symmetrieebene.** Der README sagt *„Half-model of a prismatic cell"*, Domaene `Box([0,ymin,zmin],[0.0219,ymax,zmax])`. x = 0 … 0.0219 ist die halbe Zelldicke | `ghost_lo := T1` — Spiegelung, exakt |
| x = 0.0219, Gehaeusewand | **Domaenengrenze, hier tritt Waerme aus.** Kuehlplatten liegen auf den ±x-Flaechen bei x = ±0.0238, also 1.9 mm dahinter | `ghost_hi` aus dem konvektiven Fluss (§6) |
| y/z-Umfang | **adiabat / Symmetrie.** Die Kuehlung sitzt nur auf ±x; die Coolant-Inlets bei y = −0.1265 / +0.14605 liegen im Fluidkanal der Kuehlplatte, nicht an der Zellseitenflaeche. Im Modul liegt seitlich die Nachbarzelle | **`reflect`-Padding** |

Und, weil §1 es misst: der Randring liegt **auf** der Flaechenkante
(0.198089 × 0.104441 m gegen `dy=0.198`, `dz=0.104` im README). Der Randknoten
sitzt also auf der Grenze, nicht davor.

> **Korrektur an der Auskunft.** Die Antwort auf Frage 1 lautete „vermutlich ein
> separates Probe-Raster, nicht die Domaenengrenze", begruendet damit, dass die
> Coolant-Flaechen bei y = −0.1265 / +0.14605 weit ausserhalb des Rasters
> liegen. Das stimmt, ist aber kein Beleg: diese Flaechen gehoeren zum
> **Fluidkanal in der Kuehlplatte**, der als Zu- und Ablauf ueber die Zelle
> hinausragt — nicht zur Zellkontur. Die Zellkontur steht zwei Absaetze weiter
> oben im selben README und stimmt mit dem Raster ueberein.

### Warum `reflect` und nicht `replicate`

Weil der Randknoten **auf** der Grenze liegt und dort `dT/dn = 0` gilt:

* `reflect` spiegelt um den Randknoten, Geist := erster innerer Knoten. Die
  zentrale Differenz ueber den Randknoten wird damit **exakt null**. Das ist die
  uebliche Art, eine Neumann-0-Bedingung auf einem knotenzentrierten Gitter zu
  setzen.
* `replicate` setzt Geist := Randwert. Das ist ein *einseitiger* Nullgradient,
  eine groebere und andere Diskretisierung derselben Bedingung.

Mein Vorschlag aus der letzten Runde (`replicate`) war die vorsichtige Wahl
unter Unwissen. Mit der geklaerten Geometrie ist `reflect` die richtige.

### Das Layout ist damit entschieden: x als kurze Achse

Laege x in den Kanaelen, gaebe es in x kein Padding und **keine** der beiden
x-Bedingungen waere strukturell — die Symmetrie muesste als Strafterm
zurueckkommen, wie heute. Also:

```
x-Stapel, 5 tief:   [ghost_lo,  T0(Mitte),  T1(JR1),  T2(Wand),  ghost_hi]

ghost_lo := T1                                  # Spiegelung an x=0 -- exakt
ghost_hi := T2 - (dx2 / lam_xx) * q_wall        # konvektiver Austritt
q_wall   := h(V_dot) * (T2 - T_fluid(t))

y/z:  reflect-Padding, beide Richtungen
```

Gefaltet wird ueber (y, z); in x laeuft ein expliziter 3-Punkt-Stencil ueber den
gepaddeten Stapel, ein Schritt je Block (drei Ebenen sind zu flach zum Stapeln).

### Zwei Details, die man nicht naiv machen darf

**Der x-Abstand ist nicht aequidistant.** Δx₁ = 10.786 mm (Mitte→JR1),
Δx₂ = 11.114 mm (JR1→Wand), 3 % Unterschied. An Ebene 0 ist der gespiegelte
Stencil *uniform* (Δx₁ beidseitig), an Ebene 2 per Konstruktion auch (Δx₂). Nur
**Ebene 1 braucht die nicht-aequidistante Zweite-Ableitungs-Formel** — die
naive Form `(T₀ − 2T₁ + T₂)/Δx²` mischt dort einen Erste-Ableitungs-Anteil von
~3 % ein.

**λ_XY ist auf JR1 nicht null.** Der README fuehrt fuer JR1 `XX/XY/YY` aus CSV
(nur `XZ = YZ = 0`), und `physics.py:171` kontrahiert entsprechend voll:

```python
aniso = fo00*Txx + fo11*Tyy + fo22*Tzz + 2*(fo01*Txy + fo02*Txz + fo12*Tyz)
```

`Txy` ist eine **gemischte Ableitung ueber die kurze x-Achse und eine
Faltungsachse**. Als FD ist das ein Kreuz-Stencil ueber den ghost-gepaddeten
Stapel — machbar, aber es ist die eine Stelle, an der die FD-Variante fummeliger
ist als der Autograd-Hessian, den sie ersetzt. Nicht vergessen und nicht
stillschweigend weglassen: `fo01` ist genau auf der geheizten Ebene ungleich
null.

## 6. Der Wandterm ist keine Zutat, er ist die fehlende BC

`grep -rin "convect|robin|htc|h_conv|wall.*flux"` ueber `PINNmodulusTwo/*.py`
findet **nichts**. `heat_residual` ist reine Leitung plus Quelle, ohne Senke,
und der einzige BC-Term sitzt an der Symmetrieebene. Der Waermeaustritt an der
Gehaeusewand ist physikalisch voellig unbeschraenkt -- das Netz lernt ihn allein
aus dem Datenterm.

Genau das sieht der `energy_balance_report`: **0.5-0.9x, und es folgt dem
Volumenstrom.** V_dot = 0 -> 0.9x (fast geschlossen, fast adiabat).
V_dot = 0.0026 -> 0.5x, also die halbe Quellenergie verlaesst das System durch
eine Wand, an der die PDE nichts stehen hat.

Auf dem Gitter ist der Term hinschreibbar (§5, `ghost_hi`), und er setzt die
Fluid-Treiber `fluid_inlet_temp` / `fluid_mass_flow` **dorthin, wo sie wirken**,
statt sie als globale Skalare gleichverteilt einzustreuen. Das ist der einzige
Punkt des Entwurfs, an dem der CNN nicht nur anders rechnet, sondern mehr
Physik enthaelt als das bestehende Modell.

> **Was das NICHT erklaert.** §11.5 sagt, V_dot = 0 sei der *schwierigere* Fall
> (Mittel 5.374 C gegen 2.928 C bei V_dot > 0), und der schlechteste
> ausgehaltene OP ist OP06 -- ohne Kuehlung. Waere der fehlende Wandterm die
> Ursache der val-Fehler, muesste es andersherum sein. Es bleibt also bei O14:
> Abdeckung, nicht Physik. Der Wandterm ist eine echte Luecke, aber ihm sind die
> 6.270 C nicht anzuhaengen.

**`h(V_dot)` wird gelernt** -- entschieden, nicht gewaehlt: die Auskunft vom
02.09. sagt, der StarCCM+-interne Waermeuebergangskoeffizient ist nirgends
exportiert und steckt in den Original-Projektdateien ausserhalb dieses Repos.
Es gibt also keine Korrelation zum Nachschlagen. Eine kleine monotone Funktion
(z. B. `softplus(a) * V_dot**b + c`, drei Parameter) ist der sparsamste Ansatz;
bei elf Trajektorien ist jeder freie Parameter einer zu viel, aber drei fuer den
einzigen Senkenpfad des Modells sind vertretbar.

**Und es gibt keinen zweiten Senkenpfad.** Die Kuehlung sitzt nur auf ±x (§5),
y/z sind adiabat. Damit ist `ghost_hi` *der* Weg, auf dem Energie das Gebiet
verlaesst -- was genau zu `energy_balance_report` passt: V_dot = 0 -> 0.9x
(fast nichts geht raus), V_dot = 0.0026 -> 0.5x. Der Term ist nicht eine
Verbesserung unter mehreren, er ist die fehlende Haelfte der Bilanz.

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
