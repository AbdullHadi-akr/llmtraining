# Steckbrief der Betriebspunkte — wer ist wer, und warum

> **Wofür diese Seite da ist:** Wenn du dich fragst *„warum ist OP15 so
> speziell?"*, *„kann ich OP13 nicht einfach ins Training nehmen?"* oder
> *„warum ist OP06 so schwer?"* — dann steht die Antwort hier.
>
> **Was hier NICHT steht:** die Kritik am Versuchsplan. Die hat ihre eigene
> Seite, [`README_OPS.md`](README_OPS.md), und sie ist älter (02.09.) und
> ausführlicher. Diese Seite ist das Nachschlagewerk, jene das Urteil.
>
> **Die Quelle der Wahrheit ist keine von beiden.** Sie ist
> `PINNmodulusTwo/op_registry.py`. Die Tabellen hier sind daraus **erzeugt**,
> nicht abgeschrieben — und sie laufen ohne Daten:
>
> ```bash
> python3 PINNmodulusTwo/op_registry.py
> ```

---

## 1. Die vier Stufen

Jeder OP trägt eine `tier`. Sie ist keine Etikette, sie sagt **wie weit eine
Zahl trägt**:

| Stufe | Name | Bedeutung |
|---|---|---|
| **T0** | `T0-in-time` | Trainings-OP. Bewertet auf der eigenen Zeitachse — in-sample, außer hinter `split_t` |
| **T1** | `T1-interp` | Ausgehalten. Jeder Treiber liegt **innerhalb** des trainierten Bereichs |
| **T2** | `T2-profile` | Ausgehalten. Hat Profile, deren **Typ** im Training vorkommt |
| **T3** | `T3-extrap` | Ausgehalten und **außerhalb** des Envelopes: ein Treiberwert oder ein Profiltyp, den das Training nicht kennt |

Die Ordnung ist eine Leiter. Ein Modell, das auf T1 scheitert, braucht bei T3
gar nicht erst nachgesehen zu werden.

## 2. Der Split — und die drei Regeln dahinter

```
train (11)         OP01 OP02 OP03 OP04 OP05 OP07 OP08 OP10 OP11 OP12 OP14
val   (2)          OP06 OP09          ← hierauf wird GERANKT
test  (3)          OP13 OP15 OP16     ← nur berichtet, nie ausgewählt
```

`op_registry.py` nennt die Regeln ausdrücklich:

1. **Jeder Profil-TYP, auf den ein Selektions-OP angewiesen ist, muss im
   Training vorkommen** — sonst misst die val-Zahl Extrapolation, und die
   Auswahl optimiert auf das Falsche.
2. **Selektion fasst die Extrapolationsstufe nie an.** Eine Konfiguration auf
   OP13/OP15/OP16 zu wählen würde die einzige Evidenz außerhalb des Envelopes
   in eine gefittete Größe verwandeln.
3. **Jeder einzelne von OP01–OP16 wird genau einmal benutzt**, damit nichts
   stillschweigend wegfällt. (Nachgeprüft: stimmt.)

## 3. Das Envelope-Gitter

Starttemperatur gegen Volumenstrom, alle sechzehn OPs.
`*` = Haltemenge · `#` = Testmenge · alles andere ist Training.

| T0 [°C] ↓ / V̇ [l/min] → | **0** | **15** | **30** | **90** | **Profil** |
|---|---|---|---|---|---|
| **0** | OP14 | — | — | — | — |
| **10** | OP07 | — | — | — | — |
| **15** | — | OP02, **OP09\***, OP11 | — | — | — |
| **25** | **OP06\*** | OP01, OP08, OP10, OP12, OP13# | OP04 | OP16# | OP15# |
| **30** | — | OP03 | — | — | — |
| **40** | — | — | OP05 | — | — |

Zwei Dinge springen daraus heraus, und beide sind gemessene Befunde, keine
Vermutungen:

> **Die Nullfluss-Spalte hört bei 10 °C auf.** Das Training kennt keine Kühlung
> nur kalt — OP14 bei 0 °C, OP07 bei 10 °C. **OP06 sitzt bei 25 °C.** Das
> Modell muss den Nullfluss-Fall **15 K über den wärmsten gesehenen Fall
> hinaus** fortschreiben. Das ist **O14**, und es ist keine Datenmenge, sondern
> ein Loch an einer Stelle.

> **Die 15-l/min-Spalte bei 25 °C ist überbelegt** — fünf OPs. Die Achse
> „Volumenstrom" hat im Training drei Stufen (0, 15, 30), und der Test fragt
> nach 90.

---

## 4. Steckbriefe — die fünf OPs, auf die es ankommt

### OP06 — Haltemenge, `T1-interp`
`CC · 2.0 C · T0 = 25 °C · T_fluid = 25 °C · V̇ = 0 l/min`

**Was ihn schwer macht:** das Loch oben. Die Registry selbst notiert
*„the flow=0 regime is trained via OP07/OP14"* — beide aber im Kälteextrem.
Formal Interpolation, praktisch der Rand.

**Stand 22.09.:** 6.571 ± 0.383 °C (GridCNN A), 6.270 °C (PINN, ein Seed).
`README.md:903` ordnet den Rest als **Envelope-Problem** ein: was hier fehlt,
holt keine Architektur.

### OP09 — Haltemenge, `T2-profile`
`CC mit Fluidtemperaturprofil · 2.5 C · T0 = 15 °C · T_fluid = Profil · V̇ = 15 l/min`

**Gut abgedeckt** — und trotzdem schwer, aus einem *anderen* Grund. Die Zelle
(T0 = 15) und der Volumenstrom (15) haben beide Nachbarn im Training, und die
C-Rate 2.5 liegt zwischen den trainierten 2 und 3. Der Profiltyp
„Fluidtemperatur" kommt zweimal im Training vor.

**Aber:** OP09 ist der **einzige** Profil-OP mit `T0 ≠ 25 °C` — und er ist
ausgehalten. In allen elf Trainings-OPs mit konstantem Fluid gilt
`T0 = T_fluid` (Befund 1 in [`README_OPS.md`](README_OPS.md)). Das Modell kann
„die Zelle ist kalt" nicht von „das Kühlmittel ist kalt" trennen, weil es die
beiden nie getrennt gesehen hat. **OP09 ist genau der Fall, der das verlangt.**

> **Zwei Halte-OPs, zwei verschiedene Krankheiten.** OP06 leidet an einer
> Envelope-Lücke (Befund 3), OP09 an einer Konfundierung (Befund 1). Sie
> brauchen verschiedene Behandlungen, und eine gemeinsame val-MAE verdeckt das.

### OP13 — Testmenge, `T3-extrap`
`4.0 C · T0 = 25 °C · T_fluid = Profil · V̇ = 15 l/min · zwei Profile gleichzeitig`

**Zwei Dinge auf einmal neu:** die C-Rate 4.0 liegt über jeder trainierten
(das Training kennt nur 2.0 ×9 und 3.0 ×2), **und** es laufen zwei Profile
gleichzeitig. Fällt OP13 durch, sagt die Zahl nicht, woran es lag.

### OP15 — Testmenge, `T3-extrap` — **der speziellste von allen**
`2.0 C · T0 = 25 °C · T_fluid = Profil · V̇ = Profil · drei Profile gleichzeitig`

Alle anderen fünfzehn OPs haben einen Volumenstrom, der eine **Zahl** ist.
OP15 hat einen, der eine **Funktion der Zeit** ist — und diesen Treibertyp gibt
es im Training **überhaupt nicht**, an keiner Stelle, in keiner Ausprägung.

Das ist kategorisch anders als OP16: dort ist der Wert zu groß (90 statt 30),
hier ist die **Art** neu. Ein Modell kann einen zu großen Wert noch
fortschreiben; einen Treiber, der sich bewegt, wo er immer stillstand, nicht.

Dazu kommen zwei weitere Profile obendrauf (Fluidtemperatur, CC-CV-Strom), also
**drei gleichzeitig** — die höchste Stufe der Profilleiter, und die einzige
Stelle, an der sie das Training verlässt.

> **Deshalb ist OP15 nicht attribuierbar.** Er prüft vier Dinge in einem, und
> [`README_OPS.md`](README_OPS.md) nennt das bei einem Extrapolationstest „die
> teuerste Sorte Sparsamkeit". Erwarte, dass er verliert; das ist Information,
> kein Fehler.

### OP16 — Testmenge, `T3-extrap` — der saubere Gegenentwurf
`CC · 2.0 C · T0 = 25 °C · T_fluid = 25 °C · V̇ = 90 l/min`

Gegen OP01 ändert sich **genau eine Größe**: der Volumenstrom, 90 statt 15.
Sechsfach der häufigste, dreifach der größte trainierte Wert.

**So soll ein Extrapolationspunkt aussehen.** Scheitert OP16, ist der Grund
benannt. Er ist die Kontrollgruppe zu OP13 und OP15.

### Außerhalb des Plansheets: OP17, OP18, OP19
Mess-OPs für den Vergleich mit einer echten Zelle, **nicht** ausgehaltene
Simulationszahlen — sie mischen Modellfehler, Messfehler und den
Sim-zu-Test-Abstand, und nichts trennt die drei.

* **OP17** (Entladung) und **OP18**: **nie simuliert.** Nicht „nicht
  unterstützt" — die Rechnungen wurden nicht gemacht.
* **OP19** (Fahrzyklus): simuliert und vorhanden. Alle sechzehn Plansheet-OPs
  sind **Ladevorgänge**; ein gemischtes Vorzeichen hat das Modell nie gesehen.
  Dass OP19 *schlechter* wird, je besser das Modell auf Ladungen wird, ist das
  erwartete Verhalten einer enger werdenden Extrapolation.

---

## 5. Es gibt keine übrigen OPs

Die naheliegende Idee — *„nimm die Test-OPs einfach ins Training, dann gibt es
mehr Daten"* — kostet mehr, als sie bringt:

| | Gewinn | Preis |
|---|---|---|
| OP13, OP15, OP16 ins Training | +3 Trajektorien (**+27 %**) | die **einzige** Evidenz außerhalb des Envelopes ist weg. Regel 2 oben verbietet es ausdrücklich |
| OP17, OP18 | — | **existieren nicht als Daten** |
| OP19 | +1 | Fahrzyklus — ein Regime, das nirgends sonst vorkommt. Als Trainings-OP eine Achse, kein Zuwachs |

**Mehr Daten heißt hier: neu simulieren, nicht umsortieren.**
[`README_OPS.md`](README_OPS.md) führt die Bestellliste in der richtigen
Reihenfolge; die beiden obersten sind zwei Läufe und beheben die zwei Befunde,
die heute die val-Zahlen dominieren:

1. `T0 = 25`, `T_fluid = 15`, `V̇ = 15` — bricht die T0/T_fluid-Konfundierung
   (das ist OP09s Krankheit)
2. `T0 = 25…30`, `V̇ = 0`, als **Trainings**-OP — schließt O14 (das ist OP06s
   Krankheit)

## 6. Die effektive Stichprobe ist elf

Sechzehn OPs × tausende Zeitschritte klingt nach viel. Innerhalb einer
Trajektorie ist `T(t)` aber eine glatte diffusive Relaxation — zwei benachbarte
Zeitschritte tragen fast dieselbe Information. Was das Modell an
**unabhängigen** Beispielen sieht, sind **elf Trajektorien**.

Das ist die Zahl, die jede Architekturdiskussion bindet, und kein Netz ändert
sie.

---

## Wohin weiter

| | |
|---|---|
| **Urteil über den Plan** | [`README_OPS.md`](README_OPS.md) — die sieben Befunde und die Bestellliste |
| **Die Leiter** | [`FAHRPLAN.md`](FAHRPLAN.md) — was als Nächstes gemessen wird |
| **Warum überhaupt ein CNN** | [`README.md`](README.md) |
| **Die Zahlen von heute** | [`../TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md`](../TRAININGS_BERICHT_2026-09-22_KonfigA_POC.md) |
| **Die Rohtabelle, live** | `python3 PINNmodulusTwo/op_registry.py` |
