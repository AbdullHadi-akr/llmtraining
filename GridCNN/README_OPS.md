# Kritik am Versuchsplan OP01–OP16

> Aufgeschrieben am 02.09.2026 beim Entwurf von `GridCNN/`. Alle Zahlen unten
> sind aus `PINNmodulusTwo/op_registry.py` ausgezählt, nicht geschätzt — das
> Skript läuft ohne Daten.
>
> **Das ist keine Kritik an der Modellierungsarbeit.** Der Versuchsplan wurde
> festgelegt, bevor es ein Modell gab, gegen das man ihn hätte prüfen können.
> Was hier steht, ist die Liste dessen, was der Datensatz **nicht hergeben
> kann** — damit niemand wochenlang Gewichte gegen eine Datenlücke tunt.

## Das Plansheet, ausgezählt

| Achse | Werte (alle 16 OPs) | nur die 11 Trainings-OPs |
|---|---|---|
| `c_rate` | 2.0 ×12, 2.5 ×1, 3.0 ×2, 4.0 ×1 | **2.0 ×9, 3.0 ×2** |
| `T0` [°C] | 0, 10, 15 ×3, **25 ×9**, 30, 40 | — |
| `V̇` [l/min] | 0 ×3, **15 ×9**, 30 ×2, 90 ×1, Profil ×1 | **0 ×2, 15 ×7, 30 ×2** |
| Richtung | **16 × Ladung, 0 × Entladung** | dito |
| SOC-Fenster | 10–90 % in allen sechzehn | dito |

---

## Was der Plan gut macht

Zuerst, weil es untergeht:

* **Die Profilleiter ist sauber gebaut.** Ein einzelnes Fluidprofil (OP08/09),
  ein einzelnes Stromprofil (OP10/11), beide zusammen (OP12), und erst dann ein
  neuer Typ obendrauf (OP15). Das ist eine bewusste Steigerung, keine Sammlung.
* **Zwei verschieden geformte Stromprofile**, CC-CV und CC-CV_anode, statt
  einem. Damit ist „das Modell kennt CV-Abregelung" von „das Modell kennt
  *diese eine* CV-Kurve" trennbar.
* **OP16 ist ein sauberer Ein-Faktor-Test.** Gegen OP01 ändert sich genau eine
  Größe, der Volumenstrom. So soll ein Extrapolationspunkt aussehen.
* **Die Temperaturspanne 0–40 °C** deckt echte Extreme ab, nicht nur
  Laborbedingungen.
* **OP01 ist als Referenzpunkt benannt**, nicht bloß der erste in der Liste.

---

## Die Befunde, nach Schwere

### 1. `T0` und `T_fluid` sind vollständig konfundiert — 11 von 11

**In jedem OP mit konstantem Fluid ist die Starttemperatur des Festkörpers
gleich der Fluidtemperatur.** Ausgezählt: 11 von 11.

```
OP01 25/25   OP02 15/15   OP03 30/30   OP04 25/25   OP05 40/40
OP06 25/25   OP07 10/10   OP10 25/25   OP11 15/15   OP14  0/0    OP16 25/25
```

Und die Fluidprofil-OPs helfen nicht: die beiden im **Training** (OP08, OP12)
haben beide `T0 = 25 °C`. Der einzige Profil-OP mit anderem `T0` ist OP09 — und
der ist ein val-OP, auf dem nicht trainiert wird.

> **Folge:** Das Modell kann „die Zelle ist kalt gestartet" nicht von „das
> Kühlmittel ist kalt" unterscheiden. Es sind zwei physikalisch verschiedene
> Dinge — eine Anfangsbedingung und eine Randbedingung — und der Datensatz zeigt
> sie nie getrennt. `solid_initial_temp` und `fluid_initial_temp` sind zwei
> Eingangskanäle, deren Gewichte beliebig gegeneinander verschoben werden
> können, ohne dass der Trainingsverlust es merkt.

Das ist **nicht akademisch**. Im Fahrzeug ist der übliche Fall genau der
getrennte: warme Zelle, kaltes Kühlmittel nach einer Standzeit, oder umgekehrt
nach einer Schnellladung. Das Regime, das die Anwendung am häufigsten sieht,
kommt im Training kein einziges Mal vor.

**Der teuerste Befund der Liste**, und der billigste zu beheben: **ein einziger
zusätzlicher Simulationslauf** mit `T0 = 25`, `T_fluid = 15` würde die Achse
aufbrechen.

> **Und kein Report fängt es.** `coverage_report` meldet Kanäle *ohne* Varianz
> (O5). Zwei Kanäle mit Varianz, die aber perfekt miteinander korrelieren, sind
> ein anderer Fehlermodus — und dafür gibt es heute keine Prüfung. Vorschlag am
> Ende dieser Datei.

### 2. `c_rate` ist im Training fast eine Konstante

**9 der 11 Trainings-OPs fahren `c_rate = 2.0`**, die anderen beiden 3.0. Sonst
nichts.

`c_rate` setzt über `q_dot` die Wärmeerzeugung — es ist der Treiber, an dem das
Modell am meisten hängt. Er hat im Training **zwei Stufen**.

Und **OP13 fragt nach 4.0**. Das ist eine Extrapolation über eine Achse, die
gerade zwei Stützstellen hat. Dass OP13 mit 4.097 °C trotzdem gut aussieht, ist
eher ein Hinweis darauf, dass `c_rate` in diesem Bereich schwach wirkt, als ein
Beleg für Verallgemeinerung.

### 3. Der Volumenstrom hat drei Stufen, und die Lücke dazwischen ist O14

Training: `V̇ = 15` in 7 OPs, `0` in 2, `30` in 2. Der Test fragt nach **90** —
dem Sechsfachen des häufigsten Werts und dem Dreifachen des größten trainierten.

Dazu kommt die bekannte Lücke: die beiden No-Flow-Trainings-OPs sind
Kälteextreme (OP07 bei 10 °C, OP14 bei 0 °C), der ausgehaltene No-Flow-OP (OP06)
fährt 25 °C. **„Keine Kühlung bei mittlerer Starttemperatur" kommt im Training
nicht vor** — das ist O14, und es ist der schlechteste ausgehaltene Wert des
ganzen Laufs.

### 4. Zwei der drei Test-OPs ändern mehr als eine Sache

| OP | was neu ist | attribuierbar? |
|---|---|---|
| OP16 | nur `V̇` | ✅ ja |
| OP13 | `c_rate` 4.0 **und** zwei gleichzeitige Profile | ❌ nein |
| OP15 | ein **ungesehener Profiltyp** (Volumenstrom) **und** zwei weitere Profile | ❌ nein |

Wenn OP13 oder OP15 schlecht abschneidet, sagt die Zahl nicht, woran es lag. Für
einen Extrapolationstest — dessen einziger Zweck es ist, eine Ursache zu
benennen — ist das die teuerste Sorte Sparsamkeit.

### 5. Nur Ladung, nie Entladung — und das macht OP19 unerreichbar

Alle sechzehn sind Ladevorgänge. Der einzige Entladefall im ganzen Plansheet ist
OP17, und der **wurde nie simuliert**. OP19 ist ein Fahrzyklus mit gemischtem
Vorzeichen.

Damit ist O11 keine Modellschwäche, sondern Arithmetik: das Modell hat ein
Regime nie gesehen und wird darin bewertet. Dass OP19 *schlechter* wird, je
besser das Modell auf den Ladungen wird, ist genau das erwartete Verhalten einer
Extrapolation, die enger wird.

### 6. `soc_start` trägt null Information

10–90 % in allen sechzehn OPs. Der Kanal ist konstant, `coverage_report` meldet
`DEAD -> forced to 0`. Das ist O5 und richtig eingeordnet — hier nur, um die
Liste vollständig zu machen: **einer der sieben Konfigurationskanäle ist ein
Nullkanal.**

### 7. Die effektive Stichprobe ist 11, nicht 80 000

Sechzehn OPs × ~7400 Zeitschritte klingt nach viel. Aber innerhalb einer
Trajektorie ist `T(t)` eine glatte diffusive Relaxation — zwei benachbarte
Zeitschritte tragen fast dieselbe Information. Was das Modell an *unabhängigen*
Beispielen sieht, sind **elf Trajektorien**.

Für ein Netz mit ~70–100 k Parametern ist das die eigentliche Zahl. Sie bindet
jede Architekturdiskussion, und `GridCNN` ändert daran nichts.

---

## Was zu bestellen wäre, in dieser Reihenfolge

Der Datensatz ist fix, **weil niemand mehr gerechnet hat** — nicht, weil es
unmöglich wäre. `op_registry.py:161` sagt es über OP17/OP18 ausdrücklich: sie
fehlen, *„BECAUSE the runs have not been done, not because anything about them
is unsupported."*

Falls je wieder simuliert wird, in dieser Reihenfolge:

| # | Lauf | was er öffnet |
|---|---|---|
| **1** | `T0 = 25`, `T_fluid = 15`, `V̇ = 15`, `c_rate 2` | bricht die **T0/T_fluid-Konfundierung** (Befund 1). Ein Lauf, größter Hebel, und er macht zwei tote Eingangskanäle lebendig |
| **2** | `T0 = 25…30`, `V̇ = 0`, `c_rate 2` — als **Trainings**-OP | schließt **O14**. OP06 wird damit ein echter Interpolationsfall statt eines Regimes ohne Beispiel |
| **3** | OP17, die Entladung | macht OP19 überhaupt bewertbar (O11) |
| **4** | ein Trainings-OP bei `c_rate 4` | OP13 hört auf zu extrapolieren (Befund 2) |
| 5 | OP13/OP15 in Ein-Faktor-Varianten zerlegen | macht den Extrapolationstest attribuierbar (Befund 4) |

Punkt 1 und 2 sind zusammen **zwei Läufe** und beheben die beiden Befunde, die
heute die val-Zahlen dominieren.

---

## Ein Vorschlag für den Code

`data.coverage_report` meldet heute Kanäle **ohne** Trainingsvarianz. Befund 1
ist ein anderer Fehlermodus: zwei Kanäle **mit** Varianz, die perfekt
korrelieren. Nichts fängt das heute.

Ein Zusatz von wenigen Zeilen — die Korrelationsmatrix der aktiven
Konfigurationskanäle über den Trainingssatz, mit einer Warnzeile bei `|r| > 0.95`
— hätte die T0/T_fluid-Konfundierung beim ersten `python3 data.py` gemeldet,
statt sie beim Aufschreiben eines Versuchsplans auffallen zu lassen.

Gehört nach `PINNmodulusTwo/data.py`, nicht hierher, und ist nicht Teil dieses
Entwurfs.
