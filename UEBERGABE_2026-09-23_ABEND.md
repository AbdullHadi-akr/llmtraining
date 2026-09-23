# Neustart — Stand 23.09.2026, abends (nach PR #51)

> **Wenn du gar nichts mehr weißt, lies nur diese Seite.** Sie sagt, was heute
> passiert ist, ob sich das Modell geändert hat, und womit du anfängst.
> Unten steht ein Prompt zum Einfügen in eine neue Session.

---

## 1. Worum es geht (30 Sekunden)

Ein kleines CNN (`GridCNN`) sagt das Temperaturfeld einer Batteriezelle über
die Zeit voraus: 3 × 11 × 11 Punkte, Schritt für Schritt, und es füttert sich
dabei selbst. Trainiert wird auf 11 Betriebspunkten (OPs), bewertet auf
**OP06 und OP09**.

Die Güte wird gegen eine triviale Latte gemessen: **unter 1.0x heißt
„gelernt"**. Ein Ergebnis gilt nur mit **3 Seeds** und einer Streuung unter
~1 °C.

Vier Arme werden verglichen: **A** ist die reine Blackbox, **B** hat Physik in
der Architektur, **C** ist B mit Physik-Strafterm, **D** ist B ohne
Koordinatenkarten. Bisher ist **nur A** gemessen.

## 2. Was heute passiert ist

| | was | Ergebnis |
|---|---|---|
| **Lauf 16** | A auf voller Auflösung (dt = 0.2 s) | OP06 0.62x, OP09 **1.11x**, kein Ergebnis. Ursache: das Fenster `k` stand in Schritten statt Sekunden |
| **Lauf 17** | dasselbe, Fenster in Sekunden (`--tbptt-start 20 --tbptt 80`) | OP06 0.62x, OP09 **0.79x**, besser, aber die Streuung liegt bei 1.8–2.0 °C, also **kein Ergebnis** |
| Befund | Fehlerprofil über die Zeit | Das Modell ist **anfangs zu warm und am Ende zu kalt** (Pegelfehler), in allen Seeds |
| Befund | Materialdaten | echt, also ist das CFL-Problem echt: Arm B konnte mit dem alten Rechenschritt nie laufen |
| **PR #48, #50** | Messwerkzeuge (`nachmessen.py`), Warnungen, Berichte | gemerged |
| **PR #51** | drei neue Schalter (siehe unten) | gemerged |

Details: [`TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md`](TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md).

## 3. Hat sich das Modell geändert?

**Ja und nein.** Ohne neue Schalter rechnet der Code **exakt wie vorher**.
Lauf 17 ist also weiter gültig. Neu sind drei **Schalter**:

| Schalter | was er tut | in einem Satz |
|---|---|---|
| `--integrator exp` | rechnet die Physik exakt über jeden Zeitschritt | Damit kann Arm **B zum ersten Mal laufen** |
| `--karten kompakt` | wirft statische Karten weg, die nichts Neues sagen | weniger Gewichte, gleiche Ausdruckskraft |
| `--treiber film` | Treiber (Strom, Fluss, …) wirken in jedem Block statt nur am Eingang | 10 659 statt 11 427 Parameter |

**Noch nicht gebaut:** die „virtuellen OPs" (Physik-Loss dort, wo keine Daten
sind). Dafür braucht es zuerst den Wandterm, siehe
[`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md), Abschnitt „Virtuelle OPs".

## 4. Womit du anfängst — auf der Maschine mit den Daten

**Immer zuerst:**

```bash
git checkout main && git pull
python3 -m pytest GridCNN/tests -q          # erwartet: 157 passed
```

**Schritt 1: over- oder underfit?** Das dauert Minuten und nutzt die Gewichte
von Lauf 17 unter `GridCNN/artifacts/A`:

```bash
python3 GridCNN/tools/nachmessen.py --no-physics --subsample 2 \
    --device cuda --cache data_cache --laeufe GridCNN/artifacts/A \
    --val-ops OP01 OP02 OP03 OP04 OP05 OP07 OP08 OP10 OP11 OP12 OP14 \
    > 17_konfigA_insample.txt 2>&1
```

**Schritt 2: Lauf 18 und Lauf 19.** Beide dauern je ~1 h 45 min, laufen im
Hintergrund und dürfen parallel laufen:

```bash
nohup python3 -u GridCNN/train.py --integrator exp --seeds 3 --epochs 60 \
    --subsample 2 --inner-steps 25 --tbptt-start 20 --tbptt 80 \
    --val-every 2 --device cuda --cache data_cache \
    > 18_konfigB_exp.txt 2>&1 &

nohup python3 -u GridCNN/train.py --no-physics --karten kompakt --treiber film \
    --seeds 3 --epochs 60 --subsample 2 --inner-steps 25 \
    --tbptt-start 20 --tbptt 80 --val-every 2 --device cuda \
    --cache data_cache > 19_konfigA_schlank.txt 2>&1 &
```

Nach einer Minute `head -15` auf beide Logs. Dort muss stehen:
- `[protokoll] k=20->80 (4->16 s)`
- **kein** `!! [fenster]`
- `[artefakte] .../B-exp` bzw. `.../A-kompakt-film`
- Bei Lauf 19 zusätzlich eine Zeile `[karten] kompakt: N von 17 behalten`

**Nicht abbrechen**, bis unten `3 Seed(s)` in der Schlusstafel steht.

## 5. Wo was steht

| Datei | wofür |
|---|---|
| **diese Seite** | Neustart |
| [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md), Kopf | was als Nächstes kommt und warum |
| [`README_MODELLSTAND.md`](README_MODELLSTAND.md) | jede Modellversion und jedes Experiment, chronologisch (GridCNN jetzt **G4.3**) |
| [`TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md`](TRAININGS_BERICHT_2026-09-23_KonfigA_k_sekunden.md) | Lauf 17 und das Fehlerprofil |
| [`UEBERGABE_2026-09-23.md`](UEBERGABE_2026-09-23.md) | Lauf 16, der Vormittag im Detail |
| `GridCNN/laeufe/` | Roh-Logs und Parameterblätter aller Läufe |

## 6. Prompt für die nächste Session

```text
Repo AbdullHadi-akr/llmtraining, Branch main (nach PR #51).
Lies zuerst UEBERGABE_2026-09-23_ABEND.md, dann den Kopf von
GridCNN/FAHRPLAN.md.

Stand: GridCNN Konfiguration A, Lauf 17 (--subsample 2, k 4->16 s):
OP06 0.62x, OP09 0.79x, Streuung 1.8-2.0 C -> kein Ergebnis. Das Modell ist
anfangs zu warm, am Ende zu kalt (Pegelfehler). PR #51 hat drei Schalter
gebracht (--integrator exp, --karten kompakt, --treiber film), Defaults
unveraendert. Virtuelle OPs sind entworfen, nicht gebaut (brauchen erst den
Wandterm).

Ich lade dir hoch: 17_konfigA_insample.txt (over-/underfit), und wenn fertig
18_konfigB_exp.txt (erster Lauf von Arm B) und 19_konfigA_schlank.txt.

Werte aus, dokumentiere (Bericht, Fahrplan, README_MODELLSTAND, Uebergabe),
mach einen PR. Ich fahre die Kommandos auf der Maschine mit den Daten und lade
dir die Ausgabe hoch. Sag mir immer nur eine Sache: was der naechste Schritt
ist.
```
