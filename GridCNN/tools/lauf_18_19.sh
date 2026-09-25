#!/usr/bin/env bash
# Lauf 18 (Arm B, exakter Schritt) und Lauf 19 (A schlank) parallel unter
# NVIDIA MPS -- und die ganze Auswertung dazu, ohne dass jemand zurueckkommen
# muss.
#
#   nohup bash GridCNN/tools/lauf_18_19.sh > auswertung_18_19.txt 2>&1 &
#
# Ablauf:
#   1. MPS-Daemon an (ohne ihn wechselt der Treiber zwischen den Prozessen
#      hin und her, statt sie nebeneinander zu rechnen: README_GPU_SERVER,
#      "MPS allein bringt 2.52x").
#   2. Lauf 18 und 19 im Hintergrund, gleiches Protokoll wie Lauf 17.
#   3. Waehrenddessen Lauf 17 nachmessen: Halte-OPs und Trainings-OPs
#      (over-/underfit), dann Bilder fuer Lauf 17 -- nach ~10 min fertig.
#   4. Warten auf 18 und 19, dann dieselben Messungen und Bilder fuer alle
#      drei, mit Vergleichsbildern.
#
# Ergebnisse (Repo-Wurzel):
#   17_konfigA_nachgemessen.txt  17_konfigA_insample.txt
#   18_konfigB_exp.txt           18_konfigB_nachgemessen.txt  18_konfigB_insample.txt
#   19_konfigA_schlank.txt       19_konfigA_schlank_nachgemessen.txt  19_..._insample.txt
#   bilder/*.png                 (erst Lauf 17, am Ende alle drei)
#
# Kein `set -e`: scheitert eine Messung, sollen die anderen trotzdem laufen.
# Jeder Schritt meldet seinen Rueckgabewert.

set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
PY=${PY:-python3}
# Drei Prozesse auf vier Kernen: ohne Grenze nimmt jeder torch-Prozess alle
# Kerne fuer seine CPU-Seite, und sie treten sich gegenseitig auf die Fuesse
# (Probelauf 25.09. auf der CPU: 10x langsamer). Die GPU-Arbeit betrifft das
# nicht.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
ART=GridCNN/artifacts
AUS=${AUS:-bilder}
TRAIN_OPS="OP01 OP02 OP03 OP04 OP05 OP07 OP08 OP10 OP11 OP12 OP14"
# Ueberschreibbar nur fuer einen Probelauf auf der CPU -- auf der Maschine
# mit den Daten gilt das Protokoll von Lauf 17 unveraendert.
PROTOKOLL=${PROTOKOLL:-"--seeds 3 --epochs 60 --subsample 2 --inner-steps 25 \
--tbptt-start 20 --tbptt 80 --val-every 2 --device cuda --cache data_cache"}
MESSUNG=${MESSUNG:-"--subsample 2 --device cuda --cache data_cache"}

stempel() { date '+%H:%M:%S'; }
schritt() { echo; echo "[$(stempel)] == $* =="; }
meldung() { echo "[$(stempel)]    $1 -> Rueckgabe $2"; }

messen() {       # messen <artefakte> <prefix> <flags...>
    local verz=$1 pre=$2; shift 2
    $PY GridCNN/tools/nachmessen.py "$@" $MESSUNG --laeufe "$ART/$verz" \
        > "${pre}_nachgemessen.txt" 2>&1
    meldung "${pre}_nachgemessen.txt" $?
    $PY GridCNN/tools/nachmessen.py "$@" $MESSUNG --laeufe "$ART/$verz" \
        --val-ops $TRAIN_OPS --json "$ART/$verz/nachgemessen_insample.json" \
        > "${pre}_insample.txt" 2>&1
    meldung "${pre}_insample.txt" $?
}

schritt "Vorab"
git log --oneline -1
$PY -c "import matplotlib" 2>/dev/null \
    || echo "!! matplotlib fehlt -- '$PY -m pip install matplotlib', sonst keine Bilder"
if pgrep -af "GridCNN/train.py" >/dev/null; then
    echo "!! Es laeuft schon ein GridCNN/train.py:"; pgrep -af "GridCNN/train.py"
fi

schritt "1. MPS"
if pgrep -x nvidia-cuda-mps >/dev/null; then
    echo "MPS laeuft schon."
else
    nvidia-cuda-mps-control -d && echo "MPS gestartet." \
        || echo "!! MPS liess sich nicht starten -- die Laeufe gehen trotzdem, nur langsamer."
fi
pgrep -af nvidia-cuda-mps || true

schritt "2. Lauf 18 und 19 starten"
$PY -u GridCNN/train.py --integrator exp $PROTOKOLL \
    > 18_konfigB_exp.txt 2>&1 &
PID18=$!
$PY -u GridCNN/train.py --no-physics --karten kompakt --treiber film \
    $PROTOKOLL > 19_konfigA_schlank.txt 2>&1 &
PID19=$!
echo "Lauf 18 PID $PID18 -> 18_konfigB_exp.txt"
echo "Lauf 19 PID $PID19 -> 19_konfigA_schlank.txt"

schritt "3. Lauf 17 nachmessen (Gewichte unter $ART/A)"
if [ -f "$ART/A/seed0/model.pt" ]; then
    messen A 17_konfigA --no-physics
    $PY GridCNN/tools/bilder.py "$ART/A" --aus "$AUS" > 17_bilder.txt 2>&1
    meldung "Bilder Lauf 17 -> $AUS/" $?
else
    echo "!! $ART/A/seed0/model.pt fehlt -- Lauf 17 wird nicht nachgemessen."
fi

schritt "4. Warten auf Lauf 18 und 19 (je ~1 h 45 min, parallel eher mehr)"
wait $PID18; meldung "Lauf 18 (18_konfigB_exp.txt)" $?
wait $PID19; meldung "Lauf 19 (19_konfigA_schlank.txt)" $?

schritt "5. Lauf 18 und 19 nachmessen"
messen B-exp 18_konfigB --integrator exp
messen A-kompakt-film 19_konfigA_schlank --no-physics --karten kompakt \
    --treiber film

schritt "6. Bilder fuer alle drei"
LAEUFE=()
for v in A B-exp A-kompakt-film; do
    [ -d "$ART/$v" ] && LAEUFE+=("$ART/$v")
done
$PY GridCNN/tools/bilder.py "${LAEUFE[@]}" --aus "$AUS" > bilder.txt 2>&1
meldung "Bilder -> $AUS/ (Liste in bilder.txt)" $?
tar czf auswertung_17_18_19.tgz ./1[789]_*.txt bilder.txt "$AUS" 2>/dev/null
meldung "Paket zum Hochladen: auswertung_17_18_19.tgz" $?

schritt "FERTIG"
