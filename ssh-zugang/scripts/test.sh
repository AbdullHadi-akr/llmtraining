#!/usr/bin/env bash
# Prueft die Kette in zwei Stufen: erst der Sprunghost, dann die Instanz
# dahinter. Getrennt, damit man sieht, welche Haelfte klemmt.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

FEHLERDATEI="$(mktemp)"
trap 'rm -f "$FEHLERDATEI"' EXIT

STUFE1_OK=0

titel "Stufe 1: Sprunghost $JUMP_ALIAS ($JUMP_USER@$JUMP_HOST)"
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$JUMP_ALIAS" 'echo "erreichbar als $(whoami)@$(hostname)"' 2>"$FEHLERDATEI"; then
    gut "Sprunghost antwortet"
    STUFE1_OK=1
else
    warn "Sprunghost nicht erreichbar:"
    sed 's/^/       /' "$FEHLERDATEI" >&2
    echo
    echo "   Moegliche Ursachen:"
    echo "     - Key fehlt oder falsche Rechte   -> make status"
    echo "     - IP des Sprunghosts geaendert    -> make ip-jump"
    echo "     - deine oeffentliche IP ist nicht freigeschaltet"
    echo "       -> make myip  und die Adresse dem Chef schicken"
fi

if (( STUFE1_OK == 0 )); then
    exit 1
fi

titel "Stufe 2: GPU-Instanz $TARGET_ALIAS ($TARGET_USER@$TARGET_HOST, via $JUMP_ALIAS)"
if ssh -o BatchMode=yes -o ConnectTimeout=20 "$TARGET_ALIAS" '
        echo "erreichbar als $(whoami)@$(hostname)"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        else
            echo "(nvidia-smi nicht gefunden -- Treiber noch nicht installiert?)"
        fi
    ' 2>"$FEHLERDATEI"; then
    gut "GPU-Instanz antwortet"
    echo
    gut "Alles steht. Ab jetzt reicht:  ssh $TARGET_ALIAS"
    exit 0
fi

warn "GPU-Instanz nicht erreichbar:"
sed 's/^/       /' "$FEHLERDATEI" >&2
echo
echo "   Der Sprunghost geht, also liegt es an der Instanz selbst:"
if grep -qi 'permission denied' "$FEHLERDATEI"; then
    echo "     - 'Permission denied': dein Public Key liegt noch nicht auf der"
    echo "       Instanz. Mit 'make pubkey' anzeigen und dem Chef schicken."
elif grep -qiE 'timed out|timeout|no route' "$FEHLERDATEI"; then
    echo "     - Timeout: die IP stimmt vermutlich nicht mehr -> make ip"
elif grep -qi 'connection refused' "$FEHLERDATEI"; then
    echo "     - 'Connection refused': Instanz laeuft, aber kein SSH-Dienst,"
    echo "       oder es ist die falsche IP -> make ip"
else
    echo "     - IP nicht mehr aktuell?  -> make ip"
    echo "     - Public Key noch nicht hinterlegt?  -> make pubkey"
fi
exit 1
