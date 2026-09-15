#!/usr/bin/env bash
# Zeigt den eigenen oeffentlichen Key -- das, was der Chef braucht.
# Sucht ihn selbst: erst die .pub-Datei, sonst aus dem privaten Key ableiten,
# sonst alles in ~/.ssh nach dem passenden Kommentar durchsehen.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

titel "Dein oeffentlicher Key fuer $TARGET_ALIAS"

if PUB="$(finde_pubkey "$TARGET_KEY" 2>/dev/null)" && [[ -n "$PUB" ]]; then
    echo
    echo "$PUB"
    echo
    if FP="$(fingerabdruck "$TARGET_KEY")"; then
        info "Fingerabdruck: $FP"
    fi
    echo
    echo "   Diese eine Zeile an den Chef schicken. Der private Teil"
    echo "   ($(entfalte "$TARGET_KEY")) bleibt bei dir und geht an niemanden."
    exit 0
fi

warn "Zu $TARGET_KEY liegt kein Key (weder privat noch .pub)."

# Notfall: was liegt sonst so in ~/.ssh?
shopt -s nullglob
KANDIDATEN=("$SSH_DIR"/*.pub)
shopt -u nullglob

if (( ${#KANDIDATEN[@]} == 0 )); then
    echo
    echo "   In $SSH_DIR liegt gar kein oeffentlicher Key."
    echo "   -> make keygen  erzeugt einen."
    exit 1
fi

echo
echo "   Gefunden in $SSH_DIR (vielleicht ist einer davon gemeint):"
for datei in "${KANDIDATEN[@]}"; do
    kommentar="$(awk '{ $1=""; $2=""; sub(/^  */, ""); print }' "$datei")"
    printf '     %-40s %s\n' "${datei/#$HOME/~}" "${kommentar:-<ohne Kommentar>}"
done
echo
echo "   Passt einer: TARGET_KEY in ssh-zugang/zugang.env auf den Pfad OHNE"
echo "   .pub-Endung setzen, dann 'make config'. Sonst: make keygen"
exit 1
