#!/usr/bin/env bash
# Erzeugt das eigene Schluesselpaar fuer die GPU-Instanz.
# Der oeffentliche Teil geht an den Chef, der private bleibt hier.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

titel "Schluesselpaar fuer die GPU-Instanz"

frage TARGET_KEY   "Pfad fuer den privaten Key"
frage KEY_COMMENT  "Kommentar im Key (vorname.nachname)"

PRIVAT="$(entfalte "$TARGET_KEY")"

if [[ -f "$PRIVAT" ]]; then
    warn "$PRIVAT existiert bereits."
    if ! ja "Ueberschreiben? Der alte Key ist danach weg." n; then
        info "Behalte den vorhandenen Key."
        pruefe_keyrechte "$TARGET_KEY"
        speichere_einstellungen
        exec "$(dirname "${BASH_SOURCE[0]}")/pubkey.sh"
    fi
    rm -f "$PRIVAT" "$PRIVAT.pub"
fi

mkdir -p "$(dirname "$PRIVAT")"
chmod 700 "$(dirname "$PRIVAT")" 2>/dev/null || true

info "ssh-keygen -t ed25519 -f $TARGET_KEY -C \"$KEY_COMMENT\""
echo "   (Passphrase leer lassen ist ok, wenn der Rechner nur dir gehoert --"
echo "    sonst eine setzen und den Key in den ssh-agent laden.)"
echo
ssh-keygen -t ed25519 -f "$PRIVAT" -C "$KEY_COMMENT"

chmod 600 "$PRIVAT"
chmod 644 "$PRIVAT.pub"
gut "Key erzeugt: $PRIVAT (privat) + $PRIVAT.pub (oeffentlich)"

speichere_einstellungen
echo
exec "$(dirname "${BASH_SOURCE[0]}")/pubkey.sh"
