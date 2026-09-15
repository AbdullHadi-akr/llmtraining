#!/usr/bin/env bash
# Was ist gerade eingestellt, und was fehlt noch?

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

zeile() { printf '   %-16s %s\n' "$1" "$2"; }

titel "Einstellungen"
if [[ -f "$ENV_DATEI" ]]; then
    zeile "Quelle" "$ENV_DATEI"
else
    zeile "Quelle" "$ENV_VORLAGE (Vorlage -- noch kein 'make setup' gelaufen)"
fi
echo
zeile "Sprunghost" "$JUMP_USER@$JUMP_HOST   (ssh $JUMP_ALIAS)"
zeile "GPU-Instanz" "$TARGET_USER@$TARGET_HOST   (ssh $TARGET_ALIAS, via $JUMP_ALIAS)"
zeile "Key-Kommentar" "$KEY_COMMENT"

titel "Schluessel"
for paar in "JUMP_KEY:$JUMP_KEY" "TARGET_KEY:$TARGET_KEY"; do
    name="${paar%%:*}"; pfad="${paar#*:}"
    datei="$(entfalte "$pfad")"
    if [[ -f "$datei" ]]; then
        pruefe_keyrechte "$pfad" >/dev/null
        rechte="$(stat -c '%a' "$datei" 2>/dev/null || stat -f '%Lp' "$datei" 2>/dev/null || echo '?')"
        fp="$(fingerabdruck "$pfad" || echo 'Fingerabdruck nicht lesbar')"
        gut "$name  $pfad  (Rechte $rechte)"
        printf '        %s\n' "$fp"
    else
        warn "$name  $pfad  -- fehlt"
        if [[ "$name" == "JUMP_KEY" ]]; then
            printf '        -> make import-jump-key FILE=/pfad/zum/key\n'
        else
            printf '        -> make keygen\n'
        fi
    fi
done

titel "~/.ssh/config"
if [[ -f "$SSH_CONFIG" ]] && grep -Fq "$MARKER_START" "$SSH_CONFIG"; then
    gut "Block vorhanden in $SSH_CONFIG"
    awk -v start="$MARKER_START" -v ende="$MARKER_ENDE" '
        $0 == start { drin = 1; next }
        $0 == ende  { drin = 0 }
        drin && /^(Host|    (HostName|User|IdentityFile|ProxyJump|StrictHostKeyChecking)) / { print "        " $0 }
    ' "$SSH_CONFIG"
else
    warn "Kein Block in $SSH_CONFIG -> make config"
fi

titel "OpenSSH"
zeile "Version" "$(ssh -V 2>&1)"
zeile "Hostkey-Modus" "StrictHostKeyChecking $(strict_modus)"
pruefe_proxyjump && zeile "ProxyJump" "unterstuetzt"

echo
echo "   Naechster Schritt:  make test"
