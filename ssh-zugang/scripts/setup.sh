#!/usr/bin/env bash
# Einmaliges Einrichten: fragt alles ab, legt Keys an, schreibt ~/.ssh/config.
# Mehrfach aufrufbar -- die vorhandenen Werte sind jeweils die Vorgabe.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

HIER="$(dirname "${BASH_SOURCE[0]}")"

cat <<'KOPF'

  SSH-Zugang zur GPU-Instanz einrichten
  =====================================

  Aufbau:   du  ->  Sprunghost (durch die Firewall)  ->  GPU-Instanz

  Enter uebernimmt jeweils den Wert in den eckigen Klammern.

KOPF

pruefe_proxyjump || true

# --- 1. Sprunghost -----------------------------------------------------------

titel "1/5  Sprunghost -- die Kiste, die von aussen erreichbar ist"
frage JUMP_ALIAS "Kurzname fuer ssh"
frage JUMP_HOST  "Adresse (IP oder Hostname)"
frage JUMP_USER  "Benutzername"
frage JUMP_KEY   "Privater Key"
pruefe_adresse "$JUMP_HOST" || fehler "'$JUMP_HOST' ist keine gueltige Adresse."

# --- 2. Zielhost -------------------------------------------------------------

titel "2/5  GPU-Instanz -- nur ueber den Sprunghost erreichbar"
frage TARGET_ALIAS "Kurzname fuer ssh"
frage TARGET_HOST  "Adresse (aendert sich taeglich -> spaeter 'make ip')"
frage TARGET_USER  "Benutzername"
frage TARGET_KEY   "Privater Key"
frage KEY_COMMENT  "Kommentar im eigenen Key (vorname.nachname)"
pruefe_adresse "$TARGET_HOST" || fehler "'$TARGET_HOST' ist keine gueltige Adresse."

speichere_einstellungen

# --- 3. Sprunghost-Key -------------------------------------------------------

titel "3/5  Sprunghost-Key"
JUMP_PFAD="$(entfalte "$JUMP_KEY")"
if [[ -f "$JUMP_PFAD" ]]; then
    pruefe_keyrechte "$JUMP_KEY"
    gut "liegt schon da: $JUMP_PFAD"
    fingerabdruck "$JUMP_KEY" | sed 's/^/        /' || true
else
    warn "fehlt: $JUMP_PFAD"
    echo "   Diesen Key bekommst du vom Administrator; er wird nicht selbst erzeugt."
    if interaktiv && ja "Jetzt importieren?" j; then
        echo
        echo "   Pfad zu einer Key-Datei angeben, oder leer lassen zum Einfuegen."
        read -r -p "   Datei: " KEY_DATEI < /dev/tty || true
        if [[ -n "${KEY_DATEI:-}" ]]; then
            FILE="$KEY_DATEI" "$HIER/import-key.sh" jump
        else
            "$HIER/import-key.sh" jump
        fi
    else
        info "Spaeter mit:  make import-jump-key FILE=/pfad/zum/key"
    fi
fi

# --- 4. Eigener Key ----------------------------------------------------------

titel "4/5  Eigener Key fuer die GPU-Instanz"
ZIEL_PFAD="$(entfalte "$TARGET_KEY")"
if [[ -f "$ZIEL_PFAD" ]]; then
    pruefe_keyrechte "$TARGET_KEY"
    gut "liegt schon da: $ZIEL_PFAD"
    fingerabdruck "$TARGET_KEY" | sed 's/^/        /' || true
else
    warn "fehlt: $ZIEL_PFAD"
    if interaktiv && ja "Jetzt erzeugen (ed25519)?" j; then
        ssh-keygen -t ed25519 -f "$ZIEL_PFAD" -C "$KEY_COMMENT"
        chmod 600 "$ZIEL_PFAD"; chmod 644 "$ZIEL_PFAD.pub"
        gut "erzeugt: $ZIEL_PFAD"
    else
        info "Spaeter mit:  make keygen"
    fi
fi

# --- 5. Config ---------------------------------------------------------------

titel "5/5  ~/.ssh/config schreiben"
schreibe_config
echo "        Host-Keys werden mit 'StrictHostKeyChecking $(strict_modus)' automatisch"
echo "        angenommen -- das haendische 'yes' beim ersten Verbinden entfaellt."

# --- Abschluss ---------------------------------------------------------------

titel "Was der Administrator jetzt von dir braucht"
echo
if PUB="$(finde_pubkey "$TARGET_KEY" 2>/dev/null)" && [[ -n "$PUB" ]]; then
    echo "   1) Deinen oeffentlichen Key (muss auf die GPU-Instanz):"
    echo
    printf '      %s\n' "$PUB"
    echo
else
    warn "   1) Noch kein oeffentlicher Key da -> make keygen"
fi
echo "   2) Deine aktuelle oeffentliche IP (Firewall-Freigabe fuer den Sprunghost):"
if ADRESSE="$("$HIER/myip.sh" 2>/dev/null)"; then
    printf '      %s\n' "$ADRESSE"
else
    echo "      (nicht ermittelbar -- spaeter: make myip)"
fi

titel "Ab jetzt"
printf '   %-38s %s\n' "ssh $TARGET_ALIAS"                      "direkt auf die GPU"
printf '   %-38s %s\n' "rsync -avz data_cache $TARGET_ALIAS:~/"  "Daten hochschieben"
printf '   %-38s %s\n' "make ip"                                 "wenn sich die IP geaendert hat"
printf '   %-38s %s\n' "make test"                               "Verbindung pruefen"
echo

if interaktiv && ja "Verbindung jetzt testen?" j; then
    echo
    exec "$HIER/test.sh"
fi
