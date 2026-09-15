#!/usr/bin/env bash
# Taegliche Aufgabe: die IP der GPU-Instanz hat sich geaendert.
#
#   make ip                 -> fragt nach, Enter behaelt die alte
#   make ip IP=1.2.3.4      -> ohne Nachfrage
#   make ip-jump            -> dasselbe fuer den Sprunghost

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

WELCHER="${1:-target}"

case "$WELCHER" in
    target)
        NAME_VAR="TARGET_HOST"; ALIAS="$TARGET_ALIAS"; BESCHRIFTUNG="GPU-Instanz" ;;
    jump)
        NAME_VAR="JUMP_HOST";   ALIAS="$JUMP_ALIAS";   BESCHRIFTUNG="Sprunghost" ;;
    *)
        fehler "Unbekanntes Ziel '$WELCHER' (erlaubt: target, jump)" ;;
esac

ALT="${!NAME_VAR}"

titel "Neue Adresse fuer $BESCHRIFTUNG ($ALIAS)"
echo "   bisher: $ALT"
echo "   (AWS-Konsole -> Instanz -> \"Oeffentliche IPv4-Adresse\")"
echo

NEU="${IP:-}"
if [[ -z "$NEU" ]]; then
    if interaktiv; then
        while :; do
            read -r -p "   Neue Adresse [${ALT}]: " NEU < /dev/tty || true
            NEU="${NEU#"${NEU%%[![:space:]]*}"}"
            NEU="${NEU%"${NEU##*[![:space:]]}"}"
            NEU="${NEU:-$ALT}"
            pruefe_adresse "$NEU" && break
            warn "'$NEU' sieht nicht nach einer IP oder einem Hostnamen aus. Nochmal."
        done
    else
        NEU="$ALT"
    fi
fi

pruefe_adresse "$NEU" || fehler "'$NEU' ist keine gueltige Adresse."

if [[ "$NEU" == "$ALT" ]]; then
    info "Adresse unveraendert ($ALT) -- schreibe die Config trotzdem neu."
else
    printf -v "$NAME_VAR" '%s' "$NEU"
    gut "$BESCHRIFTUNG: $ALT -> $NEU"

    # Der alte Eintrag in known_hosts gehoert zu einer IP, die jetzt jemand
    # anderem gehoeren kann. Weg damit, sonst kommt irgendwann die grosse
    # "REMOTE HOST IDENTIFICATION HAS CHANGED"-Warnung.
    if [[ -f "$BEKANNTE_HOSTS" ]] && [[ -n "$ALT" ]]; then
        if ssh-keygen -R "$ALT" -f "$BEKANNTE_HOSTS" >/dev/null 2>&1; then
            info "Alten known_hosts-Eintrag fuer $ALT entfernt"
        fi
    fi
    speichere_einstellungen
fi

schreibe_config

if [[ -n "${NO_TEST:-}" ]]; then
    exit 0
fi

if interaktiv && ! ja "Verbindung jetzt testen?" j; then
    echo
    info "Gut. Test spaeter mit: make test"
    exit 0
fi

echo
exec "$(dirname "${BASH_SOURCE[0]}")/test.sh"
