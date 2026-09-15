#!/usr/bin/env bash
# Legt einen fertigen privaten Key (z.B. den vom Chef geschickten Sprunghost-Key)
# an der richtigen Stelle mit den richtigen Rechten ab.
#
#   make import-jump-key FILE=~/Downloads/systemtest_key
#   make import-jump-key            -> Key einfuegen, Ende mit Strg-D
#
# Der Key wird direkt nach ~/.ssh geschrieben und landet zu keinem Zeitpunkt
# im Repo.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

WELCHER="${1:-jump}"
QUELLE="${2:-${FILE:-}}"

case "$WELCHER" in
    jump)   ZIEL_VAR="JUMP_KEY";   BESCHRIFTUNG="Sprunghost-Key ($JUMP_ALIAS)" ;;
    target) ZIEL_VAR="TARGET_KEY"; BESCHRIFTUNG="Instanz-Key ($TARGET_ALIAS)" ;;
    *)      fehler "Unbekanntes Ziel '$WELCHER' (erlaubt: jump, target)" ;;
esac

ZIEL="$(entfalte "${!ZIEL_VAR}")"

titel "$BESCHRIFTUNG importieren -> $ZIEL"

umask 077
TEMP="$(mktemp)"
trap 'rm -f "$TEMP"' EXIT

if [[ -n "$QUELLE" ]]; then
    QUELLE="$(entfalte "$QUELLE")"
    [[ -f "$QUELLE" ]] || fehler "Datei nicht gefunden: $QUELLE"
    cat "$QUELLE" > "$TEMP"
    info "Gelesen aus $QUELLE"
else
    interaktiv || fehler "Keine Quelle angegeben. Nutze FILE=/pfad/zum/key"
    echo
    echo "   Den privaten Key hier einfuegen (inklusive der BEGIN-/END-Zeilen)."
    echo "   Danach Enter und Strg-D."
    echo
    cat > "$TEMP"
fi

# Kaputte Copy-Paste-Faelle abfangen, bevor sie zu einer halben Stunde
# Fehlersuche werden.
[[ -s "$TEMP" ]] || fehler "Es kam nichts an -- Datei/Eingabe war leer."
if ! grep -q 'BEGIN .*PRIVATE KEY' "$TEMP"; then
    fehler "Das sieht nicht nach einem privaten Key aus (keine BEGIN-Zeile). Abgebrochen."
fi
if grep -q $'\r' "$TEMP"; then
    warn "Windows-Zeilenenden gefunden -- werden entfernt."
    tr -d '\r' < "$TEMP" > "$TEMP.rein" && mv "$TEMP.rein" "$TEMP"
fi
# OpenSSH bricht ohne Zeilenumbruch am Ende ab.
[[ -n "$(tail -c1 "$TEMP")" ]] && echo >> "$TEMP"

if ssh-keygen -y -f "$TEMP" < /dev/null > /dev/null 2>&1; then
    gut "Key ist lesbar und formal in Ordnung"
else
    warn "Key liess sich nicht auslesen -- entweder passphrase-geschuetzt"
    warn "(dann ist alles gut) oder beim Kopieren beschaedigt."
    if ! ja "Trotzdem ablegen?" j; then
        fehler "Abgebrochen."
    fi
fi

mkdir -p "$(dirname "$ZIEL")"
chmod 700 "$(dirname "$ZIEL")" 2>/dev/null || true

if [[ -f "$ZIEL" ]] && ! cmp -s "$TEMP" "$ZIEL"; then
    warn "$ZIEL existiert bereits und ist anders."
    if ! ja "Ueberschreiben?" n; then
        fehler "Abgebrochen -- vorhandener Key bleibt."
    fi
    cp -p "$ZIEL" "$ZIEL.bak.$(date '+%Y%m%d%H%M%S')"
fi

cat "$TEMP" > "$ZIEL"
chmod 600 "$ZIEL"
gut "Abgelegt: $ZIEL (Rechte 600)"

# .pub daneben legen, falls ableitbar -- praktisch fuer spaetere Vergleiche.
if [[ ! -f "$ZIEL.pub" ]] && ssh-keygen -y -f "$ZIEL" < /dev/null > "$ZIEL.pub" 2>/dev/null; then
    chmod 644 "$ZIEL.pub"
    info "Passenden oeffentlichen Key abgeleitet: $ZIEL.pub"
else
    rm -f "$ZIEL.pub" 2>/dev/null || true
fi

if FP="$(fingerabdruck "$ZIEL")"; then
    info "Fingerabdruck: $FP"
fi

echo
warn "Wenn dieser Key jemals durch einen Chat, eine Mail oder ein Ticket"
warn "gelaufen ist, gilt er als kompromittiert. Dann beim Chef einen neuen"
warn "anfordern -- ein privater Key gehoert auf genau eine Festplatte."
