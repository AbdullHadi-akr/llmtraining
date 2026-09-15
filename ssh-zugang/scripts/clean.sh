#!/usr/bin/env bash
# Aufraeumen.
#   clean.sh config -- den erzeugten Block aus ~/.ssh/config entfernen
#   clean.sh hosts  -- die known_hosts-Eintraege beider Hosts vergessen
#   clean.sh all    -- beides

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

WAS="${1:-config}"

entferne_block() {
    if [[ ! -f "$SSH_CONFIG" ]] || ! grep -Fq "$MARKER_START" "$SSH_CONFIG"; then
        info "Kein Block in $SSH_CONFIG -- nichts zu tun."
        return 0
    fi
    local temp; temp="$(mktemp)"
    awk -v start="$MARKER_START" -v ende="$MARKER_ENDE" '
        $0 == start { drin = 1; next }
        $0 == ende  { drin = 0; next }
        !drin       { print }
    ' "$SSH_CONFIG" | sed '/./,$!d' > "$temp"
    cp -p "$SSH_CONFIG" "$SSH_CONFIG.bak.$(date '+%Y%m%d%H%M%S')"
    cat "$temp" > "$SSH_CONFIG"
    rm -f "$temp"
    chmod 600 "$SSH_CONFIG"
    gut "Block aus $SSH_CONFIG entfernt (Sicherung liegt daneben)"
}

vergiss_hosts() {
    [[ -f "$BEKANNTE_HOSTS" ]] || { info "Keine known_hosts-Datei."; return 0; }
    local h
    for h in "$JUMP_HOST" "$TARGET_HOST" "$JUMP_ALIAS" "$TARGET_ALIAS"; do
        [[ -n "$h" ]] || continue
        if ssh-keygen -R "$h" -f "$BEKANNTE_HOSTS" >/dev/null 2>&1; then
            info "known_hosts-Eintrag fuer $h entfernt"
        fi
    done
    gut "Beim naechsten Verbinden werden die Host-Keys neu angenommen."
}

case "$WAS" in
    config) entferne_block ;;
    hosts)  vergiss_hosts ;;
    all)    entferne_block; vergiss_hosts ;;
    *)      fehler "Unbekannt: '$WAS' (erlaubt: config, hosts, all)" ;;
esac

info "Die zugang.env bleibt liegen -- sie loeschst du bei Bedarf von Hand."
