#!/usr/bin/env bash
# Die eigene oeffentliche IP -- das, was der Chef fuer die Firewall-Freigabe
# des Sprunghosts braucht.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for dienst in https://checkip.amazonaws.com https://ifconfig.me/ip https://api.ipify.org; do
    if ADRESSE="$(curl -fsS --max-time 8 "$dienst" 2>/dev/null)"; then
        ADRESSE="$(printf '%s' "$ADRESSE" | tr -d '[:space:]')"
        if pruefe_adresse "$ADRESSE"; then
            echo "$ADRESSE"
            exit 0
        fi
    fi
done

fehler "Konnte die eigene oeffentliche IP nicht ermitteln (kein Netz?). Von Hand: https://checkip.amazonaws.com"
