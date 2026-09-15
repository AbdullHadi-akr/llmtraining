#!/usr/bin/env bash
# Schreibt den ~/.ssh/config-Block aus der zugang.env neu. Fragt nichts.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen
pruefe_proxyjump || true
schreibe_config
