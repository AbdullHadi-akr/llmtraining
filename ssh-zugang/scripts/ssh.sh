#!/usr/bin/env bash
# `make ssh` / `make ssh CMD="nvidia-smi"` -- verbindet mit dem Alias aus der
# zugang.env, ohne dass man ihn kennen muss.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lade_einstellungen

if [[ -n "${CMD:-}" ]]; then
    exec ssh "$TARGET_ALIAS" "$CMD"
fi
exec ssh "$TARGET_ALIAS" "$@"
