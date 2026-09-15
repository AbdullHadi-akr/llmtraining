#!/usr/bin/env bash
# Gemeinsame Helfer fuer die Skripte in diesem Ordner. Wird von jedem Skript
# per `source` eingebunden, nicht direkt aufgerufen.

set -euo pipefail

BASIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DATEI="${ENV_DATEI:-$BASIS/zugang.env}"
ENV_VORLAGE="$BASIS/zugang.env.example"

SSH_DIR="${SSH_DIR:-$HOME/.ssh}"
SSH_CONFIG="${SSH_CONFIG:-$SSH_DIR/config}"
BEKANNTE_HOSTS="$SSH_DIR/known_hosts"

MARKER_START="# >>> llmtraining ssh-zugang BEGIN >>>"
MARKER_ENDE="# <<< llmtraining ssh-zugang END <<<"

# --- Ausgabe ----------------------------------------------------------------

if [[ -t 1 ]]; then
    F_ROT=$'\033[31m'; F_GRUEN=$'\033[32m'; F_GELB=$'\033[33m'
    F_BLAU=$'\033[36m'; F_FETT=$'\033[1m'; F_AUS=$'\033[0m'
else
    F_ROT=""; F_GRUEN=""; F_GELB=""; F_BLAU=""; F_FETT=""; F_AUS=""
fi

info()  { printf '%s==>%s %s\n' "$F_BLAU" "$F_AUS" "$*"; }
gut()   { printf '%s  ok%s  %s\n' "$F_GRUEN" "$F_AUS" "$*"; }
warn()  { printf '%s  !!%s  %s\n' "$F_GELB" "$F_AUS" "$*" >&2; }
fehler(){ printf '%sFehler:%s %s\n' "$F_ROT" "$F_AUS" "$*" >&2; exit 1; }
titel() { printf '\n%s%s%s\n' "$F_FETT" "$*" "$F_AUS"; }

# --- .env lesen und schreiben ------------------------------------------------

# Bewusst kein `source`: die Datei wird geparst, nicht ausgefuehrt. Ein
# Tippfehler in der zugang.env soll keinen Befehl starten koennen.
lade_env() {
    local datei="$1" zeile schluessel wert
    [[ -f "$datei" ]] || return 0
    while IFS= read -r zeile || [[ -n "$zeile" ]]; do
        zeile="${zeile%$'\r'}"
        [[ "$zeile" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "$zeile" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
        schluessel="${BASH_REMATCH[1]}"
        wert="${BASH_REMATCH[2]}"
        wert="${wert%"${wert##*[![:space:]]}"}"          # Leerzeichen am Ende weg
        [[ "$wert" == \"*\" ]] && wert="${wert:1:${#wert}-2}"
        [[ "$wert" == \'*\' ]] && wert="${wert:1:${#wert}-2}"
        printf -v "$schluessel" '%s' "$wert"
    done < "$datei"
    return 0
}

# Erst die Vorlage (Defaults), dann die echte Datei (gewinnt), dann die
# Umgebung (gewinnt endgueltig -- so wirkt `IP=... make ip` ohne Nachfrage).
lade_einstellungen() {
    local k
    for k in JUMP_ALIAS JUMP_HOST JUMP_USER JUMP_KEY \
             TARGET_ALIAS TARGET_HOST TARGET_USER TARGET_KEY KEY_COMMENT; do
        printf -v "UMGEBUNG_$k" '%s' "${!k:-}"
    done

    lade_env "$ENV_VORLAGE"
    lade_env "$ENV_DATEI"

    for k in JUMP_ALIAS JUMP_HOST JUMP_USER JUMP_KEY \
             TARGET_ALIAS TARGET_HOST TARGET_USER TARGET_KEY KEY_COMMENT; do
        local aus_umgebung="UMGEBUNG_$k"
        if [[ -n "${!aus_umgebung:-}" ]]; then
            printf -v "$k" '%s' "${!aus_umgebung}"
        fi
    done
    return 0
}

speichere_einstellungen() {
    local temp
    temp="$(mktemp "${ENV_DATEI}.XXXXXX")"
    {
        echo "# Erzeugt von ssh-zugang/. Wird NICHT eingecheckt."
        echo "# Zuletzt geaendert: $(date '+%Y-%m-%d %H:%M:%S')"
        echo
        echo "JUMP_ALIAS=$JUMP_ALIAS"
        echo "JUMP_HOST=$JUMP_HOST"
        echo "JUMP_USER=$JUMP_USER"
        echo "JUMP_KEY=$JUMP_KEY"
        echo
        echo "TARGET_ALIAS=$TARGET_ALIAS"
        echo "TARGET_HOST=$TARGET_HOST"
        echo "TARGET_USER=$TARGET_USER"
        echo "TARGET_KEY=$TARGET_KEY"
        echo
        echo "KEY_COMMENT=$KEY_COMMENT"
    } > "$temp"
    chmod 600 "$temp"
    mv "$temp" "$ENV_DATEI"
    gut "Einstellungen gespeichert: $ENV_DATEI"
}

# --- Eingaben ----------------------------------------------------------------

interaktiv() { [[ -z "${NONINTERACTIVE:-}" ]] && [[ -r /dev/tty ]]; }

# frage VARNAME "Fragetext"  -- der aktuelle Wert ist die Vorgabe, Enter behaelt ihn
frage() {
    local name="$1" text="$2" vorgabe="${!1:-}" eingabe
    if ! interaktiv; then
        printf '   %s: %s\n' "$text" "${vorgabe:-<leer>}"
        return 0
    fi
    read -r -p "   $text [${vorgabe}]: " eingabe < /dev/tty || true
    eingabe="${eingabe#"${eingabe%%[![:space:]]*}"}"
    eingabe="${eingabe%"${eingabe##*[![:space:]]}"}"
    [[ -n "$eingabe" ]] && printf -v "$name" '%s' "$eingabe"
    return 0
}

# ja "Frage?" [j|n]  -- Rueckgabe 0 = ja
ja() {
    local text="$1" vorgabe="${2:-j}" eingabe
    if ! interaktiv; then
        [[ "$vorgabe" == "j" ]]
        return $?
    fi
    local hinweis="[J/n]"; [[ "$vorgabe" == "n" ]] && hinweis="[j/N]"
    read -r -p "   $text $hinweis " eingabe < /dev/tty || true
    eingabe="${eingabe:-$vorgabe}"
    [[ "${eingabe,,}" =~ ^(j|ja|y|yes)$ ]]
}

# --- Pfade und Pruefungen ----------------------------------------------------

entfalte() { local p="$1"; printf '%s' "${p/#\~/$HOME}"; }

pruefe_adresse() {
    local wert="$1"
    [[ -n "$wert" ]] || return 1
    [[ "$wert" =~ [[:space:]] ]] && return 1
    if [[ "$wert" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]]; then
        local i
        for i in 1 2 3 4; do
            (( BASH_REMATCH[i] <= 255 )) || return 1
        done
        return 0
    fi
    # kein IPv4 -> als DNS-Namen durchgehen lassen
    [[ "$wert" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]]
}

ssh_version() {
    ssh -V 2>&1 | sed -n 's/^OpenSSH_\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p'
}

# accept-new nimmt einen unbekannten Hostkey automatisch an -- genau das
# "yes", das man sonst von Hand tippt. Gibt es erst ab OpenSSH 7.6.
strict_modus() {
    local haupt neben
    read -r haupt neben <<< "$(ssh_version)"
    if [[ -z "${haupt:-}" ]]; then
        echo "accept-new"
    elif (( haupt > 7 )) || { (( haupt == 7 )) && (( ${neben:-0} >= 6 )); }; then
        echo "accept-new"
    else
        echo "no"
    fi
}

pruefe_proxyjump() {
    local haupt neben
    read -r haupt neben <<< "$(ssh_version)"
    [[ -z "${haupt:-}" ]] && return 0
    if (( haupt < 7 )) || { (( haupt == 7 )) && (( ${neben:-0} < 3 )); }; then
        warn "Dein OpenSSH ist aelter als 7.3 und kennt kein ProxyJump."
        warn "Update auf eine neuere Version, sonst greift der Sprung nicht."
        return 1
    fi
    return 0
}

# --- ~/.ssh/config schreiben --------------------------------------------------

erzeuge_block() {
    local strict; strict="$(strict_modus)"
    cat <<BLOCK
$MARKER_START
# Erzeugt von ssh-zugang/, zuletzt geaendert $(date '+%Y-%m-%d %H:%M:%S').
# Nicht von Hand aendern -- jedes \`make ip\` / \`make config\` schreibt den Block
# neu. Die Werte stehen in ssh-zugang/zugang.env.

# Sprunghost: die einzige Kiste, die von aussen durch die Firewall darf.
Host $JUMP_ALIAS
    HostName $JUMP_HOST
    User $JUMP_USER
    IdentityFile $JUMP_KEY
    IdentitiesOnly yes
    StrictHostKeyChecking $strict
    ServerAliveInterval 30
    ServerAliveCountMax 4
    TCPKeepAlive yes

# GPU-Instanz: nur ueber den Sprunghost erreichbar. Die HostName-Zeile ist die,
# die sich taeglich aendert -> \`make ip\`.
Host $TARGET_ALIAS
    HostName $TARGET_HOST
    User $TARGET_USER
    IdentityFile $TARGET_KEY
    IdentitiesOnly yes
    ProxyJump $JUMP_ALIAS
    StrictHostKeyChecking $strict
    ServerAliveInterval 30
    ServerAliveCountMax 4
    TCPKeepAlive yes
    ForwardAgent no
$MARKER_ENDE
BLOCK
}

schreibe_config() {
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR" 2>/dev/null || true
    [[ -f "$SSH_CONFIG" ]] || : > "$SSH_CONFIG"

    local rest neu
    rest="$(mktemp)"; neu="$(mktemp)"
    trap 'rm -f "$rest" "$neu"' RETURN

    # alten Block herausschneiden, Rest behalten
    awk -v start="$MARKER_START" -v ende="$MARKER_ENDE" '
        $0 == start { drin = 1; next }
        $0 == ende  { drin = 0; next }
        !drin       { print }
    ' "$SSH_CONFIG" | sed '/./,$!d' > "$rest"

    # Der Block kommt nach OBEN. SSH nimmt pro Option den ersten Treffer -- ein
    # spaeter im File stehendes `Host *` wuerde sonst unsere IdentityFile
    # ueberstimmen.
    erzeuge_block > "$neu"
    if [[ -s "$rest" ]]; then
        echo >> "$neu"
        cat "$rest" >> "$neu"
    fi

    if diff -q -I '^# Erzeugt von ssh-zugang/, zuletzt geaendert ' \
            "$neu" "$SSH_CONFIG" > /dev/null 2>&1; then
        gut "~/.ssh/config war schon aktuell"
        return 0
    fi

    if [[ -s "$SSH_CONFIG" ]]; then
        local sicherung="$SSH_CONFIG.bak.$(date '+%Y%m%d%H%M%S')"
        cp -p "$SSH_CONFIG" "$sicherung"
        info "Sicherung abgelegt: $sicherung"
        # nur die letzten fuenf Sicherungen behalten
        ls -1t "$SSH_CONFIG".bak.* 2>/dev/null | tail -n +6 | while read -r alt; do
            rm -f "$alt"
        done
    fi

    cat "$neu" > "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
    gut "~/.ssh/config geschrieben (Hosts: $JUMP_ALIAS, $TARGET_ALIAS)"
}

# --- Keys ---------------------------------------------------------------------

# Findet den oeffentlichen Key zu einem privaten -- egal ob die .pub-Datei
# daneben liegt oder nicht.
finde_pubkey() {
    local privat; privat="$(entfalte "$1")"
    if [[ -f "$privat.pub" ]]; then
        cat "$privat.pub"
        return 0
    fi
    if [[ -f "$privat" ]] && ssh-keygen -y -f "$privat" < /dev/null 2>/dev/null; then
        return 0
    fi
    return 1
}

fingerabdruck() {
    local datei; datei="$(entfalte "$1")"
    [[ -f "$datei" ]] || return 1
    ssh-keygen -l -f "$datei" 2>/dev/null
}

pruefe_keyrechte() {
    local datei; datei="$(entfalte "$1")"
    [[ -f "$datei" ]] || return 1
    local rechte; rechte="$(stat -c '%a' "$datei" 2>/dev/null || stat -f '%Lp' "$datei" 2>/dev/null || echo "")"
    if [[ -n "$rechte" && "$rechte" != "600" && "$rechte" != "400" ]]; then
        chmod 600 "$datei"
        info "Rechte von $datei auf 600 gesetzt (waren $rechte)"
    fi
    return 0
}
