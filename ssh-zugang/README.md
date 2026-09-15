# SSH-Zugang zur GPU-Instanz

Die GPU-Instanz hängt hinter einer Firewall und ist nicht direkt erreichbar.
Der Weg führt über einen Sprunghost:

```
   du  ──ssh──▶  Sprunghost  ──ssh──▶  GPU-Instanz
                 (systest)             (pinn)
                 feste IP              IP ändert sich täglich
```

OpenSSH kann das selbst (`ProxyJump`), man muss es nur einmal richtig in
`~/.ssh/config` schreiben. Genau das macht dieser Ordner — plus den einen
Handgriff, der täglich anfällt: die neue IP der Instanz eintragen.

---

## Einmal einrichten

```bash
cd ssh-zugang
make setup
```

Fragt der Reihe nach ab: Kurznamen, Adressen, Benutzernamen, Key-Pfade. Enter
übernimmt jeweils den vorgeschlagenen Wert. Danach

- liegt `~/.ssh/config` mit beiden Hosts und dem `ProxyJump` dazwischen,
- ist ein eigenes Schlüsselpaar erzeugt (falls noch keins da war),
- ist der Sprunghost-Key an der richtigen Stelle mit Rechten `600`,
- steht am Ende, was der Administrator von dir braucht.

Die Antworten landen in `zugang.env`. Die Datei ist **nicht** im Repo — dort
stehen IPs, Benutzernamen und Key-Pfade drin. Im Repo liegt nur
`zugang.env.example` mit Platzhaltern.

---

## Täglich: die IP hat sich geändert

```bash
cd ssh-zugang
make
```

Mehr nicht. `make` ohne Ziel fragt genau eine Sache:

```
Neue Adresse fuer GPU-Instanz (pinn)
   bisher: 203.0.113.20
   (AWS-Konsole -> Instanz -> "Oeffentliche IPv4-Adresse")

   Neue Adresse [203.0.113.20]:
```

IP eintippen, Enter — der Rest läuft durch: `zugang.env` aktualisiert,
`~/.ssh/config` neu geschrieben, der veraltete `known_hosts`-Eintrag der alten
IP entfernt, Verbindung getestet. Enter ohne Eingabe behält die alte Adresse.

Wenn du die IP schon weißt und keine Rückfrage willst:

```bash
make ip IP=203.0.113.20
```

---

## Alle Befehle

| Befehl | tut |
|---|---|
| `make` / `make ip` | Tages-IP der Instanz ändern (Standard) |
| `make ip-jump` | Adresse des Sprunghosts ändern |
| `make setup` | einmalige Einrichtung, fragt alles ab |
| `make config` | `~/.ssh/config` aus `zugang.env` neu schreiben, ohne Rückfrage |
| `make status` | was ist eingestellt, welcher Key fehlt |
| `make test` | Verbindung prüfen — erst Sprunghost, dann Instanz |
| `make ssh` | auf der Instanz einloggen |
| `make ssh CMD="nvidia-smi"` | einen einzelnen Befehl dort ausführen |
| `make keygen` | eigenes Schlüsselpaar erzeugen |
| `make pubkey` | eigenen öffentlichen Key anzeigen |
| `make import-jump-key FILE=…` | Sprunghost-Key ablegen (oder ohne `FILE=` einfügen) |
| `make myip` | eigene öffentliche IP (für die Firewall-Freigabe) |
| `make forget` | `known_hosts`-Einträge vergessen (nach Neuaufsetzen der Instanz) |
| `make clean` | den erzeugten Block wieder aus `~/.ssh/config` entfernen |
| `make help` | diese Liste im Terminal |

---

## Was der Administrator von dir braucht

**1. Deinen öffentlichen Key.** `make pubkey` sucht ihn selbst — erst die
`.pub`-Datei, sonst leitet er ihn aus dem privaten Key ab, sonst listet er
auf, was sonst in `~/.ssh` liegt. Eine Zeile, sieht so aus:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA<...der lange Teil...> vorname.nachname
```

**2. Deine öffentliche IP**, damit der Sprunghost dich durchlässt:

```bash
make myip
```

Die ändert sich bei den meisten Internetanschlüssen auch mal. Wenn der
Sprunghost plötzlich in einen Timeout läuft, ist das der erste Verdacht.

> **Der private Key geht an niemanden.** Nicht per Chat, nicht per Mail, nicht
> ins Ticket. Ist ein privater Key einmal durch so einen Kanal gelaufen, ist er
> verbrannt und gehört ersetzt — auch wenn nichts passiert ist.

---

## Was in `~/.ssh/config` landet

Ein abgegrenzter Block ganz oben in der Datei:

```
# >>> llmtraining ssh-zugang BEGIN >>>
Host systest
    HostName <sprunghost-ip>
    User <benutzer>
    IdentityFile ~/.ssh/systemtest_key
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    …

Host pinn
    HostName <instanz-ip>
    User <benutzer>
    IdentityFile ~/.ssh/pinn-student
    IdentitiesOnly yes
    ProxyJump systest
    StrictHostKeyChecking accept-new
    …
# <<< llmtraining ssh-zugang END <<<
```

Drei Entscheidungen, die dahinterstecken:

- **Der Block kommt nach oben.** OpenSSH nimmt pro Option den *ersten* Treffer.
  Ein `Host *` weiter unten in deiner Config kann unsere `IdentityFile`-Zeile
  dann nicht mehr überstimmen.
- **`StrictHostKeyChecking accept-new`** nimmt einen unbekannten Hostkey
  automatisch an. Das ist das `yes`, das man sonst beim ersten Verbinden von
  Hand tippt. Ein *geänderter* Key bei bekanntem Host wird weiterhin abgelehnt
  — die Warnung, auf die es ankommt, bleibt also scharf. (Bei OpenSSH älter als
  7.6 gibt es `accept-new` nicht, dann wird `no` eingetragen.)
- **Alles außerhalb der Marker bleibt unangetastet.** Vor jedem Schreiben wird
  eine Sicherung `~/.ssh/config.bak.<zeitstempel>` angelegt (die letzten fünf
  bleiben liegen), und `make clean` stellt den Zustand ohne Block wieder her.

Danach funktioniert alles, was SSH benutzt, mit dem Kurznamen:

```bash
ssh pinn
scp ergebnisse.csv pinn:~/
rsync -avz data_cache pinn:~/llmtraining/
```

Zum Datenhochladen siehe [`../PINNmodulusTwo/README_GPU_SERVER.md`](../PINNmodulusTwo/README_GPU_SERVER.md).

---

## Wenn es klemmt

`make test` prüft in zwei Stufen und sagt, welche Hälfte hängt.

| Symptom | Ursache | Abhilfe |
|---|---|---|
| Stufe 1, `Connection timed out` | deine öffentliche IP ist nicht (mehr) freigegeben | `make myip`, Adresse an den Administrator |
| Stufe 1, `Permission denied` | Sprunghost-Key fehlt oder falsche Rechte | `make status`, ggf. `make import-jump-key FILE=…` |
| Stufe 2, `Connection timed out` | Instanz-IP nicht mehr aktuell | `make ip` |
| Stufe 2, `Permission denied` | dein Public Key liegt noch nicht auf der Instanz | `make pubkey`, Zeile an den Administrator |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Instanz wurde neu aufgesetzt | `make forget`, dann `make test` |
| `Bad configuration option: proxyjump` | OpenSSH älter als 7.3 | OpenSSH aktualisieren |

Hilft das nicht weiter, zeigt `ssh -vvv pinn` die komplette Kette inklusive der
Stelle, an der sie abbricht.
