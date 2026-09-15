# ---------------------------------------------------------------------------
# SSH-Zugang zur GPU-Instanz -- ueber den Sprunghost, weil die Firewall nicht
# direkt durchlaesst.
#
#   make setup   einmal am Anfang
#   make         taeglich, wenn sich die IP der Instanz geaendert hat
#
# Alle Werte (IPs, Benutzer, Key-Pfade) stehen in zugang.env. Die Datei wird
# von `make setup` angelegt und ist nicht im Repo.
# ---------------------------------------------------------------------------

SHELL := /bin/bash
S     := $(CURDIR)/scripts

# Ohne Argument die taegliche Aufgabe: IP nachziehen.
.DEFAULT_GOAL := ip

.PHONY: help ip ip-jump setup config status test ssh keygen pubkey \
        import-jump-key import-key myip forget clean

help: ## Diese Liste anzeigen
	@echo
	@echo "  SSH-Zugang zur GPU-Instanz"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Beispiele:"
	@echo "    make                          IP abfragen und eintragen"
	@echo "    make ip IP=203.0.113.20       IP ohne Nachfrage setzen"
	@echo "    make ssh CMD=\"nvidia-smi\"     einen Befehl auf der Instanz"
	@echo "    make import-jump-key FILE=~/Downloads/systemtest_key"
	@echo

ip: ## Tages-IP der GPU-Instanz aendern (Standardziel)
	@IP='$(IP)' NO_TEST='$(NO_TEST)' $(S)/set-ip.sh target

ip-jump: ## Adresse des Sprunghosts aendern
	@IP='$(IP)' NO_TEST='$(NO_TEST)' $(S)/set-ip.sh jump

setup: ## Einmalige Einrichtung: fragt alles ab, legt Keys an, schreibt die Config
	@$(S)/setup.sh

config: ## ~/.ssh/config aus der zugang.env neu schreiben (fragt nichts)
	@$(S)/write-config.sh

status: ## Zeigen, was eingestellt ist und was noch fehlt
	@$(S)/status.sh

test: ## Verbindung pruefen -- erst Sprunghost, dann Instanz
	@$(S)/test.sh

ssh: ## Auf der GPU-Instanz einloggen (make ssh CMD="nvidia-smi")
	@CMD='$(CMD)' $(S)/ssh.sh

keygen: ## Eigenes Schluesselpaar fuer die Instanz erzeugen
	@$(S)/keygen.sh

pubkey: ## Eigenen oeffentlichen Key anzeigen (den bekommt der Administrator)
	@$(S)/pubkey.sh

import-jump-key: ## Sprunghost-Key ablegen (FILE=... oder einfuegen)
	@FILE='$(FILE)' $(S)/import-key.sh jump

import-key: ## Fertigen Instanz-Key ablegen (FILE=... oder einfuegen)
	@FILE='$(FILE)' $(S)/import-key.sh target

myip: ## Eigene oeffentliche IP (fuer die Firewall-Freigabe)
	@$(S)/myip.sh

forget: ## known_hosts-Eintraege beider Hosts vergessen (nach Neuaufsetzen)
	@$(S)/clean.sh hosts

clean: ## Den erzeugten Block wieder aus ~/.ssh/config entfernen
	@$(S)/clean.sh config
