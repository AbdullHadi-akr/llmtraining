# Übergabe — Sitzung vom 22.09.2026

> **Wenn du nur eine Sache liest:** Stufe 1 ist abgeschlossen, `U(V̇)` ist
> kalibriert, Stufe 2 ist **gebaut aber nicht gelaufen**. Der nächste Schritt
> ist ein Kommando auf der Maschine mit `data_raw/`, und es steht unten unter
> „Das Nächste".

**Branch:** `claude/nifty-faraday-wupjv2` · **PR:** [#42](https://github.com/AbdullHadi-akr/llmtraining/pull/42) · **Basis:** `main` nach PR #41

---

## 1. Was am Anfang der Sitzung galt — und was sich daran sofort geändert hat

Die Sitzung startete mit der Annahme, PR #41 sei offen und `main` trage noch
die alten Geometriezahlen. **PR #41 war um 09:53 bereits gemergt** (`2cffaea`).
`main` trägt seitdem die korrigierte Spannweite, die zwei reparierten
Schranken, `_assert_shared_geometry` und Ablationsarm D.

---

## 2. Was gemessen wurde

Drei Läufe von `GridCNN/tools/balance_check.py` auf der Maschine mit
`data_raw/` (`10`/`11`/`12_bilanz.txt`), sieben Konstant-Treiber-Trainings-OPs,
drei Flusslevel.

### 2.1 Stufe 1 — abgeschlossen

| Größe | Ergebnis | Tor |
|---|---|---|
| Fluidbilanz `dT_ber / dT_gem` | **1.030 … 1.062** | 🟢 `ghost_hi` steht, kein Halbmodell-Faktor auf dem Fluidpfad |
| `Q_ht/tot` | **0.700 … 0.772** mit Fluss, 0.11 ohne | 🔴 **nicht ≈ 1** — die Hypothese aus 1a ist widerlegt |
| `Q_ht/jr1` | **2.286 … 2.566** | 🟡 flussabhängig, siehe 2.2 |
| `tot/jr1` | **2.99 … 3.44** | ⅓ bis 40 % der Erzeugung liegt außerhalb der Wickel → **O17 belegt** |
| `jr2/jr1` | 1.000 | zwei gleiche Wickel, bestätigt |

### 2.2 `Q_ht/jr1` ist keine Konstante — das Tor war falsch gestellt

| V̇ | `Q_ht/jr1` | `Q_ht/tot` |
|---|---|---|
| 0 | 0.334 / 0.336 | 0.109 / 0.112 |
| 15 | 2.286 / 2.302 / 2.304 | 0.709 / 0.739 / 0.700 |
| 30 | 2.488 / 2.566 | 0.772 / 0.746 |

**Eine Bezugsfläche weiß nicht, wie schnell das Kühlmittel fließt.** Ein
Konventionsfaktor ist Geometrie und damit konstant; dieses Verhältnis ist es
nicht. Das Tor „`Q` halbieren **oder** `A` verdoppeln" setzt eine Konstante
voraus, die es nicht gibt — es ist als Abschnitt **1d** im `FAHRPLAN.md`
umgeschrieben, nicht abgehakt.

`Q_ht/tot` sättigt gegen **~0.8, nicht gegen 1**: auch im späten Fenster wird
noch gespeichert. Das Fenster ist quasistationär genug für einen Vergleich,
die Zelle ist es nicht.

### 2.3 `U(V̇)` ist kalibriert — der Wandterm ist frei

| V̇ [l/min] | ṁ [kg/s] | **`U(T_mittel)`** [W/m²K] | Streuung |
|---|---|---|---|
| 0 | 0 | **50.4** | 2.10 % |
| 15 | 0.0013 | **421.0** | 1.38 % |
| 30 | 0.0026 | **501.7** | 1.09 % |

Die OPs je Level unterscheiden sich in C-Rate, SOC und Starttemperatur und
liegen trotzdem auf 1.1–2.1 % zusammen: **`U` hängt am Fluss und an sonst
nichts.** Das sind genau die drei trainierten Level, also genau die
Stützstellen von `physics.UCurve` — sie interpoliert linear und klemmt
außerhalb. **Null freie Parameter, keine Funktionsform nötig.**

### 2.4 Die Bezugstemperatur war der größere Hebel als die Halbmodellfrage

`physics.UCurve` schreibt die Kalibrierung selbst vor:
`U = Q̇ / (A · (T2_mittel − T_fluid_mittel))` — gegen die **mittlere**
Fluidtemperatur, passend zu `WallModel._advective`, das entlang +y von `T_in`
bis `T_out` marschiert. Das Werkzeug rechnete gegen `T_in`.

| Bezug | OP01 | |
|---|---|---|
| `T_in` | 229.25 | untere Schranke |
| `T_mittel` | **421.66** | das, was `UCurve` will |
| `T_out` | **−8684** | **unbrauchbar** |

⚠ **`U(T_out)` ist keine obere Schranke, sondern unbrauchbar.** Bei V̇ = 15
kommen −8684 / +21933 / −12729 heraus: `Tw` ist ein **Flächenmittel** der
Wand, `T_out` das **heiße Ende** des Kanals, die Differenz wechselt das
Vorzeichen. Bei V̇ = 30 wechselt sie *nicht* und liefert brave 1129.94 /
1132.60 — **und die Default-OPs vom 09.09. waren genau OP04/OP05/OP07/OP14.**
Deshalb sah der Fehler damals wie eine Messung aus. Die „~1130" und die „~50"
aus der Stand-Tabelle sind zurückgezogen, samt „Faktor ~23" aus 1b.

### 2.5 Die Kurvenform bleibt offen — und blockiert nichts

V̇ = 0 ist nach 1b ein **anderer Mechanismus** (stehendes Fluid lädt seine
Wärmekapazität, Modus `capacity`). Für die Advektionsform bleiben **zwei**
Stützstellen, und durch zwei Punkte geht jedes Zweiparametergesetz exakt.
Gemessen: `U ∝ ṁ^0.253`.

**OP16 (V̇ = 90) entscheidet, und der Test ist scharf:**

| Lesart | `U` bei OP16 |
|---|---|
| Potenzgesetz `ṁ^0.253` | **662** |
| Serienwiderstand `ṁ^{-0.8}` | **591** |

12 % auseinander gegen 1.1–1.4 % Streuung. OP16 bleibt **Test**-OP:
Gegenprobe, nie Stützstelle.

### 2.6 Nebenbefund

`A = 0.0206 m²` ist die **y × z-Fläche des Gitters selbst**:
`0.198094368 × 0.104431991 = 0.0206874`. Die benutzte `0.0206` liegt 0.42 %
darunter — unter der Streuung je Level, aber nachzuziehen.

---

## 3. Was gebaut wurde

### 3.1 `balance_check.py` — jede CSV bringt ihre eigene Zeitachse mit

Abschnitt 4 stürzte ab: `np.interp(tw, r["t"], r["t_in"])` →
`fp and xp are not of the same length`. Das Skript liest fünf CSVs je OP und
hatte die Zeitachse von **zwei** mitgelesen; `t_out` (aus `*_Temperaturen.csv`)
und `t_in` (aus `*_Input Signale.csv`, dort ein **Skalar** — die Datei hat gar
keine Zeitspalte) wurden behandelt, als lägen sie auf der Achse von
`*_Heat Source.csv`.

> **In Abschnitt 2 ist es nicht aufgefallen**, weil dort `t_out - t_in` steht
> und numpy Länge 1 still auf Länge N broadcastet. Die Zahl war richtig, aber
> aus dem falschen Grund. **Dieselbe Klasse wie `xyz = raw[0]` in `data.py`:**
> eine Zusage, die stimmt, die aber niemand prüft.

Repariert: `auf_achse()` verlangt, dass man die Achse einer Reihe nennt; ein
Skalar trägt keine und gilt überall; eine Reihe **ohne** Achse wird abgelehnt
statt geraten. `als_konstante()` prüft den Rückfallwert. Abschnitt 4 rechnet
jetzt gegen alle drei Bezugstemperaturen nebeneinander.

### 3.2 Stufe 2 — Schema v3, der Wandpfad im Cache ✅ gebaut, ⬜ nicht gelaufen

**Drei der vier Größen lagen schon im Bundle** — nachgesehen, nicht nachgebaut:

| Größe | liegt in |
|---|---|
| `mdot` | kanonischer Kanal `fluid_mass_flow` → `sim_config` |
| `cp_fluid` | `fluid_props` |
| `total_w` | `q_source[:, 2]` |

**Neu sind zwei**, aus Dateien, die `assemble_op` nie geöffnet hat:

| Größe | Einheit | Quelle |
|---|---|---|
| `wall_ts["q_solid_to_fluid"]` | W | `*_Heat Transfer.csv` |
| `wall_ts["fluid_out_temp"]` | °C | `*_Temperaturen.csv` |

**Jede Reihe trägt ihre eigene Zeitachse.** `meta["wall_ts_axes"]` hält je
Reihe `n`, `t0`, `t1` und `gleich_t_slow` fest — festgehalten, nie
angeglichen. Das ist direkt die Lehre aus 3.1.

`fluid_props` wird **benannt** gelesen statt positionell, Namen in
`fluid_props_names`. Der v2-Vertrag riet die Reihenfolge („typically density,
specific heat, thermal conductivity"); der Export ist **Conductivity, Density,
Specific Heat**. Wer die erste Spalte als Dichte las, rechnete mit
0.42 kg/m³.

`schema_version` **2 → 3**, mit Eintrag in `schema_versions.md` — ohne den
verweigert `save_bundle` das Schreiben.

**Rückwärtskompatibel in die Richtung, die zählt:** ein v2-Bundle lädt weiter,
mit leerem `wall_ts`. Das heißt „noch nicht neu gebaut", nicht „null" — eine
stille Null sähe wie eine Messung aus.

`generate_cache.py`: `--all` baut die **siebzehn** (Liste aus `op_registry`,
nicht aus dem Gedächtnis), `--check` prüft in Sekunden vor, ob jeder OP alle
acht Dateien hat. Der stille Default `OP05 OP06 OP07` ist weg — er hätte beim
Rebuild vierzehn OPs auf dem alten Schema zurückgelassen.

### 3.3 MPS und Parallelität im PINN-Fahrplan

Die Zahlen lagen in `README_GPU_SERVER.md` §6.4, aber der FAHRPLAN ist der
Einstieg und nannte MPS nur als eine Zeile. Jetzt steht dort: **g4dn.2xlarge,
Tesla T4** (15.6 GiB, sm_75), 8 vCPU / 4 physisch, `-j 4` + MPS = **3.69×**,
MPS allein 2.52 ×, und warum das ab O18 zählt (jeder Arm kostet drei Seeds,
`w_phys`/`w_bc` wächst multiplikativ).

⚠ Dabei aufgefallen: §6.4 sagt
`sudo cp PINNmodulusTwo/deploy/nvidia-mps.service …` — **die Datei gibt es
nicht** und kann es nicht geben, weil die `.gitignore` nur `.py`/`.md` und
eine Handvoll Namen durchlässt. Als Warnung im Kasten vermerkt.

---

## 4. Das Nächste

**Auf der Maschine mit `data_raw/`:**

```bash
cd ~/llmtraining
git fetch origin claude/nifty-faraday-wupjv2
git checkout claude/nifty-faraday-wupjv2

# 1. Sekunden, baut nichts -- hat jeder OP alle acht Dateien?
python3 PINNmodulusTwo/generate_cache.py --all --check

# 2. Wenn das grün ist: der Rebuild, 10-30 min, alle SIEBZEHN
python3 PINNmodulusTwo/generate_cache.py --all 2>&1 | tee 13_cache.txt
```

**Das Tor von Stufe 2:** `profile_report`, `coverage_report` und
`energy_balance_report` müssen **exakt dieselben Zahlen** liefern wie vorher.
Also die alten Zahlen **vor** dem Rebuild sichern.

**Danach**, in dieser Reihenfolge:

1. **Ladepfad in `train.py`** anschließen. `train.modell_kwargs(args)`
   übersetzt die Ablationsflags schon an **einer** Stelle — der Ladepfad muss
   sie nur aufrufen. Und `derive_layout` mit `bundle.xn` (float64) × `L_ref`
   füttern, **nicht** mit `op.xn` (float32, halbiert die Reserve der
   Äquidistanzprüfung).
2. **Konfiguration A** fahren — sie ist die Latte für B, C und D.
   `--seeds ≥ 3`, ein Seed ist keine Streuung.
3. Die Ablation **A/B/C/D**. D (`--no-coord-maps`) entscheidet, ob der
   Ortsprior aus README §2b echt ist.

**Für `PINNmodulusTwo` offen:** O18 einbauen, `(∇λ)·(∇T)`. Die Geometrie ist
bestätigt. Der λ-Sprung ist **anisotrop — 256× in x, 8.6× in z**, nicht die
skalaren 167 (die stehen in `tools/make_synthetic_cache.py`, dem Testcache).
Korrektur als Kasten in `PINNmodulusTwo/FAHRPLAN.md`.

---

## 5. Offene Punkte, die in dieser Sitzung aufgefallen sind

| | |
|---|---|
| `A = 0.0206` gegen gemessene `0.0206874` | 0.42 %, unter der Streuung. Nachziehen, wenn `balance_check.py` ohnehin läuft |
| ~~`deploy/nvidia-mps.service` existiert nicht~~ | ✅ **erledigt 22.09.** — Unit liegt im Repo, `.gitignore` um eine Zeile ergänzt, §6.4 funktioniert wie geschrieben. MPS überlebt jetzt den Reboot |
| `GridCNN` kann nicht parallel | kein `sweep.py`, keine Artefakt-Trennung, `--seeds` dreht keine Schleife. Zwölf Ablationsläufe bleiben seriell, bis der Ladepfad steht. Siehe `GridCNN/FAHRPLAN.md`, Kasten bei der Ablation |
| Die Legacy-Suite ist rot | `legacy/.../tests`, `PYTHONPATH=src`: **11 failed / 55 passed / 9 skipped** — **vor** dieser Sitzung genauso, per `git stash` nachgeprüft. Läuft in keiner CI-Stufe |
| `q_source` wird mit `keep` aus `t_fast` indiziert | `data.py:421`. `q_source` liegt auf `t_slow`. Funktioniert nur, solange beide Achsen gleich lang sind — geprüft wird es nicht. Dieselbe Klasse wie 3.1 |
| OP16 als `U(V̇)`-Gegenprobe | 662 gegen 591, 12 % — die Messung steht aus |

---

## 6. Wie du mich in der nächsten Sitzung wieder anwirfst

Neue Session im Repo `AbdullHadi-akr/llmtraining` starten und das hier
schicken:

> Repo AbdullHadi-akr/llmtraining, Branch `claude/nifty-faraday-wupjv2`
> (PR #42). Lies **zuerst** `UEBERGABE_2026-09-22.md` im Wurzelverzeichnis,
> dann `GridCNN/FAHRPLAN.md` (Kopf und Abschnitt 1d/1e) und
> `PINNmodulusTwo/FAHRPLAN.md` (Kopf, O18).
>
> Stand: Stufe 1 ist durch, `U(V̇)` kalibriert. Stufe 2 (Schema v3) ist
> gebaut, aber der Rebuild ist noch nicht gelaufen.
>
> Sag mir **immer nur eine Sache**: was der nächste Schritt ist. Ich fahre
> die Kommandos auf der Maschine mit `data_raw/` und schicke dir die Ausgabe
> als Datei zurück.

**Was du mir dazugeben musst**, weil ich es nicht selbst sehen kann: die
Ausgabedateien der Läufe (`13_cache.txt` usw.) — hochladen genügt, ich lese
sie. `data_raw/` und `data_cache/` liegen nicht im Repo.

**Was gut funktioniert hat und wiederholt werden sollte:** ein Schritt pro
Antwort, das Ergebnis als Datei zurück, und jedes Ergebnis wandert sofort in
`FAHRPLAN.md` (Stand-Tabelle + Erledigt). Ein Haken ohne Zahl ist wertlos.
