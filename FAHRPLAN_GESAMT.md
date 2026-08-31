# Fahrplan bis zum fertigen Modell

> **Update 31.08.2026 — die acht Benchmark-Skripte sind gelöscht.**
> `smallBench.py`, `bench_common.py`, `benchmark_balance.py`,
> `benchmark_arch.py`, `benchmark_wphys_wbc.py`, `smokeBench.py`,
> `profileBench.py`, `bench_profiles.py`. Sie werden Schritt für Schritt neu
> aufgebaut, sobald feststeht, was gemessen werden soll. Die Phasen E0–E5 unten
> bleiben inhaltlich gültig; wo sie ein gelöschtes Skript aufrufen, steht jetzt
> der `train.py`-Aufruf, der dasselbe leistet. E6 und die Profil-Stages sind
> gegenstandslos, bis der Neuaufbau ansteht —
> [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) §4 hat die
> Reihenfolge dafür.

**Stand: 2026-08-31.** Geltungsbereich: **beide** Projekte — `PINNmodulusTwo/`
(Basis, OP01–OP07) und `PINNmodulusTwoExtProfiles/` (Erweiterung, OP01–OP16 mit
Profilen) — vom heutigen Stand bis zu einem abgeschlossenen, zitierfähigen
Ergebnis.

**Verhältnis zu `PINNmodulusTwo/FAHRPLAN.md`:** die Datei bleibt gültig und
bleibt der Einstieg für die Etappen E1–E5. Sie ist bewusst nur auf OP01–OP07
zugeschnitten und endet bei Tor G5. **Diese** Datei ist die Klammer darum: was
davor noch zu tun ist, was nach G5 kommt, und woran „fertig" gemessen wird. Wo
sie ins Detail geht, verweise ich dorthin statt es zu wiederholen.

---

## 0. Was „fertig" heißt

Ohne diese Definition ist jede Etappe unten beliebig lang. Sechs Punkte, alle
prüfbar:

| # | Kriterium | Woran man es sieht |
|---|---|---|
| **F1** | **Zielgenauigkeit dokumentiert** | Eine Zahl in °C, von der Fachseite, als Anforderung des Alterungsmodells. Steht als `README_MODEL_CRITIQUE.md` **O5** seit Wochen offen |
| **F2** | **Modell schlägt beide trivialen Vorhersager** | `train.py` druckt persistence + Trainings-Mittel neben jeder Held-out-MAE; die MAE liegt darunter — auf **echtem**, gehaltenem OP |
| **F3** | **Rollout stabil** | `[SATURATED]` steht in der letzten Epoche bei 0, über die volle Trajektorie, nicht nur 60 Schritte |
| **F4** | **Ergebnis reproduzierbar** | Ein Commit-Hash + eine `config.yaml` + **ein gespeicherter Checkpoint** + eine `metrics.txt`, die zusammengehören |
| **F5** | **Genauigkeit je Tier getrennt** | T1-interp / T2-profile / T3-extrap einzeln berichtet, nie eine Zahl für alles (`op_registry` hält die Tiers; der Bericht muss neu gebaut werden) |
| **F6** | **Grenzen dokumentiert** | Wo das Modell *nicht* gilt: außerhalb des trainierten Envelopes, an Materialgrenzen, bei Profiltypen ohne Trainingsbeispiel |

**F1 ist keine Code-Frage und blockiert F6.** Ohne Zielgenauigkeit kann kein
Messung sagen, ob ein Ergebnis gut ist — 8 °C unterbietet die trivialen Vorhersager
und ist als Eingang eines Alterungsmodells vermutlich trotzdem zu grob. Diese
Zahl bitte **jetzt** besorgen, parallel zu allem anderen; sie kostet dich ein
Gespräch und entscheidet später, ob Etappe E6 zwei Tage oder zwei Wochen dauert.

**F4 ist seit dem 31.08. erfüllbar.** `train.py` schreibt am Ende jedes Laufes
`artifacts/model.pt` — Gewichte plus vollständige `RecurrentField`-Konfiguration
plus `T_mu`/`T_sigma`, also alles, was ein Reload braucht
(`--save-checkpoint ""` schaltet es ab). Vorher lag die einzige `torch.save` in
`bench_common.py`; mit dessen Löschung wäre F4 sonst unerreichbar geworden.
Siehe **E0.3**.

---

## 1. Wo wir jetzt stehen

Aus `PINNmodulusTwo/FAHRPLAN.md` §1, plus dem, was seither dazugekommen ist:

| | Stand | Beleg |
|---|---|---|
| Datenpipeline CSV → NPZ → Training | ✅ läuft | Bericht 28.08. §1 |
| 17 NPZ-Bündel lokal erzeugt (OP01–OP16, OP19) | ✅ | Bericht §1 |
| `A = 118.9 / 29.7`, `dTdt_scale = 2.479` | ✅ verifiziert, SNR > 2000 | Bericht §1 |
| Training läuft ohne Abbruch | ✅ | Bericht §4 |
| Rollout zahm | ❌ 342 saturierte Schritte auf einem **Trainings**-OP | `train.py:674` |
| Loss-Balance | ❌ beide Läufe `FAIL` | Bericht §4 |
| Physik-Term | ❌ kollabiert (`L_phys_bal = 2.7e-06`) | Bericht §6 |
| **Schlägt das Modell „nichts tun"?** | ❓ **unbekannt** | kein Maßstab auf echten Daten |
| Schritt A (A/B gegen alten Stand) | ❌ nie gelaufen | `README_MODEL_CRITIQUE.md:159` |
| Drift-Test | ❌ nie gelaufen | ebd. Z. 206 |
| Grenzflächenanteil gemessen | ❌ nie gelaufen | `ARCHITECTURE.md` 4.1 |
| Modell speicherbar | ❌ **nicht implementiert** | kein `torch.save` in `train.py` |
| Erweiterung OP01–OP16 | ❌ kein einziges Ergebnis gemessen | `PINNmodulusTwoExtProfiles/README.md` |
| PR #18 (Diagnostik) | ✅ gemergt 31.08. | siehe E0.1 |
| Überschreibschutz für den Synthetik-Cache | 🟡 offen in PR #19 | siehe E0.2 |

Die Zeile mit dem Fragezeichen ist weiterhin die wichtigste. **Alles andere zu
optimieren, bevor sie beantwortet ist, ist Blindflug.**

---

## 2. Die Etappen im Überblick

| | Etappe | Läuft wo | Aufwand¹ | Tor |
|---|---|---|---|---|
| **E0** | Aufräumen vor dem ersten Lauf | Repo + lokal | ~1 h | G0 |
| **E1** | Selbsttest + Datenprüfung | lokal, kein GPU | ~30 min | G1 |
| **E2** | Die Latte messen | lokal, kein GPU | Minuten | G2 |
| **E3** | Schritt A: das A/B + Drift-Test | lokal CPU | ~30 min | G3 |
| **E4** | Rollout zahm bekommen | lokal / GPU | Stunden | G4 |
| **E5** | **Die erste Zahl, die zählt** | lokal / GPU | — | **G5** |
| **E6** | Benchmarks auf dem Basisdatensatz | GPU | Tage | G6 |
| **E7** | Erweiterung OP01–OP16 mit Profilen | GPU | Tage | G7 |
| **E8** | Abschluss: festschreiben und berichten | überall | ~1 Tag | F1–F6 |

¹ Alle Angaben außer E0 stammen aus den Projektdokumenten, **nicht nachgemessen**.
Die belastbare Zahl ist die Sekunden-pro-Epoche, die das Trainingslog nach der
ersten Konfiguration druckt — jede ETA sollte von dort kommen, nicht von hier.

**Die eine Regel, die über allem steht:**

> Nichts, was Stunden kostet, bevor das Billige gelaufen ist, das es entwerten
> könnte.

Jede Etappe hat ein Tor. Ist es rot, geht es nicht weiter — dann wird das Tor
repariert, nicht die nächste Etappe gestartet.

---

## E0 — Aufräumen vor dem ersten Lauf

**Warum zuerst:** drei Dinge, die später teuer sind und jetzt eine Stunde
kosten. Punkt E0.2 ist der einzige im ganzen Plan, der Daten vernichten kann.

### E0.1 — ✅ Die Diagnostik ist in `main`

[PR #18](https://github.com/AbdullHadi-akr/llmtraining/pull/18) ist am 31.08.
gemergt und bringt genau die Diagnostik, die E3 und E4 brauchen:

* **`spread_space` / `spread_time` + `[FLAT]`-Zeile** — ein in Raum und Zeit
  konstantes Feld erfüllt Wärmeresiduum **und** Neumann-BC exakt. Erst damit ist
  unterscheidbar, ob `L_phys_bal = 2.7e-06` konvergierte Physik oder die
  triviale Lösung ist. Das ist die offene Frage aus Bericht §6, und sie ist in
  keiner Verlustkurve sichtbar.
* **`div_data` / `div_phys` / `div_bc`** — trennt „Term ist wirklich gefallen"
  von „EMA-Divisor ist aus einem alten Regime stale". Gegenteilige Reaktionen.
* **BC-Masken-`[WARN]`** — bei `n_bc = 0` ist `L_bc` identisch 0 und der Lauf
  fällt den ganzen Weg durch die Balance-Prüfung, ohne dass irgendwo steht, dass
  die BC nie ausgewertet wurde.
* **Modulus-Stub in `selftest.py`** — ohne das kann E1 auf keiner Maschine ohne
  Modulus laufen, obwohl es reine Arithmetik prüft.

### E0.2 — ⚠️ Der Überschreibschutz, **bevor** `main` lokal ausgecheckt wird

`tools/make_synthetic_cache.py:266-268` schreibt mit `np.savez_compressed`
**ungeprüft** in `--out`, und der Default ist das oberste `data_cache/` — genau
der Ordner, in dem deine gemessenen `OP01.npz … OP16.npz` liegen. Das
`material_properties/` ist gegen Überschreiben geschützt (Existenzprüfung +
`--force-materials`), die NPZ-Bündel sind es nicht. Der Ordner ist gitignored,
also nicht über git wiederherstellbar.

Der dokumentierte Copy-Paste-Befehl aus `README_LOKALER_LAUF.md` ist derselbe,
den du auf dem Rechner **mit** den Daten ausführen würdest.

**Zu tun, in dieser Reihenfolge:**

1. **Backup zuerst**, außerhalb des Repos und außerhalb von `data_cache/`:
   ```bash
   cp -r data_cache ~/data_cache_backup_2026-08-31
   ```
2. ✅ **Umgesetzt**, hängt in [PR #19](https://github.com/AbdullHadi-akr/llmtraining/pull/19):
   `measured_bundles()` prüft alle Ziele, **bevor** irgendetwas
   geschrieben wird — auch vor dem Material-Ersatz, damit ein abgelehnter Lauf
   keine halbe Fixture hinterlässt. Ein Bündel ohne `synthetic`-Markierung gilt
   als Messung, ein unlesbares ebenfalls (fail closed). Abbruch statt
   Überspringen: eine Cache-Verzeichnis aus beidem ist laut Docstring des
   Werkzeugs selbst kein Datensatz. Ausweg ist ein anderes `--out`.
3. Erst danach `make_synthetic_cache.py` auf dem Rechner mit den Daten anfassen.
   **`main` trägt das Werkzeug seit #18, den Guard aber erst mit #19** — in dem
   Fenster dazwischen ist der Default-Aufruf scharf.

> Du hast die echten Daten lokal. Für dich ist `make_synthetic_cache.py` ein
> Werkzeug, das du **nie brauchst** — es existiert für Maschinen ohne Daten. Der
> Guard ist trotzdem nötig, weil der Befehl in der Doku steht und der Ordner
> nicht wiederherstellbar ist.

### E0.3 — Checkpoint-Speicherung in `train.py` — ✅ erledigt 31.08.2026

`train.py` schreibt `artifacts/model.pt`: `state_dict`, die vollständigen
`RecurrentField`-Konstruktorargumente (Breite, Tiefe, `k_max`, History-Modus,
`rate_lags` in **normierter** Zeit), `T_mu`/`T_sigma`/`T_span_ref` für die
Rückrechnung in °C, und der Lauf selbst (OPs, Epochen, Seed, ob der Cache
synthetisch war). Damit ist ein Reload
`RecurrentField(**ckpt["model_config"])` + `load_state_dict(..., strict=True)`
und hängt nicht daran, ob `config.yaml` später noch dasselbe sagt.
`tests/test_local_smoke.py::test_checkpoint_round_trips_without_config_yaml`
hält das fest.

Offen bleibt der **Commit-Hash** in der Datei — dafür braucht es einen
`git`-Aufruf aus `train.py` heraus, und der Lauf soll nicht daran scheitern,
dass er außerhalb eines Checkouts startet.

### E0.4 — Zielgenauigkeit anfragen (F1)

Parallel, blockiert nichts, wird aber in E6 gebraucht. Eine Zahl in °C.

**Tor G0:**
- [ ] Backup von `data_cache/` liegt außerhalb des Repos
- [x] PR #18 gemergt — die Diagnostik ist in `main`
- [ ] Guard in `make_synthetic_cache.py` gemergt ([PR #19](https://github.com/AbdullHadi-akr/llmtraining/pull/19))
- [ ] `train.py` schreibt einen Checkpoint mit Config und Commit
- [ ] Anfrage zur Zielgenauigkeit raus

---

## E1 — Selbsttest und Datenprüfung

Entspricht Phase 0 + Phase 1 in `PINNmodulusTwo/FAHRPLAN.md`. Kein GPU, kein
Training.

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate

# Phase 0 -- prueft die Mathematik, nicht das Ergebnis
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/tools/rollout_divergence.py

# Phase 1 -- prueft die Daten
python3 PINNmodulusTwo/tools/data_probe.py
python3 PINNmodulusTwo/tools/interface_probe.py | tee PINNmodulusTwo/artifacts/interface.txt
```

> `generate_cache.py` steht in `FAHRPLAN.md` Phase 1 mit drin, ist bei dir aber
> **erledigt** — die 17 NPZ-Bündel sind laut Bericht §1 am 28.08. erzeugt worden.
> Nur nötig, falls der Cache fehlt oder die Rohdaten sich geändert haben.

**Tor G1:**

| Prüfung | Muss | Sonst |
|---|---|---|
| `A` für `[5, 20]` | ≈ 119 / 30 | Datenaufbereitung stimmt nicht — nicht weiter |
| `dTdt_scale` | ≈ 2.479 | dito |
| SNR | > 100 | der kurze Rate-Kanal misst Rauschen |
| **Grenzflächenanteil** | **notieren** | entscheidet in E6 über `ARCHITECTURE.md` 4.1 Option A vs. B |
| `RMS(∇λ·∇T) / RMS(λ:∇²T)` | notieren | unter ~1 % ist der fehlende Term im Regionsinneren irrelevant |

Die letzten beiden Zeilen sind neu gegenüber dem 28.08. und sind der Grund,
warum `interface_probe.py` überhaupt existiert. **Diese Zahlen jetzt notieren**,
auch wenn sie erst in E6 gebraucht werden — der Lauf kostet Minuten und ist
später mitten im Sweep unbequem.

---

## E2 — Die Latte

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 \
        --val-ops OP06 --test-ops OP07 | tee latte.txt
```

Zwei Epochen reichen: die Latte hängt nicht am Modell, sondern nur an den Daten.

Interessant ist **nicht** die MAE des Modells, sondern was in Klammern daneben
steht:

```
  [val ] OP06: MAE=?? C  (beats|LOSES TO the trivial baselines:
                          persistence=?? C, train-mean=?? C)
  [test] OP07: MAE=?? C  (...)
```

**Tor G2:** die Latte steht als Zahl fest, gemessen auf dem **echten** OP07.

> Die synthetischen 11.96 / 6.60 aus `README_ERSTER_TEST.md` sind **nicht** diese
> Zahl und dürfen sie nicht ersetzen. Der schärfere der beiden Vorhersager ist
> der konstante Mittelwert, nicht die Persistenz — an dem muss sich das Modell
> messen lassen.

Alles ab hier wird gegen diese eine Zahl gelesen. Trag sie in
`README_ERSTER_TEST.md` Kapitel 6 ein und entferne dabei den Synthetik-Hinweis.

---

## E3 — Schritt A: das A/B und der Drift-Test

Der Lauf, den `README_MODEL_CRITIQUE.md:159` „den wichtigsten im ganzen
Dokument" nennt und der bis heute nicht gemacht wurde.

```bash
# neuer Stand
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
                                                     | tee A_neu.txt

# alter Stand als Baseline
python3 PINNmodulusTwo/train.py --epochs 10 --val-ops OP06 --test-ops OP07 \
        --inner-steps 1 --learn-gains --loss-balance legacy \
                                                     | tee A_alt.txt

# Drift-Test -- pred_OP07.npz schreibt der zweite Lauf oben schon, weil OP07
# als --test-ops mitlaeuft
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP07.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
frueh, spaet = e[1:n//5].mean(), e[-(n//5):].mean()
print(f'frueh {frueh:.3f} C  spaet {spaet:.3f} C  Wachstum {spaet/frueh:.2f}x')
"
```

> ⚠️ **Beide A/B-Läufe schreiben in dieselbe `artifacts/metrics.txt`** — der
> zweite überschreibt den ersten, ebenso `artifacts/pred_OP07.npz`. Das `tee`
> oben fängt stdout ab und genügt für den Vergleich; wer die Dateien selbst
> braucht, kopiert sie zwischen den Läufen weg (der Drift-Test liest
> `pred_OP07.npz` des **zuletzt** gelaufenen Standes).

**Tor G3 — drei Ablesungen, drei Entscheidungen:**

**(a) Der A/B-Vergleich**, `Test MAE` gegeneinander:

| Ergebnis | Heißt | Dann |
|---|---|---|
| neu deutlich besser | Unterversorgung war der Engpass | E4 |
| neu ≈ Baseline | Hauptdiagnose war falsch — Kapazität, History-Struktur oder O4 | Architektur zuerst (E6b), Gewichte später |
| neu schlechter | Overfitting durch 100× Updates (O3), oder der Residual-Ausgang schadet | `--no-residual-output` einzeln testen, um beide Ursachen zu trennen |
| beide > 20 °C | lernt grundsätzlich nicht | **Daten prüfen, nicht Hyperparameter** — zurück zu E1 |

**(b) Der Drift-Faktor:**

| Wachstum | Deutung | Konsequenz |
|---|---|---|
| nahe 1 | kein Drift, der Fehler ist Bias | **O1 lohnt nicht** — BPTT-Fenster spart man sich |
| 1,5–3 | moderate Drift | O1 ist ein sinnvoller Hebel in E6 |
| > 3 | Drift dominiert | **O1 vorziehen**, vor jedem Gewichte-Sweep |

**(c) Neu, dank PR #18: die `spread`-Zeile und `[FLAT]`.**
Fällt `spread_space` oder `spread_time` unter 0.2, ist der `L_phys`-Kollaps aus
Bericht §6 **die triviale Lösung** — ein flaches Feld, das Residuum und BC
gratis erfüllt. Dann ist die Antwort *nicht* „mehr Physik", sondern `--w-phys`
und `--w-bc` runter und prüfen, ob der Datenterm den Optimierer überhaupt
erreicht. Bleiben beide Werte nahe 1, ist der Kollaps echte Konvergenz und die
Hypothese aus `ARCHITECTURE.md` 4.1 („Zug zur Überglättung") wird in E6
relevant.

Außerdem im Baseline-Lauf ablesen: `src_gain(final)` und `diff_gain(final)` aus
`artifacts/metrics.txt`. Nahe 0 ist der degenerierte Fall — `L_phys` wurde durch
Abschalten der Physik minimiert.

**Vier Dateien fallen hier an**, und sie sind genau das, was ich brauche, um
weiterzurechnen statt zu raten: `latte.txt`, `A_neu.txt`, `A_alt.txt`,
`interface.txt`.

---

## E4 — Rollout zahm bekommen

**Tor G4: `[SATURATED]` steht in der letzten Epoche bei 0.**

Aktuell 342, und zwar auf einem *Trainings*-OP. `train.py:674` sagt dazu
wörtlich: *„it is not a prediction, and a run that only survives because of this
is not trained."* Ein Sweep über Konfigurationen, deren Rollout wegläuft, rankt
Guard-Verhalten, nicht Physik.

Hebel **in dieser Reihenfolge**, eine Änderung nach der anderen:

1. mehr Epochen
2. `lr` runter
3. `rollout_clamp` prüfen
4. `A` senken über längere `rate_lags`

Kommt aus E3(b) ein Drift-Faktor > 3, gehört **O1** (BPTT-Fenster 20–50 Schritte)
hier vorgezogen — es ist die einzige Änderung, die Fehlerakkumulation *direkt*
adressiert. Der Loss bestraft sie heute nirgends.

---

## E5 — Die erste Zahl, die zählt

Keine eigene Arbeit, sondern der Moment, in dem E2–E4 sich auszahlen.

**Tor G5: ein Lauf unterbietet die Latte aus E2.**

Ab hier ist das Modell zum ersten Mal mehr wert als „nichts tun". Vorher ist
jede Rangfolge zwischen Konfigurationen eine Rangfolge zwischen Verlierern.

**Das ist der Punkt, an dem sich der GPU-Server lohnt — vorher kostet er nur
Geld.** Und es ist der Punkt, an dem sich das Zwischenergebnis berichten lässt:
Datum, Commit, Config, Latte, Test-MAE. Ein Satz, der vorher nicht sagbar war.

---

## E6 — Benchmarks auf dem Basisdatensatz

Erst jetzt. Vorher hat kein Sweep eine Aussage.

### E6.0 — Vorher: die Messmaschinerie neu bauen

Die vier Skripte sind am 31.08. gelöscht worden (Banner oben). Der Neuaufbau
kommt **hierher**, nicht früher, und in vier Schritten — die Reihenfolge steht
in `PINNmodulusTwo/FAHRPLAN.md` §4:

1. Zwei `train.py`-Läufe von Hand vergleichen. Solange das reicht, reicht das.
2. **Seeds.** Der erste echte Bedarf: eine MAE-Differenz zwischen zwei
   Konfigurationen ist wertlos ohne die Streuung über Seeds daneben. Eine
   Schleife über `--seed` plus Mittelwert/Std.
3. **Eine Achse.** Liste von `fit()`-Overrides, eine CSV-Zeile je Punkt.
4. Plots und Resume ganz zuletzt, nur für die Achse, die wirklich Stunden läuft.

Aus dem alten Code übernommen wird die **Bewertungslogik** — Auswahl auf
`--val-ops`, Bericht auf `--test-ops`, Kriterium MAE und nie `L_data`, plus das
Seed-Rausch-Urteil —, nicht die Infrastruktur.

### E6a — Loss-Balance (~4 h GPU)

Die erste Achse. Das in E3/E4 gerissene Kriterium **ist** die Balance, und
`w_phys` bedeutet nichts, solange nicht feststeht, wie die Terme skaliert
werden. Die Diagnosen dafür liegen schon in der History (`spread_space`,
`spread_time`, `div_*`) und im `[FLAT]`-Log.

### E6b — Architektur (~1 Tag GPU bei `--epochs 20`)

Breite, Tiefe, `rate_lags`, `delta_grid` — eine Achse nach der anderen. Nur wenn
E6a die Balance geklärt hat. Vorgezogen, falls E3(a) „neu ≈ Baseline" ergab.

### E6c — Gewichte, klein (3×3), **nicht** das 10×10-Gitter

Das volle Gitter waren „100 trainings (~6-8 days)" und misst Gewichte, bevor
feststeht, was ein Gewicht hier bedeutet.

### E6d — Die Materialgrenzen (`ARCHITECTURE.md` 4.1)

Jetzt, nicht früher — eine Änderung am Physik-Term vor E3 wäre eine zweite
unabhängige Variable im A/B gewesen. Entschieden wird nach der Zahl aus **E1**:

| Grenzflächenanteil der 363 Punkte | Option |
|---|---|
| klein | nichts tun, dokumentieren |
| erheblich | **A**: Grenzflächenpunkte aus dem `batch_phys`-Sampling ausschließen (Maske über `region`). Klein, in sich korrekt, entfernt den falschen Druck |
| erheblich **und** A reicht nicht | **B**: Flusskopplung `λ₁∂T/∂n\|₁ = λ₂∂T/∂n\|₂` als eigener Loss-Term. Physikalisch richtig, aber braucht ein neues Gewicht, das selbst kalibriert werden muss |

Option **C** (`(∇λ)·∇T` über geglättetes `λ` nachrüsten) **nicht als Erstes** —
und nur, falls Messung 2 aus E1 zeigt, dass die glatte Variation im Inneren
überhaupt zählt.

Getrennt davon der Nebenbefund: `_static_features` gibt `λ` nur als isotropen
Mittelwert weiter, das Residuum nutzt den vollen Tensor. Eigener Effekt, eigene
Änderung — **nicht mit der Grenzflächensache vermischen**, sonst ist hinterher
nicht trennbar, was gewirkt hat.

### E6e — Die kleinen offenen Punkte

Aus `README_MODEL_CRITIQUE.md` §2, jetzt fällig, weil sie erst bei 30 000
Schritten zählen:

* **O2** — kein LR-Schedule. Cosine-Decay lohnt sich erst jetzt. Klein.
* **O3** — Auswahl nach *Trainings*-Loss, berichtet wird der Endzustand, nicht
  der beste. Bei 30 000 Schritten ist Overfitting erstmals möglich. Klein.
* **O4** — fünf Trainings-OPs sind eine harte Grenze für Cross-OP-Generalisierung.
  **Nicht durch Code lösbar** — das ist genau das, was E7 adressiert.

**Tor G6:**
- [ ] Balance hält über den ganzen Lauf, `L_phys_bal` und `L_bc_bal` in `[0.01, 100]`
- [ ] `[SATURATED]` = 0, `[FLAT]` feuert nicht
- [ ] Test-MAE unter der Latte aus E2, mit Abstand
- [ ] Falls F1 vorliegt: Test-MAE gegen die Zielgenauigkeit eingeordnet
- [ ] Eine `config.yaml`, die als Basis für E7 taugt

---

## E7 — Die Erweiterung: OP01–OP16 mit Profilen

Ab OP08 werden die Treiber zeitabhängig: Fluidtemperaturprofil, CC-CV-Strom,
Volumenstromprofil. Gleiches Modell, gleiche Physik, gleiche Rekurrenz — was
sich ändert, sind **Vorverarbeitung, Normierung und was zu messen ist**.

> ⚠️ **Nichts aus E6 überträgt sich als Zahl.** Die Normierung poolt hier über
> OP01–OP16, das verschiebt `T_sigma`, `dTdt_scale` und `A`. Die
> Konfiguration aus E6 ist ein *Startpunkt*, kein Ergebnis. `FAHRPLAN.md` §8
> sagt zu Recht: nicht nach ExtProfiles schielen, solange die Basis nicht steht
> — umgekehrt gilt aber genauso, dass die Basis-Zahlen hier neu zu messen sind.

Die Reihenfolge ist **nicht** frei: `resample` ändert `q_dot` → `Qsrc_scale` →
`phys_scale`, und das sind die Divisoren des Physik-Residuums. Wer die Gewichte
vor dem Resampling einstellt, stellt sie zweimal ein.

Die **Reihenfolge der Stages** bleibt gültig; die Skripte, die sie gefahren
hätten, sind gelöscht. Was heute schon geht, und in welcher Reihenfolge der
Rest nachkommt:

```bash
# Stage 0 -- Tor, Minuten. Ersetzt smokeBench: Plan-Sheet gegen die Buendel,
# Abdeckung der Held-out-Treiber, und ein kurzer Lauf.
python3 PINNmodulusTwoExtProfiles/op_registry.py     # der Split, ohne Daten
python3 PINNmodulusTwoExtProfiles/data.py            # profile_report + coverage_report
python3 PINNmodulusTwoExtProfiles/train.py --epochs 5 --subsample 40

# Stage 1 -- Vorverarbeitung: resample, use_driver_history, driver_rate_lags.
# Bis die Seed-Schleife aus E6.0 steht, von Hand: je zwei Laeufe, verglichen
# ueber die val-MAE, die train.py ohnehin druckt.
python3 PINNmodulusTwoExtProfiles/train.py --resample mean  --epochs 20
python3 PINNmodulusTwoExtProfiles/train.py --resample point --epochs 20

# --> Gewinner in config.yaml schreiben, BEVOR es weitergeht

# Stage 2 -- Loss-Gewichte. Stage 3 -- Architektur, zuletzt und optional.
# Beide erst, wenn die Achsen-Maschinerie aus E6.0 existiert: ohne Streuung
# ueber Seeds ist eine MAE-Differenz zwischen zwei Gewichten nicht lesbar.
```

Aus Stage 0 zwei Dinge mitnehmen: ob die Bündel die Profile wirklich tragen, und
das finale `L_data` — daraus kommt der Gewichtsbereich für Stage 2. `w_phys` ist
hier **kein** relatives Gewicht, sondern der nahezu konstante Boden, auf dem der
Physik-Term liegt; das Verhältnis am Ende ist etwa `w_phys / L_data_final`. Also
grob `[0, L_data_final, 10 × L_data_final]` sweepen.

**Die Entscheidungsregel an jeder Stage** — der neu gebaute Bericht muss sie drucken
automatisch:

| Ausgabe | Zu tun |
|---|---|
| Spanne **unter** dem Seed-Spread | Der Regler zählt auf diesen Daten nicht. Billigste Einstellung nehmen, aufhören zu tunen |
| Spanne **über** dem Seed-Spread | Lohnt sich, um den Gewinner herum verfeinern |
| „seed spread unknown (single seed)" | Nichts schlussfolgern. Mit `--seeds 0 1 2` neu |

Zusätzlich beachten: `noise_verdict` (schlägt der Abstand zum Zweiten den
Seed-Spread?) und `split_verdict` (hat der Gewinner seinen Score aus **einer**
Hälfte der Auswahlmenge gekauft?). Eine Konfiguration, die den Fehler auf OP06
halbiert und auf OP09 verdoppelt, kann den Mittelwert gewinnen und ist trotzdem
die falsche Antwort auf die Frage, für die diese Erweiterung existiert.

**Tor G7:**
- [ ] Bewertet wird auf **T1 und T2** — darauf lief die Auswahl
- [ ] **T3 wird berichtet, nicht optimiert.** Ein Gewinner auf T1/T2, der T3
      verliert, ist nicht kaputt — T3 ist Extrapolation und war nie
      Auswahlkriterium
- [ ] Keine T3-Zahl wird ohne den OP zitiert, aus dem sie kommt
      (per-OP getrennt halten, nie eine Zahl für alles)

---

## E8 — Abschluss

### E8.1 — Der eine Lauf, der zählt

Mit der festgeschriebenen Konfiguration, mehreren Seeds, Checkpoint aus E0.3.
Ergebnis: ein Commit-Hash, eine `config.yaml`, ein Checkpoint, eine
`metrics.txt` — die vier gehören zusammen und werden zusammen abgelegt (**F4**).

### E8.2 — Der Ergebnisbericht

Nicht der Trainingsbericht vom 28.08. fortgeschrieben, sondern ein eigenes
Dokument, das eine Frage beantwortet: *Was kann dieses Modell und was nicht?*

| Abschnitt | Inhalt |
|---|---|
| Was es ist | Hybrid-PINN, BDF2, Hybrid-History, 52 485 Parameter — aus `ARCHITECTURE.md` |
| Genauigkeit | MAE je Tier: T0-in-time, T1-interp, T2-profile, T3-extrap. Getrennt (**F5**) |
| Gegen was | Die Latte aus E2 und die Zielgenauigkeit aus F1. Beide Zahlen, beide benannt |
| Wo es nicht gilt | Außerhalb des Envelopes, an Materialgrenzen (`ARCHITECTURE.md` 4.1), bei Profiltypen ohne Trainingsbeispiel (**F6**) |
| Reproduktion | Commit, Config, Checkpoint, Kommandos |

### E8.3 — Doku-Endstand

Nichts löschen, umetikettieren — `FAHRPLAN.md` §7 hat die Tabelle. Zusätzlich:

* `FAHRPLAN.md` und **diese Datei** werden Archiv, sobald F1–F6 stehen.
  Der Einstieg ist dann der Ergebnisbericht.
* `README_ERSTER_TEST.md`: die synthetischen Zahlen sind durch die echten aus E2
  ersetzt, der Synthetik-Hinweis ist mit weg.
* `UEBERGABE_2026-08-27.txt`, `TRAININGS_BERICHT_2026-08-28.md`: Archiv,
  historischer Stand, bleiben unverändert.

---

## 3. Was nicht zu tun ist

Gilt über den ganzen Plan.

| Nicht | Warum |
|---|---|
| `make_synthetic_cache.py` auf deinem Rechner laufen lassen | Überschreibt `data_cache/OP*.npz` ungeprüft. Du hast die echten Daten — das Werkzeug ist für Maschinen ohne sie |
| `w_phys` auf 1.0 / 10.0 erhöhen | `L_phys_bal = L_phys/EMA(L_phys)` ist selbstnormiert, `w_phys` kommt darin nicht vor (`train.py:600`) |
| Das 10×10-Gewichtsgitter | ~6–8 Tage, und misst Gewichte, bevor feststeht, was eines bedeutet |
| Gegen 11.96 °C vergleichen | Synthetisch, und es ist der Persistenz-, nicht der Mittelwert-Vorhersager |
| „Full Grid 6358 Punkte" | Existiert nicht. 363 ist die native Sensorzahl |
| Am Physik-Term schrauben vor G3 | Zweite Variable in einem A/B, das ohnehin noch aussteht |
| Zwei Änderungen in einem Lauf | Gilt besonders für E6d: Grenzflächen und Anisotropie sind zwei Effekte |
| Eine MAE-Zahl für alles berichten | T1, T2, T3 messen verschiedene Dinge (**F5**) |
| GPU-Server vor G5 buchen | Vorher rankt jeder Sweep Verlierer |

---

## 4. Abbruchkriterien

Ehrlichkeitsklausel — wann dieser Plan selbst falsch ist:

* **G3 sagt „beide > 20 °C"** → das Problem sind die Daten oder die Modellklasse,
  nicht die Hyperparameter. Zurück auf E1, nicht weiter nach unten.
* **G4 lässt sich nicht auf 0 bringen** → der Rollout ist instabil, nicht
  untertrainiert. `ARCHITECTURE.md` 3.1, nicht mehr Epochen.
* **`[FLAT]` feuert dauerhaft** → das Modell löst die triviale Aufgabe. Physik
  runter, Datenterm prüfen — kein Gewichte-Sweep repariert das.
* **G5 bleibt nach E6a+E6b rot** → das Modell schlägt einen konstanten Mittelwert
  nicht. Dann ist die Frage nicht mehr, welches Gewicht gewinnt, sondern ob
  History-Struktur und Kapazität für diese Aufgabe reichen. Das ist der Punkt,
  an dem O4 (fünf Trainings-OPs) und die Modellklasse selbst auf den Tisch
  kommen — und an dem E7 als Antwort auf O4 vorzuziehen ist, statt E6 weiter
  auszureizen.
* **G7: T3 bricht weg, während T1/T2 stehen** → kein Fehler, sondern ein Befund.
  Er gehört als Grenze in den Bericht (**F6**), nicht in eine weitere
  Optimierungsrunde.

---

## 5. Die kurze Fassung

1. **E0 heute:** Backup, Guard, PR #18, Checkpoint-Speicherung, Zielgenauigkeit anfragen.
2. **E1–E3 in einer Sitzung, ~1 h, ohne GPU.** Es fallen vier Textdateien an:
   `latte.txt`, `A_neu.txt`, `A_alt.txt`, `interface.txt`. Damit lässt sich
   weiterrechnen statt raten.
3. **E4/E5:** Rollout auf 0 saturierte Schritte, dann die Latte unterbieten. Das
   ist der erste echte Meilenstein.
4. **E6/E7:** GPU, Tage. Erst Basis, dann Profile — in dieser Reihenfolge, weil
   sich keine Zahl überträgt.
5. **E8:** festschreiben, berichten, Grenzen benennen.

Der Engpass ist heute nicht der Code. Es sind die ~60 Minuten an deinem Rechner
für E1–E3 und die eine Zahl von der Fachseite.
