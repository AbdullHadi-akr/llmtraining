# PINNmodulusTwoExtProfiles — the profile extension of PINNmodulusTwo

> **Update 31.08.2026 — die Benchmarks sind gelöscht.** `profileBench.py`,
> `bench_profiles.py` und `smokeBench.py` gibt es nicht mehr, ebenso wenig die
> vier im Basisprojekt. Sie werden Schritt für Schritt neu aufgebaut, sobald
> feststeht, was gemessen werden soll —
> [`../PINNmodulusTwo/FAHRPLAN.md`](../PINNmodulusTwo/FAHRPLAN.md) §0 und §4.
> `train.py` bleibt vollständig lauffähig und wertet Train-, Val- und Test-OPs
> weiterhin über `op_metrics.py` aus; was fehlt, ist der Vergleich über Seeds.
> Siehe [Measuring this extension](#measuring-this-extension--the-benchmark-is-gone-deliberately).

> **Update 27.08.2026.** `residual_output` wurde hier nie an das Modell
> übergeben, lief also still mit dem alten Default `true` — einem Integrator
> ohne Leck, der den Rollout in Epoche 1 nach `inf` treibt. Er ist jetzt ein
> Schalter und steht auf `false`, dazu kommt `rollout_clamp`. Die `rate_lags`
> bleiben bei `[5, 20]`; die Verstärkung `A` ist hier durch das Pooling über
> OP01–OP16 zwar *größer* als im Basisprojekt, aber sie ist nicht die Ursache.
> [`README_UPDATE_2026-08-27.md`](README_UPDATE_2026-08-27.md) hat den Befund,
> die Änderungen und die Messung dazu.

`PINNmodulusTwo` trains on **OP01–OP05** and holds out OP06/OP07. All seven are
*constant* operating points: one C-rate, one fluid inlet temperature and one
volume flow, held for the whole charge.

This folder extends that model to the **full plan sheet, OP01–OP16**, where from
OP08 on the drivers become **profiles** — they vary in time:

| driver | which OPs vary it | what it looks like |
|---|---|---|
| `fluid_inlet_temp` | OP08, OP09, OP12, OP13, OP15 | *Fluidtemperaturprofil* |
| `cell_current` | OP10, OP11, OP12, OP13, OP15 | CC‑CV / CC‑CV_anode, current pre-simulated in Batemo — the CV phase tapers the current away |
| `fluid_mass_flow` | OP15 only | *Volumenstromprofil* |

Same model, same physics, same recurrence. What changes is the **preprocessing,
the normalisation, and what has to be measured** — because a mean over a driver that moves
is not the same quantity as a sample of a driver that does not, and because
every pooled statistic is refitted on a much more heterogeneous set.

> **The data is not in this repository and never will be.** Nothing here can be
> run without the `OP*.npz` bundles and the material-property CSVs. See
> [Data](#data--not-in-git) below. Every number in this README is a description
> of what the code computes, **not a measured result** — see
> [Caveats](#caveats).

---

## The operating points and the split

`op_registry.py` holds the plan sheet in code and is the single source of truth
for the split. Run it — it needs no data:

```bash
python3 PINNmodulusTwoExtProfiles/op_registry.py
```

Held-out OPs are graded into **tiers** by what they ask of the model, not by
whether they happen to contain a profile:

| tier | meaning |
|---|---|
| `T0-in-time` | a training OP, scored on its own timeline |
| `T1-interp` | held out, every driver inside the trained range |
| `T2-profile` | held out, profiles whose **type** appears in training |
| `T3-extrap` | held out and outside the trained envelope: a driver value beyond the trained range, or a profile type never trained on |

Default split — every one of OP01–OP16 is used exactly once:

```
train (11)          OP01 OP02 OP03 OP04 OP05 OP07 OP08 OP10 OP11 OP12 OP14
val   (selection)   OP06 (T1)  OP09 (T2)
test  (report only) OP13 (T3)  OP15 (T3)  OP16 (T3)
```

Why these three are `T3`:

* **OP13** — C‑rate 4, above every trained C‑rate (2 … 3), *and* two profiles at
  once (fluid temperature + CC‑CV_anode).
* **OP15** — carries a **volume-flow profile**, a profile *type* that appears in
  no training OP. Its rate channels were dead during training and are being
  asked to mean something for the first time.
* **OP16** — constant, but the sheet's `15*6` = 90 l/min is three times the
  trained maximum.

The rules the split follows (enforced advisorily by `op_registry.check_split`):
every profile type a **selection** OP relies on must occur in training, and
selection never touches the extrapolation tier — otherwise the ranking optimises
for extrapolation and the only out-of-envelope evidence is spent.

**OP17–OP19 are deliberately out of scope.** They are the *Abgleich mit
Minimodul-Test*: measured mini-module data, partly discharge, one synthetic WLTP
cycle, with "Test Data" in every driver column. That is a measurement-vs-
simulation exercise, not this surrogate's training set. Nothing here reads them.

---

## What actually changed, and why

### 1. Drivers are resampled, not point-sampled

The base loader takes `[::subsample]` of everything. For a constant OP that is
exact — a constant equals its own mean. For a profile it is **aliasing**: at
`subsample=2`, nineteen of every twenty raw samples of a CC‑CV current taper are
discarded and *which* twentieth survives decides what the model is told the
current was.

Every **driver** (the heat source `q_dot` and the config profiles) is now
reduced with a **backward window mean**: sample `j` is the average over the raw
interval that *ends* at it. That is the quantity the step from `t_{j-1}` to
`t_j` is actually driven by, and for the heat source it preserves the energy
that went into the window. Backward rather than centred or forward on purpose —
a forward window lets a step that happens *after* a sample influence that
sample, which hands a free-running autoregressive rollout information from its
own future.

**Temperature stays point-sampled.** It is a state, and the rollout has to match
it at the sample instants, not average over them.

`resample: point` restores the base behaviour exactly. Which of the two is
better is an open question and one of the first axes the rebuilt benchmark
should measure -- so far the choice of `mean` is an argument, not a
measurement.

### 2. The drivers get their own history channels

The recurrence feeds back the model's own past **temperature** — that is what
disambiguates two OPs sharing an instantaneous config, and it is unchanged. But
the drivers are exogenous and fully known in advance, so their history costs
nothing to compute and never has to be predicted. For each driver `d` the loader
appends causal rate channels over the same cumulative segments the hybrid
temperature history uses:

```
rate_1 = [d(t)     - d(t-L1)]     / L1
rate_2 = [d(t-L1)  - d(t-L1-L2)]  / L2
```

with `d(t) := d(t₀)` for `t < t₀` and each rate divided by its own **nominal**
segment length — the same choice, for the same reason, as
`model._history_hybrid`: dividing by the clamped *elapsed* span is a singularity
at the start of the trajectory.

Drivers covered: `q_dot`, `c_rate`, `cell_current`, `fluid_inlet_temp`,
`fluid_mass_flow` (5 × 2 lags = 10 extra channels by default). A constant OP
gets rates that are identically zero, which is itself informative. The rates are
divided by their **pooled training RMS**, not a per-OP scale — a per-OP scale
would make "the current is changing fast" mean something different in every OP,
which is exactly the comparison the model has to be able to make.

These are plain extra columns of `forcing_feat`. **The model, the recurrence and
the physics residual are untouched** — only `n_forcing` changes, and it was
already a constructor argument.

### 3. The normalisation constants are different, and not by a little

Pooling OP01–OP16 instead of OP01–OP05 widens every pooled statistic. `T_sigma`
now spans a 0 °C start (OP14) and a 4 C charge (OP13); `Qsrc_scale` is dominated
by the high-C-rate OPs. `phys_scale`, `dTdt_scale`, `aniso_scale` and
`Qsrc_scale` are the **divisors of the physics residual**, so:

> **`w_phys` / `w_bc` tuned in `PINNmodulusTwo` do not carry over.** The same
> number is a different mixing ratio here. `config.yaml` ships placeholders,
> not inherited values, and they have to be re-measured here. This is the main
> reason the extension needs its own measurement rather than inheriting numbers.

`data.normalisation_report()` prints the constants next to the per-OP spread of
`Qsrc` RMS at the top of every run, so the shift is visible instead of assumed.

### 4. A consequence of (3) that bites: hybrid history amplification

`model._history_hybrid` feeds the network `(T_end − T_start) / (lag_n ·
rate_scale)` with `rate_scale = dTdt_scale`. For a genuine rate that lands the
channel at O(1). Early in a free-running rollout it is **not** a genuine rate:
step 1 differences the untrained network's first output against the imposed
initial condition, so the numerator is a *level* jump, returned multiplied by

```
A = 1 / (lag_n · rate_scale)
```

Widening the pooled `T_sigma` shrinks `Tn`, shrinks its time derivative, and
therefore shrinks `dTdt_scale` — **raising A**. Pooling OP01–OP16 does exactly
that: a large part of `T_sigma` is now between-OP offset that contributes
nothing to any single OP's own rate, while the level jumps do not shrink,
because they are what `T_sigma` is made of. Past a point the opening steps
saturate and `L_data` goes non-finite in epoch 1.

`A` is printed at every startup and warned about when large. `--max-rate-amp N`
raises `rate_scale` just far enough to cap `A` at `N`. **It is off by default**
(`0`), because silently rescaling a channel would make this extension's model
quietly different from the one the base results were produced with — and when it
does engage it says so in the log and in `metrics.txt`, so the run can be
reported honestly. `--history-mode raw` removes the rate channels entirely and
is the escape hatch. See `data.effective_rate_scale`.

*(This was found while exercising the pipeline on synthetic data, and it
reproduces identically in the unmodified `PINNmodulusTwo` when that is pointed
at eleven OPs. It is a property of the shared model code that the wider pooling
makes more likely, not a bug introduced here.)*

### 5. Per-OP loss balancing, and a shuffled OP order

The base loop keeps **one** EMA of `L_phys` and one of `L_bc`, updated as it
walks the OPs, stepping the optimiser after each. Harmless among five
near-identical constant OPs. Among eleven OPs whose physics term differs by more
than an order of magnitude, a single EMA balances each OP against *whichever OP
happened to come before it* — an OP-order-dependent weighting nobody chose. Each
OP now carries its own EMA, so `w_phys` means the same thing everywhere.

For the same reason the OP order is reshuffled every epoch from a seeded RNG
(`shuffle_ops: true`): a fixed order otherwise lets the last OP in the list
always take the last word of every epoch.

Each OP contributes `inner_steps` optimiser updates per epoch, all against that
epoch's frozen rollout — the same scheme the base project moved to. The rollout
is the expensive part, so the update count rises at roughly constant cost;
`--inner-steps 1` reproduces the old one-update-per-OP behaviour.

### 6. The late-window metric is labelled honestly

The data loss samples the whole rollout, so the base project's "MAE test" on a
*training* OP is in-sample — it is a split of the metric, not of the training
data. That is now labelled as such everywhere. `--holdout-tail` makes it real by
truncating the training rollout at `split_t`.

**It is off by default, and think before turning it on:** with CC‑CV OPs in the
set, the late window *is* the CV taper, so holding it out removes the very
regime the extension exists to learn. The honest held-out signal here is the
**OP-level tiers**, not the time split.

### 7. Extrapolation is reported, not discovered later

`build_op` reuses the training constants — that is what makes a held-out OP a
genuine test, and also what makes an out-of-range driver silently z-score to
something the network never saw. `data.coverage_report()` says which channel of
which OP leaves the trained range and by how many training sigmas, and it runs
in `train.py`, and it is worth keeping in whatever replaces the benchmark.

---

## Measuring this extension — the benchmark is gone, deliberately

`profileBench.py`, `bench_profiles.py` and `smokeBench.py` were deleted on
31.08.2026, together with the four benchmark scripts in the base project. They
are being rebuilt step by step; see
[`../PINNmodulusTwo/FAHRPLAN.md`](../PINNmodulusTwo/FAHRPLAN.md) §0 and §4 for
why and in what order.

The reason bites hardest here: **not one number in this folder was ever
measured.** The extension needs all sixteen bundles, they are not in git, and
the benchmark that was supposed to produce the first result never ran. A tiered
sweep with seed aggregation, checkpoints and per-tier boxplots is a lot of
machinery to carry for zero measurements — and the base project it inherits its
loss weights from has not settled them either.

**What still works, without any of it:**

```bash
python3 PINNmodulusTwoExtProfiles/op_registry.py     # the split (no data needed)
python3 PINNmodulusTwoExtProfiles/data.py            # constants + coverage + profile report
python3 PINNmodulusTwoExtProfiles/train.py --epochs 60
```

`train.py` here already does what the base project's `train.py` only learned on
31.08.: it trains on `--ops`, then rolls out `--val-ops` and `--test-ops` with
the training bundle's normalisation and reports every OP through
`op_metrics.py` — `mae`, the transient/quiescent split, `peak_err`, `late_mae`
and the tier each held-out OP belongs to. That is a complete per-OP evaluation.
What it is not is a *comparison* between configurations over several seeds, and
that is the one thing the rebuilt benchmark has to add first.

**The checks that lived in `smokeBench` and are worth having again**, in the
order they should come back:

1. **Are the profiles actually in the bundles?** `op_registry` is a
   transcription of the plan sheet and can be wrong; `data.profile_report()`
   prints what the `.npz` files really contain next to what the sheet claims.
   That comparison still exists — it just no longer has a script wrapped
   around it.
2. **Does any held-out driver leave the training range?**
   `data.coverage_report()`, which `train.py` already runs.
3. **Does a short run finish at all** before a long one is started.

**What the rebuilt benchmark must keep from the deleted one** — the scoring
rules, not the infrastructure:

* **Selection over a SET of validation OPs, not one.** Two things trade off
  here — staying accurate on constant OPs, and following a driver that moves —
  and one OP can only measure one of them. Rank on the mean over `--val-ops`
  (OP06 constant, OP09 profile), unweighted, and carry every per-OP number
  alongside so a configuration that wins the mean by wrecking one of the two is
  visible rather than hidden.
* **Results grouped by tier.** A single averaged test MAE over OP13, OP15 and
  OP16 mixes a C-rate extrapolation, an unseen profile type and a 3x flow. The
  average of three different questions answers none of them. `op_registry`
  still holds the tiers.
* **A seed-noise verdict.** A ranking whose spread between configurations is
  smaller than the spread between seeds of one configuration is not a ranking.
* **The mean over seeds, with failed seeds dropped rather than averaged in.**


## Files

| file | role |
|---|---|
| `op_registry.py` | the plan sheet in code: OP01–OP16, profiles, tiers, the split, split sanity checks. **Runs without data.** |
| `data.py` | profile-aware loader: window-mean driver resampling, driver rate channels, refitted normalisation, amplification diagnostic, coverage/profile reports |
| `train.py` | training loop for the heterogeneous OP set; evaluates train + val + test OPs |
| `op_metrics.py` | the per-OP rollout metrics: mae/rmse, the transient/quiescent split, `peak_err`, `late_mae` |
| `config.yaml` | defaults; the single source for them |
| `_paths.py` | imports the unchanged `model.py`, `physics.py`, `materials.py`, `device_utils.py` from `PINNmodulusTwo/` |

### Reused from `PINNmodulusTwo/`, not copied

`model.py`, `physics.py`, `materials.py` and `device_utils.py` are imported
through `_paths.py` rather than duplicated. They needed no change — the profile
features arrive as extra input channels, and `n_config` / `n_forcing` were
already constructor arguments — and two copies would only drift apart.
`materials.py` also owns the path to `PINNmodulusTwo/material_properties/`,
which keeps that one copy authoritative.

Consequence worth knowing: this directory is inserted **ahead** of
`PINNmodulusTwo` on `sys.path`, so a file here shadows its sibling of the same
name. That is deliberate for `data.py`, and it is why nothing here may be named
`model.py`, `physics.py`, `materials.py` or `device_utils.py`.

---

## Run

```bash
source .venv/bin/activate      # the same environment as PINNmodulusTwo

python3 PINNmodulusTwoExtProfiles/op_registry.py     # the split (no data needed)
python3 PINNmodulusTwoExtProfiles/data.py            # the constants + coverage
python3 PINNmodulusTwoExtProfiles/train.py --epochs 5     # does it start
python3 PINNmodulusTwoExtProfiles/train.py --epochs 60    # the real run
```

`--device` defaults to `auto` (CUDA when available). Full server setup — driver,
CUDA PyTorch, Modulus, data transfer — is unchanged and documented in
[`../PINNmodulusTwo/README_GPU_SERVER.md`](../PINNmodulusTwo/README_GPU_SERVER.md).

Useful flags when something goes wrong:

| symptom | first thing to try |
|---|---|
| `L_data` non-finite in epoch 1 | `--max-rate-amp 50`, then `--history-mode raw` |
| still diverging | `--grad-clip 1.0`, lower `--lr`, larger `--subsample`, `--no-driver-history` |
| unsure whether the profiles are even being read | `python3 data.py` — it prints the profile report and the coverage report, no training |
| want the base project's preprocessing back | `--resample point --no-driver-history` |

---

## Data — not in git

`.gitignore` keeps only source, README and the two `config.yaml` files. The
`OP*.npz` bundles and `PINNmodulusTwo/material_properties/` are **not tracked
and never arrive with a clone.** They have to be present on each machine.

`data.py` searches these locations and takes the first that exists:

1. `PINNmodulusTwoExtProfiles/data_cache/` — extension-local override
2. `PINNmodulusTwo/data_cache/` — base-project override
3. `data_cache/` — **preferred**: shared, top level
4. `legacy/battery_surrogate_agenticWorkflow/data_cache/`
5. `battery_surrogate_agenticWorkflow/data_cache/` — pre-restructure location

Build missing bundles from the raw CSVs with the base project's generator (the
one place the active code still depends on the legacy assembly):

```bash
python3 PINNmodulusTwo/generate_cache.py OP08 OP09 OP10 OP11 OP12 OP13 OP15 OP16
```

`require_ops()` fails immediately and lists what is available, rather than after
the first training run.

---

## Caveats

Read these before quoting anything from this folder.

1. **No result has been produced.** The real `OP*.npz` bundles were not
   available where this code was written, so **no number in this extension has
   been measured on the real data.** `config.yaml`'s `w_phys` / `w_bc` are
   placeholders carried over as a starting point, explicitly *not* tuned values.
   The first real deliverable is a run on the sixteen real bundles; until then
   this folder is a pipeline, not a result.

2. **The pipeline was exercised on synthetic bundles, not validated by them.**
   Every path here — window-mean resampling, driver rate channels, per-OP EMAs,
   the tier machinery, checkpoints, plots, CSVs — was run end to end against
   fabricated `.npz` files with the right keys and shapes and physically
   meaningless contents. That proves the plumbing runs and the reports render.
   It proves **nothing** about accuracy, convergence at the real `subsample=2`,
   or whether any of the new features help. The synthetic data is not in the
   repository.

3. **`op_registry.py` is a transcription of the plan sheet and can be wrong.**
   It drives tier labels and reports, never a feature — every model input comes
   from the `.npz`. `data.profile_report()` compares the sheet against what the
   bundles actually contain -- run `python3 data.py` and believe the bundles,
   not the table.

4. **The tier assignments are a judgement, not a measurement.** OP16 is called
   `T3` because 90 l/min is 3× the trained maximum, and OP09 `T2` because C‑rate
   2.5 interpolates 2…3. If the real bundles disagree with the sheet, the tiers
   move. `coverage_report` checks the actual numbers per channel; where it and
   the sheet disagree, it wins.

5. **OP15 asks for something training never showed.** It carries the only
   volume-flow *profile* in the set, so its `fluid_mass_flow` rate channels were
   dead during training. Expect it to be the worst test OP, and do not read that
   as a defect in the method. The alternative — training on OP15 — would leave
   no way to measure profile-type generalisation at all.

6. **`soc_start` is likely a dead channel.** The sheet has 10–90 % for every
   training OP, so it has zero variance and is forced to 0 like any other dead
   channel. Whether the real bundles agree is printed by
   `normalisation_report`.

7. **Selecting on two OPs is still selecting on two OPs.** OP06 and OP09 are
   one constant and one profile case. A configuration can be tuned to those two
   specific held-out points. The tiers reduce the risk; they do not remove it.

8. **A short run finishing means the pipeline is sound, not that the model is
   accurate.** No check in this folder currently says whether a MAE is good --
   that needs a target accuracy from the aging model, which nobody has
   supplied.

9. **`bc_V`, voltage and aging remain out of scope**, exactly as in the base
   project. Temperature only.

10. **`--max-rate-amp` changes the model.** When it engages, the run is not
    comparable with an unguarded one. The log and `metrics.txt` record the
    effective `rate_scale`; quote it whenever a guarded run is reported.
