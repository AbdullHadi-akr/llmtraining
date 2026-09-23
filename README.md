# Battery temperature surrogate

> **Nach einer Pause hier anfangen:** [`README_NAECHSTE_SITZUNG.md`](README_NAECHSTE_SITZUNG.md)
> (wo du stehst, was als Nächstes kommt). **Welches Modell welche Zahl erzeugt
> hat:** [`README_MODELLSTAND.md`](README_MODELLSTAND.md) (Modellversionen und
> Experimente, chronologisch).
> **Neues PINN-Modell P3 und sein POC (Priorität 1):**
> [`PINNmodulusTwo/README_MODELL_P3_POC.md`](PINNmodulusTwo/README_MODELL_P3_POC.md).

## Project status — as of 23.09.2026

**Two folders are active and up to date: [`GridCNN/`](GridCNN/) and
[`PINNmodulusTwo/`](PINNmodulusTwo/).** They are two independent models for the
same problem, trained on the same data with the same split. Both were last
changed on 23.09.2026, and CI tests both. Everything else is either data you have
to copy in, session documentation, or outdated.

| Folder | Status | Last change | What it is and where it stands |
|---|---|---|---|
| [`GridCNN/`](GridCNN/) | 🟢 **active, current** | 23.09.2026 (PR #51) | Model 2: a CNN over the whole 3 × 11 × 11 grid, `T + Δt·(L(T) + Qsrc + g_θ)`. Version **G4.3**. Trained on real data, and the latest run beats the trivial predictor but is not yet a result. [Details below](#gridcnn--where-it-stands) |
| [`PINNmodulusTwo/`](PINNmodulusTwo/) | 🟢 **active, current** | 23.09.2026 (PR #49) | Model 1: a recurrent PINN (Modulus MLP plus its own recurrence). The code is version **P2.1**. The new model **P3** is built and waiting for its POC on real data. [Details below](#pinnmodulustwo--where-it-stands) |
| `data_cache/` | ⚠ **required, not in git** | — | `OP*.npz` bundles, cache schema **v3** (since 22.09.). Both models read it, and it has to be copied in by hand ([TODO below](#todo--the-data-is-missing-from-a-fresh-clone)) |
| [`legacy/`](legacy/) | ⚪ **outdated, not maintained** | 22.09.2026 | Earlier approaches, kept so old results stay reproducible. **One part is still in use:** the raw-CSV assembly in `legacy/battery_surrogate_agenticWorkflow/src/` (plus `op_matrix.yaml`, `build.yaml` and `coordinates/`). `PINNmodulusTwo/generate_cache.py` builds the cache through it, which is why it got the schema-v3 update. See [`legacy/README.md`](legacy/README.md) |
| `modulus-sym/` | ⚪ **empty, unused** | 02.09.2026 | A submodule pointer with no `.gitmodules` entry, so a clone gets an empty folder. The active models don't use it (they import `modulus` from the `modulus_env` venv). Only `legacy/pinnANDmodulus/` refers to it |
| [`.github/workflows/`](.github/workflows/) | 🟢 **active** | 14.09.2026 | `tests.yml` runs the PINN tests and the GridCNN tests as two separate invocations (both projects have a `physics` module), GridCNN benchmark stages 0 + 2 (dry run), `selftest.py`, an import check and `op_registry.py` |
| root `*.md` | 🟢 **current** (German) | 23.09.2026 | Session documentation. [`README_NAECHSTE_SITZUNG.md`](README_NAECHSTE_SITZUNG.md) is where to start. [`README_MODELLSTAND.md`](README_MODELLSTAND.md) lists every model version and experiment. `TRAININGS_BERICHT_*.md` are the GridCNN run reports and `UEBERGABE_*.md` the GridCNN handovers, both from 22.–23.09. |

### `PINNmodulusTwo/` — where it stands

* **Code: P2.1** (PR #49). With the defaults it computes exactly the same result as
  P2, bit for bit, so every earlier number is still valid. It adds three things:
  the `--phys-stencil {buffer,live}` switch, the diagnostic tool
  `tools/residual_decomposition.py`, and a refusal of `--time-deriv autograd`.
  163 tests.
* **Best real-data numbers** (T4, 3 seeds, mean over val OP06 + OP09): axis 1,
  δ = 0.2 s, gives **4.868 ± 0.650 °C**. The null measurement without physics
  (axis 0) gives **5.248 ± 0.518 °C**. Both are `[NOT SEPARATED]`: the physics
  term has not yet shown a gain that is larger than the seed spread.
* **New model P3** = P2.1 with `--phys-stencil live`. The MLP is unchanged. The
  only difference is that the physics term takes `T(t−δ)` and `T(t−2δ)` from the
  live network instead of from the rollout frozen at the start of the epoch. On
  the synthetic cache it reached val OP06 3.330 ± 0.264 °C against
  6.585 ± 0.179 °C, which confirms the mechanism but is not a result. **It has
  never run on real data.**
* **Next, priority 1:** the P3 POC on the T4 (`buffer` against `live`, dt 1 s,
  3 seeds), described in
  [`PINNmodulusTwo/README_MODELL_P3_POC.md`](PINNmodulusTwo/README_MODELL_P3_POC.md).
  After that, the residual decomposition of the checkpoints, which decides
  between axis 5 (P3 at full resolution) and P4 (spatial derivatives by finite
  differences).

### `GridCNN/` — where it stands

* **Code: G4.3** (PR #51, evening of 23.09.). It adds three switches, and all
  defaults compute the same as before: `--integrator exp` (an exact,
  unconditionally stable physics step, which makes arms B/C/D runnable),
  `--karten kompakt` and `--treiber film`. 157 tests.
* **Real-data runs** (config A, no physics, T4, 3 seeds). The ratio is the MAE
  divided by the trivial-predictor bar, so below 1× beats the bar:

  | run | OP06 | OP09 | verdict |
  |---|---|---|---|
  | 15 — POC, dt 1 s | 6.571 ± 0.383 °C · 0.61× | 5.813 ± 0.867 °C · 0.75× | first readable result |
  | 16 — full resolution | 6.64 ± 1.52 °C · 0.62× | 8.60 ± 1.76 °C · 1.11× | no result (TBPTT window counted in steps) |
  | **17 — full resolution, window in seconds** | **6.64 ± 1.76 °C · 0.62×** | **6.16 ± 1.97 °C · 0.79×** | beats the bar, but the ~2 °C seed spread means no result yet. The remaining error is a level error: too warm early, too cold late |

* **Next:** first check over- or underfitting with `tools/nachmessen.py` on the
  training OPs (minutes). Then run 18 (arm B with `--integrator exp`, physics
  in the architecture for the first time) and run 19 (config A with
  `--karten kompakt --treiber film`). After that comes the wall term in the
  integrator, then "virtual OPs". Plan:
  [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md), top section.
* ⚠ The current state is in the top section of `GridCNN/FAHRPLAN.md`. The
  quick-start part of `GridCNN/README.md` is out of date: it still says the load
  path is missing and lists 99 tests. The stage table inside the FAHRPLAN is
  also not updated for stages 2, 4 and 5.

> **PINN and GridCNN numbers can't be compared yet.** They use different dt,
> and GridCNN reports per OP where the PINN reports an average. Making them
> comparable is GridCNN stage 6.

## TODO — the data is missing from a fresh clone

Nothing in `data_cache/`, `data_raw/` or `material_properties/` is tracked in git
(the `.gitignore` keeps only sources, documentation and a few configs), so a
fresh clone **cannot train**. Everything below has to be copied onto the machine by hand. Tick a box
once that OP is on disk.

### 1. Fast path — the `.npz` bundles (this is what training reads)

Copy the finished bundles into the **top-level** `data_cache/`, one file per OP,
named exactly `OP01.npz`, `OP02.npz`, …:

```
llmtraining/
└── data_cache/          <- create this folder, put the .npz files here
    ├── OP01.npz
    ├── OP02.npz
    └── ...
```

`PINNmodulusTwo/data.py` searches its own folder first and falls back to this
shared one (full list under [`data_cache/`](#data_cache--the-op-bundles) below).
**`data_cache/` at the top level is the one to use.**

### 1b. The material-property CSVs — equally missing, equally required

`PINNmodulusTwo/materials.py` reads these from
`PINNmodulusTwo/material_properties/`. They are untracked too, so they also have
to be put there by hand:

```
PINNmodulusTwo/material_properties/
├── constants.yaml                                  <- jr1 + housing scalars
├── Cell Center/
│   ├── Density_Grid_CellCenter.csv
│   ├── SpecificHeat_Grid_CellCenter.csv
│   └── ThermalConductivity{XX,YY,ZZ}_Grid_CellCenter.csv
└── JR1 Center/
    └── ThermalConductivity{XX,XY,YY}_Grid_JR1Center.csv
```

- [ ] `material_properties/` present

Without it nothing runs at all — `materials.py` is imported by `data.py`, so
this fails before the first OP is even looked up.

**All sixteen are needed, not a subset.** There is one project and it trains on
the whole plan sheet; the split is in
[`PINNmodulusTwo/op_registry.py`](PINNmodulusTwo/op_registry.py) (run it, it
needs no data):

| | OPs | role |
|---|---|---|
| train | OP01–05, 07, 08, 10, 11, 12, 14 | eleven; every profile type a val OP needs occurs here |
| val | OP06, OP09 | one constant, one profile — what a tuning decision may look at |
| test | OP13, OP15, OP16 | the extrapolation tier; read once, never selected on |

- [ ] `OP01.npz` … `OP16.npz` — all sixteen

Missing any one of them fails immediately with the list of what is available.

**`OP19.npz` — optional, and a different question.** OP17–OP19 are the
mini-module *measurement* comparison: measured data rather than a
Batemo/StarCCM+ simulation, partly discharge where OP01–OP16 are all charge,
drivers read from test data, and OP19 is a synthetic drive cycle. Of the three
only OP19 exists in this pipeline — `op_matrix.yaml` has no OP17 or OP18 at all.
Pass it as `--measurement-ops OP19` to roll it out and report it; it is never
trained on and never selected on.

- [ ] `OP19.npz` — optional; the sim-vs-measurement check

### 2. Full path — the raw CSVs (only if the cache has to be rebuilt)

`PINNmodulusTwo/generate_cache.py` builds the `.npz` from raw exports through
the legacy assembly. Those raw OP folders go **two levels deep**, `OP<NN>/OP<NN>/`
(that nesting is what `assemble_op()` expects, not a typo):

```
legacy/battery_surrogate_agenticWorkflow/
└── data_raw/            <- create this folder
    └── OP01/
        └── OP01/
            ├── *_Batemo FMU1.csv
            ├── *_Heat Source.csv
            ├── *_Fluidstoffwerte.csv
            ├── *_T_grid_cc_i.csv
            ├── *_T_grid_g_i.csv
            ├── *_T_grid_jr1c_i.csv
            ├── *_Inputsignale.csv        (OP03–OP07 export it as "Input Signale.csv")
            └── profile files, only where Inputsignale points at one:
                *_CellCurrent(t).csv
                *_FluidMassFlow(t).csv
                *_FluidInletTemperature(t).csv   (OP08: only as .xlsx)
                *_ModuleTestData*.csv            (OP19)
```

- [ ] `data_raw/OP01/OP01/` … `data_raw/OP16/OP16/` — the sixteen the model uses
- [ ] `data_raw/OP19/OP19/` — optional, the measurement comparison

Already in git, so nothing to copy: the three coordinate CSVs under
`legacy/battery_surrogate_agenticWorkflow/coordinates/`, `op_matrix.yaml`,
`build.yaml`.

Then, from the repo root:

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
```

which writes straight into the top-level `data_cache/`, where both projects
find it.

### 3. Check it worked

```bash
ls data_cache/
python3 PINNmodulusTwo/op_registry.py    # the split; needs no data
python3 PINNmodulusTwo/data.py           # constants, profile report, coverage
```

The last one needs the training environment (numpy, torch, pandas) and prints
the pooled constants next to a **profile report**: what the bundles actually
contain, against what the plan sheet claims. A `MISMATCH` line there means the
sheet is wrong or a bundle was built from the wrong export — believe the
bundles.

---
Predicting the internal temperature field of a battery cell from operating-point
inputs, so cell temperature can feed an aging model without measuring inside the
cell. High-fidelity Batemo + StarCCM+ simulations are run offline to produce
training data; the trained surrogate then runs online under automotive
controller constraints.

## Layout

> **Start here:** [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md) —
> the gated order of what to run, and what changed on 31.08.2026 when the two
> projects were merged and the eight benchmark scripts deleted.

```
PINNmodulusTwo/   <- model 1: the recurrent PINN, its training, its tests
GridCNN/          <- model 2: the CNN on the 3 x 11 x 11 grid; independent, same data and split
data_cache/       <- OP*.npz bundles (not in git, see above)
legacy/           <- earlier approaches, kept for reference only
modulus-sym/      <- empty submodule pointer, unused
```

Status of every folder: [Project status](#project-status--as-of-23092026) above.

There is exactly one dependency list: `PINNmodulusTwo/requirements-gpu.txt`. The
former root-level `requirements.txt` is deleted — a UTF-16 encoded Windows
`pip freeze` that installed the CPU wheels, and which the GPU guide already
listed as a trap. Recoverable from the git history if anyone still needs it.

### `PINNmodulusTwo/` — the model

A recurrent PINN: a Modulus `FCLayer` MLP with a per-layer learnable swish,
wrapped in a PyTorch recurrence that feeds the model's own past predictions back
in as temperature history. The loss combines a data term, the anisotropic heat
residual, and the symmetry boundary condition `dT/dx = 0` at the cell centre.
Training is free-running: the data loss is taken on the model's own
autoregressive rollout, seeded only by the measured initial condition. There is
no teacher forcing anywhere.

It trains on the **whole plan sheet, OP01–OP16** — constant drivers and profiles
together. Until 31.08.2026 this was split into a constant-driver project and a
profile extension next door; they are merged, because the profile pipeline is a
strict superset (a constant driver is a profile that does not move) and two
copies of the same model only drift apart. `--resample point
--no-driver-history` reproduces the old constant-only preprocessing exactly.

From OP08 the drivers become **profiles** that vary in time — a fluid
temperature profile, a pre-simulated CC-CV current whose CV phase tapers the
current away, and in OP15 a volume-flow profile. That is what the recurrence is
for: two OPs can share the same instantaneous driver values at some time `t` and
have very different temperatures because their *history* differed.

| file | what it is |
|---|---|
| `model.py` | `RecurrentField` + the Modulus MLP + `rollout` |
| `physics.py` | the nondimensional anisotropic heat residual and the Neumann BC |
| `data.py` | loading, normalisation, driver resampling, the three reports |
| `op_registry.py` | the plan sheet in code: OP01–OP16, tiers, the split. Runs without data |
| `op_metrics.py` | per-OP metrics: MAE, RMSE, peak error, transient vs. quiescent |
| `train.py` | the training loop, evaluation, checkpoint |
| `tests/`, `selftest.py` | seconds, no data, no GPU |

Start with [`PINNmodulusTwo/README.md`](PINNmodulusTwo/README.md). How it works
internally — control flow, the model, where to extend it — is in
[`PINNmodulusTwo/ARCHITECTURE.md`](PINNmodulusTwo/ARCHITECTURE.md) (in German);
the GPU server setup is
[`PINNmodulusTwo/README_GPU_SERVER.md`](PINNmodulusTwo/README_GPU_SERVER.md).

*Outdated, kept as history: real-data runs started on 31.08.2026, and the
current numbers are under [Project status](#pinnmodulustwo--where-it-stands).*
Originally written on 31.08.2026: **nothing here has been measured on the real data yet.** Every MAE in the
repository came off a synthetic fixture, which is why the benchmarks were
deleted rather than kept: a sweep that ranks configurations none of which has
beaten a trivial predictor is a ranking between losers. `train.py` now prints
both trivial predictors next to every OP's MAE, so one run answers that.

### `GridCNN/` — the second model

An independent CNN that predicts the whole 3 × 11 × 11 field at once instead of
point by point. It uses the same data, split and metrics as the PINN so the two
can be compared. Start with the top section of
[`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md) for the current state. The design
rationale is in [`GridCNN/README.md`](GridCNN/README.md) (both in German).

### `data_cache/` — the OP bundles

One `OP*.npz` per operating point. **Not tracked in git**, so it never arrives
with a fresh clone. `data.py` searches, first hit wins:

1. `PINNmodulusTwo/data_cache/` — project-local override
2. `PINNmodulusTwoExtProfiles/data_cache/` — where a pre-merge cache may still sit
3. `data_cache/` — **preferred**: shared, top level
4. `legacy/battery_surrogate_agenticWorkflow/data_cache/`

The material-property CSVs in `PINNmodulusTwo/material_properties/` are equally
untracked. Requesting an OP with no bundle fails immediately and lists what is
available, instead of surfacing after the first training run.

### `legacy/` — earlier approaches

Superseded, kept only so earlier results stay reproducible. Not maintained, not
part of the active pipeline. See [`legacy/README.md`](legacy/README.md).
