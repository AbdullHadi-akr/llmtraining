# Battery temperature surrogate

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
PINNmodulusTwo/   <- the model, the training, the tests. One project.
data_cache/       <- OP*.npz bundles (not in git, see above)
legacy/           <- earlier approaches, kept for reference only
```

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

**Two function classes, one pipeline.** `--arch mlp` (default) is the Modulus
coordinate network described above. `--arch cnn` predicts the whole field per
step with convolutions: the 363 points are three x-planes carrying the same
regular 11 × 11 raster in (y, z), so the field is an image and heat conduction —
being local — is what a 3 × 3 kernel already is. Same data, same split, same
recurrence, same inputs, same metrics; the physics residual then comes from grid
stencils rather than autograd, which is required rather than optional. Nothing
about it has been measured on the real data yet. See
[`PINNmodulusTwo/README.md`](PINNmodulusTwo/README.md) *Why a CNN, and what it
changes* and `FAHRPLAN.md`.

From OP08 the drivers become **profiles** that vary in time — a fluid
temperature profile, a pre-simulated CC-CV current whose CV phase tapers the
current away, and in OP15 a volume-flow profile. That is what the recurrence is
for: two OPs can share the same instantaneous driver values at some time `t` and
have very different temperatures because their *history* differed.

| file | what it is |
|---|---|
| `model.py` | `RecurrentField` + the Modulus MLP + `rollout` |
| `physics.py` | the nondimensional anisotropic heat residual and the Neumann BC |
| `grid.py` | the 363 points as the structured 3 × 11 × 11 raster they actually are |
| `cnn_model.py` | `--arch cnn`: the same recurrence, convolutions instead of the MLP |
| `physics_grid.py` | that residual again, differenced on the raster instead of by autograd |
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

**Nothing here has been measured on the real data yet.** Every MAE in the
repository came off a synthetic fixture, which is why the benchmarks were
deleted rather than kept: a sweep that ranks configurations none of which has
beaten a trivial predictor is a ranking between losers. `train.py` now prints
both trivial predictors next to every OP's MAE, so one run answers that.

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
