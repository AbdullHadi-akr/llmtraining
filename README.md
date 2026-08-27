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

Both projects find it there: `PINNmodulusTwo/data.py` and
`PINNmodulusTwoExtProfiles/data.py` each search their own folder first and fall
back to this shared one (full list under [`data_cache/`](#data_cache--the-op-bundles)
below). **`data_cache/` at the top level is the one to use** for anything new —
one copy, both projects.

### 1b. The material-property CSVs — equally missing, equally required

`PINNmodulusTwo/materials.py` reads these from `PINNmodulusTwo/material_properties/`,
and the extension reads that same copy. They are untracked too, so they also have
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

Without it nothing runs at all — `materials.py` is imported by `data.py` in both
projects, so this fails before the first OP is even looked up.

Which OPs are needed, in priority order — the defaults in
[`PINNmodulusTwo/config.yaml`](PINNmodulusTwo/config.yaml) are
`ops: [OP01…OP05]`, validation on OP06, report on OP07:

**`PINNmodulusTwo/` — the constant-driver model** (`ops` in its `config.yaml`):

- [ ] `OP01.npz` — train (25 °C, ṁ 0.0013)
- [ ] `OP02.npz` — train (15 °C, ṁ 0.0013)
- [ ] `OP03.npz` — train (30 °C, ṁ 0.0013)
- [ ] `OP04.npz` — train (25 °C, ṁ 0.0026)
- [ ] `OP05.npz` — train (40 °C, ṁ 0.0026)
- [ ] `OP06.npz` — **validation**, benchmark selection ranks on this one (25 °C, ṁ 0)
- [ ] `OP07.npz` — **report only**, never part of any selection (10 °C, ṁ 0)

Without OP01–OP05 nothing trains; without OP06 the benchmarks cannot select a
model; without OP07 there is no report number.

**`PINNmodulusTwoExtProfiles/` — the profile extension** needs **all sixteen**,
not a subset: OP01–OP05, OP07, OP08, OP10, OP11, OP12, OP14 to train, OP06 and
OP09 to select on, OP13, OP15 and OP16 to report. Its `README.md` states that
**no result has been measured yet**, and this cache is the only reason why.

- [ ] `OP08.npz` … `OP16.npz` — required by the extension, on top of the seven above
- [ ] `OP19.npz` — optional; used by neither config, module-test data (see `op_matrix.yaml`)

`OP16` is also what `python3 PINNmodulusTwo/data.py` uses as its held-out demo.

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

- [ ] `data_raw/OP01/OP01/` … `data_raw/OP07/OP07/` — the seven `PINNmodulusTwo` uses
- [ ] `data_raw/OP08/OP08/` … `data_raw/OP16/OP16/` — the rest of what the extension needs
- [ ] `data_raw/OP19/OP19/` — optional, module-test data, in neither config

Already in git, so nothing to copy: the three coordinate CSVs under
`legacy/battery_surrogate_agenticWorkflow/coordinates/`, `op_matrix.yaml`,
`build.yaml`.

Then, from the repo root:

```bash
# what PINNmodulusTwo needs
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07
# the rest, for PINNmodulusTwoExtProfiles
python3 PINNmodulusTwo/generate_cache.py OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
```

which writes straight into the top-level `data_cache/`, where both projects
find it.

### 3. Check it worked

```bash
ls data_cache/
python3 -c "import sys; sys.path.insert(0, 'PINNmodulusTwo'); import data; print(data.available_ops())"
python3 -c "import sys; sys.path.insert(0, 'PINNmodulusTwoExtProfiles'); import data; print(data.available_ops())"
```

The last two need the training environment (numpy, torch) and should list every
OP you copied — the second from the base project's view, the third from the
extension's. A missing OP fails immediately with the list of what is available,
rather than after the first training run.

---
Predicting the internal temperature field of a battery cell from operating-point
inputs, so cell temperature can feed an aging model without measuring inside the
cell. High-fidelity Batemo + StarCCM+ simulations are run offline to produce
training data; the trained surrogate then runs online under automotive
controller constraints.

## Layout

```
PINNmodulusTwo/              <- the active model, constant operating points
PINNmodulusTwoExtProfiles/   <- extension of it to the OPs whose drivers are PROFILES
data_cache/                  <- OP*.npz bundles (not in git, see below)
legacy/                      <- earlier approaches, kept for reference only
```

There is exactly one dependency list: `PINNmodulusTwo/requirements-gpu.txt`. The
former root-level `requirements.txt` is deleted — a UTF-16 encoded Windows
`pip freeze` that installed the CPU wheels, and which the GPU guide already
listed as a trap. Recoverable from the git history if anyone still needs it.

### `PINNmodulusTwo/` — the active approach

A recurrent PINN: a Modulus `FCLayer` MLP with a per-layer learnable swish,
wrapped in a PyTorch recurrence that feeds the model's own past predictions back
in as temperature history. The loss combines a data term, the anisotropic heat
residual, and the symmetry boundary condition `dT/dx = 0` at the cell centre.

Start with [`PINNmodulusTwo/README.md`](PINNmodulusTwo/README.md). How it works
internally — control flow, the model, where to extend it — is in
[`PINNmodulusTwo/ARCHITECTURE.md`](PINNmodulusTwo/ARCHITECTURE.md) (in German);
for the GPU server setup and the full benchmark session see
[`PINNmodulusTwo/README_GPU_SERVER.md`](PINNmodulusTwo/README_GPU_SERVER.md).
Where the model currently stands — what was broken, what is fixed, and what to
look for in each test before spending GPU days — is in
[`PINNmodulusTwo/README_MODEL_CRITIQUE.md`](PINNmodulusTwo/README_MODEL_CRITIQUE.md).

The recurrence is deliberately **not** adaptive: the history spacing `δ`, the lag
count `k`, the lag gates and the hybrid `rate_lags` are all fixed
hyperparameters. Learned are the MLP weights and the per-layer swish `β`. The
physics gains `src_gain` / `diff_gain` are pinned at 1.0 (`--learn-gains` frees
them again).

### `PINNmodulusTwoExtProfiles/` — the profile extension

`PINNmodulusTwo` trains on OP01–OP05, where every driver (C-rate, fluid inlet
temperature, volume flow) is held constant for the whole charge. From OP08 the
plan sheet turns those drivers into **profiles** that vary in time — a fluid
temperature profile, a pre-simulated CC-CV current, and in OP15 a volume-flow
profile.

This folder extends the same model to the full **OP01–OP16**. The network, the
physics residual and the recurrence are imported unchanged from
`PINNmodulusTwo/`; what it owns is everything the profiles force: anti-aliased
driver resampling, causal driver-rate feature channels, normalisation constants
refitted on the wider set, and its own benchmark — **`profileBench`, the Profile
Tier Benchmark** — because the loss weights are only meaningful relative to
physics scales that have moved.

Start with
[`PINNmodulusTwoExtProfiles/README.md`](PINNmodulusTwoExtProfiles/README.md).
Note that **no result from it has been measured yet**: it needs the data cache
below, which is not in git.

### `data_cache/` — the OP bundles

One `OP*.npz` per operating point. **Not tracked in git** (the `.gitignore` keeps
only sources and documentation), so it never arrives with a fresh clone — it has to
be present on each machine.

`PINNmodulusTwo/data.py` searches these locations and takes the first that
exists, so an existing cache keeps working wherever it already sits:

1. `PINNmodulusTwo/data_cache/` — project-local override
2. `data_cache/` — **preferred**: shared, top level
3. `legacy/battery_surrogate_agenticWorkflow/data_cache/`
4. `battery_surrogate_agenticWorkflow/data_cache/` — pre-restructure location

`PINNmodulusTwoExtProfiles/data.py` searches the same list, with its own
`PINNmodulusTwoExtProfiles/data_cache/` ahead of it. The material-property CSVs
in `PINNmodulusTwo/material_properties/` are equally untracked, and both
projects read that one copy.

Requesting an OP that has no bundle fails immediately and lists what is
available, instead of surfacing after the first training run.

### `legacy/` — earlier approaches

Superseded, kept only so earlier results stay reproducible. Not maintained, not
part of the active pipeline. See [`legacy/README.md`](legacy/README.md).
