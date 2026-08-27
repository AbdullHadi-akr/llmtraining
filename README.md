# Battery temperature surrogate

## TODO — the data is missing from a fresh clone

Nothing in `data_cache/` or `data_raw/` is tracked in git (the `.gitignore`
keeps only sources, READMEs and a few configs), so a fresh clone **cannot
train**. Everything below has to be copied onto the machine by hand. Tick a box
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

`PINNmodulusTwo/data.py` also accepts three fallback locations (see
[`data_cache/`](#data_cache--the-op-bundles) below), but **`data_cache/` at the
top level is the one to use** for anything new.

Which OPs are needed, in priority order — the defaults in
[`PINNmodulusTwo/config.yaml`](PINNmodulusTwo/config.yaml) are
`ops: [OP01…OP05]`, validation on OP06, report on OP07:

- [ ] `OP01.npz` — train (25 °C, ṁ 0.0013)
- [ ] `OP02.npz` — train (15 °C, ṁ 0.0013)
- [ ] `OP03.npz` — train (30 °C, ṁ 0.0013)
- [ ] `OP04.npz` — train (25 °C, ṁ 0.0026)
- [ ] `OP05.npz` — train (40 °C, ṁ 0.0026)
- [ ] `OP06.npz` — **validation**, benchmark selection ranks on this one (25 °C, ṁ 0)
- [ ] `OP07.npz` — **report only**, never part of any selection (10 °C, ṁ 0)
- [ ] `OP08.npz` … `OP16.npz`, `OP19.npz` — optional, only for wider sweeps.
      `OP16` is what `python3 PINNmodulusTwo/data.py` uses as its held-out demo.

Without OP01–OP05 nothing trains; without OP06 the benchmarks cannot select a
model; without OP07 there is no report number.

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

- [ ] `data_raw/OP01/OP01/` … `data_raw/OP07/OP07/` — the seven OPs the defaults use
- [ ] `data_raw/OP08/OP08/` … `data_raw/OP19/OP19/` — optional, for the wider sweeps

Already in git, so nothing to copy: the three coordinate CSVs under
`legacy/battery_surrogate_agenticWorkflow/coordinates/`, `op_matrix.yaml`,
`build.yaml`.

Then, from the repo root:

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07
```

which writes straight into the top-level `data_cache/`.

### 3. Check it worked

```bash
ls data_cache/
python3 -c "import sys; sys.path.insert(0, 'PINNmodulusTwo'); import data; print(data.available_ops())"
```

The second command needs the training environment (numpy) and should list every
OP you copied. A missing OP fails immediately with the list of
what is available, rather than after the first training run.

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

Es gibt nur eine Abhängigkeitsliste: `PINNmodulusTwo/requirements-gpu.txt`. Die
frühere `requirements.txt` im Root ist gelöscht — ein UTF-16-kodierter
Windows-`pip freeze`, der die CPU-Wheels installierte und den die
GPU-Anleitung ohnehin als Falle führte. Aus der Git-Historie holbar, falls
jemand sie noch braucht.

### `PINNmodulusTwo/` — the active approach

A recurrent PINN: a Modulus `FCLayer` MLP with a per-layer learnable swish,
wrapped in a PyTorch recurrence that feeds the model's own past predictions back
in as temperature history. The loss combines a data term, the anisotropic heat
residual, and the symmetry boundary condition `dT/dx = 0` at the cell centre.

Start with [`PINNmodulusTwo/README.md`](PINNmodulusTwo/README.md). Wie das Ganze
intern abläuft — Kontrollfluss, Modell, Erweiterungspunkte — steht in
[`PINNmodulusTwo/ARCHITECTURE.md`](PINNmodulusTwo/ARCHITECTURE.md); für das
GPU-Server-Setup und die volle Benchmark-Session siehe
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
only source and README files), so it never arrives with a fresh clone — it has to
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
