# Battery temperature surrogate

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

### `PINNmodulusTwo/` — the active approach

A recurrent PINN: a Modulus `FCLayer` MLP with a per-layer learnable swish,
wrapped in a PyTorch recurrence that feeds the model's own past predictions back
in as temperature history. The loss combines a data term, the anisotropic heat
residual, and the symmetry boundary condition `dT/dx = 0` at the cell centre.

Start with [`PINNmodulusTwo/README.md`](PINNmodulusTwo/README.md); for the GPU
server setup and the full benchmark session see
[`PINNmodulusTwo/README_GPU_SERVER.md`](PINNmodulusTwo/README_GPU_SERVER.md).

The recurrence is deliberately **not** adaptive: the history spacing `δ`, the lag
count `k`, the lag gates and the hybrid `rate_lags` are all fixed
hyperparameters. Learned are the MLP weights, the per-layer swish `β`, and the
two physics gains `src_gain` / `diff_gain`.

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
