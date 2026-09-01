# PINNmodulusTwo — Approach 2 (Modulus-as-a-tool + own recurrence)

> **Neu hier? → [`FAHRPLAN.md`](FAHRPLAN.md).** Ein Einstieg, gegatterte
> Reihenfolge, und was lokal zu tun ist. Diese Datei bleibt Nachschlagewerk.

> **Update 31.08.2026 — ein Projekt, ein Datensatz, keine Benchmarks.**
>
> * `PINNmodulusTwoExtProfiles/` ist hier aufgegangen. Die Profil-Pipeline ist
>   eine echte Obermenge der konstanten — ein konstanter Treiber ist ein Profil,
>   das sich nicht bewegt — also gibt es keinen Grund für zwei Projekte.
>   Trainiert wird auf dem ganzen Plansheet **OP01–OP16**.
> * Die acht Benchmark-Skripte sind gelöscht und werden Schritt für Schritt neu
>   aufgebaut. Was sie an Nützlichem konnten, kann `train.py` selbst: `--val-ops`
>   / `--test-ops`, per-OP-Metriken je Tier, Coverage-Report und die beiden
>   trivialen Vorhersager neben jeder Zeile.
> * `train.py` schreibt jetzt `artifacts/model.pt` — vorher lag die einzige
>   `torch.save` im gelöschten `bench_common.py`.
>
> Warum, und in welcher Reihenfolge die Messungen zurückkommen:
> [`FAHRPLAN.md`](FAHRPLAN.md) §0 und §3.

Implementation of **method #2** from the Notion page *"Battery Model with NVIDIA
MODULUS"*: use Modulus as much as practical, but bring our **own recurrence** in
PyTorch. Roughly a 50:50 Modulus / PyTorch split. **Temperature only** — `bc_V`
is deliberately out of scope.

Trains on eleven of the sixteen plan-sheet OPs, validates on **OP06 + OP09**
(`--val-ops`: what a tuning decision may look at) and reports **OP13, OP15,
OP16** (`--test-ops`: the extrapolation tier, read once, never selected on).
`op_registry.py` holds that split and argues for it; run it, it needs no data.

**OP17–OP19 are not part of this.** They are the mini-module *measurement*
comparison — measured data rather than simulation, partly discharge where
OP01–OP16 are all charge — and of the three only OP19 exists in this pipeline at
all. `--measurement-ops OP19` rolls it out and reports it, never trains or
selects on it.

## What comes from Modulus vs. PyTorch

| Modulus (`modulus_env`) | PyTorch |
|---|---|
| `modulus.models.module.Module` base (save/load, device, meta) | Learnable swish `x·sigmoid(β·x)`, one **β per layer** |
| `modulus.models.layers.FCLayer` (weight-norm linear blocks) | Recurrence: raw history or hybrid history (`T(t-Δgrid)` + rate segments) |
| MLP function approximator for the field | Differentiable time interpolation, so a lag may land between two grid points |
| — | **Raw** or **hybrid** history mode via config/CLI |
| — | Physics: autograd Hessian in space + **finite-difference in time** (`bdf1`/`bdf2`) or `autograd` |

## Files

- `data.py` — loads the OPs from the cached `.npz` (JR1 heat = `q_source[:,0]`,
  a total power in W, made volumetric by one division: `q_dot = jr1_w / V_JR1`),
  pooled z-score for temperature, shared `L_ref`/`T_span_ref` non-dimensionalisation,
  anisotropic Fourier tensor, and the per-timestep **config feature block**.
  Owns everything the profiles force: anti-aliased driver resampling
  (`--resample mean`), causal driver-rate channels (`--driver-rate-lags`), and
  the four reports — `normalisation_report`, `profile_report` (what the bundles
  actually contain, against what the plan sheet claims), `coverage_report`
  (which held-out driver leaves the trained range, and by how many sigmas —
  including channels with NO training variance, where the value is forced to 0
  and the network is never told it differs) and `energy_balance_report` (can the
  source account for the temperature rise? It is the only check that can see a
  uniform factor on `Qsrc`, and on 31.08.2026 it found one: 121x).
- `grid.py` — the 363 points as a structured **3 × 11 × 11** raster (three
  x-planes, the same regular 11 × 11 in (y, z) on each). Derives the permutation
  from `bundle.xn` at runtime and **checks** it: a geometry that is not a tensor
  product fails here with the reason instead of training on a scrambled image.
  Only used by `--arch cnn`.
- `cnn_model.py` — `ConvRecurrentField`: the same recurrence with the Modulus MLP
  replaced by convolutions over (y, z), the three x-planes folded into the
  channel axis. See **Why a CNN, and what it changes** below.
- `physics_grid.py` — the same heat residual, differenced on that raster instead
  of taken by autograd. Required, not optional: see the same section.
- `op_registry.py` — the plan sheet in code: OP01–OP16, which OP carries which
  profile, the tiers, and the split. **Runs without data.**
- `op_metrics.py` — per-OP rollout metrics. Not just a mean: a CC-CV OP spends
  most of its samples in the easy CC phase and a short window in the CV taper,
  so `mae_transient` / `mae_quiescent`, `peak_err` (the number an aging model
  consumes) and `late_mae` are reported alongside `mae`/`rmse`.
- `model.py` — `LearnableSwish`, `ModulusMLP` (Modulus `FCLayer`s), and
  `RecurrentField`. The recurrence is deliberately **not** adaptive: `δ`, `k`,
  `delta_grid`, `rate_lags` and the lag gates are all fixed hyperparameters —
  configurable, never trained. Learned are only the MLP weights and the per-layer
  swish `β`; `src_gain`/`diff_gain` are pinned at 1.0 unless `--learn-gains`.
  `residual_output` is **off by default** — the net predicts the absolute
  normalised temperature. Switching it on makes `field()` return
  `level(t) + net(...)`, which carries the level through an integrator of gain
  exactly 1 with no leak and makes every rollout run away; see below.
- `physics.py` — nondimensional anisotropic heat residual; space via autograd,
  time via the finite-difference `(T(t) − T(t−δ))/δ` over the recurrence. The
  assembled residual is divided by **one** scale, not each term by its own.
- `train.py` — training loop + evaluation, plots, metrics. One free-running
  rollout per OP per epoch, then `--inner-steps` minibatch updates against it.
- `selftest.py` — seconds-long arithmetic checks on the loss balancing and the
  residual scaling. No data, no GPU.
- `tests/` — rollout stability, the history fast path (bit-exactness against the
  general path), the checkpoint round-trip, and the loss bookkeeping. Needs
  neither Modulus nor the data cache; runs in seconds.
- `config.yaml` — hyperparameters. Since the benchmark scripts are gone this is
  the ONLY place a default lives (CLI overrides available).


## Why a CNN, and what it changes (`--arch cnn`)

The 363 measurement points are **not** a point cloud. They are three x-planes —
cell centre, JR1 centre, housing wall — each carrying the same regular 11 × 11
raster in (y, z); the three `coordinates/*.csv` agree to the last digit on y and
z and differ only in x. The temperature field is therefore literally a
`3 × 11 × 11` image, and the process generating it is **local**: a point's next
temperature depends on its neighbours and on the source there. A 3 × 3
convolution is that stencil, with the locality built in instead of learned.

The MLP has to discover locality from raw coordinates, one point at a time, from
sixteen operating points. That is a lot to ask of this much data. It is the
argument for trying the other function class — not a claim that it wins.
**Nothing here has been measured on the real data**; `FAHRPLAN.md` says when the
comparison is worth running and how to read it.

**Held identical on purpose.** Dataset, split, recurrence (history layout,
`delta`/`delta_grid`/`rate_lags`, the causality clamp, the rollout fast path, the
residual level), `op_metrics`, both trivial predictors — and the *inputs*:
`[xn(3), static(S), config(C), forcing(F), history(k)]` per point, exactly what
the MLP is fed, with the scalars broadcast across the image. Not even the obvious
extra channel (the volumetric source field `Qsrc`, which the MLP currently has to
reconstruct from `q_dot` and the JR1 indicator) is added. A difference in MAE
should be attributable to the architecture and to nothing else.

**Layout.** `(B, F, nx, ny, nz)` is folded to `(B, nx·F, ny, nz)`: 3 × 3 kernels
over y and z (11 levels each, genuinely local; four layers reach 9 of 11 cells,
five reach all of them — no pooling, an 11 × 11 image has nothing to pool), and x
folded into the channels so every layer is fully connected across the three
planes. With three levels there is no locality in x to exploit. Edge padding is
`replicate` (zero gradient one cell out); `zeros` would assert the normalised
zero temperature there, a Dirichlet boundary nobody measured.

**What necessarily changes: the physics term.** `physics.py` takes its spatial
derivatives by autograd with respect to `xn`, which is meaningful only for a
continuous function of `xn`. A convolution reads a lattice. It *does* take `xn`
as three channels, so autograd returns a finite, plausible-looking number — and
that is the danger, because the number answers "how does the prediction respond
to relabelling a pixel's coordinates while its neighbours' temperatures stay
put?", which has nothing to do with conduction. Nothing would raise and `L_phys`
would fall like any other run. So `physics.py` now **refuses** a model carrying a
`grid` attribute, and `physics_grid.py` differences the lattice instead.

What that discretisation costs, stated plainly:

| | autograd (`--arch mlp`) | grid (`--arch cnn`) |
|---|---|---|
| y, z | any point | central differences on **9 of 11** levels per axis |
| x | any position | three planes ⇒ **one quadratic**; `d²T/dx²` is constant in x |
| residual points per step | 1 per forward pass | **243**, from *one* forward pass |

The outer y/z ring carries no residual: there is no boundary condition on the
cell's outer faces to close it with, and a one-sided stencil there would be
inventing one. The x limit is a limit of the **data**, not the method — three
planes cannot yield a third x-mode. The autograd Laplacian only *looks* more
accurate in x because it is free to invent curvature between planes that nothing
measured. `--time-deriv autograd` is unavailable with `--arch cnn` (there is no
continuous time input); `bdf1`/`bdf2` are unchanged, and difference the
recurrence exactly as before.

**Minibatching.** A convolution has no meaning on scattered pixels, so the batch
is over *times* and each one carries the whole field. That is not a concession:
one forward pass supervises all 363 points instead of the single pixel a
pointwise draw uses. `--batch-grid 0` (the default) derives the time counts from
`--batch-data`/`--batch-phys`/`--batch-bc` so each term keeps roughly the point
count it has under `--arch mlp`, and the derived numbers are printed at startup.

Guards live in `tests/test_cnn_grid.py` — seconds, no data, no GPU: the real
coordinates really are 3 × 11 × 11, the permutation round-trips, every second
derivative of a quadratic is exact, `field` and `field_batch` agree, the
convolution is translation-equivariant in y, and `physics.py` refuses the grid
model.

## Why recurrence (profiles)

The simulation configs can be **profiles** (vary in time). Two OPs may share the
same instantaneous config at some time `t` yet have very different temperatures
because their *history* differed. Feeding the temperature history disambiguates
these cases — this is the whole reason method #2 needs recurrence.

`k` (how many history points) and `δ` (their spacing) are **fixed
hyperparameters**, not learned — as are the `rate_lags` in hybrid mode and the
lag gates, which are permanently on. The history layout is configured once and
stays put; only the network trains. Sweep the layout by running `train.py` at
two settings and comparing the held-out MAE, rather than expecting the model to
find it.

Hybrid history keeps the same raw interpolation for the physics residual, but
feeds the network a more compact feature block:

- `T(t-Δgrid)` as an absolute anchor. `Δgrid` is `--delta-grid` (default
  `0.2 s`), a free knob independent of `--subsample`, and used **only** in
  hybrid mode -- raw mode spaces its lags by `δ` instead.
- One rate channel per entry in `rate_lags` (`5 s` and `20 s` by default). The
  segments are cumulative, each starting where the previous ended, and each rate
  is divided by **its own segment length** — the actual distance between the two
  points being differenced:

      Rate 1: [T(t-Δgrid)   - T(t-Δgrid-5)]  / 5
      Rate 2: [T(t-Δgrid-5) - T(t-Δgrid-25)] / 20

  `Δgrid` shifts where the window sits but is not part of any span: the endpoints
  of rate 1 are 5 s apart however far back the anchor is. Dividing by the clamped
  *elapsed* span instead is a singularity: early in the rollout that span
  collapses to one grid step and the rate explodes.

  Dividing by `lag_n * rate_scale` means the channel multiplies everything
  non-smooth — including an untrained net's step-to-step jitter — by
  `A = 1/(lag_n * rate_scale)`, which at `5 s` against a ~1474 s reference span
  is `A ≈ 119`. `A` is printed at startup. It is tolerable here; longer segments
  lower it and generalise worse — see the stability notes below.

Set `history_mode: raw` if you want the original lag stack, or `history_mode:
hybrid` (the default) for the anchor + rates layout.

The physics term supports `time_deriv: bdf1`, `bdf2`, or `autograd`. `bdf2` is
the default and remains the recommended choice when the history buffer is long
enough; `history_at()` always uses raw interpolation so the derivative is not
coupled to the hybrid feature layout.

## Training budget (`--inner-steps`)

The rollout is what costs time: ~7000 *sequential* steps per OP per epoch that
cannot be parallelised. The loop used to spend one of those on a single optimiser
step, so a 60-epoch run over 5 OPs finished after **300 Adam updates** — far too
few for a ~70k-parameter MLP, and the main reason the rollout error stayed large.

Now each rollout is computed once under `no_grad` and reused for `--inner-steps`
minibatch updates of `batch_data` random `(t, point)` pairs. That is not an
approximation of the old objective: the recurrence always detached its history
between steps, so the old full-sequence gradient was already a plain sum of
independent per-`(t, point)` gradients against a trajectory it held constant — a
minibatch estimates the same quantity. At the default 100 the same run takes
**30 000** updates instead of 300, for one rollout's worth of extra cost.

The tradeoff is staleness: after a few updates the frozen buffer is no longer
quite what the current weights would produce. It is refreshed every epoch, so
keep `--inner-steps` in the hundreds. `--inner-steps 1` reproduces the old
budget exactly, which is the honest baseline to compare against.

Measure the new per-epoch time with step 6.3 of `README_GPU_SERVER.md` before
starting a long sweep — every runtime estimate in chapters 7 and 8 hangs on that
one number, and the inner loop shifts it.

## Loss balancing, and what it does to older numbers

Each loss term is divided by a running estimate of its own magnitude before its
weight is applied, so `w_data:w_phys:w_bc` is a ratio between **terms** and not
between their accidental units. `--loss-balance` picks which terms:

| mode | what is divided | consequence |
|---|---|---|
| `ema` *(default)* | all three, `L_data` included | the ratio means the same in epoch 1 and epoch 60 |
| `legacy` | only `L_phys` and `L_bc`; `L_data` stays raw | `L_data` falls by orders of magnitude during a run, so the mixture drifts towards physics and the best `w_phys` becomes a function of `--epochs` |
| `fixed` | all three, divisors frozen after `--balance-warmup` epochs | |

**`ema` is the default, and it changes what a weight means.** Any `w_phys` /
`w_bc` result produced under the old scheme was measured with `L_data` raw. Those
numbers do not carry over — not because either scheme is wrong, but because the
quantity the weight multiplies is a different one. Before comparing against an
older sweep, either re-run it or reproduce the old scheme explicitly:

```bash
python3 PINNmodulusTwo/train.py --loss-balance legacy
```

Two things that are *not* restored by that flag, because they were removed for a
reason rather than switched off:

- **The EMA horizon** is now corrected for `len(ops)` × `--inner-steps`. It used
  to be a per-step decay, so its real horizon in *epochs* silently depended on
  how many OPs were trained. `--loss-balance legacy` keeps the corrected horizon;
  it restores the scheme, not the bug.
- **The per-term residual normalisation** is gone from `heat_residual` in every
  mode. `--residual-norm legacy` restores only the old *overall* divisor
  `sqrt(phys_scale)`; it does not bring back dividing `dTdt`, `aniso` and `Qsrc`
  by three different numbers, which changed the equation rather than scaling it
  (see `physics.heat_residual` and `ARCHITECTURE.md` 1.3).

`config.yaml` holds the defaults and `train.py` is the only entry point, so
there is no longer a second configuration that some other script would run.

## Run

```bash
source .venv/bin/activate
python3 PINNmodulusTwo/train.py --epochs 60
```

Everything — the OP set, the split, the preprocessing — comes from
`config.yaml`. A bare run trains on OP01–OP16 and reports val and test.

Everything not passed comes from `config.yaml`, which is the single source of
defaults. Add `--val-ops OP06 --test-ops OP07` (they are already the config
defaults) to get a held-out number rather than only the in-time tail of an OP
the model trained on. Note that `subsample: 2` makes this slow (~1.5–2.5 h at 60 epochs); use
`--epochs 5` for a quick check, or `--subsample 40` for a pure "does it start"
run.

`--device` steht auf `auto`: läuft auf der GPU, wenn eine verfügbar ist, sonst
auf der CPU. Explizit erzwingen mit `--device cuda` / `--device cuda:1` /
`--device cpu`. Komplettes Server-Setup (Treiber, CUDA-PyTorch, Modulus,
Datenübertragung): **[README_GPU_SERVER.md](README_GPU_SERVER.md)**.

Outputs land in `PINNmodulusTwo/artifacts/`: `metrics.txt`, `training_curves.png`,
`timeseries.png`, and `pred_OP0*.npz` (one per training OP and per held-out OP).

`metrics.txt` reports three groups, and the difference between them is the
point: the training OPs (`MAE train` in-sample, `MAE test` the in-time tail past
`split_t`), `[val ]` — whole unseen OPs a tuning decision may look at, and
`[test]` — whole unseen OPs nothing selected on. Only the last is a report
number, and only if you read it once.

## Notes

- The spatial part of the physics residual uses autograd, so `batch_phys` was
  kept small for the original CPU runs. On a GPU it can be raised considerably —
  that is where most of the speed-up comes from (see `README_GPU_SERVER.md`).
- No AMP/fp16: the residual differentiates the network twice, and reduced
  precision degrades those second derivatives. TF32 is available as an opt-in
  `--tf32` flag but is off by default for the same reason.
- Training is free-running: the data loss is taken on the model's own
  autoregressive rollout (seeded only by the measured initial condition), never
  on ground-truth history. There is NO teacher forcing anywhere in train/eval.
- **`residual_output` is off, and that is not optional.** With it on every run
  aborted in epoch 1 with `L_data = nan`, before a single optimiser step.

  The training loop computes one rollout per OP per epoch under `no_grad` and
  then takes `--inner-steps` updates against that frozen buffer, so the buffer is
  an *input* to the first gradient step. Once it holds `inf` there is no gradient
  left to recover from.

  `residual_output` made `field()` return `level(t) + net(...)`, and
  `level(t) ~ level(t - delta_grid) + mean(net)` is an integrator of gain exactly
  1 with no leak -- any one-signed component of the output accumulates over
  ~7000 steps and nothing pulls it back.

  Measured end to end (synthetic bundle, 20 epochs, 3 seeds, no guards):
  `residual_output: true` aborted **9/9 in every history configuration**, `raw`
  included -- which is where there are no rate channels at all. That is what
  identifies the integrator rather than the rate channel as the cause.
  ARCHITECTURE.md 3.1 has the tables and the confirmation at `n_t = 4000`.

  A better initialisation does NOT fix this: zeroing the output layer makes the
  rollout perfectly stable at init (0/5 over 7000 steps), and twenty Adam steps
  later the next rollout reaches 1e4. The stable region of weight space is small
  and training walks out of it. It is a layout problem, not a starting point.
- **`rate_lags` stays at `[5, 20]`.** The rate channel divides by
  `lag_n * rate_scale`, so a 5 s segment amplifies anything non-smooth by
  `A ~ 119` (printed at startup). That is real, but it is not what aborted the
  runs -- `residual_output` was. With the integrator gone and `--rollout-clamp`
  on, `A ~ 119` is tolerable, and the short segment carries the better signal: a
  rate over 600 s on a ~1474 s trajectory is closer to a progress indicator than
  to a rate, and it generalises worse.

  Measured at the REAL geometry (`n_t = 7000`, synthetic `dTdt_scale` 2.467
  against the real 2.479, so `A` matches the real 119/30), 10 epochs, 3 seeds,
  MAE in degrees C on the held-out tail:

  | `rate_lags` | `A` | MAE test |
  |---|---|---|
  | **`[5, 20]`** | 119 / 30 | **1.207** — best on all three seeds |
  | `[50, 150]` | 12 / 4 | 2.102 |
  | `[200, 600]` | 3 / 1 | 2.507 |
  | `raw` | — | 2.601 |

  No aborts in any of the twelve runs. `--max-rate-amp` caps `A` by rescaling the
  channel and made things worse still (MAE test 0.72 -> 1.08 -> 1.57 as the cap
  tightened), so it stays off.
- **Is the result any good?** `L_data` is a z-scored training loss on the train
  portion; the deliverable is MAE in degrees C on the held-out tail, and the two
  do NOT rank configurations the same way -- `[200, 600]` wins on `L_data` and
  loses on MAE. Against trivial baselines the model is worth several times over:
  6.60 C for "predict the training mean", 11.96 C for "hold the initial
  condition". Those two numbers are printed next to every `--val-ops` /
  `--test-ops` MAE, computed on the OP in hand — see `trivial_baselines`. Choose
  `rate_lags` or `history_mode` on that MAE, never on `L_data`.
- **What transfers between datasets is `A`, not the lag in seconds.** `A` depends
  on `rate_scale = dTdt_scale`, which is a property of the data. The same
  seconds give a different `A` on a different OP set -- check the startup line.
- `--rollout-clamp` (default `50.0`) saturates the rollout buffer in normalised
  temperature units. With `w_phys: 0` it is only a diagnostic -- it keeps a
  runaway rollout finite so the log reports how much of the trajectory ran away
  (`[SATURATED]`, with a count) instead of a single `nan` line carrying no
  information. With the physics term on it is **load-bearing**: the physics
  gradient walks the weights out of the stable region faster, and over 3 seeds
  the clamp turned a 1-in-3 abort at width 128/depth 4 (and a 2-in-3 abort in
  `raw`) into three converging runs. It still does not make a saturated
  trajectory a prediction -- watch the count: falling is the model pulling
  itself together, flat or rising is not.
- `train.py` prints a simple CFL sanity check after loading the data. If the
  estimated `Δt_max` is below the current step, the log warns that the rollout
  may be unstable.
