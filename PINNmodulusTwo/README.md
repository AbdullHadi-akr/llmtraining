# PINNmodulusTwo — Approach 2 (Modulus-as-a-tool + own recurrence)

Implementation of **method #2** from the Notion page *"Battery Model with NVIDIA
MODULUS"*: use Modulus as much as practical, but bring our **own recurrence** in
PyTorch. Roughly a 50:50 Modulus / PyTorch split. **Temperature only** — `bc_V`
is deliberately out of scope. Trains on **OP01–OP05**, validates on **OP06**
(that is what benchmark selection ranks on) and reports **OP07**, which never
takes part in any selection.

## What comes from Modulus vs. PyTorch

| Modulus (`modulus_env`) | PyTorch |
|---|---|
| `modulus.models.module.Module` base (save/load, device, meta) | Learnable swish `x·sigmoid(β·x)`, one **β per layer** |
| `modulus.models.layers.FCLayer` (weight-norm linear blocks) | Recurrence: raw history or hybrid history (`T(t-Δgrid)` + rate segments) |
| MLP function approximator for the field | Differentiable time interpolation, so a lag may land between two grid points |
| — | **Raw** or **hybrid** history mode via config/CLI |
| — | Physics: autograd Hessian in space + **finite-difference in time** (`bdf1`/`bdf2`) or `autograd` |

## Files

- `data.py` — loads the OPs from the cached `.npz` (JR1 heat = `q_source[:,0]`),
  pooled z-score for temperature, shared `L_ref`/`T_span_ref` non-dimensionalisation,
  anisotropic Fourier tensor, and a per-timestep **config feature block** that
  already supports time-varying **profiles**.
- `model.py` — `LearnableSwish`, `ModulusMLP` (Modulus `FCLayer`s), and
  `RecurrentField`. The recurrence is deliberately **not** adaptive: `δ`, `k`,
  `delta_grid`, `rate_lags` and the lag gates are all fixed hyperparameters —
  configurable, never trained. Learned are only the MLP weights, the per-layer
  swish `β`, and the physics gains `src_gain`/`diff_gain`.
- `physics.py` — nondimensional anisotropic heat residual; space via autograd,
  time via the finite-difference `(T(t) − T(t−δ))/δ` over the recurrence.
- `train.py` — training loop + evaluation, plots, metrics.
- `bench_common.py` — shared benchmark machinery: per-seed training, mean/std
  aggregation over seeds, the val/test split, and the seed-noise verdict. A
  benchmark only describes its own sweep axis.
- `benchmark_wphys_wbc.py` — 2D sweep of the loss weights `w_phys` x `w_bc`.
- `benchmark_balance.py` — **run this before the weight sweep**: how the loss
  terms are scaled relative to each other, and whether the optional input
  channels help. A weight only means something once the scaling under it is
  settled (see *Loss balancing* below).
- `benchmark_arch.py` — width, depth, history lags and anchor lag (`delta_grid`),
  one axis at a time.
- `smallBench.py` — 2-5 minute smoke test; run it before any long sweep.
- `selftest.py` — seconds-long arithmetic checks on the loss balancing and
  residual scaling. No data, no GPU. The properties it guards are invisible
  in a training log, and the next chance to notice is a multi-hour sweep.
- `config.yaml` — hyperparameters, matching what the benchmarks run
  (CLI overrides available).
- `ARCHITECTURE.md` — **[Kontrollfluss und Modellbeschreibung](ARCHITECTURE.md)**:
  wie ein Lauf abläuft, was gelernt wird und was fest ist, wo man für eine
  Erweiterung ansetzt, und welche Fallstricke schon einmal zugeschlagen haben.

## Why recurrence (profiles)

The simulation configs can be **profiles** (vary in time). Two OPs may share the
same instantaneous config at some time `t` yet have very different temperatures
because their *history* differed. Feeding the temperature history disambiguates
these cases — this is the whole reason method #2 needs recurrence.

`k` (how many history points) and `δ` (their spacing) are **fixed
hyperparameters**, not learned — as are the `rate_lags` in hybrid mode and the
lag gates, which are permanently on. The history layout is configured once and
stays put; only the network and the two physics gains train. Sweep the layout
with `benchmark_arch.py` rather than expecting the model to find it.

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
  collapses to one grid step and the rate explodes, which is what made every
  sweep point diverge to NaN.

Set `history_mode: raw` if you want the original lag stack, or `history_mode:
hybrid` (the default) for the anchor + rates layout.

The physics term supports `time_deriv: bdf1`, `bdf2`, or `autograd`. `bdf2` is
the default and remains the recommended choice when the history buffer is long
enough; `history_at()` always uses raw interpolation so the derivative is not
coupled to the hybrid feature layout.

## Loss balancing — what `w_phys` and `w_bc` actually mean

The three loss terms live on completely different scales, so a raw weight is not
a mixing ratio. Each term is therefore divided by a running estimate of its own
magnitude before its weight is applied (`--loss-balance`, `config.yaml:
loss_balance`):

| mode | which terms are divided | consequence |
|---|---|---|
| `ema` (default) | `L_data`, `L_phys`, `L_bc` | `w_data:w_phys:w_bc` is a genuine ratio. `w_phys = 1` means "physics contributes as much as data", and it means that in epoch 1 as much as in epoch 60. |
| `legacy` | only `L_phys`, `L_bc` | the historical scheme. `L_data` stays raw and falls by orders of magnitude during a run, while the two normalised terms stay pinned near 1 — so the mixture drifts steadily towards physics, and the best `w_phys` becomes a function of `--epochs`. |
| `fixed` | all three, frozen after warm-up | deterministic, no feedback between a loss and its own scale. |

This is why `benchmark_balance.py` runs first. It trains at one fixed weight
point and logs `w_phys*L_phys_bal / (w_data*L_data_bal)` per epoch — the mixture
the optimiser actually saw. Under `legacy` that number moves during the run;
under `ema` it does not. Tuning weights before knowing which regime you are in
measures the drift as much as the weights.

Two related scaling knobs:

- `--residual-norm` (default `rms`). The `*_scale` constants in `data.py` are RMS
  values, so `x / scale` is what gives `mean(res²) = 1`. The original code
  divided by `sqrt(scale)`, leaving `mean(res²) = scale` — i.e. the `dTdt`,
  `aniso` and `Qsrc` terms kept their original size gap despite the
  normalisation. `legacy` restores that behaviour for comparison.
- `bc_scale` is now **measured** from the training data (the RMS spatial
  gradient across x-neighbouring grid points) instead of the old `1 / L_ref`
  guess, which never involved a temperature at all. `--residual-norm legacy`
  does *not* restore the old value, so pre-fix `L_bc` numbers are not
  reproducible.

## Optional input features (off by default)

Both widen the network input, so they are measured rather than assumed —
`benchmark_balance.py --part 2` is the axis for them.

- `--forcing-energy`: cumulative injected heat as a second forcing channel,
  expressed as an adiabatic temperature rise in sigmas. A thermal system
  *integrates* power; the instantaneous `q_dot` the net gets says how hard it is
  being heated right now, never how much heat is already in the cell.
- `--config-rates`: `d(config)/dt` for the config channels that are genuine time
  profiles. Same argument as the temperature recurrence, applied to the configs.

`train.py` also reports which config channels are **profiles** (varying in time
within an OP) and which are **labels** (constant per OP, differing between them).
Label channels are constants the network can memorise per OP — with five
training OPs they can act as an OP identifier that cannot transfer to OP06/OP07.

## Run

```bash
source .venv/bin/activate
python3 PINNmodulusTwo/train.py --epochs 60
```

Everything not passed comes from `config.yaml`, which now holds the same
settings the benchmarks use — so a bare run trains the model the benchmarks
measure. Note that `subsample: 2` makes this slow (~1.5–2.5 h at 60 epochs); use
`--epochs 5` for a quick check, or `--subsample 40` for a pure "does it start"
run.

`--device` steht auf `auto`: läuft auf der GPU, wenn eine verfügbar ist, sonst
auf der CPU. Explizit erzwingen mit `--device cuda` / `--device cuda:1` /
`--device cpu`. Komplettes Server-Setup (Treiber, CUDA-PyTorch, Modulus,
Datenübertragung): **[README_GPU_SERVER.md](README_GPU_SERVER.md)**.

Outputs land in `PINNmodulusTwo/artifacts/`: `metrics.txt`, `training_curves.png`,
`timeseries.png`, and `pred_OP0*.npz`.

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
- `train.py` prints a simple CFL sanity check after loading the data. If the
  estimated `Δt_max` is below the current step, the log warns that the rollout
  may be unstable.
