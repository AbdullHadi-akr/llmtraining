# PINNmodulusTwo — Approach 2 (Modulus-as-a-tool + own recurrence)

> **Update 27.08.2026 — der erste durchlaufende Test.** Bis dahin brach
> jeder Lauf in Epoche 1 mit `L_data = nan` ab.
> [`README_ERSTER_TEST.md`](README_ERSTER_TEST.md) beschreibt Modell,
> Architektur, Training, Daten, Loss und Ergebnisse vollständig, dazu die
> Ursache des Abbruchs und den Vergleich `hybrid` gegen `raw`.

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
  configurable, never trained. Learned are only the MLP weights and the per-layer
  swish `β`; `src_gain`/`diff_gain` are pinned at 1.0 unless `--learn-gains`.
  With `residual_output` (default) the net predicts the deviation from the
  anchor's spatially averaged temperature level rather than the absolute value.
- `physics.py` — nondimensional anisotropic heat residual; space via autograd,
  time via the finite-difference `(T(t) − T(t−δ))/δ` over the recurrence. The
  assembled residual is divided by **one** scale, not each term by its own.
- `train.py` — training loop + evaluation, plots, metrics. One free-running
  rollout per OP per epoch, then `--inner-steps` minibatch updates against it.
- `bench_common.py` — shared benchmark machinery: per-seed training, mean/std
  aggregation over seeds, the val/test split, and the seed-noise verdict. A
  benchmark only describes its own sweep axis.
- `benchmark_wphys_wbc.py` — 2D sweep of the loss weights `w_phys` x `w_bc`.
- `benchmark_arch.py` — width, depth, history lags and anchor lag (`delta_grid`),
  one axis at a time.
- `smallBench.py` — 2-5 minute smoke test; run it before any long sweep.
- `config.yaml` — hyperparameters, matching what the benchmarks run
  (CLI overrides available).
- [`README_MODEL_CRITIQUE.md`](README_MODEL_CRITIQUE.md) — what was wrong with the
  model, what is fixed, what is still open, and **what you have to see in which
  test to know which step comes next**. Everything in it is so far verified
  mathematically only, not measured on real data; it names the run that settles
  that. Read it before committing GPU days to a sweep.

## Why recurrence (profiles)

The simulation configs can be **profiles** (vary in time). Two OPs may share the
same instantaneous config at some time `t` yet have very different temperatures
because their *history* differed. Feeding the temperature history disambiguates
these cases — this is the whole reason method #2 needs recurrence.

`k` (how many history points) and `δ` (their spacing) are **fixed
hyperparameters**, not learned — as are the `rate_lags` in hybrid mode and the
lag gates, which are permanently on. The history layout is configured once and
stays put; only the network trains. Sweep the layout with `benchmark_arch.py`
rather than expecting the model to find it.

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
  (see `README_MODEL_CRITIQUE.md` §1.3).

`smallBench.py` and both benchmark scripts pass all of these through to `fit()`,
and `config.yaml` holds the defaults — so a bare run, a smoke test and a
benchmark all balance the same way.

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
- **`residual_output` is off and `rate_lags` are long, and neither is optional.**
  With the old settings (`residual_output: true`, `rate_lags: [5, 20]`) every run
  aborted in epoch 1 with `L_data = nan`, before a single optimiser step.

  The training loop computes one rollout per OP per epoch under `no_grad` and
  then takes `--inner-steps` updates against that frozen buffer, so the buffer is
  an *input* to the first gradient step. Once it holds `inf` there is no gradient
  left to recover from.

  Two things drove it there. `residual_output` made `field()` return
  `level(t) + net(...)`, and `level(t) ~ level(t - delta_grid) + mean(net)` is an
  integrator of gain exactly 1 with no leak -- any one-signed component of the
  output accumulates over ~7000 steps and nothing pulls it back. Separately, the
  hybrid rate channel divides by `lag_n * rate_scale`, which at 5 s against a
  ~1474 s reference span is a gain of `A ~ 119` on anything non-smooth. `A` is
  printed at startup and warned about above ~100.

  Measured end to end (synthetic bundle, 20 epochs, 3 seeds, no guards):
  `residual_output: true` aborted **9/9 in every history configuration**, `raw`
  included -- which is where there are no rate channels at all, so the integrator
  is the primary driver. With it off, all of them ran; `hybrid` + `[200, 600]`
  reached `L_data ~ 1e-4`, two orders of magnitude better than anything else.
  ARCHITECTURE.md 3.1 has the tables and the confirmation at `n_t = 4000`.

  A better initialisation does NOT fix this: zeroing the output layer makes the
  rollout perfectly stable at init (0/5 over 7000 steps), and twenty Adam steps
  later the next rollout reaches 1e4. The stable region of weight space is small
  and training walks out of it. It is a layout problem, not a starting point.
- **Is the result any good?** `L_data` is a z-scored training loss on the train
  portion; the deliverable is MAE in degrees C on the held-out tail, and the two
  do NOT rank configurations the same way. Measured at 128/4 with `w_phys: 0.1`
  over 3 seeds: MAE test 0.38-2.43 C for `hybrid [200, 600]` and 0.43-1.70 C for
  `raw`, against 6.60 C for "predict the training mean" and 11.96 C for "hold the
  initial condition". So the model is worth 3-17x over a trivial baseline -- but
  `hybrid [200, 600]`'s 100x advantage in `L_data` does NOT survive the switch to
  MAE, where `raw` is marginally ahead and the gap is inside the seed spread. Use
  `benchmark_arch.py`, not `L_data`, to choose `rate_lags` or `history_mode`.
- **What transfers to real data is `A`, not the lag in seconds.** These numbers
  come from a synthetic bundle whose `rate_scale` is 16.38 against 2.479 for the
  real OP01-05. The shipped `[200.0, 600.0]` gives `A = 2.97 / 0.99` on real data
  and was measured at `A = 0.45 / 0.15`. The rule is "get `A` to O(1)", not "use
  200 and 600 seconds".
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
