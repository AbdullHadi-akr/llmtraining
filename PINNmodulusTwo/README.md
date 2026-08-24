# PINNmodulusTwo — Approach 2 (Modulus-as-a-tool + own recurrence)

Implementation of **method #2** from the Notion page *"Battery Model with NVIDIA
MODULUS"*: use Modulus as much as practical, but bring our **own recurrence** in
PyTorch. Roughly a 50:50 Modulus / PyTorch split. **Temperature only** — `bc_V`
is deliberately out of scope. Trains on **OP01, OP02, OP03**.

## What comes from Modulus vs. PyTorch

| Modulus (`modulus_env`) | PyTorch |
|---|---|
| `modulus.models.module.Module` base (save/load, device, meta) | Learnable swish `x·sigmoid(β·x)`, one **β per layer** |
| `modulus.models.layers.FCLayer` (weight-norm linear blocks) | Recurrence: raw history or hybrid history (`T(t-Δgrid)` + rate segments) |
| MLP function approximator for the field | **Learnable δ** via differentiable time interpolation |
| — | **Raw** or **hybrid** history mode via config/CLI |
| — | Physics: autograd Hessian in space + **finite-difference in time** (`bdf1`/`bdf2`) or `autograd` |

## Files

- `data.py` — loads OP01/02/03 from the cached `.npz` (JR1 heat = `q_source[:,0]`),
  pooled z-score for temperature, shared `L_ref`/`T_span_ref` non-dimensionalisation,
  anisotropic Fourier tensor, and a per-timestep **config feature block** that
  already supports time-varying **profiles** (constant for OP01-03).
- `model.py` — `LearnableSwish`, `ModulusMLP` (Modulus `FCLayer`s), and
  `RecurrentField` (recurrence with learnable `δ` and lag gates).
- `physics.py` — nondimensional anisotropic heat residual; space via autograd,
  time via the finite-difference `(T(t) − T(t−δ))/δ` over the recurrence.
- `train.py` — training loop on OP01/02/03 + evaluation, plots, metrics.
- `config.yaml` — hyperparameters (CLI overrides available).

## Why recurrence (profiles)

The simulation configs can be **profiles** (vary in time). Two OPs may share the
same instantaneous config at some time `t` yet have very different temperatures
because their *history* differed. Feeding the temperature history disambiguates
these cases — this is the whole reason method #2 needs recurrence.

`k` (how many history points) and `δ` (their spacing) are **learned** in the
raw history mode: `δ` is a positive `softplus` parameter used through a
differentiable interpolation of the history, and each lag has a sigmoid gate so
unused lags fade to zero weight (effective, learned `k`).

Hybrid history keeps the same raw interpolation for the physics residual, but
feeds the network a more compact feature block:

- `T(t-Δgrid)` as an absolute anchor.
- Two rate channels from `rate_lags` (default `5 s` and `25 s`) computed with
  per-endpoint padding and actual elapsed span.

Set `history_mode: raw` if you want the original lag stack, or `history_mode:
hybrid` if you want the anchor + rates layout.

The physics term supports `time_deriv: bdf1`, `bdf2`, or `autograd`. `bdf2` is
the default and remains the recommended choice when the history buffer is long
enough; `history_at()` always uses raw interpolation so the derivative is not
coupled to the hybrid feature layout.

## Run

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate
python3 PINNmodulusTwo/train.py --epochs 60 --subsample 40
```

Outputs land in `PINNmodulusTwo/artifacts/`: `metrics.txt`, `training_curves.png`,
`timeseries.png`, and `pred_OP0*.npz`.

## Notes

- CPU-first (per the repo Modulus workflow). The spatial part of the physics
  residual uses autograd, so `batch_phys` is kept small.
- Training is free-running: the data loss is taken on the model's own
  autoregressive rollout (seeded only by the measured initial condition), never
  on ground-truth history. There is NO teacher forcing anywhere in train/eval.
- `train.py` prints a simple CFL sanity check after loading the data. If the
  estimated `Δt_max` is below the current step, the log warns that the rollout
  may be unstable.
