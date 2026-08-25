# Point-wise MLP — Data-Driven Battery Surrogate

A small, self-contained baseline that maps **one spatial point at one time** to
**temperature and cell voltage**:

$$
(x, y, z, t, \text{sim\_config}) \; \mapsto \; (\hat{T}, \; \widehat{bc\_V})
$$

This is the *data-driven MLP* from the project's model outline. It deliberately
ignores geometry and physics — it is the baseline that later models
(MLP + recurrent, PINN) are compared against.

---

## Quick facts

| Property | Value |
|----------|-------|
| Input features | **11** — `x, y, z, t` + 7 canonical sim-config channels |
| Outputs | **2** — `T` (°C) and `bc_V` (V) from one shared network |
| Activation | Swish with **one learnable β per hidden layer** |
| Normalization | **z-score** on all inputs and outputs (fit on train OPs only) |
| Spatial coverage | **all sensors** (363 = 121 cc + 121 g + 121 jr1c), never subsampled |
| Temporal subsampling | keep every `subsample_time`-th timestep (speed) |
| Framework | pure PyTorch + NumPy |

The 7 canonical sim-config channels (fixed order) are:
`c_rate, cell_current, fluid_initial_temp, fluid_inlet_temp, fluid_mass_flow,
soc_start, solid_initial_temp`.

---

## Package layout

```
battery_surrogate/model/
├── mlp_pointwise.py      # PointwiseMLP + LearnableSwish
├── features_pointwise.py # OpBundle → (X, Y) feature blocks + ts interpolation
├── normalizer.py         # streaming z-score fit / transform / inverse / save-load
├── dataset_pointwise.py  # IterableDataset streaming one OP at a time
├── split.py              # config-driven train/val/test OP split + validation
├── trainer.py            # training loop (bc_V de-inflation, early stop, scheduler)
└── evaluate.py           # metrics + bc_V spatial-independence check
```

CLI entry points live in `battery_surrogate/cli/`:

- `train_mlp_pointwise.py` — train from a YAML config.
- `optuna_mlp_pointwise.py` — phased hyperparameter search.

Default config: `configs/model/mlp_pointwise.yaml`.

---

## Configuration

Everything is controlled from one YAML file. Change the OP lists to decide what
trains and what is held out — **no code changes needed**.

```yaml
seed: 42

data:
  train_ops: [OP01, OP02, OP08, OP09, OP10, OP11, OP12, OP13]
  val_ops:   [OP14, OP15]
  test_ops:  [OP16, OP19]
  subsample_time: 50        # time only; space is ALWAYS full
  ts_extrapolation: clamp   # edge-clamp time-series config to t_fast range

model:
  n_hidden_layers: 3
  hidden_size: 128
  swish_beta_init: 1.0
  swish_beta_learnable: true

train:
  epochs: 50
  batch_size: 4096
  lr: 1.0e-3
  weight_decay: 0.0
  grad_clip: 1.0
  early_stopping_patience: 10

loss:
  T_weight: 1.0
  bc_V_weight: 1.0          # de-inflated by 1/n_sensors internally

output:
  ckpt_dir: artifacts/mlp_pointwise/{timestamp}
```

> **bc_V redundancy**: at each timestep all 363 sensors share the same `bc_V`
> target, so `bc_V` appears 363× per step. The trainer divides the `bc_V` loss
> by `n_sensors` so it can't dominate — while still learning that `bc_V` is
> independent of `x, y, z`. That independence is *measured* after training via
> `bc_V_spatial_variance` (should be ≈ 0).

---

## Command-line usage

Run from the workflow root (`battery_surrogate_agenticWorkflow/`) with the
package importable (`PYTHONPATH=src` or an editable install).

Train with defaults:

```bash
python -m battery_surrogate.cli.train_mlp_pointwise \
    --config configs/model/mlp_pointwise.yaml
```

Fast smoke run (1 epoch, aggressive time subsampling):

```bash
python -m battery_surrogate.cli.train_mlp_pointwise \
    --config configs/model/mlp_pointwise.yaml \
    --epochs 1 --data.subsample_time 200
```

The command prints a JSON summary (best val loss, checkpoint dir, parameter
count, per-layer β) and writes to `output.ckpt_dir`:

- `best.pt` — best model state dict + history
- `normalizer.json` — z-score statistics
- `history.json` — train/val loss per epoch

Hyperparameter search (needs `optuna`):

```bash
python -m battery_surrogate.cli.optuna_mlp_pointwise \
    --config configs/model/mlp_pointwise.yaml --n-trials 30
```

---

## Programmatic usage

The whole pipeline is callable so you don't have to shell out each time.

```python
from battery_surrogate.cli.train_mlp_pointwise import train_from_config
import yaml, pathlib

config = yaml.safe_load(
    pathlib.Path("configs/model/mlp_pointwise.yaml").read_text()
)
config["data"]["subsample_time"] = 100   # override anything inline
config["train"]["epochs"] = 10

summary = train_from_config(config)
print(summary["best_val_loss"], summary["ckpt_dir"])
```

Lower-level building blocks (used by the notebook for plots):

```python
import torch, yaml, pathlib
from battery_surrogate.model.split import resolve_split
from battery_surrogate.model.normalizer import PointwiseNormalizer
from battery_surrogate.model.mlp_pointwise import PointwiseMLP
from battery_surrogate.model.evaluate import evaluate_on_ops

# rebuild a trained model
model = PointwiseMLP(n_features=11, n_hidden_layers=3, hidden_size=128)
ckpt = torch.load("artifacts/mlp_pointwise/<run>/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
normalizer = PointwiseNormalizer.load(
    pathlib.Path("artifacts/mlp_pointwise/<run>/normalizer.json")
)

metrics = evaluate_on_ops(
    model, ["OP16"], normalizer, subsample_time=100
)
print(metrics["T"], metrics["bc_V"], metrics["bc_V_spatial_variance"])
```

### Predict at arbitrary points

```python
from battery_surrogate.model.features_pointwise import assemble_pointwise_block
from battery_surrogate.data.loader import load_op

bundle = load_op("OP16")
X, Y_true, sensor_ids = assemble_pointwise_block(bundle, time_index=1000)
X_norm = normalizer.transform_X(X)
with torch.no_grad():
    Y_pred = normalizer.inverse_Y(model(torch.from_numpy(X_norm)).numpy())
# Y_pred[:, 0] = T (°C) per sensor, Y_pred[:, 1] = bc_V (V)
```

---

## Notebook

`notebooks/mlp_pointwise_demo.ipynb` runs the full loop end-to-end on a small OP
subset: fit normalizer → train → plot loss curves and β evolution → evaluate on
held-out data → plot predicted vs. actual `T` and `bc_V` → run a short Optuna
search and plot trial results. It is the recommended starting point.

---

## Outputs & metrics

`evaluate_on_ops` returns, in physical units (°C, V):

| Key | Meaning |
|-----|---------|
| `T` / `bc_V` | dict of `mse`, `mae`, `max_error`, `r2` |
| `per_op` | same metrics broken down per OP |
| `bc_V_spatial_variance` | variance of `bc_V` across sensors at fixed t (≈0 is good) |

Target ballpark for a healthy baseline: `T` R² > 0.95, `T` max error < 5 °C,
`bc_V` MSE < 0.01 V², `bc_V_spatial_variance` < 1e-3, and per-layer β staying in
roughly `[0.5, 2.0]` (outside that range suggests the Swish is overfitting).

---

## Notes & limitations

- The model does **not** encode physics or geometry — expect it to struggle to
  extrapolate to unseen operating regimes (e.g. `OP19`, which is module-test
  data with a different input distribution).
- The raw OP source tree contains a few naming drifts: some folders use
  `Input Signale.csv` instead of `Inputsignale.csv`, and `OP11` has shorter
  `Inputsignale` headers. The data pipeline now accepts those variants, so the
  problem is documented here but handled automatically during loading.
- `subsample_time` trades accuracy for speed; set it to `1` for full resolution.
- Never set a spatial subsample — the point-wise model needs the full sensor set
  to learn the spatial field.
