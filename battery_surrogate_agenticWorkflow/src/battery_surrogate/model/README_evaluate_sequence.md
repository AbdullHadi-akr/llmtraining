# evaluate_sequence Module

Evaluation helpers for recurrent sequence models with autoregressive rollout.

## Overview

This module provides evaluation functions for `RecurrentPointwise` models that predict battery thermal variables (T, bc_V) using sequence-to-sequence autoregressive rollout.

Unlike the pointwise MLP which predicts each time step independently, recurrent models use their own predictions as history for future steps—revealing **exposure bias** patterns where errors accumulate over time.

## Key Functions

### `evaluate_sequence_model()`

Evaluate a recurrent model on test OPs using autoregressive rollout.

```python
from battery_surrogate.model.evaluate_sequence import evaluate_sequence_model

metrics = evaluate_sequence_model(
    model,              # Trained RecurrentPointwise
    op_ids=["OP16"],    # Test OP IDs
    normalizer=norm,    # Fitted normalizer
    config=config,      # Config with data/model params
    device=device,
    max_sensors=50,     # Limit for faster eval (optional)
)

print(f"Temperature R²: {metrics['r2_T']:.4f}")
print(f"Voltage R²: {metrics['r2_bc_V']:.4f}")
```

**Returns:**
- `mae_T`, `mse_T`, `r2_T`, `max_error_T`: Temperature metrics
- `mae_bc_V`, `mse_bc_V`, `r2_bc_V`, `max_error_bc_V`: Voltage metrics
- `per_op`: Per-OP breakdown dictionary
- `error_curves`: Per-sensor error trajectories for visualization

### `history_length_benchmark()`

Sweep different history lengths (k) to find the accuracy-cost sweet spot.

```python
from battery_surrogate.model.evaluate_sequence import history_length_benchmark

df = history_length_benchmark(
    config,
    k_values=[1, 2, 4, 8, 16, 32],  # History lengths to test
    epochs_per_k=1,                  # Training epochs per config
    max_sensors=10,                  # Faster evaluation
    device=device,
)

# Find recommended k
recommended = df[df["recommended"]]
print(f"Recommended k: {recommended['k'].values[0]}")
```

**Returns DataFrame with columns:**
| Column | Description |
|--------|-------------|
| `k` | History length |
| `mae_T`, `mse_T`, `r2_T` | Temperature metrics |
| `mae_bc_V`, `mse_bc_V`, `r2_bc_V` | Voltage metrics |
| `lookback_seconds_median` | Actual lookback window in seconds |
| `lookback_seconds_min`, `lookback_seconds_max` | Range bounds |
| `param_count` | Model parameter count (11 + 2*k input width) |
| `train_time_s` | Training time in seconds |
| `recommended` | True for accuracy knee (smallest k within 5% of best) |

### `compute_lookback_seconds()`

Translate history length k to actual lookback window in seconds (important for non-uniform Δt).

```python
from battery_surrogate.model.evaluate_sequence import compute_lookback_seconds

lookback = compute_lookback_seconds(
    op_ids=["OP01", "OP02"],
    history_length=8,
    subsample_time=50,
)
print(f"Lookback: {lookback['median']:.2f}s (min: {lookback['min']:.2f}s, max: {lookback['max']:.2f}s)")
```

### `plot_error_curves()`

Visualize error-vs-time trajectories to diagnose exposure bias.

```python
from battery_surrogate.model.evaluate_sequence import plot_error_curves

fig = plot_error_curves(
    metrics["error_curves"],  # From evaluate_sequence_model
    op_id="OP16",
    sensor_ids=[0, 1, 2, 3, 4],  # Sensors to plot
)
plt.show()
```

## Integration with Training Pipeline

The evaluation functions integrate with the unified training pipeline:

```python
from battery_surrogate.cli.train import train_from_config
from battery_surrogate.model.evaluate_sequence import evaluate_sequence_model

# Train recurrent model
config["model"]["type"] = "recurrent"
summary = train_from_config(config)

# Load and evaluate
model = build_model(config, n_sensors=363, seed=42)
model.load_state_dict(torch.load(summary["best_ckpt"]))

metrics = evaluate_sequence_model(
    model,
    config["data"]["test_ops"],
    PointwiseNormalizer.load(summary["normalizer"]),
    config,
)
```

## Warm-Up Period

Recurrent evaluation **masks the first k steps** (history_length) when computing metrics, since these steps use padded/replicated history rather than true predictions.

```python
# In evaluate_sequence_model:
k = config["model"]["history_length"]
y_true_slice = targets[k:]  # Skip warm-up
y_pred_slice = preds[k:]
```

## Error Curves for Exposure Bias Analysis

The `error_curves` output helps diagnose where recurrent models struggle:

- **Early divergence**: Model fails to capture initial dynamics
- **Late divergence**: Error accumulates over long rollouts (exposure bias)
- **Steady error**: Model has learned robust autoregressive behavior

```python
# Analyze error growth
for sensor_id, curve in metrics["error_curves"]["OP16"].items():
    times = [c[0] for c in curve]
    errors_T = [c[1] for c in curve]
    
    # Check if error grows with time
    early_error = np.mean(errors_T[:10])
    late_error = np.mean(errors_T[-10:])
    print(f"Sensor {sensor_id}: early={early_error:.4f}, late={late_error:.4f}")
```

## See Also

- [model_launcher.ipynb](../notebooks/model_launcher.ipynb): Interactive notebook for training and evaluation
- [recurrent_pointwise.py](recurrent_pointwise.py): RecurrentPointwise model architecture
- [trainer_sequence.py](trainer_sequence.py): Sequence training loop with teacher forcing
