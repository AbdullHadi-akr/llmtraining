# Legacy — superseded approaches

> **Outdated.** Nothing in this folder is part of the active pipeline. It is kept
> so earlier results stay reproducible and so the reasoning behind the current
> design stays traceable. It is not maintained, and it is not a place to add new
> work.
>
> The active model is [`../PINNmodulusTwo/`](../PINNmodulusTwo/).

## What is here, and why it was superseded

| Folder | What it was | Why it was left behind |
| --- | --- | --- |
| `battery_surrogate_agenticWorkflow/` | Data-driven MLP pipeline (pointwise and sequence models), plus the original data ingest and the `data_cache/` bundles | Purely data-driven: no physics constraint, so nothing keeps the prediction consistent with the heat equation |
| `battery_surrogate_agenticWorkflow_PINN/` | First attempt at adding a physics loss to that pipeline | Superseded by `PINNmodulusTwo/`, which reorganised the recurrence and the residual scaling |
| `pinnANDmodulus/` | Early Modulus experiments on OP01 (continuous, no recurrence) | No recurrence, so time-varying configurations that coincide at one instant become indistinguishable despite different histories |
| `first tries/` | Exploratory notebooks | Exploratory only |
| `stuffTolookat/` | Reference snippets (Navier-Stokes, naive MLP, Modulus smoke tests) | Reference only |
| `*.ipynb` | Grid/coordinate and input/output helper notebooks | Their job now sits in `PINNmodulusTwo/data.py` and `generate_cache.py` |

## The data cache

`battery_surrogate_agenticWorkflow/data_cache/` may still hold the `OP*.npz`
bundles on an existing machine. It is not tracked in git, so the restructure did
not move it, and `PINNmodulusTwo/data.py` still finds it there.

The preferred location is now the top-level `data_cache/`. Moving it is a
one-time local step — nothing in git changes:

```bash
mkdir -p data_cache
mv legacy/battery_surrogate_agenticWorkflow/data_cache/*.npz data_cache/
```
