# Modulus-native OP01 PINN (Option A)

**End-to-end NVIDIA Modulus-Sym (PhysicsNeMo)** implementation of the OP01 solid heat
equation — the counterpart to the hand-written PyTorch PINN in
`battery_surrogate_agenticWorkflow_PINN/`. The current PyTorch system is kept unchanged;
this folder is a parallel experiment to see how much Modulus actually shortens the work
and how its loss/results compare.

## What is different from the current (PyTorch) approach
| Aspect | Current (PyTorch) | Here (Modulus-Sym, Option A) |
|---|---|---|
| Framework use | ~3% Modulus (only `FullyConnected`) | **end-to-end**: `Solver`+`Domain`+`Constraint`+`PDE` |
| Time | **discrete, recurrent** (autoregressive rollout, history channels) | **continuous coordinate** `t` (no recurrence) |
| Teacher forcing | N/A — no teacher forcing, rolls out on own predictions | **concept does not exist**: there is no time-stepping, `t` is just an input; see §"no teacher forcing" in prompt 009 |
| Time derivative | finite difference (total) | **autograd** `T__t` (continuous) |
| Space derivative | autograd 2nd order | autograd 2nd order (same) |
| IC | hard (`T = T_ic + t·N`) | soft IC constraint (Modulus); hard possible via output transform |
| Collocation | 363 sensor points, subsampled | same 363 points × time provided via `from_numpy` |
| Physics coeffs | per-point `Fo` tensor | per-point `Fo` tensor passed as input Keys |

## Environment status (important)
The **true end-to-end** `physicsnemo.sym` Solver script (`run_modulus_op01.py`) currently
**cannot execute here**: the env only has `nvidia-modulus 0.9.0` (no `modulus.sym`), while the
workspace `modulus-sym/` is the newer *physicsnemo-branded* rewrite that needs a separate
`physicsnemo` core package (not installed, offline). See prompt `009` for details.

To still get real comparison numbers, `run_continuous_op01.py` reproduces the **same modelling
paradigm** (continuous field, autograd ∂t/∂², soft IC, anisotropic-heat residual) using the
installed `modulus.models` + a plain PyTorch loop.

## Run
```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate

# runnable comparison (works today):
python3 pinnANDmodulus/run_continuous_op01.py --steps 2000

# true end-to-end Modulus-Sym Solver (needs a physicsnemo core install):
python3 pinnANDmodulus/run_modulus_op01.py
```

## Result (800-step quick run vs current recurrent model, 30 ep)
| Metric | Current (recurrent) | Continuous (Option A paradigm) |
|---|---|---|
| MAE train | 0.918 °C | 0.902 °C |
| MAE test | 1.342 °C | 1.579 °C |
| L_phys (norm) | ≈1.0 | ≈0.6–0.8 |

Outputs go to `pinnANDmodulus/outputs/` (Modulus/Hydra run dir, when the Solver runs) and
`pinnANDmodulus/artifacts/` (`continuous_metrics.txt`, predictions `.npz`, single-grid-point plot).
