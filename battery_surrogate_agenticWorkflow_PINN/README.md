# OP01 Battery PINN — Model Description

First physics-informed surrogate for operating point **OP01**, built with NVIDIA Modulus /
PhysicsNeMo (`modulus_env`, CPU-first). Full build plan: [.github/promptsPINN/002_builder-op01-pinn-plan.md](../.github/promptsPINN/002_builder-op01-pinn-plan.md).

> Status: **planning / Feinschliff**. This README describes the intended model; code is not built yet.

## Goal
Predict, for OP01:
- `T(t, x, y, z)` — temperature field on the 3-layer grid (physics-informed).
- `bc_V(t)` — terminal voltage (data-only for now; electrical residual for `Diff` deferred).

Both outputs are **recurrent** (history length `k`, default `k=2`, `dt=1 s`): teacher forcing in
training, free-running rollout at inference.

## Geometry
Half-model of a prismatic cell, three planar 11×11 grids (121 points each = 363) at fixed x:
| Layer | x [m] | role |
|-------|-------|------|
| Cell Center | 0.0 | jelly-roll temperature |
| JR1 Center | 0.0108 | jelly-roll temperature |
| Gehäusewand (housing) | 0.0219 | housing wall temperature |

Face spans `dy=0.198 m`, `dz=0.104 m`. Modulus `Box([0,ymin,zmin],[0.0219,ymax,zmax])` is the solid
domain for interior PDE collocation; time axis via a `Parameterization`.

Coolant **inlet** faces at `y=-0.1265`, **outlet** faces at `y=0.14605` (flow along +y), cooling
plates on the ±x faces (x=±0.0238). 20 inlet + 20 outlet face centers stored in
[data/inlet_outlet_faces.csv](data/inlet_outlet_faces.csv).

## Physics
Transient anisotropic heat conduction in the solid:
$$\rho\,C_p\,\partial_t T = \nabla\cdot(\lambda\nabla T) + \dot q,\qquad \lambda = \lambda^\top \in \mathbb{R}^{3\times3}$$

Encoded as a Modulus symbolic PDE (`AnisotropicHeatTransient(PDE)`, mirroring `eq/pdes/diffusion.py`),
so the residual becomes a loss term.

### Material properties (Material_Properties_Gridpoints.pdf, MAHLE 18.06.2026)
| Layer | ρ [kg/m³] | C_p [J/kgK] | λ [W/mK] |
|-------|-----------|-------------|----------|
| Housing | 2700 | 893 | 193 (isotropic → 193·I) |
| JR1 | 2468.13 | 938.05 | XX/XY/YY from CSV; XZ=YZ=0; **ZZ=22.4** (transverse isotropic) |
| Cell Center | per-point CSV | per-point CSV | XX/YY/ZZ from CSV; XY=XZ=YZ=0 |

`λ` is symmetric; unspecified off-diagonals are 0.

### Heat source
`q̇(t) = heatSourceJr1(t) / V_JR1` for all jelly-roll points (JR1 + CC), `q̇ = 0` in the housing.
- `V_JR1 = 4.394793e-04 m³` (single jelly-roll volume; used for CC too).
- Source from `OP1_Heat Source.csv` (`dt=0.1 s`, resampled to the T grid at `dt=1 s`); JR1 ≈ JR2.

### Initial & boundary conditions
- **IC:** `T(0,x) = ` solid initial temperature (OP01 config); `bc_V(0)` from config.
- **BC:** inlet points `T = ` fluid inlet temp; outlet points `T ≈ Tm_avg_fluid_out`.

The fluid relation `ṁ·c·(T_out − T_in) = Q` is **ignored** in this first model.

## Losses (Modulus constraints)
$$L = w_\text{data}L_\text{data} + w_\text{phys}L_\text{phys} + w_\text{IC}L_\text{IC} + w_\text{BC}L_\text{BC}$$
- `L_phys` → `PointwiseInteriorConstraint` (PDE residual on collocation points).
- `L_BC` → `PointwiseBoundaryConstraint` (inlet/outlet faces).
- `L_data`, `L_IC` → `PointwiseConstraint.from_numpy` (363 labeled sensor points; `t=0` slice).

**IC/BC enforcement:** implement both **soft** (loss terms above) and **hard**
(`T = g(t,x) + h(t,x)·N`, with `g` matching IC/BC and `h=0` there) and compare.

## Environment
- WSL venv **`modulus_env`** (`source modulus_env/bin/activate`), CPU-first.
- Reference pattern: [first tries/testerOFMod.py](../first%20tries/testerOFMod.py).
