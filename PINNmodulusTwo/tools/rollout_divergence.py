"""Does the free-running rollout of an UNTRAINED RecurrentField diverge?

Answers the question ``L_data=nan`` in epoch 1 raises, without needing Modulus,
the data cache or a GPU. Reproduces smallBench.py's production geometry:
subsample 2 -> dt = 0.2 s, T_span_ref ~ 1474 s -> dtn = 1.357e-4, n_t = 7000,
rate_lags = [5, 20] s, delta_grid = 0.2 s, rate_scale = dTdt_scale = 2.479,
width 64, depth 3, P = 363.

    python3 PINNmodulusTwo/tools/rollout_divergence.py

Sweeps history_mode x residual_output over several seeds and reports where each
combination first goes non-finite. What it showed (5 seeds, float32):

    history  residual   diverged at step
    hybrid   yes        36-720    (5/5)
    hybrid   no         36-1770   (5/5)
    raw      yes        1394-5068 (3/5)
    raw      no         --        (0/5)

So there are TWO drivers. The hybrid rate channel is the strong one -- it takes
the rollout down in both output parameterisations -- and residual_output is a
second, independent one: T(t) = net(...) + level(t) is an integrator, so a
systematic bias in the untrained net accumulates step by step, which is enough
on its own (raw, 3/5).

``--stub conftest`` restores the OLD test substitute for Modulus (nn.Linear's
default init) to show what that understated: it reports hybrid/residual=no as
diverging on only 3/5 seeds, and at width 128 / depth 4 the original measurement
against it called that combination stable outright. The default ``faithful``
copies the real FCLayer's xavier_uniform + zero bias. See ARCHITECTURE.md 3.1.

This is a diagnostic, not a test -- tests/test_rollout_stability.py is the part
that runs in CI.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # model.py
sys.path.insert(0, str(HERE))                 # _modulus_stub.py

p = argparse.ArgumentParser()
p.add_argument("--stub", choices=["faithful", "conftest"], default="faithful")
p.add_argument("--steps", type=int, default=7000)
p.add_argument("--width", type=int, default=64)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--dtype", default="float32")
args = p.parse_args()

import _modulus_stub
_modulus_stub.install(faithful=(args.stub == "faithful"))

import torch
from model import RecurrentField, rollout

DT_S, T_SPAN = 0.2, 1474.0
DTN = DT_S / T_SPAN
RATE_LAGS_N = [5.0 / T_SPAN, 20.0 / T_SPAN]
DELTA_GRID_N = DT_S / T_SPAN
RATE_SCALE = 2.479
P, N_CFG, N_STAT, N_FRC = 363, 6, 3, 1
DTYPE = getattr(torch, args.dtype)


def run(history_mode: str, residual_output: bool, seed: int):
    torch.manual_seed(seed)
    m = RecurrentField(
        n_config=N_CFG, n_static=N_STAT, n_forcing=N_FRC, k_max=2,
        history_mode=history_mode, rate_lags=RATE_LAGS_N,
        layer_size=args.width, num_layers=args.depth,
        delta_seconds=1.0, dtn=DTN, t_span_ref=T_SPAN,
        rate_scale=RATE_SCALE, delta_grid=DELTA_GRID_N,
        residual_output=residual_output,
    ).to(DTYPE)
    n_t = args.steps
    g = torch.Generator().manual_seed(seed + 1000)
    xn = torch.rand(P, 3, generator=g, dtype=DTYPE)
    static = torch.randn(P, N_STAT, generator=g, dtype=DTYPE)
    # z-scored config, forcing a smooth positive ramp: realistic magnitudes
    cfg = torch.randn(n_t, N_CFG, generator=g, dtype=DTYPE) * 0.1 + torch.linspace(-1, 1, n_t, dtype=DTYPE)[:, None]
    frc = torch.linspace(0.0, 1.0, n_t, dtype=DTYPE)[:, None]
    Tn_ic = torch.randn(P, generator=g, dtype=DTYPE) * 0.05 - 1.0
    tn = torch.arange(n_t, dtype=DTYPE) * DTN
    buf = rollout(m, xn, static, cfg, frc, Tn_ic, tn, DTN)
    amax = buf.abs().amax(dim=1)
    bad = ~torch.isfinite(amax)
    first = int(bad.nonzero()[0]) if bad.any() else None
    return first, amax


print(f"stub={args.stub}  dtype={args.dtype}  width={args.width} depth={args.depth} "
      f"steps={args.steps}  P={P}")
print(f"dtn={DTN:.4g}  rate divisors (normalised): "
      + ", ".join(f"{l*RATE_SCALE:.4g} (gain {1/(l*RATE_SCALE):.1f}x)" for l in RATE_LAGS_N))
print()
hdr = f"{'history':8s} {'residual':9s} {'diverged at step':>18s}   {'|T| max at end':>16s}"
print(hdr); print("-" * len(hdr))
for hm in ("hybrid", "raw"):
    for ro in (True, False):
        firsts, ends = [], []
        for s in range(args.seeds):
            f, amax = run(hm, ro, s)
            firsts.append(f)
            ends.append(float(amax[-1]) if f is None else float("inf"))
        n_div = sum(f is not None for f in firsts)
        fin = [f for f in firsts if f is not None]
        if fin:
            where = f"{min(fin)}-{max(fin)} ({n_div}/{args.seeds} seeds)"
        else:
            where = f"-- (0/{args.seeds})"
        end = "inf/nan" if n_div == args.seeds else f"{max(e for e in ends if e != float('inf')) if any(e != float('inf') for e in ends) else float('inf'):.4g}"
        print(f"{hm:8s} {str(int(ro)):9s} {where:>18s}   {end:>16s}")
