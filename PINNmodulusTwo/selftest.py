#!/usr/bin/env python3
"""Seconds-long checks on the loss balancing and residual scaling.

Everything here is pure arithmetic on the scaling machinery -- no data, no
training, no GPU. It exists because the properties it checks are invisible in a
training log: a poisoned EMA, a divisor whose horizon quietly depends on how
many OPs are being trained, or a "normalisation" that does not normalise all
look like an ordinary run that merely converged badly. On this project the next
opportunity to notice is a multi-hour benchmark.

Run before any long sweep:
    python3 PINNmodulusTwo/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Nothing below touches a network, but ``physics`` imports ``model``, which
# imports Modulus at module scope -- so this file could only ever run on the one
# machine that has Modulus installed, despite testing pure arithmetic. The
# faithful stand-in makes the import succeed; the checks themselves never build
# a layer, so which FCLayer is in scope cannot affect a single result here.
try:
    import modulus  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    import _modulus_stub
    _modulus_stub.install(faithful=True)

from physics import _term_norm
from train import _LossBalancer

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def balancer(mode: str, *, decay: float = 0.9, warmup: int = 1) -> _LossBalancer:
    return _LossBalancer(mode=mode, decay=decay, warmup_steps=warmup,
                         phys_norm=0.0, bc_norm=0.0, data_floor=1e-8)


def main() -> int:
    print("Loss-balancing selftest\n")

    # A spike must show up at full size. Folding the current sample into the
    # average before dividing by it lets a 10x jump report as ~5x -- damping
    # exactly the signal the balancing is supposed to expose.
    b = balancer("ema")
    b.divisor("phys", 100.0)
    b.end_step()
    check("a 10x spike is reported as 10x, not damped",
          abs(1000.0 / b.divisor("phys", 1000.0) - 10.0) < 1e-9)

    # legacy leaves L_data raw; ema does not. This is the whole difference
    # between "w_phys is a ratio" and "w_phys depends on how far the fit got".
    check("legacy leaves L_data unnormalised",
          balancer("legacy").divisor("data", 4e-3) == 1.0)
    check("ema normalises L_data too",
          balancer("ema").divisor("data", 4e-3) == 4e-3)

    # decay * nan == nan, so one bad step would pin the divisor at nan forever.
    b = balancer("ema")
    b.divisor("phys", 4.0)
    b.end_step()
    b.divisor("phys", float("nan"))
    b.end_step()
    check("a non-finite sample never enters the average",
          b.divisor("phys", 4.0) == 4.0)

    b = balancer("fixed", warmup=2)
    for value in (10.0, 10.0, 1e6, 1e-6):
        frozen = b.divisor("phys", value)
        b.end_step()
    check("fixed mode freezes after warm-up", frozen == 10.0)

    b = _LossBalancer(mode="ema", decay=0.9, warmup_steps=1, phys_norm=0.0,
                      bc_norm=0.0, data_floor=1e-3)
    check("the data divisor respects its floor", b.divisor("data", 1e-9) == 1e-3)

    # The EMA is stepped once per OP, so a per-step decay would give a horizon
    # of 1/(1-d) STEPS -- ~2 epochs at five OPs but ~10 at one, silently making
    # runs with different --ops incomparable. train.py corrects for that.
    estimates = []
    for n_ops in (1, 3, 5):
        b = balancer("ema", decay=0.9 ** (1.0 / n_ops))
        b._ema["phys"] = 1.0
        for _ in range(n_ops):            # exactly one EPOCH at 10x
            b.divisor("phys", 10.0)
            b.end_step()
        estimates.append(b._ema["phys"])
    check("EMA horizon is independent of the number of OPs",
          max(estimates) - min(estimates) < 1e-9,
          f"after one epoch at 10x: {[round(v, 4) for v in estimates]}")

    # The *_scale constants are RMS values, so x/scale is what gives unit RMS.
    # The original x/sqrt(scale) leaves mean(x**2) == scale instead, i.e. the
    # three residual terms keep the size gap the scales exist to remove.
    x = np.random.default_rng(0).normal(0.0, 7.5, 200_000)
    scale = float(np.sqrt((x ** 2).mean()))
    rms_ms = float(((x / _term_norm(scale, "rms")) ** 2).mean())
    legacy_ms = float(((x / _term_norm(scale, "legacy")) ** 2).mean())
    check("residual_norm=rms puts a term at unit RMS", abs(rms_ms - 1.0) < 1e-3,
          f"mean(res^2)={rms_ms:.4f}")
    check("residual_norm=legacy leaves it at the scale itself",
          abs(legacy_ms - scale) < 1e-2,
          f"mean(res^2)={legacy_ms:.3f}, scale={scale:.3f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
