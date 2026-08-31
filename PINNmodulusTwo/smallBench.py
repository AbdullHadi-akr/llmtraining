#!/usr/bin/env python3
"""Small convergence benchmark: quick sanity check before full sweeps.

Runs a short training (few epochs, 2 w_phys values) to verify:
1. CFL stability (no inf/NaN losses)
2. Loss convergence (L_data decreasing)
3. Balanced losses are ~O(1) -- reported per term, L_phys_bal AND L_bc_bal
4. Test MAE is reasonable (< 20°C)

All four are smoke tests: they ask whether the RUN is usable, not whether the
MODEL is good. Check 4 in particular is a 20 °C bound and passing it means very
little. The line that answers "is this worth more than doing nothing" is the
trivial-predictor comparison printed underneath it, computed on the same
held-out OP -- see ``_trivial_baselines``. A run can pass all four checks and
still lose to predicting a constant.

No ``data_cache/``? ``python3 PINNmodulusTwo/tools/make_synthetic_cache.py``
writes a synthetic one, and every command below then runs on a bare checkout --
``--modulus-stub`` covers a machine without Modulus on top of that. Both are
announced in a banner and recorded in the artifacts, because a number off either
must never be quoted as a measurement.

This is also "step A" of PINNmodulusTwo/README_MODEL_CRITIQUE.md: run it once as
it stands and once as

    --inner-steps 1 --no-residual-output --learn-gains

which is the configuration from before the training-budget, residual-output and
physics-residual fixes. Comparing the two Test MAEs is what decides whether those
fixes actually helped -- so far they are only verified mathematically -- and the
critique file turns each possible outcome into the next step to take.

Run:
    cd /mnt/c/Users/M0245635/batterysurrogatemodell
    source modulus_env/bin/activate
    python3 PINNmodulusTwo/smallBench.py

Expected runtime: ~2-5 minutes on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# --modulus-stub has to be handled before ``model`` is imported, because that
# import is what raises when Modulus is missing -- same ordering as
# tools/rollout_divergence.py. It is opt-in, never a silent fallback: the stub
# replaces the network's building block, and a benchmark that swapped that out
# without saying so would produce numbers nobody could attribute.
USE_MODULUS_STUB = "--modulus-stub" in sys.argv
if USE_MODULUS_STUB:
    sys.path.insert(0, str(THIS_DIR / "tools"))
    import _modulus_stub
    _modulus_stub.install(faithful=True)

from data import build_op
from device_utils import resolve_device
from model import rollout
from train import fit

ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml_defaults() -> dict:
    cfg_path = THIS_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    d = _load_yaml_defaults()
    p = argparse.ArgumentParser(description="Small convergence benchmark")
    p.add_argument("--ops", nargs="+", default=d.get("ops", ["OP01", "OP02"]))
    p.add_argument("--test-op", default=d.get("test_op", "OP07"))
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 2))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--width", type=int, default=d.get("layer_size", 64))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 3))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--history-mode", default=d.get("history_mode", "hybrid"))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 20.0]),
                   help="hybrid rate segments in SECONDS. What matters is "
                        "A = 1/(lag_n * rate_scale), printed at startup: short "
                        "segments divide a small difference by a small number "
                        "and amplify everything non-smooth by A. 5 s gives "
                        "A ~ 119 and the rollout diverges")
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2),
                   help="anchor lag of the hybrid history in seconds")
    p.add_argument("--time-deriv", default=d.get("time_deriv", "bdf2"))
    p.add_argument("--lr", type=float, default=d.get("lr", 0.002))
    p.add_argument("--w-phys", type=float, nargs="+", default=[0.0, 0.1])
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 1024))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 128))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 64))
    p.add_argument("--inner-steps", type=int, default=d.get("inner_steps", 100),
                   help="optimiser steps per OP per epoch against that epoch's "
                        "frozen rollout")
    p.add_argument("--rollout-clamp", type=float,
                   default=d.get("rollout_clamp", 50.0),
                   help="saturate the rollout buffer at +/-this many normalised "
                        "temperature units; 0 disables")
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0),
                   help="cap the hybrid history amplification A by raising "
                        "rate_scale; 0 = leave it at dTdt_scale")
    p.add_argument("--residual-output", action=argparse.BooleanOptionalAction,
                   default=d.get("residual_output", True))
    p.add_argument("--learn-gains", action=argparse.BooleanOptionalAction,
                   default=d.get("learn_gains", False))
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", True))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", True))
    p.add_argument("--loss-balance", choices=["ema", "legacy", "fixed"],
                   default=d.get("loss_balance", "ema"))
    p.add_argument("--ema-decay", type=float, default=d.get("ema_decay", 0.9))
    p.add_argument("--balance-warmup", type=int, default=d.get("balance_warmup", 1))
    p.add_argument("--data-floor", type=float, default=d.get("data_floor", 1e-8))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--bc-norm", type=float, default=d.get("bc_norm", 0.0))
    p.add_argument("--residual-norm", choices=["rms", "legacy"],
                   default=d.get("residual_norm", "rms"))
    p.add_argument("--zero-weight-terms", choices=["skip", "compute"],
                   default=d.get("zero_weight_terms", "skip"))
    p.add_argument("--subsample-mode", choices=["stride", "mean"],
                   default=d.get("subsample_mode", "stride"))
    p.add_argument("--forcing-energy", action="store_true",
                   default=d.get("forcing_energy", False))
    p.add_argument("--config-rates", action="store_true",
                   default=d.get("config_rates", False))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N (auto = cuda when available)")
    p.add_argument("--modulus-stub", action="store_true",
                   help="run without a Modulus install, substituting the "
                        "faithful-at-init FCLayer stand-in from "
                        "tools/_modulus_stub.py. For getting a laptop to run "
                        "the pipeline at all -- never for a quoted result")
    p.add_argument("--quick", action="store_true",
                   help="cut the run down for a laptop CPU: 3 epochs, 25 inner "
                        "steps, one w_phys. Enough to see every check fire and "
                        "the artifacts written; far too short to judge accuracy")
    cli = p.parse_args()
    if cli.quick:
        # Only override what the user did not ask for explicitly, so
        # --quick --epochs 5 still runs 5 epochs. argparse also accepts
        # --epochs=5 as a single token, so a plain set membership test would
        # miss it and silently overrule the value the user typed.
        def given(flag: str) -> bool:
            return any(a == flag or a.startswith(flag + "=")
                       for a in sys.argv[1:])

        if not given("--epochs"):
            cli.epochs = 3
        if not given("--inner-steps"):
            cli.inner_steps = 25
        if not given("--w-phys"):
            cli.w_phys = [0.1]
    return cli


class Args:
    """Namespace-like class for fit() compatibility."""
    pass


def _make_args(cli, w_phys: float) -> Args:
    args = Args()
    args.ops = cli.ops
    args.subsample = cli.subsample
    args.epochs = cli.epochs
    args.k_max = cli.k_max
    args.history_mode = cli.history_mode
    args.rate_lags = cli.rate_lags
    # fit() liest beide nur via getattr. Ohne diese Zeilen faellt delta_grid
    # still auf den Datenschritt zurueck statt auf den konfigurierten Wert --
    # bei subsample=2 zufaellig identisch, bei jedem anderen Wert nicht.
    args.delta_grid = cli.delta_grid
    args.gain_lr_mult = cli.gain_lr_mult
    args.time_deriv = cli.time_deriv
    args.width = cli.width
    args.depth = cli.depth
    args.lr = cli.lr
    args.w_data = 1.0
    args.w_phys = float(w_phys)
    args.w_bc = cli.w_bc
    args.batch_data = cli.batch_data
    args.batch_phys = cli.batch_phys
    args.batch_bc = cli.batch_bc
    args.inner_steps = cli.inner_steps
    args.rollout_clamp = cli.rollout_clamp
    args.max_rate_amp = cli.max_rate_amp
    args.residual_output = cli.residual_output
    args.learn_gains = cli.learn_gains
    args.weight_decay = 0.0
    args.grad_clip = cli.grad_clip
    args.early_stopping_patience = 0
    # Same trap as delta_grid above: fit() reads all of these via getattr, so a
    # field missing here does not raise -- the smoke test would quietly run a
    # different balancing than config.yaml asks for, which is exactly what this
    # file exists to rule out.
    args.phys_norm = cli.phys_norm
    args.bc_norm = cli.bc_norm
    args.loss_balance = cli.loss_balance
    args.ema_decay = cli.ema_decay
    args.balance_warmup = cli.balance_warmup
    args.data_floor = cli.data_floor
    args.residual_norm = cli.residual_norm
    args.zero_weight_terms = cli.zero_weight_terms
    args.subsample_mode = cli.subsample_mode
    args.forcing_energy = cli.forcing_energy
    args.config_rates = cli.config_rates
    args.use_static = cli.use_static
    args.use_forcing = cli.use_forcing
    args.seed = cli.seed
    args.device = cli.device
    args.test_op = cli.test_op
    return args


@torch.no_grad()
def _rollout_mae(model, op, bundle, device) -> float:
    """Free-running rollout MAE in physical °C."""
    xn = torch.as_tensor(op.xn, dtype=torch.float32, device=device)
    static = torch.as_tensor(op.static_feat, dtype=torch.float32, device=device)
    forcing = torch.as_tensor(op.forcing_feat, dtype=torch.float32, device=device)
    cfg = torch.as_tensor(op.config_feat, dtype=torch.float32, device=device)
    tn = torch.as_tensor(op.tn, dtype=torch.float32, device=device)
    Tn_ic = torch.as_tensor(op.Tn_ic, dtype=torch.float32, device=device)
    static = static[:, :model.n_static]
    forcing = forcing[:, :model.n_forcing]
    buf = rollout(model, xn, static, cfg, forcing, Tn_ic, tn, op.dtn)
    T_pred = buf.cpu().numpy() * bundle.T_sigma + bundle.T_mu
    return float(np.abs(T_pred - op.T_lab).mean())


def _trivial_baselines(op, bundle) -> tuple[float, float]:
    """MAE of the two trivial predictors on ``op``, in physical °C.

    Without these a Test MAE is a number with no scale. The pair is the one
    README_ERSTER_TEST.md chapter 6 uses, and the model has to beat the BETTER
    of the two to have learned anything:

    * persistence -- ``T(t) = T(0)``, i.e. the field never changes;
    * mean -- the constant mean of the training labels. That constant is
      ``bundle.T_mu`` by construction (``data.py`` pools it over the training
      portion of the training OPs), so this is not a second definition of the
      same thing, it IS the same thing.

    The numbers in README_ERSTER_TEST.md (11.96 °C and 6.60 °C) come from a
    SYNTHETIC bundle and its chapter 9.1 says the magnitudes do not carry over
    to the real OPs -- so they must not be used as the yardstick here. These are
    computed on whatever ``op`` actually is, which is the point.
    """
    persistence = float(np.abs(op.T_lab - op.T_lab[0][None, :]).mean())
    mean = float(np.abs(op.T_lab - bundle.T_mu).mean())
    return persistence, mean


def _cache_is_synthetic() -> bool:
    """True when the loaded OP bundles were written by make_synthetic_cache.py.

    Absolute MAE off that fixture means nothing about the real OPs, so a run on
    it must never be quoted as a result. Cheap to check and easy to forget,
    hence the banner rather than a line in a README.
    """
    try:
        import data as _data
        for path in sorted(Path(_data.DATA_CACHE).glob("OP*.npz")):
            with np.load(path, allow_pickle=True) as npz:
                # Any synthetic bundle in the cache is enough to disqualify the
                # run: a directory holding both kinds is not a dataset, and the
                # banner has to fire on the mixture too, not only when the file
                # that happens to sort first is the synthetic one.
                if "synthetic" in npz.files:
                    return True
    except Exception:
        pass
    return False


def main():
    cli = parse_args()
    device = resolve_device(cli.device)
    cli.device = str(device)  # hand the resolved device down to fit()
    dt_s = 0.1 * cli.subsample

    synthetic = _cache_is_synthetic()

    print("=" * 70)
    print("SMALL CONVERGENCE BENCHMARK")
    print("=" * 70)
    if USE_MODULUS_STUB:
        print("  *** MODULUS STUB (tools/_modulus_stub.py) ***")
        print("  The real modulus.models.layers.FCLayer is NOT in use. The")
        print("  stand-in matches it at initialisation, which is what rollout")
        print("  stability depends on, but this is not a Modulus run.")
        print("-" * 70)
    if synthetic:
        print("  *** SYNTHETIC CACHE (tools/make_synthetic_cache.py) ***")
        print("  Smoke fixture. It answers 'does the pipeline run and do the")
        print("  checks fire', not 'how accurate is the model'. Absolute MAE")
        print("  from it says nothing about the real OPs.")
        print("-" * 70)
    print(f"  Training OPs:  {cli.ops}")
    print(f"  Test OP:       {cli.test_op}")
    print(f"  Δt:            {dt_s:.1f}s (subsample={cli.subsample})")
    print(f"  Epochs:        {cli.epochs}")
    print(f"  Architecture:  width={cli.width}, depth={cli.depth}")
    print(f"  History:       mode={cli.history_mode}, rate_lags={cli.rate_lags}")
    print(f"  Physics sweep: w_phys={cli.w_phys}")
    print(f"  BC weight:     w_bc={cli.w_bc}")
    print(f"  Grad clip:     {cli.grad_clip}")
    print("=" * 70)
    print()

    results = []
    all_passed = True
    all_histories = []  # Store histories for convergence plot

    for w_phys in cli.w_phys:
        print(f"\n{'='*50}")
        print(f"Training with w_phys = {w_phys}")
        print(f"{'='*50}")

        args = _make_args(cli, w_phys)
        model, bundle, ops_packed, dtn, hist = fit(args)
        model.eval()

        # A term whose weight is 0 is not computed at all (train.py
        # --zero-weight-terms), and is recorded as NaN on purpose. That is an
        # absent measurement, not a failed one: checking it would fail the very
        # w_phys=0 reference point this smoke test needs in order to show what
        # the physics term is worth.
        phys_on = float(w_phys) != 0.0
        bc_on = float(cli.w_bc) != 0.0

        # Check 1: No inf/NaN in final losses
        L_data_final = hist["L_data"][-1] if hist["L_data"] else float("nan")
        L_phys_final = hist["L_phys"][-1] if hist["L_phys"] else float("nan")
        stable = np.isfinite(L_data_final) and (not phys_on
                                                or np.isfinite(L_phys_final))

        # Check 2: Loss decreased (convergence)
        if len(hist["L_data"]) >= 2:
            L_data_first = hist["L_data"][0]
            converged = L_data_final < L_data_first
        else:
            converged = False

        # Check 3: Balanced losses are ~O(1) -- only where a term is switched on
        L_phys_bal = hist.get("L_phys_bal", [1.0])[-1]
        L_bc_bal = hist.get("L_bc_bal", [1.0])[-1]
        # Kept apart on purpose: a single `balanced` flag cannot say WHICH term
        # broke it, and a run reading only L_phys_bal then blames the physics
        # weight for a bc failure -- or, at w_phys=0 where the physics term is
        # skipped entirely, has nothing left to blame and reaches for the MAE.
        phys_balanced = (not phys_on) or 0.01 < L_phys_bal < 100
        bc_balanced = (not bc_on) or 0.01 < L_bc_bal < 100
        balanced = phys_balanced and bc_balanced

        # The divisors those balanced losses came out of, and how much structure
        # the rollout still carries (train.py records both per epoch).
        #
        # They answer the two questions a bare L_phys_bal cannot. A value far
        # below 1 means either the term genuinely fell or the EMA divisor is
        # stale from an earlier regime, and those call for opposite responses --
        # the divisor separates them. And a field constant in space and time
        # satisfies the heat residual AND the Neumann BC exactly, so a physics
        # loss going to zero is only good news while the spread stays up.
        #
        # Reported, never gated: a short smoke run is entitled to a flat-ish
        # rollout, and turning that into a FAIL would cry wolf on every one.
        spread_space = hist.get("spread_space", [float("nan")])[-1]
        spread_time = hist.get("spread_time", [float("nan")])[-1]
        div_phys = hist.get("div_phys", [float("nan")])[-1]
        div_bc = hist.get("div_bc", [float("nan")])[-1]

        # Check 4: Test MAE is reasonable
        held = build_op(cli.test_op, bundle, subsample_time=cli.subsample)
        test_mae = _rollout_mae(model, held, bundle, device)
        mae_ok = test_mae < 20.0

        # The two trivial predictors on the SAME held-out OP. `mae_ok` only asks
        # whether the rollout stayed finite-ish (< 20 °C); it is a smoke-test
        # bound, not a quality bar, and passing it says nothing about whether
        # the model beats doing nothing. `beats_trivial` is that question.
        mae_persist, mae_mean = _trivial_baselines(held, bundle)
        mae_trivial = min(mae_persist, mae_mean)
        beats_trivial = test_mae < mae_trivial

        # In-time MAE for training OPs
        train_maes = [_rollout_mae(model, op, bundle, device) 
                      for op in [build_op(oid, bundle, cli.subsample) for oid in cli.ops]]
        train_mae = float(np.mean(train_maes))

        passed = stable and converged and balanced and mae_ok

        results.append({
            "w_phys": w_phys,
            "L_data_final": L_data_final,
            "L_phys_bal": L_phys_bal,
            "L_bc_bal": L_bc_bal,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "mae_persist": mae_persist,
            "mae_mean": mae_mean,
            "mae_trivial": mae_trivial,
            "beats_trivial": beats_trivial,
            "spread_space": spread_space,
            "spread_time": spread_time,
            "div_phys": div_phys,
            "div_bc": div_bc,
            "stable": stable,
            "converged": converged,
            "phys_balanced": phys_balanced,
            "bc_balanced": bc_balanced,
            "balanced": balanced,
            "mae_ok": mae_ok,
            "passed": passed,
        })
        all_histories.append({"w_phys": w_phys, "hist": hist})

        if not passed:
            all_passed = False

        print(f"\n  Results for w_phys={w_phys}:")
        print(f"    L_data(final):  {L_data_final:.4e}  {'✓' if stable else '✗ (inf/NaN)'}")
        phys_note = ("(w_phys=0: term not computed)" if not phys_on
                     else "✓" if phys_balanced else "✗ (not ~O(1))")
        bc_note = ("(w_bc=0: term not computed)" if not bc_on
                   else "✓" if bc_balanced else "✗ (not ~O(1))")
        print(f"    L_phys_bal:     {L_phys_bal:.4e}  {phys_note}")
        print(f"    L_bc_bal:       {L_bc_bal:.4e}  {bc_note}")
        print(f"      divisors:     phys={div_phys:.4e}  bc={div_bc:.4e}")
        flat_note = ("  <- near-constant field: both residual terms are "
                     "satisfied for free"
                     if min(spread_space, spread_time) < 0.2 else "")
        print(f"    Rollout spread: space={spread_space:.3g}x  "
              f"time={spread_time:.3g}x  of the labels'{flat_note}")
        print(f"    Train MAE:      {train_mae:.2f}°C")
        print(f"    Test MAE:       {test_mae:.2f}°C  {'✓' if mae_ok else '✗ (>20°C)'}")
        print(f"      vs. persistence T(t)=T(0):     {mae_persist:.2f}°C")
        print(f"      vs. constant mean of train:    {mae_mean:.2f}°C")
        beat_note = ("✓" if beats_trivial
                     else f"✗ -- {test_mae:.2f}°C >= {mae_trivial:.2f}°C")
        print(f"      beats the better of the two:   {beat_note}")
        print(f"    Converged:      {'✓' if converged else '✗'}")
        print(f"    Overall:        {'✓ PASS' if passed else '✗ FAIL'}")
        if not passed:
            # Which check actually failed. Without this the four booleans have to
            # be reconstructed from four separate lines above, and the guess that
            # comes out is usually "MAE too high" -- the one bound that is hardest
            # to fail, because it is 20 °C.
            why = []
            if not stable:
                why.append("L_data/L_phys not finite")
            if not converged:
                why.append("L_data did not decrease over the run")
            if not phys_balanced:
                why.append(f"L_phys_bal={L_phys_bal:.4e} outside [0.01, 100]")
            if not bc_balanced:
                why.append(f"L_bc_bal={L_bc_bal:.4e} outside [0.01, 100]")
            if not mae_ok:
                why.append(f"Test MAE {test_mae:.2f}°C >= 20°C")
            print(f"    FAIL reason:    {'; '.join(why)}")
        if passed and not beats_trivial:
            print("    NOTE:           all four checks pass, but the model does "
                  "not beat the trivial predictor -- PASS here means 'the run is "
                  "usable', not 'the model is good'.")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    # L_bc_bal belongs in this table: it is one of the two terms Check 3 tests,
    # so a FAIL with a healthy L_phys_bal is otherwise unreadable from here.
    print(f"{'w_phys':>8} | {'L_data':>10} | {'L_phys_bal':>10} | {'L_bc_bal':>10} | "
          f"{'Train MAE':>10} | {'Test MAE':>10} | {'Status':>8}")
    print("-" * 88)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['w_phys']:>8.3f} | {r['L_data_final']:>10.4e} | {r['L_phys_bal']:>10.4e} | "
              f"{r['L_bc_bal']:>10.4e} | {r['train_mae']:>9.2f}C | {r['test_mae']:>9.2f}C | "
              f"{status:>8}")
    print("-" * 88)

    # The yardstick, once, on the real held-out OP. Identical across rows (it
    # does not depend on the model), which is exactly why it belongs here and
    # not in a per-row column.
    if results:
        r0 = results[0]
        best = min(results, key=lambda r: r["test_mae"])
        print(f"\nTrivial predictors on {cli.test_op} (no training involved):")
        print(f"  persistence T(t)=T(0):        {r0['mae_persist']:>8.2f}C")
        print(f"  constant mean of train labels:{r0['mae_mean']:>8.2f}C")
        print(f"  -> the bar to beat:           {r0['mae_trivial']:>8.2f}C")
        if best["beats_trivial"]:
            print(f"  best run (w_phys={best['w_phys']:g}) at {best['test_mae']:.2f}C "
                  f"beats it by {r0['mae_trivial'] - best['test_mae']:.2f}C. The model "
                  f"has learned something.")
        else:
            print(f"  best run (w_phys={best['w_phys']:g}) at {best['test_mae']:.2f}C does "
                  f"NOT beat it. Whatever else passed, the model is not yet worth "
                  f"more than doing nothing.")

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Ready for full benchmark!")
        # NOT the 10x10 grid: that is 100 trainings (~6-8 days) and it would
        # sweep weights before anything has established what a weight means
        # here. The balancing benchmark is ~4 h and settles that first.
        print("  Next: python3 PINNmodulusTwo/benchmark_balance.py --part 1 "
              "--epochs 20 --device cuda")
        print("        (settles the loss balancing; the weight probe in "
              "benchmark_wphys_wbc.py comes after it)")
    else:
        print("\n✗ SOME CHECKS FAILED - Review issues above before full benchmark")

    # Plot convergence curves
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for item in all_histories:
        w = item["w_phys"]
        h = item["hist"]
        epochs = h["epoch"]
        label = f"w_phys={w:.3f}"
        
        # Top: data loss
        axes[0].plot(epochs, h["L_data"], marker="o", label=label)
        axes[0].set_ylabel("L_data (MSE)", fontsize=11)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        axes[0].set_title(f"Convergence: {len(cli.ops)} OPs, {cli.epochs} epochs", fontsize=12)
        
        # Bottom: balanced physics and BC losses (if applicable)
        if "L_phys_bal" in h and len(h["L_phys_bal"]) > 0:
            axes[1].plot(epochs, h["L_phys_bal"], marker="s", linestyle="--", 
                        label=f"{label} L_phys_bal", alpha=0.7)
        if "L_bc_bal" in h and len(h["L_bc_bal"]) > 0:
            axes[1].plot(epochs, h["L_bc_bal"], marker="^", linestyle=":", 
                        label=f"{label} L_bc_bal", alpha=0.7)
    
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Balanced Loss", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    convergence_plot = ART_DIR / "smallBench_convergence.png"
    plt.savefig(convergence_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved convergence plot to {convergence_plot}")

    # Save results
    out_file = ART_DIR / "smallBench_results.txt"
    with open(out_file, "w") as f:
        f.write("Small Benchmark Results\n")
        f.write(f"OPs: {cli.ops}, Test: {cli.test_op}\n")
        f.write(f"Δt: {dt_s}s, Epochs: {cli.epochs}\n")
        f.write(f"Architecture: width={cli.width}, depth={cli.depth}\n")
        f.write(f"w_bc: {cli.w_bc}, loss_balance: {cli.loss_balance}, "
                f"ema_decay: {cli.ema_decay}\n")
        # Provenance travels with the numbers or it is lost. A results file that
        # does not say it came off the smoke fixture, or off the Modulus stub,
        # is one copy-paste away from being quoted as a measurement.
        if synthetic:
            f.write("SYNTHETIC CACHE (tools/make_synthetic_cache.py) -- smoke "
                    "fixture; absolute MAE says nothing about the real OPs\n")
        if USE_MODULUS_STUB:
            f.write("MODULUS STUB (tools/_modulus_stub.py) -- the real FCLayer "
                    "was not in use\n")
        f.write("\n")
        for r in results:
            f.write(f"w_phys={r['w_phys']}: L_data={r['L_data_final']:.4e}, "
                    f"L_phys_bal={r['L_phys_bal']:.4e}, L_bc_bal={r['L_bc_bal']:.4e}, "
                    f"div_phys={r['div_phys']:.4e}, div_bc={r['div_bc']:.4e}, "
                    f"spread_space={r['spread_space']:.3g}, "
                    f"spread_time={r['spread_time']:.3g}, "
                    f"train_mae={r['train_mae']:.2f}C, test_mae={r['test_mae']:.2f}C, "
                    f"{'PASS' if r['passed'] else 'FAIL'}\n")
        # README_MODEL_CRITIQUE.md step A compares this file between two runs, so
        # the yardstick has to travel with it -- otherwise the comparison is
        # between two numbers whose scale is only known in the terminal.
        if results:
            r0 = results[0]
            f.write(f"\nTrivial predictors on {cli.test_op} (no training):\n")
            f.write(f"  persistence T(t)=T(0):         {r0['mae_persist']:.2f}C\n")
            f.write(f"  constant mean of train labels: {r0['mae_mean']:.2f}C\n")
            f.write(f"  bar to beat:                   {r0['mae_trivial']:.2f}C\n")
    print(f"  Saved results to {out_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
