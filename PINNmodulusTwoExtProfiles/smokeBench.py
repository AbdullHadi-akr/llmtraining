#!/usr/bin/env python3
"""smokeBench - the few-minute check that gates every long profileBench run.

The base project's ``smallBench.py`` answers "does training converge at all".
That is still question one here, but the profile extension adds failure modes
that a converging loss does not catch, and every one of them silently produces a
plausible-looking number:

1. **The profiles are not actually there.** ``op_registry`` is a transcription
   of the plan sheet; the ``.npz`` bundles are the truth. If a bundle that
   should carry a fluid-temperature profile carries a scalar instead -- wrong
   export, stale cache, a profile file that never got assembled -- every driver
   rate channel for it is zero and the model quietly trains on the constant
   case while the report says "profile".
2. **The profile does not cover the run.** ``np.interp`` holds the first and
   last profile value outside the profile's own time range, so a profile that
   stops at 600 s in a 1400 s OP looks like a profile that goes flat. That is a
   data problem, not a modelling one, and it has to surface here.
3. **A driver rate channel is dead across the whole training set.** Then the
   input width is being spent on a column of zeros -- harmless for the loss,
   misleading in any conclusion about whether driver history helps.
4. **The hybrid history amplification is large.** Pooling OP01-OP16 widens
   ``T_sigma`` and shrinks ``dTdt_scale``, which raises the factor by which the
   temperature history magnifies the opening steps of the rollout. Past a point
   the free-running rollout diverges in epoch 1 -- see
   ``data.effective_rate_scale``.
5. **Held-out OPs sit outside the trained envelope.** True by construction for
   the T3 tier, but worth printing before hours are spent, and a surprise on any
   other OP means the split is not what it is documented to be.

Checks 1-3 and 5 need no training at all, so they run first and fail fast.

Run:
    python3 PINNmodulusTwoExtProfiles/smokeBench.py
    python3 PINNmodulusTwoExtProfiles/smokeBench.py --subsample 40 --epochs 3

Expect a few minutes. Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from data import (
    PROFILE_CHANNELS, available_ops, build_op, coverage_report,
    effective_rate_scale, hybrid_rate_amplification, load_ops,
    normalisation_report, profile_report, require_ops,
)
from device_utils import resolve_device
from op_metrics import op_metrics, rollout_phys
from op_registry import (
    DEFAULT_TEST_OPS, DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS, check_split,
    profiles_of, tier_of,
)

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# A rollout MAE above this on the TRAINING OPs after a handful of epochs means
# something is wrong beyond "not trained long enough". Deliberately loose: this
# is a smoke test, not a quality bar.
MAE_SANITY_C = 25.0


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
    bo = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="Profile-extension smoke test")
    p.add_argument("--ops", nargs="+", default=d.get("ops", list(DEFAULT_TRAIN_OPS)))
    p.add_argument("--val-ops", nargs="+", default=d.get("val_ops", list(DEFAULT_VAL_OPS)))
    p.add_argument("--test-ops", nargs="+", default=d.get("test_ops", list(DEFAULT_TEST_OPS)))
    # Coarser than the training default on purpose: this has to finish in
    # minutes, and every check except the loss numbers is subsample-independent.
    p.add_argument("--subsample", type=int, default=40)
    p.add_argument("--train-frac", type=float, default=d.get("train_frac", 0.8))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--resample", choices=["mean", "point"],
                   default=d.get("resample", "mean"))
    p.add_argument("--driver-history", action=bo,
                   default=d.get("use_driver_history", True))
    p.add_argument("--driver-rate-lags", nargs="+", type=float,
                   default=d.get("driver_rate_lags", [5.0, 20.0]))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 20.0]))
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2))
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "hybrid"))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--w-phys", nargs="+", type=float, default=[0.0, 0.1],
                   help="one short training per value")
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--batch-data", type=int, default=1024)
    p.add_argument("--batch-phys", type=int, default=128)
    p.add_argument("--batch-bc", type=int, default=64)
    p.add_argument("--use-static", action=bo, default=d.get("use_static", True))
    p.add_argument("--use-forcing", action=bo, default=d.get("use_forcing", True))
    p.add_argument("--shuffle-ops", action=bo, default=d.get("shuffle_ops", True))
    p.add_argument("--holdout-tail", action=bo, default=d.get("holdout_tail", False))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=d.get("device", "auto"))
    p.add_argument("--tf32", action="store_true", default=d.get("tf32", False))
    p.add_argument("--skip-training", action="store_true",
                   help="run only the data checks (seconds, no torch training)")
    return p.parse_args()


def _args_for_fit(cli, w_phys: float):
    from argparse import Namespace
    return Namespace(
        ops=list(cli.ops), val_ops=list(cli.val_ops), test_ops=list(cli.test_ops),
        subsample=cli.subsample, train_frac=cli.train_frac, epochs=cli.epochs,
        resample=cli.resample, driver_history=cli.driver_history,
        driver_rate_lags=list(cli.driver_rate_lags),
        k_max=cli.k_max, time_deriv=cli.time_deriv,
        history_mode=cli.history_mode, rate_lags=list(cli.rate_lags),
        delta_grid=cli.delta_grid, max_rate_amp=cli.max_rate_amp,
        width=cli.width, depth=cli.depth, lr=cli.lr,
        w_data=1.0, w_phys=float(w_phys), w_bc=cli.w_bc,
        batch_data=cli.batch_data, batch_phys=cli.batch_phys,
        batch_bc=cli.batch_bc, weight_decay=cli.weight_decay,
        grad_clip=cli.grad_clip, gain_lr_mult=cli.gain_lr_mult,
        early_stopping_patience=cli.early_stopping_patience,
        phys_norm=cli.phys_norm, use_static=cli.use_static,
        use_forcing=cli.use_forcing, shuffle_ops=cli.shuffle_ops,
        holdout_tail=cli.holdout_tail, seed=cli.seed, device=cli.device,
        tf32=cli.tf32,
    )


def data_checks(cli, out: list) -> bool:
    """Everything that can be decided without training. Returns pass/fail."""
    ok = True
    all_ops = list(cli.ops) + list(cli.val_ops) + list(cli.test_ops)
    require_ops(*all_ops)
    out.append(f"  cache holds: {', '.join(available_ops())}")

    for w in check_split(cli.ops, cli.val_ops, cli.test_ops):
        out.append(f"  [SPLIT] {w}")

    bundle = load_ops(
        op_ids=list(cli.ops), subsample_time=cli.subsample,
        train_frac=cli.train_frac, resample=cli.resample,
        driver_rate_lags=[float(v) for v in cli.driver_rate_lags],
        use_driver_history=bool(cli.driver_history),
    )
    held = [build_op(o, bundle, subsample_time=cli.subsample,
                     train_frac=cli.train_frac)
            for o in list(cli.val_ops) + list(cli.test_ops)]
    out += ["", *normalisation_report(bundle), "", *profile_report(bundle, held)]

    # --- 1. do the bundles carry the profiles the plan sheet claims? ---------
    out.append("")
    out.append("CHECK 1 - profiles present as the plan sheet describes them")
    for op in list(bundle.ops) + held:
        detected = {p for p in op.profiles_detected if p in PROFILE_CHANNELS}
        claimed = set(profiles_of(op.op_id))
        if detected != claimed:
            ok = False
            out.append(
                f"  FAIL {op.op_id}: bundle has {sorted(detected) or '[]'}, the "
                f"sheet says {sorted(claimed) or '[]'}. Either the cache was "
                f"built from the wrong export or op_registry.OPS is wrong; "
                f"resolve it before trusting any tier label."
            )
    if ok:
        out.append("  PASS every OP carries exactly the profiles the sheet claims")

    # --- 2. does each profile cover its OP's timeline? -----------------------
    out.append("")
    out.append("CHECK 2 - each profile spans the whole OP it belongs to")
    gaps = 0
    for op in list(bundle.ops) + held:
        t_lo, t_hi = float(op.t[0]), float(op.t[-1])
        for name, (lo, hi) in sorted(op.profile_coverage.items()):
            if lo > t_lo + 1e-6 or hi < t_hi - 1e-6:
                gaps += 1
                out.append(
                    f"  WARN {op.op_id}.{name}: profile covers {lo:.1f}..{hi:.1f} s, "
                    f"OP runs {t_lo:.1f}..{t_hi:.1f} s -> held flat outside. "
                    f"Not fatal, but the model is being taught a plateau that "
                    f"the simulation may not have had."
                )
    if not gaps:
        out.append("  PASS every profile covers its own OP end to end")

    # --- 3. is any driver rate channel dead across the training set? ---------
    out.append("")
    out.append("CHECK 3 - driver rate channels carry signal")
    if not bundle.use_driver_history:
        out.append("  SKIP driver history is off")
    else:
        dead = [(bundle.driver_names[d], bundle.driver_rate_lags[i])
                for d in range(bundle.driver_rate_active.shape[0])
                for i in range(bundle.driver_rate_active.shape[1])
                if not bundle.driver_rate_active[d, i]]
        live = int(bundle.driver_rate_active.sum())
        if dead:
            out.append(
                f"  NOTE {len(dead)} of {bundle.n_driver_rate} rate channels are "
                f"flat across the whole training set and are forced to 0: "
                + ", ".join(f"{n}@{lag:g}s" for n, lag in dead))
            out.append(
                "  That is correct behaviour for a driver no training OP varies "
                "(c_rate is a label, not a signal, and fluid_mass_flow is a "
                "profile only in OP15), but it means the model has no trained "
                "meaning for that channel if a held-out OP does vary it.")
        if live == 0:
            ok = False
            out.append("  FAIL no rate channel carries signal at all - either no "
                       "training OP has a profile, or the profiles are not being "
                       "read. Driver history is pure overhead in this state.")
        else:
            out.append(f"  PASS {live} rate channel(s) carry signal")

    # --- 4. hybrid history amplification ------------------------------------
    out.append("")
    out.append("CHECK 4 - hybrid history amplification of the opening steps")
    if cli.history_mode != "hybrid":
        out.append(f"  SKIP history_mode={cli.history_mode}")
    else:
        lags_n = [float(v) / bundle.T_span_ref for v in cli.rate_lags]
        amps = hybrid_rate_amplification(bundle.dTdt_scale, lags_n)
        scale, lines = effective_rate_scale(bundle.dTdt_scale, lags_n,
                                            float(cli.max_rate_amp))
        out += ["  " + ln.strip() for ln in lines]
        worst = float(amps.max()) if amps.size else 0.0
        eff = hybrid_rate_amplification(scale, lags_n)
        eff_worst = float(eff.max()) if eff.size else 0.0
        if eff_worst > 200.0:
            out.append(
                f"  WARN effective amplification is {eff_worst:.4g}. Training may "
                f"abort with a non-finite L_data in epoch 1; --max-rate-amp 50 "
                f"is the guard, --history-mode raw the escape hatch.")
        else:
            out.append(f"  PASS effective amplification {eff_worst:.4g} "
                       f"(unguarded {worst:.4g})")

    # --- 5. coverage of the held-out OPs ------------------------------------
    out.append("")
    out.append("CHECK 5 - where the held-out OPs leave the trained envelope")
    for op in held:
        out.append(f"  {op.op_id} [{tier_of(op.op_id)}]:")
        out += ["    " + ln.strip() for ln in coverage_report(bundle, op)]
    return ok


def main() -> int:
    cli = parse_args()
    device = resolve_device(cli.device)
    cli.device = str(device)

    out = ["=" * 72,
           "PROFILE-EXTENSION SMOKE TEST",
           "=" * 72,
           f"  train    : {', '.join(cli.ops)}",
           f"  val      : {', '.join(cli.val_ops)}",
           f"  test     : {', '.join(cli.test_ops)}",
           f"  dt       : {0.1*cli.subsample:.1f}s (subsample={cli.subsample})",
           f"  epochs   : {cli.epochs}   width={cli.width} depth={cli.depth}",
           f"  resample : {cli.resample}   driver_history={cli.driver_history} "
           f"lags={cli.driver_rate_lags}s",
           "=" * 72]
    print("\n".join(out), flush=True)

    ok = data_checks(cli, out)
    print("\n".join(out[10:]), flush=True)

    if cli.skip_training:
        out.append("")
        out.append("training checks SKIPPED (--skip-training)")
    else:
        from train import fit  # imported late so --skip-training needs no model

        out.append("")
        out.append("CHECK 6 - short trainings converge and stay finite")
        l_data_finals = []
        for w_phys in cli.w_phys:
            print(f"\n{'='*56}\nshort training, w_phys={w_phys}\n{'='*56}",
                  flush=True)
            try:
                model, bundle, _packed, _dtn, hist = fit(_args_for_fit(cli, w_phys))
            except Exception as exc:
                ok = False
                out.append(f"  FAIL w_phys={w_phys}: training raised {exc!r}")
                continue
            model.eval()
            L_first = hist["L_data"][0] if hist["L_data"] else float("nan")
            L_last = hist["L_data"][-1] if hist["L_data"] else float("nan")
            if np.isfinite(L_last):
                l_data_finals.append(float(L_last))
            stable = np.isfinite(L_last) and np.isfinite(
                hist["L_phys"][-1] if hist["L_phys"] else np.nan)
            converged = np.isfinite(L_first) and np.isfinite(L_last) and L_last < L_first
            bal = hist["L_phys_bal"][-1] if hist["L_phys_bal"] else float("nan")
            balanced = np.isfinite(bal) and 0.01 < bal < 100

            train_mae = float(np.mean([
                op_metrics(rollout_phys(model, op, bundle, device), op,
                           late_is_holdout=cli.holdout_tail)["mae"]
                for op in bundle.ops]))
            # The profile OPs are the point, so they are scored separately: a
            # mean over eleven OPs where eight are constant would let a model
            # that ignores the profiles entirely still look fine.
            prof_ops = [op for op in bundle.ops if op.profiles_detected]
            prof_mae = float(np.mean([
                op_metrics(rollout_phys(model, op, bundle, device), op,
                           late_is_holdout=cli.holdout_tail)["mae"]
                for op in prof_ops])) if prof_ops else float("nan")
            val_maes = []
            for op_id in cli.val_ops:
                held_op = build_op(op_id, bundle, subsample_time=cli.subsample,
                                   train_frac=cli.train_frac)
                val_maes.append(op_metrics(
                    rollout_phys(model, held_op, bundle, device), held_op,
                    late_is_holdout=True)["mae"])
            val_mae = float(np.mean(val_maes))
            mae_ok = np.isfinite(train_mae) and train_mae < MAE_SANITY_C
            passed = stable and converged and balanced and mae_ok
            ok = ok and passed
            out.append(
                f"  {'PASS' if passed else 'FAIL'} w_phys={w_phys:g}: "
                f"L_data {L_first:.4e} -> {L_last:.4e}"
                f"{'' if converged else ' (NOT decreasing)'}, "
                f"L_phys_bal={bal:.4e}{'' if balanced else ' (not ~O(1))'}, "
                f"train MAE={train_mae:.2f} C, profile-OP MAE={prof_mae:.2f} C, "
                f"val MAE={val_mae:.2f} C"
                f"{'' if mae_ok else f' (above the {MAE_SANITY_C:g} C sanity bar)'}"
            )

    out += ["", "=" * 72]
    if ok:
        out.append("ALL CHECKS PASSED")
        out.append("  next (stage 1, preprocessing - settle this BEFORE the "
                   "weights, because")
        out.append("  --resample changes q_dot and therefore Qsrc_scale, which is "
                   "what w_phys is")
        out.append("  measured against):")
        out.append("    python3 PINNmodulusTwoExtProfiles/profileBench.py "
                   "--axes resample drivhist drlags --epochs 20 --seeds 0 1 2")
        if l_data_finals:
            # w_phys multiplies a loss that is divided by its own EMA and so sits
            # at ~1: it is the floor the physics term holds at, while
            # w_data * L_data falls. The end-of-training ratio between the two is
            # therefore ~ w_phys / L_data_final, which is what makes a short run's
            # L_data the right thing to pick the sweep range from.
            ref = float(np.median(l_data_finals))

            def _sig2(v: float) -> float:
                """Round to 2 significant figures - a suggested bracket has no
                business printing six digits it cannot justify."""
                if v <= 0:
                    return 0.0
                return float(f"{v:.2g}")

            grid = sorted({_sig2(v) for v in
                           (0.0, ref / 10.0, ref, 3.0 * ref, 10.0 * ref)})
            out.append(f"  then (stage 2, weights) - L_data ended at ~{ref:.3g} "
                       f"here, and w_phys is the")
            out.append("  near-constant FLOOR the physics term holds at, so the "
                       "physics-to-data ratio")
            out.append("  at the end is ~ w_phys / L_data_final. A range built "
                       "around that value:")
            out.append("    python3 PINNmodulusTwoExtProfiles/profileBench.py "
                       "--axes wphys wbc --epochs 20 --seeds 0 1 2 \\")
            out.append("        --w-phys-values "
                       + " ".join(f"{v:g}" for v in grid))
            out.append("  (this L_data comes from a deliberately short, coarse "
                       "run - treat the range as")
            out.append("  a starting bracket, and re-derive it from a full-length "
                       "run before refining.)")
        out.append("  NOTE a pass here means the pipeline is sound, NOT that the "
                   "model is accurate. Accuracy is what profileBench measures.")
    else:
        out.append("SOME CHECKS FAILED - fix them before spending hours on a sweep")
    out.append("=" * 72)

    (ART_DIR / "smokeBench_results.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[-20:]), flush=True)
    print(f"\n  wrote {ART_DIR / 'smokeBench_results.txt'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
