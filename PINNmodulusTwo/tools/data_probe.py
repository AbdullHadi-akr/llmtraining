"""What do the real bundles say about A, and does anything change in 5 seconds?

Answers TODO-1 and the first check of TODO-2 from UEBERGABE_2026-08-27.txt
WITHOUT training anything: no Modulus, no torch, no GPU, no material_properties.
It needs numpy and the ``data_cache/OP*.npz`` bundles, and it reads only ``t_fast``
and ``T`` out of them.

    python3 PINNmodulusTwo/tools/data_probe.py

Three questions, in the order the handover asks them:

1. Is ``dTdt_scale`` really ~2.479, i.e. does ``A = 1/(lag_n * rate_scale)`` come
   out at ~119/30 for rate_lags [5, 20]? If not, chapter 4 of the handover has to
   be recomputed against these numbers.
2. How large is the true temperature change over each rate window, in millikelvin?
   A = 119 predicts ~42 mK over 5 s at T_sigma = 5 K.
3. Is that above the resolution of the stored simulation output, or is the first
   rate channel mostly measuring discretisation noise? Reported as the ratio of
   the window change to a per-sample noise estimate, plus the quantisation step
   the values are actually stored with.

The statistics follow data.py: ``T_span_ref`` is the longest OP timeline,
``T_mu``/``T_sigma`` are pooled over the first ``train_frac`` of every training
OP, and ``dTdt_scale`` is the RMS of the central difference ``dTn/dtn`` over the
same part. Numbers therefore compare directly against train.py's start-up line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_ROOT = _PROJECT.parent

# Same search order as data.py, most specific first.
_CACHE_CANDIDATES = (
    _PROJECT / "data_cache",
    _ROOT / "data_cache",
    _ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "data_cache",
    _ROOT / "battery_surrogate_agenticWorkflow" / "data_cache",
)


def _resolve_cache(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"[ABORT] --data-cache {p} is not a directory")
        return p
    for c in _CACHE_CANDIDATES:
        if c.is_dir() and any(c.glob("*.npz")):
            return c
    sys.exit("[ABORT] no data_cache with OP*.npz found. Looked in:\n  "
             + "\n  ".join(str(c) for c in _CACHE_CANDIDATES)
             + "\nPass --data-cache <path> if it lives elsewhere.")


def _decimate(a: np.ndarray, step: int) -> np.ndarray:
    """Stride subsampling, the config default (subsample_mode: stride)."""
    return a if step <= 1 else a[::step]


def _read(cache: Path, op_id: str, step: int) -> dict:
    npz = np.load(cache / f"{op_id}.npz", allow_pickle=True)
    t_raw = np.asarray(npz["t_fast"], dtype=np.float64)
    T_raw = np.asarray(npz["T"])
    return dict(op_id=op_id, t=_decimate(t_raw, step), T=_decimate(T_raw, step),
                stored_dtype=T_raw.dtype, n_t_raw=T_raw.shape[0],
                dt_raw=float(np.median(np.diff(t_raw))) if t_raw.size > 1 else float("nan"))


def _quantisation_step(T: np.ndarray) -> float:
    """Smallest gap between distinct stored values -- the resolution of the export.

    For a float grid this lands at the float spacing (nothing was quantised); for
    values written out through a rounded text format it lands at that rounding.
    """
    v = np.unique(np.asarray(T, dtype=np.float64).ravel())
    if v.size < 2:
        return float("nan")
    d = np.diff(v)
    d = d[d > 0]
    return float(d.min()) if d.size else float("nan")


def _noise_sigma(T: np.ndarray) -> float:
    """Per-sample high-frequency scale, in the units of ``T``.

    ``T[i-1] - 2 T[i] + T[i+1]`` annihilates anything locally linear, so for a
    smooth trajectory plus white per-sample jitter of sigma its RMS is
    sqrt(6)*sigma. Real curvature also survives it, so this is an UPPER bound on
    the noise: if the window change beats it, the change is certainly real. The
    median over grid points keeps one loud point from setting the level.
    """
    if T.shape[0] < 3:
        return float("nan")
    d2 = T[:-2] - 2.0 * T[1:-1] + T[2:]
    per_point = np.sqrt((d2.astype(np.float64) ** 2).mean(axis=0) / 6.0)
    return float(np.median(per_point))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-cache", default=None, help="override the data_cache location")
    ap.add_argument("--ops", nargs="+", default=["OP01", "OP02", "OP03", "OP04", "OP05"],
                    help="training OPs the statistics are pooled over (config.yaml: ops)")
    ap.add_argument("--subsample", type=int, default=2,
                    help="config.yaml subsample_time; the model sees this grid")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="data.py build_bundle(train_frac=...)")
    ap.add_argument("--rate-lags", type=float, nargs="+", default=[5.0, 20.0],
                    help="seconds, config.yaml rate_lags")
    args = ap.parse_args()

    cache = _resolve_cache(args.data_cache)
    have = sorted(p.stem for p in cache.glob("*.npz"))
    print(f"data_cache: {cache}")
    print(f"  bundles present: {', '.join(have) if have else '(none)'}")
    missing = [o for o in args.ops if o not in have]
    if missing:
        sys.exit(f"[ABORT] requested OPs not in the cache: {', '.join(missing)}")

    raw = [_read(cache, op, args.subsample) for op in args.ops]

    print(f"\n=== 0. geometry (subsample {args.subsample}, stride) ===")
    print(f"{'OP':6} {'n_t':>7} {'dt/s':>7} {'span/s':>9} {'P':>6} "
          f"{'T_min':>8} {'T_max':>8}  stored as")
    for r in raw:
        t, T = r["t"], np.asarray(r["T"], dtype=np.float64)
        dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
        print(f"{r['op_id']:6} {T.shape[0]:7d} {dt:7.3f} {t[-1] - t[0]:9.1f} "
              f"{T.shape[1]:6d} {T.min():8.3f} {T.max():8.3f}  {r['stored_dtype']}"
              f"  (raw n_t={r['n_t_raw']}, dt={r['dt_raw']:.3f}s)")

    # ---- shared statistics, exactly as data.py pools them ---------------------
    T_span_ref = float(max(r["t"].max() - r["t"][0] for r in raw))
    pool, splits = [], []
    for r in raw:
        split_t = int(args.train_frac * r["T"].shape[0])
        splits.append(split_t)
        pool.append(np.asarray(r["T"][:split_t], dtype=np.float64).ravel())
    flat = np.concatenate(pool)
    T_mu = float(flat.mean())
    T_sigma = float(flat.std()) + 1e-12

    dTdt_pool = []
    for r, split_t in zip(raw, splits):
        tn = (r["t"] - r["t"][0]) / (T_span_ref + 1e-12)
        Tn = (np.asarray(r["T"], dtype=np.float64) - T_mu) / T_sigma
        dTdt = (Tn[2:] - Tn[:-2]) / (tn[2:] - tn[:-2])[:, None]
        dTdt_pool.append(dTdt[:split_t].ravel())
    dTdt_scale = float(np.sqrt((np.concatenate(dTdt_pool) ** 2).mean())) + 1e-6

    print(f"\n=== 1. is A really ~119/30? ===")
    print(f"  T_span_ref  = {T_span_ref:.1f} s        (handover assumed ~1474 s)")
    print(f"  T_mu        = {T_mu:.4f}")
    print(f"  T_sigma     = {T_sigma:.4f}            (handover assumed ~5 K)")
    print(f"  dTdt_scale  = {dTdt_scale:.4f}            (handover assumed ~2.479)")
    print(f"  = rate_scale, with max_rate_amp 0.0")
    print(f"\n  {'lag/s':>8} {'lag_n':>10} {'A':>10}")
    amps = {}
    for lag in args.rate_lags:
        lag_n = lag / (T_span_ref + 1e-12)
        A = 1.0 / (max(lag_n, 1e-30) * max(dTdt_scale, 1e-30))
        amps[lag] = A
        print(f"  {lag:8.1f} {lag_n:10.3e} {A:10.1f}")
    ratio = dTdt_scale / 2.479
    if not 0.8 < ratio < 1.25:
        print(f"\n  [WARN] dTdt_scale is {ratio:.2f}x the 2.479 the handover assumed."
              f"\n         Chapter 4 of UEBERGABE_2026-08-27.txt has to be recomputed"
              f"\n         against these A values before the lag choice is trusted.")
    else:
        print(f"\n  [OK] dTdt_scale within {abs(1 - ratio) * 100:.0f}% of the assumed"
              f" 2.479; the handover's A values hold.")

    print(f"\n=== 2./3. does anything change over a rate window? (TODO-1) ===")
    print("  |dT| over the window against a per-sample noise estimate, per OP.")
    print("  sigma_n is an UPPER bound (curvature counts into it), so the SNR")
    print("  column is a LOWER bound. Everything in K unless marked mK.\n")
    print(f"  {'OP':6} {'lag/s':>6} {'|dT| med':>11} {'|dT| p10':>11} "
          f"{'sigma_n':>11} {'SNR':>8}  {'quantum':>11}")
    verdicts = {lag: [] for lag in args.rate_lags}
    for r in raw:
        T = np.asarray(r["T"], dtype=np.float64)
        t = r["t"]
        dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
        sigma_n = _noise_sigma(T)
        quantum = _quantisation_step(T)
        for lag in args.rate_lags:
            k = max(int(round(lag / dt)), 1)
            if k >= T.shape[0]:
                print(f"  {r['op_id']:6} {lag:6.1f}   window longer than the trajectory")
                continue
            d = np.abs(T[k:] - T[:-k])
            med = float(np.median(d))
            p10 = float(np.percentile(d, 10))
            # a window difference carries two independent samples of the jitter
            snr = med / (sigma_n * np.sqrt(2.0)) if sigma_n > 0 else float("inf")
            verdicts[lag].append(snr)
            print(f"  {r['op_id']:6} {lag:6.1f} {med * 1e3:8.1f} mK {p10 * 1e3:8.1f} mK "
                  f"{sigma_n * 1e3:8.1f} mK {snr:8.1f}  {quantum:8.2e} K"
                  f"  (k={k} steps)")

    print("\n  verdict per lag (median SNR over the OPs):")
    for lag in args.rate_lags:
        if not verdicts[lag]:
            continue
        s = float(np.median(verdicts[lag]))
        if s >= 10.0:
            note = "real signal -- the channel measures temperature, keep the lag"
        elif s >= 3.0:
            note = "signal, but thin. Worth comparing against a longer first segment"
        else:
            note = ("mostly discretisation noise -- try longer first segments"
                    " (e.g. [20, 60]), but see chapter 4: too long and the channel"
                    " turns into a progress indicator")
        print(f"    lag {lag:g} s (A = {amps[lag]:.0f}): SNR ~ {s:.1f}  -> {note}")

    print("\nDone. Paste this whole output back into the session; it is aggregate"
          "\nnumbers only, no trajectory data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
