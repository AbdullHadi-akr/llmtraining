#!/usr/bin/env python3
"""Read ``artifacts/history.csv`` the way the roadmap says results must be read.

Why this exists
---------------
Twice now a finding in this project came from the LAST ROW of a 60-row file, and
once it was wrong:

* 5b read the ``spread`` trend off three epochs and concluded the physics term
  collapses the field. Step 6 refuted it -- ``spread_time`` runs to 0.968.
* O12 read ``ratio_bc = 0.0178`` off one epoch and concluded the BC term does
  nothing. The median over the last thirty epochs is 0.0581, the series scatters
  by 59 % relative, and five of thirty epochs sit below that final value. There
  was no trend to read.

The series in this file are noisy enough that a single row is a sample, not a
result. So this prints the MEDIAN and the SPREAD over a window, puts the final
value next to it, and says out loud when the two disagree. That is the roadmap's
rule (FAHRPLAN.md, "Die Auswahlregeln stehen fest") expressed as a program, so it
is applied rather than remembered.

Standard library only, on purpose: it has to run on a machine that has the CSV
but not the training environment.

Usage
-----
    python3 tools/analyse_history.py [artifacts/history.csv] [--last 30]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

# Series that carry a per-epoch measurement worth summarising. ``epoch`` is the
# index and ``delta`` is a constant of the run, so neither is summarised here --
# delta is reported separately because a CHANGING delta would mean the run was
# not the one configuration it claims to be.
SUMMARY_KEYS = (
    "L_data", "L_phys", "L_bc",
    "L_phys_bal", "L_bc_bal",
    "ratio_phys", "ratio_bc",
    "div_data", "div_phys", "div_bc",
    "spread_space", "spread_time",
)

# The loss/divisor pairs the balancing is supposed to keep at the same order.
# ``L_data`` has no ``L_data_bal`` column, so the pairing is stated here rather
# than derived from the names.
BALANCE_PAIRS = (("data", "L_data", "div_data"),
                 ("phys", "L_phys", "div_phys"),
                 ("bc", "L_bc", "div_bc"))

# spread is a SIDE CONDITION, not a target (FAHRPLAN §10): a configuration with a
# better MAE at spread 0.2 has stopped predicting rather than won.
SPREAD_OK = (0.7, 1.3)

# A divisor more than this far above its own loss is not an estimate of that
# loss any more. 10x is generous -- the EMA lags by design -- and O15 was found
# at 1140x.
DIVISOR_STALE_FACTOR = 10.0

# How far into a tail the final epoch has to sit before the report says so.
# 0.2 rather than 0.15 because O12 -- the case this whole tool exists for -- had
# 5 of 30 epochs below its final value, i.e. 0.167. A threshold that misses the
# one worked example is the wrong threshold.
TAIL_FRAC = 0.2

# A series whose steps go one way this often has a direction, so its last value
# is its state and not a sample.
MONOTONE_FRAC = 0.8


def _finite(values):
    return [v for v in values if v is not None and math.isfinite(v)]


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("epoch")]
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    out = []
    for r in rows:
        rec = {}
        for k, v in r.items():
            if k is None:
                continue
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = float("nan")
        out.append(rec)
    return out


def column(rows, key):
    return [r.get(key, float("nan")) for r in rows]


def summarise(values):
    """Median, spread and the position of the FINAL value inside the window.

    ``below`` is the point of the whole exercise: it says how many epochs of the
    window sit under the value someone would have quoted by reading the last
    line. O12's 0.0178 had five of thirty below it.
    """
    v = _finite(values)
    if not v:
        return None
    last = v[-1]
    med = st.median(v)
    mean = st.mean(v)
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    steps = [b - a for a, b in zip(v, v[1:])]
    down = sum(1 for d in steps if d < 0) / len(steps) if steps else 0.5
    return {
        "n": len(v), "last": last, "median": med, "mean": mean, "std": sd,
        "min": min(v), "max": max(v),
        "rel_std": (sd / abs(mean)) if mean else float("nan"),
        "below": sum(1 for x in v if x < last),
        "down_frac": down,
    }


def geometric_rate(values):
    """Per-step factor implied by the first and last finite value.

    Used on the divisors: a series that decays at a constant geometric rate,
    independent of the loss it is meant to estimate, is not estimating anything.
    """
    v = _finite(values)
    v = [x for x in v if x > 0.0]
    if len(v) < 2:
        return float("nan")
    return (v[-1] / v[0]) ** (1.0 / (len(v) - 1))


def fmt(x, width=11):
    if x is None or not math.isfinite(x):
        return f"{'n/a':>{width}}"
    a = abs(x)
    if a and (a < 1e-3 or a >= 1e5):
        return f"{x:>{width}.3e}"
    return f"{x:>{width}.4f}"


def analyse(rows, *, last=30, ema_decay=0.9, spike_factor=5.0):
    """Return ``(stats, notes, window)`` for already-parsed rows.

    Separate from :func:`main` so the flags can be tested as values. A check
    that only exists inside a print statement is a check nothing can assert on,
    and the flags are the actual product of this tool.
    """
    win = rows[-last:] if last > 0 else rows
    stats = {key: summarise(column(win, key)) for key in SUMMARY_KEYS}
    notes: list[str] = []

    # --- the last-line trap --------------------------------------------------
    # "below" near 0 or near n means the final epoch sits in a tail of its own
    # window. Quoting it as the run's value is exactly how O12 happened.
    #
    # A MONOTONE series is exempt, and that exemption is what makes the flag
    # useful rather than noisy: the last value of a series with a direction is
    # its current state, not an outlier. The divisors fall monotonically for the
    # whole run, so without this they would raise the loudest warning in the
    # report and drown the one that matters.
    for key, s in stats.items():
        if s is None or s["n"] < 8:
            continue
        if s["down_frac"] >= MONOTONE_FRAC or s["down_frac"] <= 1.0 - MONOTONE_FRAC:
            continue
        frac = s["below"] / s["n"]
        if frac <= TAIL_FRAC or frac >= 1.0 - TAIL_FRAC:
            side = "low" if frac <= TAIL_FRAC else "high"
            notes.append(
                f"[LAST-LINE] {key}: the final epoch ({fmt(s['last'], 0)}) sits in the "
                f"{side} tail of its own window -- {s['below']} of {s['n']} epochs are "
                f"below it, median is {fmt(s['median'], 0)}, spread is "
                f"{s['rel_std'] * 100:.0f}% relative. Quote the median, not this.")

    # --- O15: is the balancing actually balancing? ---------------------------
    for name, lkey, dkey in BALANCE_PAIRS:
        lw, dw = _finite(column(win, lkey)), _finite(column(win, dkey))
        if not lw or not dw:
            continue
        ratios = [d / l for d, l in zip(dw, lw) if l > 0]
        if not ratios:
            continue
        med = st.median(ratios)
        rate = geometric_rate(column(rows, dkey))
        pure = math.isfinite(rate) and abs(rate - ema_decay) < 5e-3
        if med > DIVISOR_STALE_FACTOR:
            why = (f" and it is falling at {rate:.4f}/epoch, which IS --ema-decay "
                   f"({ema_decay:g}): it is decaying, not tracking"
                   if pure else f" (falling at {rate:.4f}/epoch)")
            notes.append(
                f"[O15] {dkey} sits {med:,.0f}x above {lkey} over the window{why}. "
                f"The balancing has not engaged, so w_{name} does not mean what it says "
                f"and a weight sweep would measure this offset. See FAHRPLAN §11.6.")

    # A divisor RATIO that is constant means the mix between terms is frozen,
    # whatever the individual terms did. That is the part Adam does not cancel.
    #
    # Only worth saying when a divisor is ALSO stale: three divisors that sit on
    # their terms and shrink together are the balancing working, and flagging
    # that would train the reader to ignore the flag.
    rates = {d: geometric_rate(column(rows, d))
             for _, _, d in BALANCE_PAIRS if _finite(column(rows, d))}
    if len(rates) > 1 and notes:
        vals = [r for r in rates.values() if math.isfinite(r)]
        if len(vals) > 1 and (max(vals) - min(vals)) < 5e-3:
            notes.append(
                f"[O15] all divisors decay at the same rate "
                f"({', '.join(f'{k}={v:.4f}' for k, v in rates.items())}), so their "
                f"RATIO is frozen for the whole run. The mix between terms is then set "
                f"by whatever the first optimiser step produced, not by the terms.")

    # --- spread as a side condition -----------------------------------------
    for key in ("spread_time", "spread_space"):
        s = stats.get(key)
        if s is None:
            continue
        if not (SPREAD_OK[0] <= s["median"] <= SPREAD_OK[1]):
            notes.append(
                f"[SPREAD] {key} median {s['median']:.3f} is outside "
                f"[{SPREAD_OK[0]}, {SPREAD_OK[1]}]. A configuration that wins on MAE "
                f"here has stopped predicting rather than won (FAHRPLAN §10).")

    # --- excursions ----------------------------------------------------------
    lp = _finite(column(rows, "L_phys"))
    if len(lp) > 10:
        med = st.median(lp[len(lp) // 6:])
        spikes = [(int(r["epoch"]), r["L_phys"]) for r in rows
                  if math.isfinite(r.get("L_phys", float("nan")))
                  and r["L_phys"] > spike_factor * med and int(r["epoch"]) > 3]
        if spikes:
            notes.append(
                f"[EXCURSION] L_phys exceeds {spike_factor:g}x its median "
                f"({med:.3g}) in {len(spikes)} epochs: "
                f"{', '.join(f'Ep{e} ({v:.3g})' for e, v in spikes[:8])}"
                f"{' ...' if len(spikes) > 8 else ''}. A run that happens to END on one "
                f"of these reports nonsense that looks like a result -- one more reason "
                f"to compare medians over a window.")

    # --- delta must be a constant of the run --------------------------------
    dlt = sorted({round(v, 12) for v in _finite(column(rows, "delta"))})
    if len(dlt) > 1:
        notes.append(f"[DELTA] delta is not constant over the run: {dlt}. "
                     f"The rows are then not one configuration.")

    return stats, notes, win


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="artifacts/history.csv",
                    help="path to history.csv (default: artifacts/history.csv)")
    ap.add_argument("--last", type=int, default=30,
                    help="window of trailing epochs to summarise (default 30)")
    ap.add_argument("--ema-decay", type=float, default=0.9,
                    help="the run's --ema-decay, to recognise a divisor that is "
                         "in free fall rather than tracking (default 0.9)")
    ap.add_argument("--spike-factor", type=float, default=5.0,
                    help="L_phys above this multiple of its median counts as an "
                         "excursion worth naming (default 5)")
    args = ap.parse_args(argv)

    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"{path}: not found. Run training first, or pass the path.")
    rows = read_history(path)
    stats, notes, win = analyse(rows, last=args.last, ema_decay=args.ema_decay,
                               spike_factor=args.spike_factor)
    lo_ep, hi_ep = int(win[0]["epoch"]), int(win[-1]["epoch"])

    print(f"{path}  --  {len(rows)} epochs, window = Ep{lo_ep}-{hi_ep} ({len(win)} rows)")
    print()
    print(f"{'series':<13}{'last':>11}{'median':>11}{'mean':>11}{'std':>11}"
          f"{'rel.std':>9}{'min':>11}{'max':>11}   below")
    print("-" * 99)
    for key in SUMMARY_KEYS:
        s_ = stats[key]
        if s_ is None:
            print(f"{key:<13}{'-- not recorded (weight 0?) --':>50}")
            continue
        rel = f"{s_['rel_std'] * 100:>8.0f}%" if math.isfinite(s_["rel_std"]) else "     n/a"
        print(f"{key:<13}{fmt(s_['last'])}{fmt(s_['median'])}{fmt(s_['mean'])}"
              f"{fmt(s_['std'])}{rel}{fmt(s_['min'])}{fmt(s_['max'])}"
              f"   {s_['below']}/{s_['n']}")

    dlt = sorted({round(v, 12) for v in _finite(column(rows, "delta"))})
    if len(dlt) == 1 and dlt[0]:
        print(f"\ndelta (normalised) = {dlt[0]:.6g}"
              f"  ->  T_span_ref = {1.0 / dlt[0]:.1f} s")

    print()
    if notes:
        for note in notes:
            print(note)
            print()
    else:
        print("no flags raised over this window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
