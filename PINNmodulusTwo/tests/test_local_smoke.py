"""Regression tests for the pieces that make a local run readable.

Everything here guards a failure this project has actually had:

* a training report that attributed a FAIL to a check which had passed, because
  the benchmark printed ``FAIL`` and nothing else;
* a BC term whose contribution was invisible in every summary table;
* an MAE quoted against a baseline measured on a different dataset;
* a history series added to ``train.fit`` and not to ``bench_common.EMPTY_HIST``,
  which only shows up when a benchmark aggregates a crashed seed.

The fixture is the synthetic cache, so these run on a bare checkout. conftest.py
has already substituted Modulus by the time this module is imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(PKG_DIR / "tools"))

import bench_common  # noqa: E402
import data as data_mod  # noqa: E402
import make_synthetic_cache as msc  # noqa: E402
import smallBench  # noqa: E402
import train as train_mod  # noqa: E402


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_baselines_are_measured_on_the_op_in_hand():
    """Both trivial predictors, computed from T_lab and the bundle mean.

    The numbers in README_ERSTER_TEST chapter 6 came off a synthetic bundle and
    its own text says they do not transfer as absolute values, so the baselines
    have to be recomputed per run rather than quoted.
    """
    lab = np.array([[10.0, 20.0],
                    [12.0, 26.0],
                    [14.0, 32.0]])
    op = SimpleNamespace(T_lab=lab)
    bundle = SimpleNamespace(T_mu=20.0)

    base = smallBench._baseline_maes(op, bundle)

    # persistence holds row 0: errors are 0,0 / 2,6 / 4,12 -> mean 4.0
    assert base["persistence"] == pytest.approx(4.0)
    # train mean 20: errors 10,0 / 8,6 / 6,12 -> mean 7.0
    assert base["train_mean"] == pytest.approx(7.0)


def test_persistence_baseline_is_zero_for_a_constant_field():
    """A field that never moves is predicted exactly by "it never moves".

    This is the degenerate case that makes the skill number readable: if the
    held-out OP were flat, beating persistence would be impossible rather than
    merely hard, and the run should not silently report a huge skill.
    """
    lab = np.full((5, 3), 7.5)
    base = smallBench._baseline_maes(SimpleNamespace(T_lab=lab),
                                     SimpleNamespace(T_mu=7.5))
    assert base["persistence"] == pytest.approx(0.0)
    assert base["train_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# synthetic cache
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    """A short synthetic cache in a tmp dir, with data.DATA_CACHE pointed at it.

    ``--seconds 60`` keeps the rollout to a few hundred steps so a real ``fit``
    call is a test and not a benchmark. The amplification A is off the measured
    regime at that length by construction (see make_synthetic_cache), which is
    fine here: nothing in this file asserts anything about stability.
    """
    msc.ensure_material_properties()
    out = tmp_path_factory.mktemp("data_cache")
    for op_id in ("OP01", "OP02"):
        msc.write_op(op_id, out, seconds=60.0, seed=0)
    old = data_mod.DATA_CACHE
    data_mod.DATA_CACHE = out
    yield out
    data_mod.DATA_CACHE = old


def test_synthetic_bundle_loads_through_the_real_loader(synthetic_cache):
    bundle = data_mod.load_ops(op_ids=["OP01", "OP02"], subsample_time=40)
    assert len(bundle.ops) == 2
    assert bundle.T_sigma > 0
    assert np.isfinite(bundle.phys_scale)
    # bc_scale must come from measured x-neighbour pairs, not the 1/L_ref
    # fallback: the grid puts three x planes on every (y, z) line precisely so
    # the measurement has something to work with.
    assert bundle.bc_pairs > 0


def test_synthetic_labels_satisfy_the_neumann_bc(synthetic_cache):
    """dT/dx = 0 at x = 0 in the fixture itself.

    physics.boundary_condition_loss drives the model's dT/dx to zero on that
    plane. If the labels disagreed, the BC term and the data term would be
    pulling against each other and a BC term that never converges would be
    unattributable.
    """
    bundle = data_mod.load_ops(op_ids=["OP01"], subsample_time=40)
    op = bundle.ops[0]
    xn = bundle.xn
    at_zero = np.abs(xn[:, 0]) < 1e-6
    assert at_zero.sum() > 0

    # Compare the x=0 plane against the next plane out, at the same (y, z).
    planes = np.unique(np.round(xn[:, 0], 12))
    second = planes[1]
    nxt = np.abs(xn[:, 0] - second) < 1e-9
    key = lambda m: np.lexsort((xn[m, 2], xn[m, 1]))  # noqa: E731
    a = op.T_lab[:, at_zero][:, key(at_zero)]
    b = op.T_lab[:, nxt][:, key(nxt)]
    # A cosine flat at x=0 differs from the next plane by O(dx^2), so the drop
    # across the first gap must be far smaller than the field's own spread.
    assert np.abs(a - b).mean() < 0.25 * op.T_lab.std()


def test_cache_is_flagged_as_synthetic(synthetic_cache):
    """smallBench must be able to tell the fixture from measured data.

    ``_cache_is_synthetic`` re-reads ``data.DATA_CACHE`` on each call, so the
    fixture's monkeypatched cache is what gets inspected here.
    """
    assert smallBench._cache_is_synthetic() is True


def test_a_cache_without_the_marker_is_not_flagged(tmp_path, monkeypatch):
    """The banner must not fire on measured bundles, which carry no marker."""
    np.savez_compressed(tmp_path / "OP01.npz", T=np.zeros((2, 2)))
    monkeypatch.setattr(data_mod, "DATA_CACHE", tmp_path)
    assert smallBench._cache_is_synthetic() is False


# --------------------------------------------------------------------------
# history bookkeeping
# --------------------------------------------------------------------------

def _tiny_args(cache_dir, **over):
    argv = ["train.py", "--ops", "OP01", "--epochs", "1", "--subsample", "40",
            "--inner-steps", "1", "--batch-data", "64", "--batch-phys", "16",
            "--batch-bc", "8", "--width", "8", "--depth", "2",
            "--device", "cpu", "--no-residual-output"]
    old = sys.argv
    sys.argv = argv
    try:
        args = train_mod.parse_args()
    finally:
        sys.argv = old
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_fit_history_matches_empty_hist_keys(synthetic_cache):
    """Every series fit() records must exist in bench_common.EMPTY_HIST.

    The benchmarks aggregate a crashed seed's EMPTY_HIST alongside a successful
    seed's real history. A key present in only one of them is a KeyError hours
    into a sweep, which is the worst possible moment to find out.
    """
    args = _tiny_args(synthetic_cache)
    _, _, _, _, hist = train_mod.fit(args)

    assert set(hist) == set(bench_common.EMPTY_HIST)
    # and every per-epoch series is the same length as the epoch column
    n = len(hist["epoch"])
    for key, series in hist.items():
        if key == "aborted":
            continue
        assert len(series) == n, f"{key} has {len(series)} rows, epoch has {n}"


def test_fit_records_divisors_and_spread(synthetic_cache):
    """The two diagnostics a collapsed physics term would otherwise hide."""
    args = _tiny_args(synthetic_cache)
    _, _, _, _, hist = train_mod.fit(args)

    assert hist["div_phys"][-1] > 0
    assert hist["div_bc"][-1] > 0
    # Spread is a ratio against the labels, so it is positive and finite for any
    # rollout that produced numbers at all.
    assert np.isfinite(hist["spread_space"][-1])
    assert np.isfinite(hist["spread_time"][-1])
    assert hist["spread_space"][-1] >= 0.0


def test_balanced_loss_is_the_raw_loss_over_the_recorded_divisor(synthetic_cache):
    """div_* has to be the divisor that produced L_*_bal, not a second estimate.

    Recording a separately computed magnitude would drift away from the number
    the optimiser actually saw and make L_phys_bal unattributable -- which is
    the exact question 'is the divisor stale or did the term really fall?'
    needs answered.
    """
    args = _tiny_args(synthetic_cache)
    _, _, _, _, hist = train_mod.fit(args)

    assert (hist["L_phys"][-1] / hist["div_phys"][-1]
            == pytest.approx(hist["L_phys_bal"][-1], rel=1e-9))
    assert (hist["L_bc"][-1] / hist["div_bc"][-1]
            == pytest.approx(hist["L_bc_bal"][-1], rel=1e-9))


def test_zero_weight_bc_is_recorded_as_nan_not_zero(synthetic_cache):
    """A skipped term is an absent measurement, not a measurement of zero.

    smallBench excludes a zero-weight term from the 'balanced ~ O(1)' check on
    exactly this basis; if the series came back 0.0 the check would fail the
    reference run it exists to provide.
    """
    args = _tiny_args(synthetic_cache, w_bc=0.0)
    _, _, _, _, hist = train_mod.fit(args)

    assert np.isnan(hist["L_bc_bal"][-1])
    assert np.isnan(hist["div_bc"][-1])
