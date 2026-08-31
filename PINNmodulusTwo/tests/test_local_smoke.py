"""Regression tests for the pieces that make a local run readable.

Everything here guards a failure this project has actually had:

* an MAE quoted against a baseline measured on a different dataset;
* an absolute MAE quoted off the synthetic fixture as though it were a
  measurement;
* a history series added to ``train.fit`` and not to every place that appends
  to it, which misaligns the CSV and the plots by a row and raises nothing.

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

import data as data_mod  # noqa: E402
import make_synthetic_cache as msc  # noqa: E402
import train as train_mod  # noqa: E402


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_baselines_are_measured_on_the_op_in_hand():
    """Both trivial predictors, computed from T_lab and the bundle mean.

    The numbers in README_ERSTER_TEST chapter 6 came off a synthetic bundle and
    its own text says they do not transfer as absolute values, so the baselines
    have to be recomputed per run rather than quoted. ``train.evaluate`` prints
    them next to every held-out MAE for exactly that reason.
    """
    lab = np.array([[10.0, 20.0],
                    [12.0, 26.0],
                    [14.0, 32.0]])
    op = SimpleNamespace(T_lab=lab)
    bundle = SimpleNamespace(T_mu=20.0)

    persistence, mean = train_mod.trivial_baselines(op, bundle)

    # persistence holds row 0: errors are 0,0 / 2,6 / 4,12 -> mean 4.0
    assert persistence == pytest.approx(4.0)
    # train mean 20: errors 10,0 / 8,6 / 6,12 -> mean 7.0
    assert mean == pytest.approx(7.0)


def test_persistence_baseline_is_zero_for_a_constant_field():
    """A field that never moves is predicted exactly by "it never moves".

    The degenerate case behind "the bar to beat": on a flat held-out OP the bar
    is 0 C, so beating it is impossible rather than merely hard. Worth pinning
    down, because that is the one case where losing to the trivial predictor
    says nothing at all about the model.
    """
    lab = np.full((5, 3), 7.5)
    persistence, mean = train_mod.trivial_baselines(
        SimpleNamespace(T_lab=lab), SimpleNamespace(T_mu=7.5))
    assert persistence == pytest.approx(0.0)
    assert mean == pytest.approx(0.0)


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
    """``train.py`` must be able to tell the fixture from measured data.

    ``cache_is_synthetic`` re-reads ``data.DATA_CACHE`` on each call, so the
    fixture's monkeypatched cache is what gets inspected here.
    """
    assert data_mod.cache_is_synthetic() is True


def test_a_cache_without_the_marker_is_not_flagged(tmp_path, monkeypatch):
    """The banner must not fire on measured bundles, which carry no marker."""
    np.savez_compressed(tmp_path / "OP01.npz", T=np.zeros((2, 2)))
    monkeypatch.setattr(data_mod, "DATA_CACHE", tmp_path)
    assert data_mod.cache_is_synthetic() is False


# --------------------------------------------------------------------------
# the overwrite guard
# --------------------------------------------------------------------------

def test_measured_bundles_are_left_alone(tmp_path):
    """A bundle without the synthetic marker is off limits.

    ``--out`` defaults to the shared ``data_cache/``, which on the machine that
    has the measured data is where OP01.npz ... OP16.npz live. That folder is
    gitignored, so an overwrite there has nothing to restore from.
    """
    # measured: no marker. Ours: the marker write_op stores.
    np.savez_compressed(tmp_path / "OP01.npz", T=np.zeros((2, 2)))
    np.savez_compressed(tmp_path / "OP02.npz", T=np.zeros((2, 2)),
                        synthetic=np.array(True))

    # OP03 is not on disk at all and must not be reported as an obstacle.
    hits = msc.measured_bundles(tmp_path, ["OP01", "OP02", "OP03"])

    assert hits == [tmp_path / "OP01.npz"]


def test_an_unreadable_bundle_counts_as_measured(tmp_path):
    """Fail closed: what this script cannot parse, it must not overwrite."""
    (tmp_path / "OP01.npz").write_bytes(b"not an npz at all")
    assert msc.measured_bundles(tmp_path, ["OP01"]) == [tmp_path / "OP01.npz"]


# --------------------------------------------------------------------------
# history bookkeeping
# --------------------------------------------------------------------------

def _tiny_args(cache_dir, **over):
    # --val-ops/--test-ops empty: the defaults come from config.yaml and name
    # OPs the synthetic fixture does not contain.
    argv = ["train.py", "--ops", "OP01", "--epochs", "1", "--subsample", "40",
            "--inner-steps", "1", "--batch-data", "64", "--batch-phys", "16",
            "--batch-bc", "8", "--width", "8", "--depth", "2",
            "--device", "cpu", "--no-residual-output",
            "--val-ops", "--test-ops"]
    old = sys.argv
    sys.argv = argv
    try:
        args = train_mod.parse_args()
    finally:
        sys.argv = old
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_fit_history_matches_the_declared_keys(synthetic_cache):
    """fit() records exactly train.HISTORY_KEYS, and every series is aligned.

    ``fit`` has two places that append a row -- the normal epoch and the abort
    path -- and both are written out by hand. A series added to one and not the
    other shifts the plots and the CSV by a row with nothing raised, so the
    declared key list is the contract and this is what holds it.
    """
    args = _tiny_args(synthetic_cache)
    _, _, _, _, hist = train_mod.fit(args)

    assert set(hist) == set(train_mod.HISTORY_KEYS) | {"aborted"}
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

    A 0.0 would plot as a flat line the run never produced, and would read as
    'the BC term converged perfectly' in any later comparison.
    """
    args = _tiny_args(synthetic_cache, w_bc=0.0)
    _, _, _, _, hist = train_mod.fit(args)

    assert np.isnan(hist["L_bc_bal"][-1])
    assert np.isnan(hist["div_bc"][-1])


# --------------------------------------------------------------------------
# checkpoints and held-out evaluation
# --------------------------------------------------------------------------

def test_checkpoint_round_trips_without_config_yaml(synthetic_cache, tmp_path):
    """The saved file alone has to rebuild the model, weights and all.

    Until 31.08.2026 the only ``torch.save`` in the project lived in
    ``bench_common.py``, so a plain ``train.py`` run left the finished model in
    RAM and nothing else. What makes the file usable is not the ``state_dict``
    -- it is the layout next to it: ``RecurrentField(**model_config)`` has to
    accept every key, and ``load_state_dict`` has to be happy with
    ``strict=True``. A layout key that drifts out of step with the constructor
    fails exactly here and nowhere else.
    """
    import torch
    from model import RecurrentField

    args = _tiny_args(synthetic_cache)
    model, bundle, _packed, dtn, hist = train_mod.fit(args)
    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    ckpt = torch.load(path, weights_only=False)
    rebuilt = RecurrentField(**ckpt["model_config"])
    rebuilt.load_state_dict(ckpt["model_state_dict"], strict=True)

    # de-normalisation has to travel too, or the weights predict a z-score
    # against a normalisation nothing recorded
    assert ckpt["bundle_stats"]["T_sigma"] == pytest.approx(bundle.T_sigma)
    assert ckpt["bundle_stats"]["T_mu"] == pytest.approx(bundle.T_mu)
    # and the run must be identifiable as synthetic after the fact
    assert ckpt["run"]["synthetic_cache"] is True


def test_held_out_op_uses_the_training_normalisation(synthetic_cache):
    """``build_op`` re-fits nothing -- that is what makes it out-of-sample.

    A held-out OP normalised against its own statistics would be an easier
    problem than the one the model was trained on, and the resulting MAE would
    not be a generalisation estimate at all.
    """
    bundle = data_mod.load_ops(op_ids=["OP01"], subsample_time=40)
    held = data_mod.build_op("OP02", bundle, subsample_time=40)

    # Tn is the held-out OP put through the TRAINING transform, exactly.
    expected = (held.T_lab - bundle.T_mu) / bundle.T_sigma
    assert np.allclose(held.Tn, expected, atol=1e-5)
    assert held.xn.shape == bundle.xn.shape


def test_trivial_baselines_are_what_evaluate_compares_against(synthetic_cache):
    """The bar printed next to a held-out MAE is measured on that OP."""
    bundle = data_mod.load_ops(op_ids=["OP01"], subsample_time=40)
    held = data_mod.build_op("OP02", bundle, subsample_time=40)

    persistence, mean = train_mod.trivial_baselines(held, bundle)

    assert persistence == pytest.approx(
        float(np.abs(held.T_lab - held.T_lab[0][None, :]).mean()))
    assert mean == pytest.approx(float(np.abs(held.T_lab - bundle.T_mu).mean()))
    assert persistence > 0 and mean > 0


# --------------------------------------------------------------------------
# device selection
# --------------------------------------------------------------------------

def test_ask_never_blocks_without_a_terminal(monkeypatch, capsys):
    """``--device ask`` must fall back, not hang, when nobody can answer.

    It is the default, so every CI job, every ``nohup``ed run and every piped
    invocation goes through this branch. A training script that blocks on a
    prompt no one can see is strictly worse than one that picks a default and
    says which.
    """
    import device_utils

    monkeypatch.setattr(device_utils.sys.stdin, "isatty", lambda: False)
    # input() must never be reached; make it loud if it is.
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("ask blocked"))

    dev = device_utils.resolve_device("ask")

    assert dev.type in {"cpu", "cuda"}
    assert "not an interactive terminal" in capsys.readouterr().out


def test_ask_accepts_an_index_or_the_spec(monkeypatch):
    """A prompt that only takes '2' is worse than one that also takes 'cuda:0'."""
    import device_utils

    monkeypatch.setattr(device_utils.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(device_utils, "available_devices",
                        lambda: [("cpu", "CPU"), ("cuda:0", "fake"), ("cuda:1", "fake")])

    monkeypatch.setattr("builtins.input", lambda *a: "3")
    assert device_utils._prompt_for_device() == "cuda:1"
    monkeypatch.setattr("builtins.input", lambda *a: "cuda:1")
    assert device_utils._prompt_for_device() == "cuda:1"
    monkeypatch.setattr("builtins.input", lambda *a: "")      # Enter = default
    assert device_utils._prompt_for_device() == "cuda:0"
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError))
    assert device_utils._prompt_for_device() == "cuda:0"


def test_an_unlisted_op_gets_a_label_not_an_exception():
    """A bundle outside the plan sheet must not cost a finished training run.

    The DATA path already accepts one: ``build_op`` reads any bundle that
    exists and detects which channels are profiles from the bundle itself,
    never from the table. Only the TIER is unknown, and ``evaluate`` runs after
    training -- raising there would throw away hours over a label.
    """
    import op_registry

    assert op_registry.tier_of("OP06") == op_registry.TIER_INTERP
    assert op_registry.tier_or_unknown("OP06") == op_registry.TIER_INTERP
    assert op_registry.tier_or_unknown("OP17") == op_registry.TIER_UNKNOWN
    # the strict lookup still raises, because the split checks depend on it
    with pytest.raises(KeyError):
        op_registry.tier_of("OP17")
