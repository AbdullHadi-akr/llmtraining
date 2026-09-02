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

import copy
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
import op_metrics as op_metrics_mod  # noqa: E402


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


def test_a_missing_measurement_op_skips_instead_of_aborting(synthetic_cache, capsys):
    """A measurement OP with no bundle must never cost a training run.

    OP17 and OP18 have not been simulated yet, so a config naming them is the
    normal case, not an error. They are a bonus report -- unlike ops/val_ops/
    test_ops, which the evaluation depends on and which fail fast. Getting this
    backwards would mean a GPU run refusing to start over a report it could
    simply have left out.
    """
    args = _tiny_args(synthetic_cache, measurement_ops=["OP17", "OP18"])
    model, bundle, ops, dtn, hist = train_mod.fit(args)

    # train() is what filters; call the same guard the way train() does.
    train_mod.require_ops(*args.ops)          # the hard ones still pass
    have = set(data_mod.available_ops())
    assert "OP17" not in have and "OP18" not in have
    # and nothing above raised, which is the whole claim
    assert hist["epoch"], "training produced no epochs"


def test_val_ops_still_fail_fast(synthetic_cache):
    """The other direction: a bad --val-ops must NOT be silently dropped.

    It is what the ranking is read off. Skipping it would leave a run that
    looks complete and reports nothing held out.
    """
    with pytest.raises(SystemExit):
        train_mod.require_ops("OP01", "OP99")


# --------------------------------------------------------------------------
# environment guard
# --------------------------------------------------------------------------

def test_env_check_names_the_interpreter_not_the_import(monkeypatch):
    """The message has to point at the venv, not at whichever import failed.

    The real failure was `ModuleNotFoundError: No module named 'pandas'` raised
    four imports deep in materials.py. Every word true, none of it the problem
    -- and the obvious reading ("pip install pandas") half-populates the system
    interpreter and makes the NEXT error worse. So the guard must name the
    interpreter in use and the activate command.
    """
    import env_check

    monkeypatch.setattr(env_check, "_REQUIRED", (("no_such_module_xyz", "no-such"),))
    monkeypatch.setattr(env_check.sys, "prefix", "/usr")
    monkeypatch.setattr(env_check.sys, "base_prefix", "/usr")   # not a venv

    with pytest.raises(SystemExit) as exc:
        env_check.require_training_env()

    msg = str(exc.value)
    assert "no_such_module_xyz" in msg
    assert env_check.sys.executable in msg      # WHICH python is running
    assert "activate" in msg                     # and what to do about it
    assert "Do NOT pip install" in msg


def test_env_check_is_silent_when_the_env_is_fine():
    """A no-op in a working environment -- it runs at every entry point."""
    import env_check
    env_check.require_training_env()             # must not raise


def test_env_check_imports_nothing_third_party():
    """It has to survive exactly the situation it diagnoses.

    A guard that itself needs numpy cannot report that numpy is missing.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "env_check.py").read_text()
    stdlib_only = {"__future__", "importlib", "importlib.util", "sys", "pathlib"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in stdlib_only, f"env_check imports {a.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module in stdlib_only, f"env_check imports {node.module}"


def test_history_csv_is_written_every_epoch(synthetic_cache, monkeypatch, tmp_path):
    """Hours of training must not depend on the process reaching the end.

    Before this, metrics.txt and the checkpoint were both written after the
    training loop returned, so a 60-epoch run killed at epoch 55 -- a closed
    terminal, a full disk, a stopped session -- left nothing behind at all.
    """
    monkeypatch.setattr(train_mod, "ART_DIR", tmp_path)

    args = _tiny_args(synthetic_cache)
    args.epochs = 3
    train_mod.fit(args)

    csv = tmp_path / "history.csv"
    assert csv.exists(), "history.csv was not written during training"
    rows = csv.read_text().strip().split("\n")
    assert rows[0].split(",") == list(train_mod.HISTORY_KEYS)
    assert len(rows) == 1 + 3, "one header plus one row per epoch"


def test_a_checkpoint_says_whether_the_run_finished(synthetic_cache, tmp_path):
    """A mid-run checkpoint must not be mistaken for a completed one."""
    import torch

    args = _tiny_args(synthetic_cache)
    args.epochs = 1
    model, bundle, _packed, dtn, hist = train_mod.fit(args)

    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)
    assert torch.load(path, weights_only=False)["run"]["complete"] is True

    # a run that stopped early carries the same key, saying so
    hist_short = dict(hist)
    hist_short["epoch"] = []
    train_mod.save_checkpoint(model, bundle, args, dtn, hist_short, path)
    assert torch.load(path, weights_only=False)["run"]["complete"] is False


# --------------------------------------------------------------------------
# physics sanity
# --------------------------------------------------------------------------

def test_q_dot_is_the_jr1_power_spread_over_the_jr1_volume(synthetic_cache):
    """The units error that made every Qsrc 121x too small, pinned.

    ``q_source[:, 0]`` is ``jr1_w`` and carries WATTS -- the bundle contract
    says so. ``Qsrc`` multiplies ``q_dot`` by a plain 0/1 region mask, so
    ``q_dot`` has to arrive as a VOLUMETRIC source in W/m^3, and the conversion
    is one division by the volume the heat sits in.

    Until 01.09.2026 the loader divided by ``V_JR1 * N_JR1_POINTS`` on a
    "Gleichverteilung" reading inherited from the base project. Spreading the
    total power over the 121 JR1 points is what a uniform source over V_JR1
    already does, so the point count was counted twice and the source came out
    121x short. Nothing failed: a uniform factor is divided straight back out by
    the EMA balancer and ``L_phys`` lands at O(1) either way. The statement that
    does see it is the one asserted here -- integrate the volumetric source over
    the volume and the total power has to come back.
    """
    op_id = "OP01"
    npz = np.load(Path(synthetic_cache) / f"{op_id}.npz", allow_pickle=True)
    jr1_w = np.asarray(npz["q_source"], dtype=np.float64)[:, 0]

    # subsample 1 with point sampling: no window averaging between the two, so
    # the round trip is exact rather than approximate.
    raw = data_mod._read_raw(op_id, 1, "point")

    recovered = raw["q_dot"] * data_mod.V_JR1          # W/m^3 * m^3 -> W
    assert recovered == pytest.approx(jr1_w, rel=1e-9), (
        "q_dot * V_JR1 must give back the cached watts; an extra factor here is "
        "a uniform error on every Qsrc and invisible everywhere else"
    )
    assert np.max(np.abs(recovered - jr1_w / data_mod.N_JR1_POINTS)) > 0, (
        "the JR1 point count must not appear in the source conversion"
    )


def test_energy_balance_report_compares_the_source_against_the_rise(synthetic_cache):
    """The check that would catch a units error in the heat column.

    ``Qsrc`` is built from ``q_dot`` with a 0/1 region mask, i.e. ``q_dot`` is
    treated as W/m^3, and the single division in ``_read_raw`` is what makes it
    so. An error in that one number is a UNIFORM factor on every ``Qsrc`` --
    exactly the kind that hides: the EMA balancer divides it straight back out
    and ``L_phys`` still lands at O(1). Only an energy argument sees it, and on
    31.08.2026 this report is what saw it (~147x on the real bundles).
    """
    bundle = data_mod.load_ops(op_ids=["OP01", "OP02"], subsample_time=40)
    lines = data_mod.energy_balance_report(bundle)

    assert any("energy balance" in l for l in lines)
    rows = [l for l in lines if "ratio=" in l]
    assert len(rows) == len(bundle.ops)
    assert all("<|dTn/dtn|>" in l and "<|Qsrc|>" in l for l in rows)


def test_coverage_report_speaks_up_for_a_dead_channel(synthetic_cache):
    """The channel that cannot be extrapolated to was the one nothing reported.

    ``soc_start`` is constant across all sixteen OPs, so ``config_active`` marks
    it dead and ``_normalise_config`` forces it to 0 no matter what a held-out
    OP carries. That makes a differing value the WORST case, not a harmless one:
    the network is handed an identical feature and never learns the difference
    exists. ``coverage_report`` used to skip dead channels outright, so the only
    channel where the envelope is a single point was also the only silent one.
    """
    # One OP is the smallest bundle in which a scalar channel has no variance
    # at all -- which is what soc_start is across the real sixteen.
    bundle = data_mod.load_ops(op_ids=["OP01"], subsample_time=40)
    idx = data_mod.CONFIG_ORDER.index("soc_start")
    assert not bundle.config_active[idx], "soc_start has to be dead here"

    op = bundle.ops[0]
    inside = data_mod.coverage_report(bundle, op)
    assert not any("soc_start" in l for l in inside), (
        "an OP that carries the trained value is not off the envelope"
    )

    moved = copy.deepcopy(op)
    moved.cfg_phys = np.array(moved.cfg_phys, dtype=np.float64, copy=True)
    moved.cfg_phys[:, idx] += 40.0
    lines = data_mod.coverage_report(bundle, moved)

    hits = [l for l in lines if "soc_start" in l]
    assert hits, "a dead channel moving off its single trained value has to be reported"
    assert "DEAD" in hits[0] and "invisible" in hits[0], (
        "and the line has to say why it is worse than an ordinary overshoot"
    )


def test_energy_balance_report_flags_a_shrunken_source(synthetic_cache):
    """And the detector has to fire, or the guard above guards nothing.

    Reintroducing the old divisor by hand -- ``Qsrc`` scaled down by the JR1
    point count -- has to move every ratio by exactly that factor and raise the
    ``[ENERGY]`` verdict. This is the failure mode the 121 was, reproduced
    without reverting the loader.

    The absolute ratio is not asserted: the synthetic field is not built from
    its own source (``make_synthetic_cache`` picks the rise and the watts
    independently), so only real bundles can be expected to balance. What has to
    hold on any bundle is the SENSITIVITY -- a source scaled by k moves every
    ratio by exactly k, which is what makes the report able to see a uniform
    factor at all.
    """
    bundle = data_mod.load_ops(op_ids=["OP01", "OP02"], subsample_time=40)

    def ratios(b):
        return [float(l.split("ratio=")[1].split("x")[0])
                for l in data_mod.energy_balance_report(b) if "ratio=" in l]

    before = ratios(bundle)
    for op in bundle.ops:
        op.Qsrc = op.Qsrc / data_mod.N_JR1_POINTS
    after = ratios(bundle)

    assert after == pytest.approx(
        [r * data_mod.N_JR1_POINTS for r in before], rel=1e-3)
    assert any("[ENERGY]" in l for l in data_mod.energy_balance_report(bundle))


def test_cfl_check_looks_at_the_physics_stencil_not_only_the_data_step(capsys):
    """δ is its own knob and was never checked -- that is how it stayed at 4x.

    ``_check_cfl_stability`` used to be handed only the data grid step. The BDF
    stencil in ``physics.py`` uses ``model.delta``, which does NOT follow
    ``--subsample``: a fine data grid printed "CFL OK" while the residual's time
    derivative was taken over a lag four times the limit.
    """
    from types import SimpleNamespace

    bundle = SimpleNamespace(
        Fo=np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]]),
        L_ref=0.0768, T_span_ref=1604.0,
        xn=np.array([[0.0, 0, 0], [1.0, 1.0, 1.0]]),
    )
    train_mod._check_cfl_stability(bundle, 1e-6, None, phys_delta_s=1e9)
    out = capsys.readouterr().out
    assert "CFL OK" in out, "the data step alone should still read OK"
    assert "PHYSICS stencil" in out, "the physics lag must be flagged separately"
    assert "--delta-phys" in out, "and the message must name the knob"


def test_delta_phys_reaches_the_model(synthetic_cache):
    """The knob has to change the BDF lag, not just parse."""
    args = _tiny_args(synthetic_cache, delta_phys=0.25)
    model, bundle, _packed, _dtn, _hist = train_mod.fit(args)
    assert float(model.delta) == pytest.approx(0.25 / bundle.T_span_ref, rel=1e-6)


# --------------------------------------------------------------------------
# the loss balancer (O15) -- and why the divisor is watched, not just recorded
# --------------------------------------------------------------------------

def _feed(balancer, key, values):
    """Push a series through the balancer, returning the divisor it handed out."""
    out = []
    for v in values:
        out.append(balancer.divisor(key, v))
        balancer.end_step()
    return out


def _balancer(decay_per_epoch=0.9, steps_per_epoch=100):
    return train_mod._LossBalancer(
        mode="ema", decay=decay_per_epoch ** (1.0 / steps_per_epoch),
        warmup_steps=1, phys_norm=0.0, bc_norm=0.0, data_floor=1e-12)


def test_the_divisor_converges_on_a_constant_loss():
    """The EMA has to move towards the value it is fed, monotonically.

    This is the mechanism O15 turns on, so it is worth holding separately from
    the outcome: if the update itself ever breaks, every ratio in history.csv
    becomes unreadable and nothing else in the suite would notice.
    """
    bal = _balancer()
    # 100 epochs at 100 steps: 0.9**100 = 2.7e-5, so a factor-100 anchor is fully
    # unwound and what is left is the value being fed. That it takes that long is
    # the subject of the next test.
    seen = _feed(bal, "phys", [100.0] + [1.0] * (100 * 100))

    assert seen[0] == pytest.approx(100.0)      # first call anchors on the value
    assert seen[-1] < seen[len(seen) // 2] < seen[1]   # and then only falls
    assert seen[-1] == pytest.approx(1.0, rel=0.05)


@pytest.mark.xfail(strict=True, reason=(
    "O15: the EMA is anchored on the FIRST optimiser step, whose loss is ~325x "
    "the epoch-1 mean, and at ema_decay 0.9/epoch unwinding that anchor alone "
    "takes ~10*ln(325) = 58 epochs. Step 6 ran 60, so div_* was still 960-10000x "
    "above its own term and all three divisors fell at exactly 0.9000/epoch. "
    "See FAHRPLAN.md §11.6. When this XPASSes, O15 is fixed: drop the marker and "
    "close the point."))
def test_the_divisor_tracks_its_loss_within_one_run():
    """After a realistic run length, div_* must be an estimate of L_*.

    Not a style point: ``w_data:w_phys:w_bc`` is documented as a ratio between
    TERMS (config.yaml, "loss balancing"), and that claim holds only while each
    divisor is close to the term it divides. While it is not, the weights mean
    something nobody chose, and a weight sweep measures the offset rather than
    the physics -- which is why O15 blocks O6.

    The first value is the spike a randomly initialised net produces on step 1;
    the rest is a converged loss. 60 epochs at 100 steps each is Step 6.
    """
    bal = _balancer()
    seen = _feed(bal, "phys", [3.3e4] + [0.1] * (60 * 100))

    assert seen[-1] == pytest.approx(0.1, rel=10.0)


def test_analyse_history_flags_a_divisor_in_free_fall():
    """The reporting side of O15, on a CSV that reproduces the pattern.

    The tool is the thing that would have caught O15 on day one, so it gets a
    fixture that decays exactly like Step 6 did and one that does not.
    """
    import analyse_history as ah

    # The losses fall FASTER than 0.9/epoch -- that is the whole mechanism: an
    # EMA at 0.9 cannot keep up, so the gap widens instead of closing.
    def rows_for(gap_of_epoch):
        out = []
        for e in range(1, 61):
            ld, lp, lb = 0.5 * 0.85 ** e, 5e5 * 0.85 ** e, 5e-9 * 0.85 ** e
            g = gap_of_epoch(e)
            out.append({"epoch": float(e), "L_data": ld, "L_phys": lp, "L_bc": lb,
                        "L_phys_bal": 1e-3, "L_bc_bal": 1e-4,
                        "ratio_phys": 0.5, "ratio_bc": 0.05,
                        "div_data": ld * g, "div_phys": lp * g, "div_bc": lb * g,
                        "spread_space": 1.1, "spread_time": 0.92,
                        "delta": 6.23e-4})
        return out

    # Step 6: divisors at 0.9/epoch against losses at 0.85 -> the gap grows.
    stale = ah.analyse(rows_for(lambda e: 300.0 * (0.9 / 0.85) ** e))[1]
    assert any(n.startswith("[O15]") and "div_phys" in n for n in stale)
    assert any("RATIO is frozen" in n for n in stale)

    # a divisor that sits on its term raises nothing -- not even the frozen-ratio
    # note, although the three rates agree there too
    healthy = ah.analyse(rows_for(lambda e: 1.0))[1]
    assert not [n for n in healthy if n.startswith("[O15]")]


def test_analyse_history_calls_out_the_last_line_but_not_a_trend():
    """The O12 mistake, and the case that must NOT be flagged as one.

    ``ratio_bc`` scatters, so its final epoch is a sample; ``div_*`` falls
    monotonically, so its final epoch is the state. Flagging both would bury the
    first under the second -- which is why the monotone exemption exists.
    """
    import analyse_history as ah

    noisy = [0.06, 0.02, 0.11, 0.04, 0.09, 0.03, 0.10, 0.05, 0.08, 0.0178]
    rows = [{"epoch": float(i + 1), "ratio_bc": v, "div_phys": 1e9 * 0.9 ** i,
             "L_phys": 5e4, "spread_time": 0.9, "spread_space": 1.1}
            for i, v in enumerate(noisy * 3)]

    notes = ah.analyse(rows, last=30)[1]
    assert any(n.startswith("[LAST-LINE]") and "ratio_bc" in n for n in notes)
    assert not [n for n in notes if "[LAST-LINE]" in n and "div_phys" in n]


# --------------------------------------------------------------------------
# the signed late error (O13)
# --------------------------------------------------------------------------

def _ramp_op(n_t=100, n_p=3, split_t=80):
    true = np.linspace(20.0, 50.0, n_t)[:, None] * np.ones((1, n_p))
    return SimpleNamespace(T_lab=true, n_t=n_t, split_t=split_t,
                           transient=np.zeros(n_t, bool))


def test_a_late_offset_reads_as_drift_not_scatter():
    """Same |error|, opposite diagnosis -- which is the point of the metric.

    ``late_mae`` is 2.0 C in both halves of this test. Only the sign separates a
    rollout whose level has walked away (O13, mechanism 1) from one that is
    merely imprecise in a hard regime, and they call for opposite responses.
    """
    op = _ramp_op()
    drift = op.T_lab.copy()
    drift[80:] += 2.0                       # consistently too warm, late
    m = op_metrics_mod.op_metrics(drift, op, late_is_holdout=True)

    assert m["late_mae"] == pytest.approx(2.0)
    assert m["late_bias"] == pytest.approx(+2.0)
    assert m["late_bias_frac"] == pytest.approx(1.0)
    assert m["bias_end"] > m["bias_early"] == pytest.approx(0.0)

    scatter = op.T_lab.copy()
    scatter[80::2] += 2.0                   # same size, no direction
    scatter[81::2] -= 2.0
    m2 = op_metrics_mod.op_metrics(scatter, op, late_is_holdout=True)

    assert m2["late_mae"] == pytest.approx(2.0)
    assert abs(m2["late_bias"]) < 1e-9
    assert m2["late_bias_frac"] == pytest.approx(0.0)


def test_a_cold_drift_keeps_its_sign():
    """+ is too warm, - is too cold. A metric that lost that would be an MAE."""
    op = _ramp_op()
    cold = op.T_lab.copy()
    cold[80:] -= 3.5
    m = op_metrics_mod.op_metrics(cold, op, late_is_holdout=True)

    assert m["late_bias"] == pytest.approx(-3.5)
    assert m["late_bias_frac"] == pytest.approx(1.0)   # magnitude only


def test_the_bias_fraction_cannot_leave_the_unit_interval():
    """|mean(x)| <= mean(|x|) is what makes the number readable as a fraction.

    Asserted rather than assumed because every report prints it as a percentage,
    and a value above 1 there would be read as a bug in the model rather than in
    the metric.
    """
    rng = np.random.default_rng(0)
    op = _ramp_op()
    for scale in (0.01, 1.0, 25.0):
        pred = op.T_lab + rng.normal(0.0, scale, op.T_lab.shape)
        m = op_metrics_mod.op_metrics(pred, op, late_is_holdout=False)
        assert 0.0 <= m["late_bias_frac"] <= 1.0


def test_every_op_gets_the_signed_metrics(synthetic_cache):
    """A real rollout, so the keys cannot drift away from what evaluate prints.

    ``_report_op`` formats these for every OP in every role; a key renamed here
    and not there raises a KeyError only once a full run reaches the report,
    which is two hours in.
    """
    args = _tiny_args(synthetic_cache)
    model, bundle, _packed, _dtn, _hist = train_mod.fit(args)
    device = next(model.parameters()).device
    op = bundle.ops[0]
    pred = op_metrics_mod.rollout_phys(model, op, bundle, device)
    m = op_metrics_mod.op_metrics(pred, op, late_is_holdout=False)

    for key in ("bias_early", "bias_mid", "bias_end", "late_bias", "late_bias_frac"):
        assert key in m, key
    # and the formatter has to survive them
    assert "late_bias" in op_metrics_mod.format_op_metrics(op.op_id, "T0-in-time", m)


# --------------------------------------------------------------------------
# sweep.py -- the seed loop
# --------------------------------------------------------------------------

def test_sweep_runs_an_axis_and_records_one_row_per_seed(synthetic_cache, tmp_path):
    """Two values, two seeds, four rows -- and the CSV written as they finish.

    The fixture is tiny (1 epoch, width 8), so this is a wiring test, not a
    measurement: what it holds is that the loop varies the axis, varies the
    seed, scores the HELD-OUT OP and writes a row for every combination.
    """
    import sweep as sweep_mod

    out = tmp_path / "sweep.csv"
    argv = ["--axis", "w_phys", "--values", "0", "0.1", "--seeds", "0", "1",
            "--out", str(out),
            "--ops", "OP01", "--val-ops", "OP02", "--test-ops",
            "--epochs", "1", "--subsample", "40", "--inner-steps", "1",
            "--batch-data", "64", "--batch-phys", "16", "--batch-bc", "8",
            "--width", "8", "--depth", "2", "--device", "cpu",
            "--no-residual-output"]
    assert sweep_mod.main(argv) == 0

    import csv as csv_mod
    rows = list(csv_mod.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(rows) == 4
    assert {r["value"] for r in rows} == {"0", "0.1"}
    assert {r["seed"] for r in rows} == {"0", "1"}
    # the held-out OP is scored, the training OP is not
    assert all(r["mae_OP02"] for r in rows)
    assert "mae_OP01" not in rows[0]


def test_sweep_skips_rows_it_already_has(synthetic_cache, tmp_path, capsys):
    """A crash at hour nine must cost one run, not nine.

    Six 60-epoch runs are twelve hours. Appending on completion and skipping
    what is already recorded is the difference between restarting and starting
    over -- and it is the only bookkeeping this file is allowed to carry.
    """
    import sweep as sweep_mod

    out = tmp_path / "sweep.csv"
    base = ["--axis", "w_phys", "--values", "0", "--seeds", "0",
            "--out", str(out),
            "--ops", "OP01", "--val-ops", "OP02", "--test-ops",
            "--epochs", "1", "--subsample", "40", "--inner-steps", "1",
            "--batch-data", "64", "--batch-phys", "16", "--batch-bc", "8",
            "--width", "8", "--depth", "2", "--device", "cpu",
            "--no-residual-output"]
    sweep_mod.main(base)
    capsys.readouterr()

    sweep_mod.main(base)                      # same call again
    assert "0 to go" in capsys.readouterr().out

    import csv as csv_mod
    rows = list(csv_mod.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(rows) == 1, "the second call must not duplicate the row"


def test_sweep_refuses_to_score_on_training_ops(synthetic_cache, tmp_path):
    """Without --val-ops there is nothing to select on.

    Scoring a sweep on the OPs it trained on ranks memorisation. Failing fast
    is the point: the alternative is twelve hours producing a table that means
    nothing.
    """
    import sweep as sweep_mod

    with pytest.raises(SystemExit, match="nothing to select on"):
        sweep_mod.main(["--axis", "w_phys", "--values", "0", "--seeds", "0",
                        "--out", str(tmp_path / "s.csv"),
                        "--ops", "OP01", "--val-ops", "--test-ops",
                        "--epochs", "1", "--device", "cpu"])


def test_sweep_rejects_an_axis_that_is_not_a_flag(synthetic_cache, tmp_path):
    """A typo in --axis must not silently sweep nothing.

    ``setattr`` on an argparse namespace accepts any name, so an unchecked axis
    would run the whole grid at the default value and report a clean, wrong
    'no difference between configurations'.
    """
    import sweep as sweep_mod

    with pytest.raises(SystemExit, match="not a train.py argument"):
        sweep_mod.main(["--axis", "w_phsy", "--values", "0", "--seeds", "0",
                        "--out", str(tmp_path / "s.csv"),
                        "--ops", "OP01", "--val-ops", "OP02", "--test-ops",
                        "--epochs", "1", "--device", "cpu"])


# --------------------------------------------------------------------------
# evaluate.py -- the load half of save_checkpoint
# --------------------------------------------------------------------------

def test_evaluate_scores_a_checkpoint_without_training(synthetic_cache, tmp_path,
                                                       capsys):
    """Train once, save, and score from the file alone.

    The claim is narrow and load-bearing: the bundle evaluate.py rebuilds has to
    be the one the weights were trained under. It reads train_frac, resample and
    the rate lags from the checkpoint's ``preprocessing`` section -- they are not
    in ``run``, and taking them from the wrong place would quietly fall back to
    config.yaml and score the model against a normalisation it never saw.
    """
    import evaluate as eval_mod

    args = _tiny_args(synthetic_cache, val_ops=["OP02"])
    model, bundle, _ops, dtn, hist = train_mod.fit(args)
    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    assert eval_mod.main([str(path), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "OP02" in out and "late_bias" in out
    # and the configuration travels with the weights, so the printed numbers can
    # be attributed to a configuration without a log file next to them
    assert f"w_phys={float(args.w_phys)}" in out


def test_evaluate_rebuilds_the_runs_own_normalisation(synthetic_cache, tmp_path):
    """The reloaded bundle matches the trained one, not config.yaml's defaults.

    T_sigma is the number every MAE is expressed in. A checkpoint scored against
    re-fitted statistics would be an easier problem than the one the model
    solved, and its MAE would not be comparable to the run it came from.
    """
    import evaluate as eval_mod

    args = _tiny_args(synthetic_cache, val_ops=["OP02"], train_frac=0.6)
    model, bundle, _ops, dtn, hist = train_mod.fit(args)
    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    _m, reloaded, run, _ckpt = eval_mod.load_checkpoint(path, "cpu")

    assert reloaded.T_sigma == pytest.approx(bundle.T_sigma)
    assert reloaded.T_mu == pytest.approx(bundle.T_mu)
    assert reloaded.ops[0].split_t == bundle.ops[0].split_t   # train_frac travelled
    assert run["ops"] == list(args.ops)


def test_evaluate_scores_an_unlisted_op_without_crashing(synthetic_cache, tmp_path,
                                                         capsys):
    """OP19 is not in the OP01-OP16 plan sheet, and tier_of raises for it.

    The measurement OP is exactly the one somebody will point this at, so the
    label has to degrade to 'unlisted' instead of ending the run.
    """
    import evaluate as eval_mod
    import op_registry

    args = _tiny_args(synthetic_cache, val_ops=["OP02"])
    model, bundle, _ops, dtn, hist = train_mod.fit(args)
    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    with pytest.raises(Exception):
        op_registry.tier_of("OP19")          # the raising one
    assert op_registry.tier_or_unknown("OP19") == op_registry.TIER_UNKNOWN

    assert eval_mod.main([str(path), "--ops", "OP02", "--device", "cpu"]) == 0
    assert "OP02" in capsys.readouterr().out


def test_the_checkpoint_records_the_loss_weights(synthetic_cache, tmp_path):
    """Which weights produced these numbers is part of the result.

    Until 02.09.2026 the file recorded the architecture, the normalisation and
    the OP list but not w_data/w_phys/w_bc, so model.pt could not say whether it
    came from w_phys 0 or 0.1 -- while that was precisely the open question
    (FAHRPLAN §11.8). A result whose configuration is only in a log file next to
    it is a result that gets mislabelled the first time the log is lost.
    """
    args = _tiny_args(synthetic_cache, w_phys=0.25, w_bc=0.0)
    model, bundle, _ops, dtn, hist = train_mod.fit(args)
    path = tmp_path / "model.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    import torch
    loss = torch.load(path, weights_only=False)["loss"]
    assert loss["w_phys"] == pytest.approx(0.25)
    assert loss["w_bc"] == pytest.approx(0.0)
    assert loss["loss_balance"] == args.loss_balance


def test_evaluate_says_so_when_the_weights_are_missing(synthetic_cache, tmp_path,
                                                       capsys):
    """An old checkpoint must not read as w_phys = 0.

    model_schritt6.pt predates the section, and a printed ``None`` next to
    ``w_phys`` is one glance away from being read as a measured zero -- which
    would invert the conclusion it is being used to check.
    """
    import evaluate as eval_mod
    import torch

    args = _tiny_args(synthetic_cache, val_ops=["OP02"])
    model, bundle, _ops, dtn, hist = train_mod.fit(args)
    path = tmp_path / "old.pt"
    train_mod.save_checkpoint(model, bundle, args, dtn, hist, path)

    ckpt = torch.load(path, weights_only=False)
    del ckpt["loss"]                       # an August file
    torch.save(ckpt, path)

    assert eval_mod.main([str(path), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "NOT RECORDED" in out
    assert "w_phys=None" not in out
