#!/usr/bin/env python3
"""Score a saved checkpoint on any OP, without training anything.

Why this exists
---------------
``save_checkpoint`` has written ``artifacts/model.pt`` since 31.08., and
``test_checkpoint_round_trips_without_config_yaml`` proves the file alone
rebuilds the model. Nothing ever loaded it back outside that test, so every
question put to a trained model -- "does the run WITH physics drift the same
way?", "what does this OP look like under that configuration?" -- cost a fresh
two-hour run. It costs a rollout now.

The immediate use is O13. The signed late-error metrics are younger than Step 6,
so the drift verdict in FAHRPLAN §11.9 rests on the physics-free run alone.
``model_schritt6.pt`` still holds the other half of that comparison.

Usage
-----
    python3 PINNmodulusTwo/evaluate.py artifacts/model_schritt6.pt
    python3 PINNmodulusTwo/evaluate.py artifacts/model.pt --ops OP06 OP09 OP13

What it does NOT do
-------------------
Re-fit anything. The bundle is rebuilt with the checkpoint's own subsample and
train_frac, so the normalisation is the one the weights were trained under. A
checkpoint scored against re-fitted statistics would be an easier problem than
the one the model actually solved, and its MAE would not be comparable to the
run it came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from data import build_op, load_ops  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from model import RecurrentField  # noqa: E402
from op_metrics import format_op_metrics, op_metrics, rollout_phys  # noqa: E402
from op_registry import tier_or_unknown  # noqa: E402


def load_checkpoint(path: Path, device):
    """Rebuild model and bundle from the file, using the run's own settings."""
    ckpt = torch.load(path, weights_only=False)
    run = ckpt.get("run", {})
    # The preprocessing is part of the model in everything but name -- the
    # checkpoint says so itself -- and it lives in its own section, not in
    # ``run``. Reading it from the wrong place would silently rebuild the
    # bundle with config.yaml's defaults instead of the run's own settings.
    pre = ckpt.get("preprocessing", {})

    model = RecurrentField(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    bundle = load_ops(
        op_ids=list(run.get("ops", [])),
        subsample_time=int(run.get("subsample", 2)),
        train_frac=float(pre.get("train_frac", 0.8)),
        resample=str(pre.get("resample", "mean")),
        driver_rate_lags=[float(v) for v in pre.get("driver_rate_lags", [])],
        use_driver_history=bool(pre.get("use_driver_history", True)),
    )
    return model, bundle, run, ckpt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to a model.pt written by train.py")
    ap.add_argument("--ops", nargs="*", default=None,
                    help="OPs to score (default: the run's val + test OPs)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"{path}: not found")

    device = resolve_device(args.device)
    model, bundle, run, ckpt = load_checkpoint(path, device)

    trained_on = list(run.get("ops", []))
    want = args.ops if args.ops is not None else (
        list(run.get("val_ops", [])) + list(run.get("test_ops", [])))
    if not want:
        raise SystemExit("the checkpoint names no val/test OPs; pass --ops explicitly")

    cfg = ckpt.get("model_config", {})
    print(f"{path}")
    print(f"  seed={run.get('seed')}  epochs_run={run.get('epochs_run')}"
          f"/{run.get('epochs')}  complete={run.get('complete')}  "
          f"aborted={run.get('aborted')}")
    print(f"  delta_phys={cfg.get('delta_seconds')}s  delta_grid={cfg.get('delta_grid')}  "
          f"width={cfg.get('layer_size')} depth={cfg.get('num_layers')}  "
          f"subsample={run.get('subsample')}")
    # Checkpoints written before 02.09.2026 carry no "loss" section, so this
    # file cannot say whether they came from w_phys 0 or 0.1 -- the question the
    # project was open on. Say that outright rather than printing a None that
    # reads like a measured zero.
    loss = ckpt.get("loss")
    if loss:
        print(f"  w_data={loss.get('w_data')} w_phys={loss.get('w_phys')} "
              f"w_bc={loss.get('w_bc')}  balance={loss.get('loss_balance')}"
              f"/{loss.get('ema_decay')}  time_deriv={loss.get('time_deriv')}")
    else:
        print("  loss weights: NOT RECORDED (checkpoint written before 02.09.2026) "
              "-- take them from the run's log")
    print(f"  synthetic_cache={run.get('synthetic_cache')}")
    print(f"  scoring {want} against the training normalisation of {trained_on}")
    print()

    for op_id in want:
        # An OP the run trained on is in-sample here; saying so is the whole
        # reason late_is_holdout exists, and a report that guessed would put an
        # in-sample number next to a held-out one with no way to tell.
        in_train = op_id in trained_on
        op_data = build_op(op_id, bundle, subsample_time=int(run.get("subsample", 2)))
        pred = rollout_phys(model, op_data, bundle, device)
        m = op_metrics(pred, op_data, late_is_holdout=not in_train)
        print(format_op_metrics(op_id, tier_or_unknown(op_id), m), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
