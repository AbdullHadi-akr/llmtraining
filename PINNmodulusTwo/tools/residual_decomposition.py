#!/usr/bin/env python3
"""Split L_phys into its terms, and say which of them carries the 1/delta^2.

Why this exists
---------------
Achse 1 (FAHRPLAN 11.9) measured ``L_phys`` = 4.6e4 / 2.5e5 / 1.0e6 for
delta = 1.0 / 0.4 / 0.2 s. Fitting ``L = A + B / delta^2`` to those three
numbers leaves at most 5 % residual, and the 1/delta^2 part is 91 / 98 / 99.6 %
of ``L_phys``. The FAHRPLAN reads that as O18, the missing ``(grad lambda) .
(grad T)`` term. **Algebraically it cannot be.** ``heat_residual`` assembles

    residual = (3 T - 4 T_1 + T_2) / (2 delta)  -  aniso  -  Qsrc

and a spatial error lives in ``aniso``: it is added to the residual, it is not
divided by delta, so it lands in ``A``. Only something INSIDE the BDF numerator
that does not shrink with delta can scale like 1/delta^2.

There is one obvious candidate. ``T`` is the LIVE network at ``t``; ``T_1`` and
``T_2`` are read from the FROZEN rollout buffer the epoch started with
(``train.py``: one rollout per OP per epoch, then ``--inner-steps`` updates
against it). Right after the rollout the live network reproduces the buffer
exactly at grid times, so ``T - T_buf(t)`` is zero. Every data step then pulls
the live one-step prediction towards the LABEL at ``t`` while the buffer stays
on the free-running trajectory -- which in Achse 0/1 sits several C off the
labels (O17). That jump is delta-independent, it enters the numerator with a
factor 3, and it is divided by ``2 delta``.

What the tool measures
----------------------
For a trained checkpoint and the cached OPs it rebuilds the frozen rollout and
evaluates ``heat_residual`` (through ``return_parts``, so there is no second
copy of the assembly) at many ``(t, point)`` samples, for several delta on the
SAME weights, in two states:

``fresh``  the rollout the current weights produce -- no jump by construction.
``stale``  the same rollout, after ``--stale-steps`` data-term updates against
           it, averaged over the way there -- the state ``L_phys`` is logged in.

It also checks whether the autograd Laplacian sees the field at all
(:func:`laplacian_visibility`, ``[BLIND]``) and how rough the fresh rollout is
from step to step (:func:`roughness`).

For each state and delta it prints ``L_phys``, the share of the jump term, the
share of everything else, the RMS of the jump in degrees C, and ``L`` split by
x-plane (the three x-planes are the three materials, so an interface error in
x would show up as one plane dominating). Then it fits ``A + B / delta^2`` to
both columns and names what it sees -- several of these can hold at once:

``[STALE]``           in the stale state the jump is at least half of L_phys.
                      The 1/delta^2 in the Achse-1 numbers is then the training
                      loop, not the heat equation -- and not O18.
``[JITTER]``          the fresh rollout itself is not smooth on the delta
                      scale (``roughness``: its one-step second difference
                      against the labels'). Also 1/delta^2, also not O18.
``[BLIND]``           the autograd Laplacian sees under 10 % of the curvature
                      the field has (``laplacian_visibility``). Then the
                      conduction term -- the only place O16/O18 could act --
                      is taken of almost nothing.
``[NO 1/delta^2]``    the stale column does not scale. Then Achse 1 is not
                      reproduced on these weights and the question stays open.

The delta-independent ``A`` -- in both states -- is where O16/O18 can live, and
its per-plane split is printed next to it.

The decomposition always measures the BUFFER stencil -- the ``L_phys`` every run
up to 23.09.2026 logged. ``--phys-stencil live`` (train.py) removes the jump
term by construction, so a checkpoint trained with it still shows ``[STALE]``
here: that says what the buffer stencil WOULD log on those weights, not what the
run logged. ``tests/test_residual_decomposition.py`` holds the cancellation.

Usage
-----
On the T4 instance, venv active and the MPS daemon up (``-j`` is four processes
on ONE card; without MPS the driver time-slices them, measured 3.69x -> 1.46x)::

    nvidia-cuda-mps-control -d
    python PINNmodulusTwo/tools/residual_decomposition.py \
        artifacts/achse1/*/model.pt artifacts/achse0/*/model.pt -j 4

Every checkpoint gets ``residual_decomposition.txt`` (the full report) and
``residual_decomposition.json`` (the summary) next to it; the last lines are one
table over all of them. One checkpoint alone runs in-process and prints its
report directly. ``--device auto`` takes the card when one answers.

CPU works too (one checkpoint on the real data: minutes, not seconds). The
checkpoint carries its OPs, subsample and preprocessing; the bundle is rebuilt
from the cache with exactly those, and a normalisation that does not match the
checkpoint's is refused rather than silently used.

The stale state is an EMULATION: data term only, Adam started fresh (warmed up
on ``--warmup-steps`` before the measured rollout, so its first bias-corrected
steps do not inflate the drift), gradient clipped like ``train.py``. It is the
mechanism, not a replay of one specific epoch.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # model.py, physics.py, data.py
sys.path.insert(0, str(HERE))                 # _modulus_stub.py

import numpy as np  # noqa: E402

try:  # the real Modulus when it is there, the faithful stub otherwise
    import modulus  # noqa: F401
except ModuleNotFoundError:
    import _modulus_stub
    _modulus_stub.install(faithful=True)

import torch  # noqa: E402

from model import RecurrentField, interp_history, rollout  # noqa: E402
from physics import heat_residual  # noqa: E402


# --------------------------------------------------------------------------
# the measurement, importable (tests/test_residual_decomposition.py)
# --------------------------------------------------------------------------

def _plane_ids(xn: torch.Tensor) -> torch.Tensor:
    """Index of the x-plane every grid point sits on (0 = the lowest x)."""
    xs = torch.unique(torch.round(xn[:, 0] * 1e9) / 1e9)
    return torch.bucketize(torch.round(xn[:, 0] * 1e9) / 1e9, xs)


def decompose(model: RecurrentField, op: dict, buf: torch.Tensor,
              delta_n: float, phys_scale: float, *, n_samples: int,
              gen: torch.Generator, time_deriv: str = "bdf2",
              t_min: int = 1) -> dict:
    """``heat_residual`` at random ``(t, point)``, split into its terms.

    ``delta_n`` is the BDF lag in NORMALISED time and is set on the model for
    the duration of the call. ``t_min = 1`` skips row 0 on purpose: that row is
    the imposed initial condition, never a prediction, so the live network and
    the buffer differ there by the whole first step. It is measured on its own
    by :func:`row0_jump` rather than mixed in.

    Returns mean squares (so they compare directly with ``L_phys``) of the
    residual, of the jump term, of the residual without it, of the other terms,
    and ``L`` per x-plane.
    """
    if time_deriv not in ("bdf1", "bdf2"):
        raise ValueError("the jump is a property of the BDF stencil; "
                         f"time_deriv={time_deriv!r} has none")
    n_t, n_pts = int(op["n_t"]), int(op["n_points"])
    device = buf.device
    old = model._delta.clone()
    model._delta.fill_(float(delta_n))
    try:
        pt = torch.randint(t_min, n_t, (n_samples,), generator=gen).to(device)
        pp = torch.randint(0, n_pts, (n_samples,), generator=gen).to(device)
        tq = op["tn"][pt]
        res, parts = heat_residual(
            model, op["xn"], op["static"], op["cfg"][pt], op["forcing"][pt],
            op["Fo"], op["Qsrc"][pt, pp], buf, float(op["dtn"]), tq, pp,
            phys_scale, time_deriv=time_deriv, return_parts=True,
        )
        # What the buffer itself holds at t. At a grid time this is exactly the
        # row the rollout wrote, so ``T - T_buf`` is the live network's
        # departure from its own frozen trajectory.
        T_buf = interp_history(buf, float(op["dtn"]), tq, pp)
        T = parts["T"]
        div = parts["divisor"]
        if time_deriv == "bdf2":
            jump = 3.0 * (T - T_buf) / (2.0 * model.delta + 1e-8) / div
        else:
            jump = (T - T_buf) / (model.delta + 1e-8) / div
        dg, sg = parts["diff_gain"], parts["src_gain"]
        terms = {
            "aniso_xx": dg * parts["aniso_xx"] / div,
            "aniso_yy": dg * parts["aniso_yy"] / div,
            "aniso_zz": dg * parts["aniso_zz"] / div,
            "aniso_cross": dg * parts["aniso_cross"] / div,
            "Qsrc": sg * parts["Qsrc"] / div,
            "dTdt_smooth": parts["dTdt"] / div - jump,
        }
        res = res.detach()
        jump = jump.detach()
        plane = _plane_ids(op["xn"])[pp]
        out = {
            "L": float(torch.mean(res ** 2)),
            "L_jump": float(torch.mean(jump ** 2)),
            "L_without_jump": float(torch.mean((res - jump) ** 2)),
            "jump_rms_z": float(torch.sqrt(torch.mean((T - T_buf).detach() ** 2))),
            "terms": {k: float(torch.mean(v.detach() ** 2)) for k, v in terms.items()},
            "L_by_plane": {int(k): float(torch.mean(res[plane == k] ** 2))
                           for k in torch.unique(plane).tolist()},
        }
        return out
    finally:
        model._delta.copy_(old)


def row0_jump(model: RecurrentField, op: dict, buf: torch.Tensor) -> float:
    """RMS of ``T_live(0) - IC`` in z-units: the first step, which training's
    physics sampling includes (``pt`` from 0) and :func:`decompose` does not."""
    with torch.no_grad():
        P = int(op["n_points"])
        p = torch.arange(P, device=buf.device)
        tq = op["tn"][0].expand(P)
        hist = model._history(buf, float(op["dtn"]), tq, p)
        T = model.field(op["xn"], op["static"], op["cfg"][0].expand(P, -1),
                        op["forcing"][0].expand(P, -1), hist,
                        model.level(buf, float(op["dtn"]), tq))
        return float(torch.sqrt(torch.mean((T - buf[0]) ** 2)))


def roughness(buf: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """RMS of the one-step second difference of the rollout, and its ratio to
    the labels' over the same rows: ``(rms_z, ratio)``.

    A rollout that alternates by ``e`` from step to step puts ``~5 e`` into the
    BDF2 numerator at ANY delta (three rows of it), so it scales ``L_phys`` like
    1/delta^2 without any staleness. This is the direct measure of that: a
    ratio near 1 is a trajectory as smooth as the data, far above 1 is jitter.
    """
    n = min(buf.shape[0], labels.shape[0])
    d2b = buf[2:n] - 2.0 * buf[1:n - 1] + buf[:n - 2]
    d2l = labels[2:n] - 2.0 * labels[1:n - 1] + labels[:n - 2]
    rb = float(torch.sqrt(torch.mean(d2b ** 2)))
    rl = float(torch.sqrt(torch.mean(d2l ** 2)))
    return rb, (rb / rl if rl > 0 else float("inf"))


def _fd_second(values: torch.Tensor, coord: torch.Tensor,
               line_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Second derivative along one axis by the non-equidistant 3-point stencil.

    ``line_key`` groups the points into lines along that axis (all other
    coordinates equal). Returns ``(d2, index)`` for the INTERIOR points of every
    line -- the ends have no stencil and are left out rather than padded.
    """
    d2, idx = [], []
    for key in torch.unique(line_key):
        members = torch.where(line_key == key)[0]
        if members.numel() < 3:
            continue
        order = members[torch.argsort(coord[members])]
        c, f = coord[order], values[order]
        hm, hp = c[1:-1] - c[:-2], c[2:] - c[1:-1]
        d2.append(2.0 * ((f[2:] - f[1:-1]) / hp - (f[1:-1] - f[:-2]) / hm) / (hm + hp))
        idx.append(order[1:-1])
    return torch.cat(d2), torch.cat(idx)


def laplacian_visibility(model: RecurrentField, op: dict, buf: torch.Tensor,
                         rows, phys_scale: float) -> dict:
    """Does the autograd Laplacian see the curvature the field actually has?

    ``heat_residual`` differentiates the network with respect to ``xn`` and
    nothing else. The history channels -- in the default hybrid mode the
    per-point anchor ``T(t - delta_grid)`` -- are read from the buffer by POINT
    INDEX, so autograd treats them as constants. A network that has learnt
    ``T(t) ~ T(t - 0.2 s) + small`` then carries the spatial structure through
    an input autograd cannot see, and ``Fo : grad^2 T`` is taken of the small
    rest. ``model.py``'s ``level()`` docstring says exactly this about the
    per-point anchor under ``residual_output``; the history anchor is the same
    situation with the flag off.

    Measured by comparing, at the same time rows, the autograd ``T_yy``/``T_zz``
    against the finite-difference second derivative of the SAME field values on
    the grid lines (11 points in y and in z). ``ratio`` near 1: autograd sees the
    curvature. Near 0: the conduction term is taken of almost nothing, and
    ``L_phys`` reduces to ``dT/dt - Qsrc``.
    """
    P = int(op["n_points"])
    xn = op["xn"]
    # Integer line keys. In float32 (the dtype xn arrives in) a combined key of
    # this size rounds, merges lines, and the stencil then divides by a zero
    # spacing -- which is how the first version of this reported nan.
    r = lambda v: torch.round(v.double() * 1e6).long()  # noqa: E731
    key_y = r(xn[:, 0]) * 10**9 + r(xn[:, 2])
    key_z = r(xn[:, 0]) * 10**9 + r(xn[:, 1])
    ag = {"yy": [], "zz": []}
    fd = {"yy": [], "zz": []}
    lab = {"yy": [], "zz": []}
    p_all = torch.arange(P, device=buf.device)
    for t in rows:
        pt = torch.full((P,), int(t), device=buf.device, dtype=torch.long)
        _, parts = heat_residual(
            model, xn, op["static"], op["cfg"][pt], op["forcing"][pt], op["Fo"],
            op["Qsrc"][pt, p_all], buf, float(op["dtn"]), op["tn"][pt], p_all,
            phys_scale, time_deriv="bdf2", return_parts=True,
        )
        T = parts["T"].detach()
        for name, coord, key, auto in (("yy", xn[:, 1], key_y, parts["Tyy"]),
                                       ("zz", xn[:, 2], key_z, parts["Tzz"])):
            d2, idx = _fd_second(T, coord, key)
            fd[name].append(d2)
            ag[name].append(auto.detach()[idx])
            # The labels' own curvature at the same row: says whether the
            # rollout's curvature is the field's or the rollout's.
            lab[name].append(_fd_second(op["Tn"][int(t)], coord, key)[0])
    out = {}
    for name in ("yy", "zz"):
        a, f = torch.cat(ag[name]), torch.cat(fd[name])
        ra = float(torch.sqrt(torch.mean(a ** 2)))
        rf = float(torch.sqrt(torch.mean(f ** 2)))
        rl = float(torch.sqrt(torch.mean(torch.cat(lab[name]) ** 2)))
        out[name] = {"autograd_rms": ra, "fd_rms": rf, "labels_fd_rms": rl,
                     "ratio": ra / rf if rf > 0 else float("inf")}
    return out


def data_steps(model: RecurrentField, op: dict, buf: torch.Tensor,
               opt: torch.optim.Optimizer, n: int, *, batch: int,
               gen: torch.Generator, grad_clip: float) -> None:
    """``n`` data-term updates against a FROZEN buffer, exactly as train.py
    draws them: ``t`` in ``[1, split_t)``, labels never past ``split_t``."""
    dtn = float(op["dtn"])
    for _ in range(n):
        bt = torch.randint(1, int(op["split_t"]), (batch,), generator=gen).to(buf.device)
        bp = torch.randint(0, int(op["n_points"]), (batch,), generator=gen).to(buf.device)
        tq = op["tn"][bt]
        hist = model._history(buf, dtn, tq, bp)
        pred = model.field(op["xn"][bp], op["static"][bp], op["cfg"][bt],
                           op["forcing"][bt], hist, model.level(buf, dtn, tq))
        loss = torch.mean((pred - op["Tn"][bt, bp]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()


def fit_inverse_square(deltas, values) -> tuple[float, float, float]:
    """Least squares ``L = A + B / delta^2``; returns ``(A, B, max rel. error)``."""
    d = np.asarray(deltas, dtype=float)
    v = np.asarray(values, dtype=float)
    X = np.column_stack([np.ones_like(d), 1.0 / d ** 2])
    (A, B), *_ = np.linalg.lstsq(X, v, rcond=None)
    err = float(np.max(np.abs(X @ np.array([A, B]) - v) / np.abs(v)))
    return float(A), float(B), err


def share_inverse_square(deltas, values) -> float:
    """Share of ``B / delta^2`` in the fitted ``L`` at the SMALLEST delta."""
    A, B, _ = fit_inverse_square(deltas, values)
    d = float(min(deltas))
    total = A + B / d ** 2
    return float((B / d ** 2) / total) if total > 0 else 0.0


def tags(deltas, fresh, stale, stale_jump_share: float, roughness_ratio: float,
         visibility: float, threshold: float = 0.5) -> list[str]:
    """Name what the numbers show. Several can hold at once, and on the
    synthetic cache they do -- which is why this is a list and not one verdict.

    ``[STALE]``   in the state L_phys is LOGGED in, the jump between the live
                  network and the frozen buffer is at least half of it (at the
                  smallest delta).
    ``[JITTER]``  the fresh rollout alone already scales like 1/delta^2, or its
                  one-step second difference is 10x the labels'.
    ``[BLIND]``   autograd sees under 10 % of the curvature the field has.
    ``[NO 1/delta^2]``  the logged state does not scale like 1/delta^2 at all:
                  Achse 1 is not reproduced on these weights.
    """
    out = []
    if share_inverse_square(deltas, stale) < threshold:
        out.append("[NO 1/delta^2]")
    if stale_jump_share >= threshold:
        out.append("[STALE]")
    if share_inverse_square(deltas, fresh) >= threshold or roughness_ratio >= 10.0:
        out.append("[JITTER]")
    if visibility < 0.1:
        out.append("[BLIND]")
    return out


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def _load(ckpt_path: Path, device: torch.device, ops_override):
    import data as D
    from train import _to_tensor_ops

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["model_config"])
    model = RecurrentField(**cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    run, pre = ck["run"], ck.get("preprocessing", {})
    # ALWAYS all training OPs of the checkpoint: T_mu/T_sigma/phys_scale are
    # pooled over them, and a subset would pool to other numbers. --ops only
    # selects which of them are then measured.
    bundle = D.load_ops(
        op_ids=run["ops"], subsample_time=int(run["subsample"]),
        train_frac=float(pre.get("train_frac", 0.8)),
        resample=str(pre.get("resample", "mean")),
        driver_rate_lags=pre.get("driver_rate_lags", D.DEFAULT_DRIVER_RATE_LAGS),
        use_driver_history=bool(pre.get("use_driver_history", True)),
    )
    # The weights predict a z-score against the normalisation they were trained
    # with. A bundle built from other OPs has another T_mu/T_sigma, and every
    # number below would then be measured in the wrong units without an error.
    stats = ck["bundle_stats"]
    for key in ("T_mu", "T_sigma", "phys_scale"):
        a, b = float(stats[key]), float(getattr(bundle, key))
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            raise SystemExit(
                f"normalisation mismatch: checkpoint {key}={a:.6g}, rebuilt "
                f"bundle {key}={b:.6g}. The cache no longer holds what this "
                f"checkpoint was trained on (rebuilt since?). Measure it against "
                f"the cache it was trained with.")
    ops = _to_tensor_ops(bundle, device)
    if ops_override:
        unknown = sorted(set(ops_override) - {op["op_id"] for op in ops})
        if unknown:
            raise SystemExit(f"--ops {' '.join(unknown)}: not a training OP of this "
                             f"checkpoint ({' '.join(run['ops'])})")
        ops = [op for op in ops if op["op_id"] in set(ops_override)]
    for op in ops:
        op["static"] = op["static"][:, :model.n_static]
        op["forcing"] = op["forcing"][:, :model.n_forcing]
    return model, bundle, ops, ck


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", type=Path, nargs="+",
                   help="one or more model.pt; several run as parallel processes")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="checkpoints measured at once (on the T4: 4, with MPS on)")
    p.add_argument("--deltas", nargs="+", type=float, default=[1.0, 0.4, 0.2],
                   help="BDF lags in SECONDS, all on the same weights")
    p.add_argument("--ops", nargs="*", default=None,
                   help="measure only these of the checkpoint's training OPs "
                        "(the normalisation is always pooled over all of them)")
    p.add_argument("--samples", type=int, default=4096,
                   help="(t, point) samples per OP and delta")
    p.add_argument("--stale-steps", type=int, default=100,
                   help="data updates against the frozen buffer (train.py: --inner-steps)")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--measure-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-data", type=int, default=2048)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2"], default="bdf2")
    p.add_argument("--visibility-rows", type=int, default=8,
                   help="time rows at which the autograd Laplacian is checked")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="auto (cuda when a card answers), cuda or cpu")
    return p


# Every option a child process must see again, in the parent's values.
_PASS = ("deltas", "ops", "samples", "stale_steps", "warmup_steps", "measure_every",
         "lr", "batch_data", "grad_clip", "time_deriv", "visibility_rows", "seed",
         "device")


def _child_argv(args, ckpt: Path) -> list[str]:
    out = [str(ckpt), "-j", "1"]
    for name in _PASS:
        v = getattr(args, name)
        if v is None:
            continue
        flag = "--" + name.replace("_", "-")
        out += [flag] + [str(x) for x in v] if isinstance(v, list) else [flag, str(v)]
    return out


def _resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def run_one(ckpt: Path, args) -> dict:
    """Measure one checkpoint, print the report, write it next to the checkpoint
    as ``residual_decomposition.json`` and return the summary."""
    device = torch.device(args.device)
    model0, bundle, ops, ck = _load(ckpt, device, args.ops)
    T_sigma, T_span = float(bundle.T_sigma), float(bundle.T_span_ref)
    phys_scale = float(bundle.phys_scale)
    deltas_n = [d / T_span for d in args.deltas]
    gen = torch.Generator().manual_seed(args.seed)
    print(f"checkpoint {ckpt}  (trained with delta = "
          f"{ck['model_config']['delta_seconds']:g} s, OPs {' '.join(ck['run']['ops'])})")
    print(f"T_sigma = {T_sigma:.3f} C   T_span = {T_span:.1f} s   "
          f"phys_scale = {phys_scale:.4g}   dt = {float(ops[0]['dtn']) * T_span:.3g} s")
    if ck["run"].get("synthetic_cache"):
        print("  *** trained on the SYNTHETIC cache: the mechanism is measurable, "
              "the numbers are not results ***")

    acc = {st: {d: [] for d in args.deltas} for st in ("fresh", "stale")}
    row0, rough, vis = [], [], []
    for op in ops:
        # fresh: the rollout these weights produce, measured on the same weights
        with torch.no_grad():
            buf = rollout(model0, op["xn"], op["static"], op["cfg"], op["forcing"],
                          op["Tn_ic"], op["tn"], float(op["dtn"]))
        row0.append(row0_jump(model0, op, buf))
        rough.append((op["op_id"], *roughness(buf, op["Tn"])))
        n_t = int(op["n_t"])
        rows = np.linspace(n_t // 4, n_t - 1, args.visibility_rows).astype(int)
        vis.append((op["op_id"], laplacian_visibility(model0, op, buf, rows,
                                                      phys_scale)))
        for d, dn in zip(args.deltas, deltas_n):
            acc["fresh"][d].append(decompose(model0, op, buf, dn, phys_scale,
                                             n_samples=args.samples, gen=gen,
                                             time_deriv=args.time_deriv))
        # stale: warm Adam up, roll out fresh, then drift against that buffer
        m = copy.deepcopy(model0)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        data_steps(m, op, buf, opt, args.warmup_steps, batch=args.batch_data,
                   gen=gen, grad_clip=args.grad_clip)
        with torch.no_grad():
            buf = rollout(m, op["xn"], op["static"], op["cfg"], op["forcing"],
                          op["Tn_ic"], op["tn"], float(op["dtn"]))
        done = 0
        while done < args.stale_steps:
            k = min(args.measure_every, args.stale_steps - done)
            data_steps(m, op, buf, opt, k, batch=args.batch_data, gen=gen,
                       grad_clip=args.grad_clip)
            done += k
            for d, dn in zip(args.deltas, deltas_n):
                acc["stale"][d].append(decompose(m, op, buf, dn, phys_scale,
                                                 n_samples=args.samples, gen=gen,
                                                 time_deriv=args.time_deriv))

    def mean_of(rows, key):
        return float(np.mean([r[key] for r in rows]))

    cols = {}
    for st in ("fresh", "stale"):
        print(f"\n== {st} " + "=" * 60)
        print(f"  {'delta':>6} {'L_phys':>11} {'jump':>7} {'rest':>7} "
              f"{'jump RMS':>10}   L by x-plane (0 = x min)")
        cols[st] = []
        for d in args.deltas:
            rows = acc[st][d]
            L = mean_of(rows, "L")
            cols[st].append(L)
            share_j = mean_of(rows, "L_jump") / L if L > 0 else 0.0
            share_r = mean_of(rows, "L_without_jump") / L if L > 0 else 0.0
            jrms = mean_of(rows, "jump_rms_z") * T_sigma
            planes = sorted({k for r in rows for k in r["L_by_plane"]})
            by_plane = "  ".join(
                f"{k}: {np.mean([r['L_by_plane'][k] for r in rows if k in r['L_by_plane']]):.3g}"
                for k in planes)
            print(f"  {d:>6g} {L:>11.4g} {100 * share_j:>6.1f}% {100 * share_r:>6.1f}% "
                  f"{jrms:>8.3f} C   {by_plane}")
        A, B, err = fit_inverse_square(args.deltas, cols[st])
        print(f"  fit L = A + B/delta^2: A = {A:.4g}  B = {B:.4g}  "
              f"(max rel. error {100 * err:.1f} %)  -> 1/delta^2 share at "
              f"delta = {min(args.deltas):g} s: "
              f"{100 * share_inverse_square(args.deltas, cols[st]):.1f} %")
        d_big = max(args.deltas)
        terms = {k: np.mean([r["terms"][k] for r in acc[st][d_big]])
                 for k in acc[st][d_big][0]["terms"]}
        print(f"  mean squares of the other terms at delta = {d_big:g} s: "
              + "  ".join(f"{k} {v:.3g}" for k, v in terms.items()))

    print("\nLaplacian visibility (fresh rollout; autograd T_yy/T_zz against the "
          "finite difference of the same field):")
    print("  (autograd / finite difference of the field = ratio; labels' finite "
          "difference in brackets)")
    for op_id, v in vis:
        print(f"  {op_id}: yy {v['yy']['autograd_rms']:.3g} / {v['yy']['fd_rms']:.3g} "
              f"= {v['yy']['ratio']:.3g} [{v['yy']['labels_fd_rms']:.3g}]   "
              f"zz {v['zz']['autograd_rms']:.3g} / {v['zz']['fd_rms']:.3g} "
              f"= {v['zz']['ratio']:.3g} [{v['zz']['labels_fd_rms']:.3g}]")
    print("\nroughness of the fresh rollout (one-step 2nd difference, rollout / labels):")
    for op_id, rb, ratio in rough:
        print(f"  {op_id}: {rb * T_sigma:.4f} C  = {ratio:.3g} x the labels'")
    print(f"\nrow 0 (imposed IC, sampled by train.py, not above): "
          f"|T_live(0) - IC| RMS = {np.mean(row0) * T_sigma:.3f} C")
    d_min = min(args.deltas)
    rows = acc["stale"][d_min]
    jump_share = mean_of(rows, "L_jump") / mean_of(rows, "L")
    found = tags(args.deltas, cols["fresh"], cols["stale"], jump_share,
                 max(r for _, _, r in rough),
                 min(min(v["yy"]["ratio"], v["zz"]["ratio"]) for _, v in vis))
    print("\n" + " ".join(found))
    guidance = {
        "[STALE]": "the logged L_phys is mostly the jump between the live network "
                   "and the frozen rollout it is differenced against -- a property "
                   "of the training loop. O18 cannot produce it.",
        "[JITTER]": "the fresh rollout is not smooth on the delta scale; its own "
                    "BDF numerator does not shrink with delta. Also not O18.",
        "[BLIND]": "the conduction term is taken of the network's explicit "
                   "xn-dependence, not of the field. L_phys is effectively "
                   "dT/dt - Qsrc, and no change inside the conduction term "
                   "(O16, O18) can act on it.",
        "[NO 1/delta^2]": "the stale state does not scale like 1/delta^2; Achse 1 "
                          "is not reproduced on these weights.",
    }
    for t in found:
        print(f"  {t} {guidance[t]}")
    summary = {
        "checkpoint": str(ckpt),
        "trained_delta_s": float(ck["model_config"]["delta_seconds"]),
        "synthetic_cache": bool(ck["run"].get("synthetic_cache")),
        "deltas_s": list(args.deltas),
        "L_fresh": cols["fresh"], "L_stale": cols["stale"],
        "stale_jump_share_at_min_delta": jump_share,
        "roughness_ratio_max": max(r for _, _, r in rough),
        "visibility_min": min(min(v["yy"]["ratio"], v["zz"]["ratio"]) for _, v in vis),
        "tags": found,
    }
    out = Path(ckpt).with_name("residual_decomposition.json")
    try:
        out.write_text(json.dumps(summary, indent=2))
        print(f"\n  wrote {out}")
    except OSError as exc:  # a read-only artifacts dir must not cost the report
        print(f"\n  [WARN] could not write {out}: {exc}")
    return summary


def _summary_table(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"  {'checkpoint':<44} {'trained':>7} {'jump':>6} {'vis':>7} {'rough':>7}  tags")
    for r in rows:
        name = r["checkpoint"][-44:]
        print(f"  {name:<44} {r['trained_delta_s']:>6g}s "
              f"{100 * r['stale_jump_share_at_min_delta']:>5.0f}% "
              f"{r['visibility_min']:>7.3g} {r['roughness_ratio_max']:>7.3g}  "
              f"{' '.join(r['tags'])}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.device = _resolve_device(args.device)
    if len(args.checkpoints) == 1 or args.jobs <= 1:
        rows = [run_one(c, args) for c in args.checkpoints]
        if len(rows) > 1:
            _summary_table(rows)
        return 0

    # Several checkpoints in parallel: one process each, exactly like sweep.py,
    # so a crashed checkpoint costs its own line and not the others. On the T4
    # that is only worth it with the MPS daemon up -- the rollout is ~50 tiny
    # kernels per step, and without MPS the driver time-slices the contexts.
    if args.device.startswith("cuda"):
        from sweep import mps_is_running  # the check sweep.py already trusts
        if mps_is_running():
            print("  [MPS] control daemon detected.", flush=True)
        else:
            print("  [WARN] -j > 1 on CUDA without the MPS daemon. Measured on this "
                  "card: 3.69x -> 1.46x.\n         Start it once per boot:  "
                  "nvidia-cuda-mps-control -d", flush=True)
    tools_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tools_dir.parent), str(tools_dir), env.get("PYTHONPATH", "")])

    def one(ckpt: Path) -> tuple[Path, int]:
        log = Path(ckpt).with_name("residual_decomposition.txt")
        with open(log, "w") as fh:
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 *_child_argv(args, ckpt)],
                                stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        print(f"  [{'ok' if rc == 0 else 'FAIL'}] {ckpt}  -> {log}", flush=True)
        return ckpt, rc

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(one, args.checkpoints))
    rows = []
    for ckpt, rc in results:
        js = Path(ckpt).with_name("residual_decomposition.json")
        if rc == 0 and js.exists():
            rows.append(json.loads(js.read_text()))
    if rows:
        _summary_table(rows)
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
