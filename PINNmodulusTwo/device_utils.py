"""Device selection helpers shared by the PINNmodulusTwo entry points.

Four modes, and the difference between them is who decides:

* ``ask``  -- the default. Lists what the machine has and prompts. This project
  is run on a laptop CPU and on a GPU server by the same person from the same
  checkout, and the wrong one is a wasted afternoon in either direction.
  Falls back to ``auto`` without blocking when there is no terminal to ask
  (CI, ``nohup``, a pipe).
* ``auto`` -- CUDA when available, else CPU. No prompt.
* ``cpu``  -- forced.
* ``cuda`` / ``cuda:N`` -- forced, and it FAILS LOUDLY if the card is not there
  rather than falling back to the CPU. A silent fallback on a GPU box is very
  easy to miss in a log and costs the whole run's speed.

See ``README_GPU_SERVER.md`` for the full server setup.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


def available_devices() -> list[tuple[str, str]]:
    """``[(spec, human description)]`` for every device this machine can train on.

    CPU is always first and always present. Each visible CUDA card follows with
    its name and memory, because "cuda:1" on its own tells you nothing about
    which card you are about to spend hours on.
    """
    out = [("cpu", f"CPU  ({os.cpu_count() or '?'} threads visible)")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            out.append((
                f"cuda:{i}",
                f"{pr.name}  {pr.total_memory / 1024**3:.1f} GiB  "
                f"sm_{pr.major}{pr.minor}",
            ))
    return out


def _prompt_for_device() -> str:
    """Ask which device to use, and return the chosen ``--device`` spec.

    Only ever called for ``--device ask``. Falls back to ``auto`` without
    blocking whenever there is nobody to answer -- a CI job, a nohup'd run, a
    pipe -- because a training script that hangs on a prompt nobody can see is
    strictly worse than one that picks a sane default and says so.
    """
    devices = available_devices()

    if not sys.stdin.isatty():
        pick = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[device] --device ask, but this is not an interactive terminal; "
              f"falling back to auto -> {pick}. Pass --device explicitly to be "
              f"sure (it is also a config.yaml key).", flush=True)
        return pick

    if len(devices) == 1:
        print("[device] --device ask: no CUDA device is visible, so there is "
              "nothing to choose -- using the CPU.", flush=True)
        return "cpu"

    print("\nWhich device should this run use?", flush=True)
    for n, (spec, desc) in enumerate(devices, start=1):
        default = "   <- default" if spec.startswith("cuda") and n == 2 else ""
        print(f"  [{n}] {spec:<8} {desc}{default}", flush=True)
    default_spec = devices[1][0]

    while True:
        try:
            raw = input(f"Choice [1-{len(devices)}, Enter = {default_spec}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[device] no answer, using {default_spec}", flush=True)
            return default_spec
        if not raw:
            return default_spec
        # Accept the index or the spec itself, so "cuda:1" and "3" both work.
        if raw in {spec for spec, _ in devices}:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(devices):
            return devices[int(raw) - 1][0]
        print(f"  not one of 1..{len(devices)} or "
              f"{', '.join(spec for spec, _ in devices)} -- try again", flush=True)


def resolve_device(spec: str) -> torch.device:
    """Turn a ``--device`` string into a ``torch.device`` and log what was picked.

    Args:
        spec: ``ask`` (prompt, when there is a terminal to prompt), ``auto``
            (CUDA when available, else CPU), ``cpu``, ``cuda`` or ``cuda:N``.

    Raises:
        RuntimeError: if CUDA was requested explicitly but is unavailable or the
            requested index does not exist. Failing here is deliberate: a silent
            fallback to the CPU on a GPU box is very easy to miss in the logs.
    """
    spec = str(spec).strip().lower()

    # Resolved first, and to another spec rather than to a device, so everything
    # below -- the loud failure on a missing card included -- applies to what
    # was chosen interactively exactly as it does to what was typed.
    if spec == "ask":
        spec = _prompt_for_device()

    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"

    if spec == "cpu":
        print("[device] cpu", flush=True)
        return torch.device("cpu")

    if not spec.startswith("cuda"):
        raise RuntimeError(
            f"unknown --device {spec!r}; expected 'ask', 'auto', 'cpu', 'cuda' "
            f"or 'cuda:N'"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {spec!r} was requested but torch.cuda.is_available() is False.\n"
            f"  torch={torch.__version__}, torch.version.cuda={torch.version.cuda}\n"
            "  Check the NVIDIA driver with 'nvidia-smi' and make sure a CUDA build of\n"
            "  torch is installed (see PINNmodulusTwo/README_GPU_SERVER.md).\n"
            "  Use '--device auto' if a CPU fallback is acceptable."
        )

    device = torch.device(spec)
    index = device.index if device.index is not None else torch.cuda.current_device()
    n_devices = torch.cuda.device_count()
    if index >= n_devices:
        raise RuntimeError(
            f"--device {spec!r} requested but only {n_devices} CUDA device(s) visible "
            f"(valid indices 0..{n_devices - 1}). Check CUDA_VISIBLE_DEVICES."
        )

    props = torch.cuda.get_device_properties(index)
    print(
        f"[device] cuda:{index} {props.name}  "
        f"{props.total_memory / 1024**3:.1f} GiB  "
        f"sm_{props.major}{props.minor}  "
        f"torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )
    return torch.device(f"cuda:{index}")


def seed_everything(seed: int) -> None:
    """Seed torch (CPU + all CUDA devices) and numpy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_tf32(flag: bool) -> None:
    """Toggle TF32 matmuls on Ampere+ cards.

    Off by default on purpose: the physics residual differentiates the network
    twice (``physics.py`` autograd Hessian in space), and the reduced TF32
    mantissa noticeably degrades second derivatives.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(flag)
    torch.backends.cudnn.allow_tf32 = bool(flag)
    if flag:
        print("[device] TF32 matmuls ENABLED (faster, less precise derivatives)", flush=True)
