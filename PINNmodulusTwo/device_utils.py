"""Device selection helpers shared by the PINNmodulusTwo entry points.

The training/benchmark scripts used to default to ``--device cpu`` (the code was
developed CPU-first in WSL). On a GPU server that silently wastes the card, so
the default is now ``auto`` and an explicit ``cuda`` request fails loudly instead
of falling back to the CPU.

See ``README_GPU_SERVER.md`` for the full server setup.
"""

from __future__ import annotations

import numpy as np
import torch


def resolve_device(spec: str) -> torch.device:
    """Turn a ``--device`` string into a ``torch.device`` and log what was picked.

    Args:
        spec: ``auto`` (CUDA when available, else CPU), ``cpu``, ``cuda`` or
            ``cuda:N``.

    Raises:
        RuntimeError: if CUDA was requested explicitly but is unavailable or the
            requested index does not exist. Failing here is deliberate: a silent
            fallback to the CPU on a GPU box is very easy to miss in the logs.
    """
    spec = str(spec).strip().lower()

    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"

    if spec == "cpu":
        print("[device] cpu", flush=True)
        return torch.device("cpu")

    if not spec.startswith("cuda"):
        raise RuntimeError(
            f"unknown --device {spec!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'"
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
