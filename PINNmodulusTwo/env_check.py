"""Fail with a useful sentence when the training environment is not active.

Why this file exists
--------------------
Running an entry point with the system interpreter instead of the project
virtualenv produces this::

    File ".../PINNmodulusTwo/materials.py", line 18, in <module>
        import pandas as pd
    ModuleNotFoundError: No module named 'pandas'

Every word of that is true and none of it is the actual problem. The problem is
that ``source modulus_env/bin/activate`` was not run, and nothing in the message
says so -- it points at ``materials.py``, four imports deep, which is the first
place the missing dependency happened to be needed rather than the place
anything is wrong. The obvious reading is "pandas is not installed, let me pip
install pandas", which then half-populates the system interpreter and makes the
next failure harder to read, not easier.

So: check up front, name the interpreter actually in use, and say what to do.

This module imports NOTHING outside the standard library, which is the whole
point -- it has to survive exactly the situation it diagnoses. It is also why
``op_registry.py`` deliberately does not import it: that file is pure stdlib and
runs anywhere, and the roadmap leans on that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# What an entry point needs before it can do anything. Import name first (that
# is what fails), then the pip name where the two differ.
_REQUIRED = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
)

# Candidate virtualenvs, in the order the project's own docs mention them. Only
# used to print a command that will actually work on this machine.
_VENV_NAMES = ("modulus_env", ".venv", "venv", "physics_env_fixed")


def _missing() -> list[tuple[str, str]]:
    out = []
    for mod, pip_name in _REQUIRED:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append((mod, pip_name))
    return out


def _venv_hint(project_root: Path) -> str:
    """An activate line for a venv that exists here, or a generic one."""
    for name in _VENV_NAMES:
        if (project_root / name / "bin" / "activate").exists():
            return f"source {name}/bin/activate"
        if (project_root / name / "Scripts" / "activate").exists():   # Windows
            return f"{name}\\Scripts\\activate"
    return "source modulus_env/bin/activate    # or whatever the venv is called"


def require_training_env() -> None:
    """Exit with an explanation if the third-party dependencies are missing.

    Called at the top of every entry point that needs them. A no-op in a
    correctly activated environment, so it costs one ``find_spec`` per module
    and nothing else.
    """
    missing = _missing()
    if not missing:
        return

    root = Path(__file__).resolve().parent.parent
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    names = ", ".join(m for m, _ in missing)
    pips = " ".join(sorted({p for _, p in missing}))

    lines = [
        "",
        "=" * 72,
        "  The training environment is not active.",
        "=" * 72,
        f"  missing     : {names}",
        f"  interpreter : {sys.executable}",
        f"  virtualenv  : {'yes -- ' + sys.prefix if in_venv else 'NO (this is the system Python)'}",
        "",
    ]
    if not in_venv:
        lines += [
            "  That is almost certainly the whole problem. Activate the project",
            "  environment and run the same command again:",
            "",
            f"      cd {root}",
            f"      {_venv_hint(root)}",
            "",
            "  Do NOT pip install these into the system Python -- a half-populated",
            "  system interpreter makes the next error harder to read, not easier.",
        ]
    else:
        lines += [
            "  A virtualenv IS active, so it is missing packages rather than being",
            "  the wrong environment. Install them into it:",
            "",
            f"      pip install {pips}",
            "",
            "  Full setup (driver, CUDA torch, Modulus):",
            "      PINNmodulusTwo/README_GPU_SERVER.md",
        ]
    lines += ["=" * 72, ""]
    raise SystemExit("\n".join(lines))
