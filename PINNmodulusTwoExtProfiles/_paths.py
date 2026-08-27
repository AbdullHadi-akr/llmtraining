"""Make the unchanged PINNmodulusTwo modules importable from this extension.

Why a path shim instead of copies
---------------------------------
This folder is an *extension* of ``PINNmodulusTwo``, not a fork. Three of its
modules are untouched by the profile work and copying them would only create two
versions that drift apart:

* ``model.py``      -- ``RecurrentField`` needs no change. The profile features
  arrive as extra ``n_config`` / ``n_forcing`` input channels, and both are
  already constructor arguments.
* ``physics.py``    -- the residual is the same PDE.
* ``materials.py``  -- reads ``PINNmodulusTwo/material_properties/``, which is
  where the material CSVs actually live. Importing it from here keeps that one
  copy authoritative.
* ``device_utils.py`` -- device resolution / seeding / TF32.

What this extension *does* own is everything the profiles change:
``data.py`` (preprocessing + normalisation), ``train.py`` (the loop over a
heterogeneous OP set), ``op_registry.py``, ``bench_profiles.py`` and
``profileBench.py``.

Import order matters
--------------------
This directory is inserted *ahead* of ``PINNmodulusTwo``, so ``import data``
resolves to the profile-aware loader in this folder while ``import model``
falls through to ``PINNmodulusTwo/model.py``. Any module this extension adds
therefore shadows the sibling of the same name -- that is deliberate, and it is
the reason no file here may be named ``model.py``, ``physics.py``,
``materials.py`` or ``device_utils.py`` unless it really means to replace one.

Import this module first, before any of the shared modules:

    import _paths  # noqa: F401
    from model import RecurrentField
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
BASE_DIR = PROJECT_ROOT / "PINNmodulusTwo"

if not BASE_DIR.exists():
    raise SystemExit(
        f"the base project is missing: {BASE_DIR}\n"
        "  This extension reuses PINNmodulusTwo/model.py, physics.py, materials.py\n"
        "  and device_utils.py. It has to sit next to that folder, not replace it."
    )

# Base first, then this directory -- so THIS_DIR ends up at index 0 and wins any
# name collision. See the module docstring.
for _p in (BASE_DIR, THIS_DIR):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)
