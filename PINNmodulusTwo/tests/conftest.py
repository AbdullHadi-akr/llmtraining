"""Test bootstrap: import ``model``/``physics`` without a Modulus install.

``model.py`` imports Modulus at module scope and raises a helpful setup error when
it is missing. That is right for training, but it would make the history tests
unrunnable anywhere except the GPU server -- and these tests are exactly the ones
that need to run cheaply and often, because they guard a bit-exactness claim.

So: use the real Modulus when it is installed, and otherwise substitute the three
symbols ``model.py`` actually imports. The substitute ``FCLayer`` is a plain
``nn.Linear`` plus the activation, which is all the history path needs -- none of
these tests assert anything about the backbone's numerics.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - exercised only where Modulus is installed
    import modulus  # noqa: F401
except ModuleNotFoundError:

    class ModelMetaData:  # noqa: D101
        name: str = "stub"

    class Module(nn.Module):  # noqa: D101
        def __init__(self, meta=None) -> None:
            super().__init__()
            self.meta = meta

    class FCLayer(nn.Module):  # noqa: D101
        def __init__(self, in_features, out_features, activation_fn=None,
                     weight_norm=False) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.activation_fn = activation_fn

        def forward(self, x):
            x = self.linear(x)
            return x if self.activation_fn is None else self.activation_fn(x)

    root = types.ModuleType("modulus")
    models = types.ModuleType("modulus.models")
    layers = types.ModuleType("modulus.models.layers")
    meta_mod = types.ModuleType("modulus.models.meta")
    module_mod = types.ModuleType("modulus.models.module")

    layers.FCLayer = FCLayer
    meta_mod.ModelMetaData = ModelMetaData
    module_mod.Module = Module
    models.layers, models.meta, models.module = layers, meta_mod, module_mod
    root.models = models

    sys.modules.update({
        "modulus": root,
        "modulus.models": models,
        "modulus.models.layers": layers,
        "modulus.models.meta": meta_mod,
        "modulus.models.module": module_mod,
    })
