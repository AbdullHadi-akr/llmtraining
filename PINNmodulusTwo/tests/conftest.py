"""Test bootstrap: import ``model``/``physics`` without a Modulus install.

``model.py`` imports Modulus at module scope and raises a helpful setup error when
it is missing. That is right for training, but it would make the history tests
unrunnable anywhere except the GPU server -- and these tests are exactly the ones
that need to run cheaply and often, because they guard a bit-exactness claim.

So: use the real Modulus when it is installed, and otherwise substitute the three
symbols ``model.py`` actually imports. The substitute ``FCLayer`` is a plain
``nn.Linear`` plus the activation.

The INITIALISATION is copied from the real ``FCLayer`` on purpose, even though
the history tests do not assert anything about the backbone's numerics. Modulus
zeroes the bias and uses ``xavier_uniform_``; ``nn.Linear``'s own default is
``kaiming_uniform_(a=sqrt(5))`` with a random bias, which for a square layer is
a factor ``sqrt(1/3)`` less expansive per layer. Over a 4-layer stack that is
~9x, and it is the difference between "the untrained rollout stays bounded" and
"it reaches inf": a stability measurement taken against the old stub reported
``history_mode=hybrid, residual_output=False`` as stable when the real model
diverges on every seed. ``weight_norm(dim=0)`` initialises ``g`` to the current
row norms, so it reproduces the xavier weights exactly and only changes the
parameterisation -- but it is applied here too, so the parameter NAMES match a
real checkpoint.
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
            nn.init.constant_(self.linear.bias, 0)
            nn.init.xavier_uniform_(self.linear.weight)
            if weight_norm:
                nn.utils.parametrizations.weight_norm(
                    self.linear, name="weight", dim=0)

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
