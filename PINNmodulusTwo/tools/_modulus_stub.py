"""Faithful-at-init stand-in for modulus.models.layers.FCLayer.

The real FCLayer does:
    nn.init.constant_(self.linear.bias, 0)
    nn.init.xavier_uniform_(self.linear.weight)
    if weight_norm: nn.utils.weight_norm(self.linear, name="weight", dim=0)

weight_norm(dim=0) initialises g to the current row norms, so it reproduces the
xavier weights EXACTLY at init -- it changes the parameterisation, not the
values. The only init difference from tests/conftest.py's stub is therefore:
    xavier_uniform  vs  nn.Linear default (kaiming_uniform, a=sqrt(5))
    bias = 0        vs  bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
Both are reproduced here.
"""
from __future__ import annotations
import sys, types
import torch.nn as nn


def install(faithful: bool = True) -> None:
    class ModelMetaData:
        name = "stub"

    class Module(nn.Module):
        def __init__(self, meta=None):
            super().__init__()
            self.meta = meta

    class FCLayer(nn.Module):
        def __init__(self, in_features, out_features, activation_fn=None,
                     weight_norm=False):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.activation_fn = activation_fn
            if faithful:
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
        "modulus": root, "modulus.models": models,
        "modulus.models.layers": layers, "modulus.models.meta": meta_mod,
        "modulus.models.module": module_mod,
    })
