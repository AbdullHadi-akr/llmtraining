from __future__ import annotations

import numpy as np
import torch

from battery_surrogate.model.dataset_pointwise import PointwiseDataset
from battery_surrogate.model.features_pointwise import assemble_pointwise_block
from battery_surrogate.model.mlp_pointwise import LearnableSwish, PointwiseMLP
from battery_surrogate.model.normalizer import PointwiseNormalizer
from battery_surrogate.model.trainer import _compute_loss
from battery_surrogate.schema.columns import CANONICAL_CHANNELS


def _bundle_with_all_channels(make_minimal_bundle):
    bundle = make_minimal_bundle(op_id="PTW01")

    scalar_names = list(bundle.sim_config_scalar_names)
    scalar_values = list(bundle.sim_config_scalar.tolist())

    sim_ts = dict(bundle.sim_config_ts)
    for channel in CANONICAL_CHANNELS:
        if channel in scalar_names:
            continue
        sim_ts[channel] = (
            np.array([bundle.t_fast[0], bundle.t_fast[-1]], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )

    return bundle.__class__(
        op_id=bundle.op_id,
        schema_version=bundle.schema_version,
        cache_key=bundle.cache_key,
        t_fast=bundle.t_fast,
        t_slow=bundle.t_slow,
        bc_V=bundle.bc_V,
        bc_OCV=bundle.bc_OCV,
        bc_I=bundle.bc_I,
        pe_P_loss=bundle.pe_P_loss,
        T=bundle.T,
        q_source=bundle.q_source,
        xyz=bundle.xyz,
        layer=bundle.layer,
        sensor_id=bundle.sensor_id,
        fluid_props=bundle.fluid_props,
        sim_config_scalar=np.asarray(scalar_values, dtype=np.float32),
        sim_config_scalar_names=tuple(scalar_names),
        sim_config_ts=sim_ts,
        meta=bundle.meta,
    )


def test_swish_beta_per_layer() -> None:
    model = PointwiseMLP(
        n_features=11,
        n_hidden_layers=4,
        hidden_size=16,
        swish_beta_init=1.0,
        swish_beta_learnable=True,
    )

    swish_layers = [module for module in model.modules() if isinstance(module, LearnableSwish)]
    assert len(swish_layers) == 4
    assert all(layer.beta.requires_grad for layer in swish_layers)


def test_swish_beta_frozen() -> None:
    model = PointwiseMLP(
        n_features=11,
        n_hidden_layers=2,
        hidden_size=8,
        swish_beta_init=1.0,
        swish_beta_learnable=False,
    )

    swish_layers = [module for module in model.modules() if isinstance(module, LearnableSwish)]
    assert all(not layer.beta.requires_grad for layer in swish_layers)


def test_layer_order() -> None:
    model = PointwiseMLP(
        n_features=11,
        n_hidden_layers=1,
        hidden_size=8,
        swish_beta_init=1.0,
        swish_beta_learnable=True,
    )

    block = list(model.backbone.children())
    assert isinstance(block[0], torch.nn.Linear)
    assert isinstance(block[1], torch.nn.LayerNorm)
    assert isinstance(block[2], LearnableSwish)


def test_zscore_roundtrip() -> None:
    normalizer = PointwiseNormalizer()
    x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=np.float32)

    normalizer.partial_fit(x, y)
    normalizer.finalize()

    y_norm = normalizer.transform_Y(y)
    y_back = normalizer.inverse_Y(y_norm)
    np.testing.assert_allclose(y_back, y, atol=1.0e-5)


def test_features_dim(make_minimal_bundle) -> None:
    bundle = _bundle_with_all_channels(make_minimal_bundle)
    x_block, y_block, _sensor_ids = assemble_pointwise_block(bundle, 0)

    assert x_block.shape[1] == 11
    assert y_block.shape[1] == 2


def test_full_spatial_coverage(make_minimal_bundle) -> None:
    bundle = _bundle_with_all_channels(make_minimal_bundle)

    def loader(_op_id: str):
        return bundle

    dataset = PointwiseDataset(
        ["PTW01"],
        subsample_time=100,
        loader=loader,
        shuffle_ops=False,
        shuffle_time=False,
    )

    assert dataset.n_sensors == bundle.xyz.shape[0]


def test_bcV_deinflation() -> None:
    pred = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    loss = _compute_loss(
        pred,
        target,
        n_sensors=10,
        t_weight=1.0,
        bc_v_weight=1.0,
    )

    # MSE for first column = 1.0; second column = 4.0, scaled by 1/10.
    assert torch.isclose(loss, torch.tensor(1.4, dtype=torch.float32))
