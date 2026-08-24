"""Unit tests for recurrent sequence model."""

import torch

from battery_surrogate.model.recurrent_pointwise import RecurrentPointwise


def _build_config(rnn_type: str) -> dict:
    return {
        "rnn_type": rnn_type,
        "n_layers": 2,
        "hidden_size": 32,
        "history_length": 8,
    }


def test_forward_shape_gru():
    model = RecurrentPointwise(_build_config("gru"), n_sensors=50, seed=42)
    features = torch.randn(2, 100, 11)
    history = torch.randn(2, 100, 16)
    out = model(features, history)
    assert out.shape == (2, 100, 2)


def test_forward_shape_lstm():
    model = RecurrentPointwise(_build_config("lstm"), n_sensors=50, seed=42)
    features = torch.randn(2, 100, 11)
    history = torch.randn(2, 100, 16)
    out = model(features, history)
    assert out.shape == (2, 100, 2)


def test_rollout_shape():
    model = RecurrentPointwise(_build_config("gru"), n_sensors=50, seed=42)
    features = torch.randn(100, 11)
    y0 = torch.randn(2)
    out = model.rollout(features, y0)
    assert out.shape == (100, 2)


def test_rollout_deterministic_for_seed():
    cfg = _build_config("gru")
    model_a = RecurrentPointwise(cfg, n_sensors=50, seed=123)
    model_b = RecurrentPointwise(cfg, n_sensors=50, seed=123)

    features = torch.randn(30, 11)
    y0 = torch.randn(2)
    out_a = model_a.rollout(features, y0)
    out_b = model_b.rollout(features, y0)
    torch.testing.assert_close(out_a, out_b)


def test_forward_not_equal_rollout():
    model = RecurrentPointwise(_build_config("gru"), n_sensors=50, seed=42)
    features = torch.randn(1, 20, 11)
    history = torch.randn(1, 20, 16)
    forward_out = model(features, history).squeeze(0)

    rollout_out = model.rollout(features.squeeze(0), y0=torch.zeros(2))
    assert not torch.allclose(forward_out, rollout_out)
