"""Tests for per-target history length support in recurrent models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from battery_surrogate.model.features_sequence import (
    resolve_history_lengths,
    build_history_lags_per_target,
)
from battery_surrogate.model.recurrent_pointwise import RecurrentPointwise


def test_resolve_history_lengths_scalar():
    """Test scalar history_length resolves to (k, k)."""
    config = {"history_length": 8}
    k_T, k_V = resolve_history_lengths(config)
    assert k_T == 8
    assert k_V == 8


def test_resolve_history_lengths_dict():
    """Test dict history_length resolves correctly."""
    config = {"history_length": {"T": 6, "bc_V": 3}}
    k_T, k_V = resolve_history_lengths(config)
    assert k_T == 6
    assert k_V == 3


def test_resolve_history_lengths_default():
    """Test default history_length."""
    config = {}
    k_T, k_V = resolve_history_lengths(config)
    assert k_T == 8
    assert k_V == 8


def test_resolve_history_lengths_invalid():
    """Test invalid history_length raises ValueError."""
    config = {"history_length": 0}
    with pytest.raises(ValueError):
        resolve_history_lengths(config)


def test_build_history_lags_per_target_shape():
    """Test build_history_lags_per_target output shape."""
    targets = np.random.randn(100, 2).astype(np.float32)
    history = build_history_lags_per_target(targets, k_T=6, k_V=3)
    assert history.shape == (100, 9)  # 6 + 3


def test_build_history_lags_per_target_grouped():
    """Test grouped layout of build_history_lags_per_target."""
    targets = np.arange(20).reshape(10, 2).astype(np.float32)  # [[0,1], [2,3], ...]
    history = build_history_lags_per_target(targets, k_T=2, k_V=1)
    
    # At t=2, should have T-block = [0, 2] (T values at t=0,1) and V-block = [1] (V at t=1)
    # T values at positions 0, 2, 4, ... | V values at positions 1, 3, 5, ...
    assert history.shape == (10, 3)  # 2 + 1
    
    # Check warm-up: at t=0, should be padded with y_0
    y_0_T = targets[0, 0]  # 0.0
    y_0_V = targets[0, 1]  # 1.0
    assert history[0, 0] == y_0_T
    assert history[0, 1] == y_0_T
    assert history[0, 2] == y_0_V


def test_recurrent_model_per_target_history():
    """Test RecurrentPointwise with per-target history."""
    config = {
        "model": {
            "history_length": {"T": 4, "bc_V": 2},
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 32,
        }
    }
    model = RecurrentPointwise(config, n_sensors=10, seed=42)
    
    assert model.k_T == 4
    assert model.k_V == 2
    assert model.history_length == 4  # max(4, 2)
    
    # Input size should be 11 + 4 + 2 = 17
    # Check the first layer of RNN
    assert model.rnn.input_size == 17


def test_recurrent_model_forward():
    """Test forward pass with per-target history."""
    config = {
        "model": {
            "history_length": {"T": 3, "bc_V": 2},
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 16,
        }
    }
    model = RecurrentPointwise(config, n_sensors=10, seed=42)
    
    batch_size = 2
    seq_len = 5
    features = torch.randn(batch_size, seq_len, 11)
    history = torch.randn(batch_size, seq_len, 5)  # 3 + 2
    
    output = model.forward(features, history)
    assert output.shape == (batch_size, seq_len, 2)


def test_recurrent_model_rollout():
    """Test rollout with per-target history."""
    config = {
        "model": {
            "history_length": {"T": 2, "bc_V": 1},
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 16,
        }
    }
    model = RecurrentPointwise(config, n_sensors=10, seed=42)
    
    seq_len = 10
    features = torch.randn(seq_len, 11)
    y0 = torch.tensor([0.0, 1.0])
    
    output = model.rollout(features, y0)
    assert output.shape == (seq_len, 2)
    
    # Check that rollout is deterministic
    output2 = model.rollout(features, y0)
    assert torch.allclose(output, output2)


def test_recurrent_backward_compat_scalar_history():
    """Test that scalar history_length still works (backward compatibility)."""
    config = {
        "model": {
            "history_length": 8,  # scalar
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 16,
        }
    }
    model = RecurrentPointwise(config, n_sensors=10, seed=42)
    
    assert model.k_T == 8
    assert model.k_V == 8
    assert model.history_length == 8
    assert model.rnn.input_size == 11 + 16  # 11 + 8 + 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
