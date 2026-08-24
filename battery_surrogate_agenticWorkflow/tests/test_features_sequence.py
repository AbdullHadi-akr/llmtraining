"""Tests for sequence features and dataset."""

import numpy as np
import pytest

from battery_surrogate.model.features_sequence import build_sequence_for_sensor, build_history_lags
from battery_surrogate.model.dataset_sequence import SequenceDataset
from battery_surrogate.model.normalizer import PointwiseNormalizer


def test_build_sequence_for_sensor_shape(make_minimal_bundle):
    """Test that sequence features/targets have correct shapes."""
    bundle = make_minimal_bundle("TEST01")
    
    config = {
        "data": {
            "subsample_time": 5,
            "ts_extrapolation": "clamp",
        }
    }
    
    features, targets, seq_len = build_sequence_for_sensor(
        bundle, sensor_idx=0, config=config
    )
    
    # Features: (seq_len, 11) — [x, y, z, t, 7×sim_config]
    assert features.shape == (seq_len, 11)
    assert features.dtype == np.float32
    
    # Targets: (seq_len, 2) — [T, bc_V]
    assert targets.shape == (seq_len, 2)
    assert targets.dtype == np.float32
    
    # seq_len should match subsampled time
    expected_seq_len = (bundle.t_fast.shape[0] + 4) // 5  # ceil divide
    assert seq_len == expected_seq_len or seq_len == expected_seq_len + 1


def test_build_sequence_coordinates_constant():
    """Test that coordinates are constant across time in sequence."""
    from battery_surrogate.data.models import OpBundle
    
    n_fast = 50
    n_sensors = 10
    n_scalar = 3
    
    bundle = OpBundle(
        op_id="TEST_COORDS",
        schema_version=2,
        cache_key="test",
        t_fast=np.arange(n_fast, dtype=np.float32),
        t_slow=np.arange(10, dtype=np.float32) * 10.0,
        bc_V=np.ones(n_fast, dtype=np.float32) * 3.7,
        bc_OCV=np.ones(n_fast, dtype=np.float32) * 3.6,
        bc_I=np.ones(n_fast, dtype=np.float32) * 100,
        pe_P_loss=np.ones(n_fast, dtype=np.float32) * 10,
        T=np.random.randn(n_fast, n_sensors).astype(np.float32) + 25.0,
        q_source=np.ones((10, 3), dtype=np.float32),
        xyz=np.random.randn(n_sensors, 3).astype(np.float32),
        layer=np.array(["cc"] * (n_sensors // 2) + ["g"] * (n_sensors // 2)),
        sensor_id=np.array([f"s_{i:03d}" for i in range(n_sensors)]),
        fluid_props=np.array([[1000.0, 4180.0, 0.6]], dtype=np.float32),
        sim_config_scalar=np.array([2.0, 316.0, 25.0], dtype=np.float32),
        sim_config_scalar_names=("c_rate", "cell_current", "fluid_initial_temp"),
        sim_config_ts={},
        meta={
            "op_id": "TEST_COORDS",
            "sim_config_scalar_names": ["c_rate", "cell_current", "fluid_initial_temp"],
        },
    )
    
    config = {"data": {"subsample_time": 1}}
    sensor_idx = 5
    
    features, targets, seq_len = build_sequence_for_sensor(bundle, sensor_idx, config)
    
    # x, y, z should be constant (columns 0, 1, 2)
    expected_xyz = bundle.xyz[sensor_idx, :]
    for t in range(seq_len):
        np.testing.assert_array_almost_equal(features[t, :3], expected_xyz)


def test_build_history_lags_shape():
    """Test that history lags have correct shape and alignment."""
    seq_len = 20
    targets_seq = np.random.randn(seq_len, 2).astype(np.float32)
    history_length = 8
    k = history_length
    
    history = build_history_lags(targets_seq, history_length)
    
    # Should be (seq_len, 2*k)
    assert history.shape == (seq_len, 2 * k)
    assert history.dtype == np.float32


def test_build_history_lags_warm_up():
    """Test that history lag warm-up (t < k) uses y_0 padding."""
    targets_seq = np.array([
        [25.0, 3.7],   # t=0, y_0
        [26.0, 3.8],   # t=1
        [27.0, 3.9],   # t=2
    ], dtype=np.float32)
    
    history = build_history_lags(targets_seq, history_length=2)
    
    # At t=0, history should be [y_0, y_0] (replicated warm-up)
    expected_t0 = np.tile(targets_seq[0, :], 2)  # [25.0, 3.7, 25.0, 3.7]
    np.testing.assert_array_almost_equal(history[0, :], expected_t0)
    
    # At t=1, still warm-up (t < k=2), history should be [y_0, y_0]
    np.testing.assert_array_almost_equal(history[1, :], expected_t0)
    
    # At t=2, full history: [y_0, y_1]
    expected_t2 = np.array([25.0, 3.7, 26.0, 3.8], dtype=np.float32)
    np.testing.assert_array_almost_equal(history[2, :], expected_t2)


def test_build_history_lags_alignment():
    """Test that history at step t contains [y_{t-k}, ..., y_{t-1}]."""
    seq_len = 10
    targets_seq = np.arange(seq_len * 2).reshape(seq_len, 2).astype(np.float32)
    history_length = 3
    k = history_length
    
    history = build_history_lags(targets_seq, history_length)
    
    # At t=5 (when t >= k), history should be [y_2, y_3, y_4] flattened
    t = 5
    expected = targets_seq[t - k:t, :].flatten()
    np.testing.assert_array_almost_equal(history[t, :], expected)


def test_sequence_dataset_iter_yields_tuples(make_minimal_bundle, monkeypatch):
    """Test that SequenceDataset yields 6-tuples with correct shapes."""
    def mock_load_op(op_id):
        return make_minimal_bundle(op_id)
    
    monkeypatch.setattr("battery_surrogate.model.dataset_sequence.load_op", mock_load_op)
    
    config = {
        "data": {
            "subsample_time": 10,
            "ts_extrapolation": "clamp",
        },
        "model": {
            "history_length": 4,
        }
    }
    
    normalizer = PointwiseNormalizer()
    # Fit on dummy data
    x_dummy = np.random.randn(100, 11).astype(np.float32)
    y_dummy = np.random.randn(100, 2).astype(np.float32)
    normalizer.partial_fit(x_dummy, y_dummy)
    normalizer.finalize()
    
    dataset = SequenceDataset(
        ["TEST01", "TEST02"],
        normalizer=normalizer,
        config=config,
        shuffle_ops=False,
    )
    
    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= 2:  # Collect just a couple samples
            break
    
    # Each sample should be a 6-tuple
    assert len(samples) >= 1
    sample = samples[0]
    assert len(sample) == 6, f"Expected 6-tuple, got {len(sample)}"
    
    features, history, targets, seq_len, op_id, sensor_id = sample
    
    # Verify shapes and types
    assert features.shape[1] == 11
    assert history.shape[1] == 8  # 2 * history_length=4
    assert targets.shape[1] == 2
    assert isinstance(seq_len, (int, np.integer))
    assert isinstance(op_id, str)
    assert isinstance(sensor_id, (int, np.integer))


def test_sequence_dataset_n_sensors_inferred(make_minimal_bundle, monkeypatch):
    """Test that n_sensors is correctly inferred from first OP."""
    def mock_load_op(op_id):
        return make_minimal_bundle(op_id)
    
    monkeypatch.setattr("battery_surrogate.model.dataset_sequence.load_op", mock_load_op)
    
    config = {
        "data": {"subsample_time": 1},
        "model": {"history_length": 4},
    }
    
    normalizer = PointwiseNormalizer()
    x_dummy = np.random.randn(50, 11).astype(np.float32)
    y_dummy = np.random.randn(50, 2).astype(np.float32)
    normalizer.partial_fit(x_dummy, y_dummy)
    normalizer.finalize()
    
    dataset = SequenceDataset(
        ["TEST01"],
        normalizer=normalizer,
        config=config,
    )
    
    # n_sensors should match minimal bundle
    bundle = make_minimal_bundle("TEST01")
    expected_n_sensors = bundle.xyz.shape[0]
    assert dataset.n_sensors == expected_n_sensors
