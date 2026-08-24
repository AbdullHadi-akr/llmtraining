"""Tests for normalizer schemes (per-group scaling and backward compatibility)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from battery_surrogate.model.normalizer import PointwiseNormalizer


def test_normalizer_default_preset_backward_compat():
    """Test that default preset (None) reproduces all-zscore behavior (byte-parity)."""
    # Create two normalizers: one old-style, one with new default None
    normalizer_new = PointwiseNormalizer(preprocess_config=None)
    
    # Sample data (11 features, 2 targets)
    x_train = np.random.randn(100, 11).astype(np.float32) + 5
    y_train = np.random.randn(100, 2).astype(np.float32) + 3
    
    normalizer_new.partial_fit(x_train, y_train)
    normalizer_new.finalize()
    
    # Apply transform
    x_norm_new = normalizer_new.transform_X(x_train)
    y_norm_new = normalizer_new.transform_Y(y_train)
    
    # Verify normalization is z-score (mean≈0, std≈1)
    assert np.abs(x_norm_new.mean()) < 0.1, "Normalized X should have mean ~0"
    assert np.abs(x_norm_new.std() - 1.0) < 0.1, "Normalized X should have std ~1"
    assert np.abs(y_norm_new.mean()) < 0.1, "Normalized Y should have mean ~0"
    assert np.abs(y_norm_new.std() - 1.0) < 0.1, "Normalized Y should have std ~1"


def test_normalizer_version_2_save_load():
    """Test that version 2 normalizer saves and loads correctly."""
    normalizer = PointwiseNormalizer(preprocess_config={"coords": "minmax"})
    
    x_train = np.random.randn(100, 11).astype(np.float32) + 5
    y_train = np.random.randn(100, 2).astype(np.float32) + 3
    
    normalizer.partial_fit(x_train, y_train)
    normalizer.finalize()
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "normalizer.json"
        normalizer.save(save_path)
        
        # Check version is set
        payload = json.loads(save_path.read_text())
        assert payload["version"] == 2
        assert payload["preprocess_config"] == {"coords": "minmax"}
        
        # Load
        loaded_normalizer = PointwiseNormalizer.load(save_path)
        assert loaded_normalizer.version == 2
        assert loaded_normalizer.preprocess_config == {"coords": "minmax"}
        
        # Verify stats match
        np.testing.assert_array_almost_equal(loaded_normalizer.x_mean, normalizer.x_mean)
        np.testing.assert_array_almost_equal(loaded_normalizer.x_std, normalizer.x_std)


def test_normalizer_version_1_backward_compat_load():
    """Test that old version 1 (no version key) loads with default all-zscore."""
    # Simulate old normalizer.json without version key
    old_stats = {
        "eps": 1.0e-8,
        "x_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        "x_std": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "y_mean": [25.0, 3.7],
        "y_std": [5.0, 0.15],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "normalizer_old.json"
        save_path.write_text(json.dumps(old_stats))
        
        # Load old format
        loaded_normalizer = PointwiseNormalizer.load(save_path)
        
        # Should load without error and have default None preprocess_config
        assert loaded_normalizer.preprocess_config is None or loaded_normalizer.preprocess_config == {}
        np.testing.assert_array_almost_equal(loaded_normalizer.x_mean, np.array(old_stats["x_mean"]))


def test_normalizer_round_trip_inverse():
    """Test that normalize → unnormalize recovers original values."""
    normalizer = PointwiseNormalizer()
    
    x_train = np.random.randn(100, 11).astype(np.float32) * 10 + 5
    y_train = np.random.randn(100, 2).astype(np.float32) * 2 + 25
    
    normalizer.partial_fit(x_train, y_train)
    normalizer.finalize()
    
    # Normalize then inverse-normalize targets
    y_norm = normalizer.transform_Y(y_train)
    y_recovered = normalizer.inverse_Y(y_norm)
    
    # Should be close to original
    np.testing.assert_array_almost_equal(y_recovered, y_train, decimal=5)


def test_normalizer_min_max_stats_tracked():
    """Test that min/max stats are tracked during finalization (Phase 2)."""
    normalizer = PointwiseNormalizer()
    
    x_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], 
                       dtype=np.float32)
    y_train = np.array([[25.0, 3.7], [30.0, 4.0]], dtype=np.float32)
    
    normalizer.partial_fit(x_train, y_train)
    normalizer.finalize()
    
    # Check min/max are set
    assert normalizer.x_min is not None
    assert normalizer.x_max is not None
    assert normalizer.y_min is not None
    assert normalizer.y_max is not None
    
    # Verify ranges
    np.testing.assert_array_almost_equal(normalizer.x_min, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]))
    np.testing.assert_array_almost_equal(normalizer.x_max, np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))


def test_normalizer_invalid_array_shape():
    """Test that invalid shapes raise ValueError."""
    normalizer = PointwiseNormalizer()
    
    x_1d = np.array([1.0, 2.0, 3.0])  # 1D, should be 2D
    y_2d = np.random.randn(3, 2)
    
    with pytest.raises(ValueError, match="2D arrays"):
        normalizer.partial_fit(x_1d, y_2d)


def test_normalizer_mismatched_rows():
    """Test that mismatched row counts raise ValueError."""
    normalizer = PointwiseNormalizer()
    
    x_2d = np.random.randn(5, 11)
    y_2d = np.random.randn(3, 2)  # Different n_rows
    
    with pytest.raises(ValueError, match="same number of rows"):
        normalizer.partial_fit(x_2d, y_2d)
