"""Tests for split coverage validation (Phase 2)."""

import pytest

from battery_surrogate.model.split import validate_coverage


def test_validate_coverage_with_no_stats():
    """Test that validate_coverage returns empty list when stats are None (deferred validation)."""
    split = {
        "train": ["OP01", "OP02"],
        "val": ["OP03"],
        "test": ["OP04"],
    }
    
    warnings = validate_coverage(split, normalizer_stats=None)
    assert warnings == [], "Should return empty list when stats are None"


def test_validate_coverage_returns_list():
    """Test that validate_coverage always returns a list (never raises)."""
    split = {
        "train": ["OP01"],
        "val": ["OP02"],
        "test": ["OP03"],
    }
    
    # Valid stats
    stats = {
        "x_mean": [5.0] * 11,
        "x_std": [1.0] * 11,
        "x_min": [3.0] * 11,
        "x_max": [7.0] * 11,
    }
    
    warnings = validate_coverage(split, normalizer_stats=stats)
    assert isinstance(warnings, list), "Should always return a list"


def test_validate_coverage_with_missing_stats():
    """Test that validate_coverage handles missing stats gracefully (returns empty list)."""
    split = {
        "train": ["OP01"],
        "val": ["OP02"],
        "test": ["OP03"],
    }
    
    # Incomplete stats (missing required keys)
    stats = {}
    
    warnings = validate_coverage(split, normalizer_stats=stats)
    assert isinstance(warnings, list), "Should return list even with incomplete stats"


def test_validate_coverage_non_raising():
    """Test that validate_coverage never raises, even with invalid OPs (Phase 2 guarantee)."""
    split = {
        "train": ["OP01"],
        "val": ["INVALID_OP_THAT_DOES_NOT_EXIST"],
        "test": ["OP03"],
    }
    
    stats = {
        "x_mean": [5.0] * 11,
        "x_std": [1.0] * 11,
    }
    
    # Should NOT raise, even though OP cannot be loaded
    warnings = validate_coverage(split, normalizer_stats=stats)
    assert isinstance(warnings, list)
