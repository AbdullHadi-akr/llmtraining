from pathlib import Path

import numpy as np

from battery_surrogate.data.filter import clip_array, clip_temperature, clip_voltage
from battery_surrogate.data.paths import deep_merge, load_yaml
from battery_surrogate.data.time_axes import as_float32_axis, is_strictly_increasing
from battery_surrogate.schema.inputsignale import InputSentinel, parse_inputsignale_value


def test_parse_inputsignale_comma_decimal() -> None:
    assert parse_inputsignale_value("3,5", "c_rate") == 3.5


def test_parse_inputsignale_soc_start_alias() -> None:
    assert parse_inputsignale_value("nicht def.", "soc_start") is InputSentinel.DERIVE_FROM_OCV


def test_parse_inputsignale_unknown_value_raises() -> None:
    try:
        parse_inputsignale_value("mystery", "c_rate")
    except ValueError as exc:
        assert "Unknown Inputsignale value" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown Inputsignale value")


def test_load_yaml_missing_file_returns_empty_dict() -> None:
    assert load_yaml(Path("does-not-exist.yaml")) == {}


def test_deep_merge_preserves_base_mapping() -> None:
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}

    merged = deep_merge(base, override)

    assert merged == {"a": {"b": 1, "c": 2}}


def test_clip_array_limits_values() -> None:
    values = np.array([-2.0, 0.5, 5.0], dtype=np.float32)

    clipped = clip_array(values, 0.0, 1.0)

    assert np.array_equal(clipped, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_clip_temperature_uses_configured_limits() -> None:
    values = np.array([-60.0, 10.0, 200.0], dtype=np.float32)

    clipped = clip_temperature(values, (-50.0, 150.0))

    assert np.array_equal(clipped, np.array([-50.0, 10.0, 150.0], dtype=np.float32))


def test_clip_voltage_uses_configured_limits() -> None:
    values = np.array([-1.0, 2.5, 6.0], dtype=np.float32)

    clipped = clip_voltage(values, (0.0, 5.0))

    assert np.array_equal(clipped, np.array([0.0, 2.5, 5.0], dtype=np.float32))


def test_as_float32_axis_flattens_to_one_dimension() -> None:
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)

    axis = as_float32_axis(values)

    assert axis.shape == (4,)


def test_is_strictly_increasing_rejects_duplicates() -> None:
    values = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)

    assert is_strictly_increasing(values) is False