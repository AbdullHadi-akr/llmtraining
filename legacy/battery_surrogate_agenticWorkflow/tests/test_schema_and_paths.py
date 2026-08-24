from battery_surrogate.schema.inputsignale import InputSentinel, parse_inputsignale_value
from battery_surrogate.data.paths import deep_merge


def test_parse_inputsignale_numeric() -> None:
    assert parse_inputsignale_value("3.5", "c_rate") == 3.5


def test_parse_inputsignale_sentinel() -> None:
    assert parse_inputsignale_value("File Table", "cell_current") is InputSentinel.FILE_TABLE


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
