"""Helpers that map raw OP folders to their companion files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_op_matrix(path: Path) -> dict[str, Any]:
    """Load the OP matrix YAML file as a plain dictionary."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def get_op_record(op_matrix: dict[str, Any], op_id: str) -> dict[str, Any]:
    """Return the stored record for one OP id."""

    record = op_matrix.get(op_id, {})
    if not isinstance(record, dict):
        raise TypeError(f"OP matrix entry for {op_id} must be a mapping")
    return record


def resolve_profile_source(op_dir: Path, channel: str) -> Path | None:
    """Find the profile file for one channel inside one OP folder."""

    patterns: list[str]
    if channel == "cell_current":
        patterns = ["*_CellCurrent(t).csv"]
    elif channel == "fluid_mass_flow":
        patterns = ["*_FluidMassFlow(t).csv"]
    elif channel == "fluid_inlet_temp":
        patterns = ["*_FluidInletTemperature(t).csv", "*_FluidInletTemperatur(t).xlsx"]
    else:
        return None

    for pattern in patterns:
        matches = sorted(op_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_inputsignale_source(op_dir: Path) -> Path | None:
    """Find the Inputsignale CSV, accepting spacing variants from raw exports."""

    patterns = [
        "*_Inputsignale.csv",
        "*_Inputsignale*.csv",
        "*_Input Signale.csv",
        "*_Input Signale*.csv",
    ]
    for pattern in patterns:
        matches = sorted(op_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def profile_source_name(path: Path | None) -> str | None:
    """Return the file name for metadata storage."""

    return None if path is None else path.name


def serialise_op_matrix(op_matrix: dict[str, Any]) -> str:
    """Return a stable JSON view of the OP matrix for hashing or debug output."""

    return json.dumps(op_matrix, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
