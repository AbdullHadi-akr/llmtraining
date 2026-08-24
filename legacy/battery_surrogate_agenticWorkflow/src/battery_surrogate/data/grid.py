"""Coordinate ingestion and thermal sensor ordering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema.columns import COORDINATE_FILES
from .errors import MissingCoordinatesError
from .paths import coordinates_dir


def _read_coordinate_file(path: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def read_coordinates(
    root: Path | None = None,
    encoding: str = "cp1252",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the three coordinate tables and return xyz, layer, and sensor ids."""

    base_dir = coordinates_dir() if root is None else root
    parts: list[pd.DataFrame] = []
    layers: list[str] = []
    sensor_ids: list[str] = []

    for layer_name, file_name in COORDINATE_FILES:
        path = base_dir / file_name
        if not path.exists():
            raise MissingCoordinatesError(f"Missing coordinate file: {path}")

        frame = _read_coordinate_file(path, encoding=encoding)
        if len(frame.index) != 121:
            raise MissingCoordinatesError(f"Coordinate file {path} must have 121 rows")

        parts.append(frame)
        layers.extend([layer_name] * len(frame.index))
        sensor_ids.extend([f"{layer_name}_{index + 1:03d}" for index in range(len(frame.index))])

    combined = pd.concat(parts, ignore_index=True)
    xyz = combined.iloc[:, :3].to_numpy(dtype=np.float32)
    layer = np.asarray(layers, dtype=object)
    sensor_id = np.asarray(sensor_ids, dtype=object)
    return xyz, layer, sensor_id
