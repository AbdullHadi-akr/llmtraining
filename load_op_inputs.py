from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable

import numpy as np


def _find_op_folder(dataset_root: Path, op_id: int) -> Path:
    """Find the folder that contains files for a given OP id."""
    candidates = [
        dataset_root / f"OP{op_id}" / f"OP{op_id}",
        dataset_root / f"OP{op_id}",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find folder for OP{op_id}. Tried: {[str(c) for c in candidates]}"
    )


def _read_csv_as_array(file_path: Path) -> tuple[list[str], np.ndarray]:
    """Read a CSV with a header row and return (headers, numeric rows)."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {file_path}")

    headers = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        # Return empty 2D array with correct number of columns.
        return headers, np.empty((0, len(headers)), dtype=float)

    data = np.array(data_rows, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return headers, data


def load_op_input_and_fluid(
    op_ids: Iterable[int],
    dataset_root: str | Path = r"c:\Users\M0245635\Downloads",
) -> dict[int, dict[str, np.ndarray | list[str]]]:
    """
    Load Input Signale and Fluidstoffwerte for selected OPs.

    Example
    -------
    result = load_op_input_and_fluid([1])
    op1_inputs = result[1]["input_signals"]
    op1_fluids = result[1]["fluidstoffwerte"]
    """
    root = Path(dataset_root)
    out: dict[int, dict[str, np.ndarray | list[str]]] = {}

    for op_id in op_ids:
        op_folder = _find_op_folder(root, op_id)
        input_file = op_folder / f"OP{op_id}_Input Signale.csv"
        fluid_file = op_folder / f"OP{op_id}_Fluidstoffwerte.csv"

        if not input_file.exists():
            raise FileNotFoundError(f"Missing file: {input_file}")
        if not fluid_file.exists():
            raise FileNotFoundError(f"Missing file: {fluid_file}")

        input_headers, input_data = _read_csv_as_array(input_file)
        fluid_headers, fluid_data = _read_csv_as_array(fluid_file)

        out[op_id] = {
            "input_headers": input_headers,
            "input_signals": input_data,
            "fluid_headers": fluid_headers,
            "fluidstoffwerte": fluid_data,
        }

    return out


if __name__ == "__main__":
    selected_ops = [1]
    data = load_op_input_and_fluid(selected_ops)

    for op_id in selected_ops:
        print(f"OP{op_id} input_signals shape: {data[op_id]['input_signals'].shape}")
        print(data[op_id]["input_signals"])
        print(f"OP{op_id} fluidstoffwerte shape: {data[op_id]['fluidstoffwerte'].shape}")
        print(data[op_id]["fluidstoffwerte"])
