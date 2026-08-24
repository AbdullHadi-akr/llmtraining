"""Generate or refresh the OP matrix from raw data deterministically."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from ..data.paths import build_config_path, data_raw_dir, load_yaml, op_matrix_path
from ..data.raw_readers import read_inputsignale
from ..schema.columns import CANONICAL_CHANNELS
from ..schema.inputsignale import InputSentinel
from ..schema.mapping import (
    load_op_matrix,
    resolve_inputsignale_source,
    serialise_op_matrix,
)


def _build_op_matrix(source_root: Path) -> dict[str, dict[str, Any]]:
    """Scan source_root/OPxx/OPxx folders and extract canonical channels."""

    result: dict[str, dict[str, Any]] = {}
    op_folders = sorted(
        d for d in source_root.iterdir() if d.is_dir() and d.name.startswith("OP")
    )

    if not op_folders:
        raise ValueError(f"No OP folders found in {source_root}")

    build_config = load_yaml(build_config_path())
    encoding = build_config.get("csv_encoding", "cp1252")

    for op_folder in op_folders:
        op_id = op_folder.name
        op_inner = op_folder / op_id

        if not op_inner.exists():
            continue

        inputsignale_path = resolve_inputsignale_source(op_inner)
        if inputsignale_path is None:
            continue

        try:
            inputsignale = read_inputsignale(inputsignale_path, encoding=encoding)
        except Exception:
            continue

        op_record: dict[str, Any] = {}
        for channel in CANONICAL_CHANNELS:
            raw_value = inputsignale.get(channel)
            if raw_value is None:
                continue
            if isinstance(raw_value, float):
                op_record[channel] = raw_value
            elif isinstance(raw_value, InputSentinel):
                op_record[channel] = raw_value.value

        op_record.setdefault("regime", "train")
        op_record.setdefault("charge_discharge", "mixed")
        result[op_id] = op_record

    if not result:
        raise ValueError(f"No usable OPs found in {source_root}")

    return result


def build_op_matrix(source_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Public API: generate op_matrix dict from source_root (defaults to data_raw/)."""

    if source_root is None:
        source_root = data_raw_dir()

    return _build_op_matrix(source_root)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Generate or validate the OP matrix from raw data."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Path to data_raw/ directory (default: auto-detected).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout without writing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify that regenerated matrix matches committed one (exit 0 on "
            "match, 1 on drift, 2 on error)."
        ),
    )

    args = parser.parse_args()
    source_root = args.source_root or data_raw_dir()

    if not source_root.exists():
        print(f"Error: source root not found: {source_root}", file=sys.stderr)
        return 2

    try:
        regenerated = build_op_matrix(source_root)
    except Exception as exc:
        print(f"Error generating OP matrix: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            committed = load_op_matrix(op_matrix_path())
        except Exception as exc:
            print(f"Error loading committed matrix: {exc}", file=sys.stderr)
            return 2

        if serialise_op_matrix(regenerated) == serialise_op_matrix(committed):
            print("OP matrix is in sync.")
            return 0
        print("OP matrix has drifted from source.", file=sys.stderr)
        return 1

    rendered = yaml.safe_dump(regenerated, sort_keys=True, allow_unicode=True)
    op_matrix_path().write_text(rendered, encoding="utf-8")
    print(f"Wrote OP matrix to {op_matrix_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
