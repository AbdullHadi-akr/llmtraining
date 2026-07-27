"""Generate or refresh the OP matrix from raw data deterministically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ..data.paths import data_raw_dir, op_matrix_path
from ..data.raw_readers import read_inputsignale
from ..schema.columns import CANONICAL_CHANNELS
from ..schema.inputsignale import InputSentinel
from ..schema.mapping import serialise_op_matrix, load_op_matrix


def _build_op_matrix(source_root: Path) -> dict[str, dict[str, Any]]:
    """Scan source_root/OPxx/OPxx folders and extract canonical channels."""
    
    result: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    
    # Find all OPxx folders
    op_folders = sorted([d for d in source_root.iterdir() if d.is_dir() and d.name.startswith("OP")])
    
    if not op_folders:
        raise ValueError(f"No OP folders found in {source_root}")
    
    # Read build config once
    from ..data.paths import build_config_path, load_yaml
    build_config = load_yaml(build_config_path())
    encoding = build_config.get("csv_encoding", "cp1252")
    
    for op_folder in op_folders:
        op_id = op_folder.name
        op_inner = op_folder / op_id
        
        if not op_inner.exists():
            skipped.append(f"{op_id} (missing inner folder)")
            continue
        
        # Find Inputsignale CSV
        inputsignale_files = sorted(op_inner.glob("*_Inputsignale.csv"))
        if not inputsignale_files:
            skipped.append(f"{op_id} (missing Inputsignale.csv)")
            continue
        
        try:
            inputsignale = read_inputsignale(inputsignale_files[0], encoding=encoding)
        except Exception as e:
            skipped.append(f"{op_id} (error reading Inputsignale: {e})")
            continue
        
        # Extract canonical channels using control flow from A1
        op_record: dict[str, Any] = {}
        
        for channel in CANONICAL_CHANNELS:
            raw_value = inputsignale.get(channel)
            
            if raw_value is None:
                # Not in inputsignale, skip for now (will be filled by derived/fallback at assemble time)
                continue
            
            # Apply control flow from A1
            if isinstance(raw_value, float):
                # Numeric scalar
                op_record[channel] = raw_value
            elif isinstance(raw_value, InputSentinel):
                # Sentinel: emit the enum value string
                op_record[channel] = raw_value.value
            else:
                # Should not happen (read_inputsignale returns only float | InputSentinel)
                raise TypeError(f"Unexpected value type for {channel}: {type(raw_value)}")
        
        # Add other required matrix fields (regime, charge_discharge from OP metadata)
        # Default to train regime, mixed charge_discharge
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
        help="Verify that regenerated matrix matches committed one (exit 0 on match, 1 on drift, 2 on error).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite op_matrix.yaml without prompting.",
    )
    
    args = parser.parse_args()
    
    source_root = args.source_root or data_raw_dir()
    
    if not source_root.exists():
        print(f"Error: source root not found: {source_root}", file=sys.stderr)
        return 2
    
    try:
        regenerated = build_op_matrix(source_root)
    except Exception as e:
        print(f"Error generating OP matrix: {e}", file=sys.stderr)
        return 2
    
    if args.check:
        # Drift guard: canonical-semantic compare
        try:
            committed = load_op_matrix(op_matrix_path())
        except Exception as e:
            print(f"Error loading committed matrix: {e}", file=sys.stderr)
            return 2
        
        if serialise_op_matrix(regenerated) == serialise_op_matrix(committed):
            print("OP matrix is in sync.")
            return 0
        else:
            print("OP matrix has drifted from source.", file=sys.stderr)
            return 1
    
    if args.dry_run:
        # Print to stdout
        print(yaml.safe_dump(regenerated, sort_keys=True, allow_unicode=True))
        return 0
    
    # Write to file
    output_path = op_matrix_path()
    
    if output_path.exists() and not args.force:
        # Check if content differs
        committed = load_op_matrix(output_path)
        if serialise_op_matrix(regenerated) == serialise_op_matrix(committed):
            print(f"OP matrix already up-to-date: {output_path}")
            return 0
        else:
            print(
                f"OP matrix would be overwritten. Use --force to confirm: {output_path}",
                file=sys.stderr,
            )
            return 1
    
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(regenerated, f, sort_keys=True, allow_unicode=True)
    
    print(f"OP matrix written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
