"""Copy raw OP folders into the workflow tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..data.errors import InsufficientDiskError
from ..data.paths import ingest_manifest_path, load_raw_paths_config, data_raw_dir


def _free_disk_ok(target: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(target).free
    if free < required_bytes:
        raise InsufficientDiskError(f"Need {required_bytes} bytes, only {free} bytes free")


def ingest_raw(source_root: Path | None = None) -> dict[str, object]:
    """Copy raw OP folders into data_raw and write an ingest manifest."""

    config = load_raw_paths_config()
    source = Path(source_root or config.get("source_root") or "")
    if not source.exists():
        raise FileNotFoundError(f"Source root not found: {source}")

    target = data_raw_dir()
    target.mkdir(parents=True, exist_ok=True)
    _free_disk_ok(target, 50 * 1024 * 1024)

    copied: list[str] = []
    for op_dir in sorted(source.glob("OP*/OP*")):
        destination = target / op_dir.parent.name / op_dir.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(op_dir, destination)
        copied.append(op_dir.parent.name)

    manifest = {"source_root": str(source), "copied_ops": copied}
    ingest_manifest_path().write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args()
    ingest_raw(args.source_root)


if __name__ == "__main__":
    main()
