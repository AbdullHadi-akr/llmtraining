"""Check whether the cache fits the selected storage format."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.paths import data_cache_dir


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def size_audit(op_ids: list[str] | None = None) -> dict[str, float]:
    """Measure cached bundle sizes in megabytes."""

    cache_dir = data_cache_dir()
    result: dict[str, float] = {}
    for path in sorted(cache_dir.glob("*.*z*")):
        result[path.stem] = _file_size_mb(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op_ids", nargs="*")
    args = parser.parse_args()
    print(size_audit(args.op_ids))


if __name__ == "__main__":
    main()
