"""Build cached model-ready bundles for one or more OPs."""

from __future__ import annotations

import argparse

from ..data.cache import build_many
from ..data.paths import build_config_path, load_yaml, op_matrix_path
from ..schema.mapping import load_op_matrix


def build_cache(op_ids: list[str] | None = None) -> list[str]:
    """Build cache files for the requested OP ids."""

    if op_ids is None:
        op_ids = sorted(load_op_matrix(op_matrix_path()).keys())
    format_name = load_yaml(build_config_path()).get("format", "npz")
    paths = build_many(op_ids, format_name=format_name)
    return [str(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op_ids", nargs="*")
    args = parser.parse_args()
    build_cache(args.op_ids or None)


if __name__ == "__main__":
    main()
