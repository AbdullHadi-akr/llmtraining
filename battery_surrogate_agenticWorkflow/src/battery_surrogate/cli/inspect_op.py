"""Print a short summary for one OP bundle."""

from __future__ import annotations

import argparse
import json

from ..data.loader import load_op


def inspect_op(op_id: str) -> dict[str, object]:
    """Load one OP and return a readable summary dictionary."""

    bundle = load_op(op_id)
    return {
        "op_id": bundle.op_id,
        "schema_version": bundle.schema_version,
        "cache_key": bundle.cache_key,
        "scalar_names": list(bundle.sim_config_scalar_names),
        "profile_names": list(bundle.sim_config_ts),
        "shape_T": list(bundle.T.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op_id")
    args = parser.parse_args()
    print(json.dumps(inspect_op(args.op_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
