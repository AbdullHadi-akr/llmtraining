#!/usr/bin/env python3
"""Build the .npz OP bundles PINNmodulusTwo trains on.

Usage:
    python3 PINNmodulusTwo/generate_cache.py            # OP05 OP06 OP07
    python3 PINNmodulusTwo/generate_cache.py OP08 OP09  # specific OPs

Writes to the top-level ``data_cache/``. The raw-CSV assembly still comes from
the legacy workflow -- this is the only place the active code depends on it.
"""

import dataclasses
import sys
from pathlib import Path

# The raw-CSV assembly still lives in the legacy workflow; this is the one place
# the active code depends on it. Both layouts are accepted so a checkout that has
# not been restructured keeps working.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_CANDIDATES = (
    _PROJECT_ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "src",
    _PROJECT_ROOT / "battery_surrogate_agenticWorkflow" / "src",
)
for _src in _SRC_CANDIDATES:
    if _src.exists():
        sys.path.insert(0, str(_src))
        break
else:
    raise SystemExit(
        "cannot find the legacy battery_surrogate sources needed to build the cache.\n"
        "  searched:\n"
        + "".join(f"    {c}\n" for c in _SRC_CANDIDATES)
    )

from battery_surrogate.data.assemble import assemble_op
from battery_surrogate.data.cache import save_bundle, compute_cache_key

def generate_cache_for_ops(op_ids):
    """Generate .npz cache files for the given OP IDs."""
    # Preferred location: shared and top-level, matching data._CACHE_CANDIDATES.
    output_dir = _PROJECT_ROOT / "data_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for op_id in op_ids:
        print(f"\n=== Processing {op_id} ===")
        try:
            # Assemble the OP from raw CSVs
            bundle = assemble_op(op_id)
            
            # Compute cache key (bundle is frozen; use dataclasses.replace)
            cache_key = compute_cache_key(op_id)
            bundle = dataclasses.replace(bundle, cache_key=cache_key)
            
            # Save to .npz
            path = save_bundle(bundle, target_dir=output_dir)
            print(f"✓ Created: {path}")
            print(f"  T shape: {bundle.T.shape}")
            print(f"  xyz shape: {bundle.xyz.shape}")
            print(f"  t_fast points: {len(bundle.t_fast)}")
            
        except Exception as e:
            print(f"✗ Error processing {op_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # OP ids on the command line, e.g.:
    #   python3 PINNmodulusTwo/generate_cache.py OP08
    ops_to_process = sys.argv[1:] or ["OP05", "OP06", "OP07"]
    print(f"Generating .npz cache files for: {ops_to_process}")
    generate_cache_for_ops(ops_to_process)
    print(f"\n✓ Done! Cache files created in {_PROJECT_ROOT / 'data_cache'}")
