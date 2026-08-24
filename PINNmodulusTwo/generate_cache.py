#!/usr/bin/env python3
"""Generate .npz cache files for OP05 and OP07 for PINNmodulusTwo."""

import dataclasses
import sys
from pathlib import Path

# Add battery_surrogate to path
sys.path.insert(0, str(Path(__file__).parent.parent / "battery_surrogate_agenticWorkflow" / "src"))

from battery_surrogate.data.assemble import assemble_op
from battery_surrogate.data.cache import save_bundle, compute_cache_key

def generate_cache_for_ops(op_ids):
    """Generate .npz cache files for the given OP IDs."""
    output_dir = Path(__file__).parent / "data_cache"
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
    ops_to_process = ["OP05", "OP06", "OP07"]
    print(f"Generating .npz cache files for: {ops_to_process}")
    generate_cache_for_ops(ops_to_process)
    print("\n✓ Done! Cache files created in PINNmodulusTwo/data_cache/")
