#!/usr/bin/env python3
"""
Phase 1-2 Test Runner with graceful dependency handling
"""
import subprocess
import sys
import os
from pathlib import Path

# Setup paths
work_dir = Path("/mnt/c/Users/M0245635/batterysurrogatemodell/battery_surrogate_agenticWorkflow")
os.chdir(work_dir)

# Add src to PYTHONPATH
sys.path.insert(0, str(work_dir / "src"))
os.environ["PYTHONPATH"] = f"{work_dir / 'src'}:{os.environ.get('PYTHONPATH', '')}"

# Try importing key deps
missing_deps = []
try:
    import torch
except ImportError:
    missing_deps.append("torch")

try:
    import numpy
except ImportError:
    missing_deps.append("numpy")

try:
    import pytest
except ImportError:
    missing_deps.append("pytest")

try:
    import yaml
except ImportError:
    missing_deps.append("pyyaml")

if missing_deps:
    print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
    print("Attempting to install...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + missing_deps,
        timeout=300
    )
    print("Installation completed.\n")

# Now run tests
print("="*70)
print("PHASE 1-2 VERIFICATION - TEST SUITE")
print("="*70)
print()

test_files = [
    ("test_registry.py", "Registry & Dispatch Tests"),
    ("test_train_dispatch_smoke.py::test_train_from_config_mlp_smoke", "CLI Smoke Test"),
    ("test_normalizer_schemes.py", "Normalizer Schemes Tests"),
    ("test_split_coverage.py", "Split Coverage Tests"),
]

results = {}
for test_path, description in test_files:
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Test: {test_path}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/{test_path}", "-v", "--tb=short"],
        capture_output=False
    )
    results[description] = "PASS ✅" if result.returncode == 0 else "FAIL ❌"

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for desc, status in results.items():
    print(f"{desc:.<50} {status}")
print()

all_passed = all("PASS" in status for status in results.values())
if all_passed:
    print("✅ All tests PASSED!")
    sys.exit(0)
else:
    print("❌ Some tests FAILED")
    sys.exit(1)
