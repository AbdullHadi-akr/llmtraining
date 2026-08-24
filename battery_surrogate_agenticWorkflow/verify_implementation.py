#!/usr/bin/env python3
"""Comprehensive verification of implementation phases."""

import sys
sys.path.insert(0, 'src')

print("=== IMPLEMENTATION STATUS REPORT ===\n")

# Phase 1
print("[Phase 1] Registry + CLI")
try:
    from battery_surrogate.cli.train import train_from_config
    from battery_surrogate.model.registry import build_model, build_datasets
    print("  ✓ train_from_config")
    print("  ✓ registry.build_model")
    print("  ✓ registry.build_datasets")
except Exception as e:
    print(f"  ✗ {e}")
print()

# Phase 2
print("[Phase 2] Normalizer Extension")
try:
    from battery_surrogate.model.normalizer import PointwiseNormalizer
    from battery_surrogate.model.split import validate_coverage
    print("  ✓ PointwiseNormalizer (extended)")
    print("  ✓ validate_coverage")
except Exception as e:
    print(f"  ✗ {e}")
print()

# Phase 3
print("[Phase 3] Sequence Features")
try:
    from battery_surrogate.model.features_sequence import SequenceFeatureBuilder
    from battery_surrogate.model.dataset_sequence import SequenceDataset
    print("  ✓ SequenceFeatureBuilder")
    print("  ✓ SequenceDataset")
except Exception as e:
    print(f"  ✗ {e}")
print()

# Phase 4
print("[Phase 4] Recurrent Model")
try:
    from battery_surrogate.model.recurrent_pointwise import RecurrentPointwise
    print("  ✓ RecurrentPointwise")
except Exception as e:
    print(f"  ✗ {e}")
print()

# Phase 5
print("[Phase 5] Sequence Trainer & Evaluation")
try:
    from battery_surrogate.model.trainer_sequence import train_sequence_model
    print("  ✓ train_sequence_model")
except Exception as e:
    print(f"  ✗ {e}")
    
try:
    from battery_surrogate.model.evaluate_sequence import evaluate_sequence_model, history_length_benchmark
    print("  ✓ evaluate_sequence_model")
    print("  ✓ history_length_benchmark")
except Exception as e:
    print(f"  ✗ Missing: evaluate_sequence (needed for Phase 5 completion)")
print()

# Phase 6
print("[Phase 6] Integration")
try:
    from battery_surrogate.model import build_model
    from battery_surrogate.cli import train_from_config as train_cli
    print("  ✓ Exports in model/__init__")
    print("  ✓ Exports in cli/__init__")
except Exception as e:
    print(f"  ✗ {e}")
print()

# Phase 7
print("[Phase 7] Notebook")
import os
nb_exists = os.path.exists('notebooks/model_launcher.ipynb')
print(f"  {'✓' if nb_exists else '✗'} model_launcher.ipynb {'exists' if nb_exists else 'MISSING'}")
print()

print("=== END REPORT ===")
