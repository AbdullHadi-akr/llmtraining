#!/usr/bin/env python3
"""Pipeline smoke test for MLP and Recurrent models."""

import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("PIPELINE SMOKE TEST")
print("=" * 60)
print()

# Test 1: Imports
print("[1] Testing imports...")
try:
    from battery_surrogate.cli.train import train_from_config
    from battery_surrogate.model.registry import build_model, build_datasets
    from battery_surrogate.model.normalizer import PointwiseNormalizer
    from battery_surrogate.model.recurrent_pointwise import RecurrentPointwise
    from battery_surrogate.model.evaluate_sequence import (
        evaluate_sequence_model,
        history_length_benchmark,
        compute_lookback_seconds,
    )
    print("    ✓ All imports successful")
except Exception as e:
    print(f"    ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Registry dispatch
print("\n[2] Testing registry dispatch...")
try:
    mlp_config = {
        "model": {"type": "mlp_pointwise", "n_hidden_layers": 1, "hidden_size": 16},
        "seed": 42,
    }
    model = build_model(mlp_config, n_sensors=363, seed=42)
    print(f"    ✓ MLP model created: {type(model).__name__}")
    
    rec_config = {
        "model": {"type": "recurrent", "rnn_type": "gru", "n_layers": 1, "hidden_size": 16, "history_length": 4},
        "seed": 42,
    }
    model_rec = build_model(rec_config, n_sensors=363, seed=42)
    print(f"    ✓ Recurrent model created: {type(model_rec).__name__}")
except Exception as e:
    print(f"    ✗ Registry error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Recurrent forward + rollout
print("\n[3] Testing recurrent forward/rollout...")
try:
    import torch
    features = torch.randn(2, 50, 11)
    history = torch.randn(2, 50, 8)  # 4 lags * 2 outputs = 8
    output = model_rec(features, history)
    print(f"    ✓ Forward: input (2, 50, 11+8) → output {tuple(output.shape)}")
    
    # Rollout
    features_single = torch.randn(50, 11)
    y0 = torch.randn(2)
    preds = model_rec.rollout(features_single, y0)
    print(f"    ✓ Rollout: input (50, 11) + y0 (2,) → preds {tuple(preds.shape)}")
except Exception as e:
    print(f"    ✗ Forward/rollout error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Normalizer
print("\n[4] Testing normalizer...")
try:
    import numpy as np
    norm = PointwiseNormalizer()
    X = np.random.randn(100, 11).astype(np.float32)
    Y = np.random.randn(100, 2).astype(np.float32)
    norm.partial_fit(X, Y)
    norm.finalize()
    X_norm = norm.transform_X(X)
    Y_norm = norm.transform_Y(Y)
    X_back = norm.inverse_X(X_norm)
    Y_back = norm.inverse_Y(Y_norm)
    assert np.allclose(X, X_back, atol=1e-5), "X round-trip failed"
    assert np.allclose(Y, Y_back, atol=1e-5), "Y round-trip failed"
    print("    ✓ Normalizer round-trip successful")
except Exception as e:
    print(f"    ✗ Normalizer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SMOKE TEST PASSED")
print("=" * 60)
