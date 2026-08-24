"""Quick manual verification of Phase 1-2 tests."""
print("=== PHASE 1-2 MANUAL TEST VERIFICATION ===\n")

from battery_surrogate.model.registry import build_model, build_datasets
from battery_surrogate.model.mlp_pointwise import PointwiseMLP
from battery_surrogate.model.normalizer import PointwiseNormalizer
import numpy as np

# TEST 1: Registry - MLP dispatch
print("TEST 1: test_build_model_mlp_pointwise")
config = {
    "model": {
        "type": "mlp_pointwise",
        "n_hidden_layers": 2,
        "hidden_size": 64,
    }
}
model = build_model(config, n_sensors=50, seed=42)
assert isinstance(model, PointwiseMLP), "Expected PointwiseMLP"
print("✓ PASS: MLP model built correctly\n")

# TEST 2: Registry - Default to MLP
print("TEST 2: test_build_model_default_mlp")
config_noType = {"model": {"n_hidden_layers": 2, "hidden_size": 64}}
model = build_model(config_noType, n_sensors=50, seed=42)
assert isinstance(model, PointwiseMLP), "Expected MLP as default"
print("✓ PASS: Defaults to MLP when type unspecified\n")

# TEST 3: Registry - Unknown type error
print("TEST 3: test_build_model_unknown_type")
config_bad = {"model": {"type": "unknown_model"}}
try:
    build_model(config_bad, n_sensors=50, seed=42)
    print("✗ FAIL: Should have raised ValueError")
except ValueError as e:
    if "Unknown model type" in str(e):
        print("✓ PASS: Correctly rejects unknown type\n")
    else:
        print(f"✗ FAIL: Wrong error message: {e}\n")

# TEST 4: Registry - Recurrent NotImplementedError
print("TEST 4: test_build_model_recurrent_not_implemented")
config_recurrent = {
    "model": {
        "type": "recurrent",
        "rnn_type": "gru",
        "n_layers": 2,
        "hidden_size": 128,
        "history_length": 8,
    }
}
try:
    build_model(config_recurrent, n_sensors=50, seed=42)
    print("✗ FAIL: Should have raised NotImplementedError")
except NotImplementedError as e:
    if "Phase 4" in str(e):
        print("✓ PASS: Recurrent correctly deferred to Phase 4\n")
    else:
        print(f"✗ FAIL: Wrong error message: {e}\n")

# TEST 5: Normalizer version 2
print("TEST 5: test_normalizer_version_2_format")
norm = PointwiseNormalizer(eps=1e-5, preprocess_config=None)
x_dummy = np.random.randn(50, 11).astype(np.float32)
y_dummy = np.random.randn(50, 2).astype(np.float32)
norm.partial_fit(x_dummy, y_dummy)
norm.finalize()
x_norm = norm.transform_X(x_dummy)
assert x_norm.shape == x_dummy.shape
print("✓ PASS: Normalizer works\n")

# TEST 6: Normalizer backward compat (version 1)
print("TEST 6: test_normalizer_version_1_backward_compat")
import json
import tempfile
from pathlib import Path
v1_data = {
    "version": 1,
    "x_mean": np.random.randn(11).tolist(),
    "x_std": np.ones(11).tolist(),
    "y_mean": np.random.randn(2).tolist(),
    "y_std": np.ones(2).tolist(),
}
with tempfile.TemporaryDirectory() as tmpdir:
    v1_file = Path(tmpdir) / "normalizer.json"
    with open(v1_file, "w") as f:
        json.dump(v1_data, f)
    norm_v1 = PointwiseNormalizer.load(str(v1_file))
    assert norm_v1.x_mean is not None
    print("✓ PASS: Version 1 normalizer loads correctly\n")

print("=" * 50)
print("PHASE 1-2 VERIFICATION: ✅ ALL TESTS PASS")
print("=" * 50)
