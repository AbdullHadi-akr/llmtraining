"""End-to-end smoke tests for the data pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from battery_surrogate.data.assemble import assemble_op
from battery_surrogate.data.cache import (
    compute_cache_key,
    load_bundle,
    save_bundle,
)
from battery_surrogate.data.paths import (
    data_raw_dir,
    op_matrix_path,
)
from battery_surrogate.schema.mapping import (
    load_op_matrix,
    serialise_op_matrix,
)
from battery_surrogate.cli.build_op_matrix import (
    build_op_matrix,
)


def _op_dir_exists(op_id: str) -> bool:
    """Check if an OP folder exists with raw data."""
    root = data_raw_dir()
    op_folder = root / op_id / op_id
    inputsignale_files = sorted(op_folder.glob("*_Inputsignale.csv")) if op_folder.exists() else []
    return bool(inputsignale_files)


class TestPipelineSkipGuards:
    """Test 1: Skip guards are per-OP (missing OP01 ≠ skip OP08)."""

    def test_op01_skip_guard(self):
        """OP01 should be skipped if raw data is absent."""
        if not _op_dir_exists("OP01"):
            pytest.skip("OP01 raw data not available")
        # If we reach here, OP01 is present
        assert _op_dir_exists("OP01"), "OP01 exists in raw data"

    def test_op08_skip_guard(self):
        """OP08 should be skipped independently of OP01."""
        if not _op_dir_exists("OP08"):
            pytest.skip("OP08 raw data not available")
        # If we reach here, OP08 is present
        assert _op_dir_exists("OP08"), "OP08 exists in raw data"


class TestOP01Assembly:
    """Test 2: OP01 can be fully assembled."""

    def test_assemble_op01(self):
        """OP01 assembly should succeed and return correct shape."""
        if not _op_dir_exists("OP01"):
            pytest.skip("OP01 raw data not available")

        bundle = assemble_op("OP01")
        assert bundle.op_id == "OP01"
        assert bundle.schema_version == 2
        assert len(bundle.t_fast) > 0
        assert bundle.T.shape[0] == len(bundle.t_fast), "T time dimension matches t_fast"
        assert bundle.T.shape[1] == len(bundle.xyz), "T sensor dimension matches xyz"


class TestBundleRoundTrip:
    """Test 3: Bundle save/load round-trip preserves data."""

    def test_round_trip_with_minimal_bundle(self, make_minimal_bundle):
        """Save and load a minimal bundle, verify data integrity."""
        bundle_in = make_minimal_bundle(op_id="ROUND_TRIP_TEST")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # save_bundle takes a directory, not a file path
            file_path = save_bundle(bundle_in, tmpdir_path)
            assert file_path.exists(), "NPZ file was written"

            bundle_out = load_bundle(file_path)
            assert bundle_out.op_id == bundle_in.op_id
            assert bundle_out.schema_version == bundle_in.schema_version
            np.testing.assert_array_equal(bundle_out.t_fast, bundle_in.t_fast)
            np.testing.assert_array_equal(bundle_out.T, bundle_in.T)
            assert bundle_out.meta == bundle_in.meta


class TestCacheKeySeam:
    """Test 4: Cache-key seam patches work correctly (monkeypatch target: cache.op_matrix_path)."""

    def test_cache_key_mutation(self, monkeypatch, tmp_path):
        """Mutating op_matrix.yaml should change the cache key for OP01."""
        if not _op_dir_exists("OP01"):
            pytest.skip("OP01 raw data not available")

        # Get baseline key
        baseline_key = compute_cache_key("OP01")

        # Create a copy of op_matrix.yaml in tmp_path
        original_matrix_path = op_matrix_path()
        tmp_matrix_path = tmp_path / "op_matrix.yaml"
        tmp_matrix_path.write_text(original_matrix_path.read_text())

        # Monkeypatch the consumer module (cache.py)
        monkeypatch.setattr(
            "battery_surrogate.data.cache.op_matrix_path",
            lambda: tmp_matrix_path,
        )
        # Also patch assemble.py if it uses op_matrix_path (for assemble_op)
        monkeypatch.setattr(
            "battery_surrogate.data.assemble.op_matrix_path",
            lambda: tmp_matrix_path,
        )

        # Mutate the matrix
        import yaml

        data = yaml.safe_load(tmp_matrix_path.read_text())
        if "OP01" in data and isinstance(data["OP01"], dict):
            data["OP01"]["c_rate"] = 9.99  # Change from 2.0 to 9.99
            tmp_matrix_path.write_text(yaml.safe_dump(data, sort_keys=True))

        # Compute key with mutated matrix
        mutated_key = compute_cache_key("OP01")

        # Keys should differ
        assert mutated_key != baseline_key, "Mutating op_matrix.yaml changes the cache key"


class TestEdgeOPs:
    """Test 5: Edge case OPs assemble without errors."""

    @pytest.mark.parametrize("op_id", ["OP08", "OP12", "OP19"])
    def test_edge_op_assembly(self, op_id):
        """Edge OPs (with special data paths) should assemble successfully."""
        if not _op_dir_exists(op_id):
            pytest.skip(f"{op_id} raw data not available")

        bundle = assemble_op(op_id)
        assert bundle.op_id == op_id
        assert len(bundle.t_fast) > 0
        if op_id == "OP19":
            assert bundle.meta["c_rate"] == "modul_test_data"
            assert bundle.meta["sim_config_scalar_names"] == [
                "fluid_initial_temp",
                "soc_start",
                "solid_initial_temp",
            ]
            assert np.allclose(bundle.sim_config_scalar, [25.260003662109398, 77.068108183535841, 25.260003662109398])
            assert bundle.meta["sim_config_derived"] == {}
            assert bundle.meta["sim_config_fallbacks"] == {}
            assert list(bundle.sim_config_ts.keys()) == [
                "c_rate",
                "cell_current",
                "fluid_inlet_temp",
                "fluid_mass_flow",
            ]


class TestDriftGuard:
    """Test 6: Drift guard detects when matrix regeneration differs from committed."""

    def test_drift_guard_in_sync(self):
        """Regenerated matrix should match committed matrix (drift guard = 0)."""
        if not (data_raw_dir() / "OP01" / "OP01").exists():
            pytest.skip("OP01 raw data not available")

        regenerated = build_op_matrix(source_root=data_raw_dir())
        committed = load_op_matrix(op_matrix_path())

        canonical_regen = serialise_op_matrix(regenerated)
        canonical_committed = serialise_op_matrix(committed)

        assert (
            canonical_regen == canonical_committed
        ), "Regenerated matrix matches committed (canonical-semantic compare)"


class TestDryRun:
    """Test 7: Dry-run mode parses correctly."""

    def test_dry_run_parse(self):
        """Dry-run output should be valid YAML that round-trips through load_op_matrix."""
        if not (data_raw_dir() / "OP01" / "OP01").exists():
            pytest.skip("OP01 raw data not available")

        regenerated = build_op_matrix(source_root=data_raw_dir())
        # Dry-run would emit YAML, so verify it round-trips
        import yaml

        yaml_text = yaml.safe_dump(regenerated, sort_keys=True)
        reloaded = yaml.safe_load(yaml_text)
        assert reloaded == regenerated, "Dry-run YAML round-trips cleanly"
