"""Pytest fixtures and shared test utilities."""

from __future__ import annotations

import numpy as np
import pytest

from battery_surrogate.data.models import OpBundle


@pytest.fixture
def make_minimal_bundle():
    """Create a minimal in-memory OpBundle for testing (no raw-data dependency)."""

    def _bundle(op_id: str = "TEST01") -> OpBundle:
        """Build a small, valid OpBundle with correct dtypes and shapes."""
        n_fast = 100
        n_slow = 10
        n_sensors = 50
        n_scalar = 3

        return OpBundle(
            op_id=op_id,
            schema_version=2,
            cache_key="test_key_placeholder",
            t_fast=np.arange(n_fast, dtype=np.float32),
            t_slow=np.arange(n_slow, dtype=np.float32) * 10.0,
            bc_V=np.random.randn(n_fast).astype(np.float32) + 3.7,
            bc_OCV=np.random.randn(n_fast).astype(np.float32) + 3.6,
            bc_I=np.random.randn(n_fast).astype(np.float32) * 100,
            pe_P_loss=np.random.randn(n_fast).astype(np.float32) * 10,
            T=np.random.randn(n_fast, n_sensors).astype(np.float32) + 25.0,
            q_source=np.random.randn(n_slow, 3).astype(np.float32) * 100,
            xyz=np.random.randn(n_sensors, 3).astype(np.float32),
            # Use Unicode strings (U4) instead of object dtype for pickle-free NPZ serialization
            layer=np.array(["cc"] * (n_sensors // 2) + ["g"] * (n_sensors // 2)),
            # Use Unicode strings for sensor IDs (e.g., "cc_001", "g_026")
            sensor_id=np.array([f"{'cc' if i < n_sensors // 2 else 'g'}_{i % (n_sensors // 2) + 1:03d}" 
                               for i in range(n_sensors)]),
            fluid_props=np.array([[1000.0, 4180.0, 0.6]], dtype=np.float32),
            sim_config_scalar=np.array([2.0, 316.0, 25.0], dtype=np.float32),
            sim_config_scalar_names=("c_rate", "cell_current", "fluid_initial_temp"),
            sim_config_ts={},
            meta={
                "op_id": op_id,
                "charge_discharge": "mixed",
                "profile_flags": {
                    "c_rate": False,
                    "cell_current": False,
                    "fluid_inlet_temp": False,
                    "fluid_mass_flow": False,
                },
                "sim_config_sentinels": {},
                "sim_config_derived": {},
                "sim_config_scalar_names": ["c_rate", "cell_current", "fluid_initial_temp"],
                "sim_config_ts_names": [],
                "c_rate": 2.0,
                "source_file_hashes": {},
                "schema_version": 2,
            },
        )

    return _bundle
