## OpBundle Contract Specification

This document is the authoritative reference for every field in `OpBundle`, the frozen interface between the raw data pipeline and downstream models.

### Source-of-Truth Hierarchy

| File | Owns | Reference |
|------|------|-----------|
| `data/models.py` | Field names and Python types | `@dataclass OpBundle` |
| `data/assemble.py` | Population semantics, units, `meta` contents | `assemble_op()` function body |
| `data/cache.py` | On-disk shape and dtype round-trip | `_bundle_to_npz_payload()`, `load_bundle()` |
| `schema/mapping.py` | Canonical serialization of OP matrix | `serialise_op_matrix()` |

Every row below cites the file(s) that own its fact.

---

### Core Fields (Direct from Simulation/Grid)

| Field | Type | Shape | Dtype | Unit | Meaning | Source |
|-------|------|-------|-------|------|---------|--------|
| `op_id` | `str` | scalar | — | — | OP identifier (e.g., `"OP01"`) | `assemble_op()` line: `op_id` parameter |
| `schema_version` | `int` | scalar | — | — | Data schema version (currently `2`) | `build.yaml` → `assemble_op()` |
| `cache_key` | `str` | scalar | — | — | Stable hash of input config (empty until cached) | `cache.py` → `compute_cache_key()` |
| `t_fast` | `np.ndarray` | `(n_fast,)` | float32 | s | Time axis for FMU1 electrical signals (Batemo); read from `*_Batemo FMU1.csv` `Physical Time` column | `data/raw_readers.py` → `read_batemo_fmu1()` |
| `t_slow` | `np.ndarray` | `(n_slow,)` | float32 | s | Time axis for heat source; read from `*_Heat Source.csv` `Physical Time` column | `data/raw_readers.py` → `read_heat_source()` |

---

### Boundary Conditions (Fast Axis, `n_fast` samples)

| Field | Type | Shape | Dtype | Unit | Meaning | Source |
|-------|------|-------|-------|------|---------|--------|
| `bc_V` | `np.ndarray` | `(n_fast,)` | float32 | V | Battery cell voltage from FMU1 monitor | `data/raw_readers.py` → `read_batemo_fmu1()` column `bc_V Monitor` |
| `bc_OCV` | `np.ndarray` | `(n_fast,)` | float32 | V | Open-circuit voltage from FMU1 monitor | `data/raw_readers.py` → `read_batemo_fmu1()` column `bc_OCV Monitor` |
| `bc_I` | `np.ndarray` | `(n_fast,)` | float32 | A | Cell current from FMU1 monitor | `data/raw_readers.py` → `read_batemo_fmu1()` column `bc_I Monitor` |
| `pe_P_loss` | `np.ndarray` | `(n_fast,)` | float32 | W | Power electronics loss from FMU1 monitor | `data/raw_readers.py` → `read_batemo_fmu1()` column `pe_P_loss Monitor` |

---

### Thermal Fields

| Field | Type | Shape | Dtype | Unit | Meaning | Source |
|-------|------|-------|-------|------|---------|--------|
| `T` | `np.ndarray` | `(n_fast, n_sensors)` | float32 | °C | Grid temperatures, one column per sensor (grid + wall + JR layers concatenated) | `data/assemble.py` → reads three `*_T_grid_*.csv` files and stacks columns |
| `q_source` | `np.ndarray` | `(n_slow, 3)` | float32 | W | Heat source for three monitors: `[jr1_w, jr2_w, total_w]` from `*_Heat Source.csv` | `data/raw_readers.py` → `read_heat_source()` |

---

### Spatial (Grid) Metadata

| Field | Type | Shape | Dtype | Unit | Meaning | Source |
|-------|------|-------|-------|------|---------|--------|
| `xyz` | `np.ndarray` | `(n_sensors, 3)` | float32 | m | 3D coordinates of each sensor grid point | `data/grid.py` → `read_coordinates()` |
| `layer` | `np.ndarray` | `(n_sensors,)` | str or int | — | Layer label for each sensor (e.g., `"cc"`, `"g"`, `"jr1c"` for Cell Center, Gehäusewand, JR1 Center) | `data/grid.py` → `read_coordinates()` |
| `sensor_id` | `np.ndarray` | `(n_sensors,)` | int | — | Unique ID for each sensor within its layer | `data/grid.py` → `read_coordinates()` |
| `fluid_props` | `np.ndarray` | `(1, 3)` | float32 | — | Fluid property row (typically density, specific heat, thermal conductivity) from `*_Fluidstoffwerte.csv` | `data/raw_readers.py` → `read_fluidstoffwerte()` |

---

### Configuration & Simulation Parameters

| Field | Type | Shape/Keys | Dtype | Unit | Meaning | Source |
|-------|------|-----------|-------|------|---------|--------|
| `sim_config_scalar` | `np.ndarray` | `(n_scalar,)` | float32 | various | Numeric scalar configuration values in column order matching `sim_config_scalar_names` | `data/assemble.py` → loop over `CANONICAL_CHANNELS` |
| `sim_config_scalar_names` | `tuple[str, ...]` | tuple of `n_scalar` strings | — | — | Names of scalar columns in `sim_config_scalar`, e.g., `("c_rate", "cell_current", "fluid_initial_temp", …)` | `data/assemble.py` → accumulates during channel processing |
| `sim_config_ts` | `dict[str, tuple[np.ndarray, np.ndarray]]` | — | — | — | Time-series inputs: maps channel name (e.g., `"cell_current"`, `"fluid_inlet_temp"`) to `(times, values)` tuple | `data/assemble.py` → `read_time_series_input()` for profiles, `read_module_test_data()` for test data |

**Invariant**: The order of `sim_config_scalar_names` must exactly match column order of `sim_config_scalar`; downstream code relies on this correspondence (no re-indexing).

---

### Metadata Dictionary

The `meta` dict captures derived state and source provenance. All keys below are populated by `assemble_op()`:

| Key | Type | Meaning | Source |
|-----|------|---------|--------|
| `op_id` | `str` | OP identifier (redundant with top-level field) | `assemble_op()` parameter |
| `charge_discharge` | `str` | Charge/discharge regime (default: `"mixed"`, from `op_matrix.yaml` or OP record) | `data/assemble.py` → `op_record.get("charge_discharge")` |
| `profile_flags` | `dict[str, bool]` | For each profile channel (`"cell_current"`, `"fluid_inlet_temp"`, `"fluid_mass_flow"`, `"c_rate"`), whether a profile file was loaded (`True`) or scalar used (`False`) | `data/assemble.py` → set to `True` when sentinel resolved to file |
| `sim_config_sentinels` | `dict[str, dict[str, str]]` | Maps sentinel-resolved channels to source metadata: `{"channel": {"sentinel": "value", "source_file": "name.csv"}}` | `data/assemble.py` → populated when InputSentinel encountered |
| `sim_config_fallbacks` | `dict[str, dict[str, str]]` | Maps ModuleTestData fallbacks to their resolution source, including `source`, `module_test_file`, `source_kind`, and `fallback_value` | `data/assemble.py` → populated when a ModuleTestData column is missing and the OP matrix provides the fallback |
| `sim_config_derived` | `dict[str, str]` | Maps derived-channel names to how they were computed, e.g., `{"soc_start": "derive_from_ocv"}` or `{"c_rate": "label:0.5C"}` | `data/assemble.py` → set when `InputSentinel.DERIVE_FROM_OCV` or fallback label used |
| `sim_config_ts_names` | `list[str]` | Keys of `sim_config_ts` in resolution order (for round-tripping) | `data/assemble.py` → `list(sim_config_ts.keys())` |
| `c_rate` | `float \| str \| None` | C-rate value (numeric if scalar, string label if `MODUL_TEST_DATA` fallback to label, `None` if derived) | `op_matrix.yaml` or fallback during assembly |
| `source_file_hashes` | `dict[str, str]` | Cryptographic hashes of source files (computed at cache time) | `data/cache.py` → `_compute_source_hashes()` |
| `schema_version` | `int` | Schema version (currently `2`) | `build.yaml` |

---

### Time Axis Correspondence

- **Fast axis (`t_fast`, `n_fast ≈ 14,000–35,000` samples)**: FMU1 electrical simulation with ~1 s sampling.
  - Used by: `bc_V`, `bc_OCV`, `bc_I`, `pe_P_loss`, `T`
  
- **Slow axis (`t_slow`, `n_slow ≈ 350–3,500` samples)**: Heat-source external logging with ~100–140 s sampling.
  - Used by: `q_source`

When training models, ensure resampling/interpolation if both axes are used together.

---

### Canonical Channels (from schema/columns.py)

Every OP must account for all seven canonical channels in one of four ways (per `data/assemble.py` logic):

1. **Numeric scalar** → stored in `sim_config_scalar`, name in `sim_config_scalar_names`
2. **Time-series profile** (sentinel `FILE_TABLE`, `TEMPERATURPROFIL`, `VOLUMENSTROMPROFIL`) → stored in `sim_config_ts`, flag in `profile_flags`
3. **Derived** (sentinel `DERIVE_FROM_OCV`) → marked in `sim_config_derived`, not in scalars or profiles
4. **Module-test data** (sentinel `MODUL_TEST_DATA` with fallback) → stored in `sim_config_ts` or `sim_config_derived`

Channels: `"c_rate"`, `"cell_current"`, `"fluid_initial_temp"`, `"fluid_inlet_temp"`, `"fluid_mass_flow"`, `"soc_start"`, `"solid_initial_temp"`

---

### Shape Consistency Rules

- `T.shape[0]` must equal `t_fast.shape[0]` (both index along fast time axis)
- `q_source.shape[0]` must equal `t_slow.shape[0]` (both index along slow time axis)
- `T.shape[1]` must equal `xyz.shape[0]` (sensor count consistency)
- `sim_config_scalar.shape[0]` must equal `len(sim_config_scalar_names)` (scalar count consistency)
- `fluid_props.shape` must be `(1, 3)` (fixed fluid property row)

---

### On-Disk Representation (NPZ format)

When `save_bundle()` writes to NPZ, every array is stored with its name as the key, plus a `meta.json` for the metadata dict.  
See `data/cache.py` → `_bundle_to_npz_payload()` and `load_bundle()` for the round-trip contract.

