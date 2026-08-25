"""Assemble one OP's raw inputs into a frozen bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schema.columns import CANONICAL_CHANNELS
from ..schema.inputsignale import InputSentinel
from ..schema.mapping import (
    get_op_record,
    load_op_matrix,
    resolve_inputsignale_source,
    profile_source_name,
    resolve_profile_source,
)
from .errors import MissingOpError, UnknownInputsignaleValueError
from .filter import clip_temperature, clip_voltage
from .grid import read_coordinates
from .models import OpBundle
from .paths import build_config_path, data_raw_dir, load_yaml, op_matrix_path
from .raw_readers import (
    read_batemo_fmu1,
    read_fluidstoffwerte,
    read_heat_source,
    read_inputsignale,
    read_module_test_data,
    read_t_grid,
    read_time_series_input,
)
from .time_axes import as_float32_axis, is_strictly_increasing


def _single_glob(op_dir: Path, pattern: str) -> Path:
    matches = sorted(op_dir.glob(pattern))
    if not matches:
        raise MissingOpError(f"Missing file for pattern {pattern!r} in {op_dir}")
    return matches[0]


def _read_build_config() -> dict[str, Any]:
    return load_yaml(build_config_path())


def _read_schema_version() -> int:
    config = _read_build_config()
    return int(config.get("schema_version", 1))


def _read_op_matrix() -> dict[str, Any]:
    return load_op_matrix(op_matrix_path())


def _record_scalar(
    scalar_names: list[str],
    scalar_values: list[float],
    channel: str,
    value: float,
) -> None:
    scalar_names.append(channel)
    scalar_values.append(float(value))


def _record_module_test_fallback(
    fallback_provenance: dict[str, dict[str, str]],
    channel: str,
    fallback: str | float,
    module_test_file: str,
    source_kind: str,
) -> None:
    fallback_provenance[channel] = {
        "source": "op_matrix",
        "module_test_file": module_test_file,
        "source_kind": source_kind,
        "fallback_value": str(fallback),
    }


def assemble_op(op_id: str, root: Path | None = None) -> OpBundle:
    """Read raw files for one OP and combine them into one bundle."""

    root_dir = data_raw_dir() if root is None else root
    op_dir = root_dir / op_id / op_id
    if not op_dir.exists():
        raise MissingOpError(f"Missing OP folder: {op_dir}")

    build_config = _read_build_config()
    op_matrix = _read_op_matrix()
    op_record = get_op_record(op_matrix, op_id)
    schema_version = int(build_config.get("schema_version", 1))
    encoding = build_config.get("csv_encoding", "cp1252")

    xyz, layer, sensor_id = read_coordinates()
    fmu1 = read_batemo_fmu1(_single_glob(op_dir, "*_Batemo FMU1.csv"), encoding=encoding)
    heat_source = read_heat_source(_single_glob(op_dir, "*_Heat Source.csv"), encoding=encoding)
    fluid_props = read_fluidstoffwerte(
        _single_glob(op_dir, "*_Fluidstoffwerte.csv"),
        encoding=encoding,
    )

    t_fast = as_float32_axis(fmu1["physical_time_s"].to_numpy())
    t_slow = as_float32_axis(heat_source["physical_time_s"].to_numpy())
    if not is_strictly_increasing(t_fast):
        raise ValueError(f"Fast time axis is not strictly increasing for {op_id}")
    if not is_strictly_increasing(t_slow):
        raise ValueError(f"Slow time axis is not strictly increasing for {op_id}")

    bc_V = fmu1["bc_V"].to_numpy(dtype=np.float32)
    bc_OCV = fmu1["bc_OCV"].to_numpy(dtype=np.float32)
    bc_I = fmu1["bc_I"].to_numpy(dtype=np.float32)
    pe_P_loss = fmu1["pe_P_loss"].to_numpy(dtype=np.float32)

    clip_limits = build_config.get("filters", {})
    if "V_clip_V" in clip_limits:
        bc_V = clip_voltage(bc_V, tuple(clip_limits["V_clip_V"]))

    if "T_clip_C" in clip_limits:
        t_clip = tuple(clip_limits["T_clip_C"])
    else:
        t_clip = (-50.0, 150.0)

    t_grid_parts = {
        layer_name: read_t_grid(
            _single_glob(op_dir, f"*_T_grid_{layer_name}_i.csv"),
            layer=layer_name,
            encoding=encoding,
        )
        for layer_name in ("cc", "g", "jr1c")
    }
    t_columns: list[np.ndarray] = []
    for layer_name in ("cc", "g", "jr1c"):
        frame = t_grid_parts[layer_name]
        cols = [column for column in frame.columns if column != "physical_time_s"]
        t_columns.extend([frame[column].to_numpy(dtype=np.float32) for column in cols])
    T = np.column_stack(t_columns)
    T = clip_temperature(T, t_clip)
    q_source = heat_source[["jr1_w", "jr2_w", "total_w"]].to_numpy(dtype=np.float32)

    inputsignale_path = resolve_inputsignale_source(op_dir)
    if inputsignale_path is None:
        raise MissingOpError(f"Missing Inputsignale file in {op_dir}")
    inputsignale = read_inputsignale(inputsignale_path, encoding=encoding)
    scalar_names: list[str] = []
    scalar_values: list[float] = []
    sim_config_ts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    derived: dict[str, str] = {}
    sentinels: dict[str, dict[str, str]] = {}
    fallback_provenance: dict[str, dict[str, str]] = {}
    resolved_c_rate = op_record.get("c_rate")
    profile_flags = {
        channel: False
        for channel in ("c_rate", "cell_current", "fluid_inlet_temp", "fluid_mass_flow")
    }

    module_test_path = resolve_profile_source(op_dir, "fluid_inlet_temp")
    module_test_data = None

    for channel in CANONICAL_CHANNELS:
        value = inputsignale.get(channel, op_record.get(channel))
        if isinstance(value, float):
            _record_scalar(scalar_names, scalar_values, channel, value)
            continue

        if value == InputSentinel.DERIVE_FROM_OCV:
            derived[channel] = InputSentinel.DERIVE_FROM_OCV.value
            continue

        if value in (
            InputSentinel.FILE_TABLE,
            InputSentinel.TEMPERATURPROFIL,
            InputSentinel.VOLUMENSTROMPROFIL,
        ):
            profile_path = resolve_profile_source(op_dir, channel)
            if profile_path is None:
                raise MissingOpError(f"Missing profile file for {channel} in {op_dir}")
            times, series = read_time_series_input(
                profile_path,
                channel=channel,
                encoding=encoding,
            )
            sim_config_ts[channel] = (times, series)
            sentinels[channel] = {
                "sentinel": value.value,
                "source_file": profile_source_name(profile_path) or "",
            }
            profile_flags[channel] = True
            continue

        if value == InputSentinel.MODUL_TEST_DATA:
            if module_test_data is None:
                # Accept both canonical and suffixed exports, e.g. *_ModuleTestData(in).csv
                module_test_path = _single_glob(op_dir, "*_ModuleTestData*.csv")
                module_test_data = read_module_test_data(module_test_path, encoding=encoding)
            if channel in module_test_data:
                sim_config_ts[channel] = module_test_data[channel]
                sentinels[channel] = {
                    "sentinel": value.value,
                    "source_file": module_test_path.name,
                }
                profile_flags[channel] = True
            else:
                fallback = op_record.get(channel)
                if isinstance(fallback, (int, float)):
                    _record_scalar(scalar_names, scalar_values, channel, float(fallback))
                    _record_module_test_fallback(
                        fallback_provenance,
                        channel,
                        float(fallback),
                        module_test_path.name,
                        "scalar",
                    )
                elif isinstance(fallback, str) and fallback.strip():
                    meta_label = fallback.strip()
                    derived[channel] = f"label:{meta_label}"
                    _record_module_test_fallback(
                        fallback_provenance,
                        channel,
                        meta_label,
                        module_test_path.name,
                        "label",
                    )
                    if channel == "c_rate":
                        resolved_c_rate = meta_label
                else:
                    raise UnknownInputsignaleValueError(
                        f"OP {op_id} needs a fallback for {channel} but none was found"
                    )
            continue

        if value is None:
            raise UnknownInputsignaleValueError(f"OP {op_id} has no value for {channel}")

        raise UnknownInputsignaleValueError(
            f"Unsupported Inputsignale value for {channel}: {value!r}"
        )

    meta = {
        "op_id": op_id,
        "charge_discharge": op_record.get("charge_discharge", "mixed"),
        "profile_flags": profile_flags,
        "sim_config_sentinels": sentinels,
        "sim_config_fallbacks": fallback_provenance,
        "sim_config_derived": derived,
        "sim_config_scalar_names": list(scalar_names),
        "sim_config_ts_names": list(sim_config_ts.keys()),
        "c_rate": resolved_c_rate,
        "source_file_hashes": {},
        "schema_version": schema_version,
    }

    accounted = set(scalar_names) | set(sim_config_ts) | set(derived)
    missing = [channel for channel in CANONICAL_CHANNELS if channel not in accounted]
    if missing:
        raise UnknownInputsignaleValueError(f"OP {op_id} missing channels: {missing!r}")

    bundle = OpBundle(
        op_id=op_id,
        schema_version=schema_version,
        cache_key="",
        t_fast=t_fast,
        t_slow=t_slow,
        bc_V=bc_V,
        bc_OCV=bc_OCV,
        bc_I=bc_I,
        pe_P_loss=pe_P_loss,
        T=T,
        q_source=q_source,
        xyz=xyz,
        layer=layer,
        sensor_id=sensor_id,
        fluid_props=fluid_props,
        sim_config_scalar=np.asarray(scalar_values, dtype=np.float32),
        sim_config_scalar_names=tuple(scalar_names),
        sim_config_ts=sim_config_ts,
        meta=meta,
    )
    return bundle
