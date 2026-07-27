"""Cache key computation and bundle persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.mapping import load_op_matrix, serialise_op_matrix
from .assemble import assemble_op
from .errors import SchemaChangelogError
from .models import OpBundle
from .paths import (
    build_config_path,
    data_cache_dir,
    data_raw_dir,
    coordinates_dir,
    op_matrix_path,
    schema_versions_path,
)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_ignore(path: Path, patterns: list[str]) -> bool:
    rel = path.as_posix()
    name = path.name
    for pattern in patterns:
        if path.match(pattern) or name == pattern or rel.endswith(pattern.replace("**", "")):
            return True
    return False


def _raw_hashes_json(op_id: str, ignored_files: list[str]) -> str:
    op_dir = data_raw_dir() / op_id / op_id
    hashes: dict[str, str] = {}
    if not op_dir.exists():
        return json.dumps(hashes, sort_keys=True)
    for path in sorted(op_dir.rglob("*")):
        if not path.is_file():
            continue
        if _matches_ignore(path.relative_to(op_dir), ignored_files):
            continue
        hashes[path.relative_to(op_dir).as_posix()] = _hash_file(path)
    return json.dumps(hashes, sort_keys=True, separators=(",", ":"))


def _coordinates_hash() -> str:
    coord_dir = coordinates_dir()
    files = [
        coord_dir / "Coordinates - Grid Cell Center.csv",
        coord_dir / "Coordinates - Grid Gehäusewand.csv",
        coord_dir / "Coordinates - Grid JR1 Center.csv",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_yaml_hash() -> str:
    return _hash_file(build_config_path())


def _op_matrix_slice_hash(op_id: str) -> str:
    op_matrix = load_op_matrix(op_matrix_path())
    record = op_matrix.get(op_id, {})
    return _hash_bytes(serialise_op_matrix(record).encode("utf-8"))


def compute_cache_key(op_id: str) -> str:
    """Compute the stable cache key for one OP."""

    ignored_files = []
    config = {}
    if build_config_path().exists():
        with build_config_path().open("r", encoding="utf-8") as handle:
            import yaml

            config = yaml.safe_load(handle) or {}
    ignored_files = list(config.get("ignored_files", []))
    schema_version = int(config.get("schema_version", 1))
    payload = "|".join(
        [
            _raw_hashes_json(op_id, ignored_files),
            _coordinates_hash(),
            _build_yaml_hash(),
            _op_matrix_slice_hash(op_id),
            str(schema_version),
        ]
    )
    return _hash_bytes(payload.encode("utf-8"))


def _ensure_schema_changelog(schema_version: int) -> None:
    schema_versions_file = schema_versions_path()
    text = (
        schema_versions_file.read_text(encoding="utf-8")
        if schema_versions_file.exists()
        else ""
    )
    if f"## v{schema_version}" not in text and f"- {schema_version}" not in text:
        raise SchemaChangelogError(
            f"Schema version {schema_version} is not listed in schema_versions.md"
        )


def _bundle_to_npz_payload(bundle: OpBundle) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "op_id": np.asarray(bundle.op_id),
        "schema_version": np.asarray(bundle.schema_version),
        "cache_key": np.asarray(bundle.cache_key),
        "t_fast": bundle.t_fast,
        "t_slow": bundle.t_slow,
        "bc_V": bundle.bc_V,
        "bc_OCV": bundle.bc_OCV,
        "bc_I": bundle.bc_I,
        "pe_P_loss": bundle.pe_P_loss,
        "T": bundle.T,
        "q_source": bundle.q_source,
        "xyz": bundle.xyz,
        "layer": np.asarray(bundle.layer),
        "sensor_id": np.asarray(bundle.sensor_id),
        "fluid_props": bundle.fluid_props,
        "sim_config_scalar": bundle.sim_config_scalar,
        "sim_config_scalar_names_json": np.asarray(
            json.dumps(list(bundle.sim_config_scalar_names))
        ),
        "sim_config_ts_names_json": np.asarray(json.dumps(list(bundle.sim_config_ts.keys()))),
        "meta_json": np.asarray(json.dumps(bundle.meta, sort_keys=True)),
    }
    for name, (times, values) in bundle.sim_config_ts.items():
        payload[f"sim_config_ts_{name}_t"] = np.asarray(times, dtype=np.float32)
        payload[f"sim_config_ts_{name}_v"] = np.asarray(values, dtype=np.float32)
    return payload


def save_bundle(
    bundle: OpBundle,
    target_dir: Path | None = None,
    format_name: str = "npz",
) -> Path:
    """Write one bundle to disk and return the file path."""

    _ensure_schema_changelog(bundle.schema_version)
    cache_dir = data_cache_dir() if target_dir is None else target_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    if format_name == "zarr":
        try:
            import zarr  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("zarr is not installed") from exc
        path = cache_dir / f"{bundle.op_id}.zarr"
        store = zarr.open_group(str(path), mode="w")
        for key, value in _bundle_to_npz_payload(bundle).items():
            store.array(key, value, overwrite=True)
        return path

    path = cache_dir / f"{bundle.op_id}.npz"
    tmp_path = path.with_suffix(".npz.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **_bundle_to_npz_payload(bundle))
    os.replace(tmp_path, path)
    return path


def load_bundle(path: Path) -> OpBundle:
    """Load a bundle from an NPZ or Zarr file."""

    if path.suffix == ".zarr":
        import zarr  # type: ignore

        store = zarr.open_group(str(path), mode="r")
        arrays = {name: np.asarray(store[name]) for name in store.array_keys()}
    else:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}

    scalar_names = tuple(json.loads(str(arrays["sim_config_scalar_names_json"].item())))
    ts_names = list(json.loads(str(arrays["sim_config_ts_names_json"].item())))
    meta = json.loads(str(arrays["meta_json"].item()))
    sim_config_ts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ts_names:
        sim_config_ts[name] = (
            np.asarray(arrays[f"sim_config_ts_{name}_t"], dtype=np.float32),
            np.asarray(arrays[f"sim_config_ts_{name}_v"], dtype=np.float32),
        )
    return OpBundle(
        op_id=str(arrays["op_id"].item()),
        schema_version=int(arrays["schema_version"].item()),
        cache_key=str(arrays["cache_key"].item()),
        t_fast=np.asarray(arrays["t_fast"], dtype=np.float32),
        t_slow=np.asarray(arrays["t_slow"], dtype=np.float32),
        bc_V=np.asarray(arrays["bc_V"], dtype=np.float32),
        bc_OCV=np.asarray(arrays["bc_OCV"], dtype=np.float32),
        bc_I=np.asarray(arrays["bc_I"], dtype=np.float32),
        pe_P_loss=np.asarray(arrays["pe_P_loss"], dtype=np.float32),
        T=np.asarray(arrays["T"], dtype=np.float32),
        q_source=np.asarray(arrays["q_source"], dtype=np.float32),
        xyz=np.asarray(arrays["xyz"], dtype=np.float32),
        layer=np.asarray(arrays["layer"]),
        sensor_id=np.asarray(arrays["sensor_id"]),
        fluid_props=np.asarray(arrays["fluid_props"], dtype=np.float32),
        sim_config_scalar=np.asarray(arrays["sim_config_scalar"], dtype=np.float32),
        sim_config_scalar_names=scalar_names,
        sim_config_ts=sim_config_ts,
        meta=meta,
    )


def cache_index_path() -> Path:
    return data_cache_dir() / "cache_index.json"


def build_report_path() -> Path:
    return data_cache_dir() / "build_report.json"


def load_cache_index() -> dict[str, Any]:
    path = cache_index_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache_index(index: dict[str, Any]) -> None:
    path = cache_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_build_report(report: dict[str, Any]) -> None:
    path = build_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def build_one(op_id: str, format_name: str = "npz") -> Path:
    """Assemble and write one OP bundle, then update the cache index."""

    bundle = assemble_op(op_id)
    cache_key = compute_cache_key(op_id)
    object.__setattr__(bundle, "cache_key", cache_key)
    path = save_bundle(bundle, format_name=format_name)
    index = load_cache_index()
    index[op_id] = {
        "cache_key": cache_key,
        "schema_version": bundle.schema_version,
        "format": format_name,
        "path": str(path),
    }
    write_cache_index(index)
    write_build_report({"op_id": op_id, "cache_key": cache_key, "path": str(path)})
    return path


def build_many(op_ids: list[str], format_name: str = "npz") -> list[Path]:
    """Build several OP bundles."""

    return [build_one(op_id, format_name=format_name) for op_id in op_ids]
