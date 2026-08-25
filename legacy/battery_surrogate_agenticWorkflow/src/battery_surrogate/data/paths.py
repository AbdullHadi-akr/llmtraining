"""Path helpers for the workflow package."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the workflow package root directory."""

    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Return the workflow folder that holds data and configs."""

    return project_root()


def data_raw_dir() -> Path:
    """Return the raw-data folder inside the workflow tree."""

    return data_root() / "data_raw"


def data_cache_dir() -> Path:
    """Return the cache folder inside the workflow tree."""

    return data_root() / "data_cache"


def coordinates_dir() -> Path:
    """Return the preferred coordinate folder inside the workflow tree."""

    return data_root() / "coordinates"


def docs_dir() -> Path:
    """Return the documentation folder inside the workflow tree."""

    return data_root() / "docs"


def configs_dir() -> Path:
    """Return the configuration folder inside the workflow tree."""

    return data_root() / "configs"


def build_config_path() -> Path:
    """Return the build configuration file path."""

    return data_root() / "build.yaml"


def op_matrix_path() -> Path:
    """Return the OP matrix file path."""

    return data_root() / "op_matrix.yaml"


def ingest_manifest_path() -> Path:
    """Return the ingest manifest file path."""

    return data_root() / "ingest_manifest.json"


def schema_versions_path() -> Path:
    """Return the schema-version changelog path."""

    return docs_dir() / "schema_versions.md"


def raw_paths_default_path() -> Path:
    """Return the committed raw-path defaults file path."""

    return configs_dir() / "data" / "raw_paths.default.yaml"


def raw_paths_local_path() -> Path:
    """Return the machine-local raw-path override file path."""

    return configs_dir() / "data" / "raw_paths.local.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dictionaries without mutating either one."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_paths_config() -> dict[str, Any]:
    """Load the default raw-path config and overlay the local override."""

    return deep_merge(load_yaml(raw_paths_default_path()), load_yaml(raw_paths_local_path()))
