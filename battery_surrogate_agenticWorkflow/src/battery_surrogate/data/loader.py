"""Public loader for cached OP bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema.mapping import load_op_matrix
from .cache import build_one, compute_cache_key, load_bundle, load_cache_index
from .models import OpBundle
from .paths import data_cache_dir, op_matrix_path


@dataclass(frozen=True)
class OpAvailability:
    """Availability summary for one OP regime."""

    available: set[str]
    missing_from_cache: set[str]
    missing_from_config: set[str]
    stale_schema: set[str]
    stale_cache_key: set[str]


def _cache_path_for(op_id: str) -> Path:
    index = load_cache_index()
    cached = index.get(op_id, {})
    path = cached.get("path")
    if path:
        return Path(path)
    return data_cache_dir() / f"{op_id}.npz"


def _op_ids_from_matrix(regime: str) -> set[str]:
    op_matrix = load_op_matrix(op_matrix_path())
    if regime == "all":
        return set(op_matrix)
    return {
        op_id
        for op_id, record in op_matrix.items()
        if record.get("regime", "train") == regime
    }


def list_ops(regime: str = "all") -> OpAvailability:
    """List available OPs and classify any stale cache entries."""

    op_ids = _op_ids_from_matrix(regime)
    index = load_cache_index()
    available: set[str] = set()
    missing_from_cache: set[str] = set()
    missing_from_config: set[str] = set()
    stale_schema: set[str] = set()
    stale_cache_key: set[str] = set()

    for op_id in op_ids:
        path = _cache_path_for(op_id)
        if not path.exists():
            missing_from_cache.add(op_id)
            continue
        cached = index.get(op_id, {})
        bundle = load_bundle(path)
        expected_key = compute_cache_key(op_id)
        if bundle.schema_version != int(cached.get("schema_version", bundle.schema_version)):
            stale_schema.add(op_id)
            continue
        if bundle.cache_key != expected_key:
            stale_cache_key.add(op_id)
            continue
        available.add(op_id)

    for op_id in index:
        if op_id not in op_ids:
            missing_from_config.add(op_id)

    return OpAvailability(
        available=available,
        missing_from_cache=missing_from_cache,
        missing_from_config=missing_from_config,
        stale_schema=stale_schema,
        stale_cache_key=stale_cache_key,
    )


def load_op(op_id: str) -> OpBundle:
    """Load one OP bundle from cache or rebuild it if the cache is stale."""

    path = _cache_path_for(op_id)
    if not path.exists():
        build_one(op_id)
        path = _cache_path_for(op_id)
    bundle = load_bundle(path)
    expected_key = compute_cache_key(op_id)
    if bundle.cache_key != expected_key:
        build_one(op_id)
        path = _cache_path_for(op_id)
        bundle = load_bundle(path)
    return bundle
