"""Preregistered, fresh-data protocol for the one-shot v4 evaluation.

Importing this package never generates data or creates an output directory.
The formal runner must first load a frozen ``release_v4_locked`` digest and
then call the data helpers explicitly.
"""

from .data import (
    TEST_V4_ROLES,
    ComparisonSource,
    audit_freshness,
    dataset_contract,
    default_comparison_sources,
    derive_seed,
    generate_locked_datasets,
    query_keys,
    query_sha256,
    validate_dataset_contract,
)

__all__ = [
    "TEST_V4_ROLES",
    "ComparisonSource",
    "audit_freshness",
    "dataset_contract",
    "default_comparison_sources",
    "derive_seed",
    "generate_locked_datasets",
    "query_keys",
    "query_sha256",
    "validate_dataset_contract",
]
