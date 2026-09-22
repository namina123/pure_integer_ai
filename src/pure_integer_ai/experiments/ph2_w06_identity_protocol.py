"""W-06 runtime identity primitives with no course or training imports.

The read-only trained graph needs only the frozen direction field identities.
Keeping them in this small module prevents the production query path from
importing W-05/W-06 course catalogs while preserving the exact integer keys
used by the training adapter.
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    VersionBundle,
    concept_identity,
)


W06_NAMESPACE = 60606
W06_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _direction_field_identity(field_kind: int) -> ObjectIdentity:
    return concept_identity(
        (W06_NAMESPACE, 400, field_kind),
        versions=W06_IDENTITY_VERSIONS,
    )


def _direction_value_identity(directionality: int) -> ObjectIdentity:
    return concept_identity(
        (W06_NAMESPACE, 401, directionality),
        versions=W06_IDENTITY_VERSIONS,
    )


def w06_directionality_binding_predicate() -> ObjectIdentity:
    """Return the graph identity for the directionality binding field."""
    return _direction_field_identity(3)


def w06_directionality_value(directionality: int) -> ObjectIdentity:
    """Return one frozen integer direction value identity."""
    return _direction_value_identity(directionality)


__all__ = [
    "W06_IDENTITY_VERSIONS",
    "W06_NAMESPACE",
    "w06_directionality_binding_predicate",
    "w06_directionality_value",
]
