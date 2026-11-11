"""Extract guarded role/literal generation parts from licensed train teachers."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_identity
from pure_integer_ai.experiments.ph2_authored_relation_schema import LICENSE_ID, AuthoredRelationSeed
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    PART_CLAIM_PREDICATE, PART_CLAIM_ROLE, PART_LITERAL, MultiClaimCompositionVariant,
)


def compile_train_conjoined_generation(
        seeds: tuple[AuthoredRelationSeed, ...], compound_surface: str, *,
        separator: str, variant_id: int, source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        ) -> MultiClaimCompositionVariant:
    """Keep trained grammar intervals and predicates; strip every teacher filler.

    The caller owns the audited train permission. No held-out input, expected
    answer or proposition-specific result table is accepted by this compiler.
    """
    if (type(seeds) is not tuple or len(seeds) < 2
            or type(compound_surface) is not str or not compound_surface
            or type(separator) is not str or not separator
            or license_identity != tuple(map(ord, LICENSE_ID))
            or any(seed.split != "train" or seed.label_owner != "teacher"
                   or seed.sample_role != "support" for seed in seeds)):
        raise ValueError("conjoined generation requires licensed train teachers")
    if compound_surface.split(separator) != [seed.surface for seed in seeds]:
        raise ValueError("compound generation surface differs from its train teachers")
    parts = []
    for claim, seed in enumerate(seeds, 1):
        parts.append((PART_CLAIM_PREDICATE, claim,
                      authored_relation_identity(seed.relation_kind).stable_key()))
        coordinates = {binding.endpoint_id: ordinal
                       for ordinal, binding in enumerate(seed.bindings, 1)}
        cursor = 0
        for endpoint in sorted(seed.endpoints, key=lambda item: item.start):
            if (endpoint.start < cursor
                    or seed.surface[endpoint.start:endpoint.end] != endpoint.surface_fragment):
                raise ValueError("train generation endpoint intervals are incomplete")
            literal = seed.surface[cursor:endpoint.start]
            if literal:
                parts.append((PART_LITERAL, tuple(map(ord, literal))))
            parts.append((PART_CLAIM_ROLE, claim, coordinates[endpoint.endpoint_id]))
            cursor = endpoint.end
        if cursor < len(seed.surface):
            parts.append((PART_LITERAL, tuple(map(ord, seed.surface[cursor:]))))
        if claim < len(seeds):
            parts.append((PART_LITERAL, tuple(map(ord, separator))))
    return MultiClaimCompositionVariant(
        variant_id, len(seeds), tuple(parts), source_identity, license_identity)


__all__ = ["compile_train_conjoined_generation"]
