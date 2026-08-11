"""Bridge learned predicate replacements into FT11 raw-question matching."""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    RawQuestionFeatureCatalog,
    raw_question_feature_catalog,
    run_raw_question_feature_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    LearnedPredicateAliasBridge,
    LearnedPredicateAliasOption,
    LearnedPredicateAliasResolution,
    LearnedPredicateAliasRoute,
    RawQuestionPredicateAliasAnswerResult,
    RawQuestionPredicateAliasMatch,
    W03W04W05RawQuestionAliasError,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
    RawQuestionRequest,
)


PREDICATE_ALIAS_BRIDGE_SHA256 = (
    "e1c893e17874e462198fe930ae28f3dfed926a0641751b75d0ffe08dc51d9556")
PREDICATE_ALIAS_ANSWER_SHA256S = (
    "2f2d95d22dcc86469c41087826553f37b0b1c2bc90f88eb4d8adf7b861ff234a",
    "5ab622c99f7be7dc84065b8bf2c72737c8ca259e77588a3e884281f54cd4c05c",
    "ca15c4fefb1880e57aad001b8d009dc8bfaeef9084e381025132446fafd073a8",
    "f43127f5b61aa7ee8e52d74c699a0a51508e700f287e0f615ba7715c6bfd43d4",
)
PREDICATE_ALIAS_EXPRESSION_BOUNDARY = (
    ("explicit_predicate_alias", "SUPPORTED_BY_TWO_SOURCE_BOUND_ROUTES"),
    ("missing_alias_route", "UNKNOWN"),
    ("non_equivalent_alias_routes", "CLARIFY"),
    ("implicit_predicate", "UNKNOWN_UNTIL_A_LEARNED_STRUCTURE_EXISTS"),
    ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _record_key(record) -> tuple[int, ...]:
    return record.stable_key.stable_key()


def _pair_by_key(batch: EvaluationPublicBatch) -> dict[tuple[int, ...], object]:
    return {_record_key(item.observation): item for item in batch.pairs}


def _source_by_key(batch: EvaluationPublicBatch) -> dict[object, object]:
    return {item.record.stable_key: item.record for item in batch.source_records}


def _payload(pair, *, stage: str) -> dict[str, object]:
    observation = pair.observation
    value = observation.typed_payload.to_value()
    if (observation.w_stage != stage or not isinstance(value, dict)):
        raise W03W04W05RawQuestionAliasError(
            f"predicate alias {stage} payload drifted")
    return value


def _one_predicate_surface(
        construction: RawQuestionConstruction,
        ) -> str:
    values = tuple(
        item.surface for item in construction.segments
        if item.kind == "PREDICATE")
    if len(values) != 1:
        raise W03W04W05RawQuestionAliasError(
            "predicate alias construction lacks one predicate")
    return values[0]


def _route_for_vertical(
        feature_catalog: RawQuestionFeatureCatalog,
        vertical,
        ) -> LearnedPredicateAliasRoute:
    if vertical.link is None:
        raise W03W04W05RawQuestionAliasError(
            "predicate alias route requires a vertical link")
    w03_pairs = _pair_by_key(feature_catalog.w03_batch)
    w04_pairs = _pair_by_key(feature_catalog.w04_batch)
    current_w04 = w04_pairs.get(vertical.link.w04_observation_key)
    if current_w04 is None or current_w04.observation.supersedes_key is None:
        raise W03W04W05RawQuestionAliasError(
            "current predicate has no learned replacement predecessor")
    previous_w04 = w04_pairs.get(
        current_w04.observation.supersedes_key.stable_key())
    if previous_w04 is None:
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement predecessor is unavailable")
    if (len(current_w04.observation.prerequisite_keys) != 1
            or len(previous_w04.observation.prerequisite_keys) != 1):
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement W04 prerequisites drifted")
    current_w03 = w03_pairs.get(
        current_w04.observation.prerequisite_keys[0].stable_key())
    previous_w03 = w03_pairs.get(
        previous_w04.observation.prerequisite_keys[0].stable_key())
    if current_w03 is None or previous_w03 is None:
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement W03 prerequisite is unavailable")
    if (current_w03.observation.supersedes_key
            != previous_w03.observation.stable_key):
        raise W03W04W05RawQuestionAliasError(
            "W03 and W04 replacement chains are not parallel")

    old_w03 = _payload(previous_w03, stage="W-03")
    new_w03 = _payload(current_w03, stage="W-03")
    old_w04 = _payload(previous_w04, stage="W-04")
    new_w04 = _payload(current_w04, stage="W-04")
    alias_surface = old_w03.get("surface")
    predicate_surface = new_w03.get("surface")
    alias_context = old_w03.get("context")
    predicate_context = new_w03.get("context")
    sense_label = old_w03.get("candidate_sense")
    primitive = old_w04.get("candidate_primitive")
    if (not isinstance(alias_surface, str)
            or not isinstance(predicate_surface, str)
            or not isinstance(alias_context, str)
            or not isinstance(predicate_context, str)
            or not isinstance(sense_label, str)
            or not isinstance(primitive, dict)):
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement lexical projection drifted")
    expected_parallel = (
        old_w04.get("surface_form") == alias_surface,
        new_w04.get("surface_form") == predicate_surface,
        old_w04.get("context") == alias_context,
        new_w04.get("context") == predicate_context,
        new_w03.get("candidate_sense") == sense_label,
        new_w04.get("candidate_primitive") == primitive,
        current_w03.observation.source_ref_key
        == current_w04.observation.source_ref_key,
        previous_w03.observation.source_ref_key
        == previous_w04.observation.source_ref_key,
        current_w04.observation.perturbation_kind == "CUE_REPLACEMENT",
        current_w04.observation.sample_role == "supersede",
        previous_w04.observation.sample_role == "support",
    )
    if not all(expected_parallel):
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement records do not agree")
    registry = primitive.get("registry")
    kind = primitive.get("kind")
    if ((registry, kind)
            != (vertical.link.primitive_registry, vertical.link.primitive_kind)):
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement primitive escaped the vertical link")
    candidate = tuple(
        item for item in vertical.w04_w05.w05_result.candidates
        if item.proposition_key == vertical.link.proposition_key)
    if len(candidate) != 1:
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement Proposition is not unique")
    occurrence = tuple(
        item for item in candidate[0].occurrences
        if item.identity_key == vertical.link.predicate_occurrence_key)
    if (len(occurrence) != 1
            or occurrence[0].surface_fragment != predicate_surface):
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement did not reach the Proposition occurrence")
    old_sources = _source_by_key(feature_catalog.w03_batch)
    new_sources = _source_by_key(feature_catalog.w04_batch)
    alias_source = old_sources.get(previous_w03.observation.source_ref_key)
    predicate_source = new_sources.get(current_w04.observation.source_ref_key)
    if alias_source is None or predicate_source is None:
        raise W03W04W05RawQuestionAliasError(
            "predicate replacement SourceRef is unavailable")
    evidence_keys = tuple(sorted((
        _record_key(previous_w03.evidence),
        _record_key(previous_w04.evidence),
        _record_key(current_w03.evidence),
        _record_key(current_w04.evidence),
    )))
    return LearnedPredicateAliasRoute(
        alias_surface,
        predicate_surface,
        sense_label,
        registry,
        kind,
        alias_context,
        predicate_context,
        _record_key(alias_source),
        _record_key(predicate_source),
        _sha(alias_source.to_dict()),
        _sha(predicate_source.to_dict()),
        _record_key(previous_w03.observation),
        _record_key(previous_w04.observation),
        _record_key(current_w03.observation),
        _record_key(current_w04.observation),
        evidence_keys,
        vertical.link.proposition_key,
        vertical.link.predicate_occurrence_key,
    )


def build_learned_predicate_alias_bridge(
        source: object,
        bundle: object | None = None,
        *,
        expected_identity_sha256: str = PREDICATE_ALIAS_BRIDGE_SHA256,
        ) -> LearnedPredicateAliasBridge:
    """Recover lexical replacement routes from any shared question catalog."""
    feature_catalog = raw_question_feature_catalog(
        source if bundle is None else bundle,
        expected_overlay=None if bundle is None else source,
    )
    routes = tuple(sorted(
        (_route_for_vertical(feature_catalog, item)
         for item in feature_catalog.vertical_results),
        key=LearnedPredicateAliasRoute.sha256,
    ))
    value = {
        "overlay_validation_sha256": (
            feature_catalog.overlay_validation_sha256),
        "raw_question_bundle_sha256": (
            feature_catalog.bundle_identity_sha256),
        "routes": [item.to_dict() for item in routes],
        "w03_source_binding_sha256": (
            feature_catalog.w03_batch.source_binding.sha256()),
        "w04_source_binding_sha256": (
            feature_catalog.w04_batch.source_binding.sha256()),
    }
    identity = _sha(value)
    bridge = LearnedPredicateAliasBridge(
        feature_catalog.overlay_validation_sha256,
        feature_catalog.bundle_identity_sha256,
        feature_catalog.w03_batch.source_binding.sha256(),
        feature_catalog.w04_batch.source_binding.sha256(),
        routes,
        identity,
    )
    if identity != expected_identity_sha256:
        raise W03W04W05RawQuestionAliasError(
            "predicate alias bridge commitment drifted")
    return bridge


def resolve_learned_predicate_alias(
        bridge: LearnedPredicateAliasBridge,
        alias_surface: str,
        ) -> LearnedPredicateAliasResolution:
    """Group complete public routes by semantic target without stable-sort winning."""
    if not isinstance(bridge, LearnedPredicateAliasBridge):
        raise TypeError("predicate alias resolver bridge is invalid")
    if (not isinstance(alias_surface, str) or not alias_surface
            or alias_surface.strip() != alias_surface):
        raise W03W04W05RawQuestionAliasError(
            "predicate alias resolver surface is not canonical")
    grouped: dict[tuple[str, str, int, str], list] = {}
    for route in bridge.routes:
        if route.alias_surface == alias_surface:
            grouped.setdefault(route.semantic_key(), []).append(route)
    options = tuple(sorted(
        (
            LearnedPredicateAliasOption(
                alias_surface,
                semantic[3],
                semantic[0],
                semantic[1],
                semantic[2],
                tuple(sorted(
                    routes, key=LearnedPredicateAliasRoute.sha256)),
            )
            for semantic, routes in grouped.items()
        ),
        key=LearnedPredicateAliasOption.sha256,
    ))
    status = (
        "MISSING" if not options
        else "SELECTED" if len(options) == 1
        else "AMBIGUOUS"
    )
    return LearnedPredicateAliasResolution(alias_surface, status, options)


def _extract_predicate_alias(
        construction: RawQuestionConstruction,
        question_surface: str,
        ) -> str | None:
    predicate_indexes = tuple(
        ordinal for ordinal, item in enumerate(construction.segments)
        if item.kind == "PREDICATE")
    if len(predicate_indexes) != 1:
        raise W03W04W05RawQuestionAliasError(
            "predicate alias matcher requires one predicate segment")
    ordinal = predicate_indexes[0]
    prefix = "".join(
        item.surface for item in construction.segments[:ordinal])
    suffix = "".join(
        item.surface for item in construction.segments[ordinal + 1:])
    if (not question_surface.startswith(prefix)
            or not question_surface.endswith(suffix)
            or len(question_surface) <= len(prefix) + len(suffix)):
        return None
    end = len(question_surface) - len(suffix) if suffix else None
    alias_surface = question_surface[len(prefix):end]
    if (not alias_surface or alias_surface.strip() != alias_surface
            or alias_surface == _one_predicate_surface(construction)):
        return None
    return alias_surface


def _alias_match(
        bridge: LearnedPredicateAliasBridge,
        construction: RawQuestionConstruction,
        question_surface: str,
        ) -> RawQuestionPredicateAliasMatch | None:
    alias_surface = _extract_predicate_alias(construction, question_surface)
    if alias_surface is None:
        return None
    resolution = resolve_learned_predicate_alias(bridge, alias_surface)
    vertical = construction.vertical_result
    assert vertical.link is not None
    predicate_surface = _one_predicate_surface(construction)
    aligned = tuple(sorted(
        route.sha256()
        for option in resolution.options
        for route in option.routes
        if (option.predicate_surface == predicate_surface
            and (option.primitive_registry, option.primitive_kind)
            == (construction.pattern.primitive_registry,
                construction.pattern.primitive_kind)
            and route.proposition_key == vertical.link.proposition_key
            and route.predicate_occurrence_key
            == vertical.link.predicate_occurrence_key)
    ))
    return RawQuestionPredicateAliasMatch(
        construction, alias_surface, resolution, aligned)


def run_question_feature_predicate_alias_answer(
        bridge: LearnedPredicateAliasBridge,
        feature_catalog: RawQuestionFeatureCatalog,
        request: RawQuestionRequest,
        ) -> RawQuestionPredicateAliasAnswerResult:
    """Preserve FT11 exact matching, then rewrite only a learned predicate slot."""
    if (not isinstance(bridge, LearnedPredicateAliasBridge)
            or not isinstance(feature_catalog, RawQuestionFeatureCatalog)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("predicate alias raw question inputs are invalid")
    if (bridge.overlay_validation_sha256
            != feature_catalog.overlay_validation_sha256
            or bridge.raw_question_bundle_sha256
            != feature_catalog.bundle_identity_sha256):
        raise W03W04W05RawQuestionAliasError(
            "predicate alias bridge escaped its feature catalog")
    exact = run_raw_question_feature_answer(feature_catalog, request)
    if exact.status != "UNKNOWN":
        return RawQuestionPredicateAliasAnswerResult(
            request, exact.status, exact.answer_surface, exact, (), None, None)
    matches = tuple(sorted(
        (
            match
            for construction in feature_catalog.catalog
            if (request.source_record_key is None
                or construction.source_record_key
                == request.source_record_key)
            for match in (
                _alias_match(
                    bridge, construction, request.question_surface),)
            if match is not None
        ),
        key=RawQuestionPredicateAliasMatch.sha256,
    ))
    if any(item.resolution.status == "AMBIGUOUS" for item in matches):
        return RawQuestionPredicateAliasAnswerResult(
            request, "CLARIFY", None, exact, matches, None, None)
    selected = tuple(item for item in matches if item.selected)
    if not selected:
        return RawQuestionPredicateAliasAnswerResult(
            request, "UNKNOWN", None, exact, matches, None, None)
    if len(selected) > 1:
        return RawQuestionPredicateAliasAnswerResult(
            request, "CLARIFY", None, exact, matches, None, None)
    match = selected[0]
    normalized = run_raw_question_feature_answer(
        feature_catalog,
        RawQuestionRequest(
            match.construction.question_surface,
            request.source_record_key,
        ),
    )
    return RawQuestionPredicateAliasAnswerResult(
        request,
        normalized.status,
        normalized.answer_surface,
        exact,
        matches,
        match,
        normalized,
    )


def run_raw_question_predicate_alias_answer(
        bridge: LearnedPredicateAliasBridge,
        catalog: tuple[RawQuestionConstruction, ...],
        w03_batch,
        w04_batch,
        w05_batch,
        request: RawQuestionRequest,
        *,
        overlay_validation_sha256: str,
        ) -> RawQuestionPredicateAliasAnswerResult:
    """Compatibility facade for callers that still pass six runtime parts."""
    vertical_by_sha = {
        item.vertical_result.sha256(): item.vertical_result
        for item in catalog
    }
    pattern_by_sha = {
        item.pattern.sha256(): item.pattern
        for item in catalog
    }
    feature_catalog = RawQuestionFeatureCatalog(
        bridge.raw_question_bundle_sha256,
        overlay_validation_sha256,
        tuple(vertical_by_sha[key] for key in sorted(vertical_by_sha)),
        tuple(pattern_by_sha[key] for key in sorted(pattern_by_sha)),
        catalog,
        w03_batch,
        w04_batch,
        w05_batch,
    )
    return run_question_feature_predicate_alias_answer(
        bridge,
        feature_catalog,
        request,
    )


__all__ = [
    "PREDICATE_ALIAS_ANSWER_SHA256S",
    "PREDICATE_ALIAS_BRIDGE_SHA256",
    "PREDICATE_ALIAS_EXPRESSION_BOUNDARY",
    "build_learned_predicate_alias_bridge",
    "resolve_learned_predicate_alias",
    "run_question_feature_predicate_alias_answer",
    "run_raw_question_predicate_alias_answer",
]
