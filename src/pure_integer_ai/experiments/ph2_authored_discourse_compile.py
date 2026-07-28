"""把 D-02E discourse seed 编译为来源化 span/occurrence/reference/revision payload。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    CorpusVersion,
    CurriculumVersion,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.parser_revision import (
    ParserAnchorRevision,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_discourse_schema import (
    AuthoredDiscourseSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 210


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 family/seed id 压为稳定正整数来源段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _versions(parser_version: int) -> VersionBundle:
    """构造只改变 ParserVersion 的冻结 PH2 版本束。"""
    return VersionBundle(
        CorpusVersion(1),
        ParserVersion(parser_version),
        PrimitiveVersion(1),
        CurriculumVersion(1),
    )


def _source(seed: AuthoredDiscourseSeed, parser_version: int) -> SourceRef:
    """同一 seed 的跨 parser 来源保持 lineage，只改变版本束。"""
    return SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("discourse-family", seed.family),
        _stable_positive_int("discourse-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _versions(parser_version),
    )


def _identity_key(value) -> list[int]:
    """把来源、scope 或对象身份投影为规范整数列表。"""
    return list(value.stable_key())


def _occurrence_map(seed: AuthoredDiscourseSeed, source: SourceRef):
    """按 seed id 返回真实 occurrence_identity。"""
    return {
        item.occurrence_id: occurrence_identity(
            source,
            start=item.start,
            end=item.end,
            ordinal=item.ordinal,
        )
        for item in seed.occurrences
    }


def _reference_payload(seed: AuthoredDiscourseSeed, identities) -> dict | None:
    """投影显式有界窗口、候选和后文局部替代。"""
    plan = seed.reference_plan
    if plan is None:
        return None
    return {
        "candidate_occurrence_ids": list(plan.candidate_occurrence_ids),
        "candidate_occurrence_keys": [
            _identity_key(identities[item])
            for item in plan.candidate_occurrence_ids
        ],
        "impacted_query_ids": list(plan.impacted_query_ids),
        "reference_occurrence_id": plan.reference_occurrence_id,
        "reference_occurrence_key": _identity_key(
            identities[plan.reference_occurrence_id]),
        "rejected_occurrence_id": plan.rejected_occurrence_id,
        "rejected_occurrence_key": (
            None if not plan.rejected_occurrence_id
            else _identity_key(identities[plan.rejected_occurrence_id])
        ),
        "replacement_occurrence_id": plan.replacement_occurrence_id,
        "replacement_occurrence_key": (
            None if not plan.replacement_occurrence_id
            else _identity_key(identities[plan.replacement_occurrence_id])
        ),
        "window_occurrence_ids": list(plan.window_occurrence_ids),
        "window_occurrence_keys": [
            _identity_key(identities[item])
            for item in plan.window_occurrence_ids
        ],
    }


def _revision_payload(seed: AuthoredDiscourseSeed) -> dict | None:
    """构造跨 SourceRef/ParserVersion 的真实 ParserAnchorRevision。"""
    revision = seed.parser_revision
    if revision is None:
        return None
    old_source = _source(seed, revision.old_parser_version)
    new_source = _source(seed, revision.new_parser_version)
    old_identities = _occurrence_map(seed, old_source)
    new_identities = _occurrence_map(seed, new_source)
    anchors = tuple(ParserAnchorRevision(
        old_identities[item.old_occurrence_id],
        tuple(new_identities[value]
              for value in item.replacement_occurrence_ids),
    ) for item in revision.mappings)
    return {
        "affected_occurrences": [
            {
                "occurrence_id": item.old_occurrence_id,
                "old_occurrence_key": _identity_key(
                    old_identities[item.old_occurrence_id]),
                "replacement_occurrence_ids": list(
                    item.replacement_occurrence_ids),
                "replacement_occurrence_keys": [
                    _identity_key(new_identities[value])
                    for value in item.replacement_occurrence_ids
                ],
                "revision_key": _identity_key(anchor),
            }
            for item, anchor in zip(revision.mappings, anchors)
        ],
        "new_document_scope_key": _identity_key(document_scope(new_source)),
        "new_parser_version": revision.new_parser_version,
        "new_source_key": _identity_key(new_source),
        "old_document_scope_key": _identity_key(document_scope(old_source)),
        "old_parser_version": revision.old_parser_version,
        "old_source_key": _identity_key(old_source),
        "recompute_query_ids": list(revision.recompute_query_ids),
        "unaffected_occurrences": [
            {
                "new_occurrence_key": _identity_key(new_identities[item]),
                "occurrence_id": item,
                "old_occurrence_key": _identity_key(old_identities[item]),
            }
            for item in revision.unaffected_occurrence_ids
        ],
    }


def compile_discourse_seed(seed: AuthoredDiscourseSeed) -> AuthoredCompiledSeed:
    """生成只含 Observation 可见内容的 discourse/revision 执行查询。"""
    if not isinstance(seed, AuthoredDiscourseSeed):
        raise TypeError("compile_discourse_seed 需要 AuthoredDiscourseSeed")
    parser_version = (
        1 if seed.parser_revision is None
        else seed.parser_revision.new_parser_version
    )
    source = _source(seed, parser_version)
    scope = document_scope(source)
    occurrences = _occurrence_map(seed, source)
    payload = {
        "bare_reference_fifo_authoritative": 0,
        "consumer_request": {
            "max_candidates": seed.consumer_request.max_candidates,
            "max_context_chars": seed.consumer_request.max_context_chars,
            "max_occurrences": seed.consumer_request.max_occurrences,
            "max_paragraphs": seed.consumer_request.max_paragraphs,
            "max_recompute_queries": (
                seed.consumer_request.max_recompute_queries),
        },
        "document_scope_key": _identity_key(scope),
        "evidence_state": {
            "refute": seed.evidence_refute,
            "support": seed.evidence_support,
        },
        "occurrences": [
            {
                "end": item.end,
                "occurrence_id": item.occurrence_id,
                "occurrence_key": _identity_key(
                    occurrences[item.occurrence_id]),
                "ordinal": item.ordinal,
                "semantic_kind": item.semantic_kind,
                "start": item.start,
            }
            for item in seed.occurrences
        ],
        "old_occurrences_rewritten": 0,
        "paragraph_spans": [
            {
                "end": item.end,
                "ordinal": item.ordinal,
                "span_id": item.span_id,
                "span_key": _identity_key(span_identity(
                    source,
                    members=((item.start, item.end),),
                    ordinal=item.ordinal,
                )),
                "span_kind": item.span_kind,
                "start": item.start,
            }
            for item in seed.paragraphs
        ],
        "parser_revision_plan": _revision_payload(seed),
        "query_kind": "typed_discourse_revision_candidate",
        "reference_plan": _reference_payload(seed, occurrences),
        "source_ref_key": _identity_key(source),
        "surface": seed.surface,
        "surface_cue_authoritative": 0,
        "unaffected_recomputed": 0,
        "variant_kind": seed.variant_kind,
        "whole_document_recomputed": 0,
    }
    typed_payload = CanonicalJsonObject.from_value(payload)
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "DiscourseRevisionQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.seed_id, payload),
        (
            seed.surface,
            payload["paragraph_spans"],
            payload["occurrences"],
            payload["reference_plan"],
            payload["parser_revision_plan"],
        ),
        (
            "typed_discourse_revision_v1",
            seed.variant_kind,
            len(seed.paragraphs),
            len(seed.occurrences),
            int(seed.reference_plan is not None),
            int(seed.parser_revision is not None),
        ),
    )


__all__ = ["compile_discourse_seed"]
