"""FT28 来源绑定 W-05 proposition 的不可变公开合同。

本模块把 W-04 ``source_claim`` primitive 结构化为“某来源如此声称”的
proposition。W-05 typed projection 中的 AUTHORIZED 只授权该来源声明的存在，
不把其 asserted value 裁定为客观事实。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_STATUSES,
    W03PublicSenseQuery,
    W03PublicSenseSourceRevision,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive_contract import (
    W03W04SourceBoundPrimitiveQueryResult,
    W04_SOURCE_BOUND_EPISTEMIC_STATUS,
    W04_SOURCE_BOUND_PRIMITIVE_KINDS,
    W04_SOURCE_BOUND_TRUTH_STATUS,
    W04SourceBoundPrimitive,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicCandidateProjection,
    W05V2PublicOccurrenceProjection,
    W05V2PublicRoleBindingProjection,
)


W05_SOURCE_BOUND_PROPOSITION_REGISTRY = "source_claim_proposition"
W05_SOURCE_BOUND_PROPOSITION_KINDS = dict(
    W04_SOURCE_BOUND_PRIMITIVE_KINDS)
W05_SOURCE_BOUND_SOURCE_REF_KIND = 52801
W05_SOURCE_BOUND_UNDERSTANDING_STATUS = "SOURCE_CLAIM_STRUCTURED"
W05_SOURCE_BOUND_REASONING_AUTHORIZED = "AUTHORIZED"
W05_SOURCE_BOUND_REASONING_SUPERSEDED = "SUPERSEDED"


# object-model: exception
class W04W05SourceBoundPropositionError(ValueError):
    """来源绑定 proposition 或其 W-05 typed projection 漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity_key(value: object) -> tuple[int, ...]:
    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    values = tuple(
        int.from_bytes(digest[offset:offset + 8], "big")
        & ((1 << 63) - 1)
        for offset in range(0, len(digest), 8)
    )
    return (1, 28, 5, *(item or 1 for item in values))


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W04W05SourceBoundPropositionError(
            f"{where} is not canonical text")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W04W05SourceBoundPropositionError(
            f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W04W05SourceBoundPropositionError(
            f"{where} is not a SHA-256 digest")
    return value


def source_ref_projection_key(
        primitive: W04SourceBoundPrimitive,
        ) -> tuple[int, ...]:
    """返回兼容既有 W-05 typed contract 的确定性 SourceRef 键。"""
    if not isinstance(primitive, W04SourceBoundPrimitive):
        raise TypeError("source-ref projection requires a W04 primitive")
    source = primitive.source_ref
    source_id = int(
        _sha({"source_record_key": list(source.stable_key)})[:15], 16) or 1
    document_id = int(_sha({
        "revision_id": source.revision_id,
        "snapshot_id": source.snapshot_id,
        "source_identity": source.source_identity,
    })[:15], 16) or 1
    key = (
        W05_SOURCE_BOUND_SOURCE_REF_KIND,
        source_id,
        document_id,
        0, 0, 0, 1,
        1, 1, 1, 1,
    )
    SourceRef.from_stable_key(key)
    return key


def source_bound_proposition_key(
        primitive: W04SourceBoundPrimitive,
        ) -> tuple[int, ...]:
    """从完整 primitive 身份导出不可伪造的 proposition 键。"""
    if not isinstance(primitive, W04SourceBoundPrimitive):
        raise TypeError("proposition key requires a W04 primitive")
    return _identity_key({
        "primitive": primitive.to_dict(),
        "projection_version": 1,
        "proposition_kind": W05_SOURCE_BOUND_PROPOSITION_KINDS[
            primitive.relation_kind],
        "proposition_registry": W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
    })


def source_bound_candidate_projection(
        primitive: W04SourceBoundPrimitive,
        proposition_key: tuple[int, ...],
        ) -> W05V2PublicCandidateProjection:
    """从 primitive 重建完整 W05 compatibility candidate。"""
    if not isinstance(primitive, W04SourceBoundPrimitive):
        raise TypeError("candidate projection requires a W04 primitive")
    if proposition_key != source_bound_proposition_key(primitive):
        raise W04W05SourceBoundPropositionError(
            "candidate proposition identity drifted")
    relation = primitive.relation_kind
    surface = f"{primitive.surface}\t{relation}\t{primitive.asserted_value}"
    subject_end = len(primitive.surface)
    predicate_start = subject_end + 1
    predicate_end = predicate_start + len(relation)
    value_start = predicate_end + 1
    value_end = value_start + len(primitive.asserted_value)
    predicate_key = _identity_key({
        "kind": W05_SOURCE_BOUND_PROPOSITION_KINDS[relation],
        "registry": W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
        "role": "PREDICATE",
    })
    value_key = _identity_key({
        "primitive_key": list(primitive.primitive_key),
        "role": "ASSERTED_VALUE",
        "value": primitive.asserted_value,
    })
    subject_occurrence_key = _identity_key({
        "proposition_key": list(proposition_key),
        "role": "SUBJECT_OCCURRENCE",
    })
    predicate_occurrence_key = _identity_key({
        "proposition_key": list(proposition_key),
        "role": "PREDICATE_OCCURRENCE",
    })
    value_occurrence_key = _identity_key({
        "proposition_key": list(proposition_key),
        "role": "VALUE_OCCURRENCE",
    })
    occurrences = (
        W05V2PublicOccurrenceProjection(
            subject_occurrence_key,
            primitive.concept_key,
            0,
            subject_end,
            0,
            primitive.surface,
        ),
        W05V2PublicOccurrenceProjection(
            predicate_occurrence_key,
            predicate_key,
            predicate_start,
            predicate_end,
            1,
            relation,
        ),
        W05V2PublicOccurrenceProjection(
            value_occurrence_key,
            value_key,
            value_start,
            value_end,
            2,
            primitive.asserted_value,
        ),
    )
    subject_role_key = _identity_key({
        "registry": W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
        "role": "SUBJECT",
    })
    value_role_key = _identity_key({
        "registry": W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
        "role": "ASSERTED_VALUE",
    })
    bindings = (
        W05V2PublicRoleBindingProjection(
            _identity_key({
                "filler": list(primitive.concept_key),
                "proposition_key": list(proposition_key),
                "role": list(subject_role_key),
            }),
            subject_role_key,
            primitive.concept_key,
            0,
        ),
        W05V2PublicRoleBindingProjection(
            _identity_key({
                "filler": list(value_key),
                "proposition_key": list(proposition_key),
                "role": list(value_role_key),
            }),
            value_role_key,
            value_key,
            1,
        ),
    )
    active = primitive.active
    return W05V2PublicCandidateProjection(
        surface,
        proposition_key,
        predicate_key,
        predicate_occurrence_key,
        _identity_key({
            "source_ref": primitive.source_ref.to_dict(),
            "role": "SOURCE_ASSERTION_CONTEXT",
        }),
        tuple(item.identity_key for item in occurrences),
        occurrences,
        bindings,
        source_ref_projection_key(primitive),
        primitive.source_ref.stable_key,
        primitive.source_ref.source_key,
        primitive.source_ref.source_commitment_sha256,
        primitive.source_ref.license_id,
        "ACTIVE" if active == 1 else "SUPERSEDED",
        active,
        int(active == 0),
        W05_SOURCE_BOUND_UNDERSTANDING_STATUS,
        (
            W05_SOURCE_BOUND_REASONING_AUTHORIZED
            if active == 1 else W05_SOURCE_BOUND_REASONING_SUPERSEDED
        ),
        1,
    )


def source_bound_query_record_commitment(
        query: W03PublicSenseQuery,
        primitive_result: W03W04SourceBoundPrimitiveQueryResult,
        propositions: tuple["W05SourceBoundProposition", ...],
        proposition_projection_sha256: str,
        ) -> str:
    """绑定一次 query、FT27 结果和 proposition 投影身份。"""
    if (not isinstance(query, W03PublicSenseQuery)
            or not isinstance(
                primitive_result, W03W04SourceBoundPrimitiveQueryResult)
            or not isinstance(propositions, tuple)
            or any(not isinstance(item, W05SourceBoundProposition)
                   for item in propositions)):
        raise TypeError("query record commitment inputs are invalid")
    _sha256(
        proposition_projection_sha256,
        where="query proposition projection",
    )
    return _sha({
        "primitive_result_sha256": primitive_result.sha256(),
        "proposition_keys": [
            list(item.proposition_key) for item in propositions],
        "proposition_projection_sha256": proposition_projection_sha256,
        "query": query.to_dict(),
    })


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05SourceBoundProposition:
    """一个完整承接 FT27 身份且不越过来源真值边界的 proposition。"""

    proposition_key: tuple[int, ...]
    primitive: W04SourceBoundPrimitive
    proposition_registry: str
    proposition_kind: int
    supersedes_proposition_keys: tuple[tuple[int, ...], ...]
    w05_candidate: W05V2PublicCandidateProjection
    epistemic_status: str = W04_SOURCE_BOUND_EPISTEMIC_STATUS
    truth_status: str = W04_SOURCE_BOUND_TRUTH_STATUS

    def __post_init__(self) -> None:
        _key(self.proposition_key, where="proposition key")
        if not isinstance(self.primitive, W04SourceBoundPrimitive):
            raise TypeError("proposition primitive type is invalid")
        if self.proposition_key != source_bound_proposition_key(
                self.primitive):
            raise W04W05SourceBoundPropositionError(
                "proposition identity drifted")
        _text(self.proposition_registry, where="proposition registry")
        expected_kind = W05_SOURCE_BOUND_PROPOSITION_KINDS.get(
            self.primitive.relation_kind)
        if (self.proposition_registry
                != W05_SOURCE_BOUND_PROPOSITION_REGISTRY
                or self.proposition_kind != expected_kind):
            raise W04W05SourceBoundPropositionError(
                "proposition relation coordinate drifted")
        if (self.epistemic_status != W04_SOURCE_BOUND_EPISTEMIC_STATUS
                or self.truth_status != W04_SOURCE_BOUND_TRUTH_STATUS):
            raise W04W05SourceBoundPropositionError(
                "proposition truth boundary drifted")
        if (not isinstance(self.supersedes_proposition_keys, tuple)
                or any(not isinstance(item, tuple)
                       for item in self.supersedes_proposition_keys)):
            raise W04W05SourceBoundPropositionError(
                "proposition supersede identity drifted")
        for item in self.supersedes_proposition_keys:
            _key(item, where="superseded proposition key")
        if tuple(sorted(set(self.supersedes_proposition_keys))) != (
                self.supersedes_proposition_keys
                ) or self.proposition_key in self.supersedes_proposition_keys:
            raise W04W05SourceBoundPropositionError(
                "proposition supersede identity is not canonical")
        if not isinstance(self.w05_candidate, W05V2PublicCandidateProjection):
            raise TypeError("proposition W05 candidate type is invalid")
        expected_candidate = source_bound_candidate_projection(
            self.primitive, self.proposition_key)
        if self.w05_candidate.to_dict() != expected_candidate.to_dict():
            raise W04W05SourceBoundPropositionError(
                "proposition W05 candidate projection drifted")

    @property
    def definition_text(self) -> str | None:
        return self.primitive.definition_text

    @property
    def relation_kind(self) -> str:
        return self.primitive.relation_kind

    def to_dict(self) -> dict[str, object]:
        return {
            "epistemic_status": self.epistemic_status,
            "primitive": self.primitive.to_dict(),
            "proposition_key": list(self.proposition_key),
            "proposition_kind": self.proposition_kind,
            "proposition_registry": self.proposition_registry,
            "supersedes_proposition_keys": [
                list(item) for item in self.supersedes_proposition_keys],
            "truth_status": self.truth_status,
            "w05_candidate": self.w05_candidate.to_dict(),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04W05SourceBoundPropositionQueryResult:
    """保留 W03/FT27 状态并公开对应 W-05 source-claim propositions。"""

    query: W03PublicSenseQuery
    primitive_result: W03W04SourceBoundPrimitiveQueryResult
    status: str
    propositions: tuple[W05SourceBoundProposition, ...]
    source_revisions: tuple[W03PublicSenseSourceRevision, ...]
    primitive_projection_sha256: str
    proposition_projection_sha256: str
    record_commitment_sha256: str
    clarify_required: int
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W03PublicSenseQuery)
                or not isinstance(
                    self.primitive_result,
                    W03W04SourceBoundPrimitiveQueryResult,
                )):
            raise TypeError("proposition query inputs are invalid")
        if (self.status not in W03_PUBLIC_SENSE_STATUSES
                or self.status != self.primitive_result.status):
            raise W04W05SourceBoundPropositionError(
                "proposition query weakened the W03 status")
        if (not isinstance(self.propositions, tuple)
                or any(not isinstance(item, W05SourceBoundProposition)
                       for item in self.propositions)
                or any(item.primitive.active != 1
                       for item in self.propositions)):
            raise W04W05SourceBoundPropositionError(
                "proposition query candidate inventory drifted")
        expected = tuple(
            item.primitive_key
            for item in self.primitive_result.primitives)
        actual = tuple(
            item.primitive.primitive_key for item in self.propositions)
        if actual != expected:
            raise W04W05SourceBoundPropositionError(
                "proposition query lost or reordered a primitive")
        if self.status == "UNKNOWN" and self.propositions:
            raise W04W05SourceBoundPropositionError(
                "UNKNOWN proposition query returned candidates")
        if self.status != "UNKNOWN" and not self.propositions:
            raise W04W05SourceBoundPropositionError(
                "non-UNKNOWN proposition query lost candidates")
        if (not isinstance(self.source_revisions, tuple)
                or self.source_revisions
                != self.primitive_result.source_revisions):
            raise W04W05SourceBoundPropositionError(
                "proposition source revisions drifted")
        for name in (
                "primitive_projection_sha256",
                "proposition_projection_sha256",
                "record_commitment_sha256"):
            _sha256(getattr(self, name), where=f"proposition query {name}")
        if self.primitive_projection_sha256 != (
                self.primitive_result.projection_sha256):
            raise W04W05SourceBoundPropositionError(
                "proposition query primitive commitment drifted")
        if self.record_commitment_sha256 != (
                source_bound_query_record_commitment(
                    self.query,
                    self.primitive_result,
                    self.propositions,
                    self.proposition_projection_sha256,
                )):
            raise W04W05SourceBoundPropositionError(
                "proposition query record commitment drifted")
        expected_clarify = int(
            self.status in {"AMBIGUOUS", "CONFLICT", "CLARIFY"})
        if self.clarify_required != expected_clarify:
            raise W04W05SourceBoundPropositionError(
                "proposition query clarify boundary drifted")
        if (self.experimental, self.formal_mastery_claim,
                self.w03_started, self.w04_started,
                self.w05_started) != (1, 0, 0, 0, 0):
            raise W04W05SourceBoundPropositionError(
                "proposition query formal boundary drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "clarify_required": self.clarify_required,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "primitive_projection_sha256": (
                self.primitive_projection_sha256),
            "primitive_result": self.primitive_result.to_dict(),
            "proposition_projection_sha256": (
                self.proposition_projection_sha256),
            "propositions": [item.to_dict() for item in self.propositions],
            "query": self.query.to_dict(),
            "record_commitment_sha256": self.record_commitment_sha256,
            "source_revisions": [
                item.to_dict() for item in self.source_revisions],
            "status": self.status,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W04W05SourceBoundPropositionError",
    "W04W05SourceBoundPropositionQueryResult",
    "W05_SOURCE_BOUND_PROPOSITION_KINDS",
    "W05_SOURCE_BOUND_PROPOSITION_REGISTRY",
    "W05_SOURCE_BOUND_REASONING_AUTHORIZED",
    "W05_SOURCE_BOUND_REASONING_SUPERSEDED",
    "W05_SOURCE_BOUND_SOURCE_REF_KIND",
    "W05_SOURCE_BOUND_UNDERSTANDING_STATUS",
    "W05SourceBoundProposition",
    "source_bound_candidate_projection",
    "source_bound_proposition_key",
    "source_bound_query_record_commitment",
    "source_ref_projection_key",
]
