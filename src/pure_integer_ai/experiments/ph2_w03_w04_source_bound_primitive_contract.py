"""FT27 W-03 词义到来源绑定 W-04 primitive 的不可变合同。

本模块记录公开来源声称的内容，不把 definition、label 或 alias 提升为已经裁定的事实命题。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_STATUSES,
    W03PublicSenseEntry,
    W03PublicSenseQueryResult,
    W03PublicSenseSourceRef,
    W03PublicSenseSourceRevision,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicCandidateProjection,
)


W04_SOURCE_BOUND_PRIMITIVE_REGISTRY = "source_claim"
W04_SOURCE_BOUND_PRIMITIVE_KINDS = {
    "DEFINITION": 1,
    "LABEL": 2,
    "ALIAS": 3,
}
W04_SOURCE_BOUND_EPISTEMIC_STATUS = "SOURCE_ASSERTED"
W04_SOURCE_BOUND_TRUTH_STATUS = "NOT_ADJUDICATED"


# object-model: exception
class W03W04SourceBoundPrimitiveError(ValueError):
    """来源绑定 primitive 或公开投影发生合同漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04SourceBoundPrimitiveError(
            f"{where} is not canonical text")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04SourceBoundPrimitiveError(
            f"{where} is not a strict integer key")
    return value


def asserted_value(entry: W03PublicSenseEntry) -> str:
    """返回一个 entry 所表达的来源声明类型化对象。"""
    if not isinstance(entry, W03PublicSenseEntry):
        raise TypeError("source-bound primitive entry type is invalid")
    if entry.relation_kind == "DEFINITION":
        if entry.definition_text is None:
            raise W03W04SourceBoundPrimitiveError(
                "DEFINITION entry lacks definition text")
        return entry.definition_text
    if entry.relation_kind == "LABEL":
        return entry.surface
    if entry.relation_kind == "ALIAS":
        return entry.canonical_surface
    raise W03W04SourceBoundPrimitiveError(
        "source-bound primitive relation kind is unsupported")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04SourceBoundPrimitive:
    """一个只表达来源所述内容且感知 revision 的 primitive。"""

    primitive_key: tuple[int, ...]
    entry_key: tuple[int, ...]
    relation_kind: str
    surface: str
    canonical_surface: str
    language: str
    asserted_value: str
    definition_text: str | None
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    source_ref: W03PublicSenseSourceRef
    field_roles: tuple[str, ...]
    active: int
    supersedes_entry_keys: tuple[tuple[int, ...], ...]
    w04_candidate: W04V2PublicCandidateProjection
    primitive_registry: str = W04_SOURCE_BOUND_PRIMITIVE_REGISTRY
    primitive_kind: int = 0
    epistemic_status: str = W04_SOURCE_BOUND_EPISTEMIC_STATUS
    truth_status: str = W04_SOURCE_BOUND_TRUTH_STATUS

    def __post_init__(self) -> None:
        for name in (
                "primitive_key", "entry_key", "sense_key", "concept_key",
                "observation_key"):
            _key(getattr(self, name), where=f"primitive {name}")
        for name in (
                "relation_kind", "surface", "canonical_surface", "language",
                "asserted_value", "primitive_registry", "epistemic_status",
                "truth_status"):
            _text(getattr(self, name), where=f"primitive {name}")
        expected_kind = W04_SOURCE_BOUND_PRIMITIVE_KINDS.get(
            self.relation_kind)
        if (expected_kind is None
                or self.primitive_registry
                != W04_SOURCE_BOUND_PRIMITIVE_REGISTRY
                or self.primitive_kind != expected_kind):
            raise W03W04SourceBoundPrimitiveError(
                "primitive relation coordinate drifted")
        if (self.epistemic_status != W04_SOURCE_BOUND_EPISTEMIC_STATUS
                or self.truth_status != W04_SOURCE_BOUND_TRUTH_STATUS):
            raise W03W04SourceBoundPrimitiveError(
                "primitive truth boundary drifted")
        if self.definition_text is not None:
            _text(self.definition_text, where="primitive definition text")
        if self.relation_kind == "DEFINITION":
            expected_value = self.definition_text
        elif self.relation_kind == "LABEL":
            expected_value = self.surface
        else:
            expected_value = self.canonical_surface
        if self.asserted_value != expected_value:
            raise W03W04SourceBoundPrimitiveError(
                "primitive asserted value does not match its typed relation")
        if not isinstance(self.source_ref, W03PublicSenseSourceRef):
            raise TypeError("primitive SourceRef type is invalid")
        if (not isinstance(self.field_roles, tuple)
                or not self.field_roles
                or any(not isinstance(item, str) or not item
                       for item in self.field_roles)
                or tuple(sorted(set(self.field_roles))) != self.field_roles):
            raise W03W04SourceBoundPrimitiveError(
                "primitive field roles drifted")
        if self.active not in {0, 1}:
            raise W03W04SourceBoundPrimitiveError(
                "primitive active flag is not binary")
        if (not isinstance(self.supersedes_entry_keys, tuple)
                or any(not isinstance(item, tuple)
                       for item in self.supersedes_entry_keys)):
            raise W03W04SourceBoundPrimitiveError(
                "primitive supersede identity drifted")
        for item in self.supersedes_entry_keys:
            _key(item, where="primitive superseded entry")
        if tuple(sorted(set(self.supersedes_entry_keys))) != (
                self.supersedes_entry_keys):
            raise W03W04SourceBoundPrimitiveError(
                "primitive supersede identity is not canonical")
        if not isinstance(self.w04_candidate, W04V2PublicCandidateProjection):
            raise TypeError("primitive W04 candidate type is invalid")
        candidate = self.w04_candidate
        if (
            candidate.surface,
            candidate.context_text,
            candidate.primitive_registry,
            candidate.primitive_kind,
            candidate.candidate_key,
            candidate.source_ref_key,
            candidate.source_key,
            candidate.source_commitment,
            candidate.license_id,
            candidate.active,
            candidate.superseded,
        ) != (
            self.surface,
            self.asserted_value,
            self.primitive_registry,
            self.primitive_kind,
            self.primitive_key,
            self.source_ref.stable_key,
            self.source_ref.source_key,
            self.source_ref.source_commitment_sha256,
            self.source_ref.license_id,
            self.active,
            int(self.active == 0),
        ):
            raise W03W04SourceBoundPrimitiveError(
                "primitive W04 compatibility projection drifted")

    def to_dict(self) -> dict[str, object]:
        """导出完整的类型化来源声明。"""
        return {
            "active": self.active,
            "asserted_value": self.asserted_value,
            "canonical_surface": self.canonical_surface,
            "concept_key": list(self.concept_key),
            "definition_text": self.definition_text,
            "entry_key": list(self.entry_key),
            "epistemic_status": self.epistemic_status,
            "field_roles": list(self.field_roles),
            "language": self.language,
            "observation_key": list(self.observation_key),
            "primitive_key": list(self.primitive_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "relation_kind": self.relation_kind,
            "sense_key": list(self.sense_key),
            "source_ref": self.source_ref.to_dict(),
            "supersedes_entry_keys": [
                list(item) for item in self.supersedes_entry_keys],
            "surface": self.surface,
            "truth_status": self.truth_status,
            "w04_candidate": self.w04_candidate.to_dict(),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04SourceBoundPrimitiveQueryResult:
    """原样保留 W-03 状态及其 W-04 来源声明 primitive。"""

    sense_result: W03PublicSenseQueryResult
    status: str
    primitives: tuple[W04SourceBoundPrimitive, ...]
    source_revisions: tuple[W03PublicSenseSourceRevision, ...]
    source_binding_sha256: str
    projection_sha256: str
    clarify_required: int
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sense_result, W03PublicSenseQueryResult):
            raise TypeError("primitive query sense result type is invalid")
        if (self.status not in W03_PUBLIC_SENSE_STATUSES
                or self.status != self.sense_result.status):
            raise W03W04SourceBoundPrimitiveError(
                "primitive query weakened the W03 status")
        if (not isinstance(self.primitives, tuple)
                or any(not isinstance(item, W04SourceBoundPrimitive)
                       for item in self.primitives)
                or any(item.active != 1 for item in self.primitives)):
            raise W03W04SourceBoundPrimitiveError(
                "primitive query candidates drifted")
        expected_entries = tuple(
            item.entry.entry_key for item in self.sense_result.candidates)
        actual_entries = tuple(item.entry_key for item in self.primitives)
        if actual_entries != expected_entries:
            raise W03W04SourceBoundPrimitiveError(
                "primitive query lost or reordered a sense candidate")
        if self.status == "UNKNOWN" and self.primitives:
            raise W03W04SourceBoundPrimitiveError(
                "UNKNOWN primitive query returned candidates")
        if self.status != "UNKNOWN" and not self.primitives:
            raise W03W04SourceBoundPrimitiveError(
                "non-UNKNOWN primitive query lost candidates")
        if (not isinstance(self.source_revisions, tuple)
                or any(not isinstance(item, W03PublicSenseSourceRevision)
                       for item in self.source_revisions)):
            raise W03W04SourceBoundPrimitiveError(
                "primitive query source revisions drifted")
        for name in ("source_binding_sha256", "projection_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise W03W04SourceBoundPrimitiveError(
                    f"primitive query {name} drifted")
        expected_clarify = int(
            self.status in {"AMBIGUOUS", "CONFLICT", "CLARIFY"})
        if self.clarify_required != expected_clarify:
            raise W03W04SourceBoundPrimitiveError(
                "primitive query clarify boundary drifted")
        if (self.experimental, self.formal_mastery_claim,
                self.w03_started, self.w04_started) != (1, 0, 0, 0):
            raise W03W04SourceBoundPrimitiveError(
                "primitive query formal boundary drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "clarify_required": self.clarify_required,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "primitives": [item.to_dict() for item in self.primitives],
            "projection_sha256": self.projection_sha256,
            "sense_result": self.sense_result.to_dict(),
            "source_binding_sha256": self.source_binding_sha256,
            "source_revisions": [
                item.to_dict() for item in self.source_revisions],
            "status": self.status,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W03W04SourceBoundPrimitiveError",
    "W03W04SourceBoundPrimitiveQueryResult",
    "W04_SOURCE_BOUND_EPISTEMIC_STATUS",
    "W04_SOURCE_BOUND_PRIMITIVE_KINDS",
    "W04_SOURCE_BOUND_PRIMITIVE_REGISTRY",
    "W04_SOURCE_BOUND_TRUTH_STATUS",
    "W04SourceBoundPrimitive",
    "asserted_value",
]
