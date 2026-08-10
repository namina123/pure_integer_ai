"""W-05 来源绑定公开查询的不可变外部合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)


W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN = "NOT_RUN"
W05_V2_PUBLIC_QUERY_STATUSES = {"UNIQUE", "MULTI", "UNKNOWN", "CONFLICT"}
W05_V2_PUBLIC_QUERY_LIFECYCLES = {
    "ACTIVE", "REFUTED", "SUPERSEDED", "CONFLICT", "UNKNOWN"}


# object-model: exception
class W05V2PublicQueryError(ValueError):
    """W-05 公开查询或其来源绑定投影发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W05V2PublicQueryError(f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W05V2PublicQueryError(f"{where} is not a strict integer key")
    return value


def _source_ref_key(
        value: tuple[int, ...],
        *,
        where: str,
        ) -> tuple[int, ...]:
    _strict_key(value, where=where)
    try:
        SourceRef.from_stable_key(value)
    except (TypeError, ValueError) as exc:
        raise W05V2PublicQueryError(
            f"{where} is not a canonical SourceRef key") from exc
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W05V2PublicQueryError(f"{where} is not a SHA-256 digest")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicQuery:
    """不含预期答案的表层文本与可选 SourceRef 过滤条件。"""

    surface: str
    source_ref_key: tuple[int, ...] | None = None
    allow_generation: int = 1

    def __post_init__(self) -> None:
        _text(self.surface, where="query surface")
        if self.source_ref_key is not None:
            _source_ref_key(self.source_ref_key, where="query SourceRef key")
        if self.allow_generation not in {0, 1}:
            raise W05V2PublicQueryError(
                "query allow_generation must be zero or one")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "source_ref_key": (
                None if self.source_ref_key is None
                else list(self.source_ref_key)
            ),
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicOccurrenceProjection:
    """一个精确 occurrence 及其来源内语义对象。"""

    identity_key: tuple[int, ...]
    semantic_object_key: tuple[int, ...]
    start: int
    end: int
    ordinal: int
    surface_fragment: str

    def __post_init__(self) -> None:
        _strict_key(self.identity_key, where="occurrence identity")
        _strict_key(self.semantic_object_key, where="occurrence semantic object")
        if (type(self.start) is not int or type(self.end) is not int
                or type(self.ordinal) is not int or self.start < 0
                or self.end <= self.start or self.ordinal < 0):
            raise W05V2PublicQueryError("occurrence span or ordinal drifted")
        _text(self.surface_fragment, where="occurrence surface fragment")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "identity_key": list(self.identity_key),
            "ordinal": self.ordinal,
            "semantic_object_key": list(self.semantic_object_key),
            "start": self.start,
            "surface_fragment": self.surface_fragment,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicRoleBindingProjection:
    """一个不依赖宿主语言角色标签的一等 RoleBinding。"""

    identity_key: tuple[int, ...]
    role_key: tuple[int, ...]
    filler_key: tuple[int, ...]
    ordinal: int

    def __post_init__(self) -> None:
        _strict_key(self.identity_key, where="RoleBinding identity")
        _strict_key(self.role_key, where="RoleBinding role")
        _strict_key(self.filler_key, where="RoleBinding filler")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise W05V2PublicQueryError("RoleBinding ordinal drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "filler_key": list(self.filler_key),
            "identity_key": list(self.identity_key),
            "ordinal": self.ordinal,
            "role_key": list(self.role_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicCandidateProjection:
    """一个携带精确结构、来源和生命周期的已学习 Proposition。"""

    surface: str
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_anchor_key: tuple[int, ...]
    context_key: tuple[int, ...]
    occurrence_order: tuple[tuple[int, ...], ...]
    occurrences: tuple[W05V2PublicOccurrenceProjection, ...]
    role_bindings: tuple[W05V2PublicRoleBindingProjection, ...]
    source_ref_key: tuple[int, ...]
    source_record_key: tuple[int, ...]
    source_key: str
    source_commitment: str
    license_id: str
    lifecycle_status: str
    active: int
    superseded: int
    understanding_status: str
    reasoning_status: str
    evidence_count: int

    def __post_init__(self) -> None:
        for name in ("surface", "source_key", "license_id",
                     "understanding_status", "reasoning_status"):
            _text(getattr(self, name), where=f"candidate {name}")
        for name in ("proposition_key", "predicate_key", "source_anchor_key",
                     "context_key", "source_record_key"):
            _strict_key(getattr(self, name), where=f"candidate {name}")
        _source_ref_key(self.source_ref_key, where="candidate SourceRef key")
        if (not isinstance(self.occurrence_order, tuple)
                or any(not isinstance(item, tuple) for item in self.occurrence_order)):
            raise W05V2PublicQueryError("candidate occurrence order drifted")
        for item in self.occurrence_order:
            _strict_key(item, where="candidate occurrence order item")
        if (not isinstance(self.occurrences, tuple) or not self.occurrences
                or any(not isinstance(item, W05V2PublicOccurrenceProjection)
                       for item in self.occurrences)
                or tuple(item.identity_key for item in self.occurrences)
                != self.occurrence_order):
            raise W05V2PublicQueryError("candidate occurrence projection drifted")
        if (not isinstance(self.role_bindings, tuple) or not self.role_bindings
                or any(not isinstance(item, W05V2PublicRoleBindingProjection)
                       for item in self.role_bindings)):
            raise W05V2PublicQueryError("candidate RoleBinding projection drifted")
        _sha256(self.source_commitment, where="candidate source commitment")
        if (self.lifecycle_status not in W05_V2_PUBLIC_QUERY_LIFECYCLES
                or self.active not in {0, 1}
                or self.superseded not in {0, 1}
                or self.active + self.superseded > 1
                or type(self.evidence_count) is not int
                or self.evidence_count < 0):
            raise W05V2PublicQueryError("candidate lifecycle projection drifted")
        if self.lifecycle_status == "ACTIVE" and self.active != 1:
            raise W05V2PublicQueryError("ACTIVE candidate lacks active state")
        if self.lifecycle_status == "SUPERSEDED" and self.superseded != 1:
            raise W05V2PublicQueryError(
                "SUPERSEDED candidate lacks superseded state")

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "context_key": list(self.context_key),
            "evidence_count": self.evidence_count,
            "license_id": self.license_id,
            "lifecycle_status": self.lifecycle_status,
            "occurrence_order": [list(item) for item in self.occurrence_order],
            "occurrences": [item.to_dict() for item in self.occurrences],
            "predicate_key": list(self.predicate_key),
            "proposition_key": list(self.proposition_key),
            "reasoning_status": self.reasoning_status,
            "role_bindings": [item.to_dict() for item in self.role_bindings],
            "source_anchor_key": list(self.source_anchor_key),
            "source_commitment": self.source_commitment,
            "source_key": self.source_key,
            "source_record_key": list(self.source_record_key),
            "source_ref_key": list(self.source_ref_key),
            "superseded": self.superseded,
            "surface": self.surface,
            "understanding_status": self.understanding_status,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicGenerationProjection:
    """一个同时绑定模板来源与目标来源的 construction 选项。"""

    surface: str
    construction_key: tuple[int, ...]
    construction_source_proposition_key: tuple[int, ...]
    construction_source_ref_key: tuple[int, ...]
    construction_source_commitment: str
    target_proposition_key: tuple[int, ...]
    target_predicate_key: tuple[int, ...]
    target_source_ref_key: tuple[int, ...]
    target_source_commitment: str
    context_key: tuple[int, ...]
    occurrence_order: tuple[tuple[int, ...], ...]
    role_binding_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _text(self.surface, where="generation surface")
        for name in (
                "construction_key", "construction_source_proposition_key",
                "target_proposition_key", "target_predicate_key", "context_key"):
            _strict_key(getattr(self, name), where=f"generation {name}")
        _source_ref_key(
            self.construction_source_ref_key,
            where="generation construction SourceRef key",
        )
        _source_ref_key(
            self.target_source_ref_key,
            where="generation target SourceRef key",
        )
        for name in ("occurrence_order", "role_binding_keys"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise W05V2PublicQueryError(f"generation {name} is empty")
            for item in value:
                _strict_key(item, where=f"generation {name} item")
        _sha256(
            self.construction_source_commitment,
            where="generation construction source commitment",
        )
        _sha256(
            self.target_source_commitment,
            where="generation target source commitment",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "construction_key": list(self.construction_key),
            "construction_source_commitment": self.construction_source_commitment,
            "construction_source_proposition_key": list(
                self.construction_source_proposition_key),
            "construction_source_ref_key": list(
                self.construction_source_ref_key),
            "context_key": list(self.context_key),
            "occurrence_order": [list(item) for item in self.occurrence_order],
            "role_binding_keys": [list(item) for item in self.role_binding_keys],
            "surface": self.surface,
            "target_predicate_key": list(self.target_predicate_key),
            "target_proposition_key": list(self.target_proposition_key),
            "target_source_commitment": self.target_source_commitment,
            "target_source_ref_key": list(self.target_source_ref_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicQueryResult:
    """不含 evaluator 标签或预写答案的安全 W-05 查询结果。"""

    query: W05V2PublicQuery
    status: str
    candidates: tuple[W05V2PublicCandidateProjection, ...]
    selected_proposition_key: tuple[int, ...] | None
    clarify_required: int
    selected_reasoning_status: str
    generation_status: str
    generation_options: tuple[W05V2PublicGenerationProjection, ...]
    source_binding_sha256: str
    record_commitment: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W05V2PublicQuery)
                or self.status not in W05_V2_PUBLIC_QUERY_STATUSES
                or not isinstance(self.candidates, tuple)
                or any(not isinstance(item, W05V2PublicCandidateProjection)
                       for item in self.candidates)
                or not isinstance(self.generation_options, tuple)
                or any(not isinstance(item, W05V2PublicGenerationProjection)
                       for item in self.generation_options)):
            raise W05V2PublicQueryError("query result projection drifted")
        if self.status == "UNIQUE":
            if self.selected_proposition_key is None:
                raise W05V2PublicQueryError("UNIQUE query lacks Proposition")
            _strict_key(
                self.selected_proposition_key,
                where="selected Proposition key",
            )
            selected = tuple(
                item for item in self.candidates
                if item.proposition_key == self.selected_proposition_key)
            if (len(selected) != 1 or selected[0].lifecycle_status != "ACTIVE"
                    or self.selected_reasoning_status != "AUTHORIZED"):
                raise W05V2PublicQueryError(
                    "UNIQUE query selected a non-authorized Proposition")
        elif self.selected_proposition_key is not None:
            raise W05V2PublicQueryError(
                "non-UNIQUE query selected a Proposition")
        if self.clarify_required != int(self.status == "MULTI"):
            raise W05V2PublicQueryError("query clarify flag drifted")
        _text(self.selected_reasoning_status, where="selected reasoning status")
        _text(self.generation_status, where="generation status")
        _sha256(self.source_binding_sha256, where="source binding")
        _sha256(self.record_commitment, where="record commitment")
        if ((self.status != "UNIQUE"
             or not self.query.allow_generation)
                and (self.generation_status
                     != W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
                     or self.generation_options)):
            raise W05V2PublicQueryError(
                "unauthorized query ran Generation")
        if (self.generation_status
                == W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
                and self.generation_options):
            raise W05V2PublicQueryError(
                "NOT_RUN generation cannot publish options")
        if (self.generation_status == "READY"
                and not self.generation_options):
            raise W05V2PublicQueryError(
                "READY generation lacks source-bound options")
        if (self.experimental, self.formal_mastery_claim,
                self.w05_started) != (1, 0, 0):
            raise W05V2PublicQueryError("query result boundary drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "clarify_required": self.clarify_required,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "generation_options": [
                item.to_dict() for item in self.generation_options],
            "generation_status": self.generation_status,
            "query": self.query.to_dict(),
            "record_commitment": self.record_commitment,
            "selected_proposition_key": (
                None if self.selected_proposition_key is None
                else list(self.selected_proposition_key)
            ),
            "selected_reasoning_status": self.selected_reasoning_status,
            "source_binding_sha256": self.source_binding_sha256,
            "status": self.status,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN",
    "W05_V2_PUBLIC_QUERY_LIFECYCLES",
    "W05_V2_PUBLIC_QUERY_STATUSES",
    "W05V2PublicCandidateProjection",
    "W05V2PublicGenerationProjection",
    "W05V2PublicOccurrenceProjection",
    "W05V2PublicQuery",
    "W05V2PublicQueryError",
    "W05V2PublicQueryResult",
    "W05V2PublicRoleBindingProjection",
]
