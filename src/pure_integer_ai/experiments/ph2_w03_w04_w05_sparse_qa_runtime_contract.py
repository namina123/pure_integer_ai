"""FT22 reusable sparse question-answer runtime value contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    RawQuestionFeatureRegistryAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch_contract import (
    RawQuestionSparseDispatchIndex,
    RawQuestionSparseDispatchProbe,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_STATUSES,
    RawQuestionRequest,
)


SPARSE_QA_RUNTIME_SHA256 = (
    "254d2288324e7821b945d6cdb604eca31881be3c5b1cffe968c0bfc07b35218d")
SPARSE_QA_RUNTIME_EXPRESSION_BOUNDARY = (
    ("knowledge_source", "PUBLIC_W03_W04_W05_LEARNED_ARTIFACTS"),
    ("lifecycle", "BUILD_ONCE_QUERY_MANY"),
    ("default_dispatch", "FT20_SPARSE_HOT_PATH"),
    ("audit_projection", "EXPLICIT_ONLY"),
    ("source_binding", "OPTIONAL_INPUT_AND_RESOLVED_OUTPUT"),
    ("answer_templates", "FORBIDDEN"),
    ("manual_role_keys", "FORBIDDEN"),
    ("formal_claim", "FORBIDDEN"),
    ("wall_clock_gate", "FORBIDDEN"),
)


# object-model: exception
class W03W04W05SparseQARuntimeError(ValueError):
    """The FT22 runtime, result, or deterministic probe drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05SparseQARuntimeError(
            f"{where} is not a canonical SHA-256")
    return value


def _strict_key(
        value: tuple[int, ...],
        *,
        where: str,
        ) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05SparseQARuntimeError(
            f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQAFrozenIdentities:
    """Frozen FT16-FT20 identities consumed by one runtime."""

    registry_sha256: str
    feature_index_sha256: str
    construction_index_sha256: str
    alias_frame_anchor_sha256: str
    sparse_dispatch_sha256: str

    def __post_init__(self) -> None:
        for name in (
                "registry_sha256",
                "feature_index_sha256",
                "construction_index_sha256",
                "alias_frame_anchor_sha256",
                "sparse_dispatch_sha256"):
            _sha256(getattr(self, name), where=f"FT22 {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_frame_anchor_sha256": self.alias_frame_anchor_sha256,
            "construction_index_sha256": self.construction_index_sha256,
            "feature_index_sha256": self.feature_index_sha256,
            "registry_sha256": self.registry_sha256,
            "sparse_dispatch_sha256": self.sparse_dispatch_sha256,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQARuntimeBuildProbe:
    """Deterministic build and learned-object inventory counts."""

    runtime_build_count: int
    registry_build_count: int
    feature_index_build_count: int
    construction_index_build_count: int
    alias_frame_anchor_build_count: int
    sparse_dispatch_build_count: int
    registry_entry_count: int
    explicit_construction_count: int
    implicit_construction_count: int
    learned_alias_route_count: int
    exact_index_row_count: int
    alias_index_row_count: int
    alias_frame_count: int
    implicit_index_row_count: int
    dispatch_entry_count: int

    def __post_init__(self) -> None:
        build_counts = (
            self.runtime_build_count,
            self.registry_build_count,
            self.feature_index_build_count,
            self.construction_index_build_count,
            self.alias_frame_anchor_build_count,
            self.sparse_dispatch_build_count,
        )
        object_counts = (
            self.registry_entry_count,
            self.explicit_construction_count,
            self.implicit_construction_count,
            self.learned_alias_route_count,
            self.exact_index_row_count,
            self.alias_index_row_count,
            self.alias_frame_count,
            self.implicit_index_row_count,
            self.dispatch_entry_count,
        )
        if any(type(item) is not int or item != 1 for item in build_counts):
            raise W03W04W05SparseQARuntimeError(
                "FT22 runtime layers were not built exactly once")
        if any(type(item) is not int or item <= 0 for item in object_counts):
            raise W03W04W05SparseQARuntimeError(
                "FT22 learned-object inventory is invalid")
        if self.registry_entry_count != self.dispatch_entry_count:
            raise W03W04W05SparseQARuntimeError(
                "FT22 dispatch inventory escaped registry ownership")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_frame_anchor_build_count": (
                self.alias_frame_anchor_build_count),
            "alias_frame_count": self.alias_frame_count,
            "alias_index_row_count": self.alias_index_row_count,
            "construction_index_build_count": (
                self.construction_index_build_count),
            "dispatch_entry_count": self.dispatch_entry_count,
            "exact_index_row_count": self.exact_index_row_count,
            "explicit_construction_count": (
                self.explicit_construction_count),
            "feature_index_build_count": self.feature_index_build_count,
            "implicit_construction_count": (
                self.implicit_construction_count),
            "implicit_index_row_count": self.implicit_index_row_count,
            "learned_alias_route_count": self.learned_alias_route_count,
            "registry_build_count": self.registry_build_count,
            "registry_entry_count": self.registry_entry_count,
            "runtime_build_count": self.runtime_build_count,
            "sparse_dispatch_build_count": self.sparse_dispatch_build_count,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQAEntryPublicStateMemo:
    """Identity-neutral answer-state commitment for one registry entry."""

    entry_sha256: str
    public_state_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.entry_sha256, where="FT24A memo entry")
        _sha256(self.public_state_sha256, where="FT24A public state")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQARuntime:
    """One immutable public learned runtime built for repeated queries."""

    dispatch_index: RawQuestionSparseDispatchIndex
    frozen_identities: SparseQAFrozenIdentities
    build_probe: SparseQARuntimeBuildProbe
    identity_sha256: str
    entry_public_state_memo: tuple[SparseQAEntryPublicStateMemo, ...] = ()
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(
                self.dispatch_index, RawQuestionSparseDispatchIndex)
                or not isinstance(
                    self.frozen_identities, SparseQAFrozenIdentities)
                or not isinstance(
                    self.build_probe, SparseQARuntimeBuildProbe)
                or not isinstance(self.entry_public_state_memo, tuple)
                or any(not isinstance(item, SparseQAEntryPublicStateMemo)
                       for item in self.entry_public_state_memo)):
            raise TypeError("FT22 runtime inputs are invalid")
        construction = self.dispatch_index.anchor_index.construction_index
        feature = construction.feature_index
        registry = feature.registry
        expected = SparseQAFrozenIdentities(
            registry.identity_sha256,
            feature.identity_sha256,
            construction.identity_sha256,
            self.dispatch_index.anchor_index.identity_sha256,
            self.dispatch_index.identity_sha256,
        )
        if self.frozen_identities != expected:
            raise W03W04W05SparseQARuntimeError(
                "FT22 identities escaped the assembled index chain")
        if (self.build_probe.registry_entry_count != len(registry.entries)
                or self.build_probe.dispatch_entry_count
                != len(self.dispatch_index.entries)):
            raise W03W04W05SparseQARuntimeError(
                "FT22 build probe escaped the assembled runtime")
        entry_sha256s = tuple(item.sha256() for item in registry.entries)
        memo_entry_sha256s = tuple(
            item.entry_sha256 for item in self.entry_public_state_memo)
        if (memo_entry_sha256s != tuple(sorted(entry_sha256s))
                or len(memo_entry_sha256s) != len(entry_sha256s)):
            raise W03W04W05SparseQARuntimeError(
                "FT24A public-state memo escaped registry ownership")
        _sha256(self.identity_sha256, where="FT22 runtime")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05SparseQARuntimeError(
                "FT22 runtime identity drifted")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05SparseQARuntimeError(
                "FT22 formal-state boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "build_probe": self.build_probe.to_dict(),
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "frozen_identities": self.frozen_identities.to_dict(),
            "runtime_boundary": [
                {"capability": key, "status": status}
                for key, status in SPARSE_QA_RUNTIME_EXPRESSION_BOUNDARY
            ],
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQAResult:
    """Short typed answer with resolved source and optional FT16 audit."""

    runtime_identity_sha256: str
    frozen_identities: SparseQAFrozenIdentities
    request: RawQuestionRequest
    status: str
    answer_surface: str | None
    decisive_phase: str | None
    selected_entry_sha256: str | None
    selected_source_record_key: tuple[int, ...] | None
    execution_record_sha256: str
    dispatch_probe: RawQuestionSparseDispatchProbe
    audit_result: RawQuestionFeatureRegistryAnswerResult | None = None
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        _sha256(self.runtime_identity_sha256, where="FT22 result runtime")
        _sha256(self.execution_record_sha256, where="FT22 execution record")
        if (not isinstance(self.frozen_identities, SparseQAFrozenIdentities)
                or not isinstance(self.request, RawQuestionRequest)
                or self.status not in RAW_QUESTION_STATUSES
                or not isinstance(
                    self.dispatch_probe, RawQuestionSparseDispatchProbe)):
            raise W03W04W05SparseQARuntimeError(
                "FT22 short result projection drifted")
        if self.status == "ANSWER":
            if (not isinstance(self.answer_surface, str)
                    or not self.answer_surface
                    or self.decisive_phase not in {
                        "EXACT", "ALIAS", "IMPLICIT"}
                    or self.selected_entry_sha256 is None
                    or self.selected_source_record_key is None):
                raise W03W04W05SparseQARuntimeError(
                    "FT22 ANSWER lacks typed answer or source")
            _sha256(self.selected_entry_sha256, where="FT22 selected entry")
            _strict_key(
                self.selected_source_record_key,
                where="FT22 selected SourceRef",
            )
        elif self.status == "CLARIFY":
            if (self.answer_surface is not None
                    or self.decisive_phase not in {
                        "EXACT", "ALIAS", "IMPLICIT"}
                    or self.selected_entry_sha256 is not None
                    or self.selected_source_record_key is not None):
                raise W03W04W05SparseQARuntimeError(
                    "FT22 CLARIFY selected an answer or source")
        elif (self.answer_surface is not None
                or self.decisive_phase is not None
                or self.selected_entry_sha256 is not None
                or self.selected_source_record_key is not None):
            raise W03W04W05SparseQARuntimeError(
                "FT22 UNKNOWN published an answer or source")
        if self.audit_result is not None:
            audit = self.audit_result
            if (not isinstance(audit, RawQuestionFeatureRegistryAnswerResult)
                    or audit.request != self.request
                    or audit.status != self.status
                    or audit.answer_surface != self.answer_surface
                    or audit.decisive_phase != self.decisive_phase
                    or audit.selected_entry_sha256
                    != self.selected_entry_sha256
                    or audit.registry_identity_sha256
                    != self.frozen_identities.registry_sha256):
                raise W03W04W05SparseQARuntimeError(
                    "FT22 audit escaped its sparse decision")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05SparseQARuntimeError(
                "FT22 result formal-state boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        value = {
            "answer_surface": self.answer_surface,
            "decisive_phase": self.decisive_phase,
            "dispatch_probe": self.dispatch_probe.to_dict(),
            "execution_record_sha256": self.execution_record_sha256,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "frozen_identities": self.frozen_identities.to_dict(),
            "request": self.request.to_dict(),
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "selected_entry_sha256": self.selected_entry_sha256,
            "selected_source_record_key": (
                None if self.selected_source_record_key is None
                else list(self.selected_source_record_key)
            ),
            "status": self.status,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }
        if self.audit_result is not None:
            value["audit"] = self.audit_result.to_dict()
        return value

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQAQueryProbe:
    """Deterministic counts for repeated warm queries on one runtime."""

    runtime_build_count: int
    query_count: int
    execution_record_count: int
    result_object_count: int
    audit_projection_count: int
    created_sparse_trace_count: int
    created_audit_trace_count: int
    unique_request_count: int
    unique_result_sha256_count: int
    result_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        counts = (
            self.runtime_build_count,
            self.query_count,
            self.execution_record_count,
            self.result_object_count,
            self.audit_projection_count,
            self.created_sparse_trace_count,
            self.created_audit_trace_count,
            self.unique_request_count,
            self.unique_result_sha256_count,
        )
        if any(type(item) is not int or item < 0 for item in counts):
            raise W03W04W05SparseQARuntimeError(
                "FT22 query probe count is invalid")
        if (self.runtime_build_count != 1 or self.query_count <= 0
                or self.execution_record_count != self.query_count
                or self.result_object_count != self.query_count
                or not 1 <= self.unique_request_count <= self.query_count
                or not 1 <= self.unique_result_sha256_count
                <= self.query_count
                or len(self.result_sha256s) != self.query_count
                or len(set(self.result_sha256s))
                != self.unique_result_sha256_count):
            raise W03W04W05SparseQARuntimeError(
                "FT22 query/result counts diverged")
        for item in self.result_sha256s:
            _sha256(item, where="FT22 query result")

    @property
    def bit_identical(self) -> bool:
        return self.unique_request_count == self.unique_result_sha256_count == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_projection_count": self.audit_projection_count,
            "bit_identical": self.bit_identical,
            "created_audit_trace_count": self.created_audit_trace_count,
            "created_sparse_trace_count": self.created_sparse_trace_count,
            "execution_record_count": self.execution_record_count,
            "query_count": self.query_count,
            "result_object_count": self.result_object_count,
            "result_sha256s": list(self.result_sha256s),
            "runtime_build_count": self.runtime_build_count,
            "unique_request_count": self.unique_request_count,
            "unique_result_sha256_count": (
                self.unique_result_sha256_count),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQAQueryBatch:
    """Ordered warm-query results and their deterministic probe."""

    results: tuple[SparseQAResult, ...]
    probe: SparseQAQueryProbe

    def __post_init__(self) -> None:
        if (not isinstance(self.results, tuple) or not self.results
                or any(not isinstance(item, SparseQAResult)
                       for item in self.results)
                or not isinstance(self.probe, SparseQAQueryProbe)
                or len(self.results) != self.probe.query_count
                or tuple(item.sha256() for item in self.results)
                != self.probe.result_sha256s):
            raise W03W04W05SparseQARuntimeError(
                "FT22 query batch escaped its deterministic probe")

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe.to_dict(),
            "results": [item.to_dict() for item in self.results],
        }


__all__ = [
    "SPARSE_QA_RUNTIME_EXPRESSION_BOUNDARY",
    "SPARSE_QA_RUNTIME_SHA256",
    "SparseQAFrozenIdentities",
    "SparseQAEntryPublicStateMemo",
    "SparseQAQueryBatch",
    "SparseQAQueryProbe",
    "SparseQAResult",
    "SparseQARuntime",
    "SparseQARuntimeBuildProbe",
    "W03W04W05SparseQARuntimeError",
]
