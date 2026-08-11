"""冻结 FT20 稀疏问题分派索引、执行记录与审计探针。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor_contract import (
    RawQuestionAliasFrameAnchorIndex,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    RawQuestionFeatureDispatchTrace,
    RawQuestionFeatureInterpretation,
    RawQuestionFeatureRegistryEntry,
    raw_question_feature_trace_interpretation,
    resolve_question_feature_interpretations,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_STATUSES,
    RawQuestionRequest,
)


QUESTION_SPARSE_DISPATCH_SHA256 = (
    "0ba334b34b0863d588103460a800fa3b2de0256fe788c95ec622bef325d3066c")
QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY = (
    ("index_source", "FT19_ANCHOR_AND_FT16_REGISTRY"),
    ("hot_execution", "CANDIDATE_ENTRIES_ONLY"),
    ("missing_unknowns", "OMITTED_FROM_HOT_RECORD"),
    ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
    ("audit_projection", "ON_DEMAND_BYTE_IDENTICAL_TO_FT16"),
    ("source_binding", "PRESERVED"),
    ("wall_clock_gate", "FORBIDDEN"),
    ("handwritten_dispatch", "FORBIDDEN"),
)


# object-model: exception
class W03W04W05QuestionSparseDispatchError(ValueError):
    """FT20 稀疏 entry、阶段记录、决议或审计投影发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionSparseDispatchError(
            f"{where} is not a canonical SHA-256")
    return value


def _sha_tuple(values: tuple[str, ...], *, where: str) -> tuple[str, ...]:
    if (not isinstance(values, tuple)
            or values != tuple(sorted(set(values)))):
        raise W03W04W05QuestionSparseDispatchError(
            f"{where} identities are not canonical")
    for value in values:
        _sha256(value, where=where)
    return values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparseDispatchEntry:
    """供热路径按 SHA 二分访问的一个既有 registry entry。"""

    entry_sha256: str
    entry: RawQuestionFeatureRegistryEntry

    def __post_init__(self) -> None:
        _sha256(self.entry_sha256, where="sparse dispatch entry")
        if not isinstance(self.entry, RawQuestionFeatureRegistryEntry):
            raise TypeError("sparse dispatch registry entry is invalid")
        if self.entry_sha256 != self.entry.sha256():
            raise W03W04W05QuestionSparseDispatchError(
                "sparse dispatch entry identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {"entry_sha256": self.entry_sha256}


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparseDispatchIndex:
    """FT19 anchor 与 FT16 registry entry 的不可变稀疏分派索引。"""

    anchor_index: RawQuestionAliasFrameAnchorIndex
    entries: tuple[RawQuestionSparseDispatchEntry, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_index, RawQuestionAliasFrameAnchorIndex):
            raise TypeError("sparse dispatch anchor index is invalid")
        if (not isinstance(self.entries, tuple)
                or any(not isinstance(item, RawQuestionSparseDispatchEntry)
                       for item in self.entries)
                or self.entries != tuple(sorted(
                    self.entries, key=lambda item: item.entry_sha256))
                or len({item.entry_sha256 for item in self.entries})
                != len(self.entries)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse dispatch entries are not canonical")
        registry_entries = self.anchor_index.construction_index.registry.entries
        if (tuple(item.entry for item in self.entries) != registry_entries
                or tuple(item.entry_sha256 for item in self.entries)
                != tuple(item.sha256() for item in registry_entries)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse dispatch entries escaped registry ownership")
        _sha256(self.identity_sha256, where="sparse dispatch index")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05QuestionSparseDispatchError(
                "sparse dispatch index identity drifted")

    @property
    def registry_identity_sha256(self) -> str:
        return self.anchor_index.construction_index.registry.identity_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_index_identity_sha256": self.anchor_index.identity_sha256,
            "entries": [item.to_dict() for item in self.entries],
            "expression_boundary": [
                {"capability": key, "status": status}
                for key, status in QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY
            ],
            "registry_identity_sha256": self.registry_identity_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_raw_question_sparse_dispatch_index(
        anchor_index: RawQuestionAliasFrameAnchorIndex,
        *,
        expected_identity_sha256: str | None = None,
        ) -> RawQuestionSparseDispatchIndex:
    """从 FT19 与既有 canonical registry 顺序派生 entry 二分索引。"""
    if not isinstance(anchor_index, RawQuestionAliasFrameAnchorIndex):
        raise TypeError("sparse dispatch index input is invalid")
    entries = tuple(
        RawQuestionSparseDispatchEntry(entry.sha256(), entry)
        for entry in anchor_index.construction_index.registry.entries
    )
    payload = {
        "anchor_index_identity_sha256": anchor_index.identity_sha256,
        "entries": [item.to_dict() for item in entries],
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY
        ],
        "registry_identity_sha256": (
            anchor_index.construction_index.registry.identity_sha256),
    }
    identity = _sha(payload)
    value = RawQuestionSparseDispatchIndex(anchor_index, entries, identity)
    if (expected_identity_sha256 is not None
            and identity != _sha256(
                expected_identity_sha256,
                where="expected sparse dispatch index")):
        raise W03W04W05QuestionSparseDispatchError(
            "sparse dispatch index commitment drifted")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparseDecision:
    """不依赖完整 UNKNOWN 轨迹的类型化全局阶段决议。"""

    status: str
    answer_surface: str | None
    decisive_phase: str | None
    interpretations: tuple[RawQuestionFeatureInterpretation, ...]
    selected_entry_sha256: str | None

    def __post_init__(self) -> None:
        if (self.status not in RAW_QUESTION_STATUSES
                or self.decisive_phase not in {
                    None, *QUESTION_FEATURE_DISPATCH_PHASES}
                or not isinstance(self.interpretations, tuple)
                or any(not isinstance(item, RawQuestionFeatureInterpretation)
                       for item in self.interpretations)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse question decision projection drifted")
        ordered = tuple(sorted(
            {item.sha256(): item for item in self.interpretations}.values(),
            key=RawQuestionFeatureInterpretation.sha256,
        ))
        if ordered != self.interpretations:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse decision interpretations are not canonical")
        if self.status == "UNKNOWN":
            if (self.answer_surface is not None
                    or self.decisive_phase is not None
                    or self.interpretations
                    or self.selected_entry_sha256 is not None):
                raise W03W04W05QuestionSparseDispatchError(
                    "sparse UNKNOWN decision published an answer")
        elif self.status == "ANSWER":
            if (not isinstance(self.answer_surface, str)
                    or not self.answer_surface
                    or self.decisive_phase is None
                    or len(self.interpretations) != 1
                    or self.selected_entry_sha256 is None):
                raise W03W04W05QuestionSparseDispatchError(
                    "sparse ANSWER decision is incomplete")
            _sha256(
                self.selected_entry_sha256,
                where="sparse selected entry",
            )
        elif (self.answer_surface is not None
                or self.decisive_phase is None
                or self.selected_entry_sha256 is not None):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse CLARIFY decision selected an answer")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "decisive_phase": self.decisive_phase,
            "interpretations": [
                item.to_dict() for item in self.interpretations],
            "selected_entry_sha256": self.selected_entry_sha256,
            "status": self.status,
        }


def project_sparse_question_phase_decision(
        traces: tuple[RawQuestionFeatureDispatchTrace, ...],
        phase: str,
        ) -> RawQuestionSparseDecision | None:
    """把一个阶段的候选结果投影为全局决议；缺失 entry 等价 UNKNOWN。"""
    if (not isinstance(traces, tuple)
            or any(not isinstance(item, RawQuestionFeatureDispatchTrace)
                   for item in traces)
            or traces != tuple(sorted(
                traces, key=lambda item: item.entry_sha256))
            or len({item.entry_sha256 for item in traces}) != len(traces)
            or phase not in QUESTION_FEATURE_DISPATCH_PHASES):
        raise W03W04W05QuestionSparseDispatchError(
            "sparse phase decision inputs are not canonical")
    if not traces:
        return None
    statuses = tuple(item.status_at(phase) for item in traces)
    if all(item == "UNKNOWN" for item in statuses):
        return None
    answered = tuple(
        item for item in traces if item.status_at(phase) == "ANSWER")
    interpretations = tuple(sorted(
        {
            value.sha256(): value
            for value in (
                raw_question_feature_trace_interpretation(item, phase)
                for item in answered)
        }.values(),
        key=RawQuestionFeatureInterpretation.sha256,
    ))
    resolution = resolve_question_feature_interpretations(interpretations)
    if "CLARIFY" in statuses or resolution == "AMBIGUOUS":
        return RawQuestionSparseDecision(
            "CLARIFY", None, phase, interpretations, None)
    if resolution != "SELECTED":
        raise W03W04W05QuestionSparseDispatchError(
            "decisive sparse phase lacks an answer interpretation")
    selected = min(answered, key=lambda item: item.entry_sha256)
    return RawQuestionSparseDecision(
        "ANSWER",
        selected.answer_at(phase),
        phase,
        interpretations,
        selected.entry_sha256,
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparsePhaseVisit:
    """一个已到达阶段实际候选的 entry 与 construction 计数。"""

    phase: str
    entry_sha256s: tuple[str, ...]
    construction_count: int

    def __post_init__(self) -> None:
        if self.phase not in QUESTION_FEATURE_DISPATCH_PHASES:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse phase visit is invalid")
        _sha_tuple(
            self.entry_sha256s,
            where="sparse phase visit entry",
        )
        if (type(self.construction_count) is not int
                or self.construction_count < len(self.entry_sha256s)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse phase construction count is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "construction_count": self.construction_count,
            "entry_sha256s": list(self.entry_sha256s),
            "phase": self.phase,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparseExecutionRecord:
    """只保存候选 entry 真实执行结果的不可变热路径记录。"""

    dispatch_index_identity_sha256: str
    registry_identity_sha256: str
    request: RawQuestionRequest
    decision: RawQuestionSparseDecision
    phase_visits: tuple[RawQuestionSparsePhaseVisit, ...]
    traces: tuple[RawQuestionFeatureDispatchTrace, ...]
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        _sha256(
            self.dispatch_index_identity_sha256,
            where="sparse execution index",
        )
        _sha256(
            self.registry_identity_sha256,
            where="sparse execution registry",
        )
        if (not isinstance(self.request, RawQuestionRequest)
                or not isinstance(self.decision, RawQuestionSparseDecision)
                or not isinstance(self.phase_visits, tuple)
                or not self.phase_visits
                or any(not isinstance(item, RawQuestionSparsePhaseVisit)
                       for item in self.phase_visits)
                or not isinstance(self.traces, tuple)
                or any(not isinstance(item, RawQuestionFeatureDispatchTrace)
                       for item in self.traces)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution record projection drifted")
        phases = tuple(item.phase for item in self.phase_visits)
        if phases != QUESTION_FEATURE_DISPATCH_PHASES[:len(phases)]:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution phase order drifted")
        expected_phase_count = {
            "EXACT": 1,
            "ALIAS": 2,
            "IMPLICIT": 3,
            None: 3,
        }[self.decision.decisive_phase]
        if len(self.phase_visits) != expected_phase_count:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution did not stop at its decisive phase")
        if (self.traces != tuple(sorted(
                self.traces, key=lambda item: item.entry_sha256))
                or len({item.entry_sha256 for item in self.traces})
                != len(self.traces)
                or any(item.exact_result.request != self.request
                       for item in self.traces)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution traces are not canonical")
        visits = {item.phase: set(item.entry_sha256s)
                  for item in self.phase_visits}
        visited = set().union(*visits.values())
        if visited != {item.entry_sha256 for item in self.traces}:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse traces escaped candidate entries")
        expected_alias = visits.get("ALIAS", set()) | visits.get(
            "IMPLICIT", set())
        actual_alias = {
            item.entry_sha256 for item in self.traces
            if item.alias_result is not None
        }
        expected_implicit = visits.get("IMPLICIT", set())
        actual_implicit = {
            item.entry_sha256 for item in self.traces
            if item.implicit_result is not None
        }
        if (actual_alias != expected_alias
                or actual_implicit != expected_implicit):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse trace depth escaped candidate phases")
        by_entry = {item.entry_sha256: item for item in self.traces}
        projected = None
        for visit in self.phase_visits:
            projected = project_sparse_question_phase_decision(
                tuple(by_entry[key] for key in visit.entry_sha256s),
                visit.phase,
            )
            if projected is not None:
                break
        if projected is None:
            projected = RawQuestionSparseDecision(
                "UNKNOWN", None, None, (), None)
        if projected != self.decision:
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution decision escaped its candidate results")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse execution boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "dispatch_index_identity_sha256": (
                self.dispatch_index_identity_sha256),
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "phase_visits": [item.to_dict() for item in self.phase_visits],
            "registry_identity_sha256": self.registry_identity_sha256,
            "request": self.request.to_dict(),
            "traces": [item.to_dict() for item in self.traces],
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionSparseDispatchProbe:
    """不用墙钟表达稀疏执行与完整审计规模的确定性计数。"""

    visited_entry_sha256s: tuple[str, ...]
    phase_candidate_entry_counts: tuple[int, ...]
    phase_candidate_construction_counts: tuple[int, ...]
    created_exact_result_count: int
    created_alias_result_count: int
    created_implicit_result_count: int
    created_dispatch_trace_count: int
    sparse_trace_count: int
    projected_trace_count: int

    def __post_init__(self) -> None:
        _sha_tuple(
            self.visited_entry_sha256s,
            where="sparse probe visited entry",
        )
        for values in (
                self.phase_candidate_entry_counts,
                self.phase_candidate_construction_counts):
            if (not isinstance(values, tuple)
                    or not 1 <= len(values) <= 3
                    or any(type(item) is not int or item < 0
                           for item in values)):
                raise W03W04W05QuestionSparseDispatchError(
                    "sparse probe phase counts are invalid")
        if (len(self.phase_candidate_entry_counts)
                != len(self.phase_candidate_construction_counts)):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse probe phase count depths diverged")
        counts = (
            self.created_exact_result_count,
            self.created_alias_result_count,
            self.created_implicit_result_count,
            self.created_dispatch_trace_count,
            self.sparse_trace_count,
            self.projected_trace_count,
        )
        if any(type(item) is not int or item < 0 for item in counts):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse probe object counts are invalid")
        if (self.sparse_trace_count != len(self.visited_entry_sha256s)
                or self.created_exact_result_count != self.sparse_trace_count
                or self.created_dispatch_trace_count
                != sum(self.phase_candidate_entry_counts)
                or self.created_dispatch_trace_count < self.sparse_trace_count
                or self.projected_trace_count < self.sparse_trace_count):
            raise W03W04W05QuestionSparseDispatchError(
                "sparse probe trace counts diverged")

    def to_dict(self) -> dict[str, object]:
        return {
            "created_alias_result_count": self.created_alias_result_count,
            "created_dispatch_trace_count": (
                self.created_dispatch_trace_count),
            "created_exact_result_count": self.created_exact_result_count,
            "created_implicit_result_count": (
                self.created_implicit_result_count),
            "phase_candidate_construction_counts": list(
                self.phase_candidate_construction_counts),
            "phase_candidate_entry_counts": list(
                self.phase_candidate_entry_counts),
            "projected_trace_count": self.projected_trace_count,
            "sparse_trace_count": self.sparse_trace_count,
            "visited_entry_sha256s": list(self.visited_entry_sha256s),
        }


__all__ = [
    "QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY",
    "QUESTION_SPARSE_DISPATCH_SHA256",
    "RawQuestionSparseDecision",
    "RawQuestionSparseDispatchEntry",
    "RawQuestionSparseDispatchIndex",
    "RawQuestionSparseDispatchProbe",
    "RawQuestionSparseExecutionRecord",
    "RawQuestionSparsePhaseVisit",
    "W03W04W05QuestionSparseDispatchError",
    "build_raw_question_sparse_dispatch_index",
    "project_sparse_question_phase_decision",
]
