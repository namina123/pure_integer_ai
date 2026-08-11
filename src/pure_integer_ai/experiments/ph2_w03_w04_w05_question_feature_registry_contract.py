"""FT16 已学问题特征注册表与分派结果合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer_contract import (
    W03W04W05QuestionAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    RawQuestionFeatureCatalog,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    LearnedPredicateAliasBridge,
    RawQuestionPredicateAliasAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_STATUSES,
    RawQuestionAnswerResult,
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    RawQuestionImplicitPredicateAnswerResult,
    W03W04W05ImplicitQuestionBundle,
)


QUESTION_FEATURE_DISPATCH_PHASES = ("EXACT", "ALIAS", "IMPLICIT")
QUESTION_FEATURE_REGISTRY_SHA256 = (
    "5b1067d49baaa1465cce48d120717e63e49eea291e1d92dc13d3bb8df6134244")
QUESTION_FEATURE_INTERPRETATION_STATUSES = {
    "AMBIGUOUS", "MISSING", "SELECTED"}
QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY = (
    ("caller_input", "RAW_QUESTION_AND_OPTIONAL_SOURCE_REF"),
    ("catalog_selection", "READ_ONLY_REGISTRY"),
    ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
    ("equivalent_provenance", "CONVERGES_BY_TYPED_INTERPRETATION"),
    ("non_equivalent_interpretations", "CLARIFY"),
    ("missing_structure", "UNKNOWN"),
    ("source_binding", "NEVER_MERGED_ACROSS_PUBLIC_BATCHES"),
)


# object-model: exception
class W03W04W05QuestionFeatureRegistryError(ValueError):
    """注册表、阶段轨迹或类型化解释发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionFeatureRegistryError(
            f"{where} is not a canonical SHA-256")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05QuestionFeatureRegistryError(
            f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureRegistryEntry:
    """保存一个目录及其既有 alias 与隐式运行时。"""

    feature_catalog: RawQuestionFeatureCatalog
    alias_bridge: LearnedPredicateAliasBridge
    implicit_bundle: W03W04W05ImplicitQuestionBundle

    def __post_init__(self) -> None:
        if (not isinstance(self.feature_catalog, RawQuestionFeatureCatalog)
                or not isinstance(
                    self.alias_bridge, LearnedPredicateAliasBridge)
                or not isinstance(
                    self.implicit_bundle, W03W04W05ImplicitQuestionBundle)):
            raise TypeError("question feature registry entry is invalid")
        if (self.implicit_bundle.explicit_catalog != self.feature_catalog
                or self.alias_bridge.overlay_validation_sha256
                != self.feature_catalog.overlay_validation_sha256
                or self.alias_bridge.raw_question_bundle_sha256
                != self.feature_catalog.bundle_identity_sha256):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry entry crossed catalog ownership")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_bridge_sha256": self.alias_bridge.identity_sha256,
            "feature_catalog_sha256": self.feature_catalog.sha256(),
            "implicit_bundle_sha256": self.implicit_bundle.identity_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureRegistry:
    """独立来源绑定目录的规范只读清单。"""

    entries: tuple[RawQuestionFeatureRegistryEntry, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.entries, tuple) or len(self.entries) < 2
                or any(not isinstance(item, RawQuestionFeatureRegistryEntry)
                       for item in self.entries)):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry requires multiple entries")
        if (self.entries != tuple(sorted(
                self.entries, key=RawQuestionFeatureRegistryEntry.sha256))
                or len({item.sha256() for item in self.entries})
                != len(self.entries)):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry entries are not canonical")
        _sha256(self.identity_sha256, where="question feature registry")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [item.to_dict() for item in self.entries],
            "expression_boundary": [
                {"capability": key, "status": status}
                for key, status in (
                    QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY)
            ],
            "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_raw_question_feature_registry(
        entries: tuple[RawQuestionFeatureRegistryEntry, ...],
        *,
        expected_identity_sha256: str | None = None,
        ) -> RawQuestionFeatureRegistry:
    """冻结独立所有权目录，但不合并其公开批次。"""
    if (not isinstance(entries, tuple)
            or any(not isinstance(item, RawQuestionFeatureRegistryEntry)
                   for item in entries)):
        raise TypeError("question feature registry inputs are invalid")
    ordered = tuple(sorted(
        entries, key=RawQuestionFeatureRegistryEntry.sha256))
    identity = _sha({
        "entries": [item.to_dict() for item in ordered],
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY
        ],
        "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
    })
    value = RawQuestionFeatureRegistry(ordered, identity)
    if (expected_identity_sha256 is not None
            and identity != _sha256(
                expected_identity_sha256,
                where="expected question feature registry")):
        raise W03W04W05QuestionFeatureRegistryError(
            "question feature registry commitment drifted")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureInterpretation:
    """刻意排除来源轨迹的类型化回答含义。"""

    primitive_registry: str
    primitive_kind: int
    proposition_key: tuple[int, ...]
    target_role_key: tuple[int, ...]
    answer_filler_key: tuple[int, ...]
    answer_surface: str

    def __post_init__(self) -> None:
        if (not isinstance(self.primitive_registry, str)
                or not self.primitive_registry
                or self.primitive_registry.strip() != self.primitive_registry
                or type(self.primitive_kind) is not int
                or self.primitive_kind <= 0
                or not isinstance(self.answer_surface, str)
                or not self.answer_surface
                or self.answer_surface.strip() != self.answer_surface):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature interpretation text or primitive drifted")
        _strict_key(
            self.proposition_key,
            where="question feature interpretation Proposition",
        )
        _strict_key(
            self.target_role_key,
            where="question feature interpretation target Role",
        )
        _strict_key(
            self.answer_filler_key,
            where="question feature interpretation answer filler",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_filler_key": list(self.answer_filler_key),
            "answer_surface": self.answer_surface,
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "proposition_key": list(self.proposition_key),
            "target_role_key": list(self.target_role_key),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def resolve_question_feature_interpretations(
        values: tuple[RawQuestionFeatureInterpretation, ...],
        ) -> str:
    """收敛等价含义，并拒绝任一类型或回答差异。"""
    if (not isinstance(values, tuple)
            or any(not isinstance(item, RawQuestionFeatureInterpretation)
                   for item in values)):
        raise TypeError("question feature interpretations are invalid")
    identities = {item.sha256() for item in values}
    if not identities:
        return "MISSING"
    if len(identities) == 1:
        return "SELECTED"
    return "AMBIGUOUS"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureDispatchTrace:
    """一个注册项在已到达深度上的既有运行时结果。"""

    entry_sha256: str
    exact_result: RawQuestionAnswerResult
    alias_result: RawQuestionPredicateAliasAnswerResult | None
    implicit_result: RawQuestionImplicitPredicateAnswerResult | None

    def __post_init__(self) -> None:
        _sha256(self.entry_sha256, where="question feature trace entry")
        if not isinstance(self.exact_result, RawQuestionAnswerResult):
            raise TypeError("question feature trace exact result is invalid")
        if self.alias_result is not None:
            if (not isinstance(
                    self.alias_result, RawQuestionPredicateAliasAnswerResult)
                    or self.exact_result.status != "UNKNOWN"
                    or self.alias_result.exact_result != self.exact_result):
                raise W03W04W05QuestionFeatureRegistryError(
                    "question feature alias trace escaped global exact phase")
        if self.implicit_result is not None:
            if (self.alias_result is None
                    or not isinstance(
                        self.implicit_result,
                        RawQuestionImplicitPredicateAnswerResult)
                    or self.alias_result.status != "UNKNOWN"
                    or self.implicit_result.predicate_result
                    != self.alias_result):
                raise W03W04W05QuestionFeatureRegistryError(
                    "question feature implicit trace escaped global alias phase")
        request = self.exact_result.request
        if (self.alias_result is not None
                and self.alias_result.request != request
                or self.implicit_result is not None
                and self.implicit_result.request != request):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature trace requests diverged")

    def status_at(self, phase: str) -> str:
        if phase == "EXACT":
            return self.exact_result.status
        if phase == "ALIAS" and self.alias_result is not None:
            return self.alias_result.status
        if phase == "IMPLICIT" and self.implicit_result is not None:
            return self.implicit_result.status
        raise W03W04W05QuestionFeatureRegistryError(
            "question feature trace does not contain the requested phase")

    def answer_at(self, phase: str) -> str | None:
        if phase == "EXACT":
            return self.exact_result.answer_surface
        if phase == "ALIAS" and self.alias_result is not None:
            return self.alias_result.answer_surface
        if phase == "IMPLICIT" and self.implicit_result is not None:
            return self.implicit_result.answer_surface
        raise W03W04W05QuestionFeatureRegistryError(
            "question feature trace does not contain the requested phase")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_result": (
                None if self.alias_result is None
                else self.alias_result.to_dict()
            ),
            "entry_sha256": self.entry_sha256,
            "exact_result": self.exact_result.to_dict(),
            "implicit_result": (
                None if self.implicit_result is None
                else self.implicit_result.to_dict()
            ),
        }


def _trace_typed_answer(
        trace: RawQuestionFeatureDispatchTrace,
        phase: str,
        ) -> W03W04W05QuestionAnswerResult:
    typed = None
    if phase == "EXACT":
        typed = trace.exact_result.typed_result
    elif phase == "ALIAS" and trace.alias_result is not None:
        normalized = trace.alias_result.normalized_result
        typed = None if normalized is None else normalized.typed_result
    elif phase == "IMPLICIT" and trace.implicit_result is not None:
        implicit = trace.implicit_result.implicit_result
        typed = None if implicit is None else implicit.typed_result
    if (not isinstance(typed, W03W04W05QuestionAnswerResult)
            or typed.status != "ANSWER" or typed.proof is None
            or typed.vertical_result.link is None):
        raise W03W04W05QuestionFeatureRegistryError(
            "registry ANSWER lacks an existing typed answer path")
    return typed


def raw_question_feature_trace_interpretation(
        trace: RawQuestionFeatureDispatchTrace,
        phase: str,
        ) -> RawQuestionFeatureInterpretation:
    typed = _trace_typed_answer(trace, phase)
    proof = typed.proof
    link = typed.vertical_result.link
    if proof is None or link is None or typed.answer_surface is None:
        raise W03W04W05QuestionFeatureRegistryError(
            "registry typed answer projection is unavailable")
    return RawQuestionFeatureInterpretation(
        link.primitive_registry,
        link.primitive_kind,
        proof.proposition_key,
        proof.role_key,
        proof.filler_key,
        typed.answer_surface,
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureRegistryAnswerResult:
    """全局优先回答及逐目录完整阶段证据。"""

    registry_identity_sha256: str
    request: RawQuestionRequest
    status: str
    answer_surface: str | None
    decisive_phase: str | None
    interpretations: tuple[RawQuestionFeatureInterpretation, ...]
    selected_entry_sha256: str | None
    traces: tuple[RawQuestionFeatureDispatchTrace, ...]
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        _sha256(
            self.registry_identity_sha256,
            where="question feature result registry",
        )
        if (not isinstance(self.request, RawQuestionRequest)
                or self.status not in RAW_QUESTION_STATUSES
                or self.decisive_phase not in {
                    None, *QUESTION_FEATURE_DISPATCH_PHASES}
                or not isinstance(self.interpretations, tuple)
                or any(not isinstance(item, RawQuestionFeatureInterpretation)
                       for item in self.interpretations)
                or not isinstance(self.traces, tuple)
                or len(self.traces) < 2
                or any(not isinstance(item, RawQuestionFeatureDispatchTrace)
                       for item in self.traces)):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry result projection drifted")
        if (self.traces != tuple(sorted(
                self.traces, key=lambda item: item.entry_sha256))
                or len({item.entry_sha256 for item in self.traces})
                != len(self.traces)
                or any(item.exact_result.request != self.request
                       for item in self.traces)):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature result traces are not canonical")
        ordered_interpretations = tuple(sorted(
            {item.sha256(): item for item in self.interpretations}.values(),
            key=RawQuestionFeatureInterpretation.sha256,
        ))
        if ordered_interpretations != self.interpretations:
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature interpretations are not canonical")
        self._validate_depth_and_outcome()
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry boundary flags drifted")

    def _validate_depth_and_outcome(self) -> None:
        alias_depth = all(item.alias_result is not None for item in self.traces)
        implicit_depth = all(
            item.implicit_result is not None for item in self.traces)
        if (any(item.alias_result is not None for item in self.traces)
                != alias_depth
                or any(item.implicit_result is not None for item in self.traces)
                != implicit_depth
                or implicit_depth and not alias_depth):
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature registry phase depth is inconsistent")
        expected_depth = {
            "EXACT": (False, False),
            "ALIAS": (True, False),
            "IMPLICIT": (True, True),
            None: (True, True),
        }[self.decisive_phase]
        if (alias_depth, implicit_depth) != expected_depth:
            raise W03W04W05QuestionFeatureRegistryError(
                "question feature result does not stop at its decisive phase")
        phase = self.decisive_phase
        if phase is None:
            if (self.status != "UNKNOWN" or self.answer_surface is not None
                    or self.interpretations
                    or self.selected_entry_sha256 is not None
                    or any(item.status_at("IMPLICIT") != "UNKNOWN"
                           for item in self.traces)):
                raise W03W04W05QuestionFeatureRegistryError(
                    "missing registry result published a decision")
            return
        statuses = tuple(item.status_at(phase) for item in self.traces)
        if all(item == "UNKNOWN" for item in statuses):
            raise W03W04W05QuestionFeatureRegistryError(
                "decisive question feature phase has no decision")
        if self.status == "ANSWER":
            selected = tuple(
                item for item in self.traces
                if item.entry_sha256 == self.selected_entry_sha256)
            if (len(self.interpretations) != 1 or len(selected) != 1
                    or selected[0].status_at(phase) != "ANSWER"
                    or self.answer_surface != selected[0].answer_at(phase)
                    or raw_question_feature_trace_interpretation(
                        selected[0], phase)
                    != self.interpretations[0]):
                raise W03W04W05QuestionFeatureRegistryError(
                    "selected registry answer escaped its typed interpretation")
        elif (self.status != "CLARIFY" or self.answer_surface is not None
                or self.selected_entry_sha256 is not None
                or not ("CLARIFY" in statuses
                        or len(self.interpretations) > 1)):
            raise W03W04W05QuestionFeatureRegistryError(
                "ambiguous registry result selected an answer")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "decisive_phase": self.decisive_phase,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "interpretations": [
                item.to_dict() for item in self.interpretations],
            "registry_identity_sha256": self.registry_identity_sha256,
            "request": self.request.to_dict(),
            "selected_entry_sha256": self.selected_entry_sha256,
            "status": self.status,
            "traces": [item.to_dict() for item in self.traces],
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "QUESTION_FEATURE_DISPATCH_PHASES",
    "QUESTION_FEATURE_INTERPRETATION_STATUSES",
    "QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY",
    "QUESTION_FEATURE_REGISTRY_SHA256",
    "RawQuestionFeatureDispatchTrace",
    "RawQuestionFeatureInterpretation",
    "RawQuestionFeatureRegistry",
    "RawQuestionFeatureRegistryAnswerResult",
    "RawQuestionFeatureRegistryEntry",
    "W03W04W05QuestionFeatureRegistryError",
    "build_raw_question_feature_registry",
    "raw_question_feature_trace_interpretation",
    "resolve_question_feature_interpretations",
]
