"""GG-01 生成上下文与分层选择候选合同。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_HYPOTHESIS,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_EPISODE,
    SCOPE_QUERY,
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1

CHOICE_KINDS = (
    "CONTENT_CHOICE",
    "PROPOSITION_STRUCTURE_CHOICE",
    "DISCOURSE_REFERENCE_CHOICE",
    "LEXICAL_REALIZATION_CHOICE",
    "COMMUNICATIVE_TASK_CHOICE",
)
OBLIGATION_REQUIREMENTS = ("FORBIDDEN", "OPTIONAL", "REQUIRED")
USE_KINDS = ("CORE_USE", "MEMORY_USE")

GG01_CONTEXT_FIELDS = (
    "ADDRESSEE_CONTEXT",
    "COMMUNICATIVE_GOAL",
    "CONTENT_OBLIGATIONS",
    "DISCOURSE_STATE",
    "EXPRESSION_CONSTRAINTS",
    "GOAL_BINDING",
    "OWNER_SCOPE_VERSION",
)
GG01_HYPOTHESIS_FIELDS = (
    "AUTHORIZED_SCOPE",
    "CHOICE_KIND",
    "COMPETITION_KEY",
    "CONDITION",
    "EXACT_USES",
    "FORMING_EVIDENCE",
    "SELECTED_DECLARATIVE_OBJECT",
    "TARGET_OBLIGATION",
    "TYPED_OUTCOMES",
)
GG01_INVARIANTS_V1 = (
    "ANSWER_EPISODE_NEVER_BECOMES_TEMPLATE",
    "CANDIDATE_LIFECYCLE_REUSES_H04_H05",
    "CONDITION_AND_ACTION_REMAIN_DISTINCT",
    "CONTEXT_BINDS_EXACT_GOAL",
    "EVALUATOR_HELD_OUT_ZERO_HOST_WRITE",
    "OWNER_SCOPE_VERSION_FAIL_CLOSED",
    "TEACHER_CALL_ZERO",
)
GG01_INVARIANTS = tuple(sorted((
    *GG01_INVARIANTS_V1,
    "EXACT_KEYS_PRESERVE_ZERO_BEARING_CORE_MEMORY_IDENTITIES",
)))
GG01_VERIFIER_DIMENSIONS = (
    "CANDIDATE_PREFLIGHT_ZERO_WRITE",
    "CANONICAL_ROUND_TRIP",
    "CONDITION_ACTION_SEPARATION",
    "CONTEXT_FIELD_COMPLETENESS",
    "FIVE_LAYER_CHOICE_COVERAGE",
    "GOAL_CONTEXT_BINDING",
    "OWNER_SCOPE_VERSION_ISOLATION",
)
GG01_NE_CONDITIONS = (
    "GG02_USE_OUTCOME_BRIDGE_NOT_CONNECTED",
    "GG03_COMBINATION_COURSE_NOT_FROZEN",
    "RUNTIME_GENERALIZATION_REQUESTED",
    "W_TRAINING_NOT_EXECUTED",
)
EXECUTION_STATE_KEYS = (
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "formal_training_runs",
    "mastered_claims",
    "memory_learning_writes",
    "readiness_claims",
    "teacher_calls",
    "use_learning_writes",
    "w01_started",
)


class GenerationChoiceContractError(RuntimeError):
    """GG-01 合同不完整、有歧义或越过授权边界。"""


@dataclass(frozen=True, order=True)
class LosslessIntegerKey:
    """无摘要保存 Core、Memory、scope 和 verifier 的开放严格整数键。"""

    components: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or not self.components:
            raise GenerationChoiceContractError(
                "lossless integer key must be a non-empty tuple")
        if any(type(value) is not int for value in self.components):
            raise GenerationChoiceContractError(
                "lossless integer key must use strict integers")
        assert_int(*self.components, _where="LosslessIntegerKey.components")

    def to_list(self) -> list[int]:
        return list(self.components)

    def stable_key(self) -> tuple[int, ...]:
        return self.components

    @classmethod
    def from_value(cls, value: Any, *, where: str) -> "LosslessIntegerKey":
        if not isinstance(value, list) or not value:
            raise GenerationChoiceContractError(
                f"{where} must be a non-empty integer list")
        try:
            return cls(tuple(value))
        except (TypeError, ValueError, GenerationChoiceContractError) as error:
            raise GenerationChoiceContractError(f"{where} is invalid") from error


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise GenerationChoiceContractError(f"{where} fields are not exact")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationChoiceContractError(f"{where} must be non-empty text")
    return value


def _text_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise GenerationChoiceContractError(f"{where} must be a text tuple")
    if value != tuple(sorted(set(value))):
        raise GenerationChoiceContractError(f"{where} must be sorted and unique")
    return value


def _strict_int(value: Any, *, where: str, minimum: int = 0) -> int:
    assert_int(value, _where=where)
    if type(value) is not int or value < minimum:
        raise GenerationChoiceContractError(f"{where} is invalid")
    return value


def _binary(value: Any, *, where: str) -> int:
    value = _strict_int(value, where=where)
    if value not in (0, 1):
        raise GenerationChoiceContractError(f"{where} must be 0 or 1")
    return value


def _int_key(value: Any, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise GenerationChoiceContractError(f"{where} must be a non-empty tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise GenerationChoiceContractError(f"{where} must use strict integers")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _object(value: Any, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise GenerationChoiceContractError(f"{where} must be ObjectIdentity")
    return value


def _objects(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(value, tuple) or (not value and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in value)):
        raise GenerationChoiceContractError(f"{where} has invalid objects")
    normalized = tuple(sorted(value, key=ObjectIdentity.stable_key))
    if len(set(normalized)) != len(normalized):
        raise GenerationChoiceContractError(f"{where} has duplicate objects")
    return normalized


def _sources(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> tuple[SourceRef, ...]:
    if (not isinstance(value, tuple) or (not value and not allow_empty)
            or any(not isinstance(item, SourceRef) for item in value)):
        raise GenerationChoiceContractError(f"{where} has invalid sources")
    normalized = tuple(sorted(value, key=SourceRef.stable_key))
    if len(set(normalized)) != len(normalized):
        raise GenerationChoiceContractError(f"{where} has duplicate sources")
    return normalized


def _same_boundary(scope: ScopeIdentity, value: ObjectIdentity | SourceRef,
                   *, where: str) -> None:
    if value.owner != scope.owner or value.versions != scope.versions:
        raise GenerationChoiceContractError(
            f"{where} crosses owner or version boundary")


def _object_value(value: ObjectIdentity) -> list[int]:
    return list(value.stable_key())


def _source_value(value: SourceRef) -> list[int]:
    return list(value.stable_key())


def _scope_value(value: ScopeIdentity) -> list[int]:
    return list(value.stable_key())


def _object_from(value: Any, *, where: str) -> ObjectIdentity:
    if not isinstance(value, list):
        raise GenerationChoiceContractError(f"{where} must be an integer list")
    try:
        return ObjectIdentity.from_stable_key(tuple(value))
    except Exception as error:
        raise GenerationChoiceContractError(f"{where} is invalid") from error


def _source_from(value: Any, *, where: str) -> SourceRef:
    if not isinstance(value, list):
        raise GenerationChoiceContractError(f"{where} must be an integer list")
    try:
        return SourceRef.from_stable_key(tuple(value))
    except Exception as error:
        raise GenerationChoiceContractError(f"{where} is invalid") from error


def _scope_from(value: Any, *, where: str) -> ScopeIdentity:
    if not isinstance(value, list):
        raise GenerationChoiceContractError(f"{where} must be an integer list")
    try:
        return ScopeIdentity.from_stable_key(tuple(value))
    except Exception as error:
        raise GenerationChoiceContractError(f"{where} is invalid") from error


@dataclass(frozen=True)
class GenerationAddresseeContext:
    """当前 episode/query 的受众与可恢复指称边界。"""

    addressee: ObjectIdentity
    shared_visible_objects: tuple[ObjectIdentity, ...]
    shared_visible_events: tuple[ObjectIdentity, ...]
    recoverable_references: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _object(self.addressee, where="addressee")
        for name in (
                "shared_visible_objects", "shared_visible_events",
                "recoverable_references"):
            object.__setattr__(self, name, _objects(
                getattr(self, name), where=name, allow_empty=True))
        visible = set(self.shared_visible_objects) | set(self.shared_visible_events)
        if not set(self.recoverable_references).issubset(visible):
            raise GenerationChoiceContractError(
                "recoverable references must be shared-visible")

    def stable_key(self) -> tuple[int, ...]:
        values = [1, *_pack(self.addressee.stable_key())]
        for items in (
                self.shared_visible_objects, self.shared_visible_events,
                self.recoverable_references):
            values.append(len(items))
            for item in items:
                values.extend(_pack(item.stable_key()))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "addressee": _object_value(self.addressee),
            "recoverable_references": [
                _object_value(item) for item in self.recoverable_references],
            "shared_visible_events": [
                _object_value(item) for item in self.shared_visible_events],
            "shared_visible_objects": [
                _object_value(item) for item in self.shared_visible_objects],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationAddresseeContext":
        raw = _exact(value, {
            "addressee", "recoverable_references", "shared_visible_events",
            "shared_visible_objects",
        }, where="GenerationAddresseeContext")
        return cls(
            _object_from(raw["addressee"], where="addressee"),
            tuple(_object_from(item, where="shared visible object")
                  for item in raw["shared_visible_objects"]),
            tuple(_object_from(item, where="shared visible event")
                  for item in raw["shared_visible_events"]),
            tuple(_object_from(item, where="recoverable reference")
                  for item in raw["recoverable_references"]),
        )


@dataclass(frozen=True)
class GenerationContentObligation:
    """携带来源和不确定性的必需、可选或禁止命题。"""

    obligation: ObjectIdentity
    requirement: str
    proposition: ObjectIdentity
    source_constraints: tuple[SourceRef, ...]
    uncertainty: ObjectIdentity | None

    def __post_init__(self) -> None:
        _object(self.obligation, where="content obligation")
        if self.requirement not in OBLIGATION_REQUIREMENTS:
            raise GenerationChoiceContractError("obligation requirement is invalid")
        _object(self.proposition, where="content proposition")
        if self.proposition.object_kind != OBJECT_PROPOSITION:
            raise GenerationChoiceContractError(
                "content obligation must reference a Proposition")
        object.__setattr__(self, "source_constraints", _sources(
            self.source_constraints, where="source constraints", allow_empty=True))
        if self.uncertainty is not None:
            _object(self.uncertainty, where="uncertainty")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1, OBLIGATION_REQUIREMENTS.index(self.requirement),
            *_pack(self.obligation.stable_key()),
            *_pack(self.proposition.stable_key()),
            len(self.source_constraints),
        ]
        for source in self.source_constraints:
            values.extend(_pack(source.stable_key()))
        uncertainty = () if self.uncertainty is None else self.uncertainty.stable_key()
        values.extend(_pack(uncertainty))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation": _object_value(self.obligation),
            "proposition": _object_value(self.proposition),
            "requirement": self.requirement,
            "source_constraints": [
                _source_value(item) for item in self.source_constraints],
            "uncertainty": (
                None if self.uncertainty is None
                else _object_value(self.uncertainty)),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationContentObligation":
        raw = _exact(value, {
            "obligation", "proposition", "requirement", "source_constraints",
            "uncertainty",
        }, where="GenerationContentObligation")
        return cls(
            _object_from(raw["obligation"], where="obligation"),
            str(raw["requirement"]),
            _object_from(raw["proposition"], where="proposition"),
            tuple(_source_from(item, where="source constraint")
                  for item in raw["source_constraints"]),
            None if raw["uncertainty"] is None else _object_from(
                raw["uncertainty"], where="uncertainty"),
        )


@dataclass(frozen=True)
class GenerationExpressionConstraints:
    """目标语言、声明性分支、显式度与输出预算。"""

    target_language: ObjectIdentity
    allowed_structure_families: tuple[ObjectIdentity, ...]
    allowed_lexical_branches: tuple[ObjectIdentity, ...]
    require_explicit_source: int
    allow_ellipsis: int
    allow_pronoun: int
    max_output_units: int

    def __post_init__(self) -> None:
        _object(self.target_language, where="target language")
        if self.target_language.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise GenerationChoiceContractError(
                "target language must be LanguageBranch")
        for name in ("allowed_structure_families", "allowed_lexical_branches"):
            object.__setattr__(self, name, _objects(
                getattr(self, name), where=name, allow_empty=True))
        _binary(self.require_explicit_source, where="require explicit source")
        _binary(self.allow_ellipsis, where="allow ellipsis")
        _binary(self.allow_pronoun, where="allow pronoun")
        _strict_int(self.max_output_units, where="max output units", minimum=1)

    def stable_key(self) -> tuple[int, ...]:
        values = [1, *_pack(self.target_language.stable_key())]
        for items in (
                self.allowed_structure_families,
                self.allowed_lexical_branches):
            values.append(len(items))
            for item in items:
                values.extend(_pack(item.stable_key()))
        values.extend((
            self.require_explicit_source, self.allow_ellipsis,
            self.allow_pronoun, self.max_output_units,
        ))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_ellipsis": self.allow_ellipsis,
            "allow_pronoun": self.allow_pronoun,
            "allowed_lexical_branches": [
                _object_value(item) for item in self.allowed_lexical_branches],
            "allowed_structure_families": [
                _object_value(item) for item in self.allowed_structure_families],
            "max_output_units": self.max_output_units,
            "require_explicit_source": self.require_explicit_source,
            "target_language": _object_value(self.target_language),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationExpressionConstraints":
        raw = _exact(value, {
            "allow_ellipsis", "allow_pronoun", "allowed_lexical_branches",
            "allowed_structure_families", "max_output_units",
            "require_explicit_source", "target_language",
        }, where="GenerationExpressionConstraints")
        return cls(
            _object_from(raw["target_language"], where="target language"),
            tuple(_object_from(item, where="structure family")
                  for item in raw["allowed_structure_families"]),
            tuple(_object_from(item, where="lexical branch")
                  for item in raw["allowed_lexical_branches"]),
            raw["require_explicit_source"], raw["allow_ellipsis"],
            raw["allow_pronoun"], raw["max_output_units"],
        )


@dataclass(frozen=True)
class GenerationDiscourseState:
    """开放问题、当前话题、既有表达与修正依赖。"""

    open_questions: tuple[ObjectIdentity, ...]
    current_topic: ObjectIdentity | None
    prior_expressed: tuple[ObjectIdentity, ...]
    revision_dependencies: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        for name in ("open_questions", "prior_expressed", "revision_dependencies"):
            object.__setattr__(self, name, _objects(
                getattr(self, name), where=name, allow_empty=True))
        if self.current_topic is not None:
            _object(self.current_topic, where="current topic")

    def stable_key(self) -> tuple[int, ...]:
        topic = () if self.current_topic is None else self.current_topic.stable_key()
        values = [1, *_pack(topic)]
        for items in (
                self.open_questions, self.prior_expressed,
                self.revision_dependencies):
            values.append(len(items))
            for item in items:
                values.extend(_pack(item.stable_key()))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_topic": (
                None if self.current_topic is None
                else _object_value(self.current_topic)),
            "open_questions": [
                _object_value(item) for item in self.open_questions],
            "prior_expressed": [
                _object_value(item) for item in self.prior_expressed],
            "revision_dependencies": [
                _object_value(item) for item in self.revision_dependencies],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationDiscourseState":
        raw = _exact(value, {
            "current_topic", "open_questions", "prior_expressed",
            "revision_dependencies",
        }, where="GenerationDiscourseState")
        return cls(
            tuple(_object_from(item, where="open question")
                  for item in raw["open_questions"]),
            None if raw["current_topic"] is None else _object_from(
                raw["current_topic"], where="current topic"),
            tuple(_object_from(item, where="prior expression")
                  for item in raw["prior_expressed"]),
            tuple(_object_from(item, where="revision dependency")
                  for item in raw["revision_dependencies"]),
        )


@dataclass(frozen=True)
class GenerationContextContract:
    """绑定精确目标且不建立长期用户画像的当前 query 生成上下文。"""

    context: ObjectIdentity
    goal: ObjectIdentity
    communicative_goal: ObjectIdentity
    scope: ScopeIdentity
    addressee: GenerationAddresseeContext
    content_obligations: tuple[GenerationContentObligation, ...]
    expression_constraints: GenerationExpressionConstraints
    discourse_state: GenerationDiscourseState

    def __post_init__(self) -> None:
        for name in ("context", "goal", "communicative_goal"):
            _object(getattr(self, name), where=name)
        if not isinstance(self.scope, ScopeIdentity):
            raise GenerationChoiceContractError("context scope is invalid")
        if self.scope.scope_kind not in {SCOPE_EPISODE, SCOPE_QUERY}:
            raise GenerationChoiceContractError(
                "generation context must be query or episode scoped")
        if not isinstance(self.addressee, GenerationAddresseeContext):
            raise GenerationChoiceContractError("addressee context is invalid")
        if (not isinstance(self.content_obligations, tuple)
                or not self.content_obligations
                or any(not isinstance(item, GenerationContentObligation)
                       for item in self.content_obligations)):
            raise GenerationChoiceContractError("content obligations are invalid")
        obligations = tuple(sorted(
            self.content_obligations,
            key=lambda item: item.obligation.stable_key()))
        if len({item.obligation for item in obligations}) != len(obligations):
            raise GenerationChoiceContractError("content obligation identity repeats")
        proposition_requirements: dict[ObjectIdentity, str] = {}
        for item in obligations:
            prior = proposition_requirements.setdefault(
                item.proposition, item.requirement)
            if prior != item.requirement:
                raise GenerationChoiceContractError(
                    "one proposition has competing obligation requirements")
        if not any(item.requirement == "REQUIRED" for item in obligations):
            raise GenerationChoiceContractError(
                "generation context needs a required content obligation")
        object.__setattr__(self, "content_obligations", obligations)
        if not isinstance(
                self.expression_constraints, GenerationExpressionConstraints):
            raise GenerationChoiceContractError("expression constraints are invalid")
        if not isinstance(self.discourse_state, GenerationDiscourseState):
            raise GenerationChoiceContractError("discourse state is invalid")
        boundary_values: list[ObjectIdentity | SourceRef] = [
            self.context, self.goal, self.communicative_goal,
            self.addressee.addressee,
            *self.addressee.shared_visible_objects,
            *self.addressee.shared_visible_events,
            *self.addressee.recoverable_references,
            self.expression_constraints.target_language,
            *self.expression_constraints.allowed_structure_families,
            *self.expression_constraints.allowed_lexical_branches,
            *self.discourse_state.open_questions,
            *self.discourse_state.prior_expressed,
            *self.discourse_state.revision_dependencies,
        ]
        if self.discourse_state.current_topic is not None:
            boundary_values.append(self.discourse_state.current_topic)
        for obligation in obligations:
            boundary_values.extend((obligation.obligation, obligation.proposition))
            boundary_values.extend(obligation.source_constraints)
            if obligation.uncertainty is not None:
                boundary_values.append(obligation.uncertainty)
        for value in boundary_values:
            _same_boundary(self.scope, value, where="generation context value")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            *_pack(self.context.stable_key()),
            *_pack(self.goal.stable_key()),
            *_pack(self.communicative_goal.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.addressee.stable_key()),
            len(self.content_obligations),
        ]
        for item in self.content_obligations:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.expression_constraints.stable_key()))
        values.extend(_pack(self.discourse_state.stable_key()))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "addressee": self.addressee.to_dict(),
            "communicative_goal": _object_value(self.communicative_goal),
            "content_obligations": [
                item.to_dict() for item in self.content_obligations],
            "context": _object_value(self.context),
            "discourse_state": self.discourse_state.to_dict(),
            "expression_constraints": self.expression_constraints.to_dict(),
            "goal": _object_value(self.goal),
            "scope": _scope_value(self.scope),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationContextContract":
        raw = _exact(value, {
            "addressee", "communicative_goal", "content_obligations", "context",
            "discourse_state", "expression_constraints", "goal", "scope",
        }, where="GenerationContextContract")
        return cls(
            _object_from(raw["context"], where="context"),
            _object_from(raw["goal"], where="goal"),
            _object_from(raw["communicative_goal"], where="communicative goal"),
            _scope_from(raw["scope"], where="context scope"),
            GenerationAddresseeContext.from_dict(raw["addressee"]),
            tuple(GenerationContentObligation.from_dict(item)
                  for item in raw["content_obligations"]),
            GenerationExpressionConstraints.from_dict(
                raw["expression_constraints"]),
            GenerationDiscourseState.from_dict(raw["discourse_state"]),
        )


@dataclass(frozen=True)
class GenerationChoiceCondition:
    """与所选动作分离的可复用上下文条件。"""

    condition: ObjectIdentity
    context: ObjectIdentity
    required_context_objects: tuple[ObjectIdentity, ...]
    forbidden_context_objects: tuple[ObjectIdentity, ...]
    authorized_scope: ScopeIdentity

    def __post_init__(self) -> None:
        _object(self.condition, where="choice condition")
        _object(self.context, where="choice context")
        if not isinstance(self.authorized_scope, ScopeIdentity):
            raise GenerationChoiceContractError("condition scope is invalid")
        for name in ("required_context_objects", "forbidden_context_objects"):
            object.__setattr__(self, name, _objects(
                getattr(self, name), where=name, allow_empty=True))
        if set(self.required_context_objects) & set(self.forbidden_context_objects):
            raise GenerationChoiceContractError(
                "condition cannot require and forbid the same object")
        for item in (
                self.condition, self.context, *self.required_context_objects,
                *self.forbidden_context_objects):
            _same_boundary(self.authorized_scope, item, where="choice condition")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1, *_pack(self.condition.stable_key()),
            *_pack(self.context.stable_key()),
            *_pack(self.authorized_scope.stable_key()),
        ]
        for items in (
                self.required_context_objects, self.forbidden_context_objects):
            values.append(len(items))
            for item in items:
                values.extend(_pack(item.stable_key()))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_scope": _scope_value(self.authorized_scope),
            "condition": _object_value(self.condition),
            "context": _object_value(self.context),
            "forbidden_context_objects": [
                _object_value(item) for item in self.forbidden_context_objects],
            "required_context_objects": [
                _object_value(item) for item in self.required_context_objects],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationChoiceCondition":
        raw = _exact(value, {
            "authorized_scope", "condition", "context",
            "forbidden_context_objects", "required_context_objects",
        }, where="GenerationChoiceCondition")
        return cls(
            _object_from(raw["condition"], where="condition"),
            _object_from(raw["context"], where="condition context"),
            tuple(_object_from(item, where="required context object")
                  for item in raw["required_context_objects"]),
            tuple(_object_from(item, where="forbidden context object")
                  for item in raw["forbidden_context_objects"]),
            _scope_from(raw["authorized_scope"], where="authorized scope"),
        )


@dataclass(frozen=True, order=True)
class GenerationChoiceUseRef:
    """一次有界生成 episode 中实际选择的精确 Core 或 Memory Use。"""

    use_kind: str
    use_key: LosslessIntegerKey
    selection_key: LosslessIntegerKey
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if self.use_kind not in USE_KINDS:
            raise GenerationChoiceContractError("choice use kind is invalid")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise GenerationChoiceContractError("choice use key is invalid")
        if not isinstance(self.selection_key, LosslessIntegerKey):
            raise GenerationChoiceContractError("choice selection key is invalid")
        if not isinstance(self.scope, ScopeIdentity):
            raise GenerationChoiceContractError("choice use scope is invalid")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1, USE_KINDS.index(self.use_kind),
            *_pack(self.use_key.components),
            *_pack(self.selection_key.components),
            *_pack(self.scope.stable_key()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": _scope_value(self.scope),
            "selection_key": self.selection_key.to_list(),
            "use_key": self.use_key.to_list(),
            "use_kind": self.use_kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationChoiceUseRef":
        raw = _exact(value, {
            "scope", "selection_key", "use_key", "use_kind",
        }, where="GenerationChoiceUseRef")
        return cls(
            str(raw["use_kind"]),
            LosslessIntegerKey.from_value(raw["use_key"], where="use key"),
            LosslessIntegerKey.from_value(
                raw["selection_key"], where="selection key"),
            _scope_from(raw["scope"], where="use scope"),
        )


@dataclass(frozen=True, order=True)
class GenerationChoiceOutcomeRef:
    """由 verifier 持有且只归因到一个精确 Use 的分型结果。"""

    outcome_key: LosslessIntegerKey
    use_key: LosslessIntegerKey
    dimension_key: LosslessIntegerKey
    verifier_key: LosslessIntegerKey
    result_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        for name in (
                "outcome_key", "use_key", "dimension_key", "verifier_key",
                "result_key"):
            if not isinstance(getattr(self, name), LosslessIntegerKey):
                raise GenerationChoiceContractError(
                    f"choice outcome {name} is invalid")

    def stable_key(self) -> tuple[int, ...]:
        values = [1]
        for item in (
                self.outcome_key, self.use_key, self.dimension_key,
                self.verifier_key, self.result_key):
            values.extend(_pack(item.components))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_key": self.dimension_key.to_list(),
            "outcome_key": self.outcome_key.to_list(),
            "result_key": self.result_key.to_list(),
            "use_key": self.use_key.to_list(),
            "verifier_key": self.verifier_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationChoiceOutcomeRef":
        raw = _exact(value, {
            "dimension_key", "outcome_key", "result_key", "use_key",
            "verifier_key",
        }, where="GenerationChoiceOutcomeRef")
        return cls(*(LosslessIntegerKey.from_value(raw[name], where=name)
                     for name in (
                         "outcome_key", "use_key", "dimension_key",
                         "verifier_key", "result_key")))


@dataclass(frozen=True)
class GenerationChoiceHypothesis:
    """分层有条件选择候选，不是完整回答模板。"""

    candidate: ObjectIdentity
    choice_kind: str
    target_obligation: ObjectIdentity
    condition: GenerationChoiceCondition
    selected_object: ObjectIdentity
    forming_sources: tuple[SourceRef, ...]
    competition_key: tuple[int, ...]
    authorized_scope: ScopeIdentity
    exact_uses: tuple[GenerationChoiceUseRef, ...] = ()
    typed_outcomes: tuple[GenerationChoiceOutcomeRef, ...] = ()

    def __post_init__(self) -> None:
        _object(self.candidate, where="choice candidate")
        if self.candidate.object_kind != OBJECT_HYPOTHESIS:
            raise GenerationChoiceContractError(
                "choice candidate must use hypothesis object identity")
        if self.choice_kind not in CHOICE_KINDS:
            raise GenerationChoiceContractError("choice kind is invalid")
        _object(self.target_obligation, where="choice target obligation")
        if not isinstance(self.condition, GenerationChoiceCondition):
            raise GenerationChoiceContractError("choice condition is invalid")
        _object(self.selected_object, where="selected declarative object")
        object.__setattr__(self, "forming_sources", _sources(
            self.forming_sources, where="choice forming sources"))
        _int_key(self.competition_key, where="choice competition key")
        if not isinstance(self.authorized_scope, ScopeIdentity):
            raise GenerationChoiceContractError("choice scope is invalid")
        if (self.condition.authorized_scope != self.authorized_scope):
            raise GenerationChoiceContractError(
                "condition and choice authorization scopes differ")
        for item in (
                self.candidate, self.target_obligation,
                self.condition.context, self.condition.condition,
                self.selected_object, *self.forming_sources):
            _same_boundary(self.authorized_scope, item, where="choice value")
        if (not isinstance(self.exact_uses, tuple)
                or any(not isinstance(item, GenerationChoiceUseRef)
                       for item in self.exact_uses)):
            raise GenerationChoiceContractError("exact uses are invalid")
        uses = tuple(sorted(self.exact_uses, key=GenerationChoiceUseRef.stable_key))
        if len({item.use_key for item in uses}) != len(uses):
            raise GenerationChoiceContractError("exact use identity repeats")
        if any(item.scope != self.authorized_scope for item in uses):
            raise GenerationChoiceContractError(
                "exact use crosses choice authorization scope")
        object.__setattr__(self, "exact_uses", uses)
        if (not isinstance(self.typed_outcomes, tuple)
                or any(not isinstance(item, GenerationChoiceOutcomeRef)
                       for item in self.typed_outcomes)):
            raise GenerationChoiceContractError("typed outcomes are invalid")
        outcomes = tuple(sorted(
            self.typed_outcomes, key=GenerationChoiceOutcomeRef.stable_key))
        if len({item.outcome_key for item in outcomes}) != len(outcomes):
            raise GenerationChoiceContractError("outcome identity repeats")
        use_keys = {item.use_key for item in uses}
        if any(item.use_key not in use_keys for item in outcomes):
            raise GenerationChoiceContractError(
                "typed outcome does not reference an exact Use")
        object.__setattr__(self, "typed_outcomes", outcomes)

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1, CHOICE_KINDS.index(self.choice_kind),
            *_pack(self.candidate.stable_key()),
            *_pack(self.target_obligation.stable_key()),
            *_pack(self.condition.stable_key()),
            *_pack(self.selected_object.stable_key()),
            *_pack(self.competition_key),
            *_pack(self.authorized_scope.stable_key()),
            len(self.forming_sources),
        ]
        for source in self.forming_sources:
            values.extend(_pack(source.stable_key()))
        values.append(len(self.exact_uses))
        for item in self.exact_uses:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.typed_outcomes))
        for item in self.typed_outcomes:
            values.extend(_pack(item.stable_key()))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_scope": _scope_value(self.authorized_scope),
            "candidate": _object_value(self.candidate),
            "choice_kind": self.choice_kind,
            "competition_key": list(self.competition_key),
            "condition": self.condition.to_dict(),
            "exact_uses": [item.to_dict() for item in self.exact_uses],
            "forming_sources": [
                _source_value(item) for item in self.forming_sources],
            "selected_object": _object_value(self.selected_object),
            "target_obligation": _object_value(self.target_obligation),
            "typed_outcomes": [item.to_dict() for item in self.typed_outcomes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationChoiceHypothesis":
        raw = _exact(value, {
            "authorized_scope", "candidate", "choice_kind", "competition_key",
            "condition", "exact_uses", "forming_sources", "selected_object",
            "target_obligation", "typed_outcomes",
        }, where="GenerationChoiceHypothesis")
        return cls(
            _object_from(raw["candidate"], where="candidate"),
            str(raw["choice_kind"]),
            _object_from(raw["target_obligation"], where="target obligation"),
            GenerationChoiceCondition.from_dict(raw["condition"]),
            _object_from(raw["selected_object"], where="selected object"),
            tuple(_source_from(item, where="forming source")
                  for item in raw["forming_sources"]),
            tuple(raw["competition_key"]),
            _scope_from(raw["authorized_scope"], where="authorized scope"),
            tuple(GenerationChoiceUseRef.from_dict(item)
                  for item in raw["exact_uses"]),
            tuple(GenerationChoiceOutcomeRef.from_dict(item)
                  for item in raw["typed_outcomes"]),
        )


@dataclass(frozen=True)
class GenerationChoiceCandidateProtocol:
    """把 GG-01 选择装入通用候选引擎的分型绑定协议。"""

    candidate_kind_predicate: ObjectIdentity
    choice_kind_predicate: ObjectIdentity
    target_predicate: ObjectIdentity
    context_predicate: ObjectIdentity
    condition_predicate: ObjectIdentity
    selected_predicate: ObjectIdentity
    candidate_kind: ObjectIdentity
    choice_kind_objects: tuple[ObjectIdentity, ...]
    competition_namespace: tuple[int, ...]

    def __post_init__(self) -> None:
        predicates = (
            self.candidate_kind_predicate, self.choice_kind_predicate,
            self.target_predicate, self.context_predicate,
            self.condition_predicate, self.selected_predicate,
        )
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_CONCEPT for item in predicates):
            raise GenerationChoiceContractError(
                "choice candidate predicates must be Concepts")
        if len(set(predicates)) != len(predicates):
            raise GenerationChoiceContractError(
                "choice candidate predicates must be distinct")
        _object(self.candidate_kind, where="choice candidate kind")
        if (not isinstance(self.choice_kind_objects, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.choice_kind_objects)
                or len(set(self.choice_kind_objects)) != (
                    len(self.choice_kind_objects))):
            raise GenerationChoiceContractError(
                "choice kind objects must be typed and unique")
        if len(self.choice_kind_objects) != len(CHOICE_KINDS):
            raise GenerationChoiceContractError(
                "choice kind objects do not cover five layers")
        _int_key(self.competition_namespace, where="competition namespace")

    def kind_object(self, choice_kind: str) -> ObjectIdentity:
        if choice_kind not in CHOICE_KINDS:
            raise GenerationChoiceContractError("choice kind is invalid")
        return self.choice_kind_objects[CHOICE_KINDS.index(choice_kind)]


class GenerationChoiceCandidateMapper:
    """把分层选择映射到 H-00/H-04/H-05，不建立第二套生命周期。"""

    def __init__(self, protocol: GenerationChoiceCandidateProtocol) -> None:
        if not isinstance(protocol, GenerationChoiceCandidateProtocol):
            raise TypeError("choice candidate protocol is invalid")
        self.protocol = protocol

    def competition_key(
            self, choice: GenerationChoiceHypothesis,
            ) -> tuple[int, ...]:
        """返回 H-05 使用的分层竞争组全键。"""
        if not isinstance(choice, GenerationChoiceHypothesis):
            raise TypeError("choice hypothesis is invalid")
        return (
            *self.protocol.competition_namespace,
            CHOICE_KINDS.index(choice.choice_kind),
            *_pack(choice.competition_key),
            *_pack(choice.condition.context.stable_key()),
        )

    def candidate_identity(
            self, choice: GenerationChoiceHypothesis,
            ) -> ObjectIdentity:
        """把 GG-01 choice 身份无损投影为图可物化的 Hypothesis 身份。"""
        if not isinstance(choice, GenerationChoiceHypothesis):
            raise TypeError("choice hypothesis is invalid")
        observation = choice.forming_sources[0]
        return HypothesisKey(
            self.protocol.candidate_kind.stable_key(),
            choice.candidate.stable_key(),
            self.competition_key(choice),
            document_scope(observation),
            observation,
        ).object_identity()

    def context_identity(
            self, choice: GenerationChoiceHypothesis,
            ) -> ObjectIdentity:
        """把 GG-01 抽象 context 无损投影为来源化图 ContextScope。"""
        if not isinstance(choice, GenerationChoiceHypothesis):
            raise TypeError("choice hypothesis is invalid")
        return context_scope_identity(
            choice.forming_sources[0], choice.condition.context.stable_key())

    def definition(
            self, choice: GenerationChoiceHypothesis,
            ) -> EvidenceCandidateDefinition:
        if not isinstance(choice, GenerationChoiceHypothesis):
            raise TypeError("choice hypothesis is invalid")
        bindings = (
            CandidateBinding(
                self.protocol.candidate_kind_predicate,
                self.protocol.candidate_kind),
            CandidateBinding(
                self.protocol.choice_kind_predicate,
                self.protocol.kind_object(choice.choice_kind)),
            CandidateBinding(
                self.protocol.target_predicate, choice.target_obligation),
            CandidateBinding(
                self.protocol.context_predicate, self.context_identity(choice)),
            CandidateBinding(
                self.protocol.condition_predicate, choice.condition.condition),
            CandidateBinding(
                self.protocol.selected_predicate, choice.selected_object),
        )
        return EvidenceCandidateDefinition(
            self.candidate_identity(choice),
            self.competition_key(choice),
            bindings,
            choice.forming_sources,
        )

    def preflight_load(
            self,
            choice: GenerationChoiceHypothesis,
            learning: CandidateLearningRuntime,
            *,
            timestamp_base: int = 0,
            ) -> HypothesisKey:
        """以零候选写和零图写证明通用候选装载能力。"""
        if not isinstance(learning, CandidateLearningRuntime):
            raise TypeError("candidate learning runtime is invalid")
        return learning.preflight_register(
            self.definition(choice), timestamp_base=timestamp_base)


@dataclass(frozen=True)
class GG01ContractManifest:
    """不可覆盖的 GG-01 合同冻结与 LC-13 初始方向登记。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_keys: tuple[str, ...]
    prerequisite_manifest_relative_path: str
    prerequisite_manifest_sha256: str
    choice_kind_keys: tuple[str, ...]
    context_field_keys: tuple[str, ...]
    hypothesis_field_keys: tuple[str, ...]
    invariant_keys: tuple[str, ...]
    reused_component_refs: tuple[str, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    lc13_directional_consumption: CanonicalJsonObject
    runtime_status: str
    results_observed: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise GenerationChoiceContractError("GG-01 format version is invalid")
        _text(self.artifact_version, where="GG-01 artifact version")
        if self.artifact_status != "CONTRACT_FROZEN":
            raise GenerationChoiceContractError("GG-01 artifact status is invalid")
        if self.task_keys != ("GG-01", "LC-13"):
            raise GenerationChoiceContractError("GG-01 task keys are invalid")
        _relative_path(
            self.prerequisite_manifest_relative_path,
            where="GG-01 prerequisite path")
        _sha256(self.prerequisite_manifest_sha256, where="GG-01 prerequisite hash")
        if self.artifact_version == "GG-01-generation-context-choice-contract-v1":
            expected_invariants = GG01_INVARIANTS_V1
        elif self.artifact_version == (
                "GG-01-generation-context-choice-contract-v2-supersedes-v1"):
            expected_invariants = GG01_INVARIANTS
        else:
            raise GenerationChoiceContractError(
                "GG-01 artifact version is unsupported")
        for actual, expected, label in (
                (self.choice_kind_keys, CHOICE_KINDS, "choice kinds"),
                (self.context_field_keys, GG01_CONTEXT_FIELDS, "context fields"),
                (self.hypothesis_field_keys, GG01_HYPOTHESIS_FIELDS,
                 "hypothesis fields"),
                (self.invariant_keys, expected_invariants, "invariants"),
                (self.verifier_dimensions, GG01_VERIFIER_DIMENSIONS,
                 "verifier dimensions"),
                (self.verifier_ne_conditions, GG01_NE_CONDITIONS,
                 "verifier NE conditions")):
            if actual != expected:
                raise GenerationChoiceContractError(
                    f"GG-01 {label} are incomplete")
        _text_tuple(self.reused_component_refs, where="GG-01 reused refs")
        directions = self.lc13_directional_consumption.to_value()
        if tuple(directions) != ("GENERATION", "REASONING", "UNDERSTANDING"):
            raise GenerationChoiceContractError("GG-01 LC-13 directions are incomplete")
        expected_direction_fields = (
            "applicability", "consumer_refs", "fact_state", "write_permissions")
        for value in directions.values():
            if not isinstance(value, dict) or tuple(value) != expected_direction_fields:
                raise GenerationChoiceContractError(
                    "GG-01 LC-13 direction fields are incomplete")
        if directions["GENERATION"]["fact_state"] != "CONTRACT_FROZEN":
            raise GenerationChoiceContractError(
                "GG-01 generation direction is not contract-frozen")
        if directions["REASONING"]["fact_state"] != "ABSENT":
            raise GenerationChoiceContractError(
                "GG-01 must keep the reasoning consumer absent")
        if "NO_HOST_LEARNING_WRITE" not in (
                directions["GENERATION"]["write_permissions"]):
            raise GenerationChoiceContractError(
                "GG-01 generation writes are not bounded")
        if self.runtime_status != "NOT_CONNECTED":
            raise GenerationChoiceContractError(
                "GG-01 must not claim runtime connection")
        _strict_int(self.results_observed, where="GG-01 results observed")
        if self.results_observed != 0:
            raise GenerationChoiceContractError(
                "GG-01 must not claim observed runtime results")
        state = self.execution_state.to_value()
        if tuple(state) != EXECUTION_STATE_KEYS or any(state.values()):
            raise GenerationChoiceContractError(
                "GG-01 execution state must be exact and zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_GG01_GENERATION_CHOICE_CONTRACT",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "choice_kind_keys": list(self.choice_kind_keys),
            "context_field_keys": list(self.context_field_keys),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "hypothesis_field_keys": list(self.hypothesis_field_keys),
            "invariant_keys": list(self.invariant_keys),
            "lc13_directional_consumption": (
                self.lc13_directional_consumption.to_value()),
            "prerequisite_manifest_relative_path": (
                self.prerequisite_manifest_relative_path),
            "prerequisite_manifest_sha256": self.prerequisite_manifest_sha256,
            "results_observed": self.results_observed,
            "reused_component_refs": list(self.reused_component_refs),
            "runtime_status": self.runtime_status,
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GG01ContractManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "choice_kind_keys", "context_field_keys", "execution_state",
            "format_version", "hypothesis_field_keys", "invariant_keys",
            "lc13_directional_consumption",
            "prerequisite_manifest_relative_path",
            "prerequisite_manifest_sha256", "results_observed",
            "reused_component_refs", "runtime_status", "task_keys",
            "verifier_dimensions", "verifier_ne_conditions",
        }, where="GG01ContractManifest")
        if raw["artifact_kind"] != "PH2_GG01_GENERATION_CHOICE_CONTRACT":
            raise GenerationChoiceContractError("GG-01 artifact kind is invalid")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            str(raw["prerequisite_manifest_relative_path"]),
            str(raw["prerequisite_manifest_sha256"]),
            tuple(str(item) for item in raw["choice_kind_keys"]),
            tuple(str(item) for item in raw["context_field_keys"]),
            tuple(str(item) for item in raw["hypothesis_field_keys"]),
            tuple(str(item) for item in raw["invariant_keys"]),
            tuple(str(item) for item in raw["reused_component_refs"]),
            tuple(str(item) for item in raw["verifier_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(raw["lc13_directional_consumption"]),
            str(raw["runtime_status"]), raw["results_observed"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise GenerationChoiceContractError(f"{where} is not a safe POSIX path")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise GenerationChoiceContractError(f"{where} is not SHA-256")
    return text


def zero_execution_state() -> CanonicalJsonObject:
    return CanonicalJsonObject.from_value({key: 0 for key in EXECUTION_STATE_KEYS})


def build_gg01_contract_manifest(
        *, prerequisite_manifest_relative_path: str,
        prerequisite_manifest_sha256: str,
        ) -> GG01ContractManifest:
    direction = CanonicalJsonObject.from_value({
        "GENERATION": {
            "applicability": "REQUIRED",
            "consumer_refs": [
                "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py"],
            "fact_state": "CONTRACT_FROZEN",
            "write_permissions": ["CANDIDATE_ONLY", "NO_HOST_LEARNING_WRITE"],
        },
        "REASONING": {
            "applicability": "REQUIRED",
            "consumer_refs": [],
            "fact_state": "ABSENT",
            "write_permissions": [],
        },
        "UNDERSTANDING": {
            "applicability": "REQUIRED",
            "consumer_refs": [
                "src/pure_integer_ai/experiments/language_semantic_query.py"],
            "fact_state": "DESIGNED",
            "write_permissions": ["NO_HOST_LEARNING_WRITE"],
        },
    })
    return GG01ContractManifest(
        FORMAT_VERSION,
        "GG-01-generation-context-choice-contract-v1",
        "CONTRACT_FROZEN",
        ("GG-01", "LC-13"),
        prerequisite_manifest_relative_path,
        prerequisite_manifest_sha256,
        CHOICE_KINDS,
        GG01_CONTEXT_FIELDS,
        GG01_HYPOTHESIS_FIELDS,
        GG01_INVARIANTS_V1,
        tuple(sorted((
            "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
            "src/pure_integer_ai/cognition/shared/evidence_candidate.py",
            "src/pure_integer_ai/cognition/shared/generation_content.py",
            "src/pure_integer_ai/cognition/shared/generation_plan.py",
            "src/pure_integer_ai/cognition/shared/generation_structure_plan.py",
            "src/pure_integer_ai/cognition/shared/generation_surface.py",
            "src/pure_integer_ai/experiments/language_generation_connector.py",
        ))),
        GG01_VERIFIER_DIMENSIONS,
        GG01_NE_CONDITIONS,
        direction,
        "NOT_CONNECTED",
        0,
        zero_execution_state(),
    )


def build_gg01_contract_manifest_v2(
        *, superseded_manifest_sha256: str,
        ) -> GG01ContractManifest:
    """发布绑定 v1 的零值 stable-key 修订，保留原 artifact 不变。"""
    manifest = build_gg01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/gg01_generation_choice_contract_v1.json"),
        prerequisite_manifest_sha256=superseded_manifest_sha256,
    )
    return GG01ContractManifest(
        manifest.format_version,
        "GG-01-generation-context-choice-contract-v2-supersedes-v1",
        manifest.artifact_status,
        manifest.task_keys,
        manifest.prerequisite_manifest_relative_path,
        manifest.prerequisite_manifest_sha256,
        manifest.choice_kind_keys,
        manifest.context_field_keys,
        manifest.hypothesis_field_keys,
        GG01_INVARIANTS,
        manifest.reused_component_refs,
        manifest.verifier_dimensions,
        manifest.verifier_ne_conditions,
        manifest.lc13_directional_consumption,
        manifest.runtime_status,
        manifest.results_observed,
        manifest.execution_state,
    )


def write_gg01_contract_manifest(
        manifest: GG01ContractManifest, path: str | Path) -> Path:
    if not isinstance(manifest, GG01ContractManifest):
        raise GenerationChoiceContractError("GG-01 manifest type is invalid")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise GenerationChoiceContractError(
                "GG-01 manifest already exists with different content")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise GenerationChoiceContractError(
            "GG-01 manifest could not be published") from error
    return target


def read_gg01_contract_manifest(path: str | Path) -> GG01ContractManifest:
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise GenerationChoiceContractError("GG-01 manifest newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = GG01ContractManifest.from_dict(value)
    except GenerationChoiceContractError:
        raise
    except Exception as error:
        raise GenerationChoiceContractError("GG-01 manifest is damaged") from error
    if manifest.canonical_bytes() != payload:
        raise GenerationChoiceContractError("GG-01 manifest is not canonical")
    return manifest


__all__ = [
    "CHOICE_KINDS",
    "GG01ContractManifest",
    "GG01_CONTEXT_FIELDS",
    "GG01_HYPOTHESIS_FIELDS",
    "GG01_INVARIANTS",
    "GG01_INVARIANTS_V1",
    "GG01_NE_CONDITIONS",
    "GG01_VERIFIER_DIMENSIONS",
    "GenerationAddresseeContext",
    "GenerationChoiceCandidateMapper",
    "GenerationChoiceCandidateProtocol",
    "GenerationChoiceCondition",
    "GenerationChoiceContractError",
    "GenerationChoiceHypothesis",
    "GenerationChoiceOutcomeRef",
    "GenerationChoiceUseRef",
    "GenerationContentObligation",
    "GenerationContextContract",
    "GenerationDiscourseState",
    "GenerationExpressionConstraints",
    "LosslessIntegerKey",
    "OBLIGATION_REQUIREMENTS",
    "USE_KINDS",
    "build_gg01_contract_manifest",
    "build_gg01_contract_manifest_v2",
    "read_gg01_contract_manifest",
    "write_gg01_contract_manifest",
    "zero_execution_state",
]
