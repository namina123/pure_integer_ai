"""G-04 surface 反解析观察和显式 postcheck 请求对象。

本模块不解析 Unicode、不判断来源可信度，也不执行任务。调用方必须把同一次 typed
generation、实际 Artifact attachment、来源要求和任务要求显式放进请求；后续 verifier
只能消费这些对象，不能从隐藏全局状态补齐语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.generation_content import (
    ContentArtifactAttachment,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验失败原因或 stance 使用注入的 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


@dataclass(frozen=True)
class RecoveredGenerationProposition:
    """surface 反解析后恢复的候选键、BoundProposition 和运行归属。"""

    candidate_key: tuple[int, ...]
    proposition: BoundProposition
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.candidate_key, label="recovered candidate key")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("recovered proposition 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("recovered proposition source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("recovered proposition scope 类型错误")
        if semantic_source(self.proposition.template) != self.source:
            raise ValueError("recovered Proposition 与 source 不一致")
        _strict_key(self.trace, label="recovered proposition trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、命题、来源、scope 和反解析 trace。"""
        return (
            *_packed(self.candidate_key),
            *_packed(self.proposition.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class GenerationSourceRequirement:
    """一个 planned Proposition 对 citation 和独立可信度的显式要求。"""

    candidate_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    citation_required: bool
    trust_required: bool
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.candidate_key, label="source requirement candidate key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("source requirement source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("source requirement scope 类型错误")
        if (type(self.citation_required) is not bool
                or type(self.trust_required) is not bool):
            raise TypeError("source requirement 标志必须是严格 bool")
        if not self.citation_required and not self.trust_required:
            raise ValueError("source requirement 至少要求 citation 或 trust")
        _strict_key(self.trace, label="source requirement trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、来源、scope、两类要求和 trace。"""
        return (
            *_packed(self.candidate_key),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            int(self.citation_required),
            int(self.trust_required),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class GenerationTaskRequirement:
    """postcheck 调用方声明的任务、要求和期望结果键。"""

    task: ObjectIdentity
    requirement: ObjectIdentity
    expected_result_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, identity in (
                ("task requirement task", self.task),
                ("task requirement kind", self.requirement)):
            if not isinstance(identity, ObjectIdentity):
                raise TypeError(f"{label} 必须是 ObjectIdentity")
        _strict_key(
            self.expected_result_key, label="task expected result key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("task requirement source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("task requirement scope 类型错误")
        _strict_key(self.trace, label="task requirement trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回任务、要求、期望结果、归属和 trace。"""
        return (
            *_packed(self.task.stable_key()),
            *_packed(self.requirement.stable_key()),
            *_packed(self.expected_result_key),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class GenerationTaskObservation:
    """surface 或外部执行边界返回的实际任务结果观察。"""

    task: ObjectIdentity
    result_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, ObjectIdentity):
            raise TypeError("task observation task 类型错误")
        _strict_key(self.result_key, label="task observation result key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("task observation source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("task observation scope 类型错误")
        _strict_key(self.trace, label="task observation trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回任务、实际结果、归属和观察 trace。"""
        return (
            *_packed(self.task.stable_key()),
            *_packed(self.result_key),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class GenerationSurfaceObservation:
    """分支注入 parser 对实际 surface 产生的完整 typed 观察。"""

    execution_key: tuple[int, ...]
    rendered_key: tuple[int, ...]
    representations: tuple[ObjectIdentity, ...]
    branch: ObjectIdentity
    stance: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    propositions: tuple[RecoveredGenerationProposition, ...]
    artifact_keys: tuple[tuple[int, ...], ...]
    cited_sources: tuple[SourceRef, ...]
    structure_payload: tuple[int, ...]
    task_observations: tuple[GenerationTaskObservation, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.execution_key, label="surface observation execution key")
        _strict_key(self.rendered_key, label="surface observation rendered key")
        if not isinstance(self.representations, tuple) or not self.representations:
            raise ValueError("surface observation representations 必须非空")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.representations):
            raise TypeError("surface observation representations 类型错误")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("surface observation branch 必须是 LanguageBranch")
        _require_instruction(self.stance, label="surface observation stance")
        if not isinstance(self.source, SourceRef):
            raise TypeError("surface observation source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("surface observation scope 类型错误")
        if not isinstance(self.propositions, tuple) or any(
                not isinstance(item, RecoveredGenerationProposition)
                for item in self.propositions):
            raise TypeError("surface observation propositions 类型错误")
        proposition_keys = tuple(item.candidate_key for item in self.propositions)
        if len(set(proposition_keys)) != len(proposition_keys):
            raise ValueError("surface observation candidate key 不得重复")
        if not isinstance(self.artifact_keys, tuple):
            raise TypeError("surface observation artifact_keys 必须是 tuple")
        for key in self.artifact_keys:
            _strict_key(key, label="surface observation artifact key")
        if len(set(self.artifact_keys)) != len(self.artifact_keys):
            raise ValueError("surface observation artifact key 不得重复")
        if not isinstance(self.cited_sources, tuple) or any(
                not isinstance(item, SourceRef) for item in self.cited_sources):
            raise TypeError("surface observation cited_sources 类型错误")
        if len(set(self.cited_sources)) != len(self.cited_sources):
            raise ValueError("surface observation cited source 不得重复")
        _strict_key(self.structure_payload, label="surface structure payload")
        if not isinstance(self.task_observations, tuple) or any(
                not isinstance(item, GenerationTaskObservation)
                for item in self.task_observations):
            raise TypeError("surface observation task_observations 类型错误")
        task_keys = tuple(
            item.task.stable_key() for item in self.task_observations)
        if len(set(task_keys)) != len(task_keys):
            raise ValueError("surface observation task 不得重复")
        _strict_key(self.trace, label="surface observation trace")
        object.__setattr__(self, "propositions", tuple(sorted(
            self.propositions, key=lambda item: item.candidate_key)))
        object.__setattr__(self, "artifact_keys", tuple(sorted(
            self.artifact_keys)))
        object.__setattr__(self, "cited_sources", tuple(sorted(
            self.cited_sources, key=lambda item: item.stable_key())))
        object.__setattr__(self, "task_observations", tuple(sorted(
            self.task_observations,
            key=lambda item: item.task.stable_key(),
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回执行绑定、恢复语义、来源、结构和任务观察完整键。"""
        result = [
            *_packed(self.execution_key),
            *_packed(self.rendered_key),
            len(self.representations),
        ]
        for representation in self.representations:
            result.extend(_packed(representation.stable_key()))
        result.extend((
            *_packed(self.branch.stable_key()),
            *_packed(self.stance.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self.propositions),
        ))
        for proposition in self.propositions:
            result.extend(_packed(proposition.stable_key()))
        result.append(len(self.artifact_keys))
        for key in self.artifact_keys:
            result.extend(_packed(key))
        result.append(len(self.cited_sources))
        for source in self.cited_sources:
            result.extend(_packed(source.stable_key()))
        result.extend(_packed(self.structure_payload))
        result.append(len(self.task_observations))
        for observation in self.task_observations:
            result.extend(_packed(observation.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


@dataclass(frozen=True)
class GenerationSurfaceParseResult:
    """parser 的成功观察或分型失败原因。"""

    reason: ObjectIdentity
    trace: tuple[int, ...]
    observation: GenerationSurfaceObservation | None = None

    def __post_init__(self) -> None:
        _require_instruction(self.reason, label="surface parse reason")
        _strict_key(self.trace, label="surface parse trace")
        if (self.observation is not None
                and not isinstance(self.observation, GenerationSurfaceObservation)):
            raise TypeError("surface parse observation 类型错误")

    @property
    def succeeded(self) -> bool:
        """仅在 parser 返回完整 typed observation 时返回真。"""
        return self.observation is not None

    def stable_key(self) -> tuple[int, ...]:
        """返回 reason、trace 和可选观察键。"""
        result = [
            *_packed(self.reason.stable_key()),
            *_packed(self.trace),
            0 if self.observation is None else 1,
        ]
        if self.observation is not None:
            result.extend(_packed(self.observation.stable_key()))
        return tuple(result)


class GenerationSurfaceParser(Protocol):
    """按目标 LanguageBranch 反解析实际 Representation 和 renderer 输出。"""

    def parse(
            self, execution: TypedGenerationExecution,
            ) -> GenerationSurfaceParseResult:
        """返回 typed 观察或分型失败，不从码点名称猜语义。"""
        ...


@dataclass(frozen=True)
class GenerationPostcheckRequest:
    """同一次 generation 及其 Artifact、来源和任务复核要求。"""

    execution: TypedGenerationExecution
    artifacts: tuple[ContentArtifactAttachment, ...]
    source_requirements: tuple[GenerationSourceRequirement, ...]
    task_requirements: tuple[GenerationTaskRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.execution, TypedGenerationExecution):
            raise TypeError("postcheck execution 类型错误")
        if not self.execution.complete:
            raise ValueError("G-04 只复核已形成实际 surface 和 renderer 的执行")
        if not isinstance(self.artifacts, tuple) or any(
                not isinstance(item, ContentArtifactAttachment)
                for item in self.artifacts):
            raise TypeError("postcheck artifacts 类型错误")
        artifact_keys = tuple(item.stable_key() for item in self.artifacts)
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("postcheck Artifact attachment 不得重复")
        selection = self.execution.surface.preview.request.structure.selection
        if set(artifact_keys) != set(selection.selected_artifact_keys):
            raise ValueError("postcheck Artifact 必须精确覆盖 G-01 已采用 attachment")
        if not isinstance(self.source_requirements, tuple) or any(
                not isinstance(item, GenerationSourceRequirement)
                for item in self.source_requirements):
            raise TypeError("postcheck source_requirements 类型错误")
        source_map = {
            item.candidate_key: item for item in self.source_requirements}
        if len(source_map) != len(self.source_requirements):
            raise ValueError("同一 candidate 不得重复 source requirement")
        planned = {
            item.candidate_key: item
            for item in self.execution.surface.preview.request.structure
            .propositions.propositions
        }
        if set(source_map) != set(planned):
            raise ValueError("source requirement 必须逐点覆盖 planned Proposition")
        for key, requirement in source_map.items():
            proposition = planned[key]
            if (requirement.source != proposition.source
                    or requirement.scope != proposition.scope):
                raise ValueError("source requirement 与 planned Proposition 归属不一致")
        if not isinstance(self.task_requirements, tuple) or any(
                not isinstance(item, GenerationTaskRequirement)
                for item in self.task_requirements):
            raise TypeError("postcheck task_requirements 类型错误")
        task_keys = tuple(
            item.task.stable_key() for item in self.task_requirements)
        if len(set(task_keys)) != len(task_keys):
            raise ValueError("同一 task 不得重复 requirement")
        goal = self.execution.plan.request.goal
        for requirement in self.task_requirements:
            if requirement.scope != goal.scope:
                raise ValueError(
                    "task requirement 与 generation goal query scope 不一致")
        object.__setattr__(self, "artifacts", tuple(sorted(
            self.artifacts, key=lambda item: item.stable_key())))
        object.__setattr__(self, "source_requirements", tuple(sorted(
            self.source_requirements, key=lambda item: item.candidate_key)))
        object.__setattr__(self, "task_requirements", tuple(sorted(
            self.task_requirements, key=lambda item: item.task.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation、全部 attachment、来源和任务要求完整键。"""
        result = [*_packed(self.execution.stable_key()), len(self.artifacts)]
        for artifact in self.artifacts:
            result.extend(_packed(artifact.stable_key()))
        result.append(len(self.source_requirements))
        for requirement in self.source_requirements:
            result.extend(_packed(requirement.stable_key()))
        result.append(len(self.task_requirements))
        for requirement in self.task_requirements:
            result.extend(_packed(requirement.stable_key()))
        return tuple(result)


__all__ = [
    "GenerationPostcheckRequest",
    "GenerationSourceRequirement",
    "GenerationSurfaceObservation",
    "GenerationSurfaceParseResult",
    "GenerationSurfaceParser",
    "GenerationTaskObservation",
    "GenerationTaskRequirement",
    "RecoveredGenerationProposition",
]
