"""G-04 多维生成后复核 runtime。

本模块复用 R-09 VerificationReport，分别裁决结构、核心 Proposition、scope、Artifact、
来源和任务结果。全部复核只读且无综合 reward；失败只保留分维 verdict、claim 和 trace。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactVerificationObservation,
    ArtifactVerificationRequest,
    FormalArtifactVerifier,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSourceRequirement,
    GenerationSurfaceObservation,
    GenerationSurfaceParseRequest,
    GenerationSurfaceParseResult,
    GenerationSurfaceParser,
    GenerationTaskRequirement,
    RecoveredGenerationProposition,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    MultiVerifierOrchestrator,
    VerificationEvaluation,
    VerificationReport,
    VerificationResult,
    VerifierRegistration,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _require_reason(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验内置分型原因使用注入的 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


def _detail(reason: ObjectIdentity, *keys: tuple[int, ...]) -> tuple[int, ...]:
    """把分型原因和可选完整键编码为 verification detail。"""
    result = [*_packed(reason.stable_key()), len(keys)]
    for key in keys:
        result.extend(_packed(key))
    return tuple(result)


@dataclass(frozen=True)
class GenerationPostcheckProtocol:
    """注入六个 dimension/verifier 键和共享分型 reason。"""

    structure_dimension: ProtocolKey
    proposition_dimension: ProtocolKey
    scope_dimension: ProtocolKey
    artifact_dimension: ProtocolKey
    source_dimension: ProtocolKey
    task_dimension: ProtocolKey
    structure_verifier: ProtocolKey
    proposition_verifier: ProtocolKey
    scope_verifier: ProtocolKey
    artifact_verifier: ProtocolKey
    source_verifier: ProtocolKey
    task_verifier: ProtocolKey
    parse_contract_failure: ObjectIdentity
    external_contract_failure: ObjectIdentity
    proposition_match: ObjectIdentity
    proposition_mismatch: ObjectIdentity
    scope_match: ObjectIdentity
    scope_mismatch: ObjectIdentity
    artifact_match: ObjectIdentity
    artifact_missing: ObjectIdentity
    artifact_rejected: ObjectIdentity
    artifact_unknown: ObjectIdentity
    source_match: ObjectIdentity
    source_citation_missing: ObjectIdentity
    source_unknown: ObjectIdentity
    task_mismatch: ObjectIdentity
    task_unknown: ObjectIdentity

    def __post_init__(self) -> None:
        dimensions = self.dimensions()
        verifiers = self.verifiers()
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("G-04 dimension ProtocolKey 必须互不相同")
        if len(set(verifiers)) != len(verifiers):
            raise ValueError("G-04 verifier ProtocolKey 必须互不相同")
        reasons = self.reasons()
        if len(set(reasons)) != len(reasons):
            raise ValueError("G-04 内置 reason 必须互不相同")
        for reason in reasons:
            _require_reason(reason, label="G-04 reason")

    def dimensions(self) -> tuple[ProtocolKey, ...]:
        """返回结构、命题、scope、Artifact、来源和任务维度。"""
        return (
            self.structure_dimension,
            self.proposition_dimension,
            self.scope_dimension,
            self.artifact_dimension,
            self.source_dimension,
            self.task_dimension,
        )

    def verifiers(self) -> tuple[ProtocolKey, ...]:
        """返回六个维度各自的 verifier 协议键。"""
        return (
            self.structure_verifier,
            self.proposition_verifier,
            self.scope_verifier,
            self.artifact_verifier,
            self.source_verifier,
            self.task_verifier,
        )

    def bindings(self) -> tuple[tuple[ProtocolKey, ProtocolKey], ...]:
        """返回六个 dimension/verifier 的固定一一绑定。"""
        return tuple(zip(self.dimensions(), self.verifiers()))

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """返回 runtime 内置的全部分型原因身份。"""
        return (
            self.parse_contract_failure,
            self.external_contract_failure,
            self.proposition_match,
            self.proposition_mismatch,
            self.scope_match,
            self.scope_mismatch,
            self.artifact_match,
            self.artifact_missing,
            self.artifact_rejected,
            self.artifact_unknown,
            self.source_match,
            self.source_citation_missing,
            self.source_unknown,
            self.task_mismatch,
            self.task_unknown,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回六维绑定和全部分型 reason 的完整协议键。"""
        result = [len(self.bindings())]
        for dimension, verifier in self.bindings():
            result.extend(_packed(dimension.stable_key()))
            result.extend(_packed(verifier.stable_key()))
        result.append(len(self.reasons()))
        for reason in self.reasons():
            result.extend(_packed(reason.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class GenerationStructureCheckRequest:
    """结构 verifier 可见的原计划、实际 surface 和反解析结构载荷。"""

    postcheck: GenerationPostcheckRequest
    observation: GenerationSurfaceObservation


@dataclass(frozen=True)
class GenerationSourceCheckRequest:
    """来源 verifier 可见的逐命题要求、恢复结果和 citation 集。"""

    postcheck: GenerationPostcheckRequest
    observation: GenerationSurfaceObservation
    requirements: tuple[GenerationSourceRequirement, ...]
    propositions: tuple[RecoveredGenerationProposition, ...]


@dataclass(frozen=True)
class GenerationTaskCheckRequest:
    """任务 verifier 可见的显式要求和实际任务观察。"""

    postcheck: GenerationPostcheckRequest
    observation: GenerationSurfaceObservation
    requirements: tuple[GenerationTaskRequirement, ...]


class GenerationStructureVerifier(Protocol):
    """比较 G-02/S-07 structure plan 与反解析结构载荷。"""

    def verify(
            self, request: GenerationStructureCheckRequest,
            ) -> VerificationEvaluation:
        """返回独立结构 verdict，不写 effect。"""
        ...


class GenerationSourceVerifier(Protocol):
    """独立裁决 citation 之外的来源可信度要求。"""

    def verify(
            self, request: GenerationSourceCheckRequest,
            ) -> VerificationEvaluation:
        """返回来源 verdict；citation 存在本身不得强制 support。"""
        ...


class GenerationTaskVerifier(Protocol):
    """按显式 task requirement 核验实际任务结果。"""

    def verify(
            self, request: GenerationTaskCheckRequest,
            ) -> VerificationEvaluation:
        """返回任务结果 verdict，不从 goal_kind 猜任务语义。"""
        ...


@dataclass(frozen=True)
class _GenerationPostcheckContext:
    """一次 parser 结果和 postcheck request 的内部只读组合。"""

    request: GenerationPostcheckRequest
    parse_request: GenerationSurfaceParseRequest
    parsed: GenerationSurfaceParseResult


def _verification_result_key(result: VerificationResult) -> tuple[int, ...]:
    """把无 effect、无运行异常的分维结果编码为确定性键。"""
    if result.operational_failure is not None:
        raise ValueError("G-04 领域失败不得退化为 operational_failure")
    if result.proposed_effects or result.committed_effects or result.artifact is not None:
        raise ValueError("G-04 只读结果不得携带 effect 或临时 artifact")
    values = [
        *_packed(result.dimension.stable_key()),
        *_packed(result.verifier.stable_key()),
        result.applicability,
        result.verdict,
        len(result.claim_keys),
    ]
    for key in result.claim_keys:
        values.extend(_packed(key))
    values.extend(_packed(result.detail))
    for identity in (result.source, result.scope):
        values.append(0 if identity is None else 1)
        if identity is not None:
            values.extend(_packed(identity.stable_key()))
    return tuple(values)


@dataclass(frozen=True)
class GenerationPostcheckRun:
    """一次 G-04 parse 和六维只读 VerificationReport。"""

    protocol: GenerationPostcheckProtocol
    request: GenerationPostcheckRequest
    parsed: GenerationSurfaceParseResult
    report: VerificationReport

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, GenerationPostcheckProtocol):
            raise TypeError("postcheck run protocol 类型错误")
        if not isinstance(self.request, GenerationPostcheckRequest):
            raise TypeError("postcheck run request 类型错误")
        if not isinstance(self.parsed, GenerationSurfaceParseResult):
            raise TypeError("postcheck run parsed 类型错误")
        if not isinstance(self.report, VerificationReport):
            raise TypeError("postcheck run report 类型错误")
        if not self.report.read_only:
            raise ValueError("G-04 report 必须只读")
        actual = tuple(
            (item.dimension, item.verifier) for item in self.report.results)
        if set(actual) != set(self.protocol.bindings()) or len(actual) != 6:
            raise ValueError("G-04 report 未逐点覆盖六个 dimension/verifier")
        for result in self.report.results:
            _verification_result_key(result)
            if (result.applicability == APPLICABILITY_APPLICABLE
                    and not result.claim_keys):
                raise ValueError("G-04 applicable 结果必须携带非空 claim")

    @property
    def complete(self) -> bool:
        """仅在 parse 成功且全部 applicable 维度 support 时返回真。"""
        if not self.parsed.succeeded:
            return False
        for result in self.report.results:
            if result.applicability == APPLICABILITY_UNKNOWN:
                return False
            if (result.applicability == APPLICABILITY_APPLICABLE
                    and result.verdict != VERDICT_SUPPORT):
                return False
        required = {
            self.protocol.structure_dimension,
            self.protocol.proposition_dimension,
            self.protocol.scope_dimension,
        }
        applicable = {
            item.dimension for item in self.report.applicable_results()}
        return required.issubset(applicable)

    def stable_key(self) -> tuple[int, ...]:
        """返回协议、请求、parse 和六维 verdict 的完整键。"""
        result = [
            len(self.protocol.bindings()),
            *_packed(self.request.stable_key()),
            *_packed(self.parsed.stable_key()),
            len(self.report.results),
        ]
        for dimension, verifier in self.protocol.bindings():
            result.extend(_packed(dimension.stable_key()))
            result.extend(_packed(verifier.stable_key()))
        result.append(len(self.protocol.reasons()))
        for reason in self.protocol.reasons():
            result.extend(_packed(reason.stable_key()))
        for verification in self.report.results:
            result.extend(_packed(_verification_result_key(verification)))
        return tuple(result)


class GenerationPostcheckRuntime:
    """运行 parser 和六个隔离 verifier，不写 Memory、Use、effect 或 reward。"""

    def __init__(
            self,
            protocol: GenerationPostcheckProtocol,
            parser: GenerationSurfaceParser,
            structure_verifier: GenerationStructureVerifier,
            source_verifier: GenerationSourceVerifier,
            *,
            artifact_verifier: FormalArtifactVerifier | None = None,
            task_verifier: GenerationTaskVerifier | None = None,
            ) -> None:
        if not isinstance(protocol, GenerationPostcheckProtocol):
            raise TypeError("postcheck protocol 类型错误")
        if not hasattr(parser, "parse"):
            raise TypeError("postcheck parser 必须实现 parse")
        if not hasattr(structure_verifier, "verify"):
            raise TypeError("structure verifier 必须实现 verify")
        if not hasattr(source_verifier, "verify"):
            raise TypeError("source verifier 必须实现 verify")
        if artifact_verifier is not None and not hasattr(
                artifact_verifier, "verify"):
            raise TypeError("artifact verifier 必须实现 verify")
        if task_verifier is not None and not hasattr(task_verifier, "verify"):
            raise TypeError("task verifier 必须实现 verify")
        self.protocol = protocol
        self.parser = parser
        self.structure_verifier = structure_verifier
        self.source_verifier = source_verifier
        self.artifact_verifier = artifact_verifier
        self.task_verifier = task_verifier
        self.orchestrator = MultiVerifierOrchestrator()

    def run(self, request: GenerationPostcheckRequest) -> GenerationPostcheckRun:
        """反解析同一次 typed execution，并生成六维只读报告。"""
        if not isinstance(request, GenerationPostcheckRequest):
            raise TypeError("postcheck runtime 只接受 GenerationPostcheckRequest")
        parse_request = GenerationSurfaceParseRequest.from_execution(
            request.execution)
        parsed = self.parser.parse(parse_request)
        if not isinstance(parsed, GenerationSurfaceParseResult):
            raise TypeError("surface parser 必须返回 GenerationSurfaceParseResult")
        context = _GenerationPostcheckContext(request, parse_request, parsed)
        report = self.orchestrator.run(
            context, self._registrations(), read_only=True)
        return GenerationPostcheckRun(self.protocol, request, parsed, report)

    def _registrations(self) -> tuple[VerifierRegistration, ...]:
        """建立六个无 committer 的固定分维 registration。"""
        protocol = self.protocol
        return (
            VerifierRegistration(
                protocol.structure_dimension,
                protocol.structure_verifier,
                lambda request: True,
                self._structure_evaluation,
            ),
            VerifierRegistration(
                protocol.proposition_dimension,
                protocol.proposition_verifier,
                lambda request: True,
                self._proposition_evaluation,
            ),
            VerifierRegistration(
                protocol.scope_dimension,
                protocol.scope_verifier,
                lambda request: True,
                self._scope_evaluation,
            ),
            VerifierRegistration(
                protocol.artifact_dimension,
                protocol.artifact_verifier,
                lambda request: bool(
                    request.request.execution.surface.preview.request
                    .structure.selection.selected_artifact_keys),
                self._artifact_evaluation,
            ),
            VerifierRegistration(
                protocol.source_dimension,
                protocol.source_verifier,
                lambda request: bool(request.request.source_requirements),
                self._source_evaluation,
            ),
            VerifierRegistration(
                protocol.task_dimension,
                protocol.task_verifier,
                lambda request: bool(request.request.task_requirements),
                self._task_evaluation,
            ),
        )

    @staticmethod
    def _claim_keys(context: _GenerationPostcheckContext) -> tuple[tuple[int, ...], ...]:
        """返回 planned candidate 键；空内容以本次 execution 键作为复核 claim。"""
        planned = context.request.execution.surface.preview.request.structure
        keys = tuple(
            item.candidate_key for item in planned.propositions.propositions)
        return keys or (context.request.execution.stable_key(),)

    @staticmethod
    def _observation_binding_matches(
            context: _GenerationPostcheckContext,
            ) -> bool:
        """核验 parser 观察绑定受限请求，且恢复了实际 Representation。"""
        observation = context.parsed.observation
        if observation is None:
            return False
        execution = context.request.execution
        return (
            observation.parse_request_key == context.parse_request.stable_key()
            and observation.representations == execution.representations
        )

    def _unknown(
            self,
            context: _GenerationPostcheckContext,
            reason: ObjectIdentity,
            ) -> VerificationEvaluation:
        """返回绑定当前 generation source/scope 的 unknown 分型。"""
        goal = context.request.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_UNKNOWN,
            self._claim_keys(context),
            detail=_detail(reason),
            source=goal.source,
            scope=goal.scope,
        )

    def _external_evaluation(
            self,
            context: _GenerationPostcheckContext,
            verifier,
            request,
            *,
            missing_reason: ObjectIdentity,
            ) -> VerificationEvaluation:
        """调用注入 verifier，并把契约漂移转换为 typed unknown。"""
        if verifier is None:
            return self._unknown(context, missing_reason)
        try:
            evaluation = verifier.verify(request)
        except Exception:
            return self._unknown(
                context, self.protocol.external_contract_failure)
        goal = context.request.execution.plan.request.goal
        if (not isinstance(evaluation, VerificationEvaluation)
                or evaluation.proposed_effects
                or evaluation.artifact is not None
                or not evaluation.claim_keys
                or evaluation.source != goal.source
                or evaluation.scope != goal.scope
                or not evaluation.detail):
            return self._unknown(
                context, self.protocol.external_contract_failure)
        return evaluation

    def _structure_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """先守 execution/branch/stance，再委托独立结构 verifier。"""
        observation = context.parsed.observation
        if observation is None:
            goal = context.request.execution.plan.request.goal
            return VerificationEvaluation(
                VERDICT_REFUTE,
                self._claim_keys(context),
                detail=_detail(context.parsed.reason),
                source=goal.source,
                scope=goal.scope,
            )
        execution = context.request.execution
        structure = execution.surface.preview.request.structure
        goal = execution.plan.request.goal
        if (not self._observation_binding_matches(context)
                or observation.branch != goal.target_branch
                or observation.stance != structure.selection.stance):
            return VerificationEvaluation(
                VERDICT_REFUTE,
                self._claim_keys(context),
                detail=_detail(self.protocol.parse_contract_failure),
                source=goal.source,
                scope=goal.scope,
            )
        return self._external_evaluation(
            context,
            self.structure_verifier,
            GenerationStructureCheckRequest(context.request, observation),
            missing_reason=self.protocol.external_contract_failure,
        )

    def _proposition_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """精确核验反解析结果未缺失、替换或增加核心 BoundProposition。"""
        if (context.parsed.observation is None
                or not self._observation_binding_matches(context)):
            return self._unknown(context, context.parsed.reason)
        expected_items = context.request.execution.surface.preview.request
        expected = {
            item.candidate_key: item.proposition
            for item in expected_items.structure.propositions.propositions
        }
        recovered = {
            item.candidate_key: item.proposition
            for item in context.parsed.observation.propositions
        }
        matches = expected == recovered
        reason = (
            self.protocol.proposition_match
            if matches else self.protocol.proposition_mismatch)
        goal = context.request.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            self._claim_keys(context),
            detail=_detail(
                reason,
                tuple(value for key in sorted(expected)
                      for value in _packed(key)),
                tuple(value for key in sorted(recovered)
                      for value in _packed(key)),
            ),
            source=goal.source,
            scope=goal.scope,
        )

    def _scope_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """核验全局及逐命题 source/scope 未在 surface 往返中漂移。"""
        observation = context.parsed.observation
        if observation is None or not self._observation_binding_matches(context):
            return self._unknown(context, context.parsed.reason)
        execution = context.request.execution
        goal = execution.plan.request.goal
        expected = {
            item.candidate_key: (item.source, item.scope)
            for item in execution.surface.preview.request.structure
            .propositions.propositions
        }
        recovered = {
            item.candidate_key: (item.source, item.scope)
            for item in observation.propositions
        }
        matches = (
            observation.source == goal.source
            and observation.scope == goal.scope
            and expected == recovered
        )
        reason = self.protocol.scope_match if matches else self.protocol.scope_mismatch
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            self._claim_keys(context),
            detail=_detail(reason),
            source=goal.source,
            scope=goal.scope,
        )

    def _artifact_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """核验 surface 保留 Artifact key，并重新调用独立 S-06 verifier。"""
        observation = context.parsed.observation
        if observation is None or not self._observation_binding_matches(context):
            return self._unknown(context, context.parsed.reason)
        expected = context.request.execution.surface.preview.request.structure
        selected_keys = expected.selection.selected_artifact_keys
        goal = context.request.execution.plan.request.goal
        if observation.artifact_keys != selected_keys:
            return VerificationEvaluation(
                VERDICT_REFUTE,
                selected_keys,
                detail=_detail(self.protocol.artifact_missing),
                source=goal.source,
                scope=goal.scope,
            )
        if self.artifact_verifier is None:
            return self._unknown(context, self.protocol.artifact_unknown)
        attachments = {
            item.stable_key(): item for item in context.request.artifacts}
        has_unknown = False
        for key in selected_keys:
            result = attachments[key].result
            if (not result.succeeded
                    or result.execution is None
                    or result.value is None):
                return VerificationEvaluation(
                    VERDICT_REFUTE,
                    selected_keys,
                    detail=_detail(self.protocol.artifact_rejected),
                    source=goal.source,
                    scope=goal.scope,
                )
            try:
                observation_result = self.artifact_verifier.verify(
                    ArtifactVerificationRequest(
                        result.invocation, result.execution, result.value))
            except Exception:
                has_unknown = True
                continue
            if not isinstance(
                    observation_result, ArtifactVerificationObservation):
                has_unknown = True
                continue
            if (observation_result.authority
                    != result.invocation.definition.verifier
                    or observation_result.source != result.invocation.source
                    or observation_result.scope != result.invocation.scope):
                has_unknown = True
                continue
            if observation_result.accepted is False:
                return VerificationEvaluation(
                    VERDICT_REFUTE,
                    selected_keys,
                    detail=_detail(self.protocol.artifact_rejected),
                    source=goal.source,
                    scope=goal.scope,
                )
            if observation_result.accepted is None:
                has_unknown = True
        verdict = VERDICT_UNKNOWN if has_unknown else VERDICT_SUPPORT
        reason = (
            self.protocol.artifact_unknown
            if has_unknown else self.protocol.artifact_match)
        return VerificationEvaluation(
            verdict,
            selected_keys,
            detail=_detail(reason),
            source=goal.source,
            scope=goal.scope,
        )

    def _source_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """先核验 citation 要求，再由独立 verifier 裁决来源可信度。"""
        observation = context.parsed.observation
        if observation is None or not self._observation_binding_matches(context):
            return self._unknown(context, context.parsed.reason)
        citations = set(observation.cited_sources)
        missing = tuple(
            item.candidate_key
            for item in context.request.source_requirements
            if item.citation_required and item.source not in citations
        )
        goal = context.request.execution.plan.request.goal
        if missing:
            return VerificationEvaluation(
                VERDICT_REFUTE,
                missing,
                detail=_detail(self.protocol.source_citation_missing),
                source=goal.source,
                scope=goal.scope,
            )
        trust_requirements = tuple(
            item for item in context.request.source_requirements
            if item.trust_required
        )
        if not trust_requirements:
            return VerificationEvaluation(
                VERDICT_SUPPORT,
                tuple(
                    item.candidate_key
                    for item in context.request.source_requirements
                ),
                detail=_detail(self.protocol.source_match),
                source=goal.source,
                scope=goal.scope,
            )
        trust_keys = {
            item.candidate_key for item in trust_requirements}
        return self._external_evaluation(
            context,
            self.source_verifier,
            GenerationSourceCheckRequest(
                context.request,
                observation,
                trust_requirements,
                tuple(
                    item for item in observation.propositions
                    if item.candidate_key in trust_keys
                ),
            ),
            missing_reason=self.protocol.source_unknown,
        )

    def _task_evaluation(
            self, context: _GenerationPostcheckContext,
            ) -> VerificationEvaluation:
        """核验 task 集与归属完整，再委托显式任务 verifier。"""
        observation = context.parsed.observation
        if observation is None or not self._observation_binding_matches(context):
            return self._unknown(context, context.parsed.reason)
        requirements = {
            item.task: item for item in context.request.task_requirements}
        actual = {
            item.task: item for item in observation.task_observations}
        goal = context.request.execution.plan.request.goal
        if (set(requirements) != set(actual)
                or any(
                    actual[task].source != requirement.source
                    or actual[task].scope != requirement.scope
                    for task, requirement in requirements.items())):
            return VerificationEvaluation(
                VERDICT_REFUTE,
                tuple(item.task.stable_key() for item in requirements.values()),
                detail=_detail(self.protocol.task_mismatch),
                source=goal.source,
                scope=goal.scope,
            )
        return self._external_evaluation(
            context,
            self.task_verifier,
            GenerationTaskCheckRequest(
                context.request,
                observation,
                context.request.task_requirements,
            ),
            missing_reason=self.protocol.task_unknown,
        )


__all__ = [
    "GenerationPostcheckProtocol",
    "GenerationPostcheckRun",
    "GenerationPostcheckRuntime",
    "GenerationSourceCheckRequest",
    "GenerationSourceVerifier",
    "GenerationStructureCheckRequest",
    "GenerationStructureVerifier",
    "GenerationTaskCheckRequest",
    "GenerationTaskVerifier",
]
