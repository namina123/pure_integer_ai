"""从实际 Unicode units 恢复 grounded response act、结构和任务观察。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSurfaceObservation,
    GenerationSurfaceParseRequest,
    GenerationSurfaceParseResult,
    GenerationTaskObservation,
    GenerationTaskRequirement,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationSourceCheckRequest,
    GenerationStructureCheckRequest,
    GenerationTaskCheckRequest,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompilation,
    GroundedResponseActVariant,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationEvaluation,
)


# object-model: exception
class GroundedResponseActParserError(ValueError):
    """response-act parser catalog 或实际 readback 归属发生漂移。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedResponseActParserError(
            f"{where} 必须是非空严格整数 tuple")
    return value


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 parser 和 verifier 原因使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedResponseActParserError(
            f"{where} 必须是 MinimalInstruction")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActParserProtocol:
    """注入成功、无匹配、歧义三类 parse 结果。"""

    succeeded: ObjectIdentity
    no_match: ObjectIdentity
    ambiguous: ObjectIdentity

    def __post_init__(self) -> None:
        values = (self.succeeded, self.no_match, self.ambiguous)
        if len(set(values)) != len(values):
            raise GroundedResponseActParserError(
                "response-act parser reason 不得重复")
        for value in values:
            _instruction(value, where="response-act parser reason")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActParserGrammar:
    """受限 parser 的单个 response-act grammar，不含 selected id 或 label。"""

    pattern_id: int
    renderer: ObjectIdentity
    branch: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    stance: ObjectIdentity
    structure: ObjectIdentity
    representation: ObjectIdentity
    units: tuple[int, ...]
    task: ObjectIdentity
    task_result_key: tuple[int, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedResponseActParserError("parser pattern id 非法")
        _instruction(self.renderer, where="response-act parser renderer")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise TypeError("response-act parser branch 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("response-act parser source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("response-act parser scope 类型错误")
        _instruction(self.stance, where="response-act parser stance")
        if (not isinstance(self.structure, ObjectIdentity)
                or self.structure.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise TypeError("response-act parser structure 类型错误")
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise TypeError("response-act parser representation 类型错误")
        _strict_key(self.units, where="response-act parser units")
        _instruction(self.task, where="response-act parser task")
        _strict_key(
            self.task_result_key, where="response-act parser task result")
        if (not self.support_teacher_keys
                or self.support_teacher_keys != tuple(sorted(
                    set(self.support_teacher_keys)))):
            raise GroundedResponseActParserError(
                "response-act parser 缺 teacher Evidence 追溯")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActParserCatalog:
    """保存同一 response-act compilation 的全部合法 grammar。"""

    grammars: tuple[GroundedResponseActParserGrammar, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.grammars, tuple) or not self.grammars
                or any(not isinstance(item, GroundedResponseActParserGrammar)
                       for item in self.grammars)):
            raise GroundedResponseActParserError("parser catalog 不能为空")
        ids = tuple(item.pattern_id for item in self.grammars)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedResponseActParserError(
                "parser grammar id 必须唯一递增")


def build_grounded_response_act_parser_catalog(
        compilation: GroundedResponseActCompilation,
        renderer: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        ) -> GroundedResponseActParserCatalog:
    """从编译 grammar 和运行归属建立 read-only parser catalog。"""
    if not isinstance(compilation, GroundedResponseActCompilation):
        raise TypeError("response-act parser compilation 类型错误")
    _instruction(renderer, where="response-act parser renderer")
    if not isinstance(source, SourceRef):
        raise TypeError("response-act parser source 类型错误")
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("response-act parser scope 类型错误")
    grammars = tuple(
        GroundedResponseActParserGrammar(
            variant.pattern_id,
            renderer,
            variant.template.branch,
            source,
            scope,
            variant.template.stance,
            variant.template.slot.structure,
            variant.representation,
            representation_parts(variant.representation)[1],
            variant.task,
            variant.task_result_key,
            variant.support_teacher_keys,
        )
        for variant in compilation.variants
    )
    return GroundedResponseActParserCatalog(grammars)


# object-model: parser
class GroundedResponseActSurfaceParser:
    """只从 actual units、branch、source 和 scope 恢复 response act。"""

    def __init__(
            self,
            protocol: GroundedResponseActParserProtocol,
            catalog: GroundedResponseActParserCatalog,
            ) -> None:
        if not isinstance(protocol, GroundedResponseActParserProtocol):
            raise TypeError("response-act parser protocol 类型错误")
        if not isinstance(catalog, GroundedResponseActParserCatalog):
            raise TypeError("response-act parser catalog 类型错误")
        self.protocol = protocol
        self.catalog = catalog

    def parse(
            self,
            request: GenerationSurfaceParseRequest,
            ) -> GenerationSurfaceParseResult:
        """唯一匹配 grammar 并形成零命题、显式 task 的 typed 观察。"""
        if not isinstance(request, GenerationSurfaceParseRequest):
            raise TypeError("response-act parser request 类型错误")
        relevant = tuple(
            item for item in self.catalog.grammars
            if (item.renderer == request.renderer
                and item.branch == request.branch
                and item.source == request.source
                and item.scope == request.scope)
        )
        matches = tuple(item for item in relevant if item.units == request.units)
        if len(matches) > 1:
            return GenerationSurfaceParseResult(
                self.protocol.ambiguous,
                (1, len(matches), *request.stable_key()),
            )
        if not matches:
            return GenerationSurfaceParseResult(
                self.protocol.no_match,
                (2, len(relevant), *request.stable_key()),
            )
        grammar = matches[0]
        observation = GenerationSurfaceObservation(
            request.stable_key(),
            (grammar.representation,),
            grammar.branch,
            grammar.stance,
            grammar.source,
            grammar.scope,
            (),
            (),
            (),
            grammar.structure.stable_key(),
            (GenerationTaskObservation(
                grammar.task,
                grammar.task_result_key,
                grammar.source,
                grammar.scope,
                (1, grammar.pattern_id),
            ),),
            (3, grammar.pattern_id, *request.stable_key()),
        )
        return GenerationSurfaceParseResult(
            self.protocol.succeeded,
            (4, grammar.pattern_id, *request.stable_key()),
            observation,
        )


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedResponseActStructureVerifier:
    """独立比较 actual G-02 单句结构与 parser readback 结构。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="response-act structure match")
        _instruction(
            self.mismatched_reason, where="response-act structure mismatch")

    def verify(
            self,
            request: GenerationStructureCheckRequest,
            ) -> VerificationEvaluation:
        """不读取 surface 文本，只比较 structure payload 和零命题边界。"""
        if not isinstance(request, GenerationStructureCheckRequest):
            raise TypeError("response-act structure verifier request 类型错误")
        execution = request.postcheck.execution
        structure = execution.surface.preview.request.structure
        sentences = structure.syntax.sentences
        matches = (
            len(sentences) == 1
            and set(structure.syntax.suppressed_candidate_keys)
            == set(structure.selection.selected_candidate_keys)
            and not sentences[0].proposition_keys
            and request.observation.structure_payload
            == sentences[0].structure.stable_key()
        )
        goal = execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            (execution.stable_key(),),
            detail=(1 if matches else 2,
                    *(self.matched_reason if matches
                      else self.mismatched_reason).stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedResponseActSourceVerifier:
    """核验 non-answer readback 没有伪造命题、引用或来源要求。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        """固定成功与失败分型理由，避免 verifier 在运行时补造语义。"""
        _instruction(self.matched_reason, where="response-act source match")
        _instruction(
            self.mismatched_reason, where="response-act source mismatch")

    def verify(
            self,
            request: GenerationSourceCheckRequest,
            ) -> VerificationEvaluation:
        """要求完整 G-04 请求和 readback 都保持零来源性 non-answer 边界。"""
        if not isinstance(request, GenerationSourceCheckRequest):
            raise TypeError("response-act source verifier request 类型错误")
        matches = (
            not request.postcheck.source_requirements
            and not request.requirements
            and not request.observation.propositions
            and not request.propositions
            and not request.observation.cited_sources
        )
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            (request.postcheck.execution.stable_key(),),
            detail=(1 if matches else 2,
                    *(self.matched_reason if matches
                      else self.mismatched_reason).stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedResponseActTaskVerifier:
    """独立比较显式 task requirement 与 parser task observation。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="response-act task match")
        _instruction(self.mismatched_reason, where="response-act task mismatch")

    def verify(
            self,
            request: GenerationTaskCheckRequest,
            ) -> VerificationEvaluation:
        """逐 task 比较 result/source/scope，不从 goal kind 猜 response act。"""
        if not isinstance(request, GenerationTaskCheckRequest):
            raise TypeError("response-act task verifier request 类型错误")
        actual = {
            item.task: item for item in request.observation.task_observations}
        matches = (
            len(actual) == len(request.requirements)
            and all(
                requirement.task in actual
                and actual[requirement.task].result_key
                == requirement.expected_result_key
                and actual[requirement.task].source == requirement.source
                and actual[requirement.task].scope == requirement.scope
                for requirement in request.requirements
            )
        )
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            tuple(item.task.stable_key() for item in request.requirements),
            detail=(1 if matches else 2,
                    *(self.matched_reason if matches
                      else self.mismatched_reason).stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


# object-model: mapper; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedResponseActPostcheckMapper:
    """为同次 response-act generation 建立显式 G-04 task requirement。"""

    variant: GroundedResponseActVariant
    trace_prefix: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variant, GroundedResponseActVariant):
            raise TypeError("response-act postcheck variant 类型错误")
        _strict_key(
            self.trace_prefix, where="response-act postcheck trace prefix")

    def build(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            generation,
            ) -> GenerationPostcheckRequest:
        """绑定同次 execution，并声明 learned response act 的任务结果。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("response-act postcheck question request 类型错误")
        if not isinstance(query, QuestionQuery) or query.request != request:
            raise ValueError("response-act postcheck query 漂移")
        if (not isinstance(result, QuestionExecutionResult)
                or result.query != query
                or generation.plan.request != result.planning_request()):
            raise ValueError("response-act postcheck execution 漂移")
        goal = generation.plan.request.goal
        return GenerationPostcheckRequest(
            generation,
            (),
            (),
            (GenerationTaskRequirement(
                self.variant.task,
                self.variant.task_requirement,
                self.variant.task_result_key,
                goal.source,
                goal.scope,
                (*self.trace_prefix, self.variant.pattern_id),
            ),),
        )


__all__ = [
    "GroundedResponseActParserCatalog",
    "GroundedResponseActParserError",
    "GroundedResponseActParserGrammar",
    "GroundedResponseActParserProtocol",
    "GroundedResponseActPostcheckMapper",
    "GroundedResponseActSourceVerifier",
    "GroundedResponseActStructureVerifier",
    "GroundedResponseActSurfaceParser",
    "GroundedResponseActTaskVerifier",
    "build_grounded_response_act_parser_catalog",
]
