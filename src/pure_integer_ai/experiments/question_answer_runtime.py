"""F-00 QuestionRequest 到 G-04 的统一 typed 问答与生成编排。

runtime 只负责 route、查询执行、G-01 独立选择、G-00 至 G-03 执行和可选 G-04
复核。查询 adapter 必须返回真实 ``QuestionExecutionResult``；未注册 query kind
明确返回 unsupported，绝不回退 expected、teacher、held-out 或旧文本生成链。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSourceRequirement,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationCommitReport,
    MemoryGenerationOutcomeReport,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionExecutor,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
    GenerationPostcheckRuntime,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度前缀。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 runtime trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """核验 runtime 协议身份是一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


@dataclass(frozen=True)
class QuestionAnswerProtocol:
    """注入未注册 query kind 的 unsupported 状态、原因和编译原因。"""

    unsupported_status: ObjectIdentity
    unsupported_reason: ObjectIdentity
    compiled_reason: ObjectIdentity

    def __post_init__(self) -> None:
        """核验三个协议身份互异且均为一等最小指令。"""
        identities = (
            self.unsupported_status,
            self.unsupported_reason,
            self.compiled_reason,
        )
        if len(set(identities)) != len(identities):
            raise ValueError("question answer protocol 身份必须互不相同")
        for identity in identities:
            _require_instruction(identity, label="question answer protocol")


@dataclass(frozen=True)
class QuestionRouteRegistration:
    """把开放 query kind 显式绑定到 route 和唯一 typed executor。"""

    query_kind: ObjectIdentity
    route: ObjectIdentity
    executor: QuestionExecutor

    def __post_init__(self) -> None:
        """核验 query kind、route 和 executor 协议完整。"""
        _require_instruction(self.query_kind, label="question route query_kind")
        _require_instruction(self.route, label="question route route")
        if not hasattr(self.executor, "execute"):
            raise TypeError("question route executor 必须实现 execute")


class QuestionAnswerPostcheckMapper(Protocol):
    """为同次问题、查询结果和 generation execution 构造 G-04 请求。"""

    def build(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            generation: TypedGenerationExecution,
            ) -> GenerationPostcheckRequest:
        """返回绑定同次 execution 的来源化 G-04 请求。"""
        ...


class QuestionSelectionCommitter(Protocol):
    """在成功 surface 后提交 G-01 实际采用产生的外部状态。"""

    def commit(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            ) -> MemoryGenerationCommitReport:
        """提交同次已选项并返回可审计报告；不得处理未选或失败生成。"""
        ...


class QuestionOutcomeCommitter(Protocol):
    """在同次 G-04 后把分维结果提交到此前已形成的实际采用记录。"""

    def commit(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            selection_commit: MemoryGenerationCommitReport,
            postcheck: GenerationPostcheckRun,
            ) -> MemoryGenerationOutcomeReport:
        """返回绑定同次 Use 与 G-04 report 的分维 outcome 报告。"""
        ...


class EvidenceQuestionPostcheckMapper:
    """为无 Artifact/任务副作用的证据问答建立同次 G-04 来源要求。"""

    def __init__(
            self,
            trace_prefix: tuple[int, ...],
            *,
            citation_required: bool,
            trust_required: bool,
            ) -> None:
        """绑定来源核验强度和注入式 trace 前缀，不解释问题或关系类型。"""
        self.trace_prefix = _strict_key(
            trace_prefix,
            label="question postcheck trace prefix",
        )
        if (type(citation_required) is not bool
                or type(trust_required) is not bool):
            raise TypeError("question postcheck 来源要求必须是严格 bool")
        if not citation_required and not trust_required:
            raise ValueError("question postcheck 至少启用一种来源核验")
        self.citation_required = citation_required
        self.trust_required = trust_required

    def build(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            generation: TypedGenerationExecution,
            ) -> GenerationPostcheckRequest:
        """逐点覆盖同次 planned Proposition；空内容仍保留结构与 scope 复核。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("question postcheck request 类型错误")
        if not isinstance(query, QuestionQuery) or query.request != request:
            raise ValueError("question postcheck query 替换了原请求")
        if (not isinstance(result, QuestionExecutionResult)
                or result.query != query):
            raise ValueError("question postcheck result 替换了同次 query")
        if (not isinstance(generation, TypedGenerationExecution)
                or generation.plan.request != result.planning_request()):
            raise ValueError("question postcheck generation 替换了查询结果")
        planned = generation.surface.preview.request.structure
        candidates = {
            item.stable_key(): item
            for item in generation.plan.request.candidates
        }
        planned_keys = {
            item.candidate_key for item in planned.propositions.propositions}
        if set(candidates).intersection(planned_keys) != planned_keys:
            raise ValueError("question postcheck 无法恢复 planned candidate")
        requirements = tuple(
            GenerationSourceRequirement(
                item.candidate_key,
                item.source,
                item.scope,
                self.citation_required,
                self.trust_required,
                (*self.trace_prefix, index),
                candidates[item.candidate_key].citation_sources,
            )
            for index, item in enumerate(
                planned.propositions.propositions,
                start=1,
            )
        )
        return GenerationPostcheckRequest(
            generation,
            (),
            requirements,
        )


@dataclass(frozen=True)
class QuestionAnswerRun:
    """一次问题从 typed query 到回答选择、surface 和 G-04 的完整报告。"""

    request: QuestionRequest
    status: ObjectIdentity
    reason: ObjectIdentity
    trace: tuple[int, ...]
    query: QuestionQuery | None = None
    query_result: QuestionExecutionResult | None = None
    planning_request: GenerationPlanningRequest | None = None
    selection: AnswerContentSelection | None = None
    generation: TypedGenerationExecution | None = None
    postcheck: GenerationPostcheckRun | None = None
    selection_commit: MemoryGenerationCommitReport | None = None
    outcome_commit: MemoryGenerationOutcomeReport | None = None
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        """核验 unsupported 与完整执行两种报告形态不能混合或缺段。"""
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("question answer run request 类型错误")
        _require_instruction(self.status, label="question answer run status")
        _require_instruction(self.reason, label="question answer run reason")
        _strict_key(self.trace, label="question answer run trace")
        fields = (
            self.query,
            self.query_result,
            self.planning_request,
            self.selection,
            self.generation,
        )
        if self.query is None:
            if any(item is not None for item in fields[1:]):
                raise ValueError("unsupported question run 不得携带部分执行结果")
            if self.postcheck is not None:
                raise ValueError("unsupported question run 不得携带 postcheck")
            if self.selection_commit is not None:
                raise ValueError("unsupported question run 不得携带 selection commit")
            if self.outcome_commit is not None:
                raise ValueError("unsupported question run 不得携带 outcome commit")
            object.__setattr__(self, "_stable_key_cache", self._build_stable_key())
            return
        if any(item is None for item in fields):
            raise ValueError("已路由 question run 必须包含查询、选择和生成全段")
        if not isinstance(self.query, QuestionQuery):
            raise TypeError("question answer run query 类型错误")
        if not isinstance(self.query_result, QuestionExecutionResult):
            raise TypeError("question answer run query_result 类型错误")
        if self.query_result.query != self.query:
            raise ValueError("question answer run query_result 替换了 query")
        if not isinstance(self.planning_request, GenerationPlanningRequest):
            raise TypeError("question answer run planning_request 类型错误")
        if self.query_result.planning_request() != self.planning_request:
            raise ValueError("question answer run planning request 未来自查询结果")
        if not isinstance(self.selection, AnswerContentSelection):
            raise TypeError("question answer run selection 类型错误")
        if self.selection.request != self.planning_request:
            raise ValueError("question answer run selection 替换了 G-00 请求")
        if not isinstance(self.generation, TypedGenerationExecution):
            raise TypeError("question answer run generation 类型错误")
        if self.generation.plan.request != self.planning_request:
            raise ValueError("question answer run generation 替换了 G-00 请求")
        if self.status != self.selection.stance:
            raise ValueError("question answer run status 必须来自 G-01 stance")
        if self.reason != self.selection.reason:
            raise ValueError("question answer run reason 必须来自 G-01 decision")
        if self.selection_commit is not None:
            if not isinstance(
                    self.selection_commit, MemoryGenerationCommitReport):
                raise TypeError("question answer run selection_commit 类型错误")
            if self.selection_commit.selection_key != integer_tuple_fingerprint(
                    self.selection.stable_key(),
                    domain="question.commit.selection.v1"):
                raise ValueError("selection commit 替换了同次 G-01 selection")
            if self.selection_commit.generation_key != integer_tuple_fingerprint(
                    self.generation.stable_key(),
                    domain="question.commit.generation.v1"):
                raise ValueError("selection commit 替换了同次 generation")
        if self.postcheck is not None:
            if not isinstance(self.postcheck, GenerationPostcheckRun):
                raise TypeError("question answer run postcheck 类型错误")
            if self.postcheck.request.execution != self.generation:
                raise ValueError("question answer run postcheck 替换了同次 generation")
        if self.outcome_commit is not None:
            if not isinstance(
                    self.outcome_commit, MemoryGenerationOutcomeReport):
                raise TypeError("question answer run outcome_commit 类型错误")
            if self.selection_commit is None or self.postcheck is None:
                raise ValueError("question outcome 必须同时绑定 Use 与 G-04")
            if self.outcome_commit.selection_key != integer_tuple_fingerprint(
                    self.selection.stable_key(),
                    domain="question.commit.selection.v1"):
                raise ValueError("question outcome 替换了同次 G-01 selection")
            if self.outcome_commit.generation_key != integer_tuple_fingerprint(
                    self.generation.stable_key(),
                    domain="question.commit.generation.v1"):
                raise ValueError("question outcome 替换了同次 generation")
            if self.outcome_commit.postcheck_key != integer_tuple_fingerprint(
                    self.postcheck.stable_key(),
                    domain="memory.generation.outcome.postcheck.v1"):
                raise ValueError("question outcome 替换了同次 G-04 postcheck")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def complete(self) -> bool:
        """返回 G-00 至 G-03 完整且已安装的 G-04 也通过。"""
        if self.generation is None or not self.generation.complete:
            return False
        return self.postcheck is None or self.postcheck.complete

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、状态、trace 和全部 typed 执行段的内容引用键。"""
        if not self._stable_key_cache:
            raise RuntimeError("question answer stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在冻结构造完成时只计算一次完整问答结果键。"""
        result = [
            *_packed(integer_tuple_fingerprint(
                self.request.stable_key(), domain="question.run.request.v1")),
            *_packed(self.status.stable_key()),
            *_packed(self.reason.stable_key()),
            *_packed(integer_tuple_fingerprint(
                self.trace, domain="question.run.trace.v1")),
        ]
        for item in (
                self.query,
                self.query_result,
                self.planning_request,
                self.selection,
                self.generation,
                self.selection_commit,
                self.postcheck,
                self.outcome_commit):
            result.append(0 if item is None else 1)
            if item is not None:
                result.extend(_packed(integer_tuple_fingerprint(
                    item.stable_key(), domain="question.run.component.v1")))
        return tuple(result)


class QuestionAnswerRuntime:
    """编译 query kind，执行 typed owner，并复核 G-01 与 G-00 至 G-04 同次性。"""

    def __init__(
            self,
            protocol: QuestionAnswerProtocol,
            registrations: tuple[QuestionRouteRegistration, ...],
            selector: AnswerContentSelector,
            generator: TypedGenerationExecutor,
            *,
            selection_committer: QuestionSelectionCommitter | None = None,
            postcheck_mapper: QuestionAnswerPostcheckMapper | None = None,
            postchecker: GenerationPostcheckRuntime | None = None,
            outcome_committer: QuestionOutcomeCommitter | None = None,
            ) -> None:
        """绑定 query 路由、共享 G-01 selector、typed generator 和可选 G-04。"""
        if not isinstance(protocol, QuestionAnswerProtocol):
            raise TypeError("question answer protocol 类型错误")
        if (not isinstance(registrations, tuple)
                or any(not isinstance(item, QuestionRouteRegistration)
                       for item in registrations)):
            raise TypeError("question answer registrations 类型错误")
        query_kinds = tuple(item.query_kind for item in registrations)
        routes = tuple(item.route for item in registrations)
        if len(set(query_kinds)) != len(query_kinds):
            raise ValueError("question answer query kind 不得重复注册")
        if len(set(routes)) != len(routes):
            raise ValueError("question answer route 不得重复注册")
        if not isinstance(selector, AnswerContentSelector):
            raise TypeError("question answer selector 类型错误")
        if not isinstance(generator, TypedGenerationExecutor):
            raise TypeError("question answer generator 类型错误")
        if (selection_committer is not None
                and not hasattr(selection_committer, "commit")):
            raise TypeError("question answer selection committer 缺少 commit")
        if (postcheck_mapper is None) != (postchecker is None):
            raise ValueError("question answer G-04 mapper/runtime 必须成对安装")
        if (postcheck_mapper is not None
                and not hasattr(postcheck_mapper, "build")):
            raise TypeError("question answer postcheck mapper 缺少 build")
        if (postchecker is not None
                and not isinstance(postchecker, GenerationPostcheckRuntime)):
            raise TypeError("question answer postchecker 类型错误")
        if outcome_committer is not None:
            if not hasattr(outcome_committer, "commit"):
                raise TypeError("question answer outcome committer 缺少 commit")
            if selection_committer is None or postchecker is None:
                raise ValueError("question outcome committer 必须同时安装 Use 与 G-04")
        self.protocol = protocol
        self._registrations = {
            item.query_kind: item for item in registrations
        }
        self.selector = selector
        self.generator = generator
        self.selection_committer = selection_committer
        self.postcheck_mapper = postcheck_mapper
        self.postchecker = postchecker
        self.outcome_committer = outcome_committer

    def run(self, request: QuestionRequest) -> QuestionAnswerRun:
        """执行一条真实问答纵切；缺 route 明确 unsupported，绝不调用生成 fallback。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("question answer runtime 需要 QuestionRequest")
        registration = self._registrations.get(request.query_kind)
        if registration is None:
            trace = (
                1,
                *_packed(request.stable_key()),
                *_packed(self.protocol.unsupported_status.stable_key()),
            )
            return QuestionAnswerRun(
                request,
                self.protocol.unsupported_status,
                self.protocol.unsupported_reason,
                trace,
            )
        query_trace = (
            1,
            *_packed(request.trace),
            *_packed(self.protocol.compiled_reason.stable_key()),
            *_packed(registration.route.stable_key()),
        )
        query = QuestionQuery(request, registration.route, query_trace)
        result = registration.executor.execute(query)
        if not isinstance(result, QuestionExecutionResult):
            raise TypeError("question executor 必须返回 QuestionExecutionResult")
        if result.query != query:
            raise ValueError("question executor 替换了 compiled query")
        planning = result.planning_request()
        selection = self.selector.select(planning)
        generation = self.generator.execute(planning)
        if len(generation.plan.layers) < 2:
            raise ValueError("typed generation 缺少 stance/content 两层")
        selection_key = selection.stable_key()
        stance_layer, content_layer = generation.plan.layers[:2]
        if (not stance_layer.executed or not content_layer.executed
                or stance_layer.payload != selection_key
                or content_layer.payload != selection_key):
            raise ValueError("question G-01 选择与 generation stance/content 不一致")
        selection_commit = None
        if self.selection_committer is not None and generation.complete:
            selection_commit = self.selection_committer.commit(
                request, query, result, selection, generation)
            if not isinstance(selection_commit, MemoryGenerationCommitReport):
                raise TypeError("question selection committer 返回类型错误")
        postcheck = None
        if self.postchecker is not None and generation.complete:
            postcheck_request = self.postcheck_mapper.build(
                request, query, result, generation)
            if not isinstance(postcheck_request, GenerationPostcheckRequest):
                raise TypeError("question postcheck mapper 返回类型错误")
            if postcheck_request.execution != generation:
                raise ValueError("question postcheck request 替换了 generation")
            postcheck = self.postchecker.run(postcheck_request)
        outcome_commit = None
        if self.outcome_committer is not None:
            if selection_commit is None or postcheck is None:
                raise RuntimeError("question outcome 缺少同次 Use 或 G-04")
            outcome_commit = self.outcome_committer.commit(
                request,
                query,
                result,
                selection,
                generation,
                selection_commit,
                postcheck,
            )
            if not isinstance(
                    outcome_commit, MemoryGenerationOutcomeReport):
                raise TypeError("question outcome committer 返回类型错误")
        trace = (
            2,
            *_packed(integer_tuple_fingerprint(
                query.stable_key(), domain="question.trace.query.v1")),
            *_packed(integer_tuple_fingerprint(
                result.stable_key(), domain="question.trace.result.v1")),
            *_packed(integer_tuple_fingerprint(
                selection.stable_key(), domain="question.trace.selection.v1")),
            *_packed(integer_tuple_fingerprint(
                generation.stable_key(), domain="question.trace.generation.v1")),
            *_packed(
                () if selection_commit is None
                else integer_tuple_fingerprint(
                    selection_commit.stable_key(),
                    domain="question.trace.commit.v1")),
            *_packed(
                () if postcheck is None else integer_tuple_fingerprint(
                    postcheck.stable_key(),
                    domain="question.trace.postcheck.v1")),
            *_packed(
                () if outcome_commit is None else integer_tuple_fingerprint(
                    outcome_commit.stable_key(),
                    domain="question.trace.outcome.v1")),
        )
        return QuestionAnswerRun(
            request,
            selection.stance,
            selection.reason,
            trace,
            query,
            result,
            planning,
            selection,
            generation,
            postcheck=postcheck,
            selection_commit=selection_commit,
            outcome_commit=outcome_commit,
        )


__all__ = [
    "EvidenceQuestionPostcheckMapper",
    "QuestionAnswerPostcheckMapper",
    "QuestionAnswerProtocol",
    "QuestionAnswerRun",
    "QuestionAnswerRuntime",
    "QuestionOutcomeCommitter",
    "QuestionRouteRegistration",
    "QuestionSelectionCommitter",
]
