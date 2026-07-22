"""G-00 分层生成计划对象和纯运行期调度入口。

本模块只接收 typed 回答目标、候选 Evidence 和 S-05 reasoning trace。六层身份、
完成/失败/阻断状态及原因均由调用方以一等 MinimalInstruction 注入；规划过程不读取
PR、路径、salience、surface 名称或旧生成序列，也不写图、Memory 或 WorkMemory。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningPlanResult
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界，避免相邻字段发生拼接碰撞。"""
    return len(key), *key


def _strict_int_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放整数 tuple，拒绝 bool、字符串和浮点混入计划 trace。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验元定义身份是 MinimalInstruction，不从名称或数值猜作用。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


@dataclass(frozen=True)
class AnswerGenerationGoal:
    """当前 query 的 typed 回答目标及所需 support/refute 证据方向。"""

    goal_kind: ObjectIdentity
    proposition: BoundProposition
    required: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    target_branch: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        _require_instruction(self.goal_kind, label="generation goal kind")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("generation goal proposition 必须是 BoundProposition")
        if not isinstance(self.required, LogicEvidenceState):
            raise TypeError("generation goal required 必须是 LogicEvidenceState")
        if not self.required.support and not self.required.refute:
            raise ValueError("generation goal 至少要求一个证据方向")
        if not isinstance(self.source, SourceRef):
            raise TypeError("generation goal source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("generation goal scope 必须是 ScopeIdentity")
        if semantic_source(self.proposition.template) != self.source:
            raise ValueError("generation goal Proposition 与 source 不一致")
        if self.target_branch is not None:
            if not isinstance(self.target_branch, ObjectIdentity):
                raise TypeError("generation goal target_branch 类型错误")
            if self.target_branch.object_kind != OBJECT_LANGUAGE_BRANCH:
                raise ValueError("generation goal target_branch 必须是 LanguageBranch")

    def stable_key(self) -> tuple[int, ...]:
        """返回目标、证据方向、归属和可选目标语言分支完整键。"""
        result = [
            *_packed(self.goal_kind.stable_key()),
            *_packed(self.proposition.stable_key()),
            *self.required.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            0 if self.target_branch is None else 1,
        ]
        if self.target_branch is not None:
            result.extend(_packed(self.target_branch.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class GenerationCandidate:
    """可供内容层采用的命题、当前四态、完整 Evidence 和可选 reasoning。"""

    proposition: BoundProposition
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence: tuple[EvidenceRecord, ...]
    reasoning: ReasoningPlanResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("generation candidate proposition 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("generation candidate state 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("generation candidate source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("generation candidate scope 类型错误")
        if semantic_source(self.proposition.template) != self.source:
            raise ValueError("generation candidate Proposition 与 source 不一致")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("generation candidate 必须携带非空 Evidence tuple")
        if any(not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("generation candidate evidence 含非法项")
        evidence = tuple(sorted(self.evidence, key=lambda item: item.evidence_id))
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("generation candidate Evidence id 不得重复")
        superseded_ids = {
            item.supersedes_evidence_id
            for item in evidence
            if item.supersedes_evidence_id != 0
        }
        if superseded_ids.intersection(evidence_ids):
            raise ValueError("generation candidate 不得同时携带已被替代的 Evidence")
        for item in evidence:
            if item.hypothesis.observation != self.source:
                raise ValueError(
                    "generation candidate Evidence Hypothesis 与 Proposition 来源不一致")
        derived = LogicEvidenceState(
            any(item.stance == EVIDENCE_SUPPORT for item in evidence),
            any(item.stance == EVIDENCE_REFUTE for item in evidence),
        )
        if derived != self.state:
            raise ValueError("generation candidate 四态与所携 Evidence 不一致")
        if self.reasoning is not None:
            if not isinstance(self.reasoning, ReasoningPlanResult):
                raise TypeError("generation candidate reasoning 类型错误")
            obligation = self.reasoning.obligation
            if obligation.proposition != self.proposition:
                raise ValueError("generation candidate reasoning 目标命题不一致")
            if obligation.source != self.source or obligation.scope != self.scope:
                raise ValueError("generation candidate reasoning source/scope 不一致")
            if not set(self.reasoning.evidence_ids).issubset(set(evidence_ids)):
                raise ValueError("generation candidate 未携带 reasoning 引用的全部 Evidence")
        object.__setattr__(self, "evidence", evidence)

    @property
    def hypotheses(self) -> tuple:
        """返回 Evidence 中实际出现的完整 Hypothesis 集，按稳定键排序去重。"""
        return tuple(sorted(
            {item.hypothesis for item in self.evidence},
            key=lambda item: item.stable_key(),
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回命题、状态、来源、scope、Evidence 和 reasoning 的完整候选键。"""
        result = [
            *_packed(self.proposition.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self.evidence),
        ]
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        reasoning_key = () if self.reasoning is None else self.reasoning.stable_key()
        result.extend(_packed(reasoning_key))
        return tuple(result)


@dataclass(frozen=True)
class GenerationPlanningRequest:
    """一次生成规划的回答目标和无序 typed 候选集合。"""

    goal: AnswerGenerationGoal
    candidates: tuple[GenerationCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.goal, AnswerGenerationGoal):
            raise TypeError("generation request goal 类型错误")
        if not isinstance(self.candidates, tuple):
            raise TypeError("generation request candidates 必须是 tuple")
        if any(not isinstance(item, GenerationCandidate) for item in self.candidates):
            raise TypeError("generation request candidates 含非法项")
        for item in self.candidates:
            if item.scope != self.goal.scope:
                raise ValueError("generation candidate 必须绑定当前 query scope")
        candidates = tuple(sorted(
            self.candidates, key=lambda item: item.stable_key()))
        keys = tuple(item.stable_key() for item in candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("generation request 不得重复提交同一候选")
        object.__setattr__(self, "candidates", candidates)

    def candidate_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回排序后的完整候选键，供逐层采用归因和主动核验。"""
        return tuple(item.stable_key() for item in self.candidates)

    def stable_key(self) -> tuple[int, ...]:
        """返回目标和全部候选的确定性请求键，不接受裸 PR 或路径输入。"""
        result = [*_packed(self.goal.stable_key()), len(self.candidates)]
        for item in self.candidates:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class GenerationPlanProtocol:
    """注入六层身份、结果状态和下游阻断原因。"""

    stance_layer: ObjectIdentity
    content_layer: ObjectIdentity
    discourse_layer: ObjectIdentity
    proposition_layer: ObjectIdentity
    syntax_layer: ObjectIdentity
    surface_layer: ObjectIdentity
    complete: ObjectIdentity
    failed: ObjectIdentity
    blocked: ObjectIdentity
    downstream_blocked: ObjectIdentity

    def __post_init__(self) -> None:
        identities = self.layers() + self.outcomes() + (self.downstream_blocked,)
        if len(set(identities)) != len(identities):
            raise ValueError("generation plan 协议身份必须互不相同")
        for identity in identities:
            _require_instruction(identity, label="generation plan protocol")

    def layers(self) -> tuple[ObjectIdentity, ...]:
        """返回任务定义的六层顺序，具体图身份完全由调用方注入。"""
        return (
            self.stance_layer,
            self.content_layer,
            self.discourse_layer,
            self.proposition_layer,
            self.syntax_layer,
            self.surface_layer,
        )

    def outcomes(self) -> tuple[ObjectIdentity, ...]:
        """返回可由 planner 解释的 complete、failed 和 blocked 身份。"""
        return self.complete, self.failed, self.blocked

    def stable_key(self) -> tuple[int, ...]:
        """返回全部层、状态和阻断原因的一等身份键。"""
        result: list[int] = []
        for identity in self.layers() + self.outcomes() + (self.downstream_blocked,):
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class GenerationLayerDecision:
    """一个已执行 layer resolver 返回的采用、载荷和完整 trace。"""

    layer: ObjectIdentity
    outcome: ObjectIdentity
    reason: ObjectIdentity
    selected_candidate_keys: tuple[tuple[int, ...], ...] = ()
    payload: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()
    artifact: Any = None

    def __post_init__(self) -> None:
        _require_instruction(self.layer, label="generation decision layer")
        _require_instruction(self.outcome, label="generation decision outcome")
        _require_instruction(self.reason, label="generation decision reason")
        if not isinstance(self.selected_candidate_keys, tuple):
            raise TypeError("selected_candidate_keys 必须是 tuple")
        for key in self.selected_candidate_keys:
            _strict_int_tuple(key, label="selected candidate key")
            if not key:
                raise ValueError("selected candidate key 不能为空")
        if len(set(self.selected_candidate_keys)) != len(self.selected_candidate_keys):
            raise ValueError("selected candidate key 不得重复")
        _strict_int_tuple(self.payload, label="generation decision payload")
        _strict_int_tuple(self.trace, label="generation decision trace")
        if not self.trace:
            raise ValueError("已执行 generation layer 必须携带非空 trace")
        if self.artifact is not None:
            stable_key = getattr(self.artifact, "stable_key", None)
            if not callable(stable_key):
                raise TypeError("generation decision artifact 必须提供 stable_key")
            artifact_key = stable_key()
            _strict_int_tuple(
                artifact_key, label="generation decision artifact key")
            if artifact_key != self.payload:
                raise ValueError("generation decision artifact 与 payload 不一致")
        object.__setattr__(self, "selected_candidate_keys", tuple(sorted(
            self.selected_candidate_keys)))


class GenerationLayerResolver(Protocol):
    """单层 planner resolver；它不能读写其他层的隐藏状态。"""

    def resolve(
            self,
            request: GenerationPlanningRequest,
            prior: tuple["GenerationLayerResult", ...],
            ) -> GenerationLayerDecision:
        """根据完整请求和已提交上游结果返回当前层 decision。"""
        ...


@dataclass(frozen=True)
class GenerationLayerRegistration:
    """把一个注入 layer identity 显式绑定到唯一 resolver。"""

    layer: ObjectIdentity
    resolver: GenerationLayerResolver

    def __post_init__(self) -> None:
        _require_instruction(self.layer, label="generation registration layer")
        if not hasattr(self.resolver, "resolve"):
            raise TypeError("generation layer resolver 必须实现 resolve")


@dataclass(frozen=True)
class GenerationLayerResult:
    """一个层的输入指纹、执行状态、采用候选、载荷和归因 trace。"""

    layer: ObjectIdentity
    outcome: ObjectIdentity
    reason: ObjectIdentity
    executed: bool
    input_key: tuple[int, ...]
    selected_candidate_keys: tuple[tuple[int, ...], ...] = ()
    payload: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()
    artifact: Any = None

    def __post_init__(self) -> None:
        _require_instruction(self.layer, label="generation result layer")
        _require_instruction(self.outcome, label="generation result outcome")
        _require_instruction(self.reason, label="generation result reason")
        if type(self.executed) is not bool:
            raise TypeError("generation result executed 必须是严格 bool")
        _strict_int_tuple(self.input_key, label="generation result input_key")
        if not self.input_key:
            raise ValueError("generation result input_key 不能为空")
        if not isinstance(self.selected_candidate_keys, tuple):
            raise TypeError("generation result candidate keys 必须是 tuple")
        for key in self.selected_candidate_keys:
            _strict_int_tuple(key, label="generation result candidate key")
            if not key:
                raise ValueError("generation result candidate key 不能为空")
        if len(set(self.selected_candidate_keys)) != len(self.selected_candidate_keys):
            raise ValueError("generation result candidate key 不得重复")
        _strict_int_tuple(self.payload, label="generation result payload")
        _strict_int_tuple(self.trace, label="generation result trace")
        if not self.trace:
            raise ValueError("generation result 必须保存非空 trace")
        if self.artifact is not None:
            stable_key = getattr(self.artifact, "stable_key", None)
            if not callable(stable_key):
                raise TypeError("generation result artifact 必须提供 stable_key")
            artifact_key = stable_key()
            _strict_int_tuple(
                artifact_key, label="generation result artifact key")
            if artifact_key != self.payload:
                raise ValueError("generation result artifact 与 payload 不一致")
        if not self.executed and self.artifact is not None:
            raise ValueError("未执行 generation layer 不得携带 artifact")
        object.__setattr__(self, "selected_candidate_keys", tuple(sorted(
            self.selected_candidate_keys)))

    def stable_key(self) -> tuple[int, ...]:
        """返回层身份、结果、输入、采用候选、payload 和 trace 的完整键。"""
        result = [
            *_packed(self.layer.stable_key()),
            *_packed(self.outcome.stable_key()),
            *_packed(self.reason.stable_key()),
            int(self.executed),
            *_packed(self.input_key),
            len(self.selected_candidate_keys),
        ]
        for key in self.selected_candidate_keys:
            result.extend(_packed(key))
        result.extend(_packed(self.payload))
        result.extend(_packed(self.trace))
        return tuple(result)


@dataclass(frozen=True)
class GenerationPlan:
    """六层逐项可审计的纯运行期计划；它不是 Core 或 Memory 对象。"""

    request: GenerationPlanningRequest
    protocol: GenerationPlanProtocol
    layers: tuple[GenerationLayerResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationPlanningRequest):
            raise TypeError("generation plan request 类型错误")
        if not isinstance(self.protocol, GenerationPlanProtocol):
            raise TypeError("generation plan protocol 类型错误")
        if not isinstance(self.layers, tuple):
            raise TypeError("generation plan layers 必须是 tuple")
        if any(not isinstance(item, GenerationLayerResult) for item in self.layers):
            raise TypeError("generation plan layers 含非法项")
        expected = self.protocol.layers()
        if tuple(item.layer for item in self.layers) != expected:
            raise ValueError("generation plan 必须完整保持六层协议顺序")
        halted = False
        for item in self.layers:
            if halted:
                if item.executed or item.outcome != self.protocol.blocked:
                    raise ValueError("失败后的 generation layer 必须显式 blocked 且不执行")
            elif item.outcome == self.protocol.complete:
                if not item.executed:
                    raise ValueError("complete generation layer 必须真实执行")
            elif item.outcome == self.protocol.failed:
                if not item.executed:
                    raise ValueError("failed generation layer 必须真实执行")
                halted = True
            else:
                raise ValueError("首个非完成 generation layer 只能是 failed")

    @property
    def complete(self) -> bool:
        """返回六层是否全部真实执行并以注入 complete outcome 收口。"""
        return all(
            item.executed and item.outcome == self.protocol.complete
            for item in self.layers
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、协议和六层结果的完整确定性回放键。"""
        result = [
            *_packed(self.request.stable_key()),
            *_packed(self.protocol.stable_key()),
            len(self.layers),
        ]
        for item in self.layers:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


def _input_key(
        request: GenerationPlanningRequest,
        prior: tuple[GenerationLayerResult, ...],
        ) -> tuple[int, ...]:
    """冻结当前层可见的完整请求和全部已提交上游结果。"""
    result = [*_packed(request.stable_key()), len(prior)]
    for item in prior:
        result.extend(_packed(item.stable_key()))
    return tuple(result)


class GenerationPlanner:
    """按六层注册顺序执行 resolver，并对失败后的下游层显式 fail closed。"""

    def __init__(
            self,
            protocol: GenerationPlanProtocol,
            registrations: Sequence[GenerationLayerRegistration],
            ) -> None:
        if not isinstance(protocol, GenerationPlanProtocol):
            raise TypeError("generation planner protocol 类型错误")
        if not isinstance(registrations, Sequence):
            raise TypeError("generation registrations 必须是 Sequence")
        registration_items = tuple(registrations)
        if any(not isinstance(item, GenerationLayerRegistration)
               for item in registration_items):
            raise TypeError("generation registrations 含非法项")
        by_layer: dict[ObjectIdentity, GenerationLayerRegistration] = {}
        for item in registration_items:
            if item.layer in by_layer:
                raise ValueError("同一 generation layer 不得重复注册")
            by_layer[item.layer] = item
        expected = set(protocol.layers())
        if set(by_layer) != expected:
            raise ValueError("generation planner 必须且只能注册协议中的六层")
        self._protocol = protocol
        self._registrations = tuple(by_layer[layer] for layer in protocol.layers())

    def plan(self, request: GenerationPlanningRequest) -> GenerationPlan:
        """建立一次局部计划；planner 不保留跨调用的失败和采用记录。"""
        if not isinstance(request, GenerationPlanningRequest):
            raise TypeError("generation planner 只接受 GenerationPlanningRequest")
        candidate_keys = set(request.candidate_keys())
        results: list[GenerationLayerResult] = []
        halted = False
        blocking_result: GenerationLayerResult | None = None
        for registration in self._registrations:
            prior = tuple(results)
            input_key = _input_key(request, prior)
            if halted:
                if blocking_result is None:
                    raise RuntimeError("generation blocked 状态缺少首个 failed 结果")
                result = GenerationLayerResult(
                    registration.layer,
                    self._protocol.blocked,
                    self._protocol.downstream_blocked,
                    False,
                    input_key,
                    trace=_packed(blocking_result.stable_key()),
                )
                results.append(result)
                continue
            decision = registration.resolver.resolve(request, prior)
            if not isinstance(decision, GenerationLayerDecision):
                raise TypeError("generation resolver 必须返回 GenerationLayerDecision")
            if decision.layer != registration.layer:
                raise ValueError("generation resolver 返回了其他 layer")
            if decision.outcome not in (
                    self._protocol.complete, self._protocol.failed):
                raise ValueError("generation resolver 只能返回 complete 或 failed")
            if any(key not in candidate_keys
                   for key in decision.selected_candidate_keys):
                raise ValueError("generation resolver 采用了请求之外的候选")
            result = GenerationLayerResult(
                decision.layer,
                decision.outcome,
                decision.reason,
                True,
                input_key,
                decision.selected_candidate_keys,
                decision.payload,
                decision.trace,
                decision.artifact,
            )
            results.append(result)
            halted = decision.outcome == self._protocol.failed
            if halted:
                blocking_result = result
        return GenerationPlan(request, self._protocol, tuple(results))


__all__ = [
    "AnswerGenerationGoal",
    "GenerationCandidate",
    "GenerationLayerDecision",
    "GenerationLayerRegistration",
    "GenerationLayerResolver",
    "GenerationLayerResult",
    "GenerationPlan",
    "GenerationPlanProtocol",
    "GenerationPlanner",
    "GenerationPlanningRequest",
]
