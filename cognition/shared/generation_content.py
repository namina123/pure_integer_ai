"""G-01 回答立场、内容选择和 G-00 前两层 resolver。

policy 可以提出 answer、clarify、unknown、refuse 或 conflict，但共享 selector 会按
候选 Evidence 四态执行不可绕过的约束。S-06 Artifact 只作为来源化内容附件；形式
执行成功不能替代语言 Proposition 的独立 support Evidence。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactInvocationResult,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerDecision,
    GenerationLayerResult,
    GenerationPlanProtocol,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_int_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放整数 tuple，拒绝 bool、字符串和浮点混入 trace。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验立场、原因和层身份均为注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _optional_identity_key(identity: ObjectIdentity | None) -> tuple[int, ...]:
    """把可选一等身份编码为带空值边界的稳定键。"""
    return () if identity is None else identity.stable_key()


def _artifact_result_key(result: ArtifactInvocationResult) -> tuple[int, ...]:
    """把 S-06 完整调用、执行、验证、值、proof 和失败 trace 编入采用键。"""
    if not isinstance(result, ArtifactInvocationResult):
        raise TypeError("artifact result 必须是 ArtifactInvocationResult")
    values: list[int] = [*_packed(result.invocation.stable_key())]
    values.append(len(result.bound_arguments))
    for bound in result.bound_arguments:
        values.extend(_packed(bound.parameter.stable_key()))
        values.extend(_packed(bound.argument.parameter.stable_key()))
        values.extend(_packed(bound.argument.value.stable_key()))
        values.append(len(bound.type_support))
        for identity in bound.type_support:
            values.extend(_packed(identity.stable_key()))
        values.append(len(bound.unit_support))
        for identity in bound.unit_support:
            values.extend(_packed(identity.stable_key()))
        values.extend(_packed(bound.unit_adapter_payload))

    execution = result.execution
    values.append(0 if execution is None else 1)
    if execution is not None:
        values.extend(_packed(execution.authority.stable_key()))
        values.extend(_packed(execution.source.stable_key()))
        values.extend(_packed(execution.scope.stable_key()))
        values.append(int(execution.executed))
        values.extend(_packed(execution.output_payload))
        values.extend(_packed(execution.trace))
        values.extend(_packed(_optional_identity_key(execution.failure_reason)))

    verification = result.verification
    values.append(0 if verification is None else 1)
    if verification is not None:
        values.extend(_packed(verification.authority.stable_key()))
        values.extend(_packed(verification.source.stable_key()))
        values.extend(_packed(verification.scope.stable_key()))
        accepted = 0 if verification.accepted is None else (
            2 if verification.accepted else 1)
        values.append(accepted)
        values.extend(_packed(verification.payload))
        values.extend(_packed(verification.trace))
        values.extend(_packed(_optional_identity_key(
            verification.failure_reason)))

    for artifact in (result.value, result.proof):
        values.extend(_packed(
            () if artifact is None else artifact.stable_key()))
    values.extend(result.proposition_state.stable_key())
    values.append(len(result.failures))
    for failure in result.failures:
        values.extend(_packed(failure.reason.stable_key()))
        values.extend(_packed(failure.proposition.stable_key()))
        values.extend(_packed(_optional_identity_key(failure.parameter)))
        values.extend(_packed(_optional_identity_key(failure.expected)))
        values.extend(_packed(_optional_identity_key(failure.actual)))
        values.extend(_packed(_optional_identity_key(
            failure.upstream_reason)))
        values.extend(_packed(failure.details))
    return tuple(values)


@dataclass(frozen=True)
class AnswerContentProtocol:
    """注入 answer、clarify、unknown、refuse 和 conflict 五种立场身份。"""

    answer: ObjectIdentity
    clarify: ObjectIdentity
    unknown: ObjectIdentity
    refuse: ObjectIdentity
    conflict: ObjectIdentity

    def __post_init__(self) -> None:
        stances = self.stances()
        if len(set(stances)) != len(stances):
            raise ValueError("answer content stance 必须互不相同")
        for stance in stances:
            _require_instruction(stance, label="answer content stance")

    def stances(self) -> tuple[ObjectIdentity, ...]:
        """返回全部开放 stance identity，具体整数值不由宿主解释。"""
        return self.answer, self.clarify, self.unknown, self.refuse, self.conflict

    def stable_key(self) -> tuple[int, ...]:
        """返回五种立场的一等身份键。"""
        result: list[int] = []
        for stance in self.stances():
            result.extend(_packed(stance.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ContentArtifactAttachment:
    """把一个 S-06 调用结果显式归属到 G-00 候选键。"""

    candidate_key: tuple[int, ...]
    result: ArtifactInvocationResult

    def __post_init__(self) -> None:
        _strict_int_tuple(self.candidate_key, label="artifact candidate key")
        if not self.candidate_key:
            raise ValueError("artifact candidate key 不能为空")
        if not isinstance(self.result, ArtifactInvocationResult):
            raise TypeError("artifact attachment result 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选归属和完整 S-06 结果键。"""
        return *_packed(self.candidate_key), *_packed(
            _artifact_result_key(self.result))


@dataclass(frozen=True)
class AnswerContentDecision:
    """policy 提出的立场、原因、候选、Artifact 和非空审计 trace。"""

    stance: ObjectIdentity
    reason: ObjectIdentity
    selected_candidate_keys: tuple[tuple[int, ...], ...] = ()
    selected_artifact_keys: tuple[tuple[int, ...], ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_instruction(self.stance, label="answer content decision stance")
        _require_instruction(self.reason, label="answer content decision reason")
        for label, keys in (
                ("candidate", self.selected_candidate_keys),
                ("artifact", self.selected_artifact_keys)):
            if not isinstance(keys, tuple):
                raise TypeError(f"selected {label} keys 必须是 tuple")
            for key in keys:
                _strict_int_tuple(key, label=f"selected {label} key")
                if not key:
                    raise ValueError(f"selected {label} key 不能为空")
            if len(set(keys)) != len(keys):
                raise ValueError(f"selected {label} key 不得重复")
            object.__setattr__(self, f"selected_{label}_keys", tuple(sorted(keys)))
        _strict_int_tuple(self.trace, label="answer content decision trace")
        if not self.trace:
            raise ValueError("answer content decision 必须携带非空 trace")


class AnswerContentPolicy(Protocol):
    """按任务策略提出立场和内容集合，不拥有最终真值裁决权。"""

    def select(
            self,
            request: GenerationPlanningRequest,
            artifacts: tuple[ContentArtifactAttachment, ...],
            ) -> AnswerContentDecision:
        """基于 typed 请求和 Artifact 附件提出选择，供共享 selector 核验。"""
        ...


@dataclass(frozen=True)
class AnswerContentSelection:
    """经共享不变量核验后的回答立场和采用归因。"""

    request: GenerationPlanningRequest
    protocol: AnswerContentProtocol
    stance: ObjectIdentity
    reason: ObjectIdentity
    selected_candidate_keys: tuple[tuple[int, ...], ...]
    selected_artifact_keys: tuple[tuple[int, ...], ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationPlanningRequest):
            raise TypeError("answer content selection request 类型错误")
        if not isinstance(self.protocol, AnswerContentProtocol):
            raise TypeError("answer content selection protocol 类型错误")
        if self.stance not in self.protocol.stances():
            raise ValueError("answer content selection stance 未注册")
        _require_instruction(self.reason, label="answer content selection reason")
        for label, keys in (
                ("candidate", self.selected_candidate_keys),
                ("artifact", self.selected_artifact_keys)):
            if not isinstance(keys, tuple):
                raise TypeError(f"selection {label} keys 必须是 tuple")
            for key in keys:
                _strict_int_tuple(key, label=f"selection {label} key")
                if not key:
                    raise ValueError(f"selection {label} key 不能为空")
            if len(set(keys)) != len(keys):
                raise ValueError(f"selection {label} key 不得重复")
            object.__setattr__(self, f"selected_{label}_keys", tuple(sorted(keys)))
        request_keys = set(self.request.candidate_keys())
        if any(key not in request_keys for key in self.selected_candidate_keys):
            raise ValueError("answer content selection 含请求外候选")
        _strict_int_tuple(self.trace, label="answer content selection trace")
        if not self.trace:
            raise ValueError("answer content selection trace 不能为空")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、协议、立场、采用项和 policy trace 的完整键。"""
        result = [
            *_packed(self.request.stable_key()),
            *_packed(self.protocol.stable_key()),
            *_packed(self.stance.stable_key()),
            *_packed(self.reason.stable_key()),
            len(self.selected_candidate_keys),
        ]
        for key in self.selected_candidate_keys:
            result.extend(_packed(key))
        result.append(len(self.selected_artifact_keys))
        for key in self.selected_artifact_keys:
            result.extend(_packed(key))
        result.extend(_packed(self.trace))
        return tuple(result)


def _satisfies(state: LogicEvidenceState, required: LogicEvidenceState) -> bool:
    """判断候选四态是否覆盖回答目标要求的全部证据位。"""
    return (
        (not required.support or state.support)
        and (not required.refute or state.refute)
    )


def _group_states(candidates) -> dict[tuple[int, ...], LogicEvidenceState]:
    """按完整 BoundProposition 聚合 support/refute，避免跨命题伪造冲突。"""
    grouped: dict[tuple[int, ...], LogicEvidenceState] = {}
    for candidate in candidates:
        key = candidate.proposition.stable_key()
        previous = grouped.get(key, LogicEvidenceState(False, False))
        grouped[key] = LogicEvidenceState(
            previous.support or candidate.state.support,
            previous.refute or candidate.state.refute,
        )
    return grouped


def _ambiguous_competitions(
        candidates,
        required: LogicEvidenceState,
        ) -> tuple[frozenset[tuple[int, ...]], ...]:
    """找出同 competition+scope 下多个可回答命题，禁止按稳定序私选。"""
    grouped: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for candidate in candidates:
        if (not _satisfies(candidate.state, required)
                or (candidate.state.support and candidate.state.refute)):
            continue
        proposition_key = candidate.proposition.stable_key()
        for hypothesis in candidate.hypotheses:
            competition = (
                *_packed(hypothesis.competition_key),
                *_packed(hypothesis.scope.stable_key()),
            )
            grouped.setdefault(competition, set()).add(proposition_key)
    return tuple(sorted(
        (frozenset(values) for values in grouped.values() if len(values) > 1),
        key=lambda values: tuple(sorted(values)),
    ))


class AnswerContentSelector:
    """执行五种 stance 的 Evidence、歧义、冲突和 Artifact 采用不变量。"""

    def __init__(
            self,
            protocol: AnswerContentProtocol,
            policy: AnswerContentPolicy,
            ) -> None:
        if not isinstance(protocol, AnswerContentProtocol):
            raise TypeError("answer content protocol 类型错误")
        if not hasattr(policy, "select"):
            raise TypeError("answer content policy 必须实现 select")
        self._protocol = protocol
        self._policy = policy

    def select(
            self,
            request: GenerationPlanningRequest,
            artifacts: Sequence[ContentArtifactAttachment] = (),
            ) -> AnswerContentSelection:
        """核验 policy 选择，不允许 Artifact、排序或空模板绕过 Evidence 四态。"""
        if not isinstance(request, GenerationPlanningRequest):
            raise TypeError("answer content selector 只接受 GenerationPlanningRequest")
        if not isinstance(artifacts, Sequence):
            raise TypeError("answer content artifacts 必须是 Sequence")
        artifact_items = tuple(artifacts)
        if any(not isinstance(item, ContentArtifactAttachment)
               for item in artifact_items):
            raise TypeError("answer content artifacts 含非法项")
        candidates = {item.stable_key(): item for item in request.candidates}
        artifact_map: dict[tuple[int, ...], ContentArtifactAttachment] = {}
        for attachment in artifact_items:
            candidate = candidates.get(attachment.candidate_key)
            if candidate is None:
                raise ValueError("Artifact attachment 指向请求外候选")
            invocation = attachment.result.invocation
            if invocation.proposition != candidate.proposition.template:
                raise ValueError("Artifact attachment Proposition 与候选不一致")
            if invocation.source != candidate.source or invocation.scope != candidate.scope:
                raise ValueError("Artifact attachment source/scope 与候选不一致")
            key = attachment.stable_key()
            if key in artifact_map:
                raise ValueError("Artifact attachment 不得重复")
            artifact_map[key] = attachment
        normalized_artifacts = tuple(
            artifact_map[key] for key in sorted(artifact_map))
        decision = self._policy.select(request, normalized_artifacts)
        if not isinstance(decision, AnswerContentDecision):
            raise TypeError("answer content policy 必须返回 AnswerContentDecision")
        if decision.stance not in self._protocol.stances():
            raise ValueError("answer content policy 返回未注册 stance")
        try:
            selected = tuple(candidates[key]
                             for key in decision.selected_candidate_keys)
        except KeyError as exc:
            raise ValueError("answer content policy 选择了请求外候选") from exc
        try:
            selected_artifacts = tuple(
                artifact_map[key] for key in decision.selected_artifact_keys)
        except KeyError as exc:
            raise ValueError("answer content policy 选择了请求外 Artifact") from exc
        selected_keys = set(decision.selected_candidate_keys)
        if any(item.candidate_key not in selected_keys
               for item in selected_artifacts):
            raise ValueError("采用 Artifact 必须归属于已选择候选")

        grouped = _group_states(selected)
        request_grouped = _group_states(request.candidates)
        has_group_conflict = any(
            state.support and state.refute for state in grouped.values())
        selected_propositions = set(grouped)
        has_hidden_conflict = any(
            request_grouped[key].support and request_grouped[key].refute
            for key in selected_propositions
        )
        request_has_conflict = any(
            state.support and state.refute
            for state in request_grouped.values()
        )
        ambiguities = _ambiguous_competitions(
            request.candidates, request.goal.required)
        answerable_all = bool(selected) and all(
            _satisfies(item.state, request.goal.required)
            and not (item.state.support and item.state.refute)
            for item in selected
        )
        if decision.stance == self._protocol.answer:
            if (not answerable_all or has_group_conflict
                    or has_hidden_conflict or request_has_conflict):
                raise ValueError("answer stance 必须选择满足目标且无冲突的候选")
            if ambiguities:
                raise ValueError("answer stance 遇同竞争组多解时必须改为 clarify")
            if any(not item.result.succeeded for item in selected_artifacts):
                raise ValueError("answer stance 只能采用已独立验证成功的 Artifact")
        elif decision.stance == self._protocol.conflict:
            if not selected or not has_group_conflict:
                raise ValueError("conflict stance 必须选择同命题 support/refute 冲突")
            if selected_artifacts:
                raise ValueError("conflict stance 不得用 Artifact 掩盖语言证据冲突")
        elif decision.stance == self._protocol.clarify:
            selected_props = frozenset(grouped)
            if not any(values.issubset(selected_props) for values in ambiguities):
                raise ValueError("clarify stance 必须保留同竞争组的完整多解命题")
            if (has_group_conflict or has_hidden_conflict
                    or request_has_conflict):
                raise ValueError("已有证据冲突时不能降格为 clarify")
            if selected_artifacts:
                raise ValueError("clarify stance 尚未决议，不得采用 Artifact")
        elif decision.stance == self._protocol.unknown:
            if any(
                    _satisfies(item.state, request.goal.required)
                    and not (item.state.support and item.state.refute)
                    for item in request.candidates):
                raise ValueError("存在可回答候选时不能返回 unknown")
            if request_has_conflict:
                raise ValueError("存在冲突候选时不能用 unknown 隐藏冲突")
            if selected_artifacts:
                raise ValueError("unknown stance 不得采用 Artifact 冒充答案")
        elif decision.stance == self._protocol.refuse:
            if selected_artifacts:
                raise ValueError("refuse stance 不得采用 Artifact")
        else:
            raise ValueError("未知 answer content stance")

        return AnswerContentSelection(
            request,
            self._protocol,
            decision.stance,
            decision.reason,
            decision.selected_candidate_keys,
            decision.selected_artifact_keys,
            decision.trace,
        )


class GenerationStanceLayerResolver:
    """把 G-01 选择结果投影为 G-00 stance 层，结果本身仍不写图。"""

    def __init__(
            self,
            planner_protocol: GenerationPlanProtocol,
            selector: AnswerContentSelector,
            artifacts: Sequence[ContentArtifactAttachment] = (),
            ) -> None:
        if not isinstance(planner_protocol, GenerationPlanProtocol):
            raise TypeError("stance resolver planner protocol 类型错误")
        if not isinstance(selector, AnswerContentSelector):
            raise TypeError("stance resolver selector 类型错误")
        if not isinstance(artifacts, Sequence):
            raise TypeError("stance resolver artifacts 必须是 Sequence")
        self._planner_protocol = planner_protocol
        self._selector = selector
        self._artifacts = tuple(artifacts)

    def resolve(
            self,
            request: GenerationPlanningRequest,
            prior: tuple[GenerationLayerResult, ...],
            ) -> GenerationLayerDecision:
        """独立计算 stance/content 选择，并写入 stance 层完整选择指纹。"""
        if prior:
            raise ValueError("stance layer 不得接收上游 generation result")
        selection = self._selector.select(request, self._artifacts)
        key = selection.stable_key()
        return GenerationLayerDecision(
            self._planner_protocol.stance_layer,
            self._planner_protocol.complete,
            selection.reason,
            selection.selected_candidate_keys,
            key,
            (*_packed(selection.stance.stable_key()), *_packed(key)),
        )


class GenerationContentLayerResolver:
    """重算 G-01 选择并与 stance 层交叉核验后提交 content 层。"""

    def __init__(
            self,
            planner_protocol: GenerationPlanProtocol,
            selector: AnswerContentSelector,
            artifacts: Sequence[ContentArtifactAttachment] = (),
            ) -> None:
        if not isinstance(planner_protocol, GenerationPlanProtocol):
            raise TypeError("content resolver planner protocol 类型错误")
        if not isinstance(selector, AnswerContentSelector):
            raise TypeError("content resolver selector 类型错误")
        if not isinstance(artifacts, Sequence):
            raise TypeError("content resolver artifacts 必须是 Sequence")
        self._planner_protocol = planner_protocol
        self._selector = selector
        self._artifacts = tuple(artifacts)

    def resolve(
            self,
            request: GenerationPlanningRequest,
            prior: tuple[GenerationLayerResult, ...],
            ) -> GenerationLayerDecision:
        """拒绝 stance/content 两次 policy 结果漂移，不依赖跨层隐藏缓存。"""
        if len(prior) != 1 or prior[0].layer != self._planner_protocol.stance_layer:
            raise ValueError("content layer 必须紧随唯一 stance layer")
        if prior[0].outcome != self._planner_protocol.complete:
            raise ValueError("content layer 只接受 complete stance layer")
        selection = self._selector.select(request, self._artifacts)
        key = selection.stable_key()
        if prior[0].payload != key:
            raise ValueError("stance/content 独立选择结果不一致")
        return GenerationLayerDecision(
            self._planner_protocol.content_layer,
            self._planner_protocol.complete,
            selection.reason,
            selection.selected_candidate_keys,
            key,
            (*_packed(selection.stance.stable_key()), *_packed(key)),
        )


__all__ = [
    "AnswerContentDecision",
    "AnswerContentPolicy",
    "AnswerContentProtocol",
    "AnswerContentSelection",
    "AnswerContentSelector",
    "ContentArtifactAttachment",
    "GenerationContentLayerResolver",
    "GenerationStanceLayerResolver",
]
