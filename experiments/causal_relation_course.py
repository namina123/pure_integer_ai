"""R-07 typed CAUSES 的正式课程请求、round 编排和评测克隆。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalEndpointEvaluation,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFact
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalExecutionTrace,
    CausalRelationRuntime,
    CausalVerificationRequest,
)
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureFormationTrace,
)
from pure_integer_ai.experiments.verification_orchestration import (
    MultiVerifierOrchestrator,
    VerificationReport,
)


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    return value


@dataclass(frozen=True)
class CausalEventTimeFactRequest:
    """课程显式提供的一条 Event/Proposition 时间事实写入请求。"""

    relation: ObjectIdentity
    subject: ObjectIdentity
    object_identity: ObjectIdentity
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验课程只提交 typed 身份和纯整数写入元数据。"""
        if not isinstance(self.relation, ObjectIdentity):
            raise TypeError("causal event-time relation 类型非法")
        if self.relation.object_kind != OBJECT_CONCEPT:
            raise ValueError("causal event-time relation 必须是一等 Concept")
        if not isinstance(self.subject, ObjectIdentity):
            raise TypeError("causal event-time subject 类型非法")
        if not isinstance(self.object_identity, ObjectIdentity):
            raise TypeError("causal event-time object 类型非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("causal event-time scope 类型非法")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("causal event-time qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="CausalEventTimeFactRequest",
        )


@dataclass(frozen=True)
class CausalFormationRequest:
    """课程显式提供的一次 S-00 定义和 R-00 forming 请求。"""

    spec: RelationClosureCandidateSpec
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验 forming 使用 typed 候选、scope 和纯整数元数据。"""
        if not isinstance(self.spec, RelationClosureCandidateSpec):
            raise TypeError("causal formation spec 类型非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("causal formation scope 类型非法")
        if self.scope.source != self.spec.proposition.source:
            raise ValueError("causal formation scope 必须绑定 Proposition 来源")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("causal formation qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="CausalFormationRequest",
        )


@dataclass(frozen=True)
class CausalExecutionRequest:
    """课程提供的一次 active CAUSES 与 S-04 四态执行请求。"""

    proposition: ObjectIdentity
    temporal: EventTimeVerificationRequest
    cause: CausalEndpointEvaluation
    effect: CausalEndpointEvaluation
    use_key: tuple[int, ...]
    generation_use_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """核验执行请求没有退化为裸节点、表层 cue 或匿名 use。"""
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("causal execution proposition 类型非法")
        if self.proposition.object_kind != OBJECT_PROPOSITION:
            raise ValueError("causal execution proposition 必须是一等 Proposition")
        if not isinstance(self.temporal, EventTimeVerificationRequest):
            raise TypeError("causal execution temporal 类型非法")
        if not isinstance(self.cause, CausalEndpointEvaluation):
            raise TypeError("causal execution cause 类型非法")
        if not isinstance(self.effect, CausalEndpointEvaluation):
            raise TypeError("causal execution effect 类型非法")
        _strict_key(self.use_key, where="CausalExecutionRequest.use_key")
        if self.generation_use_key is not None:
            _strict_key(
                self.generation_use_key,
                where="CausalExecutionRequest.generation_use_key",
            )


@dataclass(frozen=True)
class CausalRoundRequest:
    """一个来源 scope 下课程注入的全部 typed causal 操作。"""

    scope: ScopeIdentity
    temporal_facts: tuple[CausalEventTimeFactRequest, ...] = ()
    formations: tuple[CausalFormationRequest, ...] = ()
    verifications: tuple[CausalVerificationRequest, ...] = ()
    executions: tuple[CausalExecutionRequest, ...] = ()

    def __post_init__(self) -> None:
        """主动拒绝错误容器、重复候选、重复 recognition 和重复 use。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("causal round scope 类型非法")
        groups = (
            ("temporal_facts", self.temporal_facts, CausalEventTimeFactRequest),
            ("formations", self.formations, CausalFormationRequest),
            ("verifications", self.verifications, CausalVerificationRequest),
            ("executions", self.executions, CausalExecutionRequest),
        )
        for name, values, expected in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"causal round {name} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"causal round {name} 元素类型非法")
        if any(item.scope != self.scope for item in self.temporal_facts):
            raise ValueError("causal round 时间事实必须绑定当前 scope")
        if any(item.scope != self.scope for item in self.formations):
            raise ValueError("causal round forming 必须绑定当前 scope")
        if any(
                item.temporal.scope != self.scope
                for item in self.verifications):
            raise ValueError("causal round verification temporal 必须绑定当前 scope")
        if any(item.temporal.scope != self.scope for item in self.executions):
            raise ValueError("causal round execution temporal 必须绑定当前 scope")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("同一 causal round 不得重复 forming Proposition")
        recognition_routes = tuple(
            (
                item.spec.proposition.proposition,
                item.observation,
                item.event_key,
            )
            for item in self.verifications
        )
        if len(set(recognition_routes)) != len(recognition_routes):
            raise ValueError("同一 causal round 不得重复 recognition 路由")
        use_keys = tuple(item.use_key for item in self.executions)
        if len(set(use_keys)) != len(use_keys):
            raise ValueError("同一 causal round 不得重复 execution use_key")
        generation_keys = tuple(
            item.generation_use_key
            for item in self.executions
            if item.generation_use_key is not None
        )
        if len(set(generation_keys)) != len(generation_keys):
            raise ValueError("同一 causal round 不得重复 generation use_key")


@dataclass(frozen=True)
class CausalRoundReport:
    """保存课程请求对应的事实写入、forming、核验和执行报告。"""

    scope: ScopeIdentity
    read_only: bool
    temporal_facts: tuple[OrderFact, ...]
    formations: tuple[RelationClosureFormationTrace, ...]
    verifications: tuple[VerificationReport, ...]
    executions: tuple[CausalExecutionTrace, ...]

    def __post_init__(self) -> None:
        """核验正式结果容器完整且 held-out 标志为严格布尔值。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("causal report scope 类型非法")
        if type(self.read_only) is not bool:
            raise TypeError("causal report read_only 必须是 bool")
        groups = (
            (self.temporal_facts, OrderFact, "temporal_facts"),
            (self.formations, RelationClosureFormationTrace, "formations"),
            (self.verifications, VerificationReport, "verifications"),
            (self.executions, CausalExecutionTrace, "executions"),
        )
        for values, expected, name in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"causal report {name} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"causal report {name} 元素类型非法")
        if self.read_only and self.formations:
            raise ValueError("read-only causal report 不得包含 forming 写入")


@runtime_checkable
class CausalRelationProtocol(Protocol):
    """由项目课程注入低层 R-00/R-06/R-07 typed runtime 的构造协议。"""

    def build(self, ctx) -> CausalRelationRuntime:
        """在指定 TrainContext 的图上构造完整 causal owner。"""
        ...

    def stable_key(self) -> tuple[int, ...]:
        """返回协议身份、schema、Role 和 resolver 版本的完整整数键。"""
        ...


@runtime_checkable
class CausalRelationCourse(Protocol):
    """把来源 scope 映射为 typed 课程请求，不向宿主暴露表层规则。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> CausalRoundRequest:
        """返回当前训练或 held-out observation 的 typed causal 请求。"""
        ...

    def clone_for_evaluation(self) -> "CausalRelationCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程 mapper 的可比较纯整数状态。"""
        ...


class CausalRelationCourseRuntime:
    """让 formal round 只提交 scope，由课程和 typed owner 完成 R-07。"""

    def __init__(
            self,
            owner: CausalRelationRuntime,
            protocol_key: tuple[int, ...],
            course: CausalRelationCourse,
            ) -> None:
        """绑定唯一 causal owner、协议身份和可克隆课程 mapper。"""
        if not isinstance(owner, CausalRelationRuntime):
            raise TypeError("causal course runtime owner 类型非法")
        self.protocol_key = _strict_key(
            protocol_key,
            where="CausalRelationCourseRuntime.protocol_key",
        )
        if not isinstance(course, CausalRelationCourse):
            raise TypeError("course 未实现 CausalRelationCourse")
        _strict_key(course.state_key(), where="CausalRelationCourse.state_key")
        self.owner = owner
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> CausalRoundReport:
        """按时间事实、forming、纯核验和 provisional 执行的顺序处理请求。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("causal process scope 类型非法")
        if type(read_only) is not bool:
            raise TypeError("causal process read_only 必须是 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, CausalRoundRequest):
            raise TypeError("course.request 返回类型非法")
        if request.scope != scope:
            raise ValueError("course.request 替换了 round scope")
        if read_only and request.formations:
            raise ValueError("held-out causal 请求不得形成新候选")

        temporal_facts = tuple(
            self.owner.event_time_facts.record(
                item.relation,
                item.subject,
                item.object_identity,
                scope=item.scope,
                provenance_kind=item.provenance_kind,
                epistemic_origin=item.epistemic_origin,
                content_version=item.content_version,
                qualifiers=item.qualifiers,
            )
            for item in request.temporal_facts
        )
        formations = tuple(
            self.owner.form(
                item.spec,
                scope=item.scope,
                provenance_kind=item.provenance_kind,
                epistemic_origin=item.epistemic_origin,
                content_version=item.content_version,
                qualifiers=item.qualifiers,
            )
            for item in request.formations
        )
        orchestrator = MultiVerifierOrchestrator()
        verifications = tuple(
            orchestrator.run(
                item,
                (self.owner.registration(),),
                read_only=read_only,
            )
            for item in request.verifications
        )
        executions = tuple(
            self.owner.execute(
                item.proposition,
                item.temporal,
                item.cause,
                item.effect,
                use_key=item.use_key,
                generation_use_key=item.generation_use_key,
            )
            for item in request.executions
        )
        return CausalRoundReport(
            scope,
            read_only,
            temporal_facts,
            formations,
            verifications,
            executions,
        )

    def clone_for_context(self, ctx) -> "CausalRelationCourseRuntime":
        """在评测 context 上克隆 owner 与课程，宿主 ledger 保持不变。"""
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_course, CausalRelationCourse):
            raise TypeError("course clone 未实现 CausalRelationCourse")
        return CausalRelationCourseRuntime(
            self.owner.clone_for_context(ctx),
            self.protocol_key,
            cloned_course,
        )

    def state_key(self) -> tuple:
        """返回协议、低层 owner 和课程 mapper 的完整隔离状态。"""
        course_key = _strict_key(
            self.course.state_key(),
            where="CausalRelationCourse.state_key",
        )
        return self.protocol_key, self.owner.state_key(), course_key


def install_causal_relation_runtime(
        ctx,
        protocol: CausalRelationProtocol,
        course: CausalRelationCourse,
        ) -> CausalRelationCourseRuntime:
    """在 TrainContext 上安装成对注入的 R-07 protocol/course。"""
    if not isinstance(protocol, CausalRelationProtocol):
        raise TypeError("protocol 未实现 CausalRelationProtocol")
    if not isinstance(course, CausalRelationCourse):
        raise TypeError("course 未实现 CausalRelationCourse")
    if getattr(ctx, "causal_relation_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 causal relation runtime")
    protocol_key = _strict_key(
        protocol.stable_key(),
        where="CausalRelationProtocol.stable_key",
    )
    _strict_key(course.state_key(), where="CausalRelationCourse.state_key")
    owner = protocol.build(ctx)
    if not isinstance(owner, CausalRelationRuntime):
        raise TypeError("protocol.build 必须返回 CausalRelationRuntime")
    if owner.semantic_graph.ontology is not ctx.graph_ontology:
        raise ValueError("causal protocol.build 未绑定 TrainContext 图")
    if owner.event_time_facts.ontology is not ctx.graph_ontology:
        raise ValueError("causal event-time facade 未绑定 TrainContext 图")
    runtime = CausalRelationCourseRuntime(
        owner,
        protocol_key,
        course,
    )
    ctx.causal_relation_runtime = runtime
    return runtime


__all__ = [
    "CausalEventTimeFactRequest",
    "CausalExecutionRequest",
    "CausalFormationRequest",
    "CausalRelationCourse",
    "CausalRelationCourseRuntime",
    "CausalRelationProtocol",
    "CausalRoundReport",
    "CausalRoundRequest",
    "install_causal_relation_runtime",
]
