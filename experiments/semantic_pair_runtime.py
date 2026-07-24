"""R-05 SIMILAR/ANTONYM 双独立 owner 的课程、预算和 V-06 编排。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairKnowledge,
    SymmetricRelationBudgetExceeded,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
)
from pure_integer_ai.experiments.symmetric_relation_runtime import (
    LegacySymmetricPairMapper,
    SymmetricChannelBatch,
    SymmetricChannelReport,
    SymmetricPairFormationRequest,
    SymmetricPairQuery,
    SymmetricPairRuntimeResult,
    SymmetricRelationChannelRuntime,
    SymmetricRelationRuntimeError,
)
from pure_integer_ai.experiments.train_context import TrainContext


class SemanticPairRuntimeError(RuntimeError):
    """R-05 双 owner、课程、预算或跨 channel 边界不完整。"""


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验 builder、课程和 mapper 的纯整数稳定键。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SemanticPairBudget:
    """限制两个独立 channel 当前直接 Evidence 的总量。"""

    max_total_direct_facts: int

    def __post_init__(self) -> None:
        """要求总直接事实预算为严格正整数。"""
        assert_int(
            self.max_total_direct_facts,
            _where="SemanticPairBudget.max_total_direct_facts",
        )
        if (type(self.max_total_direct_facts) is not int
                or self.max_total_direct_facts <= 0):
            raise ValueError("semantic pair 总直接事实预算必须为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回双 channel 直接事实总预算。"""
        return (self.max_total_direct_facts,)


class SemanticPairRuntime:
    """组合两个 hypothesis kind 独立的对称 relation channel。"""

    def __init__(
            self,
            similar: SymmetricRelationChannelRuntime,
            antonym: SymmetricRelationChannelRuntime,
            budget: SemanticPairBudget,
            ) -> None:
        """核验两个 channel 的 owner、kind、relation、schema 和 ledger 均独立。"""
        if not isinstance(similar, SymmetricRelationChannelRuntime):
            raise TypeError("semantic pair similar channel 类型错误")
        if not isinstance(antonym, SymmetricRelationChannelRuntime):
            raise TypeError("semantic pair antonym channel 类型错误")
        if not isinstance(budget, SemanticPairBudget):
            raise TypeError("semantic pair budget 类型错误")
        if similar is antonym:
            raise SemanticPairRuntimeError("两个 semantic pair channel 不得共用 facade")
        if similar.relation_runtime is antonym.relation_runtime:
            raise SemanticPairRuntimeError("两个 channel 不得共用 RelationClosureRuntime")
        if (similar.relation_runtime.candidate_runtime
                is antonym.relation_runtime.candidate_runtime):
            raise SemanticPairRuntimeError("两个 channel 不得共用 CandidateLearningRuntime")
        if similar.hypothesis_kind == antonym.hypothesis_kind:
            raise SemanticPairRuntimeError("SIMILAR/ANTONYM hypothesis kind 必须不同")
        if similar.protocol.relation == antonym.protocol.relation:
            raise SemanticPairRuntimeError("两个 channel relation 身份必须不同")
        if similar.protocol.schema.schema == antonym.protocol.schema.schema:
            raise SemanticPairRuntimeError("两个 channel schema 身份必须不同")
        if (similar.semantic_graph.ontology
                is not antonym.semantic_graph.ontology):
            raise SemanticPairRuntimeError("两个 channel 必须共享同一图本体")
        self.similar = similar
        self.antonym = antonym
        self.budget = budget

    @property
    def ontology(self):
        """返回两个 channel 共享的权威图本体。"""
        return self.similar.semantic_graph.ontology

    def knowledge(
            self,
            ) -> tuple[SymmetricPairKnowledge, SymmetricPairKnowledge]:
        """读取两个 owner 快照，并执行跨 channel 直接事实总预算。"""
        similar = self.similar.knowledge()
        antonym = self.antonym.knowledge()
        total = len(similar.evidence) + len(antonym.evidence)
        if total > self.budget.max_total_direct_facts:
            raise SymmetricRelationBudgetExceeded(
                "semantic pair 双 channel 直接事实总预算耗尽")
        return similar, antonym

    def query_similar(
            self, query: SymmetricPairQuery,
            ) -> SymmetricPairRuntimeResult:
        """在双 owner 总预算下查询 SIMILAR channel。"""
        similar, _antonym = self.knowledge()
        selections = self.similar.select_many((query,), knowledge=similar)
        uses = self.similar.consume_selections((query,), selections)
        return SymmetricPairRuntimeResult(selections[0], uses[0])

    def query_antonym(
            self, query: SymmetricPairQuery,
            ) -> SymmetricPairRuntimeResult:
        """在双 owner 总预算下查询 ANTONYM channel。"""
        _similar, antonym = self.knowledge()
        selections = self.antonym.select_many((query,), knowledge=antonym)
        uses = self.antonym.consume_selections((query,), selections)
        return SymmetricPairRuntimeResult(selections[0], uses[0])

    def state_key(self) -> tuple:
        """返回两个独立 owner、协议、kind 和总预算完整状态。"""
        return (
            self.similar.state_key(),
            self.antonym.state_key(),
            self.budget.stable_key(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "SemanticPairRuntime":
        """在 V-06 克隆图上分别重建两个独立 channel owner。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("semantic pair clone ctx 类型错误")
        return SemanticPairRuntime(
            self.similar.clone_for_context(ctx),
            self.antonym.clone_for_context(ctx),
            self.budget,
        )


@dataclass(frozen=True)
class SemanticPairRoundRequest:
    """同一来源 scope 中两个 channel 的纯学习轮或纯查询轮。"""

    scope: ScopeIdentity
    similar: SymmetricChannelBatch = field(
        default_factory=SymmetricChannelBatch)
    antonym: SymmetricChannelBatch = field(
        default_factory=SymmetricChannelBatch)

    def __post_init__(self) -> None:
        """核验 scope、跨 channel 混轮、Proposition 和 recognition 唯一性。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("semantic pair round scope 类型错误")
        if not isinstance(self.similar, SymmetricChannelBatch):
            raise TypeError("semantic pair similar batch 类型错误")
        if not isinstance(self.antonym, SymmetricChannelBatch):
            raise TypeError("semantic pair antonym batch 类型错误")
        batches = (self.similar, self.antonym)
        writes = any(
            batch.legacy_records or batch.formations or batch.recognitions
            for batch in batches
        )
        queries = any(batch.queries for batch in batches)
        if writes and queries:
            raise ValueError("semantic pair 学习写入轮与查询采用轮必须分开")
        for batch in batches:
            if any(item.scope != self.scope for item in batch.legacy_records):
                raise ValueError("legacy symmetric record 必须绑定当前 scope")
            if any(item.scope != self.scope for item in batch.formations):
                raise ValueError("symmetric formation 必须绑定当前 scope")
            if any(item.scope != self.scope for item in batch.recognitions):
                raise ValueError("symmetric recognition 必须绑定当前 scope")
            contexts = tuple(
                item.context for item in batch.queries
                if item.context is not None
            )
            if any(item.scope != self.scope for item in contexts):
                raise ValueError("symmetric query context 必须绑定当前 scope")
        propositions = tuple(
            item.spec.proposition.proposition
            for batch in batches for item in batch.formations
        )
        if len(set(propositions)) != len(propositions):
            raise ValueError("两个 semantic pair channel 不得重复 Proposition")
        routes = tuple(
            item.route_key()
            for batch in batches for item in batch.recognitions
        )
        if len(set(routes)) != len(routes):
            raise ValueError("两个 semantic pair channel 不得重复 recognition 路由")


@dataclass(frozen=True)
class SemanticPairRoundReport:
    """一次 R-05 课程轮的两个独立 channel 报告。"""

    scope: ScopeIdentity
    read_only: bool
    similar: SymmetricChannelReport
    antonym: SymmetricChannelReport

    def __post_init__(self) -> None:
        """核验报告 scope、只读标志和两个 channel 报告类型。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("semantic pair report scope 类型错误")
        if type(self.read_only) is not bool:
            raise TypeError("semantic pair report read_only 必须是严格 bool")
        if not isinstance(self.similar, SymmetricChannelReport):
            raise TypeError("semantic pair similar report 类型错误")
        if not isinstance(self.antonym, SymmetricChannelReport):
            raise TypeError("semantic pair antonym report 类型错误")
        if self.read_only and (
                self.similar.formations
                or self.similar.recognitions
                or self.antonym.formations
                or self.antonym.recognitions):
            raise ValueError("read-only semantic pair report 不得含学习写入")


@runtime_checkable
class SemanticPairRuntimeBuilder(Protocol):
    """由项目课程注入两个独立 R-00 owner 和 R-05 纯语义组件。"""

    def build(self, ctx: TrainContext) -> SemanticPairRuntime:
        """在指定 TrainContext 图上构造双 channel owner。"""
        ...

    def clone_for_evaluation(self) -> "SemanticPairRuntimeBuilder":
        """返回清除宿主可变引用的评测 builder。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回两个 kind、协议、schema、规则和预算版本完整键。"""
        ...


@runtime_checkable
class SemanticPairCourse(Protocol):
    """把来源 scope 映射为两个 channel 的 typed 学习轮或查询轮。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> SemanticPairRoundRequest:
        """返回当前来源的 typed R-05 请求。"""
        ...

    def similar_legacy_mapper(self) -> LegacySymmetricPairMapper | None:
        """需要迁移旧 SIMILAR 输入时返回显式 mapper。"""
        ...

    def antonym_legacy_mapper(self) -> LegacySymmetricPairMapper | None:
        """需要迁移旧 ANTONYM 输入时返回显式 mapper。"""
        ...

    def clone_for_evaluation(self) -> "SemanticPairCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程、mapper 和来源策略的完整整数键。"""
        ...


@dataclass(frozen=True)
class _ChannelPlan:
    """一个 channel 经 legacy 映射和全量逻辑序分配后的提交计划。"""

    formations: tuple[SymmetricPairFormationRequest, ...]
    recognitions: tuple[RelationClosureRecognitionInput, ...]
    formation_requests: tuple[tuple, ...]
    recognition_requests: tuple[tuple, ...]


class SemanticPairCourseRuntime:
    """让 formal round 只提交 scope，由课程编排两个独立 R-05 owner。"""

    def __init__(
            self,
            ctx: TrainContext,
            owner: SemanticPairRuntime,
            builder: SemanticPairRuntimeBuilder,
            course: SemanticPairCourse,
            ) -> None:
        """绑定当前 context、双 owner、可克隆 builder 和课程。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("semantic pair course ctx 类型错误")
        if not isinstance(owner, SemanticPairRuntime):
            raise TypeError("semantic pair owner 类型错误")
        if not isinstance(builder, SemanticPairRuntimeBuilder):
            raise TypeError("semantic pair builder 协议不完整")
        if not isinstance(course, SemanticPairCourse):
            raise TypeError("semantic pair course 协议不完整")
        _strict_key(builder.state_key(), label="SemanticPairRuntimeBuilder.state_key")
        _strict_key(course.state_key(), label="SemanticPairCourse.state_key")
        self.ctx = ctx
        self.owner = owner
        self.builder = builder
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> SemanticPairRoundReport:
        """全量预检后执行双 channel 纯学习轮或纯 context 查询轮。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("semantic pair process scope 类型错误")
        if type(read_only) is not bool:
            raise TypeError("semantic pair process read_only 必须是严格 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, SemanticPairRoundRequest):
            raise TypeError("semantic pair course.request 返回类型错误")
        if request.scope != scope:
            raise ValueError("semantic pair course.request 替换了 round scope")
        if read_only and self._has_writes(request):
            raise ValueError("read-only semantic pair 请求不得学习或迁移旧边")

        similar_plan = self._prepare_channel(
            self.owner.similar,
            request.similar,
            self.course.similar_legacy_mapper(),
            scope,
        )
        antonym_plan = self._prepare_channel(
            self.owner.antonym,
            request.antonym,
            self.course.antonym_legacy_mapper(),
            scope,
        )
        self._validate_cross_plans(similar_plan, antonym_plan)

        similar_selections = ()
        antonym_selections = ()
        if request.similar.queries or request.antonym.queries:
            similar_knowledge, antonym_knowledge = self.owner.knowledge()
            if request.similar.queries:
                similar_selections = self.owner.similar.select_many(
                    request.similar.queries,
                    knowledge=similar_knowledge,
                )
            if request.antonym.queries:
                antonym_selections = self.owner.antonym.select_many(
                    request.antonym.queries,
                    knowledge=antonym_knowledge,
                )

        for owner, plan in (
                (self.owner.similar, similar_plan),
                (self.owner.antonym, antonym_plan)):
            for item in plan.formations:
                owner.semantic_graph.define_atomic(
                    item.spec.proposition,
                    scope=item.scope,
                    **item.metadata(),
                )
        similar_report = self._commit_channel(
            self.owner.similar,
            similar_plan,
            request.similar.queries,
            similar_selections,
        )
        antonym_report = self._commit_channel(
            self.owner.antonym,
            antonym_plan,
            request.antonym.queries,
            antonym_selections,
        )
        return SemanticPairRoundReport(
            scope,
            read_only,
            similar_report,
            antonym_report,
        )

    def clone_for_context(self, ctx: TrainContext) -> "SemanticPairCourseRuntime":
        """用克隆 builder/course 在评测 context 上重建独立双 owner。"""
        cloned_builder = self.builder.clone_for_evaluation()
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_builder, SemanticPairRuntimeBuilder):
            raise TypeError("semantic pair builder clone 协议不完整")
        if not isinstance(cloned_course, SemanticPairCourse):
            raise TypeError("semantic pair course clone 协议不完整")
        if cloned_builder.state_key() != self.builder.state_key():
            raise ValueError("semantic pair builder clone 改变协议状态")
        if cloned_course.state_key() != self.course.state_key():
            raise ValueError("semantic pair course clone 改变课程状态")
        return SemanticPairCourseRuntime(
            ctx,
            self.owner.clone_for_context(ctx),
            cloned_builder,
            cloned_course,
        )

    def state_key(self) -> tuple:
        """返回 builder、课程、双 owner 和两个可选 mapper 完整状态。"""
        mapper_keys = []
        for label, mapper in (
                ("similar", self.course.similar_legacy_mapper()),
                ("antonym", self.course.antonym_legacy_mapper())):
            if mapper is None:
                mapper_keys.append(())
                continue
            if not isinstance(mapper, LegacySymmetricPairMapper):
                raise TypeError(f"{label} legacy mapper 协议不完整")
            key = mapper.state_key()
            _strict_key(key, label=f"{label} LegacySymmetricPairMapper.state_key")
            mapper_keys.append(key)
        return (
            self.builder.state_key(),
            self.course.state_key(),
            tuple(mapper_keys),
            self.owner.state_key(),
        )

    @staticmethod
    def _has_writes(request: SemanticPairRoundRequest) -> bool:
        """判断双 channel 请求是否包含 forming、recognition 或 legacy 迁移。"""
        return any(
            batch.legacy_records or batch.formations or batch.recognitions
            for batch in (request.similar, request.antonym)
        )

    def _prepare_channel(
            self,
            owner: SymmetricRelationChannelRuntime,
            batch: SymmetricChannelBatch,
            mapper: LegacySymmetricPairMapper | None,
            scope: ScopeIdentity,
            ) -> _ChannelPlan:
        """完成 legacy 映射、schema 预检和 recognition 逻辑序分配。"""
        formations = list(batch.formations)
        recognitions = list(batch.recognitions)
        if batch.legacy_records:
            if mapper is None:
                raise SemanticPairRuntimeError(
                    "legacy symmetric 记录缺少当前 channel 显式 mapper")
            if not isinstance(mapper, LegacySymmetricPairMapper):
                raise TypeError("legacy symmetric mapper 协议不完整")
            for record in batch.legacy_records:
                mapped = owner.map_legacy(record, mapper)
                if mapped is None:
                    raise SemanticPairRuntimeError(
                        "legacy symmetric mapper 无法补全 typed pair")
                formations.append(mapped.formation)
                recognitions.append(mapped.recognition)
        self._validate_channel_combined(formations, recognitions)
        if any(item.scope != scope for item in formations):
            raise ValueError("mapped symmetric formation 必须绑定当前 scope")
        if any(item.scope != scope for item in recognitions):
            raise ValueError("mapped symmetric recognition 必须绑定当前 scope")

        formation_requests = tuple(
            (item.spec, item.timestamp_base) for item in formations)
        timestamps = ()
        if recognitions:
            next_timestamp = (
                owner.relation_runtime.candidate_runtime.next_timestamps(1)[0]
            )
            formation_end = max((
                item.timestamp_base + len(item.spec.forming_sources) - 1
                for item in formations
            ), default=next_timestamp - 1)
            start = max(next_timestamp, formation_end + 1)
            timestamps = tuple(range(start, start + len(recognitions) * 3))
        recognition_requests = tuple(
            (
                recognition,
                timestamps[index * 3],
                timestamps[index * 3 + 1],
                timestamps[index * 3 + 2],
            )
            for index, recognition in enumerate(recognitions)
        )
        for item in formations:
            owner.pair_from_definition(item.spec.proposition)
            if item.spec.schema != owner.protocol.schema:
                raise SymmetricRelationRuntimeError(
                    "symmetric formation 未使用当前 typed schema")
            owner.semantic_graph.preflight_atomic(
                item.spec.proposition,
                scope=item.scope,
                **item.metadata(),
            )
        if formation_requests or recognition_requests:
            owner.relation_runtime.preflight_many(
                formation_requests,
                recognition_requests,
            )
        return _ChannelPlan(
            tuple(formations),
            tuple(recognitions),
            formation_requests,
            recognition_requests,
        )

    @staticmethod
    def _commit_channel(
            owner: SymmetricRelationChannelRuntime,
            plan: _ChannelPlan,
            queries: tuple[SymmetricPairQuery, ...],
            selections: tuple,
            ) -> SymmetricChannelReport:
        """提交一个已全量预检的 channel，并按预计算 selection 写 context Use。"""
        formation_traces = (
            owner.relation_runtime.form_many(plan.formation_requests)
            if plan.formation_requests else ()
        )
        recognition_traces = (
            owner.relation_runtime.recognize_many_at(
                plan.recognition_requests)
            if plan.recognition_requests else ()
        )
        query_results = ()
        if queries:
            uses = owner.consume_selections(queries, selections)
            query_results = tuple(
                SymmetricPairRuntimeResult(selection, group)
                for selection, group in zip(selections, uses, strict=True)
            )
        return SymmetricChannelReport(
            formation_traces,
            recognition_traces,
            query_results,
        )

    @staticmethod
    def _validate_channel_combined(
            formations: list[SymmetricPairFormationRequest],
            recognitions: list[RelationClosureRecognitionInput],
            ) -> None:
        """在 legacy 映射后拒绝单 channel 重复 Proposition 和 recognition 路由。"""
        propositions = tuple(
            item.spec.proposition.proposition for item in formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("symmetric combined forming Proposition 重复")
        routes = tuple(item.route_key() for item in recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("symmetric combined recognition 路由重复")

    @staticmethod
    def _validate_cross_plans(
            similar: _ChannelPlan, antonym: _ChannelPlan,
            ) -> None:
        """拒绝 legacy 映射后两个 channel 复用 Proposition 或 recognition 路由。"""
        propositions = tuple(
            item.spec.proposition.proposition
            for plan in (similar, antonym) for item in plan.formations
        )
        if len(set(propositions)) != len(propositions):
            raise ValueError("两个 semantic pair channel 不得复用 Proposition")
        routes = tuple(
            item.route_key()
            for plan in (similar, antonym) for item in plan.recognitions
        )
        if len(set(routes)) != len(routes):
            raise ValueError("两个 semantic pair channel 不得复用 recognition 路由")


def install_semantic_pair_runtime(
        ctx: TrainContext,
        builder: SemanticPairRuntimeBuilder,
        course: SemanticPairCourse,
        ) -> SemanticPairCourseRuntime:
    """在 TrainContext 上安装显式成对注入且默认关闭的 R-05 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install semantic pair ctx 类型错误")
    if not isinstance(builder, SemanticPairRuntimeBuilder):
        raise TypeError("semantic pair builder 协议不完整")
    if not isinstance(course, SemanticPairCourse):
        raise TypeError("semantic pair course 协议不完整")
    if getattr(ctx, "semantic_pair_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 semantic pair runtime")
    _strict_key(builder.state_key(), label="SemanticPairRuntimeBuilder.state_key")
    _strict_key(course.state_key(), label="SemanticPairCourse.state_key")
    owner = builder.build(ctx)
    if not isinstance(owner, SemanticPairRuntime):
        raise TypeError("semantic pair builder.build 返回类型错误")
    if owner.ontology is not ctx.graph_ontology:
        raise ValueError("semantic pair owner 未绑定当前 TrainContext 图")
    runtime = SemanticPairCourseRuntime(ctx, owner, builder, course)
    ctx.semantic_pair_runtime = runtime
    return runtime


__all__ = [
    "SemanticPairBudget",
    "SemanticPairCourse",
    "SemanticPairCourseRuntime",
    "SemanticPairRoundReport",
    "SemanticPairRoundRequest",
    "SemanticPairRuntime",
    "SemanticPairRuntimeBuilder",
    "SemanticPairRuntimeError",
    "install_semantic_pair_runtime",
]
