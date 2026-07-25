"""A-03 同原文跨 ParserVersion 修正的事务协调入口。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ArchiveDirective,
    HypothesisResolver,
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.parser_revision import (
    MaterializedParserRevision,
    ParserHypothesisRevision,
    ParserRevisionError,
    ParserRevisionGraph,
    ParserRevisionRequest,
    parser_lineage_key,
)
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.storage.source_record import SourceRecordRepository


class ParserRevisionRuntimeError(RuntimeError):
    """A-03 来源、当前 lineage、训练历史或事务介质违反契约。"""


class ParserRevisionTrainingSink:
    """按 Hypothesis observation 路由多个 parser 来源的统一 Core 训练历史 sink。"""

    def __init__(self, routes: tuple[tuple[SourceRef, object], ...]) -> None:
        """绑定开放 SourceRef 到训练 sink 的唯一映射，并要求全部共享 backend。"""
        if not isinstance(routes, tuple) or not routes:
            raise ValueError("A-03 training sink routes 必须是非空 tuple")
        checked = {}
        backends = set()
        for source, sink in routes:
            if not isinstance(source, SourceRef):
                raise TypeError("A-03 training sink route source 类型错误")
            if source in checked:
                raise ValueError("A-03 training sink source 不得重复")
            if any(not callable(getattr(sink, name, None)) for name in (
                    "append_hypothesis",
                    "append_evidence",
                    "append_transition",
                    "append_decision")):
                raise TypeError("A-03 route sink 缺少 H-00/H-04 追加协议")
            history = getattr(sink, "history", None)
            backend = getattr(history, "backend", None)
            if backend is None:
                raise TypeError("A-03 route sink 没有可恢复训练 backend")
            checked[source] = sink
            backends.add(backend)
        if len(backends) != 1:
            raise ValueError("A-03 route sinks 必须共享同一 Core backend")
        self._routes = checked
        self.backend = next(iter(backends))

    def append_hypothesis(self, hypothesis: HypothesisKey) -> None:
        """把候选声明路由到其 observation 对应的训练历史。"""
        self._sink_for(hypothesis).append_hypothesis(hypothesis)

    def append_evidence(self, evidence: EvidenceRecord) -> None:
        """按被说明候选而非 Evidence 来源路由不可变证据。"""
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("A-03 routed Evidence 类型错误")
        self._sink_for(evidence.hypothesis).append_evidence(evidence)

    def append_transition(self, transition: HypothesisTransition) -> None:
        """把生命周期转换路由到被转换候选的训练历史。"""
        if not isinstance(transition, HypothesisTransition):
            raise TypeError("A-03 routed transition 类型错误")
        self._sink_for(transition.hypothesis).append_transition(transition)

    def append_decision(self, decision: ResolverDecision) -> None:
        """要求一个 H-04 competition 只有一个 observation，再路由完整决策。"""
        if not isinstance(decision, ResolverDecision):
            raise TypeError("A-03 routed decision 类型错误")
        hypotheses = tuple(item.hypothesis for item in decision.candidates)
        if not hypotheses:
            raise ParserRevisionRuntimeError(
                "A-03 routed decision 不得没有候选")
        observations = {item.observation for item in hypotheses}
        if len(observations) != 1:
            raise ParserRevisionRuntimeError(
                "A-03 H-04 decision 跨越 parser observation")
        self._sink_for(hypotheses[0]).append_decision(decision)

    def _sink_for(self, hypothesis: HypothesisKey):
        """按完整 observation 精确返回训练 sink，缺路由时 fail closed。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("A-03 routed hypothesis 类型错误")
        sink = self._routes.get(hypothesis.observation)
        if sink is None:
            raise ParserRevisionRuntimeError(
                "A-03 training sink 缺少 hypothesis observation 路由")
        return sink


@dataclass(frozen=True)
class ParserRevisionResult:
    """一次提交或精确重放得到的 revision 图对象和 H-04 决策。"""

    materialized: MaterializedParserRevision
    decisions: tuple[ResolverDecision, ...]
    replayed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.materialized, MaterializedParserRevision):
            raise TypeError("A-03 result materialized 类型错误")
        if (not isinstance(self.decisions, tuple)
                or any(not isinstance(item, ResolverDecision)
                       for item in self.decisions)):
            raise TypeError("A-03 result decisions 类型错误")
        if type(self.replayed) is not bool:
            raise TypeError("A-03 result replayed 必须是严格 bool")

    def stable_key(self) -> tuple[int, ...]:
        """返回图结果、决策和重放标记的完整确定性键。"""
        result = [*self.materialized.stable_key(), len(self.decisions)]
        for decision in self.decisions:
            key = decision.stable_key()
            result.extend((len(key), *key))
        result.append(int(self.replayed))
        return tuple(result)


class ParserRevisionRuntime:
    """核验同原文重解析，并原子提交 revision 图和旧候选退出历史。"""

    def __init__(
            self,
            source_records: SourceRecordRepository,
            graph: ParserRevisionGraph,
            ledger: HypothesisLedger,
            resolver: HypothesisResolver,
            ) -> None:
        """绑定共享 Core backend、ParserRevision 图和同一 H-00/H-04 状态。"""
        if not isinstance(source_records, SourceRecordRepository):
            raise TypeError("A-03 source_records 类型错误")
        if not isinstance(graph, ParserRevisionGraph):
            raise TypeError("A-03 graph 类型错误")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("A-03 ledger 类型错误")
        if not isinstance(resolver, HypothesisResolver):
            raise TypeError("A-03 resolver 类型错误")
        if resolver.ledger is not ledger:
            raise ValueError("A-03 resolver 必须绑定同一 ledger")
        if source_records.backend is not graph.ontology.backend:
            raise ValueError("A-03 SourceRecord 与 revision 图必须共享 backend")
        self.source_records = source_records
        self.graph = graph
        self.ledger = ledger
        self.resolver = resolver
        self._validate_transaction_sinks()

    def apply(self, request: ParserRevisionRequest) -> ParserRevisionResult:
        """整批预检来源、影响集和历史后提交；任一步异常恢复调用前状态。"""
        if not isinstance(request, ParserRevisionRequest):
            raise TypeError("A-03 request 类型错误")
        self._validate_sources(request)
        existing = self.graph.preflight(request)
        self._validate_lineage(request, existing is not None)
        if existing is not None:
            decisions = self._validate_exact_history(request)
            return ParserRevisionResult(existing, decisions, True)

        self._validate_materialized_inputs(request)
        groups = self._validate_live_history(request)
        preview_ledger = self.ledger.clone()
        preview_resolver = self.resolver.clone(ledger=preview_ledger)
        preview_decisions = self._apply_history(
            request, groups, preview_ledger, preview_resolver)

        backend = self.graph.ontology.backend
        backend_state = backend.recovery_state_snapshot()
        ledger_state = self.ledger.clone()
        resolver_state = self.resolver.clone(ledger=ledger_state)
        try:
            materialized = self.graph.materialize(request)
            decisions = self._apply_history(
                request, groups, self.ledger, self.resolver)
            if tuple(item.stable_key() for item in decisions) != tuple(
                    item.stable_key() for item in preview_decisions):
                raise ParserRevisionRuntimeError(
                    "A-03 宿主决策与隔离 preview 漂移")
            backend.commit()
            return ParserRevisionResult(materialized, decisions, False)
        except BaseException:
            backend.restore_recovery_state(backend_state)
            backend.commit()
            self.graph.ontology.clear_runtime_caches()
            self.source_records.clear_runtime_caches()
            self.ledger._restore_runtime_state(ledger_state)
            self.resolver._restore_runtime_state(resolver_state)
            raise

    def clone_for_evaluation(self) -> "ParserRevisionRuntime":
        """复制 backend、来源、图和领域历史，后续 revision 不回写宿主。"""
        ontology = self.graph.ontology
        backend = clone_backend(ontology.backend)
        scoped = ScopedIdentityStore(backend)
        cloned_ontology = GraphOntology(
            backend,
            space_id=ontology.space_id,
            space_identity=ontology.space_identity,
            scoped_identities=scoped,
        )
        cloned_graph = ParserRevisionGraph(
            cloned_ontology, self.graph.protocol)
        cloned_ledger = self.ledger.clone()
        cloned_resolver = self.resolver.clone(ledger=cloned_ledger)
        return ParserRevisionRuntime(
            SourceRecordRepository(backend),
            cloned_graph,
            cloned_ledger,
            cloned_resolver,
        )

    def _validate_transaction_sinks(self) -> None:
        """只放行纯内存状态或与图共享可恢复 Core backend 的统一训练 sink。"""
        ledger_sink = self.ledger.event_sink
        resolver_sink = self.resolver.event_sink
        if ledger_sink is None and resolver_sink is None:
            return
        if ledger_sink is None or resolver_sink is not ledger_sink:
            raise ParserRevisionRuntimeError(
                "A-03 H-00/H-04 必须共用同一训练历史 sink")
        history = getattr(ledger_sink, "history", None)
        sink_backend = getattr(
            ledger_sink, "backend", getattr(history, "backend", None))
        if sink_backend is not self.graph.ontology.backend:
            raise ParserRevisionRuntimeError(
                "A-03 sink 必须绑定 revision 图的可恢复 Core backend")

    def _validate_sources(self, request: ParserRevisionRequest) -> None:
        """要求新旧 SourceRecord 均已留档，且原文逐码点完全一致。"""
        old_record = self.source_records.find(request.old_source.stable_key())
        new_record = self.source_records.find(request.new_source.stable_key())
        if old_record is None or new_record is None:
            raise ParserRevisionRuntimeError(
                "A-03 新旧 ParserVersion 都必须已有 SourceRecord")
        if old_record.raw_text != new_record.raw_text:
            raise ParserRevisionRuntimeError(
                "A-03 新旧 ParserVersion 原文逐码点不一致")

    def _validate_lineage(
            self, request: ParserRevisionRequest, exact_replay: bool,
            ) -> None:
        """要求请求接在唯一 current head，或精确命中已经提交的同一 revision。"""
        edges = tuple(
            item for item in self.graph.lineages()
            if parser_lineage_key(item.old_source)
            == parser_lineage_key(request.old_source)
        )
        revision = request.revision_identity(self.graph.protocol)
        matches = tuple(item for item in edges if item.revision == revision)
        if exact_replay:
            if (len(matches) != 1
                    or matches[0].old_source != request.old_source
                    or matches[0].new_source != request.new_source):
                raise ParserRevisionRuntimeError(
                    "A-03 已有 revision 没有唯一精确 lineage 边")
            return
        if matches:
            raise ParserRevisionRuntimeError(
                "A-03 新提交意外命中既有 revision lineage")
        if not edges:
            return
        outgoing = {item.old_source: item.new_source for item in edges}
        incoming = {item.new_source: item.old_source for item in edges}
        nodes = set(outgoing) | set(incoming)
        roots = tuple(item for item in nodes if item not in incoming)
        heads = tuple(item for item in nodes if item not in outgoing)
        if (len(roots) != 1 or len(heads) != 1
                or len(nodes) != len(edges) + 1):
            raise ParserRevisionRuntimeError(
                "A-03 parser lineage 无唯一 current")
        if request.old_source != heads[0]:
            raise ParserRevisionRuntimeError(
                "A-03 只能从当前 ParserVersion head 继续")
        if request.new_source in nodes:
            raise ParserRevisionRuntimeError(
                "A-03 新 ParserVersion 已在当前 lineage 中")

    def _validate_materialized_inputs(
            self, request: ParserRevisionRequest) -> None:
        """要求 anchor/Hypothesis 均由真实上游先行物化，禁止 revision 伪造产物。"""
        identities = []
        for mapping in request.anchors:
            identities.append(mapping.old)
            identities.extend(mapping.replacements)
        for mapping in request.hypotheses:
            identities.append(mapping.old.object_identity())
            identities.extend(
                item.object_identity() for item in mapping.replacements)
        for identity in identities:
            if self.graph.ontology.resolve(identity) is None:
                raise ParserRevisionRuntimeError(
                    "A-03 anchor/Hypothesis 尚未由上游物化")

    def _validate_live_history(
            self, request: ParserRevisionRequest,
            ) -> tuple[tuple[ParserHypothesisRevision, ...], ...]:
        """核验旧 active 竞争组完整列出，且全部新候选有未撤销 Evidence。"""
        by_old = {item.old: item for item in request.hypotheses}
        for mapping in request.hypotheses:
            if not self.ledger.has_hypothesis(mapping.old):
                raise ParserRevisionRuntimeError(
                    "A-03 old Hypothesis 未在 ledger 登记")
            snapshot = self.ledger.snapshot(mapping.old)
            if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                raise ParserRevisionRuntimeError(
                    "A-03 非重放 old Hypothesis 必须仍为 active")
            if mapping.refute.timestamp_seq > request.timestamp_seq:
                raise ParserRevisionRuntimeError(
                    "A-03 提交逻辑序早于 refute Evidence")
            for replacement in mapping.replacements:
                if not self.ledger.has_hypothesis(replacement):
                    raise ParserRevisionRuntimeError(
                        "A-03 new Hypothesis 未在 ledger 登记")
                new_snapshot = self.ledger.snapshot(replacement)
                active_evidence = (
                    *new_snapshot.support_evidence_ids,
                    *new_snapshot.refute_evidence_ids,
                    *new_snapshot.unknown_evidence_ids,
                )
                if (new_snapshot.lifecycle != LIFECYCLE_ACTIVE
                        or not active_evidence
                        or new_snapshot.epistemic_status == EPISTEMIC_REFUTED):
                    raise ParserRevisionRuntimeError(
                        "A-03 new Hypothesis 必须 active 且有非纯反驳 Evidence")

        groups: dict[
            tuple[HypothesisKey, ...], tuple[ParserHypothesisRevision, ...]
        ] = {}
        remaining = set(by_old)
        while remaining:
            anchor = min(remaining, key=HypothesisKey.stable_key)
            competition = self.ledger.competition(anchor)
            active = tuple(
                item.hypothesis for item in competition
                if item.lifecycle == LIFECYCLE_ACTIVE)
            if not active or any(item not in by_old for item in active):
                raise ParserRevisionRuntimeError(
                    "A-03 遗漏旧 competition 的 active 候选")
            groups[active] = tuple(by_old[item] for item in active)
            remaining.difference_update(active)
        return tuple(
            groups[key]
            for key in sorted(
                groups,
                key=lambda values: tuple(
                    item.stable_key() for item in values),
            )
        )

    @staticmethod
    def _apply_history(
            request: ParserRevisionRequest,
            groups: tuple[tuple[ParserHypothesisRevision, ...], ...],
            ledger: HypothesisLedger,
            resolver: HypothesisResolver,
            ) -> tuple[ResolverDecision, ...]:
        """追加新来源 refute，并按旧竞争组用 archive 而非跨版本 replacement 退出。"""
        for mapping in request.hypotheses:
            ledger.append_evidence(mapping.refute)
        decisions = []
        for group in groups:
            anchor = min(
                (item.old for item in group),
                key=HypothesisKey.stable_key,
            )
            archives = tuple(
                ArchiveDirective(item.old, item.refute.evidence_id)
                for item in sorted(
                    group, key=lambda value: value.old.stable_key())
            )
            decisions.append(resolver.resolve(
                anchor,
                timestamp_seq=request.timestamp_seq,
                archives=archives,
            ))
        return tuple(decisions)

    def _validate_exact_history(
            self, request: ParserRevisionRequest,
            ) -> tuple[ResolverDecision, ...]:
        """核验已有 revision 的 refute、archive 和 decision 已完整存在且不追加事件。"""
        for mapping in request.hypotheses:
            if not self.ledger.has_hypothesis(mapping.old):
                raise ParserRevisionRuntimeError(
                    "A-03 重放缺少 old Hypothesis")
            snapshot = self.ledger.snapshot(mapping.old)
            if snapshot.lifecycle != LIFECYCLE_ARCHIVED:
                raise ParserRevisionRuntimeError(
                    "A-03 已有 revision 的 old Hypothesis 未归档")
            evidence = tuple(
                item for item in self.ledger.evidence_history(mapping.old)
                if item.evidence_id == mapping.refute.evidence_id)
            if evidence != (mapping.refute,):
                raise ParserRevisionRuntimeError(
                    "A-03 已有 revision 的 refute Evidence 漂移")
            transitions = tuple(
                item for item in self.ledger.transition_history(mapping.old)
                if (item.to_state == LIFECYCLE_ARCHIVED
                    and item.reason_evidence_id == mapping.refute.evidence_id
                    and item.timestamp_seq == request.timestamp_seq
                    and item.replacement is None))
            if len(transitions) != 1:
                raise ParserRevisionRuntimeError(
                    "A-03 已有 revision 的 archive transition 漂移")
        groups: dict[tuple[HypothesisKey, ...], HypothesisKey] = {}
        remaining = {item.old for item in request.hypotheses}
        while remaining:
            anchor = min(remaining, key=HypothesisKey.stable_key)
            competition = tuple(
                item.hypothesis for item in self.ledger.competition(anchor))
            mapped = tuple(item for item in competition if item in remaining)
            if not mapped:
                raise ParserRevisionRuntimeError(
                    "A-03 重放无法恢复旧 competition")
            groups[competition] = anchor
            remaining.difference_update(mapped)

        decisions = []
        for competition in sorted(
                groups,
                key=lambda values: tuple(
                    item.stable_key() for item in values)):
            history = tuple(
                item for item in self.resolver.decision_history(
                    groups[competition])
                if item.timestamp_seq == request.timestamp_seq
            )
            matching = tuple(
                item for item in history
                if all(any(
                    trace.hypothesis == mapping.old
                    and trace.after.lifecycle == LIFECYCLE_ARCHIVED
                    and trace.transition_event_id > 0
                    for trace in item.candidates)
                    for mapping in request.hypotheses
                    if mapping.old in competition)
            )
            if len(matching) != 1:
                raise ParserRevisionRuntimeError(
                    "A-03 已有 revision 的 H-04 decision 漂移")
            decisions.append(matching[0])
        return tuple(decisions)


__all__ = [
    "ParserRevisionResult",
    "ParserRevisionRuntime",
    "ParserRevisionRuntimeError",
    "ParserRevisionTrainingSink",
]
