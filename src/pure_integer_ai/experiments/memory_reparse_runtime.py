"""A-08 消费 ParserRevision 并重建长期 Memory 当前派生的协调入口。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.hypothesis import (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_HYPOTHESIS,
    INTAKE_DERIVED_MANIFEST,
    INTAKE_OUTCOME_SUCCESS,
    MEMORY_EVENT_DERIVATION,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_USE,
    DerivationTransitionPayload,
    HypothesisPayload,
    IntakeManifestPayload,
    MemoryObjectRef,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.parser_revision import (
    MaterializedParserRevision,
    ParserRevisionGraph,
    ParserRevisionRequest,
    parser_lineage_key,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    MemoryIntakeParser,
    MemoryIntakeResult,
    MemorySourceIntake,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.train_context import TrainContext


_A08_VERSION = 1


class MemoryParserRevisionError(RuntimeError):
    """A-08 图、manifest、映射、事务或局部派生违反契约。"""


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """为复合稳定键添加长度边界。"""
    return len(key), *key


def _access(source: SourceRef) -> MemoryAccessContext:
    """从来源 owner 构造同层 Memory ACL。"""
    owner = source.owner
    return MemoryAccessContext(
        owner.tenant_id,
        owner.user_id,
        owner.session_id,
    )


@dataclass(frozen=True)
class _ManifestHypothesis:
    """manifest 中一个 Hypothesis 的完整键、引用和派生 lineage。"""

    hypothesis: HypothesisKey
    reference: MemoryObjectRef
    lineage_key: tuple[int, ...]


@dataclass(frozen=True)
class _MemoryHistorySnapshot:
    """A-08 首写前需要保持不变的旧事件和 Use 计数。"""

    events: tuple[MaterializedMemoryEvent, ...]
    use_counts: tuple[tuple[MemoryObjectRef, int], ...]


@dataclass(frozen=True)
class MemoryParserRevisionResult:
    """一次长期重解析提交或精确重放的可审计结果。"""

    revision: MaterializedParserRevision
    intake: MemoryIntakeResult
    old_hypothesis_refs: tuple[MemoryObjectRef, ...]
    new_hypothesis_refs: tuple[MemoryObjectRef, ...]
    preserved_use_refs: tuple[MemoryObjectRef, ...]
    replayed: bool

    def __post_init__(self) -> None:
        """核验结果只携带完整对象引用和严格重放标记。"""
        if not isinstance(self.revision, MaterializedParserRevision):
            raise TypeError("A-08 result revision 类型错误")
        if not isinstance(self.intake, MemoryIntakeResult):
            raise TypeError("A-08 result intake 类型错误")
        for name, refs in (
                ("old_hypothesis_refs", self.old_hypothesis_refs),
                ("new_hypothesis_refs", self.new_hypothesis_refs),
                ("preserved_use_refs", self.preserved_use_refs)):
            if (not isinstance(refs, tuple)
                    or any(not isinstance(item, MemoryObjectRef)
                           for item in refs)):
                raise TypeError(f"A-08 result {name} 类型错误")
            if len(set(refs)) != len(refs):
                raise ValueError(f"A-08 result {name} 含重复引用")
        if type(self.replayed) is not bool:
            raise TypeError("A-08 result replayed 必须是严格 bool")

    def stable_key(self) -> tuple[int, ...]:
        """返回 revision、摄入对象、Use 和重放状态的确定性键。"""
        result = [
            _A08_VERSION,
            *_pack(self.revision.stable_key()),
            *_pack(self.intake.manifest_ref.stable_key()),
            self.intake.outcome_kind,
        ]
        for refs in (
                self.old_hypothesis_refs,
                self.new_hypothesis_refs,
                self.preserved_use_refs):
            result.append(len(refs))
            for ref in refs:
                result.extend(_pack(ref.stable_key()))
        result.append(int(self.replayed))
        return tuple(result)


class _ValidatedReparseParser:
    """在 M-05 首个 Memory event 前核验 A-03 与 manifest 映射基数。"""

    def __init__(
            self,
            runtime: "MemoryParserRevisionRuntime",
            parser: MemoryIntakeParser,
            request: ParserRevisionRequest,
            old_manifest: IntakeManifestPayload,
            ) -> None:
        """绑定实际 parser 和本次不可变 revision/旧 manifest。"""
        if not callable(getattr(parser, "parse", None)):
            raise TypeError("A-08 parser 必须实现 parse")
        self.runtime = runtime
        self.parser = parser
        self.request = request
        self.old_manifest = old_manifest
        self.calls = 0

    def parse(self, source_slice) -> ObservationIntakeDraft:
        """运行一次真实 parser，并在返回 M-05 前完成映射预检。"""
        self.calls += 1
        if self.calls != 1:
            raise MemoryParserRevisionError("A-08 parser 不得重复执行")
        draft = self.parser.parse(source_slice)
        if not isinstance(draft, ObservationIntakeDraft):
            raise MemoryParserRevisionError(
                "A-08 已提交 revision 必须产生成功 Observation 草案")
        self.runtime._validate_draft(
            self.request, self.old_manifest, draft)
        return draft


class MemoryParserRevisionRuntime:
    """协调 A-03 图、M-05/M-10 摄入和 M-04 局部派生更新。"""

    def __init__(
            self,
            ctx: TrainContext,
            graph: ParserRevisionGraph,
            intake: MemorySourceIntake,
            ) -> None:
        """绑定同一可恢复 backend 上的 Core 图和一个长期 Memory 通道。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("A-08 ctx 必须是 TrainContext")
        if not isinstance(graph, ParserRevisionGraph):
            raise TypeError("A-08 graph 必须是 ParserRevisionGraph")
        if not isinstance(intake, MemorySourceIntake):
            raise TypeError("A-08 intake 必须是 MemorySourceIntake")
        if graph.ontology is not ctx.graph_ontology:
            raise ValueError("A-08 graph 必须绑定当前 TrainContext ontology")
        if intake is ctx.memory_read_intake:
            aggregates = ctx.memory_read_aggregates
            channel = 1
        elif intake is ctx.memory_interact_intake:
            aggregates = ctx.memory_interact_aggregates
            channel = 2
        else:
            raise ValueError("A-08 intake 不属于当前 TrainContext")
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise ValueError("A-08 缺少当前 Memory aggregate")
        if (intake.event_log is not aggregates.event_log
                or intake.batch_runtime is None
                or intake.batch_runtime.event_log is not intake.event_log
                or intake.batch_runtime.aggregates is not aggregates):
            raise ValueError("A-08 必须绑定 M-10 与同一 M-04 aggregate")
        backend = ctx.backend
        if any(item is not backend for item in (
                graph.ontology.backend,
                intake.event_log.backend,
                intake.source_intake.repository.backend,
                intake.batch_runtime.event_log.backend)):
            raise ValueError("A-08 图、来源、Memory 和 M-10 必须共享 backend")
        self.ctx = ctx
        self.graph = graph
        self.intake = intake
        self.aggregates = aggregates
        self._channel = channel

    def apply(
            self,
            request: ParserRevisionRequest,
            *,
            raw_text: str,
            license_id: str,
            batch_id: int,
            parser: MemoryIntakeParser,
            materialize: Callable[[ObservationIntakeDraft],
                                  ObservationIntakeDraft] | None = None,
            batch_fault_injector=None,
            ) -> MemoryParserRevisionResult:
        """预检已提交 revision，执行完整 reparse，并核验当前派生和旧历史。"""
        if not isinstance(request, ParserRevisionRequest):
            raise TypeError("A-08 request 必须是 ParserRevisionRequest")
        if not isinstance(raw_text, str):
            raise TypeError("A-08 raw_text 必须是字符串")
        if not isinstance(license_id, str) or not license_id:
            raise ValueError("A-08 license_id 必须非空")
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("A-08 batch_id 必须是严格正整数")
        if not callable(getattr(parser, "parse", None)):
            raise TypeError("A-08 parser 必须实现 parse")
        revision = self._require_committed_revision(request)
        self._validate_source_records(
            request,
            raw_text=raw_text,
            license_id=license_id,
            batch_id=batch_id,
        )
        access = _access(request.old_source)
        self.aggregates.require_clean(access=access)
        old_manifest = self.intake.manifest(request.old_source)
        if old_manifest is None:
            raise MemoryParserRevisionError(
                "A-08 旧 ParserVersion 没有长期 Memory manifest")

        existing = self.intake.manifest(request.new_source)
        if existing is not None:
            result = self.intake.result_for_source(request.new_source)
            return self._validated_result(
                revision,
                request,
                old_manifest,
                result,
                replayed=True,
            )

        current = self.intake.require_current_manifest(request.old_source)
        if current != old_manifest:
            raise MemoryParserRevisionError("A-08 旧 manifest current 状态漂移")
        snapshot = self._snapshot_old_history(old_manifest, access=access)
        checked_parser = _ValidatedReparseParser(
            self, parser, request, old_manifest)
        result = self.intake.ingest(
            request.new_source,
            raw_text,
            license_id=license_id,
            batch_id=batch_id,
            parser=checked_parser,
            supersedes_source=request.old_source,
            materialize=materialize,
            failure_classifier=self._reject_parse_failure,
            batch_fault_injector=batch_fault_injector,
        )
        validated = self._validated_result(
            revision,
            request,
            old_manifest,
            result,
            replayed=checked_parser.calls == 0,
        )
        self._require_snapshot_unchanged(snapshot, access=access)
        return validated

    @staticmethod
    def _reject_parse_failure(exc: Exception):
        """A-03 已提交后拒绝把 parser/映射异常降格为长期 ParseFailure。"""
        if isinstance(exc, MemoryParserRevisionError):
            raise exc
        raise MemoryParserRevisionError(
            "A-08 已提交 revision 的 parser 执行失败") from exc

    def clone_for_evaluation(self) -> "MemoryParserRevisionRuntime":
        """克隆完整 backend/context，并在隔离 Memory 通道重建 A-08 runtime。"""
        backend = clone_backend(self.ctx.backend)
        cloned = clone_train_context(
            self.ctx,
            backend,
            label="memory-parser-revision",
        )
        graph = ParserRevisionGraph(
            cloned.graph_ontology,
            self.graph.protocol,
        )
        intake = (
            cloned.memory_read_intake
            if self._channel == 1
            else cloned.memory_interact_intake
        )
        return MemoryParserRevisionRuntime(cloned, graph, intake)

    def state_key(self) -> tuple[int, ...]:
        """返回图协议、Memory 空间和 M-10 依赖的完整装配键。"""
        batch = self.intake.batch_runtime
        if batch is None:
            raise MemoryParserRevisionError("A-08 M-10 runtime 已脱离")
        return (
            _A08_VERSION,
            self._channel,
            *_pack(self.graph.protocol.stable_key()),
            *_pack(self.intake.event_log.memory_space_identity.stable_key()),
            *_pack(batch.core_dependency.stable_key()),
            batch.write_budget.object_limit,
            batch.write_budget.byte_limit,
        )

    def _require_committed_revision(
            self,
            request: ParserRevisionRequest,
            ) -> MaterializedParserRevision:
        """要求图中存在精确 revision，且其新来源仍是当前 lineage head。"""
        materialized = self.graph.preflight(request)
        if materialized is None:
            raise MemoryParserRevisionError(
                "A-08 不得消费尚未由 A-03 提交的 revision")
        edges = tuple(
            item for item in self.graph.lineages()
            if parser_lineage_key(item.old_source)
            == parser_lineage_key(request.old_source)
        )
        matches = tuple(
            item for item in edges
            if (item.revision == materialized.revision
                and item.old_source == request.old_source
                and item.new_source == request.new_source)
        )
        if len(matches) != 1:
            raise MemoryParserRevisionError(
                "A-08 revision 没有唯一精确 lineage 边")
        if any(item.old_source == request.new_source for item in edges):
            raise MemoryParserRevisionError(
                "A-08 不得对已非 current head 的 parser revision 补写 Memory")
        return materialized

    def _validate_source_records(
            self,
            request: ParserRevisionRequest,
            *,
            raw_text: str,
            license_id: str,
            batch_id: int,
            ) -> None:
        """核验新旧完整 SourceRecord 共用同一原文、许可和 Companion。"""
        repository = self.intake.source_intake.repository
        old = repository.find(request.old_source.stable_key())
        new = repository.find(request.new_source.stable_key())
        if old is None or new is None:
            raise MemoryParserRevisionError(
                "A-08 新旧 ParserVersion 必须已有 SourceRecord")
        if (old.raw_text != new.raw_text
                or new.raw_text != raw_text):
            raise MemoryParserRevisionError(
                "A-08 新旧 SourceRecord 原文逐码点不一致")
        if (not old.metadata_complete
                or not new.metadata_complete
                or old.license_id != license_id
                or new.license_id != license_id
                or new.batch_id != batch_id):
            raise MemoryParserRevisionError(
                "A-08 SourceRecord 许可、批次或 Companion metadata 漂移")
        old_assoc = (
            old.companion_type_hash,
            old.companion_name_hash,
            old.companion_assoc_id,
        )
        new_assoc = (
            new.companion_type_hash,
            new.companion_name_hash,
            new.companion_assoc_id,
        )
        if old_assoc != new_assoc:
            raise MemoryParserRevisionError(
                "A-08 新旧 ParserVersion 未共享同一 Companion 原文")

    def _validate_draft(
            self,
            request: ParserRevisionRequest,
            old_manifest: IntakeManifestPayload,
            draft: ObservationIntakeDraft,
            ) -> None:
        """核验草案覆盖 Memory 内 A-03 目标，并正确降阶多对多 mapping。"""
        old_items = {
            item.hypothesis: item
            for item in self._manifest_hypotheses(old_manifest)
        }
        new_items = {}
        for item in draft.hypotheses:
            if not isinstance(item, HypothesisIntakeDraft):
                raise TypeError("A-08 draft hypothesis 类型错误")
            hypothesis = HypothesisKey(
                item.hypothesis_kind,
                item.candidate_key,
                item.competition_key,
                request.new_scope,
                request.new_source,
            )
            new_items[hypothesis] = item
        reverse_count: dict[HypothesisKey, int] = {}
        for mapping in request.hypotheses:
            for replacement in mapping.replacements:
                reverse_count[replacement] = (
                    reverse_count.get(replacement, 0) + 1)
        all_new_lineages = {
            item.lineage_key for item in draft.hypotheses
        }
        for mapping in request.hypotheses:
            old_item = old_items.get(mapping.old)
            if old_item is None:
                continue
            missing = tuple(
                item for item in mapping.replacements
                if item not in new_items
            )
            if missing:
                raise MemoryParserRevisionError(
                    "A-08 新 manifest 缺少 A-03 声明的 replacement")
            unique = (
                len(mapping.replacements) == 1
                and reverse_count[mapping.replacements[0]] == 1
            )
            if unique:
                replacement = new_items[mapping.replacements[0]]
                if replacement.lineage_key != old_item.lineage_key:
                    raise MemoryParserRevisionError(
                        "A-08 双向唯一 mapping 必须保留 M-05 lineage")
            elif old_item.lineage_key in all_new_lineages:
                raise MemoryParserRevisionError(
                    "A-08 删除、拆分或合并不得私选单个 Memory replacement")

    def _validated_result(
            self,
            revision: MaterializedParserRevision,
            request: ParserRevisionRequest,
            old_manifest: IntakeManifestPayload,
            result: MemoryIntakeResult,
            *,
            replayed: bool,
            ) -> MemoryParserRevisionResult:
        """核验提交后的 lifecycle、局部索引、Use 和 current 候选。"""
        if result.outcome_kind != INTAKE_OUTCOME_SUCCESS:
            raise MemoryParserRevisionError(
                "A-08 已提交 revision 不得形成 ParseFailure manifest")
        current = self.intake.require_current_manifest(request.new_source)
        if (current.source != request.new_source
                or current.batch_id != result.source_record.batch_id
                or current.supersedes_manifest_ref
                != self.intake.result_for_source(
                    request.old_source).manifest_ref):
            raise MemoryParserRevisionError(
                "A-08 新 manifest current、批次或前驱漂移")
        old_items = self._manifest_hypotheses(old_manifest)
        new_items = self._manifest_hypotheses(current)
        old_by_key = {item.hypothesis: item for item in old_items}
        new_by_key = {item.hypothesis: item for item in new_items}
        reverse_count: dict[HypothesisKey, int] = {}
        mappings = {item.old: item for item in request.hypotheses}
        for mapping in request.hypotheses:
            for replacement in mapping.replacements:
                reverse_count[replacement] = (
                    reverse_count.get(replacement, 0) + 1)

        access = _access(request.new_source)
        expected_targets = {
            item.object_ref for item in old_manifest.bindings
        }
        expected_targets.add(
            self.intake.result_for_source(request.old_source).manifest_ref)
        by_target: dict[MemoryObjectRef, list[DerivationTransitionPayload]] = {}
        for target in sorted(expected_targets, key=MemoryObjectRef.stable_key):
            entries = self.intake.event_log.query(
                access=access,
                event_kind=MEMORY_EVENT_DERIVATION,
                object_ref=target,
            )
            for entry in entries:
                payload = entry.event.payload
                if (isinstance(payload, DerivationTransitionPayload)
                        and payload.prior_source == request.old_source
                        and payload.replacement_source == request.new_source):
                    by_target.setdefault(
                        payload.target_ref, []).append(payload)
        if set(by_target) != expected_targets or any(
                len(items) != 1 for items in by_target.values()):
            raise MemoryParserRevisionError(
                "A-08 derivation 没有精确覆盖旧 manifest")

        use_refs = []
        for old_item in old_items:
            transition = by_target[old_item.reference][0]
            mapping = mappings.get(old_item.hypothesis)
            if mapping is not None:
                missing = tuple(
                    item for item in mapping.replacements
                    if item not in new_by_key
                )
                if missing:
                    raise MemoryParserRevisionError(
                        "A-08 已提交 manifest 缺少图中 replacement")
                unique = (
                    len(mapping.replacements) == 1
                    and reverse_count[mapping.replacements[0]] == 1
                )
                if unique:
                    expected_ref = new_by_key[
                        mapping.replacements[0]].reference
                    if (transition.to_state != LIFECYCLE_SUPERSEDED
                            or transition.replacement_ref != expected_ref):
                        raise MemoryParserRevisionError(
                            "A-08 一对一 Memory supersede 与图映射漂移")
                elif (transition.to_state != LIFECYCLE_ARCHIVED
                      or transition.replacement_ref is not None):
                    raise MemoryParserRevisionError(
                        "A-08 多对多 Memory 派生必须 archive")
            self.aggregates.require_hypothesis_clean(
                old_item.reference, access=access)
            aggregate = self.aggregates.read(
                old_item.reference, access=access)
            if aggregate is None or aggregate.lifecycle_state != (
                    transition.to_state):
                raise MemoryParserRevisionError(
                    "A-08 旧 Hypothesis aggregate lifecycle 未更新")
            indexed = self.aggregates.events(
                old_item.reference, access=access)
            uses = tuple(
                entry.event.object_ref for entry in indexed
                if (entry.event.event_kind == MEMORY_EVENT_USE
                    and isinstance(entry.event.payload, UsePayload)
                    and entry.event.payload.memory_ref
                    == old_item.reference)
            )
            if aggregate.use_count != len(uses):
                raise MemoryParserRevisionError(
                    "A-08 旧 Use 与 aggregate 计数漂移")
            use_refs.extend(uses)

        for new_item in new_items:
            self.aggregates.require_hypothesis_clean(
                new_item.reference, access=access)
            aggregate = self.aggregates.read(
                new_item.reference, access=access)
            if aggregate is None or aggregate.lifecycle_state != LIFECYCLE_ACTIVE:
                raise MemoryParserRevisionError(
                    "A-08 新 Hypothesis 未形成 active aggregate")
            self.aggregates.events(new_item.reference, access=access)
            self.aggregates.sources(new_item.reference, access=access)
        self.aggregates.require_clean(access=access)

        old_refs = tuple(sorted(
            (item.reference for item in old_items),
            key=MemoryObjectRef.stable_key,
        ))
        new_refs = tuple(sorted(
            (item.reference for item in new_items),
            key=MemoryObjectRef.stable_key,
        ))
        return MemoryParserRevisionResult(
            revision,
            result,
            old_refs,
            new_refs,
            tuple(sorted(set(use_refs), key=MemoryObjectRef.stable_key)),
            replayed,
        )

    def _manifest_hypotheses(
            self,
            manifest: IntakeManifestPayload,
            ) -> tuple[_ManifestHypothesis, ...]:
        """从 manifest 绑定和声明事件恢复完整 HypothesisKey。"""
        access = _access(manifest.source)
        result = []
        for binding in manifest.bindings:
            if binding.binding_kind != INTAKE_DERIVED_HYPOTHESIS:
                continue
            entries = self.intake.event_log.query(
                access=access,
                event_kind=MEMORY_EVENT_HYPOTHESIS,
                object_ref=binding.object_ref,
            )
            matches = tuple(
                item for item in entries
                if (item.event.object_ref == binding.object_ref
                    and isinstance(item.event.payload, HypothesisPayload))
            )
            if len(matches) != 1:
                raise MemoryParserRevisionError(
                    "A-08 manifest Hypothesis 没有唯一声明")
            result.append(_ManifestHypothesis(
                matches[0].event.payload.hypothesis,
                binding.object_ref,
                binding.lineage_key,
            ))
        if len({item.hypothesis for item in result}) != len(result):
            raise MemoryParserRevisionError(
                "A-08 manifest 含重复 Hypothesis 身份")
        return tuple(sorted(
            result,
            key=lambda item: item.hypothesis.stable_key(),
        ))

    def _snapshot_old_history(
            self,
            manifest: IntakeManifestPayload,
            *,
            access: MemoryAccessContext,
            ) -> _MemoryHistorySnapshot:
        """冻结旧声明、索引事件、Use 输出 Episode 和 aggregate Use 计数。"""
        indexed: dict[int, MaterializedMemoryEvent] = {}
        use_counts = []
        for ref in (
                *(item.object_ref for item in manifest.bindings),
                self.intake.result_for_source(manifest.source).manifest_ref):
            for entry in self.intake.event_log.query(
                    access=access, object_ref=ref):
                indexed[entry.event_hash] = entry
        for item in self._manifest_hypotheses(manifest):
            self.aggregates.require_hypothesis_clean(
                item.reference, access=access)
            aggregate = self.aggregates.read(item.reference, access=access)
            if aggregate is None:
                raise MemoryParserRevisionError(
                    "A-08 旧 Hypothesis 缺少干净 aggregate")
            use_counts.append((item.reference, aggregate.use_count))
            for entry in self.aggregates.events(
                    item.reference, access=access):
                indexed[entry.event_hash] = entry
                payload = entry.event.payload
                if isinstance(payload, UsePayload):
                    for episode in self.intake.event_log.query(
                            access=access,
                            object_ref=payload.episode_ref):
                        indexed[episode.event_hash] = episode
        return _MemoryHistorySnapshot(
            tuple(sorted(
                indexed.values(),
                key=lambda item: item.event_hash,
            )),
            tuple(sorted(
                use_counts,
                key=lambda item: item[0].stable_key(),
            )),
        )

    def _require_snapshot_unchanged(
            self,
            snapshot: _MemoryHistorySnapshot,
            *,
            access: MemoryAccessContext,
            ) -> None:
        """逐事件核验旧历史/输出未改写，并核对旧 Use 计数守恒。"""
        for expected in snapshot.events:
            actual = self.intake.event_log.read(
                expected.event_hash, access=access)
            if actual != expected:
                raise MemoryParserRevisionError(
                    "A-08 原地修改或隐藏了旧 Memory event")
        for ref, expected_count in snapshot.use_counts:
            aggregate = self.aggregates.read(ref, access=access)
            if aggregate is None or aggregate.use_count != expected_count:
                raise MemoryParserRevisionError(
                    "A-08 改变了旧 Hypothesis 的 Use 归因")


__all__ = [
    "MemoryParserRevisionError",
    "MemoryParserRevisionResult",
    "MemoryParserRevisionRuntime",
]
