"""断奶后来源到 Memory Observation/Hypothesis 的两阶段摄入协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    TypedRef,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.memory_event import (
    DerivationTransitionPayload,
    EvidencePayload,
    INTAKE_DERIVED_EVIDENCE,
    INTAKE_DERIVED_FAILURE,
    INTAKE_DERIVED_HYPOTHESIS,
    INTAKE_DERIVED_MANIFEST,
    INTAKE_DERIVED_OBSERVATION,
    INTAKE_OUTCOME_FAILURE,
    INTAKE_OUTCOME_SUCCESS,
    IntakeDerivedBinding,
    IntakeManifestPayload,
    MEMORY_EVENT_DERIVATION,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_INTAKE_MANIFEST,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_PARSE_FAILURE,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_INTAKE_MANIFEST,
    MEMORY_OBJECT_OBSERVATION,
    MEMORY_OBJECT_PARSE_FAILURE,
    MemoryEvent,
    MemoryLinkedRef,
    MemoryObjectRef,
    ObservationPayload,
    ParseFailurePayload,
    HypothesisPayload,
    RETENTION_EPISODIC,
    memory_object_ref,
    source_reparse_lineage_key,
)
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchFaultInjector,
    MemoryBatchRuntime,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_LIFECYCLE,
    CLOCK_MEMORY_OBSERVED,
    LogicalClockIdentity,
    LogicalTimestamp,
    document_scope,
)
from pure_integer_ai.cognition.understanding.source_intake import (
    SourceIntake,
    SourceSlice,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.write_guard import forbid_backend_writes
from pure_integer_ai.storage.source_record import SourceRecordStorage
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.memory_batch import source_record_dependency
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
    SpaceRegistry,
)


INTAKE_CHANNEL_READING = 1
INTAKE_CHANNEL_INTERACTION = 2
_VALID_CHANNELS = frozenset({
    INTAKE_CHANNEL_READING,
    INTAKE_CHANNEL_INTERACTION,
})


class MemoryIntakeIntegrityError(RuntimeError):
    """来源摄入的阶段、ACL、事件或版本谱系不一致。"""


@dataclass(frozen=True)
class MemoryIntakePolicy:
    """共享摄入协议的通道和 owner visibility 约束。"""

    channel: int
    accepted_visibilities: tuple[int, ...]
    memory_space: SpaceIdentity

    def __post_init__(self) -> None:
        """核验通道和 visibility 集合为明确的基础设施策略。"""
        if self.channel not in _VALID_CHANNELS:
            raise ValueError("MemoryIntakePolicy.channel 未注册")
        if (not isinstance(self.accepted_visibilities, tuple)
                or not self.accepted_visibilities
                or any(value not in {
                    VISIBILITY_GLOBAL,
                    VISIBILITY_TENANT,
                    VISIBILITY_USER,
                    VISIBILITY_SESSION,
                } for value in self.accepted_visibilities)):
            raise ValueError("MemoryIntakePolicy visibility 策略非法")
        if (self.channel == INTAKE_CHANNEL_READING
                and VISIBILITY_SESSION in self.accepted_visibilities):
            raise ValueError("阅读层不得接受 session visibility")
        if (self.channel == INTAKE_CHANNEL_INTERACTION
                and self.accepted_visibilities != (VISIBILITY_SESSION,)):
            raise ValueError("交互层必须只接受 session visibility")
        if (not isinstance(self.memory_space, SpaceIdentity)
                or self.memory_space.space_type != SPACE_TYPE_MEMORY):
            raise ValueError("MemoryIntakePolicy 缺少稳定 Memory 空间身份")

    def accepts(self, source: SourceRef) -> bool:
        """判断来源 owner 是否能进入当前 Memory 通道。"""
        return source.owner.visibility in self.accepted_visibilities


def reading_intake_policy() -> MemoryIntakePolicy:
    """返回拒绝 session 来源的阅读层策略。"""
    return MemoryIntakePolicy(
        INTAKE_CHANNEL_READING,
        (VISIBILITY_GLOBAL, VISIBILITY_TENANT, VISIBILITY_USER),
        SpaceRegistry.identity_for(SPACE_TYPE_MEMORY, "memory_read"),
    )


def interaction_intake_policy() -> MemoryIntakePolicy:
    """返回只接受 session 来源的交互层策略。"""
    return MemoryIntakePolicy(
        INTAKE_CHANNEL_INTERACTION,
        (VISIBILITY_SESSION,),
        SpaceRegistry.identity_for(SPACE_TYPE_MEMORY, "memory_interact"),
    )


@dataclass(frozen=True)
class HypothesisIntakeDraft:
    """一个来源内唯一候选及其单条来源 Evidence 草案。"""

    lineage_key: tuple[int, ...]
    hypothesis_kind: tuple[int, ...]
    candidate_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    stance: int
    signal_ref: MemoryLinkedRef | None = None
    reason_key: tuple[int, ...] = ()
    detail: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验候选身份开放、证据方向已注册且不存在隐式频次计数。"""
        for name, value in (
                ("lineage_key", self.lineage_key),
                ("hypothesis_kind", self.hypothesis_kind),
                ("candidate_key", self.candidate_key),
                ("competition_key", self.competition_key)):
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"{name} 必须是非空整数 tuple")
            assert_int(*value, _where=f"HypothesisIntakeDraft.{name}")
            if any(type(item) is not int for item in value):
                raise ValueError(f"{name} 必须使用严格整数")
        for name, value in (("reason_key", self.reason_key),
                            ("detail", self.detail)):
            if not isinstance(value, tuple):
                raise ValueError(f"{name} 必须是整数 tuple")
            if value:
                assert_int(*value, _where=f"HypothesisIntakeDraft.{name}")
                if any(type(item) is not int for item in value):
                    raise ValueError(f"{name} 必须使用严格整数")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("HypothesisIntakeDraft.stance 未注册")
        if (self.signal_ref is not None
                and not isinstance(self.signal_ref, MemoryLinkedRef)):
            raise TypeError("HypothesisIntakeDraft.signal_ref 类型错误")
        if (self.signal_ref is None) == (not self.reason_key):
            raise ValueError("每条来源 Evidence 必须且只能有 signal 或 reason")


@dataclass(frozen=True)
class ObservationIntakeDraft:
    """解析成功后、Memory 写入前的来源化 Observation 草案。"""

    lineage_key: tuple[int, ...]
    context: MemoryLinkedRef
    concept_refs: tuple[TypedRef, ...] = ()
    ordered_refs: tuple[TypedRef, ...] = ()
    structure_ref: TypedRef | None = None
    proposition_refs: tuple[TypedRef, ...] = ()
    relation_occurrences: tuple[MemoryLinkedRef, ...] = ()
    hypotheses: tuple[HypothesisIntakeDraft, ...] = ()

    def __post_init__(self) -> None:
        """核验 Observation 草案只携带已分型引用，不写入任何 Core。"""
        if not isinstance(self.lineage_key, tuple) or not self.lineage_key:
            raise ValueError("ObservationIntakeDraft.lineage_key 非法")
        assert_int(*self.lineage_key, _where="ObservationIntakeDraft.lineage_key")
        if not isinstance(self.context, MemoryLinkedRef):
            raise TypeError("ObservationIntakeDraft.context 必须是一等引用")
        for name, refs, expected in (
                ("concept_refs", self.concept_refs, TypedRef),
                ("ordered_refs", self.ordered_refs, TypedRef),
                ("proposition_refs", self.proposition_refs, TypedRef),
                ("relation_occurrences", self.relation_occurrences,
                 MemoryLinkedRef),
                ("hypotheses", self.hypotheses, HypothesisIntakeDraft)):
            if not isinstance(refs, tuple) or any(
                    not isinstance(ref, expected) for ref in refs):
                raise TypeError(f"ObservationIntakeDraft.{name} 类型错误")
        if (self.structure_ref is not None
                and not isinstance(self.structure_ref, TypedRef)):
            raise TypeError("ObservationIntakeDraft.structure_ref 类型错误")
        lineages = [item.lineage_key for item in self.hypotheses]
        if len(set(lineages)) != len(lineages):
            raise ValueError("同一来源不得重复声明 Hypothesis lineage")
        identities = [(
            item.hypothesis_kind,
            item.candidate_key,
            item.competition_key,
        ) for item in self.hypotheses]
        if len(set(identities)) != len(identities):
            raise ValueError("同一来源不得用不同 lineage 重复同一 Hypothesis")


@dataclass(frozen=True)
class ParseFailureDraft:
    """解析器在 Core 物化前返回的结构化失败，不含自由文本。"""

    lineage_key: tuple[int, ...]
    failure_kind: MemoryLinkedRef
    diagnostic_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验失败类别和诊断标识由调用方注入且可纯整数恢复。"""
        for name, value in (
                ("lineage_key", self.lineage_key),
                ("diagnostic_key", self.diagnostic_key)):
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"ParseFailureDraft.{name} 非法")
            assert_int(*value, _where=f"ParseFailureDraft.{name}")
            if any(type(item) is not int for item in value):
                raise ValueError(f"ParseFailureDraft.{name} 必须使用严格整数")
        if not isinstance(self.failure_kind, MemoryLinkedRef):
            raise TypeError("ParseFailureDraft.failure_kind 必须是一等引用")


class MemoryIntakeParser(Protocol):
    """只解析 SourceSlice，成功返回草案，失败返回 ParseFailureDraft。"""

    def parse(self, source: SourceSlice) -> ObservationIntakeDraft | ParseFailureDraft:
        """把来源切片转换为纯草案，不得写 Core 或 Memory。"""
        ...


@dataclass(frozen=True)
class MemoryIntakeResult:
    """一次幂等摄入的来源、manifest、派生对象和替代记录。"""

    source_record: SourceRecordStorage
    manifest_ref: MemoryObjectRef
    outcome_kind: int
    observation_ref: MemoryObjectRef | None
    hypothesis_refs: tuple[MemoryObjectRef, ...]
    evidence_refs: tuple[MemoryObjectRef, ...]
    failure_ref: MemoryObjectRef | None
    superseded_refs: tuple[MemoryObjectRef, ...]


class MemorySourceIntake:
    """执行 SourceRecord、纯解析、Memory 事件和显式 reparse 替代。"""

    def __init__(
            self, source_intake: SourceIntake,
            event_log: MemoryEventLog,
            policy: MemoryIntakePolicy,
            batch_runtime: MemoryBatchRuntime | None = None,
            ) -> None:
        if not isinstance(source_intake, SourceIntake):
            raise TypeError("source_intake 必须是 SourceIntake")
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 必须是 MemoryEventLog")
        if not isinstance(policy, MemoryIntakePolicy):
            raise TypeError("policy 必须是 MemoryIntakePolicy")
        if source_intake.repository.backend is not event_log.backend:
            raise MemoryIntakeIntegrityError("来源和 Memory 必须使用同一 backend")
        if event_log.memory_space_identity != policy.memory_space:
            raise MemoryIntakeIntegrityError("摄入通道与 Memory 空间身份不一致")
        self.source_intake = source_intake
        self.event_log = event_log
        self.policy = policy
        self.batch_runtime: MemoryBatchRuntime | None = None
        if batch_runtime is not None:
            self.attach_batch_runtime(batch_runtime)

    def attach_batch_runtime(self, runtime: MemoryBatchRuntime) -> None:
        """安装与当前 event log 相同空间的 M-10 K-02 批次 runtime。"""
        if not isinstance(runtime, MemoryBatchRuntime):
            raise TypeError("runtime 必须是 MemoryBatchRuntime")
        if runtime.event_log is not self.event_log:
            raise MemoryIntakeIntegrityError(
                "Memory intake batch runtime 绑定了其他 event log")
        if self.batch_runtime is not None and self.batch_runtime is not runtime:
            raise MemoryIntakeIntegrityError(
                "Memory intake 已绑定其他 batch runtime")
        self.batch_runtime = runtime

    def ingest(
            self, source: SourceRef, raw_text: str, *, license_id: str,
            batch_id: int, parser: MemoryIntakeParser,
            supersedes_source: SourceRef | None = None,
            materialize: Callable[[ObservationIntakeDraft],
                                  ObservationIntakeDraft] | None = None,
            failure_classifier: Callable[[Exception],
                                         ParseFailureDraft] | None = None,
            batch_fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> MemoryIntakeResult:
        """先保存来源，再运行纯解析，成功后才允许 Core 物化和 Memory 写入。"""
        self._validate_request(
            source, batch_id=batch_id, parser=parser,
            supersedes_source=supersedes_source)
        record = self.source_intake.ensure(
            source, raw_text, license_id=license_id, batch_id=batch_id)
        dependency = source_record_dependency(record)
        recovered = None
        if self.batch_runtime is not None:
            recovered = self.batch_runtime.recover_unit(
                source,
                batch_id,
                source_dependency=dependency,
                fault_injector=batch_fault_injector,
            )
        if recovered is None:
            existing = self._manifest_for(source)
        else:
            with self.batch_runtime.visibility.preview(recovered.batch_hash):
                existing = self._manifest_for(source)
        if existing is not None:
            self._validate_existing_predecessor(
                existing, supersedes_source=supersedes_source)
            return self._result_from_manifest(record, existing)

        prior = self._resolve_active_predecessor(
            source, supersedes_source=supersedes_source)
        pending_events = [] if self.batch_runtime is not None else None

        source_slice = self.source_intake.read_slice(
            source, 0, len(record.raw_text))
        try:
            outcome = parser.parse(source_slice)
        except Exception as exc:
            if failure_classifier is None:
                raise MemoryIntakeIntegrityError(
                    "parser 异常必须由调用方分类为 ParseFailureDraft") from exc
            outcome = failure_classifier(exc)
            if not isinstance(outcome, ParseFailureDraft):
                raise MemoryIntakeIntegrityError(
                    "failure_classifier 必须返回 ParseFailureDraft") from exc
        if isinstance(outcome, ParseFailureDraft):
            manifest, failure_ref = self._append_failure(
                source, batch_id, outcome, prior,
                pending_events=pending_events)
            replaced = self._supersede_prior(
                source, prior, manifest, new_bindings=manifest.bindings,
                pending_events=pending_events)
            self._publish_pending(
                source,
                batch_id,
                dependency,
                pending_events,
                fault_injector=batch_fault_injector,
            )
            return MemoryIntakeResult(
                record, self._manifest_ref(manifest),
                INTAKE_OUTCOME_FAILURE, None, (), (),
                failure_ref, replaced)
        if not isinstance(outcome, ObservationIntakeDraft):
            raise MemoryIntakeIntegrityError(
                "parser 必须返回 ObservationIntakeDraft 或 ParseFailureDraft")
        if materialize is not None:
            with forbid_backend_writes():
                outcome = materialize(outcome)
            if not isinstance(outcome, ObservationIntakeDraft):
                raise MemoryIntakeIntegrityError(
                    "Core materialize 必须返回 ObservationIntakeDraft")
        result = self._append_success(
            source,
            batch_id,
            outcome,
            prior,
            pending_events=pending_events,
        )
        replaced = self._supersede_prior(
            source, prior, result[0], new_bindings=result[1],
            pending_events=pending_events)
        self._publish_pending(
            source,
            batch_id,
            dependency,
            pending_events,
            fault_injector=batch_fault_injector,
        )
        return MemoryIntakeResult(
            record, self._manifest_ref(result[0]),
            INTAKE_OUTCOME_SUCCESS, result[2],
            result[3], result[4], None, replaced)

    def _validate_request(
            self, source: SourceRef, *, batch_id: int,
            parser: MemoryIntakeParser,
            supersedes_source: SourceRef | None) -> None:
        """核验来源版本、通道 ACL 和显式 parser 替代方向。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if source.versions.parser.value <= 0:
            raise ValueError("M-05 来源必须声明正 parser version")
        assert_int(batch_id, _where="MemorySourceIntake.batch_id")
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("M-05 batch_id 必须是严格正整数")
        if not callable(getattr(parser, "parse", None)):
            raise TypeError("parser 必须实现 parse")
        if not self.policy.accepts(source):
            raise MemoryIntakeIntegrityError(
                "来源 owner visibility 不符合当前 Memory 通道")
        if supersedes_source is not None:
            if not isinstance(supersedes_source, SourceRef):
                raise TypeError("supersedes_source 必须是 SourceRef")
            if (source_reparse_lineage_key(source)
                    != source_reparse_lineage_key(supersedes_source)
                    or source.versions.parser.value
                    <= supersedes_source.versions.parser.value):
                raise MemoryIntakeIntegrityError(
                    "supersedes_source 必须是同谱系更早 parser version")

    @staticmethod
    def _validate_existing_predecessor(
            manifest: IntakeManifestPayload, *,
            supersedes_source: SourceRef | None,
            ) -> None:
        """核验幂等重放声明的前驱与既有 manifest 完全相同。"""
        prior_ref = manifest.supersedes_manifest_ref
        expected = (
            None if prior_ref is None
            else SourceRef.from_stable_key(prior_ref.object_key)
        )
        if supersedes_source != expected:
            raise MemoryIntakeIntegrityError(
                "幂等摄入的 supersedes_source 与既有 manifest 漂移")

    def _resolve_active_predecessor(
            self, source: SourceRef, *,
            supersedes_source: SourceRef | None,
            ) -> IntakeManifestPayload | None:
        """恢复同谱系唯一活跃 manifest，并要求调用方显式指定该前驱。"""
        manifests: list[IntakeManifestPayload] = []
        for record in self.source_intake.repository.versions_for(
                source.stable_key()):
            candidate_source = SourceRef.from_stable_key(record.source_key)
            if candidate_source == source:
                continue
            candidate = self._manifest_for(candidate_source)
            if candidate is not None:
                manifests.append(candidate)

        if not manifests:
            if supersedes_source is not None:
                raise MemoryIntakeIntegrityError(
                    "重新解析必须显式指向已存在的旧 manifest")
            return None

        transitions = self.event_log.query(
            access=self._access(source),
            event_kind=MEMORY_EVENT_DERIVATION,
        )
        inactive_manifest_refs = {
            item.event.payload.target_ref
            for item in transitions
            if (isinstance(item.event.payload, DerivationTransitionPayload)
                and item.event.payload.binding_kind
                == INTAKE_DERIVED_MANIFEST
                and source_reparse_lineage_key(
                    item.event.payload.prior_source)
                == source_reparse_lineage_key(source))
        }
        active = tuple(
            item for item in manifests
            if self._manifest_ref(item) not in inactive_manifest_refs
        )
        if len(active) != 1:
            raise MemoryIntakeIntegrityError(
                "同一来源谱系没有唯一活跃 intake manifest")
        if supersedes_source is None:
            raise MemoryIntakeIntegrityError(
                "检测到旧 manifest，重新解析必须显式声明 supersedes_source")
        if active[0].source != supersedes_source:
            raise MemoryIntakeIntegrityError(
                "supersedes_source 不是同谱系当前活跃 manifest")
        return active[0]

    def _append_failure(
            self, source: SourceRef, batch_id: int,
            draft: ParseFailureDraft,
            prior: IntakeManifestPayload | None,
            *,
            pending_events: list[MemoryEvent] | None,
            ) -> tuple[IntakeManifestPayload, MemoryObjectRef]:
        """追加结构化 ParseFailure 和失败 manifest，不写 Observation/Core。"""
        scope = document_scope(source)
        clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
        failed_at = clock.advance()
        payload = ParseFailurePayload(
            source, draft.failure_kind, batch_id,
            source.versions.parser.value, draft.diagnostic_key, failed_at)
        failure_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_PARSE_FAILURE,
            payload.identity_key(),
            owner=source.owner,
            versions=source.versions,
        )
        self._emit_event(MemoryEvent(
            MEMORY_EVENT_PARSE_FAILURE, failure_ref, scope, payload),
            pending_events)
        binding = IntakeDerivedBinding(
            INTAKE_DERIVED_FAILURE, draft.lineage_key, failure_ref)
        manifest = self._append_manifest(
            source, batch_id, INTAKE_OUTCOME_FAILURE, (binding,), prior,
            clock, pending_events=pending_events)
        return manifest, failure_ref

    def _append_success(
            self, source: SourceRef, batch_id: int,
            draft: ObservationIntakeDraft,
            prior: IntakeManifestPayload | None,
            *,
            pending_events: list[MemoryEvent] | None,
            ) -> tuple[
                IntakeManifestPayload, tuple[IntakeDerivedBinding, ...],
                MemoryObjectRef, tuple[MemoryObjectRef, ...],
                tuple[MemoryObjectRef, ...]]:
        """追加 Observation、每候选一条 Evidence 和成功 manifest。"""
        scope = document_scope(source)
        observed_clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
        observation_payload = ObservationPayload(
            source,
            draft.context,
            draft.concept_refs,
            draft.ordered_refs,
            draft.structure_ref,
            draft.proposition_refs,
            draft.relation_occurrences,
            observed_clock.advance(),
        )
        observation_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_OBSERVATION,
            observation_payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        self._emit_event(MemoryEvent(
            MEMORY_EVENT_OBSERVATION, observation_ref, scope,
            observation_payload), pending_events)

        created_clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(scope, CLOCK_MEMORY_CREATED))
        hypothesis_refs: list[MemoryObjectRef] = []
        evidence_refs: list[MemoryObjectRef] = []
        bindings: list[IntakeDerivedBinding] = [IntakeDerivedBinding(
            INTAKE_DERIVED_OBSERVATION, draft.lineage_key, observation_ref)]
        for item in draft.hypotheses:
            hypothesis = HypothesisKey(
                item.hypothesis_kind,
                item.candidate_key,
                item.competition_key,
                scope,
                source,
            )
            hypothesis_payload = self._hypothesis_payload(
                hypothesis, created_clock.advance())
            hypothesis_ref = memory_object_ref(
                self.event_log.memory_space_identity,
                MEMORY_OBJECT_HYPOTHESIS,
                hypothesis.stable_key(),
                owner=source.owner,
                versions=source.versions,
            )
            self._emit_event(MemoryEvent(
                MEMORY_EVENT_HYPOTHESIS, hypothesis_ref, scope,
                hypothesis_payload), pending_events)
            evidence_payload = EvidencePayload(
                hypothesis_ref,
                item.stance,
                item.signal_ref,
                item.reason_key,
                source,
                None,
                item.detail,
                None,
                observed_clock.advance(),
            )
            evidence_ref = memory_object_ref(
                self.event_log.memory_space_identity,
                MEMORY_OBJECT_EVIDENCE,
                evidence_payload.stable_key(),
                owner=source.owner,
                versions=source.versions,
            )
            self._emit_event(MemoryEvent(
                MEMORY_EVENT_EVIDENCE, evidence_ref, scope,
                evidence_payload), pending_events)
            hypothesis_refs.append(hypothesis_ref)
            evidence_refs.append(evidence_ref)
            bindings.extend((
                IntakeDerivedBinding(
                    INTAKE_DERIVED_HYPOTHESIS, item.lineage_key,
                    hypothesis_ref),
                IntakeDerivedBinding(
                    INTAKE_DERIVED_EVIDENCE, item.lineage_key,
                    evidence_ref),
            ))
        manifest = self._append_manifest(
            source, batch_id, INTAKE_OUTCOME_SUCCESS, tuple(bindings), prior,
            observed_clock, pending_events=pending_events)
        return (
            manifest, tuple(bindings), observation_ref,
            tuple(hypothesis_refs), tuple(evidence_refs),
        )

    @staticmethod
    def _hypothesis_payload(hypothesis: HypothesisKey,
                            created_at: LogicalTimestamp):
        """构造不带统计派生字段的初始 Hypothesis 声明。"""
        return HypothesisPayload(
            hypothesis, RETENTION_EPISODIC, LIFECYCLE_ACTIVE, created_at)

    def _append_manifest(
            self, source: SourceRef, batch_id: int, outcome_kind: int,
            bindings: tuple[IntakeDerivedBinding, ...],
            prior: IntakeManifestPayload | None,
            clock, *,
            pending_events: list[MemoryEvent] | None,
            ) -> IntakeManifestPayload:
        """追加成功或失败 manifest，并把 batch/parser 绑定到同一事件链。"""
        scope = document_scope(source)
        prior_ref = None if prior is None else memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_INTAKE_MANIFEST,
            prior.identity_key(),
            owner=prior.source.owner,
            versions=prior.source.versions,
        )
        payload = IntakeManifestPayload(
            source,
            batch_id,
            source.versions.parser.value,
            outcome_kind,
            bindings,
            prior_ref,
            clock.advance(),
        )
        manifest_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_INTAKE_MANIFEST,
            payload.identity_key(),
            owner=source.owner,
            versions=source.versions,
        )
        self._emit_event(MemoryEvent(
            MEMORY_EVENT_INTAKE_MANIFEST, manifest_ref, scope, payload),
            pending_events)
        return payload

    def _supersede_prior(
            self, source: SourceRef,
            prior: IntakeManifestPayload | None,
            manifest: IntakeManifestPayload,
            *, new_bindings: tuple[IntakeDerivedBinding, ...],
            pending_events: list[MemoryEvent] | None,
            ) -> tuple[MemoryObjectRef, ...]:
        """按 manifest lineage 显式替代旧派生对象，缺 successor 则归档。"""
        if prior is None:
            return ()
        new_by_key = {
            (item.binding_kind, item.lineage_key): item.object_ref
            for item in new_bindings
        }
        old_manifest_ref = self._manifest_ref(prior)
        new_manifest_ref = self._manifest_ref(manifest)
        transitions: list[tuple[MemoryObjectRef, int, tuple[int, ...]]] = []
        for item in self._bindings_for_manifest(prior):
            replacement = new_by_key.get((item.binding_kind, item.lineage_key))
            to_state = (
                LIFECYCLE_SUPERSEDED
                if replacement is not None else LIFECYCLE_ARCHIVED)
            transitions.append((item.object_ref, to_state,
                                item.lineage_key))
        transitions.append((
            old_manifest_ref, LIFECYCLE_SUPERSEDED,
            source_reparse_lineage_key(prior.source),
        ))
        replaced: list[MemoryObjectRef] = []
        old_scope = document_scope(prior.source)
        lifecycle_clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(old_scope, CLOCK_MEMORY_LIFECYCLE))
        for target, to_state, lineage_key in transitions:
            if target.object_kind == MEMORY_OBJECT_INTAKE_MANIFEST:
                replacement = new_manifest_ref
            else:
                replacement = new_by_key.get((
                    self._binding_kind_for_ref(target), lineage_key))
            payload = DerivationTransitionPayload(
                target,
                LIFECYCLE_ACTIVE,
                to_state,
                replacement if to_state == LIFECYCLE_SUPERSEDED else None,
                prior.source,
                source,
                self._binding_kind_for_ref(target),
                lineage_key,
                lifecycle_clock.advance(),
            )
            self._emit_event(MemoryEvent(
                MEMORY_EVENT_DERIVATION,
                target,
                old_scope,
                payload,
            ), pending_events)
            replaced.append(target)
        return tuple(sorted(replaced, key=lambda ref: ref.stable_key()))

    def _emit_event(
            self,
            event: MemoryEvent,
            pending_events: list[MemoryEvent] | None,
            ) -> None:
        """未安装 M-10 时立即追加；安装后只收集到有序 staged 事件集。"""
        if pending_events is None:
            self.event_log.append(event)
            return
        pending_events.append(event)

    def _publish_pending(
            self,
            source: SourceRef,
            batch_id: int,
            dependency: SegmentDependency,
            pending_events: list[MemoryEvent] | None,
            *,
            fault_injector: MemoryBatchFaultInjector | None,
            ) -> None:
        """把本次构造的完整事件集交给 M-10 runtime，activation 后才返回。"""
        if self.batch_runtime is None:
            if pending_events is not None:
                raise MemoryIntakeIntegrityError(
                    "未安装 batch runtime 却存在 staged 事件")
            return
        if pending_events is None or not pending_events:
            raise MemoryIntakeIntegrityError("M-10 摄入没有形成事件集")
        self.batch_runtime.publish(
            source,
            batch_id,
            tuple(pending_events),
            source_dependency=dependency,
            fault_injector=fault_injector,
        )

    @staticmethod
    def _binding_kind_for_ref(ref: MemoryObjectRef) -> int:
        """把派生对象种类映射为 manifest 绑定种类。"""
        mapping = {
            MEMORY_OBJECT_OBSERVATION: INTAKE_DERIVED_OBSERVATION,
            MEMORY_OBJECT_HYPOTHESIS: INTAKE_DERIVED_HYPOTHESIS,
            MEMORY_OBJECT_EVIDENCE: INTAKE_DERIVED_EVIDENCE,
            MEMORY_OBJECT_PARSE_FAILURE: INTAKE_DERIVED_FAILURE,
            MEMORY_OBJECT_INTAKE_MANIFEST: INTAKE_DERIVED_MANIFEST,
        }
        try:
            return mapping[ref.object_kind]
        except KeyError as exc:
            raise MemoryIntakeIntegrityError(
                "manifest 不能替代未注册派生对象") from exc

    def _manifest_for(self, source: SourceRef) -> IntakeManifestPayload | None:
        """按完整来源版本读取唯一 manifest，不因查询登记新身份。"""
        manifest_ref = self._manifest_ref_for_source(source)
        entries = self.event_log.query(
            access=self._access(source),
            event_kind=MEMORY_EVENT_INTAKE_MANIFEST,
            object_ref=manifest_ref,
        )
        matches = tuple(
            entry.event.payload for entry in entries
            if isinstance(entry.event.payload, IntakeManifestPayload)
            and entry.event.payload.source == source
        )
        if len(matches) > 1:
            raise MemoryIntakeIntegrityError("同一来源版本存在多个 intake manifest")
        return matches[0] if matches else None

    @staticmethod
    def _bindings_for_manifest(
            manifest: IntakeManifestPayload,
            ) -> tuple[IntakeDerivedBinding, ...]:
        """返回 manifest 的不可变派生绑定并核验成功/失败结果。"""
        return manifest.bindings

    def _manifest_ref(self, manifest: IntakeManifestPayload) -> MemoryObjectRef:
        """把 manifest payload 身份映射为当前 Memory 空间引用。"""
        return self._manifest_ref_for_source(manifest.source)

    def _manifest_ref_for_source(self, source: SourceRef) -> MemoryObjectRef:
        """按完整 SourceRef 构造不触发登记的确定性 manifest 引用。"""
        return memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_INTAKE_MANIFEST,
            source.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )

    def _result_from_manifest(
            self, record: SourceRecordStorage,
            manifest: IntakeManifestPayload,
            ) -> MemoryIntakeResult:
        """从既有 manifest 恢复幂等结果，不重新解析或追加 Core。"""
        by_kind = {
            kind: tuple(item.object_ref for item in manifest.bindings
                        if item.binding_kind == kind)
            for kind in {
                INTAKE_DERIVED_OBSERVATION,
                INTAKE_DERIVED_HYPOTHESIS,
                INTAKE_DERIVED_EVIDENCE,
                INTAKE_DERIVED_FAILURE,
            }
        }
        return MemoryIntakeResult(
            record,
            self._manifest_ref(manifest),
            manifest.outcome_kind,
            (by_kind[INTAKE_DERIVED_OBSERVATION] or (None,))[0],
            by_kind[INTAKE_DERIVED_HYPOTHESIS],
            by_kind[INTAKE_DERIVED_EVIDENCE],
            (by_kind[INTAKE_DERIVED_FAILURE] or (None,))[0],
            self._derivation_targets_for(manifest.source),
        )

    def _derivation_targets_for(
            self, source: SourceRef,
            ) -> tuple[MemoryObjectRef, ...]:
        """恢复以当前来源为 replacement source 的全部旧派生目标。"""
        entries = self.event_log.query(
            access=self._access(source),
            event_kind=MEMORY_EVENT_DERIVATION,
        )
        return tuple(sorted((
            item.event.payload.target_ref for item in entries
            if (isinstance(
                item.event.payload, DerivationTransitionPayload)
                and item.event.payload.replacement_source == source)
        ), key=lambda ref: ref.stable_key()))

    @staticmethod
    def _access(source: SourceRef) -> MemoryAccessContext:
        """从来源 owner 生成同层 ACL 查询上下文。"""
        owner = source.owner
        return MemoryAccessContext(
            owner.tenant_id, owner.user_id, owner.session_id)


def install_memory_source_intakes(ctx, companion: CompanionSpace) -> None:
    """为上下文装配共享 SourceIntake 和彼此隔离的阅读/交互 Memory writer。"""
    if not isinstance(companion, CompanionSpace):
        raise TypeError("companion 必须是 CompanionSpace")
    if (ctx.memory_read_events is None
            or ctx.memory_interact_events is None):
        raise MemoryIntakeIntegrityError("M-05 装配缺少双层 Memory event log")
    repository = SourceRecordRepository(
        ctx.backend, registry=ctx.scoped_identity_store.registry)
    source_intake = SourceIntake(repository, companion)
    ctx.memory_read_intake = MemorySourceIntake(
        source_intake, ctx.memory_read_events, reading_intake_policy())
    ctx.memory_interact_intake = MemorySourceIntake(
        source_intake, ctx.memory_interact_events,
        interaction_intake_policy())


__all__ = [
    "INTAKE_CHANNEL_INTERACTION",
    "INTAKE_CHANNEL_READING",
    "HypothesisIntakeDraft",
    "MemoryIntakeIntegrityError",
    "MemoryIntakeParser",
    "MemoryIntakePolicy",
    "MemoryIntakeResult",
    "MemorySourceIntake",
    "ObservationIntakeDraft",
    "ParseFailureDraft",
    "interaction_intake_policy",
    "install_memory_source_intakes",
    "reading_intake_policy",
]
