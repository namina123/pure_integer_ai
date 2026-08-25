"""用户新资料的来源化 Runtime 摄入、按 item 读取和显式 promotion request。

该入口把原文留在既有 ``SourceRecordRepository`` 伴随层，把 capsule/event 留在
Runtime Memory；没有 CoreDelta 或本模块生成的隐式晋升。调用方必须显式提供来源、
scope、版本、许可 metadata 和 authority/consent，才能得到可审计结果。
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    LearningInputCapsule,
    LearningReplayReceipt,
    PromotionRequest,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    PROJECTION_RUNTIME,
    STATUS_OBSERVED,
    append_runtime_event,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
    RawPropositionQualification,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
    RawPropositionRelationBinding,
    bind_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import RawLexicalEvidence
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
    RawTextSpanUnit,
    compile_raw_text_observation,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_coverage import (
    RawTextCandidateCoverageAudit,
    audit_raw_text_candidate_coverage,
)
from pure_integer_ai.experiments.conversation_broad_dialogue_persistence import (
    write_broad_dialogue_checkpoint,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import BroadDialogueState
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.storage.k_run_boundary import KRunRoot
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
    SourceRecordStorage,
)
from pure_integer_ai.cognition.understanding.source_intake import SourceIntake


RUNTIME_MATERIAL_INGEST_PROTOCOL_V1 = 1
RUNTIME_MATERIAL_MEMORY_ITEM_KIND_V1 = 1
RUNTIME_MATERIAL_READ_HIT = "HIT"
RUNTIME_MATERIAL_READ_UNKNOWN = "UNKNOWN"


class RuntimeMaterialIngestError(ValueError):
    """用户资料 capsule、来源留档、Runtime append 或 promotion 边界错误。"""


def _strict_key(value: tuple[int, ...], *, label: str,
                empty: bool = False) -> tuple[int, ...]:
    if (not isinstance(value, tuple)
            or (not empty and not value)
            or any(type(item) is not int for item in value)):
        raise RuntimeMaterialIngestError(f"{label} 必须是严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    checked = _strict_key(value, label="record field", empty=True)
    result.extend((len(checked), *checked))


def _memory_item_key(capsule: LearningInputCapsule) -> tuple[int, ...]:
    return (RUNTIME_MATERIAL_MEMORY_ITEM_KIND_V1, *capsule.identity_key)


def _replay_key(capsule: LearningInputCapsule,
                event: RuntimeMemoryEvent,
                source_record: SourceRecordStorage,
                ) -> tuple[int, ...]:
    result = [RUNTIME_MATERIAL_INGEST_PROTOCOL_V1]
    for value in (
            capsule.canonical_record, event.event_key,
            source_record.source_key,
            (source_record.text_hash, source_record.codepoint_count),
    ):
        _pack(result, value)
    return digest_bytes(encode_integer_tuple(tuple(result)))


@dataclass(frozen=True, slots=True)
class RuntimeMaterialIngest:
    """一次用户资料进入 Runtime Memory 的完整结果。"""

    capsule: LearningInputCapsule
    event: RuntimeMemoryEvent
    source_record: SourceRecordStorage
    memory_before: RuntimeMemoryState
    memory_after: RuntimeMemoryState
    admission_status: int
    receipt: LearningReplayReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("material capsule 类型错误")
        if not isinstance(self.event, RuntimeMemoryEvent):
            raise TypeError("material event 类型错误")
        if self.event.capsule != self.capsule:
            raise RuntimeMaterialIngestError("event/capsule 漂移")
        if not isinstance(self.source_record, SourceRecordStorage):
            raise TypeError("material source record 类型错误")
        if self.source_record.source_key != self.capsule.source.stable_key():
            raise RuntimeMaterialIngestError("source record/source 漂移")
        if not isinstance(self.memory_before, RuntimeMemoryState):
            raise TypeError("material memory_before 类型错误")
        if not isinstance(self.memory_after, RuntimeMemoryState):
            raise TypeError("material memory_after 类型错误")
        if self.memory_before.scope_key != self.memory_after.scope_key:
            raise RuntimeMaterialIngestError("Runtime Memory scope 漂移")
        if self.event.capsule.scope.stable_key() != self.memory_before.scope_key:
            raise RuntimeMaterialIngestError("event 不在 Runtime Memory scope")
        if self.admission_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
            raise RuntimeMaterialIngestError("admission status 未注册")
        if self.receipt.projection_kind != PROJECTION_RUNTIME:
            raise RuntimeMaterialIngestError("receipt projection kind 漂移")
        if self.receipt.input_identity != self.capsule.identity_key:
            raise RuntimeMaterialIngestError("receipt input identity 漂移")
        if self.receipt.output_identity != self.event.event_key:
            raise RuntimeMaterialIngestError("receipt output identity 漂移")

    def promotion_request(
            self,
            *,
            authority_key: tuple[int, ...],
            consent_key: tuple[int, ...],
            evidence_keys: tuple[tuple[int, ...], ...] | None = None,
            ) -> PromotionRequest:
        """只构造显式 promotion request，不修改 Core。"""
        authority = _strict_key(authority_key, label="promotion authority")
        consent = _strict_key(consent_key, label="promotion consent")
        evidence = evidence_keys or (
            self.event.event_key,
            (self.source_record.source_hash, self.source_record.text_hash),
        )
        if (not isinstance(evidence, tuple) or not evidence
                or any(not isinstance(item, tuple) or not item for item in evidence)):
            raise RuntimeMaterialIngestError("promotion evidence 不得为空")
        checked_evidence = tuple(
            _strict_key(item, label="promotion evidence") for item in evidence)
        return PromotionRequest(
            self.event.event_key,
            self.capsule.source,
            self.capsule.scope,
            checked_evidence,
            authority,
            self.receipt.replay_key,
            consent,
        )


def ingest_runtime_material(
        runtime_state: RuntimeMemoryState,
        *,
        source: SourceRef,
        scope: ScopeIdentity,
        raw_text: str,
        source_records: SourceRecordRepository,
        metadata: SourceRecordMetadata,
        source_intake: SourceIntake | None = None,
        version_key: tuple[int, ...],
        authority_key: tuple[int, ...],
        parent_version_key: tuple[int, ...] = (),
        language: int = 1,
        modality: int = 1,
        split: int = 1,
        delta_sequence: int = 1,
        ) -> RuntimeMaterialIngest:
    """留档一份新资料并追加 Runtime event；重复输入保持幂等。"""
    if not isinstance(runtime_state, RuntimeMemoryState):
        raise TypeError("runtime_state 类型错误")
    if not isinstance(source, SourceRef) or not isinstance(scope, ScopeIdentity):
        raise TypeError("source/scope 类型错误")
    if scope.source != source:
        raise RuntimeMaterialIngestError("scope 必须绑定 source")
    if runtime_state.scope_key != scope.stable_key():
        raise RuntimeMaterialIngestError("runtime_state 与 scope 不一致")
    if type(raw_text) is not str or not raw_text:
        raise RuntimeMaterialIngestError("raw_text 必须是非空字符串")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    if not isinstance(metadata, SourceRecordMetadata) or not metadata.complete:
        raise RuntimeMaterialIngestError(
            "用户资料必须携带完整 license/batch/Companion metadata")
    if source_intake is not None:
        if not isinstance(source_intake, SourceIntake):
            raise TypeError("source_intake 类型错误")
        if source_intake.repository is not source_records:
            raise RuntimeMaterialIngestError(
                "source_intake 必须绑定同一 SourceRecordRepository")
        companion_identity = source_intake.companion.identity
        if (metadata.companion_type_hash != companion_identity.type_hash
                or metadata.companion_name_hash != companion_identity.name_hash):
            raise RuntimeMaterialIngestError(
                "metadata Companion identity 与 SourceIntake 不一致")
    version = _strict_key(version_key, label="material version")
    parent = _strict_key(parent_version_key, label="material parent", empty=True)
    authority = _strict_key(authority_key, label="material authority")
    for value, label in (
            (language, "language"), (modality, "modality"),
            (split, "split"), (delta_sequence, "delta_sequence")):
        if type(value) is not int or value <= 0:
            raise RuntimeMaterialIngestError(f"{label} 必须为正整数")
    if source_intake is None:
        source_record = source_records.put_complete(
            source.stable_key(), raw_text, metadata=metadata)
    else:
        source_record = source_intake.ensure(
            source,
            raw_text,
            license_id=metadata.license_id,
            batch_id=metadata.batch_id,
        )
    raw = raw_text.encode("utf-8")
    capsule = LearningInputCapsule(
        source, scope, version, parent, language, modality,
        digest_bytes(raw), ((1, len(raw)),), authority,
        metadata.license_id, split, delta_sequence,
    )
    event = RuntimeMemoryEvent(capsule, _memory_item_key(capsule))
    memory_after, admission = append_runtime_event(runtime_state, event)
    if admission not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
        raise RuntimeMaterialIngestError(f"Runtime Memory append 被拒绝: {admission}")
    replay_key = _replay_key(capsule, event, source_record)
    receipt = LearningReplayReceipt(
        PROJECTION_RUNTIME,
        capsule.identity_key,
        event.event_key,
        STATUS_OBSERVED,
        replay_key,
    )
    return RuntimeMaterialIngest(
        capsule, event, source_record, runtime_state, memory_after,
        admission, receipt,
    )


@dataclass(frozen=True, slots=True)
class RuntimeMaterialRead:
    """一次按 item key 的 Runtime 资料证据读取。"""

    status: str
    event: RuntimeMemoryEvent | None
    source_record: SourceRecordStorage | None
    physical_read_count: int = 1

    def __post_init__(self) -> None:
        if self.status not in {RUNTIME_MATERIAL_READ_HIT,
                               RUNTIME_MATERIAL_READ_UNKNOWN}:
            raise RuntimeMaterialIngestError("material read status 未注册")
        if self.status == RUNTIME_MATERIAL_READ_HIT:
            if not isinstance(self.event, RuntimeMemoryEvent):
                raise TypeError("HIT 缺少 Runtime event")
            if not isinstance(self.source_record, SourceRecordStorage):
                raise TypeError("HIT 缺少 SourceRecord")
        elif self.event is not None or self.source_record is not None:
            raise RuntimeMaterialIngestError("UNKNOWN 不得携带资料")
        if type(self.physical_read_count) is not int or self.physical_read_count < 0:
            raise RuntimeMaterialIngestError("physical read count 非法")


@dataclass(frozen=True, slots=True)
class RuntimeMaterialReadIndex:
    """一次建立的 Runtime item 热索引；查询不重新扫描事件账本。"""

    state: RuntimeMemoryState
    item_index: tuple[tuple[tuple[int, ...], int], ...]

    @classmethod
    def build(cls, state: RuntimeMemoryState) -> "RuntimeMaterialReadIndex":
        if not isinstance(state, RuntimeMemoryState):
            raise TypeError("state 类型错误")
        by_item: dict[tuple[int, ...], tuple[int, RuntimeMemoryEvent]] = {}
        for ordinal, event in enumerate(state.events):
            previous = by_item.get(event.memory_item_key)
            if previous is None or event.revision > previous[1].revision:
                by_item[event.memory_item_key] = (ordinal, event)
            elif (event.revision == previous[1].revision
                  and event.event_key != previous[1].event_key):
                raise RuntimeMaterialIngestError("Runtime item 存在竞争 revision")
        return cls(state, tuple(sorted(
            (key, ordinal) for key, (ordinal, _event) in by_item.items())))

    def read(
            self,
            memory_item_key: tuple[int, ...],
            source_records: SourceRecordRepository,
            ) -> RuntimeMaterialRead:
        key = _strict_key(memory_item_key, label="material item key")
        if not isinstance(source_records, SourceRecordRepository):
            raise TypeError("source_records 类型错误")
        position = bisect_left(self.item_index, (key, -1))
        if position >= len(self.item_index) or self.item_index[position][0] != key:
            return RuntimeMaterialRead(RUNTIME_MATERIAL_READ_UNKNOWN, None, None)
        event = self.state.events[self.item_index[position][1]]
        source_record = source_records.find(event.capsule.source.stable_key())
        if source_record is None:
            raise RuntimeMaterialIngestError("Runtime event 缺少 SourceRecord")
        if digest_bytes(source_record.raw_text.encode("utf-8")) != event.capsule.raw_content_digest:
            raise RuntimeMaterialIngestError("SourceRecord 与 capsule raw digest 漂移")
        return RuntimeMaterialRead(
            RUNTIME_MATERIAL_READ_HIT, event, source_record)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialQualificationGate:
    """Runtime item 与既有 proposition qualification 的一致性闸门。"""

    material_evidence_id: str
    proposition_binding: RawPropositionRelationBinding
    qualification: RawPropositionQualification
    consumer_result: RawPropositionConsumerResult

    def __post_init__(self) -> None:
        if (type(self.material_evidence_id) is not str
                or not self.material_evidence_id.strip()):
            raise RuntimeMaterialIngestError("material evidence id 不能为空")
        if not isinstance(self.proposition_binding, RawPropositionRelationBinding):
            raise TypeError("qualification proposition binding 类型错误")
        if not isinstance(self.qualification, RawPropositionQualification):
            raise TypeError("qualification 类型错误")
        if not isinstance(self.consumer_result, RawPropositionConsumerResult):
            raise TypeError("qualification consumer result 类型错误")
        if self.material_evidence_id not in self.qualification.evidence_ids:
            raise RuntimeMaterialIngestError(
                "qualification 未包含 Runtime material evidence")
        expected = consume_raw_proposition_relation(
            self.proposition_binding, self.qualification)
        if expected != self.consumer_result:
            raise RuntimeMaterialIngestError(
                "qualification consumer result 与既有 consumer 漂移")

    @classmethod
    def from_relation(
            cls,
            material_evidence_id: str,
            proposition_binding: RawPropositionRelationBinding,
            qualification: RawPropositionQualification,
            ) -> "RuntimeMaterialQualificationGate":
        """消费既有 qualification，不重新解释 relation 或文本。"""
        consumer = consume_raw_proposition_relation(
            proposition_binding, qualification)
        return cls(material_evidence_id, proposition_binding, qualification, consumer)

    @classmethod
    def from_annotation(
            cls,
            material_evidence_id: str,
            observation: RawTextObservation,
            lexical_evidence: tuple[RawLexicalEvidence, ...],
            proposition: RawPropositionRelationEvidence,
            qualification: RawPropositionQualification,
            ) -> "RuntimeMaterialQualificationGate":
        """从真实 observation/evidence annotation 走既有 binding/consumer。"""
        relation = bind_raw_proposition_relation(
            observation, lexical_evidence, proposition)
        return cls.from_relation(material_evidence_id, relation, qualification)

    @property
    def response_act(self) -> str:
        return self.consumer_result.response_act


@dataclass(frozen=True, slots=True)
class RuntimeMaterialAnswerBinding:
    """由理解/路由层显式授权的一条问题到 Runtime item 绑定。"""

    question: str
    memory_item_key: tuple[int, ...]
    qualification_gate: RuntimeMaterialQualificationGate
    source_title: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if type(self.question) is not str or not self.question.strip():
            raise RuntimeMaterialIngestError("answer binding question 不能为空")
        _strict_key(self.memory_item_key, label="answer binding item")
        if not isinstance(self.qualification_gate, RuntimeMaterialQualificationGate):
            raise TypeError("answer binding qualification gate 类型错误")
        for value, label in ((self.source_title, "source title"),
                             (self.source_url, "source url")):
            if value is not None and (type(value) is not str or not value.strip()):
                raise RuntimeMaterialIngestError(f"{label} 非法")


@dataclass(frozen=True, slots=True)
class RuntimeMaterialAnswerProvider:
    """只消费显式唯一绑定的 Runtime 原文，不执行词面猜测。"""

    index: RuntimeMaterialReadIndex
    source_records: SourceRecordRepository
    bindings: tuple[RuntimeMaterialAnswerBinding, ...]

    def __post_init__(self) -> None:
        index = self.index
        source_records = self.source_records
        bindings = self.bindings
        if not isinstance(index, RuntimeMaterialReadIndex):
            raise TypeError("material answer index 类型错误")
        if not isinstance(source_records, SourceRecordRepository):
            raise TypeError("material answer source repository 类型错误")
        if (not isinstance(bindings, tuple)
                or any(not isinstance(item, RuntimeMaterialAnswerBinding)
                       for item in bindings)):
            raise TypeError("material answer bindings 类型错误")
        questions = tuple(item.question for item in bindings)
        if len(set(questions)) != len(questions):
            raise RuntimeMaterialIngestError("answer binding question 不得重复")

    def answer(self, question: str) -> tuple[str, str | None, str | None] | None:
        """精确命中 binding 才返回 ``(evidence, title, url)``。"""
        if type(question) is not str or not question.strip():
            raise ValueError("material answer question 不能为空")
        binding = next((item for item in self.bindings
                        if item.question == question), None)
        if binding is None:
            return None
        read = self.index.read(binding.memory_item_key, self.source_records)
        if (binding.qualification_gate is None
                or binding.qualification_gate.response_act != "ANSWER"):
            return None
        if read.status != RUNTIME_MATERIAL_READ_HIT or read.source_record is None:
            return None
        return read.source_record.raw_text, binding.source_title, binding.source_url


def persist_runtime_material_checkpoint(
        root: KRunRoot,
        dialogue_state: BroadDialogueState,
        ingest: RuntimeMaterialIngest,
        ):
    """把 Runtime-only 资料操作写入独立 operation checkpoint。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 类型错误")
    if not isinstance(dialogue_state, BroadDialogueState):
        raise TypeError("dialogue_state 类型错误")
    if not isinstance(ingest, RuntimeMaterialIngest):
        raise TypeError("ingest 类型错误")
    return write_broad_dialogue_checkpoint(
        root,
        dialogue_state,
        runtime_memory_state=ingest.memory_after,
    )


def compile_runtime_material_observation(
        ingest: RuntimeMaterialIngest,
        *,
        observation_id: str,
        context_id: str,
        family_id: str,
        source_namespace: str,
        split: str,
        unit_id: str,
        unit_role: str,
        ) -> RawTextObservation:
    """将已留档 Runtime 原文接入 G0 observation，不猜结构标签。"""
    if not isinstance(ingest, RuntimeMaterialIngest):
        raise TypeError("ingest 类型错误")
    raw = tuple(ingest.source_record.raw_text.encode("utf-8"))
    scalars = tuple(ord(item) for item in ingest.source_record.raw_text)
    return compile_raw_text_observation(
        raw,
        observation_id=observation_id,
        source_id=str(ingest.source_record.source_hash),
        context_id=context_id,
        family_id=family_id,
        source_namespace=source_namespace,
        split=split,
        units=(RawTextSpanUnit(
            unit_id, unit_role, 0, len(scalars), 0, len(raw)),),
    )


def compile_runtime_material_structure_observation(
        ingest: RuntimeMaterialIngest,
        *,
        observation_id: str,
        context_id: str,
        family_id: str,
        source_namespace: str,
        split: str,
        unit_role: str = "sentence",
        ) -> tuple[RawTextObservation, RawTextCandidateExtraction,
                   RawTextCandidateCoverageAudit]:
    """把 Runtime 原文接入真实机械结构观察，再交给 G1-G3 标注。

    T1-G16 只按冻结句界标点产生物理候选；本函数只投影候选为 observation
    units，并要求 G17 覆盖闭合。它不分配 relation、argument 或 qualification
    状态，因此不会把结构边界误当成语义或真值。
    """
    if not isinstance(ingest, RuntimeMaterialIngest):
        raise TypeError("ingest 类型错误")
    if type(unit_role) is not str or not unit_role.strip():
        raise RuntimeMaterialIngestError("unit_role 必须是非空文本")
    raw = tuple(ingest.source_record.raw_text.encode("utf-8"))
    extraction = extract_raw_text_candidate_spans(raw)
    if not extraction.accepted or not extraction.candidates:
        raise RuntimeMaterialIngestError(
            "Runtime material 没有可用的机械结构候选")
    units = tuple(
        RawTextSpanUnit(
            f"unit-{candidate.ordinal}", unit_role,
            candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte,
        )
        for candidate in extraction.candidates
    )
    observation = compile_raw_text_observation(
        raw,
        observation_id=observation_id,
        source_id=str(ingest.source_record.source_hash),
        context_id=context_id,
        family_id=family_id,
        source_namespace=source_namespace,
        split=split,
        units=units,
    )
    coverage = audit_raw_text_candidate_coverage(extraction, observation)
    if not coverage.complete:
        raise RuntimeMaterialIngestError(
            "Runtime material 结构候选未完整覆盖 observation units")
    return observation, extraction, coverage


__all__ = [
    "RUNTIME_MATERIAL_INGEST_PROTOCOL_V1",
    "RUNTIME_MATERIAL_MEMORY_ITEM_KIND_V1",
    "RUNTIME_MATERIAL_READ_HIT",
    "RUNTIME_MATERIAL_READ_UNKNOWN",
    "RuntimeMaterialIngest",
    "RuntimeMaterialIngestError",
    "RuntimeMaterialAnswerBinding",
    "RuntimeMaterialAnswerProvider",
    "RuntimeMaterialQualificationGate",
    "RuntimeMaterialRead",
    "RuntimeMaterialReadIndex",
    "compile_runtime_material_observation",
    "compile_runtime_material_structure_observation",
    "ingest_runtime_material",
    "persist_runtime_material_checkpoint",
]
