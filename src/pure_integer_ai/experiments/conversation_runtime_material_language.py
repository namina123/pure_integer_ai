"""Runtime 资料到既有语言观察/结构关系输出的只读适配。

资料已经由 ``conversation_runtime_material_ingest`` 留档并进入 Runtime Memory。
本模块只把同一份原文投影为现有语言观察入口所接受的 ``CollectedItem``，调用
既有 ``_split_item_to_segments``/``observe``，并把机械句界 units 绑定为 G1 lexical
evidence。它不把资料写入 Core，不从 relation candidate 推出真值，也不生成回答；
G2 proposition 的消费仍须经外部 qualification gate。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    InputPayload,
    LANG_ZH,
    MODALITY_LANGUAGE,
    STAGE_USER_INTERACTION,
    ObserveResult,
    Segment,
    SpaceContext,
)
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.experiments.collection import COLLECT_PRECEDES, CollectedItem
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
    RawRelationArgument,
    bind_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    RuntimeMaterialIngest,
    RuntimeMaterialQualificationGate,
    compile_runtime_material_structure_observation,
)
from pure_integer_ai.experiments.language_observation import (
    _materialize_item_spans,
    _run_item_predictions,
    _run_item_semantic_course,
    _run_item_sense_candidates,
    _split_item_to_segments,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


RUNTIME_MATERIAL_LANGUAGE_PROTOCOL_V1 = 1
RUNTIME_MATERIAL_RELATION_PRECEDES = "PRECEDES"


class RuntimeMaterialLanguageError(ValueError):
    """Runtime 资料无法闭合为既有语言观察输入。"""


@dataclass(frozen=True, slots=True)
class _MechanicalBoundary:
    """由 T1-G16 候选产生的句界投影；不读取或解释 surface。"""

    scalar_ends: tuple[int, ...]

    def token_cuts(self, token_spans: tuple[tuple[int, int, int], ...]) -> tuple[int, ...]:
        if not token_spans or not self.scalar_ends:
            raise RuntimeMaterialLanguageError("机械边界/token span 不能为空")
        cuts: list[int] = []
        for scalar_end in self.scalar_ends:
            matches = [index + 1 for index, (_start, end, _ordinal)
                       in enumerate(token_spans) if end == scalar_end]
            if len(matches) != 1:
                raise RuntimeMaterialLanguageError(
                    "机械边界没有唯一 winner token 端点")
            cut = matches[0]
            if cuts and cut <= cuts[-1]:
                raise RuntimeMaterialLanguageError("机械边界切点顺序漂移")
            cuts.append(cut)
        return tuple(cuts)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialObservedRelation:
    """来自现有 observe 结构序的 relation candidate，尚未资格化。"""

    proposition: RawPropositionRelationEvidence
    evidence: tuple[RawLexicalEvidence, ...]
    source_observation: RawTextObservation

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, RawPropositionRelationEvidence):
            raise TypeError("relation proposition 类型错误")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise RuntimeMaterialLanguageError("relation evidence 不能为空")
        if not isinstance(self.source_observation, RawTextObservation):
            raise TypeError("relation source observation 类型错误")
        ids = tuple(item.evidence_id for item in self.evidence)
        argument_ids = tuple(item.evidence_id for item in self.proposition.arguments)
        if argument_ids != ids:
            raise RuntimeMaterialLanguageError("relation/evidence 顺序漂移")

    def qualification_gate(
            self,
            qualification: RawPropositionQualification,
            *,
            material_evidence_id: str | None = None,
            ) -> RuntimeMaterialQualificationGate:
        """将真实 observe relation 交给既有 G2/G3 qualification consumer。"""
        if not isinstance(qualification, RawPropositionQualification):
            raise TypeError("qualification 类型错误")
        binding = bind_raw_proposition_relation(
            self.source_observation, self.evidence, self.proposition)
        evidence_id = material_evidence_id or self.evidence[0].evidence_id
        return RuntimeMaterialQualificationGate.from_relation(
            evidence_id, binding, qualification)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialLanguageObservation:
    """一次资料经既有语言观察入口后的完整 Runtime 产物。"""

    ingest: RuntimeMaterialIngest
    item: CollectedItem
    input_payload: InputPayload
    segments: tuple[Segment, ...]
    observation: ObserveResult
    raw_observation: RawTextObservation
    lexical_evidence: tuple[RawLexicalEvidence, ...]
    relation_candidates: tuple[RuntimeMaterialObservedRelation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ingest, RuntimeMaterialIngest):
            raise TypeError("ingest 类型错误")
        if not isinstance(self.item, CollectedItem):
            raise TypeError("item 类型错误")
        if not isinstance(self.input_payload, InputPayload):
            raise TypeError("input_payload 类型错误")
        if not isinstance(self.observation, ObserveResult):
            raise TypeError("observe result 类型错误")
        if self.item.source_ref != self.ingest.capsule.source:
            raise RuntimeMaterialLanguageError("item/source 漂移")
        if self.input_payload.source_ref != self.ingest.capsule.source:
            raise RuntimeMaterialLanguageError("input/source 漂移")
        if tuple(self.input_payload.segments) != self.segments:
            raise RuntimeMaterialLanguageError("input/segments 漂移")
        if tuple(item.evidence_id for item in self.lexical_evidence) != tuple(
                f"runtime-material-evidence-{unit.unit_id}"
                for unit in self.raw_observation.units):
            raise RuntimeMaterialLanguageError("lexical evidence identity 漂移")

    @property
    def runtime_space_only(self) -> bool:
        """该适配器的 stage 合同：观察进入 MemoryInteract，不是 Core。"""
        return self.input_payload.stage == STAGE_USER_INTERACTION

    def stable_key(self) -> tuple[int, ...]:
        """返回可跨语言重建的整数摘要，不携带宿主对象地址。"""
        values = (
            self.ingest.receipt.replay_key,
            self.raw_observation.canonical_record(),
            tuple(item.canonical_record() for item in self.lexical_evidence),
            tuple(item.proposition.canonical_record()
                  for item in self.relation_candidates),
            tuple(self.observation.struct_refs),
        )
        result = [RUNTIME_MATERIAL_LANGUAGE_PROTOCOL_V1]
        for value in values:
            if value and isinstance(value[0], tuple):
                result.append(len(value))
                for nested in value:
                    result.extend((len(nested), *nested))
            else:
                result.extend((len(value), *value))
        return digest_bytes(encode_integer_tuple(tuple(result)))


def _unit_text(raw: str, start: int, end: int) -> str:
    return raw[start:end]


def _make_item(ingest: RuntimeMaterialIngest,
               raw_observation: RawTextObservation) -> CollectedItem:
    raw = ingest.source_record.raw_text
    tokens = [
        _unit_text(raw, unit.start_scalar, unit.end_scalar)
        for unit in raw_observation.units
    ]
    if any(not token for token in tokens):
        raise RuntimeMaterialLanguageError("mechanical unit 不能为空")
    return CollectedItem(
        tokens=tokens,
        raw_text=raw,
        collect_type=COLLECT_PRECEDES,
        source=ingest.capsule.source.source_kind,
        domain=DOMAIN_TEXT,
        lang=ingest.capsule.language or LANG_ZH,
        modality=MODALITY_LANGUAGE,
        source_ref=ingest.capsule.source,
        boundary_decision=_MechanicalBoundary(tuple(
            unit.end_scalar for unit in raw_observation.units)),
    )


def _lexical_evidence(observation: RawTextObservation) -> tuple[RawLexicalEvidence, ...]:
    return tuple(
        RawLexicalEvidence(
            f"runtime-material-evidence-{unit.unit_id}",
            observation.observation_id,
            observation.source_id,
            observation.context_id,
            observation.family_id,
            observation.source_namespace,
            observation.split,
            unit.unit_id,
            unit.role,
            "runtime-observe-v1",
            unit.start_scalar,
            unit.end_scalar,
            unit.start_byte,
            unit.end_byte,
        )
        for unit in observation.units
    )


def _relation_candidates(
        observation: RawTextObservation,
        evidence: tuple[RawLexicalEvidence, ...],
        observed: ObserveResult,
        ) -> tuple[RuntimeMaterialObservedRelation, ...]:
    """把 observe 的跨句 PRECEDES 结构投影为未资格化 relation candidates。"""
    if len(evidence) > 1 and observed.built_edges <= 0:
        return ()
    result = []
    for index in range(len(evidence) - 1):
        left, right = evidence[index:index + 2]
        proposition_key = digest_bytes(encode_integer_tuple((
            RUNTIME_MATERIAL_LANGUAGE_PROTOCOL_V1,
            *tuple(ord(item) for item in RUNTIME_MATERIAL_RELATION_PRECEDES),
            index,
            *observation.canonical_record(),
        )))
        proposition_id = "runtime-material-proposition-v1:" + ".".join(
            str(item) for item in proposition_key)
        proposition = RawPropositionRelationEvidence(
            proposition_id,
            observation.observation_id,
            observation.source_id,
            observation.context_id,
            observation.family_id,
            observation.source_namespace,
            observation.split,
            RUNTIME_MATERIAL_RELATION_PRECEDES,
            "runtime-observe-v1",
            (
                RawRelationArgument(left.evidence_id, left.unit_id, "before", 1),
                RawRelationArgument(right.evidence_id, right.unit_id, "after", 2),
            ),
        )
        result.append(RuntimeMaterialObservedRelation(
            proposition, (left, right), observation))
    return tuple(result)


def observe_runtime_material_language(
        ctx: Any,
        ingest: RuntimeMaterialIngest,
        *,
        observation_id: str,
        context_id: str,
        family_id: str,
        source_namespace: str,
        split: str = "train",
        typed_payload: Any = None,
        payload_kind: str | None = None,
        ) -> RuntimeMaterialLanguageObservation:
    """用既有语言 observation 管线处理一份 Runtime 资料。

    ``ctx`` 应是专用的 Runtime/会话训练上下文；此函数只使用
    ``STAGE_USER_INTERACTION``，因此不会向 Core 写入。没有配置词形或语义课程时，
    仍会诚实返回机械句界和 observe 结构结果，缺少的高阶关系不会被伪造。
    """
    if not isinstance(ingest, RuntimeMaterialIngest):
        raise TypeError("ingest 类型错误")
    for value, label in ((observation_id, "observation_id"),
                         (context_id, "context_id"),
                         (family_id, "family_id"),
                         (source_namespace, "source_namespace")):
        if type(value) is not str or not value.strip():
            raise RuntimeMaterialLanguageError(f"{label} 必须是非空文本")
    raw_observation, _extraction, coverage = (
        compile_runtime_material_structure_observation(
            ingest,
            observation_id=observation_id,
            context_id=context_id,
            family_id=family_id,
            source_namespace=source_namespace,
            split=split,
        ))
    if not coverage.complete:
        raise RuntimeMaterialLanguageError("Runtime 资料结构覆盖未闭合")
    if (payload_kind is None) != (typed_payload is None):
        raise RuntimeMaterialLanguageError(
            "typed_payload 与 payload_kind 必须成对提供")
    if payload_kind is not None and (
            not isinstance(payload_kind, str)
            or not payload_kind
            or payload_kind.strip() != payload_kind
            or not hasattr(typed_payload, "to_value")):
        raise RuntimeMaterialLanguageError(
            "typed semantic payload 必须是成对的 canonical object")
    item = _make_item(ingest, raw_observation)
    providers = getattr(ctx, "word_form_providers", None)
    if (providers is not None and item.raw_text is not None
            and providers.supports(item.lang)):
        from pure_integer_ai.experiments.language_observation import (
            _apply_word_form_providers,
        )
        _apply_word_form_providers([item], providers, commit_evidence=True)
    item = replace(item, typed_payload=typed_payload,
                   payload_kind=payload_kind)
    scope = document_scope(ingest.capsule.source)
    segments = tuple(_split_item_to_segments(
        item,
        backend=ctx.backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
        language_signal_runtime=ctx.language_signal_runtime,
        language_signal_compatibility_enabled=(
            ctx.language_signal_compatibility_enabled),
    ))
    if not segments:
        raise RuntimeMaterialLanguageError("Runtime 资料没有语言 Segment")
    payload = InputPayload(
        segments=list(segments),
        source=ingest.capsule.source.source_kind,
        stage=STAGE_USER_INTERACTION,
        modality=MODALITY_LANGUAGE,
        lang=item.lang,
        domain=DOMAIN_TEXT,
        item_key=ingest.event.memory_item_key,
        scope_identity=scope,
        source_ref=ingest.capsule.source,
        occurrence_scope_identity=scope,
        raw_text=ingest.source_record.raw_text,
        source_license_id=ingest.source_record.metadata.license_id,
        source_batch_id=ingest.source_record.metadata.batch_id,
    )
    source_intake = None
    memory_read_intake = getattr(ctx, "memory_read_intake", None)
    if memory_read_intake is not None:
        source_intake = getattr(memory_read_intake, "source_intake", None)
    companion = (
        getattr(source_intake, "companion", None)
        if source_intake is not None else None
    )
    if source_intake is None or companion is None:
        raise RuntimeMaterialLanguageError(
            "Runtime 资料语言观察需要已安装 Companion 来源入口")
    space_context = SpaceContext(
        core=ctx.core_space,
        memory_read=ctx.memory_read,
        memory_interact=ctx.memory_interact,
        companion=companion,
        stage=STAGE_USER_INTERACTION,
        memory_active=False,
    )
    observed = observe(
        payload,
        space_context,
        concept_index=ctx.concept_index,
        work_memory=ctx.work_memory,
        word_form_providers=ctx.word_form_providers,
        occurrence_index=ctx.occurrence_index,
        language_signal_runtime=ctx.language_signal_runtime,
        language_signal_compatibility_enabled=(
            ctx.language_signal_compatibility_enabled),
        write_legacy_language_sequences=True,
        source_intake=source_intake,
    )
    if ctx.occurrence_index is not None:
        _materialize_item_spans(ctx, item, observed)
        _run_item_predictions(ctx, item, observed)
        _run_item_sense_candidates(ctx, item, observed)
    semantic_run = _run_item_semantic_course(ctx, item, payload, observed)
    _ = semantic_run
    lexical = _lexical_evidence(raw_observation)
    return RuntimeMaterialLanguageObservation(
        ingest,
        item,
        payload,
        segments,
        observed,
        raw_observation,
        lexical,
        _relation_candidates(raw_observation, lexical, observed),
    )


__all__ = [
    "RUNTIME_MATERIAL_LANGUAGE_PROTOCOL_V1",
    "RUNTIME_MATERIAL_RELATION_PRECEDES",
    "RuntimeMaterialLanguageError",
    "RuntimeMaterialLanguageObservation",
    "RuntimeMaterialObservedRelation",
    "observe_runtime_material_language",
]
