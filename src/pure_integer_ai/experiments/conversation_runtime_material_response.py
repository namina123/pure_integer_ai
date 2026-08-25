"""Runtime 资料证据到既有多段回答组织协议的只读接线。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.experiments.conversation_capsule_response_organization import (
    SEGMENT_CLAIM,
    SEGMENT_QUALIFIER,
    SEGMENT_REPAIR,
    SEGMENT_SUPPORT,
    ResponseOrganizationPlan,
    ResponseSegment,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    RuntimeMaterialIngest,
    RuntimeMaterialAnswerBinding,
    RuntimeMaterialAnswerProvider,
    RuntimeMaterialQualificationGate,
    RUNTIME_MATERIAL_READ_HIT,
    RuntimeMaterialReadIndex,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    DialogueCitation,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


def _has_followup_reference(surface: str) -> bool:
    """复用对话层已冻结的指代边界，不把普通词内子串当作追问。"""
    from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
        _has_followup_reference as broad_has_followup_reference,
    )
    return broad_has_followup_reference(surface)


RUNTIME_MATERIAL_RESPONSE_PROTOCOL_V1 = 1
_RELATED_QUERY_MIN_TERMS = 3
_RELATED_QUERY_MIN_COVERAGE = 45


class RuntimeMaterialResponseError(ValueError):
    """Runtime 资料不能形成守恒的 response organization。"""


def _candidate_evidence_units(
        candidate: "_RuntimeMaterialResponseCandidate",
        ) -> tuple[tuple[str, str, str], ...]:
    """从已资格化 relation binding 投影有序证据单元。

    该函数只消费 binding 中已经闭合的 argument u8/scalar，不重新扫描或解释
    SourceRecord。不同 ``relation_index`` 因此会得到不同的相邻证据窗口。
    """
    arguments = candidate.binding.qualification_gate.proposition_binding.arguments
    if not arguments:
        raise RuntimeMaterialResponseError("relation binding 缺少 evidence arguments")
    result: list[tuple[str, str, str]] = []
    for argument in arguments:
        decoded = decode_utf8_v1(tuple(argument.unit_bytes))
        if decoded is None or decoded != tuple(argument.unit_scalars):
            raise RuntimeMaterialResponseError(
                "relation evidence u8/scalar identity 漂移")
        try:
            surface = "".join(chr(item) for item in decoded)
        except (TypeError, ValueError) as error:
            raise RuntimeMaterialResponseError(
                "relation evidence surface 无法投影") from error
        if not surface:
            raise RuntimeMaterialResponseError("relation evidence unit 为空")
        result.append((argument.observation_id, argument.evidence_id, surface))
    if not result:
        raise RuntimeMaterialResponseError("relation evidence surface 为空")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialResponseSpec:
    """一条问题到真实 Runtime observation/candidate 的显式绑定。"""

    observation: object
    qualification: RawPropositionQualification
    question: str
    relation_index: int = 0
    source_title: str | None = None
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class _RuntimeMaterialResponseCandidate:
    """已消费 gate 的单条只读资料候选。"""

    question: str
    index: RuntimeMaterialReadIndex
    binding: RuntimeMaterialAnswerBinding
    relation_index: int = 0

    def sort_key(self) -> tuple[object, ...]:
        return (
            tuple(ord(item) for item in self.question),
            self.binding.memory_item_key,
            self.relation_index,
            self.binding.qualification_gate.qualification.canonical_record(),
            tuple(ord(item) for item in (self.binding.source_title or "")),
            tuple(ord(item) for item in (self.binding.source_url or "")),
        )


@dataclass(frozen=True, slots=True)
class RuntimeMaterialResponseProvider:
    """多资料 Runtime response consumer，冲突和未知均阻断外部回退。"""

    source_records: SourceRecordRepository
    candidates: tuple[_RuntimeMaterialResponseCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_records, SourceRecordRepository):
            raise TypeError("source_records 类型错误")
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or any(not isinstance(item, _RuntimeMaterialResponseCandidate)
                       for item in self.candidates)):
            raise RuntimeMaterialResponseError("response candidates 不能为空")
        if self.candidates != tuple(sorted(self.candidates,
                                           key=lambda item: item.sort_key())):
            raise RuntimeMaterialResponseError("response candidates 未规范排序")

    def _respond_matches_with_citations(
            self,
            matches: tuple[_RuntimeMaterialResponseCandidate, ...],
            ) -> tuple[
                str, str | None, str | None, str | None,
                tuple[DialogueCitation, ...],
            ] | None:
        """消费已经通过问题或焦点选择的候选，不重新解释资料。"""
        if not matches:
            return None
        acts = tuple(item.binding.qualification_gate.response_act
                     for item in matches)
        if "CLARIFY" in acts:
            return "CLARIFY", None, None, None, ()
        if any(item != "ANSWER" for item in acts):
            return "UNKNOWN", None, None, None, ()
        source_answers: list[
            tuple[tuple[int, ...], list[tuple[str, str]], list[str],
                  str | None, str | None]] = []
        titles: list[str] = []
        urls: list[str] = []
        for candidate in matches:
            binding = candidate.binding
            read = candidate.index.read(
                binding.memory_item_key, self.source_records)
            if read.status != RUNTIME_MATERIAL_READ_HIT or read.source_record is None:
                return "UNKNOWN", None, None, None, ()
            source_key = read.source_record.source_key
            group = next((item for item in source_answers
                          if item[0] == source_key), None)
            if group is None:
                group = (source_key, [], [], binding.source_title, binding.source_url)
                source_answers.append(group)
            seen, surfaces = group[1], group[2]
            for observation_id, evidence_id, surface in (
                    _candidate_evidence_units(candidate)):
                evidence_key = (observation_id, evidence_id)
                if evidence_key in seen:
                    continue
                seen.append(evidence_key)
                surfaces.append(surface)
            if binding.source_title and binding.source_title not in titles:
                titles.append(binding.source_title)
            if binding.source_url and binding.source_url not in urls:
                urls.append(binding.source_url)
        answers = tuple("".join(item[2]) for item in source_answers if item[2])
        if not answers:
            return "UNKNOWN", None, None, None, ()
        citations = tuple(
            DialogueCitation(
                "".join(group[2]), group[3], group[4])
            for group in source_answers if group[2])
        title = "；".join(titles) if titles else None
        url = urls[0] if len(urls) == 1 else None
        return "ANSWER", "\n".join(answers), title, url, citations

    def response_with_citations(
            self,
            question: str,
            ) -> tuple[
                str, str | None, str | None, str | None,
                tuple[DialogueCitation, ...],
            ] | None:
        """返回答案及逐来源 evidence surface；旧 ``response`` 保持四元兼容。"""
        if type(question) is not str or not question.strip():
            raise ValueError("question 必须是非空文本")
        matches = tuple(item for item in self.candidates
                        if item.question == question)
        return self._respond_matches_with_citations(matches)

    def _respond_matches(
            self,
            matches: tuple[_RuntimeMaterialResponseCandidate, ...],
            ) -> tuple[str, str | None, str | None, str | None] | None:
        result = self._respond_matches_with_citations(matches)
        if result is None:
            return None
        return result[:4]

    def response(
            self,
            question: str,
            ) -> tuple[str, str | None, str | None, str | None] | None:
        """精确问题命中后聚合资料；命中 UNKNOWN/CONFLICT 时不回退广域 QA。"""
        if type(question) is not str or not question.strip():
            raise ValueError("question 必须是非空文本")
        matches = tuple(item for item in self.candidates
                        if item.question == question)
        return self._respond_matches(matches)

    def response_followup_with_citations(
            self,
            question: str,
            source_title: str,
            ) -> tuple[
                str, str | None, str | None, str | None,
                tuple[DialogueCitation, ...],
            ] | None:
        """在上一轮来源标题唯一时消费紧邻的显式指代追问。

        该入口只扩展已登记来源候选的焦点，不把追问改写成新事实查询；
        多候选或无指代时返回 ``None``，让上层保持原有广域/澄清边界。
        """
        if (type(question) is not str or not question.strip()
                or type(source_title) is not str or not source_title.strip()):
            raise ValueError("followup question/source_title 必须是非空文本")
        if not _has_followup_reference(question):
            return None
        matches = tuple(item for item in self.candidates
                        if item.binding.source_title == source_title)
        if len(matches) != 1:
            return "CLARIFY", None, None, None, ()
        return self._respond_matches_with_citations(matches)

    def response_followup(
            self,
            question: str,
            source_title: str,
            ) -> tuple[str, str | None, str | None, str | None] | None:
        """兼容旧四元接口的紧邻指代入口。"""
        result = self.response_followup_with_citations(question, source_title)
        if result is None:
            return None
        return result[:4]

    def response_related_with_citations(
            self,
            question: str,
            source_title: str | None = None,
            ) -> tuple[
                str, str | None, str | None, str | None,
                tuple[DialogueCitation, ...],
            ] | None:
        """在唯一高重合绑定上消费自然改写，不进行开放域猜测。

        问题表面只投影为既有确定性 n-gram 特征；至少三个特征且覆盖率达到
        45%，并且最高分唯一时才允许复用已资格化资料。候选相近或不足时返回
        ``None``，由终端继续保持 UNKNOWN/CLARIFY 边界。
        """
        if type(question) is not str or not question.strip():
            raise ValueError("related question 必须是非空文本")
        if source_title is not None and (
                type(source_title) is not str or not source_title.strip()):
            raise ValueError("related source_title 必须是非空文本")
        query_features = set(broad_qa_terms(question))
        if len(query_features) < _RELATED_QUERY_MIN_TERMS:
            return None
        candidates = tuple(
            item for item in self.candidates
            if source_title is None or item.binding.source_title == source_title)
        ranked: list[tuple[int, int, int, _RuntimeMaterialResponseCandidate]] = []
        for candidate in candidates:
            binding_features = set(broad_qa_terms(candidate.question))
            overlap = len(query_features & binding_features)
            if overlap < _RELATED_QUERY_MIN_TERMS:
                continue
            coverage = (overlap * 100) // len(query_features)
            if coverage < _RELATED_QUERY_MIN_COVERAGE:
                continue
            ranked.append((overlap, coverage, -len(binding_features), candidate))
        ranked.sort(key=lambda item: (
            -item[0], -item[1], -item[2], item[3].sort_key()))
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][:3] == best[:3]:
            return "CLARIFY", None, None, None, ()
        return self._respond_matches_with_citations((best[3],))

    def response_related(
            self,
            question: str,
            source_title: str | None = None,
            ) -> tuple[str, str | None, str | None, str | None] | None:
        """兼容旧四元接口的自然改写入口。"""
        result = self.response_related_with_citations(question, source_title)
        if result is None:
            return None
        return result[:4]

    def answer(self, question: str) -> tuple[str, str | None, str | None] | None:
        """兼容旧单资料 callback，仅把 ANSWER 投影为三元组。"""
        result = self.response(question)
        if result is None or result[0] != "ANSWER":
            return None
        return result[1], result[2], result[3]


def build_runtime_material_answer_provider(
        observation,
        qualification: RawPropositionQualification,
        *,
        source_records: SourceRecordRepository,
        question: str,
        source_title: str | None = None,
        source_url: str | None = None,
        relation_index: int = 0,
        ) -> RuntimeMaterialAnswerProvider:
    """把真实语言观察的 relation candidate 接到对话 provider。

    ``observation`` 必须来自 ``observe_runtime_material_language``；本适配器只
    选择已有 candidate 并消费调用方提供的 qualification，不从问题或原文重新
    推断命题。资料仍留在 Runtime/Companion，provider 只读 SourceRecord。
    """
    from pure_integer_ai.experiments.conversation_runtime_material_language import (
        RuntimeMaterialLanguageObservation,
    )
    if not isinstance(observation, RuntimeMaterialLanguageObservation):
        raise TypeError("observation 必须是 RuntimeMaterialLanguageObservation")
    if not isinstance(qualification, RawPropositionQualification):
        raise TypeError("qualification 类型错误")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    if type(question) is not str or not question.strip():
        raise RuntimeMaterialResponseError("question 必须是非空文本")
    if type(relation_index) is not int or relation_index < 0:
        raise RuntimeMaterialResponseError("relation_index 必须是非负整数")
    candidates = observation.relation_candidates
    if relation_index >= len(candidates):
        raise RuntimeMaterialResponseError("relation_index 超出真实 candidate")
    relation = candidates[relation_index]
    gate = relation.qualification_gate(qualification)
    binding = RuntimeMaterialAnswerBinding(
        question,
        observation.ingest.event.memory_item_key,
        gate,
        source_title,
        source_url,
    )
    return RuntimeMaterialAnswerProvider(
        RuntimeMaterialReadIndex.build(observation.ingest.memory_after),
        source_records,
        (binding,),
    )


def build_runtime_material_response_provider(
        specs: tuple[RuntimeMaterialResponseSpec, ...],
        *,
        source_records: SourceRecordRepository,
        ) -> RuntimeMaterialResponseProvider:
    """从多个真实 observation/candidate 构造 fail-closed response provider。"""
    if not isinstance(specs, tuple) or not specs:
        raise RuntimeMaterialResponseError("response specs 不能为空 tuple")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    from pure_integer_ai.experiments.conversation_runtime_material_language import (
        RuntimeMaterialLanguageObservation,
    )
    candidates: list[_RuntimeMaterialResponseCandidate] = []
    for spec in specs:
        if not isinstance(spec, RuntimeMaterialResponseSpec):
            raise TypeError("response spec 类型错误")
        if not isinstance(spec.observation, RuntimeMaterialLanguageObservation):
            raise TypeError("response observation 类型错误")
        if not isinstance(spec.qualification, RawPropositionQualification):
            raise TypeError("response qualification 类型错误")
        if type(spec.question) is not str or not spec.question.strip():
            raise RuntimeMaterialResponseError("response question 必须是非空文本")
        if type(spec.relation_index) is not int or spec.relation_index < 0:
            raise RuntimeMaterialResponseError("relation_index 必须是非负整数")
        observations = spec.observation.relation_candidates
        if spec.relation_index >= len(observations):
            raise RuntimeMaterialResponseError("relation_index 超出真实 candidate")
        relation = observations[spec.relation_index]
        gate = relation.qualification_gate(spec.qualification)
        binding = RuntimeMaterialAnswerBinding(
            spec.question,
            spec.observation.ingest.event.memory_item_key,
            gate,
            spec.source_title,
            spec.source_url,
        )
        candidates.append(_RuntimeMaterialResponseCandidate(
            spec.question,
            RuntimeMaterialReadIndex.build(spec.observation.ingest.memory_after),
            binding,
            spec.relation_index,
        ))
    ordered = tuple(sorted(candidates, key=lambda item: item.sort_key()))
    identities = tuple((item.question, item.binding.memory_item_key,
                        item.binding.qualification_gate.proposition_binding
                        .proposition_id) for item in ordered)
    if len(set(identities)) != len(identities):
        raise RuntimeMaterialResponseError("response spec 重复绑定")
    return RuntimeMaterialResponseProvider(source_records, ordered)


def _surface(value: str, *, label: str) -> tuple[int, ...]:
    if type(value) is not str or not value.strip():
        raise RuntimeMaterialResponseError(f"{label} 必须是非空文本")
    return tuple(encode_utf8_v1(tuple(ord(item) for item in value.strip())))


def organize_runtime_material_response(
        ingest: RuntimeMaterialIngest,
        qualification_gate: RuntimeMaterialQualificationGate,
        *,
        support_surfaces: tuple[str, ...] = (),
        fallback_surfaces: tuple[str, ...] = (),
        ) -> ResponseOrganizationPlan:
    """消费已资格化 Runtime 资料，复用 claim/support/repair 组织协议。

    原文只作为已留档 SourceRecord 的 claim 载荷；关系、资格和 response-act
    全部来自上游 gate。此函数不检索、不猜测、不写 Core/Runtime，也不把支持
    或修复文本硬编码进系统。
    """
    if not isinstance(ingest, RuntimeMaterialIngest):
        raise TypeError("ingest 类型错误")
    if not isinstance(qualification_gate, RuntimeMaterialQualificationGate):
        raise TypeError("qualification_gate 类型错误")
    if not isinstance(support_surfaces, tuple) or not isinstance(
            fallback_surfaces, tuple):
        raise TypeError("support/fallback surfaces 必须是 tuple")
    act = qualification_gate.response_act
    if act not in {"ANSWER", "UNKNOWN", "CLARIFY"}:
        raise RuntimeMaterialResponseError("response_act 未注册")
    source_key = ingest.event.event_key
    segments: list[ResponseSegment] = []
    if act == "ANSWER":
        segments.append(ResponseSegment(
            SEGMENT_CLAIM,
            _surface(ingest.source_record.raw_text, label="material claim"),
            source_key,
            0,
        ))
        for ordinal, support in enumerate(support_surfaces):
            segments.append(ResponseSegment(
                SEGMENT_SUPPORT, _surface(support, label="support"),
                source_key, ordinal,
            ))
    else:
        for ordinal, fallback in enumerate(fallback_surfaces):
            segments.append(ResponseSegment(
                SEGMENT_REPAIR if act == "CLARIFY" else SEGMENT_QUALIFIER,
                _surface(fallback, label="fallback"), source_key, ordinal,
            ))
    if not segments:
        raise RuntimeMaterialResponseError("没有可组织的回答段")
    ordered = tuple(sorted(
        segments,
        key=lambda item: (item.segment_kind, item.ordinal,
                          item.canonical_record()),
    ))
    replay: list[int] = [RUNTIME_MATERIAL_RESPONSE_PROTOCOL_V1]
    for value in (
            ingest.event.event_key,
            qualification_gate.consumer_result.integer_record,
            tuple(item.canonical_record() for item in ordered)):
        if value and isinstance(value[0], tuple):
            replay.append(len(value))
            for nested in value:
                replay.extend((len(nested), *nested))
        else:
            replay.extend((len(value), *value))
    return ResponseOrganizationPlan(
        act,
        ordered,
        source_key,
        digest_bytes(encode_integer_tuple(tuple(replay))),
    )


__all__ = [
    "build_runtime_material_answer_provider",
    "build_runtime_material_response_provider",
    "RUNTIME_MATERIAL_RESPONSE_PROTOCOL_V1",
    "RuntimeMaterialResponseProvider",
    "RuntimeMaterialResponseError",
    "RuntimeMaterialResponseSpec",
    "organize_runtime_material_response",
]
