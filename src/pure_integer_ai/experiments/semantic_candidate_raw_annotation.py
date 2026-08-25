"""S-02 semantic candidate 到 T1-G0..G3 raw 资格链的严格适配。

该模块只做来源和证据身份的投影，不从文本推断 relation、argument 或真值。
``RawPropositionRelationEvidence`` 仍必须由上游显式提供；适配器仅要求它的
proposition identity 与 SemanticPropositionCandidate 完整一致，然后交给既有
``RuntimeMaterialQualificationGate`` 消费。核心稳定身份保持整数 tuple，可在
不具备 Python 对象模型的实现中按同样的字段复现。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticPropositionCandidate,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    RuntimeMaterialIngest,
    RuntimeMaterialQualificationGate,
)


SEMANTIC_RAW_ANNOTATION_PROTOCOL_V1 = 1


class SemanticRawAnnotationError(ValueError):
    """semantic candidate、raw observation 或资格链身份不闭合。"""


def semantic_candidate_proposition_id(
        candidate: SemanticPropositionCandidate,
        ) -> str:
    """把完整 Proposition stable key 投影为可跨语言回读的 metadata id。

    这里不使用 hash，也不携带 surface；十进制整数序列带长度边界由
    ``ObjectIdentity.stable_key`` 自身提供。该字符串只在 raw 合同中作
    identity bridge，事实内容仍不进入它。
    """
    if not isinstance(candidate, SemanticPropositionCandidate):
        raise TypeError("candidate 类型错误")
    key = candidate.definition.proposition.stable_key()
    if not key or any(type(item) is not int or item < 0 for item in key):
        raise SemanticRawAnnotationError("candidate Proposition stable key 非法")
    return "semantic-proposition-v1:" + ".".join(str(item) for item in key)


@dataclass(frozen=True, slots=True)
class SemanticCandidateRawAnnotation:
    """一个已有语义候选与完整 G0-G3 raw annotation 的只读闭包。"""

    candidate: SemanticPropositionCandidate
    source: SourceRef
    scope: ScopeIdentity
    observation: RawTextObservation
    lexical_evidence: tuple[RawLexicalEvidence, ...]
    proposition: RawPropositionRelationEvidence
    qualification: RawPropositionQualification
    material_evidence_id: str
    anchor_unit_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SemanticPropositionCandidate):
            raise TypeError("candidate 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("scope 类型错误")
        if self.scope.source != self.source:
            raise SemanticRawAnnotationError("scope/source 漂移")
        if not isinstance(self.observation, RawTextObservation):
            raise TypeError("observation 类型错误")
        if (not isinstance(self.lexical_evidence, tuple)
                or not self.lexical_evidence
                or any(not isinstance(item, RawLexicalEvidence)
                       for item in self.lexical_evidence)):
            raise TypeError("lexical_evidence 必须是非空 tuple")
        if not isinstance(self.proposition, RawPropositionRelationEvidence):
            raise TypeError("proposition 类型错误")
        if not isinstance(self.qualification, RawPropositionQualification):
            raise TypeError("qualification 类型错误")
        for value, label in ((self.material_evidence_id, "material_evidence_id"),
                             (self.anchor_unit_id, "anchor_unit_id")):
            if type(value) is not str or not value.strip():
                raise SemanticRawAnnotationError(f"{label} 不能为空")

        if semantic_source(self.candidate.definition.proposition) != self.source:
            raise SemanticRawAnnotationError("candidate Proposition/source 漂移")
        if self.candidate.hypothesis.scope != self.scope:
            raise SemanticRawAnnotationError("candidate hypothesis/scope 漂移")
        if self.proposition.proposition_id != self.proposition_id:
            raise SemanticRawAnnotationError(
                "raw proposition_id 未绑定 semantic candidate")
        if self.qualification.proposition_id != self.proposition.proposition_id:
            raise SemanticRawAnnotationError("qualification/proposition 漂移")
        if self.anchor_unit_id not in {
                item.unit_id for item in self.observation.units}:
            raise SemanticRawAnnotationError("anchor_unit_id 不在 observation units")
        argument_units = {item.unit_id for item in self.proposition.arguments}
        if self.anchor_unit_id not in argument_units:
            raise SemanticRawAnnotationError(
                "proposition 未携带 semantic anchor 的 lexical evidence")

    @property
    def proposition_id(self) -> str:
        return semantic_candidate_proposition_id(self.candidate)

    @property
    def observation_digest(self) -> tuple[int, ...]:
        return digest_bytes(bytes(self.observation.raw_bytes))

    @classmethod
    def from_runtime_material(
            cls,
            ingest: RuntimeMaterialIngest,
            candidate: SemanticPropositionCandidate,
            observation: RawTextObservation,
            lexical_evidence: tuple[RawLexicalEvidence, ...],
            proposition: RawPropositionRelationEvidence,
            qualification: RawPropositionQualification,
            *,
            material_evidence_id: str,
            anchor_unit_id: str,
            ) -> "SemanticCandidateRawAnnotation":
        """从 Runtime 资料建立严格来源化适配，不执行 Core promotion。"""
        if not isinstance(ingest, RuntimeMaterialIngest):
            raise TypeError("ingest 类型错误")
        if ingest.capsule.source != semantic_source(candidate.definition.proposition):
            raise SemanticRawAnnotationError("Runtime material/candidate source 漂移")
        if ingest.capsule.scope != candidate.hypothesis.scope:
            raise SemanticRawAnnotationError("Runtime material/candidate scope 漂移")
        if tuple(observation.raw_bytes) != tuple(
                ingest.source_record.raw_text.encode("utf-8")):
            raise SemanticRawAnnotationError("Runtime material/observation raw 漂移")
        return cls(
            candidate,
            ingest.capsule.source,
            ingest.capsule.scope,
            observation,
            lexical_evidence,
            proposition,
            qualification,
            material_evidence_id,
            anchor_unit_id,
        )

    def qualification_gate(self) -> RuntimeMaterialQualificationGate:
        """把已闭合 annotation 交给既有 G0-G3 consumer。"""
        return RuntimeMaterialQualificationGate.from_annotation(
            self.material_evidence_id,
            self.observation,
            self.lexical_evidence,
            self.proposition,
            self.qualification,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回仅含整数的适配记录，供跨语言 receipt/回放使用。"""
        result = [SEMANTIC_RAW_ANNOTATION_PROTOCOL_V1]
        for value in (
                self.candidate.definition.proposition.stable_key(),
                self.source.stable_key(), self.scope.stable_key(),
                self.observation.canonical_record(),
                tuple(item.canonical_record() for item in self.lexical_evidence),
                self.proposition.canonical_record(),
                self.qualification.canonical_record(),
                tuple(ord(item) for item in self.material_evidence_id),
                tuple(ord(item) for item in self.anchor_unit_id)):
            if value and isinstance(value[0], tuple):
                result.append(len(value))
                for nested in value:
                    result.extend((len(nested), *nested))
            else:
                result.extend((len(value), *value))
        return tuple(result)


__all__ = [
    "SEMANTIC_RAW_ANNOTATION_PROTOCOL_V1",
    "SemanticCandidateRawAnnotation",
    "SemanticRawAnnotationError",
    "semantic_candidate_proposition_id",
]
