"""H-05 可复用的来源隔离候选、预测揭示和 H-04 消费投影。

本模块不解释 structure、cue、Sense、relation 或 boundary 的领域含义。调用方提交
完整一等对象绑定；forming 只登记 unknown Evidence，独立 observation 必须先形成
Prediction，再由外部 verifier 揭示 support/refute/unknown。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ArchiveDirective,
    HypothesisResolver,
    ReplacementDirective,
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_EVIDENCE_HASHER = Hasher("evidence_candidate.evidence.v1")
_DEFINITION_VERSION = 1
_PREDICTION_VERSION = 1
_BINDING_VERSION = 2

CANDIDATE_AS_SUBJECT = 1
CANDIDATE_AS_OBJECT = 2
_CANDIDATE_ENDPOINTS = frozenset({
    CANDIDATE_AS_SUBJECT,
    CANDIDATE_AS_OBJECT,
})


class EvidenceCandidateError(RuntimeError):
    """候选来源边界、预测揭示或 resolver 投影不一致。"""


def _strict_key(value, *, where: str,
                allow_empty: bool = False) -> tuple[int, ...]:
    """校验由课程、图或领域 mapper 注入的严格整数键。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给完整可变长整数键增加长度前缀。"""
    return len(value), *value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False) -> tuple[tuple[int, ...], int]:
    """从长度前缀整数流读取一个完整键段并返回新游标。"""
    if cursor >= len(values):
        raise ValueError(f"候选稳定键缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"候选稳定键 {label} 长度非法")
    if cursor + size > len(values):
        raise ValueError(f"候选稳定键 {label} 被截断")
    return values[cursor:cursor + size], cursor + size


@dataclass(frozen=True, order=True)
class CandidateBinding:
    """候选对象在一等 predicate 边的一端绑定一个完整 typed 对象。"""

    predicate: ObjectIdentity
    value: ObjectIdentity
    ordinal: int = 0
    candidate_endpoint: int = CANDIDATE_AS_SUBJECT

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, ObjectIdentity):
            raise TypeError("CandidateBinding.predicate 必须是 ObjectIdentity")
        if self.predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("CandidateBinding.predicate 必须是一等 Concept")
        if not isinstance(self.value, ObjectIdentity):
            raise TypeError("CandidateBinding.value 必须是 ObjectIdentity")
        assert_int(
            self.ordinal,
            self.candidate_endpoint,
            _where="CandidateBinding",
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("CandidateBinding.ordinal 必须为非负严格整数")
        if (type(self.candidate_endpoint) is not int
                or self.candidate_endpoint not in _CANDIDATE_ENDPOINTS):
            raise ValueError("CandidateBinding.candidate_endpoint 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回保留 predicate、对端、方向和值序号的无歧义完整键。"""
        predicate = self.predicate.stable_key()
        value = self.value.stable_key()
        return (
            _BINDING_VERSION,
            *_pack(predicate),
            *_pack(value),
            self.ordinal,
            self.candidate_endpoint,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "CandidateBinding":
        """从完整键恢复 binding，拒绝截断、尾随和损坏身份。"""
        values = _strict_key(key, where="CandidateBinding.stable_key")
        if values[0] != _BINDING_VERSION:
            raise ValueError("CandidateBinding 稳定键版本非法")
        predicate_key, cursor = _take(values, 1, label="predicate")
        value_key, cursor = _take(values, cursor, label="value")
        if cursor + 2 != len(values):
            raise ValueError(
                "CandidateBinding 稳定键缺 ordinal/endpoint 或含尾随数据")
        return cls(
            ObjectIdentity.from_stable_key(predicate_key),
            ObjectIdentity.from_stable_key(value_key),
            values[cursor],
            values[cursor + 1],
        )

    def subject(self, candidate: ObjectIdentity) -> ObjectIdentity:
        """按稳定端点方向返回 statement subject。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        if self.candidate_endpoint == CANDIDATE_AS_SUBJECT:
            return candidate
        return self.value

    def object(self, candidate: ObjectIdentity) -> ObjectIdentity:
        """按稳定端点方向返回 statement object。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        if self.candidate_endpoint == CANDIDATE_AS_OBJECT:
            return candidate
        return self.value


@dataclass(frozen=True)
class EvidenceCandidateProtocol:
    """候选 kind、形成 reason、aggregate 来源和形成样本条件。"""

    hypothesis_kind_key: tuple[int, ...]
    formation_reason_key: tuple[int, ...]
    aggregate_source: SourceRef
    aggregate_scope: ScopeIdentity
    minimum_forming_sources: int

    def __post_init__(self) -> None:
        _strict_key(
            self.hypothesis_kind_key,
            where="EvidenceCandidateProtocol.hypothesis_kind_key",
        )
        _strict_key(
            self.formation_reason_key,
            where="EvidenceCandidateProtocol.formation_reason_key",
        )
        if not isinstance(self.aggregate_source, SourceRef):
            raise TypeError("aggregate_source 必须是 SourceRef")
        if not isinstance(self.aggregate_scope, ScopeIdentity):
            raise TypeError("aggregate_scope 必须是 ScopeIdentity")
        if self.aggregate_scope.source != self.aggregate_source:
            raise ValueError("aggregate_scope 必须指向 aggregate_source")
        assert_int(
            self.minimum_forming_sources,
            _where="EvidenceCandidateProtocol.minimum_forming_sources",
        )
        if (type(self.minimum_forming_sources) is not int
                or self.minimum_forming_sources <= 0):
            raise ValueError("minimum_forming_sources 必须为严格正整数")


@dataclass(frozen=True)
class EvidenceCandidateDefinition:
    """一个一等候选、完整图绑定、竞争组和独立 forming 来源集合。"""

    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    bindings: tuple[CandidateBinding, ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        _strict_key(
            self.competition_key,
            where="EvidenceCandidateDefinition.competition_key",
        )
        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise ValueError("候选必须包含至少一个一等图绑定")
        if any(not isinstance(item, CandidateBinding)
               for item in self.bindings):
            raise TypeError("bindings 只能包含 CandidateBinding")
        binding_slots = tuple(
            (item.predicate, item.candidate_endpoint, item.ordinal)
            for item in self.bindings)
        if len(set(binding_slots)) != len(binding_slots):
            raise ValueError("同一 predicate/endpoint/ordinal 不得绑定多个对象")
        if not isinstance(self.forming_sources, tuple):
            raise TypeError("forming_sources 必须是 SourceRef tuple")
        if any(not isinstance(item, SourceRef)
               for item in self.forming_sources):
            raise TypeError("forming_sources 只能包含 SourceRef")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("forming_sources 不得重复同一来源")
        object.__setattr__(self, "bindings", tuple(sorted(
            self.bindings, key=CandidateBinding.stable_key)))
        object.__setattr__(self, "forming_sources", tuple(sorted(
            self.forming_sources, key=SourceRef.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """完整编码候选、绑定和 forming 来源，不以 hash 替代身份。"""
        candidate = self.candidate.stable_key()
        values: list[int] = [
            _DEFINITION_VERSION,
            *_pack(candidate),
            *_pack(self.competition_key),
            len(self.bindings),
        ]
        for binding in self.bindings:
            values.extend(_pack(binding.stable_key()))
        values.append(len(self.forming_sources))
        for source in self.forming_sources:
            values.extend(_pack(source.stable_key()))
        return tuple(values)

    def hypothesis(self, protocol: EvidenceCandidateProtocol) -> HypothesisKey:
        """把完整定义映射到 aggregate 来源的 H-00 候选。"""
        if not isinstance(protocol, EvidenceCandidateProtocol):
            raise TypeError("protocol 必须是 EvidenceCandidateProtocol")
        return HypothesisKey(
            protocol.hypothesis_kind_key,
            self.stable_key(),
            self.competition_key,
            protocol.aggregate_scope,
            protocol.aggregate_source,
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...]) -> "EvidenceCandidateDefinition":
        """从 Hypothesis candidate key 恢复全部绑定和 forming 来源。"""
        values = _strict_key(
            key, where="EvidenceCandidateDefinition.stable_key")
        if values[0] != _DEFINITION_VERSION:
            raise ValueError("候选定义稳定键版本非法")
        candidate_key, cursor = _take(values, 1, label="candidate")
        competition_key, cursor = _take(
            values, cursor, label="competition")
        if cursor >= len(values):
            raise ValueError("候选定义缺少 binding 数量")
        binding_count = values[cursor]
        cursor += 1
        if binding_count <= 0:
            raise ValueError("候选定义 binding 数量非法")
        bindings: list[CandidateBinding] = []
        for index in range(binding_count):
            binding_key, cursor = _take(
                values, cursor, label=f"binding[{index}]")
            bindings.append(CandidateBinding.from_stable_key(binding_key))
        if cursor >= len(values):
            raise ValueError("候选定义缺少 forming source 数量")
        source_count = values[cursor]
        cursor += 1
        if source_count < 0:
            raise ValueError("候选定义 forming source 数量非法")
        sources: list[SourceRef] = []
        for index in range(source_count):
            source_key, cursor = _take(
                values, cursor, label=f"forming_source[{index}]")
            sources.append(SourceRef.from_stable_key(source_key))
        if cursor != len(values):
            raise ValueError("候选定义稳定键含尾随数据")
        restored = cls(
            ObjectIdentity.from_stable_key(candidate_key),
            competition_key,
            tuple(bindings),
            tuple(sources),
        )
        if restored.stable_key() != values:
            raise ValueError("候选定义稳定键不能规范化重放")
        return restored


@dataclass(frozen=True)
class CandidatePrediction:
    """在揭示结果前冻结的候选预测及其可见输入。"""

    hypothesis: HypothesisKey
    observation: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    visible_inputs: tuple[ObjectIdentity, ...]
    predicted: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("prediction.hypothesis 必须是 HypothesisKey")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("prediction.observation 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("prediction.scope 必须是 ScopeIdentity")
        if self.scope.source != self.observation:
            raise ValueError("prediction scope 必须指向 observation")
        _strict_key(self.event_key, where="CandidatePrediction.event_key")
        if not isinstance(self.visible_inputs, tuple) or not self.visible_inputs:
            raise ValueError("prediction 必须保存非空可见输入")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.visible_inputs):
            raise TypeError("visible_inputs 只能包含 ObjectIdentity")
        if not isinstance(self.predicted, ObjectIdentity):
            raise TypeError("predicted 必须是 ObjectIdentity")

    def stable_key(self) -> tuple[int, ...]:
        """返回不包含揭示标签的完整 prediction identity。"""
        hypothesis = self.hypothesis.stable_key()
        observation = self.observation.stable_key()
        scope = self.scope.stable_key()
        predicted = self.predicted.stable_key()
        values: list[int] = [
            _PREDICTION_VERSION,
            *_pack(hypothesis),
            *_pack(observation),
            *_pack(scope),
            *_pack(self.event_key),
            len(self.visible_inputs),
        ]
        for item in self.visible_inputs:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(predicted))
        return tuple(values)

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "CandidatePrediction":
        """从 Evidence 内嵌的完整稳定键恢复已揭示 prediction。"""
        key = _strict_key(key, where="CandidatePrediction.stable_key")
        if key[0] != _PREDICTION_VERSION:
            raise ValueError("CandidatePrediction 版本未注册")
        hypothesis_key, cursor = _take(
            key, 1, label="CandidatePrediction.hypothesis")
        observation_key, cursor = _take(
            key, cursor, label="CandidatePrediction.observation")
        scope_key, cursor = _take(
            key, cursor, label="CandidatePrediction.scope")
        event_key, cursor = _take(
            key, cursor, label="CandidatePrediction.event_key")
        if cursor >= len(key):
            raise ValueError("CandidatePrediction 缺 visible input 数量")
        visible_count = key[cursor]
        if visible_count <= 0:
            raise ValueError("CandidatePrediction visible input 数量非法")
        cursor += 1
        visible_inputs: list[ObjectIdentity] = []
        for index in range(visible_count):
            item_key, cursor = _take(
                key,
                cursor,
                label=f"CandidatePrediction.visible_inputs[{index}]",
            )
            visible_inputs.append(ObjectIdentity.from_stable_key(item_key))
        predicted_key, cursor = _take(
            key, cursor, label="CandidatePrediction.predicted")
        if cursor != len(key):
            raise ValueError("CandidatePrediction 稳定键含尾随数据")
        return cls(
            HypothesisKey.from_stable_key(hypothesis_key),
            SourceRef.from_stable_key(observation_key),
            ScopeIdentity.from_stable_key(scope_key),
            event_key,
            tuple(visible_inputs),
            ObjectIdentity.from_stable_key(predicted_key),
        )


@dataclass(frozen=True)
class CandidateVerification:
    """独立 verifier 对一条已冻结 prediction 的三态揭示。"""

    stance: int
    reason_key: tuple[int, ...]
    source: SourceRef
    authority: ObjectIdentity
    authority_version: tuple[int, ...]
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        assert_int(self.stance, _where="CandidateVerification.stance")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("verification stance 未注册")
        _strict_key(self.reason_key, where="CandidateVerification.reason_key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("verification.source 必须是 SourceRef")
        if not isinstance(self.authority, ObjectIdentity):
            raise TypeError("verification.authority 必须是 ObjectIdentity")
        _strict_key(
            self.authority_version,
            where="CandidateVerification.authority_version",
        )
        _strict_key(
            self.trace, where="CandidateVerification.trace", allow_empty=True)

    def payload_for(self, prediction: CandidatePrediction) -> tuple[int, ...]:
        """把 prediction、authority、版本和 trace 完整编码进 Evidence。"""
        prediction_key = prediction.stable_key()
        authority = self.authority.stable_key()
        return (
            *_pack(prediction_key),
            *_pack(authority),
            *_pack(self.authority_version),
            *_pack(self.trace),
        )


@dataclass(frozen=True)
class ActiveEvidenceCandidate:
    """供下游采用的 active+supported+adopted 候选投影。"""

    definition: EvidenceCandidateDefinition
    hypothesis: HypothesisKey
    snapshot: HypothesisSnapshot
    decision: ResolverDecision


@dataclass(frozen=True)
class CandidateRecognitionRecord:
    """一条可从持久 Evidence 无损恢复的 prediction 与 verifier 结果。"""

    prediction: CandidatePrediction
    verification: CandidateVerification
    evidence: EvidenceRecord


class EvidenceCandidateEngine:
    """统一管理 forming、prediction、Evidence、H-04 和只读采用投影。"""

    def __init__(
            self, protocol: EvidenceCandidateProtocol, *,
            ledger: HypothesisLedger | None = None,
            resolver: HypothesisResolver | None = None) -> None:
        if not isinstance(protocol, EvidenceCandidateProtocol):
            raise TypeError("protocol 必须是 EvidenceCandidateProtocol")
        self.protocol = protocol
        self.ledger = ledger or HypothesisLedger()
        decision_sink = self.ledger.event_sink
        if decision_sink is not None and not callable(
                getattr(decision_sink, "append_decision", None)):
            decision_sink = None
        self.resolver = resolver or HypothesisResolver(
            self.ledger, sink=decision_sink)
        if self.resolver.ledger is not self.ledger:
            raise ValueError("候选 resolver 必须绑定同一 H-00 ledger")
        self._definitions: dict[
            HypothesisKey, EvidenceCandidateDefinition] = {}
        self._predictions: dict[
            tuple[HypothesisKey, SourceRef, tuple[int, ...]],
            CandidatePrediction,
        ] = {}

    @classmethod
    def from_history(
            cls, protocol: EvidenceCandidateProtocol, *,
            definitions: tuple[EvidenceCandidateDefinition, ...],
            ledger: HypothesisLedger,
            decisions: tuple[ResolverDecision, ...],
            ) -> "EvidenceCandidateEngine":
        """从持久 ledger/decision 和调用方已核验定义恢复候选 engine。"""
        if not isinstance(protocol, EvidenceCandidateProtocol):
            raise TypeError("from_history.protocol 类型错误")
        if (not isinstance(definitions, tuple)
                or any(not isinstance(item, EvidenceCandidateDefinition)
                       for item in definitions)):
            raise TypeError("definitions 必须是 EvidenceCandidateDefinition tuple")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("from_history.ledger 类型错误")
        by_hypothesis = {
            item.hypothesis(protocol): item for item in definitions
        }
        if len(by_hypothesis) != len(definitions):
            raise ValueError("恢复定义重复同一 Hypothesis")
        if frozenset(by_hypothesis) != frozenset(ledger.hypotheses()):
            raise ValueError("恢复定义与持久 ledger Hypothesis 集合不一致")
        decision_sink = ledger.event_sink
        if decision_sink is not None and not callable(
                getattr(decision_sink, "append_decision", None)):
            decision_sink = None
        resolver = HypothesisResolver.from_history(
            ledger, decisions, sink=decision_sink)
        engine = cls(protocol, ledger=ledger, resolver=resolver)
        engine._definitions = by_hypothesis
        for hypothesis, definition in by_hypothesis.items():
            formation_payloads = {
                (ordinal, len(definition.forming_sources))
                for ordinal in range(len(definition.forming_sources))
            }
            for evidence in ledger.evidence_history(hypothesis):
                if (evidence.source in definition.forming_sources
                        and evidence.reason_key
                        == protocol.formation_reason_key
                        and evidence.payload in formation_payloads):
                    continue
                prediction = cls._prediction_from_evidence(evidence)
                route = (
                    prediction.hypothesis,
                    prediction.observation,
                    prediction.event_key,
                )
                existing = engine._predictions.get(route)
                if existing is not None and existing != prediction:
                    raise EvidenceCandidateError(
                        "恢复 Evidence 把同一 prediction route 绑定到不同输入")
                engine._predictions[route] = prediction
        return engine

    def predictions(self) -> tuple[CandidatePrediction, ...]:
        """返回已由 Evidence 揭示的完整 prediction 历史。"""
        return tuple(sorted(
            self._predictions.values(),
            key=lambda item: item.stable_key(),
        ))

    def recognition_history(
            self,
            hypothesis: HypothesisKey,
            ) -> tuple[CandidateRecognitionRecord, ...]:
        """恢复指定候选全部非 forming recognition，供领域 runtime 重建幂等游标。"""
        self.definition(hypothesis)
        formation_sources = self._definitions[hypothesis].forming_sources
        formation_payloads = {
            (ordinal, len(formation_sources))
            for ordinal in range(len(formation_sources))
        }
        result = []
        for evidence in self.ledger.evidence_history(hypothesis):
            if (evidence.source in formation_sources
                    and evidence.reason_key
                    == self.protocol.formation_reason_key
                    and evidence.payload in formation_payloads):
                continue
            prediction, verification = self._recognition_from_evidence(
                evidence)
            result.append(CandidateRecognitionRecord(
                prediction,
                verification,
                evidence,
            ))
        return tuple(result)

    def definitions(self) -> tuple[EvidenceCandidateDefinition, ...]:
        """返回当前 owner 已核验的全部候选完整定义。"""
        return tuple(
            self._definitions[key]
            for key in sorted(
                self._definitions,
                key=lambda item: item.stable_key(),
            )
        )

    def register(
            self, definition: EvidenceCandidateDefinition, *,
            timestamp_base: int = 0) -> HypothesisKey:
        """形成候选并只追加 unknown Evidence，不把样本数当支持或掌握。"""
        return self.register_many(((definition, timestamp_base),))[0]

    def register_many(
            self,
            requests: tuple[tuple[EvidenceCandidateDefinition, int], ...],
            ) -> tuple[HypothesisKey, ...]:
        """整批预检并登记 forming，使 owner 复制次数不随候选数线性增长。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("register_many requests 必须是非空 tuple")
        normalized: list[
            tuple[EvidenceCandidateDefinition, int, HypothesisKey]
        ] = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("register_many request 必须是 definition/timestamp 对")
            definition, timestamp_base = request
            if not isinstance(definition, EvidenceCandidateDefinition):
                raise TypeError("definition 必须是 EvidenceCandidateDefinition")
            assert_int(timestamp_base, _where="EvidenceCandidateEngine.register")
            if type(timestamp_base) is not int or timestamp_base < 0:
                raise ValueError("timestamp_base 必须为非负严格整数")
            if len(definition.forming_sources) < (
                    self.protocol.minimum_forming_sources):
                raise EvidenceCandidateError(
                    "形成（forming）独立来源数未达到注入条件")
            if self.protocol.aggregate_source in definition.forming_sources:
                raise EvidenceCandidateError(
                    "aggregate manifest 不得冒充 forming observation")
            normalized.append((
                definition,
                timestamp_base,
                definition.hypothesis(self.protocol),
            ))
        hypotheses = tuple(item[2] for item in normalized)
        if len(set(hypotheses)) != len(hypotheses):
            raise EvidenceCandidateError("同批 forming 不得重复 Hypothesis")

        pending = []
        probe = self.ledger.clone()
        for definition, timestamp_base, hypothesis in normalized:
            existing = self._definitions.get(hypothesis)
            if existing is not None:
                if existing != definition:
                    raise EvidenceCandidateError(
                        "同一 Hypothesis 绑定了不同候选定义")
                self._validate_existing_formation(
                    definition,
                    hypothesis,
                    timestamp_base=timestamp_base,
                )
                continue
            self._register_into(
                probe,
                definition,
                hypothesis,
                timestamp_base=timestamp_base,
            )
            pending.append((definition, timestamp_base, hypothesis))
        for definition, timestamp_base, hypothesis in pending:
            self._register_into(
                self.ledger,
                definition,
                hypothesis,
                timestamp_base=timestamp_base,
            )
            self._definitions[hypothesis] = definition
        return hypotheses

    def predict(
            self, hypothesis: HypothesisKey, *, observation: SourceRef,
            scope: ScopeIdentity, event_key: tuple[int, ...],
            visible_inputs: tuple[ObjectIdentity, ...],
            predicted: ObjectIdentity) -> CandidatePrediction:
        """在读取核验结果前冻结 prediction，并拒绝 forming 来源泄入识别。"""
        definition = self.definition(hypothesis)
        if observation in definition.forming_sources:
            raise EvidenceCandidateError(
                "forming observation 不得再次作为 recognition Evidence")
        prediction = CandidatePrediction(
            hypothesis,
            observation,
            scope,
            _strict_key(event_key, where="predict.event_key"),
            visible_inputs,
            predicted,
        )
        route = (hypothesis, observation, prediction.event_key)
        existing = self._predictions.get(route)
        if existing is not None and existing != prediction:
            raise EvidenceCandidateError(
                "同一候选、来源和事件不得在揭示前改写 prediction")
        self._predictions[route] = prediction
        return prediction

    def reveal(
            self, prediction: CandidatePrediction,
            verification: CandidateVerification, *,
            timestamp_seq: int) -> EvidenceRecord:
        """把独立 verifier 结果追加为 Evidence，同一 observation/event 只计一次。"""
        if not isinstance(prediction, CandidatePrediction):
            raise TypeError("prediction 必须是 CandidatePrediction")
        if not isinstance(verification, CandidateVerification):
            raise TypeError("verification 必须是 CandidateVerification")
        assert_int(timestamp_seq, _where="EvidenceCandidateEngine.reveal")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        definition = self.definition(prediction.hypothesis)
        if prediction.observation in definition.forming_sources:
            raise EvidenceCandidateError("forming observation 不得产生揭示 Evidence")
        route = (
            prediction.hypothesis,
            prediction.observation,
            prediction.event_key,
        )
        frozen = self._predictions.get(route)
        if frozen != prediction:
            raise EvidenceCandidateError("prediction 未在当前 engine 中先行冻结")
        payload = verification.payload_for(prediction)
        evidence_id = _EVIDENCE_HASHER.h63((
            prediction.hypothesis.stable_key(),
            prediction.observation.stable_key(),
            prediction.event_key,
        )) or 1
        evidence = EvidenceRecord(
            evidence_id,
            prediction.hypothesis,
            verification.stance,
            verification.reason_key,
            verification.source,
            timestamp_seq,
            payload=payload,
        )
        return self.ledger.append_evidence(evidence)

    def resolve(
            self, hypothesis: HypothesisKey, *, timestamp_seq: int,
            scorers=(), archive_refuted: bool = False,
            replacement: HypothesisKey | None = None) -> ResolverDecision:
        """提交 H-04 决策；显式归档和替代均要求当前定向 refute。"""
        self.definition(hypothesis)
        snapshot = self.ledger.snapshot(hypothesis)
        refute_id = (
            snapshot.refute_evidence_ids[-1]
            if snapshot.refute_evidence_ids else 0)
        if archive_refuted and replacement is not None:
            raise EvidenceCandidateError("同一候选不得同时 archive 和 supersede")
        archives = (
            (ArchiveDirective(hypothesis, refute_id),)
            if archive_refuted else ())
        replacements = (
            (ReplacementDirective(hypothesis, replacement, refute_id),)
            if replacement is not None else ())
        if (archive_refuted or replacement is not None) and refute_id == 0:
            raise EvidenceCandidateError("退出候选必须引用当前 active refute")
        if replacement is not None:
            self.definition(replacement)
        return self.resolver.resolve(
            hypothesis,
            timestamp_seq=timestamp_seq,
            scorers=tuple(scorers),
            archives=archives,
            replacements=replacements,
        )

    def active(self, hypothesis: HypothesisKey) -> ActiveEvidenceCandidate | None:
        """仅投影当前 active+supported 且被最新未陈旧决策 adopted 的候选。"""
        definition = self.definition(hypothesis)
        snapshot = self.ledger.snapshot(hypothesis)
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or snapshot.epistemic_status != EPISTEMIC_SUPPORTED):
            return None
        history = self.resolver.decision_history(hypothesis)
        if not history:
            return None
        decision = history[-1]
        try:
            trace = decision.candidate(hypothesis)
        except KeyError:
            return None
        if trace.after != snapshot or hypothesis not in decision.adopted_hypotheses:
            return None
        return ActiveEvidenceCandidate(
            definition, hypothesis, snapshot, decision)

    def active_competition(
            self, hypothesis: HypothesisKey
            ) -> tuple[ActiveEvidenceCandidate, ...]:
        """返回同一竞争组中的全部可采用候选，不强制选择唯一 winner。"""
        active: list[ActiveEvidenceCandidate] = []
        for snapshot in self.ledger.competition(hypothesis):
            projected = self.active(snapshot.hypothesis)
            if projected is not None:
                active.append(projected)
        return tuple(sorted(
            active, key=lambda item: item.hypothesis.stable_key()))

    def require_unique(
            self, hypothesis: HypothesisKey) -> ActiveEvidenceCandidate:
        """为要求唯一语义的消费者返回单候选，多解或无解均 fail closed。"""
        candidates = self.active_competition(hypothesis)
        if len(candidates) != 1:
            raise LookupError("当前竞争组没有唯一 active adopted 候选")
        return candidates[0]

    def definition(
            self, hypothesis: HypothesisKey) -> EvidenceCandidateDefinition:
        """按完整 Hypothesis 返回候选定义，未知候选不得从键值猜测。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        definition = self._definitions.get(hypothesis)
        if definition is None:
            raise KeyError("Hypothesis 未在当前候选 owner 中登记")
        if definition.hypothesis(self.protocol) != hypothesis:
            raise EvidenceCandidateError("候选定义与 Hypothesis 完整键不一致")
        return definition

    def clone(self) -> "EvidenceCandidateEngine":
        """复制 ledger、resolver、定义和冻结 prediction，供 held-out 写隔离。"""
        ledger = self.ledger.clone()
        cloned = EvidenceCandidateEngine(
            self.protocol,
            ledger=ledger,
            resolver=self.resolver.clone(ledger=ledger),
        )
        cloned._definitions = dict(self._definitions)
        cloned._predictions = dict(self._predictions)
        return cloned

    def state_key(self) -> tuple:
        """返回定义、prediction、H-00 和 H-04 的完整可比较状态。"""
        definitions = tuple(sorted(
            (hypothesis.stable_key(), definition.stable_key())
            for hypothesis, definition in self._definitions.items()
        ))
        predictions = tuple(sorted(
            prediction.stable_key()
            for prediction in self._predictions.values()
        ))
        return (
            definitions,
            predictions,
            self.ledger.state_key(),
            self.resolver.state_key(),
        )

    def _register_into(
            self, ledger: HypothesisLedger,
            definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey, *, timestamp_base: int) -> None:
        """向指定 ledger 登记候选及形成 unknown，供 clone 预检和正式提交复用。"""
        ledger.register(hypothesis)
        for ordinal, source in enumerate(definition.forming_sources):
            ledger.append_evidence(self._formation_evidence(
                definition,
                hypothesis,
                source=source,
                ordinal=ordinal,
                timestamp_base=timestamp_base,
            ))

    def _validate_existing_formation(
            self,
            definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey,
            *,
            timestamp_base: int,
            ) -> None:
        """核验既有 forming Evidence 的来源、ordinal 和逻辑序与课程完全一致。"""
        existing = {
            item.evidence_id: item
            for item in self.ledger.evidence_history(hypothesis)
        }
        for ordinal, source in enumerate(definition.forming_sources):
            expected = self._formation_evidence(
                definition,
                hypothesis,
                source=source,
                ordinal=ordinal,
                timestamp_base=timestamp_base,
            )
            if existing.get(expected.evidence_id) != expected:
                raise EvidenceCandidateError(
                    "既有 forming Evidence 与课程逻辑序或来源不一致")

    def _formation_evidence(
            self,
            definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey,
            *,
            source: SourceRef,
            ordinal: int,
            timestamp_base: int,
            ) -> EvidenceRecord:
        """构造一条来源化 forming unknown Evidence，供预检和提交共用。"""
        evidence_id = _EVIDENCE_HASHER.h63((
            hypothesis.stable_key(),
            self.protocol.formation_reason_key,
            source.stable_key(),
        )) or 1
        return EvidenceRecord(
            evidence_id,
            hypothesis,
            EVIDENCE_UNKNOWN,
            self.protocol.formation_reason_key,
            source,
            timestamp_base + ordinal,
            payload=(ordinal, len(definition.forming_sources)),
        )

    @staticmethod
    def _prediction_from_evidence(
            evidence: EvidenceRecord,
            ) -> CandidatePrediction:
        """从 recognition Evidence 恢复 prediction 并核验剩余 verifier 载荷。"""
        return EvidenceCandidateEngine._recognition_from_evidence(evidence)[0]

    @staticmethod
    def _recognition_from_evidence(
            evidence: EvidenceRecord,
            ) -> tuple[CandidatePrediction, CandidateVerification]:
        """从 Evidence payload 恢复完整 prediction 和独立 verifier 结果。"""
        try:
            prediction_key, cursor = _take(
                evidence.payload, 0, label="Evidence.prediction")
            authority_key, cursor = _take(
                evidence.payload, cursor, label="Evidence.authority")
            authority_version, cursor = _take(
                evidence.payload, cursor, label="Evidence.authority_version")
            trace, cursor = _take(
                evidence.payload,
                cursor,
                label="Evidence.verification_trace",
                allow_empty=True,
            )
            if cursor != len(evidence.payload):
                raise ValueError("recognition Evidence payload 含尾随数据")
            prediction = CandidatePrediction.from_stable_key(prediction_key)
            authority = ObjectIdentity.from_stable_key(authority_key)
            _strict_key(
                authority_version,
                where="Evidence.authority_version",
            )
            _strict_key(
                trace,
                where="Evidence.verification_trace",
                allow_empty=True,
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceCandidateError(
                "非 forming Evidence 无法恢复完整 prediction/verifier 载荷"
            ) from exc
        if prediction.hypothesis != evidence.hypothesis:
            raise EvidenceCandidateError(
                "恢复 prediction 与 Evidence 候选身份不一致")
        if evidence.source is None:
            raise EvidenceCandidateError("recognition Evidence 缺少 verifier 来源")
        verification = CandidateVerification(
            evidence.stance,
            evidence.reason_key,
            evidence.source,
            authority,
            authority_version,
            trace,
        )
        if verification.payload_for(prediction) != evidence.payload:
            raise EvidenceCandidateError(
                "恢复 verifier 与 Evidence payload 不一致")
        return prediction, verification


def binding_from_ref(
        predicate: ObjectIdentity, value: TypedRef, ontology, *,
        ordinal: int = 0,
        candidate_endpoint: int = CANDIDATE_AS_SUBJECT) -> CandidateBinding:
    """把经同一 GraphOntology 核验的 TypedRef 转成候选图绑定。"""
    if not isinstance(value, TypedRef):
        raise TypeError("value 必须是 TypedRef")
    return CandidateBinding(
        predicate,
        ontology.identity_of(value),
        ordinal,
        candidate_endpoint,
    )


__all__ = [
    "CandidateRecognitionRecord",
    "ActiveEvidenceCandidate",
    "CANDIDATE_AS_OBJECT",
    "CANDIDATE_AS_SUBJECT",
    "CandidateBinding",
    "CandidatePrediction",
    "CandidateVerification",
    "EvidenceCandidateDefinition",
    "EvidenceCandidateEngine",
    "EvidenceCandidateError",
    "EvidenceCandidateProtocol",
    "binding_from_ref",
]
