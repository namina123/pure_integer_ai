"""不绑定具体语言或目标类型的纯整数条件预测协议。

预测上下文只保存可见前缀、可见后缀和调用方注入的条件键，目标始终单独保存；
因此 predictor 不能通过上下文对象读取被遮蔽真值。条件计数保留完整来源和事件身份，
重复重放幂等，同一来源可在查询时整体排除。具体 token、span、篇章或成分含义由
上层用一等 ``ObjectIdentity`` 和开放整数键注入。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
    PREFERENCE_EQUIVALENT,
    PREFERENCE_INCOMPARABLE,
    PREFERENCE_LEFT_BETTER,
    PREFERENCE_RIGHT_BETTER,
    ResolverPreference,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_EVIDENCE_HASHER = Hasher("pure_integer_ai.prediction_evidence.v1")
_CONTEXT_VERSION = 1
_TARGET_VERSION = 1
_HYPOTHESIS_VERSION = 1


def _integer_key(value, *, where: str) -> tuple[int, ...]:
    """校验调用方注入的开放整数键，禁止空键、bool 和整数子类。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _object_sequence(value, *, where: str) -> tuple[ObjectIdentity, ...]:
    """校验预测单元序列，确保目标不退化为字符串、码点或运行时 ref。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是 ObjectIdentity tuple")
    if any(not isinstance(item, ObjectIdentity) for item in value):
        raise TypeError(f"{where} 只能包含 ObjectIdentity")
    return value


def _pack_object_sequence(
        values: tuple[ObjectIdentity, ...]) -> tuple[int, ...]:
    """把完整对象身份按长度前缀打包，hash 不参与权威身份。"""
    packed: list[int] = [len(values)]
    for value in values:
        key = value.stable_key()
        packed.extend((len(key), *key))
    return tuple(packed)


@dataclass(frozen=True)
class PredictionProtocol:
    """注入预测 Hypothesis、三类 Evidence 理由和整数分数单位。"""

    hypothesis_kind_key: tuple[int, ...]
    support_reason_key: tuple[int, ...]
    refute_reason_key: tuple[int, ...]
    unknown_reason_key: tuple[int, ...]
    score_scale: int

    def __post_init__(self) -> None:
        keys = tuple(
            _integer_key(value, where=f"PredictionProtocol.{name}")
            for name, value in (
                ("hypothesis_kind_key", self.hypothesis_kind_key),
                ("support_reason_key", self.support_reason_key),
                ("refute_reason_key", self.refute_reason_key),
                ("unknown_reason_key", self.unknown_reason_key),
            )
        )
        if len(set(keys)) != len(keys):
            raise ValueError("预测 kind 与 Evidence 理由键必须互不相同")
        assert_int(self.score_scale, _where="PredictionProtocol.score_scale")
        if type(self.score_scale) is not int or self.score_scale <= 0:
            raise ValueError("score_scale 必须为严格正整数")

    def resolver_scorer_key(self) -> tuple[int, ...]:
        """由完整预测协议组成 scorer 身份，不生成领域常量或摘要。"""
        keys = (
            self.hypothesis_kind_key,
            self.support_reason_key,
            self.refute_reason_key,
            self.unknown_reason_key,
        )
        return tuple(
            value for key in keys for value in (len(key), *key))


@dataclass(frozen=True, order=True)
class PredictionTarget:
    """一个可预测目标的完整键及其一等对象单元。"""

    target_key: tuple[int, ...]
    units: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _integer_key(self.target_key, where="PredictionTarget.target_key")
        _object_sequence(self.units, where="PredictionTarget.units")
        if not self.units:
            raise ValueError("预测目标必须至少包含一个一等对象单元")

    def stable_key(self) -> tuple[int, ...]:
        """返回同时保存目标语义键和完整对象身份的整数键。"""
        units = _pack_object_sequence(self.units)
        return (
            _TARGET_VERSION,
            len(self.target_key),
            *self.target_key,
            len(units),
            *units,
        )


def target_from_units(
        units: tuple[ObjectIdentity, ...]) -> PredictionTarget:
    """以完整对象身份构造无摘要目标键，供 token、span 或成分共用。"""
    units = _object_sequence(units, where="target_from_units.units")
    if not units:
        raise ValueError("目标单元不能为空")
    packed = _pack_object_sequence(units)
    return PredictionTarget((len(packed), *packed), units)


@dataclass(frozen=True, order=True)
class PredictionContext:
    """一个不含目标的遮蔽上下文及有序条件回退键。"""

    objective_key: tuple[int, ...]
    visible_prefix: tuple[ObjectIdentity, ...]
    visible_suffix: tuple[ObjectIdentity, ...]
    masked_width: int
    condition_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        _integer_key(self.objective_key, where="PredictionContext.objective_key")
        _object_sequence(
            self.visible_prefix, where="PredictionContext.visible_prefix")
        _object_sequence(
            self.visible_suffix, where="PredictionContext.visible_suffix")
        assert_int(self.masked_width, _where="PredictionContext.masked_width")
        if type(self.masked_width) is not int or self.masked_width <= 0:
            raise ValueError("masked_width 必须为严格正整数")
        if not isinstance(self.condition_keys, tuple):
            raise TypeError("condition_keys 必须是整数 tuple 的 tuple")
        normalized = tuple(
            _integer_key(value, where="PredictionContext.condition_key")
            for value in self.condition_keys
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("condition_keys 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含目标的完整上下文身份。"""
        prefix = _pack_object_sequence(self.visible_prefix)
        suffix = _pack_object_sequence(self.visible_suffix)
        packed_conditions: list[int] = [len(self.condition_keys)]
        for key in self.condition_keys:
            packed_conditions.extend((len(key), *key))
        return (
            _CONTEXT_VERSION,
            len(self.objective_key),
            *self.objective_key,
            self.masked_width,
            len(prefix),
            *prefix,
            len(suffix),
            *suffix,
            len(packed_conditions),
            *packed_conditions,
        )

    def bucket_keys(self) -> tuple[tuple[int, ...], ...]:
        """按精确上下文到调用方回退条件返回计数桶身份。"""
        exact = self.stable_key()
        buckets: list[tuple[int, ...]] = [(1, len(exact), *exact)]
        for condition in self.condition_keys:
            buckets.append((
                2,
                len(self.objective_key),
                *self.objective_key,
                self.masked_width,
                len(condition),
                *condition,
            ))
        return tuple(buckets)


@dataclass(frozen=True, order=True)
class PredictionExample:
    """一条训练观察，目标与上下文分离且带可重放事件身份。"""

    context: PredictionContext
    target: PredictionTarget
    source: SourceRef
    event_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, PredictionContext):
            raise TypeError("PredictionExample.context 类型错误")
        if not isinstance(self.target, PredictionTarget):
            raise TypeError("PredictionExample.target 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("PredictionExample.source 必须是 SourceRef")
        _integer_key(self.event_key, where="PredictionExample.event_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、事件、上下文与目标的完整观察身份。"""
        source = self.source.stable_key()
        context = self.context.stable_key()
        target = self.target.stable_key()
        return (
            len(source),
            *source,
            len(self.event_key),
            *self.event_key,
            len(context),
            *context,
            len(target),
            *target,
        )


def build_masked_example(
        units: tuple[ObjectIdentity, ...], *,
        target_start: int,
        target_width: int,
        objective_key: tuple[int, ...],
        source: SourceRef,
        reveal_suffix: bool,
        condition_keys: tuple[tuple[int, ...], ...] = (),
        event_key: tuple[int, ...] | None = None,
        ) -> PredictionExample:
    """从完整单元序列构造遮蔽样本，返回对象中不保留目标位置内容。"""
    units = _object_sequence(units, where="build_masked_example.units")
    assert_int(
        target_start,
        target_width,
        _where="build_masked_example.target",
    )
    if (type(target_start) is not int or type(target_width) is not int
            or target_start < 0 or target_width <= 0
            or target_start + target_width > len(units)):
        raise ValueError("预测目标区间非法")
    if type(reveal_suffix) is not bool:
        raise TypeError("reveal_suffix 必须是 bool")
    target_units = units[target_start:target_start + target_width]
    context = PredictionContext(
        objective_key,
        units[:target_start],
        units[target_start + target_width:] if reveal_suffix else (),
        target_width,
        condition_keys,
    )
    derived_event_key = (
        target_start,
        target_width,
        1 if reveal_suffix else 0,
    )
    return PredictionExample(
        context,
        target_from_units(target_units),
        source,
        derived_event_key if event_key is None else event_key,
    )


@dataclass(frozen=True, order=True)
class PredictionScore:
    """一个候选在首个有证据条件层上的纯整数分数明细。"""

    condition_level: int
    condition_key: tuple[int, ...]
    support_count: int
    total_count: int
    scaled_score: int


class ConditionalPredictionModel:
    """按完整观察累计条件计数，并支持排除指定来源的确定性查询。"""

    def __init__(self, score_scale: int) -> None:
        assert_int(score_scale, _where="ConditionalPredictionModel.score_scale")
        if type(score_scale) is not int or score_scale <= 0:
            raise ValueError("score_scale 必须为严格正整数")
        self.score_scale = score_scale
        self._events: dict[tuple[int, ...], PredictionExample] = {}
        self._event_slots: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._targets: dict[tuple[int, ...], PredictionTarget] = {}
        self._buckets: dict[
            tuple[int, ...], dict[PredictionTarget, set[tuple[int, ...]]]
        ] = {}

    def observe(self, example: PredictionExample) -> PredictionExample:
        """幂等追加训练观察，同一来源事件槽绑定不同内容时 fail closed。"""
        if not isinstance(example, PredictionExample):
            raise TypeError("observe 需要 PredictionExample")
        source_key = example.source.stable_key()
        slot = (
            len(source_key),
            *source_key,
            len(example.context.objective_key),
            *example.context.objective_key,
            len(example.event_key),
            *example.event_key,
        )
        event_identity = example.stable_key()
        existing_identity = self._event_slots.get(slot)
        if existing_identity is not None:
            existing = self._events[existing_identity]
            if existing != example:
                raise ValueError("同一来源预测事件槽已绑定不同观察")
            return existing
        target_key = example.target.target_key
        registered_target = self._targets.get(target_key)
        if registered_target is not None and registered_target != example.target:
            raise ValueError("同一预测目标键已绑定不同完整对象身份")
        self._targets[target_key] = example.target
        self._events[event_identity] = example
        self._event_slots[slot] = event_identity
        for bucket_key in example.context.bucket_keys():
            targets = self._buckets.setdefault(bucket_key, {})
            targets.setdefault(example.target, set()).add(event_identity)
        return example

    def score(
            self, context: PredictionContext, target: PredictionTarget, *,
            excluded_sources: tuple[SourceRef, ...] = (),
            ) -> PredictionScore:
        """在首个有可用观察的条件层计算整数条件比例，不混合任意权重。"""
        if not isinstance(context, PredictionContext):
            raise TypeError("score.context 类型错误")
        if not isinstance(target, PredictionTarget):
            raise TypeError("score.target 类型错误")
        excluded = self._excluded_source_keys(excluded_sources)
        for level, bucket_key in enumerate(context.bucket_keys()):
            counts = self._bucket_counts(bucket_key, excluded)
            total = sum(counts.values())
            if total <= 0:
                continue
            support = counts.get(target, 0)
            return PredictionScore(
                level,
                bucket_key,
                support,
                total,
                (support * self.score_scale) // total,
            )
        return PredictionScore(0, (), 0, 0, 0)

    def known_targets(
            self, context: PredictionContext, *, candidate_limit: int,
            excluded_sources: tuple[SourceRef, ...] = (),
            ) -> tuple[PredictionTarget, ...]:
        """从首个有证据条件层返回按计数和完整身份稳定排序的目标候选。"""
        assert_int(candidate_limit, _where="known_targets.candidate_limit")
        if type(candidate_limit) is not int or candidate_limit <= 0:
            raise ValueError("candidate_limit 必须为严格正整数")
        excluded = self._excluded_source_keys(excluded_sources)
        for bucket_key in context.bucket_keys():
            counts = self._bucket_counts(bucket_key, excluded)
            if not counts:
                continue
            ranked = sorted(
                counts,
                key=lambda item: (-counts[item], item.stable_key()),
            )
            return tuple(ranked[:candidate_limit])
        return ()

    def observation_count(self) -> int:
        """返回去重后的完整训练观察数。"""
        return len(self._events)

    def clone(self) -> "ConditionalPredictionModel":
        """复制完整观察和条件桶，供 V-06 评测沙箱独立试算。"""
        cloned = ConditionalPredictionModel(self.score_scale)
        for event_identity in sorted(self._events):
            cloned.observe(self._events[event_identity])
        return cloned

    def state_key(self) -> tuple:
        """返回可比较的完整模型状态，不以摘要替代观察身份。"""
        return (
            self.score_scale,
            tuple(
                self._events[key].stable_key()
                for key in sorted(self._events)
            ),
        )

    def _bucket_counts(
            self, bucket_key: tuple[int, ...],
            excluded: frozenset[tuple[int, ...]],
            ) -> dict[PredictionTarget, int]:
        """按事件来源过滤一个计数桶，保留 occurrence 级重复而去除重放。"""
        counts: dict[PredictionTarget, int] = {}
        for target, event_ids in self._buckets.get(bucket_key, {}).items():
            count = sum(
                self._events[event_id].source.stable_key() not in excluded
                for event_id in event_ids
            )
            if count:
                counts[target] = count
        return counts

    @staticmethod
    def _excluded_source_keys(
            sources: tuple[SourceRef, ...]) -> frozenset[tuple[int, ...]]:
        """校验并展开查询期排除来源。"""
        if not isinstance(sources, tuple):
            raise TypeError("excluded_sources 必须是 SourceRef tuple")
        if any(not isinstance(source, SourceRef) for source in sources):
            raise TypeError("excluded_sources 只能包含 SourceRef")
        return frozenset(source.stable_key() for source in sources)


@dataclass(frozen=True)
class PredictionCandidateResult:
    """一个预测候选的 Hypothesis、目标、条件分数和当前证据状态。"""

    hypothesis: HypothesisKey
    target: PredictionTarget
    score: PredictionScore
    snapshot: HypothesisSnapshot


@dataclass(frozen=True)
class PredictionResult:
    """一次不读取真值的候选预测结果；并列最优全部保留。"""

    context: PredictionContext
    observation: SourceRef
    candidates: tuple[PredictionCandidateResult, ...]
    selected_hypotheses: tuple[HypothesisKey, ...]
    selection_decision_id: int = 0
    evaluation_decision_id: int = 0


class _PredictionResolverScorer:
    """把条件计数比例适配为不受整数缩放截断影响的候选对关系。"""

    def __init__(
            self, protocol: PredictionProtocol,
            candidates: tuple[PredictionCandidateResult, ...],
            ) -> None:
        """绑定本次预测候选，scorer 身份完全来自注入协议。"""
        self.scorer_key = protocol.resolver_scorer_key()
        self._scores = {
            item.hypothesis: item.score for item in candidates
        }
        if len(self._scores) != len(candidates):
            raise ValueError("预测 resolver scorer 不得重复绑定 Hypothesis")

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]:
        """逐对交叉乘法比较比例；无分母时保持不可比。"""
        if any(item not in self._scores for item in hypotheses):
            raise ValueError("预测 scorer 缺少 resolver eligible 候选")
        preferences: list[ResolverPreference] = []
        for left_index in range(len(hypotheses)):
            for right_index in range(left_index + 1, len(hypotheses)):
                left = hypotheses[left_index]
                right = hypotheses[right_index]
                left_score = self._scores[left]
                right_score = self._scores[right]
                if (left_score.total_count <= 0
                        or right_score.total_count <= 0):
                    preference = PREFERENCE_INCOMPARABLE
                else:
                    left_product = (
                        left_score.support_count * right_score.total_count)
                    right_product = (
                        right_score.support_count * left_score.total_count)
                    if left_product > right_product:
                        preference = PREFERENCE_LEFT_BETTER
                    elif right_product > left_product:
                        preference = PREFERENCE_RIGHT_BETTER
                    else:
                        preference = PREFERENCE_EQUIVALENT
                preferences.append(ResolverPreference(
                    self.scorer_key,
                    left,
                    right,
                    preference,
                    (
                        left_score.condition_level,
                        len(left_score.condition_key),
                        *left_score.condition_key,
                        left_score.support_count,
                        left_score.total_count,
                        left_score.scaled_score,
                        right_score.condition_level,
                        len(right_score.condition_key),
                        *right_score.condition_key,
                        right_score.support_count,
                        right_score.total_count,
                        right_score.scaled_score,
                    ),
                ))
        return tuple(preferences)


class PredictionEngine:
    """把条件模型的候选结果映射为 H-00 Hypothesis/Evidence。"""

    def __init__(
            self, protocol: PredictionProtocol, *,
            model: ConditionalPredictionModel | None = None,
            ledger: HypothesisLedger | None = None,
            resolver: HypothesisResolver | None = None,
            ) -> None:
        if not isinstance(protocol, PredictionProtocol):
            raise TypeError("protocol 必须是 PredictionProtocol")
        self.protocol = protocol
        self.model = model or ConditionalPredictionModel(protocol.score_scale)
        if self.model.score_scale != protocol.score_scale:
            raise ValueError("预测模型与协议 score_scale 不一致")
        self.ledger = ledger or HypothesisLedger()
        self.resolver = resolver or HypothesisResolver(self.ledger)
        if self.resolver.ledger is not self.ledger:
            raise ValueError("预测 resolver 必须绑定同一 H-00 ledger")

    def observe(self, example: PredictionExample) -> PredictionExample:
        """把已揭示目标追加到条件模型，不产生自评 Evidence。"""
        return self.model.observe(example)

    def predict(
            self, context: PredictionContext, *, observation: SourceRef,
            scope: ScopeIdentity, candidate_limit: int,
            excluded_sources: tuple[SourceRef, ...] | None = None,
            ) -> PredictionResult:
        """只根据既有模型形成候选；expected target 不属于本函数输入。"""
        if not isinstance(observation, SourceRef):
            raise TypeError("observation 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        excluded = (
            (observation,) if excluded_sources is None else excluded_sources)
        targets = self.model.known_targets(
            context,
            candidate_limit=candidate_limit,
            excluded_sources=excluded,
        )
        context_key = context.stable_key()
        competition_key = (
            _HYPOTHESIS_VERSION,
            len(context_key),
            *context_key,
        )
        candidates: list[PredictionCandidateResult] = []
        for target in targets:
            target_key = target.stable_key()
            hypothesis = self.ledger.register(HypothesisKey(
                self.protocol.hypothesis_kind_key,
                (
                    _HYPOTHESIS_VERSION,
                    len(context_key),
                    *context_key,
                    len(target_key),
                    *target_key,
                ),
                competition_key,
                scope,
                observation,
            ))
            candidates.append(PredictionCandidateResult(
                hypothesis,
                target,
                self.model.score(
                    context,
                    target,
                    excluded_sources=excluded,
                ),
                self.ledger.snapshot(hypothesis),
            ))
        ranked = tuple(sorted(
            candidates,
            key=lambda item: (
                -item.score.scaled_score,
                -item.score.support_count,
                item.target.stable_key(),
            ),
        ))
        if not ranked:
            return PredictionResult(context, observation, (), ())
        decision = self.resolver.resolve(
            ranked[0].hypothesis,
            timestamp_seq=0,
            scorers=(_PredictionResolverScorer(self.protocol, ranked),),
        )
        return PredictionResult(
            context,
            observation,
            ranked,
            decision.adopted_hypotheses,
            decision.decision_id,
        )

    def evaluate(
            self, result: PredictionResult, *,
            expected: PredictionTarget | None,
            evidence_source: SourceRef,
            timestamp_seq: int,
            ) -> PredictionResult:
        """在预测完成后揭示目标，并只更新本次相关候选的 H-00 Evidence。"""
        if not isinstance(result, PredictionResult):
            raise TypeError("result 必须是 PredictionResult")
        if expected is not None and not isinstance(expected, PredictionTarget):
            raise TypeError("expected 必须是 PredictionTarget 或 None")
        if not isinstance(evidence_source, SourceRef):
            raise TypeError("evidence_source 必须是 SourceRef")
        assert_int(timestamp_seq, _where="PredictionEngine.timestamp_seq")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        evaluated: list[PredictionCandidateResult] = []
        expected_key = () if expected is None else expected.stable_key()
        for candidate in result.candidates:
            if expected is None:
                stance = EVIDENCE_UNKNOWN
                reason_key = self.protocol.unknown_reason_key
            elif candidate.target == expected:
                stance = EVIDENCE_SUPPORT
                reason_key = self.protocol.support_reason_key
            else:
                stance = EVIDENCE_REFUTE
                reason_key = self.protocol.refute_reason_key
            score = candidate.score
            evidence_id = _EVIDENCE_HASHER.h63((
                candidate.hypothesis.stable_key(),
                stance,
                reason_key,
                evidence_source.stable_key(),
                timestamp_seq,
                expected_key,
                score.condition_level,
                score.support_count,
                score.total_count,
                score.scaled_score,
            )) or 1
            self.ledger.append_evidence(EvidenceRecord(
                evidence_id,
                candidate.hypothesis,
                stance,
                reason_key,
                evidence_source,
                timestamp_seq,
                payload=(
                    score.condition_level,
                    score.support_count,
                    score.total_count,
                    score.scaled_score,
                    len(expected_key),
                    *expected_key,
                ),
            ))
            evaluated.append(PredictionCandidateResult(
                candidate.hypothesis,
                candidate.target,
                candidate.score,
                self.ledger.snapshot(candidate.hypothesis),
            ))
        decision_id = 0
        if evaluated:
            decision = self.resolver.resolve(
                evaluated[0].hypothesis,
                timestamp_seq=timestamp_seq,
            )
            decision_id = decision.decision_id
            evaluated = [
                PredictionCandidateResult(
                    candidate.hypothesis,
                    candidate.target,
                    candidate.score,
                    self.ledger.snapshot(candidate.hypothesis),
                )
                for candidate in evaluated
            ]
        return PredictionResult(
            result.context,
            result.observation,
            tuple(evaluated),
            result.selected_hypotheses,
            result.selection_decision_id,
            decision_id,
        )

    def clone(self) -> "PredictionEngine":
        """复制条件模型和 Evidence ledger，供评测沙箱隔离运行。"""
        ledger = self.ledger.clone()
        return PredictionEngine(
            self.protocol,
            model=self.model.clone(),
            ledger=ledger,
            resolver=self.resolver.clone(ledger=ledger),
        )

    def state_key(self) -> tuple:
        """返回模型、H-00 事件和 H-04 决策链的完整可比较状态。"""
        return (
            self.model.state_key(),
            self.ledger.state_key(),
            self.resolver.state_key(),
        )

    @staticmethod
    def _same_ratio(left: PredictionScore, right: PredictionScore) -> bool:
        """用交叉乘法比较条件比例，避免 scaled_score 截断制造伪并列。"""
        if left.total_count <= 0 or right.total_count <= 0:
            return left.total_count == right.total_count == 0
        return (
            left.support_count * right.total_count
            == right.support_count * left.total_count
        )


__all__ = [
    "ConditionalPredictionModel",
    "PredictionCandidateResult",
    "PredictionContext",
    "PredictionEngine",
    "PredictionExample",
    "PredictionProtocol",
    "PredictionResult",
    "PredictionScore",
    "PredictionTarget",
    "build_masked_example",
    "target_from_units",
]
