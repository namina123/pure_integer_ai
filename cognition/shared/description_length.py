"""H-03 纯整数前缀描述长度、递归复用和候选成本分解。

本模块不解释语言、结构类别或现实真值。调用方提供完整一等对象观察、H-00 候选和
可递归 fragment 定义；engine 先验证每条编码能无损展开回原观察，再以统一前缀码 bit
长度计算模型、数据、边界和当前 active 反例成本。核心没有分项权重或通过阈值。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


TERM_LITERAL = 1
TERM_FRAGMENT = 2
_TERM_KINDS = frozenset({TERM_LITERAL, TERM_FRAGMENT})
_PROBLEM_VERSION = 1


def _integer_key(value, *, where: str) -> tuple[int, ...]:
    """校验来源事件等调用方开放键为非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _object_sequence(value, *, where: str) -> tuple[ObjectIdentity, ...]:
    """校验待编码内容只由完整一等对象身份组成。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是 ObjectIdentity tuple")
    if any(not isinstance(item, ObjectIdentity) for item in value):
        raise TypeError(f"{where} 只能包含 ObjectIdentity")
    return value


def nonnegative_prefix_bit_cost(value: int) -> int:
    """返回非负整数 ``value`` 经 ``value+1`` Elias gamma 编码的 bit 长度。"""
    assert_int(value, _where="nonnegative_prefix_bit_cost")
    if type(value) is not int or value < 0:
        raise ValueError("前缀码输入必须是非负严格整数")
    return 2 * (value + 1).bit_length() - 1


def signed_prefix_bit_cost(value: int) -> int:
    """先以 zigzag 映射有符号整数，再返回统一 gamma 前缀码 bit 长度。"""
    assert_int(value, _where="signed_prefix_bit_cost")
    if type(value) is not int:
        raise TypeError("有符号前缀码输入必须是严格整数")
    mapped = 2 * value if value >= 0 else -2 * value - 1
    return nonnegative_prefix_bit_cost(mapped)


def integer_tuple_bit_cost(values: tuple[int, ...]) -> int:
    """返回带元素数量前缀的严格整数 tuple 可恢复编码长度。"""
    if not isinstance(values, tuple):
        raise TypeError("整数序列成本只接受 tuple")
    assert_int(*values, _where="integer_tuple_bit_cost")
    if any(type(value) is not int for value in values):
        raise TypeError("整数序列成本只接受严格整数")
    return nonnegative_prefix_bit_cost(len(values)) + sum(
        signed_prefix_bit_cost(value) for value in values)


def _identity_bit_cost(identity: ObjectIdentity) -> int:
    """以完整稳定键而非 hash 计算一等对象身份成本。"""
    return integer_tuple_bit_cost(identity.stable_key())


@dataclass(frozen=True)
class DescriptionTerm:
    """一个 literal 对象或对候选模型内一等 fragment 的引用。"""

    term_kind: int
    identity: ObjectIdentity

    def __post_init__(self) -> None:
        """核验 codec tag 和对象身份，不允许 surface 或运行时 ref 混入。"""
        assert_int(self.term_kind, _where="DescriptionTerm.term_kind")
        if type(self.term_kind) is not int or self.term_kind not in _TERM_KINDS:
            raise ValueError("DescriptionTerm.term_kind 未注册")
        if not isinstance(self.identity, ObjectIdentity):
            raise TypeError("DescriptionTerm.identity 必须是 ObjectIdentity")

    @classmethod
    def literal(cls, identity: ObjectIdentity) -> "DescriptionTerm":
        """构造直接编码完整对象身份的 literal term。"""
        return cls(TERM_LITERAL, identity)

    @classmethod
    def fragment(cls, identity: ObjectIdentity) -> "DescriptionTerm":
        """构造指向候选模型内一等 fragment 的引用 term。"""
        return cls(TERM_FRAGMENT, identity)

    def stable_key(self) -> tuple[int, ...]:
        """返回保留 term 类型和完整对象身份的整数键。"""
        identity = self.identity.stable_key()
        return self.term_kind, len(identity), *identity


@dataclass(frozen=True)
class DescriptionObservation:
    """一条来源化、作用域化且保留重复与空序列的对象观察。"""

    source: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    units: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验来源/scope 一致和观察对象完整性。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("DescriptionObservation.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("DescriptionObservation.scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise ValueError("描述长度观察 scope 必须指向同一 SourceRef")
        _integer_key(
            self.event_key, where="DescriptionObservation.event_key")
        _object_sequence(
            self.units, where="DescriptionObservation.units")

    def slot_key(self) -> tuple[int, ...]:
        """返回来源、scope 和事件组成的完整观察槽身份。"""
        source = self.source.stable_key()
        scope = self.scope.stable_key()
        return (
            len(source),
            *source,
            len(scope),
            *scope,
            len(self.event_key),
            *self.event_key,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回包含观察槽和全部对象身份的完整问题项键。"""
        slot = self.slot_key()
        values: list[int] = [len(slot), *slot, len(self.units)]
        for unit in self.units:
            identity = unit.stable_key()
            values.extend((len(identity), *identity))
        return tuple(values)


@dataclass(frozen=True)
class DescriptionLengthProblem:
    """所有候选必须完整编码的同一观察全集。"""

    observations: tuple[DescriptionObservation, ...]

    def __post_init__(self) -> None:
        """拒绝空问题、重复事件槽和非来源化观察。"""
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("描述长度问题必须包含至少一条观察")
        if any(not isinstance(item, DescriptionObservation)
               for item in self.observations):
            raise TypeError("problem.observations 只能包含 DescriptionObservation")
        slots = tuple(item.slot_key() for item in self.observations)
        if len(set(slots)) != len(slots):
            raise ValueError("描述长度问题不得重复来源事件槽")

    def ordered_observations(self) -> tuple[DescriptionObservation, ...]:
        """按完整观察身份排序，消除调用方输入顺序对成本的影响。"""
        return tuple(sorted(
            self.observations, key=lambda item: item.stable_key()))

    def stable_key(self) -> tuple[int, ...]:
        """返回不丢来源、事件和对象内容的完整问题身份。"""
        values: list[int] = [_PROBLEM_VERSION, len(self.observations)]
        for observation in self.ordered_observations():
            key = observation.stable_key()
            values.extend((len(key), *key))
        return tuple(values)


@dataclass(frozen=True)
class DescriptionFragment:
    """候选模型中的一等 fragment 及其可递归、非空展开。"""

    fragment: ObjectIdentity
    expansion: tuple[DescriptionTerm, ...]

    def __post_init__(self) -> None:
        """核验 fragment 身份和展开 term；DAG/引用闭包由 engine 联合检查。"""
        if not isinstance(self.fragment, ObjectIdentity):
            raise TypeError("DescriptionFragment.fragment 必须是 ObjectIdentity")
        if not isinstance(self.expansion, tuple) or not self.expansion:
            raise ValueError("DescriptionFragment.expansion 必须是非空 term tuple")
        if any(not isinstance(item, DescriptionTerm) for item in self.expansion):
            raise TypeError("fragment.expansion 只能包含 DescriptionTerm")


@dataclass(frozen=True)
class DescriptionModel:
    """一个 H-00 候选及其可复用一等 fragment 定义。"""

    hypothesis: HypothesisKey
    fragments: tuple[DescriptionFragment, ...] = ()

    def __post_init__(self) -> None:
        """核验候选和 fragment 身份唯一性，不预设领域对象类型。"""
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("DescriptionModel.hypothesis 必须是 HypothesisKey")
        if not isinstance(self.fragments, tuple):
            raise TypeError("DescriptionModel.fragments 必须是 tuple")
        if any(not isinstance(item, DescriptionFragment) for item in self.fragments):
            raise TypeError("model.fragments 只能包含 DescriptionFragment")
        identities = tuple(item.fragment for item in self.fragments)
        if len(set(identities)) != len(identities):
            raise ValueError("候选模型不得重复定义同一 fragment")

    def ordered_fragments(self) -> tuple[DescriptionFragment, ...]:
        """按完整 fragment 身份返回确定性局部 ordinal 顺序。"""
        return tuple(sorted(
            self.fragments, key=lambda item: item.fragment.stable_key()))


@dataclass(frozen=True)
class DescriptionEncoding:
    """一个候选对某个完整观察槽的 literal/fragment 编码。"""

    source: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    terms: tuple[DescriptionTerm, ...]

    def __post_init__(self) -> None:
        """核验编码槽和 term 序列；空编码只可能匹配空观察。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("DescriptionEncoding.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("DescriptionEncoding.scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise ValueError("描述长度编码 scope 必须指向同一 SourceRef")
        _integer_key(self.event_key, where="DescriptionEncoding.event_key")
        if not isinstance(self.terms, tuple):
            raise TypeError("DescriptionEncoding.terms 必须是 tuple")
        if any(not isinstance(item, DescriptionTerm) for item in self.terms):
            raise TypeError("encoding.terms 只能包含 DescriptionTerm")

    def slot_key(self) -> tuple[int, ...]:
        """返回与 DescriptionObservation 相同格式的完整事件槽键。"""
        source = self.source.stable_key()
        scope = self.scope.stable_key()
        return (
            len(source),
            *source,
            len(scope),
            *scope,
            len(self.event_key),
            *self.event_key,
        )


@dataclass(frozen=True)
class DescriptionCandidate:
    """一个模型及其对问题全集的候选编码。"""

    model: DescriptionModel
    encodings: tuple[DescriptionEncoding, ...]

    def __post_init__(self) -> None:
        """核验模型和编码容器，完整覆盖关系由 engine 对问题检查。"""
        if not isinstance(self.model, DescriptionModel):
            raise TypeError("DescriptionCandidate.model 必须是 DescriptionModel")
        if not isinstance(self.encodings, tuple):
            raise TypeError("DescriptionCandidate.encodings 必须是 tuple")
        if any(not isinstance(item, DescriptionEncoding) for item in self.encodings):
            raise TypeError("candidate.encodings 只能包含 DescriptionEncoding")


@dataclass(frozen=True)
class DescriptionLengthBreakdown:
    """候选的可加和 bit 成本、复用诊断和算术消融结果。"""

    hypothesis: HypothesisKey
    problem_key: tuple[int, ...]
    model_cost: int
    encoded_data_cost: int
    boundary_cost: int
    exception_cost: int
    total_cost: int
    literal_baseline_cost: int
    recursive_reuse_gain: int
    exception_count: int
    fragment_count: int
    fragment_reference_count: int
    recursive_fragment_reference_count: int
    without_model_cost: int
    without_exception_cost: int
    without_boundary_cost: int
    without_reuse_cost: int


def _term_bit_cost(
        term: DescriptionTerm,
        fragment_ordinals: dict[ObjectIdentity, int],
        ) -> int:
    """计算 term tag 加 literal 完整身份或模型内局部 ordinal 的成本。"""
    cost = nonnegative_prefix_bit_cost(term.term_kind)
    if term.term_kind == TERM_LITERAL:
        return cost + _identity_bit_cost(term.identity)
    ordinal = fragment_ordinals.get(term.identity)
    if ordinal is None:
        raise ValueError("fragment term 引用了候选模型外的对象")
    return cost + nonnegative_prefix_bit_cost(ordinal)


def _validate_fragment_dag(
        fragments: dict[ObjectIdentity, DescriptionFragment]) -> None:
    """核验 fragment 引用闭包并以确定性拓扑消除法拒绝任意环。"""
    dependencies: dict[ObjectIdentity, set[ObjectIdentity]] = {}
    dependents: dict[ObjectIdentity, set[ObjectIdentity]] = {
        identity: set() for identity in fragments
    }
    for identity, fragment in fragments.items():
        refs = {
            term.identity for term in fragment.expansion
            if term.term_kind == TERM_FRAGMENT
        }
        unknown = refs - fragments.keys()
        if unknown:
            raise ValueError("fragment 定义引用了候选模型外的对象")
        dependencies[identity] = refs
        for dependency in refs:
            dependents[dependency].add(identity)

    ready = [
        (identity.stable_key(), identity)
        for identity, refs in dependencies.items()
        if not refs
    ]
    heapq.heapify(ready)
    resolved = 0
    while ready:
        _stable_key, identity = heapq.heappop(ready)
        resolved += 1
        for dependent in sorted(
                dependents[identity], key=lambda item: item.stable_key()):
            dependencies[dependent].remove(identity)
            if not dependencies[dependent]:
                heapq.heappush(
                    ready, (dependent.stable_key(), dependent))
    if resolved != len(fragments):
        raise ValueError("fragment 递归引用必须构成可终止 DAG")


def _validate_exact_expansion(
        terms: tuple[DescriptionTerm, ...],
        expected: tuple[ObjectIdentity, ...],
        fragments: dict[ObjectIdentity, DescriptionFragment],
        ) -> None:
    """迭代展开候选编码并与原观察逐点比对，超长时立即失败。"""
    stack = list(reversed(terms))
    position = 0
    while stack:
        term = stack.pop()
        if term.term_kind == TERM_FRAGMENT:
            fragment = fragments.get(term.identity)
            if fragment is None:
                raise ValueError("encoding 引用了候选模型外的 fragment")
            stack.extend(reversed(fragment.expansion))
            continue
        if position >= len(expected) or term.identity != expected[position]:
            raise ValueError("候选编码不能无损展开回原观察")
        position += 1
    if position != len(expected):
        raise ValueError("候选编码不能完整覆盖原观察")


def _exception_semantic_key(evidence: EvidenceRecord) -> tuple[int, ...]:
    """返回排除 hash/时钟、保留 reason/source/payload 的完整例外语义键。"""
    reason = evidence.reason_key
    source = evidence.source.stable_key()
    payload = evidence.payload
    return (
        len(reason),
        *reason,
        len(source),
        *source,
        len(payload),
        *payload,
    )


class DescriptionLengthEngine:
    """验证无损候选，并以统一整数前缀码对同一问题排序。"""

    def __init__(self, ledger: HypothesisLedger) -> None:
        """绑定真正拥有候选和 active Evidence 的 H-00 ledger。"""
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger 必须是 HypothesisLedger")
        self.ledger = ledger

    def score(
            self, problem: DescriptionLengthProblem,
            candidate: DescriptionCandidate,
            ) -> DescriptionLengthBreakdown:
        """核验候选完整覆盖问题后，返回无权重的可审计 bit 成本分解。"""
        if not isinstance(problem, DescriptionLengthProblem):
            raise TypeError("problem 必须是 DescriptionLengthProblem")
        if not isinstance(candidate, DescriptionCandidate):
            raise TypeError("candidate 必须是 DescriptionCandidate")
        model = candidate.model
        snapshot = self.ledger.snapshot(model.hypothesis)
        ordered_fragments = model.ordered_fragments()
        fragments = {item.fragment: item for item in ordered_fragments}
        _validate_fragment_dag(fragments)
        fragment_ordinals = {
            fragment.fragment: ordinal
            for ordinal, fragment in enumerate(ordered_fragments)
        }

        encodings: dict[tuple[int, ...], DescriptionEncoding] = {}
        for encoding in candidate.encodings:
            slot = encoding.slot_key()
            if slot in encodings:
                raise ValueError("候选不得重复编码同一来源事件槽")
            encodings[slot] = encoding
        observations = problem.ordered_observations()
        observation_slots = {item.slot_key() for item in observations}
        if set(encodings) != observation_slots:
            raise ValueError("候选必须且只能编码问题中的全部观察槽")
        for observation in observations:
            _validate_exact_expansion(
                encodings[observation.slot_key()].terms,
                observation.units,
                fragments,
            )

        model_cost = _identity_bit_cost(model.hypothesis.object_identity())
        boundary_cost = nonnegative_prefix_bit_cost(len(ordered_fragments))
        recursive_references = 0
        for fragment in ordered_fragments:
            model_cost += _identity_bit_cost(fragment.fragment)
            boundary_cost += nonnegative_prefix_bit_cost(
                len(fragment.expansion))
            for term in fragment.expansion:
                model_cost += _term_bit_cost(term, fragment_ordinals)
                if term.term_kind == TERM_FRAGMENT:
                    recursive_references += 1

        encoded_data_cost = 0
        data_references = 0
        boundary_cost += nonnegative_prefix_bit_cost(len(observations))
        for observation in observations:
            encoding = encodings[observation.slot_key()]
            boundary_cost += nonnegative_prefix_bit_cost(len(encoding.terms))
            for term in encoding.terms:
                encoded_data_cost += _term_bit_cost(term, fragment_ordinals)
                if term.term_kind == TERM_FRAGMENT:
                    data_references += 1

        history = {
            evidence.evidence_id: evidence
            for evidence in self.ledger.evidence_history(model.hypothesis)
        }
        exception_keys = tuple(sorted({
            _exception_semantic_key(history[evidence_id])
            for evidence_id in snapshot.refute_evidence_ids
        }))
        exception_cost = 0
        if exception_keys:
            exception_cost += nonnegative_prefix_bit_cost(len(exception_keys))
            exception_cost += sum(
                integer_tuple_bit_cost(key) for key in exception_keys)

        literal_model_cost = _identity_bit_cost(
            model.hypothesis.object_identity())
        literal_boundary_cost = (
            nonnegative_prefix_bit_cost(0)
            + nonnegative_prefix_bit_cost(len(observations))
        )
        literal_data_cost = 0
        for observation in observations:
            literal_boundary_cost += nonnegative_prefix_bit_cost(
                len(observation.units))
            literal_data_cost += sum(
                nonnegative_prefix_bit_cost(TERM_LITERAL)
                + _identity_bit_cost(unit)
                for unit in observation.units
            )
        literal_baseline = (
            literal_model_cost + literal_boundary_cost + literal_data_cost)

        total = (
            model_cost + encoded_data_cost + boundary_cost + exception_cost)
        reuse_gain = literal_baseline - (
            model_cost + encoded_data_cost + boundary_cost)
        return DescriptionLengthBreakdown(
            hypothesis=model.hypothesis,
            problem_key=problem.stable_key(),
            model_cost=model_cost,
            encoded_data_cost=encoded_data_cost,
            boundary_cost=boundary_cost,
            exception_cost=exception_cost,
            total_cost=total,
            literal_baseline_cost=literal_baseline,
            recursive_reuse_gain=reuse_gain,
            exception_count=len(exception_keys),
            fragment_count=len(ordered_fragments),
            fragment_reference_count=(
                recursive_references + data_references),
            recursive_fragment_reference_count=recursive_references,
            without_model_cost=total - model_cost,
            without_exception_cost=total - exception_cost,
            without_boundary_cost=total - boundary_cost,
            without_reuse_cost=literal_baseline + exception_cost,
        )

    def rank(
            self, problem: DescriptionLengthProblem,
            candidates: tuple[DescriptionCandidate, ...],
            ) -> tuple[DescriptionLengthBreakdown, ...]:
        """按 total 升序、再按完整 Hypothesis 稳定键确定性排列候选。"""
        if not isinstance(candidates, tuple) or not candidates:
            raise ValueError("rank 至少需要一个 DescriptionCandidate")
        if any(not isinstance(item, DescriptionCandidate) for item in candidates):
            raise TypeError("rank.candidates 只能包含 DescriptionCandidate")
        hypotheses = tuple(item.model.hypothesis for item in candidates)
        if len(set(hypotheses)) != len(hypotheses):
            raise ValueError("rank 不得重复评分同一 Hypothesis")
        scored = tuple(self.score(problem, candidate) for candidate in candidates)
        return tuple(sorted(
            scored,
            key=lambda item: (
                item.total_cost,
                item.hypothesis.stable_key(),
            ),
        ))


__all__ = [
    "DescriptionCandidate",
    "DescriptionEncoding",
    "DescriptionFragment",
    "DescriptionLengthBreakdown",
    "DescriptionLengthEngine",
    "DescriptionLengthProblem",
    "DescriptionModel",
    "DescriptionObservation",
    "DescriptionTerm",
    "TERM_FRAGMENT",
    "TERM_LITERAL",
    "integer_tuple_bit_cost",
    "nonnegative_prefix_bit_cost",
    "signed_prefix_bit_cost",
]
