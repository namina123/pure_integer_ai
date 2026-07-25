"""来源化句界候选的 Hypothesis/Evidence 竞争协议。

本模块只接收上游显式提供的码点锚点和证据，不读取字符内容、Unicode 属性或语言
字面量。没有唯一受支持候选时返回未决结果，消费者必须保留完整输入段。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
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
    HypothesisResolver,
    ReplacementDirective,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_EVIDENCE_HASHER = Hasher("boundary_hypothesis.evidence.v1")


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或图协议注入的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True, order=True)
class BoundaryCandidate:
    """一个来源文档中的有序内部码点边界集合。"""

    anchors: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.anchors, tuple):
            raise TypeError("BoundaryCandidate.anchors 必须是整数 tuple")
        assert_int(*self.anchors, _where="BoundaryCandidate.anchors")
        if any(type(anchor) is not int for anchor in self.anchors):
            raise ValueError("BoundaryCandidate.anchors 必须使用严格整数")
        if any(anchor <= 0 for anchor in self.anchors):
            raise ValueError("边界锚点必须为正码点位置")
        if tuple(sorted(set(self.anchors))) != self.anchors:
            raise ValueError("边界锚点必须严格递增且不重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含字符内容和证据状态的候选身份键。"""
        return len(self.anchors), *self.anchors


@dataclass(frozen=True, order=True)
class BoundaryEvidenceSpec:
    """由外部来源、课程或已学结构提供的一条边界证据。"""

    candidate: BoundaryCandidate
    stance: int
    reason_key: tuple[int, ...]
    timestamp_seq: int = 0
    source: SourceRef | None = None
    payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, BoundaryCandidate):
            raise TypeError("BoundaryEvidenceSpec.candidate 类型非法")
        assert_int(self.stance, self.timestamp_seq,
                   _where="BoundaryEvidenceSpec")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("BoundaryEvidenceSpec.stance 未注册")
        _strict_key(self.reason_key, where="BoundaryEvidenceSpec.reason_key")
        if self.timestamp_seq < 0:
            raise ValueError("边界证据逻辑序不得为负")
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("BoundaryEvidenceSpec.source 必须是 SourceRef 或 None")
        if not isinstance(self.payload, tuple):
            raise TypeError("BoundaryEvidenceSpec.payload 必须是整数 tuple")
        assert_int(*self.payload, _where="BoundaryEvidenceSpec.payload")
        if any(type(value) is not int for value in self.payload):
            raise ValueError("BoundaryEvidenceSpec.payload 必须使用严格整数")


@dataclass(frozen=True)
class BoundaryEvidenceProfile:
    """一个文档可见的全部边界候选证据，不包含字符作用白名单。"""

    evidence: tuple[BoundaryEvidenceSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, tuple):
            raise TypeError("BoundaryEvidenceProfile.evidence 必须是 tuple")
        if any(not isinstance(item, BoundaryEvidenceSpec)
               for item in self.evidence):
            raise TypeError("边界 profile 只能包含 BoundaryEvidenceSpec")


@dataclass(frozen=True)
class BoundaryHypothesisProtocol:
    """指定句界候选使用的开放 Hypothesis kind。"""

    hypothesis_kind_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(
            self.hypothesis_kind_key,
            where="BoundaryHypothesisProtocol.hypothesis_kind_key",
        )


@dataclass(frozen=True)
class BoundaryCandidateSnapshot:
    """一个码点边界方案及其当前 H-00 快照。"""

    candidate: BoundaryCandidate
    hypothesis: HypothesisKey
    snapshot: HypothesisSnapshot


@dataclass(frozen=True)
class BoundaryDecision:
    """供分段消费者使用的来源化已决边界；未决时 anchors 为空。"""

    text: str
    observation: SourceRef
    scope: ScopeIdentity
    language_key: tuple[int, ...]
    selected_hypothesis: HypothesisKey | None
    anchors: tuple[int, ...]

    def token_cuts(
            self, token_spans: tuple[tuple[int, int, int], ...]
            ) -> tuple[int, ...]:
        """把内部码点锚点严格映射到当前 winner token 的右边界。"""
        if not isinstance(token_spans, tuple):
            raise TypeError("token_spans 必须是三元整数 tuple")
        previous_end = -1
        for index, span in enumerate(token_spans):
            if not isinstance(span, tuple) or len(span) != 3:
                raise ValueError("token_spans 必须包含 (start,end,ordinal)")
            start, end, ordinal = span
            assert_int(start, end, ordinal,
                       _where=f"BoundaryDecision.token_spans[{index}]")
            if (type(start) is not int or type(end) is not int
                    or type(ordinal) is not int
                    or start < 0 or end <= start or ordinal < 0
                    or start < previous_end or end > len(self.text)):
                raise ValueError("token span 顺序、范围或 ordinal 非法")
            previous_end = end
        cuts: list[int] = []
        for anchor in self.anchors:
            matches = [
                index + 1 for index, (_, end, _) in enumerate(token_spans)
                if end == anchor
            ]
            if len(matches) != 1:
                raise ValueError("已决边界无法唯一对齐当前 winner token")
            cut = matches[0]
            if cut < len(token_spans) and token_spans[cut][0] < anchor:
                raise ValueError("已决边界落入后续 token 内部")
            cuts.append(cut)
        return tuple(cuts)


@dataclass(frozen=True)
class BoundaryResult:
    """保留全部边界候选和当前唯一可消费决定。"""

    text: str
    observation: SourceRef
    scope: ScopeIdentity
    language_key: tuple[int, ...]
    candidates: tuple[BoundaryCandidateSnapshot, ...]
    selected_hypothesis: HypothesisKey | None
    adopted_hypotheses: tuple[HypothesisKey, ...] = ()
    resolver_decision_id: int = 0

    @property
    def selected(self) -> BoundaryCandidateSnapshot | None:
        """返回唯一受支持 active 候选；未决时返回 None。"""
        if self.selected_hypothesis is None:
            return None
        for candidate in self.candidates:
            if candidate.hypothesis == self.selected_hypothesis:
                return candidate
        raise ValueError("selected_hypothesis 不在边界候选集合中")

    def decision(self) -> BoundaryDecision:
        """投影为不携带 Evidence 细节的分段消费对象。"""
        selected = self.selected
        return BoundaryDecision(
            self.text,
            self.observation,
            self.scope,
            self.language_key,
            self.selected_hypothesis,
            () if selected is None else selected.candidate.anchors,
        )


def _evidence_sort_key(spec: BoundaryEvidenceSpec,
                       observation: SourceRef) -> tuple:
    """生成与 profile 原始排列无关的证据规范顺序。"""
    source = observation if spec.source is None else spec.source
    return (
        spec.candidate.stable_key(),
        spec.stance,
        spec.reason_key,
        source.stable_key(),
        spec.timestamp_seq,
        spec.payload,
    )


def _candidate_from_hypothesis(
        hypothesis: HypothesisKey) -> BoundaryCandidate:
    """从边界 Hypothesis 的完整 candidate key 恢复候选锚点。"""
    values = hypothesis.candidate_key
    if not values or values[0] < 0 or len(values) != values[0] + 1:
        raise ValueError("边界 Hypothesis candidate key 无法恢复")
    candidate = BoundaryCandidate(tuple(values[1:]))
    if candidate.stable_key() != values:
        raise ValueError("边界 Hypothesis candidate key 非规范")
    return candidate


class BoundaryHypothesisEngine:
    """登记边界候选和证据，只在唯一受支持时产生消费者决定。"""

    def __init__(self, protocol: BoundaryHypothesisProtocol, *,
                 ledger: HypothesisLedger | None = None,
                 resolver: HypothesisResolver | None = None) -> None:
        if not isinstance(protocol, BoundaryHypothesisProtocol):
            raise TypeError("protocol 必须是 BoundaryHypothesisProtocol")
        self.protocol = protocol
        self.ledger = ledger or HypothesisLedger()
        self.resolver = resolver or HypothesisResolver(self.ledger)
        if self.resolver.ledger is not self.ledger:
            raise ValueError("句界 resolver 必须绑定同一 H-00 ledger")

    def resolve(
            self, text: str, *, observation: SourceRef,
            scope: ScopeIdentity, language_key: tuple[int, ...],
            profile: BoundaryEvidenceProfile | None = None,
            commit: bool = True,
            ) -> BoundaryResult:
        """把显式证据变成来源化候选；无唯一支持时保持整段。"""
        if not isinstance(text, str):
            raise TypeError("边界解析输入必须是字符串")
        if not isinstance(observation, SourceRef):
            raise TypeError("observation 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity) or scope.source != observation:
            raise ValueError("边界解析 scope 必须指向同一 observation")
        _strict_key(language_key, where="BoundaryHypothesisEngine.language_key")
        if profile is None:
            profile = BoundaryEvidenceProfile()
        if not isinstance(profile, BoundaryEvidenceProfile):
            raise TypeError("profile 必须是 BoundaryEvidenceProfile 或 None")
        if type(commit) is not bool:
            raise TypeError("commit 必须是 bool")
        if not commit:
            return self.clone().resolve(
                text,
                observation=observation,
                scope=scope,
                language_key=language_key,
                profile=profile,
                commit=True,
            )

        competition_key = (
            len(language_key),
            *language_key,
            len(text),
        )
        by_candidate: dict[BoundaryCandidate, list[BoundaryEvidenceSpec]] = {}
        for spec in profile.evidence:
            if any(anchor >= len(text) for anchor in spec.candidate.anchors):
                raise ValueError("边界锚点必须位于原文内部")
            by_candidate.setdefault(spec.candidate, []).append(spec)

        snapshots: list[BoundaryCandidateSnapshot] = []
        for candidate in sorted(by_candidate):
            hypothesis = HypothesisKey(
                self.protocol.hypothesis_kind_key,
                candidate.stable_key(),
                competition_key,
                scope,
                observation,
            )
            self.ledger.register(hypothesis)
            duplicate_counts: dict[tuple, int] = {}
            for spec in sorted(
                    by_candidate[candidate],
                    key=lambda item: _evidence_sort_key(item, observation)):
                source = observation if spec.source is None else spec.source
                evidence_key = _evidence_sort_key(spec, observation)
                duplicate_ordinal = duplicate_counts.get(evidence_key, 0)
                duplicate_counts[evidence_key] = duplicate_ordinal + 1
                evidence_id = _EVIDENCE_HASHER.h63((
                    hypothesis.stable_key(),
                    evidence_key,
                    duplicate_ordinal,
                )) or 1
                self.ledger.append_evidence(EvidenceRecord(
                    evidence_id,
                    hypothesis,
                    spec.stance,
                    spec.reason_key,
                    source,
                    spec.timestamp_seq,
                    payload=spec.payload,
                ))
            snapshots.append(BoundaryCandidateSnapshot(
                candidate,
                hypothesis,
                self.ledger.snapshot(hypothesis),
            ))

        ordered = tuple(sorted(
            snapshots,
            key=lambda item: (
                item.candidate.stable_key(),
                item.hypothesis.stable_key(),
            ),
        ))
        if not ordered:
            return BoundaryResult(
                text,
                observation,
                scope,
                language_key,
                (),
                None,
            )
        decision = self.resolver.resolve(
            ordered[0].hypothesis,
            timestamp_seq=(
                max((
                    spec.timestamp_seq for spec in profile.evidence
                ), default=0) + 1
            ),
        )
        refreshed = tuple(
            BoundaryCandidateSnapshot(
                _candidate_from_hypothesis(item.hypothesis),
                item.hypothesis,
                item,
            )
            for item in self.ledger.competition(ordered[0].hypothesis)
        )
        adopted_supported = tuple(
            item.hypothesis for item in refreshed
            if (item.hypothesis in decision.adopted_hypotheses
                and item.snapshot.lifecycle == LIFECYCLE_ACTIVE
                and item.snapshot.epistemic_status == EPISTEMIC_SUPPORTED)
        )
        selected = (
            adopted_supported[0]
            if len(adopted_supported) == 1 else None)
        return BoundaryResult(
            text,
            observation,
            scope,
            language_key,
            refreshed,
            selected,
            decision.adopted_hypotheses,
            decision.decision_id,
        )

    def record_feedback(
            self, hypothesis: HypothesisKey, *, stance: int,
            source: SourceRef, reason_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: HypothesisKey | None = None,
            ) -> HypothesisSnapshot:
        """追加反馈；反驳可把错误候选归档或替代，但不删除历史。"""
        self.validate_feedback(
            hypothesis,
            stance=stance,
            source=source,
            reason_key=reason_key,
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )
        evidence_id = _EVIDENCE_HASHER.h63((
            hypothesis.stable_key(),
            stance,
            source.stable_key(),
            reason_key,
            timestamp_seq,
            () if replacement is None else replacement.stable_key(),
        )) or 1
        evidence = self.ledger.append_evidence(EvidenceRecord(
            evidence_id,
            hypothesis,
            stance,
            reason_key,
            source,
            timestamp_seq,
        ))
        replacements = (
            () if replacement is None
            else (ReplacementDirective(
                hypothesis,
                replacement,
                evidence.evidence_id,
            ),)
        )
        self.resolver.resolve(
            hypothesis,
            timestamp_seq=timestamp_seq,
            replacements=replacements,
        )
        return self.ledger.snapshot(hypothesis)

    def next_timestamp(self, result: BoundaryResult | None = None) -> int:
        """从当前边界 Evidence、transition 和 H-04 decision 派生下一逻辑序。"""
        if result is not None and not isinstance(result, BoundaryResult):
            raise TypeError("result 必须是 BoundaryResult 或 None")
        if result is None or not result.candidates:
            return 1
        values = [0]
        for candidate in result.candidates:
            values.extend(
                item.timestamp_seq
                for item in self.ledger.evidence_history(candidate.hypothesis)
            )
            values.extend(
                item.timestamp_seq
                for item in self.ledger.transition_history(candidate.hypothesis)
            )
            values.extend(
                item.timestamp_seq
                for item in self.resolver.decision_history(candidate.hypothesis)
            )
        return max(values) + 1

    def validate_feedback(
            self, hypothesis: HypothesisKey, *, stance: int,
            source: SourceRef, reason_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: HypothesisKey | None = None,
            ) -> None:
        """在写入前核验反馈、replacement 和完整竞争边界。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("反馈 hypothesis 必须是 HypothesisKey")
        if stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("反馈 stance 未注册")
        if not isinstance(source, SourceRef):
            raise TypeError("反馈 source 必须是 SourceRef")
        _strict_key(reason_key, where="Boundary feedback.reason_key")
        assert_int(timestamp_seq, _where="Boundary feedback.timestamp_seq")
        if timestamp_seq < 0:
            raise ValueError("反馈逻辑序不得为负")
        self.ledger.snapshot(hypothesis)
        if replacement is None:
            return
        if stance != EVIDENCE_REFUTE:
            raise ValueError("只有反驳反馈可以指定 replacement")
        replacement_snapshot = self.ledger.snapshot(replacement)
        if replacement_snapshot.lifecycle != LIFECYCLE_ACTIVE:
            raise ValueError("replacement 必须是已登记的 active 候选")
        if (
                replacement == hypothesis
                or replacement.hypothesis_kind != hypothesis.hypothesis_kind
                or replacement.competition_key != hypothesis.competition_key
                or replacement.scope != hypothesis.scope
                or replacement.observation != hypothesis.observation):
            raise ValueError("replacement 必须属于同一竞争组")

    def clone(self) -> "BoundaryHypothesisEngine":
        """复制协议和 ledger，供 probe 预览与评测写隔离。"""
        ledger = self.ledger.clone()
        return BoundaryHypothesisEngine(
            self.protocol,
            ledger=ledger,
            resolver=self.resolver.clone(ledger=ledger),
        )

    def state_key(self) -> tuple:
        """返回句界 H-00 事件和 H-04 决策链的完整可比较状态。"""
        return self.ledger.state_key(), self.resolver.state_key()


__all__ = [
    "BoundaryCandidate",
    "BoundaryCandidateSnapshot",
    "BoundaryDecision",
    "BoundaryEvidenceProfile",
    "BoundaryEvidenceSpec",
    "BoundaryHypothesisEngine",
    "BoundaryHypothesisProtocol",
    "BoundaryResult",
]
