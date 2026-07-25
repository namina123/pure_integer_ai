"""把分词边界候选接到通用 Hypothesis/Evidence 协议。

词形命中只产生支持 Evidence，连续 OOV 只产生 unknown Evidence；候选排序和生命周期
保持分离。调用方可用反例把错误候选 supersede/archive，旧 Evidence 与旧候选仍保留审计。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
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
    PREFERENCE_EQUIVALENT,
    PREFERENCE_INCOMPARABLE,
    PREFERENCE_LEFT_BETTER,
    PREFERENCE_RIGHT_BETTER,
    ReplacementDirective,
    ResolverPreference,
)
from pure_integer_ai.cognition.shared.identity import SourceRef, TypedRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.segmentation_candidates import (
    SegmentationCandidate,
    build_segmentation_candidates,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_EVIDENCE_HASHER = Hasher("segmentation_hypothesis.evidence.v1")


def _protocol_key(value, *, where: str) -> tuple[int, ...]:
    """校验由图内 MinimalInstruction 身份使用的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SegmentationProtocol:
    """由调用方注入的分词 Hypothesis、Evidence 理由和候选预算。"""

    hypothesis_kind_key: tuple[int, ...]
    lexical_match_reason_key: tuple[int, ...]
    oov_reason_key: tuple[int, ...]
    candidate_limit: int

    def __post_init__(self) -> None:
        _protocol_key(
            self.hypothesis_kind_key,
            where="SegmentationProtocol.hypothesis_kind_key",
        )
        _protocol_key(
            self.lexical_match_reason_key,
            where="SegmentationProtocol.lexical_match_reason_key",
        )
        _protocol_key(
            self.oov_reason_key,
            where="SegmentationProtocol.oov_reason_key",
        )
        assert_int(self.candidate_limit, _where="SegmentationProtocol")
        if self.candidate_limit < 3:
            raise ValueError("分词候选预算必须至少为 3")

    def instruction_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回需要在图中物化的最小协议符号键。"""
        return tuple(sorted({
            self.hypothesis_kind_key,
            self.lexical_match_reason_key,
            self.oov_reason_key,
        }))

    def resolver_scorer_key(self) -> tuple[int, ...]:
        """由完整注入协议组成 scorer 身份，不生成领域常量或摘要。"""
        keys = (
            self.hypothesis_kind_key,
            self.lexical_match_reason_key,
            self.oov_reason_key,
        )
        return tuple(
            value for key in keys for value in (len(key), *key))


@dataclass(frozen=True)
class SegmentationHypothesisCandidate:
    """一个边界方案及其通用 Hypothesis 当前快照。"""

    segmentation: SegmentationCandidate
    hypothesis: HypothesisKey
    snapshot: HypothesisSnapshot


@dataclass(frozen=True)
class SegmentationResult:
    """保留全部候选和当前消费者可采用的确定性排序。"""

    text: str
    candidates: tuple[SegmentationHypothesisCandidate, ...]
    selected_hypothesis: HypothesisKey | None
    adopted_hypotheses: tuple[HypothesisKey, ...] = ()
    resolver_decision_id: int = 0

    @property
    def selected(self) -> SegmentationHypothesisCandidate | None:
        """返回当前可采用 winner；全部候选退出时返回 None。"""
        if self.selected_hypothesis is None:
            return None
        for candidate in self.candidates:
            if candidate.hypothesis == self.selected_hypothesis:
                return candidate
        raise ValueError("selected_hypothesis 不在候选集合中")

    @property
    def tokens(self) -> tuple[str, ...]:
        """返回当前 winner token；无可采用候选时 fail closed。"""
        selected = self.selected
        if selected is None:
            if not self.candidates and not self.text.strip():
                return ()
            raise LookupError("当前输入没有可采用的分词候选")
        return selected.segmentation.tokens

    @property
    def consumer_candidates(
            self) -> tuple[SegmentationHypothesisCandidate, ...]:
        """返回 resolver 本次采用的全部候选，语义并列时可多于一个。"""
        adopted = frozenset(
            self.adopted_hypotheses
            if self.adopted_hypotheses
            else (() if self.selected_hypothesis is None
                  else (self.selected_hypothesis,))
        )
        return tuple(
            candidate for candidate in self.candidates
            if candidate.hypothesis in adopted
        )


def _candidate_rank(
        candidate: SegmentationHypothesisCandidate) -> tuple:
    """先按生命周期/证据状态，再按相互独立的边界统计排序。"""
    lifecycle_rank = (
        0 if candidate.snapshot.lifecycle == LIFECYCLE_ACTIVE else 1)
    epistemic_rank = {
        EPISTEMIC_SUPPORTED: 0,
        EPISTEMIC_UNKNOWN: 1,
        EPISTEMIC_CONFLICTED: 2,
        EPISTEMIC_REFUTED: 3,
    }[candidate.snapshot.epistemic_status]
    segmentation = candidate.segmentation
    return (
        lifecycle_rank,
        epistemic_rank,
        -segmentation.known_codepoints,
        segmentation.unknown_spans,
        len(segmentation.parts),
        segmentation.stable_key(),
    )


class _SegmentationResolverScorer:
    """把边界结构统计适配为无权重 Pareto 候选对关系。"""

    def __init__(
            self, protocol: SegmentationProtocol,
            candidates: tuple[SegmentationHypothesisCandidate, ...],
            ) -> None:
        """绑定本次候选，scorer 身份完全来自调用方注入协议。"""
        self.scorer_key = protocol.resolver_scorer_key()
        self._candidates = {
            item.hypothesis: item.segmentation for item in candidates
        }
        if len(self._candidates) != len(candidates):
            raise ValueError("分词 resolver scorer 不得重复绑定 Hypothesis")

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]:
        """按已知覆盖高、未知段少、part 少三轴生成保守支配关系。"""
        if any(item not in self._candidates for item in hypotheses):
            raise ValueError("分词 scorer 缺少 resolver eligible 候选")
        preferences: list[ResolverPreference] = []
        for left_index in range(len(hypotheses)):
            for right_index in range(left_index + 1, len(hypotheses)):
                left = hypotheses[left_index]
                right = hypotheses[right_index]
                left_profile = self._profile(self._candidates[left])
                right_profile = self._profile(self._candidates[right])
                left_better = self._dominates(left_profile, right_profile)
                right_better = self._dominates(right_profile, left_profile)
                if left_better:
                    preference = PREFERENCE_LEFT_BETTER
                elif right_better:
                    preference = PREFERENCE_RIGHT_BETTER
                elif left_profile == right_profile:
                    preference = PREFERENCE_EQUIVALENT
                else:
                    preference = PREFERENCE_INCOMPARABLE
                preferences.append(ResolverPreference(
                    self.scorer_key,
                    left,
                    right,
                    preference,
                    (*left_profile, *right_profile),
                ))
        return tuple(preferences)

    @staticmethod
    def _profile(candidate: SegmentationCandidate) -> tuple[int, int, int]:
        """返回三个独立统计轴，不预乘权重或拼成综合 strength。"""
        return (
            candidate.known_codepoints,
            candidate.unknown_spans,
            len(candidate.parts),
        )

    @staticmethod
    def _dominates(
            left: tuple[int, int, int], right: tuple[int, int, int],
            ) -> bool:
        """仅在三轴均不差且至少一轴更优时建立 Pareto 支配。"""
        no_worse = (
            left[0] >= right[0]
            and left[1] <= right[1]
            and left[2] <= right[2]
        )
        return no_worse and left != right


class SegmentationHypothesisEngine:
    """生成分词候选、追加 Evidence，并从 ledger 选择当前消费者方案。"""

    def __init__(self, protocol: SegmentationProtocol, *,
                 ledger: HypothesisLedger | None = None,
                 resolver: HypothesisResolver | None = None) -> None:
        if not isinstance(protocol, SegmentationProtocol):
            raise TypeError("protocol 必须是 SegmentationProtocol")
        self.protocol = protocol
        self.ledger = ledger or HypothesisLedger()
        self.resolver = resolver or HypothesisResolver(self.ledger)
        if self.resolver.ledger is not self.ledger:
            raise ValueError("分词 resolver 必须绑定同一 H-00 ledger")

    def parse(
            self, text: str, *, lattice: tuple[tuple[str, ...], ...],
            branch: TypedRef, observation: SourceRef,
            scope: ScopeIdentity,
            visible_form: Callable[[str], object | None],
            sequence_base: int = 0,
            commit: bool = True,
            ) -> SegmentationResult:
        """把 lattice 转成来源化候选，并为词形/OOV 追加支持或未知 Evidence。"""
        if not isinstance(branch, TypedRef):
            raise TypeError("branch 必须是 TypedRef")
        if not isinstance(observation, SourceRef):
            raise TypeError("observation 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        assert_int(sequence_base, _where="SegmentationHypothesisEngine.parse")
        if sequence_base < 0:
            raise ValueError("sequence_base 不得为负")
        if type(commit) is not bool:
            raise TypeError("commit 必须是 bool")
        if not commit:
            preview = self.clone()
            return preview.parse(
                text,
                lattice=lattice,
                branch=branch,
                observation=observation,
                scope=scope,
                visible_form=visible_form,
                sequence_base=sequence_base,
                commit=True,
            )
        base_candidates = build_segmentation_candidates(
            text,
            lattice,
            candidate_limit=self.protocol.candidate_limit,
        )
        competition_key = (
            *branch.stable_key(),
            len(text),
            *(ord(char) for char in text),
        )
        candidates: list[SegmentationHypothesisCandidate] = []
        for segmentation in base_candidates:
            hypothesis = HypothesisKey(
                self.protocol.hypothesis_kind_key,
                (
                    *competition_key,
                    *segmentation.stable_key(),
                ),
                competition_key,
                scope,
                observation,
            )
            self.ledger.register(hypothesis)
            evidence_hash_prefix = _EVIDENCE_HASHER.prepare_tuple_prefix(
                7,
                (hypothesis.stable_key(),),
            )
            for part_index, part in enumerate(segmentation.parts):
                visible = visible_form(part.surface) if part.known_word_form else None
                source = (
                    visible.source_ref
                    if visible is not None and hasattr(visible, "source_ref")
                    else observation)
                stance = (
                    EVIDENCE_SUPPORT if visible is not None
                    else EVIDENCE_UNKNOWN)
                reason_key = (
                    self.protocol.lexical_match_reason_key
                    if visible is not None
                    else self.protocol.oov_reason_key)
                evidence_id = evidence_hash_prefix.h63((
                    part_index,
                    part.start,
                    part.end,
                    stance,
                    reason_key,
                    source.stable_key(),
                )) or 1
                self.ledger.append_evidence(EvidenceRecord(
                    evidence_id,
                    hypothesis,
                    stance,
                    reason_key,
                    source,
                    sequence_base + part_index + 1,
                    payload=(
                        part.start,
                        part.end,
                        1 if part.known_word_form else 0,
                    ),
                ))
            candidates.append(SegmentationHypothesisCandidate(
                segmentation,
                hypothesis,
                self.ledger.snapshot(hypothesis),
            ))
        ranked_before = tuple(sorted(candidates, key=_candidate_rank))
        if not ranked_before:
            return SegmentationResult(text, (), None, (), 0)
        decision = self.resolver.resolve(
            ranked_before[0].hypothesis,
            timestamp_seq=(
                sequence_base
                + max(len(item.segmentation.parts) for item in ranked_before)
                + 1
            ),
            scorers=(
                _SegmentationResolverScorer(self.protocol, ranked_before),),
        )
        refreshed = tuple(
            SegmentationHypothesisCandidate(
                item.segmentation,
                item.hypothesis,
                self.ledger.snapshot(item.hypothesis),
            )
            for item in ranked_before
        )
        ranked = tuple(sorted(refreshed, key=_candidate_rank))
        adopted = decision.adopted_hypotheses
        selected = adopted[0] if len(adopted) == 1 else None
        return SegmentationResult(
            text,
            ranked,
            selected,
            adopted,
            decision.decision_id,
        )

    def record_feedback(
            self, hypothesis: HypothesisKey, *, stance: int,
            source: SourceRef, reason_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: HypothesisKey | None = None,
            ) -> HypothesisSnapshot:
        """追加边界反馈；反例会使错误候选退出消费者但保留完整历史。"""
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

    def validate_feedback(
            self, hypothesis: HypothesisKey, *, stance: int,
            source: SourceRef, reason_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: HypothesisKey | None = None,
            ) -> None:
        """在任何 ledger 或图写入前核验反馈和替代候选的完整竞争边界。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("反馈 hypothesis 必须是 HypothesisKey")
        if stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("反馈 stance 未注册")
        if not isinstance(source, SourceRef):
            raise TypeError("反馈 source 必须是 SourceRef")
        _protocol_key(reason_key, where="record_feedback.reason_key")
        assert_int(timestamp_seq, _where="record_feedback.timestamp_seq")
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

    def clone(self) -> "SegmentationHypothesisEngine":
        """复制协议和 ledger，供评测写隔离。"""
        ledger = self.ledger.clone()
        return SegmentationHypothesisEngine(
            self.protocol,
            ledger=ledger,
            resolver=self.resolver.clone(ledger=ledger),
        )

    def state_key(self) -> tuple:
        """返回 H-00 事件和 H-04 决策链的完整可比较状态。"""
        return self.ledger.state_key(), self.resolver.state_key()

    def resolution_history(self, hypothesis: HypothesisKey) -> tuple:
        """返回候选所在竞争组的完整 H-04 append-only 决策历史。"""
        return self.resolver.decision_history(hypothesis)


__all__ = [
    "SegmentationHypothesisCandidate",
    "SegmentationHypothesisEngine",
    "SegmentationProtocol",
    "SegmentationResult",
]
