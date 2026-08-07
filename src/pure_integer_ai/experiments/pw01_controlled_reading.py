"""PW-01 受控读后问答的双 Memory 装配和可审计测试语义。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivationProposal,
    AttractorDependency,
    AttractorScoreReason,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONTEXT_SCOPE,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_HYPOTHESIS,
    MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningObligation,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _ACCESS,
    _goals,
    _instruction,
    _question_dialogue,
)
from pure_integer_ai.experiments.cross_memory_use_runtime import (
    install_cross_memory_use_runtime,
)
from pure_integer_ai.experiments.memory_resolver_runtime import (
    federate_hypothesis_memory_runtimes,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
)


PW01_HYPOTHESIS_KIND = (7201,)
PW01_CANDIDATE_KEY = (3,)
PW01_COMPETITION_KEY = (8603,)


def pw01_source(*, parser_version: int, document_id: int = 901) -> SourceRef:
    """构造只属于一个用户、可跨其 session 读取的受控自有来源。"""
    if type(parser_version) is not int or parser_version <= 0:
        raise ValueError("PW-01 parser_version 必须是正严格整数")
    if type(document_id) is not int or document_id <= 0:
        raise ValueError("PW-01 document_id 必须是正严格整数")
    return SourceRef(
        SOURCE_BARE_TEXT,
        2026080701,
        document_id,
        OwnerScope(1, 2, 0, VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(parser_version),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW01ControlledReadingParser:
    """把受控自有材料解析成 held-out 目标所需的一个来源化候选。"""

    source: SourceRef
    stance: int
    lineage_id: int

    def __post_init__(self) -> None:
        """冻结来源、证据立场和正整数 lineage，拒绝延迟到写入后失败。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("PW-01 parser source 必须是 SourceRef")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("PW-01 parser stance 未注册")
        if type(self.lineage_id) is not int or self.lineage_id <= 0:
            raise ValueError("PW-01 parser lineage_id 必须是正严格整数")

    def parse(self, source_slice: Any) -> ObservationIntakeDraft:
        """不解释 expected，只按注入来源和立场形成 M-05 草案。"""
        if source_slice.source != self.source:
            raise ValueError("PW-01 parser 收到其他来源")
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (2026080701, self.lineage_id),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(
            _instruction(self.source, 2026080710 + self.lineage_id))
        return ObservationIntakeDraft(
            (2026080720, self.lineage_id),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (2026080730, self.lineage_id),
                PW01_HYPOTHESIS_KIND,
                PW01_CANDIDATE_KEY,
                PW01_COMPETITION_KEY,
                self.stance,
                signal_ref=signal,
            ),),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW01HypothesisMapper:
    """保留原设施候选，并把新增阅读候选只映射到第三个 held-out 目标。"""

    memory_object_kind = MEMORY_OBJECT_HYPOTHESIS

    delegate: object
    memory_space: SpaceIdentity

    def __post_init__(self) -> None:
        """绑定原 Hypothesis mapper，使候选 1/2 的既有行为不变。"""
        if (not callable(getattr(self.delegate, "project", None))
                or not callable(getattr(self.delegate, "state_key", None))):
            raise TypeError("PW-01 delegate 缺少 Attractor mapper 协议")
        if (not isinstance(self.memory_space, SpaceIdentity)
                or self.memory_space.space_type != SPACE_TYPE_MEMORY):
            raise ValueError("PW-01 mapper memory_space 必须是 Memory 身份")

    def project(
            self,
            request: Any,
            candidate: Any,
            obligations: tuple[ReasoningObligation, ...],
            ) -> tuple[AttractorActivationProposal, ...]:
        """只识别完整 candidate 3；其余候选委托原 mapper。"""
        if (candidate.hypothesis is None
                or candidate.memory_ref is None
                or candidate.memory_ref.memory_space != self.memory_space
                or request.hypothesis_kind != PW01_HYPOTHESIS_KIND
                or candidate.hypothesis.candidate_key != PW01_CANDIDATE_KEY):
            return self.delegate.project(request, candidate, obligations)
        if len(obligations) < 3:
            raise ValueError("PW-01 held-out query 缺少第三个目标义务")
        obligation = obligations[2]
        dependency = AttractorDependency(
            request.query_kind, candidate.hypothesis)
        adjustment = 6000
        reason = AttractorScoreReason(
            _instruction(request.source, 2026080741),
            adjustment,
            (dependency,),
        )
        return (AttractorActivationProposal(
            _instruction(request.source, 2026080742),
            obligation,
            adjustment,
            (reason,),
            (dependency,),
        ),)

    def clone_for_context(self, ctx: TrainContext) -> "PW01HypothesisMapper":
        """按既有 clone 协议复制 delegate，不共享评测上下文可变状态。"""
        clone = getattr(self.delegate, "clone_for_context", None)
        delegate = self.delegate if clone is None else clone(ctx)
        result = PW01HypothesisMapper(
            delegate, ctx.memory_read_events.memory_space_identity)
        if result.state_key() != self.state_key():
            raise ValueError("PW-01 mapper clone 改变了协议状态")
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回联邦候选语义、目标序位和原 mapper 协议承诺。"""
        delegate_key = self.delegate.state_key()
        return (
            1,
            MEMORY_OBJECT_HYPOTHESIS,
            *PW01_HYPOTHESIS_KIND,
            *PW01_CANDIDATE_KEY,
            3,
            *self.memory_space.stable_key(),
            len(delegate_key),
            *delegate_key,
        )


def install_pw01_controlled_query(ctx: TrainContext) -> None:
    """在 query 外把阅读空间和第三目标 mapper 接入现有正式设施。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("PW-01 ctx 必须是 TrainContext")
    if ctx.attractor_runtime is None:
        raise ValueError("PW-01 安装前缺少 A-10 runtime")
    install_cross_memory_use_runtime(ctx, recovery_access=_ACCESS)
    mapper = PW01HypothesisMapper(
        ctx.attractor_runtime.mapper,
        ctx.memory_read_events.memory_space_identity,
    )
    federate_hypothesis_memory_runtimes(ctx, ctx.memory_read_aggregates)
    ctx.attractor_runtime.register_mapper_route(mapper)


def pw01_obligations(
        source: SourceRef,
        scope: Any,
        ) -> tuple[ReasoningObligation, ...]:
    """返回保留既有两个目标并追加 held-out 第三目标的义务集。"""
    return _goals(source, scope, count=3)


def build_pw01_question_dialogue(
        ctx: TrainContext,
        source: SourceRef,
        observation: Any,
        ) -> tuple[Any, Any]:
    """装配目标序位 3 的完整 Memory->Attractor->生成->Use 对话。"""
    return _question_dialogue(
        ctx,
        source,
        observation,
        target_index=2,
        obligation_factory=pw01_obligations,
    )


__all__ = [
    "PW01_CANDIDATE_KEY",
    "PW01_COMPETITION_KEY",
    "PW01ControlledReadingParser",
    "PW01_HYPOTHESIS_KIND",
    "build_pw01_question_dialogue",
    "install_pw01_controlled_query",
    "pw01_obligations",
    "pw01_source",
]
