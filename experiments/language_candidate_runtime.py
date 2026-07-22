"""正式训练上下文中的 H-05 structure/Sense 候选协议装配。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.understanding.language_candidate import (
    ActiveCueStructureConsumer,
    ActiveSenseConsumer,
    CueStructureCandidateProtocol,
    SenseCandidateProtocol,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.language_structure_candidate_runtime import (
    StructureCandidateCourseMapper,
)
from pure_integer_ai.experiments.language_structure_boundary_runtime import (
    StructureBoundaryEvidenceMapper,
)
from pure_integer_ai.experiments.language_sense_candidate_runtime import (
    SenseCandidateCourseMapper,
    SenseCandidateCourseRuntime,
)


@dataclass(frozen=True)
class CandidateDomainRuntimeProtocol:
    """一个候选领域的 H-00 forming 协议和独立 verifier 协议。"""

    evidence: EvidenceCandidateProtocol
    verifier: IndependentVerifierProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceCandidateProtocol):
            raise TypeError("domain evidence 必须是 EvidenceCandidateProtocol")
        if not isinstance(self.verifier, IndependentVerifierProtocol):
            raise TypeError("domain verifier 必须是 IndependentVerifierProtocol")


@dataclass(frozen=True)
class LanguageCandidateRuntimeProtocol:
    """共享图 lifecycle 与 structure/Sense 各自 owner、字段的完整注入协议。"""

    projection: CandidateProjectionProtocol
    metadata: CandidateProjectionMetadata
    structure_domain: CandidateDomainRuntimeProtocol
    structure_fields: CueStructureCandidateProtocol
    structure_mapper: StructureCandidateCourseMapper
    sense_domain: CandidateDomainRuntimeProtocol
    sense_fields: SenseCandidateProtocol
    sense_mapper: SenseCandidateCourseMapper
    structure_boundary_mapper: StructureBoundaryEvidenceMapper | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.projection, CandidateProjectionProtocol):
            raise TypeError("projection 必须是 CandidateProjectionProtocol")
        if not isinstance(self.metadata, CandidateProjectionMetadata):
            raise TypeError("metadata 必须是 CandidateProjectionMetadata")
        if not isinstance(
                self.structure_domain, CandidateDomainRuntimeProtocol):
            raise TypeError("structure_domain 类型非法")
        if not isinstance(
                self.structure_fields, CueStructureCandidateProtocol):
            raise TypeError("structure_fields 类型非法")
        if not isinstance(self.structure_mapper, StructureCandidateCourseMapper):
            raise TypeError("structure_mapper 未实现课程映射协议")
        if not isinstance(self.sense_domain, CandidateDomainRuntimeProtocol):
            raise TypeError("sense_domain 类型非法")
        if not isinstance(self.sense_fields, SenseCandidateProtocol):
            raise TypeError("sense_fields 类型非法")
        if not isinstance(self.sense_mapper, SenseCandidateCourseMapper):
            raise TypeError("sense_mapper 未实现课程映射协议")
        if (self.structure_boundary_mapper is not None
                and not isinstance(
                    self.structure_boundary_mapper,
                    StructureBoundaryEvidenceMapper)):
            raise TypeError("structure_boundary_mapper 未实现课程映射协议")
        if (self.structure_domain.evidence.hypothesis_kind_key
                == self.sense_domain.evidence.hypothesis_kind_key):
            raise ValueError("structure 与 Sense 必须使用不同 Hypothesis kind")


def _domain_runtime(
        domain: CandidateDomainRuntimeProtocol,
        graph: CandidateProjectionGraph,
        metadata: CandidateProjectionMetadata) -> CandidateLearningRuntime:
    """为一个领域建立独立 H-00/H-04 owner 并复用共享 lifecycle 图。"""
    return CandidateLearningRuntime(
        EvidenceCandidateEngine(domain.evidence),
        graph,
        IndependentObjectVerifier(domain.verifier),
        metadata,
    )


def install_language_candidate_runtime(
        ctx: TrainContext,
        protocol: LanguageCandidateRuntimeProtocol) -> None:
    """在 occurrence/span 地基上安装 structure/Sense 候选 writer 和 typed reader。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(protocol, LanguageCandidateRuntimeProtocol):
        raise TypeError("protocol 必须是 LanguageCandidateRuntimeProtocol")
    if ctx.occurrence_index is None or ctx.span_index is None:
        raise ValueError("H-05 候选协议必须同时启用 L-03 occurrence 和 L-04 Span")
    graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        protocol.projection,
    )
    ctx.candidate_projection_graph = graph
    ctx.structure_candidate_runtime = _domain_runtime(
        protocol.structure_domain,
        graph,
        protocol.metadata,
    )
    ctx.structure_candidate_consumer = ActiveCueStructureConsumer(
        graph,
        protocol.structure_fields,
    )
    ctx.structure_candidate_mapper = protocol.structure_mapper
    ctx.structure_boundary_evidence_mapper = (
        protocol.structure_boundary_mapper)
    ctx.sense_candidate_runtime = _domain_runtime(
        protocol.sense_domain,
        graph,
        protocol.metadata,
    )
    ctx.sense_candidate_consumer = ActiveSenseConsumer(
        graph,
        protocol.sense_fields,
    )
    ctx.sense_candidate_course_runtime = SenseCandidateCourseRuntime(
        protocol.sense_mapper)


__all__ = [
    "CandidateDomainRuntimeProtocol",
    "LanguageCandidateRuntimeProtocol",
    "install_language_candidate_runtime",
]
