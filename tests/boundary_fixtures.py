"""测试专用的显式句界 Evidence 构造器。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
    BoundaryCandidate,
    BoundaryEvidenceProfile,
    BoundaryEvidenceSpec,
    BoundaryHypothesisEngine,
    BoundaryHypothesisProtocol,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher

_SOURCE_HASHER = Hasher("tests.boundary_fixture.source.v1")
_HYPOTHESIS_KIND = (999700, 1)
_EVIDENCE_REASON = (999700, 2)


def attach_boundary_fixture(
        item, *, cut_after: tuple[int, ...],
        source_ref: SourceRef | None = None):
    """给旧 token fixture 附加来源化句界决定，不按字符内容推断作用。"""
    raw_text = "".join(item.tokens)
    source_id = _SOURCE_HASHER.h63((
        item.source,
        item.lang,
        item.domain,
        item.modality,
        raw_text,
    )) or 1
    source = source_ref or SourceRef(
        item.source, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    if source.source_kind != item.source:
        raise ValueError("句界 fixture 来源类型与 CollectedItem 不一致")
    token_ends: list[int] = []
    cursor = 0
    for token in item.tokens:
        cursor += len(token)
        token_ends.append(cursor)
    internal_anchors = tuple(
        token_ends[cut - 1]
        for cut in sorted(set(cut_after))
        if 0 < cut < len(item.tokens)
    )
    profile = BoundaryEvidenceProfile((BoundaryEvidenceSpec(
        BoundaryCandidate(internal_anchors),
        EVIDENCE_SUPPORT,
        _EVIDENCE_REASON,
    ),))
    result = BoundaryHypothesisEngine(BoundaryHypothesisProtocol(
        _HYPOTHESIS_KIND)).resolve(
            raw_text,
            observation=source,
            scope=document_scope(source),
            language_key=(item.lang,),
            profile=profile,
        )
    item.raw_text = raw_text
    item.source_ref = source
    item.boundary_profile = profile
    item.boundary_parse = result
    item.boundary_decision = result.decision()
    return item


__all__ = ["attach_boundary_fixture"]
