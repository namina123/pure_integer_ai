"""用 FT19 prefix/suffix anchor 交集缩小 FT18 alias frame。"""
from __future__ import annotations

from bisect import bisect_left

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor_contract import (
    QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY,
    QUESTION_ALIAS_FRAME_ANCHOR_SHA256,
    RawQuestionAliasFrameAnchorEdge,
    RawQuestionAliasFrameAnchorIndex,
    RawQuestionAliasFrameAnchorNode,
    RawQuestionAliasFrameAnchorTerminal,
    W03W04W05QuestionAliasFrameAnchorError,
    build_raw_question_alias_frame_anchor_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    lookup_indexed_question_feature_alias_frame_constructions,
    lookup_indexed_question_feature_direct_constructions,
    merge_indexed_question_construction_candidates,
    run_indexed_question_construction_registry_answer_with_lookup,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index_contract import (
    RawQuestionConstructionCandidateSet,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    RawQuestionFeatureRegistryAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)


def _terminal_frame_ordinals(
        terminals: tuple[RawQuestionAliasFrameAnchorTerminal, ...],
        source_record_key: tuple[int, ...] | None,
        ) -> tuple[int, ...]:
    if source_record_key is None:
        return tuple(item.frame_ordinal for item in terminals)
    values = []
    for terminal in terminals:
        ordinal = bisect_left(
            terminal.source_record_keys,
            source_record_key,
        )
        if (ordinal < len(terminal.source_record_keys)
                and terminal.source_record_keys[ordinal]
                == source_record_key):
            values.append(terminal.frame_ordinal)
    return tuple(values)


def _walk_alias_frame_anchor(
        nodes: tuple[RawQuestionAliasFrameAnchorNode, ...],
        surface: str,
        source_record_key: tuple[int, ...] | None,
        ) -> tuple[int, ...]:
    node = nodes[0]
    values = set(_terminal_frame_ordinals(
        node.terminals,
        source_record_key,
    ))
    for character in surface:
        edge_ordinal = bisect_left(
            node.edges,
            character,
            key=lambda item: item.surface,
        )
        if (edge_ordinal >= len(node.edges)
                or node.edges[edge_ordinal].surface != character):
            break
        node = nodes[node.edges[edge_ordinal].child_ordinal]
        values.update(_terminal_frame_ordinals(
            node.terminals,
            source_record_key,
        ))
    return tuple(sorted(values))


def lookup_indexed_alias_frame_ordinals(
        index: RawQuestionAliasFrameAnchorIndex,
        request: RawQuestionRequest,
        ) -> tuple[int, ...]:
    """返回同时满足 prefix、suffix 与可选 SourceRef 的 frame。"""
    if (not isinstance(index, RawQuestionAliasFrameAnchorIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("alias frame anchor lookup inputs are invalid")
    prefix = set(_walk_alias_frame_anchor(
        index.prefix_nodes,
        request.question_surface,
        request.source_record_key,
    ))
    suffix = set(_walk_alias_frame_anchor(
        index.suffix_nodes,
        request.question_surface[::-1],
        request.source_record_key,
    ))
    return tuple(sorted(prefix & suffix))


def lookup_indexed_alias_frame_anchor_constructions(
        index: RawQuestionAliasFrameAnchorIndex,
        phase: str,
        request: RawQuestionRequest,
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    """exact/implicit 复用 FT18；alias 只访问 anchor 交集 frame。"""
    if (not isinstance(index, RawQuestionAliasFrameAnchorIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("alias frame anchor construction inputs are invalid")
    construction_index = index.construction_index
    direct = lookup_indexed_question_feature_direct_constructions(
        construction_index,
        phase,
        request,
    )
    if phase != "ALIAS":
        return direct
    structural = lookup_indexed_question_feature_alias_frame_constructions(
        construction_index,
        request,
        lookup_indexed_alias_frame_ordinals(index, request),
    )
    return merge_indexed_question_construction_candidates(
        (*direct, *structural))


def indexed_alias_frame_anchor_candidate_counts(
        index: RawQuestionAliasFrameAnchorIndex,
        request: RawQuestionRequest,
        ) -> tuple[int, int, int]:
    """返回 visited frame、registry entry 与 construction 候选数。"""
    frame_ordinals = lookup_indexed_alias_frame_ordinals(index, request)
    candidates = lookup_indexed_alias_frame_anchor_constructions(
        index,
        "ALIAS",
        request,
    )
    return (
        len(frame_ordinals),
        len(candidates),
        sum(len(item.constructions) for item in candidates),
    )


def run_indexed_alias_frame_anchor_registry_answer(
        index: RawQuestionAliasFrameAnchorIndex,
        request: RawQuestionRequest,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """复用 FT18 三阶段 runtime，只替换 alias frame 候选发现。"""
    if (not isinstance(index, RawQuestionAliasFrameAnchorIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("alias frame anchor dispatch inputs are invalid")
    return run_indexed_question_construction_registry_answer_with_lookup(
        index.construction_index,
        request,
        index,
        lookup_indexed_alias_frame_anchor_constructions,
    )


__all__ = [
    "QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY",
    "QUESTION_ALIAS_FRAME_ANCHOR_SHA256",
    "RawQuestionAliasFrameAnchorEdge",
    "RawQuestionAliasFrameAnchorIndex",
    "RawQuestionAliasFrameAnchorNode",
    "RawQuestionAliasFrameAnchorTerminal",
    "W03W04W05QuestionAliasFrameAnchorError",
    "build_raw_question_alias_frame_anchor_index",
    "indexed_alias_frame_anchor_candidate_counts",
    "lookup_indexed_alias_frame_anchor_constructions",
    "lookup_indexed_alias_frame_ordinals",
    "run_indexed_alias_frame_anchor_registry_answer",
]
