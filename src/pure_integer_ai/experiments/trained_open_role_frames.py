"""从现役关系 Span 图恢复开放角色框架，不加载课程或宿主语言规则。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceFrame,
)


def restore_open_role_frames(
        frames: tuple[RelationSurfaceFrame, ...],
        facts: tuple[ActiveRelationSurface, ...] = (),
                             ) -> tuple[OpenRoleFrame, ...]:
    """把已存在的槽位生成框架反向用于来源化解析。

    The generated envelope covers only the proposition and role spans. Any
    trained carrier material immediately outside that envelope remains part of
    the graph structure and must be represented as a gap, otherwise an open
    final role would consume a sentence terminator or carrier suffix.
    """
    surfaces = {
        (fact.proposition.stable_key(), fact.source_hash): fact
        for fact in facts
    }
    result = []
    for frame in frames:
        fact = surfaces.get((frame.proposition.stable_key(), frame.source_hash))
        surface = None if fact is None else fact.evidence_surface
        if frame.carrier_prefix is not None or frame.carrier_suffix is not None:
            from pure_integer_ai.experiments.relation_language_carrier import (
                validate_carrier_boundaries, validate_carrier_frame,
            )
            if fact is None:
                if any(item.proposition == frame.proposition for item in facts):
                    raise ValueError("开放框架来源与同命题成员不一致")
                # 输入语言图覆盖全部框架；当前事实视图可按查询资格过滤。
                # 此路径仅恢复语法，不创建或授权缺失事实。
                validate_carrier_boundaries(frame)
            else:
                validate_carrier_frame(fact, frame)
            gaps_text = (frame.carrier_prefix + frame.gaps[0], *frame.gaps[1:-1],
                         frame.gaps[-1] + frame.carrier_suffix)
        elif fact is not None and not surface:
            raise ValueError("无正文开放框架缺少已学习的载体边缘")
        elif surface is not None:
            if not (0 <= frame.envelope_start <= frame.envelope_end <= len(surface)):
                raise ValueError("开放框架来源 envelope 越界")
            gaps_text = (
                surface[:frame.envelope_start] + frame.gaps[0],
                *frame.gaps[1:-1],
                frame.gaps[-1] + surface[frame.envelope_end:],
            )
        else:
            gaps_text = frame.gaps
        gaps = tuple(tuple(ord(value) for value in gap) for gap in gaps_text)
        # 无语言结构约束的空框架没有开放输入资格；原生成框架仍保留。
        if not any(gaps):
            continue
        result.append(OpenRoleFrame(
            frame.proposition.stable_key(), frame.predicate.stable_key(),
            (frame.source_hash,), tuple(role.stable_key() for role in frame.roles),
            gaps, (frame.envelope_start, frame.envelope_end)))
    return tuple(sorted(set(result), key=lambda item: item.stable_key()))
