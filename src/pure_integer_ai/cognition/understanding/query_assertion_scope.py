"""用来源化训练框架绑定陈述作用域；不从表面词猜测立场或事实真值。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRoleFrame, match_open_roles, pack_record, unpack_record,
)

if TYPE_CHECKING:
    from pure_integer_ai.cognition.understanding.query_input_structure import QueryInputStructure


SCOPE_NEGATED = 1
SCOPE_POSSIBLE = 2
SCOPE_QUOTED = 3
SCOPE_FRAME_VERSION = 92380
SCOPE_BINDING_VERSION = 92381
SCOPE_QUERY_PROOF = 92382
SCOPE_MEMORY_COMPETITION = 92387


def _key(key: tuple[int, ...]) -> None:
    """限定为可由规范整数记录恢复的非空有界键。"""
    if (type(key) is not tuple or not key
            or any(type(value) is not int or not 0 <= value < 2**63 for value in key)):
        raise ValueError("scope identity 必须是非空有界整数元组")


@dataclass(frozen=True, slots=True)
class TrainedAssertionScopeFrame:
    """一个许可课程学到的作用域算子与完整开放语法，不包含事实答案。"""

    syntax: OpenRoleFrame
    mode: int
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.syntax, OpenRoleFrame)
                or type(self.mode) is not int
                or self.mode not in (SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED)):
            raise ValueError("作用域框架缺少已注册算子或开放角色语法")
        _key(self.source_identity)
        _key(self.license_identity)
        _key(self.syntax.stable_key())

    def stable_key(self) -> tuple[int, ...]:
        """身份完整保留语法、算子、来源和许可，不以摘要替代本体。"""
        return pack_record(SCOPE_FRAME_VERSION, (self.mode,), self.syntax.stable_key(),
                           self.source_identity, self.license_identity)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> TrainedAssertionScopeFrame:
        """从跨语言整数协议恢复完整作用域课程记录。"""
        fields = unpack_record(key, SCOPE_FRAME_VERSION)
        if len(fields) != 4 or len(fields[0]) != 1:
            raise ValueError("scope frame 字段不完整")
        return cls(OpenRoleFrame.from_stable_key(fields[1]), fields[0][0],
                   fields[2], fields[3])


@dataclass(frozen=True, slots=True)
class InputAssertionScopeBinding:
    """当前输入对完整课程框架和一个内层关系的精确绑定证明。"""

    frame_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    relation_key: tuple[int, ...]
    slot_span_refs: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        frame = TrainedAssertionScopeFrame.from_stable_key(self.frame_key)
        _key(self.source_ref)
        _key(self.relation_key)
        if (type(self.slot_span_refs) is not tuple
                or len(self.slot_span_refs) != len(frame.syntax.roles)
                or len(set(self.slot_span_refs)) != len(self.slot_span_refs)):
            raise ValueError("scope binding 缺少唯一有序角色跨度")
        for ref in self.slot_span_refs:
            _key(ref)

    @property
    def mode(self) -> int:
        """算子由完整训练记录恢复，不接受额外可漂移的模式字段。"""
        return TrainedAssertionScopeFrame.from_stable_key(self.frame_key).mode

    def stable_key(self) -> tuple[int, ...]:
        """保留当前来源、课程、内层关系及所有角色身份。"""
        return pack_record(SCOPE_BINDING_VERSION, self.frame_key, self.source_ref,
                           self.relation_key, *self.slot_span_refs)


def match_assertion_scope_frames(
        frames: tuple[TrainedAssertionScopeFrame, ...], structure: QueryInputStructure,
        ) -> tuple[InputAssertionScopeBinding, ...]:
    """只接受完整输入及同一来源关系的精确角色，竞争解释保持未授权。"""
    from pure_integer_ai.cognition.understanding.query_input_structure import (
        PROJECTION_FILLER, STRUCTURE_CLOSED,
    )

    if (not frames or not structure.tokens or structure.relation_question_bindings
            or structure.discourse_compositions or structure.discourse_gaps
            or structure.operation_request_bindings
            or structure.pending_relation_composition_bindings):
        return ()
    # 同一语法可有多个来源或模式，不能用 dict 覆盖而隐去竞争解释。
    by_syntax = {}
    for frame in frames:
        by_syntax.setdefault(frame.syntax.stable_key(), []).append(frame)
    parses = match_open_roles(tuple(
        OpenRoleFrame.from_stable_key(key) for key in sorted(by_syntax)),
        structure.token_values, structure.source_ref)
    spans = {(span.start, span.end, span.values): span.ref_key for span in structure.spans}
    result = set()
    for parsed in parses:
        refs = tuple(spans.get((slot.start, slot.end, slot.values)) for slot in parsed.bindings)
        if any(ref is None for ref in refs):
            continue
        for frame in by_syntax[parsed.frame.stable_key()]:
            for relation in structure.relation_candidates:
                if (relation.state != STRUCTURE_CLOSED or relation.conflict_count
                        or relation.required_open or relation.required_count != relation.support_count
                        or relation.predicate_key != frame.syntax.predicate
                        or relation.role_coverage != len(frame.syntax.roles)):
                    continue
                for role, ref in zip(frame.syntax.roles, refs):
                    targets = {item.candidate_key for item in structure.semantic_candidates
                        if item.projection_kind == PROJECTION_FILLER
                        and item.proposition_key == relation.proposition_key
                        and item.predicate_key == relation.predicate_key
                        and item.source_ref == relation.source_ref and item.role_key == role
                        and item.span_ref == ref and ref in relation.span_refs}
                    if len(targets) != 1:
                        break
                else:
                    result.add(InputAssertionScopeBinding(
                        frame.stable_key(), structure.source_ref, relation.stable_key(), refs))
    # 初版没有嵌套和竞争作用域裁决；必须完整保留为未解析而不是选第一项。
    return tuple(result) if len(result) == 1 else ()


def attach_assertion_scope_frames(frames, structure):
    """生产与 Memory 共用同一绑定，拒绝外部预装的未授权作用域。"""
    matches = match_assertion_scope_frames(frames, structure)
    if structure.assertion_scope_bindings == matches:
        return structure
    return replace(structure, assertion_scope_bindings=matches)


def scope_relation_stance(structure, candidate_key, *, relation_candidate_key):
    """仅精确内层关系可取得来源化反驳立场，其余解释一律 UNKNOWN。"""
    from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE, EVIDENCE_UNKNOWN

    if len(structure.assertion_scope_bindings) != 1:
        return EVIDENCE_UNKNOWN
    binding = structure.assertion_scope_bindings[0]
    if binding.mode == SCOPE_NEGATED:
        for relation in structure.relation_candidates:
            if (relation.stable_key() == binding.relation_key
                    and relation_candidate_key(relation) == candidate_key):
                return EVIDENCE_REFUTE
    return EVIDENCE_UNKNOWN


def assertion_scope_competition(structure, competition_key):
    """来源化说话行为与普通内容身份分离，历史立场不变成无作用域真值。"""
    if len(structure.assertion_scope_bindings) != 1:
        return competition_key
    return pack_record(SCOPE_MEMORY_COMPETITION, competition_key,
                       structure.assertion_scope_bindings[0].stable_key())
