"""完整请求的未落地作用域；语法、当前变量和事实资格分别保留。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRoleFrame, match_open_roles, pack_record, unpack_record,
)
from pure_integer_ai.cognition.understanding.query_operation_requests import (
    OperationRequestFrame, READER, SEQUENCE, _key, match_operation_request_frames,
)


PENDING_REQUEST_SCOPE = 92410
PENDING_REQUEST_READING = 92411


@dataclass(frozen=True, slots=True)
class PendingOperationRequestScope:
    """当前全输入及每个 reader 的精确区间/全部投影，不填入未知对象或事实。"""

    frame_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    input_values: tuple[int, ...]
    readings: tuple[tuple[int, int, int, tuple[tuple[int, ...], ...]], ...]

    def __post_init__(self):
        frame = OperationRequestFrame.from_stable_key(self.frame_key)
        _key(self.source_ref)
        _key(self.input_values)
        if (type(self.readings) is not tuple
                or tuple(row[0] for row in self.readings) != tuple(row[0] for row in frame.readers)):
            raise ValueError("pending request loses its complete ordered reader contract")
        cursor = 0
        for gap, (variable, start, end, projections) in zip(frame.gaps, self.readings):
            if (type(variable) is not int or type(start) is not int or type(end) is not int
                    or start < cursor or not start < end <= len(self.input_values)
                    or self.input_values[cursor:start] != gap
                    or type(projections) is not tuple or len(projections) > 64
                    or projections != tuple(sorted(set(projections)))):
                raise ValueError("pending request changes its exact input interval or provenance")
            for projection in projections:
                _key(projection)
            cursor = end
        if self.input_values[cursor:] != frame.gaps[-1]:
            raise ValueError("pending request does not cover the whole current input")

    def stable_key(self):
        """完整来源、原输入和所有缺槽区间进入可逆整数身份。"""
        return pack_record(PENDING_REQUEST_SCOPE, self.frame_key, self.source_ref, self.input_values,
            pack_record(SEQUENCE, *(pack_record(PENDING_REQUEST_READING, (variable, start, end),
                pack_record(SEQUENCE, *projections)) for variable, start, end, projections in self.readings)))

    @classmethod
    def from_stable_key(cls, key):
        """恢复全部字段并核验规范编码，不能只凭操作 id 或摘要授权缺槽。"""
        fields = unpack_record(key, PENDING_REQUEST_SCOPE)
        if len(fields) != 4:
            raise ValueError("pending request scope is truncated")
        readings = []
        for row in unpack_record(fields[3], SEQUENCE):
            parts = unpack_record(row, PENDING_REQUEST_READING)
            if len(parts) != 2 or len(parts[0]) != 3:
                raise ValueError("pending request reader is truncated")
            readings.append((*parts[0], unpack_record(parts[1], SEQUENCE)))
        result = cls(fields[0], fields[1], fields[2], tuple(readings))
        if result.stable_key() != key:
            raise ValueError("pending request scope is not canonical")
        return result


def match_pending_operation_request_scopes(frames, structure, *, max_steps=65536):
    """复用开放角色核匹配完整训练语法，保留所有未获唯一完整绑定的解析。"""
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("pending request matching budget must be positive")
    if not structure.tokens or not frames or structure.operation_request_bindings:
        return ()
    if match_operation_request_frames(frames, structure):
        return ()
    syntaxes = tuple(OpenRoleFrame(frame.stable_key(), frame.operation_key, frame.source.stable_key(),
        tuple(pack_record(READER, (variable,), predicate, role, kinds)
              for variable, predicate, role, kinds in frame.readers), frame.gaps,
        (0, sum(map(len, frame.gaps)) + len(frame.readers))) for frame in frames)
    spans = {span.ref_key: span for span in structure.spans}
    projection_index = {}
    for item in structure.semantic_candidates:
        if item.projection_kind == 1:
            span = spans[item.span_ref]
            projection_index.setdefault((span.start, span.end, item.predicate_key,
                item.role_key, item.object_kind), set()).add(item.stable_key())
    scopes = set()
    for parsed in match_open_roles(syntaxes, structure.token_values, structure.source_ref, max_steps=max_steps):
        frame = OperationRequestFrame.from_stable_key(parsed.frame.proposition)
        readings = []
        for (variable, predicate, role, kinds), span in zip(frame.readers, parsed.bindings):
            projections = set()
            for kind in kinds:
                projections.update(projection_index.get((span.start, span.end, predicate, role, kind), ()))
            readings.append((variable, span.start, span.end, tuple(sorted(projections))))
        scopes.add(PendingOperationRequestScope(frame.stable_key(), structure.source_ref,
                                                structure.token_values, tuple(readings)))
    return tuple(sorted(scopes, key=lambda item: item.stable_key()))


def pending_operation_request_input_view(frames, structure):
    """完整未落地请求只激活全部reader投影，语法间隔不成为独立事实竞争。"""
    if (structure.operation_request_bindings or structure.relation_question_bindings
            or structure.assertion_scope_bindings or structure.pending_relation_composition_bindings
            or structure.discourse_compositions or structure.discourse_gaps):
        return structure
    scopes = match_pending_operation_request_scopes(frames, structure)
    if not scopes:
        return structure
    members = {key for scope in scopes for _variable, _start, _end, keys in scope.readings for key in keys}
    return replace(structure,
        semantic_candidates=tuple(item for item in structure.semantic_candidates if item.stable_key() in members),
        relation_candidates=(), open_relations=(), concept_links=(), concept_routes=())
