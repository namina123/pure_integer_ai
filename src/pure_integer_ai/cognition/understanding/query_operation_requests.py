"""图内请求变量 reader：完整整数输入匹配，不预先要求存在答案命题。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


FRAME = 92320
READER = 92321
BINDING = 92322
READING = 92323
SEQUENCE = 92324


def _key(value, *, empty=False):
    """所有协议字段只接受规范非负严格整数。"""
    if type(value) is not tuple or (not value and not empty) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("operation request needs a complete integer key")
    return value


@dataclass(frozen=True, slots=True)
class OperationRequestFrame:
    """许可 train 输入形状；不保存 exemplar filler、claim 或答案。"""

    operation_key: tuple[int, ...]
    source: SourceRef
    source_sha256: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    # input occurrence order: variable, predicate, semantic role, allowed kinds
    readers: tuple[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
    gaps: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        for key in (self.operation_key, self.source_identity, self.license_identity):
            _key(key)
        digest = _key(self.source_sha256)
        if (not isinstance(self.source, SourceRef) or len(digest) != 32 or any(value > 255 for value in digest)
                or self.source.source_kind != FRAME
                or self.source.source_id != ((int.from_bytes(bytes(digest[:8]), "big") & ((1 << 63) - 1)) or 1)):
            raise ValueError("request frame lacks its exact physical source digest")
        if (type(self.readers) is not tuple or not 1 <= len(self.readers) <= 8
                or len({row[0] for row in self.readers}) != len(self.readers)
                or type(self.gaps) is not tuple or len(self.gaps) != len(self.readers) + 1
                or not any(self.gaps)):
            raise ValueError("request frame has ambiguous variables or incomplete syntax")
        for variable, predicate, role, kinds in self.readers:
            if type(variable) is not int or variable <= 0 or kinds != tuple(sorted(set(kinds))):
                raise ValueError("request reader variable or kinds are not canonical")
            for key in (predicate, role, kinds):
                _key(key)
        for gap in self.gaps:
            _key(gap, empty=True)
            if len(gap) > 256 or any(value > 0x10FFFF or 0xD800 <= value <= 0xDFFF for value in gap):
                raise ValueError("request grammar exceeds its Unicode scalar budget")

    def stable_key(self):
        """完整来源、reader 与 grammar 均为一等可恢复整数记录。"""
        return pack_record(FRAME, self.operation_key, self.source.stable_key(), self.source_sha256,
            self.source_identity, self.license_identity,
            pack_record(SEQUENCE, *(pack_record(READER, (variable,), predicate, role, kinds)
                                   for variable, predicate, role, kinds in self.readers)),
            pack_record(SEQUENCE, *self.gaps))

    @classmethod
    def from_stable_key(cls, key):
        """严格冷恢复完整 frame，不补缺失字段或推断默认语言。"""
        fields = unpack_record(key, FRAME)
        if len(fields) != 7:
            raise ValueError("request frame is truncated")
        readers = []
        for row in unpack_record(fields[5], SEQUENCE):
            parts = unpack_record(row, READER)
            if len(parts) != 4 or len(parts[0]) != 1:
                raise ValueError("request reader is truncated")
            readers.append((parts[0][0], *parts[1:]))
        result = cls(fields[0], SourceRef.from_stable_key(fields[1]), fields[2], fields[3], fields[4],
                     tuple(readers), unpack_record(fields[6], SEQUENCE))
        if result.stable_key() != key:
            raise ValueError("request frame is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class InputOperationRequestBinding:
    """当前输入各变量及全部图投影来源；不是事实 support。"""

    frame_key: tuple[int, ...]
    operation_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    # input order: variable, full filler, span ref, all matching projection keys
    readings: tuple[tuple[int, tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]], ...]

    def __post_init__(self):
        frame = OperationRequestFrame.from_stable_key(self.frame_key)
        _key(self.source_ref)
        if (frame.operation_key != self.operation_key or type(self.readings) is not tuple
                or tuple(row[0] for row in self.readings) != tuple(row[0] for row in frame.readers)):
            raise ValueError("request binding differs from its complete frame")
        for _variable, target, span, projections in self.readings:
            _key(target); _key(span)
            if (type(projections) is not tuple or not projections or len(projections) > 64
                    or projections != tuple(sorted(set(projections)))):
                raise ValueError("request binding lacks canonical projection provenance")
            for projection in projections:
                _key(projection)

    def stable_key(self):
        """冻结全部当前 variable/span/alias-reader 关系而非 operation 标签。"""
        return pack_record(BINDING, self.frame_key, self.operation_key, self.source_ref,
            pack_record(SEQUENCE, *(pack_record(READING, (variable,), target, span, pack_record(SEQUENCE, *projections))
                                   for variable, target, span, projections in self.readings)))

    @classmethod
    def from_stable_key(cls, key):
        """冷恢复当前请求的完整变量和来源，不丢失输入 occurrence 顺序。"""
        fields = unpack_record(key, BINDING)
        if len(fields) != 4:
            raise ValueError("request input binding is truncated")
        readings = []
        for row in unpack_record(fields[3], SEQUENCE):
            parts = unpack_record(row, READING)
            if len(parts) != 4 or len(parts[0]) != 1:
                raise ValueError("request input reader is truncated")
            readings.append((parts[0][0], parts[1], parts[2], unpack_record(parts[3], SEQUENCE)))
        result = cls(fields[0], fields[1], fields[2], tuple(readings))
        if result.stable_key() != key:
            raise ValueError("request input binding is not canonical")
        return result


def match_operation_request_frames(frames, structure, *, max_combinations=4096):
    """仅有唯一完整跨度/identity 组合时绑定；同名歧义保留 UNKNOWN。"""
    if type(max_combinations) is not int or max_combinations <= 0:
        raise ValueError("request matching budget must be positive")
    if not structure.tokens or structure.relation_question_bindings or structure.pending_relation_composition_bindings:
        return ()
    spans = {span.ref_key: span for span in structure.spans}
    values, matches, searched = structure.token_values, set(), 0
    for frame in frames:
        options = []
        for variable, predicate, role, kinds in frame.readers:
            grouped = {}
            for item in structure.semantic_candidates:
                if (item.projection_kind == 1 and item.object_kind in kinds
                        and item.predicate_key == predicate and item.role_key == role):
                    grouped.setdefault((item.span_ref, item.candidate_key), set()).add(item.stable_key())
            options.append(tuple((variable, target, span, tuple(sorted(projections)))
                for (span, target), projections in sorted(grouped.items())))
        if any(not choice for choice in options):
            continue
        for readings in product(*options):
            searched += 1
            if searched > max_combinations:
                raise ValueError("complete operation request matching budget exceeded")
            positions = tuple(spans[row[2]] for row in readings)
            cursor, valid = 0, True
            for gap, span in zip(frame.gaps, positions):
                if span.start < cursor or values[cursor:span.start] != gap:
                    valid = False
                    break
                cursor = span.end
            if valid and values[cursor:] == frame.gaps[-1]:
                matches.add(InputOperationRequestBinding(frame.stable_key(), frame.operation_key,
                                                        structure.source_ref, readings))
    return tuple(matches) if len(matches) == 1 else ()


def attach_operation_request(frames, structure):
    """输入结构只接当前冷图重算得到的 request，拒绝调用方注入变量。"""
    bindings = match_operation_request_frames(frames, structure)
    if structure.operation_request_bindings and structure.operation_request_bindings != bindings:
        raise ValueError("operation request binding does not match the current trained graph input")
    if not bindings:
        return structure
    operations = tuple(sorted(set(structure.operation_keys) | {bindings[0].operation_key}))
    if len(operations) != 1:
        raise ValueError("operation request has competing learned input operations")
    return replace(structure, operation_keys=operations, operation_request_bindings=bindings)


def operation_request_input_view(structure):
    """A full request activates exactly its readers, without guessing any answer.

    Every original token/span and every alias provenance member of each current
    reader is retained. Statement/open-role hypotheses inside the licensed
    request grammar are not separate input assertions or anaphora requests.
    """
    if not structure.operation_request_bindings:
        return structure
    request, = structure.operation_request_bindings
    frame = OperationRequestFrame.from_stable_key(request.frame_key)
    if match_operation_request_frames((frame,), structure) != (request,):
        raise ValueError("request scope requires one exact complete current frame match")
    members = {key for _variable, _target, _span, keys in request.readings for key in keys}
    if not members <= {item.stable_key() for item in structure.semantic_candidates}:
        raise ValueError("request scope lacks complete current reader provenance")
    return replace(structure,
        semantic_candidates=tuple(item for item in structure.semantic_candidates if item.stable_key() in members),
        relation_candidates=(), open_relations=(), concept_links=(), concept_routes=(),
        discourse_compositions=(), discourse_gaps=())
