"""把一次已回读问答压成不含 surface 文本的 run-local 会话状态。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


_DIGEST_SIZE = 32
_CONTEXT_VERSION = 2


class ConversationContextError(ValueError):
    """会话上下文缺少实际回读、键漂移或 revision 链不闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空、只含严格整数的内容键。"""
    if not isinstance(value, tuple) or not value:
        raise ConversationContextError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ConversationContextError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键增加边界。"""
    return len(value), *value


def _digest(value: tuple[int, ...]) -> tuple[int, ...]:
    """从当前 snapshot 形成固定 32 字节整数摘要。"""
    fingerprint = integer_tuple_fingerprint(
        value, domain="conversation.context.snapshot.v1")
    return tuple(fingerprint[2:])


def _digest_key(value: tuple[int, ...], *, empty: bool) -> tuple[int, ...]:
    """核验 revision 0 空指针或后续 revision 的摘要。"""
    if empty:
        if value != ():
            raise ConversationContextError(
                "conversation context 初始 revision 不得带 previous digest")
        return value
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise ConversationContextError("conversation context previous digest 非法")
    return value


def _strict_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验已提交 snapshot/read 的固定摘要。"""
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise ConversationContextError(f"{label} 非法")
    return value


def _sorted_keys(
        values: tuple[tuple[int, ...], ...], *, label: str,
        ) -> tuple[tuple[int, ...], ...]:
    """规范化排序并拒绝重复 candidate/discourse key。"""
    if not isinstance(values, tuple):
        raise ConversationContextError(f"{label} 必须是 tuple")
    for value in values:
        _strict_key(value, label=label)
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise ConversationContextError(f"{label} 不得重复")
    return ordered


def _ordered_keys(
        values: tuple[tuple[int, ...], ...], *, label: str,
        ) -> tuple[tuple[int, ...], ...]:
    """核验非空整数键并保留调用方提供的语义顺序。"""
    if not isinstance(values, tuple):
        raise ConversationContextError(f"{label} 必须是 tuple")
    for value in values:
        _strict_key(value, label=label)
    if len(set(values)) != len(values):
        raise ConversationContextError(f"{label} 不得重复")
    return values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationTurnState:
    """一次实际问答的 typed 状态；不含问题或回答的原始文字。"""

    turn_ordinal: int
    request_key: tuple[int, ...]
    target_key: tuple[int, ...]
    query_key: tuple[int, ...]
    planning_key: tuple[int, ...]
    response_stance: ObjectIdentity
    selected_candidate_keys: tuple[tuple[int, ...], ...]
    cited_sources: tuple[SourceRef, ...]
    discourse_sentence_keys: tuple[tuple[int, ...], ...]
    parser_revision: tuple[int, ...]
    readback_key: tuple[int, ...]
    context_read: "ConversationContextRead | None" = None

    def __post_init__(self) -> None:
        if type(self.turn_ordinal) is not int or self.turn_ordinal < 0:
            raise ConversationContextError("conversation turn ordinal 非法")
        for label, value in (
                ("request_key", self.request_key),
                ("target_key", self.target_key),
                ("query_key", self.query_key),
                ("planning_key", self.planning_key),
                ("parser_revision", self.parser_revision),
                ("readback_key", self.readback_key)):
            _strict_key(value, label=f"conversation turn {label}")
        if not isinstance(self.response_stance, ObjectIdentity):
            raise TypeError("conversation turn response stance 类型错误")
        self_candidates = _sorted_keys(
            self.selected_candidate_keys,
            label="conversation turn selected candidate key",
        )
        if (not isinstance(self.cited_sources, tuple)
                or any(not isinstance(item, SourceRef)
                       for item in self.cited_sources)):
            raise TypeError("conversation turn cited sources 类型错误")
        citations = tuple(sorted(set(self.cited_sources), key=SourceRef.stable_key))
        if len(citations) != len(self.cited_sources):
            raise ConversationContextError(
                "conversation turn cited sources 不得重复")
        sentence_keys = _ordered_keys(
            self.discourse_sentence_keys,
            label="conversation turn discourse sentence key",
        )
        if (self.context_read is not None
                and not isinstance(self.context_read, ConversationContextRead)):
            raise TypeError("conversation turn context read 类型错误")
        object.__setattr__(self, "selected_candidate_keys", self_candidates)
        object.__setattr__(self, "cited_sources", citations)

    def typed_key(self) -> tuple[int, ...]:
        """返回本轮 typed payload，不递归展开之前的 context read。"""
        result = [_CONTEXT_VERSION, self.turn_ordinal]
        for value in (
                self.request_key,
                self.target_key,
                self.query_key,
                self.planning_key,
                self.parser_revision,
                self.readback_key):
            result.extend(_packed(value))
        result.extend(_packed(self.response_stance.stable_key()))
        result.append(len(self.selected_candidate_keys))
        for value in self.selected_candidate_keys:
            result.extend(_packed(value))
        result.append(len(self.cited_sources))
        for source in self.cited_sources:
            result.extend(_packed(source.stable_key()))
        result.append(len(self.discourse_sentence_keys))
        for value in self.discourse_sentence_keys:
            result.extend(_packed(value))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回完整整数状态键，不包含任何 surface payload。"""
        result = list(self.typed_key())
        result.append(0 if self.context_read is None else 1)
        if self.context_read is not None:
            result.extend(_packed(self.context_read.stable_key()))
        return tuple(result)

    @classmethod
    def from_run(
            cls,
            run: QuestionAnswerRun,
            turn_ordinal: int,
            *,
            context_read: "ConversationContextRead | None" = None,
            ) -> "ConversationTurnState":
        """只从完整 QuestionAnswerRun 的 typed/readback 段建立状态。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("conversation turn 需要 QuestionAnswerRun")
        if (not run.complete or run.query is None
                or run.planning_request is None
                or run.selection is None or run.generation is None
                or run.postcheck is None
                or run.postcheck.parsed.observation is None):
            raise ConversationContextError(
                "conversation turn 必须来自完整 actual parser/G-04 readback")
        syntax = run.generation.surface.preview.request.structure.syntax
        sentence_keys = tuple(sentence.stable_key() for sentence in syntax.sentences)
        if not sentence_keys:
            raise ConversationContextError(
                "conversation turn 缺少实际 discourse sentence")
        observation = run.postcheck.parsed.observation
        return cls(
            turn_ordinal,
            run.request.stable_key(),
            run.request.target.stable_key(),
            run.query.stable_key(),
            run.planning_request.stable_key(),
            run.selection.stance,
            run.selection.selected_candidate_keys,
            observation.cited_sources,
            sentence_keys,
            run.postcheck.parsed.trace,
            observation.stable_key(),
            context_read,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationContextRead:
    """一次显式、有限的会话状态读取；不携带任何 surface 文本。"""

    conversation_key: tuple[int, ...]
    revision: int
    digest: tuple[int, ...]
    turns: tuple[ConversationTurnState, ...]

    def __post_init__(self) -> None:
        _strict_key(self.conversation_key, label="conversation read key")
        if type(self.revision) is not int or self.revision < 0:
            raise ConversationContextError("conversation read revision 非法")
        _strict_digest(self.digest, label="conversation read digest")
        if (not isinstance(self.turns, tuple)
                or any(not isinstance(item, ConversationTurnState)
                       for item in self.turns)):
            raise TypeError("conversation read turns 类型错误")
        expected = tuple(range(self.revision - len(self.turns), self.revision))
        if tuple(item.turn_ordinal for item in self.turns) != expected:
            raise ConversationContextError(
                "conversation read 必须是当前 revision 的连续尾部")

    def stable_key(self) -> tuple[int, ...]:
        """返回读取边界、摘要和 typed turns 的整数键。"""
        result = [_CONTEXT_VERSION]
        result.extend(_packed(self.conversation_key))
        result.append(self.revision)
        result.extend(_packed(self.digest))
        result.append(len(self.turns))
        for turn in self.turns:
            result.extend(_packed(integer_tuple_fingerprint(
                turn.typed_key(),
                domain="conversation.context.read.turn.v1",
            )))
        return tuple(result)

    def request_trace_suffix(self) -> tuple[int, ...]:
        """返回把本次 typed read 绑定到下一问题的固定整数摘要。"""
        return integer_tuple_fingerprint(
            self.stable_key(),
            domain="conversation.context.request.read.v1",
        )

    def bind_request(self, request: QuestionRequest) -> QuestionRequest:
        """把本次显式读取绑定到下一问题 trace，不加入 surface payload。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("conversation read 只能绑定 QuestionRequest")
        suffix = self.request_trace_suffix()
        if request.trace[-len(suffix):] == suffix:
            raise ConversationContextError(
                "conversation read 不得重复绑定同一 request")
        return replace(request, trace=(*request.trace, *suffix))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationContextState:
    """append-only run-local 会话快照，只保留可显式查询的 typed turn。"""

    conversation_key: tuple[int, ...]
    revision: int
    previous_digest: tuple[int, ...]
    turns: tuple[ConversationTurnState, ...] = ()

    def __post_init__(self) -> None:
        _strict_key(self.conversation_key, label="conversation key")
        if type(self.revision) is not int or self.revision < 0:
            raise ConversationContextError("conversation context revision 非法")
        _digest_key(self.previous_digest, empty=self.revision == 0)
        if (not isinstance(self.turns, tuple)
                or any(not isinstance(item, ConversationTurnState)
                       for item in self.turns)):
            raise TypeError("conversation context turns 类型错误")
        ordinals = tuple(item.turn_ordinal for item in self.turns)
        if self.revision != len(self.turns):
            raise ConversationContextError(
                "conversation context revision 必须等于 turn 数")
        if ordinals != tuple(range(len(ordinals))):
            raise ConversationContextError(
                "conversation context turn ordinal 必须连续")
        for turn in self.turns:
            if turn.context_read is None:
                raise ConversationContextError(
                    "conversation context turn 缺少显式 context read")
            if (turn.context_read.conversation_key != self.conversation_key
                    or turn.context_read.revision != turn.turn_ordinal):
                raise ConversationContextError(
                    "conversation context turn 读取了其他 revision")
        if self.turns and self.previous_digest != self.turns[-1].context_read.digest:
            raise ConversationContextError(
                "conversation context previous digest 与末轮读取不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回快照的整数键和所有 turn typed 状态。"""
        result = [_CONTEXT_VERSION]
        result.extend(_packed(self.conversation_key))
        result.append(self.revision)
        result.extend(_packed(self.previous_digest))
        result.append(len(self.turns))
        for turn in self.turns:
            result.extend(_packed(turn.stable_key()))
        return tuple(result)

    def digest(self) -> tuple[int, ...]:
        """返回当前快照摘要，供下一 revision 形成前驱链。"""
        return _digest(self.stable_key())

    def append(self, run: QuestionAnswerRun) -> "ConversationContextState":
        """提交无前序状态的首轮 actual run。"""
        if self.revision != 0:
            raise ConversationContextError(
                "后续 conversation turn 必须显式绑定 context read")
        return self._append(run, self.read(0))

    def append_consumed(
            self,
            run: QuestionAnswerRun,
            context_read: ConversationContextRead,
            ) -> "ConversationContextState":
        """提交已绑定当前显式 typed read 的后续 actual run。"""
        if not isinstance(context_read, ConversationContextRead):
            raise TypeError("conversation append 缺少 typed context read")
        expected = self.read(len(context_read.turns))
        if context_read != expected:
            raise ConversationContextError(
                "conversation append 使用了过期或其他会话的 context read")
        suffix = context_read.request_trace_suffix()
        if (not isinstance(run, QuestionAnswerRun)
                or run.request.trace[-len(suffix):] != suffix):
            raise ConversationContextError(
                "conversation append 的 run 未绑定同次 context read")
        return self._append(run, context_read)

    def _append(
            self,
            run: QuestionAnswerRun,
            context_read: ConversationContextRead,
            ) -> "ConversationContextState":
        """在 read/bind 合同核验后形成下一不可变 snapshot。"""
        turn = ConversationTurnState.from_run(
            run, len(self.turns), context_read=context_read)
        return ConversationContextState(
            self.conversation_key,
            self.revision + 1,
            self.digest(),
            (*self.turns, turn),
        )

    def read(self, limit: int) -> "ConversationContextRead":
        """只返回调用方明确请求的 typed 会话尾部和当前摘要。"""
        if type(limit) is not int or limit < 0:
            raise ConversationContextError("conversation read limit 非法")
        return ConversationContextRead(
            self.conversation_key,
            self.revision,
            self.digest(),
            self.visible_turns(limit),
        )

    def visible_turns(self, limit: int) -> tuple[ConversationTurnState, ...]:
        """只返回调用方显式请求的最近 typed turns，不返回 surface 文本。"""
        if type(limit) is not int or limit < 0:
            raise ConversationContextError("conversation visible limit 非法")
        return self.turns[-limit:] if limit else ()


def start_conversation_context(
        conversation_key: tuple[int, ...],
        ) -> ConversationContextState:
    """创建 revision 0 的空会话上下文。"""
    return ConversationContextState(conversation_key, 0, ())


__all__ = [
    "ConversationContextError",
    "ConversationContextRead",
    "ConversationContextState",
    "ConversationTurnState",
    "start_conversation_context",
]
