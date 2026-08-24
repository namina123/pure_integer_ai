"""DLG-RAW-01：公开 Frame catalog 的纯读词汇和构式入口。

这个阶段只把已接受的 DLG-RAW-00 integer record 映射为完整的
``QuestionRequest``。它不执行问答、不读取 terminal history、不推测词义，且
所有结果都声明空 ``state_delta``。回答执行属于独立的 RAW-02 runtime bridge。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    PublicFrame,
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    ConversationRawIntake,
)


RAW_LEXICAL_INGRESS_RECORD_V1 = 1

_RESULT_CODES = frozenset({
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
})


# object-model: exception; interop=DLG-RAW-01
class ConversationRawLexicalIngressError(ValueError):
    """DLG-RAW-01 输入、catalog 或显式 context 的闭合关系发生漂移。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验调用方传入的稳定整数 operation/context key。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise ConversationRawLexicalIngressError(f"{label} 必须是非空严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长整数段加入 canonical result record。"""
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=DLG-RAW-01
@dataclass(frozen=True, slots=True)
class ConversationRawLexicalIngressResult:
    """一条 RAW-01 的只读理解结果，成功时才携带完整 QuestionRequest。"""

    result_code: int
    intake: ConversationRawIntake
    catalog: PublicFrameCatalog
    matched_frame_count: int = 0
    frame: PublicFrame | None = None
    representations: tuple[ObjectIdentity, ...] = ()
    language_atoms: tuple[ObjectIdentity, ...] = ()
    request: QuestionRequest | None = None
    context_read: ConversationContextRead | None = None
    state_delta: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验各失败层没有越级携带语义请求或持久副作用。"""
        if type(self.result_code) is not int:
            raise ConversationRawLexicalIngressError("RAW-01 result code 必须是严格整数")
        if not isinstance(self.intake, ConversationRawIntake):
            raise TypeError("RAW-01 缺少 ConversationRawIntake")
        if not isinstance(self.catalog, PublicFrameCatalog):
            raise TypeError("RAW-01 缺少 PublicFrameCatalog")
        if type(self.matched_frame_count) is not int or self.matched_frame_count < 0:
            raise ConversationRawLexicalIngressError("RAW-01 frame count 非法")
        if (not isinstance(self.representations, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.representations)
                or not isinstance(self.language_atoms, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.language_atoms)):
            raise TypeError("RAW-01 lexical identities 类型错误")
        if self.context_read is not None and not isinstance(
                self.context_read, ConversationContextRead):
            raise TypeError("RAW-01 context read 类型错误")
        if (not isinstance(self.state_delta, tuple)
                or any(type(item) is not int for item in self.state_delta)
                or self.state_delta):
            raise ConversationRawLexicalIngressError("RAW-01 必须保持零 state delta")
        has_frame = self.frame is not None
        if has_frame and not isinstance(self.frame, PublicFrame):
            raise TypeError("RAW-01 frame 类型错误")
        has_request = self.request is not None
        if has_request and not isinstance(self.request, QuestionRequest):
            raise TypeError("RAW-01 request 类型错误")
        if not self.intake.accepted:
            if (self.result_code != self.intake.result_code or self.matched_frame_count
                    or has_frame or self.representations or self.language_atoms
                    or has_request or self.context_read is not None):
                raise ConversationRawLexicalIngressError("RAW-01 raw 拒绝 record 非法")
            return
        if self.result_code not in _RESULT_CODES:
            raise ConversationRawLexicalIngressError("RAW-01 result code 未注册")
        if self.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
            if (self.matched_frame_count != 0 or has_frame
                    or self.representations or self.language_atoms
                    or has_request or self.context_read is not None):
                raise ConversationRawLexicalIngressError("RAW-01 lexical miss record 非法")
            return
        if self.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
            if (self.matched_frame_count < 2 or has_frame
                    or self.representations or self.language_atoms
                    or has_request or self.context_read is not None):
                raise ConversationRawLexicalIngressError("RAW-01 lexical ambiguity record 非法")
            return
        if self.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT:
            if (self.matched_frame_count < 1 or has_frame
                    or self.representations or self.language_atoms
                    or has_request or self.context_read is not None):
                raise ConversationRawLexicalIngressError(
                    "RAW-01 source conflict record 非法")
            return
        if (self.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS
                and not has_frame):
            # 组合器可在确定唯一输入候选后、物化其 frame 前发现完整性漂移。
            # 这仍是可审计的 RAW-01 code 9：不得虚构 route/request/context，且
            # 与一般 lexical miss/ambiguity 保持严格分账。
            if (self.matched_frame_count < 1 or self.representations
                    or self.language_atoms or has_request
                    or self.context_read is not None):
                raise ConversationRawLexicalIngressError(
                    "pre-frame construction miss record 非法")
            return
        if not has_frame or self.matched_frame_count != 1:
            raise ConversationRawLexicalIngressError("RAW-01 selected frame 不闭合")
        expected_representations = tuple(item.representation for item in self.frame.routes)
        expected_atoms = tuple(item.atom for item in self.frame.routes)
        if (self.representations != expected_representations
                or self.language_atoms != expected_atoms):
            raise ConversationRawLexicalIngressError("RAW-01 lexical route 本体漂移")
        if self.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS:
            if has_request:
                raise ConversationRawLexicalIngressError("construction miss 不得产生 request")
            return
        if self.result_code == DLG_RAW_REJECT_CONTEXT:
            if has_request:
                raise ConversationRawLexicalIngressError("context miss 不得产生 request")
            return
        if (self.result_code != DLG_RAW_ACCEPT or not has_request
                or self.request.target != self.frame.question.target):
            raise ConversationRawLexicalIngressError("RAW-01 accept request 不闭合")
        if self.frame.context_requirement == PUBLIC_FRAME_CONTEXT_NONE:
            if self.context_read is not None:
                raise ConversationRawLexicalIngressError("NONE frame 不得携带 context read")
        elif self.frame.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR:
            if self.context_read is None:
                raise ConversationRawLexicalIngressError("TARGET_ANCHOR frame 缺 context read")
        else:
            raise ConversationRawLexicalIngressError("RAW-01 frame context tag 未注册")

    @property
    def accepted(self) -> bool:
        """仅完整 request 已形成、但尚未调用回答 runtime 时返回真。"""
        return self.result_code == DLG_RAW_ACCEPT

    def canonical_record(self) -> tuple[int, ...]:
        """导出不依赖宿主对象或加速索引的完整 integer result record。"""
        result = [
            RAW_LEXICAL_INGRESS_RECORD_V1,
            self.result_code,
            self.matched_frame_count,
        ]
        for value in (
                self.intake.canonical_record(),
                self.catalog.canonical_record(),
                (() if self.frame is None else self.frame.canonical_record()),
                tuple(item for identity in self.representations
                      for item in identity.stable_key()),
                tuple(item for identity in self.language_atoms
                      for item in identity.stable_key()),
                (() if self.request is None else self.request.stable_key()),
                (() if self.context_read is None
                 else self.context_read.stable_key()),
                self.state_delta):
            _pack(result, value)
        return tuple(result)


def _frame_payload(frame: PublicFrame) -> tuple[tuple[ObjectIdentity, ...], tuple[ObjectIdentity, ...]]:
    """从已经过目录验证的有序 routes 提取 representation 和 semantic atom。"""
    return (
        tuple(item.representation for item in frame.routes),
        tuple(item.atom for item in frame.routes),
    )


def _context_matches(frame: PublicFrame, context_read: ConversationContextRead | None) -> bool:
    """只允许显式尾轮 target anchor 作为省略输入的语义上下文。"""
    if frame.context_requirement == PUBLIC_FRAME_CONTEXT_NONE:
        return context_read is None
    if frame.context_requirement != PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR:
        raise ConversationRawLexicalIngressError("RAW-01 frame context tag 未注册")
    return bool(context_read and context_read.turns
                and context_read.turns[-1].target_key == frame.context_target_key)


def ingress_raw_lexical_frame(
        intake: ConversationRawIntake,
        catalog: PublicFrameCatalog,
        occurrence_key: tuple[int, ...],
        *,
        context_read: ConversationContextRead | None = None,
        ) -> ConversationRawLexicalIngressResult:
    """将一条 accepted raw record 映射到唯一公开 Frame，绝不执行自然语言 fallback。"""
    if not isinstance(intake, ConversationRawIntake):
        raise TypeError("RAW-01 intake 类型错误")
    if not isinstance(catalog, PublicFrameCatalog):
        raise TypeError("RAW-01 catalog 类型错误")
    occurrence = _strict_key(occurrence_key, label="RAW-01 occurrence key")
    if context_read is not None and not isinstance(context_read, ConversationContextRead):
        raise TypeError("RAW-01 context read 类型错误")
    if not intake.accepted:
        return ConversationRawLexicalIngressResult(
            intake.result_code, intake, catalog)
    matches = catalog.matching_frames(intake.unicode_scalars)
    if not matches:
        return ConversationRawLexicalIngressResult(
            DLG_RAW_REJECT_LEXICAL_MISS, intake, catalog)
    if len(matches) != 1:
        return ConversationRawLexicalIngressResult(
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS, intake, catalog,
            matched_frame_count=len(matches))
    frame = matches[0]
    representations, atoms = _frame_payload(frame)
    if not catalog.has_construction(frame.construction):
        return ConversationRawLexicalIngressResult(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, intake, catalog, 1, frame,
            representations, atoms)
    if not _context_matches(frame, context_read):
        return ConversationRawLexicalIngressResult(
            DLG_RAW_REJECT_CONTEXT, intake, catalog, 1, frame,
            representations, atoms, context_read=context_read)
    request = frame.question.request_for(occurrence)
    if context_read is not None:
        request = context_read.bind_request(request)
    return ConversationRawLexicalIngressResult(
        DLG_RAW_ACCEPT, intake, catalog, 1, frame, representations, atoms,
        request, context_read)


__all__ = [
    "RAW_LEXICAL_INGRESS_RECORD_V1",
    "ConversationRawLexicalIngressError",
    "ConversationRawLexicalIngressResult",
    "ingress_raw_lexical_frame",
]
