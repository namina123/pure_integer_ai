"""把公开完整句运行时与 K 盘来源约束广域问答组合为可选对话入口。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3
from typing import Callable, Iterable

from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_contract import BroadQaResult
from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    LearnedTypedObligation,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    LearnedRelationRoleEvidenceModel,
    project_qualified_relation_value,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    LearnedRelationMarkerEvidenceModel,
    project_marker_relation_value,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    LearnedRelationAnswerFrameModel,
    render_relation_answer_frame,
)


@dataclass(frozen=True, slots=True)
class DialogueCitation:
    """用户可见 evidence surface 与单一来源的值记录。"""

    surface: str
    source_title: str | None
    source_url: str | None

    def __post_init__(self) -> None:
        if type(self.surface) is not str or not self.surface.strip():
            raise ValueError("citation surface 必须是非空文本")
        for label, value in (("source title", self.source_title),
                             ("source url", self.source_url)):
            if value is not None and (
                    type(value) is not str or not value.strip()):
                raise ValueError(f"citation {label} 非法")


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """一次对话轮次的纯值摘要；不持有 SQLite、缓存或宿主对象。"""

    ordinal: int
    question: str
    answer: str | None
    display_answer: str | None
    status: str
    source_title: str | None
    source_url: str | None
    turn_key: tuple[int, ...]
    retrieval_question: str | None = None
    citations: tuple[DialogueCitation, ...] = ()


@dataclass(frozen=True, slots=True)
class BroadDialogueState:
    """只保留最近有限轮次，避免无关长期历史使查询线性变慢。"""

    conversation_key: tuple[int, ...]
    next_ordinal: int = 0
    turns: tuple[DialogueTurn, ...] = ()

    def append(self, turn: DialogueTurn, *, keep: int = 8) -> "BroadDialogueState":
        if turn.ordinal != self.next_ordinal:
            raise ValueError("对话 ordinal 不连续")
        if keep <= 0:
            raise ValueError("对话 hot history 必须为正")
        return BroadDialogueState(
            self.conversation_key,
            self.next_ordinal + 1,
            (*self.turns, turn)[-keep:],
        )


def _turn_key(conversation_key: tuple[int, ...], ordinal: int,
              question: str, answer: str | None, status: str) -> tuple[int, ...]:
    """用固定域分离 SHA 投影轮次身份，跨语言可由 raw bytes 重建。"""
    payload = (b"PURE-INTEGER-AI/BROAD-DIALOGUE-TURN/V1" +
               bytes(conversation_key) + ordinal.to_bytes(8, "big") +
               question.encode("utf-8") + b"\0" +
               (answer or "").encode("utf-8") + b"\0" + status.encode("ascii"))
    return tuple(hashlib.sha256(payload).digest())


_FOLLOWUP_REFERENCE_PREFIXES = (
    "它", "他", "她", "其", "该", "这座", "这项", "这个", "这种",
    "上述", "前者", "后者",
)
_FOLLOWUP_REFERENCE_MARKERS = (
    "它", "他", "她", "其", "该条目", "该桥", "该机场", "该项目",
    "这座桥", "这项工程", "这个条目", "这种做法", "上述条目",
)
_REFERENCE_BOUNDARIES = frozenset(" \t，,：:；;。！？!?（(【[")
_REFERENCE_SINGLE_MARKERS = frozenset(("它", "他", "她", "其"))
_REFERENCE_EXTENSIONS = frozenset("们們俩倆")
_EMPTY_LABELED_PARENTHESES_RE = re.compile(
    r"（[^（）()]{1,64}[:：][ \t]*）|\([^()]{1,64}[:：][ \t]*\)")


def _humanize_display_surface(surface: str) -> str:
    """清除证据表面中没有值的标签括号，不改动原始证据载荷。

    Wikipedia 摘录偶尔保留 ``（学名：）`` 这类空模板投影。它不是事实，
    但会破坏人类可读性；只在 display projection 中删除这一种结构，避免
    对来源正文、citation 或 Runtime/Core 数据做不可审计的改写。
    """
    if type(surface) is not str:
        raise TypeError("display surface 必须是字符串")
    cleaned = _EMPTY_LABELED_PARENTHESES_RE.sub("", surface)
    cleaned = cleaned.strip()
    return cleaned or surface.strip()


def _has_followup_reference(surface: str) -> bool:
    """只在词首或明确分句边界识别指代，避免误命中普通词内子串。"""
    if any(surface.startswith(prefix) for prefix in _FOLLOWUP_REFERENCE_PREFIXES):
        return True
    for marker in _FOLLOWUP_REFERENCE_MARKERS:
        start = surface.find(marker)
        while start >= 0:
            end = start + len(marker)
            if marker not in _REFERENCE_SINGLE_MARKERS:
                return True
            extended = (marker in _REFERENCE_SINGLE_MARKERS
                        and end < len(surface)
                        and surface[end] in _REFERENCE_EXTENSIONS)
            if ((start == 0 or surface[start - 1] in _REFERENCE_BOUNDARIES)
                    and not extended):
                return True
            start = surface.find(marker, start + 1)
    return False


def _resolve_source_followup(state: BroadDialogueState,
                             question: str) -> str:
    """把紧接上一来源的指代追问还原为可检索的标题锚定问句。

    这里只做确定性的 discourse 解析：仅在问题以登记的指代前缀开始、且
    最近一条已回答轮次带有来源标题时注入标题。没有来源或不是指代问句时
    保持原文，避免把 hot history 变成隐式事实生成器。
    """
    surface = question.strip()
    if _has_followup_reference(surface):
        # 只接受真正紧邻的上一轮；跳过 UNKNOWN/CLARIFY 会把旧焦点错误
        # 泄漏到新话题，尤其在长会话中会产生看似合理但无来源的回答。
        if state.turns:
            turn = state.turns[-1]
            if turn.status == "ANSWER" and turn.source_title:
                if turn.source_title not in surface:
                    return f"{turn.source_title}，{surface}"
    return question


def answer_broad_dialogue_turn(
        state: BroadDialogueState,
        question: str,
        database: sqlite3.Connection,
        *,
        narrow_answer: Callable[[str], tuple[str, str] | None] | None = None,
        defer_narrow: bool = False,
        surface_consumer: Callable[[str, str, str | None], str | None]
        | None = None,
        runtime_material_answer: Callable[
            [str], tuple[str, str | None, str | None] | None] | None = None,
        runtime_material_response: Callable[
            [str], tuple[object, ...] | None]
        | None = None,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None = None,
        learned_typed_obligation: LearnedTypedObligation | None = None,
        learned_relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        learned_relation_role_evidence_model: LearnedRelationRoleEvidenceModel
        | None = None,
        learned_relation_marker_evidence_model: LearnedRelationMarkerEvidenceModel
        | None = None,
        learned_relation_answer_frame_model: LearnedRelationAnswerFrameModel
        | None = None,
        ) -> tuple[BroadDialogueState, DialogueTurn]:
    """先尝试窄域消费者，再查询来源约束数据库，最后可选消费回答表面。

    ``surface_consumer`` 只接收已经产生的用户可见 ANSWER 表面、状态和来源
    标题；它不能改变完整证据链、来源身份或 UNKNOWN/CLARIFY 结果。返回空值
    表示保持原表面，适合把训练后的结构组织接到真实回答侧而不替代事实检索。
    """
    if type(state) is not BroadDialogueState or type(question) is not str:
        raise TypeError("dialogue state/question 类型错误")
    if type(defer_narrow) is not bool:
        raise TypeError("defer_narrow 必须是严格 bool")
    if not question.strip():
        raise ValueError("question 不能为空")
    answer = None
    status = "UNKNOWN"
    source_title = None
    source_url = None
    display_answer = None
    retrieval_question = None
    citations: tuple[DialogueCitation, ...] = ()
    runtime_material_decided = False
    # 显式 Runtime response 是资格化资料的权威决定；它必须先于窄域/广域
    # fallback，避免同一问题被未资格化或冲突资料绕过。
    if runtime_material_response is not None:
        material_response = runtime_material_response(question)
        if material_response is not None:
            if (not isinstance(material_response, tuple)
                    or len(material_response) not in (4, 5)
                    or material_response[0] not in {"ANSWER", "UNKNOWN", "CLARIFY"}
                    or (material_response[1] is not None
                        and type(material_response[1]) is not str)
                    or (material_response[2] is not None
                        and type(material_response[2]) is not str)
                    or (material_response[3] is not None
                        and type(material_response[3]) is not str)):
                raise TypeError("runtime material response 返回值非法")
            status, answer, source_title, source_url = material_response[:4]
            if len(material_response) == 5:
                raw_citations = material_response[4]
                if (not isinstance(raw_citations, tuple)
                        or any(not isinstance(item, DialogueCitation)
                               for item in raw_citations)):
                    raise TypeError("runtime material citations 返回值非法")
                citations = raw_citations
            if status == "ANSWER" and (answer is None or not answer.strip()):
                raise TypeError("runtime material ANSWER 必须携带 answer")
            if status != "ANSWER" and answer is not None:
                raise TypeError("runtime material 非 ANSWER 不得携带 answer")
            if status != "ANSWER" and citations:
                raise TypeError("runtime material 非 ANSWER 不得携带 citations")
            if citations and answer != "\n".join(
                    item.surface for item in citations):
                raise TypeError("runtime material answer/citation surface 漂移")
            display_answer = answer
            retrieval_question = question
            runtime_material_decided = True
    if (answer is None and not runtime_material_decided
            and not defer_narrow and narrow_answer is not None):
        narrow = narrow_answer(question)
        if narrow is not None:
            answer, status = narrow
            display_answer = answer
    if (answer is None and not runtime_material_decided
            and runtime_material_answer is not None):
        material = runtime_material_answer(question)
        if material is not None:
            if (not isinstance(material, tuple) or len(material) != 3
                    or type(material[0]) is not str or not material[0].strip()
                    or (material[1] is not None and type(material[1]) is not str)
                    or (material[2] is not None and type(material[2]) is not str)):
                raise TypeError("runtime material answer provider 返回值非法")
            answer, source_title, source_url = material
            status = "ANSWER"
            display_answer = answer
            retrieval_question = question
    if answer is None and not runtime_material_decided:
        retrieval_question = _resolve_source_followup(state, question)
        if (learned_evidence_term_weights is None
                and learned_typed_obligation is None
                and learned_relation_evidence_model is None):
            # Preserve the narrow callable contract used by existing embedders;
            # the learned path is opt-in and must not alter legacy callers.
            result: BroadQaResult = query_broad_qa(
                database, retrieval_question)
        else:
            result = query_broad_qa(
                database, retrieval_question,
                learned_evidence_term_weights=learned_evidence_term_weights,
                learned_typed_obligation=learned_typed_obligation,
                learned_relation_evidence_model=learned_relation_evidence_model)
        status = result.status
        answer = result.answer
        # 保留 BroadQaResult.answer 的完整证据链；终端只展示主证据窗口，
        # 使完整句回答可读且不把类别/邻接段落误当成一个答案。
        evidence_chain = getattr(result, "evidence_chain", ())
        display_answer = (
            _humanize_display_surface(evidence_chain[0].selected_text)
            if result.status == "ANSWER" and evidence_chain
            else result.answer
        )
        if (result.status == "ANSWER" and display_answer
                and learned_relation_evidence_model is not None):
            projected = None
            if learned_relation_role_evidence_model is not None:
                projected = project_qualified_relation_value(
                    learned_relation_evidence_model,
                    learned_relation_role_evidence_model,
                    retrieval_question,
                    display_answer,
                    anchor_text=result.title,
                )
            if projected is None and learned_relation_marker_evidence_model is not None:
                projected = project_marker_relation_value(
                    learned_relation_evidence_model,
                    learned_relation_marker_evidence_model,
                    retrieval_question,
                    display_answer,
                    anchor_text=result.title,
                )
            if projected is not None:
                display_answer = projected[0]
                if (learned_relation_answer_frame_model is not None
                        and result.title):
                    family = learned_relation_evidence_model.relation_family(
                        retrieval_question)
                    sentence = render_relation_answer_frame(
                        learned_relation_answer_frame_model, family,
                        result.title, projected[0])
                    if sentence is not None:
                        display_answer = sentence
        if result.title is not None:
            source_title = result.title
            source_url = result.source_url
    # Fast/deferred mode queries the broad index first.  The narrow runtime is
    # initialized lazily only when broad retrieval cannot answer; this removes
    # several seconds of snapshot construction from ordinary broad questions
    # without changing the strict mode ordering above.
    if (defer_narrow and answer is None and status == "UNKNOWN"
            and not runtime_material_decided and narrow_answer is not None):
        narrow = narrow_answer(question)
        if narrow is not None:
            answer, status = narrow
            display_answer = answer
    if (surface_consumer is not None and status == "ANSWER"
            and display_answer):
        consumed = surface_consumer(display_answer, status, source_title)
        if consumed is not None and consumed.strip():
            display_answer = consumed
    turn = DialogueTurn(
        state.next_ordinal, question, answer, display_answer,
        status, source_title, source_url,
        _turn_key(state.conversation_key, state.next_ordinal,
                  question, answer, status),
        retrieval_question,
        citations,
    )
    return state.append(turn), turn


__all__ = [
    "BroadDialogueState", "DialogueCitation", "DialogueTurn",
    "answer_broad_dialogue_turn",
]
