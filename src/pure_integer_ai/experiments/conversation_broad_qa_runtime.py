"""把公开完整句运行时与 K 盘来源约束广域问答组合为可选对话入口。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3
from typing import Callable, Iterable

from pure_integer_ai.experiments.ph2_broad_qa_query import (
    BroadQaQueryCache,
    SurfaceVariantProvider,
    has_exact_broad_qa_title,
    has_explicit_non_real_constraint,
    query_broad_qa,
)
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
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)


@dataclass(frozen=True, slots=True)
class DialogueCitation:
    """用户可见 evidence surface 与单一来源的值记录。"""

    surface: str
    source_title: str | None
    source_url: str | None
    license_id: str | None = None
    attribution: str | None = None
    source_ref: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.surface) is not str or not self.surface.strip():
            raise ValueError("citation surface 必须是非空文本")
        for label, value in (("source title", self.source_title),
                             ("source url", self.source_url),
                             ("license id", self.license_id),
                             ("attribution", self.attribution)):
            if value is not None and (
                    type(value) is not str or not value.strip()):
                raise ValueError(f"citation {label} 非法")
        if self.source_ref is not None and (
                not isinstance(self.source_ref, tuple)
                or len(self.source_ref) != 11
                or any(type(item) is not int or item < 0
                       for item in self.source_ref)):
            raise ValueError("citation source_ref 必须是十一非负整数 tuple")


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


def _resolve_source_followup(state: BroadDialogueState,
                             question: str,
                             reference_resolver: Callable[[str, DialogueTurn], bool]
                             | None = None) -> str:
    """用调用方学习到的指代解析器恢复来源追问。

    通用对话核心不登记任何语言、脚本或代词表。解析器必须来自当前语言
    分支/图中的已学习关系；缺少解析器时保持原问句，避免把来源标题或
    hot history 当作隐式事实生成器。
    """
    surface = question.strip()
    if reference_resolver is None or not state.turns:
        return question
    if not callable(reference_resolver):
        raise TypeError("source followup resolver 必须是可调用对象")
    # 只接受真正紧邻的上一轮；跳过 UNKNOWN/CLARIFY 会把旧焦点错误
    # 泄漏到新话题，尤其在长会话中会产生看似合理但无来源的回答。
    turn = state.turns[-1]
    if (turn.status == "ANSWER" and turn.source_title
            and reference_resolver(surface, turn)
            and turn.source_title not in surface):
        return f"{turn.source_title}，{surface}"
    return question


def build_index_evidence_source_followup_resolver(
        database: sqlite3.Connection,
        *,
        query_cache: BroadQaQueryCache | None = None,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None = None,
        learned_typed_obligation: LearnedTypedObligation | None = None,
        learned_relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        fast_path: bool = False,
        ) -> Callable[[str, DialogueTurn], bool]:
    """建立基于来源索引证据的紧邻焦点解析器。

    解析器不包含语言、脚本或代词表。它只验证一个可复现的检索事实：把
    上一轮已经确认的来源标题作为上下文前缀后，当前问题是否得到同一来源
    的 ANSWER。若没有该证据，返回 ``False``，上层继续使用原问题并保持
    UNKNOWN/CLARIFY 边界。前缀使用 ASCII 空格作为跨语言 transport 分隔符，
    不改变任何来源正文或答案载荷。
    """
    if not isinstance(database, sqlite3.Connection):
        raise TypeError("source followup database 类型错误")
    if query_cache is not None and not isinstance(query_cache, BroadQaQueryCache):
        raise TypeError("source followup query_cache 类型错误")
    if type(fast_path) is not bool:
        raise TypeError("source followup fast_path 必须是严格 bool")
    if (surface_variant_provider is not None
            and not callable(surface_variant_provider)):
        raise TypeError("source followup surface_variant_provider 必须可调用")

    def resolve(question: str, turn: DialogueTurn) -> bool:
        if (type(question) is not str or not question.strip()
                or not isinstance(turn, DialogueTurn)
                or turn.status != "ANSWER"
                or not turn.source_title
                or turn.source_title in question
                or has_exact_broad_qa_title(database, question)):
            return False
        candidate = f"{turn.source_title} {question.strip()}"
        result = query_broad_qa(
            database,
            candidate,
            learned_evidence_term_weights=learned_evidence_term_weights,
            learned_typed_obligation=learned_typed_obligation,
            learned_relation_evidence_model=learned_relation_evidence_model,
            surface_variant_provider=surface_variant_provider,
            fast_path=fast_path,
            query_cache=query_cache,
        )
        return bool(
            result.status == "ANSWER"
            and result.title == turn.source_title
        )

    return resolve


def _validated_source_passage_response(
        provider: Callable[[str], tuple[object, ...] | None],
        question: str,
        ) -> tuple[str, str, str, str, tuple[DialogueCitation, ...]] | None:
    """调用来源段消费者并核验完整引用合同，拒绝无来源回答进入对话。"""
    source_response = provider(question)
    if source_response is None:
        return None
    if (not isinstance(source_response, tuple)
            or len(source_response) != 5
            or source_response[0] != "ANSWER"
            or type(source_response[1]) is not str
            or not source_response[1].strip()
            or type(source_response[2]) is not str
            or not source_response[2].strip()
            or type(source_response[3]) is not str
            or not source_response[3].startswith("https://")
            or not isinstance(source_response[4], tuple)
            or not source_response[4]
            or any(not isinstance(item, DialogueCitation)
                   for item in source_response[4])):
        raise TypeError("source passage response 返回值非法")
    status, answer, source_title, source_url, citations = source_response
    if answer != "\n".join(item.surface for item in citations):
        raise TypeError("source passage answer/citations 漂移")
    return status, answer, source_title, source_url, citations


def answer_broad_dialogue_turn(
        state: BroadDialogueState,
        question: str,
        database: sqlite3.Connection,
        *,
        narrow_answer: Callable[[str], tuple[str, str] | None] | None = None,
        defer_narrow: bool = False,
        query_cache: BroadQaQueryCache | None = None,
        fast_path: bool = False,
        prefer_source_passage: bool = False,
        prefer_learned_dialogue: bool = False,
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
        learned_dialogue_answer: Callable[[str], str | None] | None = None,
        learned_dialogue_clarify_answer: Callable[[str], str | None]
        | None = None,
        memory_recall_response: Callable[[str], str | None] | None = None,
        source_passage_response: Callable[
            [str], tuple[object, ...] | None] | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        source_followup_resolver: Callable[[str, DialogueTurn], bool]
        | None = None,
        ) -> tuple[BroadDialogueState, DialogueTurn]:
    """先尝试窄域消费者，再查询来源约束数据库，最后可选消费回答表面。

    ``surface_consumer`` 只接收已经产生的用户可见 ANSWER 表面、状态和来源
    标题；它不能改变完整证据链、来源身份或 UNKNOWN/CLARIFY 结果。返回空值
    表示保持原表面，适合把训练后的结构组织接到真实回答侧而不替代事实检索。
    ``source_followup_resolver`` 是可选的语言/图侧指代判定；核心不内置任何
    代词词表，未提供时不会擅自把上一来源标题注入当前问题。
    """
    if type(state) is not BroadDialogueState or type(question) is not str:
        raise TypeError("dialogue state/question 类型错误")
    if type(defer_narrow) is not bool:
        raise TypeError("defer_narrow 必须是严格 bool")
    if query_cache is not None and not isinstance(query_cache, BroadQaQueryCache):
        raise TypeError("query_cache 类型错误")
    if type(fast_path) is not bool:
        raise TypeError("fast_path 必须是严格 bool")
    if type(prefer_source_passage) is not bool:
        raise TypeError("prefer_source_passage 必须是严格 bool")
    if type(prefer_learned_dialogue) is not bool:
        raise TypeError("prefer_learned_dialogue 必须是严格 bool")
    if (surface_variant_provider is not None
            and not callable(surface_variant_provider)):
        raise TypeError("surface_variant_provider 必须是可调用对象")
    if (source_followup_resolver is not None
            and not callable(source_followup_resolver)):
        raise TypeError("source_followup_resolver 必须是可调用对象")
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
    source_passage_queried = False
    learned_dialogue_queried = False
    question_answer_kinds = (
        load_broad_qa_question_slots().answer_kinds(
            question, surface_variant_provider))
    learned_dialogue_eligible = (
        not has_explicit_non_real_constraint(question)
        and not question_answer_kinds
    )
    source_passage_precedes_broad = (
        prefer_source_passage
        and (
            not question_answer_kinds
            or (
                fast_path
                and not has_exact_broad_qa_title(database, question)
            )
        )
    )
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
    if (answer is None and not runtime_material_decided
            and prefer_learned_dialogue and learned_dialogue_eligible
            and learned_dialogue_answer is not None):
        learned_dialogue_queried = True
        learned = learned_dialogue_answer(question)
        if learned is not None:
            if type(learned) is not str or not learned.strip():
                raise TypeError("learned dialogue answer 必须是非空文本或 None")
            answer = learned.strip()
            display_answer = answer
            status = "ANSWER"
            retrieval_question = question
    if (answer is None and not runtime_material_decided
            and source_passage_precedes_broad
            and source_passage_response is not None):
        source_passage_queried = True
        source_response = _validated_source_passage_response(
            source_passage_response, question)
        if source_response is not None:
            status, answer, source_title, source_url, citations = source_response
            display_answer = answer
            retrieval_question = question
    bounded_dialogue_miss = (
        answer is None and not runtime_material_decided and fast_path
        and learned_dialogue_eligible and learned_dialogue_queried
        and source_passage_queried)
    if answer is None and not runtime_material_decided and not bounded_dialogue_miss:
        retrieval_question = _resolve_source_followup(
            state, question, source_followup_resolver)
        if (learned_evidence_term_weights is None
                and learned_typed_obligation is None
                and learned_relation_evidence_model is None
                and query_cache is None
                and surface_variant_provider is None
                and not fast_path):
            # Preserve the narrow callable contract used by existing embedders;
            # the learned path is opt-in and must not alter legacy callers.
            result: BroadQaResult = query_broad_qa(
                database, retrieval_question)
        else:
            query_kwargs = {
                "learned_evidence_term_weights": learned_evidence_term_weights,
                "learned_typed_obligation": learned_typed_obligation,
                "learned_relation_evidence_model": learned_relation_evidence_model,
                "surface_variant_provider": surface_variant_provider,
                "fast_path": fast_path,
            }
            if query_cache is not None:
                query_kwargs["query_cache"] = query_cache
            result = query_broad_qa(
                database, retrieval_question, **query_kwargs)
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
    elif bounded_dialogue_miss:
        retrieval_question = question
    # 训练侧来源 passage 是广域检索的证据补充，而不是固定 QA fallback。
    # 仅在既有检索未回答时查询；它必须返回完整来源引用，且只能把结果提升为
    # ANSWER，不能用低置信候选覆盖 UNKNOWN/CLARIFY 门。
    if (answer is None and status == "UNKNOWN"
            and not runtime_material_decided
            and source_passage_response is not None
            and not source_passage_queried):
        source_response = _validated_source_passage_response(
            source_passage_response, question)
        if source_response is not None:
            status, answer, source_title, source_url, citations = source_response
            display_answer = answer
            retrieval_question = question
    # Runtime/Core memory is a read-only fallback after factual and source
    # routes have declined.  The callback owns language-independent recall
    # ranking and may return only text derived from an already persisted turn
    # or from a learned dialogue model; this layer does not inspect words,
    # scripts, or internal status labels.
    if (answer is None and status == "UNKNOWN"
            and not runtime_material_decided
            and memory_recall_response is not None):
        recalled = memory_recall_response(question)
        if recalled is not None:
            if type(recalled) is not str or not recalled.strip():
                raise TypeError(
                    "memory recall response 必须是非空文本或 None")
            answer = recalled.strip()
            display_answer = answer
            status = "ANSWER"
            retrieval_question = question
    # Fast/deferred mode queries the broad index first.  The narrow runtime is
    # initialized lazily only when broad retrieval cannot answer; this removes
    # several seconds of snapshot construction from ordinary broad questions
    # without changing the strict mode ordering above.
    if (defer_narrow and answer is None and status == "UNKNOWN"
            and not runtime_material_decided and narrow_answer is not None
            and not bounded_dialogue_miss
            and not (fast_path
                     and has_explicit_non_real_constraint(question))):
        narrow = narrow_answer(question)
        if narrow is not None:
            answer, status = narrow
            display_answer = answer
    # 检索型 CLARIFY 只允许更严格的高置信对话片段接管。Runtime 资料作出的
    # CLARIFY 永不覆盖；调用方应为此入口设置比普通 UNKNOWN fallback 更高的
    # 相似度门槛。
    if (answer is None and status == "CLARIFY"
            and not runtime_material_decided
            and learned_dialogue_eligible
            and learned_dialogue_clarify_answer is not None):
        learned = learned_dialogue_clarify_answer(question)
        if learned is not None:
            if type(learned) is not str or not learned.strip():
                raise TypeError(
                    "learned dialogue clarify answer 必须是非空文本或 None")
            answer = learned.strip()
            display_answer = answer
            status = "ANSWER"
            retrieval_question = question
    # 人工对话模型是最低优先级表层消费者：只有 Runtime 资料、来源约束广域
    # 查询和窄域模型均明确 UNKNOWN 后才可生成。它不携带事实来源，也不能覆盖
    # CLARIFY、冲突或已有 ANSWER。
    if (answer is None and status == "UNKNOWN"
            and not runtime_material_decided
            and learned_dialogue_eligible
            and learned_dialogue_answer is not None
            and not learned_dialogue_queried):
        learned = learned_dialogue_answer(question)
        if learned is not None:
            if type(learned) is not str or not learned.strip():
                raise TypeError("learned dialogue answer 必须是非空文本或 None")
            answer = learned.strip()
            display_answer = answer
            status = "ANSWER"
            retrieval_question = question
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
    "build_index_evidence_source_followup_resolver",
]
