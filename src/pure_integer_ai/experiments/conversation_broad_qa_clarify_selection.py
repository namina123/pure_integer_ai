"""广域问答 CLARIFY 后的来源候选选择（opt-in）。

该层只保存不可变候选身份和一轮 pending。首次歧义由真实 broad-QA
检索产生，选择输入必须再次经过同一来源查询，并命中 pending 候选的
page identity；否则 pending 立即失效。它不向默认对话入口注入答案。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_multiturn_boundary import (
    _sha256,
    _turn_digest,
    _turn_dict,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    BroadQaRetrievalCandidate,
    query_broad_qa,
    retrieve_broad_qa_candidates,
)


CLARIFY_SELECTION_PROTOCOL_V1 = 1
CLARIFY_SELECTION_PASS = "PASS"
CLARIFY_SELECTION_NE = "NE"
_IDENTITY_DOMAIN = b"PURE-INTEGER-AI/BROAD-QA/CLARIFY-SELECTION/V1"


class ClarifySelectionError(ValueError):
    """候选 pending、重检索或选择状态不闭合。"""


def _u8(value: tuple[int, ...], label: str) -> tuple[int, ...]:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise ClarifySelectionError(f"{label} 必须是非空 raw-u8 tuple")
    return value


def _candidate_identity(candidate: BroadQaRetrievalCandidate) -> tuple[int, ...]:
    """用候选的来源/段落字段形成可跨语言重算的 raw-u8 identity。"""
    if not isinstance(candidate, BroadQaRetrievalCandidate):
        raise TypeError("candidate 类型错误")
    payload = bytearray(_IDENTITY_DOMAIN)
    unsigned_values = (
        candidate.doc_id, candidate.page_id, candidate.revision_id,
        candidate.passage_id, candidate.raw_start, candidate.raw_end,
        candidate.matched_term_count)
    for value in unsigned_values:
        if type(value) is not int or value < 0 or value >= 1 << 64:
            raise ClarifySelectionError("candidate 无符号整数字段非法")
        payload.extend(value.to_bytes(8, "big", signed=False))
    if (type(candidate.score) is not int
            or not -(1 << 63) <= candidate.score < 1 << 63):
        raise ClarifySelectionError("candidate 有符号整数字段非法")
    payload.extend(candidate.score.to_bytes(8, "big", signed=True))
    title = candidate.title.encode("utf-8")
    payload.extend(len(title).to_bytes(4, "big"))
    payload.extend(title)
    return tuple(hashlib.sha256(bytes(payload)).digest())


@dataclass(frozen=True, slots=True)
class BroadClarifyCandidate:
    """来源候选的纯值投影，不持有 SQLite 或宿主对象。"""

    doc_id: int
    page_id: int
    revision_id: int
    title: str
    candidate_identity_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        if (type(self.doc_id) is not int or self.doc_id < 0
                or type(self.page_id) is not int or self.page_id < 0
                or type(self.revision_id) is not int or self.revision_id < 0
                or type(self.title) is not str or not self.title.strip()
                or len(_u8(self.candidate_identity_u8,
                           "candidate identity")) != 32):
            raise ClarifySelectionError("候选值投影非法")

    @classmethod
    def from_retrieval(
            cls, candidate: BroadQaRetrievalCandidate) -> "BroadClarifyCandidate":
        return cls(candidate.doc_id, candidate.page_id, candidate.revision_id,
                   candidate.title, _candidate_identity(candidate))


@dataclass(frozen=True, slots=True)
class BroadClarifyPending:
    """只对下一轮选择有效的来源候选集合。"""

    opened_question: str
    candidates: tuple[BroadClarifyCandidate, ...]

    def __post_init__(self) -> None:
        if (type(self.opened_question) is not str
                or not self.opened_question.strip()
                or type(self.candidates) is not tuple
                or len(self.candidates) < 2):
            raise ClarifySelectionError("pending 候选集合非法")
        identities = tuple(item.candidate_identity_u8 for item in self.candidates)
        if len(set(identities)) != len(identities):
            raise ClarifySelectionError("pending 候选 identity 重复")

    @property
    def page_ids(self) -> frozenset[int]:
        return frozenset(item.page_id for item in self.candidates)


@dataclass(frozen=True, slots=True)
class BroadClarifySession:
    """对话热状态与一轮候选 pending 的结构体包装。"""

    state: BroadDialogueState
    pending: BroadClarifyPending | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not BroadDialogueState:
            raise TypeError("clarify session state 类型错误")
        if (self.pending is not None
                and type(self.pending) is not BroadClarifyPending):
            raise TypeError("clarify session pending 类型错误")


def _retrieve(
        connection: sqlite3.Connection,
        question: str,
        *,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None,
        learned_typed_obligation: Any | None,
        learned_relation_evidence_model: Any | None,
        ) -> tuple[tuple[BroadQaRetrievalCandidate, ...], Any]:
    return retrieve_broad_qa_candidates(
        connection, question,
        learned_evidence_term_weights=learned_evidence_term_weights,
        learned_typed_obligation=learned_typed_obligation,
        learned_relation_evidence_model=learned_relation_evidence_model,
    )


def answer_broad_clarify_selection_turn(
        session: BroadClarifySession,
        question: str,
        database: sqlite3.Connection,
        *,
        narrow_answer: Any | None = None,
        surface_consumer: Any | None = None,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None = None,
        learned_typed_obligation: Any | None = None,
        learned_relation_evidence_model: Any | None = None,
        learned_relation_role_evidence_model: Any | None = None,
        learned_relation_marker_evidence_model: Any | None = None,
        learned_relation_answer_frame_model: Any | None = None,
        ) -> tuple[BroadClarifySession, DialogueTurn, bool]:
    """运行一轮 broad 对话并返回是否提交了候选选择。"""
    if type(session) is not BroadClarifySession or type(question) is not str:
        raise TypeError("clarify selection 输入类型错误")
    if not isinstance(database, sqlite3.Connection):
        raise TypeError("clarify selection database 类型错误")
    selected = False
    pending = session.pending
    if pending is not None:
        # 选择必须先重检索；不接受仅凭 title、位置或字符串相等的外部断言。
        result = query_broad_qa(
            database, question,
            learned_evidence_term_weights=learned_evidence_term_weights,
            learned_typed_obligation=learned_typed_obligation,
            learned_relation_evidence_model=learned_relation_evidence_model,
        )
        selected = (result.status == "ANSWER"
                    and result.page_id in pending.page_ids)
    _, turn = answer_broad_dialogue_turn(
        session.state, question, database,
        narrow_answer=narrow_answer,
        surface_consumer=surface_consumer,
        learned_evidence_term_weights=learned_evidence_term_weights,
        learned_typed_obligation=learned_typed_obligation,
        learned_relation_evidence_model=learned_relation_evidence_model,
        learned_relation_role_evidence_model=learned_relation_role_evidence_model,
        learned_relation_marker_evidence_model=learned_relation_marker_evidence_model,
        learned_relation_answer_frame_model=learned_relation_answer_frame_model,
    )
    selected = selected and turn.status == "ANSWER"
    next_pending = None
    if pending is None and turn.status == "CLARIFY":
        candidates, trace = _retrieve(
            database, question,
            learned_evidence_term_weights=learned_evidence_term_weights,
            learned_typed_obligation=learned_typed_obligation,
            learned_relation_evidence_model=learned_relation_evidence_model,
        )
        if trace.candidate_document_count > 1 and len(candidates) >= 2:
            unique: dict[tuple[int, int], BroadClarifyCandidate] = {}
            for candidate in candidates:
                value = BroadClarifyCandidate.from_retrieval(candidate)
                unique.setdefault((value.doc_id, value.page_id), value)
            if len(unique) >= 2:
                next_pending = BroadClarifyPending(
                    question, tuple(unique.values()))
    elif pending is not None and not selected:
        next_pending = None
    return BroadClarifySession(session.state.append(turn), next_pending), turn, selected


def candidate_selection_question(
        pending: BroadClarifyPending, *, ordinal: int = 0) -> tuple[str, BroadClarifyCandidate]:
    """从 pending 候选本身构造一个完整来源锚定问句，仅用于 opt-in probe。"""
    if type(pending) is not BroadClarifyPending:
        raise TypeError("pending 类型错误")
    if type(ordinal) is not int or not 0 <= ordinal < len(pending.candidates):
        raise ValueError("candidate ordinal 越界")
    candidate = pending.candidates[ordinal]
    return f"{candidate.title}，{pending.opened_question}", candidate


def _path_kwargs(models: tuple[Any, ...] | None) -> dict[str, Any]:
    if models is None:
        return {}
    (_training_pack, evidence_model, obligation_model, relation_model,
     role_model, marker_model, frame_model, _observation, consumer, _used) = models
    return {
        "surface_consumer": consumer,
        "learned_evidence_term_weights": evidence_model.weights,
        "learned_typed_obligation": obligation_model,
        "learned_relation_evidence_model": relation_model,
        "learned_relation_role_evidence_model": role_model,
        "learned_relation_marker_evidence_model": marker_model,
        "learned_relation_answer_frame_model": frame_model,
    }


def _run_selection_path(
        database: Path,
        questions: tuple[str, ...],
        selection_question: str | None = None,
        *,
        models: tuple[Any, ...] | None = None,
        ) -> tuple[tuple[DialogueTurn, ...], BroadClarifySession,
                   tuple[bool, ...], BroadClarifyPending | None]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        session = BroadClarifySession(BroadDialogueState((6, 8, 1)))
        turns: list[DialogueTurn] = []
        selected: list[bool] = []
        for ordinal, question in enumerate(questions):
            if ordinal == 2 and selection_question is not None:
                question = selection_question
            session, turn, was_selected = answer_broad_clarify_selection_turn(
                session, question, connection, **_path_kwargs(models))
            turns.append(turn)
            selected.append(was_selected)
        return tuple(turns), session, tuple(selected), session.pending
    finally:
        connection.close()


def build_clarify_selection_report(
        *, project_root: str | Path,
        pack_dir: str | Path,
        database_path: str | Path,
        training_run_root: str | Path,
        item_id: str,
        ) -> dict[str, object]:
    """运行真实三轮：ANSWER -> CLARIFY(candidate offer) -> selection ANSWER。"""
    from pure_integer_ai.experiments.conversation_broad_qa_scale_audit import (
        _verify_pack,
    )
    from pure_integer_ai.experiments.conversation_multiturn_clarify_boundary import (
        _ambiguity_question,
        _models,
    )

    root = Path(project_root).resolve()
    pack = Path(pack_dir).resolve()
    database = Path(database_path).resolve()
    training_root = Path(training_run_root).resolve()
    if any(path.drive.upper() != "K:" for path in (pack, database, training_root)):
        raise ClarifySelectionError("pack、database、training run 必须位于 K 盘")
    if not database.is_file() or not training_root.is_dir():
        raise ClarifySelectionError("K 盘输入缺失")
    _manifest, questions, _labels, _dimensions = _verify_pack(pack)
    selected = tuple(item for item in questions
                     if str(item.get("item_id")) == item_id)
    if len(selected) != 1:
        raise ClarifySelectionError("item_id 必须唯一命中冻结 pack")
    real_question = str(selected[0]["question"])

    # 只用真实首轮来源标题生成歧义输入；标题/候选都来自当前检索结果。
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        first_result = query_broad_qa(connection, real_question)
    finally:
        connection.close()
    if first_result.status != "ANSWER" or not first_result.title:
        raise ClarifySelectionError("真实首轮未形成来源 ANSWER")
    ambiguity = _ambiguity_question(real_question, first_result.title)
    baseline_questions = (real_question, ambiguity, "")

    _initial_turns, initial_session, _initial_selected, _ = _run_selection_path(
        database, (real_question, ambiguity))
    if initial_session.pending is None:
        raise ClarifySelectionError("baseline 未形成候选 pending")
    selection_question, selected_candidate = candidate_selection_question(
        initial_session.pending)
    baseline_turns, baseline_session, baseline_selected, _ = _run_selection_path(
        database, baseline_questions, selection_question)
    models = _models(root, training_root)
    trained_turns, trained_session, trained_selected, trained_pending = (
        _run_selection_path(database, baseline_questions, selection_question,
                            models=models))
    replay_turns, replay_session, replay_selected, replay_pending = (
        _run_selection_path(database, baseline_questions, selection_question,
                            models=models))
    candidate_page_ids = tuple(item.page_id for item in initial_session.pending.candidates)
    statuses = tuple(item.status for item in trained_turns)
    source_contract = all(
        (before.status, before.source_title, before.source_url,
         before.retrieval_question)
        == (after.status, after.source_title, after.source_url,
            after.retrieval_question)
        for before, after in zip(baseline_turns, trained_turns))
    selection_contract = (
        statuses == ("ANSWER", "CLARIFY", "ANSWER")
        and baseline_selected == (False, False, True)
        and trained_selected == (False, False, True)
        and replay_selected == (False, False, True)
        and trained_pending is None and replay_pending is None
        and trained_turns[2].status == "ANSWER"
        and trained_turns[2].source_title == selected_candidate.title
        and trained_turns[2].source_title is not None
    )
    replay_identical = (
        trained_turns == replay_turns
        and trained_session == replay_session
        and trained_selected == replay_selected)
    passed = bool(selection_contract and source_contract and replay_identical)
    training_pack, _evidence, _obligation, _relation, _role, _marker, _frame, observation, _consumer, used = models
    return {
        "artifact_kind": "PH2_BROAD_QA_CLARIFY_SELECTION_RUNTIME_V1",
        "baseline_candidate_count": len(initial_session.pending.candidates),
        "baseline_candidate_page_ids": list(candidate_page_ids),
        "database_sha256": _sha256(database),
        "format_version": CLARIFY_SELECTION_PROTOCOL_V1,
        "item_id": item_id,
        "opened_question": ambiguity,
        "replay_bit_identical": replay_identical,
        "run_id": observation.run_id,
        "selected_candidate_identity_u8": list(
            selected_candidate.candidate_identity_u8),
        "selected_candidate_page_id": selected_candidate.page_id,
        "selection_contract": selection_contract,
        "selection_question": selection_question,
        "source_contract": source_contract,
        "status": CLARIFY_SELECTION_PASS if passed else CLARIFY_SELECTION_NE,
        "status_counts": {
            status: sum(item.status == status for item in trained_turns)
            for status in sorted({item.status for item in trained_turns})
        },
        "trained_surface_consumer_used_count": used[0],
        "training_observation": observation.to_dict(),
        "training_pack_sha256": training_pack.pack_sha256,
        "turns": [_turn_dict(item) for item in trained_turns],
        "turns_sha256": _turn_digest(trained_turns),
    }


def write_clarify_selection_report(
        value: dict[str, object], output_path: str | Path) -> str:
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ClarifySelectionError("selection output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
    output.write_bytes(canonical_json_line(value))
    return str(output)


__all__ = [
    "BroadClarifyCandidate", "BroadClarifyPending", "BroadClarifySession",
    "CLARIFY_SELECTION_NE", "CLARIFY_SELECTION_PASS",
    "ClarifySelectionError", "answer_broad_clarify_selection_turn",
    "build_clarify_selection_report", "candidate_selection_question",
    "write_clarify_selection_report",
]
