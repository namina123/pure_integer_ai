"""公开多轮对话开发切片：来源焦点、未知/澄清和训练表层组合。

该切片把真实 K 盘广域 SQLite、公开窄域完整句运行时和训练表层消费者放在
同一条只读对话链中。它刻意包含 ``ANSWER -> UNKNOWN -> CLARIFY -> ANSWER``：
未知轮之后的指代不得继承旧来源；明确的新问题又必须恢复回答。报告只描述
开发能力，不改变训练图或断奶状态。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Callable

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)


MULTITURN_PROTOCOL_V1 = 1
MULTITURN_PASS = "PASS"
MULTITURN_FAIL = "FAIL"
MULTITURN_NE = "NE"
_TRACE_DOMAIN = "pure_integer_ai.dialogue.multiturn.scale.v1"
_QUESTIONS = (
    "从矮寨大桥的工程时间线看，矮寨大桥何时建成通车？",
    "火星上的矮寨大桥何时通车？",
    "它分布在哪些地区？",
    "什么使得河水上涨？",
    "请只依据黄山松条目中关于地理分布的公开资料回答：黄山松分布在哪些地区？",
    "请用一个完整句子说明该条目的地理分布。",
)
_EXPECTED_STATUS = ("ANSWER", "UNKNOWN", "CLARIFY", "ANSWER", "ANSWER", "ANSWER")


class MultiturnScaleError(ValueError):
    """多轮开发切片合同或 K 盘边界无效。"""


def _text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise MultiturnScaleError(f"{where} 必须是规范字符串")
    if not allow_empty and not value:
        raise MultiturnScaleError(f"{where} 不能为空")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise MultiturnScaleError(f"{where} 含非 Unicode scalar")
    return value


@dataclass(frozen=True, slots=True)
class MultiturnTurn:
    """可写入报告的轮次纯值投影。"""

    ordinal: int
    question: str
    status: str
    answer: str | None
    display_answer: str | None
    source_title: str | None
    retrieval_question: str | None
    turn_key: tuple[int, ...]

    @classmethod
    def from_turn(cls, turn: DialogueTurn) -> "MultiturnTurn":
        return cls(
            turn.ordinal, turn.question, turn.status, turn.answer,
            turn.display_answer, turn.source_title, turn.retrieval_question,
            tuple(turn.turn_key),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "display_answer": self.display_answer,
            "ordinal": self.ordinal,
            "question": self.question,
            "retrieval_question": self.retrieval_question,
            "source_title": self.source_title,
            "status": self.status,
            "turn_key_u8": list(self.turn_key),
        }

    def canonical_record(self) -> tuple[int, ...]:
        result = [MULTITURN_PROTOCOL_V1, self.ordinal, len(self.turn_key)]
        for value in (self.question, self.status, self.answer or "",
                      self.display_answer or "", self.source_title or "",
                      self.retrieval_question or ""):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        result.extend(self.turn_key)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MultiturnScaleReport:
    """固定六轮开发切片的摘要。"""

    status: str
    run_id: str
    pack_sha256: str
    database_name: str
    question_count: int
    answer_count: int
    unknown_count: int
    clarify_count: int
    long_answer_count: int
    trained_surface_used_count: int
    focus_injection_count: int
    focus_not_crossed_unknown: int
    replay_bit_identical: bool
    turns_sha256: str
    turns: tuple[MultiturnTurn, ...]
    trace: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_count": self.answer_count,
            "clarify_count": self.clarify_count,
            "database_name": self.database_name,
            "focus_injection_count": self.focus_injection_count,
            "focus_not_crossed_unknown": bool(self.focus_not_crossed_unknown),
            "format_version": MULTITURN_PROTOCOL_V1,
            "long_answer_count": self.long_answer_count,
            "pack_sha256": self.pack_sha256,
            "question_count": self.question_count,
            "replay_bit_identical": self.replay_bit_identical,
            "run_id": self.run_id,
            "status": self.status,
            "trace_u": list(self.trace),
            "turns": [item.to_dict() for item in self.turns],
            "turns_sha256": self.turns_sha256,
            "trained_surface_used_count": self.trained_surface_used_count,
            "unknown_count": self.unknown_count,
        }


def multiturn_questions() -> tuple[str, ...]:
    """返回固定的来源焦点/未知/恢复序列。"""
    return _QUESTIONS


def _turns_digest(turns: tuple[MultiturnTurn, ...]) -> str:
    payload = json.dumps([item.to_dict() for item in turns],
                         ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _narrow_answer_factory(
        trained_surface: TrainedSurfaceRuntime,
        ) -> tuple[Callable[[str], tuple[str, str] | None], list[int]]:
    narrow_runtime = load_or_rebuild_public_sparse_qa_runtime()
    catalog = build_public_sentence_demo_catalog(narrow_runtime)
    used_count = [0]

    def answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            narrow_runtime, catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        rendered = trained_surface.render(surface, response_act="ANSWER")
        if rendered.used:
            used_count[0] += 1
            surface = rendered.surface
        return surface, "ANSWER"

    return answer, used_count


def _run_once(database: sqlite3.Connection,
              questions: tuple[str, ...],
              narrow_answer: Callable[[str], tuple[str, str] | None],
              ) -> tuple[MultiturnTurn, ...]:
    state = BroadDialogueState((2, 8, 1))
    turns: list[MultiturnTurn] = []
    for question in questions:
        state, turn = answer_broad_dialogue_turn(
            state, question, database, narrow_answer=narrow_answer)
        turns.append(MultiturnTurn.from_turn(turn))
    return tuple(turns)


def build_multiturn_scale_report(*, project_root: str | Path,
                                 database_path: str | Path,
                                 training_run_root: str | Path,
                                 expected_pack_sha256: str | None = None,
                                 ) -> MultiturnScaleReport:
    """在同一真实 K 盘 SQLite 上运行并重放六轮开发切片。"""
    root = Path(project_root).resolve()
    database = Path(database_path).resolve()
    if database.drive.upper() != "K:" or not database.is_file():
        raise MultiturnScaleError("database 必须是存在的 K 盘文件")
    pack = load_dialogue_training_pack(default_course_paths(root))
    expected = pack.pack_sha256 if expected_pack_sha256 is None else expected_pack_sha256
    trained_surface = load_trained_surface_runtime(
        project_root=root, training_run_root=training_run_root,
        expected_pack_sha256=expected,
    )
    observation = load_training_observation(
        training_run_root, expected_pack_sha256=expected)
    narrow_answer, used_count = _narrow_answer_factory(trained_surface)
    questions = multiturn_questions()
    first_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        first = _run_once(first_connection, questions, narrow_answer)
    finally:
        first_connection.close()
    used_first = used_count[0]
    replay_narrow, _ = _narrow_answer_factory(trained_surface)
    replay_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        replay = _run_once(replay_connection, questions, replay_narrow)
    finally:
        replay_connection.close()
    digest = _turns_digest(first)
    replay_digest = _turns_digest(replay)
    statuses = tuple(item.status for item in first)
    # The third turn is a pronoun after UNKNOWN. It must remain an unanchored
    # query, while the last turn is the legitimate immediate Huangshan follow-up.
    focus_not_crossed_unknown = int(
        first[1].status == "UNKNOWN"
        and first[2].status == "CLARIFY"
        and first[2].retrieval_question == first[2].question
        and first[5].retrieval_question != first[5].question
        and first[5].source_title == "黄山松"
    )
    focus_injection_count = sum(
        item.retrieval_question is not None
        and item.retrieval_question != item.question
        for item in first)
    answer_count = sum(item.status == "ANSWER" for item in first)
    unknown_count = sum(item.status == "UNKNOWN" for item in first)
    clarify_count = sum(item.status == "CLARIFY" for item in first)
    long_answer_count = sum(
        len((item.display_answer or "").encode("utf-8")) >= 48
        for item in first if item.status == "ANSWER")
    passed = (
        statuses == _EXPECTED_STATUS
        and first[0].source_title == "矮寨大桥"
        and first[1].source_title is None
        and first[2].source_title is None
        and first[3].display_answer == "暴雨使得河水上涨。"
        and first[4].source_title == "黄山松"
        and first[5].source_title == "黄山松"
        and focus_not_crossed_unknown
        and used_first >= 1
        and first == replay
    )
    status = MULTITURN_PASS if passed else MULTITURN_FAIL
    trace_values = [MULTITURN_PROTOCOL_V1, len(first), answer_count,
                    unknown_count, clarify_count, focus_injection_count,
                    focus_not_crossed_unknown, used_first]
    for item in first:
        trace_values.extend(item.canonical_record())
    trace = integer_tuple_fingerprint(tuple(trace_values), domain=_TRACE_DOMAIN)
    return MultiturnScaleReport(
        status, observation.run_id, observation.pack_sha256, database.name,
        len(first), answer_count, unknown_count, clarify_count,
        long_answer_count, used_first, focus_injection_count,
        focus_not_crossed_unknown,
        first == replay and digest == replay_digest,
        digest, first, trace,
    )


def write_multiturn_scale_report(report: MultiturnScaleReport,
                                 output_path: str | Path) -> str:
    """只创建 K 盘摘要，不覆盖既有产物。"""
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("multiturn output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")) + "\n",
                       encoding="utf-8")
    return str(output)


__all__ = [
    "MULTITURN_FAIL", "MULTITURN_NE", "MULTITURN_PASS",
    "MultiturnScaleError", "MultiturnScaleReport", "MultiturnTurn",
    "build_multiturn_scale_report", "multiturn_questions",
    "write_multiturn_scale_report",
]
