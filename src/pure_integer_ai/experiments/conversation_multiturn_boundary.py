"""真实冻结问答 pack 的多轮焦点边界切片。

本模块只验证对话状态，不制造事实。首问和后续问题均从冻结公开 pack
读取；唯一额外输入是无答案标签的指代探针 ``它是什么？``。训练侧只
消费已存在的 K 盘课程和表层模型，来源、证据、状态仍由同一 SQLite
检索入口产生。
"""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_broad_qa_scale_audit import (
    _verify_pack,
)
from pure_integer_ai.experiments.conversation_broad_qa_training_contrast import (
    _consumer_factory,
    _frozen_training_course_paths,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_evidence_learning import (
    learn_evidence_term_weights,
)
from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    learn_typed_obligations,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    learn_relation_role_evidence_model,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


BOUNDARY_KIND = "PH2_BROAD_QA_DIALOGUE_MULTITURN_BOUNDARY_V1"
BOUNDARY_FORMAT_VERSION = 1
BOUNDARY_PASS = "PASS"
BOUNDARY_NE = "NE"
_PROBE = "它是什么？"


class MultiturnBoundaryError(ValueError):
    """多轮边界输入或运行不变量非法。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _turn_dict(turn: DialogueTurn) -> dict[str, object]:
    return {
        "answer": turn.answer,
        "display_answer": turn.display_answer,
        "ordinal": turn.ordinal,
        "question": turn.question,
        "retrieval_question": turn.retrieval_question,
        "source_title": turn.source_title,
        "source_url": turn.source_url,
        "status": turn.status,
        "turn_key_u8": list(turn.turn_key),
    }


def _turn_digest(turns: tuple[DialogueTurn, ...]) -> str:
    payload = canonical_json_line([_turn_dict(turn) for turn in turns])
    return hashlib.sha256(payload).hexdigest()


def _run_sequence(
        database_path: Path,
        questions: tuple[str, ...],
        *,
        surface_consumer: Any | None = None,
        evidence_weights: tuple[tuple[str, int], ...] | None = None,
        typed_obligation: Any | None = None,
        relation_model: Any | None = None,
        role_model: Any | None = None,
        marker_model: Any | None = None,
        frame_model: Any | None = None,
        ) -> tuple[tuple[DialogueTurn, ...], BroadDialogueState]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        state = BroadDialogueState((3, 8, 1))
        turns: list[DialogueTurn] = []
        for question in questions:
            state, turn = answer_broad_dialogue_turn(
                state, question, connection,
                surface_consumer=surface_consumer,
                learned_evidence_term_weights=evidence_weights,
                learned_typed_obligation=typed_obligation,
                learned_relation_evidence_model=relation_model,
                learned_relation_role_evidence_model=role_model,
                learned_relation_marker_evidence_model=marker_model,
                learned_relation_answer_frame_model=frame_model,
            )
            turns.append(turn)
        return tuple(turns), state
    finally:
        connection.close()


def _course_paths(root: Path, training_root: Path) -> tuple[Path, ...]:
    paths = _frozen_training_course_paths(training_root, root)
    if not paths or len(paths) != len(set(paths)):
        raise MultiturnBoundaryError("训练课程清单为空或重复")
    return paths


def _relation_paths(root: Path, names: Iterable[str]) -> tuple[Path, ...]:
    paths = tuple((root / "data/ph2" / name).resolve() for name in names)
    if any(not path.is_file() for path in paths) or len(paths) != len(set(paths)):
        raise MultiturnBoundaryError("关系课程缺失或重复")
    return paths


def _question_sequence(
        pack_dir: Path) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    _manifest, values, _labels, _dimensions = _verify_pack(pack_dir)
    if len(values) < 9:
        raise MultiturnBoundaryError("冻结 pack 至少需要九条真实问题")
    real = tuple(str(item["question"]) for item in values[:9])
    if any(not item.strip() for item in real):
        raise MultiturnBoundaryError("冻结 pack 含空问题")
    # 探针不携带实体、页面、答案或评测标签；其余问题全部来自 pack。
    return (real[0], _PROBE, *real[1:]), (
        str(values[0]["item_id"]), None,
        *(str(item["item_id"]) for item in values[1:9]),
    )


def _same_source_contract(
        baseline: tuple[DialogueTurn, ...],
        trained: tuple[DialogueTurn, ...],
        ) -> bool:
    return all(
        (before.status, before.source_title, before.source_url,
         before.retrieval_question)
        == (after.status, after.source_title, after.source_url,
            after.retrieval_question)
        for before, after in zip(baseline, trained)
    )


def _source_contract_differences(
        baseline: tuple[DialogueTurn, ...],
        trained: tuple[DialogueTurn, ...],
        ) -> tuple[dict[str, object], ...]:
    differences = []
    for before, after in zip(baseline, trained):
        before_value = (before.status, before.source_title, before.source_url,
                        before.retrieval_question)
        after_value = (after.status, after.source_title, after.source_url,
                       after.retrieval_question)
        if before_value != after_value:
            differences.append({
                "after": {
                    "retrieval_question": after.retrieval_question,
                    "source_title": after.source_title,
                    "source_url": after.source_url,
                    "status": after.status,
                },
                "before": {
                    "retrieval_question": before.retrieval_question,
                    "source_title": before.source_title,
                    "source_url": before.source_url,
                    "status": before.status,
                },
                "ordinal": before.ordinal,
            })
    return tuple(differences)


def build_multiturn_boundary_report(
        *, project_root: str | Path,
        pack_dir: str | Path,
        database_path: str | Path,
        training_run_root: str | Path,
        ) -> dict[str, object]:
    """运行真实十轮序列：真实首问、指代探针、八条真实后续问题。"""
    root = Path(project_root).resolve()
    pack = Path(pack_dir).resolve()
    database = Path(database_path).resolve()
    training_root = Path(training_run_root).resolve()
    if any(path.drive.upper() != "K:" for path in (pack, database, training_root)):
        raise MultiturnBoundaryError("pack、database、training run 必须位于 K 盘")
    if not database.is_file() or not training_root.is_dir():
        raise MultiturnBoundaryError("K 盘输入缺失")
    questions, source_item_ids = _question_sequence(pack)
    courses = _course_paths(root, training_root)
    training_pack = load_dialogue_training_pack(courses)
    evidence_model = learn_evidence_term_weights(training_pack)
    obligation_model = learn_typed_obligations(courses)
    relation_model = learn_relation_evidence_model(_relation_paths(root, (
        "dlg_raw_public_relation_evidence_v1.jsonl.sample",
        "dlg_raw_public_relation_evidence_v2.jsonl.sample",
    )))
    role_model = learn_relation_role_evidence_model(_relation_paths(root, (
        "dlg_raw_public_relation_role_evidence_v1.jsonl.sample",
        "dlg_raw_public_relation_role_evidence_v2.jsonl.sample",
    )))
    marker_model = learn_relation_marker_evidence_model(_relation_paths(root, (
        "dlg_raw_public_relation_marker_evidence_v1.jsonl.sample",
        "dlg_raw_public_relation_marker_evidence_v2.jsonl.sample",
    )))
    frame_model = learn_relation_answer_frame_model(_relation_paths(root, (
        "dlg_raw_public_relation_answer_frame_v1.jsonl.sample",
    )))
    observation, consumer_bundle = _consumer_factory(
        training_root, root, training_pack.pack_sha256)
    consumer, used = consumer_bundle

    baseline, baseline_state = _run_sequence(database, questions)
    trained, trained_state = _run_sequence(
        database, questions,
        surface_consumer=consumer,
        evidence_weights=evidence_model.weights,
        typed_obligation=obligation_model,
        relation_model=relation_model,
        role_model=role_model,
        marker_model=marker_model,
        frame_model=frame_model,
    )
    replay, replay_state = _run_sequence(
        database, questions,
        surface_consumer=consumer,
        evidence_weights=evidence_model.weights,
        typed_obligation=obligation_model,
        relation_model=relation_model,
        role_model=role_model,
        marker_model=marker_model,
        frame_model=frame_model,
    )
    first, probe = trained[0], trained[1]
    probe_expected = (
        f"{first.source_title}，{_PROBE}"
        if first.status == "ANSWER" and first.source_title else _PROBE)
    injection_indexes = tuple(
        turn.ordinal for turn in trained
        if turn.retrieval_question is not None
        and turn.retrieval_question != turn.question)
    history_bounded = (
        len(baseline_state.turns) == 8
        and len(trained_state.turns) == 8
        and len(replay_state.turns) == 8
    )
    focus_contract = (
        probe.retrieval_question == probe_expected
        and injection_indexes == (1,)
        and first.source_title is not None
    )
    replay_identical = trained == replay and _turn_digest(trained) == _turn_digest(replay)
    source_contract = _same_source_contract(baseline, trained)
    source_differences = _source_contract_differences(baseline, trained)
    evidence_differences = tuple(
        turn.ordinal for before, turn in zip(baseline, trained)
        if before.answer != turn.answer)
    ordinals_contiguous = tuple(turn.ordinal for turn in trained) == tuple(range(10))
    passed = bool(
        focus_contract and history_bounded and replay_identical
        and source_contract and ordinals_contiguous
        and all(turn.turn_key for turn in trained)
    )
    statuses = {status: sum(turn.status == status for turn in trained)
                for status in sorted({turn.status for turn in trained})}
    trace_values: list[int] = [BOUNDARY_FORMAT_VERSION, len(questions),
                               len(injection_indexes), int(focus_contract),
                               int(history_bounded), int(replay_identical),
                               int(source_contract)]
    for turn in trained:
        trace_values.extend((turn.ordinal, len(turn.turn_key),
                             int(turn.status == "ANSWER"),
                             int(turn.status == "UNKNOWN"),
                             int(turn.status == "CLARIFY")))
    return {
        "artifact_kind": BOUNDARY_KIND,
        "baseline_status_counts": {
            status: sum(turn.status == status for turn in baseline)
            for status in sorted({turn.status for turn in baseline})
        },
        "database_sha256": _sha256(database),
        "focus_contract": focus_contract,
        "evidence_changed_ordinals": list(evidence_differences),
        "focus_injection_ordinals": list(injection_indexes),
        "format_version": BOUNDARY_FORMAT_VERSION,
        "hot_history_limit": 8,
        "hot_history_lengths": {
            "baseline": len(baseline_state.turns),
            "replay": len(replay_state.turns),
            "trained": len(trained_state.turns),
        },
        "ordinals_contiguous": ordinals_contiguous,
        "probe_question": _PROBE,
        "probe_retrieval_question": probe.retrieval_question,
        "questions": list(questions),
        "question_source_item_ids": list(source_item_ids),
        "replay_bit_identical": replay_identical,
        "run_id": observation.run_id,
        "source_contract": source_contract,
        "source_contract_differences": list(source_differences),
        "status": BOUNDARY_PASS if passed else BOUNDARY_NE,
        "status_counts": statuses,
        "trained_surface_consumer_used_count": used[0],
        "training_observation": observation.to_dict(),
        "training_pack_sha256": training_pack.pack_sha256,
        "training_run_root": training_root.as_posix(),
        "turns": [_turn_dict(turn) for turn in trained],
        "turns_sha256": _turn_digest(trained),
        "trace_u": list(integer_tuple_fingerprint(
            tuple(trace_values), domain="pure_integer_ai.dialogue.multiturn.boundary.v1")),
    }


def write_multiturn_boundary_report(
        value: dict[str, object], output_path: str | Path) -> str:
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise MultiturnBoundaryError("boundary output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_line(value))
    return str(output)


__all__ = [
    "BOUNDARY_KIND", "BOUNDARY_NE", "BOUNDARY_PASS",
    "MultiturnBoundaryError", "build_multiturn_boundary_report",
    "write_multiturn_boundary_report",
]
