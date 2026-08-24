"""真实来源问题的 CLARIFY -> UNKNOWN -> ANSWER 多轮边界切片。

运行器从冻结公开 pack 读取一条问题，先由真实来源检索取得标题，再把标题
从原问题中移除形成结构性歧义探针。探针不携带答案或评测标签；所有状态、
来源和表面仍由既有广域入口产生。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    DialogueTurn,
)
from pure_integer_ai.experiments.conversation_broad_qa_scale_audit import (
    _verify_pack,
)
from pure_integer_ai.experiments.conversation_multiturn_boundary import (
    _PROBE,
    _course_paths,
    _relation_paths,
    _run_sequence,
    _same_source_contract,
    _sha256,
    _turn_digest,
    _turn_dict,
)
from pure_integer_ai.experiments.conversation_broad_qa_training_contrast import (
    _consumer_factory,
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
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


CLARIFY_BOUNDARY_KIND = "PH2_BROAD_QA_DIALOGUE_MULTITURN_CLARIFY_BOUNDARY_V1"
CLARIFY_BOUNDARY_FORMAT_VERSION = 1
CLARIFY_BOUNDARY_PASS = "PASS"
CLARIFY_BOUNDARY_NE = "NE"


class ClarifyBoundaryError(ValueError):
    """澄清边界输入或多轮状态合同非法。"""


def _ambiguity_question(question: str, source_title: str) -> str:
    """从真实问题移除已检索标题，形成不含答案的来源绑定探针。"""
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(source_title, str) or not source_title.strip()
            or source_title not in question):
        raise ClarifyBoundaryError("真实问题未包含可移除的来源标题")
    value = question.replace(source_title, "", 1)
    while "  " in value:
        value = value.replace("  ", " ")
    value = value.strip()
    if not value or value == question:
        raise ClarifyBoundaryError("来源歧义探针为空或未改变")
    return value


def _models(root: Path, training_root: Path) -> tuple[Any, ...]:
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
    return (training_pack, evidence_model, obligation_model, relation_model,
            role_model, marker_model, frame_model, observation, consumer, used)


def _run_trained(
        database: Path, questions: tuple[str, ...], models: tuple[Any, ...],
        ) -> tuple[tuple[DialogueTurn, ...], Any, int]:
    (_training_pack, evidence_model, obligation_model, relation_model,
     role_model, marker_model, frame_model, _observation, consumer, used) = models
    turns, state = _run_sequence(
        database, questions,
        surface_consumer=consumer,
        evidence_weights=evidence_model.weights,
        typed_obligation=obligation_model,
        relation_model=relation_model,
        role_model=role_model,
        marker_model=marker_model,
        frame_model=frame_model,
    )
    return turns, state, used[0]


def _ambiguity_stats(database: Path, question: str) -> dict[str, object]:
    """只读回读歧义探针的候选计数，不读取标签或生成答案。"""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        result = query_broad_qa(connection, question)
        return {
            "candidate_document_count": result.candidate_document_count,
            "matched_term_count": result.matched_term_count,
            "status": result.status,
            "source_title": result.title,
        }
    finally:
        connection.close()


def build_clarify_boundary_report(
        *, project_root: str | Path,
        pack_dir: str | Path,
        database_path: str | Path,
        training_run_root: str | Path,
        item_id: str,
        ) -> dict[str, object]:
    """运行真实五轮：ANSWER -> CLARIFY -> UNKNOWN -> ANSWER -> follow-up。"""
    root = Path(project_root).resolve()
    pack = Path(pack_dir).resolve()
    database = Path(database_path).resolve()
    training_root = Path(training_run_root).resolve()
    if any(path.drive.upper() != "K:" for path in (pack, database, training_root)):
        raise ClarifyBoundaryError("pack、database、training run 必须位于 K 盘")
    if not database.is_file() or not training_root.is_dir():
        raise ClarifyBoundaryError("K 盘输入缺失")
    _manifest, questions, _labels, _dimensions = _verify_pack(pack)
    selected = tuple(item for item in questions
                     if str(item.get("item_id")) == item_id)
    if len(selected) != 1:
        raise ClarifyBoundaryError("item_id 必须唯一命中冻结 pack")
    question = str(selected[0]["question"])

    # 先用真实入口获取标题；标题缺失时直接失败闭合，不猜测探针。
    baseline_first, _ = _run_sequence(database, (question,))
    first = baseline_first[0]
    if first.status != "ANSWER" or not first.source_title:
        raise ClarifyBoundaryError("所选真实问题未形成来源 ANSWER")
    ambiguity = _ambiguity_question(question, first.source_title)
    sequence = (question, ambiguity, _PROBE, question, _PROBE)
    sequence_item_ids: tuple[str | None, ...] = (
        item_id, None, None, item_id, None)

    baseline, baseline_state = _run_sequence(database, sequence)
    ambiguity_stats = _ambiguity_stats(database, ambiguity)
    models = _models(root, training_root)
    trained, trained_state, used = _run_trained(database, sequence, models)
    replay, replay_state, _ = _run_trained(database, sequence, models)
    observation = models[7]

    statuses = tuple(turn.status for turn in trained)
    expected_statuses = ("ANSWER", "CLARIFY", "UNKNOWN", "ANSWER", "ANSWER")
    focus_ordinals = tuple(
        turn.ordinal for turn in trained
        if turn.retrieval_question is not None
        and turn.retrieval_question != turn.question)
    focus_contract = (
        statuses == expected_statuses
        and ambiguity_stats["status"] == "CLARIFY"
        and isinstance(ambiguity_stats["candidate_document_count"], int)
        and ambiguity_stats["candidate_document_count"] > 1
        and trained[1].source_title is None
        and trained[2].source_title is None
        and trained[1].retrieval_question == trained[1].question
        and trained[2].retrieval_question == trained[2].question
        and trained[3].source_title == first.source_title
        and focus_ordinals == (4,)
        and trained[4].retrieval_question
        == f"{first.source_title}，{_PROBE}")
    history_bounded = all(
        len(state.turns) == len(sequence)
        and len(state.turns) <= 8
        for state in (baseline_state, trained_state, replay_state))
    replay_identical = trained == replay and _turn_digest(trained) == _turn_digest(replay)
    source_contract = _same_source_contract(baseline, trained)
    evidence_changed = tuple(
        turn.ordinal for before, turn in zip(baseline, trained)
        if before.answer != turn.answer)
    passed = bool(focus_contract and history_bounded and replay_identical
                  and source_contract)
    training_pack = models[0]
    return {
        "artifact_kind": CLARIFY_BOUNDARY_KIND,
        "ambiguity_question": ambiguity,
        "ambiguity_query_stats": ambiguity_stats,
        "baseline_status_counts": {
            status: sum(turn.status == status for turn in baseline)
            for status in sorted({turn.status for turn in baseline})
        },
        "database_sha256": _sha256(database),
        "evidence_changed_ordinals": list(evidence_changed),
        "focus_contract": focus_contract,
        "focus_injection_ordinals": list(focus_ordinals),
        "format_version": CLARIFY_BOUNDARY_FORMAT_VERSION,
        "hot_history_limit": 8,
        "hot_history_lengths": {
            "baseline": len(baseline_state.turns),
            "replay": len(replay_state.turns),
            "trained": len(trained_state.turns),
        },
        "item_id": item_id,
        "probe_question": _PROBE,
        "question_source_item_ids": list(sequence_item_ids),
        "questions": list(sequence),
        "replay_bit_identical": replay_identical,
        "run_id": observation.run_id,
        "source_contract": source_contract,
        "status": CLARIFY_BOUNDARY_PASS if passed else CLARIFY_BOUNDARY_NE,
        "status_counts": {
            status: sum(turn.status == status for turn in trained)
            for status in sorted(set(statuses))
        },
        "trained_surface_consumer_used_count": used,
        "training_observation": observation.to_dict(),
        "training_pack_sha256": training_pack.pack_sha256,
        "training_run_id": observation.run_id,
        "turns": [_turn_dict(turn) for turn in trained],
        "turns_sha256": _turn_digest(trained),
    }


def write_clarify_boundary_report(
        value: dict[str, object], output_path: str | Path) -> str:
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ClarifyBoundaryError("clarify output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_line(value))
    return str(output)


__all__ = [
    "CLARIFY_BOUNDARY_KIND", "CLARIFY_BOUNDARY_NE", "CLARIFY_BOUNDARY_PASS",
    "ClarifyBoundaryError", "build_clarify_boundary_report",
    "write_clarify_boundary_report",
]
