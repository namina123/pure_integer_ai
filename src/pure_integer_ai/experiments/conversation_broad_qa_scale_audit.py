"""公开 100 问广域对话规模审计。

该审计把已经冻结的 SOURCE_ALIGNED 开发集逐题送入真实对话入口，区分
用户可见的主证据句与完整证据链。它只读取 K 盘 SQLite 和公开 pack，
不写训练状态，也不把召回命中命名为语言理解。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from typing import Callable

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive_family import (
    INTERACTIVE_DIMENSION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_LABEL_KIND,
    JOINT_QUESTION_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    LearnedTypedObligation,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    LearnedRelationRoleEvidenceModel,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    LearnedRelationMarkerEvidenceModel,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    LearnedRelationAnswerFrameModel,
)


SCALE_AUDIT_KIND = "PH2_BROAD_QA_DIALOGUE_SCALE_AUDIT_V1"
SCALE_AUDIT_FORMAT_VERSION = 1
_DIMENSIONS = ("CAUSE", "COMPARISON", "QUANTITY", "RELATION", "TIME")
_STRUCTURAL_PREFIXES = (
    "Category:", "category:", "{|", "===", "====", "* ", "# ",
)
_SENTENCE_END = re.compile(r"[。！？!?；;]$")


class ConversationBroadQaScaleAuditError(ValueError):
    """公开 pack、数据库边界或运行时观察不合法。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversationBroadQaScaleAuditError(
            f"JSON artifact 不可回读: {path.name}") from error
    if not isinstance(value, dict) or canonical_json_line(value) != payload:
        raise ConversationBroadQaScaleAuditError(
            f"JSON artifact 非规范: {path.name}")
    return value


def _read_jsonl(path: Path, record_kind: str) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ConversationBroadQaScaleAuditError(f"JSONL 缺失: {path.name}")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                identity = value.get("item_id") if isinstance(value, dict) else None
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or value.get("record_kind") != record_kind
                        or not isinstance(identity, str) or not identity
                        or identity in identities):
                    raise ConversationBroadQaScaleAuditError(
                        f"JSONL record 非法: {path.name}:{line_number}")
                identities.add(identity)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, ConversationBroadQaScaleAuditError):
            raise
        raise ConversationBroadQaScaleAuditError(
            f"JSONL 不可回读: {path.name}") from error
    if not values:
        raise ConversationBroadQaScaleAuditError(f"JSONL 为空: {path.name}")
    return tuple(values)


def _verify_pack(pack_dir: Path) -> tuple[
        dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...]]:
    manifest_path = pack_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if (manifest.get("artifact_kind")
            != "PH2_BROAD_QA_INTERACTIVE_DEVELOPMENT_PACK_V1"
            or manifest.get("format_version") != 1
            or not isinstance(artifacts, list)):
        raise ConversationBroadQaScaleAuditError("interactive pack manifest 非法")
    for item in artifacts:
        if not isinstance(item, dict) or not {
                "bytes", "record_count", "role", "sha256"} <= set(item):
            raise ConversationBroadQaScaleAuditError("interactive pack artifact 清单非法")
        path = pack_dir / {
            "dev_questions": "dev.questions.jsonl",
            "dev_labels": "dev.labels.jsonl",
            "dev_dimensions": "dev.dimensions.jsonl",
            "source_targets": "source_targets.jsonl",
        }.get(str(item["role"]), "")
        if not path.is_file() or path.stat().st_size != item["bytes"] \
                or _sha256(path) != item["sha256"]:
            raise ConversationBroadQaScaleAuditError(
                f"interactive pack artifact 漂移: {item.get('role')}")
    questions = _read_jsonl(pack_dir / "dev.questions.jsonl", JOINT_QUESTION_KIND)
    labels = _read_jsonl(pack_dir / "dev.labels.jsonl", JOINT_LABEL_KIND)
    dimensions = _read_jsonl(
        pack_dir / "dev.dimensions.jsonl", INTERACTIVE_DIMENSION_RECORD_KIND)
    inventory = {item["item_id"] for item in questions}
    if (set(item["item_id"] for item in labels) != inventory
            or set(item["item_id"] for item in dimensions) != inventory
            or len(inventory) != manifest.get("question_count")):
        raise ConversationBroadQaScaleAuditError("interactive pack inventory 漂移")
    return manifest, questions, labels, dimensions


def _readable(surface: str | None) -> bool:
    if not isinstance(surface, str) or not surface.strip():
        return False
    text = surface.strip()
    return (not text.startswith(_STRUCTURAL_PREFIXES)
            and any(char.isalpha() or "\u4e00" <= char <= "\u9fff"
                    for char in text))


@dataclass(frozen=True, slots=True)
class DialogueScaleObservation:
    """单题用户可见表面和来源证据的纯值记录。"""

    item_id: str
    ordinal: int
    dimension: str
    question: str
    status: str
    source_title: str | None
    display_answer: str
    evidence_answer: str
    gold_in_display: bool
    gold_in_evidence: bool
    readable_surface: bool
    complete_sentence: bool
    long_surface: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "complete_sentence": self.complete_sentence,
            "dimension": self.dimension,
            "display_answer": self.display_answer or None,
            "evidence_answer": self.evidence_answer or None,
            "gold_in_display": self.gold_in_display,
            "gold_in_evidence": self.gold_in_evidence,
            "item_id": self.item_id,
            "long_surface": self.long_surface,
            "ordinal": self.ordinal,
            "question": self.question,
            "readable_surface": self.readable_surface,
            "source_title": self.source_title,
            "status": self.status,
        }


def _run_once(database_path: Path, questions: tuple[dict[str, Any], ...],
              labels: dict[str, dict[str, Any]],
              dimensions: dict[str, dict[str, Any]],
              surface_consumer: Callable[[str, str, str | None], str | None]
              | None = None,
              learned_evidence_term_weights: tuple[tuple[str, int], ...]
              | None = None,
              learned_typed_obligation: LearnedTypedObligation | None = None,
              learned_relation_evidence_model: LearnedRelationEvidenceModel
              | None = None,
              learned_relation_role_evidence_model: LearnedRelationRoleEvidenceModel
              | None = None,
              learned_relation_marker_evidence_model: LearnedRelationMarkerEvidenceModel
              | None = None,
              learned_relation_answer_frame_model: LearnedRelationAnswerFrameModel
              | None = None,
              ) -> tuple[DialogueScaleObservation, ...]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        observations = []
        for ordinal, question in enumerate(questions):
            item_id = str(question["item_id"])
            text = str(question["question"])
            state = BroadDialogueState((1, ordinal + 1, 8))
            _, turn = answer_broad_dialogue_turn(
                state, text, connection, surface_consumer=surface_consumer,
                learned_evidence_term_weights=learned_evidence_term_weights,
                learned_typed_obligation=learned_typed_obligation,
                learned_relation_evidence_model=learned_relation_evidence_model,
                learned_relation_role_evidence_model=(
                    learned_relation_role_evidence_model),
                learned_relation_marker_evidence_model=(
                    learned_relation_marker_evidence_model),
                learned_relation_answer_frame_model=(
                    learned_relation_answer_frame_model))
            display = turn.display_answer or ""
            evidence = turn.answer or ""
            gold = tuple(
                normalize_external_text(value)
                for value in labels[item_id].get("gold_answers", ())
                if isinstance(value, str) and value)
            display_normalized = normalize_external_text(display)
            evidence_normalized = normalize_external_text(evidence)
            observations.append(DialogueScaleObservation(
                item_id,
                ordinal,
                str(dimensions[item_id]["dimension"]),
                text,
                turn.status,
                turn.source_title,
                display,
                evidence,
                any(value in display_normalized for value in gold),
                any(value in evidence_normalized for value in gold),
                _readable(display),
                _readable(display) and bool(_SENTENCE_END.search(display)),
                len(display.encode("utf-8")) >= 48,
            ))
        return tuple(observations)
    finally:
        connection.close()


def build_conversation_broad_qa_scale_audit(
        *, pack_dir: str | Path, database_path: str | Path,
        learned_relation_evidence_model: LearnedRelationEvidenceModel
        | None = None,
        learned_relation_role_evidence_model: LearnedRelationRoleEvidenceModel
        | None = None,
        learned_relation_marker_evidence_model: LearnedRelationMarkerEvidenceModel
        | None = None,
        learned_relation_answer_frame_model: LearnedRelationAnswerFrameModel
        | None = None,
        ) -> dict[str, object]:
    """在真实对话入口上运行 100 问并重复回放。"""
    pack = Path(pack_dir).resolve()
    database = Path(database_path).resolve()
    if pack.drive.upper() != "K:" or database.drive.upper() != "K:":
        raise ConversationBroadQaScaleAuditError("pack 和 database 必须位于 K 盘")
    if not database.is_file():
        raise ConversationBroadQaScaleAuditError("database 缺失")
    manifest, question_values, label_values, dimension_values = _verify_pack(pack)
    labels = {item["item_id"]: item for item in label_values}
    dimensions = {item["item_id"]: item for item in dimension_values}
    first = _run_once(
        database, question_values, labels, dimensions,
        learned_relation_evidence_model=learned_relation_evidence_model,
        learned_relation_role_evidence_model=learned_relation_role_evidence_model,
        learned_relation_marker_evidence_model=learned_relation_marker_evidence_model,
        learned_relation_answer_frame_model=learned_relation_answer_frame_model,
    )
    replay = _run_once(
        database, question_values, labels, dimensions,
        learned_relation_evidence_model=learned_relation_evidence_model,
        learned_relation_role_evidence_model=learned_relation_role_evidence_model,
        learned_relation_marker_evidence_model=learned_relation_marker_evidence_model,
        learned_relation_answer_frame_model=learned_relation_answer_frame_model,
    )
    if first != replay:
        raise ConversationBroadQaScaleAuditError("对话规模回放不一致")
    statuses = Counter(item.status for item in first)
    dimensions_count = Counter(item.dimension for item in first)
    for dimension in _DIMENSIONS:
        if dimensions_count[dimension] != manifest["dimension_counts"][dimension]:
            raise ConversationBroadQaScaleAuditError("维度计数漂移")
    failure_counts = Counter()
    for item in first:
        if item.status != "ANSWER":
            failure_counts["NON_ANSWER"] += 1
        elif not item.readable_surface:
            failure_counts["UNREADABLE_SURFACE"] += 1
        elif not item.gold_in_display:
            failure_counts["GOLD_NOT_IN_DISPLAY"] += 1
        if not item.gold_in_evidence:
            failure_counts["GOLD_NOT_IN_EVIDENCE"] += 1
    observations = [item.to_dict() for item in first]
    questions_artifacts = [
        item for item in manifest["artifacts"]
        if isinstance(item, dict) and item.get("role") == "dev_questions"
    ]
    if len(questions_artifacts) != 1:
        raise ConversationBroadQaScaleAuditError(
            "interactive pack 必须恰有一个 dev_questions artifact")
    body = {
        "artifact_kind": SCALE_AUDIT_KIND,
        "database_sha256": _sha256(database),
        "dimension_counts": dict(sorted(dimensions_count.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "format_version": SCALE_AUDIT_FORMAT_VERSION,
        "long_surface_count": sum(item.long_surface for item in first),
        "manifest_sha256": _sha256(pack / "manifest.json"),
        "observation_count": len(first),
        "observations": observations,
        "questions_sha256": questions_artifacts[0]["sha256"],
        "readable_surface_count": sum(item.readable_surface for item in first),
        "complete_sentence_count": sum(item.complete_sentence for item in first),
        "replay_bit_identical": True,
        "status_counts": dict(sorted(statuses.items())),
        "surface_gold_hit_count": sum(item.gold_in_display for item in first),
        "evidence_gold_hit_count": sum(item.gold_in_evidence for item in first),
        "question_count": len(first),
    }
    return body


def write_conversation_broad_qa_scale_audit(
        value: dict[str, object], output_path: str | Path,
        ) -> str:
    """只创建 K 盘 canonical JSON 审计，不覆盖历史产物。"""
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ConversationBroadQaScaleAuditError("audit output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_line(value))
    return str(output)


__all__ = [
    "ConversationBroadQaScaleAuditError",
    "DialogueScaleObservation",
    "build_conversation_broad_qa_scale_audit",
    "write_conversation_broad_qa_scale_audit",
]
