"""把公开 100 问接入训练前/后真实对话对照。

对照的训练侧只消费已经存在的 K 盘表层结构模型。来源检索和完整证据链
仍由同一个广域问答入口产生；训练消费者最多改变用户可见表面，不能注入
事实、改变来源、改变状态或跨过 UNKNOWN/CLARIFY 边界。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.conversation_broad_qa_scale_audit import (
    DialogueScaleObservation,
    ConversationBroadQaScaleAuditError,
    _run_once,
    _verify_pack,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.ph2_broad_qa_evidence_learning import (
    evidence_learning_sha256,
    learn_evidence_term_weights,
)
from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    learn_typed_obligations,
    typed_obligation_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
    relation_evidence_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    learn_relation_role_evidence_model,
    relation_role_evidence_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
    relation_marker_evidence_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
    relation_answer_frame_sha256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


TRAINING_CONTRAST_KIND = "PH2_BROAD_QA_TRAINING_CONTRAST_V1"
TRAINING_CONTRAST_FORMAT_VERSION = 1


class ConversationBroadQaTrainingContrastError(
        ConversationBroadQaScaleAuditError):
    """训练前/后对照的输入、隔离或不变量不合法。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _frozen_training_course_paths(
        training_root: Path, project: Path,
        ) -> tuple[Path, ...]:
    """按训练 run 自己的 source inventory 回放同一 pack。"""
    manifest_path = training_root / "dialogue_pack_manifest.json"
    if not manifest_path.is_file():
        return default_course_paths(project)
    try:
        manifest = json.loads(manifest_path.read_bytes())
        source_files = manifest["source_files"]
        values = tuple(Path(item[0]).resolve() for item in source_files)
        expected = tuple(str(item[1]) for item in source_files)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as error:
        raise ConversationBroadQaTrainingContrastError(
            "training run source inventory 不可回读") from error
    if (not values or len(values) != len(expected)
            or any(not path.is_file() for path in values)
            or any(_sha256(path) != digest for path, digest in zip(values, expected))):
        raise ConversationBroadQaTrainingContrastError(
            "training run source inventory 漂移")
    return values


def _metrics(observations: tuple[DialogueScaleObservation, ...]) -> dict[str, object]:
    """只从运行观察计算大能力级表面指标。"""
    statuses = Counter(item.status for item in observations)
    return {
        "complete_sentence_count": sum(item.complete_sentence for item in observations),
        "evidence_gold_hit_count": sum(item.gold_in_evidence for item in observations),
        "long_surface_count": sum(item.long_surface for item in observations),
        "readable_surface_count": sum(item.readable_surface for item in observations),
        "status_counts": dict(sorted(statuses.items())),
        "surface_gold_hit_count": sum(item.gold_in_display for item in observations),
        "question_count": len(observations),
    }


def _select_questions(
        questions: tuple[dict[str, Any], ...],
        item_ids: tuple[str, ...],
        ) -> tuple[dict[str, Any], ...]:
    """按 pack 原序选择真实问题，拒绝未知、重复或空选择。"""
    if not item_ids:
        return questions
    if len(item_ids) != len(set(item_ids)):
        raise ConversationBroadQaTrainingContrastError(
            "contrast item_id 不能重复")
    by_id = {str(item["item_id"]): item for item in questions}
    if any(item not in by_id for item in item_ids):
        raise ConversationBroadQaTrainingContrastError(
            "contrast item_id 不在冻结 dev pack")
    selected = tuple(item for item in questions
                     if str(item["item_id"]) in set(item_ids))
    if not selected:
        raise ConversationBroadQaTrainingContrastError(
            "contrast 选择为空")
    return selected


def _selection_sha256(questions: tuple[dict[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for question in questions:
        digest.update(canonical_json_line(question))
    return digest.hexdigest()


def _consumer_factory(
        training_run_root: Path, project_root: Path, pack_sha256: str,
        *,
        extra_variant_course_paths: tuple[Path, ...] = (),
        extra_variant_evidence_paths: tuple[Path, ...] = (),
        extra_order_course_paths: tuple[Path, ...] = (),
        extra_order_evidence_paths: tuple[Path, ...] = (),
        ) -> tuple[Any, Any]:
    """加载训练表层消费者，并返回只改 display 的 callback 与计数器。"""
    observation = load_training_observation(
        training_run_root, expected_pack_sha256=pack_sha256)
    runtime = load_trained_surface_runtime(
        project_root=project_root,
        training_run_root=training_run_root,
        expected_pack_sha256=pack_sha256,
        extra_variant_course_paths=extra_variant_course_paths,
        extra_variant_evidence_paths=extra_variant_evidence_paths,
        extra_order_course_paths=extra_order_course_paths,
        extra_order_evidence_paths=extra_order_evidence_paths,
    )
    used = [0]

    def consume(surface: str, status: str,
                source_title: str | None) -> str | None:
        """对已产生的 ANSWER 做安全表层消费，不读取评测标签。"""
        if status != "ANSWER":
            return None
        rendered = runtime.render(
            surface, response_act="ANSWER", source_title=source_title)
        if not rendered.used or not rendered.surface.strip():
            return None
        used[0] += 1
        return rendered.surface

    return observation, (consume, used)


def _assert_source_invariant(
        baseline: tuple[DialogueScaleObservation, ...],
        trained: tuple[DialogueScaleObservation, ...],
        ) -> None:
    """训练表层不得改变回答状态、来源或完整证据链。"""
    if len(baseline) != len(trained):
        raise ConversationBroadQaTrainingContrastError(
            "训练前后问题数量不一致")
    for before, after in zip(baseline, trained):
        if (before.item_id, before.status, before.source_title) != (
                after.item_id, after.status, after.source_title):
            raise ConversationBroadQaTrainingContrastError(
                f"训练路径改变来源/状态: {before.item_id}")


def build_conversation_broad_qa_training_contrast(
        *, project_root: str | Path, pack_dir: str | Path,
        database_path: str | Path, training_run_root: str | Path,
        item_ids: tuple[str, ...] = (),
        extra_training_course_paths: tuple[str | Path, ...] = (),
        extra_obligation_course_paths: tuple[str | Path, ...] = (),
        extra_relation_evidence_course_paths: tuple[str | Path, ...] = (),
        extra_relation_role_evidence_course_paths: tuple[str | Path, ...] = (),
        extra_relation_marker_evidence_course_paths: tuple[str | Path, ...] = (),
        extra_relation_answer_frame_course_paths: tuple[str | Path, ...] = (),
        extra_variant_course_paths: tuple[str | Path, ...] = (),
        extra_variant_evidence_paths: tuple[str | Path, ...] = (),
        extra_order_course_paths: tuple[str | Path, ...] = (),
        extra_order_evidence_paths: tuple[str | Path, ...] = (),
        ) -> dict[str, object]:
    """在同一 100 问上运行 baseline、trained 和 trained replay。"""
    project = Path(project_root).resolve()
    pack = Path(pack_dir).resolve()
    database = Path(database_path).resolve()
    training_root = Path(training_run_root).resolve()
    if any(path.drive.upper() != "K:" for path in (pack, database, training_root)):
        raise ConversationBroadQaTrainingContrastError(
            "pack、database、training run 必须位于 K 盘")
    if not database.is_file() or not training_root.is_dir():
        raise ConversationBroadQaTrainingContrastError("K 盘输入缺失")
    manifest, question_values, label_values, dimension_values = _verify_pack(pack)
    labels = {item["item_id"]: item for item in label_values}
    dimensions = {item["item_id"]: item for item in dimension_values}
    questions_artifacts = [
        item for item in manifest["artifacts"]
        if isinstance(item, dict) and item.get("role") == "dev_questions"
    ]
    if len(questions_artifacts) != 1:
        raise ConversationBroadQaTrainingContrastError(
            "interactive pack 必须恰有一个 dev_questions artifact")
    question_values = _select_questions(question_values, item_ids)
    selected_item_ids = tuple(str(item["item_id"]) for item in question_values)
    baseline = _run_once(database, question_values, labels, dimensions)
    training_courses = (*_frozen_training_course_paths(training_root, project), *tuple(
        Path(item).resolve() for item in extra_training_course_paths))
    if len(training_courses) != len(set(training_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra training course path 重复")
    training_pack = load_dialogue_training_pack(training_courses)
    evidence_model = learn_evidence_term_weights(training_pack)
    obligation_courses = (*training_courses, *tuple(
        Path(item).resolve() for item in extra_obligation_course_paths))
    if len(obligation_courses) != len(set(obligation_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra obligation course path 重复")
    obligation_model = learn_typed_obligations(obligation_courses)
    relation_courses = tuple(
        Path(item).resolve() for item in extra_relation_evidence_course_paths)
    if len(relation_courses) != len(set(relation_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra relation evidence course path 重复")
    relation_model = (
        learn_relation_evidence_model(relation_courses)
        if relation_courses else None)
    relation_role_courses = tuple(
        Path(item).resolve()
        for item in extra_relation_role_evidence_course_paths)
    if len(relation_role_courses) != len(set(relation_role_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra relation role evidence course path 重复")
    relation_role_model = (
        learn_relation_role_evidence_model(relation_role_courses)
        if relation_role_courses else None)
    relation_marker_courses = tuple(
        Path(item).resolve()
        for item in extra_relation_marker_evidence_course_paths)
    if len(relation_marker_courses) != len(set(relation_marker_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra relation marker evidence course path 重复")
    relation_marker_model = (
        learn_relation_marker_evidence_model(relation_marker_courses)
        if relation_marker_courses else None)
    relation_frame_courses = tuple(
        Path(item).resolve()
        for item in extra_relation_answer_frame_course_paths)
    if len(relation_frame_courses) != len(set(relation_frame_courses)):
        raise ConversationBroadQaTrainingContrastError(
            "extra relation answer frame course path 重复")
    relation_frame_model = (
        learn_relation_answer_frame_model(relation_frame_courses)
        if relation_frame_courses else None)
    observation, consumer_bundle = _consumer_factory(
        training_root, project, training_pack.pack_sha256,
        extra_variant_course_paths=tuple(
            Path(item).resolve() for item in extra_variant_course_paths),
        extra_variant_evidence_paths=tuple(
            Path(item).resolve() for item in extra_variant_evidence_paths),
        extra_order_course_paths=tuple(
            Path(item).resolve() for item in extra_order_course_paths),
        extra_order_evidence_paths=tuple(
            Path(item).resolve() for item in extra_order_evidence_paths),
    )
    consumer, used = consumer_bundle
    trained = _run_once(
        database, question_values, labels, dimensions,
        surface_consumer=consumer,
        learned_evidence_term_weights=evidence_model.weights,
        learned_typed_obligation=obligation_model,
        learned_relation_evidence_model=relation_model,
        learned_relation_role_evidence_model=relation_role_model,
        learned_relation_marker_evidence_model=relation_marker_model,
        learned_relation_answer_frame_model=relation_frame_model,
    )
    replay = _run_once(
        database, question_values, labels, dimensions,
        surface_consumer=consumer,
        learned_evidence_term_weights=evidence_model.weights,
        learned_typed_obligation=obligation_model,
        learned_relation_evidence_model=relation_model,
        learned_relation_role_evidence_model=relation_role_model,
        learned_relation_marker_evidence_model=relation_marker_model,
        learned_relation_answer_frame_model=relation_frame_model,
    )
    _assert_source_invariant(baseline, trained)
    if trained != replay:
        raise ConversationBroadQaTrainingContrastError(
            "trained 对照回放不一致")
    observations = []
    for before, after in zip(baseline, trained):
        observations.append({
            "display_changed": before.display_answer != after.display_answer,
            "item_id": before.item_id,
            "status": after.status,
            "source_title": after.source_title,
        })
    baseline_metrics = _metrics(baseline)
    trained_metrics = _metrics(trained)
    delta = {
        key: trained_metrics[key] - baseline_metrics[key]
        for key in (
            "complete_sentence_count", "evidence_gold_hit_count",
            "long_surface_count", "readable_surface_count",
            "surface_gold_hit_count", "question_count",
        )
        if isinstance(trained_metrics[key], int)
    }
    return {
        "artifact_kind": TRAINING_CONTRAST_KIND,
        "baseline": baseline_metrics,
        "database_sha256": _sha256(database),
        "delta": delta,
        "display_changed_count": sum(
            int(item["display_changed"]) for item in observations),
        "evidence_changed_count": sum(
            int(before.evidence_answer != after.evidence_answer)
            for before, after in zip(baseline, trained)),
        "evidence_learning_model_sha256": evidence_learning_sha256(
            evidence_model),
        "relation_evidence_model_sha256": (
            relation_evidence_sha256(relation_model)
            if relation_model is not None else None),
        "relation_evidence_course_paths": tuple(
            item.as_posix() for item in relation_courses),
        "relation_role_evidence_model_sha256": (
            relation_role_evidence_sha256(relation_role_model)
            if relation_role_model is not None else None),
        "relation_role_evidence_course_paths": tuple(
            item.as_posix() for item in relation_role_courses),
        "relation_marker_evidence_model_sha256": (
            relation_marker_evidence_sha256(relation_marker_model)
            if relation_marker_model is not None else None),
        "relation_marker_evidence_course_paths": tuple(
            item.as_posix() for item in relation_marker_courses),
        "relation_answer_frame_model_sha256": (
            relation_answer_frame_sha256(relation_frame_model)
            if relation_frame_model is not None else None),
        "relation_answer_frame_course_paths": tuple(
            item.as_posix() for item in relation_frame_courses),
        "typed_obligation_model_sha256": typed_obligation_sha256(
            obligation_model),
        "typed_obligation_course_paths": tuple(
            item.as_posix() for item in obligation_courses),
        "format_version": TRAINING_CONTRAST_FORMAT_VERSION,
        "manifest_sha256": _sha256(pack / "manifest.json"),
        "observations": observations,
        "questions_sha256": questions_artifacts[0]["sha256"],
        "selected_item_ids": selected_item_ids,
        "selected_questions_sha256": _selection_sha256(question_values),
        "replay_bit_identical": True,
        "trained": trained_metrics,
        "trained_surface_consumer_used_count": used[0],
        "training_observation": observation.to_dict(),
        "training_pack_sha256": training_pack.pack_sha256,
        "training_run_id": observation.run_id,
    }


def write_conversation_broad_qa_training_contrast(
        value: dict[str, object], output_path: str | Path) -> str:
    """只创建 K 盘 canonical 对照报告，不覆盖历史产物。"""
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ConversationBroadQaTrainingContrastError(
            "contrast output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_line(value))
    return str(output)


__all__ = [
    "ConversationBroadQaTrainingContrastError",
    "build_conversation_broad_qa_training_contrast",
    "write_conversation_broad_qa_training_contrast",
]
