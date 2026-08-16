"""把 grounded-answer TRAIN episode 分账为统一 PH2 资料记录。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    FORMAT_VERSION,
    SCHEMA_VERSION,
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetBundleValidationReport,
    validate_dataset_bundle,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    LICENSE_ID,
    GroundedAnswerEpisode,
    read_grounded_answer_episodes,
    verify_surface_realization,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
STAGE = "W-09"
SUBSTAGE = "GROUNDED_ANSWER_Q0"
SNAPSHOT_ID = "grounded-answer-train-v1"
OFFICIAL_URL = "urn:pure-integer-ai:ph2:grounded-answer-train-v1"
ATTRIBUTION = "Pure Integer AI authored grounded answer training examples"


def _record_key(namespace: str, *parts: Any) -> StableRecordKey:
    """从规范值生成版本化正整数资料身份。"""
    payload = canonical_json_bytes({
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


def _sha256_file(path: Path) -> str:
    """流式计算公开 sample 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _observation_payload(episode: GroundedAnswerEpisode) -> dict[str, object]:
    """只保留学生可见问题、Evidence、scope 和会话，不泄漏训练标签。"""
    question = episode.question
    return {
        "context_surface": question.context_surface,
        "dialogue": episode.dialogue.to_dict(),
        "episode_id": episode.episode_id,
        "evidence": [item.to_dict() for item in question.evidence],
        "evidence_scope_id": question.evidence_scope_id,
        "question_surface": question.question_surface,
        "response_scope_id": question.response_scope_id,
        "split_clusters": episode.clusters.to_dict(),
    }


def _teacher_payload(episode: GroundedAnswerEpisode) -> dict[str, object]:
    """把 intent、plan、合法多表面和负例维度隔离在 teacher owner。"""
    accepted = []
    for realization in episode.surfaces.accepted:
        accepted.append({
            "realization": realization.to_dict(),
            "verification": verify_surface_realization(
                episode.question, realization).to_dict(),
        })
    rejected = []
    for item in episode.surfaces.rejected:
        rejected.append({
            "realization": item.realization.to_dict(),
            "verification": verify_surface_realization(
                episode.question, item.realization).to_dict(),
        })
    payload = {
        "answer_plan": episode.question.answer_plan.to_dict(),
        "minimum_legal_surfaces": episode.surfaces.minimum_legal_surfaces,
        "surface_realizations": {
            "accepted": accepted,
            "rejected": rejected,
        },
        "typed_intent": episode.question.typed_intent,
    }
    if episode.reference_course is not None:
        payload["reference_course"] = episode.reference_course.to_dict()
    return payload


def _sample_role(episode: GroundedAnswerEpisode) -> str:
    """把 response act 映射到既有统一资料 sample role。"""
    return {
        "ANSWER": "support",
        "UNKNOWN": "refute",
        "CLARIFY": "anomaly",
        "CONFLICT": "conflict",
    }[episode.question.answer_plan.response_act]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerTrainingBundle:
    """返回 TRAIN-only 的来源、学生 Observation、teacher Evidence 与审计。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]
    validation: DatasetBundleValidationReport


def compile_grounded_answer_training_records(
        sample_path: str | Path,
        ) -> GroundedAnswerTrainingBundle:
    """严格回读公开 sample，并构造不含 evaluator/held-out 的训练分账。"""
    path = Path(sample_path).resolve()
    episodes = read_grounded_answer_episodes(path, train_only=True)
    sample_sha256 = _sha256_file(path)
    dataset_key = _record_key("dataset", "PH2", SOURCE_KEY, SCHEMA_VERSION)
    artifact_key = _record_key(
        "artifact", SOURCE_KEY, SUBSTAGE, sample_sha256, 1)
    owner_key = _record_key("owner", SOURCE_KEY, SUBSTAGE, "teacher")
    sources = []
    observations = []
    teachers = []
    relative_path = f"data/ph2/{path.name}"
    for ordinal, episode in enumerate(episodes, start=1):
        source_key = _record_key(
            "source_ref", SOURCE_KEY, sample_sha256, episode.episode_id)
        observation_key = _record_key(
            "observation", SOURCE_KEY, SUBSTAGE, sample_sha256,
            episode.episode_id)
        source = SourceRefRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            1,
            dataset_key,
            artifact_key,
            source_key,
            SOURCE_KEY,
            SNAPSHOT_ID,
            episode.episode_id,
            OFFICIAL_URL,
            f"{relative_path}#{episode.episode_id}",
            "sha256:" + sample_sha256,
            sample_sha256,
            LICENSE_ID,
            "PUBLIC",
            ATTRIBUTION,
            1,
            CanonicalJsonObject.from_value({
                "line_end": ordinal,
                "line_start": ordinal,
                "relative_path": relative_path,
            }),
            ordinal,
            _record_key(
                "source_cluster", SUBSTAGE, episode.clusters.source),
        )
        observation = ObservationRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            1,
            dataset_key,
            artifact_key,
            observation_key,
            STAGE,
            SUBSTAGE,
            episode.split,
            "zh",
            "typed-grounded-answer",
            source.stable_key,
            LICENSE_ID,
            _record_key(
                "dedup", SUBSTAGE,
                episode.clusters.source,
                episode.clusters.proposition,
                episode.clusters.question_construction,
                episode.clusters.paraphrase,
            ),
            _record_key(
                "content", SUBSTAGE, episode.clusters.proposition),
            _record_key(
                "template", SUBSTAGE,
                episode.clusters.question_construction),
            _record_key(
                "shape", SUBSTAGE, episode.clusters.paraphrase),
            "forming",
            _sample_role(episode),
            "GroundedQuestionObservationV1",
            CanonicalJsonObject.from_value(_observation_payload(episode)),
            "NONE",
            None,
            (),
            ordinal,
        )
        teacher = TeacherEvidenceRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            1,
            dataset_key,
            artifact_key,
            _record_key(
                "teacher_evidence", SUBSTAGE, episode.episode_id),
            observation.stable_key,
            "GROUNDED_ANSWER_PLAN_AND_SURFACE_LABEL",
            CanonicalJsonObject.from_value(_teacher_payload(episode)),
            source.stable_key,
            STAGE,
            0,
            owner_key,
        )
        sources.append(source)
        observations.append(observation)
        teachers.append(teacher)
    source_tuple = tuple(sources)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teachers)
    validation = validate_dataset_bundle(
        source_tuple,
        observation_tuple,
        teacher_tuple,
        (),
        source_key=SOURCE_KEY,
        license_partition=LICENSE_ID,
        public_release=True,
    )
    return GroundedAnswerTrainingBundle(
        source_tuple, observation_tuple, teacher_tuple, validation)


__all__ = [
    "ATTRIBUTION",
    "GroundedAnswerTrainingBundle",
    "OFFICIAL_URL",
    "SNAPSHOT_ID",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "compile_grounded_answer_training_records",
]
