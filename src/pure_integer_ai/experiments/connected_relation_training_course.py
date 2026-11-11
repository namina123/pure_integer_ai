"""独立 train-only 来源、教师事实与开放问式课程；不消费 evaluator。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)
from pure_integer_ai.experiments.connected_relation_question_course import (
    compile_train_relation_question,
)
from pure_integer_ai.experiments.integer_relation_connected_course import (
    _CASE_FIELDS, _FACT_FIELDS, _SPEC, _read_lines, _seed,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourse, integer_course_from_w06_adapter,
)
from pure_integer_ai.experiments.integer_relation_source_audit import (
    audit_integer_relation_source_pack,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild, publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import compile_relation_seed
from pure_integer_ai.experiments.ph2_authored_relation_schema import AuthoredRelationSeed
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION, RECORD_SOURCE_REF, RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w06_adapter import adapt_w06_training_payload
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_registry import W06_RELATION_REGISTRY
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES


@dataclass(frozen=True, slots=True)
class RelationTrainingCompilation:
    """保留来源审计、纯整数教师课程和问式出处；不授权整波生产。"""

    pack: AuthoredCourseBuild
    seeds: tuple[AuthoredRelationSeed, ...]
    train_course: IntegerRelationGraphCourse
    question_frames: tuple[TrainedRelationQuestionFrame, ...]
    source_audit: dict[str, object]
    question_audit: dict[str, object]


def _digest_key(path: Path) -> tuple[int, ...]:
    """将完整 SHA-256 保存为八个有界整数，避免只以截断摘要作来源本体。"""
    digest = hashlib.sha256(path.read_bytes()).digest()
    return tuple(int.from_bytes(digest[index:index + 4], "big")
                 for index in range(0, len(digest), 4))


def compile_relation_training_course(
        source_path: str | Path, question_path: str | Path,
        release_root: str | Path,
        ) -> RelationTrainingCompilation:
    """审核独立教师来源后编译 train-only 事实及多来源问式，不读取 held-out。"""
    source = Path(source_path).resolve(strict=True)
    questions = Path(question_path).resolve(strict=True)
    rows = _read_lines(source, _FACT_FIELDS)
    cases = _read_lines(questions, _CASE_FIELDS)
    seeds = tuple(_seed(row, number) for number, row in enumerate(rows, 1))
    if any(seed.split != "train" or seed.label_owner != "teacher" for seed in seeds):
        raise ValueError("train-only course rejects evaluator observations")
    by_id = {seed.seed_id: seed for seed in seeds}
    if len(by_id) != len(seeds):
        raise ValueError("train-only source seed ids repeat")
    names_by_identity = {}
    for seed in seeds:
        for endpoint in seed.endpoints:
            key = (endpoint.object_kind, endpoint.local_id)
            name = names_by_identity.setdefault(key, endpoint.surface_fragment)
            if name != endpoint.surface_fragment:
                raise ValueError("train-only endpoint identity has competing names")
    case_ids = set()
    question_key = _digest_key(questions)
    frames = []
    for row in cases:
        if (row["split"] != "train" or row["operation"] != "ASSERT_RELATION"
                or not isinstance(row["case_id"], str) or not row["case_id"]
                or row["case_id"] in case_ids
                or not isinstance(row["input"], str) or not row["input"]
                or not isinstance(row["seed_ids"], list) or len(row["seed_ids"]) != 1
                or row["seed_ids"][0] not in by_id):
            raise ValueError("train-only question source or operation is unqualified")
        case_ids.add(row["case_id"])
        frame = compile_train_relation_question(
            by_id[row["seed_ids"][0]], row["input"],
            source_case_key=(92029, *question_key, len(frames) + 1),
            operation_key=(92004, 1))
        if frame is None:
            raise ValueError("train-only question has no complete teacher-qualified role slots")
        frames.append(frame)
    contracts = {}
    for frame in frames:
        contracts.setdefault((frame.operation_key, frame.predicate_key, frame.roles),
                             set()).add(frame.gaps)
    if any(len(gaps) < 2 for gaps in contracts.values()):
        raise ValueError("every train question contract needs two distinct surface forms")

    spec = replace(
        _SPEC, pack_name="AUTHORED_CC0_V1--CC0-1.0--relation-language-train-v1",
        substage="RELATION_LANGUAGE_TRAIN_V1",
        snapshot_id="relation-language-train-authoring-v1",
        official_url="urn:pure-integer-ai:authored:relation-language-train-v1",
        attribution="Pure Integer AI original CC0 fictional relation language course",
    )
    substages = {
        seed.seed_id: (
            W06_RELATION_PROFILES[seed.relation_family].substage_key,
            W06_RELATION_REGISTRY[
                W06_RELATION_PROFILES[seed.relation_family].substage_key
            ].teacher_evidence_kinds[0])
        for seed in seeds
    }
    pack = publish_authored_course(
        tuple(compile_relation_seed(seed, semantic_identity=True) for seed in seeds),
        source, release_root, spec, seed_substages=substages, training_only=True)
    audit = audit_integer_relation_source_pack(
        pack.pack_root, source, source_row_parser=_seed, training_only=True)
    if (audit["status"] != "PASS_SOURCE_PACK_ONLY"
            or audit["evaluator_count"] != 0
            or audit["teacher_count"] != len(seeds)):
        raise ValueError("train-only source audit did not close teacher ownership")
    records = {kind: [] for kind in (
        RECORD_SOURCE_REF, RECORD_OBSERVATION, RECORD_TEACHER_EVIDENCE)}
    for identity in pack.manifest.files:
        if identity.record_kind not in records:
            raise ValueError("train-only artifact contains an evaluation owner")
        records[identity.record_kind].extend(read_record_artifact(pack.pack_root, identity))
    adapter = adapt_w06_training_payload(W06TrainingPayload(
        tuple(records[RECORD_SOURCE_REF]), tuple(records[RECORD_OBSERVATION]),
        tuple(records[RECORD_TEACHER_EVIDENCE])), require_full_curriculum=False)
    source_identity = (92030, *pack.manifest.stable_key.components, *question_key)
    license_identity = tuple(map(ord, spec.license_id))
    course = integer_course_from_w06_adapter(
        adapter, source_identity=source_identity, license_identity=license_identity)
    return RelationTrainingCompilation(
        pack, seeds, course, tuple(frames), audit, {
            "status": "PASS_TRAIN_SOURCE_ROLE_SYNTAX_ONLY",
            "question_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
            "frame_count": len(frames), "contract_count": len(contracts),
            "source_case_keys": [list(frame.source_case_key) for frame in frames],
            "evaluator_read_count": 0, "production_authorized": 0,
            "free_dialogue_complete": 0,
        })


__all__ = ["RelationTrainingCompilation", "compile_relation_training_course"]
