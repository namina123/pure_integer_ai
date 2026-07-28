"""D-02F train/dev/held-out/adversarial 物理隔离只读 probe pack。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    FORMAT_VERSION,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    SCHEMA_VERSION,
    SPLITS,
    ArtifactManifest,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetBundleValidationReport,
    validate_artifact_manifest,
    validate_dataset_bundle,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
PACK_NAME = "PILOT_CC0_V1--CC0-1.0--split-isolation-probe-v1"
STAGE = "W-01"
SUBSTAGE = "DATASET_SPLIT_ISOLATION"


def _key(namespace: str, *parts: Any) -> StableRecordKey:
    """从规范 probe 定义产生版本化正整数稳定键。"""
    payload = canonical_json_bytes({
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


PROBE_DEFINITION = {
    "artifact_version": 1,
    "course_version": 1,
    "license_id": LICENSE_ID,
    "pack_name": PACK_NAME,
    "source_key": SOURCE_KEY,
    "splits": list(SPLITS[:4]),
    "stage": STAGE,
    "substage": SUBSTAGE,
}
PROBE_INPUT_SHA256 = hashlib.sha256(
    canonical_json_bytes(PROBE_DEFINITION)).hexdigest()


@dataclass(frozen=True)
class PilotSplitProbeBuild:
    """返回 split probe 的发布目录、manifest 和 bundle 校验结果。"""

    pack_root: Path
    manifest: ArtifactManifest
    validation: DatasetBundleValidationReport


def _clusters_for(
        observations: tuple[ObservationRecord, ...],
        sources: tuple[SourceRefRecord, ...],
        ) -> tuple[StableRecordKey, ...]:
    """返回一组 Observation 真实引用的来源簇。"""
    index = {source.stable_key: source for source in sources}
    return tuple(sorted({
        index[observation.source_ref_key].source_cluster_key
        for observation in observations
    }))


def compile_pilot_split_probe(
        release_root: str | Path) -> PilotSplitProbeBuild:
    """发布四 split 独立文件/cluster/owner 的只读极小 probe。"""
    dataset_key = _key("dataset", "PH2", "D-02F", SOURCE_KEY)
    artifact_key = _key(
        "artifact", "PH2", "D-02F", PACK_NAME, PROBE_INPUT_SHA256)
    teacher_owner = _key("owner", PACK_NAME, "teacher")
    evaluator_owner = _key("owner", PACK_NAME, "evaluator")
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    for ordinal, split in enumerate(SPLITS[:4], start=1):
        source_key = _key("source", PACK_NAME, split)
        source_cluster = _key("source_cluster", PACK_NAME, split)
        source = SourceRefRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            1,
            dataset_key,
            artifact_key,
            source_key,
            SOURCE_KEY,
            "d02f-split-probe-v1",
            split,
            "urn:pure-integer-ai:ph2:d02f-split-probe-v1",
            f"pilot/split/{split}",
            "sha256:" + PROBE_INPUT_SHA256,
            PROBE_INPUT_SHA256,
            LICENSE_ID,
            "PUBLIC",
            "Pure Integer AI PH2 D-02F authored split isolation probe",
            1,
            CanonicalJsonObject.from_value({
                "probe": "D-02F",
                "record_ordinal": ordinal,
                "split": split,
            }),
            ordinal,
            source_cluster,
        )
        observation_key = _key("observation", PACK_NAME, split)
        observation = ObservationRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            1,
            dataset_key,
            artifact_key,
            observation_key,
            STAGE,
            SUBSTAGE,
            split,
            "zh",
            "typed-dataset-split-probe",
            source_key,
            LICENSE_ID,
            _key("dedup", PACK_NAME, split),
            _key("content", PACK_NAME, split),
            _key("template", PACK_NAME, split),
            _key("shape", PACK_NAME, split),
            "forming" if split == "train" else "evaluator",
            "read_only_probe",
            "DatasetSplitIsolationProbe",
            CanonicalJsonObject.from_value({
                "authoritative_answer_present": 0,
                "mastery_signal_present": 0,
                "split": split,
                "training_state_write_allowed": 0,
            }),
            "SPLIT_ISOLATION",
            None,
            (),
            ordinal,
        )
        sources.append(source)
        observations.append(observation)
        if split == "train":
            teachers.append(TeacherEvidenceRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                1,
                dataset_key,
                artifact_key,
                _key("teacher", PACK_NAME, split),
                observation_key,
                "SPLIT_OWNER_REVEAL",
                CanonicalJsonObject.from_value({
                    "owner": "teacher",
                    "split": split,
                    "training_state_write_allowed": 0,
                }),
                source_key,
                STAGE,
                0,
                teacher_owner,
            ))
        else:
            evaluators.append(EvaluatorLabelRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                1,
                dataset_key,
                artifact_key,
                _key("evaluator", PACK_NAME, split),
                observation_key,
                _key("dimension", PACK_NAME, split),
                "TRUE",
                CanonicalJsonObject.from_value({
                    "expected_physical_split": split,
                    "owner": "evaluator",
                    "training_state_write_allowed": 0,
                }),
                1,
                1,
                STAGE,
                evaluator_owner,
            ))
    source_tuple = tuple(sources)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teachers)
    evaluator_tuple = tuple(evaluators)
    validation = validate_dataset_bundle(
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
        source_key=SOURCE_KEY,
        license_partition=LICENSE_ID,
        public_release=True,
    )
    pack_root = Path(release_root).resolve() / "packs" / PACK_NAME
    if pack_root.exists():
        raise RuntimeError("D-02F split probe 已存在，禁止覆盖")
    all_clusters = tuple(sorted(
        source.source_cluster_key for source in source_tuple))
    files = [write_record_artifact(
        source_tuple,
        pack_root,
        ArtifactWriteSpec(
            RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
            LICENSE_ID, all_clusters,
        ),
    )]
    for split in SPLITS[:4]:
        split_observations = tuple(
            item for item in observation_tuple if item.split == split)
        files.append(write_record_artifact(
            split_observations,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                f"observations/{split}.jsonl.gz", split,
                LICENSE_ID, _clusters_for(split_observations, source_tuple),
            ),
        ))
    train = tuple(item for item in observation_tuple if item.split == "train")
    files.append(write_record_artifact(
        teacher_tuple,
        pack_root,
        ArtifactWriteSpec(
            RECORD_TEACHER_EVIDENCE, "teacher",
            "owners/teacher/train.evidence.jsonl.gz", "train",
            LICENSE_ID, _clusters_for(train, source_tuple),
        ),
    ))
    for split in SPLITS[1:4]:
        split_observations = tuple(
            item for item in observation_tuple if item.split == split)
        labels = tuple(
            item for item in evaluator_tuple
            if item.observation_key == split_observations[0].stable_key)
        files.append(write_record_artifact(
            labels,
            pack_root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL, "evaluator",
                f"owners/evaluator/{split}.labels.jsonl.gz", split,
                LICENSE_ID, _clusters_for(split_observations, source_tuple),
            ),
        ))
    manifest = ArtifactManifest(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        1,
        1,
        dataset_key,
        artifact_key,
        SOURCE_KEY,
        LICENSE_ID,
        "PUBLIC",
        1,
        1,
        1,
        tuple(files),
        SPLITS[:4],
        (STAGE,),
        all_clusters,
        (),
        STAGE,
    )
    validate_artifact_manifest(manifest, source_tuple, observation_tuple)
    write_artifact_manifest(manifest, pack_root)
    return PilotSplitProbeBuild(pack_root, manifest, validation)


__all__ = [
    "LICENSE_ID",
    "PACK_NAME",
    "PROBE_DEFINITION",
    "PROBE_INPUT_SHA256",
    "PilotSplitProbeBuild",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "compile_pilot_split_probe",
]
