from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.integer_relation_evaluation_graph import (
    EVALUATION_SPLITS,
    EvaluationGraphAssignment,
    IntegerRelationEvaluationGraphSplitCourse,
    compile_w06_evaluation_graph_splits,
)
from pure_integer_ai.experiments.integer_relation_graph_admission import (
    IntegerRelationGraphAdmissionError,
    build_integer_relation_graph_admission_splits,
    validate_integer_relation_graph_admission,
)
from pure_integer_ai.experiments.integer_relation_graph_split_course import (
    IntegerRelationGraphSplitCase,
    IntegerRelationGraphSplitCourse,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from tests.test_integer_graph_novelty import _chain, _event, _record


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "ph2_w06_dataset_artifacts" / "source_semantic_overlay_v1" / "packs"
OWNER = (99180, 1)
SOURCE = (99181, 1)
LICENSE = (99182, 1)


def _train_course() -> IntegerRelationGraphSplitCourse:
    records = _chain(100, 100)
    case = IntegerRelationGraphSplitCase(
        (99183, 1), tuple(map(ord, "甲乙先后")), ((99184, 1),),
        tuple(row.observation_key for row in records),
    )
    return IntegerRelationGraphSplitCourse(
        "train", OWNER, SOURCE, LICENSE, (case,),
        tuple(sorted(records, key=lambda row: row.stable_key())),
    )


def _evaluation_courses(
        *, include_schema_rejection: bool = False,
        ) -> dict[str, IntegerRelationEvaluationGraphSplitCourse]:
    manifest_path = next(PACK_ROOT.glob(
        "AUTHORED_CC0_V1--CC0-1.0--causes-v1/manifest.json"))
    manifest = read_artifact_manifest(manifest_path)
    sources = []
    observations = []
    for identity in manifest.files:
        if identity.record_kind not in (RECORD_SOURCE_REF, RECORD_OBSERVATION):
            continue
        records = read_record_artifact(manifest_path.parent, identity)
        if identity.record_kind == RECORD_SOURCE_REF:
            sources.extend(records)
        else:
            observations.extend(
                item for item in records
                if item.w_stage == "W-06" and item.payload_kind == "TypedRelationQuery"
                and item.split == "held_out")
    observations = tuple(sorted(observations, key=lambda item: item.stable_key))
    assert len(observations) >= len(EVALUATION_SPLITS)
    selected = observations[:len(EVALUATION_SPLITS)]
    if include_schema_rejection:
        mismatch = None
        for candidate_manifest in sorted(PACK_ROOT.glob(
                "AUTHORED_CC0_V1--CC0-1.0--*/manifest.json")):
            candidate = read_artifact_manifest(candidate_manifest)
            for identity in candidate.files:
                if identity.record_kind == RECORD_OBSERVATION:
                    rows = read_record_artifact(candidate_manifest.parent, identity)
                    mismatch = next((item for item in rows
                                     if item.w_stage == "W-06"
                                     and item.payload_kind == "TypedRelationQuery"
                                     and item.split == "train"
                                     and item.perturbation_kind == "TYPE_MISMATCH"),
                                    None)
                    if mismatch is not None:
                        break
            if mismatch is not None:
                for identity in candidate.files:
                    if identity.record_kind == RECORD_SOURCE_REF:
                        sources.extend(read_record_artifact(
                            candidate_manifest.parent, identity))
                break
        assert mismatch is not None
        selected = (
            replace(mismatch, split="held_out"),
            observations[0],
            observations[1],
        )
    sources = tuple({item.stable_key: item for item in sources}.values())

    assignments = {split: [] for split in EVALUATION_SPLITS}
    for index, (split, observation) in enumerate(
            zip(EVALUATION_SPLITS, selected, strict=True), start=1):
        assignments[split].append(EvaluationGraphAssignment(
            split,
            (99185, index),
            tuple(map(ord, f"评测查询{index}")),
            ((99186, index),),
            (tuple(observation.stable_key.components),),
        ))
    return compile_w06_evaluation_graph_splits(
        selected,
        tuple(sources),
        {split: tuple(cases) for split, cases in assignments.items()},
        owner_identity=OWNER,
        source_identity=SOURCE,
        license_identity=LICENSE,
    )


def test_admission_joins_train_labels_only_with_label_free_evaluation():
    train = _train_course()
    evaluation = _evaluation_courses()

    splits = build_integer_relation_graph_admission_splits(train, evaluation)
    assert set(splits) == {"train", *EVALUATION_SPLITS}
    report = validate_integer_relation_graph_admission(
        train, evaluation,
        minimum_novel_held_out=1,
        minimum_relation_families=1,
    )
    assert report["split_course_contract"] == "PASS_TRAIN_AND_LABEL_FREE_EVALUATION"
    assert report["evaluation_record_contract"] == "PASS_LABEL_FREE_TYPED_RECORDS"
    assert report["source_license_qualification"].startswith("UNKNOWN")
    assert report["upstream_cluster_qualification"].startswith("UNKNOWN")
    assert report["operation_qualification"].startswith("UNKNOWN")
    assert report["label_owner_qualification"].startswith("UNKNOWN")
    assert report["unauthorized_qualification"].startswith("UNKNOWN")
    assert report["free_dialogue_complete"] == 0


def test_admission_rejects_legacy_label_bearing_eval_course():
    train = _train_course()
    evaluation = _evaluation_courses()
    records = (_record(900, _event(900, 1), _event(900, 2)),)
    case = IntegerRelationGraphSplitCase(
        (99187, 1), tuple(map(ord, "旧格式评测")), ((99188, 1),),
        (records[0].observation_key,),
    )
    legacy = IntegerRelationGraphSplitCourse(
        "held_out", OWNER, SOURCE, LICENSE, (case,), records)
    evaluation["held_out"] = legacy  # type: ignore[assignment]

    with pytest.raises(IntegerRelationGraphAdmissionError, match="label-free"):
        build_integer_relation_graph_admission_splits(train, evaluation)


def test_admission_rejects_identity_drift_and_graphless_eval_case():
    train = _train_course()
    evaluation = _evaluation_courses()
    evaluation["negative"] = IntegerRelationEvaluationGraphSplitCourse(
        "negative", OWNER, SOURCE, (99999, 1), evaluation["negative"].cases)
    with pytest.raises(IntegerRelationGraphAdmissionError, match="identity drifted"):
        build_integer_relation_graph_admission_splits(train, evaluation)

    evaluation = _evaluation_courses(include_schema_rejection=True)
    assert not evaluation["held_out"].cases[0].records
    assert len(evaluation["held_out"].cases[0].rejections) == 1
    with pytest.raises(IntegerRelationGraphAdmissionError, match="no graph-accepted"):
        build_integer_relation_graph_admission_splits(train, evaluation)
