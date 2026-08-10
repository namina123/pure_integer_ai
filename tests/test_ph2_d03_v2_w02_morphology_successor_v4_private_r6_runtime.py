"""W02 V4-first R6 evaluator 的无 private synthetic 测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    run_w02_morphology_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    run_w02_morphology_successor_v2_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_artifact import (
    publish_w02_morphology_successor_v4_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_language_overlay import (
    build_w02_morphology_successor_v4_from_counts,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_private_r6_runtime import (
    evaluate_w02_morphology_successor_v4_private_r6_pair_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_public_probe import (
    W02MorphologySuccessorV4PublicTraining,
)


def _digest(value: str) -> str:
    """为 synthetic SourceRef 生成稳定 SHA-256。"""
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parent_source(split: str, ordinal: int, form: str):
    """构造冻结 compiler 所需的完整公开 synthetic SourceRef。"""
    return _source_record(
        "UD_ZH_GSDSIMP_R2_18", split, ordinal,
        snapshot_id="r6-synthetic-parent",
        revision_id="r6-synthetic-parent-revision",
        official_url=(
            "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
        source_identity=f"r6-synthetic-parent:{split}:{ordinal}",
        upstream_checksum="sha256:" + _digest(f"upstream:{split}:{ordinal}"),
        local_sha256=_digest(f"local:{split}:{ordinal}"),
        license_id="CC-BY-SA-4.0",
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value=str(ordinal),
        span_end=len(form),
    )


def _expected(form: str, lemma: str, upos: str) -> dict[str, object]:
    """构造只含公开 synthetic morphology 的 W02 UD expected。"""
    return {
        "boundary_spans": [{"end": len(form), "form": form, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [], "form": form, "lemma": lemma,
            "node_id": [1], "upos": upos,
        }],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _training_pairs() -> tuple[tuple[object, object], ...]:
    """建立现代汉语 parent fixture，使古汉语命中只能来自 V4。"""
    pairs = []
    rows = (
        ("猫化", "猫"), ("纸化", "纸"), ("木化", "木"),
        ("石化", "石"), ("新词", "新词"),
    )
    for ordinal, (form, lemma) in enumerate(rows, start=1):
        source = _parent_source("train", ordinal, form)
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="support", perturbation_kind="NONE")
        evidence = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source, observation,
            _expected(form, lemma, "VERB"),
            dimension_name=W02_DEV_DIMENSIONS[2])
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))
    return tuple(pairs)


@pytest.fixture(scope="module")
def artifact_roots(tmp_path_factory: pytest.TempPathFactory):
    """发布 Candidate/V1/V2 与一个只含 synthetic lzh 的 V4 artifact。"""
    root = tmp_path_factory.mktemp("w02-r6-v4")
    candidate = run_w02_candidate_fixture(
        fixture_root=root / "candidate", pairs=_training_pairs(),
        run_id=1, requested_workers=2, mode="fresh")
    v1 = run_w02_morphology_overlay_fixture(
        fixture_root=root / "v1", candidate_artifact_root=candidate.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    v2 = run_w02_morphology_successor_v2_overlay_fixture(
        fixture_root=root / "v2", candidate_artifact_root=candidate.artifact_path,
        v1_overlay_artifact_root=v1.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    index = build_w02_morphology_successor_v4_from_counts((
        ("lzh", "爰", "於", "ADV", "[]", 5),
        ("lzh", "既", "既", "ADV", "[]", 5),
    ))
    training = W02MorphologySuccessorV4PublicTraining(
        index, 1, 10, 2, 2, ({"relative_path": "synthetic"},))
    v4 = publish_w02_morphology_successor_v4_artifact(
        training, root / "v4", run_id=1)
    return candidate.artifact_path, v1.artifact_path, v2.artifact_path, v4.artifact_path


def _private_row():
    """构造一个原始 lzh observation、label 与一致 route capability。"""
    form = "爰"
    parent = _parent_source("held_out", 1, form)
    source = replace(
        parent,
        dataset_key=StableRecordKey((97, 1)),
        artifact_key=StableRecordKey((97, 2)),
        stable_key=StableRecordKey((97, 10, 1)),
        source_cluster_key=StableRecordKey((97, 11, 1)),
        source_key="UD_LZH_TUECL_R2_18_TOKEN_SPAN_BLIND_PRIVATE",
        snapshot_id="ud-lzh-tuecl-r2.18-test-token-span-r6",
        revision_id="0d35ec4b78bba618ff621b63c57fe9542ab61240",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-TueCL"),
        upstream_checksum="sha1:9b93e591c7747758badff15051de31fb465a2cd0",
        source_identity="synthetic-r6-token-span")
    old_observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, parent,
        carrier_kind="plain_text", surface=form, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
    observation = replace(
        old_observation,
        dataset_key=source.dataset_key,
        artifact_key=source.artifact_key,
        stable_key=StableRecordKey((97, 20, 1)),
        source_ref_key=source.stable_key,
        language="lzh")
    old_label = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, parent, old_observation,
        _expected(form, "於", "ADV"),
        dimension_name=W02_DEV_DIMENSIONS[2])
    assert isinstance(old_label, EvaluatorLabelRecord)
    label = replace(
        old_label,
        dataset_key=source.dataset_key,
        artifact_key=source.artifact_key,
        stable_key=StableRecordKey((97, 40, 1)),
        observation_key=observation.stable_key)
    capability = w02_ud_morphology_source_capability({
        "annotation_provenance": "synthetic manual UD annotation",
        "commit_sha1": source.revision_id,
        "data_file": {"git_blob_sha1": source.upstream_checksum.removeprefix("sha1:")},
        "language": "lzh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": source.source_key,
    })
    return source, observation, label, capability


def test_r6_evaluator_consumes_v4_exact_morphology(artifact_roots) -> None:
    """new-content morphology 必须由 lzh V4 exact lexeme 真正命中。"""
    source, observation, label, capability = _private_row()
    report = evaluate_w02_morphology_successor_v4_private_r6_pair_stream(
        *artifact_roots, (source,), (capability,), ((observation, label),),
        V2EvaluatorResourceBudget(16, 1_000_000, 10_000_000, 10_000, 10_000, 1))
    morphology = next(
        row for row in report["dimension_results"]
        if row["dimension_key"] == W02_DEV_DIMENSIONS[2])
    assert morphology["status"] == "PASS"
    assert morphology["numerator"] == 1
    assert report["v4_exact_candidate_count"] >= 1
    assert report["base_language_original_lzh_count"] == 1
    assert report["route_authorized_count"] == 1
