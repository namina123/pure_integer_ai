"""Synthetic V6-first and language-family tests for the R5 evaluator."""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import inspect
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    KYOTO_REMAINDER_MINIMUM_ORDINAL,
    KYOTO_REMAINDER_SOURCE_KEY,
    blind_private_source_specs_v6,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import V2AccessPermit
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r5 import (
    W02_MORPH_V3_PRIVATE_R5_EVALUATOR_VERSION,
    W02MorphologySuccessorV3PrivateR5EvaluationError,
    evaluate_w02_morphology_successor_v3_private_r5_pair_stream,
    run_w02_morphology_successor_v3_private_r5_evaluation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_io_r5 import (
    read_and_close_w02_morphology_successor_v3_private_r5_sources,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_contract import (
    W02MorphologySuccessorV3PrivateR5FileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    w02_ud_morphology_source_capability,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected(form: str, *, lemma: str | None = None, upos: str = "VERB"):
    return {
        "boundary_spans": [{"end": len(form), "form": form, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [], "form": form, "lemma": lemma or form,
            "node_id": [1], "upos": upos,
        }],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _training_pairs():
    rows = (("猫化", "猫"), ("纸化", "纸"), ("木化", "木"),
            ("石化", "石"), ("新词", "新词"))
    pairs = []
    for ordinal, (form, lemma) in enumerate(rows, start=1):
        source = _source_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal,
            snapshot_id="r5-language-family-train",
            revision_id="r5-language-family-train-revision",
            official_url=(
                "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
            source_identity=f"r5-language-family-train:{ordinal}",
            upstream_checksum="sha256:" + _sha(f"upstream:{ordinal}"),
            local_sha256=_sha(f"local:{ordinal}"),
            license_id="CC-BY-SA-4.0", attribution="public synthetic fixture",
            locator_kind="record", locator_value=str(ordinal),
            span_end=len(form))
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="support", perturbation_kind="NONE")
        evidence = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source, observation,
            _expected(form, lemma=lemma,
                      upos="VERB" if form.endswith("化") else "NOUN"),
            dimension_name=W02_DEV_DIMENSIONS[2])
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))
    return tuple(pairs)


def _lzh_private_rows():
    form = "新化"
    parent = _source_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1,
        snapshot_id="r5-language-family-parent", revision_id="parent-revision",
        official_url=(
            "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
        source_identity="r5-language-family-parent:1",
        upstream_checksum="sha256:" + "1" * 64,
        local_sha256="2" * 64, license_id="CC-BY-SA-4.0",
        attribution="public synthetic fixture", locator_kind="record",
        locator_value="1", span_end=len(form))
    dataset_key = StableRecordKey((9, 9, 7, 1))
    artifact_key = StableRecordKey((9, 9, 7, 2))
    source_key = StableRecordKey((9, 9, 7, 10, 1))
    source = replace(
        parent, dataset_key=dataset_key, artifact_key=artifact_key,
        stable_key=source_key,
        source_cluster_key=StableRecordKey((9, 9, 7, 50, 1)),
        source_key="UD_LZH_PUBLIC_R5_EVALUATOR_FIXTURE",
        snapshot_id="ud-lzh-public-r5-evaluator-r1",
        revision_id="r5-evaluator-route-revision",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-PublicFixture"),
        source_identity="public-r5-evaluator:sentence:1")
    pairs = []
    for ordinal, dimension in enumerate(W02_DEV_DIMENSIONS, start=1):
        old_observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, parent,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="read_only_probe",
            perturbation_kind="HELD_OUT_DOCUMENT")
        observation = replace(
            old_observation, dataset_key=dataset_key, artifact_key=artifact_key,
            stable_key=StableRecordKey((9, 9, 7, 20, ordinal)),
            source_ref_key=source_key, language="lzh")
        old_label = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, parent,
            old_observation, _expected(form, lemma="新"),
            dimension_name=dimension)
        assert isinstance(old_label, EvaluatorLabelRecord)
        label = replace(
            old_label, dataset_key=dataset_key, artifact_key=artifact_key,
            stable_key=StableRecordKey((9, 9, 7, 40, ordinal)),
            observation_key=observation.stable_key)
        pairs.append((observation, label))
    capability = w02_ud_morphology_source_capability({
        "annotation_provenance": "public synthetic manual UD annotation",
        "commit_sha1": source.revision_id,
        "language": "lzh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": source.source_key,
        "upstream_checksum": source.upstream_checksum,
    })
    return source, tuple(pairs), capability


@pytest.fixture(scope="module")
def artifact_roots(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("w02-r5-language-family")
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
    return candidate.artifact_path, v1.artifact_path, v2.artifact_path


def _budget() -> V2EvaluatorResourceBudget:
    return V2EvaluatorResourceBudget(
        512, 9_000_000, 536_870_912, 300_000, 100_000, 4)


def test_r5_evaluator_uses_zh_base_view_and_original_lzh_route(
        artifact_roots) -> None:
    candidate, v1, v2 = artifact_roots
    source, pairs, capability = _lzh_private_rows()
    result = evaluate_w02_morphology_successor_v3_private_r5_pair_stream(
        candidate, v1, v2, (source,), (capability,), pairs, _budget())

    assert W02_MORPH_V3_PRIVATE_R5_EVALUATOR_VERSION.endswith("R5-V1")
    assert result["status"] == "PASS"
    assert result["base_language_adapter_count"] == 5
    assert result["base_language_original_lzh_count"] == 5
    assert result["base_language_temporary_scope_language"] == "zh"
    assert result["base_language_route_uses_original_observation"] == 1
    assert result["base_language_clone_adapter_count"] == 1
    assert result["base_language_clone_original_lzh_count"] == 1
    assert result["route_authorized_count"] == 5


def test_r5_evaluator_rejects_non_lzh_private_observation(
        artifact_roots) -> None:
    candidate, v1, v2 = artifact_roots
    source, pairs, capability = _lzh_private_rows()
    observation, label = pairs[0]
    with pytest.raises(W02MorphologySuccessorV3PrivateR5EvaluationError,
                       match="original language"):
        evaluate_w02_morphology_successor_v3_private_r5_pair_stream(
            candidate, v1, v2, (source,), (capability,),
            ((replace(observation, language="zh"), label),), _budget())


def test_r5_formal_entry_closes_v6_sources_before_pair_generator() -> None:
    source = inspect.getsource(
        run_w02_morphology_successor_v3_private_r5_evaluation)
    assert source.index("read_and_close_w02") < source.index("pairs = (")
    assert "blind_private_source_specs_v6" in source


def _v6_source_ref(ordinal: int, record_ordinal: int) -> dict[str, object]:
    spec = blind_private_source_specs_v6()[0]
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    locator = f"test:{ordinal}:lzh-kyoto-{ordinal:05d}"
    prefix = [7, 7, 1]
    return {
        "artifact_key": prefix,
        "attribution": "Universal Dependencies Kyoto attribution retained",
        "course_version": 2,
        "dataset_key": [7, 7],
        "format_version": 2,
        "license_id": spec["license_id"],
        "local_sha256": "1" * 64,
        "official_url": spec["repository_url"],
        "parser_version": 1,
        "record_kind": "source_ref",
        "record_ordinal": record_ordinal,
        "redistribution_policy": "PUBLIC",
        "revision_id": spec["commit_sha1"],
        "schema_version": 2,
        "snapshot_id": spec["snapshot_id"],
        "source_cluster_key": [7, 7, 1, 1, record_ordinal],
        "source_identity": f"{KYOTO_REMAINDER_SOURCE_KEY}:sentence:{locator}",
        "source_key": KYOTO_REMAINDER_SOURCE_KEY,
        "source_span": {
            "document_cluster_key": [7, 7, 1, 2, record_ordinal],
            "entity_graph_cluster_key": [7, 7, 1, 3, record_ordinal],
            "locator_kind": "sentence", "locator_value": locator,
            "span_end": 2, "span_start": 0,
        },
        "stable_key": [7, 7, 1, 10, record_ordinal],
        "upstream_checksum": "sha1:" + str(data_file["git_blob_sha1"]),
    }


def test_r5_source_reader_closes_synthetic_v6_refs_without_real_payload(
        tmp_path: Path) -> None:
    target = tmp_path / "private" / "source_refs.jsonl.gz"
    target.parent.mkdir(parents=True)
    rows = [
        canonical_json_bytes(_v6_source_ref(
            KYOTO_REMAINDER_MINIMUM_ORDINAL + index, index + 1)) + b"\n"
        for index in range(500)
    ]
    content = b"".join(rows)
    with target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(content)
    transport = target.read_bytes()
    identity = W02MorphologySuccessorV3PrivateR5FileIdentity(
        "PRIVATE_SOURCE", "PRIVATE_EVALUATOR_ROOT", "source_ref", "", 500,
        len(content), hashlib.sha256(content).hexdigest(), len(transport),
        hashlib.sha256(transport).hexdigest(), (7, 7, 1, 10, 1),
        (7, 7, 1, 10, 500), ("CC-BY-SA-4.0",))
    permit = V2AccessPermit(
        "PH2_V2_PRIVATE_EVALUATOR", "PRIVATE_EVALUATOR_ROOT", "W-02",
        "held_out", "source_ref", target, identity.transport_sha256,
        identity.transport_size_bytes, "a" * 64)

    sources = read_and_close_w02_morphology_successor_v3_private_r5_sources(
        (identity,), {"PRIVATE_SOURCE": permit})
    assert len(sources) == 500
    assert sources[0].source_key == KYOTO_REMAINDER_SOURCE_KEY
    assert sources[-1].source_span.to_value()["locator_value"].startswith(
        "test:1500:")
