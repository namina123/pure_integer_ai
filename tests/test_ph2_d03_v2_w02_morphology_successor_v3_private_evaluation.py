"""Public synthetic tests for the successor V3 blind private evaluator."""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    evaluate_w02_morphology_successor_v3_private_pair_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_io import (
    W02MorphologySuccessorV3PrivateIOError,
    iter_w02_morphology_successor_v3_private_records,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner import (
    W02MorphologySuccessorV3PrivateFileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(split: str, ordinal: int, surface: str):
    return _source_record(
        "UD_ZH_GSDSIMP_R2_18", split, ordinal,
        snapshot_id="ud-zh-gsdsimp-v3-private-train",
        revision_id="train-revision",
        official_url="https://github.com/UniversalDependencies/UD_Chinese-GSDSimp",
        source_identity=f"v3-private-train:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{split}:{ordinal}"),
        license_id="CC-BY-SA-4.0", attribution="public synthetic fixture",
        locator_kind="record", locator_value=str(ordinal),
        span_end=len(surface))


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
        source = _source("train", ordinal, form)
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


def _routed_private_rows(*, morphology_upos: str = "VERB"):
    form = "新化"
    parent = _source("held_out", 1, form)
    dataset_key = StableRecordKey((9, 9, 8, 1))
    artifact_key = StableRecordKey((9, 9, 8, 2))
    source_key = StableRecordKey((9, 9, 8, 10, 1))
    source = replace(
        parent,
        dataset_key=dataset_key,
        artifact_key=artifact_key,
        stable_key=source_key,
        source_cluster_key=StableRecordKey((9, 9, 8, 50, 1)),
        source_key="UD_ZH_PUBLIC_V3_PRIVATE_FIXTURE",
        snapshot_id="ud-zh-public-v3-private-r1",
        revision_id="v3-private-revision",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Chinese-PublicV3PrivateFixture"),
        source_identity="public-v3-private-fixture:document:1",
        license_id="CC-BY-SA-3.0",
    )
    pairs = []
    for ordinal, dimension in enumerate(W02_DEV_DIMENSIONS, start=1):
        old_observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, parent,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="read_only_probe",
            perturbation_kind="HELD_OUT_DOCUMENT")
        observation = replace(
            old_observation,
            dataset_key=dataset_key,
            artifact_key=artifact_key,
            stable_key=StableRecordKey((9, 9, 8, 20, ordinal)),
            source_ref_key=source_key,
            license_partition="CC-BY-SA-3.0",
        )
        old_evaluation = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, parent,
            old_observation,
            _expected(form, lemma="新",
                      upos=morphology_upos if dimension == W02_DEV_DIMENSIONS[2]
                      else "VERB"),
            dimension_name=dimension)
        assert isinstance(old_evaluation, EvaluatorLabelRecord)
        evaluation = replace(
            old_evaluation,
            dataset_key=dataset_key,
            artifact_key=artifact_key,
            stable_key=StableRecordKey((9, 9, 8, 40, ordinal)),
            observation_key=observation.stable_key,
        )
        pairs.append((observation, evaluation))
    capability = w02_ud_morphology_source_capability({
        "annotation_provenance": "public synthetic manual UD annotation",
        "commit_sha1": source.revision_id,
        "language": "zh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": source.source_key,
        "upstream_checksum": source.upstream_checksum,
    })
    return source, tuple(pairs), capability


def _artifacts(tmp_path: Path):
    candidate = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate", pairs=_training_pairs(),
        run_id=1, requested_workers=2, mode="fresh")
    v1 = run_w02_morphology_overlay_fixture(
        fixture_root=tmp_path / "v1",
        candidate_artifact_root=candidate.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    v2 = run_w02_morphology_successor_v2_overlay_fixture(
        fixture_root=tmp_path / "v2",
        candidate_artifact_root=candidate.artifact_path,
        v1_overlay_artifact_root=v1.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    return candidate.artifact_path, v1.artifact_path, v2.artifact_path


@pytest.fixture(scope="module")
def artifact_roots(tmp_path_factory: pytest.TempPathFactory):
    return _artifacts(tmp_path_factory.mktemp("w02-v3-private"))


def _budget(max_logic_operations: int = 9_000_000):
    return V2EvaluatorResourceBudget(
        512, max_logic_operations, 536_870_912, 300_000, 100_000, 4)


def test_v3_private_core_authorizes_every_route_and_closes_nine_gates(
        artifact_roots) -> None:
    candidate, v1, v2 = artifact_roots
    source, pairs, capability = _routed_private_rows()
    result = evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate, v1, v2, (source,), (capability,), pairs, _budget())
    assert result["status"] == "PASS"
    assert len(result["hard_conjunct_results"]) == 9
    assert all(row["status"] == "PASS"
               for row in result["hard_conjunct_results"])
    assert result["route_authorized_count"] == 5
    assert result["v1_generalized_candidate_count"] > 0
    assert result["v2_edge_candidate_count"] > 0
    assert result["max_v2_edge_candidates_per_requested_span"] <= 8


def test_v3_private_core_preserves_fail_ne_and_fails_closed_route(
        artifact_roots) -> None:
    candidate, v1, v2 = artifact_roots
    source, failed_pairs, capability = _routed_private_rows(
        morphology_upos="AUX")
    failed = evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate, v1, v2, (source,), (capability,), failed_pairs, _budget())
    assert failed["status"] == "FAIL"
    assert failed["dimension_results"][2]["failed"] == 1

    _source_row, passing_pairs, _capability = _routed_private_rows()
    stopped = evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate, v1, v2, (source,), (capability,), passing_pairs, _budget(1))
    assert stopped["status"] == "NE"
    assert stopped["evaluation_count"] == 0
    assert stopped["input_pair_count"] == 5

    unlisted_source = replace(source, stable_key=StableRecordKey((9, 9, 8, 10, 2)))
    with pytest.raises(RuntimeError, match="route-authorized"):
        evaluate_w02_morphology_successor_v3_private_pair_stream(
            candidate, v1, v2, (unlisted_source,), (capability,),
            passing_pairs, _budget())


def test_v3_private_reader_accepts_cc_by_sa_3_and_rejects_transport_drift(
        tmp_path: Path) -> None:
    target = tmp_path / "observations" / "held_out.jsonl.gz"
    target.parent.mkdir(parents=True)
    _source_row, pairs, _capability = _routed_private_rows()
    observation, _ = pairs[0]
    line = canonical_json_bytes(observation.to_dict()) + b"\n"
    with target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(line)
    transport = target.read_bytes()
    identity = W02MorphologySuccessorV3PrivateFileIdentity(
        "PRIVATE_HELD_OUT_OBSERVATION", "PRIVATE_EVALUATOR_ROOT",
        "observation", "held_out", 1, len(line),
        hashlib.sha256(line).hexdigest(), len(transport),
        hashlib.sha256(transport).hexdigest(),
        observation.stable_key.components, observation.stable_key.components,
        ("CC-BY-SA-3.0",))
    permit = V2AccessPermit(
        "PH2_V2_PRIVATE_EVALUATOR", "PRIVATE_EVALUATOR_ROOT", "W-02",
        "held_out", "observation", target, identity.transport_sha256,
        identity.transport_size_bytes, "a" * 64)
    records = tuple(iter_w02_morphology_successor_v3_private_records(
        identity, permit))
    assert len(records) == 1 and isinstance(records[0], ObservationRecord)

    mutated = bytearray(transport)
    mutated[4] ^= 1
    target.write_bytes(bytes(mutated))
    with pytest.raises(W02MorphologySuccessorV3PrivateIOError, match="transport"):
        tuple(iter_w02_morphology_successor_v3_private_records(
            identity, permit))
