from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02_CAPABILITY_OOV_BOUNDARY_LATTICE,
    W02_CAPABILITY_UD_MORPHOLOGY,
    W02_CAPABILITY_UNICODE_ANALYSIS,
    W02CarrierRule,
    boundary_lattice,
    generate_with_carrier_rules,
    learn_w02_training_pair,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02CandidatePublicationError,
    build_w02_candidate_receipt,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    W02CandidateRuntimeBudget,
    W02CandidateStoreError,
    open_w02_candidate_predictor,
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    W02_CARRIER_KINDS,
    _authored_expected,
    _carrier_serialization,
    _observation_record,
    _owner_record,
    _source_record,
    _unicode_expected,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_runtime_contract import (
    W02_RUNTIME_CODE_PATHS,
    build_w02_candidate_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import TeacherEvidenceRecord


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(source_key: str, *, ordinal: int, surface: str):
    license_id = "CC0-1.0" if source_key == "AUTHORED_CC0" else "CC-BY-SA-4.0"
    return _source_record(
        source_key, "train", ordinal,
        snapshot_id=f"fixture-{source_key.casefold()}",
        revision_id="1",
        official_url="https://example.invalid/public-fixture",
        source_identity=f"fixture:{source_key}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{source_key}:{ordinal}"),
        local_sha256=_sha(f"local:{source_key}:{ordinal}"),
        license_id=license_id,
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value=str(ordinal),
        span_end=len(surface),
    )


def _fixture_pairs():
    pairs = []
    units = ("qz", "灥", "z1")
    surface = "".join(units)
    authored_source = _source("AUTHORED_CC0", ordinal=1, surface=surface)
    for ordinal, carrier_kind in enumerate(W02_CARRIER_KINDS, start=1):
        observation = _observation_record(
            "AUTHORED_CC0", "train", ordinal, authored_source,
            carrier_kind=carrier_kind,
            surface=surface,
            family_ordinal=1,
            sample_role="support",
            perturbation_kind="NONE",
        )
        raw, start, end, _, _ = _carrier_serialization(carrier_kind, surface)
        evidence = _owner_record(
            "AUTHORED_CC0", "train", ordinal, authored_source, observation,
            _authored_expected(units, carrier_kind, raw, start, end),
            dimension_name="W-02-V2-OOV",
        )
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))

    ud_surface = "我 爱 AI"
    ud_source = _source("UD_ZH_GSDSIMP_R2_18", ordinal=1, surface=ud_surface)
    ud_observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, ud_source,
        carrier_kind="plain_text",
        surface=ud_surface,
        family_ordinal=1,
        sample_role="support",
        perturbation_kind="NONE",
    )
    ud_expected = {
        "boundary_spans": [
            {"end": 1, "form": "我", "start": 0},
            {"end": 3, "form": "爱", "start": 2},
            {"end": 6, "form": "AI", "start": 4},
        ],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [
            {"feats": [], "form": "我", "lemma": "我", "node_id": [1], "upos": "PRON"},
            {"feats": [], "form": "爱", "lemma": "爱", "node_id": [2], "upos": "VERB"},
            {"feats": [], "form": "AI", "lemma": "AI", "node_id": [3], "upos": "PROPN"},
        ],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }
    ud_evidence = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, ud_source, ud_observation,
        ud_expected, dimension_name="W-02-V2-NEW-CONTENT-MORPHOLOGY")
    assert isinstance(ud_evidence, TeacherEvidenceRecord)
    pairs.append((ud_observation, ud_evidence))

    unicode_surface = "甲A\u0301"
    unicode_source = _source("ZHWIKTIONARY_20260701", ordinal=1,
                             surface=unicode_surface)
    unicode_observation = _observation_record(
        "ZHWIKTIONARY_20260701", "train", 1, unicode_source,
        carrier_kind="plain_text",
        surface=unicode_surface,
        family_ordinal=1,
        sample_role="support",
        perturbation_kind="NONE",
    )
    unicode_expected = _unicode_expected(unicode_surface, "plain_text")
    unicode_expected["page_title_sha256"] = _sha(unicode_surface)
    unicode_evidence = _owner_record(
        "ZHWIKTIONARY_20260701", "train", 1, unicode_source,
        unicode_observation, unicode_expected,
        dimension_name="W-02-V2-BOUNDARY-WITHDRAWAL")
    assert isinstance(unicode_evidence, TeacherEvidenceRecord)
    pairs.append((unicode_observation, unicode_evidence))
    return tuple(pairs)


def _unseen_observation():
    surface = "新AIz2"
    source = _source_record(
        "AUTHORED_CC0", "held_out", 1,
        snapshot_id="fixture-unseen",
        revision_id="1",
        official_url="https://example.invalid/public-fixture",
        source_identity="fixture:unseen:1",
        upstream_checksum="sha256:" + _sha("fixture-unseen-upstream"),
        local_sha256=_sha("fixture-unseen-local"),
        license_id="CC0-1.0",
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value="1",
        span_end=len(surface),
    )
    return _observation_record(
        "AUTHORED_CC0", "held_out", 1, source,
        carrier_kind="markdown",
        surface=surface,
        family_ordinal=1,
        sample_role="read_only_probe",
        perturbation_kind="OOV_UNSEEN_FAMILY",
    )


def test_candidate_model_learns_three_evidence_families_and_generates() -> None:
    deltas = tuple(learn_w02_training_pair(*pair) for pair in _fixture_pairs())
    capabilities = {capability for delta in deltas for capability in delta.capabilities}
    assert W02_CAPABILITY_OOV_BOUNDARY_LATTICE in capabilities
    assert W02_CAPABILITY_UD_MORPHOLOGY in capabilities
    assert W02_CAPABILITY_UNICODE_ANALYSIS in capabilities

    counts = {}
    for delta in deltas:
        counts[delta.carrier_rule] = counts.get(delta.carrier_rule, 0) + 1
    rules = tuple(sorted(counts.items(), key=lambda item: item[0].key()))
    for carrier_kind in W02_CARRIER_KINDS:
        generated = generate_with_carrier_rules(
            rules, carrier_kind=carrier_kind, surface="未见内容")
        assert generated.status == "GENERATED"
        assert generated.carrier_serialization[
            generated.content_span_start:generated.content_span_end] == "未见内容"

    assert boundary_lattice("qz灥z2", observed_unit_lengths=(1, 2, 3)) == tuple(
        range(6))


def test_candidate_fixture_is_canonical_for_1_2_4_workers_and_readback(
        tmp_path: Path) -> None:
    pairs = _fixture_pairs()
    results = []
    for workers in (1, 2, 4):
        root = tmp_path / f"workers-{workers}"
        result = run_w02_candidate_fixture(
            fixture_root=root,
            pairs=pairs,
            run_id=1,
            requested_workers=workers,
            mode="fresh",
        )
        results.append(result)
        manifest = read_canonical_object(
            result.artifact_path / "candidate.artifact.json")
        assert manifest["run_scope"] == "PUBLIC_SYNTHETIC_FIXTURE"
        assert manifest["formal_training_runs"] == 0
        assert manifest["formal_private_evaluation_runs"] == 0
        assert manifest["private_payload_reads"] == 0
        with open_w02_candidate_predictor(result.artifact_path) as predictor:
            prediction = predictor.predict(_unseen_observation())
        assert prediction.status == "PREDICTED"
        assert prediction.generation.carrier_serialization == "## 词项\n\n新AIz2\n"
        assert prediction.boundary_lattice[0] == 0
        assert prediction.boundary_lattice[-1] == len("新AIz2")
        assert any(item.form == "AI" for item in prediction.morphology_candidates)
        assert tuple(item.code_point for item in prediction.unicode_units) == tuple(
            ord(char) for char in "新AIz2")
        before = tuple((path.relative_to(result.artifact_path), _sha(path.read_bytes().hex()))
                       for path in sorted(result.artifact_path.rglob("*")) if path.is_file())
        resumed = run_w02_candidate_fixture(
            fixture_root=root,
            pairs=None,
            run_id=1,
            requested_workers=workers,
            mode="resume",
        )
        after = tuple((path.relative_to(result.artifact_path), _sha(path.read_bytes().hex()))
                      for path in sorted(result.artifact_path.rglob("*")) if path.is_file())
        assert resumed.candidate_semantic_sha256 == result.candidate_semantic_sha256
        assert before == after
    assert len({item.candidate_semantic_sha256 for item in results}) == 1
    assert len({item.generated_probe_sha256 for item in results}) == 1
    assert len({item.artifact_manifest_sha256 for item in results}) == 1
    with pytest.raises(W02CandidatePublicationError, match="不是正式"):
        build_w02_candidate_receipt(
            Path(__file__).resolve().parents[1], results[0].artifact_path)


def test_candidate_fixture_restart_and_resource_stop(tmp_path: Path) -> None:
    pairs = _fixture_pairs()
    restart_root = tmp_path / "restart"
    with pytest.raises(W02CandidateStoreError, match="injected Candidate shard fault"):
        run_w02_candidate_fixture(
            fixture_root=restart_root,
            pairs=pairs,
            run_id=1,
            requested_workers=4,
            mode="fresh",
            fault_after_shard=3,
        )
    assert not (restart_root / "candidate-store" / "run-000001").exists()
    restarted = run_w02_candidate_fixture(
        fixture_root=restart_root,
        pairs=None,
        run_id=1,
        requested_workers=2,
        mode="restart",
    )
    clean = run_w02_candidate_fixture(
        fixture_root=tmp_path / "clean",
        pairs=pairs,
        run_id=1,
        requested_workers=1,
        mode="fresh",
    )
    assert restarted.candidate_semantic_sha256 == clean.candidate_semantic_sha256
    assert restarted.generated_probe_sha256 == clean.generated_probe_sha256

    stopped_root = tmp_path / "stopped"
    with pytest.raises(W02CandidateStoreError, match="pair resource stop"):
        run_w02_candidate_fixture(
            fixture_root=stopped_root,
            pairs=pairs,
            run_id=1,
            requested_workers=1,
            mode="fresh",
            budget=W02CandidateRuntimeBudget(max_pairs=2),
        )
    assert not (stopped_root / "candidate-store" / "run-000001").exists()


def test_candidate_model_keeps_conflicting_carrier_rules_ambiguous() -> None:
    rules = (
        (W02CarrierRule("plain_text", "", "", "text", "text_span"), 2),
        (W02CarrierRule("plain_text", "[", "]", "text", "text_span"), 2),
    )
    result = generate_with_carrier_rules(
        rules, carrier_kind="plain_text", surface="x")
    assert result.status == "AMBIGUOUS"


def test_candidate_runtime_freeze_binds_parent_code_and_public_tests() -> None:
    repository = Path(__file__).resolve().parents[1]
    freeze = build_w02_candidate_runtime_freeze(repository)
    assert freeze.parent_compile_freeze_sha256 == (
        "8174c923b0752dcc17abf0aca2d35385f0ad285f5b1336416555f82e738b19d9")
    assert freeze.resource_budget.max_pairs == 51_200
    assert freeze.resource_budget.max_logic_operations == 9_000_000
    assert tuple(item.repository_path for item in freeze.code_files) == (
        W02_RUNTIME_CODE_PATHS)
    assert "tests/test_ph2_d03_v2_w02_candidate_runtime.py" in W02_RUNTIME_CODE_PATHS
