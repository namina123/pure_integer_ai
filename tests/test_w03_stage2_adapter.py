"""PH2 W-03 typed Observation 到 Sense 候选的适配合同。"""
from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03TypedAdapterError,
    adapt_w03_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_adapter_extractors import (
    W03AdapterExtractionError,
)
from pure_integer_ai.experiments.ph2_w03_context import open_w03_frozen_context
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationObservation,
    formal_w03_publication_baseline,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_contract import W03RunRequest
from pure_integer_ai.experiments.ph2_w03_firewall import W03PayloadFirewall
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPOSITORY.parent / "w02_artifacts"


@lru_cache(maxsize=1)
def _formal_payload() -> W03TrainingPayload:
    baseline = formal_w03_publication_baseline()
    continuity = verify_formal_w02_continuity(REPOSITORY, ARTIFACTS_ROOT)
    context = open_w03_frozen_context(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=baseline.head_sha1,
        w02_continuity=continuity,
        publication_baseline=baseline,
        backend_profile_key=(3, 1),
    )
    request = W03RunRequest(
        run_id=context.run_id,
        parent_run_id=context.parent_run_id,
        base_run_id=context.base_run_id,
        stage_key=context.stage_key,
        owner_key=context.owner_key,
        runner_key="PH2_LANGUAGE_STAGE2",
        publication_baseline_key=context.publication_baseline.stable_key(),
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        w02_continuity_key=context.w02_continuity.stable_key(),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    observation = W03PublicationObservation(
        local_head_sha1=baseline.head_sha1,
        tracking_head_sha1=baseline.head_sha1,
        remote_head_sha1=baseline.head_sha1,
        ci_run_id=baseline.ci_run_id,
        ci_head_sha1=baseline.head_sha1,
        ci_status="completed",
        ci_conclusion="success",
        ci_jobs=baseline.ci_jobs,
    )
    return W03PayloadFirewall.open(
        REPOSITORY,
        context,
        request,
        publication_observation=observation,
    ).read_training_payload()


@lru_cache(maxsize=1)
def _formal_output():
    return adapt_w03_training_payload(_formal_payload())


def _observation_envelope(*, kind: str, logical_order: int | None = None):
    matches = tuple(
        item for item in _formal_output().observations
        if item.observation.payload_kind == kind
        and (logical_order is None
             or item.observation.logical_order == logical_order)
    )
    if len(matches) != 1:
        raise AssertionError("test fixture 没有唯一 Observation envelope")
    return matches[0]


def test_w03_typed_adapter_has_separate_public_entrypoint() -> None:
    """W03-02 使用独立 typed adapter，不借用 legacy sense 路由。"""
    assert callable(adapt_w03_training_payload)


def test_formal_payload_maps_every_w03_schema_and_retains_w02_exactly() -> None:
    """四类 W03 schema 全覆盖，W-02 records 原对象原序保留。"""
    payload = _formal_payload()
    output = _formal_output()

    assert (len(payload.source_refs), len(payload.observations),
            len(payload.teacher_evidence)) == (83, 40, 40)
    assert len(output.source_bindings) == 83
    assert len(output.retained_w02_observations) == 19
    assert len(output.retained_w02_teacher_evidence) == 19
    assert set(output.retained_w02_observations) == {
        item for item in payload.observations if item.w_stage == "W-02"}
    assert set(output.retained_w02_teacher_evidence) == {
        item for item in payload.teacher_evidence
        if item.observation_key in {
            value.stable_key for value in output.retained_w02_observations}}
    assert len(output.observations) == len(output.evidence) == 21
    assert len(output.candidates) == 59
    assert {
        kind: sum(item.observation.payload_kind == kind
                  for item in output.observations)
        for kind in {
            "SenseBoundaryQuery",
            "ConstructionCandidateV1",
            "RAW_SOURCE_OBSERVATION_V1",
        }
    } == {
        "SenseBoundaryQuery": 4,
        "ConstructionCandidateV1": 12,
        "RAW_SOURCE_OBSERVATION_V1": 5,
    }
    assert dict(output.execution_state) == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W03_STARTED": 0,
        "W04_STARTED": 0,
        "formal_w03_training_runs": 0,
        "learning_writes": 0,
        "teacher_calls": 0,
    }


def test_source_license_revision_split_owner_and_stable_keys_are_retained() -> None:
    """每个候选仍可逐层回到原 SourceRef、Observation 和 teacher owner。"""
    output = _formal_output()
    by_record = {item.record.stable_key: item for item in output.source_bindings}

    assert len(by_record) == 83
    for envelope in output.observations:
        observation = envelope.observation
        source = envelope.source.record
        assert by_record[observation.source_ref_key] == envelope.source
        assert source.license_id == observation.license_partition
        assert source.revision_id or source.snapshot_id
        assert observation.split == "train"
        evidence = next(
            item for item in output.evidence
            if item.observation.stable_key == observation.stable_key)
        assert evidence.teacher_record.observation_key == observation.stable_key
        assert evidence.teacher_record.owner_key.components
        assert evidence.teacher_record.source_ref_key == source.stable_key
        assert evidence.source_ref == envelope.source.source_ref
        assert all(candidate.source_record is source
                   for candidate in envelope.candidates)


def test_authored_same_atom_keeps_complete_competition_and_concept_split() -> None:
    """同 surface/context 的 finance 与 river Sense 共存，不按序取首项。"""
    support = _observation_envelope(
        kind="SenseBoundaryQuery", logical_order=1).candidates[0]
    refute = _observation_envelope(
        kind="SenseBoundaryQuery", logical_order=2).candidates[0]
    atom_candidates = _formal_output().candidates_for_atom(support.anchor.atom)

    assert support.anchor.atom == refute.anchor.atom
    assert support.anchor.representation == refute.anchor.representation
    assert support.competition_key == refute.competition_key
    assert support.sense != refute.sense
    assert support.concept != refute.concept
    assert len(atom_candidates) == 3
    assert all(item.selection_state == "UNSELECTED" for item in atom_candidates)
    assert EvidenceCandidateDefinition.from_stable_key(
        support.definition.stable_key()) == support.definition


def test_supersede_preserves_old_observation_and_changes_context_identity() -> None:
    """revision 只追加依赖，新旧 stable identity 与候选都可审计。"""
    old = _observation_envelope(
        kind="SenseBoundaryQuery", logical_order=1).candidates[0]
    new = _observation_envelope(
        kind="SenseBoundaryQuery", logical_order=4).candidates[0]

    assert new.supersedes_observation_key == old.observation.stable_key
    assert old in _formal_output().candidates
    assert new.anchor.atom == old.anchor.atom
    assert new.context != old.context
    assert new.competition_key != old.competition_key
    assert new.sense != old.sense
    assert new.observation.typed_payload.to_value()["candidate_sense"] == (
        old.observation.typed_payload.to_value()["candidate_sense"])


def test_teacher_evidence_maps_four_states_without_entering_candidate_envelope() -> None:
    """TRUE/FALSE/CONFLICT 分账，candidate schema 没有 expected/label 字段。"""
    evidence_by_order = {
        item.logical_order: item for item in _formal_output().evidence
        if item.observation.payload_kind == "SenseBoundaryQuery"
    }
    assert evidence_by_order[1].stances == (EVIDENCE_SUPPORT,)
    assert evidence_by_order[2].stances == (EVIDENCE_REFUTE,)
    assert evidence_by_order[3].stances == (
        EVIDENCE_SUPPORT, EVIDENCE_REFUTE)
    assert evidence_by_order[4].stances == (EVIDENCE_SUPPORT,)
    assert evidence_by_order[4].supersedes_observation_key == (
        evidence_by_order[1].observation.stable_key)
    assert evidence_by_order[4].prerequisite_keys == ()
    candidate_fields = {item.name for item in fields(
        type(_observation_envelope(
            kind="SenseBoundaryQuery", logical_order=1).candidates[0]))}
    assert not candidate_fields.intersection({
        "expected", "expected_payload", "expected_state", "label",
        "teacher_record",
    })


def test_lc03_preserves_multi_member_span_and_anti_literal_evidence_only() -> None:
    """词汇化整体 Sense 保留结构 span，anti-literal 不伪造候选。"""
    constructions = tuple(
        item for item in _formal_output().observations
        if item.observation.payload_kind == "ConstructionCandidateV1")
    anti_literal = next(
        item for item in constructions
        if item.parser_provenance.to_value()["construction_present"] == 0)
    discontinuous = next(
        item for item in constructions
        if item.observation.typed_payload.to_value()["candidate_kind"]
        == "DISCONTINUOUS")
    lexicalized = tuple(
        candidate for item in constructions for candidate in item.candidates
        if candidate.lexicalized_multiword)

    assert anti_literal.candidates == ()
    assert len(anti_literal.anchors) == 1
    assert any(item.observation == anti_literal.observation
               for item in _formal_output().evidence)
    assert len(discontinuous.candidates) == 1
    assert len(discontinuous.candidates[0].anchor.extracted.members) == 2
    assert lexicalized
    assert all(len(item.anchor.extracted.surface) > 1 for item in lexicalized)


def test_wikidata_terms_are_structured_nondefinitive_evidence() -> None:
    """三 entity 经严格 no-float parser 形成 zh term 候选，但不直接判真。"""
    values = tuple(
        item for item in _formal_output().observations
        if item.parser_provenance.to_value()["extractor"]
        == "WIKIDATA_ENTITY_TERMS_V1")
    candidates = tuple(candidate for item in values for candidate in item.candidates)
    evidence = tuple(
        item for item in _formal_output().evidence
        if item.observation in {value.observation for value in values})

    assert len(values) == 3
    assert len(candidates) == 41
    assert all(item.external_nondefinitive for item in candidates)
    assert all(item.anchor.extracted.branch_language.startswith("zh")
               for item in candidates)
    assert len({item.concept for item in candidates}) == 3
    assert all(item.stances == (EVIDENCE_UNKNOWN,) for item in evidence)
    assert all(item.withdrawal_level == 3 for item in evidence)
    assert all(item.external_nondefinitive for item in evidence)


def test_wiktionary_redirect_and_three_senses_keep_template_provenance() -> None:
    """redirect 只作 Evidence，完整页按 section/definition 形成三义竞争。"""
    values = tuple(
        item for item in _formal_output().observations
        if item.parser_provenance.to_value()["extractor"]
        == "WIKTIONARY_WIKITEXT_SENSE_V1")
    redirect = next(
        item for item in values
        if item.parser_provenance.to_value()["redirect"] == 1)
    entry = next(
        item for item in values
        if item.parser_provenance.to_value()["redirect"] == 0)

    assert redirect.candidates == ()
    assert len(redirect.anchors) == 1
    assert len(entry.candidates) == 3
    assert entry.parser_provenance.to_value()["definition_count"] == 3
    assert entry.parser_provenance.to_value()["template_spans"]
    assert len({item.anchor.atom for item in entry.candidates}) == 1
    assert len({item.competition_key for item in entry.candidates}) == 1
    assert len({item.sense for item in entry.candidates}) == 3
    assert len({item.concept for item in entry.candidates}) == 3


def test_adapter_is_order_independent_and_performs_no_writes() -> None:
    """输入文件交付顺序不进入规范输出，执行状态始终全零。"""
    payload = _formal_payload()
    reversed_payload = W03TrainingPayload(
        tuple(reversed(payload.source_refs)),
        tuple(reversed(payload.observations)),
        tuple(reversed(payload.teacher_evidence)),
    )
    repeated = adapt_w03_training_payload(reversed_payload)

    assert repeated == _formal_output()
    assert set(dict(repeated.execution_state).values()) == {0}


def test_open_surface_sense_and_context_are_derived_from_records() -> None:
    """新词、新义和新 context 无需改 Python vocabulary 即改变 typed identity。"""
    payload = _formal_payload()
    original = next(
        item for item in payload.observations
        if item.payload_kind == "SenseBoundaryQuery" and item.logical_order == 1)
    teacher = next(
        item for item in payload.teacher_evidence
        if item.observation_key == original.stable_key)
    changed_observation = replace(
        original,
        typed_payload=CanonicalJsonObject.from_value({
            "candidate_sense": "registration_hub",
            "context": "他在星港办理登记。",
            "query_kind": "sense_boundary",
            "surface": "星港",
        }),
    )
    teacher_value = teacher.typed_evidence.to_value()
    teacher_value["expected_payload"]["boundary"] = "registration_hub"
    changed_teacher = replace(
        teacher,
        typed_evidence=CanonicalJsonObject.from_value(teacher_value),
    )
    changed_payload = W03TrainingPayload(
        payload.source_refs,
        tuple(changed_observation if item is original else item
              for item in payload.observations),
        tuple(changed_teacher if item is teacher else item
              for item in payload.teacher_evidence),
    )
    changed = adapt_w03_training_payload(changed_payload)
    candidate = next(
        item for item in changed.candidates
        if item.observation.stable_key == original.stable_key)
    formal = _observation_envelope(
        kind="SenseBoundaryQuery", logical_order=1).candidates[0]

    assert candidate.anchor.extracted.surface == "星港"
    assert candidate.anchor.representation != formal.anchor.representation
    assert candidate.anchor.atom != formal.anchor.atom
    assert candidate.concept != formal.concept
    assert candidate.context != formal.context


@pytest.mark.parametrize("failure", ("license", "teacher", "kind", "construction"))
def test_identity_owner_schema_and_teacher_drift_fail_closed(failure: str) -> None:
    """任何 record/owner/schema 漂移均在 lifecycle 或写入前拒绝。"""
    payload = _formal_payload()
    observations = list(payload.observations)
    teachers = list(payload.teacher_evidence)
    target_index = next(
        index for index, item in enumerate(observations)
        if item.w_stage == "W-03")
    target = observations[target_index]
    if failure == "license":
        observations[target_index] = replace(
            target, license_partition="CC-BY-4.0")
        error = W03TypedAdapterError
    elif failure == "teacher":
        teachers = [item for item in teachers
                    if item.observation_key != target.stable_key]
        error = W03TypedAdapterError
    elif failure == "kind":
        observations[target_index] = replace(target, payload_kind="FutureSenseV2")
        error = W03AdapterExtractionError
    else:
        construction_index = next(
            index for index, item in enumerate(observations)
            if item.payload_kind == "ConstructionCandidateV1"
            and item.typed_payload.to_value()["construction_identity"]["present"]
            == 1)
        construction = observations[construction_index]
        value = construction.typed_payload.to_value()
        value["selection_state"] = "SELECTED"
        observations[construction_index] = replace(
            construction,
            typed_payload=CanonicalJsonObject.from_value(value),
        )
        error = W03AdapterExtractionError
    with pytest.raises(error):
        adapt_w03_training_payload(W03TrainingPayload(
            payload.source_refs,
            tuple(observations),
            tuple(teachers),
        ))


def test_wikidata_qid_revision_identity_drift_fails_closed() -> None:
    """typed raw identity 与 EntityData 不一致时严格 parser 拒绝。"""
    payload = _formal_payload()
    observations = list(payload.observations)
    index = next(
        index for index, item in enumerate(observations)
        if item.payload_kind == "RAW_SOURCE_OBSERVATION_V1"
        and "entity_json_utf8" in item.typed_payload.to_value()["raw_observation"])
    observation = observations[index]
    value = observation.typed_payload.to_value()
    value["raw_observation"]["qid"] = "Q1"
    observations[index] = replace(
        observation,
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    with pytest.raises(ValueError, match="hash|QID|identity|身份"):
        adapt_w03_training_payload(W03TrainingPayload(
            payload.source_refs,
            tuple(observations),
            payload.teacher_evidence,
        ))


def test_adapter_source_has_no_legacy_or_evaluator_route() -> None:
    """W03-02 不导入 legacy sense table，也没有 evaluator label 入口。"""
    source = (
        REPOSITORY
        / "src/pure_integer_ai/experiments/ph2_w03_adapter.py"
    ).read_text(encoding="utf-8")
    extractor = (
        REPOSITORY
        / "src/pure_integer_ai/experiments/ph2_w03_adapter_extractors.py"
    ).read_text(encoding="utf-8")
    combined = source + extractor

    assert "language_sense_candidate_runtime" not in combined
    assert "sense_candidates" not in combined
    assert "EvaluatorLabelRecord" not in combined
    assert "owners/evaluator" not in combined
