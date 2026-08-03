"""W07-02 typed adapter、schema rejection 与资源边界专项。"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w07_adapter import (
    W07TypedAdapterError,
    adapt_w07_training_payload,
    w07_logic_candidate_protocol,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.experiments.ph2_w07_payload import W07TrainingPayload
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]


def _request(context, *, worker_count=1, mode="fresh"):
    return W07RunRequest(
        W07_FORMAL_RUN_ID,
        W07_W06_BASE_RUN_ID,
        W07_W06_BASE_RUN_ID,
        W07_STAGE_KEY,
        context.owner_key,
        W07_RUNNER_KEY,
        context.baseline_commit_sha1,
        context.stable_key(),
        context.backend_profile_key,
        context.base_fence_key,
        worker_count,
        mode,
        tuple(sorted(W07_RESOURCE_BUDGET.items())),
        tuple(
            item.relative_path
            for item in context.candidate_payload_bindings
        ),
        tuple(
            item.relative_path
            for item in context.teacher_evidence_bindings
        ),
    )


@pytest.fixture(scope="module")
def adapter_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("w07-adapter") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        payload = W07PayloadFirewall.open(
            ROOT, context, _request(context)).read_training_payload()
        return context, payload, adapt_w07_training_payload(payload)
    finally:
        backend.close()


def _rejected_candidate_keys(rejection):
    value = rejection.observation.typed_payload.to_value()
    if rejection.observation.substage == "NESTED_SCOPE":
        return tuple(
            tuple(layer["candidate_spec"]["candidate_key"])
            for layer in value["layers"]
        )
    return (tuple(value["candidate_spec"]["candidate_key"]),)


def _replace_observation(payload, original, changed, *, teacher=None):
    teachers = payload.teacher_evidence
    if teacher is not None:
        teachers = tuple(
            teacher if item.observation_key == original.stable_key else item
            for item in teachers
        )
    return W07TrainingPayload(
        payload.source_refs,
        tuple(
            changed if item.stable_key == original.stable_key else item
            for item in payload.observations
        ),
        teachers,
    )


def test_w07_adapter_covers_seven_families_and_normalizes_three_protocols(
        adapter_fixture):
    """七类 compiler schema 收敛到一个共享 W-07 candidate protocol。"""
    _, _, output = adapter_fixture
    assert Counter(
        item.observation.substage for item in output.proposals
    ) == {
        "NOT": 9,
        "AND_OR": 11,
        "CONDITION": 9,
        "EXISTS": 8,
        "FORALL": 8,
        "MODAL": 10,
        "NESTED_SCOPE": 8,
    }
    assert tuple(dict.fromkeys(
        item.observation.substage for item in output.proposals
    )) == W07_SUBSTAGE_ORDER
    assert len({
        item.source_protocol.stable_key() for item in output.proposals
    }) == 3
    assert output.protocol == w07_logic_candidate_protocol()
    assert len(output.proposals) == len(output.evidence) == 63
    assert len(output.specs) == 71
    assert {family for item in output.proposals
            for family in item.operator_families} == {
        "NOT", "AND", "OR", "CONDITION", "EXISTS", "FORALL", "MODAL",
    }


def test_w07_adapter_rejects_three_invalid_schemas_without_candidate_evidence(
        adapter_fixture):
    """三个显式 schema 负例保留来源，但不形成 candidate 或 Evidence。"""
    _, _, output = adapter_fixture
    assert tuple(
        (item.observation.substage, item.reason)
        for item in output.rejections
    ) == (
        ("EXISTS", "DOMAIN_TYPE_MISMATCH"),
        ("FORALL", "DOMAIN_TYPE_MISMATCH"),
        ("NESTED_SCOPE", "MISSING_INNER_OPERATOR"),
    )
    accepted_observations = {
        item.observation.stable_key for item in output.proposals}
    evidence_observations = {
        item.proposal.observation.stable_key for item in output.evidence}
    accepted_candidates = {
        item.candidate.stable_key() for item in output.specs}
    for rejection in output.rejections:
        assert rejection.observation.stable_key not in accepted_observations
        assert rejection.observation.stable_key not in evidence_observations
        assert rejection.source_record.stable_key == (
            rejection.observation.source_ref_key)
        assert rejection.teacher_record.observation_key == (
            rejection.observation.stable_key)
        assert not set(_rejected_candidate_keys(rejection)) & accepted_candidates


def test_w07_adapter_preserves_role_binder_modal_and_nested_scope(
        adapter_fixture):
    """Role 顺序、Binder/domain、modal scope 与 nested 层序保持 typed。"""
    _, _, output = adapter_fixture
    for proposal in output.proposals:
        bound_by_candidate = {}

        def visit(bound):
            bound_by_candidate[bound.template] = bound
            for binding in bound.bindings:
                if hasattr(binding.filler, "bindings"):
                    visit(binding.filler)

        visit(proposal.bound_root)
        for spec in proposal.specs:
            bound = bound_by_candidate[spec.candidate]
            assert tuple(
                (item.role, item.ordinal) for item in spec.definition.slots
            ) == tuple(
                (item.role, item.ordinal) for item in bound.bindings
            )
            assert spec.forming_sources == (proposal.source_binding.source_ref,)
        for quantifier in proposal.quantifiers:
            assert quantifier.definition.binder in (
                bound_by_candidate[
                    quantifier.operator_candidate
                ].introduced_binders
            )
            assert tuple(
                item.value for item in quantifier.value_evidence
            ) == tuple(
                item.value for item in quantifier.definition.domain.values
            )
        for plan in proposal.modal_plans:
            assert plan.input_scope == proposal.request_scope
            assert plan.source == proposal.source_binding.source_ref
            if plan.status == "RESOLVED":
                assert plan.output_scope is not None
                assert plan.output_scope.source == plan.source
                assert plan.evidence_ids
            else:
                assert plan.output_scope is None
                assert plan.evidence_ids == ()
                assert not plan.state.support and not plan.state.refute

    nested = tuple(
        item for item in output.proposals
        if item.observation.substage == "NESTED_SCOPE"
    )
    assert nested
    for proposal in nested:
        value = proposal.observation.typed_payload.to_value()
        layer_ids = [item["layer_id"] for item in value["layers"]]
        assert value["derivation_order"] == list(reversed(layer_ids))
        assert proposal.operator_families == tuple(
            item["operator_family"] for item in value["layers"])


def test_w07_adapter_separates_operator_adoption_from_content_four_state(
        adapter_fixture):
    """内容四态无损保留，合法 FALSE 不反驳 operator，reject 单独反驳。"""
    _, _, output = adapter_fixture
    expected = {
        "TRUE": (EVIDENCE_SUPPORT,),
        "FALSE": (EVIDENCE_REFUTE,),
        "UNKNOWN": (EVIDENCE_UNKNOWN,),
        "CONFLICT": (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
    }
    assert {item.expected_state for item in output.evidence} == set(expected)
    assert all(item.content_stances == expected[item.expected_state]
               for item in output.evidence)
    rejected = tuple(
        item for item in output.evidence
        if item.expected_payload.to_value()["decision"].startswith("reject_")
    )
    assert len(rejected) == 7
    assert all(item.stances == (EVIDENCE_REFUTE,) for item in rejected)
    known_false = tuple(
        item for item in output.evidence
        if item.expected_state == "FALSE" and item not in rejected
    )
    assert known_false
    assert all(item.stances == (EVIDENCE_SUPPORT,) for item in known_false)
    pseudo = next(
        item for item in output.evidence
        if item.proposal.observation.perturbation_kind == "PSEUDO_OPERATOR"
    )
    assert pseudo.content_stances == (EVIDENCE_UNKNOWN,)
    assert pseudo.stances == (EVIDENCE_REFUTE,)


def test_w07_adapter_is_deterministic_across_replay_worker_and_mode(
        adapter_fixture):
    """纯 adapter replay 与 worker/mode 调度变化不改变 typed 输出。"""
    context, payload, first = adapter_fixture
    replay = adapt_w07_training_payload(payload)
    assert replay.stable_key() == first.stable_key()
    assert tuple(
        item.stable_key(replay.protocol) for item in replay.specs
    ) == tuple(
        item.stable_key(first.protocol) for item in first.specs
    )
    second_payload = W07PayloadFirewall.open(
        ROOT,
        context,
        _request(context, worker_count=4, mode="resume"),
    ).read_training_payload()
    second = adapt_w07_training_payload(second_payload)
    assert second.stable_key() == first.stable_key()
    assert tuple(
        item.stable_key(second.protocol) for item in second.specs
    ) == tuple(
        item.stable_key(first.protocol) for item in first.specs
    )


def test_w07_adapter_rejects_protocol_and_source_binding_drift(adapter_fixture):
    """compiler protocol 与 SourceRefRecord 到 typed source 必须一一绑定。"""
    _, payload, _ = adapter_fixture
    original = next(
        item for item in payload.observations if item.substage == "NOT")
    foreign = next(
        item for item in payload.observations if item.substage == "EXISTS")
    value = original.typed_payload.to_value()
    value["candidate_protocol"] = (
        foreign.typed_payload.to_value()["candidate_protocol"])
    changed = replace(
        original,
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    with pytest.raises(W07TypedAdapterError, match="protocol 与 substage"):
        adapt_w07_training_payload(
            _replace_observation(payload, original, changed))

    first, second = tuple(
        item for item in payload.observations
        if item.substage == "NOT" and item.perturbation_kind != "PARSER_REVISION"
    )[:2]
    second_teacher = next(
        item for item in payload.teacher_evidence
        if item.observation_key == second.stable_key)
    changed_observation = replace(
        second, source_ref_key=first.source_ref_key)
    changed_teacher = replace(
        second_teacher, source_ref_key=first.source_ref_key)
    with pytest.raises(W07TypedAdapterError, match="绑定多个 typed SourceRef"):
        adapt_w07_training_payload(_replace_observation(
            payload,
            second,
            changed_observation,
            teacher=changed_teacher,
        ))


def test_w07_adapter_stable_key_covers_decoded_typed_semantics(adapter_fixture):
    """稳定键显式覆盖 Evidence、量词和 modal 解码结果，不只依赖 record key。"""
    _, _, output = adapter_fixture
    evidence = output.evidence[0]
    changed_evidence = replace(
        evidence, reason_key=(*evidence.reason_key, 1))
    assert replace(
        output,
        evidence=(changed_evidence, *output.evidence[1:]),
    ).stable_key() != output.stable_key()

    proposal_index, proposal = next(
        (index, item) for index, item in enumerate(output.proposals)
        if item.quantifiers and len(item.quantifiers[0].value_evidence) > 1)
    quantifier = proposal.quantifiers[0]
    changed_quantifier = replace(
        quantifier,
        value_evidence=tuple(reversed(quantifier.value_evidence)),
    )
    changed_proposal = replace(
        proposal,
        quantifiers=(changed_quantifier, *proposal.quantifiers[1:]),
    )
    changed_proposals = list(output.proposals)
    changed_proposals[proposal_index] = changed_proposal
    assert replace(
        output, proposals=tuple(changed_proposals)
    ).stable_key() != output.stable_key()


@pytest.mark.parametrize("budget_key", ("max_records", "max_logic_operations"))
def test_w07_adapter_fails_closed_on_resource_limits(
        adapter_fixture, budget_key):
    """调用方只能收紧冻结预算，实际 record/operation 超限时拒绝全部输出。"""
    _, payload, output = adapter_fixture
    budget = dict(W07_RESOURCE_BUDGET)
    actual = (
        output.record_count
        if budget_key == "max_records"
        else output.logic_operations
    )
    budget[budget_key] = actual - 1
    with pytest.raises(W07TypedAdapterError, match="resource 超限"):
        adapt_w07_training_payload(payload, resource_budget=budget)
