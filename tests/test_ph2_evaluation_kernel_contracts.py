"""Bounded contract tests for the manifest/plugin evaluation kernel slice."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from pure_integer_ai.experiments.ph2_evaluation_kernel.aggregate import (
    build_evaluation_aggregate,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard,
    consume_guard,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationThreshold,
    build_evaluation_manifest,
    publish_evaluation_manifest,
    read_evaluation_manifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.identity import (
    evaluation_kernel_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.owner_receipt import (
    EvaluationOwnerBinding,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.preflight import (
    EvaluationPreflightCheck,
    build_formal_ready_receipt,
    build_preflight_layer,
    publish_formal_ready_receipt,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.publication import (
    build_failure_seal,
    build_publication_decision,
    build_runtime_receipt,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationDimensionResult,
    EvaluationKernelContractError,
    EvaluationRunAudit,
    EvaluationResultSet,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import V2WriteAccount
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
from pure_integer_ai.experiments.ph2_evaluation_kernel.source_binding import (
    EvaluationSourceBinding,
    EvaluationSourceSlice,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
    V2_STAGE_EVALUATION_POLICIES,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _manifest():
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-03")
    keys = policy.hard_conjunct_keys
    source = EvaluationSourceBinding(
        _sha("source-contract"),
        (
            EvaluationSourceSlice(
                "PUBLIC-A", "held_out", "cluster-a", 1, 2, 2, _sha("refs-a")),
            EvaluationSourceSlice(
                "PUBLIC-A", "held_out", "cluster-b", 1, 2, 2, _sha("refs-b")),
        ),
        _sha("all-source-refs"),
    )
    owner = EvaluationOwnerBinding(
        "PH2_V2_PRIVATE_EVALUATOR", _sha("owner-receipt"), _sha("metadata"),
        _sha("inventory"),
        _sha("payload"), _sha("cases"), _sha("labels"), _sha("clusters"),
        4, 4, 4, 4,
    )
    plugin = EvaluationPluginDeclaration(
        "W03-SYNTHETIC", "V1", "W-03", "ph2.w03.synthetic",
        "evaluate", _sha("plugin"), keys,
    )
    thresholds = tuple(EvaluationThreshold(key, 1, 1) for key in keys)
    return build_evaluation_manifest(
        release_key="PH2-D03-V2",
        stage_key="W-03",
        family_key="W03-SYNTHETIC-A",
        revision_key="R1",
        public_head_sha1=_sha1("public-head"),
        kernel_semantic_sha256=evaluation_kernel_semantic_sha256(REPOSITORY),
        stage_manifest_sha256=_sha("stage"),
        plugin=plugin,
        source_binding=source,
        owner_binding=owner,
        candidate_artifact_sha256=_sha("candidate"),
        resource_budget=V2EvaluatorResourceBudget(
            16, 1_000_000, 1_000_000, 10_000, 10_000, 1),
        consumed_lineage_sha256=_sha("lineage"),
        bearing_dimension_keys=policy.bearing_dimension_keys,
        generation_hard_conjunct_key=policy.generation_hard_conjunct_key,
        support_dimension_keys=tuple(
            key for key in policy.hard_conjunct_keys
            if key not in policy.bearing_dimension_keys
            and key != policy.generation_hard_conjunct_key),
        thresholds=thresholds,
    )


def _result(key: str, role: str, status: str) -> EvaluationDimensionResult:
    counts = {
        "PASS": (1, 0, 0, 0),
        "FAIL": (0, 1, 0, 0),
        "NE": (0, 0, 1, 0),
        "BLOCKED": (0, 0, 0, 1),
    }[status]
    return EvaluationDimensionResult(
        key, role, status, 1, *counts, _sha(f"{key}:{status}"))


def _results(status: str = "PASS") -> EvaluationResultSet:
    manifest = _manifest()
    keys = manifest.hard_conjunct_keys
    roles = (
        *("BEARING" for _ in manifest.bearing_dimension_keys),
        "GENERATION",
        *("SUPPORT" for _ in manifest.support_dimension_keys),
    )
    rows = [_result(key, role, "PASS") for key, role in zip(keys, roles, strict=True)]
    if status != "PASS":
        rows[0] = _result(keys[0], "BEARING", status)
    return EvaluationResultSet(tuple(rows))


def _audit() -> EvaluationRunAudit:
    return EvaluationRunAudit(
        "COMPLETE", 4, 4, 12, 12, 1024, 4096, 3, V2WriteAccount())


def test_four_state_counts_and_precedence_are_fail_closed() -> None:
    assert _results("PASS").status == "PASS"
    assert _results("FAIL").status == "FAIL"
    assert _results("NE").status == "NE"
    assert _results("BLOCKED").status == "BLOCKED"
    with pytest.raises(EvaluationKernelContractError):
        replace(_result("A", "BEARING", "PASS"), status="FAIL")


def test_source_slices_are_exact_sorted_and_non_overlapping() -> None:
    source = _manifest().source_binding
    assert source.record_count == 4
    with pytest.raises(EvaluationKernelContractError):
        replace(source, slices=tuple(reversed(source.slices)))
    with pytest.raises(EvaluationKernelContractError):
        EvaluationSourceSlice("A", "held_out", "C", 1, 3, 2, _sha("bad"))


def test_owner_binding_requires_zero_read_unused_closed_counts() -> None:
    owner = _manifest().owner_binding
    with pytest.raises(EvaluationKernelContractError):
        replace(owner, private_payload_reads_before=1)
    with pytest.raises(EvaluationKernelContractError):
        replace(owner, label_count=3)


def test_manifest_binds_every_required_identity_and_round_trips(tmp_path) -> None:
    manifest = _manifest()
    assert manifest.plugin.result_keys == manifest.hard_conjunct_keys
    target = publish_evaluation_manifest(manifest, tmp_path / "family.json")
    assert read_evaluation_manifest(target) == manifest
    assert publish_evaluation_manifest(manifest, target) == target
    with pytest.raises(Exception):
        publish_evaluation_manifest(
            replace(manifest, revision_key="R2", family_commitment=manifest.family_commitment),
            target,
        )


def test_manifest_rejects_plugin_or_owner_source_drift() -> None:
    manifest = _manifest()
    with pytest.raises(EvaluationKernelContractError):
        replace(manifest, plugin=replace(
            manifest.plugin, result_keys=tuple(reversed(manifest.plugin.result_keys))))
    with pytest.raises(EvaluationKernelContractError):
        replace(manifest, owner_binding=replace(
            manifest.owner_binding, source_ref_count=5))


def test_guard_is_one_shot_and_binds_manifest_plugin_owner_candidate() -> None:
    manifest = _manifest()
    available = build_available_guard(manifest)
    consumed, intent = consume_guard(available)
    assert available.state == "AVAILABLE"
    assert consumed.state == "CONSUMED"
    assert intent.consumed_guard_sha256 == consumed.sha256()
    assert consumed.plugin_semantic_sha256 == manifest.plugin.semantic_sha256
    with pytest.raises(EvaluationKernelContractError):
        consume_guard(consumed)


@pytest.mark.parametrize("status", ("PASS", "FAIL", "NE", "BLOCKED"))
def test_aggregate_and_publication_keep_four_states_separate(status: str) -> None:
    aggregate = build_evaluation_aggregate(_manifest(), _results(status), _audit())
    decision = build_publication_decision(aggregate)
    assert aggregate.status == status
    if status == "PASS":
        assert decision.runtime_receipt_allowed == 1
        assert build_runtime_receipt(aggregate).status == "PASS"
        with pytest.raises(EvaluationKernelContractError):
            build_failure_seal(aggregate)
    else:
        assert decision.failure_seal_required == 1
        assert build_failure_seal(aggregate).status == status
        with pytest.raises(EvaluationKernelContractError):
            build_runtime_receipt(aggregate)


def test_aggregate_rejects_reordered_or_wrong_role_results() -> None:
    manifest = _manifest()
    rows = _results().results
    with pytest.raises(EvaluationKernelContractError):
        build_evaluation_aggregate(
            manifest, EvaluationResultSet(tuple(reversed(rows))), _audit())
    wrong = (replace(rows[0], role="SUPPORT"), *rows[1:])
    with pytest.raises(EvaluationKernelContractError):
        build_evaluation_aggregate(manifest, EvaluationResultSet(wrong), _audit())


def test_formal_ready_receipt_refuses_nonpass_p0_p4_chain(tmp_path) -> None:
    manifest = _manifest()
    layers = []
    for layer_key, status in (("P0", "FAIL"), ("P1", "PASS"),
                              ("P2", "PASS"), ("P3", "PASS")):
        layers.append(build_preflight_layer(layer_key, (
            EvaluationPreflightCheck(
                f"{layer_key}_CHECK", status, _sha(f"{layer_key}:{status}")),
        )))
    receipt = build_formal_ready_receipt(
        manifest, *layers,
        public_dev_status="PASS", public_shadow_a_status="PASS",
        public_shadow_b_or_metamorphic_status="PASS", family_pushed=1,
        publication_evidence_sha256=_sha("publication"))
    assert receipt.status == "BLOCKED"
    assert receipt.layers[-1].status == "BLOCKED"
    with pytest.raises(EvaluationKernelContractError):
        publish_formal_ready_receipt(manifest, receipt, tmp_path / "ready.json")
