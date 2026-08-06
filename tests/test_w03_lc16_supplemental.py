"""W-03 LC-16 supplemental 合同、聚合、隔离和 append-only 测试。"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_catalog import (
    OVERLAY_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_catalog import (
    W03Lc16SupplementalCatalogError,
    build_w03_lc16_supplemental_manifest,
    verify_w03_lc16_supplemental_files,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_contract import (
    ABLATION_ORDER,
    BEARING_DIMENSIONS,
    DIRECTION_EVALUATION_COUNT,
    DIRECTIONS,
    EVALUATION_ORDER,
    GENERATION_HARD_CONJUNCT,
    MANIFEST_PATH,
    OVERLAY_SHA256,
    SupplementalAblationResult,
    SupplementalDimensionResult,
    SupplementalDirectionResult,
    W03Lc16SupplementalError,
    read_w03_lc16_supplemental_manifest,
    write_w03_lc16_supplemental_manifest,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_evaluator import (
    aggregate_w03_lc16_supplemental,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_publication import (
    W03Lc16SupplementalPublicationError,
    publish_w03_lc16_supplemental_report,
    read_w03_lc16_supplemental_report,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_runner import (
    W03Lc16SupplementalRunnerError,
    W03Lc16SupplementalSafeResultPack,
    read_w03_lc16_supplemental_safe_result_pack,
    run_w03_lc16_supplemental_safe_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = ROOT / Path(*MANIFEST_PATH.split("/"))
_DIGESTS = {
    "core": "1" * 64,
    "cursor": "2" * 64,
    "logical": "3" * 64,
    "memory": "4" * 64,
    "use": "5" * 64,
}
_PACK_BINDINGS = {
    "private_bundle_commitment_sha256": "9" * 64,
    "safe_result_pack_sha256": "8" * 64,
}


def _dimension(key: str, status: str = "PASS") -> SupplementalDimensionResult:
    """构造单方向 1/1 dimension result。"""
    return SupplementalDimensionResult(
        key,
        status,
        int(status == "PASS"),
        1,
        int(status == "FAIL"),
        int(status == "NE"),
        (key.encode("utf-8").hex() + "0" * 64)[:64],
    )


def _direction_result(carrier: str, case_key: tuple[int, ...], owner_key: tuple[int, ...], direction: str) -> SupplementalDirectionResult:
    """构造通过基线和正交消融的公开方向结果。"""
    ablations = []
    for index, key in enumerate(ABLATION_ORDER):
        target = EVALUATION_ORDER[index]
        statuses = tuple(
            (dimension, "FAIL" if dimension == target else "PASS")
            for dimension in BEARING_DIMENSIONS)
        ablations.append(SupplementalAblationResult(
            key,
            target,
            statuses,
            "FAIL" if target == GENERATION_HARD_CONJUNCT else "PASS",
            (key.encode("utf-8").hex() + "f" * 64)[:64],
        ))
    return SupplementalDirectionResult(
        carrier,
        case_key,
        owner_key,
        direction,
        tuple(_dimension(key) for key in BEARING_DIMENSIONS),
        _dimension(GENERATION_HARD_CONJUNCT),
        tuple(ablations),
        "PASS",
        (f"{carrier}:{case_key}:{direction}".encode("utf-8").hex() + "a" * 64)[:64],
    )


def _results(overlay):
    """按 overlay 顺序构造完整 189 条测试结果。"""
    result = []
    for course in overlay.carrier_courses:
        for case in course.cases:
            for direction in DIRECTIONS:
                result.append(_direction_result(
                    course.carrier_key,
                    case.case_key.stable_key(),
                    case.owner_key.stable_key(),
                    direction,
                ))
    return tuple(result)


def _safe_pack(overlay):
    """构造不含 private payload 的新 family 测试 pack。"""
    return W03Lc16SupplementalSafeResultPack(
        "9" * 64,
        _results(overlay),
        _DIGESTS,
        dict(_DIGESTS),
        14,
        14,
        100000,
        189,
        189,
        1000,
    )


@pytest.fixture(scope="module")
def overlay():
    """读取已经 append-only 发布的历史 overlay。"""
    return read_d03_lc16_successor_overlay(
        ROOT / Path(*OVERLAY_MANIFEST_PATH.split("/")))


@pytest.fixture(scope="module")
def report(overlay):
    """构造仅用于合同测试的全 PASS synthetic public result。"""
    return aggregate_w03_lc16_supplemental(
        overlay,
        _results(overlay),
        **_PACK_BINDINGS,
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS),
        private_path_reads=14,
        private_payload_bytes=100000,
        private_payload_reads=189,
        evaluator_label_reads=189,
    )


def test_formal_manifest_is_frozen_and_current_rebuild_fails_closed():
    """正式 manifest 保持冻结，证据演进后重建和严格回验均须拒绝。"""
    manifest = read_w03_lc16_supplemental_manifest(FORMAL_MANIFEST)
    assert manifest.canonical_bytes() == FORMAL_MANIFEST.read_bytes()
    with pytest.raises(W03Lc16SupplementalCatalogError, match="身份漂移"):
        verify_w03_lc16_supplemental_files(manifest, repository_root=ROOT)
    with pytest.raises(W03Lc16SupplementalCatalogError, match="overlay"):
        build_w03_lc16_supplemental_manifest(ROOT)
    assert manifest.parent_overlay_sha256 == OVERLAY_SHA256
    state = manifest.execution_state.to_value()
    assert state["runtime_observed"] == 0
    assert state["w02_lc16_supplemental_qualified"] == 1
    assert state["w03_lc16_supplemental_qualified"] == 0


def test_manifest_scope_freezes_63_cases_and_189_direction_evaluations():
    """预注册 manifest 必须明确九载体、七类样本、三向和资源硬界。"""
    value = read_w03_lc16_supplemental_manifest(FORMAL_MANIFEST).scope.to_value()
    assert value["carrier_count"] == 9
    assert value["sample_kind_count"] == 7
    assert value["case_count"] == 63
    assert value["direction_evaluations"] == 189
    assert value["ablation_count"] == 5
    assert value["max_payload_bytes"] == 201326592
    assert value["max_payload_reads"] == 196608
    assert value["max_logic_operations"] == 3000000
    assert value["max_host_writes"] == 0
    assert value["max_workers"] == 4


def test_aggregate_requires_exact_carrier_case_owner_direction_coverage(overlay):
    """缺一条、重复一条或 owner 错绑都不能形成资格 receipt。"""
    values = list(_results(overlay))
    with pytest.raises(W03Lc16SupplementalError, match="189"):
        aggregate_w03_lc16_supplemental(
            overlay, tuple(values[:-1]), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)
    values[-1] = values[0]
    with pytest.raises(W03Lc16SupplementalError, match="漂移"):
        aggregate_w03_lc16_supplemental(
            overlay, tuple(values), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)


def test_ablation_must_hit_target_and_leave_other_dimensions_unchanged(overlay):
    """正交消融不能把未目标维度的变化隐藏在 aggregate 中。"""
    values = list(_results(overlay))
    first = values[0]
    bad = list(first.ablations)
    bad[0] = SupplementalAblationResult(
        bad[0].ablation_key,
        bad[0].targeted_dimension,
        tuple((key, "PASS") for key in BEARING_DIMENSIONS),
        "PASS",
        bad[0].evidence_sha256,
    )
    values[0] = SupplementalDirectionResult(
        first.carrier_key, first.case_key, first.owner_key, first.direction,
        first.dimensions, first.generation, tuple(bad),
        first.independent_reveal_status, first.evidence_sha256,
    )
    with pytest.raises(W03Lc16SupplementalError, match="消融"):
        aggregate_w03_lc16_supplemental(
            overlay, tuple(values), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=1,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)

    values = list(_results(overlay))
    first = values[0]
    bad = list(first.ablations)
    generation_ablation = bad[-1]
    bad[-1] = SupplementalAblationResult(
        generation_ablation.ablation_key,
        generation_ablation.targeted_dimension,
        generation_ablation.dimension_statuses,
        "PASS",
        generation_ablation.evidence_sha256,
    )
    values[0] = SupplementalDirectionResult(
        first.carrier_key, first.case_key, first.owner_key, first.direction,
        first.dimensions, first.generation, tuple(bad),
        first.independent_reveal_status, first.evidence_sha256,
    )
    with pytest.raises(W03Lc16SupplementalError, match="消融"):
        aggregate_w03_lc16_supplemental(
            overlay, tuple(values), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=1,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)


def test_runtime_zero_is_blocked_and_host_or_builder_write_is_rejected(overlay):
    """未运行只能 BLOCKED，host 写或复用 consumer builder 不能发布。"""
    blocked = aggregate_w03_lc16_supplemental(
        overlay, _results(overlay), **_PACK_BINDINGS,
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS), private_path_reads=0,
        private_payload_bytes=1, private_payload_reads=1,
        evaluator_label_reads=1, runtime_observed=0)
    assert blocked.status == "BLOCKED"
    with pytest.raises(W03Lc16SupplementalError):
        aggregate_w03_lc16_supplemental(
            overlay, _results(overlay), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1, host_write_count=1)
    with pytest.raises(W03Lc16SupplementalError):
        aggregate_w03_lc16_supplemental(
            overlay, _results(overlay), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1, consumer_result_builder_reused=1)


def test_observed_runtime_requires_private_read_evidence(overlay):
    """已观察的 private family 不得用零读取摘要伪造资格。"""
    with pytest.raises(W03Lc16SupplementalError, match="private read"):
        aggregate_w03_lc16_supplemental(
            overlay, _results(overlay), **_PACK_BINDINGS,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=0,
            evaluator_label_reads=0)


def test_report_publication_is_append_only_and_private_free(tmp_path, report):
    """receipt 只含摘要且既存目标拒绝覆盖。"""
    target = tmp_path / "supplemental.json"
    publication = publish_w03_lc16_supplemental_report(target, report)
    original = target.read_bytes()
    assert publication.status == "PASS"
    assert b"expected_surface" not in original
    value, readback = read_w03_lc16_supplemental_report(target)
    assert value["case_count"] == 63
    assert value["evaluator_label_writes"] == 0
    assert value["independent_reveal_status"] == "PASS"
    assert readback == publication
    with pytest.raises(W03Lc16SupplementalPublicationError, match="禁止覆盖"):
        publish_w03_lc16_supplemental_report(target, report)
    assert target.read_bytes() == original


def test_non_pass_report_cannot_be_publicly_released(tmp_path, report):
    """FAIL/NE/BLOCKED 只能封存，不能建立公开 PASS receipt。"""
    blocked = replace(
        report,
        status="BLOCKED",
        independent_reveal_status="BLOCKED",
        runtime_observed=0,
    )
    with pytest.raises(
            W03Lc16SupplementalPublicationError, match="只有 PASS"):
        publish_w03_lc16_supplemental_report(
            tmp_path / "blocked.json", blocked)


def test_receipt_readback_revalidates_the_full_public_contract(tmp_path, report):
    """canonical bytes 通过后仍必须满足摘要合同，不能只过字段扫描。"""
    target = tmp_path / "receipt.json"
    publish_w03_lc16_supplemental_report(target, report)
    value = json.loads(target.read_text("utf-8"))
    value["private_reads"]["private_payload_reads"] = 0
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(W03Lc16SupplementalPublicationError, match="合同"):
        read_w03_lc16_supplemental_report(target)


def test_manifest_writer_is_idempotent_but_rejects_different_bytes(tmp_path):
    """预注册 manifest 写入保持追加式，不接受异内容覆盖。"""
    manifest = read_w03_lc16_supplemental_manifest(FORMAL_MANIFEST)
    target = tmp_path / "manifest.json"
    assert write_w03_lc16_supplemental_manifest(manifest, target) == target
    assert write_w03_lc16_supplemental_manifest(manifest, target) == target
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(W03Lc16SupplementalError, match="内容不同"):
        write_w03_lc16_supplemental_manifest(manifest, target)


def test_manifest_mutation_is_fail_closed():
    """改变预算、parent 或执行状态都必须拒绝回读。"""
    manifest = read_w03_lc16_supplemental_manifest(FORMAL_MANIFEST)
    for mutate in (
            lambda value: value["scope"].update({"max_host_writes": 1}),
            lambda value: value["execution_state"].update({"W04_STARTED": 1}),
            lambda value: value.update({"parent_overlay_sha256": "0" * 64}),
            lambda value: value.update({
                "w02_supplemental_receipt_sha256": "0" * 64}),
    ):
        value = copy.deepcopy(manifest.to_dict())
        mutate(value)
        with pytest.raises(W03Lc16SupplementalError):
            type(manifest).from_dict(value)


def test_safe_result_pack_round_trip_and_stale_runner_fails_closed(tmp_path, overlay):
    """安全 pack 可规范回读；历史 manifest 漂移时 runner 必须拒绝重跑。"""
    target = tmp_path / "safe-result-pack.json"
    original = _safe_pack(overlay).canonical_bytes()
    target.write_bytes(original)
    pack, sha256 = read_w03_lc16_supplemental_safe_result_pack(target)
    assert pack.canonical_bytes() == original
    assert len(sha256) == 64
    with pytest.raises(W03Lc16SupplementalRunnerError, match="无法聚合"):
        run_w03_lc16_supplemental_safe_pack(target, repository_root=ROOT)
    assert target.read_bytes() == original


def test_missing_safe_pack_is_explicitly_blocked(tmp_path):
    """缺少新 family 安全 pack 时不得借旧结果继续。"""
    outcome = run_w03_lc16_supplemental_safe_pack(
        tmp_path / "missing.json", repository_root=ROOT)
    assert outcome.status == "BLOCKED"
    assert outcome.blocker_code == "SAFE_RESULT_PACK_MISSING"
    assert outcome.report is None


def test_original_w03_family_and_private_payload_are_rejected(tmp_path):
    """原 W-03 五 case 结构及递归 private 字段不能进入新 runner。"""
    old = tmp_path / "old-w03.json"
    old.write_bytes(canonical_json_bytes({
        "artifact_version": "PH2-W03-PRIVATE-FAMILY", "case_count": 5}) + b"\n")
    with pytest.raises(W03Lc16SupplementalRunnerError, match="原 W-03/5-case"):
        read_w03_lc16_supplemental_safe_result_pack(old)
    private = tmp_path / "private.json"
    private.write_bytes(canonical_json_bytes({
        "nested": [{"raw_observation": "secret"}]}) + b"\n")
    with pytest.raises(W03Lc16SupplementalRunnerError, match="private"):
        read_w03_lc16_supplemental_safe_result_pack(private)


def test_safe_pack_rejects_parent_and_coverage_drift(tmp_path, overlay):
    """producer、parent SHA 或 189 覆盖漂移必须在聚合前失败。"""
    value = _safe_pack(overlay).to_public_dict()
    value["producer_revision"] = "unexpected"
    target = tmp_path / "producer-drift.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(W03Lc16SupplementalRunnerError, match="producer"):
        read_w03_lc16_supplemental_safe_result_pack(target)
    value = _safe_pack(overlay).to_public_dict()
    value["parent_overlay_sha256"] = "0" * 64
    target = tmp_path / "parent-drift.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(W03Lc16SupplementalRunnerError, match="parent"):
        read_w03_lc16_supplemental_safe_result_pack(target)
    value = _safe_pack(overlay).to_public_dict()
    value["direction_results"] = value["direction_results"][:-1]
    target = tmp_path / "coverage-drift.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(W03Lc16SupplementalRunnerError, match="189"):
        read_w03_lc16_supplemental_safe_result_pack(target)
