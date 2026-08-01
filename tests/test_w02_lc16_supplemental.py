"""W-02 LC-16 supplemental 合同、聚合、隔离和 append-only 测试。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_catalog import (
    OVERLAY_MANIFEST_PATH,
    build_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_catalog import (
    build_w02_lc16_supplemental_manifest,
    verify_w02_lc16_supplemental_files,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_contract import (
    ABLATION_ORDER,
    BEARING_DIMENSIONS,
    DIRECTION_EVALUATION_COUNT,
    DIRECTIONS,
    GENERATION_HARD_CONJUNCT,
    MANIFEST_PATH,
    OVERLAY_SHA256,
    SupplementalAblationResult,
    SupplementalDimensionResult,
    SupplementalDirectionResult,
    W02Lc16SupplementalError,
    read_w02_lc16_supplemental_manifest,
    write_w02_lc16_supplemental_manifest,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_evaluator import (
    aggregate_w02_lc16_supplemental,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_publication import (
    W02Lc16SupplementalPublicationError,
    publish_w02_lc16_supplemental_report,
    read_w02_lc16_supplemental_report,
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
        target = BEARING_DIMENSIONS[index]
        statuses = tuple(
            (dimension, "FAIL" if dimension == target else "PASS")
            for dimension in BEARING_DIMENSIONS)
        ablations.append(SupplementalAblationResult(
            key,
            target,
            statuses,
            "FAIL" if target == BEARING_DIMENSIONS[2] else "PASS",
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


@pytest.fixture(scope="module")
def overlay():
    """读取并严格回验当前冻结 overlay。"""
    return build_d03_lc16_successor_overlay(ROOT)


@pytest.fixture(scope="module")
def report(overlay):
    """构造仅用于合同测试的全 PASS synthetic public result。"""
    return aggregate_w02_lc16_supplemental(
        overlay,
        _results(overlay),
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS),
        private_path_reads=14,
        private_payload_bytes=100000,
        private_payload_reads=189,
        evaluator_label_reads=189,
    )


def test_formal_manifest_is_canonical_and_parent_verified():
    """正式 supplemental manifest 必须由公开 parent 重建且仍为零执行。"""
    manifest = read_w02_lc16_supplemental_manifest(FORMAL_MANIFEST)
    assert manifest == build_w02_lc16_supplemental_manifest(ROOT)
    verify_w02_lc16_supplemental_files(manifest, repository_root=ROOT)
    assert manifest.parent_overlay_sha256 == OVERLAY_SHA256
    state = manifest.execution_state.to_value()
    assert state["runtime_observed"] == 0
    assert state["w02_lc16_supplemental_qualified"] == 0


def test_manifest_scope_freezes_63_cases_and_189_direction_evaluations():
    """预注册 manifest 必须明确九载体、七类样本、三向和资源硬界。"""
    value = read_w02_lc16_supplemental_manifest(FORMAL_MANIFEST).scope.to_value()
    assert value["carrier_count"] == 9
    assert value["sample_kind_count"] == 7
    assert value["case_count"] == 63
    assert value["direction_evaluations"] == 189
    assert value["max_host_writes"] == 0
    assert value["max_workers"] == 4


def test_aggregate_requires_exact_carrier_case_owner_direction_coverage(overlay):
    """缺一条、重复一条或 owner 错绑都不能形成资格 receipt。"""
    values = list(_results(overlay))
    with pytest.raises(W02Lc16SupplementalError, match="189"):
        aggregate_w02_lc16_supplemental(
            overlay, tuple(values[:-1]), host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)
    values[-1] = values[0]
    with pytest.raises(W02Lc16SupplementalError, match="漂移"):
        aggregate_w02_lc16_supplemental(
            overlay, tuple(values), host_digests_before=_DIGESTS,
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
    with pytest.raises(W02Lc16SupplementalError, match="消融"):
        aggregate_w02_lc16_supplemental(
            overlay, tuple(values), host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=1,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1)


def test_runtime_zero_is_blocked_and_host_or_builder_write_is_rejected(overlay):
    """未运行只能 BLOCKED，host 写或复用 consumer builder 不能发布。"""
    blocked = aggregate_w02_lc16_supplemental(
        overlay, _results(overlay), host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS), private_path_reads=0,
        private_payload_bytes=1, private_payload_reads=1,
        evaluator_label_reads=1, runtime_observed=0)
    assert blocked.status == "BLOCKED"
    with pytest.raises(W02Lc16SupplementalError):
        aggregate_w02_lc16_supplemental(
            overlay, _results(overlay), host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1, host_write_count=1)
    with pytest.raises(W02Lc16SupplementalError):
        aggregate_w02_lc16_supplemental(
            overlay, _results(overlay), host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=1,
            evaluator_label_reads=1, consumer_result_builder_reused=1)


def test_observed_runtime_requires_private_read_evidence(overlay):
    """已观察的 private family 不得用零读取摘要伪造资格。"""
    with pytest.raises(W02Lc16SupplementalError, match="private read"):
        aggregate_w02_lc16_supplemental(
            overlay, _results(overlay), host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS), private_path_reads=0,
            private_payload_bytes=1, private_payload_reads=0,
            evaluator_label_reads=0)


def test_report_publication_is_append_only_and_private_free(tmp_path, report):
    """receipt 只含摘要且既存目标拒绝覆盖。"""
    target = tmp_path / "supplemental.json"
    publication = publish_w02_lc16_supplemental_report(target, report)
    original = target.read_bytes()
    assert publication.status == "PASS"
    assert b"expected_surface" not in original
    value, readback = read_w02_lc16_supplemental_report(target)
    assert value["case_count"] == 63
    assert readback == publication
    with pytest.raises(W02Lc16SupplementalPublicationError, match="禁止覆盖"):
        publish_w02_lc16_supplemental_report(target, report)
    assert target.read_bytes() == original


def test_receipt_readback_revalidates_the_full_public_contract(tmp_path, report):
    """canonical bytes 通过后仍必须满足摘要合同，不能只过字段扫描。"""
    target = tmp_path / "receipt.json"
    publish_w02_lc16_supplemental_report(target, report)
    value = json.loads(target.read_text("utf-8"))
    value["private_reads"]["private_payload_reads"] = 0
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(W02Lc16SupplementalPublicationError, match="合同"):
        read_w02_lc16_supplemental_report(target)


def test_manifest_writer_is_idempotent_but_rejects_different_bytes(tmp_path):
    """预注册 manifest 写入保持追加式，不接受异内容覆盖。"""
    manifest = build_w02_lc16_supplemental_manifest(ROOT)
    target = tmp_path / "manifest.json"
    assert write_w02_lc16_supplemental_manifest(manifest, target) == target
    assert write_w02_lc16_supplemental_manifest(manifest, target) == target
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(W02Lc16SupplementalError, match="内容不同"):
        write_w02_lc16_supplemental_manifest(manifest, target)


def test_manifest_mutation_is_fail_closed():
    """改变预算、parent 或执行状态都必须拒绝回读。"""
    manifest = read_w02_lc16_supplemental_manifest(FORMAL_MANIFEST)
    for mutate in (
            lambda value: value["scope"].update({"max_host_writes": 1}),
            lambda value: value["execution_state"].update({"W04_STARTED": 1}),
            lambda value: value.update({"parent_overlay_sha256": "0" * 64}),
    ):
        value = copy.deepcopy(manifest.to_dict())
        mutate(value)
        with pytest.raises(W02Lc16SupplementalError):
            type(manifest).from_dict(value)
