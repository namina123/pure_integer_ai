"""MD-01 中心扩域纯合同、失败关闭和不可覆盖 manifest 测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    EXPANSION_CHANNELS,
    MD_METRIC_KEYS,
    MD01ContractManifest,
    MemoryAttentionCenter,
    MemoryCenterOrigin,
    MemoryDynamicsBoundary,
    MemoryDynamicsContractError,
    MemoryDynamicsRunReport,
    MemoryDynamicsStopDecision,
    MemoryExpansionChannelBudget,
    MemoryExpansionProfile,
    MemoryRingReceipt,
    build_md01_contract_manifest,
    read_md01_contract_manifest,
    write_md01_contract_manifest,
    zero_execution_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v17.json"
MANIFEST_PATH = REPO_ROOT / "data/ph2/manifests/md01_memory_dynamics_contract_v1.json"


def _key(*values: int) -> StableRecordKey:
    """建立测试使用的一等严格整数键。"""
    return StableRecordKey(tuple(values))


def _boundary(seed: int = 1) -> MemoryDynamicsBoundary:
    """建立 owner/scope/source/version 不混叠的边界。"""
    return MemoryDynamicsBoundary(
        _key(seed, 1), _key(seed, 2), _key(seed, 3), _key(seed, 4))


def _center(seed: int = 1) -> MemoryAttentionCenter:
    """建立一个强制理解中心，召回结果只取得 activation。"""
    origin = MemoryCenterOrigin(
        "OCCURRENCE", _key(seed, 10), (_key(seed, 11),))
    return MemoryAttentionCenter(
        _key(seed, 20),
        "UNDERSTANDING",
        "MANDATORY",
        (origin,),
        _key(seed, 21),
        _key(seed, 22),
        _boundary(seed),
        _key(seed, 23),
        _key(seed, 24),
        (_key(seed, 11),),
        "ACTIVE",
        1,
    )


def _channel(channel: str, *, enabled: int = 1) -> MemoryExpansionChannelBudget:
    """建立一个有界通道，关闭时所有预算严格归零。"""
    if enabled == 0:
        values = (0,) * 8
    else:
        values = (16, 8, 4, 4, 2, 2, 32, 4096)
    return MemoryExpansionChannelBudget(channel, enabled, *values)


def _profile(seed: int = 1) -> MemoryExpansionProfile:
    """建立列全 L0-L4 与专用索引的 typed 扩域 profile。"""
    return MemoryExpansionProfile(
        _key(seed, 24),
        (_key(seed, 21),),
        EXPANSION_CHANNELS,
        tuple(_channel(channel) for channel in EXPANSION_CHANNELS),
        (_key(seed, 30), _key(seed, 31)),
        ("GRAPH_DISTANCE", "LOGICAL_TIME", "SCOPE_DISTANCE"),
        ("OBLIGATION_MATCH", "SOURCE_INDEPENDENCE", "SUPPORT_REFUTE"),
        ("EVIDENCE_STATE", "GROUNDING_BOUNDARY", "SCOPE_AUTHORIZATION"),
        ("ACCESS", "HELD_OUT", "OWNER", "REFUTE", "VERSION"),
        1,
        1,
        0,
        0,
    )


def _resolved(seed: int = 1) -> MemoryDynamicsStopDecision:
    """建立满足冲突检查和授权证据的 RESOLVED 决断。"""
    return MemoryDynamicsStopDecision(
        _key(seed, 60), _key(seed, 20), _boundary(seed), "RESOLVED",
        (_key(seed, 21),), (), (), (_key(seed, 61),), (_key(seed, 62),),
        (), (), None, 0, ("AUTHORIZED_AND_COMPLETE",), 0,
    )


def _receipt(seed: int = 1) -> MemoryRingReceipt:
    """建立一次带过滤原因、agenda、消费和停止指针的 ring receipt。"""
    return MemoryRingReceipt(
        _key(seed, 50),
        _key(seed, 20),
        _boundary(seed),
        "L2_EPISODE_DOCUMENT",
        _key(900, 1),
        10,
        15,
        (_key(seed, 10),),
        (_key(seed, 30),),
        (_key(seed, 40),),
        5,
        3,
        2,
        CanonicalJsonObject.from_value({"OWNER_MISMATCH": 1, "REFUTED": 1}),
        (_key(seed, 41),),
        ("OBLIGATION_MATCH",),
        (_key(seed, 41),),
        (_key(seed, 42),),
        (_key(seed, 11),),
        0,
        1,
        0,
        "STOP",
        _key(seed, 60),
        0,
    )


def _metrics(**overrides: int) -> CanonicalJsonObject:
    """建立列全预注册维度的严格整数指标。"""
    values = {key: 0 for key in MD_METRIC_KEYS}
    values.update(overrides)
    return CanonicalJsonObject.from_value(values)


def test_center_round_trip_and_activation_only_fail_closed():
    """center 保留完整边界和 origin，activation 不得自动授权 adopted。"""
    center = _center()
    assert MemoryAttentionCenter.from_dict(center.to_dict()) == center
    with pytest.raises(MemoryDynamicsContractError, match="activation"):
        replace(center, activation_only=0)
    with pytest.raises(MemoryDynamicsContractError, match="排序"):
        replace(center, origins=(center.origins[0], center.origins[0]))


def test_profile_lists_every_channel_and_hard_veto_cannot_be_scored_away():
    """profile 必须列全通道、逐通道预算，并禁止总分越过 veto。"""
    profile = _profile()
    assert MemoryExpansionProfile.from_dict(profile.to_dict()) == profile
    with pytest.raises(MemoryDynamicsContractError, match="全部扩域通道"):
        replace(profile, channel_order=profile.channel_order[:-1],
                channel_budgets=profile.channel_budgets[:-1])
    with pytest.raises(MemoryDynamicsContractError, match="总分"):
        replace(profile, global_score_can_override_veto=1)
    with pytest.raises(MemoryDynamicsContractError, match="预算必须全零"):
        replace(_channel("L4_SEALED_PAGE"), admission_enabled=0)


def test_receipt_round_trip_counts_and_stop_pointer_fail_closed():
    """receipt 的过滤计数、消费来源和 STOP 指针必须完整闭合。"""
    receipt = _receipt()
    assert MemoryRingReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(MemoryDynamicsContractError, match="过滤原因计数"):
        replace(receipt, filtered_count=3)
    with pytest.raises(MemoryDynamicsContractError, match="停止决断"):
        replace(receipt, stop_decision_key=None)
    with pytest.raises(MemoryDynamicsContractError, match="host writes"):
        replace(receipt, host_learning_write_count=1)


def test_two_centers_may_share_physical_read_without_identity_merge():
    """不同 center 可共享一次物理读，但 receipt 仍保留各自 center 身份。"""
    first = _receipt(1)
    second = replace(
        _receipt(2),
        physical_read_key=first.physical_read_key,
    )
    assert first.physical_read_key == second.physical_read_key
    assert first.center_key != second.center_key
    assert first.boundary != second.boundary


def test_resolved_requires_conflict_checks_and_authorization():
    """RESOLVED 缺硬冲突检查或授权 Evidence 时必须失败。"""
    decision = _resolved()
    assert MemoryDynamicsStopDecision.from_dict(decision.to_dict()) == decision
    with pytest.raises(MemoryDynamicsContractError, match="充分性"):
        replace(decision, hard_conflict_check_keys=())
    with pytest.raises(MemoryDynamicsContractError, match="充分性"):
        replace(decision, authorization_evidence_keys=())


def test_unknown_budget_blocked_and_clarify_are_distinct():
    """unknown、预算耗尽、访问阻断和澄清不得压成同一低置信状态。"""
    unresolved = (_key(1, 21),)
    unknown = MemoryDynamicsStopDecision(
        _key(1, 70), _key(1, 20), _boundary(), "UNKNOWN", (), unresolved,
        (), (), (), (), (), None, 0, ("NO_ADMISSIBLE_CANDIDATE",), 0)
    budget = MemoryDynamicsStopDecision(
        _key(1, 71), _key(1, 20), _boundary(), "BUDGET_EXHAUSTED", (),
        unresolved, (), (), (), (), ("L4_SEALED_PAGE",), None, 1,
        ("PAGE_BUDGET_EXHAUSTED",), 0)
    blocked = MemoryDynamicsStopDecision(
        _key(1, 72), _key(1, 20), _boundary(), "ACCESS_BLOCKED", (),
        unresolved, (), (), (), (_key(1, 73),), (), None, 0,
        ("OWNER_ACCESS_DENIED",), 0)
    clarify = MemoryDynamicsStopDecision(
        _key(1, 74), _key(1, 20), _boundary(), "CLARIFY", (), unresolved,
        (_key(1, 75), _key(1, 76)), (), (), (), (), None, 0,
        ("INCOMPARABLE_CANDIDATES",), 0)
    assert {item.status for item in (unknown, budget, blocked, clarify)} == {
        "ACCESS_BLOCKED", "BUDGET_EXHAUSTED", "CLARIFY", "UNKNOWN"}
    with pytest.raises(MemoryDynamicsContractError, match="UNKNOWN"):
        replace(unknown, remaining_channel_keys=("L4_SEALED_PAGE",))
    with pytest.raises(MemoryDynamicsContractError, match="预算耗尽"):
        replace(budget, budget_exhausted=0)
    with pytest.raises(MemoryDynamicsContractError, match="竞争"):
        replace(clarify, conflict_keys=(_key(1, 75),))


def test_run_report_is_result_blind_until_probe_and_pass_respects_invariants():
    """MD-01 未运行报告全零，实际 PASS 不能掩盖硬不变量失败。"""
    not_started = MemoryDynamicsRunReport(
        1, "md01-contract-only-v1", _key(80),
        "MD-00-center-expansion-preregistration-v1", "FIXED_TOP_K",
        "NOT_STARTED", (), (), (), (), _metrics(), (), 0,
        "NOT_EVALUATED", zero_execution_state())
    assert MemoryDynamicsRunReport.from_dict(not_started.to_dict()) == not_started
    complete = replace(
        not_started,
        report_version="md-probe-result-v1",
        run_status="COMPLETE",
        center_keys=(_key(1, 20),),
        profile_keys=(_key(1, 24),),
        receipt_keys=(_key(1, 50),),
        stop_decision_keys=(_key(1, 60),),
        metric_values=_metrics(ADOPTED_CORRECT=1),
        results_observed=1,
        probe_decision="PASS",
    )
    assert complete.probe_decision == "PASS"
    with pytest.raises(MemoryDynamicsContractError, match="硬不变量"):
        replace(complete, hard_invariant_failures=("OWNER_SCOPE_VERSION",))
    with pytest.raises(MemoryDynamicsContractError, match="未运行报告"):
        replace(not_started, metric_values=_metrics(SCANNED_OBJECTS=1))


def test_manifest_round_trip_nonoverwrite_and_strict_fields(tmp_path):
    """MD-01 manifest 规范回读、幂等发布并拒绝覆盖和额外字段。"""
    digest = hashlib.sha256(b"baseline").hexdigest()
    manifest = build_md01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v17.json"),
        prerequisite_manifest_sha256=digest,
    )
    output = tmp_path / "md01.json"
    write_md01_contract_manifest(manifest, output)
    assert read_md01_contract_manifest(output) == manifest
    write_md01_contract_manifest(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(MemoryDynamicsContractError, match="内容不同"):
        write_md01_contract_manifest(manifest, output)
    value = manifest.to_dict()
    value["mastered"] = 1
    output.write_bytes(canonical_json_line(value))
    with pytest.raises(MemoryDynamicsContractError, match="字段不精确"):
        read_md01_contract_manifest(output)


def test_manifest_rejects_fake_runtime_results_and_bad_prerequisite():
    """合同冻结不得伪造 runtime 结果，前置路径和 hash 必须可迁移。"""
    digest = hashlib.sha256(b"baseline").hexdigest()
    manifest = build_md01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v17.json"),
        prerequisite_manifest_sha256=digest,
    )
    with pytest.raises(MemoryDynamicsContractError, match="runtime"):
        replace(manifest, runtime_status="CONNECTED")
    with pytest.raises(MemoryDynamicsContractError, match="results observed"):
        replace(manifest, results_observed=1)
    with pytest.raises(MemoryDynamicsContractError, match="安全 POSIX"):
        replace(manifest, prerequisite_manifest_relative_path="../private.json")
    with pytest.raises(MemoryDynamicsContractError, match="SHA-256"):
        replace(manifest, prerequisite_manifest_sha256="bad")


def test_repository_md01_manifest_matches_current_baseline():
    """正式 MD-01 artifact 必须精确绑定当前 v17 基线且保持零执行。"""
    baseline_hash = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    manifest = read_md01_contract_manifest(MANIFEST_PATH)
    expected = build_md01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v17.json"),
        prerequisite_manifest_sha256=baseline_hash,
    )
    assert manifest == expected
    assert manifest.execution_state.to_value() == {
        key: 0 for key in sorted(manifest.execution_state.to_value())}


def test_manifest_constructor_rejects_missing_contract_type():
    """五类合同少任一类都不得宣称 MD-01 已冻结。"""
    digest = hashlib.sha256(b"baseline").hexdigest()
    manifest = build_md01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v17.json"),
        prerequisite_manifest_sha256=digest,
    )
    with pytest.raises(MemoryDynamicsContractError, match="contract types"):
        MD01ContractManifest(
            manifest.format_version,
            manifest.artifact_version,
            manifest.artifact_status,
            manifest.task_keys,
            manifest.md00_preregistration_version,
            manifest.prerequisite_manifest_relative_path,
            manifest.prerequisite_manifest_sha256,
            manifest.contract_type_keys[:-1],
            manifest.direction_keys,
            manifest.strength_keys,
            manifest.channel_keys,
            manifest.stop_state_keys,
            manifest.invariant_keys,
            manifest.reused_component_refs,
            manifest.verifier_dimensions,
            manifest.verifier_ne_conditions,
            manifest.runtime_status,
            manifest.results_observed,
            manifest.execution_state,
        )
