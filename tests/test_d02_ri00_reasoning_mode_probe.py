"""RI-00 temporal/abduction/counterfactual/default/deontic 裁决测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_reasoning_mode_probe_catalog import (
    RI00_MANIFEST_PATH,
    build_reasoning_mode_probe_manifest,
)
from pure_integer_ai.experiments.ph2_reasoning_mode_probe_contract import (
    EXECUTION_STATE,
    MODE_KEYS,
    ReasoningModeProbeContractError,
    ReasoningModeProbeManifest,
    evaluate_reasoning_mode_probe,
    read_reasoning_mode_probe_manifest,
    write_reasoning_mode_probe_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST_SHA256 = (
    "5d226419d523ccef1522975d66ac40f10489b4790e3f1499397743cc650fb92a")


@pytest.fixture(scope="module")
def formal_manifest() -> ReasoningModeProbeManifest:
    return build_reasoning_mode_probe_manifest(REPOSITORY)


def _decision(formal_manifest, mode_key: str):
    return next(item for item in formal_manifest.decisions
                if item.mode_key == mode_key)


def test_five_modes_have_direct_bounded_verdicts(formal_manifest):
    """五类模式必须独立裁决，不用总分掩盖四项缺口。"""
    assert tuple(item.mode_key for item in formal_manifest.decisions) == MODE_KEYS
    assert {item.mode_key: item.verdict for item in formal_manifest.decisions} == {
        "ABDUCTION": "REJECT",
        "COUNTERFACTUAL": "REJECT",
        "DEFEASIBLE_DEFAULT": "REJECT",
        "DEONTIC_NORMATIVE": "REJECT",
        "TEMPORAL": "PASS",
    }
    assert (formal_manifest.pass_count, formal_manifest.reject_count,
            formal_manifest.ne_count) == (1, 4, 0)
    assert formal_manifest.runtime_pass_authority == 0


def test_temporal_pass_is_narrow_and_not_runtime_capability_pass(formal_manifest):
    """temporal PASS 只覆盖 typed precedence/event-time 三项 invariant。"""
    temporal = _decision(formal_manifest, "TEMPORAL")
    assert temporal.scope_decision == "BOUNDED_TYPED_PRECEDENCE_EVENT_TIME_ONLY"
    assert temporal.typed_mode_available == 1
    assert set(temporal.invariant_results.to_value().values()) == {"PASS"}
    assert temporal.representation_state == "AVAILABLE_NOT_EXECUTED"
    assert formal_manifest.runtime_status == "NOT_CONNECTED"


def test_abduction_never_mints_causes_and_missing_branch_rejects(formal_manifest):
    """解释候选不能反向新造 CAUSES，缺 typed branch 必须 REJECT。"""
    abduction = _decision(formal_manifest, "ABDUCTION")
    assert abduction.invariant_results.to_value() == {
        "NO_NEW_CAUSES": "PASS",
        "TYPED_ABDUCTIVE_BRANCH": "REJECT",
    }
    assert abduction.typed_mode_available == 0
    with pytest.raises(ReasoningModeProbeContractError):
        replace(
            abduction,
            invariant_results=CanonicalJsonObject.from_value({
                "NO_NEW_CAUSES": "REJECT",
                "TYPED_ABDUCTIVE_BRANCH": "REJECT",
            }),
        )


def test_counterfactual_does_not_fake_current_projection_isolation_pass(
        formal_manifest):
    """没有隔离分支 runtime 时，current projection 零污染只能是 NE。"""
    counterfactual = _decision(formal_manifest, "COUNTERFACTUAL")
    assert counterfactual.invariant_results.to_value() == {
        "COUNTERFACTUAL_BRANCH_RUNTIME": "REJECT",
        "CURRENT_PROJECTION_POLLUTION_ZERO": "NE",
    }
    with pytest.raises(ReasoningModeProbeContractError):
        replace(
            counterfactual,
            invariant_results=CanonicalJsonObject.from_value({
                "COUNTERFACTUAL_BRANCH_RUNTIME": "REJECT",
                "CURRENT_PROJECTION_POLLUTION_ZERO": "PASS",
            }),
        )


def test_default_and_deontic_gaps_are_separate_rejects(formal_manifest):
    """revision/modal 脚手架不得冒充 default 撤销或规范事实分离。"""
    default = _decision(formal_manifest, "DEFEASIBLE_DEFAULT")
    deontic = _decision(formal_manifest, "DEONTIC_NORMATIVE")
    assert default.invariant_results.to_value()[
        "DEFAULT_EXCEPTION_REVERSAL"] == "REJECT"
    assert deontic.invariant_results.to_value()[
        "NORMATIVE_FACT_PROJECTION_SEPARATION"] == "REJECT"
    assert default.scope_decision != deontic.scope_decision


def test_evaluator_uses_worst_invariant_without_averaging():
    """一个 REJECT 或 NE 必须保留，不能被其余 PASS 平均掉。"""
    assert evaluate_reasoning_mode_probe(
        CanonicalJsonObject.from_value({"A": "PASS", "B": "REJECT"}),
        forbidden_side_effect_count=0,
        host_learning_writes=0,
    ) == "REJECT"
    assert evaluate_reasoning_mode_probe(
        CanonicalJsonObject.from_value({"A": "NE", "B": "PASS"}),
        forbidden_side_effect_count=0,
        host_learning_writes=0,
    ) == "NE"
    assert evaluate_reasoning_mode_probe(
        CanonicalJsonObject.from_value({"A": "PASS"}),
        forbidden_side_effect_count=1,
        host_learning_writes=0,
    ) == "REJECT"


def test_evidence_inventory_binds_current_typed_facilities(formal_manifest):
    """所有 PASS/REJECT/NE 裁决必须绑定当前文件，而非历史口述。"""
    refs = {path for item in formal_manifest.decisions
            for path in item.evidence_refs}
    identities = {item.relative_path: item
                  for item in formal_manifest.evidence_files}
    assert refs == set(identities)
    for relative_path, identity in identities.items():
        payload = (REPOSITORY / Path(*relative_path.split("/"))).read_bytes()
        assert len(payload) == identity.byte_count
        assert hashlib.sha256(payload).hexdigest() == identity.sha256


def test_zero_execution_and_runtime_pass_boundaries(formal_manifest):
    """bounded 决断不得写学习状态或签发 runtime 能力 PASS。"""
    assert formal_manifest.execution_state.to_value() == EXECUTION_STATE
    assert all(value == 0 for value in EXECUTION_STATE.values())
    state = dict(EXECUTION_STATE)
    state["teacher_calls"] = 1
    with pytest.raises(ReasoningModeProbeContractError):
        replace(
            formal_manifest,
            execution_state=CanonicalJsonObject.from_value(state),
        )
    with pytest.raises(ReasoningModeProbeContractError):
        replace(formal_manifest, runtime_pass_authority=1)


def test_manifest_round_trip_nonoverwrite_and_corruption(
        tmp_path, formal_manifest):
    """RI-00 artifact 可规范恢复、幂等发布并拒绝覆盖损坏。"""
    path = tmp_path / "ri00.json"
    assert write_reasoning_mode_probe_manifest(formal_manifest, path) == path
    assert write_reasoning_mode_probe_manifest(formal_manifest, path) == path
    assert read_reasoning_mode_probe_manifest(path) == formal_manifest
    path.write_bytes(b"{}\n")
    with pytest.raises(ReasoningModeProbeContractError):
        write_reasoning_mode_probe_manifest(formal_manifest, path)
    with pytest.raises(ReasoningModeProbeContractError):
        read_reasoning_mode_probe_manifest(path)


def test_manifest_rejects_missing_mode_or_evidence_file(formal_manifest):
    """漏一类模式或一个 evidence 文件都不能形成范围裁决。"""
    with pytest.raises(ReasoningModeProbeContractError):
        replace(formal_manifest, decisions=formal_manifest.decisions[:-1])
    with pytest.raises(ReasoningModeProbeContractError):
        replace(formal_manifest, evidence_files=formal_manifest.evidence_files[:-1])


def test_repository_formal_artifact_matches_builder(formal_manifest):
    """正式不可覆盖 RI-00 artifact 必须逐字节绑定当前 builder。"""
    path = REPOSITORY / RI00_MANIFEST_PATH
    assert path.is_file()
    payload = path.read_bytes()
    assert payload == formal_manifest.canonical_bytes()
    assert hashlib.sha256(payload).hexdigest() == FORMAL_MANIFEST_SHA256
    assert read_reasoning_mode_probe_manifest(path) == formal_manifest
