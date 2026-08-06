"""J-LC-PRE-W04 逐字节合取、状态边界与 append-only T0。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_j_lc_pre_w04_catalog import (
    build_j_lc_pre_w04_gate,
)
from pure_integer_ai.experiments.ph2_j_lc_pre_w04_contract import (
    DEPENDENCY_ROLES,
    MANIFEST_PATH,
    OPEN_GENERATION_SUFFIX,
    PUBLISHED_STATE,
    W04_BLOCKING_FAILURE_KEYS,
    JLcPreW04Error,
    JLcPreW04Gate,
    read_j_lc_pre_w04_gate,
    verify_j_lc_pre_w04_files,
    write_j_lc_pre_w04_gate,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    IN_SCOPE_CARRIER_KEYS,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    CarrierProjectionRuntimeContractError,
)


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / Path(*MANIFEST_PATH.split("/"))


@pytest.fixture(scope="module")
def built():
    """读取冻结的 pre-W04 gate；当前 parent 漂移不得重建旧 gate。"""
    return read_j_lc_pre_w04_gate(GATE_PATH)


def test_stored_gate_is_canonical_and_current_parent_drift_fails_closed(built):
    """stored gate 保持 canonical；当前 parent 漂移必须 fail closed。"""
    stored = read_j_lc_pre_w04_gate(GATE_PATH)
    assert stored == built
    assert stored.canonical_bytes() == GATE_PATH.read_bytes()
    with pytest.raises(JLcPreW04Error, match="漂移"):
        verify_j_lc_pre_w04_files(stored, repository_root=ROOT)


def test_current_pre_w04_builder_rejects_historical_parent_drift():
    """旧 parent 身份漂移时，catalog builder 不得伪造新的 PASS gate。"""
    with pytest.raises(CarrierProjectionRuntimeContractError, match="文件身份漂移"):
        build_j_lc_pre_w04_gate(ROOT)


def test_gate_binds_all_required_parents_and_nine_carriers(built):
    """九类 parent、mapper/runtime/overlay/receipts 均须逐字节绑定。"""
    assert tuple(item.role for item in built.dependencies) == DEPENDENCY_ROLES
    assert tuple(item.carrier_key for item in built.carrier_bindings) \
        == IN_SCOPE_CARRIER_KEYS
    assert len({item.manifest_identity.sha256
                for item in built.carrier_bindings}) == 9
    assert len({item.sample_identity.sha256
                for item in built.carrier_bindings}) == 9
    assert built.original_w02_receipt.json_field == "w02_receipt_sha256"
    original_w03 = built.dependencies[DEPENDENCY_ROLES.index(
        "ORIGINAL_W03_RECEIPT_WITH_W02_COMMITMENT")]
    assert built.original_w02_receipt.source_sha256 == original_w03.sha256


def test_gate_only_publishes_five_allowed_state_values(built):
    """gate 只允许 W04_ALLOWED，不启动 W-04 或声明 mastered/readiness。"""
    assert built.published_state == PUBLISHED_STATE == {
        "J_LC_PRE_W04": "PASS",
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W04_ALLOWED": 1,
        "W04_STARTED": 0,
    }


def test_failure_suffix_and_open_generation_ne_are_preserved(built):
    """W-04 blockers 全解决；open generation NE 只传播 W-08+。"""
    assert built.w04_blocking_failure_keys == W04_BLOCKING_FAILURE_KEYS
    assert built.resolved_w04_blocking_failure_keys == W04_BLOCKING_FAILURE_KEYS
    assert built.unresolved_w04_blocking_failure_keys == ()
    assert built.open_generation.current_status == "NE_NOT_YET_EVALUABLE"
    assert built.open_generation.failure_suffix == OPEN_GENERATION_SUFFIX
    assert built.open_generation.blocks_w04 == 0
    assert built.open_generation.aggregate_with_source_replay == 0


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["published_state"].update({"W04_STARTED": 1}),
        lambda value: value["published_state"].update({
            "LANGUAGE_CAPABILITY_MASTERED": 1}),
        lambda value: value["unresolved_w04_blocking_failure_keys"].append(
            "W03_SUPPLEMENTAL_FAIL_OR_NE"),
        lambda value: value["open_generation"].update({
            "current_status": "PASS"}),
        lambda value: value["carrier_bindings"].pop(),
        lambda value: value["supplemental_receipt_statuses"].update({
            "W03_LC16_SUPPLEMENTAL": "NE"}),
    ),
)
def test_contract_fails_closed_on_gate_drift(built, mutate):
    """状态、failure、open-generation、carrier 或 receipt 漂移均拒绝。"""
    value = deepcopy(built.to_dict())
    mutate(value)
    with pytest.raises(JLcPreW04Error):
        JLcPreW04Gate.from_dict(value)


def test_gate_writer_is_strictly_append_only(built, tmp_path):
    """gate 只可首次创建，重复同 bytes 也不得重发。"""
    target = tmp_path / "gate.json"
    write_j_lc_pre_w04_gate(built, target)
    assert target.read_bytes() == built.canonical_bytes()
    with pytest.raises(JLcPreW04Error, match="已存在"):
        write_j_lc_pre_w04_gate(built, target)
