"""LC-13 三向 consumer、exact Use/outcome 和 postcheck 测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_directional_consumer_catalog import (
    LC13_MANIFEST_PATH,
    build_directional_consumer_manifest,
)
from pure_integer_ai.experiments.ph2_directional_consumer_contract import (
    DIRECTIONS,
    EXECUTION_STATE,
    POSTCHECK_DIMENSIONS,
    DirectionalConsumerContractError,
    DirectionalConsumerManifest,
    read_directional_consumer_manifest,
    write_directional_consumer_manifest,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST_SHA256 = (
    "81fba102d31776518c2a34180ebdf3f90bdf759bd4add549bafcd69bb6f49765")


@pytest.fixture(scope="module")
def formal_manifest() -> DirectionalConsumerManifest:
    return build_directional_consumer_manifest(REPOSITORY)


def _route(formal_manifest, capability: str, direction: str):
    return next(
        item for item in formal_manifest.routes
        if item.capability_key == capability and item.direction == direction)


def test_every_capability_has_three_explicit_direction_routes(formal_manifest):
    """20 个能力的 Understanding/Reasoning/Generation 必须全列且唯一。"""
    assert formal_manifest.route_count == 60
    assert tuple(
        item.route_key for item in formal_manifest.routes) == tuple(
            f"{capability}/{direction}"
            for capability in CAPABILITY_KEYS for direction in DIRECTIONS)
    assert formal_manifest.available_not_executed_count == 11
    assert formal_manifest.missing_ne_count == 46
    assert formal_manifest.out_of_scope_count == 3
    assert formal_manifest.runtime_connected_count == 0


def test_existing_facility_is_not_runtime_connection_or_directional_pass(
        formal_manifest):
    """源码设施只能登记 AVAILABLE_NOT_EXECUTED，不能据此发能力 PASS。"""
    available = tuple(
        item for item in formal_manifest.routes
        if item.consumer_state == "AVAILABLE_NOT_EXECUTED")
    assert len(available) == 11
    assert all(item.consumer_refs for item in available)
    assert all(item.directional_verdict == "NE" for item in available)
    assert all("NO_HOST_LEARNING_WRITE" in item.write_permissions
               for item in available)
    assert all("LC13_ROUTE_NOT_EXECUTED" in item.ne_conditions
               for item in available)
    assert all(item.postcheck_state == "AVAILABLE_NOT_EXECUTED"
               for item in available)


def test_missing_consumers_are_ne_with_no_owner_or_write_permission(
        formal_manifest):
    """缺 consumer 的方向必须显式 NE，不能从课程存在推成双向。"""
    missing = tuple(
        item for item in formal_manifest.routes
        if item.consumer_state == "MISSING_NE")
    assert len(missing) == 46
    for item in missing:
        assert item.consumer_refs == ()
        assert item.owner_key == "UNASSIGNED_NE"
        assert item.write_permissions == ()
        assert item.exact_use_outcome_state == "REQUIRED_NOT_CONNECTED"
        assert item.postcheck_state == "REQUIRED_NOT_CONNECTED"
        assert item.directional_verdict == "NE"
        assert "DIRECTIONAL_CONSUMER_NOT_CONNECTED" in item.ne_conditions


def test_non_text_wall_is_out_of_scope_in_all_three_directions(formal_manifest):
    """W1/W2 非文本墙外不能被 LC-13 偷渡为文本能力。"""
    routes = tuple(
        item for item in formal_manifest.routes
        if item.capability_key == "NON_TEXT_MEDIA")
    assert len(routes) == 3
    assert {item.consumer_state for item in routes} == {"OUT_OF_SCOPE"}
    assert {item.directional_verdict for item in routes} == {"OUT_OF_SCOPE"}
    assert all(item.ne_conditions == ("NON_TEXT_WALL_OUT_OF_SCOPE",)
               for item in routes)


def test_only_gg02_generation_has_exact_use_outcome_contract(formal_manifest):
    """只有 GG-02 五层生成合同可声明 exact Use/outcome，且仍未消费。"""
    exact = tuple(
        item for item in formal_manifest.routes
        if item.exact_use_outcome_state == "CONTRACT_FROZEN_NOT_CONSUMED")
    assert formal_manifest.exact_use_outcome_contract_count == 1
    assert len(exact) == 1
    assert exact[0].route_key == "LAYERED_GENERATION/GENERATION"
    assert "ASSESSMENT_CONSUMER_NOT_CONNECTED" in exact[0].ne_conditions
    assert _route(
        formal_manifest, "LAYERED_GENERATION", "REASONING"
    ).consumer_state == "MISSING_NE"


def test_postchecks_are_direction_specific_and_never_sentence_broadcast(
        formal_manifest):
    """三层 postcheck 判据正交，任何一层都没有整句广播权限。"""
    for item in formal_manifest.routes:
        assert item.postcheck_dimensions == POSTCHECK_DIMENSIONS[item.direction]
    assert "LAYER_OUTCOME_LOCALITY" in POSTCHECK_DIMENSIONS["GENERATION"]
    assert "PROOF_DIRECTION" in POSTCHECK_DIMENSIONS["REASONING"]
    assert "OBJECT_IDENTITY" in POSTCHECK_DIMENSIONS["UNDERSTANDING"]
    assert all("SENTENCE_WIDE_BROADCAST" not in item.postcheck_dimensions
               for item in formal_manifest.routes)


def test_consumer_file_hash_inventory_is_closed(formal_manifest):
    """已有 consumer 引用必须逐字节闭合，私有绝对路径不得进入 artifact。"""
    refs = {
        path for route in formal_manifest.routes for path in route.consumer_refs}
    identities = {
        item.relative_path: item for item in formal_manifest.evidence_files}
    assert refs == set(identities)
    assert len(identities) == 11
    for relative_path, identity in identities.items():
        payload = (REPOSITORY / Path(*relative_path.split("/"))).read_bytes()
        assert len(payload) == identity.byte_count
        assert hashlib.sha256(payload).hexdigest() == identity.sha256


def test_bad_owner_write_permission_and_exact_use_scope_fail_closed(
        formal_manifest):
    """可用路由缺 owner/零写权限或跨层冒充 exact outcome 必须拒绝。"""
    available = next(
        item for item in formal_manifest.routes
        if item.consumer_state == "AVAILABLE_NOT_EXECUTED"
        and item.route_key != "LAYERED_GENERATION/GENERATION")
    with pytest.raises(DirectionalConsumerContractError):
        replace(available, owner_key="UNASSIGNED_NE")
    with pytest.raises(DirectionalConsumerContractError):
        replace(available, write_permissions=())
    with pytest.raises(DirectionalConsumerContractError):
        replace(
            available,
            exact_use_outcome_state="CONTRACT_FROZEN_NOT_CONSUMED",
            ne_conditions=tuple(sorted({
                *available.ne_conditions,
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
            })),
        )


def test_missing_route_cannot_gain_consumer_or_pass_without_full_contract(
        formal_manifest):
    """缺失槽不能只塞文件、owner 或 PASS 字样绕过 LC-13。"""
    missing = next(
        item for item in formal_manifest.routes
        if item.consumer_state == "MISSING_NE")
    with pytest.raises(DirectionalConsumerContractError):
        replace(missing, owner_key="FAKE_OWNER")
    with pytest.raises(DirectionalConsumerContractError):
        replace(missing, directional_verdict="PASS")


def test_manifest_round_trip_nonoverwrite_and_zero_execution(
        tmp_path, formal_manifest):
    """manifest 可恢复、不可覆盖，且任何运行或学习状态都固定为零。"""
    path = tmp_path / "lc13.json"
    assert write_directional_consumer_manifest(formal_manifest, path) == path
    assert write_directional_consumer_manifest(formal_manifest, path) == path
    assert read_directional_consumer_manifest(path) == formal_manifest
    assert formal_manifest.artifact_status == "COURSE_FROZEN"
    assert formal_manifest.runtime_status == "NOT_STARTED"
    assert formal_manifest.execution_state.to_value() == EXECUTION_STATE
    state = dict(EXECUTION_STATE)
    state["teacher_calls"] = 1
    with pytest.raises(DirectionalConsumerContractError):
        replace(
            formal_manifest,
            execution_state=CanonicalJsonObject.from_value(state),
        )
    path.write_bytes(b"{}\n")
    with pytest.raises(DirectionalConsumerContractError):
        write_directional_consumer_manifest(formal_manifest, path)


def test_manifest_rejects_missing_route_or_evidence_file(formal_manifest):
    """漏任一方向槽或 consumer 文件身份都必须失败关闭。"""
    with pytest.raises(DirectionalConsumerContractError):
        replace(formal_manifest, routes=formal_manifest.routes[:-1])
    with pytest.raises(DirectionalConsumerContractError):
        replace(
            formal_manifest,
            evidence_files=formal_manifest.evidence_files[:-1],
        )


def test_repository_formal_artifact_matches_builder(formal_manifest):
    """正式不可覆盖 artifact 必须逐字节等于当前 v32 builder。"""
    path = REPOSITORY / LC13_MANIFEST_PATH
    assert path.is_file()
    payload = path.read_bytes()
    assert payload == formal_manifest.canonical_bytes()
    assert hashlib.sha256(payload).hexdigest() == FORMAL_MANIFEST_SHA256
    assert read_directional_consumer_manifest(path) == formal_manifest
