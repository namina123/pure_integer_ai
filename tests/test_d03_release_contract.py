"""D-03 正式发布合同的阶段、可见性、恢复和失效反例。"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_release_contract import (
    D03ContractError,
    D03PublicationState,
    D03ReleaseIdentity,
    D03StageManifest,
    STAGE_KEYS,
    W06_SUBSTAGE_KEYS,
    W07_SUBSTAGE_KEYS,
    ZERO_EXECUTION_STATE,
    StageInvalidationGraph,
    validate_stage_manifest_set,
    write_immutable_json,
)


def _identity_payload() -> dict[str, object]:
    """返回最小但完整的 D-03 发布身份。"""
    return {
        "format_version": 1,
        "release_key": "PH2-D03-V1",
        "release_version": "PH2-D03-formal-release-v1",
        "parent_gate_path": "data/ph2/manifests/j_lg_d03_gate_v4.json",
        "parent_gate_sha256": "a" * 64,
        "capability_baseline_path": (
            "data/ph2/manifests/language_capability_baseline_v41.json"
        ),
        "capability_baseline_sha256": "b" * 64,
        "source_coverage_path": "data/ph2/manifests/d02_source_pack_coverage_v1.json",
        "source_coverage_sha256": "c" * 64,
        "version_keys": {
            "backend_version": "BACKEND-V1",
            "code_version": "CODE-V1",
            "course_version": "COURSE-V1",
            "data_version": "DATA-V1",
            "evaluator_version": "EVALUATOR-V1",
            "location_version": "LOCATION-V1",
            "parser_version": "PARSER-V1",
            "primitive_version": "PRIMITIVE-V1",
            "schema_version": "SCHEMA-V1",
            "segment_version": "SEGMENT-V1",
        },
    }


def _stage_payload(stage_key: str) -> dict[str, object]:
    """构造一个阶段合同 payload，供正反例作最小变异。"""
    ordinal = STAGE_KEYS.index(stage_key) + 1
    prerequisite = () if ordinal == 1 else (STAGE_KEYS[ordinal - 2],)
    substages: tuple[str, ...] = ()
    if stage_key == "W-06":
        substages = W06_SUBSTAGE_KEYS
    elif stage_key == "W-07":
        substages = W07_SUBSTAGE_KEYS
    train = tuple(f"PACK-{index:02d}" for index in range(2, ordinal + 1))
    future = tuple(f"PACK-{index:02d}" for index in range(ordinal + 1, 10))
    return {
        "artifact_kind": "PH2_D03_STAGE_MANIFEST",
        "artifact_version": f"PH2-D03-{stage_key.replace('-', '')}-v1",
        "execution_state": dict(ZERO_EXECUTION_STATE),
        "format_version": 1,
        "release_key": "PH2-D03-V1",
        "stage_identity": {
            "ordinal": ordinal,
            "prerequisite_stage_keys": list(prerequisite),
            "stage_key": stage_key,
            "substage_keys": list(substages),
        },
        "data_visibility": {
            "candidate_allowed_splits": ["train"],
            "candidate_forbidden_splits": [
                "dev", "held_out", "adversarial", "wall"
            ],
            "candidate_owner": "PH2_TRAIN_CANDIDATE",
            "dev_pack_keys": list(train),
            "evaluator_owner": "PH2_PRIVATE_EVALUATOR",
            "evaluator_pack_keys": list(train),
            "future_pack_keys": list(future),
            "held_out_pack_keys": list(train),
            "teacher_owner": "PH2_TRAINING_EVIDENCE",
            "train_pack_keys": list(train),
        },
        "evaluation_binding": {
            "ablation_keys": [f"{stage_key}-ABLATION"],
            "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
            "continuous_window_count": 1,
            "evaluator_key": f"{stage_key}-EVALUATOR",
            "evaluator_version": f"{stage_key}-EVALUATOR-V1",
            "owner_key": "PH2_PRIVATE_EVALUATOR",
            "thresholds": [{
                "bearing": 1,
                "dimension_key": f"{stage_key}-PROTOCOL",
                "max_fail_count": 0,
                "min_pass_denominator": 1,
                "min_pass_numerator": 1,
                "ne_policy": "BLOCK",
                "preregistered": 1,
            }],
        },
        "resource_budget": {
            "max_checkpoint_count": 64,
            "max_logic_operations": 100000,
            "max_payload_bytes": 1048576,
            "max_payload_gets": 4096,
            "max_recompute_objects": 10000,
            "max_records": 10000,
            "max_segments": 1024,
            "max_workers": 4,
        },
        "recovery_binding": {
            "allowed_worker_counts": [1, 2, 4],
            "base_fence_required": 1,
            "cursor_version": "PH2-D03-CURSOR-V1",
            "failure_point_keys": [
                "BEFORE_FIRST_SHARD", "AFTER_PARTIAL_SHARD",
                "BEFORE_MERGE_PREVIEW", "AFTER_MERGE_BEFORE_COMMIT",
                "AFTER_COMMIT_BEFORE_CURSOR", "AFTER_MANIFEST_PUBLISH",
            ],
            "fresh_resume_equivalent": 1,
            "logical_shard_count": 16,
            "merge_barrier_key": "PH2-D03-STABLE-MERGE-BARRIER-V1",
            "run_id_policy": "NEW_POSITIVE_INTEGER_REQUIRED",
        },
    }


def _graph_payload() -> dict[str, object]:
    """返回覆盖全阶段后缀的最小失效图。"""
    return {
        "artifact_kind": "PH2_D03_STAGE_INVALIDATION_GRAPH",
        "artifact_version": "PH2-D03-invalidation-graph-v1",
        "format_version": 1,
        "release_key": "PH2-D03-V1",
        "rules": [{
            "change_kind": "SCHEMA_VERSION",
            "earliest_stage": "W-01",
            "invalidated_stage_keys": list(STAGE_KEYS),
            "subject_key": "GLOBAL",
        }, {
            "change_kind": "PACK_CONTENT",
            "earliest_stage": "W-06",
            "invalidated_stage_keys": list(STAGE_KEYS[5:]),
            "subject_key": "PACK-06",
        }],
        "stage_edges": [
            {"consumer_stage": STAGE_KEYS[index],
             "prerequisite_stage": STAGE_KEYS[index - 1]}
            for index in range(1, len(STAGE_KEYS))
        ],
        "stage_keys": list(STAGE_KEYS),
    }


def test_release_identity_requires_every_frozen_version_key():
    """数据、课程、parser、原语和六类 v41 版本缺一即拒绝。"""
    payload = _identity_payload()
    identity = D03ReleaseIdentity.from_dict(payload)
    assert identity.to_dict() == payload
    for key in tuple(payload["version_keys"]):
        broken = copy.deepcopy(payload)
        del broken["version_keys"][key]
        with pytest.raises(D03ContractError, match="version"):
            D03ReleaseIdentity.from_dict(broken)


def test_stage_sequence_and_required_substage_order_are_exact():
    """九阶段、立即前置以及 W-06/W-07 子序不可缺失、跳级或乱序。"""
    stages = tuple(D03StageManifest.from_dict(_stage_payload(key)) for key in STAGE_KEYS)
    validate_stage_manifest_set(stages)

    missing = stages[:-1]
    with pytest.raises(D03ContractError, match="九阶段"):
        validate_stage_manifest_set(missing)

    reordered = list(stages)
    reordered[4], reordered[5] = reordered[5], reordered[4]
    with pytest.raises(D03ContractError, match="顺序"):
        validate_stage_manifest_set(tuple(reordered))

    for key, required in (("W-06", W06_SUBSTAGE_KEYS), ("W-07", W07_SUBSTAGE_KEYS)):
        broken = _stage_payload(key)
        broken["stage_identity"]["substage_keys"] = list(reversed(required))
        with pytest.raises(D03ContractError, match="子序"):
            D03StageManifest.from_dict(broken)


def test_visibility_rejects_future_and_private_owner_overlap():
    """future pack、私有 owner 和 candidate split 不能进入 train 白名单。"""
    payload = _stage_payload("W-04")
    payload["data_visibility"]["future_pack_keys"].append("PACK-03")
    with pytest.raises(D03ContractError, match="future"):
        D03StageManifest.from_dict(payload)

    payload = _stage_payload("W-04")
    payload["data_visibility"]["evaluator_owner"] = "PH2_TRAIN_CANDIDATE"
    with pytest.raises(D03ContractError, match="owner"):
        D03StageManifest.from_dict(payload)

    payload = _stage_payload("W-04")
    payload["data_visibility"]["candidate_allowed_splits"] = ["train", "held_out"]
    with pytest.raises(D03ContractError, match="split"):
        D03StageManifest.from_dict(payload)


def test_threshold_budget_and_recovery_are_not_optional_or_backfilled():
    """阈值必须预注册逐维承重，预算和 cursor/shard/barrier/failure 必须完整。"""
    payload = _stage_payload("W-05")
    threshold = payload["evaluation_binding"]["thresholds"][0]
    threshold["preregistered"] = 0
    with pytest.raises(D03ContractError, match="预注册"):
        D03StageManifest.from_dict(payload)

    payload = _stage_payload("W-05")
    payload["evaluation_binding"]["aggregation_policy"] = "MEAN_CAN_HIDE_FAILURE"
    with pytest.raises(D03ContractError, match="承重"):
        D03StageManifest.from_dict(payload)

    for section, key in (
        ("resource_budget", "max_payload_bytes"),
        ("recovery_binding", "cursor_version"),
        ("recovery_binding", "merge_barrier_key"),
        ("recovery_binding", "failure_point_keys"),
    ):
        payload = _stage_payload("W-05")
        del payload[section][key]
        with pytest.raises(D03ContractError, match="预算|cursor|barrier|failure|字段"):
            D03StageManifest.from_dict(payload)


def test_invalidation_graph_requires_exact_suffix_and_acyclic_chain():
    """每条变化必须返回最早阶段及完整后缀，阶段依赖不得成环。"""
    graph = StageInvalidationGraph.from_dict(_graph_payload())
    result = graph.invalidate("PACK_CONTENT", "PACK-06")
    assert result.earliest_stage == "W-06"
    assert result.invalidated_stage_keys == STAGE_KEYS[5:]

    broken = _graph_payload()
    broken["rules"][1]["invalidated_stage_keys"] = ["W-06", "W-08", "W-09"]
    with pytest.raises(D03ContractError, match="完整后缀"):
        StageInvalidationGraph.from_dict(broken)

    broken = _graph_payload()
    broken["stage_edges"].append({
        "consumer_stage": "W-01", "prerequisite_stage": "W-09"
    })
    with pytest.raises(D03ContractError, match="环|顺序"):
        StageInvalidationGraph.from_dict(broken)


def test_publication_state_separates_candidate_git_and_post_publish():
    """只有 post-publish verified 状态可声明 d03_published=1。"""
    candidate = D03PublicationState("CANDIDATE_VERIFIED", 0, "", 0)
    assert candidate.d03_published == 0
    with pytest.raises(D03ContractError, match="post-publish"):
        D03PublicationState("GIT_PUBLISHED", 1, "d" * 40, 0)
    published = D03PublicationState(
        "POST_PUBLISH_VERIFIED", 1, "d" * 40, 1
    )
    assert published.d03_published == 1


def test_immutable_writer_is_idempotent_but_never_overwrites(tmp_path: Path):
    """同版本同字节可幂等回读，同路径不同字节必须拒绝。"""
    target = tmp_path / "manifest.json"
    first = {"artifact_version": "v1", "value": 1}
    write_immutable_json(first, target)
    write_immutable_json(first, target)
    with pytest.raises(D03ContractError, match="不可覆盖"):
        write_immutable_json({"artifact_version": "v1", "value": 2}, target)
