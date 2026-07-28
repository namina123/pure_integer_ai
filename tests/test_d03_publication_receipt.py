"""D-03 两步 Git 发布的 post-publication receipt 合同反例。"""
from __future__ import annotations

import copy
import gzip
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_publication import (
    D03PostPublishGate,
    D03PublicationInventoryScan,
    D03PublicationReceipt,
    GitHubCIJob,
    read_d03_publication_receipt,
    scan_d03_publication_inventory,
    write_d03_publication_receipt,
)
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_line
from pure_integer_ai.experiments.ph2_d03_release_catalog import FORMAL_RECEIPT_PATH
from pure_integer_ai.experiments.ph2_d03_release_contract import (
    D03ContractError,
    D03FileIdentity,
    D03PublicationState,
    ZERO_EXECUTION_STATE,
)


def _file(path: str, marker: str) -> D03FileIdentity:
    """返回测试使用的稳定文件身份。"""
    return D03FileIdentity(path, 1, marker * 64)


def _actual_file(root: Path, relative: str, payload: bytes) -> D03FileIdentity:
    """写测试文件并返回真实 identity。"""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return D03FileIdentity(
        relative, len(payload), hashlib.sha256(payload).hexdigest()
    )


def _receipt() -> D03PublicationReceipt:
    """构造一个满足两步发布语义的最小正式收据。"""
    content_commit = "c" * 40
    global_identity = _file(
        "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json", "a"
    )
    inventory = (
        global_identity,
        _file("src/pure_integer_ai/experiments/ph2_d03_release_contract.py", "b"),
    )
    gate = D03PostPublishGate(
        legacy_finding_count=0,
        secret_finding_count=0,
        binary_path_count=0,
        unreadable_path_count=0,
        github_secret_alert_count=0,
        license_gate_passed=1,
        paper_gate_passed=1,
        zero_execution_gate_passed=1,
        remote_reachable=1,
        isolated_clone_verified=1,
        content_ci_run_id=123,
        content_ci_head_sha1=content_commit,
        content_ci_jobs=(
            GitHubCIJob("Python 3.11 on ubuntu-latest", "success"),
            GitHubCIJob("Python 3.14 on ubuntu-latest", "success"),
            GitHubCIJob("Python 3.14 on windows-latest", "success"),
            GitHubCIJob("Secret scan", "success"),
        ),
    )
    state = dict(ZERO_EXECUTION_STATE)
    state["d03_published"] = 1
    return D03PublicationReceipt(
        format_version=1,
        artifact_kind="PH2_D03_POST_PUBLICATION_RECEIPT",
        artifact_version="PH2-D03-post-publication-receipt-v1",
        release_key="PH2-D03-V1",
        status="POST_PUBLISH_VERIFIED",
        publication_state=D03PublicationState(
            "POST_PUBLISH_VERIFIED", 1, content_commit, 1
        ),
        remote_name="origin",
        remote_branch="master",
        remote_ref="refs/heads/master",
        content_parent_sha1="d" * 40,
        content_commit_sha1=content_commit,
        global_manifest_identity=global_identity,
        release_inventory=inventory,
        historical_hold_receipt_identity=_file(
            "data/ph2/manifests/j_lg_d03_gate_v4_git_publication_v1.json", "e"
        ),
        post_publish_gate=gate,
        receipt_relative_path=FORMAL_RECEIPT_PATH,
        receipt_self_excluded=1,
        execution_state=state,
    )


def test_receipt_round_trip_and_immutable_writer(tmp_path: Path):
    """正式 receipt 可规范回读、同字节幂等且异字节不可覆盖。"""
    receipt = _receipt()
    target = tmp_path / "receipt.json"
    write_d03_publication_receipt(receipt, target)
    write_d03_publication_receipt(receipt, target)
    assert read_d03_publication_receipt(target) == receipt
    broken = copy.deepcopy(receipt.to_dict())
    broken["content_parent_sha1"] = "f" * 40
    target.write_bytes(receipt.canonical_bytes())
    with pytest.raises(D03ContractError, match="不可覆盖"):
        write_d03_publication_receipt(
            D03PublicationReceipt.from_dict(broken), target
        )


def test_receipt_requires_content_commit_global_inventory_and_self_exclusion():
    """content SHA、global inventory 与 receipt self-exclusion 必须闭合。"""
    payload = _receipt().to_dict()
    payload["content_commit_sha1"] = "f" * 40
    with pytest.raises(D03ContractError, match="content commit"):
        D03PublicationReceipt.from_dict(payload)

    payload = _receipt().to_dict()
    payload["release_inventory"] = payload["release_inventory"][1:]
    with pytest.raises(D03ContractError, match="global"):
        D03PublicationReceipt.from_dict(payload)

    payload = _receipt().to_dict()
    payload["release_inventory"].append(
        _file(FORMAL_RECEIPT_PATH, "f").to_dict()
    )
    with pytest.raises(D03ContractError, match="self-excluded"):
        D03PublicationReceipt.from_dict(payload)


def test_receipt_rejects_scan_ci_or_zero_execution_failure():
    """任何公开/secret/CI/论文/许可/零执行门失败都不能发布 D-03。"""
    fields = (
        "legacy_finding_count",
        "secret_finding_count",
        "binary_path_count",
        "unreadable_path_count",
        "github_secret_alert_count",
    )
    for field in fields:
        payload = _receipt().to_dict()
        payload["post_publish_gate"][field] = 1
        with pytest.raises(D03ContractError, match="gate|scan"):
            D03PublicationReceipt.from_dict(payload)

    payload = _receipt().to_dict()
    payload["post_publish_gate"]["content_ci_jobs"][-1]["conclusion"] = "failure"
    with pytest.raises(D03ContractError, match="CI"):
        D03PublicationReceipt.from_dict(payload)

    payload = _receipt().to_dict()
    payload["execution_state"]["teacher_calls"] = 1
    with pytest.raises(D03ContractError, match="execution_state"):
        D03PublicationReceipt.from_dict(payload)


def test_publication_scan_decodes_canonical_gzip_and_scans_case_insensitively(
        tmp_path: Path):
    """规范 gzip 进入文本扫描，退役名称的大小写变体不能逃逸。"""
    clean = gzip.compress(canonical_json_line({"value": "clean"}), mtime=0)
    clean_identity = _actual_file(tmp_path, "pack/clean.jsonl.gz", clean)
    result = scan_d03_publication_inventory(tmp_path, (clean_identity,))
    assert result == D03PublicationInventoryScan(1, 1, 0, 0, 0, 0)

    retired = gzip.compress(
        canonical_json_line({"value": "URN:" + "ZERO" + "-AI:retired"}),
        mtime=0,
    )
    retired_identity = _actual_file(
        tmp_path, "pack/retired.jsonl.gz", retired)
    result = scan_d03_publication_inventory(tmp_path, (retired_identity,))
    assert result.gzip_jsonl_path_count == 1
    assert result.legacy_finding_count == 1
    assert result.binary_path_count == 0
    assert result.unreadable_path_count == 0


def test_publication_scan_rejects_identity_and_classifies_unscannable_files(
        tmp_path: Path):
    """identity 漂移硬拒绝；非规范 gzip 与真二进制分别记账。"""
    malformed = gzip.compress(b'{"value": 1}\n', mtime=0)
    malformed_identity = _actual_file(
        tmp_path, "pack/malformed.jsonl.gz", malformed)
    binary_identity = _actual_file(tmp_path, "pack/value.bin", b"\xff")
    result = scan_d03_publication_inventory(
        tmp_path, (malformed_identity, binary_identity))
    assert result == D03PublicationInventoryScan(2, 1, 0, 0, 1, 1)

    (tmp_path / "pack/value.bin").write_bytes(b"changed")
    with pytest.raises(D03ContractError, match="身份漂移"):
        scan_d03_publication_inventory(tmp_path, (binary_identity,))
