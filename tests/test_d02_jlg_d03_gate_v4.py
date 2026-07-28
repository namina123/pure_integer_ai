"""J-LG-D03 v4 补充吸收、文件图、扫描与停止边界 T0。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_capability_baseline_v41_catalog import (
    build_capability_baseline_v41,
)
from pure_integer_ai.experiments.ph2_capability_baseline_v41_contract import (
    MANIFEST_PATH as BASELINE_V41_PATH,
    VERSION_KEYS,
    BaselineV41EvidenceFile,
    CapabilityBaselineV41Error,
    read_capability_baseline_v41,
    verify_capability_baseline_v41_files,
    write_capability_baseline_v41,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_jlg_d03_gate_contract import (
    read_jlg_d03_gate_manifest,
)
from pure_integer_ai.experiments.ph2_jlg_d03_gate_v4_catalog import (
    build_jlg_d03_gate_v4,
)
from pure_integer_ai.experiments.ph2_jlg_d03_gate_v4_contract import (
    ARTIFACT_PATH,
    CONDITION_KEYS,
    EXECUTION_STATE_KEYS,
    REQUIRED_EDGE_PAIRS,
    REQUIRED_NODE_SPECS,
    V3_PATH,
    V3_SHA256,
    JLGD03GateV4Error,
    read_jlg_d03_gate_v4,
    verify_jlg_d03_gate_v4_files,
    write_jlg_d03_gate_v4,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    inventory_public_files,
    scan_public_patterns,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
FORMAL_V4_PATH = REPOSITORY / ARTIFACT_PATH
PUBLICATION_RECEIPT_PATH = (
    REPOSITORY
    / "data/ph2/manifests/j_lg_d03_gate_v4_git_publication_v1.json"
)
PUBLICATION_RECEIPT_RELATIVE_PATH = PUBLICATION_RECEIPT_PATH.relative_to(
    REPOSITORY).as_posix()
PUBLICATION_OVERRIDE_PATHS = {
    "src/pure_integer_ai/experiments/ph2_public_gate_rules.py",
    "tests/test_d02_jlg_d03_gate.py",
    "tests/test_d02_jlg_d03_gate_v4.py",
    "tests/test_d02_language_baseline.py",
    "tests/test_d02_public_gate_rules.py",
}


def _git(*args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def _paths(*args: str) -> tuple[str, ...]:
    payload = _git(*args, "-z")
    return tuple(sorted(
        item.decode("utf-8") for item in payload.split(b"\0") if item))


def _modified_paths() -> tuple[str, ...]:
    return _paths("diff", "--name-only", "--diff-filter=ACMRTUXB")


def _staged_paths() -> tuple[str, ...]:
    return _paths(
        "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB")


def _untracked_paths() -> tuple[str, ...]:
    return _paths("ls-files", "--others", "--exclude-standard")


def _publication_receipt() -> dict[str, object]:
    """严格回读 v4 首次进入 Git tree 的规范回执。"""
    payload = PUBLICATION_RECEIPT_PATH.read_bytes()
    value = json.loads(payload)
    expected = (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    assert payload == expected
    assert set(value) == {
        "artifact_sha256",
        "candidate_file_count",
        "current_path_overrides",
        "d03_published",
        "format_version",
        "gate_artifact_path",
        "gate_base_head_sha1",
        "publication_manifest_self_excluded",
        "release_file_count",
        "status",
    }
    return value


def _historical_identity_matches(
        relative_path: str,
        expected_size: int,
        expected_sha256: str) -> bool:
    """确认发布回执指定的精确字节仍存在于可达 Git 历史。"""
    commits = _git(
        "log", "--all", "--format=%H", "--", relative_path,
    ).decode("ascii").splitlines()
    for commit in commits:
        try:
            payload = _git("show", f"{commit}:{relative_path}")
        except subprocess.CalledProcessError:
            continue
        if (len(payload) == expected_size
                and hashlib.sha256(payload).hexdigest() == expected_sha256):
            return True
    return False


def _candidate_workspace_matches_gate(manifest) -> bool:
    """只在 v4 所冻结的提交前工作树中重建 candidate gate。"""
    return (
        _git("rev-parse", "HEAD").decode("ascii").strip()
        == manifest.head_sha1
        and _git("rev-parse", "origin/master").decode("ascii").strip()
        == manifest.origin_master_sha1
        and len(_modified_paths()) == manifest.tracked_change_count
        and len(_staged_paths()) == manifest.staged_change_count
        and len(_untracked_paths()) + 1 == manifest.untracked_file_count
    )


def _verify_published_v4_snapshot(manifest) -> None:
    """在干净 checkout 中按发布回执复验 v4 候选的当前字节。"""
    receipt = _publication_receipt()
    assert receipt["artifact_sha256"] == manifest.sha256()
    assert receipt["candidate_file_count"] == manifest.candidate_file_count
    assert receipt["d03_published"] == 0
    assert receipt["format_version"] == 1
    assert receipt["gate_artifact_path"] == ARTIFACT_PATH
    assert receipt["gate_base_head_sha1"] == manifest.head_sha1
    assert receipt["publication_manifest_self_excluded"] == 1
    assert receipt["release_file_count"] == manifest.candidate_file_count + 1
    assert receipt["status"] == "GIT_SNAPSHOT_PUBLISHED_D03_HELD"

    overrides = {
        str(item["relative_path"]): item
        for item in receipt["current_path_overrides"]
    }
    assert set(overrides) == PUBLICATION_OVERRIDE_PATHS
    tracked = set(_paths("ls-files"))
    assert {
        *manifest.candidate_paths,
        ARTIFACT_PATH,
        PUBLICATION_RECEIPT_RELATIVE_PATH,
    } <= tracked
    for item in manifest.file_inventory:
        payload = (REPOSITORY / item.relative_path).read_bytes()
        expected = overrides.get(item.relative_path)
        expected_size = (
            int(expected["size_bytes"]) if expected else item.size_bytes)
        expected_sha256 = (
            str(expected["sha256"]) if expected else item.sha256)
        if (len(payload) == expected_size
                and hashlib.sha256(payload).hexdigest() == expected_sha256):
            continue
        assert _historical_identity_matches(
            item.relative_path, expected_size, expected_sha256)
    for item in manifest.paper_files:
        payload = (REPOSITORY / item.relative_path).read_bytes()
        assert len(payload) == item.size_bytes
        assert hashlib.sha256(payload).hexdigest() == item.sha256
    artifact = FORMAL_V4_PATH.read_bytes()
    assert hashlib.sha256(artifact).hexdigest() == receipt["artifact_sha256"]


@pytest.fixture(scope="module")
def formal_v41():
    """按当前 R-01..R-06 文件重建一次 v41。"""
    return build_capability_baseline_v41(REPOSITORY, WORKSPACE)


@pytest.fixture(scope="module")
def formal_v4():
    """提交前重建 candidate；提交后回读并验证发布快照。"""
    stored = read_jlg_d03_gate_v4(FORMAL_V4_PATH)
    if not _candidate_workspace_matches_gate(stored):
        _verify_published_v4_snapshot(stored)
        return stored
    return build_jlg_d03_gate_v4(
        REPOSITORY,
        WORKSPACE,
        head_sha1=_git("rev-parse", "HEAD").decode("ascii").strip(),
        origin_master_sha1=(
            _git("rev-parse", "origin/master").decode("ascii").strip()),
        modified_relative_paths=_modified_paths(),
        staged_change_count=len(_staged_paths()),
        untracked_relative_paths=_untracked_paths(),
    )


def test_v41_round_trip_versions_file_identity_and_zero_execution(
        tmp_path, formal_v41):
    """v41 必须绑定六类版本、45 个文件身份和全零执行。"""
    target = tmp_path / "v41.json"
    assert write_capability_baseline_v41(formal_v41, target) == target
    assert read_capability_baseline_v41(target) == formal_v41
    verify_capability_baseline_v41_files(
        formal_v41, repository_root=REPOSITORY)
    assert formal_v41.version_keys.to_value() == VERSION_KEYS
    assert len(formal_v41.evidence_files) == 45
    assert all(value == 0
               for value in formal_v41.execution_state.to_value().values())


def test_v41_rejects_version_and_file_identity_drift(formal_v41):
    """伪版本键或任一 upstream 尺寸漂移不得构成新基线。"""
    versions = formal_v41.version_keys.to_value()
    versions["backend_version"] = "DRIFT"
    with pytest.raises(CapabilityBaselineV41Error, match="version keys 漂移"):
        replace(
            formal_v41,
            version_keys=CanonicalJsonObject.from_value(versions),
        )
    first = formal_v41.evidence_files[0]
    drifted = BaselineV41EvidenceFile(
        first.relative_path, first.role, first.byte_count + 1, first.sha256)
    with pytest.raises(CapabilityBaselineV41Error, match="文件身份漂移"):
        verify_capability_baseline_v41_files(
            replace(
                formal_v41,
                evidence_files=(drifted, *formal_v41.evidence_files[1:]),
            ),
            repository_root=REPOSITORY,
        )


def test_sixteen_v4_rows_are_direct_and_all_pass(formal_v4):
    """R-01..R-06 与补充门逐行 PASS，不允许旧 aggregate PASS 代签。"""
    assert tuple(item.condition_key for item in formal_v4.conditions) == (
        CONDITION_KEYS)
    assert {item.verdict for item in formal_v4.conditions} == {"PASS"}
    assert all(item.evidence_refs for item in formal_v4.conditions)
    assert formal_v4.artifact_status == "PASS"
    assert formal_v4.conjunction_passed == 1
    assert formal_v4.d03_release_decision == (
        "ALLOW_FUTURE_CONFIRMED_SESSION_TO_PUBLISH_D03")


def test_candidate_inventory_includes_tracked_and_untracked_but_not_self(
        formal_v4):
    """当前 tracked 修改与全部 untracked 文件逐字节入账，只有 v4 自排除。"""
    if not _candidate_workspace_matches_gate(formal_v4):
        _verify_published_v4_snapshot(formal_v4)
        return
    modified = set(_modified_paths())
    untracked = {*_untracked_paths(), ARTIFACT_PATH}
    expected = tuple(sorted((modified | untracked) - {ARTIFACT_PATH}))
    assert formal_v4.candidate_paths == expected
    assert set(expected) <= {
        item.relative_path for item in formal_v4.file_inventory}
    assert formal_v4.tracked_change_count == len(modified)
    assert formal_v4.untracked_file_count == len(untracked)
    assert formal_v4.candidate_file_count == len(modified | untracked)
    assert formal_v4.inventory_exclusions == (ARTIFACT_PATH,)


def test_dependency_graph_and_all_file_identities_reverify(formal_v4):
    """23 个 node 与完整 edge 集都绑定 candidate inventory 的同一文件身份。"""
    assert len(formal_v4.dependency_nodes) == len(REQUIRED_NODE_SPECS)
    assert tuple((item.consumer_key, item.dependency_key)
                 for item in formal_v4.dependency_edges) == REQUIRED_EDGE_PAIRS
    if _candidate_workspace_matches_gate(formal_v4):
        verify_jlg_d03_gate_v4_files(
            formal_v4, repository_root=REPOSITORY)
    else:
        _verify_published_v4_snapshot(formal_v4)
    assert {item.relative_path for item in formal_v4.dependency_nodes} <= {
        item.relative_path for item in formal_v4.file_inventory}


def test_p3ib_public_scan_zero_execution_and_hold_are_explicit(formal_v4):
    """P3-Ib、公开扫描和零执行各自承重，v4 自身不发布 D-03。"""
    rows = {
        item.condition_key: item.facts.to_value()
        for item in formal_v4.conditions}
    assert rows[CONDITION_KEYS[9]] == {
        "code_switch_status": "NE",
        "cross_language_pass_authority": 0,
        "p3ib_phase": "PH3",
        "p3ib_status": "NE",
    }
    public = formal_v4.final_public_gate
    assert public.legacy_finding_count == 0
    assert public.secret_finding_count == 0
    assert public.binary_paths == ()
    assert public.unreadable_paths == ()
    assert public.artifact_self_excluded == 1
    assert public.post_publish_self_scan_required == 1
    assert set(formal_v4.execution_state.to_value()) == set(
        EXECUTION_STATE_KEYS)
    assert all(value == 0
               for value in formal_v4.execution_state.to_value().values())
    assert formal_v4.d03_published == 0


def test_source_license_paper_and_v3_are_preserved(formal_v4):
    """六许可 pack、一 blocker、paper 与 v3 旧 artifact 身份不变。"""
    rows = {
        item.condition_key: item.facts.to_value()
        for item in formal_v4.conditions}
    assert rows[CONDITION_KEYS[11]] == {
        "blocked_source_count": 1,
        "frozen_license_pack_count": 6,
        "source_entry_count": 7,
    }
    assert rows[CONDITION_KEYS[14]]["v3_sha256"] == V3_SHA256
    v3 = read_jlg_d03_gate_manifest(REPOSITORY / V3_PATH)
    assert v3.artifact_status == "PASS"
    assert v3.d03_published == 0


def test_contract_rejects_missing_r06_graph_and_nonzero_teacher(formal_v4):
    """删 R-06 edge/node 或伪造 teacher 调用都必须失败关闭。"""
    r06_edge = next(
        item for item in formal_v4.dependency_edges
        if item.consumer_key == "GATE_V4" and item.dependency_key == "R06")
    with pytest.raises(JLGD03GateV4Error, match="edge 集合不完整"):
        replace(
            formal_v4,
            dependency_edges=tuple(
                item for item in formal_v4.dependency_edges
                if item != r06_edge),
        )
    execution = formal_v4.execution_state.to_value()
    execution["teacher_calls"] = 1
    with pytest.raises(JLGD03GateV4Error, match="execution state"):
        replace(
            formal_v4,
            execution_state=CanonicalJsonObject.from_value(execution),
        )


def test_contract_rejects_condition_and_inventory_loss(formal_v4):
    """删 R-01 condition 或任一 candidate identity 都不得合取。"""
    with pytest.raises(JLGD03GateV4Error, match="conditions 不完整"):
        replace(formal_v4, conditions=formal_v4.conditions[1:])
    with pytest.raises(JLGD03GateV4Error, match="inventory"):
        replace(formal_v4, file_inventory=formal_v4.file_inventory[:-1])


def test_v41_and_v4_are_idempotent_but_non_overwritable(
        tmp_path, formal_v41, formal_v4):
    """两份新 artifact 同内容幂等、同版本异内容不可覆盖。"""
    for manifest, writer, error_type, name in (
            (formal_v41, write_capability_baseline_v41,
             CapabilityBaselineV41Error, "v41.json"),
            (formal_v4, write_jlg_d03_gate_v4,
             JLGD03GateV4Error, "v4.json")):
        target = tmp_path / name
        writer(manifest, target)
        assert writer(manifest, target) == target
        target.write_bytes(b"{}\n")
        with pytest.raises(error_type, match="已存在且内容不同"):
            writer(manifest, target)


def test_stored_v41_and_v4_match_current_builders(formal_v41, formal_v4):
    """仓内 v41 保持当前；v4 保持 candidate 或发布 tree 身份。"""
    stored_v41 = read_capability_baseline_v41(
        REPOSITORY / BASELINE_V41_PATH)
    stored_v4 = read_jlg_d03_gate_v4(FORMAL_V4_PATH)
    assert stored_v41 == formal_v41
    assert stored_v4 == formal_v4
    assert stored_v41.canonical_bytes() == formal_v41.canonical_bytes()
    assert stored_v4.canonical_bytes() == formal_v4.canonical_bytes()
    if not _candidate_workspace_matches_gate(stored_v4):
        _verify_published_v4_snapshot(stored_v4)


def test_post_publish_v4_self_scan_is_clean(formal_v4):
    """自排除只解开 canonical 循环，发布后的 v4 文件仍须单独重扫。"""
    inventory = inventory_public_files(
        REPOSITORY,
        (ARTIFACT_PATH, PUBLICATION_RECEIPT_RELATIVE_PATH),
    )
    for rules in (LEGACY_RULES, SECRET_RULES):
        findings, binary, unreadable = scan_public_patterns(
            REPOSITORY, inventory, rules)
        assert findings == ()
        assert binary == ()
        assert unreadable == ()
