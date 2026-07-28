"""Final J-LG-D03 conjunction, evidence identity, and stop-boundary T0."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_jlg_d03_gate_catalog import (
    build_jlg_d03_gate_manifest,
)
from pure_integer_ai.experiments.ph2_jlg_d03_gate_contract import (
    ARTIFACT_PATH,
    EXECUTION_STATE_KEYS,
    MAIN_CONDITION_KEYS,
    SUPPLEMENTAL_CHECK_KEYS,
    GateCondition,
    JLGD03GateContractError,
    read_jlg_d03_gate_manifest,
    verify_jlg_d03_gate_files,
    write_jlg_d03_gate_manifest,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    inventory_public_files,
    scan_public_patterns,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_scan_report,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
FORMAL_PATH = REPOSITORY / ARTIFACT_PATH
PUBLICATION_RECEIPT_PATH = (
    REPOSITORY
    / "data/ph2/manifests/j_lg_d03_gate_v4_git_publication_v1.json"
)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def _untracked_paths() -> tuple[str, ...]:
    payload = _git("ls-files", "--others", "--exclude-standard", "-z")
    return tuple(sorted(
        item.decode("utf-8") for item in payload.split(b"\0") if item))


def _candidate_workspace_matches_gate(manifest) -> bool:
    """仅把 v3 冻结时的工作树视为可重建 candidate。"""
    return (
        _git("rev-parse", "HEAD").decode("ascii").strip()
        == manifest.head_sha1
        and _git("rev-parse", "origin/master").decode("ascii").strip()
        == manifest.origin_master_sha1
        and subprocess.run(
            ("git", "diff", "--quiet"), cwd=REPOSITORY).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"),
            cwd=REPOSITORY,
        ).returncode == 0
        and len(_untracked_paths()) == manifest.untracked_file_count
    )


def _identity_matches(payload: bytes, item) -> bool:
    """按冻结尺寸和摘要比较一份候选字节。"""
    return (
        len(payload) == item.size_bytes
        and hashlib.sha256(payload).hexdigest() == item.sha256
    )


def _historical_identity_matches(
        relative_path: str,
        expected_identities: tuple[tuple[int, str], ...]) -> bool:
    """确认冻结或发布回执字节仍存在于可达 Git 历史。"""
    commits = _git(
        "log", "--all", "--format=%H", "--", relative_path,
    ).decode("ascii").splitlines()
    for commit in commits:
        try:
            payload = _git("show", f"{commit}:{relative_path}")
        except subprocess.CalledProcessError:
            continue
        identity = (len(payload), hashlib.sha256(payload).hexdigest())
        if identity in expected_identities:
            return True
    return False


def _verify_historical_gate_files(manifest) -> None:
    """用当前文件或 v3 基线 blob 复验不再存在的候选工作树。"""
    receipt = json.loads(PUBLICATION_RECEIPT_PATH.read_bytes())
    overrides = {
        str(item["relative_path"]): item
        for item in receipt["current_path_overrides"]
    }
    for item in manifest.file_inventory:
        expected_identities = [(item.size_bytes, item.sha256)]
        override = overrides.get(item.relative_path)
        if override is not None:
            expected_identities.append((
                int(override["size_bytes"]), str(override["sha256"])))
        current_path = REPOSITORY / item.relative_path
        if current_path.is_file():
            current = current_path.read_bytes()
            identity = (len(current), hashlib.sha256(current).hexdigest())
            if identity in expected_identities:
                continue
        assert _historical_identity_matches(
            item.relative_path, tuple(expected_identities))
    for item in manifest.paper_files:
        assert _identity_matches(
            (REPOSITORY / item.relative_path).read_bytes(), item)
    for item in manifest.external_evidence:
        assert _identity_matches(
            (WORKSPACE / item.relative_path).read_bytes(), item)
    assert FORMAL_PATH.is_file()


@pytest.fixture(scope="module")
def formal_manifest():
    stored = read_jlg_d03_gate_manifest(FORMAL_PATH)
    if not _candidate_workspace_matches_gate(stored):
        _verify_historical_gate_files(stored)
        return stored
    return build_jlg_d03_gate_manifest(
        REPOSITORY,
        WORKSPACE,
        head_sha1=_git("rev-parse", "HEAD").decode("ascii").strip(),
        origin_master_sha1=(
            _git("rev-parse", "origin/master").decode("ascii").strip()),
        untracked_relative_paths=_untracked_paths(),
    )


def _row(manifest, key: str):
    return next(item for item in (*manifest.conditions,
                                  *manifest.supplemental_checks)
                if item.condition_key == key)


def test_twelve_conjunction_rows_are_direct_and_all_pass(formal_manifest):
    """The original 12 rows stay explicit; no aggregate score hides a gap."""
    assert tuple(item.condition_key
                 for item in formal_manifest.conditions) == MAIN_CONDITION_KEYS
    assert tuple(item.condition_key
                 for item in formal_manifest.supplemental_checks) == (
                     SUPPLEMENTAL_CHECK_KEYS)
    assert {item.verdict for item in formal_manifest.conditions} == {"PASS"}
    assert {item.verdict for item
            in formal_manifest.supplemental_checks} == {"PASS"}
    assert all(item.evidence_refs for item in formal_manifest.conditions)
    assert formal_manifest.conjunction_passed == 1
    assert formal_manifest.artifact_status == "PASS"
    assert formal_manifest.d03_release_decision == (
        "ALLOW_NEXT_SESSION_TO_PUBLISH_D03")


def test_source_exit_keeps_cc_cedict_blocked(formal_manifest):
    """A terminal source ledger PASS must not rewrite the blocked license row."""
    source = _row(formal_manifest, "J-LG-D03-01-SOURCE-EXIT").facts.to_value()
    assert source["frozen_license_pack_count"] == 6
    assert source["source_entry_count"] == 7
    assert source["blocked_sources"] == [{
        "blocker_code": "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
        "source_key": "CC_CEDICT_20260725",
        "verdict": "BLOCKED",
    }]
    cc = _row(
        formal_manifest,
        "SUP-CC-CEDICT-HISTORICAL-BLOCKER-PRESERVED",
    ).facts.to_value()
    assert cc == {
        "current_source_verdict": "BLOCKED",
        "historical_blocker_code": "LICENSE_PARTITION_MISMATCH",
        "historical_source_verdict": "BLOCKED",
        "public_source_pack_emitted": 0,
    }


def test_ri_and_nl_preserve_reject_and_ne_details(formal_manifest):
    """Scope completion is PASS while the bounded sub-verdicts remain honest."""
    ri = _row(
        formal_manifest, "SUP-RI-00-SCOPE-DECIDED").facts.to_value()
    assert ri["mode_verdicts"] == {
        "ABDUCTION": "REJECT",
        "COUNTERFACTUAL": "REJECT",
        "DEFEASIBLE_DEFAULT": "REJECT",
        "DEONTIC_NORMATIVE": "REJECT",
        "TEMPORAL": "PASS",
    }
    assert (ri["pass_count"], ri["reject_count"], ri["ne_count"]) == (1, 4, 0)
    nl = _row(
        formal_manifest, "SUP-NL-00-SCOPE-DECIDED").facts.to_value()
    assert set(nl["layer_verdicts"].values()) == {"PASS", "REJECT", "NE"}
    assert (nl["pass_count"], nl["reject_count"], nl["ne_count"]) == (1, 3, 1)


def test_wiktionary_reports_are_equal_without_rescan(formal_manifest):
    """Only the two saved 48 KiB reports are read; the dump is not scanned."""
    row = _row(
        formal_manifest, "SUP-WIKTIONARY-DOUBLE-PASS").facts.to_value()
    assert row == {
        "anomaly_codes": {"UNBALANCED_TEMPLATE": 363},
        "full_eof_verified": 1,
        "main_namespace_count": 2674506,
        "page_count": 3191659,
        "report_sha256": (
            "6d120c78438733497392a21e4ce6844aa9a982a63eb10330bd3e8ee96dbee385"),
        "valid_page_count": 2674143,
    }
    paths = tuple(
        WORKSPACE / Path(*item.relative_path.split("/"))
        for item in formal_manifest.external_evidence
        if item.scope == "RAW_EVIDENCE")
    assert len(paths) == 2
    reports = tuple(read_mediawiki_scan_report(path) for path in paths)
    assert reports[0] == reports[1]


def test_all_formal_packs_have_four_physical_owners(formal_manifest):
    """Source, observation, teacher, and evaluator files stay separated."""
    row = _row(
        formal_manifest, "J-LG-D03-04-PACK-AUDITABILITY").facts.to_value()
    assert row["formal_pack_count"] == 16
    assert row["split_probe_count"] == 3
    assert row["owner_kinds"] == [
        "evaluator", "observation", "source", "teacher"]
    assert all(item["owner_kinds"] == row["owner_kinds"]
               for item in row["pack_facts"])
    assert len({item["manifest_path"] for item in row["pack_facts"]}) == 16


def test_final_inventory_and_public_scan_are_closed(formal_manifest):
    """All current untracked files are inventoried except the self-reference."""
    if not _candidate_workspace_matches_gate(formal_manifest):
        assert len(formal_manifest.file_inventory) + 1 == (
            formal_manifest.untracked_file_count)
        _verify_historical_gate_files(formal_manifest)
        return
    current = _untracked_paths()
    expected = tuple(item for item in current if item != ARTIFACT_PATH)
    assert tuple(item.relative_path
                 for item in formal_manifest.file_inventory) == expected
    assert formal_manifest.untracked_file_count == len(current)
    gate = formal_manifest.final_public_gate
    assert gate.scope_file_count == len(expected)
    assert gate.legacy_finding_count == 0
    assert gate.secret_finding_count == 0
    assert gate.binary_paths == ()
    assert gate.unreadable_paths == ()
    assert gate.public_candidate_clear == 1


def test_published_artifact_self_scan_is_clean(formal_manifest):
    """The self-excluded canonical JSON is scanned after publication."""
    inventory = inventory_public_files(REPOSITORY, (ARTIFACT_PATH,))
    for rules in (LEGACY_RULES, SECRET_RULES):
        findings, binary, unreadable = scan_public_patterns(
            REPOSITORY, inventory, rules)
        assert findings == ()
        assert binary == ()
        assert unreadable == ()


def test_zero_execution_and_stop_boundary(formal_manifest):
    """The gate authorizes only a later session; this session publishes nothing."""
    assert set(formal_manifest.execution_state.to_value()) == set(
        EXECUTION_STATE_KEYS)
    assert all(value == 0
               for value in formal_manifest.execution_state.to_value().values())
    assert formal_manifest.d03_published == 0
    zero = _row(
        formal_manifest,
        "J-LG-D03-07-ZERO-FORMAL-EXECUTION",
    ).facts.to_value()
    assert all(value == 0 for value in zero.values())


def test_all_file_identities_reverify(formal_manifest):
    """Public files, paper, raw reports, and pack manifests all rehash exactly."""
    if _candidate_workspace_matches_gate(formal_manifest):
        verify_jlg_d03_gate_files(
            formal_manifest,
            repository_root=REPOSITORY,
            workspace_root=WORKSPACE,
        )
    else:
        _verify_historical_gate_files(formal_manifest)
    assert len(formal_manifest.external_evidence) == 18


def test_contract_rejects_aggregate_pass_missing_evidence_or_paper_drift(
        formal_manifest):
    """A hidden failure, unknown evidence path, or changed paper fails closed."""
    first = formal_manifest.conditions[0]
    with pytest.raises(JLGD03GateContractError, match="conjunction"):
        replace(
            formal_manifest,
            conditions=(replace(first, verdict="REJECT"),
                        *formal_manifest.conditions[1:]),
        )
    with pytest.raises(JLGD03GateContractError, match="without a file identity"):
        replace(
            formal_manifest,
            conditions=(replace(
                first,
                evidence_refs=tuple(sorted((
                    *first.evidence_refs, "data/ph2/missing.json"))),
            ), *formal_manifest.conditions[1:]),
        )
    with pytest.raises(JLGD03GateContractError, match="paper byte identity"):
        replace(
            formal_manifest,
            paper_files=(replace(
                formal_manifest.paper_files[0], sha256="1" * 64),
                         formal_manifest.paper_files[1]),
        )


def test_contract_rejects_nonzero_execution_and_incomplete_inventory(
        formal_manifest):
    """Training claims and a missing final candidate cannot be signed off."""
    execution = formal_manifest.execution_state.to_value()
    execution["teacher_calls"] = 1
    with pytest.raises(JLGD03GateContractError, match="forbidden execution"):
        replace(
            formal_manifest,
            execution_state=CanonicalJsonObject.from_value(execution),
        )
    with pytest.raises(JLGD03GateContractError, match="inventory is incomplete"):
        replace(
            formal_manifest,
            file_inventory=formal_manifest.file_inventory[:-1],
        )


def test_round_trip_nonoverwrite_and_formal_builder_match(
        tmp_path, formal_manifest):
    """The artifact is canonical, recoverable, deterministic, and immutable."""
    target = tmp_path / "gate.json"
    assert write_jlg_d03_gate_manifest(formal_manifest, target) == target
    assert write_jlg_d03_gate_manifest(formal_manifest, target) == target
    assert read_jlg_d03_gate_manifest(target) == formal_manifest
    target.write_bytes(b"{}\n")
    with pytest.raises(JLGD03GateContractError, match="already differs"):
        write_jlg_d03_gate_manifest(formal_manifest, target)
    assert FORMAL_PATH.is_file()
    assert FORMAL_PATH.read_bytes() == formal_manifest.canonical_bytes()
    assert read_jlg_d03_gate_manifest(FORMAL_PATH) == formal_manifest


def test_builder_is_deterministic(formal_manifest):
    """A second read-only build has the same canonical bytes."""
    if not _candidate_workspace_matches_gate(formal_manifest):
        assert read_jlg_d03_gate_manifest(
            FORMAL_PATH).canonical_bytes() == formal_manifest.canonical_bytes()
        return
    rebuilt = build_jlg_d03_gate_manifest(
        REPOSITORY,
        WORKSPACE,
        head_sha1=_git("rev-parse", "HEAD").decode("ascii").strip(),
        origin_master_sha1=(
            _git("rev-parse", "origin/master").decode("ascii").strip()),
        untracked_relative_paths=_untracked_paths(),
    )
    assert rebuilt.canonical_bytes() == formal_manifest.canonical_bytes()


def test_condition_contract_requires_canonical_sorted_evidence(formal_manifest):
    """Evidence references cannot be duplicated or unsorted."""
    first = formal_manifest.conditions[0]
    with pytest.raises(JLGD03GateContractError, match="sorted and unique"):
        GateCondition(
            first.condition_key,
            first.verdict,
            first.statement,
            (first.evidence_refs[0], first.evidence_refs[0]),
            first.facts,
        )
