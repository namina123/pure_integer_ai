"""最终 J-LG-D03 合取闸的只读证据目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_jlg_d03_gate_contract import (
    ARTIFACT_PATH,
    EXECUTION_STATE_KEYS,
    FinalPublicGate,
    GateCondition,
    GateEvidenceIdentity,
    JLGD03GateContractError,
    JLGD03GateManifest,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    inventory_public_files,
    read_language_baseline_manifest,
    scan_public_patterns,
    verify_language_baseline_files,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
    read_mediawiki_scan_report,
)
from pure_integer_ai.experiments.ph2_nonliteral_scope_probe_contract import (
    read_nonliteral_scope_probe_manifest,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
)
from pure_integer_ai.experiments.ph2_reasoning_mode_probe_contract import (
    read_reasoning_mode_probe_manifest,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack_coverage_manifest,
)


BASELINE_PATH = "data/ph2/manifests/language_capability_baseline_v39.json"
SOURCE_COVERAGE_PATH = "data/ph2/manifests/d02_source_pack_coverage_v1.json"
CC_RECONCILIATION_PATH = (
    "data/ph2/manifests/cc_cedict_20260725.license_reconciliation_v1.json")
WIKTIONARY_SNAPSHOT_PATH = (
    "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json")
RI00_PATH = "data/ph2/manifests/ri00_reasoning_mode_probe_manifest_v2.json"
NL00_PATH = "data/ph2/manifests/nl00_nonliteral_scope_probe_manifest_v1.json"
LC09_PATH = "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json"
LC10_PATH = "data/ph2/manifests/lc10_retention_rollback_manifest_v1.json"
LC13_PATH = "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json"
GG_PATHS = (
    "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
    "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
    "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
)
MD_PATHS = (
    "data/ph2/manifests/md01_memory_dynamics_contract_v1.json",
    "data/ph2/manifests/md02_situation_state_adapter_v1.json",
    "data/ph2/manifests/md03_directional_center_adapter_v1.json",
    "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
    "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
    "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
)
CORE_COURSE_PATHS = (
    "data/ph2/manifests/lc01_lc15_initial_course_v1.json",
    "data/ph2/manifests/lc02_morphology_course_v1.json",
    "data/ph2/manifests/lc03_construction_course_v1.json",
    "data/ph2/manifests/lc04_recursive_parse_course_v1.json",
    "data/ph2/manifests/lc05_event_time_aspect_course_v1.json",
    "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
    "data/ph2/manifests/lc07_discourse_information_course_v1.json",
    "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
    "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
    "data/ph2/manifests/lc15_final_learning_objectives_v1.json",
)
REQUIRED_SPLIT_AXES = {
    "COMBINATION_CLUSTER",
    "CONTENT_CLUSTER",
    "EVIDENCE_OWNER",
    "SHAPE_CLUSTER",
    "SOURCE_CLUSTER",
    "SPLIT",
    "TEMPLATE_CLUSTER",
}
LC01_SPLIT_AXES = {
    "content_group",
    "family",
    "shape_group",
    "source_cluster",
    "template_group",
}
TRANSFER_AXES = (
    "CODE_SWITCH", "DIALECT", "DOMAIN", "ERA", "GENRE", "LANGUAGE",
    "LENGTH", "REGISTER", "SCRIPT", "SOURCE",
)
FORBIDDEN_EXECUTION_KEYS = {
    "assessment_updates",
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "directional_runtime_runs",
    "evaluator_host_write_count",
    "evaluator_label_writes",
    "formal_training_runs",
    "learning_state_writes",
    "mastered_claims",
    "memory_learning_writes",
    "postcheck_runtime_runs",
    "readiness_claims",
    "retention_runtime_runs",
    "teacher_calls",
    "use_learning_writes",
    "v06_clone_runs",
    "w01_started",
}


def _resolve_under(root: Path, relative_path: str) -> Path:
    path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(root):
        raise JLGD03GateContractError("catalog path escapes its root")
    return path


def _read_json(repository: Path, relative_path: str) -> dict[str, Any]:
    path = _resolve_under(repository, relative_path)
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise JLGD03GateContractError("evidence JSON newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
    except JLGD03GateContractError:
        raise
    except Exception as error:
        raise JLGD03GateContractError(
            f"evidence JSON is damaged: {relative_path}") from error
    if canonical_json_line(value) != payload:
        raise JLGD03GateContractError(
            f"evidence JSON is non-canonical: {relative_path}")
    return value


def _hash_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError as error:
        raise JLGD03GateContractError("evidence file cannot be hashed") from error
    return size, digest.hexdigest()


def _assert_file_sha(repository: Path, relative_path: str, expected: str) -> None:
    path = _resolve_under(repository, relative_path)
    _, actual = _hash_path(path)
    if actual != expected:
        raise JLGD03GateContractError(
            f"repository evidence hash changed: {relative_path}")


def _assert_zero_execution(value: Any, *, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_EXECUTION_KEYS and item != 0:
                raise JLGD03GateContractError(
                    f"forbidden execution is nonzero in {where}: {key}")
            _assert_zero_execution(item, where=where)
    elif isinstance(value, list):
        for item in value:
            _assert_zero_execution(item, where=where)


def _external_identity(
        workspace: Path,
        relative_path: str,
        *,
        scope: str,
        ) -> GateEvidenceIdentity:
    path = _resolve_under(workspace, relative_path)
    if not path.is_file():
        raise JLGD03GateContractError(
            f"external evidence is missing: {relative_path}")
    size, digest = _hash_path(path)
    return GateEvidenceIdentity(relative_path, scope, size, digest)


def _condition(
        key: str,
        statement: str,
        evidence_refs: Iterable[str],
        facts: dict[str, Any],
        ) -> GateCondition:
    return GateCondition(
        key,
        "PASS",
        statement,
        tuple(sorted(set(evidence_refs))),
        CanonicalJsonObject.from_value(facts),
    )


def _validate_artifact_manifest(
        workspace: Path,
        relative_path: str,
        expected_sha256: str,
        ) -> dict[str, Any]:
    path = _resolve_under(workspace, relative_path)
    manifest = read_artifact_manifest(path)
    if manifest.sha256() != expected_sha256:
        raise JLGD03GateContractError("pack manifest SHA-256 changed")
    owners = {item.record_kind: item.owner_kind for item in manifest.files}
    required = {
        "evaluator_label": "evaluator",
        "observation": "observation",
        "source_ref": "source",
        "teacher_evidence": "teacher",
    }
    if owners != required:
        raise JLGD03GateContractError("pack owner isolation is incomplete")
    for item in manifest.files:
        expected_prefix = {
            "evaluator_label": "owners/evaluator/",
            "observation": "observations/",
            "source_ref": "source_refs.jsonl.gz",
            "teacher_evidence": "owners/teacher/",
        }[item.record_kind]
        if item.record_kind == "source_ref":
            valid_path = item.relative_path == expected_prefix
        else:
            valid_path = item.relative_path.startswith(expected_prefix)
        if not valid_path:
            raise JLGD03GateContractError("pack physical owner path changed")
    return {
        "license_partition": manifest.license_partition,
        "owner_kinds": sorted(set(owners.values())),
        "record_count": manifest.record_count,
        "source_cluster_count": len(manifest.source_cluster_keys),
        "splits": list(manifest.splits),
    }


def _build_inventory(
        repository: Path,
        untracked_relative_paths: tuple[str, ...],
        ) -> tuple[tuple[Any, ...], int]:
    paths = tuple(sorted(set(untracked_relative_paths)))
    if len(paths) != len(untracked_relative_paths):
        raise JLGD03GateContractError("untracked path input is duplicated")
    if any(PurePosixPath(item).parts[0] not in {"data", "src", "tests"}
           for item in paths):
        raise JLGD03GateContractError("untracked inventory left data/src/tests")
    if ARTIFACT_PATH in paths:
        inventory_paths = tuple(item for item in paths if item != ARTIFACT_PATH)
        untracked_count = len(paths)
    else:
        inventory_paths = paths
        untracked_count = len(paths) + 1
    return inventory_public_files(repository, inventory_paths), untracked_count


def build_jlg_d03_gate_manifest(
        repository_root: str | Path,
        workspace_root: str | Path,
        *,
        head_sha1: str,
        origin_master_sha1: str,
        untracked_relative_paths: tuple[str, ...],
        ) -> JLGD03GateManifest:
    """只从冻结证据和小型 manifest 构建最终闸。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if not (repository / "src" / "pure_integer_ai").is_dir():
        raise JLGD03GateContractError("repository root is invalid")
    if repository.parent != workspace:
        raise JLGD03GateContractError("workspace/repository relationship changed")

    file_inventory, untracked_count = _build_inventory(
        repository, untracked_relative_paths)
    paper_files = inventory_public_files(
        repository, ("paper/main.pdf", "paper/main.tex"))
    legacy, legacy_binary, legacy_unreadable = scan_public_patterns(
        repository, file_inventory, LEGACY_RULES)
    secret, secret_binary, secret_unreadable = scan_public_patterns(
        repository, file_inventory, SECRET_RULES)
    if (legacy_binary != secret_binary
            or legacy_unreadable != secret_unreadable):
        raise JLGD03GateContractError("public scans used different scopes")
    final_public_gate = FinalPublicGate(
        len(file_inventory),
        len(file_inventory) - len(legacy_binary) - len(legacy_unreadable),
        tuple(key for key, _ in LEGACY_RULES),
        len(legacy),
        tuple(key for key, _ in SECRET_RULES),
        len(secret),
        legacy_binary,
        legacy_unreadable,
        1,
        1,
        int(not (legacy or secret or legacy_binary or legacy_unreadable)),
    )

    baseline_path = _resolve_under(repository, BASELINE_PATH)
    baseline = read_language_baseline_manifest(baseline_path)
    verify_language_baseline_files(baseline, repo_root=repository)
    if (baseline.artifact_status != "BASELINE_FROZEN"
            or baseline.public_gate.legacy_status != "CLEAR"
            or baseline.public_gate.secret_status != "CLEAR"):
        raise JLGD03GateContractError("v39 baseline is not clean and frozen")

    coverage = read_source_pack_coverage_manifest(
        _resolve_under(repository, SOURCE_COVERAGE_PATH))
    frozen_sources = tuple(item for item in coverage.entries
                           if item.status == "PACK_FROZEN")
    blocked_sources = tuple(item for item in coverage.entries
                            if item.status == "BLOCKED")
    if len(frozen_sources) != 6 or len(blocked_sources) != 1:
        raise JLGD03GateContractError("source terminal coverage changed")
    blocked = blocked_sources[0]
    if (blocked.source_key != "CC_CEDICT_20260725"
            or blocked.blocker_code != "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"):
        raise JLGD03GateContractError("CC-CEDICT blocker was not preserved")
    for item in coverage.entries:
        _assert_file_sha(
            repository,
            item.raw_snapshot_manifest_relative_path,
            item.raw_snapshot_manifest_sha256,
        )
    cc = _read_json(repository, CC_RECONCILIATION_PATH)
    if not (
            cc["license_verdict"] == "BLOCKED"
            and cc["historical_license_verdict"] == "BLOCKED"
            and cc["historical_blocker_code"] == "LICENSE_PARTITION_MISMATCH"
            and cc["blocker_code"] == "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"
            and cc["release_eligible"] == 0
            and cc["public_source_pack_emitted"] == 0):
        raise JLGD03GateContractError("CC-CEDICT reconciliation changed")

    wiktionary = read_mediawiki_dump_snapshot(
        _resolve_under(repository, WIKTIONARY_SNAPSHOT_PATH))
    report_paths = tuple(
        f"ph2_dataset_raw/{item.relative_path}"
        for item in wiktionary.parser_reports)
    if len(report_paths) != 2:
        raise JLGD03GateContractError("Wiktionary report pair is incomplete")
    reports = tuple(read_mediawiki_scan_report(
        _resolve_under(workspace, item)) for item in report_paths)
    if reports[0] != reports[1] or reports[0] != wiktionary.final_parser_report:
        raise JLGD03GateContractError("Wiktionary double-pass reports disagree")
    report = reports[0]
    report_hashes = []
    for item, identity in zip(report_paths, wiktionary.parser_reports):
        size, digest = _hash_path(_resolve_under(workspace, item))
        if size != identity.size_bytes or digest != identity.sha256:
            raise JLGD03GateContractError("Wiktionary report identity changed")
        report_hashes.append(digest)
    if not (
            wiktionary.double_pass_equal == 1
            and report.full_eof_verified == 1
            and report.page_count == 3191659
            and report.main_namespace_count == 2674506
            and report.valid_page_count == 2674143
            and report.anomaly_count == 363
            and report.anomaly_codes.to_value() == {
                "UNBALANCED_TEMPLATE": 363}
            and set(report_hashes) == {
                "6d120c78438733497392a21e4ce6844aa9a982a63eb10330bd3e8ee96dbee385"}):
        raise JLGD03GateContractError("Wiktionary frozen scan facts changed")

    gg_values = {path: _read_json(repository, path) for path in GG_PATHS}
    for path, value in gg_values.items():
        _assert_zero_execution(value, where=path)
    gg01, gg02, gg03 = (gg_values[path] for path in GG_PATHS)
    if not (
            baseline.gg00_audit.gg03_exit_state == "COURSE_FROZEN"
            and gg01["artifact_status"] == "CONTRACT_FROZEN"
            and gg02["artifact_status"] == "BRIDGE_FROZEN"
            and gg03["course_status"] == "COURSE_FROZEN"
            and gg03["runtime_status"] == "NOT_STARTED"
            and len(gg03["combination_axes"]) == 10
            and "MULTIPLE_LEGAL_SURFACE_SET" in gg03["evaluator_dimensions"]
            and "FAILURE_LAYER_LOCALIZATION" in gg03["evaluator_dimensions"]):
        raise JLGD03GateContractError("GG-00..03 evidence is incomplete")

    md_values = {path: _read_json(repository, path) for path in MD_PATHS}
    for path, value in md_values.items():
        _assert_zero_execution(value, where=path)
    md05 = md_values[MD_PATHS[-1]]
    if not (
            baseline.md00_preregistration.results_observed == 0
            and md_values[MD_PATHS[0]]["artifact_status"] == "CONTRACT_FROZEN"
            and md_values[MD_PATHS[1]]["artifact_status"] == "ADAPTER_FROZEN"
            and md_values[MD_PATHS[2]]["artifact_status"] == "ADAPTER_FROZEN"
            and md05["results_observed"] == 1
            and md05["verdict"] in {"PASS", "REJECT"}
            and md05["evaluator_host_write_count"] == 0
            and not md05["hard_invariant_failures"]
            and len(md05["strategy_reports"]) == 4
            and len(md05["ablation_evidence"]) == 5):
        raise JLGD03GateContractError("MD-00..05 decision is incomplete")

    lc09 = _read_json(repository, LC09_PATH)
    lc10 = _read_json(repository, LC10_PATH)
    lc13 = _read_json(repository, LC13_PATH)
    for path, value in ((LC09_PATH, lc09), (LC10_PATH, lc10), (LC13_PATH, lc13)):
        _assert_zero_execution(value, where=path)
    if not (
            lc09["artifact_status"] == "CONTRACT_FROZEN"
            and lc09["runtime_status"] == "NOT_STARTED"
            and tuple(lc09["axis_keys"]) == TRANSFER_AXES
            and lc09["pack_inventory_count"] == 16
            and len(lc09["pack_audits"]) == 16
            and len(lc09["split_probes"]) == 3
            and all(item["verdict"] == "PASS"
                    and item["host_learning_writes"] == 0
                    for item in lc09["split_probes"])
            and lc09["runtime_transfer_pass_authority"] == 0):
        raise JLGD03GateContractError("LC-09 transfer axes are incomplete")
    if not (
            lc10["artifact_status"] == "COURSE_FROZEN"
            and lc10["runtime_status"] == "NOT_STARTED"
            and lc10["actual_retention_evidenced"] == 0
            and lc10["runtime_pass_authority"] == 0
            and len(lc10["fixtures"]) == 3
            and len(lc10["retention_sequence"]) == 10):
        raise JLGD03GateContractError("LC-10 protocol is incomplete")
    if not (
            lc13["artifact_status"] == "COURSE_FROZEN"
            and lc13["runtime_status"] == "NOT_STARTED"
            and lc13["route_count"] == 60
            and lc13["runtime_connected_count"] == 0
            and lc13["exact_use_outcome_contract_count"] == 1
            and (lc13["available_not_executed_count"]
                 + lc13["missing_ne_count"] + lc13["out_of_scope_count"]
                 == lc13["route_count"])):
        raise JLGD03GateContractError("LC-13 directional map is incomplete")

    core_courses = {
        path: _read_json(repository, path) for path in CORE_COURSE_PATHS}
    for path, value in core_courses.items():
        _assert_zero_execution(value, where=path)
        if value["course_status"] != "COURSE_FROZEN":
            raise JLGD03GateContractError(f"course is not frozen: {path}")
    for index, path in enumerate(CORE_COURSE_PATHS[:-1]):
        value = core_courses[path]
        expected_axes = LC01_SPLIT_AXES if index == 0 else REQUIRED_SPLIT_AXES
        if not (
                set(value["pack_splits"]) == {"train", "held_out"}
                and set(value["split_axes"]) == expected_axes
                and value["evaluator_dimensions"]
                and value["pack_record_count"] > 0):
            raise JLGD03GateContractError(
                f"course evaluator/split evidence is incomplete: {path}")
    lc15 = core_courses[CORE_COURSE_PATHS[-1]]
    if not (
            lc15["runtime_status"] == "NOT_STARTED"
            and lc15["objective_taxonomy_status"] == "FINAL_FROZEN"
            and lc15["course_source_count"] == 9
            and len(lc15["capability_bindings"]) == 12
            and len(lc15["objectives"]) == 11):
        raise JLGD03GateContractError("LC-15 final objectives are incomplete")

    if not (
            len(baseline.capability_ledger.entries) == 20
            and len(baseline.verifier_registry.records) == 31
            and len(baseline.course_coverage_ledger.records) == 20
            and all(item.verifier_keys
                    for item in baseline.capability_ledger.entries)
            and all(item.exit_state in {
                "BASELINE_ONLY", "COURSE_FROZEN", "OUT_OF_SCOPE",
                "PARTIAL_COURSE"}
                for item in baseline.course_coverage_ledger.records)):
        raise JLGD03GateContractError("LC-00/11/12 ledgers are incomplete")

    ri00 = read_reasoning_mode_probe_manifest(_resolve_under(repository, RI00_PATH))
    if not (
            ri00.artifact_status == "PROBE_DECIDED"
            and (ri00.pass_count, ri00.reject_count, ri00.ne_count) == (1, 4, 0)
            and ri00.runtime_pass_authority == 0):
        raise JLGD03GateContractError("RI-00 bounded decisions changed")
    nl00 = read_nonliteral_scope_probe_manifest(_resolve_under(repository, NL00_PATH))
    if not (
            nl00.artifact_status == "SCOPE_DECIDED"
            and (nl00.pass_count, nl00.reject_count, nl00.ne_count) == (1, 3, 1)
            and nl00.runtime_pass_authority == 0
            and nl00.capability_learned_claims == 0):
        raise JLGD03GateContractError("NL-00 bounded decisions changed")

    external: dict[str, GateEvidenceIdentity] = {}
    pack_facts: list[dict[str, Any]] = []
    for audit in lc09["pack_audits"]:
        relative_path = audit["pack_manifest_relative_path"]
        facts = _validate_artifact_manifest(
            workspace, relative_path, audit["pack_manifest_sha256"])
        pack_facts.append({
            "combination_split_state": audit["combination_split_state"],
            "manifest_path": relative_path,
            "owner_kinds": facts["owner_kinds"],
            "record_count": facts["record_count"],
            "transfer_claim_state": audit["transfer_claim_state"],
        })
        external[relative_path] = _external_identity(
            workspace, relative_path, scope="WORKSPACE_ARTIFACT")
    for item in frozen_sources:
        _validate_artifact_manifest(
            workspace,
            item.pack_manifest_relative_path,
            item.pack_manifest_sha256,
        )
    for relative_path in report_paths:
        external[relative_path] = _external_identity(
            workspace, relative_path, scope="RAW_EVIDENCE")

    all_evidence_values = (
        list(gg_values.values()) + list(md_values.values())
        + [lc09, lc10, lc13, cc]
        + list(core_courses.values())
        + [coverage.to_dict(), baseline.to_dict(), ri00.to_dict(), nl00.to_dict()]
    )
    for index, value in enumerate(all_evidence_values):
        _assert_zero_execution(value, where=f"final-evidence-{index}")

    source_refs = [SOURCE_COVERAGE_PATH, CC_RECONCILIATION_PATH,
                   WIKTIONARY_SNAPSHOT_PATH, *report_paths]
    source_refs.extend(
        item.raw_snapshot_manifest_relative_path for item in coverage.entries)
    conditions = (
        _condition(
            "J-LG-D03-01-SOURCE-EXIT",
            "Every D-02 source has a frozen pack or an explicit terminal blocker.",
            source_refs,
            {
                "blocked_sources": [{
                    "blocker_code": item.blocker_code,
                    "source_key": item.source_key,
                    "verdict": "BLOCKED",
                } for item in blocked_sources],
                "frozen_license_pack_count": len(frozen_sources),
                "source_entry_count": len(coverage.entries),
                "wiktionary_full_eof_verified": report.full_eof_verified,
            },
        ),
        _condition(
            "J-LG-D03-02-GENERATION-GENERALIZATION",
            "GG-00..03 contracts and the independent generation course are frozen.",
            (BASELINE_PATH, *GG_PATHS, gg03["pack_manifest_relative_path"]),
            {
                "combination_axis_count": len(gg03["combination_axes"]),
                "gg00_exit_state": baseline.gg00_audit.gg03_exit_state,
                "gg01_status": gg01["artifact_status"],
                "gg02_status": gg02["artifact_status"],
                "gg03_course_status": gg03["course_status"],
                "runtime_status": gg03["runtime_status"],
            },
        ),
        _condition(
            "J-LG-D03-03-MEMORY-DYNAMICS-DECISION",
            "MD-00..05 have a preregistered, observed, and reviewable decision.",
            (BASELINE_PATH, *MD_PATHS),
            {
                "ablation_count": len(md05["ablation_evidence"]),
                "hard_invariant_failure_count": len(
                    md05["hard_invariant_failures"]),
                "strategy_count": len(md05["strategy_reports"]),
                "verdict": md05["verdict"],
            },
        ),
        _condition(
            "J-LG-D03-04-PACK-AUDITABILITY",
            "All formal packs expose cluster axes and physical owner separation.",
            (LC09_PATH, SOURCE_COVERAGE_PATH, *external),
            {
                "formal_pack_count": len(pack_facts),
                "owner_kinds": ["evaluator", "observation", "source", "teacher"],
                "pack_facts": pack_facts,
                "split_probe_count": len(lc09["split_probes"]),
            },
        ),
        _condition(
            "J-LG-D03-05-LEGACY-IDENTITY-CLEAR",
            "The final candidate inventory has no legacy registered identity.",
            (BASELINE_PATH,
             "src/pure_integer_ai/experiments/ph2_public_gate_rules.py",
             "tests/test_d02_public_gate_rules.py"),
            {
                "finding_count": len(legacy),
                "inventory_file_count": len(file_inventory),
                "rehash_complete": 1,
                "rule_count": len(LEGACY_RULES),
            },
        ),
        _condition(
            "J-LG-D03-06-SECRET-CLEAR",
            "The final candidate inventory has no real secret or key finding.",
            ("src/pure_integer_ai/experiments/ph2_public_gate_rules.py",
             "tests/test_d02_public_gate_rules.py"),
            {
                "binary_count": len(secret_binary),
                "finding_count": len(secret),
                "rule_count": len(SECRET_RULES),
                "unreadable_count": len(secret_unreadable),
            },
        ),
        _condition(
            "J-LG-D03-07-ZERO-FORMAL-EXECUTION",
            "No formal training, teacher call, learning write, or readiness claim ran.",
            (BASELINE_PATH, SOURCE_COVERAGE_PATH, *GG_PATHS, *MD_PATHS,
             LC09_PATH, LC10_PATH, LC13_PATH, RI00_PATH, NL00_PATH),
            {key: 0 for key in EXECUTION_STATE_KEYS},
        ),
        _condition(
            "J-LG-D03-08-CAPABILITY-LEDGERS-FROZEN",
            "LC-00, LC-11, and LC-12 enumerate capabilities, verifiers, and gaps.",
            (BASELINE_PATH, "tests/test_d02_language_baseline.py"),
            {
                "capability_count": len(baseline.capability_ledger.entries),
                "course_coverage_count": len(
                    baseline.course_coverage_ledger.records),
                "exit_state_counts": {
                    state: sum(
                        item.exit_state == state
                        for item in baseline.course_coverage_ledger.records)
                    for state in (
                        "BASELINE_ONLY", "COURSE_FROZEN", "OUT_OF_SCOPE",
                        "PARTIAL_COURSE")
                },
                "verifier_count": len(baseline.verifier_registry.records),
            },
        ),
        _condition(
            "J-LG-D03-09-CORE-COURSES-FROZEN",
            "LC-01..08, LC-14, and LC-15 are course-frozen without runtime claims.",
            (*CORE_COURSE_PATHS,
             *(value["pack_manifest_relative_path"]
               for value in list(core_courses.values())[:-1])),
            {
                "course_count": len(core_courses),
                "course_status": "COURSE_FROZEN",
                "lc01_legacy_split_axes": sorted(LC01_SPLIT_AXES),
                "runtime_learned_claim_count": 0,
                "standard_split_axes": sorted(REQUIRED_SPLIT_AXES),
            },
        ),
        _condition(
            "J-LG-D03-10-TRANSFER-AXES-FROZEN",
            "LC-09 separates transfer axes and leaves undeclared axes explicit.",
            (LC09_PATH, "tests/test_d02_lc09_transfer_axis_manifest.py"),
            {
                "axis_keys": list(TRANSFER_AXES),
                "formal_pack_count": lc09["pack_inventory_count"],
                "runtime_pass_authority": lc09[
                    "runtime_transfer_pass_authority"],
                "split_probe_count": len(lc09["split_probes"]),
            },
        ),
        _condition(
            "J-LG-D03-11-DIRECTIONAL-CONSUMERS-FROZEN",
            "LC-13 freezes understanding, reasoning, and generation routes.",
            (LC13_PATH, GG_PATHS[0], GG_PATHS[1],
             "tests/test_d02_lc13_directional_consumer_manifest.py"),
            {
                "available_not_executed_count": lc13[
                    "available_not_executed_count"],
                "exact_use_outcome_contract_count": lc13[
                    "exact_use_outcome_contract_count"],
                "missing_ne_count": lc13["missing_ne_count"],
                "out_of_scope_count": lc13["out_of_scope_count"],
                "route_count": lc13["route_count"],
                "runtime_connected_count": lc13["runtime_connected_count"],
            },
        ),
        _condition(
            "J-LG-D03-12-RETENTION-ROLLBACK-FROZEN",
            "LC-10 freezes retention, rollback, and scope contraction ordering.",
            (LC10_PATH, "tests/test_d02_lc10_retention_rollback_manifest.py"),
            {
                "actual_retention_evidenced": lc10[
                    "actual_retention_evidenced"],
                "fixture_count": len(lc10["fixtures"]),
                "retention_sequence": lc10["retention_sequence"],
                "runtime_pass_authority": lc10["runtime_pass_authority"],
            },
        ),
    )

    supplemental = (
        _condition(
            "SUP-CC-CEDICT-HISTORICAL-BLOCKER-PRESERVED",
            "CC-CEDICT remains blocked and no historical artifact is rewritten.",
            (CC_RECONCILIATION_PATH,
             "tests/test_d02_cc_cedict_license_reconciliation.py"),
            {
                "current_source_verdict": cc["license_verdict"],
                "historical_blocker_code": cc["historical_blocker_code"],
                "historical_source_verdict": cc[
                    "historical_license_verdict"],
                "public_source_pack_emitted": cc[
                    "public_source_pack_emitted"],
            },
        ),
        _condition(
            "SUP-NL-00-SCOPE-DECIDED",
            "NL-00 preserves bounded PASS, REJECT, and NE scope decisions.",
            (NL00_PATH, "tests/test_d02_nl00_nonliteral_scope_probe.py"),
            {
                "layer_verdicts": {
                    item.layer_key: item.verdict for item in nl00.decisions},
                "ne_count": nl00.ne_count,
                "pass_count": nl00.pass_count,
                "reject_count": nl00.reject_count,
                "runtime_pass_authority": nl00.runtime_pass_authority,
            },
        ),
        _condition(
            "SUP-PAPER-BYTE-IDENTITY",
            "The required paper files retain their frozen byte identities.",
            ("paper/main.pdf", "paper/main.tex"),
            {
                item.relative_path: item.sha256 for item in paper_files
            },
        ),
        _condition(
            "SUP-RI-00-SCOPE-DECIDED",
            "RI-00 preserves one bounded PASS and four explicit REJECT decisions.",
            (RI00_PATH, "tests/test_d02_ri00_reasoning_mode_probe.py"),
            {
                "mode_verdicts": {
                    item.mode_key: item.verdict for item in ri00.decisions},
                "ne_count": ri00.ne_count,
                "pass_count": ri00.pass_count,
                "reject_count": ri00.reject_count,
                "runtime_pass_authority": ri00.runtime_pass_authority,
            },
        ),
        _condition(
            "SUP-WIKTIONARY-DOUBLE-PASS",
            "The two existing Wiktionary reports are equal, canonical, and EOF-complete.",
            (WIKTIONARY_SNAPSHOT_PATH, *report_paths),
            {
                "anomaly_codes": report.anomaly_codes.to_value(),
                "full_eof_verified": report.full_eof_verified,
                "main_namespace_count": report.main_namespace_count,
                "page_count": report.page_count,
                "report_sha256": report_hashes[0],
                "valid_page_count": report.valid_page_count,
            },
        ),
    )

    execution_state = CanonicalJsonObject.from_value({
        key: 0 for key in EXECUTION_STATE_KEYS})
    conjunction = int(
        final_public_gate.public_candidate_clear == 1
        and all(item.verdict == "PASS" for item in (*conditions, *supplemental)))
    return JLGD03GateManifest(
        1,
        "J-LG-D03-prepublication-gate-v3-supersedes-v1-v2",
        "PASS" if conjunction else "BLOCKED",
        "J-LG-D03",
        head_sha1,
        origin_master_sha1,
        0,
        0,
        untracked_count,
        (ARTIFACT_PATH,),
        file_inventory,
        paper_files,
        tuple(external.values()),
        final_public_gate,
        conditions,
        supplemental,
        execution_state,
        conjunction,
        (
            "ALLOW_NEXT_SESSION_TO_PUBLISH_D03"
            if conjunction else "DO_NOT_PUBLISH_D03"),
        0,
    )


__all__ = [
    "BASELINE_PATH",
    "CC_RECONCILIATION_PATH",
    "CORE_COURSE_PATHS",
    "GG_PATHS",
    "LC09_PATH",
    "LC10_PATH",
    "LC13_PATH",
    "MD_PATHS",
    "NL00_PATH",
    "RI00_PATH",
    "SOURCE_COVERAGE_PATH",
    "WIKTIONARY_SNAPSHOT_PATH",
    "build_jlg_d03_gate_manifest",
]
