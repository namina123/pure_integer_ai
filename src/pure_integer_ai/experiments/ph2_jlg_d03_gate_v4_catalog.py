"""从当前候选工作树与 R-01..R-06 证据构建 J-LG-D03 v4。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_authorized_center_generation_contract import (
    read_authorized_center_generation_manifest,
    verify_authorized_center_generation_files,
)
from pure_integer_ai.experiments.ph2_capability_baseline_v41_catalog import (
    build_capability_baseline_v41,
)
from pure_integer_ai.experiments.ph2_capability_baseline_v41_contract import (
    MANIFEST_PATH as BASELINE_V41_PATH,
    VERSION_EVIDENCE,
    VERSION_KEYS,
    read_capability_baseline_v41,
    verify_capability_baseline_v41_files,
)
from pure_integer_ai.experiments.ph2_correction_recovery_contract import (
    read_correction_recovery_manifest,
    verify_correction_recovery_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_jlg_d03_gate_contract import (
    FinalPublicGate,
    GateCondition,
    read_jlg_d03_gate_manifest,
    verify_jlg_d03_gate_files,
)
from pure_integer_ai.experiments.ph2_jlg_d03_gate_v4_contract import (
    ARTIFACT_KIND,
    ARTIFACT_PATH,
    ARTIFACT_VERSION,
    CONDITION_KEYS,
    EXECUTION_STATE_KEYS,
    PAPER_SHA256,
    REQUIRED_EDGE_PAIRS,
    REQUIRED_NODE_SPECS,
    V3_PATH,
    V3_SHA256,
    GateV4DependencyEdge,
    GateV4DependencyNode,
    JLGD03GateV4Error,
    JLGD03GateV4Manifest,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    inventory_public_files,
    scan_public_patterns,
)
from pure_integer_ai.experiments.ph2_long_context_absorption_contract import (
    read_long_context_absorption_manifest,
    verify_long_context_absorption_files,
)
from pure_integer_ai.experiments.ph2_p3ia_ledger_revision_contract import (
    read_p3ia_ledger_revision,
    verify_p3ia_ledger_revision_files,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack_coverage_manifest,
)
from pure_integer_ai.experiments.ph2_storage_absorption_contract import (
    read_storage_absorption_manifest,
    verify_storage_absorption_files,
)
from pure_integer_ai.experiments.ph2_typed_proof_family_contract import (
    read_typed_proof_family_manifest,
    verify_typed_proof_family_files,
)


_BASELINE_V40_PATH = (
    "data/ph2/manifests/language_capability_baseline_v40.json")
_P3IA_PATH = (
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json")
_D02_PATH = "data/ph2/manifests/d02_source_pack_coverage_v1.json"
_CC_PATH = (
    "data/ph2/manifests/cc_cedict_20260725.license_reconciliation_v1.json")
_R_PATHS = {
    "R02": "data/ph2/manifests/r02_storage_absorption_v1.json",
    "R03": "data/ph2/manifests/r03_correction_recovery_absorption_v1.json",
    "R04": "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json",
    "R05": "data/ph2/manifests/r05_typed_proof_family_absorption_v1.json",
    "R06": "data/ph2/manifests/r06_long_context_absorption_v1.json",
}
_FORBIDDEN_EXECUTION_KEYS = {
    "assessment_updates", "companion_writes", "core_learning_writes",
    "d03_published", "evaluator_host_write_count", "evaluator_label_writes",
    "formal_training_runs", "learning_state_writes", "mastered_claims",
    "memory_learning_writes", "readiness_claims", "teacher_calls",
    "use_learning_writes", "w01_started",
}


class JLGD03GateV4CatalogError(RuntimeError):
    """v4 上游、工作树 inventory、许可或扫描事实漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    result = (root / Path(*relative_path.split("/"))).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise JLGD03GateV4CatalogError("v4 catalog 路径逃逸") from error
    return result


def _json(repository: Path, relative_path: str) -> dict[str, Any]:
    try:
        payload = _path(repository, relative_path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise JLGD03GateV4CatalogError("上游 JSON newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        return value
    except JLGD03GateV4CatalogError:
        raise
    except Exception as error:
        raise JLGD03GateV4CatalogError("上游 JSON 损坏") from error


def _zero_execution(value: Any, *, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_EXECUTION_KEYS and item != 0:
                raise JLGD03GateV4CatalogError(
                    f"{where} 禁止执行状态非零: {key}")
            _zero_execution(item, where=where)
    elif isinstance(value, list):
        for item in value:
            _zero_execution(item, where=where)


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


def _build_inventory(
        repository: Path,
        modified_relative_paths: tuple[str, ...],
        untracked_relative_paths: tuple[str, ...],
        evidence_relative_paths: tuple[str, ...],
        ) -> tuple[tuple[Any, ...], tuple[str, ...], int, int, int]:
    modified = tuple(sorted(set(modified_relative_paths)))
    untracked_set = set(untracked_relative_paths)
    if (len(modified) != len(modified_relative_paths)
            or len(untracked_set) != len(untracked_relative_paths)):
        raise JLGD03GateV4CatalogError("Git candidate 路径输入重复")
    untracked_set.add(ARTIFACT_PATH)
    untracked = tuple(sorted(untracked_set))
    if set(modified) & set(untracked):
        raise JLGD03GateV4CatalogError("tracked/untracked candidate 重叠")
    candidates = tuple(sorted((*modified, *untracked)))
    if any(Path(item).parts[0] not in {"data", "src", "tests"}
           for item in candidates):
        raise JLGD03GateV4CatalogError("candidate inventory 越出 data/src/tests")
    candidate_paths = tuple(item for item in candidates if item != ARTIFACT_PATH)
    inventory_paths = tuple(sorted({*candidate_paths, *evidence_relative_paths}))
    if any(Path(item).parts[0] not in {"data", "src", "tests"}
           for item in inventory_paths):
        raise JLGD03GateV4CatalogError("evidence inventory 越出 data/src/tests")
    return (
        inventory_public_files(repository, inventory_paths),
        candidate_paths,
        len(modified),
        len(untracked),
        len(candidates),
    )


def _audit_v3(repository: Path, workspace: Path) -> None:
    v3_path = _path(repository, V3_PATH)
    payload = v3_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != V3_SHA256:
        raise JLGD03GateV4CatalogError("v3 byte identity 漂移")
    v3 = read_jlg_d03_gate_manifest(v3_path)
    verify_jlg_d03_gate_files(
        v3, repository_root=repository, workspace_root=workspace)
    if not (
            v3.artifact_status == "PASS"
            and v3.conjunction_passed == 1
            and v3.d03_published == 0):
        raise JLGD03GateV4CatalogError("v3 合取状态漂移")


def _audit_v41(repository: Path, workspace: Path) -> None:
    path = _path(repository, BASELINE_V41_PATH.as_posix())
    stored = read_capability_baseline_v41(path)
    verify_capability_baseline_v41_files(stored, repository_root=repository)
    rebuilt = build_capability_baseline_v41(repository, workspace)
    if stored.canonical_bytes() != rebuilt.canonical_bytes():
        raise JLGD03GateV4CatalogError("v41 与当前 dependency 不一致")


def _audit_absorption(
        repository: Path, workspace: Path,
        ) -> dict[str, Any]:
    r02 = read_storage_absorption_manifest(_path(repository, _R_PATHS["R02"]))
    verify_storage_absorption_files(
        r02, repository_root=repository, workspace_root=workspace)
    r03 = read_correction_recovery_manifest(_path(repository, _R_PATHS["R03"]))
    verify_correction_recovery_files(r03, repository_root=repository)
    r04 = read_authorized_center_generation_manifest(
        _path(repository, _R_PATHS["R04"]))
    verify_authorized_center_generation_files(r04, repository_root=repository)
    r05 = read_typed_proof_family_manifest(_path(repository, _R_PATHS["R05"]))
    verify_typed_proof_family_files(r05, repository_root=repository)
    r06 = read_long_context_absorption_manifest(_path(repository, _R_PATHS["R06"]))
    verify_long_context_absorption_files(r06, repository_root=repository)
    values = {"R02": r02, "R03": r03, "R04": r04, "R05": r05, "R06": r06}
    if any(item.artifact_status != "PRODUCTION_EVIDENCED"
           for item in values.values()):
        raise JLGD03GateV4CatalogError("R-02..R-06 状态漂移")
    if any(any(value != 0 for value in item.execution_state.to_value().values())
           for item in values.values()):
        raise JLGD03GateV4CatalogError("R-02..R-06 执行状态非零")
    return values


def _audit_p3ia(repository: Path, workspace: Path) -> dict[str, Any]:
    v40 = read_p3ia_ledger_revision(_path(repository, _BASELINE_V40_PATH))
    verify_p3ia_ledger_revision_files(
        v40, repository_root=repository, workspace_root=workspace)
    course = _json(repository, _P3IA_PATH)
    if not (
            v40.p3ia_course_status == "COURSE_FROZEN"
            and v40.p3ia_production_contract_status == "CONTRACT_READY"
            and v40.formal_runtime_status == "NOT_STARTED"
            and v40.focused_runtime_evidence == "PASS"
            and v40.p3ib_status == "NE"
            and v40.p3ib_phase == "PH3"
            and v40.code_switch_status == "NE"
            and v40.cross_language_pass_authority == 0
            and course.get("course_status") == "COURSE_FROZEN"
            and course.get("runtime_status") == "NOT_STARTED"):
        raise JLGD03GateV4CatalogError("P3-Ia/P3-Ib 状态漂移")
    _zero_execution(course, where="P3-Ia course")
    return course


def _audit_source_license(repository: Path, workspace: Path) -> tuple[int, int]:
    coverage = read_source_pack_coverage_manifest(_path(repository, _D02_PATH))
    frozen = tuple(item for item in coverage.entries if item.status == "PACK_FROZEN")
    blocked = tuple(item for item in coverage.entries if item.status == "BLOCKED")
    if (len(coverage.entries), len(frozen), len(blocked)) != (7, 6, 1):
        raise JLGD03GateV4CatalogError("source terminal coverage 漂移")
    if not (
            blocked[0].source_key == "CC_CEDICT_20260725"
            and blocked[0].blocker_code
            == "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"):
        raise JLGD03GateV4CatalogError("CC-CEDICT blocker 漂移")
    cc = _json(repository, _CC_PATH)
    if not (
            cc.get("license_verdict") == "BLOCKED"
            and cc.get("historical_license_verdict") == "BLOCKED"
            and cc.get("release_eligible") == 0
            and cc.get("public_source_pack_emitted") == 0):
        raise JLGD03GateV4CatalogError("CC-CEDICT license 裁决漂移")
    for item in coverage.entries:
        raw = _path(repository, item.raw_snapshot_manifest_relative_path)
        if hashlib.sha256(raw.read_bytes()).hexdigest() != (
                item.raw_snapshot_manifest_sha256):
            raise JLGD03GateV4CatalogError("raw snapshot identity 漂移")
    for item in frozen:
        pack_path = _path(workspace, item.pack_manifest_relative_path)
        pack = read_artifact_manifest(pack_path)
        if pack.sha256() != item.pack_manifest_sha256:
            raise JLGD03GateV4CatalogError("source pack identity 漂移")
        owners = {value.record_kind: value.owner_kind for value in pack.files}
        if owners != {
                "evaluator_label": "evaluator",
                "observation": "observation",
                "source_ref": "source",
                "teacher_evidence": "teacher",
        }:
            raise JLGD03GateV4CatalogError("SourceRef/Evidence owner path 漂移")
    return len(frozen), len(blocked)


def _node_status(repository: Path, node_key: str) -> str:
    expected_path, expected_status = REQUIRED_NODE_SPECS[node_key]
    value = _json(repository, expected_path)
    status = {
        "BASELINE_V40": value.get("artifact_status"),
        "BASELINE_V41": value.get("artifact_status"),
        "D02_SOURCE": "6_PACK_FROZEN_1_BLOCKED",
        "GG01": value.get("artifact_status"),
        "GG02": value.get("artifact_status"),
        "GG03": value.get("course_status"),
        "LC07_V2": value.get("artifact_status"),
        "LC09_V2": value.get("artifact_status"),
        "LC13_V2": value.get("artifact_status"),
        "LC15_V2": value.get("artifact_status"),
        "MD01": value.get("artifact_status"),
        "MD02": value.get("artifact_status"),
        "MD03": value.get("artifact_status"),
        "MD04_PLAN": "PREREGISTERED" if value.get("results_observed") == 0 else None,
        "MD04_RUNS": "RESULTS_OBSERVED" if value.get("results_observed") == 1 else None,
        "MD05": value.get("verdict"),
        "P3IA_COURSE": value.get("course_status"),
        "R02": value.get("artifact_status"),
        "R03": value.get("artifact_status"),
        "R04": value.get("artifact_status"),
        "R05": value.get("artifact_status"),
        "R06": value.get("artifact_status"),
        "V3_GATE": value.get("artifact_status"),
    }[node_key]
    if status != expected_status:
        raise JLGD03GateV4CatalogError(f"dependency status 漂移: {node_key}")
    _zero_execution(value, where=node_key)
    return status


def _dependency_nodes(repository: Path) -> tuple[GateV4DependencyNode, ...]:
    result = []
    for node_key, (relative_path, _) in REQUIRED_NODE_SPECS.items():
        path = _path(repository, relative_path)
        payload = path.read_bytes()
        result.append(GateV4DependencyNode(
            node_key,
            relative_path,
            _node_status(repository, node_key),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        ))
    return tuple(result)


def _artifact_refs(manifest: Any, artifact_path: str) -> tuple[str, ...]:
    paths = [artifact_path]
    for item in manifest.evidence_files:
        if getattr(item, "root_key", "REPOSITORY") == "REPOSITORY":
            paths.append(item.relative_path)
    return tuple(paths)


def build_jlg_d03_gate_v4(
        repository_root: str | Path,
        workspace_root: str | Path,
        *,
        head_sha1: str,
        origin_master_sha1: str,
        modified_relative_paths: tuple[str, ...],
        staged_change_count: int,
        untracked_relative_paths: tuple[str, ...],
        ) -> JLGD03GateV4Manifest:
    """回验所有补充依赖并从当前 candidate 工作树构建 v4。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if repository.parent != workspace:
        raise JLGD03GateV4CatalogError("repository/workspace 边界非法")
    absorption = _audit_absorption(repository, workspace)
    version_refs = tuple(
        path for paths in VERSION_EVIDENCE.values() for path in paths)
    all_node_paths = tuple(path for path, _ in REQUIRED_NODE_SPECS.values())
    absorption_refs = tuple(
        path
        for key, manifest in absorption.items()
        for path in _artifact_refs(manifest, _R_PATHS[key]))
    evidence_closure = tuple(sorted({
        *all_node_paths,
        *version_refs,
        *absorption_refs,
        _BASELINE_V40_PATH,
        BASELINE_V41_PATH.as_posix(),
        _P3IA_PATH,
        _D02_PATH,
        _CC_PATH,
        "src/pure_integer_ai/experiments/ph2_jlg_d03_gate_v4_catalog.py",
        "src/pure_integer_ai/experiments/ph2_jlg_d03_gate_v4_contract.py",
        "src/pure_integer_ai/experiments/ph2_public_gate_rules.py",
        "tests/test_d02_jlg_d03_gate_v4.py",
        "tests/test_d02_public_gate_rules.py",
    }))
    (file_inventory, candidate_paths, tracked_count, untracked_count,
     candidate_count) = _build_inventory(
         repository, modified_relative_paths, untracked_relative_paths,
         evidence_closure)
    paper_files = inventory_public_files(
        repository, tuple(sorted(PAPER_SHA256)))
    legacy, legacy_binary, legacy_unreadable = scan_public_patterns(
        repository, file_inventory, LEGACY_RULES)
    secret, secret_binary, secret_unreadable = scan_public_patterns(
        repository, file_inventory, SECRET_RULES)
    if (legacy_binary != secret_binary
            or legacy_unreadable != secret_unreadable):
        raise JLGD03GateV4CatalogError("public scan 范围不一致")
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

    _audit_v3(repository, workspace)
    _audit_v41(repository, workspace)
    _audit_p3ia(repository, workspace)
    frozen_sources, blocked_sources = _audit_source_license(
        repository, workspace)
    nodes = _dependency_nodes(repository)
    edges = tuple(GateV4DependencyEdge(*pair) for pair in REQUIRED_EDGE_PAIRS)
    r02, r03, r04, r05, r06 = (
        absorption[key] for key in ("R02", "R03", "R04", "R05", "R06"))
    v41_path = BASELINE_V41_PATH.as_posix()
    zero_state = {key: 0 for key in EXECUTION_STATE_KEYS}
    conditions = (
        _condition(
            CONDITION_KEYS[0],
            "P3-Ia course, private labels, production consumer, and honest state are bound.",
            (_BASELINE_V40_PATH, v41_path, _P3IA_PATH,
             "data/ph2/manifests/lc07_discourse_information_course_v2.json",
             "data/ph2/manifests/lc09_transfer_axis_manifest_v2.json",
             "data/ph2/manifests/lc13_directional_consumer_manifest_v2.json",
             "data/ph2/manifests/lc15_final_learning_objectives_v2.json",
             "src/pure_integer_ai/experiments/free_text_hierarchy_recall_evaluator.py",
             "src/pure_integer_ai/experiments/free_text_recall_runtime.py",
             "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py"),
            {
                "consumer_status": "CONTRACT_READY",
                "course_status": "COURSE_FROZEN",
                "focused_runtime_evidence": "PASS",
                "formal_runtime_status": "NOT_STARTED",
                "label_owner_isolated": 1,
            },
        ),
        _condition(
            CONDITION_KEYS[1],
            "R-02 storage absorption and bounded profile evidence are current.",
            _artifact_refs(r02, _R_PATHS["R02"]),
            {
                "artifact_status": r02.artifact_status,
                "exact_query_segment_payload_gets": 10,
                "record_count": 100000,
                "startup_segment_payload_gets": 0,
            },
        ),
        _condition(
            CONDITION_KEYS[2],
            "R-03 exact-source correction and recovery evidence are current.",
            _artifact_refs(r03, _R_PATHS["R03"]),
            {
                "artifact_status": r03.artifact_status,
                "exact_source_forget": 1,
                "sqlite_runtime_owned_commit": 1,
                "three_fault_points": 3,
            },
        ),
        _condition(
            CONDITION_KEYS[3],
            "R-04 read-before-authorization, multi-center, and strict delivery are current.",
            _artifact_refs(r04, _R_PATHS["R04"]),
            {
                "artifact_status": r04.artifact_status,
                "read_before_authorization": 1,
                "strict_generation_delivery": 1,
                "two_centers_one_physical_read": 1,
            },
        ),
        _condition(
            CONDITION_KEYS[4],
            "R-05 five mutually exclusive typed proof families are current.",
            _artifact_refs(r05, _R_PATHS["R05"]),
            {
                "artifact_status": r05.artifact_status,
                "proof_family_count": 5,
                "strict_dispatch_budget": 1,
            },
        ),
        _condition(
            CONDITION_KEYS[5],
            "R-06 long input, persistent agenda, and generation checkpoint are current.",
            _artifact_refs(r06, _R_PATHS["R06"]),
            {
                "artifact_status": r06.artifact_status,
                "cross_process_continuation": 1,
                "mechanism_count": 3,
            },
        ),
        _condition(
            CONDITION_KEYS[6],
            "D-02, MD, GG, and the new v41 baseline remain frozen and honest.",
            (v41_path, _D02_PATH,
             "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
             "data/ph2/manifests/gg03_generation_generalization_course_v1.json"),
            {
                "baseline_version": "LG-LC-MD-GG-baseline-v41-supersedes-v40",
                "d02_source_entry_count": 7,
                "gg03_course_status": "COURSE_FROZEN",
                "md05_decision": "PASS",
            },
        ),
        _condition(
            CONDITION_KEYS[7],
            "Backend, segment, location, schema, code, and evaluator versions are bound.",
            (v41_path, *version_refs),
            VERSION_KEYS,
        ),
        _condition(
            CONDITION_KEYS[8],
            "Every dependency node and candidate file has an exact identity and edge.",
            (*all_node_paths,
             "src/pure_integer_ai/experiments/ph2_jlg_d03_gate_v4_catalog.py",
             "src/pure_integer_ai/experiments/ph2_jlg_d03_gate_v4_contract.py",
             "tests/test_d02_jlg_d03_gate_v4.py"),
            {
                "candidate_file_count": candidate_count,
                "dependency_edge_count": len(edges),
                "dependency_node_count": len(nodes),
                "file_identity_complete": 1,
                "inventory_file_count": len(file_inventory),
                "tracked_change_count": tracked_count,
                "untracked_file_count": untracked_count,
            },
        ),
        _condition(
            CONDITION_KEYS[9],
            "P3-Ib and code-switch remain NE/PH3 with no PASS authority.",
            (_BASELINE_V40_PATH, v41_path),
            {
                "code_switch_status": "NE",
                "cross_language_pass_authority": 0,
                "p3ib_phase": "PH3",
                "p3ib_status": "NE",
            },
        ),
        _condition(
            CONDITION_KEYS[10],
            "The full candidate inventory is clear under public and secret rules.",
            ("src/pure_integer_ai/experiments/ph2_public_gate_rules.py",
             "tests/test_d02_public_gate_rules.py",
             "tests/test_d02_jlg_d03_gate_v4.py"),
            {
                "artifact_self_excluded": 1,
                "binary_count": len(legacy_binary),
                "legacy_finding_count": len(legacy),
                "post_publish_self_scan_required": 1,
                "secret_finding_count": len(secret),
                "unreadable_count": len(legacy_unreadable),
            },
        ),
        _condition(
            CONDITION_KEYS[11],
            "All source packs retain license, SourceRef, Evidence, and owner paths.",
            (_D02_PATH, _CC_PATH, V3_PATH),
            {
                "blocked_source_count": blocked_sources,
                "frozen_license_pack_count": frozen_sources,
                "source_entry_count": frozen_sources + blocked_sources,
            },
        ),
        _condition(
            CONDITION_KEYS[12],
            "Paper PDF and TEX retain their frozen byte identities.",
            tuple(sorted(PAPER_SHA256)),
            PAPER_SHA256,
        ),
        _condition(
            CONDITION_KEYS[13],
            "Training, teacher, learning writes, D-03, and W-01 remain zero.",
            (v41_path, *_R_PATHS.values()),
            zero_state,
        ),
        _condition(
            CONDITION_KEYS[14],
            "v3 remains byte-identical while v4 explicitly supersedes it.",
            (V3_PATH,),
            {
                "supersedes": "v3",
                "v3_artifact_status": "PASS",
                "v3_sha256": V3_SHA256,
            },
        ),
        _condition(
            CONDITION_KEYS[15],
            "This gate permits only a future confirmed D-03 publication session.",
            (v41_path,
             "src/pure_integer_ai/experiments/ph2_jlg_d03_gate_v4_contract.py"),
            {
                "d03_published": 0,
                "publication_scope": "FUTURE_CONFIRMED_SESSION_ONLY",
                "w01_started": 0,
            },
        ),
    )
    conjunction = int(
        final_public_gate.public_candidate_clear == 1
        and all(item.verdict == "PASS" for item in conditions))
    try:
        return JLGD03GateV4Manifest(
            1,
            ARTIFACT_KIND,
            ARTIFACT_VERSION,
            "PASS" if conjunction else "BLOCKED",
            "J-LG-D03",
            head_sha1,
            origin_master_sha1,
            tracked_count,
            staged_change_count,
            untracked_count,
            candidate_count,
            candidate_paths,
            (ARTIFACT_PATH,),
            file_inventory,
            paper_files,
            nodes,
            edges,
            CanonicalJsonObject.from_value(VERSION_KEYS),
            final_public_gate,
            conditions,
            CanonicalJsonObject.from_value(zero_state),
            conjunction,
            (
                "ALLOW_FUTURE_CONFIRMED_SESSION_TO_PUBLISH_D03"
                if conjunction else "DO_NOT_PUBLISH_D03"),
            0,
        )
    except JLGD03GateV4Error as error:
        raise JLGD03GateV4CatalogError("v4 构建失败") from error


__all__ = [
    "JLGD03GateV4CatalogError",
    "build_jlg_d03_gate_v4",
]
