"""唯一运行 W02 V4-first R6，并按 PASS/FAIL/NE 不可覆盖封存。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    read_v2_evaluator_boundary_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_artifact import (
    read_w02_morphology_successor_v4_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_private_r6_protocol import (
    W02_MORPH_V4_PRIVATE_R6_EXPOSURE_LEDGER,
    W02_MORPH_V4_PRIVATE_R6_FAMILY_DOCUMENT,
    W02_MORPH_V4_PRIVATE_R6_PATHS,
    W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH,
    W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT,
    consume_w02_morphology_successor_v4_private_r6_guard,
    read_w02_morphology_successor_v4_private_r6_owner_metadata,
    read_w02_morphology_successor_v4_private_r6_protocol_freeze,
    read_w02_morphology_successor_v4_private_r6_registration,
    verify_w02_morphology_successor_v4_private_r6_consumed_guard,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_private_r6_runtime import (
    publish_w02_morphology_successor_v4_private_r6_receipt,
    run_w02_morphology_successor_v4_private_r6_evaluation,
)


def _sha256_file(path: Path) -> tuple[int, str]:
    """流式计算文件长度与 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _failure_seal(
        publication: Path,
        error: BaseException,
        *, phase: str,
        report: dict[str, object] | None,
        ) -> Path:
    """发布只含错误类型、阶段与安全 aggregate 身份的失败封印。"""
    target = publication / "run-000001.failure.json"
    value = {
        "artifact_kind": "PH2_D03_V2_W02_V4_R6_FAILURE_SEAL",
        "error_evidence_sha256": hashlib.sha256(canonical_json_bytes({
            "error_type": type(error).__name__,
            "phase": phase,
        })).hexdigest(),
        "error_type": type(error).__name__,
        "formal_private_evaluation_runs": int(phase != "PRE_GUARD"),
        "phase": phase,
        "report_sha256": (
            None if report is None
            else hashlib.sha256(canonical_json_bytes(report)).hexdigest()),
        "report_status": None if report is None else report.get("status"),
        "runtime_receipt_published": 0,
        "status": "NE_NO_RECEIPT",
    }
    write_immutable_json(value, target)
    return target


def _owner_root(value: str | Path, repository: Path) -> Path:
    """要求 private owner root 位于公开仓之外且为真实目录。"""
    root = Path(value).resolve()
    if (root.is_symlink() or not root.is_dir()
            or root.is_relative_to(repository)):
        raise RuntimeError("R6 private owner root 非法")
    return root


def _validate_pre_guard(
        repository: Path,
        compile_root: Path,
        owner_root: Path,
        owner_metadata_path: Path,
        candidate_artifact_root: Path,
        v1_overlay_artifact_root: Path,
        v2_overlay_artifact_root: Path,
        v4_overlay_artifact_root: Path,
        protocol: dict[str, object],
        ) -> tuple[object, ...]:
    """零 payload 地核对 layout、transport size 与四级 artifact 身份。"""
    required = tuple(compile_root / name for name in (
        "candidate-train", "teacher-train", "dev-calibration", "shadow-audit"))
    if (compile_root.is_symlink() or not compile_root.is_dir()
            or any(path.is_symlink() or not path.is_dir() for path in required)):
        raise RuntimeError("R6 compile root layout 非法")
    owner = read_w02_morphology_successor_v4_private_r6_owner_metadata(
        owner_metadata_path)
    for identity in owner.files:
        target = owner_root / W02_MORPH_V4_PRIVATE_R6_PATHS[identity.layout_key]
        if (target.is_symlink() or not target.is_file()
                or target.stat().st_size != identity.transport_size_bytes):
            raise RuntimeError("R6 owner transport size 漂移")
    candidate = read_w02_candidate_artifact(candidate_artifact_root)
    v1 = read_w02_morphology_overlay_artifact(v1_overlay_artifact_root)
    v2 = read_w02_morphology_successor_v2_overlay_artifact(
        v2_overlay_artifact_root)
    v4 = read_w02_morphology_successor_v4_artifact(v4_overlay_artifact_root)
    r5 = read_canonical_object(repository / (
        "data/ph2/manifests/d03_v2/stages/"
        "ph2_d03_v2_w02_morphology_successor_v3_private_r5_family_freeze_v1.json"))
    v4_receipt = read_canonical_object(repository / (
        "data/ph2/manifests/d03_v2/stages/"
        "ph2_d03_v2_w02_morphology_successor_v4_artifact_receipt_v1.json"))
    chain = r5["artifact_chain"]
    v4_identity = v4_receipt["git_external_artifact"]
    if (candidate.artifact_manifest_sha256
            != chain["candidate_artifact_manifest_sha256"]
            or candidate.candidate_semantic_sha256
            != chain["candidate_semantic_sha256"]
            or v1.artifact_manifest_sha256
            != chain["v1_overlay_artifact_manifest_sha256"]
            or v1.overlay_semantic_sha256 != chain["v1_overlay_semantic_sha256"]
            or v2.artifact_manifest_sha256
            != chain["v2_overlay_artifact_manifest_sha256"]
            or v2.semantic_sha256 != chain["v2_overlay_semantic_sha256"]
            or v4.manifest_sha256 != v4_identity["manifest_sha256"]
            or v4.semantic_sha256 != v4_identity["semantic_sha256"]
            or protocol["artifact_chain"]["v4_artifact_tree_commitment"]
            != v4_identity["tree_commitment"]):
        raise RuntimeError("R6 formal artifact roots 与公共冻结不一致")
    return owner.files


def main() -> int:
    """解析显式 root，消费 guard，运行一次并发布安全结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--compile-root", required=True)
    parser.add_argument("--private-owner-root", required=True)
    parser.add_argument("--owner-metadata", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--v1-overlay-artifact-root", required=True)
    parser.add_argument("--v2-overlay-artifact-root", required=True)
    parser.add_argument("--v4-overlay-artifact-root", required=True)
    parser.add_argument("--family-root", required=True)
    arguments = parser.parse_args()

    repository = Path(arguments.repository_root).resolve()
    compile_root = Path(arguments.compile_root).resolve()
    family_root = Path(arguments.family_root).resolve()
    owner_metadata = Path(arguments.owner_metadata).resolve()
    publication = family_root / "publication"
    publication.mkdir(exist_ok=False)
    report_path = publication / "run-000001.report.json"
    if report_path.exists():
        raise RuntimeError("R6 family 已运行")

    phase = "PRE_GUARD"
    report: dict[str, object] | None = None
    try:
        protocol = read_w02_morphology_successor_v4_private_r6_protocol_freeze(
            repository)
        protocol_path = repository / Path(
            *W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH.split("/"))
        _, protocol_sha = _sha256_file(protocol_path)
        if (read_canonical_object(
                family_root / W02_MORPH_V4_PRIVATE_R6_FAMILY_DOCUMENT)
                != protocol
                or read_canonical_object(
                    family_root / W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT)
                != read_w02_morphology_successor_v4_private_r6_registration(
                    family_root).to_dict()):
            raise RuntimeError("R6 family documents 漂移")
        owner_root = _owner_root(arguments.private_owner_root, repository)
        files = _validate_pre_guard(
            repository, compile_root, owner_root, owner_metadata,
            Path(arguments.candidate_artifact_root).resolve(),
            Path(arguments.v1_overlay_artifact_root).resolve(),
            Path(arguments.v2_overlay_artifact_root).resolve(),
            Path(arguments.v4_overlay_artifact_root).resolve(), protocol)
        registration = read_w02_morphology_successor_v4_private_r6_registration(
            family_root)
        consume_w02_morphology_successor_v4_private_r6_guard(
            repository, family_root, owner_metadata, run_id=1)
        phase = "PRIVATE_AUTHORIZATION_OR_EVALUATION"
        roots = V2PhysicalRoots.from_paths(
            compile_root / "candidate-train",
            compile_root / "teacher-train",
            compile_root / "dev-calibration",
            compile_root / "shadow-audit",
            owner_root,
            family_root / W02_MORPH_V4_PRIVATE_R6_EXPOSURE_LEDGER,
        )
        report = run_w02_morphology_successor_v4_private_r6_evaluation(
            read_v2_evaluator_boundary_contract(repository), roots,
            registration, files,
            arguments.candidate_artifact_root,
            arguments.v1_overlay_artifact_root,
            arguments.v2_overlay_artifact_root,
            arguments.v4_overlay_artifact_root,
            protocol_sha256=protocol_sha, run_id=1)
        verify_w02_morphology_successor_v4_private_r6_consumed_guard(family_root)
        write_immutable_json(report, report_path)
        phase = "REPORT_PUBLISHED"
        report_sha = _sha256_file(report_path)[1]
        if report["status"] != "PASS":
            error = RuntimeError("R6 formal result FAIL or NE")
            _failure_seal(
                publication, error, phase="FORMAL_RESULT_FAIL_OR_NE",
                report=report)
            print(json.dumps({
                "formal_private_evaluation_runs": 1,
                "report_sha256": report_sha,
                "status": report["status"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 2
        receipt_path = publish_w02_morphology_successor_v4_private_r6_receipt(
            repository, report, report_sha256=report_sha)
        phase = "RUNTIME_RECEIPT_PUBLISHED"
        print(json.dumps({
            "formal_private_evaluation_runs": 1,
            "logic_operations": report["logic_operations"],
            "private_payload_reads": report["private_payload_reads"],
            "report_sha256": report_sha,
            "runtime_receipt_sha256": _sha256_file(receipt_path)[1],
            "status": "PASS",
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as error:
        _failure_seal(publication, error, phase=phase, report=report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
