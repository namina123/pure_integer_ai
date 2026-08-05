"""W09-10 private family 的 metadata-only 冻结与唯一 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALL_DIMENSION_KEYS,
    W09_DIMENSION_KEYS,
    W09_RESOURCE_BUDGET,
    W09_WALL_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_contract import open_w09_frozen_contract
from pure_integer_ai.experiments.ph2_w09_evaluator_contract import (
    W09_EVALUATOR_FAILURE_PHASES,
    W09_EVALUATOR_PHASES,
    W09_EVALUATOR_THRESHOLD,
    W09_PRIVATE_CASE_NAME,
    W09_PRIVATE_CLUSTER_NAME,
    W09_PRIVATE_FAMILY_FREEZE_NAME,
    W09_PRIVATE_FIRST_RUN_GUARD_NAME,
    W09_PRIVATE_INFERENCE_INTERFACE_VERSION,
    W09_PRIVATE_LABEL_NAME,
    W09_PRIVATE_HARD_CONJUNCT_KEYS,
    W09_PRIVATE_OWNER_KEY,
    W09_PRIVATE_SCHEMA_NAME,
    W09_PRIVATE_SOURCE_NAME,
    W09PrivateEvaluationError,
    evidence_commitment,
    strict_sha1,
    strict_sha256,
)
from pure_integer_ai.experiments.ph2_w09_rotation import W09RotationManifest

W09_PRIVATE_FAMILY_KIND = "PH2_W09_PRIVATE_FAMILY_FREEZE"
W09_PRIVATE_SOURCE_KIND = "PH2_W09_PRIVATE_SOURCE"
W09_PRIVATE_SCHEMA_KIND = "PH2_W09_PRIVATE_SCHEMA"
W09_PRIVATE_CASE_KIND = "PH2_W09_PRIVATE_CASE_INVENTORY"
W09_PRIVATE_LABEL_KIND = "PH2_W09_PRIVATE_LABEL_INVENTORY"
W09_PRIVATE_CLUSTER_KIND = "PH2_W09_PRIVATE_CLUSTER_INVENTORY"
W09_PRIVATE_GUARD_KIND = "PH2_W09_PRIVATE_FIRST_RUN_GUARD"


def _sha256(payload: bytes) -> str:
    """计算文档字节 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_head(repository: Path) -> str:
    """读取当前 public HEAD，不访问任何数据 payload。"""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise W09PrivateEvaluationError("W09 evaluator 无法读取 public HEAD") from error
    return strict_sha1(value, label="evaluator public HEAD")


def _identity(binding: object) -> dict[str, object]:
    """把 W09 binding 投影为 metadata-only identity。"""
    return {
        "access_phase": binding.access_phase,
        "identity": binding.identity.to_dict(),
        "pack_key": binding.pack_key,
        "relative_path": binding.relative_path,
    }


@dataclass(frozen=True)
class W09PrivateFamilyDocuments:
    """五份 family 文档及其交叉引用 commitment。"""

    source_bytes: bytes
    schema_bytes: bytes
    case_bytes: bytes
    label_bytes: bytes
    cluster_bytes: bytes
    family_key: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    candidate_contract_sha256: str
    candidate_guard_sha256: str
    candidate_host_sha256: str
    candidate_seal_sha256: str
    candidate_public_head_commit_sha1: str
    evaluator_public_head_commit_sha1: str
    rotation_manifest_sha256: str
    rotation_package_commitment: str
    fixed_d03_exposure_eligible: int = 0
    rotation_exposure_audit_clean: int = 1

    def files(self) -> tuple[tuple[str, bytes], ...]:
        """返回按固定顺序写入的 family 文档。"""
        return (
            (W09_PRIVATE_SOURCE_NAME, self.source_bytes),
            (W09_PRIVATE_SCHEMA_NAME, self.schema_bytes),
            (W09_PRIVATE_CASE_NAME, self.case_bytes),
            (W09_PRIVATE_LABEL_NAME, self.label_bytes),
            (W09_PRIVATE_CLUSTER_NAME, self.cluster_bytes),
        )


def _decode(payload: bytes, kind: str) -> dict[str, Any]:
    """解析并核验一份 canonical family 文档。"""
    try:
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W09PrivateEvaluationError("W09 private family document 无法解析") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload or value.get("artifact_kind") != kind or value.get("format_version") != 1:
        raise W09PrivateEvaluationError("W09 private family document identity 漂移")
    return value


def _commitment_bundle(documents: W09PrivateFamilyDocuments) -> str:
    """计算五文档的公共 payload commitment。"""
    return evidence_commitment({
        "cases": _sha256(documents.case_bytes),
        "clusters": _sha256(documents.cluster_bytes),
        "labels": _sha256(documents.label_bytes),
        "schema": _sha256(documents.schema_bytes),
        "source": _sha256(documents.source_bytes),
    })


def build_w09_private_family_documents(
    repository_root: str | Path,
    *,
    candidate_contract_sha256: str,
    candidate_guard_sha256: str,
    candidate_host_sha256: str,
    candidate_seal_sha256: str,
    candidate_public_head_commit_sha1: str,
    evaluator_public_head_commit_sha1: str | None = None,
    rotation_manifest: W09RotationManifest | None = None,
    fixed_d03_exposure_eligible: int = 0,
) -> W09PrivateFamilyDocuments:
    """只用 manifest metadata 冻结 D-03 与 rotation family，不读取 payload。"""
    repository = Path(repository_root).resolve()
    context = open_w09_frozen_contract(repository)
    head = _git_head(repository)
    supplied_head = strict_sha1(evaluator_public_head_commit_sha1 or head, label="evaluator public HEAD")
    if supplied_head != head:
        raise W09PrivateEvaluationError("W09 evaluator public HEAD 参数漂移")
    values = {
        "candidate_contract_sha256": strict_sha256(candidate_contract_sha256, label="Candidate contract"),
        "candidate_first_run_guard_sha256": strict_sha256(candidate_guard_sha256, label="Candidate guard"),
        "candidate_host_freeze_sha256": strict_sha256(candidate_host_sha256, label="Candidate host"),
        "candidate_terminal_seal_sha256": strict_sha256(candidate_seal_sha256, label="Candidate seal"),
    }
    candidate_head = strict_sha1(candidate_public_head_commit_sha1, label="Candidate public HEAD")
    if type(fixed_d03_exposure_eligible) is not int or fixed_d03_exposure_eligible not in {0, 1}:
        raise W09PrivateEvaluationError("W09 fixed D-03 exposure flag 非法")
    if fixed_d03_exposure_eligible:
        raise W09PrivateEvaluationError("历史暴露 D-03 family 不具备 blind PASS 资格")
    if not isinstance(rotation_manifest, W09RotationManifest):
        raise W09PrivateEvaluationError("W09 formal family 必须绑定独立 rotation manifest")
    rotation_sha = strict_sha256(rotation_manifest.sha256(), label="rotation manifest")
    rotation_commitment = strict_sha256(rotation_manifest.package_commitment, label="rotation package")
    observations = tuple(item for item in context.evaluator_bindings if item.identity.owner_kind == "observation")
    labels = tuple(item for item in context.evaluator_bindings if item.identity.owner_kind == "evaluator")
    if len(observations) != len(labels) or not observations:
        raise W09PrivateEvaluationError("W09 D-03 evaluator inventory 不闭合")
    family_identity = {
        **values,
        "candidate_head": candidate_head,
        "evaluator_head": supplied_head,
        "owner": W09_PRIVATE_OWNER_KEY,
        "rotation_manifest_sha256": rotation_sha,
        "rotation_package_commitment": rotation_commitment,
    }
    family_key = evidence_commitment(family_identity)
    source = {
        "artifact_kind": W09_PRIVATE_SOURCE_KIND,
        "candidate_contract_sha256": values["candidate_contract_sha256"],
        "candidate_first_run_guard_sha256": values["candidate_first_run_guard_sha256"],
        "candidate_host_freeze_sha256": values["candidate_host_freeze_sha256"],
        "candidate_terminal_seal_sha256": values["candidate_terminal_seal_sha256"],
        "candidate_public_head_commit_sha1": candidate_head,
        "d03_observation_bindings": [_identity(item) for item in observations],
        "d03_label_bindings": [_identity(item) for item in labels],
        "evaluator_public_head_commit_sha1": supplied_head,
        "family_key": family_key,
        "fixed_d03_exposure_eligible": 0,
        "format_version": 1,
        "owner_key": W09_PRIVATE_OWNER_KEY,
        "rotation_exposure_audit_clean": 1,
        "rotation_package": {
            "manifest_sha256": rotation_sha,
            "package_commitment": rotation_commitment,
            "case_commitment": rotation_manifest.rotation_case_commitment,
            "label_owner": rotation_manifest.label_identity.owner_kind,
            "observation_count": rotation_manifest.observation_identity.record_count,
        },
    }
    schema = {
        "ablation_order": list(W09_ABLATION_KEYS),
        "artifact_kind": W09_PRIVATE_SCHEMA_KIND,
        "bearing_dimension_order": list(W09_DIMENSION_KEYS),
        "dimension_order": list(W09_ALL_DIMENSION_KEYS),
        "failure_phases": list(W09_EVALUATOR_FAILURE_PHASES),
        "hard_conjunct_order": list(W09_PRIVATE_HARD_CONJUNCT_KEYS),
        "inference_interface": {"required": 1, "version": W09_PRIVATE_INFERENCE_INTERFACE_VERSION, "evaluator_label_inputs": 0},
        "resource_limits": dict(sorted(W09_RESOURCE_BUDGET.items())),
        "wall_dimension_order": list(W09_WALL_DIMENSION_KEYS),
        "thresholds": [{"dimension_key": key, **W09_EVALUATOR_THRESHOLD} for key in W09_ALL_DIMENSION_KEYS],
        "schema_key": evidence_commitment({"ablations": list(W09_ABLATION_KEYS), "dimensions": list(W09_ALL_DIMENSION_KEYS), "family": family_key}),
        "format_version": 1,
        "formal_run_count": 0,
    }
    cases = {
        "artifact_kind": W09_PRIVATE_CASE_KIND,
        "binding_count": len(observations),
        "bindings": [_identity(item) for item in observations],
        "d03_fixed_family_eligible": 0,
        "format_version": 1,
        "formal_run_count": 0,
        "rotation_package_commitment": rotation_commitment,
    }
    labels_doc = {
        "artifact_kind": W09_PRIVATE_LABEL_KIND,
        "binding_count": len(labels),
        "bindings": [_identity(item) for item in labels],
        "d03_fixed_family_eligible": 0,
        "format_version": 1,
        "formal_run_count": 0,
        "rotation_package_commitment": rotation_commitment,
    }
    clusters: dict[str, dict[str, object]] = {}
    for binding in (*observations, *labels):
        cluster = clusters.setdefault(binding.pack_key, {"pack_key": binding.pack_key, "source_cluster_keys": []})
        keys = {tuple(item) for item in cluster["source_cluster_keys"]}
        keys.update(item.components for item in binding.identity.source_cluster_keys)
        cluster["source_cluster_keys"] = [list(item) for item in sorted(keys)]
    cluster_doc = {
        "artifact_kind": W09_PRIVATE_CLUSTER_KIND,
        "clusters": [clusters[key] for key in sorted(clusters)],
        "format_version": 1,
        "formal_run_count": 0,
        "rotation_package_commitment": rotation_commitment,
    }
    source_bytes = canonical_json_bytes(source)
    schema_bytes = canonical_json_bytes(schema)
    case_bytes = canonical_json_bytes(cases)
    label_bytes = canonical_json_bytes(labels_doc)
    cluster_bytes = canonical_json_bytes(cluster_doc)
    payload_commitment = evidence_commitment({
        "cases": _sha256(case_bytes), "clusters": _sha256(cluster_bytes),
        "labels": _sha256(label_bytes), "schema": _sha256(schema_bytes),
        "source": _sha256(source_bytes),
    })
    case_commitment = _sha256(case_bytes)
    label_commitment = _sha256(label_bytes)
    cluster_commitment = _sha256(cluster_bytes)
    if source["rotation_package"]["case_commitment"] != rotation_manifest.rotation_case_commitment:
        raise W09PrivateEvaluationError("rotation case commitment drifted")
    return W09PrivateFamilyDocuments(
        source_bytes, schema_bytes, case_bytes, label_bytes, cluster_bytes,
        family_key, payload_commitment, case_commitment, label_commitment,
        cluster_commitment, values["candidate_contract_sha256"],
        values["candidate_first_run_guard_sha256"], values["candidate_host_freeze_sha256"],
        values["candidate_terminal_seal_sha256"], candidate_head, supplied_head, rotation_sha,
        rotation_commitment, 0, 1,
    )


def validate_w09_private_family_documents(documents: W09PrivateFamilyDocuments) -> None:
    """交叉验证五份文档，确认正式读取前仍为零次运行。"""
    if not isinstance(documents, W09PrivateFamilyDocuments):
        raise W09PrivateEvaluationError("W09 family documents 类型非法")
    source = _decode(documents.source_bytes, W09_PRIVATE_SOURCE_KIND)
    schema = _decode(documents.schema_bytes, W09_PRIVATE_SCHEMA_KIND)
    cases = _decode(documents.case_bytes, W09_PRIVATE_CASE_KIND)
    labels = _decode(documents.label_bytes, W09_PRIVATE_LABEL_KIND)
    _decode(documents.cluster_bytes, W09_PRIVATE_CLUSTER_KIND)
    if (
        source.get("family_key") != documents.family_key
        or source.get("candidate_contract_sha256") != documents.candidate_contract_sha256
        or source.get("candidate_first_run_guard_sha256") != documents.candidate_guard_sha256
        or source.get("candidate_host_freeze_sha256") != documents.candidate_host_sha256
        or source.get("candidate_terminal_seal_sha256") != documents.candidate_seal_sha256
        or source.get("candidate_public_head_commit_sha1") != documents.candidate_public_head_commit_sha1
        or source.get("evaluator_public_head_commit_sha1") != documents.evaluator_public_head_commit_sha1
        or schema.get("dimension_order") != list(W09_ALL_DIMENSION_KEYS)
        or schema.get("ablation_order") != list(W09_ABLATION_KEYS)
        or cases.get("formal_run_count") != 0
        or labels.get("formal_run_count") != 0
        or cases.get("binding_count") != labels.get("binding_count")
        or documents.fixed_d03_exposure_eligible != 0
        or documents.rotation_exposure_audit_clean != 1
        or documents.payload_commitment != _commitment_bundle(documents)
    ):
        raise W09PrivateEvaluationError("W09 private family cross-reference 漂移")


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    """确保 private family 不与 public/Candidate/package root 重叠。"""
    for forbidden in forbidden_roots:
        other = Path(forbidden).resolve()
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise W09PrivateEvaluationError("W09 private root 与其他 owner root 重叠")


def publish_w09_private_family(
    artifact_root: str | Path,
    documents: W09PrivateFamilyDocuments,
    *,
    forbidden_roots: tuple[str | Path, ...] = (),
) -> tuple[Path, str]:
    """排他写五份 metadata 文档和未消费 family freeze。"""
    validate_w09_private_family_documents(documents)
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise W09PrivateEvaluationError("W09 private root 必须全新且不可复用") from error
    identities = []
    for name, payload in documents.files():
        target = root / name
        with target.open("xb") as handle:
            handle.write(payload)
        identities.append({"name": name, "sha256": _sha256(payload), "size_bytes": len(payload)})
    freeze_value = {
        "ablation_order": list(W09_ABLATION_KEYS),
        "artifact_kind": W09_PRIVATE_FAMILY_KIND,
        "bearing_dimension_order": list(W09_DIMENSION_KEYS),
        "case_commitment": documents.case_commitment,
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_first_run_guard_sha256": documents.candidate_guard_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_sha256,
        "candidate_terminal_seal_sha256": documents.candidate_seal_sha256,
        "candidate_public_head_commit_sha1": documents.candidate_public_head_commit_sha1,
        "cluster_commitment": documents.cluster_commitment,
        "dimension_order": list(W09_ALL_DIMENSION_KEYS),
        "documents": identities,
        "evaluator_public_head_commit_sha1": documents.evaluator_public_head_commit_sha1,
        "failure_phases": list(W09_EVALUATOR_FAILURE_PHASES),
        "family_key": documents.family_key,
        "fixed_d03_exposure_eligible": 0,
        "formal_run_count": 0,
        "format_version": 1,
        "label_commitment": documents.label_commitment,
        "payload_commitment": documents.payload_commitment,
        "private_payload_reads": 0,
        "resource_limits": dict(sorted(W09_RESOURCE_BUDGET.items())),
        "rotation_exposure_audit_clean": 1,
        "rotation_manifest_sha256": documents.rotation_manifest_sha256,
        "rotation_package_commitment": documents.rotation_package_commitment,
    }
    payload = canonical_json_bytes(freeze_value)
    freeze_path = root / W09_PRIVATE_FAMILY_FREEZE_NAME
    with freeze_path.open("xb") as handle:
        handle.write(payload)
    return freeze_path, _sha256(payload)


def consume_w09_private_first_run_guard(
    artifact_root: str | Path,
    *,
    family_freeze_sha256: str,
) -> tuple[Path, str]:
    """在任何 private payload read 前排他消费唯一 formal guard。"""
    root = Path(artifact_root).resolve()
    expected = strict_sha256(family_freeze_sha256, label="family freeze")
    freeze = root / W09_PRIVATE_FAMILY_FREEZE_NAME
    if not freeze.is_file() or freeze.is_symlink() or _sha256(freeze.read_bytes()) != expected:
        raise W09PrivateEvaluationError("W09 private family freeze SHA 漂移")
    try:
        value = json.loads(freeze.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W09PrivateEvaluationError("W09 private family freeze 无法读取") from error
    if value.get("formal_run_count") != 0 or value.get("private_payload_reads") != 0:
        raise W09PrivateEvaluationError("W09 private family 已运行或提前读取 payload")
    payload = canonical_json_bytes({
        "artifact_kind": W09_PRIVATE_GUARD_KIND,
        "family_freeze_sha256": expected,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
    })
    target = root / W09_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W09PrivateEvaluationError("W09 private family guard 已消费，不可重跑") from error
    return target, _sha256(payload)


__all__ = [
    "W09PrivateFamilyDocuments",
    "build_w09_private_family_documents",
    "consume_w09_private_first_run_guard",
    "publish_w09_private_family",
    "validate_w09_private_family_documents",
]
