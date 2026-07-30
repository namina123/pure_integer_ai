"""W-03 candidate 冻结后创建 private family 文档、freeze 与唯一运行守卫。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_contract import W03_EVALUATION_ORDER
from pure_integer_ai.experiments.ph2_w03_evaluator_contract import (
    W03_EVALUATOR_FAILURE_PHASES,
    W03_EVALUATOR_PHASES,
    W03_PRIVATE_FAMILY_FREEZE_NAME,
    W03_PRIVATE_FIRST_RUN_GUARD_NAME,
    W03_PRIVATE_OWNER_KEY,
    decode_w03_private_documents,
)


_DOCUMENT_NAMES = (
    "private_source.json",
    "private_schema.json",
    "private_cases.json",
    "private_labels.json",
    "private_clusters.json",
)


def _strict_sha256(value: object, *, label: str) -> str:
    """验证规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"{label} 不是规范 SHA-256")
    return value


def _digest_bytes(*parts: bytes) -> bytes:
    """以长度前缀组合私有 family 派生输入，避免串接歧义。"""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def _digest_hex(*parts: bytes) -> str:
    return _digest_bytes(*parts).hex()


def _nonce_bytes(value: tuple[int, ...]) -> bytes:
    """规范编码独立 owner 在 candidate freeze 后选定的纯整数 nonce。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise RuntimeError("W-03 private family nonce 必须是非空非负整数 tuple")
    return canonical_json_bytes(list(value))


@dataclass(frozen=True)
class W03PrivateFamilyDocuments:
    """五份 owner 分离文档及其不泄漏内容的 commitment。"""

    source_bytes: bytes
    schema_bytes: bytes
    case_bytes: bytes
    label_bytes: bytes
    cluster_bytes: bytes
    family_commitment: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str

    def files(self) -> tuple[tuple[str, bytes], ...]:
        """按 source/schema/case/label/cluster 固定顺序返回文档。"""
        return tuple(zip(_DOCUMENT_NAMES, (
            self.source_bytes,
            self.schema_bytes,
            self.case_bytes,
            self.label_bytes,
            self.cluster_bytes,
        ), strict=True))


def build_w03_private_family_documents(
        *,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        family_nonce: tuple[int, ...],
        ) -> W03PrivateFamilyDocuments:
    """只在 candidate 冻结后由新 nonce 派生未消费 private owner 文档。"""
    contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    host_sha = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host freeze")
    nonce = _nonce_bytes(family_nonce)
    seed = _digest_bytes(
        b"PH2-W03-PRIVATE-FAMILY-V1",
        bytes.fromhex(contract_sha),
        bytes.fromhex(host_sha),
        nonce,
    )
    family_key = _digest_hex(b"family", seed)
    nonce_commitment = hashlib.sha256(nonce).hexdigest()
    source_key = tuple(item + 1 for item in _digest_bytes(b"source", seed))
    source = {
        "artifact_kind": "PH2_W03_PRIVATE_SOURCE",
        "candidate_contract_sha256": contract_sha,
        "candidate_host_freeze_sha256": host_sha,
        "family_key": family_key,
        "format_version": 1,
        "license_id": "CC0-1.0",
        "nonce_commitment": nonce_commitment,
        "owner_key": W03_PRIVATE_OWNER_KEY,
        "source_key": list(source_key),
    }
    schema_core = {
        "artifact_kind": "PH2_W03_PRIVATE_SCHEMA",
        "case_fields": ["case_key", "challenge_key", "dimension_key"],
        "cluster_fields": [
            "case_key", "content_cluster", "schema_cluster",
            "source_cluster", "template_cluster"],
        "evaluation_order": list(W03_EVALUATION_ORDER),
        "failure_phases": list(W03_EVALUATOR_FAILURE_PHASES),
        "fault_registry": list(W03_EVALUATOR_PHASES),
        "format_version": 1,
        "label_fields": [
            "case_key", "dimension_key", "expected_status", "fail_allowed",
            "label_key", "ne_policy", "required"],
    }
    schema = {
        **schema_core,
        "schema_key": hashlib.sha256(canonical_json_bytes(schema_core)).hexdigest(),
    }
    case_rows = []
    label_rows = []
    cluster_rows = []
    for ordinal, dimension in enumerate(W03_EVALUATION_ORDER, start=1):
        dimension_bytes = dimension.encode("utf-8")
        challenge = _digest_bytes(b"challenge", seed, dimension_bytes)
        case_key = _digest_hex(b"case", seed, dimension_bytes)
        label_key = _digest_hex(b"label", seed, dimension_bytes)
        case_rows.append({
            "case_key": case_key,
            "challenge_key": [item + 1 for item in challenge],
            "dimension_key": dimension,
        })
        label_rows.append({
            "case_key": case_key,
            "dimension_key": dimension,
            "expected_status": "PASS",
            "fail_allowed": 0,
            "label_key": label_key,
            "ne_policy": "BLOCK",
            "required": 1,
        })
        cluster_rows.append({
            "case_key": case_key,
            "content_cluster": _digest_hex(
                b"content", seed, ordinal.to_bytes(2, "big")),
            "schema_cluster": _digest_hex(
                b"schema", seed, ordinal.to_bytes(2, "big")),
            "source_cluster": _digest_hex(
                b"source-cluster", seed, ordinal.to_bytes(2, "big")),
            "template_cluster": _digest_hex(
                b"template", seed, ordinal.to_bytes(2, "big")),
        })
    source_bytes = canonical_json_bytes(source)
    schema_bytes = canonical_json_bytes(schema)
    case_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W03_PRIVATE_CASES",
        "cases": case_rows,
        "formal_run_count": 0,
        "format_version": 1,
    })
    label_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W03_PRIVATE_LABELS",
        "format_version": 1,
        "labels": label_rows,
    })
    cluster_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W03_PRIVATE_CLUSTERS",
        "clusters": cluster_rows,
        "format_version": 1,
    })
    payload_commitment = _digest_hex(
        source_bytes, schema_bytes, case_bytes, label_bytes, cluster_bytes)
    documents = W03PrivateFamilyDocuments(
        source_bytes,
        schema_bytes,
        case_bytes,
        label_bytes,
        cluster_bytes,
        _digest_hex(b"family-commitment", seed, bytes.fromhex(payload_commitment)),
        payload_commitment,
        hashlib.sha256(case_bytes).hexdigest(),
        hashlib.sha256(label_bytes).hexdigest(),
        hashlib.sha256(cluster_bytes).hexdigest(),
        contract_sha,
        host_sha,
    )
    decode_w03_private_documents(
        source_bytes, schema_bytes, case_bytes, label_bytes, cluster_bytes)
    return documents


def _file_identity(name: str, payload: bytes) -> dict[str, object]:
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    """要求 private family 与所有公开/candidate root 双向无包含。"""
    for raw in forbidden_roots:
        forbidden = Path(raw).resolve()
        if (root == forbidden or root.is_relative_to(forbidden)
                or forbidden.is_relative_to(root)):
            raise RuntimeError("W-03 private family root 未与公开/candidate owner 隔离")


def publish_w03_private_family(
        artifact_root: str | Path,
        documents: W03PrivateFamilyDocuments,
        *,
        forbidden_roots: tuple[str | Path, ...] = (),
        ) -> tuple[Path, str]:
    """依次排他写五份 owner 文档，并以 family freeze 最后发布。"""
    if not isinstance(documents, W03PrivateFamilyDocuments):
        raise TypeError("W-03 private family documents 类型非法")
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.mkdir(parents=True, exist_ok=True)
    identities = []
    try:
        for name, payload in documents.files():
            target = root / name
            with target.open("xb") as handle:
                handle.write(payload)
            identities.append(_file_identity(name, payload))
    except FileExistsError as exc:
        raise RuntimeError("W-03 private family document 不可覆盖") from exc
    freeze = {
        "artifact_kind": "PH2_W03_PRIVATE_FAMILY_FREEZE",
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_freeze_sha256,
        "case_commitment": documents.case_commitment,
        "cluster_commitment": documents.cluster_commitment,
        "document_inventory": identities,
        "failure_phase_registry": list(W03_EVALUATOR_FAILURE_PHASES),
        "family_commitment": documents.family_commitment,
        "fault_registry": list(W03_EVALUATOR_PHASES),
        "formal_run_count": 0,
        "format_version": 1,
        "label_commitment": documents.label_commitment,
        "payload_commitment": documents.payload_commitment,
        "self_excluded": 1,
    }
    payload = canonical_json_bytes(freeze)
    target = root / W03_PRIVATE_FAMILY_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-03 private family freeze 不可覆盖") from exc
    return target, hashlib.sha256(payload).hexdigest()


def consume_w03_private_first_run_guard(
        artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        ) -> tuple[Path, str]:
    """在 private payload decode 前把 family formal run count 从 0 排他推进到 1。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(
        family_freeze_sha256, label="private family freeze")
    freeze = root / W03_PRIVATE_FAMILY_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or hashlib.sha256(freeze.read_bytes()).hexdigest() != expected):
        raise RuntimeError("W-03 private family freeze identity 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": "PH2_W03_PRIVATE_FIRST_RUN_GUARD",
        "family_freeze_sha256": expected,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
    })
    target = root / W03_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-03 private family first-run 已经消费，不可重跑") from exc
    return target, hashlib.sha256(payload).hexdigest()


__all__ = [
    "W03PrivateFamilyDocuments",
    "build_w03_private_family_documents",
    "consume_w03_private_first_run_guard",
    "publish_w03_private_family",
]
