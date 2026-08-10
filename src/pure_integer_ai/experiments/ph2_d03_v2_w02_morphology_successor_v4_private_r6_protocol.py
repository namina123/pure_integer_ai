"""W02 V4-first R6 的 owner 合同、公共冻结、family root 与一次性 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    build_v2_private_family_registration,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_r6_feasibility_receipt import (
    W02_MORPH_V4_R6_DIMENSION_COUNTS,
    W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH,
    W02_MORPH_V4_R6_SPLIT_COUNTS,
    read_w02_morphology_successor_v4_r6_feasibility_receipt,
)


W02_MORPH_V4_PRIVATE_R6_PROTOCOL_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-PRIVATE-R6-PROTOCOL-V1"
)
W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_private_r6_protocol_v1.json"
)
W02_MORPH_V4_PRIVATE_R6_OWNER_METADATA_VERSION = (
    "PH2-D03-V2-W02-V4-R6-FORMAL-OWNER-METADATA-V1"
)
W02_MORPH_V4_PRIVATE_R6_FAMILY_NAME = (
    "PH2-D03-V2-W02-V4-FIRST-R6-TUECL-TOKEN-SPAN"
)
W02_MORPH_V4_PRIVATE_R6_FAMILY_DOCUMENT = "private-family-freeze.json"
W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT = (
    "private-family-registration.json"
)
W02_MORPH_V4_PRIVATE_R6_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_MORPH_V4_PRIVATE_R6_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_MORPH_V4_PRIVATE_R6_RUN_INTENT = "run-guard/run-intent.json"
W02_MORPH_V4_PRIVATE_R6_EXPOSURE_LEDGER = "exposure-ledger"
W02_MORPH_V4_PRIVATE_R6_SPLITS = ("held_out", "adversarial", "wall")
W02_MORPH_V4_PRIVATE_R6_LAYOUTS = (
    "PRIVATE_SOURCE",
    "PRIVATE_HELD_OUT_OBSERVATION",
    "PRIVATE_ADVERSARIAL_OBSERVATION",
    "PRIVATE_WALL_OBSERVATION",
    "PRIVATE_HELD_OUT_LABEL",
    "PRIVATE_ADVERSARIAL_LABEL",
    "PRIVATE_WALL_LABEL",
)
W02_MORPH_V4_PRIVATE_R6_PATHS = {
    "PRIVATE_SOURCE": "source/source_refs.jsonl.gz",
    "PRIVATE_HELD_OUT_OBSERVATION": "observations/held_out.jsonl.gz",
    "PRIVATE_ADVERSARIAL_OBSERVATION": "observations/adversarial.jsonl.gz",
    "PRIVATE_WALL_OBSERVATION": "observations/wall.jsonl.gz",
    "PRIVATE_HELD_OUT_LABEL": "evaluator/held_out.labels.jsonl.gz",
    "PRIVATE_ADVERSARIAL_LABEL": "evaluator/adversarial.labels.jsonl.gz",
    "PRIVATE_WALL_LABEL": "evaluator/wall.labels.jsonl.gz",
}
W02_MORPH_V4_PRIVATE_R6_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v7.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v4_r6_feasibility_receipt.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v4_private_r6_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v4_private_r6_runtime.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v4_private_r6.py",
)
W02_MORPH_V4_PRIVATE_R6_PARENT_R5_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_r5_family_freeze_v1.json"
)
W02_MORPH_V4_PRIVATE_R6_V4_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_artifact_receipt_v1.json"
)
W02_MORPH_V4_PRIVATE_R6_R5_AGGREGATE_SHA256 = (
    "cc5b25e3c8c9f35fca20efc882638f53c0ab0c80713b50dab2d60c76cb7c80d1"
)
W02_MORPH_V4_PRIVATE_R6_FEASIBILITY_SELECTION_COMMITMENT = (
    "812658e2b8fa1eaf925d4dfd2a101ee80dda2e6c1854fc0c4238cf200504cf3a"
)
W02_MORPH_V4_PRIVATE_R6_RESOURCE_BUDGET = V2EvaluatorResourceBudget(
    512, 9_000_000, 536_870_912, 300_000, 100_000, 4)


# object-model: exception
class W02MorphologySuccessorV4PrivateR6ProtocolError(RuntimeError):
    """R6 owner、公共冻结、registration 或一次性 guard 漂移。"""


def _sha256(value: object) -> str:
    """计算 canonical JSON 值的 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    """要求小写 SHA-256 文本。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            f"{where} 非小写 SHA-256")
    return value


def _require_positive(value: object, *, where: str) -> int:
    """要求严格正整数。"""
    if type(value) is not int or value <= 0:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            f"{where} 非正整数")
    return value


def _require_dict(
        value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    """要求 object 字段集合精确一致。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            f"{where} 字段漂移")
    return value


def _repository_file(root: Path, relative: str) -> Path:
    """解析仓库内普通文件并拒绝路径逃逸。"""
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 公开依赖路径非法")
    return target


def _file_identity(path: Path) -> tuple[int, str]:
    """流式计算普通文件的长度与 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4PrivateR6FileIdentity:
    """R6 标准七文件中一个 gzip 的公开身份。"""

    layout_key: str
    record_kind: str
    split: str
    record_count: int
    content_size_bytes: int
    content_sha256: str
    transport_size_bytes: int
    transport_sha256: str
    first_record_key: tuple[int, ...]
    last_record_key: tuple[int, ...]
    license_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.layout_key not in W02_MORPH_V4_PRIVATE_R6_PATHS:
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 文件 layout 未注册")
        if self.layout_key == "PRIVATE_SOURCE":
            expected = ("source_ref", "")
        else:
            kind = ("observation" if self.layout_key.endswith("OBSERVATION")
                    else "evaluator_label")
            split = self.layout_key.removeprefix("PRIVATE_").removesuffix(
                "_OBSERVATION").removesuffix("_LABEL").lower()
            expected = (kind, split)
        if (self.record_kind, self.split) != expected:
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 文件 kind/split 漂移")
        for name in (
                "record_count", "content_size_bytes", "transport_size_bytes"):
            _require_positive(getattr(self, name), where=f"R6 {name}")
        _require_sha256(self.content_sha256, where="R6 content")
        _require_sha256(self.transport_sha256, where="R6 transport")
        for key in (self.first_record_key, self.last_record_key):
            if (not key or any(type(item) is not int or item <= 0
                               for item in key)):
                raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                    "R6 stable key 范围非法")
        if (self.first_record_key > self.last_record_key
                or self.license_ids != ("CC-BY-SA-4.0",)):
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 文件范围或许可漂移")

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "W02MorphologySuccessorV4PrivateR6FileIdentity":
        """从 safe owner metadata 恢复一个文件身份。"""
        raw = _require_dict(value, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "relative_path", "split", "transport_sha256",
            "transport_size_bytes",
        }, where="R6 文件身份")
        if raw["relative_path"] != W02_MORPH_V4_PRIVATE_R6_PATHS.get(
                raw["layout_key"]):
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 文件相对路径漂移")
        if (not isinstance(raw["first_record_key"], list)
                or not isinstance(raw["last_record_key"], list)
                or not isinstance(raw["license_ids"], list)):
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 文件数组字段漂移")
        return cls(
            str(raw["layout_key"]), str(raw["record_kind"]), str(raw["split"]),
            raw["record_count"], raw["content_size_bytes"],
            str(raw["content_sha256"]), raw["transport_size_bytes"],
            str(raw["transport_sha256"]), tuple(raw["first_record_key"]),
            tuple(raw["last_record_key"]), tuple(raw["license_ids"]),
        )

    def to_dict(self) -> dict[str, object]:
        """返回不含物理 root 的安全文件身份。"""
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "first_record_key": list(self.first_record_key),
            "last_record_key": list(self.last_record_key),
            "layout_key": self.layout_key,
            "license_ids": list(self.license_ids),
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "relative_path": W02_MORPH_V4_PRIVATE_R6_PATHS[self.layout_key],
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4PrivateR6OwnerReceipt:
    """formal owner 的 safe metadata 投影与最终 payload commitments。"""

    owner_id: str
    owner_family_key: str
    files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...]
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        if (not self.owner_id or not self.owner_family_key
                or tuple(row.layout_key for row in self.files)
                != W02_MORPH_V4_PRIVATE_R6_LAYOUTS):
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 owner receipt 身份或 inventory 漂移")
        for value in (
                self.payload_commitment, self.case_commitment,
                self.label_commitment, self.cluster_commitment,
                self.metadata_sha256):
            _require_sha256(value, where="R6 owner commitment")


def _validate_file_inventory(
        value: object,
        ) -> tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...]:
    """闭合七文件顺序、计数与 observation/label 配对。"""
    if not isinstance(value, list) or len(value) != 7:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 owner inventory 不完整")
    files = tuple(
        W02MorphologySuccessorV4PrivateR6FileIdentity.from_dict(row)
        for row in value)
    if tuple(row.layout_key for row in files) != W02_MORPH_V4_PRIVATE_R6_LAYOUTS:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 owner inventory 顺序漂移")
    by_layout = {row.layout_key: row for row in files}
    if by_layout["PRIVATE_SOURCE"].record_count != 500:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 SourceRef 数量漂移")
    for split in W02_MORPH_V4_PRIVATE_R6_SPLITS:
        name = split.upper()
        expected = W02_MORPH_V4_R6_SPLIT_COUNTS[split]
        if (by_layout[f"PRIVATE_{name}_OBSERVATION"].record_count != expected
                or by_layout[f"PRIVATE_{name}_LABEL"].record_count != expected):
            raise W02MorphologySuccessorV4PrivateR6ProtocolError(
                "R6 split pair 数量漂移")
    return files


def read_w02_morphology_successor_v4_private_r6_owner_metadata(
        path: str | Path) -> W02MorphologySuccessorV4PrivateR6OwnerReceipt:
    """只读 formal owner safe metadata，绝不打开七个 gzip。"""
    target = Path(path).resolve()
    size, metadata_sha = _file_identity(target)
    if size <= 0:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 owner metadata 为空")
    raw = _require_dict(read_canonical_object(target), {
        "artifact_kind", "artifact_version", "candidate_calls",
        "case_selection_commitment", "commitments", "dimension_counts",
        "double_pass_equal", "feasibility_case_transport_reads",
        "feasibility_metadata_sha256", "file_inventory",
        "formal_private_evaluation_runs", "label_imputation_count",
        "main_session_private_payload_reads", "old_owner_payload_reads",
        "owner_family_key", "owner_id", "pair_count", "source_count",
        "source_key", "split_counts", "status", "teacher_calls",
        "v1_v2_v3_v4_calls",
    }, where="R6 formal owner metadata")
    commitments = _require_dict(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="R6 owner commitments")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_V4_R6_FORMAL_OWNER_METADATA"
            or raw["artifact_version"]
            != W02_MORPH_V4_PRIVATE_R6_OWNER_METADATA_VERSION
            or raw["status"] != "R6_FORMAL_OWNER_FROZEN"
            or raw["source_key"]
            != "UD_LZH_TUECL_R2_18_TOKEN_SPAN_BLIND_PRIVATE"
            or raw["case_selection_commitment"]
            != W02_MORPH_V4_PRIVATE_R6_FEASIBILITY_SELECTION_COMMITMENT
            or raw["feasibility_metadata_sha256"]
            != "5c027695c9c46e3763a1a91dc6ff126b69730c670650459b0f8a0e4c4f4c37e8"
            or raw["source_count"] != 500 or raw["pair_count"] != 500
            or raw["dimension_counts"] != W02_MORPH_V4_R6_DIMENSION_COUNTS
            or raw["split_counts"] != W02_MORPH_V4_R6_SPLIT_COUNTS
            or raw["double_pass_equal"] != 1
            or raw["label_imputation_count"] != 0
            or raw["feasibility_case_transport_reads"] != 1
            or raw["main_session_private_payload_reads"] != 0
            or raw["old_owner_payload_reads"] != 0
            or raw["formal_private_evaluation_runs"] != 0
            or any(raw[name] != 0 for name in (
                "candidate_calls", "teacher_calls", "v1_v2_v3_v4_calls"))):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 formal owner 状态漂移")
    files = _validate_file_inventory(raw["file_inventory"])
    return W02MorphologySuccessorV4PrivateR6OwnerReceipt(
        str(raw["owner_id"]), str(raw["owner_family_key"]), files,
        _require_sha256(commitments["payload_commitment"], where="payload"),
        _require_sha256(commitments["case_commitment"], where="case"),
        _require_sha256(commitments["label_commitment"], where="label"),
        _require_sha256(commitments["cluster_commitment"], where="cluster"),
        metadata_sha,
    )


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结 R6 公共协议、runtime 与 runner 的代码身份。"""
    rows = []
    for relative in W02_MORPH_V4_PRIVATE_R6_CODE_PATHS:
        path = _repository_file(repository, relative)
        size, digest = _file_identity(path)
        rows.append({
            "repository_file": relative,
            "sha256": digest,
            "size_bytes": size,
        })
    return rows, _sha256(rows)


def build_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository_root: str | Path) -> dict[str, object]:
    """冻结能力阈值、代码、V4 artifact、V7 与 R5 FAIL lineage。"""
    repository = Path(repository_root).resolve()
    feasibility = read_w02_morphology_successor_v4_r6_feasibility_receipt(
        repository)
    r5_path = _repository_file(
        repository, W02_MORPH_V4_PRIVATE_R6_PARENT_R5_FREEZE_PATH)
    v4_path = _repository_file(repository, W02_MORPH_V4_PRIVATE_R6_V4_RECEIPT_PATH)
    feasibility_path = _repository_file(
        repository, W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH)
    r5_size, r5_sha = _file_identity(r5_path)
    v4_size, v4_sha = _file_identity(v4_path)
    feasibility_size, feasibility_sha = _file_identity(feasibility_path)
    r5 = read_canonical_object(r5_path)
    v4 = read_canonical_object(v4_path)
    if (r5.get("status")
            != "W02_SUCCESSOR_V3_R5_BLIND_PRIVATE_FAMILY_FROZEN"
            or r5.get("formal_private_evaluation_runs") != 0
            or v4.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V4_PUBLIC_ARTIFACT_FROZEN"):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 parent public evidence 漂移")
    code_rows, code_sha = _code_rows(repository)
    artifact_chain = {
        "feasibility_receipt_file_sha256": feasibility_sha,
        "r5_capability_fail_aggregate_sha256":
            W02_MORPH_V4_PRIVATE_R6_R5_AGGREGATE_SHA256,
        "r5_public_family_freeze_file_sha256": r5_sha,
        "r5_public_family_freeze_internal_artifact_chain_sha256":
            r5["artifact_chain_sha256"],
        "v4_artifact_receipt_file_sha256": v4_sha,
        "v4_artifact_semantic_sha256":
            v4["git_external_artifact"]["semantic_sha256"],
        "v4_artifact_tree_commitment":
            v4["git_external_artifact"]["tree_commitment"],
        "v7_source_manifest_sha256": feasibility["v7_source_manifest_sha256"],
    }
    return {
        "artifact_chain": artifact_chain,
        "artifact_chain_sha256": _sha256(artifact_chain),
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V4_PRIVATE_R6_PROTOCOL"),
        "artifact_version": W02_MORPH_V4_PRIVATE_R6_PROTOCOL_VERSION,
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "dimension_denominator_counts": dict(W02_MORPH_V4_R6_DIMENSION_COUNTS),
        "feasibility_case_reuse_as_formal_payload_authorized": 0,
        "formal_owner_required": 1,
        "formal_private_evaluation_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "main_session_tuecl_content_reads": 0,
        "ne_policy": "BLOCK",
        "old_r2_r3_r4_r5_owner_family_payload_reuse": 0,
        "parent_evidence_files": [
            {"repository_file": W02_MORPH_V4_PRIVATE_R6_PARENT_R5_FREEZE_PATH,
             "sha256": r5_sha, "size_bytes": r5_size},
            {"repository_file": W02_MORPH_V4_PRIVATE_R6_V4_RECEIPT_PATH,
             "sha256": v4_sha, "size_bytes": v4_size},
            {"repository_file": W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH,
             "sha256": feasibility_sha, "size_bytes": feasibility_size},
        ],
        "private_payload_reads": 0,
        "resource_budget": W02_MORPH_V4_PRIVATE_R6_RESOURCE_BUDGET.to_dict(),
        "source_count": 500,
        "split_counts": dict(W02_MORPH_V4_R6_SPLIT_COUNTS),
        "stage_key": "W-02",
        "status": "W02_V4_R6_PUBLIC_PROTOCOL_FROZEN_FORMAL_OWNER_PENDING",
        "teacher_calls": 0,
        "threshold_reduction": 0,
        "zero_write_required": 1,
    }


def publish_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository_root: str | Path,
        path: str | Path | None = None,
        ) -> Path:
    """排他或幂等发布 R6 公共 protocol freeze。"""
    repository = Path(repository_root).resolve()
    target = (
        repository / Path(*PurePosixPath(
            W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH).parts)
        if path is None else Path(path).resolve())
    write_immutable_json(
        build_w02_morphology_successor_v4_private_r6_protocol_freeze(
            repository), target)
    read_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository, target)
    return target


def read_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository_root: str | Path,
        path: str | Path | None = None,
        ) -> dict[str, object]:
    """回读并重建 R6 公共 protocol freeze。"""
    repository = Path(repository_root).resolve()
    target = (
        _repository_file(repository, W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH)
        if path is None else Path(path).resolve())
    value = read_canonical_object(target)
    expected = build_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository)
    if value != expected:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 public protocol freeze 漂移")
    return value


def _family_root(value: str | Path, *, require_exists: bool) -> Path:
    """解析一个 Git 外 family root，拒绝链接与覆盖。"""
    original = Path(value)
    if original.is_symlink():
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 family root 不得为链接")
    root = original.resolve()
    if require_exists and not root.is_dir():
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 family root 不存在")
    if not require_exists and root.exists():
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 family root 必须全新")
    return root


def _guard_value(
        protocol: dict[str, object],
        registration: V2PrivateFamilyRegistration,
        owner: W02MorphologySuccessorV4PrivateR6OwnerReceipt,
        *, state: str,
        ) -> dict[str, object]:
    """构造与 owner、代码和 artifact 全绑定的一次性 guard。"""
    return {
        "artifact_kind": "PH2_D03_V2_W02_V4_R6_ONE_SHOT_GUARD",
        "artifact_version": W02_MORPH_V4_PRIVATE_R6_PROTOCOL_VERSION,
        "family_commitment": registration.family_commitment,
        "formal_run_count_before": 0,
        "owner_metadata_sha256": owner.metadata_sha256,
        "private_payload_reads_before": 0,
        "protocol_artifact_chain_sha256": protocol["artifact_chain_sha256"],
        "protocol_code_freeze_sha256": protocol["code_freeze_sha256"],
        "run_id": 1,
        "state": state,
    }


def publish_w02_morphology_successor_v4_private_r6_family_root(
        repository_root: str | Path,
        family_root: str | Path,
        owner_metadata_path: str | Path,
        ) -> Path:
    """在公共冻结后发布全新 family registration 与唯一 available guard。"""
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=False)
    protocol = read_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository)
    owner = read_w02_morphology_successor_v4_private_r6_owner_metadata(
        owner_metadata_path)
    registration = build_v2_private_family_registration(
        "W-02",
        payload_commitment=owner.payload_commitment,
        case_commitment=owner.case_commitment,
        label_commitment=owner.label_commitment,
        cluster_commitment=owner.cluster_commitment,
        candidate_freeze_sha256=str(protocol["artifact_chain_sha256"]),
        code_freeze_sha256=str(protocol["code_freeze_sha256"]),
        resource_budget=W02_MORPH_V4_PRIVATE_R6_RESOURCE_BUDGET,
    )
    root.mkdir(parents=True)
    write_immutable_json(protocol, root / W02_MORPH_V4_PRIVATE_R6_FAMILY_DOCUMENT)
    write_immutable_json(
        registration.to_dict(),
        root / W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT)
    write_immutable_json(
        _guard_value(protocol, registration, owner, state="AVAILABLE"),
        root / W02_MORPH_V4_PRIVATE_R6_GUARD_AVAILABLE)
    (root / W02_MORPH_V4_PRIVATE_R6_EXPOSURE_LEDGER).mkdir()
    return root


def read_w02_morphology_successor_v4_private_r6_registration(
        family_root: str | Path) -> V2PrivateFamilyRegistration:
    """严格恢复 family root 内的零读 registration。"""
    root = _family_root(family_root, require_exists=True)
    raw = read_canonical_object(
        root / W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT)
    policy = raw.get("policy") if isinstance(raw, dict) else None
    budget = raw.get("resource_budget") if isinstance(raw, dict) else None
    if not isinstance(policy, dict) or not isinstance(budget, dict):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 registration 字段缺失")
    registration = build_v2_private_family_registration(
        "W-02",
        payload_commitment=str(raw["payload_commitment"]),
        case_commitment=str(raw["case_commitment"]),
        label_commitment=str(raw["label_commitment"]),
        cluster_commitment=str(raw["cluster_commitment"]),
        candidate_freeze_sha256=str(raw["candidate_freeze_sha256"]),
        code_freeze_sha256=str(raw["code_freeze_sha256"]),
        resource_budget=V2EvaluatorResourceBudget.from_dict(budget),
    )
    if registration.to_dict() != raw:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 registration 漂移")
    return registration


def consume_w02_morphology_successor_v4_private_r6_guard(
        repository_root: str | Path,
        family_root: str | Path,
        owner_metadata_path: str | Path,
        *, run_id: int = 1,
        ) -> dict[str, object]:
    """原子消费唯一 available guard，并写入不可覆盖 run intent。"""
    if run_id != 1:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 formal run_id 必须为 1")
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=True)
    protocol = read_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository)
    registration = read_w02_morphology_successor_v4_private_r6_registration(root)
    owner = read_w02_morphology_successor_v4_private_r6_owner_metadata(
        owner_metadata_path)
    available = root / W02_MORPH_V4_PRIVATE_R6_GUARD_AVAILABLE
    consumed = root / W02_MORPH_V4_PRIVATE_R6_GUARD_CONSUMED
    intent = root / W02_MORPH_V4_PRIVATE_R6_RUN_INTENT
    if not available.is_file() or consumed.exists() or intent.exists():
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 guard 已消费或状态不完整")
    expected = _guard_value(protocol, registration, owner, state="AVAILABLE")
    if read_canonical_object(available) != expected:
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 available guard 漂移")
    consumed_value = dict(expected)
    consumed_value["state"] = "CONSUMED"
    write_immutable_json(consumed_value, consumed)
    available.unlink()
    intent_value = {
        "artifact_kind": "PH2_D03_V2_W02_V4_R6_RUN_INTENT",
        "consumed_guard_sha256": _sha256(consumed_value),
        "family_commitment": registration.family_commitment,
        "run_id": 1,
        "state": "FORMAL_RUN_INTENT_FROZEN",
    }
    write_immutable_json(intent_value, intent)
    return intent_value


def verify_w02_morphology_successor_v4_private_r6_consumed_guard(
        family_root: str | Path) -> dict[str, object]:
    """回读已消费 guard 与 run intent，拒绝重复发布。"""
    root = _family_root(family_root, require_exists=True)
    if (root / W02_MORPH_V4_PRIVATE_R6_GUARD_AVAILABLE).exists():
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 available guard 不应残留")
    consumed = read_canonical_object(
        root / W02_MORPH_V4_PRIVATE_R6_GUARD_CONSUMED)
    intent = read_canonical_object(root / W02_MORPH_V4_PRIVATE_R6_RUN_INTENT)
    if (consumed.get("state") != "CONSUMED"
            or intent.get("state") != "FORMAL_RUN_INTENT_FROZEN"
            or intent.get("run_id") != 1
            or intent.get("family_commitment")
            != consumed.get("family_commitment")
            or intent.get("consumed_guard_sha256") != _sha256(consumed)):
        raise W02MorphologySuccessorV4PrivateR6ProtocolError(
            "R6 consumed guard 或 intent 漂移")
    return intent


__all__ = [
    "W02_MORPH_V4_PRIVATE_R6_EXPOSURE_LEDGER",
    "W02_MORPH_V4_PRIVATE_R6_FAMILY_DOCUMENT",
    "W02_MORPH_V4_PRIVATE_R6_FAMILY_NAME",
    "W02_MORPH_V4_PRIVATE_R6_GUARD_AVAILABLE",
    "W02_MORPH_V4_PRIVATE_R6_GUARD_CONSUMED",
    "W02_MORPH_V4_PRIVATE_R6_LAYOUTS",
    "W02_MORPH_V4_PRIVATE_R6_OWNER_METADATA_VERSION",
    "W02_MORPH_V4_PRIVATE_R6_PATHS",
    "W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH",
    "W02_MORPH_V4_PRIVATE_R6_PROTOCOL_VERSION",
    "W02_MORPH_V4_PRIVATE_R6_REGISTRATION_DOCUMENT",
    "W02_MORPH_V4_PRIVATE_R6_RESOURCE_BUDGET",
    "W02_MORPH_V4_PRIVATE_R6_RUN_INTENT",
    "W02_MORPH_V4_PRIVATE_R6_SPLITS",
    "W02MorphologySuccessorV4PrivateR6FileIdentity",
    "W02MorphologySuccessorV4PrivateR6OwnerReceipt",
    "W02MorphologySuccessorV4PrivateR6ProtocolError",
    "build_w02_morphology_successor_v4_private_r6_protocol_freeze",
    "consume_w02_morphology_successor_v4_private_r6_guard",
    "publish_w02_morphology_successor_v4_private_r6_family_root",
    "publish_w02_morphology_successor_v4_private_r6_protocol_freeze",
    "read_w02_morphology_successor_v4_private_r6_owner_metadata",
    "read_w02_morphology_successor_v4_private_r6_protocol_freeze",
    "read_w02_morphology_successor_v4_private_r6_registration",
    "verify_w02_morphology_successor_v4_private_r6_consumed_guard",
]
