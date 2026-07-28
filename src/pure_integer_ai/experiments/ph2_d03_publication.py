"""D-03 内容提交后的远端核验、CI 和 post-publication receipt 合同。"""
from __future__ import annotations

import hashlib
import gzip
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    D03PublicationState,
    FORMAT_VERSION,
    exact_dict,
    flag,
    nonnegative,
    positive,
    read_canonical_object,
    sha1_text,
    string_tuple,
    text,
    validate_zero_execution_state,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
    HISTORICAL_HOLD_RECEIPT_PATH,
    RELEASE_KEY,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
)


RECEIPT_ARTIFACT_KIND = "PH2_D03_POST_PUBLICATION_RECEIPT"
RECEIPT_ARTIFACT_VERSION = "PH2-D03-post-publication-receipt-v1"
RECEIPT_STATUS = "POST_PUBLISH_VERIFIED"
REQUIRED_CI_JOB_NAMES = (
    "Python 3.11 on ubuntu-latest",
    "Python 3.14 on ubuntu-latest",
    "Python 3.14 on windows-latest",
    "Secret scan",
)


@dataclass(frozen=True)
class D03PublicationInventoryScan:
    """冻结 release inventory 的可扫描范围和发现计数。"""

    scanned_path_count: int
    gzip_jsonl_path_count: int
    legacy_finding_count: int
    secret_finding_count: int
    binary_path_count: int
    unreadable_path_count: int

    def __post_init__(self) -> None:
        positive(self.scanned_path_count, where="publication scanned path count")
        for name in (
                "gzip_jsonl_path_count", "legacy_finding_count",
                "secret_finding_count", "binary_path_count",
                "unreadable_path_count"):
            nonnegative(getattr(self, name), where=f"publication {name}")
        if self.gzip_jsonl_path_count > self.scanned_path_count:
            raise D03ContractError("publication gzip JSONL 计数超过 inventory")


def _inventory_payload(
        repository: Path,
        identity: D03FileIdentity,
        ) -> bytes:
    """在同一次读取中核对安全路径、大小和摘要。"""
    target = (repository / Path(
        *PurePosixPath(identity.relative_path).parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise D03ContractError("publication inventory 路径缺失或逃逸")
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise D03ContractError("publication inventory 无法读取") from error
    if (len(payload) != identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != identity.sha256):
        raise D03ContractError("publication inventory 文件身份漂移")
    return payload


def _canonical_gzip_jsonl_text(payload: bytes) -> str:
    """解压并逐行要求 UTF-8 canonical JSON object。"""
    try:
        decoded = gzip.decompress(payload)
    except (EOFError, OSError) as error:
        raise D03ContractError("publication gzip JSONL 损坏") from error
    if decoded and not decoded.endswith(b"\n"):
        raise D03ContractError("publication gzip JSONL 缺规范换行")
    for line in decoded.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise D03ContractError("publication gzip JSONL 行损坏")
        try:
            parse_canonical_json_bytes(line[:-1], require_object=True)
        except Exception as error:
            raise D03ContractError("publication gzip JSONL 非规范") from error
    try:
        return decoded.decode("utf-8")
    except UnicodeError as error:
        raise D03ContractError("publication gzip JSONL 非 UTF-8") from error


def _rule_finding_count(
        value: str,
        rules: tuple[tuple[str, re.Pattern[str]], ...],
        *,
        ignore_case: bool,
        ) -> int:
    """逐行逐规则计数，不复制潜在敏感命中内容。"""
    compiled = tuple(
        re.compile(pattern.pattern, pattern.flags | re.IGNORECASE)
        if ignore_case else pattern
        for _, pattern in rules
    )
    return sum(
        pattern.search(line) is not None
        for line in value.splitlines()
        for pattern in compiled
    )


def scan_d03_publication_inventory(
        repository_root: str | Path,
        inventory: tuple[D03FileIdentity, ...],
        ) -> D03PublicationInventoryScan:
    """核验并扫描 release 文件；规范 ``.jsonl.gz`` 不视为 binary。"""
    if (not isinstance(inventory, tuple) or not inventory
            or any(not isinstance(item, D03FileIdentity)
                   for item in inventory)):
        raise D03ContractError("publication scan inventory 非法")
    paths = tuple(item.relative_path for item in inventory)
    if len(paths) != len(set(paths)):
        raise D03ContractError("publication scan inventory 路径重复")
    repository = Path(repository_root).resolve()
    legacy_count = 0
    secret_count = 0
    gzip_count = 0
    binary_count = 0
    unreadable_count = 0
    for identity in sorted(inventory):
        payload = _inventory_payload(repository, identity)
        legacy_count += _rule_finding_count(
            identity.relative_path, LEGACY_RULES, ignore_case=True)
        if identity.relative_path.endswith(".jsonl.gz"):
            gzip_count += 1
            try:
                content = _canonical_gzip_jsonl_text(payload)
            except D03ContractError:
                unreadable_count += 1
                continue
        else:
            try:
                content = payload.decode("utf-8")
            except UnicodeError:
                binary_count += 1
                continue
        legacy_count += _rule_finding_count(
            content, LEGACY_RULES, ignore_case=True)
        secret_count += _rule_finding_count(
            content, SECRET_RULES, ignore_case=False)
    return D03PublicationInventoryScan(
        len(inventory), gzip_count, legacy_count, secret_count,
        binary_count, unreadable_count,
    )


@dataclass(frozen=True, order=True)
class GitHubCIJob:
    """冻结一个 content commit 上完成的 GitHub Actions job 结论。"""

    name: str
    conclusion: str

    def __post_init__(self) -> None:
        text(self.name, where="CI job name")
        if self.conclusion != "success":
            raise D03ContractError("D-03 content CI job 未成功")

    def to_dict(self) -> dict[str, str]:
        """导出 CI job 结论。"""
        return {"conclusion": self.conclusion, "name": self.name}

    @classmethod
    def from_dict(cls, value: Any) -> "GitHubCIJob":
        """从严格 object 恢复 CI job 结论。"""
        raw = exact_dict(value, {"conclusion", "name"}, where="GitHubCIJob")
        return cls(str(raw["name"]), str(raw["conclusion"]))


@dataclass(frozen=True)
class D03PostPublishGate:
    """合取隔离 clone、公开/secret/许可/论文/零执行和 content CI。"""

    legacy_finding_count: int
    secret_finding_count: int
    binary_path_count: int
    unreadable_path_count: int
    github_secret_alert_count: int
    license_gate_passed: int
    paper_gate_passed: int
    zero_execution_gate_passed: int
    remote_reachable: int
    isolated_clone_verified: int
    content_ci_run_id: int
    content_ci_head_sha1: str
    content_ci_jobs: tuple[GitHubCIJob, ...]

    def __post_init__(self) -> None:
        counts = (
            self.legacy_finding_count,
            self.secret_finding_count,
            self.binary_path_count,
            self.unreadable_path_count,
            self.github_secret_alert_count,
        )
        for name, value in zip((
                "legacy", "secret", "binary", "unreadable",
                "GitHub secret alert"), counts):
            nonnegative(value, where=f"post-publish {name} count")
        if any(counts):
            raise D03ContractError("post-publish scan/gate 发现公开或 secret 问题")
        for name in (
                "license_gate_passed", "paper_gate_passed",
                "zero_execution_gate_passed", "remote_reachable",
                "isolated_clone_verified"):
            flag(getattr(self, name), where=name)
            if getattr(self, name) != 1:
                raise D03ContractError("post-publish gate 未全部通过")
        positive(self.content_ci_run_id, where="content CI run id")
        object.__setattr__(self, "content_ci_head_sha1", sha1_text(
            self.content_ci_head_sha1, where="content CI head"))
        if (not isinstance(self.content_ci_jobs, tuple)
                or any(not isinstance(item, GitHubCIJob)
                       for item in self.content_ci_jobs)):
            raise D03ContractError("content CI jobs 类型非法")
        jobs = tuple(sorted(self.content_ci_jobs))
        if tuple(item.name for item in jobs) != REQUIRED_CI_JOB_NAMES:
            raise D03ContractError("content CI 未覆盖 Linux/Windows/secret scan")
        object.__setattr__(self, "content_ci_jobs", jobs)

    def to_dict(self) -> dict[str, Any]:
        """导出发布后门禁。"""
        return {
            "binary_path_count": self.binary_path_count,
            "content_ci_head_sha1": self.content_ci_head_sha1,
            "content_ci_jobs": [item.to_dict() for item in self.content_ci_jobs],
            "content_ci_run_id": self.content_ci_run_id,
            "github_secret_alert_count": self.github_secret_alert_count,
            "isolated_clone_verified": self.isolated_clone_verified,
            "legacy_finding_count": self.legacy_finding_count,
            "license_gate_passed": self.license_gate_passed,
            "paper_gate_passed": self.paper_gate_passed,
            "remote_reachable": self.remote_reachable,
            "secret_finding_count": self.secret_finding_count,
            "unreadable_path_count": self.unreadable_path_count,
            "zero_execution_gate_passed": self.zero_execution_gate_passed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03PostPublishGate":
        """从严格 object 恢复发布后门禁。"""
        raw = exact_dict(value, {
            "binary_path_count", "content_ci_head_sha1", "content_ci_jobs",
            "content_ci_run_id", "github_secret_alert_count",
            "isolated_clone_verified", "legacy_finding_count",
            "license_gate_passed", "paper_gate_passed", "remote_reachable",
            "secret_finding_count", "unreadable_path_count",
            "zero_execution_gate_passed",
        }, where="D03PostPublishGate")
        if not isinstance(raw["content_ci_jobs"], list):
            raise D03ContractError("content CI jobs 必须是数组")
        return cls(
            raw["legacy_finding_count"], raw["secret_finding_count"],
            raw["binary_path_count"], raw["unreadable_path_count"],
            raw["github_secret_alert_count"], raw["license_gate_passed"],
            raw["paper_gate_passed"], raw["zero_execution_gate_passed"],
            raw["remote_reachable"], raw["isolated_clone_verified"],
            raw["content_ci_run_id"], str(raw["content_ci_head_sha1"]),
            tuple(GitHubCIJob.from_dict(item) for item in raw["content_ci_jobs"]),
        )


@dataclass(frozen=True)
class D03PublicationReceipt:
    """绑定内容提交、远端、inventory、发布后门禁和最终 D-03 状态。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    status: str
    publication_state: D03PublicationState
    remote_name: str
    remote_branch: str
    remote_ref: str
    content_parent_sha1: str
    content_commit_sha1: str
    global_manifest_identity: D03FileIdentity
    release_inventory: tuple[D03FileIdentity, ...]
    historical_hold_receipt_identity: D03FileIdentity
    post_publish_gate: D03PostPublishGate
    receipt_relative_path: str
    receipt_self_excluded: int
    execution_state: dict[str, int]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise D03ContractError("receipt format_version 非法")
        if self.artifact_kind != RECEIPT_ARTIFACT_KIND:
            raise D03ContractError("receipt artifact_kind 非法")
        if self.artifact_version != RECEIPT_ARTIFACT_VERSION:
            raise D03ContractError("receipt artifact_version 非法")
        if self.release_key != RELEASE_KEY or self.status != RECEIPT_STATUS:
            raise D03ContractError("receipt release/status 非法")
        if (not isinstance(self.publication_state, D03PublicationState)
                or self.publication_state.state != "POST_PUBLISH_VERIFIED"):
            raise D03ContractError("receipt publication state 非法")
        if (self.remote_name, self.remote_branch, self.remote_ref) != (
                "origin", "master", "refs/heads/master"):
            raise D03ContractError("receipt remote identity 非法")
        object.__setattr__(self, "content_parent_sha1", sha1_text(
            self.content_parent_sha1, where="content parent"))
        object.__setattr__(self, "content_commit_sha1", sha1_text(
            self.content_commit_sha1, where="content commit"))
        if (self.content_parent_sha1 == self.content_commit_sha1
                or self.publication_state.content_commit_sha1
                != self.content_commit_sha1):
            raise D03ContractError("receipt content commit 与 publication state 不一致")
        if not isinstance(self.global_manifest_identity, D03FileIdentity):
            raise D03ContractError("receipt global manifest identity 非法")
        if (self.global_manifest_identity.relative_path
                != FORMAL_GLOBAL_MANIFEST_PATH):
            raise D03ContractError("receipt global manifest path 漂移")
        if (not isinstance(self.release_inventory, tuple)
                or not self.release_inventory
                or any(not isinstance(item, D03FileIdentity)
                       for item in self.release_inventory)):
            raise D03ContractError("receipt release inventory 不能为空")
        inventory = tuple(sorted(self.release_inventory))
        paths = tuple(item.relative_path for item in inventory)
        if len(paths) != len(set(paths)):
            raise D03ContractError("receipt release inventory 路径重复")
        if self.receipt_relative_path in paths:
            raise D03ContractError("receipt 必须 self-excluded，不能进入 content inventory")
        matches = [
            item for item in inventory
            if item.relative_path == self.global_manifest_identity.relative_path
        ]
        if matches != [self.global_manifest_identity]:
            raise D03ContractError("receipt global manifest 未精确进入 inventory")
        object.__setattr__(self, "release_inventory", inventory)
        if (not isinstance(self.historical_hold_receipt_identity, D03FileIdentity)
                or self.historical_hold_receipt_identity.relative_path
                != HISTORICAL_HOLD_RECEIPT_PATH):
            raise D03ContractError("receipt historical hold identity 漂移")
        if (not isinstance(self.post_publish_gate, D03PostPublishGate)
                or self.post_publish_gate.content_ci_head_sha1
                != self.content_commit_sha1):
            raise D03ContractError("receipt content CI head 与 content commit 不一致")
        if self.receipt_relative_path != FORMAL_RECEIPT_PATH:
            raise D03ContractError("receipt relative path 漂移")
        flag(self.receipt_self_excluded, where="receipt self-excluded")
        if self.receipt_self_excluded != 1:
            raise D03ContractError("receipt 必须 self-excluded")
        object.__setattr__(self, "execution_state", validate_zero_execution_state(
            self.execution_state, d03_published=1))

    def to_dict(self) -> dict[str, Any]:
        """导出规范 publication receipt。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "content_commit_sha1": self.content_commit_sha1,
            "content_parent_sha1": self.content_parent_sha1,
            "execution_state": dict(self.execution_state),
            "format_version": self.format_version,
            "global_manifest_identity": self.global_manifest_identity.to_dict(),
            "historical_hold_receipt_identity": (
                self.historical_hold_receipt_identity.to_dict()
            ),
            "post_publish_gate": self.post_publish_gate.to_dict(),
            "publication_state": self.publication_state.to_dict(),
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_self_excluded": self.receipt_self_excluded,
            "release_inventory": [item.to_dict() for item in self.release_inventory],
            "release_key": self.release_key,
            "remote_branch": self.remote_branch,
            "remote_name": self.remote_name,
            "remote_ref": self.remote_ref,
            "status": self.status,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 receipt 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回规范 receipt SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "D03PublicationReceipt":
        """从严格 object 恢复 publication receipt。"""
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "content_commit_sha1",
            "content_parent_sha1", "execution_state", "format_version",
            "global_manifest_identity", "historical_hold_receipt_identity",
            "post_publish_gate", "publication_state", "receipt_relative_path",
            "receipt_self_excluded", "release_inventory", "release_key",
            "remote_branch", "remote_name", "remote_ref", "status",
        }, where="D03PublicationReceipt")
        if not isinstance(raw["release_inventory"], list):
            raise D03ContractError("receipt release inventory 必须是数组")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            str(raw["status"]),
            D03PublicationState.from_dict(raw["publication_state"]),
            str(raw["remote_name"]), str(raw["remote_branch"]),
            str(raw["remote_ref"]), str(raw["content_parent_sha1"]),
            str(raw["content_commit_sha1"]),
            D03FileIdentity.from_dict(raw["global_manifest_identity"]),
            tuple(D03FileIdentity.from_dict(item)
                  for item in raw["release_inventory"]),
            D03FileIdentity.from_dict(raw["historical_hold_receipt_identity"]),
            D03PostPublishGate.from_dict(raw["post_publish_gate"]),
            str(raw["receipt_relative_path"]), raw["receipt_self_excluded"],
            raw["execution_state"],
        )


def read_d03_publication_receipt(path: str | Path) -> D03PublicationReceipt:
    """严格回读规范 publication receipt。"""
    target = Path(path)
    receipt = D03PublicationReceipt.from_dict(read_canonical_object(target))
    if receipt.canonical_bytes() != target.read_bytes():
        raise D03ContractError("publication receipt 非规范字节")
    return receipt


def write_d03_publication_receipt(
        receipt: D03PublicationReceipt,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 publication receipt，拒绝同路径异内容。"""
    if not isinstance(receipt, D03PublicationReceipt):
        raise D03ContractError("publication receipt 类型非法")
    return write_immutable_json(receipt.to_dict(), path)


__all__ = [
    "D03PostPublishGate",
    "D03PublicationInventoryScan",
    "D03PublicationReceipt",
    "GitHubCIJob",
    "RECEIPT_ARTIFACT_KIND",
    "RECEIPT_ARTIFACT_VERSION",
    "RECEIPT_STATUS",
    "REQUIRED_CI_JOB_NAMES",
    "read_d03_publication_receipt",
    "scan_d03_publication_inventory",
    "write_d03_publication_receipt",
]
