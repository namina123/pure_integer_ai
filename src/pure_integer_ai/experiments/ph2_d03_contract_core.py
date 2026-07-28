"""D-03 正式发布合同共用的严格值、文件身份和发布三态。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
STAGE_KEYS = (
    "W-01", "W-02", "W-03", "W-04", "W-05",
    "W-06", "W-07", "W-08", "W-09",
)
W06_SUBSTAGE_KEYS = (
    "PURE_ALIAS_REFERS",
    "SUBSET_MEMBER",
    "PROPERTY",
    "MEREOLOGY",
    "SIMILAR_ANTONYM",
    "PRECEDES",
    "CAUSES",
)
W07_SUBSTAGE_KEYS = (
    "NOT",
    "AND_OR",
    "CONDITION",
    "EXISTS",
    "FORALL",
    "MODAL",
    "NESTED_SCOPE",
)
VERSION_KEYS = (
    "backend_version",
    "code_version",
    "course_version",
    "data_version",
    "evaluator_version",
    "location_version",
    "parser_version",
    "primitive_version",
    "schema_version",
    "segment_version",
)
EXECUTION_STATE_KEYS = (
    "assessment_updates",
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "evaluator_label_writes",
    "formal_training_runs",
    "mastered_claims",
    "memory_learning_writes",
    "readiness_claims",
    "teacher_calls",
    "use_learning_writes",
    "w01_started",
)
ZERO_EXECUTION_STATE = {key: 0 for key in EXECUTION_STATE_KEYS}
PUBLICATION_STATES = (
    "CANDIDATE_VERIFIED",
    "GIT_PUBLISHED",
    "POST_PUBLISH_VERIFIED",
)


class D03ContractError(ValueError):
    """D-03 冻结身份、阶段合同或规范文件不满足正式协议。"""


def text(value: Any, *, where: str) -> str:
    """要求值是无首尾空白的非空文本。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise D03ContractError(f"{where} 必须是非空无首尾空白文本")
    return value


def positive(value: Any, *, where: str) -> int:
    """要求值是正严格整数。"""
    if type(value) is not int or value <= 0:
        raise D03ContractError(f"{where} 必须是正严格整数")
    return value


def nonnegative(value: Any, *, where: str) -> int:
    """要求值是非负严格整数。"""
    if type(value) is not int or value < 0:
        raise D03ContractError(f"{where} 必须是非负严格整数")
    return value


def flag(value: Any, *, where: str) -> int:
    """要求值是严格整数 0 或 1。"""
    if type(value) is not int or value not in (0, 1):
        raise D03ContractError(f"{where} 必须是严格整数 0/1")
    return value


def enum_text(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    """要求文本属于给定冻结集合。"""
    result = text(value, where=where)
    if result not in allowed:
        raise D03ContractError(f"{where} 不在允许集合")
    return result


def sha256_text(value: Any, *, where: str) -> str:
    """要求值是小写规范 SHA-256。"""
    result = text(value, where=where).lower()
    if (len(result) != 64
            or any(character not in "0123456789abcdef" for character in result)):
        raise D03ContractError(f"{where} 必须是 SHA-256")
    return result


def sha1_text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求值是小写规范 SHA-1，可按合同允许空值。"""
    if allow_empty and value == "":
        return ""
    result = text(value, where=where).lower()
    if (len(result) != 40
            or any(character not in "0123456789abcdef" for character in result)):
        raise D03ContractError(f"{where} 必须是 SHA-1")
    return result


def relative_path(value: Any, *, where: str) -> str:
    """要求值是无逃逸、无反斜杠的 POSIX 相对路径。"""
    result = text(value, where=where)
    path = PurePosixPath(result)
    if (path.is_absolute() or ".." in path.parts or "\\" in result
            or path.as_posix() != result):
        raise D03ContractError(f"{where} 必须是安全 POSIX 相对路径")
    return result


def exact_dict(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 object 字段集合精确匹配合同。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise D03ContractError(f"{where} 字段不完整或含未知字段")
    return value


def string_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        sort_values: bool = False,
        ) -> tuple[str, ...]:
    """恢复无重复文本 tuple，并可要求规范排序。"""
    if not isinstance(value, (list, tuple)):
        raise D03ContractError(f"{where} 必须是数组")
    result = tuple(text(item, where=f"{where}[]") for item in value)
    if not result and not allow_empty:
        raise D03ContractError(f"{where} 不能为空")
    if len(result) != len(set(result)):
        raise D03ContractError(f"{where} 不得重复")
    if sort_values:
        result = tuple(sorted(result))
    return result


def validate_zero_execution_state(value: Any, *, d03_published: int = 0) -> dict[str, int]:
    """校验全部执行计数，仅按发布阶段允许 D-03 位为一。"""
    raw = exact_dict(value, set(EXECUTION_STATE_KEYS), where="execution_state")
    result: dict[str, int] = {}
    for key in EXECUTION_STATE_KEYS:
        result[key] = flag(raw[key], where=f"execution_state.{key}")
    expected = dict(ZERO_EXECUTION_STATE)
    expected["d03_published"] = d03_published
    if result != expected:
        raise D03ContractError("execution_state 含训练、teacher、学习写或能力声明")
    return result


@dataclass(frozen=True, order=True)
class D03FileIdentity:
    """D-03 直接依赖或发布文件的路径、字节数和 SHA-256。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", relative_path(
            self.relative_path, where="file.relative_path"))
        nonnegative(self.size_bytes, where="file.size_bytes")
        object.__setattr__(self, "sha256", sha256_text(
            self.sha256, where="file.sha256"))

    def to_dict(self) -> dict[str, Any]:
        """导出规范文件身份。"""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03FileIdentity":
        """从严格 object 恢复文件身份。"""
        raw = exact_dict(value, {
            "relative_path", "sha256", "size_bytes",
        }, where="D03FileIdentity")
        return cls(str(raw["relative_path"]), raw["size_bytes"], str(raw["sha256"]))


@dataclass(frozen=True)
class D03ReleaseIdentity:
    """绑定父门、能力基线、D-02 覆盖账和全部正式版本键。"""

    format_version: int
    release_key: str
    release_version: str
    parent_gate_path: str
    parent_gate_sha256: str
    capability_baseline_path: str
    capability_baseline_sha256: str
    source_coverage_path: str
    source_coverage_sha256: str
    version_keys: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise D03ContractError("release format_version 非法")
        text(self.release_key, where="release_key")
        text(self.release_version, where="release_version")
        for name in (
                "parent_gate_path", "capability_baseline_path",
                "source_coverage_path"):
            object.__setattr__(self, name, relative_path(
                getattr(self, name), where=name))
        for name in (
                "parent_gate_sha256", "capability_baseline_sha256",
                "source_coverage_sha256"):
            object.__setattr__(self, name, sha256_text(
                getattr(self, name), where=name))
        if (not isinstance(self.version_keys, tuple)
                or any(not isinstance(item, tuple) or len(item) != 2
                       for item in self.version_keys)):
            raise D03ContractError("version keys 类型非法")
        mapping = {key: value for key, value in self.version_keys}
        if (len(mapping) != len(self.version_keys)
                or tuple(sorted(mapping)) != VERSION_KEYS):
            raise D03ContractError("version keys 不完整或含未知 version")
        normalized = tuple(
            (key, text(mapping[key], where=f"version_keys.{key}"))
            for key in VERSION_KEYS
        )
        object.__setattr__(self, "version_keys", normalized)

    def to_dict(self) -> dict[str, Any]:
        """导出规范发布身份。"""
        return {
            "capability_baseline_path": self.capability_baseline_path,
            "capability_baseline_sha256": self.capability_baseline_sha256,
            "format_version": self.format_version,
            "parent_gate_path": self.parent_gate_path,
            "parent_gate_sha256": self.parent_gate_sha256,
            "release_key": self.release_key,
            "release_version": self.release_version,
            "source_coverage_path": self.source_coverage_path,
            "source_coverage_sha256": self.source_coverage_sha256,
            "version_keys": dict(self.version_keys),
        }

    def canonical_bytes(self) -> bytes:
        """返回发布身份的规范 UTF-8 JSON 字节。"""
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        """返回发布身份的规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "D03ReleaseIdentity":
        """从严格 object 恢复发布身份。"""
        raw = exact_dict(value, {
            "capability_baseline_path", "capability_baseline_sha256",
            "format_version", "parent_gate_path", "parent_gate_sha256",
            "release_key", "release_version", "source_coverage_path",
            "source_coverage_sha256", "version_keys",
        }, where="D03ReleaseIdentity")
        versions = raw["version_keys"]
        if not isinstance(versions, dict):
            raise D03ContractError("version keys 必须是 object")
        return cls(
            raw["format_version"], str(raw["release_key"]),
            str(raw["release_version"]), str(raw["parent_gate_path"]),
            str(raw["parent_gate_sha256"]),
            str(raw["capability_baseline_path"]),
            str(raw["capability_baseline_sha256"]),
            str(raw["source_coverage_path"]),
            str(raw["source_coverage_sha256"]),
            tuple((str(key), str(item)) for key, item in versions.items()),
        )


@dataclass(frozen=True)
class D03PublicationState:
    """把候选、Git 内容发布和发布后核验分成不可混合的三态。"""

    state: str
    d03_published: int
    content_commit_sha1: str
    post_publish_verified: int

    def __post_init__(self) -> None:
        enum_text(self.state, PUBLICATION_STATES, where="publication state")
        flag(self.d03_published, where="d03_published")
        flag(self.post_publish_verified, where="post_publish_verified")
        object.__setattr__(self, "content_commit_sha1", sha1_text(
            self.content_commit_sha1,
            where="content_commit_sha1",
            allow_empty=self.state == "CANDIDATE_VERIFIED",
        ))
        expected = {
            "CANDIDATE_VERIFIED": (0, 0, False),
            "GIT_PUBLISHED": (0, 0, True),
            "POST_PUBLISH_VERIFIED": (1, 1, True),
        }[self.state]
        if (self.d03_published, self.post_publish_verified) != expected[:2]:
            raise D03ContractError("只有 post-publish verified 可声明 D-03 已发布")
        if bool(self.content_commit_sha1) != expected[2]:
            raise D03ContractError("Git publication state 与 content commit 不一致")

    def to_dict(self) -> dict[str, Any]:
        """导出规范发布三态。"""
        return {
            "content_commit_sha1": self.content_commit_sha1,
            "d03_published": self.d03_published,
            "post_publish_verified": self.post_publish_verified,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03PublicationState":
        """从严格 object 恢复发布三态。"""
        raw = exact_dict(value, {
            "content_commit_sha1", "d03_published",
            "post_publish_verified", "state",
        }, where="D03PublicationState")
        return cls(
            str(raw["state"]), raw["d03_published"],
            str(raw["content_commit_sha1"]), raw["post_publish_verified"],
        )


def read_canonical_object(path: str | Path) -> dict[str, Any]:
    """严格读取单换行结尾的规范 JSON object。"""
    target = Path(path)
    try:
        payload = target.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise D03ContractError("D-03 JSON newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except D03ContractError:
        raise
    except Exception as error:
        raise D03ContractError("D-03 JSON 无法读取") from error
    assert isinstance(value, dict)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise D03ContractError("D-03 JSON 不是规范字节")
    return value


def write_immutable_json(value: dict[str, Any], path: str | Path) -> Path:
    """独占或幂等写规范 JSON，绝不覆盖同路径异字节。"""
    if not isinstance(value, dict):
        raise D03ContractError("不可覆盖 writer 只接受 object")
    target = Path(path)
    payload = canonical_json_bytes(value) + b"\n"
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise D03ContractError("D-03 artifact 不可覆盖")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise D03ContractError("D-03 artifact 无法独占写入") from error
    return target


def verify_file_identity(identity: D03FileIdentity, root: str | Path) -> None:
    """在给定仓库根下逐字节回验一个安全文件身份。"""
    repository = Path(root).resolve()
    target = (repository / Path(*PurePosixPath(identity.relative_path).parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise D03ContractError("D-03 文件身份路径缺失或逃逸")
    payload = target.read_bytes()
    if (len(payload) != identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != identity.sha256):
        raise D03ContractError("D-03 文件身份漂移")


__all__ = [
    "D03ContractError",
    "D03FileIdentity",
    "D03PublicationState",
    "D03ReleaseIdentity",
    "EXECUTION_STATE_KEYS",
    "FORMAT_VERSION",
    "PUBLICATION_STATES",
    "STAGE_KEYS",
    "VERSION_KEYS",
    "W06_SUBSTAGE_KEYS",
    "W07_SUBSTAGE_KEYS",
    "ZERO_EXECUTION_STATE",
    "enum_text",
    "exact_dict",
    "flag",
    "nonnegative",
    "positive",
    "read_canonical_object",
    "relative_path",
    "sha1_text",
    "sha256_text",
    "string_tuple",
    "text",
    "validate_zero_execution_state",
    "verify_file_identity",
    "write_immutable_json",
]
