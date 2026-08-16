"""GG-03 formal family 的代码、Observation 与 private owner 安全身份。"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    read_generation_generalization_evaluation_observations,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    generation_generalization_evaluation_requirements,
)


PRIVATE_OWNER_ARTIFACT_KIND = "PH2_GG03_PRIVATE_LABEL_OWNER_RECEIPT_V1"
_CODE_ROOT_MODULES = (
    "pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family",
    "pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner",
    "pure_integer_ai.experiments.ph2_generation_generalization_formal_runner",
)
_OWNER_FIELDS = {
    "artifact_kind", "format_version", "label_commitment_sha256",
    "label_file", "observation_inventory_sha256", "status",
    "verdict_contract_sha256",
}
_OWNER_FILE_FIELDS = {
    "content_sha256", "content_size_bytes", "record_count",
    "relative_path", "transport_sha256", "transport_size_bytes",
}


class GenerationGeneralizationEvaluationFamilyError(RuntimeError):
    """GG-03 family identity、物理隔离或冻结顺序不闭合。"""


def generation_generalization_sha256_bytes(payload: bytes) -> str:
    """返回规范对象或文件字节的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def generation_generalization_sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256，避免随 inventory 规模增加常驻内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def strict_generation_generalization_relative_path(
        value: object,
        *,
        where: str,
        ) -> str:
    """核验 receipt 中不含盘符、反斜杠或上跳的 POSIX 相对路径。"""
    if (not isinstance(value, str) or not value or "\\" in value
            or Path(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 相对路径非法")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationCodeFileIdentity:
    """一份承重 Python 文件的仓库相对身份。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        strict_generation_generalization_relative_path(
            self.relative_path, where="GG-03 code file")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 code file bytes 非法")
        sha256_text(self.sha256, where="GG-03 code file SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationCodeIdentity:
    """从 runner 根模块递归形成的完整公开 Python import 闭包。"""

    files: tuple[GenerationGeneralizationCodeFileIdentity, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        if (not self.files or tuple(sorted(self.files)) != self.files
                or len({item.relative_path for item in self.files})
                != len(self.files)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 code identity 顺序或唯一性漂移")
        sha256_text(self.aggregate_sha256, where="GG-03 code aggregate")
        expected = generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict() for item in self.files]))
        if self.aggregate_sha256 != expected:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 code aggregate 内容锁漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.to_dict() for item in self.files],
        }


def _module_file(repository: Path, module: str) -> Path | None:
    """把仓库内 absolute Python module 映射为源码文件。"""
    relative = Path("src", *module.split("."))
    direct = repository / relative.with_suffix(".py")
    package = repository / relative / "__init__.py"
    if direct.is_file():
        return direct.resolve()
    if package.is_file():
        return package.resolve()
    return None


def _imported_modules(path: Path) -> tuple[str, ...]:
    """只提取 `pure_integer_ai` absolute imports，不执行被扫代码。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 code closure 无法解析") from error
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                item.name for item in node.names
                if item.name == "pure_integer_ai"
                or item.name.startswith("pure_integer_ai."))
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
              and isinstance(node.module, str)
              and (node.module == "pure_integer_ai"
                   or node.module.startswith("pure_integer_ai."))):
            modules.add(node.module)
            for item in node.names:
                modules.add(f"{node.module}.{item.name}")
    return tuple(sorted(modules))


def build_generation_generalization_code_identity(
        repository_root: str | Path,
        ) -> GenerationGeneralizationCodeIdentity:
    """递归冻结 E-05D runner 与 family 自身的公开 import 闭包。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 code repository root 非法")
    pending = list(_CODE_ROOT_MODULES)
    visited: set[Path] = set()
    while pending:
        module = pending.pop()
        path = _module_file(repository, module)
        if path is None or path in visited:
            continue
        visited.add(path)
        pending.extend(_imported_modules(path))
    if not visited:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 code closure 为空")
    files = tuple(sorted(
        GenerationGeneralizationCodeFileIdentity(
            path.relative_to(repository).as_posix(),
            path.stat().st_size,
            generation_generalization_sha256_file(path),
        )
        for path in visited
    ))
    return GenerationGeneralizationCodeIdentity(
        files,
        generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict() for item in files])),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationObservationRecordIdentity:
    """一条正式 label-free Observation 的逐条内容锁。"""

    ordinal: int
    record_sha256: str
    size_bytes: int
    stable_key_sha256: str
    requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        if (type(self.ordinal) is not int or self.ordinal <= 0
                or type(self.size_bytes) is not int or self.size_bytes <= 0):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation record identity 非法")
        sha256_text(self.record_sha256, where="GG-03 Observation record SHA")
        sha256_text(
            self.stable_key_sha256, where="GG-03 Observation stable key SHA")
        if (not self.requirements
                or any(item not in INDEPENDENT_VERIFIER_REQUIREMENTS
                       for item in self.requirements)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation requirement 投影非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "record_sha256": self.record_sha256,
            "requirements": list(self.requirements),
            "size_bytes": self.size_bytes,
            "stable_key_sha256": self.stable_key_sha256,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationObservationInventoryIdentity:
    """正式 Observation inventory 的逐条与整体双遍身份。"""

    record_count: int
    transport_size_bytes: int
    transport_sha256: str
    record_inventory_sha256: str
    resource_ceiling: GenerationGeneralizationEvaluationBudget
    records: tuple[GenerationGeneralizationObservationRecordIdentity, ...]
    requirement_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (type(self.record_count) is not int or self.record_count <= 0
                or self.record_count != len(self.records)
                or type(self.transport_size_bytes) is not int
                or self.transport_size_bytes <= 0):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation inventory count/bytes 非法")
        sha256_text(
            self.transport_sha256, where="GG-03 Observation transport SHA")
        sha256_text(
            self.record_inventory_sha256,
            where="GG-03 Observation record inventory SHA")
        if not isinstance(
                self.resource_ceiling, GenerationGeneralizationEvaluationBudget):
            raise TypeError("GG-03 Observation resource ceiling 类型错误")
        if tuple(item.ordinal for item in self.records) != tuple(
                range(1, self.record_count + 1)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation ordinal 漂移")
        expected_counts = tuple(
            (requirement, sum(
                requirement in item.requirements for item in self.records))
            for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS)
        if self.requirement_counts != expected_counts or any(
                count <= 0 for _requirement, count in expected_counts):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation inventory 未覆盖全部 requirement")
        expected_inventory = generation_generalization_sha256_bytes(
            canonical_json_bytes([item.to_dict() for item in self.records]))
        if self.record_inventory_sha256 != expected_inventory:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 Observation record inventory 内容锁漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "record_inventory_sha256": self.record_inventory_sha256,
            "records": [item.to_dict() for item in self.records],
            "requirement_counts": [
                {"count": count, "requirement": requirement}
                for requirement, count in self.requirement_counts
            ],
            "resource_ceiling": self.resource_ceiling.to_dict(),
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }


def scan_generation_generalization_observation_inventory(
        path: str | Path,
        *,
        resource_ceiling: GenerationGeneralizationEvaluationBudget,
        ) -> GenerationGeneralizationObservationInventoryIdentity:
    """严格读取一次正式 Observation inventory 并形成逐条身份。"""
    target = Path(path).resolve()
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 Observation inventory 不可读") from error
    observations = read_generation_generalization_evaluation_observations(target)
    lines = tuple(payload.splitlines(keepends=True))
    if len(lines) != len(observations):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 Observation inventory 行数漂移")
    stable_keys = tuple(item.stable_key() for item in observations)
    if stable_keys != tuple(sorted(stable_keys)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 Observation inventory 未按 stable key 冻结排序")
    if any(item.resource_budget != resource_ceiling for item in observations):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 Observation 存在隐式逐条 resource threshold")
    records = tuple(
        GenerationGeneralizationObservationRecordIdentity(
            ordinal,
            generation_generalization_sha256_bytes(line),
            len(line),
            generation_generalization_sha256_bytes(
                canonical_json_bytes(list(observation.stable_key()))),
            generation_generalization_evaluation_requirements(observation),
        )
        for ordinal, (line, observation) in enumerate(
            zip(lines, observations, strict=True), start=1)
    )
    counts = tuple(
        (requirement, sum(requirement in item.requirements for item in records))
        for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS)
    return GenerationGeneralizationObservationInventoryIdentity(
        len(records), len(payload),
        generation_generalization_sha256_bytes(payload),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict() for item in records])),
        resource_ceiling, records, counts,
    )


def double_scan_generation_generalization_observation_inventory(
        path: str | Path,
        *,
        resource_ceiling: GenerationGeneralizationEvaluationBudget,
        ) -> GenerationGeneralizationObservationInventoryIdentity:
    """执行两次独立严格扫描，并要求逐字段与整体 identity 完全相同。"""
    first = scan_generation_generalization_observation_inventory(
        path, resource_ceiling=resource_ceiling)
    second = scan_generation_generalization_observation_inventory(
        path, resource_ceiling=resource_ceiling)
    if first != second:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 Observation inventory 双遍 identity 不一致")
    return first


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationPrivateLabelOwnerReceipt:
    """freeze 可见的 private owner 安全元数据，不包含任何 label。"""

    observation_inventory_sha256: str
    label_relative_path: str
    label_transport_size_bytes: int
    label_transport_sha256: str
    label_content_size_bytes: int
    label_content_sha256: str
    label_record_count: int
    label_commitment_sha256: str
    verdict_contract_sha256: str
    status: str = "SEALED_UNREAD"

    def __post_init__(self) -> None:
        for name in (
                "observation_inventory_sha256", "label_transport_sha256",
                "label_content_sha256", "label_commitment_sha256",
                "verdict_contract_sha256"):
            sha256_text(getattr(self, name), where=f"GG-03 private owner {name}")
        strict_generation_generalization_relative_path(
            self.label_relative_path, where="GG-03 private label file")
        if any(type(value) is not int or value <= 0 for value in (
                self.label_transport_size_bytes, self.label_content_size_bytes,
                self.label_record_count)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 private owner count/bytes 非法")
        if self.status != "SEALED_UNREAD":
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 private labels 在 freeze 前已消费")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": PRIVATE_OWNER_ARTIFACT_KIND,
            "format_version": 1,
            "label_commitment_sha256": self.label_commitment_sha256,
            "label_file": {
                "content_sha256": self.label_content_sha256,
                "content_size_bytes": self.label_content_size_bytes,
                "record_count": self.label_record_count,
                "relative_path": self.label_relative_path,
                "transport_sha256": self.label_transport_sha256,
                "transport_size_bytes": self.label_transport_size_bytes,
            },
            "observation_inventory_sha256": (
                self.observation_inventory_sha256),
            "status": self.status,
            "verdict_contract_sha256": self.verdict_contract_sha256,
        }


def read_generation_generalization_private_label_owner_receipt(
        path: str | Path,
        ) -> tuple[GenerationGeneralizationPrivateLabelOwnerReceipt, str]:
    """只读 owner receipt 本身；不解析、stat 或打开其 label 文件。"""
    target = Path(path).resolve()
    try:
        payload = target.read_bytes()
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (OSError, ValueError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private owner receipt 不可读") from error
    if (not payload.endswith(b"\n") or payload.endswith(b"\n\n")
            or canonical_json_line(value) != payload):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private owner receipt 非 canonical JSON")
    raw = exact_dict(value, _OWNER_FIELDS, where="GG-03 private owner receipt")
    label = exact_dict(
        raw["label_file"], _OWNER_FILE_FIELDS,
        where="GG-03 private owner label identity")
    if (raw["artifact_kind"] != PRIVATE_OWNER_ARTIFACT_KIND
            or raw["format_version"] != 1):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private owner kind/version 漂移")
    receipt = GenerationGeneralizationPrivateLabelOwnerReceipt(
        str(raw["observation_inventory_sha256"]),
        str(label["relative_path"]),
        label["transport_size_bytes"], str(label["transport_sha256"]),
        label["content_size_bytes"], str(label["content_sha256"]),
        label["record_count"], str(raw["label_commitment_sha256"]),
        str(raw["verdict_contract_sha256"]), str(raw["status"]),
    )
    return receipt, generation_generalization_sha256_bytes(payload)


__all__ = [
    "PRIVATE_OWNER_ARTIFACT_KIND",
    "GenerationGeneralizationCodeFileIdentity",
    "GenerationGeneralizationCodeIdentity",
    "GenerationGeneralizationEvaluationFamilyError",
    "GenerationGeneralizationObservationInventoryIdentity",
    "GenerationGeneralizationObservationRecordIdentity",
    "GenerationGeneralizationPrivateLabelOwnerReceipt",
    "build_generation_generalization_code_identity",
    "double_scan_generation_generalization_observation_inventory",
    "generation_generalization_sha256_bytes",
    "generation_generalization_sha256_file",
    "read_generation_generalization_private_label_owner_receipt",
    "scan_generation_generalization_observation_inventory",
    "strict_generation_generalization_relative_path",
]
