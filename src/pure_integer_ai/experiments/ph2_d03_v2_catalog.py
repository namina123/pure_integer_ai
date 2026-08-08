"""构建、发布和严格回读 PH2-D03-V2 successor 合同 manifest。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    OWNER_POLICIES,
    V1_PUBLIC_RECEIPT_PATH,
    V1_PUBLIC_RECEIPT_SHA256,
    V1_PUBLIC_RECEIPT_SIZE_BYTES,
    V2_ADAPTER_VERSION,
    V2_ALLOWED_WORKERS,
    V2_AUTHORITY_FORMAT_VERSION,
    V2_CARRIER_SCHEMA_VERSION,
    V2_CHECKPOINT_FORMAT_VERSION,
    V2_CHECKPOINT_IDENTITY_FIELDS,
    V2_CONTRACT_KIND,
    V2_CONTRACT_PATH,
    V2_CONTRACT_VERSION,
    V2_COURSE_VERSION,
    V2_DEFERRED_P3_MAX_RECORDS,
    V2_DEFERRED_P3_MIN_RECORDS,
    V2_EXECUTION_STAGES,
    V2_GENERATOR_VERSION,
    V2_INITIAL_EXECUTION_STATE,
    V2_INVALIDATION_FORMAT_VERSION,
    V2_INVALIDATION_VERSION,
    V2_LOGICAL_SHARD_COUNT,
    V2_MERGE_BARRIER_KEY,
    V2_OWNER_KEYS,
    V2_P3_ACTIVATION_POLICY,
    V2_PACK_EARLIEST_STAGES,
    V2_PACK_INVALIDATION_KINDS,
    V2_PARSER_VERSION,
    V2_RELEASE_KEY,
    V2_RELEASE_VERSION,
    V2_RUN_ID_POLICY,
    V2_RUN_IDENTITY_FIELDS,
    V2_SCALE_BUDGETS,
    V2_SCALE_KEYS,
    V2_SCHEMA_VERSION,
    V2_SPLIT_POLICY,
    V2_UNKNOWN_INVALIDATION_POLICY,
    V2InvalidationRule,
    V2OwnerPolicy,
    V2ScaleBudget,
    V2SplitPolicy,
    build_v2_invalidation_rules,
    validate_v2_initial_state,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    V2_PUBLIC_LICENSES,
    V2_SOURCE_KEYS,
    V2_SOURCE_LICENSES,
    record_schema_bindings,
)


def _resolve(root: Path, relative: str) -> Path:
    """在仓库根内安全解析 POSIX 相对路径。"""
    repository = root.resolve()
    target = (repository / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(repository):
        raise D03ContractError("v2 catalog 路径逃逸 root")
    return target


def _identity(root: Path, relative: str) -> D03FileIdentity:
    """读取公开文件并形成不可变路径、大小和摘要身份。"""
    target = _resolve(root, relative)
    if not target.is_file():
        raise D03ContractError(f"v2 catalog 依赖文件缺失: {relative}")
    payload = target.read_bytes()
    return D03FileIdentity(relative, len(payload), hashlib.sha256(payload).hexdigest())


def _validate_prior_receipt(root: Path) -> D03FileIdentity:
    """只读核验已公开 D-03 v1 receipt，不读取 private 或候选内容。"""
    identity = _identity(root, V1_PUBLIC_RECEIPT_PATH)
    if (identity.size_bytes != V1_PUBLIC_RECEIPT_SIZE_BYTES
            or identity.sha256 != V1_PUBLIC_RECEIPT_SHA256):
        raise D03ContractError("v1 public receipt identity 漂移")
    receipt = read_canonical_object(_resolve(root, V1_PUBLIC_RECEIPT_PATH))
    if (receipt.get("release_key") != "PH2-D03-V1"
            or receipt.get("status") != "POST_PUBLISH_VERIFIED"):
        raise D03ContractError("v1 public receipt 不满足 successor 绑定条件")
    return identity


@dataclass(frozen=True)
class V2SuccessorContract:
    """合取 successor release 的全部静态 authority 和 fail-closed 政策。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    release_version: str
    schema_version: int
    course_version: int
    adapter_version: int
    generator_version: int
    parser_version: int
    carrier_schema_version: int
    prior_release_receipt: D03FileIdentity
    source_keys: tuple[str, ...]
    source_licenses: tuple[tuple[str, tuple[str, ...]], ...]
    public_license_ids: tuple[str, ...]
    schema_bindings: tuple[dict[str, Any], ...]
    owner_policies: tuple[V2OwnerPolicy, ...]
    split_policy: V2SplitPolicy
    scale_budgets: tuple[V2ScaleBudget, ...]
    run_policy: dict[str, Any]
    invalidation_format_version: int
    invalidation_version: str
    pack_invalidation_kinds: tuple[str, ...]
    pack_earliest_stages: tuple[str, ...]
    unknown_invalidation_policy: str
    execution_stages: tuple[str, ...]
    invalidation_rules: tuple[V2InvalidationRule, ...]
    initial_state: dict[str, int]
    status: str

    def __post_init__(self) -> None:
        if (type(self.format_version) is not int
                or self.format_version != V2_AUTHORITY_FORMAT_VERSION):
            raise D03ContractError("v2 contract format_version 非法")
        if self.artifact_kind != V2_CONTRACT_KIND:
            raise D03ContractError("v2 contract artifact_kind 非法")
        if self.artifact_version != V2_CONTRACT_VERSION:
            raise D03ContractError("v2 contract artifact_version 漂移")
        if self.release_key != V2_RELEASE_KEY or self.release_version != V2_RELEASE_VERSION:
            raise D03ContractError("v2 contract release identity 漂移")
        version_values = (
            self.schema_version, self.course_version, self.adapter_version,
            self.generator_version, self.parser_version, self.carrier_schema_version,
        )
        if (any(type(value) is not int for value in version_values)
                or self.schema_version != V2_SCHEMA_VERSION
                or self.course_version != V2_COURSE_VERSION
                or self.adapter_version != V2_ADAPTER_VERSION
                or self.generator_version != V2_GENERATOR_VERSION
                or self.parser_version != V2_PARSER_VERSION
                or self.carrier_schema_version != V2_CARRIER_SCHEMA_VERSION):
            raise D03ContractError("v2 contract version key 漂移")
        if not isinstance(self.prior_release_receipt, D03FileIdentity):
            raise D03ContractError("v2 prior receipt identity 类型错误")
        if (self.prior_release_receipt.relative_path != V1_PUBLIC_RECEIPT_PATH
                or self.prior_release_receipt.size_bytes != V1_PUBLIC_RECEIPT_SIZE_BYTES
                or self.prior_release_receipt.sha256 != V1_PUBLIC_RECEIPT_SHA256):
            raise D03ContractError("v2 prior receipt identity 不匹配")
        if self.source_keys != V2_SOURCE_KEYS:
            raise D03ContractError("v2 source allowlist 漂移")
        if self.source_licenses != tuple(
                (key, V2_SOURCE_LICENSES[key]) for key in V2_SOURCE_KEYS):
            raise D03ContractError("v2 source license map 漂移")
        if self.public_license_ids != tuple(sorted(V2_PUBLIC_LICENSES)):
            raise D03ContractError("v2 public license allowlist 漂移")
        if tuple(self.execution_stages) != V2_EXECUTION_STAGES:
            raise D03ContractError("v2 execution stage 顺序漂移")
        if (not isinstance(self.owner_policies, tuple)
                or any(not isinstance(item, V2OwnerPolicy)
                       for item in self.owner_policies)
                or tuple(item.owner_key for item in self.owner_policies) != V2_OWNER_KEYS
                or self.owner_policies != OWNER_POLICIES):
            raise D03ContractError("v2 owner namespace 不完整或乱序")
        if (not isinstance(self.scale_budgets, tuple)
                or any(not isinstance(item, V2ScaleBudget)
                       for item in self.scale_budgets)
                or tuple(item.scale_key for item in self.scale_budgets) != V2_SCALE_KEYS
                or self.scale_budgets != V2_SCALE_BUDGETS):
            raise D03ContractError("v2 scale budget 不完整或乱序")
        if tuple(item for item in self.schema_bindings) != record_schema_bindings():
            raise D03ContractError("v2 schema binding 漂移")
        if self.split_policy != V2_SPLIT_POLICY:
            raise D03ContractError("v2 split policy 漂移")
        validate_v2_run_policy(self.run_policy)
        if self.invalidation_format_version != V2_INVALIDATION_FORMAT_VERSION:
            raise D03ContractError("v2 invalidation format 漂移")
        if self.invalidation_version != V2_INVALIDATION_VERSION:
            raise D03ContractError("v2 invalidation version 漂移")
        if (self.pack_invalidation_kinds != V2_PACK_INVALIDATION_KINDS
                or self.pack_earliest_stages != V2_PACK_EARLIEST_STAGES
                or self.unknown_invalidation_policy != V2_UNKNOWN_INVALIDATION_POLICY):
            raise D03ContractError("v2 pack invalidation policy 漂移")
        if (not isinstance(self.invalidation_rules, tuple)
                or any(not isinstance(item, V2InvalidationRule)
                       for item in self.invalidation_rules)
                or len({(item.change_kind, item.subject_key)
                        for item in self.invalidation_rules}) != len(self.invalidation_rules)
                or self.invalidation_rules != tuple(sorted(self.invalidation_rules))):
            raise D03ContractError("v2 invalidation rules 重复或类型错误")
        required_rules = {
            (item.change_kind, item.subject_key, item.earliest_stage, item.suffix)
            for item in build_v2_invalidation_rules()
        }
        actual_rules = {
            (item.change_kind, item.subject_key, item.earliest_stage, item.suffix)
            for item in self.invalidation_rules
        }
        if not required_rules.issubset(actual_rules):
            raise D03ContractError("v2 invalidation graph 缺全局或 evaluator 后缀")
        validate_v2_initial_state(self.initial_state)
        if self.status != "DATA_CONTRACT_FROZEN":
            raise D03ContractError("v2 contract status 必须是 DATA_CONTRACT_FROZEN")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 successor contract object。"""
        return {
            "adapter_version": self.adapter_version,
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "carrier_schema_version": self.carrier_schema_version,
            "course_version": self.course_version,
            "execution_stages": list(self.execution_stages),
            "format_version": self.format_version,
            "generator_version": self.generator_version,
            "initial_state": dict(self.initial_state),
            "invalidation_format_version": self.invalidation_format_version,
            "invalidation_rules": [item.to_dict() for item in self.invalidation_rules],
            "invalidation_version": self.invalidation_version,
            "owner_policies": [item.to_dict() for item in self.owner_policies],
            "pack_earliest_stages": list(self.pack_earliest_stages),
            "pack_invalidation_kinds": list(self.pack_invalidation_kinds),
            "parser_version": self.parser_version,
            "prior_release_receipt": self.prior_release_receipt.to_dict(),
            "public_license_ids": list(self.public_license_ids),
            "release_key": self.release_key,
            "release_version": self.release_version,
            "run_policy": dict(self.run_policy),
            "scale_budgets": [item.to_dict() for item in self.scale_budgets],
            "schema_bindings": [dict(item) for item in self.schema_bindings],
            "schema_version": self.schema_version,
            "source_keys": list(self.source_keys),
            "source_licenses": [
                {"license_ids": list(license_ids), "source_key": key}
                for key, license_ids in self.source_licenses
            ],
            "split_policy": self.split_policy.to_dict(),
            "status": self.status,
            "unknown_invalidation_policy": self.unknown_invalidation_policy,
        }

    def sha256(self) -> str:
        """返回 successor contract 规范摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "V2SuccessorContract":
        """从规范 object 严格恢复 successor contract。"""
        keys = {
            "adapter_version", "artifact_kind", "artifact_version",
            "carrier_schema_version", "course_version", "execution_stages",
            "format_version", "generator_version", "initial_state",
            "invalidation_format_version", "invalidation_rules",
            "invalidation_version", "owner_policies", "parser_version",
            "pack_earliest_stages", "pack_invalidation_kinds",
            "prior_release_receipt", "public_license_ids", "release_key",
            "release_version", "run_policy", "scale_budgets", "schema_bindings",
            "schema_version", "source_keys", "source_licenses", "split_policy",
            "status", "unknown_invalidation_policy",
        }
        raw = exact_dict(value, keys, where="V2SuccessorContract")
        for name in ("execution_stages", "owner_policies", "scale_budgets",
                     "schema_bindings", "source_keys", "source_licenses",
                     "invalidation_rules", "pack_earliest_stages",
                     "pack_invalidation_kinds"):
            if not isinstance(raw[name], list):
                raise D03ContractError(f"v2 contract {name} 必须是数组")
        source_licenses = []
        for item in raw["source_licenses"]:
            pair = exact_dict(
                item, {"license_ids", "source_key"}, where="v2 source license")
            if not isinstance(pair["license_ids"], list):
                raise D03ContractError("v2 source license ids 必须是数组")
            source_licenses.append((
                str(pair["source_key"]),
                tuple(str(value) for value in pair["license_ids"]),
            ))
        owner_policies = tuple(
            V2OwnerPolicy.from_dict(item) for item in raw["owner_policies"])
        split_policy = V2SplitPolicy.from_dict(raw["split_policy"])
        budgets = tuple(V2ScaleBudget.from_dict(item) for item in raw["scale_budgets"])
        rules = tuple(V2InvalidationRule.from_dict(item) for item in raw["invalidation_rules"])
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            str(raw["release_version"]), raw["schema_version"],
            raw["course_version"], raw["adapter_version"],
            raw["generator_version"], raw["parser_version"],
            raw["carrier_schema_version"],
            D03FileIdentity.from_dict(raw["prior_release_receipt"]),
            tuple(str(item) for item in raw["source_keys"]),
            tuple(source_licenses),
            tuple(str(item) for item in raw["public_license_ids"]),
            tuple(dict(item) for item in raw["schema_bindings"]),
            owner_policies, split_policy, budgets,
            dict(raw["run_policy"]), raw["invalidation_format_version"],
            str(raw["invalidation_version"]),
            tuple(str(item) for item in raw["pack_invalidation_kinds"]),
            tuple(str(item) for item in raw["pack_earliest_stages"]),
            str(raw["unknown_invalidation_policy"]),
            tuple(str(item) for item in raw["execution_stages"]), rules,
            dict(raw["initial_state"]), str(raw["status"]),
        )


def v2_run_policy() -> dict[str, Any]:
    """返回 canonical run identity 和 checkpoint 的冻结政策。"""
    return {
        "allowed_worker_counts": list(V2_ALLOWED_WORKERS),
        "checkpoint_identity_fields": list(V2_CHECKPOINT_IDENTITY_FIELDS),
        "checkpoint_format_version": V2_CHECKPOINT_FORMAT_VERSION,
        "logical_shard_count": V2_LOGICAL_SHARD_COUNT,
        "merge_barrier_key": V2_MERGE_BARRIER_KEY,
        "p3_activation_policy": V2_P3_ACTIVATION_POLICY,
        "p3_max_records": V2_DEFERRED_P3_MAX_RECORDS,
        "p3_min_records": V2_DEFERRED_P3_MIN_RECORDS,
        "run_id_policy": V2_RUN_ID_POLICY,
        "run_identity_fields": list(V2_RUN_IDENTITY_FIELDS),
    }


def validate_v2_run_policy(value: Any) -> dict[str, Any]:
    """严格校验 run policy，拒绝缺 worker/shard/base identity 的简化对象。"""
    raw = exact_dict(value, {
        "allowed_worker_counts", "checkpoint_identity_fields",
        "checkpoint_format_version", "logical_shard_count", "merge_barrier_key",
        "p3_activation_policy", "p3_max_records", "p3_min_records",
        "run_id_policy", "run_identity_fields",
    }, where="v2 run policy")
    if (not isinstance(raw["allowed_worker_counts"], list)
            or tuple(raw["allowed_worker_counts"]) != V2_ALLOWED_WORKERS):
        raise D03ContractError("v2 run worker policy 漂移")
    if raw["checkpoint_format_version"] != V2_CHECKPOINT_FORMAT_VERSION:
        raise D03ContractError("v2 checkpoint format 漂移")
    if raw["logical_shard_count"] != V2_LOGICAL_SHARD_COUNT:
        raise D03ContractError("v2 logical shard policy 漂移")
    if raw["merge_barrier_key"] != V2_MERGE_BARRIER_KEY:
        raise D03ContractError("v2 merge barrier 漂移")
    if raw["run_id_policy"] != V2_RUN_ID_POLICY:
        raise D03ContractError("v2 run id policy 漂移")
    if (raw["p3_activation_policy"] != V2_P3_ACTIVATION_POLICY
            or raw["p3_min_records"] != V2_DEFERRED_P3_MIN_RECORDS
            or raw["p3_max_records"] != V2_DEFERRED_P3_MAX_RECORDS):
        raise D03ContractError("v2 P3 deferred policy 漂移")
    if (not isinstance(raw["run_identity_fields"], list)
            or tuple(raw["run_identity_fields"]) != V2_RUN_IDENTITY_FIELDS):
        raise D03ContractError("v2 run identity field policy 漂移")
    if (not isinstance(raw["checkpoint_identity_fields"], list)
            or tuple(raw["checkpoint_identity_fields"])
            != V2_CHECKPOINT_IDENTITY_FIELDS):
        raise D03ContractError("v2 checkpoint identity field policy 漂移")
    return dict(raw)


def build_v2_successor_contract(
        repository_root: str | Path,
        *,
        pack_earliest_stages: tuple[tuple[str, str], ...] = (),
        ) -> V2SuccessorContract:
    """从公开 v1 receipt 构建零训练状态的 D-03 v2 successor 合同。"""
    root = Path(repository_root).resolve()
    prior = _validate_prior_receipt(root)
    return V2SuccessorContract(
        V2_AUTHORITY_FORMAT_VERSION,
        V2_CONTRACT_KIND,
        V2_CONTRACT_VERSION,
        V2_RELEASE_KEY,
        V2_RELEASE_VERSION,
        V2_SCHEMA_VERSION,
        V2_COURSE_VERSION,
        V2_ADAPTER_VERSION,
        V2_GENERATOR_VERSION,
        V2_PARSER_VERSION,
        V2_CARRIER_SCHEMA_VERSION,
        prior,
        V2_SOURCE_KEYS,
        tuple((key, V2_SOURCE_LICENSES[key]) for key in V2_SOURCE_KEYS),
        tuple(sorted(V2_PUBLIC_LICENSES)),
        record_schema_bindings(),
        OWNER_POLICIES,
        V2_SPLIT_POLICY,
        V2_SCALE_BUDGETS,
        v2_run_policy(),
        V2_INVALIDATION_FORMAT_VERSION,
        V2_INVALIDATION_VERSION,
        V2_PACK_INVALIDATION_KINDS,
        V2_PACK_EARLIEST_STAGES,
        V2_UNKNOWN_INVALIDATION_POLICY,
        V2_EXECUTION_STAGES,
        build_v2_invalidation_rules(pack_earliest_stages),
        dict(V2_INITIAL_EXECUTION_STATE),
        "DATA_CONTRACT_FROZEN",
    )


def read_v2_successor_contract(
        repository_root: str | Path,
        path: str | Path | None = None,
        *,
        verify_parent: bool = True,
        ) -> V2SuccessorContract:
    """严格回读 v2 contract，并可逐字节重验公开 v1 parent receipt。"""
    root = Path(repository_root).resolve()
    target = _resolve(root, V2_CONTRACT_PATH) if path is None else Path(path).resolve()
    contract = V2SuccessorContract.from_dict(read_canonical_object(target))
    if verify_parent and contract.prior_release_receipt != _validate_prior_receipt(root):
        raise D03ContractError("v2 contract parent receipt 现场漂移")
    return contract


def publish_v2_successor_contract(
        repository_root: str | Path,
        path: str | Path | None = None,
        *,
        pack_earliest_stages: tuple[tuple[str, str], ...] = (),
        ) -> Path:
    """独占或幂等发布 v2 contract，拒绝同路径异字节覆盖。"""
    root = Path(repository_root).resolve()
    target = _resolve(root, V2_CONTRACT_PATH) if path is None else Path(path).resolve()
    contract = build_v2_successor_contract(
        root, pack_earliest_stages=pack_earliest_stages)
    write_immutable_json(contract.to_dict(), target)
    if read_v2_successor_contract(root, target) != contract:
        raise D03ContractError("v2 contract 发布后回读不一致")
    return target


__all__ = [
    "V2SuccessorContract",
    "build_v2_successor_contract",
    "publish_v2_successor_contract",
    "read_v2_successor_contract",
    "v2_run_policy",
    "validate_v2_run_policy",
]
