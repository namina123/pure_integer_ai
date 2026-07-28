"""D-03 全局课程、pack 文件分账和历史排除证据合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    D03PublicationState,
    D03ReleaseIdentity,
    FORMAT_VERSION,
    STAGE_KEYS,
    exact_dict,
    nonnegative,
    string_tuple,
    text,
    validate_zero_execution_state,
)


GLOBAL_ARTIFACT_KIND = "PH2_D03_GLOBAL_COURSE_MANIFEST"
GLOBAL_ARTIFACT_STATUS = "CANDIDATE_VERIFIED"


@dataclass(frozen=True, order=True)
class D03ArtifactReference:
    """把逻辑 artifact key 绑定到一个不可变文件身份。"""

    artifact_key: str
    file_identity: D03FileIdentity

    def __post_init__(self) -> None:
        text(self.artifact_key, where="artifact reference key")
        if not isinstance(self.file_identity, D03FileIdentity):
            raise D03ContractError("artifact reference file identity 类型非法")

    def to_dict(self) -> dict[str, Any]:
        """导出 artifact 引用。"""
        return {
            "artifact_key": self.artifact_key,
            "file_identity": self.file_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03ArtifactReference":
        """从严格 object 恢复 artifact 引用。"""
        raw = exact_dict(value, {
            "artifact_key", "file_identity",
        }, where="D03ArtifactReference")
        return cls(
            str(raw["artifact_key"]),
            D03FileIdentity.from_dict(raw["file_identity"]),
        )


@dataclass(frozen=True, order=True)
class D03PackBinding:
    """冻结一个 D-02 pack 的 manifest、许可、阶段和物理 owner 路径。"""

    pack_key: str
    source_key: str
    license_id: str
    earliest_stage: str
    manifest_identity: D03FileIdentity
    source_ref_paths: tuple[str, ...]
    train_observation_paths: tuple[str, ...]
    dev_observation_paths: tuple[str, ...]
    held_out_observation_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]
    evaluator_label_paths: tuple[str, ...]
    total_record_count: int
    source_cluster_count: int

    def __post_init__(self) -> None:
        text(self.pack_key, where="pack key")
        text(self.source_key, where="pack source key")
        text(self.license_id, where="pack license")
        if self.earliest_stage not in STAGE_KEYS:
            raise D03ContractError("pack earliest stage 非法")
        if not isinstance(self.manifest_identity, D03FileIdentity):
            raise D03ContractError("pack manifest identity 类型非法")
        path_groups: list[set[str]] = []
        for name in (
                "source_ref_paths", "train_observation_paths",
                "dev_observation_paths", "held_out_observation_paths",
                "teacher_evidence_paths", "evaluator_label_paths"):
            normalized = tuple(sorted(string_tuple(
                getattr(self, name), where=name, allow_empty=True)))
            object.__setattr__(self, name, normalized)
            path_groups.append(set(normalized))
        all_paths: set[str] = set()
        for group in path_groups:
            if all_paths & group:
                raise D03ContractError("pack owner/split 物理路径交叉")
            all_paths.update(group)
        if not self.source_ref_paths:
            raise D03ContractError("pack 必须绑定 source refs")
        if not self.teacher_evidence_paths or not self.evaluator_label_paths:
            raise D03ContractError("pack 必须物理分离 teacher/evaluator owner")
        if any("/owners/teacher/" not in item for item in self.teacher_evidence_paths):
            raise D03ContractError("teacher Evidence 路径不在专属 owner")
        if any("/owners/evaluator/" not in item for item in self.evaluator_label_paths):
            raise D03ContractError("evaluator label 路径不在专属 owner")
        positive_counts = (self.total_record_count, self.source_cluster_count)
        if any(type(item) is not int or item <= 0 for item in positive_counts):
            raise D03ContractError("pack record/source cluster 计数必须为正")
        if not (
                self.train_observation_paths or self.dev_observation_paths
                or self.held_out_observation_paths):
            raise D03ContractError("pack 至少有一个 observation split")

    @property
    def payload_paths(self) -> tuple[str, ...]:
        """返回 pack 内全部非 manifest 物理文件路径。"""
        return tuple(sorted((
            *self.source_ref_paths,
            *self.train_observation_paths,
            *self.dev_observation_paths,
            *self.held_out_observation_paths,
            *self.teacher_evidence_paths,
            *self.evaluator_label_paths,
        )))

    def to_dict(self) -> dict[str, Any]:
        """导出 pack 绑定。"""
        return {
            "dev_observation_paths": list(self.dev_observation_paths),
            "earliest_stage": self.earliest_stage,
            "evaluator_label_paths": list(self.evaluator_label_paths),
            "held_out_observation_paths": list(self.held_out_observation_paths),
            "license_id": self.license_id,
            "manifest_identity": self.manifest_identity.to_dict(),
            "pack_key": self.pack_key,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "source_ref_paths": list(self.source_ref_paths),
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
            "total_record_count": self.total_record_count,
            "train_observation_paths": list(self.train_observation_paths),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03PackBinding":
        """从严格 object 恢复 pack 绑定。"""
        raw = exact_dict(value, {
            "dev_observation_paths", "earliest_stage",
            "evaluator_label_paths", "held_out_observation_paths",
            "license_id", "manifest_identity", "pack_key",
            "source_cluster_count", "source_key", "source_ref_paths",
            "teacher_evidence_paths", "total_record_count",
            "train_observation_paths",
        }, where="D03PackBinding")
        return cls(
            str(raw["pack_key"]), str(raw["source_key"]),
            str(raw["license_id"]), str(raw["earliest_stage"]),
            D03FileIdentity.from_dict(raw["manifest_identity"]),
            string_tuple(raw["source_ref_paths"], where="source paths"),
            string_tuple(
                raw["train_observation_paths"], where="train paths",
                allow_empty=True,
            ),
            string_tuple(
                raw["dev_observation_paths"], where="dev paths",
                allow_empty=True,
            ),
            string_tuple(
                raw["held_out_observation_paths"], where="held-out paths",
                allow_empty=True,
            ),
            string_tuple(raw["teacher_evidence_paths"], where="teacher paths"),
            string_tuple(raw["evaluator_label_paths"], where="evaluator paths"),
            raw["total_record_count"], raw["source_cluster_count"],
        )


@dataclass(frozen=True, order=True)
class D03ExcludedSource:
    """保留未进入课程的来源、许可 blocker 和历史证据身份。"""

    source_key: str
    status: str
    blocker_code: str
    evidence_identity: D03FileIdentity

    def __post_init__(self) -> None:
        text(self.source_key, where="excluded source key")
        if self.status != "BLOCKED":
            raise D03ContractError("excluded source 必须保持 BLOCKED")
        text(self.blocker_code, where="excluded source blocker")
        if not isinstance(self.evidence_identity, D03FileIdentity):
            raise D03ContractError("excluded source evidence identity 非法")

    def to_dict(self) -> dict[str, Any]:
        """导出排除来源证据。"""
        return {
            "blocker_code": self.blocker_code,
            "evidence_identity": self.evidence_identity.to_dict(),
            "source_key": self.source_key,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03ExcludedSource":
        """从严格 object 恢复排除来源证据。"""
        raw = exact_dict(value, {
            "blocker_code", "evidence_identity", "source_key", "status",
        }, where="D03ExcludedSource")
        return cls(
            str(raw["source_key"]), str(raw["status"]),
            str(raw["blocker_code"]),
            D03FileIdentity.from_dict(raw["evidence_identity"]),
        )


@dataclass(frozen=True)
class D03GlobalManifest:
    """合取发布身份、九阶段、全部 pack、失效图、排除证据和零执行状态。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    release_identity: D03ReleaseIdentity
    stage_manifests: tuple[D03ArtifactReference, ...]
    invalidation_graph: D03ArtifactReference
    pack_bindings: tuple[D03PackBinding, ...]
    excluded_sources: tuple[D03ExcludedSource, ...]
    historical_hold_receipt: D03FileIdentity
    paper_files: tuple[D03FileIdentity, ...]
    publication_state: D03PublicationState
    execution_state: dict[str, int]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise D03ContractError("global format_version 非法")
        if self.artifact_kind != GLOBAL_ARTIFACT_KIND:
            raise D03ContractError("global artifact_kind 非法")
        text(self.artifact_version, where="global artifact version")
        if self.artifact_status != GLOBAL_ARTIFACT_STATUS:
            raise D03ContractError("global candidate status 非法")
        if not isinstance(self.release_identity, D03ReleaseIdentity):
            raise D03ContractError("global release identity 类型非法")
        if (not isinstance(self.stage_manifests, tuple)
                or any(not isinstance(item, D03ArtifactReference)
                       for item in self.stage_manifests)):
            raise D03ContractError("global stage references 类型非法")
        stage_refs = tuple(sorted(self.stage_manifests))
        if tuple(item.artifact_key for item in stage_refs) != STAGE_KEYS:
            raise D03ContractError("global 必须精确引用九阶段")
        object.__setattr__(self, "stage_manifests", stage_refs)
        if (not isinstance(self.invalidation_graph, D03ArtifactReference)
                or self.invalidation_graph.artifact_key != "STAGE_INVALIDATION_GRAPH"):
            raise D03ContractError("global invalidation graph 引用非法")
        if (not isinstance(self.pack_bindings, tuple) or not self.pack_bindings
                or any(not isinstance(item, D03PackBinding)
                       for item in self.pack_bindings)):
            raise D03ContractError("global pack bindings 不能为空")
        packs = tuple(sorted(self.pack_bindings))
        if len({item.pack_key for item in packs}) != len(packs):
            raise D03ContractError("global pack key 重复")
        if any(item.source_key == "CC_CEDICT_20260725" for item in packs):
            raise D03ContractError("CC-CEDICT blocker 不得进入 D-03 pack")
        object.__setattr__(self, "pack_bindings", packs)
        if (not isinstance(self.excluded_sources, tuple)
                or not self.excluded_sources
                or any(not isinstance(item, D03ExcludedSource)
                       for item in self.excluded_sources)):
            raise D03ContractError("global excluded source 证据缺失")
        excluded = tuple(sorted(self.excluded_sources))
        if "CC_CEDICT_20260725" not in {item.source_key for item in excluded}:
            raise D03ContractError("global 必须保留 CC-CEDICT blocker")
        object.__setattr__(self, "excluded_sources", excluded)
        if not isinstance(self.historical_hold_receipt, D03FileIdentity):
            raise D03ContractError("historical hold receipt identity 非法")
        if (not isinstance(self.paper_files, tuple) or len(self.paper_files) != 2
                or any(not isinstance(item, D03FileIdentity)
                       for item in self.paper_files)):
            raise D03ContractError("global paper identities 不完整")
        papers = tuple(sorted(self.paper_files))
        if {item.relative_path for item in papers} != {"paper/main.pdf", "paper/main.tex"}:
            raise D03ContractError("global paper path 漂移")
        object.__setattr__(self, "paper_files", papers)
        if (not isinstance(self.publication_state, D03PublicationState)
                or self.publication_state.state != "CANDIDATE_VERIFIED"):
            raise D03ContractError("global 只能冻结 candidate publication state")
        object.__setattr__(self, "execution_state", validate_zero_execution_state(
            self.execution_state))

    def to_dict(self) -> dict[str, Any]:
        """导出规范全局课程 manifest。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "excluded_sources": [item.to_dict() for item in self.excluded_sources],
            "execution_state": dict(self.execution_state),
            "format_version": self.format_version,
            "historical_hold_receipt": self.historical_hold_receipt.to_dict(),
            "invalidation_graph": self.invalidation_graph.to_dict(),
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "paper_files": [item.to_dict() for item in self.paper_files],
            "publication_state": self.publication_state.to_dict(),
            "release_identity": self.release_identity.to_dict(),
            "stage_manifests": [item.to_dict() for item in self.stage_manifests],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03GlobalManifest":
        """从严格 object 恢复全局课程 manifest。"""
        raw = exact_dict(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "excluded_sources", "execution_state", "format_version",
            "historical_hold_receipt", "invalidation_graph", "pack_bindings",
            "paper_files", "publication_state", "release_identity",
            "stage_manifests",
        }, where="D03GlobalManifest")
        for key in ("excluded_sources", "pack_bindings", "paper_files", "stage_manifests"):
            if not isinstance(raw[key], list):
                raise D03ContractError(f"global {key} 必须是数组")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["artifact_status"]),
            D03ReleaseIdentity.from_dict(raw["release_identity"]),
            tuple(D03ArtifactReference.from_dict(item)
                  for item in raw["stage_manifests"]),
            D03ArtifactReference.from_dict(raw["invalidation_graph"]),
            tuple(D03PackBinding.from_dict(item) for item in raw["pack_bindings"]),
            tuple(D03ExcludedSource.from_dict(item)
                  for item in raw["excluded_sources"]),
            D03FileIdentity.from_dict(raw["historical_hold_receipt"]),
            tuple(D03FileIdentity.from_dict(item) for item in raw["paper_files"]),
            D03PublicationState.from_dict(raw["publication_state"]),
            raw["execution_state"],
        )


__all__ = [
    "D03ArtifactReference",
    "D03ExcludedSource",
    "D03GlobalManifest",
    "D03PackBinding",
    "GLOBAL_ARTIFACT_KIND",
    "GLOBAL_ARTIFACT_STATUS",
]
