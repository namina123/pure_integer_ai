"""D-03 全局计划、资料可见性、恢复后缀和失效图的只读 reader。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    STAGE_KEYS,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_d03_global_contract import (
    D03GlobalManifest,
    D03PackBinding,
)
from pure_integer_ai.experiments.ph2_d03_invalidation import (
    InvalidationResult,
    StageInvalidationGraph,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import FORMAL_RECEIPT_PATH
from pure_integer_ai.experiments.ph2_d03_stage_contract import (
    D03StageManifest,
    validate_stage_manifest_set,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_dataset_manifest import ArtifactFileIdentity


VIEW_KINDS = ("candidate", "teacher", "evaluator")


@dataclass(frozen=True)
class StageVisibilityView:
    """返回一个阶段视图的允许/拒绝路径及读取前零 payload 计数。"""

    stage_key: str
    view_kind: str
    allowed_paths: tuple[str, ...]
    rejected_paths: tuple[str, ...]
    payload_reads: int
    payload_bytes: int


@dataclass(frozen=True)
class VisibleArtifactFile:
    """绑定一个可见路径到唯一 pack manifest 和正式文件身份。"""

    pack_key: str
    manifest_identity: D03FileIdentity
    relative_path: str
    file_identity: ArtifactFileIdentity


def _resolve(root: Path, relative: str) -> Path:
    """在单个根内安全解析 POSIX 相对路径。"""
    target = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(root):
        raise D03ContractError("D-03 reader 路径逃逸")
    return target


class _RepositoryOverlay:
    """优先读取候选输出根，缺失时只读回退公开依赖根。"""

    def __init__(self, primary: Path, dependency: Path) -> None:
        self._primary = primary
        self._dependency = dependency

    def path(self, relative: str) -> Path:
        """返回唯一存在的首选/依赖文件路径。"""
        primary = _resolve(self._primary, relative)
        if primary.is_file():
            return primary
        dependency = _resolve(self._dependency, relative)
        if dependency.is_file():
            return dependency
        raise D03ContractError(f"D-03 reader 文件缺失: {relative}")

    def verify(self, identity: D03FileIdentity) -> None:
        """在 overlay 中逐字节验证文件身份。"""
        path = self.path(identity.relative_path)
        payload = path.read_bytes()
        if (len(payload) != identity.size_bytes
                or hashlib.sha256(payload).hexdigest() != identity.sha256):
            raise D03ContractError("D-03 overlay 文件身份漂移")


class D03ReleaseReader:
    """以全局 manifest 为唯一真源提供只读阶段、可见性和失效解析。"""

    def __init__(
            self,
            overlay: _RepositoryOverlay,
            global_manifest: D03GlobalManifest,
            stages: tuple[D03StageManifest, ...],
            invalidation_graph: StageInvalidationGraph,
            ) -> None:
        self._overlay = overlay
        self.global_manifest = global_manifest
        self.stages = stages
        self.invalidation_graph = invalidation_graph
        self._packs = {
            item.pack_key: item for item in global_manifest.pack_bindings}
        self._pack_files: dict[str, tuple[VisibleArtifactFile, ...]] = {}

    @classmethod
    def open(
            cls,
            repository_root: str | Path,
            global_manifest_path: str,
            *,
            dependency_root: str | Path | None = None,
            require_publication: bool = True,
            ) -> "D03ReleaseReader":
        """严格回读全局、九阶段、失效图和全部直接文件身份。"""
        primary = Path(repository_root).resolve()
        dependency = (
            Path(dependency_root).resolve()
            if dependency_root is not None else primary
        )
        overlay = _RepositoryOverlay(primary, dependency)
        global_value = read_canonical_object(overlay.path(global_manifest_path))
        global_manifest = D03GlobalManifest.from_dict(global_value)
        cls._verify_release_dependencies(overlay, global_manifest)
        stages: list[D03StageManifest] = []
        for reference in global_manifest.stage_manifests:
            overlay.verify(reference.file_identity)
            stage = D03StageManifest.from_dict(read_canonical_object(
                overlay.path(reference.file_identity.relative_path)))
            if (stage.stage_identity.stage_key != reference.artifact_key
                    or stage.release_key != global_manifest.release_identity.release_key):
                raise D03ContractError("stage reference 与 release identity 漂移")
            stages.append(stage)
        stage_tuple = tuple(stages)
        validate_stage_manifest_set(stage_tuple)
        overlay.verify(global_manifest.invalidation_graph.file_identity)
        graph = StageInvalidationGraph.from_dict(read_canonical_object(
            overlay.path(global_manifest.invalidation_graph.file_identity.relative_path)))
        if graph.release_key != global_manifest.release_identity.release_key:
            raise D03ContractError("invalidation graph release identity 漂移")
        reader = cls(overlay, global_manifest, stage_tuple, graph)
        reader._verify_stage_pack_bindings()
        if require_publication:
            reader._verify_publication_presence()
        return reader

    @staticmethod
    def _verify_release_dependencies(
            overlay: _RepositoryOverlay,
            manifest: D03GlobalManifest,
            ) -> None:
        """回验 v4/v41/source coverage、held receipt、排除来源和论文身份。"""
        identity = manifest.release_identity
        direct = (
            D03FileIdentity(
                identity.parent_gate_path,
                len(overlay.path(identity.parent_gate_path).read_bytes()),
                identity.parent_gate_sha256,
            ),
            D03FileIdentity(
                identity.capability_baseline_path,
                len(overlay.path(identity.capability_baseline_path).read_bytes()),
                identity.capability_baseline_sha256,
            ),
            D03FileIdentity(
                identity.source_coverage_path,
                len(overlay.path(identity.source_coverage_path).read_bytes()),
                identity.source_coverage_sha256,
            ),
        )
        for item in (
                *direct,
                manifest.historical_hold_receipt,
                *manifest.paper_files,
                *(item.evidence_identity for item in manifest.excluded_sources)):
            overlay.verify(item)
        gate = read_canonical_object(overlay.path(identity.parent_gate_path))
        if (gate.get("artifact_status") != "PASS"
                or gate.get("d03_release_decision")
                != "ALLOW_FUTURE_CONFIRMED_SESSION_TO_PUBLISH_D03"
                or gate.get("d03_published") != 0):
            raise D03ContractError("parent v4 未授权未来 D-03 发布")
        baseline = read_canonical_object(
            overlay.path(identity.capability_baseline_path))
        state = baseline.get("execution_state")
        if (not isinstance(state, dict) or any(value != 0 for value in state.values())):
            raise D03ContractError("v41 execution state 非零")
        baseline_versions = baseline.get("version_keys")
        if not isinstance(baseline_versions, dict):
            raise D03ContractError("v41 version keys 缺失")
        identity_versions = dict(identity.version_keys)
        if baseline_versions != {
                key: identity_versions[key] for key in baseline_versions}:
            raise D03ContractError("v41 version keys 与 D-03 identity 漂移")
        hold = read_canonical_object(
            overlay.path(manifest.historical_hold_receipt.relative_path))
        if (hold.get("status") != "GIT_SNAPSHOT_PUBLISHED_D03_HELD"
                or hold.get("d03_published") != 0):
            raise D03ContractError("历史 publication receipt 不再是 D03_HELD")

    def _verify_stage_pack_bindings(self) -> None:
        """要求每阶段 pack 集精确等于按 earliest stage 推导的累计视图。"""
        pack_keys = set(self._packs)
        for stage in self.stages:
            stage_key = stage.stage_identity.stage_key
            rank = STAGE_KEYS.index(stage_key)
            available = tuple(
                item for item in self.global_manifest.pack_bindings
                if STAGE_KEYS.index(item.earliest_stage) <= rank
            )
            future = tuple(
                item for item in self.global_manifest.pack_bindings
                if STAGE_KEYS.index(item.earliest_stage) > rank
            )
            visibility = stage.data_visibility
            expected = {
                "train_pack_keys": {
                    item.pack_key for item in available
                    if item.train_observation_paths
                },
                "dev_pack_keys": {
                    item.pack_key for item in available
                    if item.dev_observation_paths
                },
                "held_out_pack_keys": {
                    item.pack_key for item in available
                    if item.held_out_observation_paths
                },
                "evaluator_pack_keys": {
                    item.pack_key for item in available
                    if item.evaluator_label_paths
                },
                "future_pack_keys": {item.pack_key for item in future},
            }
            for name, values in expected.items():
                actual = set(getattr(visibility, name))
                if actual != values or not actual.issubset(pack_keys):
                    raise D03ContractError("stage visibility 与 pack catalog 漂移")

    def _verify_publication_presence(self) -> None:
        """要求正式 post-publication receipt 存在；具体远端合同由发布模块回验。"""
        self._overlay.path(FORMAL_RECEIPT_PATH)

    def _pack_manifest(self, pack: D03PackBinding):
        """读取并验证一个 pack manifest 的文件身份。"""
        self._overlay.verify(pack.manifest_identity)
        return read_artifact_manifest(
            self._overlay.path(pack.manifest_identity.relative_path))

    def _pack_file_identities(
            self,
            pack: D03PackBinding,
            ) -> tuple[VisibleArtifactFile, ...]:
        """从正式 manifest 恢复 pack 全路径，并核对全局 binding 未漏文件。"""
        self._overlay.verify(pack.manifest_identity)
        cached = self._pack_files.get(pack.pack_key)
        if cached is not None:
            return cached
        manifest = self._pack_manifest(pack)
        prefix = PurePosixPath(pack.manifest_identity.relative_path).parent
        files = tuple(
            VisibleArtifactFile(
                pack.pack_key,
                pack.manifest_identity,
                PurePosixPath(prefix, item.relative_path).as_posix(),
                item,
            )
            for item in manifest.files
        )
        paths = tuple(item.relative_path for item in files)
        if len(set(paths)) != len(paths) or set(paths) != set(pack.payload_paths):
            raise D03ContractError("pack binding 未覆盖全部 owner 文件")
        self._pack_files[pack.pack_key] = files
        return files

    @staticmethod
    def _file_visible_in_view(
            identity: ArtifactFileIdentity,
            view_kind: str,
            ) -> bool:
        """按正式 owner_kind 和 split 判断单个文件能否进入指定视图。"""
        if identity.owner_kind == "source":
            return True
        if view_kind == "candidate":
            return identity.owner_kind == "observation" and identity.split == "train"
        if view_kind == "teacher":
            return (
                (identity.owner_kind == "observation" and identity.split == "train")
                or (identity.owner_kind == "teacher" and identity.split == "train")
            )
        return (
            (identity.owner_kind == "observation"
             and identity.split in {"dev", "held_out"})
            or (identity.owner_kind == "evaluator"
                and identity.split in {"dev", "held_out"})
        )

    def visible_file_identities(
            self,
            stage_key: str,
            view_kind: str,
            ) -> tuple[VisibleArtifactFile, ...]:
        """返回由正式 pack 文件身份直接授权的唯一可见路径清单。"""
        if stage_key not in STAGE_KEYS or view_kind not in VIEW_KINDS:
            raise D03ContractError("未知 stage/view kind")
        stage = self.stages[STAGE_KEYS.index(stage_key)]
        visible_keys = (
            stage.data_visibility.train_pack_keys
            if view_kind in {"candidate", "teacher"}
            else stage.data_visibility.evaluator_pack_keys
        )
        visible = tuple(
            item
            for key in visible_keys
            for item in self._pack_file_identities(self._packs[key])
            if self._file_visible_in_view(item.file_identity, view_kind)
        )
        paths = tuple(item.relative_path for item in visible)
        if len(set(paths)) != len(paths):
            raise D03ContractError("stage/view 可见路径映射不唯一")
        return tuple(sorted(visible, key=lambda item: item.relative_path))

    def verify_pack_files(self, pack_key: str) -> None:
        """按 pack manifest 的 transport hash 回验全部物理 owner 文件。"""
        pack = self._packs.get(pack_key)
        if pack is None:
            raise D03ContractError("未知 pack key")
        manifest = self._pack_manifest(pack)
        prefix = PurePosixPath(pack.manifest_identity.relative_path).parent
        expected_paths: set[str] = set()
        for item in manifest.files:
            relative = PurePosixPath(prefix, item.relative_path).as_posix()
            expected_paths.add(relative)
            path = self._overlay.path(relative)
            payload = path.read_bytes()
            if (len(payload) != item.transport_size_bytes
                    or hashlib.sha256(payload).hexdigest() != item.transport_sha256):
                raise D03ContractError("pack payload transport identity 漂移")
        if expected_paths != set(pack.payload_paths):
            raise D03ContractError("pack binding 未覆盖全部 owner 文件")

    def visibility(self, stage_key: str, view_kind: str) -> StageVisibilityView:
        """在不读取 payload 的前提下返回阶段视图的允许和拒绝路径。"""
        allowed = {
            item.relative_path
            for item in self.visible_file_identities(stage_key, view_kind)
        }
        all_payloads = {
            path for pack in self.global_manifest.pack_bindings
            for path in pack.payload_paths
        }
        return StageVisibilityView(
            stage_key,
            view_kind,
            tuple(sorted(allowed)),
            tuple(sorted(all_payloads - allowed)),
            0,
            0,
        )

    def require_visible_path(
            self,
            stage_key: str,
            view_kind: str,
            relative_path: str,
            ) -> Path:
        """在任何 payload 打开前要求路径属于当前阶段和 owner 视图。"""
        normalized = PurePosixPath(relative_path).as_posix()
        if normalized != relative_path or ".." in PurePosixPath(relative_path).parts:
            raise D03ContractError("路径不可见或不规范")
        view = self.visibility(stage_key, view_kind)
        if normalized not in set(view.allowed_paths):
            raise D03ContractError("路径在当前 stage/view 不可见")
        return self._overlay.path(normalized)

    def execution_suffix(self, stage_key: str, *, mode: str) -> tuple[str, ...]:
        """返回同一 D-03 identity 下 fresh/resume 共用的待执行后缀。"""
        if stage_key not in STAGE_KEYS or mode not in {"fresh", "resume"}:
            raise D03ContractError("未知 stage 或 fresh/resume mode")
        return STAGE_KEYS[STAGE_KEYS.index(stage_key):]

    def invalidation(self, change_kind: str, subject_key: str) -> InvalidationResult:
        """委托正式失效图解析变化的最早阶段和完整后缀。"""
        return self.invalidation_graph.invalidate(change_kind, subject_key)


__all__ = [
    "D03ReleaseReader",
    "StageVisibilityView",
    "VisibleArtifactFile",
    "VIEW_KINDS",
]
