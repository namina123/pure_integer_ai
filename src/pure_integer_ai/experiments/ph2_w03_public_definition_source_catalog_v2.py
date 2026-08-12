"""FT30 manifest 驱动的公开定义覆盖扩展来源目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackBuild,
    compile_or_resume_source_pack,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import SourcePackSpec
from pure_integer_ai.experiments.ph2_source_pack_mediawiki_targeted import (
    targeted_mediawiki_source_seeds_from_selection_v2,
    verify_targeted_mediawiki_raw_identities,
)
from pure_integer_ai.experiments.ph2_source_pack_runtime import SourcePackTask
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    read_ft30_public_definition_selection,
)


FT30_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT = Path(
    "ph2_ft30_dataset_artifacts/public_definition_source_v2")
FT30_PUBLIC_DEFINITION_SELECTION_MANIFEST = Path(
    "data/ph2/manifests/ft30_w03_public_definition_selection_v2.json")
FT30_WIKTIONARY_PACK_NAME = (
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v2")


# object-model: exception
class W03PublicDefinitionSourceCatalogV2Error(RuntimeError):
    """FT30 selection、snapshot、raw 或来源包身份发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(repository: Path, relative: str) -> Path:
    """解析仓内 POSIX 相对路径并拒绝缺失或逃逸。"""
    path = (repository / Path(*relative.split("/"))).resolve()
    if not path.is_relative_to(repository) or not path.is_file():
        raise W03PublicDefinitionSourceCatalogV2Error(
            "FT30 repository path 缺失或逃逸")
    return path


def build_ft30_public_definition_source_task_v2(
        repository_root: str | Path,
        raw_root: str | Path,
        ) -> SourcePackTask:
    """从冻结 selection 直接恢复 32 页公开 SourceRef/Observation。"""
    repository = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    if not (repository / "src/pure_integer_ai").is_dir() or not raw.is_dir():
        raise W03PublicDefinitionSourceCatalogV2Error(
            "FT30 repository/raw root 非法")
    selection_path = _repository_path(
        repository, FT30_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix())
    selection_sha256 = _sha256_path(selection_path)
    selection = read_ft30_public_definition_selection(selection_path)
    snapshot_path = _repository_path(
        repository, selection.snapshot_manifest_relative_path)
    base_path = _repository_path(
        repository, selection.base_selection_manifest_relative_path)
    if (_sha256_path(snapshot_path) != selection.snapshot_manifest_sha256
            or _sha256_path(base_path)
            != selection.base_selection_manifest_sha256):
        raise W03PublicDefinitionSourceCatalogV2Error(
            "FT30 snapshot/base selection SHA 漂移")
    snapshot = read_mediawiki_dump_snapshot(snapshot_path)
    verify_targeted_mediawiki_raw_identities(snapshot, raw_root=raw)
    seeds = targeted_mediawiki_source_seeds_from_selection_v2(
        snapshot,
        selection,
        raw_root=raw,
        selection_manifest_relative_path=(
            FT30_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix()),
        selection_manifest_sha256=selection_sha256,
    )
    if len(seeds) != len(selection.selected_titles):
        raise W03PublicDefinitionSourceCatalogV2Error(
            "FT30 source seed 数量漂移")
    xml = next(item for item in snapshot.raw_files if item.role == "XML")
    spec = SourcePackSpec(
        snapshot.source_key,
        snapshot.license_id,
        "PUBLIC",
        snapshot.snapshot_id,
        xml.official_url,
        snapshot.attribution_policy,
        selection.snapshot_manifest_relative_path,
        selection.snapshot_manifest_sha256,
        1,
        2,
        snapshot.adapter_version,
        2,
        snapshot.parser_version,
        FT30_WIKTIONARY_PACK_NAME,
        "W-03",
        "FT30-W03-PUBLIC-DEFINITION-ZHWIKTIONARY-V2",
        "W-03",
    )
    return SourcePackTask(1, spec, seeds)


def compile_ft30_public_definition_source_pack_v2(
        repository_root: str | Path,
        raw_root: str | Path,
        artifact_root: str | Path,
        ) -> tuple[SourcePackTask, SourcePackBuild]:
    """发布或严格恢复 FT30 v2 来源包，不启动训练。"""
    task = build_ft30_public_definition_source_task_v2(
        repository_root, raw_root)
    build = compile_or_resume_source_pack(
        task.spec, task.seeds, artifact_root)
    return task, build


__all__ = [
    "FT30_PUBLIC_DEFINITION_SELECTION_MANIFEST",
    "FT30_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT",
    "FT30_WIKTIONARY_PACK_NAME",
    "W03PublicDefinitionSourceCatalogV2Error",
    "build_ft30_public_definition_source_task_v2",
    "compile_ft30_public_definition_source_pack_v2",
]
