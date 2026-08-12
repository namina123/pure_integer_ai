"""FT31 manifest 驱动的公开定义规模切片来源目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
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
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v3 import (
    FT31PublicDefinitionSelectionManifest,
    read_ft31_public_definition_selection,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    read_ft30_public_definition_selection,
)


FT31_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT = Path(
    "ph2_ft31_dataset_artifacts/public_definition_source_v3")
FT31_PUBLIC_DEFINITION_SELECTION_MANIFEST = Path(
    "data/ph2/manifests/ft31_w03_public_definition_selection_v3.json")
FT31_WIKTIONARY_PACK_NAME = (
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v3")


# object-model: exception
class W03PublicDefinitionSourceCatalogV3Error(RuntimeError):
    """FT31 selection、predecessor、snapshot、raw 或来源包身份漂移。"""


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
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 repository path 缺失或逃逸")
    return path


def _ft26_wiktionary_titles(
        path: Path,
        *,
        snapshot_manifest_relative_path: str,
        ) -> tuple[str, ...]:
    """Strictly recover the FT26 Wiktionary title slice."""
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise W03PublicDefinitionSourceCatalogV3Error(
                "FT31 FT26 selection newline drift")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except W03PublicDefinitionSourceCatalogV3Error:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 FT26 selection unreadable") from error
    if (canonical_json_line(value) != payload
            or set(value) != {"format_version", "source_slices"}
            or value["format_version"] != 1
            or not isinstance(value["source_slices"], list)
            or len(value["source_slices"]) != 2):
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 FT26 selection contract drift")
    slice_keys = {
        "license_id", "selection_kind", "selections", "snapshot_manifest",
        "source_key", "split",
    }
    matches = []
    for item in value["source_slices"]:
        if not isinstance(item, dict) or set(item) != slice_keys:
            raise W03PublicDefinitionSourceCatalogV3Error(
                "FT31 FT26 source slice field drift")
        if (item["source_key"] == "ZHWIKTIONARY_20260701"
                and item["selection_kind"] == "TITLE"):
            matches.append(item)
    if len(matches) != 1:
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 FT26 Wiktionary title slice not unique")
    selected = matches[0]
    titles = selected["selections"]
    if (selected["license_id"] != "CC-BY-SA-4.0"
            or selected["snapshot_manifest"]
            != snapshot_manifest_relative_path
            or selected["split"] != "train"
            or not isinstance(titles, list)
            or not titles
            or any(not isinstance(item, str) or not item for item in titles)
            or len(set(titles)) != len(titles)):
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 FT26 Wiktionary title slice drift")
    return tuple(titles)


def verify_ft31_public_definition_predecessors(
        repository_root: str | Path,
        selection: FT31PublicDefinitionSelectionManifest,
        ) -> tuple[str, ...]:
    """Recompute the complete FT26+FT30 excluded-title commitment."""
    if not isinstance(selection, FT31PublicDefinitionSelectionManifest):
        raise TypeError("FT31 selection manifest type mismatch")
    repository = Path(repository_root).resolve()
    first_relative, second_relative = (
        selection.predecessor_selection_relative_paths)
    first_path = _repository_path(repository, first_relative)
    second_path = _repository_path(repository, second_relative)
    actual_sha256s = (
        _sha256_path(first_path),
        _sha256_path(second_path),
    )
    if actual_sha256s != selection.predecessor_selection_sha256s:
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 predecessor selection SHA drift")
    ft26_titles = _ft26_wiktionary_titles(
        first_path,
        snapshot_manifest_relative_path=(
            selection.snapshot_manifest_relative_path),
    )
    ft30 = read_ft30_public_definition_selection(second_path)
    if (ft30.base_selection_manifest_relative_path != first_relative
            or ft30.base_selection_manifest_sha256 != actual_sha256s[0]
            or ft30.source_key != selection.source_key
            or ft30.snapshot_id != selection.snapshot_id
            or ft30.snapshot_manifest_relative_path
            != selection.snapshot_manifest_relative_path
            or ft30.snapshot_manifest_sha256
            != selection.snapshot_manifest_sha256
            or ft30.index_raw_relative_path
            != selection.index_raw_relative_path
            or ft30.index_compressed_size_bytes
            != selection.index_compressed_size_bytes
            or ft30.index_local_sha256 != selection.index_local_sha256
            or ft30.index_upstream_sha1 != selection.index_upstream_sha1):
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 predecessor selection chain drift")
    titles = (
        *ft26_titles,
        *(item.title for item in ft30.selected_titles),
    )
    title_sha256s = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in titles
    ))
    if (len(titles) != len(set(titles))
            or title_sha256s != selection.excluded_title_sha256s):
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 predecessor excluded-title commitment drift")
    return title_sha256s


def build_ft31_public_definition_source_task_v3(
        repository_root: str | Path,
        raw_root: str | Path,
        ) -> SourcePackTask:
    """从冻结 selection 直接恢复 256 页公开 SourceRef/Observation。"""
    repository = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    if not (repository / "src/pure_integer_ai").is_dir() or not raw.is_dir():
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 repository/raw root 非法")
    selection_path = _repository_path(
        repository, FT31_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix())
    selection_sha256 = _sha256_path(selection_path)
    selection = read_ft31_public_definition_selection(selection_path)
    snapshot_path = _repository_path(
        repository, selection.snapshot_manifest_relative_path)
    predecessor_paths = tuple(
        _repository_path(repository, item)
        for item in selection.predecessor_selection_relative_paths)
    if (_sha256_path(snapshot_path) != selection.snapshot_manifest_sha256
            or tuple(_sha256_path(item) for item in predecessor_paths)
            != selection.predecessor_selection_sha256s):
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 snapshot/predecessor selection SHA 漂移")
    verify_ft31_public_definition_predecessors(repository, selection)
    snapshot = read_mediawiki_dump_snapshot(snapshot_path)
    verify_targeted_mediawiki_raw_identities(snapshot, raw_root=raw)
    seeds = targeted_mediawiki_source_seeds_from_selection_v2(
        snapshot,
        selection,
        raw_root=raw,
        selection_manifest_relative_path=(
            FT31_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix()),
        selection_manifest_sha256=selection_sha256,
    )
    if len(seeds) != len(selection.selected_titles) or len(seeds) != 256:
        raise W03PublicDefinitionSourceCatalogV3Error(
            "FT31 source seed 数量漂移")
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
        3,
        snapshot.adapter_version,
        3,
        snapshot.parser_version,
        FT31_WIKTIONARY_PACK_NAME,
        "W-03",
        "FT31-W03-PUBLIC-DEFINITION-ZHWIKTIONARY-V3",
        "W-03",
    )
    return SourcePackTask(1, spec, seeds)


def compile_ft31_public_definition_source_pack_v3(
        repository_root: str | Path,
        raw_root: str | Path,
        artifact_root: str | Path,
        ) -> tuple[SourcePackTask, SourcePackBuild]:
    """发布或严格恢复 FT31 v3 来源包，不启动训练。"""
    task = build_ft31_public_definition_source_task_v3(
        repository_root, raw_root)
    build = compile_or_resume_source_pack(
        task.spec, task.seeds, artifact_root)
    return task, build


__all__ = [
    "FT31_PUBLIC_DEFINITION_SELECTION_MANIFEST",
    "FT31_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT",
    "FT31_WIKTIONARY_PACK_NAME",
    "W03PublicDefinitionSourceCatalogV3Error",
    "build_ft31_public_definition_source_task_v3",
    "compile_ft31_public_definition_source_pack_v3",
    "verify_ft31_public_definition_predecessors",
]
