"""FT33 manifest-driven public-definition v4 source catalog."""
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
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    read_ft30_public_definition_selection,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v3 import (
    read_ft31_public_definition_selection,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v4 import (
    FT33_EXCLUDED_TITLE_COUNT,
    FT33_MAX_SELECTED_TITLES,
    FT33PublicDefinitionSelectionManifest,
    read_ft33_public_definition_selection,
)


FT33_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT = Path(
    "ph2_ft33_dataset_artifacts/public_definition_source_v4")
FT33_PUBLIC_DEFINITION_SELECTION_MANIFEST = Path(
    "data/ph2/manifests/ft33_w03_public_definition_selection_v4.json")
FT33_WIKTIONARY_PACK_NAME = (
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v4")


# object-model: exception
class W03PublicDefinitionSourceCatalogV4Error(RuntimeError):
    """FT33 predecessor, snapshot, raw, or source-pack identity drifted."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(repository: Path, relative: str) -> Path:
    path = (repository / Path(*relative.split("/"))).resolve()
    if not path.is_relative_to(repository) or not path.is_file():
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 repository path is missing or escaped")
    return path


def _ft26_wiktionary_titles(
        path: Path,
        *,
        snapshot_manifest_relative_path: str,
        ) -> tuple[str, ...]:
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise W03PublicDefinitionSourceCatalogV4Error(
                "FT33 FT26 selection newline drifted")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except W03PublicDefinitionSourceCatalogV4Error:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 FT26 selection is unreadable") from error
    if (
        canonical_json_line(value) != payload
        or set(value) != {"format_version", "source_slices"}
        or value["format_version"] != 1
        or not isinstance(value["source_slices"], list)
        or len(value["source_slices"]) != 2
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 FT26 selection contract drifted")
    slice_keys = {
        "license_id", "selection_kind", "selections", "snapshot_manifest",
        "source_key", "split",
    }
    matches = []
    for item in value["source_slices"]:
        if not isinstance(item, dict) or set(item) != slice_keys:
            raise W03PublicDefinitionSourceCatalogV4Error(
                "FT33 FT26 source slice fields drifted")
        if (
            item["source_key"] == "ZHWIKTIONARY_20260701"
            and item["selection_kind"] == "TITLE"
        ):
            matches.append(item)
    if len(matches) != 1:
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 FT26 Wiktionary slice is not unique")
    selected = matches[0]
    titles = selected["selections"]
    if (
        selected["license_id"] != "CC-BY-SA-4.0"
        or selected["snapshot_manifest"] != snapshot_manifest_relative_path
        or selected["split"] != "train"
        or not isinstance(titles, list)
        or len(titles) != 5
        or any(not isinstance(item, str) or not item for item in titles)
        or len(set(titles)) != 5
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 FT26 Wiktionary title inventory drifted")
    return tuple(titles)


def recover_ft33_public_definition_predecessor_titles(
        repository_root: str | Path,
        selection: FT33PublicDefinitionSelectionManifest,
        ) -> tuple[str, ...]:
    """Strictly recover FT26+FT30+FT31 and recompute all 293 exclusions."""
    if not isinstance(selection, FT33PublicDefinitionSelectionManifest):
        raise TypeError("FT33 selection manifest type mismatch")
    repository = Path(repository_root).resolve()
    relative_paths = selection.predecessor_selection_relative_paths
    paths = tuple(_repository_path(repository, item)
                  for item in relative_paths)
    actual_sha256s = tuple(_sha256_path(item) for item in paths)
    if actual_sha256s != selection.predecessor_selection_sha256s:
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 predecessor selection SHA drifted")
    ft26_titles = _ft26_wiktionary_titles(
        paths[0],
        snapshot_manifest_relative_path=(
            selection.snapshot_manifest_relative_path),
    )
    ft30 = read_ft30_public_definition_selection(paths[1])
    ft31 = read_ft31_public_definition_selection(paths[2])
    shared_ft30 = (
        ft30.source_key,
        ft30.snapshot_id,
        ft30.snapshot_manifest_relative_path,
        ft30.snapshot_manifest_sha256,
        ft30.index_raw_relative_path,
        ft30.index_compressed_size_bytes,
        ft30.index_local_sha256,
        ft30.index_upstream_sha1,
    )
    shared_ft33 = (
        selection.source_key,
        selection.snapshot_id,
        selection.snapshot_manifest_relative_path,
        selection.snapshot_manifest_sha256,
        selection.index_raw_relative_path,
        selection.index_compressed_size_bytes,
        selection.index_local_sha256,
        selection.index_upstream_sha1,
    )
    if (
        ft30.base_selection_manifest_relative_path != relative_paths[0]
        or ft30.base_selection_manifest_sha256 != actual_sha256s[0]
        or shared_ft30 != shared_ft33
        or ft31.predecessor_selection_relative_paths != relative_paths[:2]
        or ft31.predecessor_selection_sha256s != actual_sha256s[:2]
        or (
            ft31.source_key, ft31.snapshot_id,
            ft31.snapshot_manifest_relative_path,
            ft31.snapshot_manifest_sha256,
            ft31.index_raw_relative_path,
            ft31.index_compressed_size_bytes,
            ft31.index_local_sha256,
            ft31.index_upstream_sha1,
        ) != shared_ft33
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 predecessor selection chain drifted")
    ft26_sha256s = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in ft26_titles))
    ft26_ft30_titles = (
        *ft26_titles,
        *(item.title for item in ft30.selected_titles),
    )
    ft26_ft30_sha256s = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in ft26_ft30_titles))
    if (
        ft30.excluded_title_sha256s != ft26_sha256s
        or ft31.excluded_title_sha256s != ft26_ft30_sha256s
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 intermediate excluded-title commitment drifted")
    titles = (
        *ft26_ft30_titles,
        *(item.title for item in ft31.selected_titles),
    )
    title_sha256s = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in titles))
    if (
        len(titles) != FT33_EXCLUDED_TITLE_COUNT
        or len(set(titles)) != FT33_EXCLUDED_TITLE_COUNT
        or title_sha256s != selection.excluded_title_sha256s
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 predecessor excluded-title commitment drifted")
    return titles


def build_ft33_public_definition_source_task_v4(
        repository_root: str | Path,
        raw_root: str | Path,
        ) -> SourcePackTask:
    """Recover 512 public SourceRef/Observation seeds from frozen blocks."""
    repository = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    if not (repository / "src/pure_integer_ai").is_dir() or not raw.is_dir():
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 repository or raw root is invalid")
    selection_path = _repository_path(
        repository, FT33_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix())
    selection_sha256 = _sha256_path(selection_path)
    selection = read_ft33_public_definition_selection(selection_path)
    snapshot_path = _repository_path(
        repository, selection.snapshot_manifest_relative_path)
    if _sha256_path(snapshot_path) != selection.snapshot_manifest_sha256:
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 snapshot manifest SHA drifted")
    recover_ft33_public_definition_predecessor_titles(repository, selection)
    snapshot = read_mediawiki_dump_snapshot(snapshot_path)
    verify_targeted_mediawiki_raw_identities(snapshot, raw_root=raw)
    seeds = targeted_mediawiki_source_seeds_from_selection_v2(
        snapshot,
        selection,
        raw_root=raw,
        selection_manifest_relative_path=(
            FT33_PUBLIC_DEFINITION_SELECTION_MANIFEST.as_posix()),
        selection_manifest_sha256=selection_sha256,
    )
    if (
        len(seeds) != len(selection.selected_titles)
        or len(seeds) != FT33_MAX_SELECTED_TITLES
    ):
        raise W03PublicDefinitionSourceCatalogV4Error(
            "FT33 source seed inventory drifted")
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
        4,
        snapshot.adapter_version,
        4,
        snapshot.parser_version,
        FT33_WIKTIONARY_PACK_NAME,
        "W-03",
        "FT33-W03-PUBLIC-DEFINITION-ZHWIKTIONARY-V4",
        "W-03",
    )
    return SourcePackTask(1, spec, seeds)


def compile_ft33_public_definition_source_pack_v4(
        repository_root: str | Path,
        raw_root: str | Path,
        artifact_root: str | Path,
        ) -> tuple[SourcePackTask, SourcePackBuild]:
    """Publish or strictly resume the FT33 v4 source pack without training."""
    task = build_ft33_public_definition_source_task_v4(
        repository_root, raw_root)
    build = compile_or_resume_source_pack(task.spec, task.seeds, artifact_root)
    return task, build


__all__ = [
    "FT33_PUBLIC_DEFINITION_SELECTION_MANIFEST",
    "FT33_PUBLIC_DEFINITION_SOURCE_ARTIFACT_ROOT",
    "FT33_WIKTIONARY_PACK_NAME",
    "W03PublicDefinitionSourceCatalogV4Error",
    "build_ft33_public_definition_source_task_v4",
    "compile_ft33_public_definition_source_pack_v4",
    "recover_ft33_public_definition_predecessor_titles",
]
