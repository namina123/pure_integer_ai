"""FT26 真实公开词义来源切片与 SourceRef/Observation pack catalog。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackBuild,
    compile_or_resume_source_pack,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
    SourcePackSpec,
)
from pure_integer_ai.experiments.ph2_source_pack_mediawiki_targeted import (
    targeted_mediawiki_source_seeds,
    verify_targeted_mediawiki_raw_identities,
)
from pure_integer_ai.experiments.ph2_source_pack_runtime import SourcePackTask
from pure_integer_ai.experiments.ph2_wikidata_snapshot import (
    read_wikidata_revision_snapshot,
)


FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT = Path(
    "ph2_ft26_dataset_artifacts/public_sense_source_v1")
FT26_PUBLIC_SENSE_SELECTION_MANIFEST = Path(
    "data/ph2/manifests/ft26_w03_public_sense_source_selection_v1.json")
FT26_WIKTIONARY_PACK_NAME = (
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-sense-v1")
FT26_WIKIDATA_PACK_NAME = (
    "WIKIDATA_REVISION_V1--CC0-1.0--w03-public-sense-v1")


# object-model: exception
class W03PublicSenseSourceCatalogError(RuntimeError):
    """FT26 来源选择、snapshot 或 raw 字节发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_identity(repo: Path, relative_path: str) -> tuple[Path, str]:
    """返回仓库内 snapshot manifest 与当前字节身份。"""
    path = (repo / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise W03PublicSenseSourceCatalogError(
            "FT26 snapshot manifest 缺失或逃逸")
    return path, _sha256_path(path)


def _axes_parts(axes: dict[str, str]) -> tuple[str, ...]:
    """把完整组合轴展平为规范 key/value 元组。"""
    return tuple(
        value
        for key in sorted(axes)
        for value in (key, axes[key])
    )


def _selection_manifest(
        repo: Path,
        ) -> tuple[dict[str, dict[str, object]], str]:
    """读取字段精确、来源唯一的 FT26 source-selection manifest。"""
    path = (repo / FT26_PUBLIC_SENSE_SELECTION_MANIFEST).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise W03PublicSenseSourceCatalogError(
            "FT26 source-selection manifest 缺失或逃逸")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicSenseSourceCatalogError(
            "FT26 source-selection manifest JSON 非法") from error
    if (not isinstance(raw, dict)
            or set(raw) != {"format_version", "source_slices"}
            or raw["format_version"] != 1
            or not isinstance(raw["source_slices"], list)
            or len(raw["source_slices"]) != 2):
        raise W03PublicSenseSourceCatalogError(
            "FT26 source-selection manifest envelope 漂移")
    values = {}
    expected_keys = {
        "license_id", "selection_kind", "selections",
        "snapshot_manifest", "source_key", "split",
    }
    for item in raw["source_slices"]:
        if (not isinstance(item, dict) or set(item) != expected_keys
                or not isinstance(item["source_key"], str)
                or not isinstance(item["selections"], list)
                or not item["selections"]
                or any(not isinstance(value, str) or not value
                       for value in item["selections"])
                or len(set(item["selections"])) != len(item["selections"])
                or item["split"] != "train"):
            raise W03PublicSenseSourceCatalogError(
                "FT26 source-selection slice 漂移")
        source_key = item["source_key"]
        if source_key in values:
            raise W03PublicSenseSourceCatalogError(
                "FT26 source-selection source_key 重复")
        values[source_key] = item
    if set(values) != {"WIKIDATA_REVISION_V1", "ZHWIKTIONARY_20260701"}:
        raise W03PublicSenseSourceCatalogError(
            "FT26 source-selection 来源集合漂移")
    return values, _sha256_path(path)


def _spec(
        *,
        source_key: str,
        license_id: str,
        snapshot_id: str,
        official_url: str,
        attribution: str,
        snapshot_relative_path: str,
        snapshot_sha256: str,
        adapter_version: int,
        parser_version: int,
        pack_name: str,
        substage: str,
        ) -> SourcePackSpec:
    """构造 FT26 W-03 public capability 专用不可变 pack spec。"""
    return SourcePackSpec(
        source_key,
        license_id,
        "PUBLIC",
        snapshot_id,
        official_url,
        attribution,
        snapshot_relative_path,
        snapshot_sha256,
        1,
        1,
        adapter_version,
        1,
        parser_version,
        pack_name,
        "W-03",
        substage,
        "W-03",
    )


def _wiktionary_task(
        repo: Path,
        raw_root: Path,
        selection: dict[str, object],
        selection_sha256: str,
        ) -> SourcePackTask:
    """从双遍 snapshot 按 index block 取得五个真实词典页。"""
    if (selection["selection_kind"] != "TITLE"
            or selection["license_id"] != "CC-BY-SA-4.0"):
        raise W03PublicSenseSourceCatalogError(
            "FT26 Wiktionary selection 合同漂移")
    manifest_rel = str(selection["snapshot_manifest"])
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_mediawiki_dump_snapshot(manifest_path)
    verify_targeted_mediawiki_raw_identities(
        manifest, raw_root=raw_root)
    selected = targeted_mediawiki_source_seeds(
        manifest,
        raw_root=raw_root,
        titles=tuple(str(item) for item in selection["selections"]),
        split="train",
    )
    seeds = []
    for seed in selected:
        span = seed.source_span.to_value()
        span.update({
            "selection_manifest_relative_path": (
                FT26_PUBLIC_SENSE_SELECTION_MANIFEST.as_posix()),
            "selection_manifest_sha256": selection_sha256,
        })
        seeds.append(replace(
            seed,
            source_span=CanonicalJsonObject.from_value(span),
        ))
    xml = next(item for item in manifest.raw_files if item.role == "XML")
    return SourcePackTask(1, _spec(
        source_key=manifest.source_key,
        license_id=manifest.license_id,
        snapshot_id=manifest.snapshot_id,
        official_url=xml.official_url,
        attribution=manifest.attribution_policy,
        snapshot_relative_path=manifest_rel,
        snapshot_sha256=manifest_sha256,
        adapter_version=manifest.adapter_version,
        parser_version=manifest.parser_version,
        pack_name=FT26_WIKTIONARY_PACK_NAME,
        substage="FT26-W03-PUBLIC-SENSE-ZHWIKTIONARY-V1",
    ), tuple(seeds))


def _wikidata_seeds(
        manifest,
        raw_root: Path,
        selections: tuple[str, ...],
        selection_sha256: str,
        ) -> tuple[SourceObservationSeed, ...]:
    """把选中的 revision-pinned 实体重分为 FT26 public capability train。"""
    by_qid = {item.qid: item for item in manifest.entities}
    if any(qid not in by_qid for qid in selections):
        raise W03PublicSenseSourceCatalogError(
            "FT26 Wikidata 目标 QID 不在 snapshot")
    seeds = []
    for ordinal, qid in enumerate(selections, start=1):
        entity = by_qid[qid]
        raw = (raw_root / Path(*entity.raw_relative_path.split("/"))).resolve()
        if not raw.is_relative_to(raw_root) or not raw.is_file():
            raise W03PublicSenseSourceCatalogError(
                "FT26 Wikidata raw 缺失或逃逸")
        if (raw.stat().st_size != entity.raw_size_bytes
                or _sha256_path(raw) != entity.raw_sha256):
            raise W03PublicSenseSourceCatalogError(
                "FT26 Wikidata raw size/SHA-256 漂移")
        raw_text = raw.read_text(encoding="utf-8")
        axes = {
            "code_switch": "NONE",
            "dialect": "UNASSESSED",
            "domain": "knowledge_graph",
            "era": manifest.snapshot_id,
            "genre": "entitydata",
            "language": "multilingual_with_zh_allowlist",
            "length": "ENTITY",
            "register": "structured",
            "script_orthography": "JSON_ENTITYDATA",
            "source": manifest.source_key,
            "source_document_cluster": entity.cluster_id,
        }
        seeds.append(SourceObservationSeed(
            f"{entity.qid}-revision-{entity.revision}",
            "train",
            "zh",
            "wikidata-entity-json-raw",
            entity.raw_relative_path,
            "sha256:" + entity.raw_sha256,
            entity.raw_sha256,
            CanonicalJsonObject.from_value({
                "cluster_id": entity.cluster_id,
                "original_split": entity.split,
                "purpose_keys": list(entity.purpose_keys),
                "qid": entity.qid,
                "raw_relative_path": entity.raw_relative_path,
                "revision": entity.revision,
                "response_url": entity.http.response_url,
                "selection_manifest_relative_path": (
                    FT26_PUBLIC_SENSE_SELECTION_MANIFEST.as_posix()),
                "selection_manifest_sha256": selection_sha256,
            }),
            CanonicalJsonObject.from_value({
                "entity_json_utf8": raw_text,
                "qid": entity.qid,
                "revision": entity.revision,
            }),
            CanonicalJsonObject.from_value(axes),
            ("entity_cluster", entity.cluster_id),
            ("raw", entity.raw_sha256),
            ("entity", entity.qid, entity.revision),
            ("entitydata", *entity.purpose_keys),
            (
                "statement_count", entity.parser_report.statement_count,
                "label_language_count",
                entity.parser_report.label_language_count,
            ),
            _axes_parts(axes),
            "support",
            "NONE",
            ordinal,
        ))
    return tuple(seeds)


def _wikidata_task(
        repo: Path,
        raw_root: Path,
        selection: dict[str, object],
        selection_sha256: str,
        ) -> SourcePackTask:
    """从 pinned snapshot 取得选中的真实 label/alias/description 实体。"""
    if (selection["selection_kind"] != "QID"
            or selection["license_id"] != "CC0-1.0"):
        raise W03PublicSenseSourceCatalogError(
            "FT26 Wikidata selection 合同漂移")
    manifest_rel = str(selection["snapshot_manifest"])
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_wikidata_revision_snapshot(manifest_path)
    seeds = _wikidata_seeds(
        manifest,
        raw_root,
        tuple(str(item) for item in selection["selections"]),
        selection_sha256,
    )
    return SourcePackTask(2, _spec(
        source_key=manifest.source_key,
        license_id=manifest.license_id,
        snapshot_id=manifest.snapshot_id,
        official_url="https://www.wikidata.org/wiki/Special:EntityData",
        attribution=manifest.attribution,
        snapshot_relative_path=manifest_rel,
        snapshot_sha256=manifest_sha256,
        adapter_version=manifest.adapter_version,
        parser_version=manifest.parser_version,
        pack_name=FT26_WIKIDATA_PACK_NAME,
        substage="FT26-W03-PUBLIC-SENSE-WIKIDATA-V1",
    ), seeds)


def build_ft26_public_sense_source_tasks(
        repository_root: str | Path,
        raw_root: str | Path,
        ) -> tuple[SourcePackTask, ...]:
    """构造唯一有序的 Wiktionary/Wikidata FT26 来源任务。"""
    repo = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    if not (repo / "src/pure_integer_ai").is_dir() or not raw.is_dir():
        raise W03PublicSenseSourceCatalogError(
            "FT26 repository/raw root 非法")
    selection, selection_sha256 = _selection_manifest(repo)
    tasks = (
        _wiktionary_task(
            repo, raw, selection["ZHWIKTIONARY_20260701"],
            selection_sha256),
        _wikidata_task(
            repo, raw, selection["WIKIDATA_REVISION_V1"],
            selection_sha256),
    )
    if tuple(item.pack_id for item in tasks) != (1, 2):
        raise W03PublicSenseSourceCatalogError("FT26 task id 漂移")
    return tasks


def compile_ft26_public_sense_source_packs(
        repository_root: str | Path,
        raw_root: str | Path,
        artifact_root: str | Path,
        ) -> tuple[tuple[SourcePackTask, SourcePackBuild], ...]:
    """发布或严格恢复两个 public source pack，不启动训练。"""
    tasks = build_ft26_public_sense_source_tasks(repository_root, raw_root)
    return tuple(
        (task, compile_or_resume_source_pack(
            task.spec, task.seeds, artifact_root))
        for task in tasks
    )


__all__ = [
    "FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT",
    "FT26_PUBLIC_SENSE_SELECTION_MANIFEST",
    "FT26_WIKIDATA_PACK_NAME",
    "FT26_WIKTIONARY_PACK_NAME",
    "W03PublicSenseSourceCatalogError",
    "build_ft26_public_sense_source_tasks",
    "compile_ft26_public_sense_source_packs",
]
