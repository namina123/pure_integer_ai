"""D-02 已冻结外部来源到统一 pack task/覆盖账的只读 catalog。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_conceptnet_adapter import (
    read_conceptnet_sample,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_git_snapshot import (
    read_git_snapshot_manifest,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    read_raw_snapshot_manifest,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackBuild,
    compile_or_resume_source_pack,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
    SourcePackCoverageEntry,
    SourcePackCoverageManifest,
    SourcePackSpec,
)
from pure_integer_ai.experiments.ph2_source_pack_mediawiki import (
    bounded_mediawiki_source_seeds,
)
from pure_integer_ai.experiments.ph2_source_pack_runtime import SourcePackTask
from pure_integer_ai.experiments.ph2_ud_gsdsimp_adapter import scan_ud_conllu
from pure_integer_ai.experiments.ph2_wikidata_snapshot import (
    read_wikidata_revision_snapshot,
)


SOURCE_PACK_COVERAGE_PATH = Path(
    "data/ph2/manifests/d02_source_pack_coverage_v1.json")
SOURCE_PACK_ARTIFACT_RELATIVE_ROOT = "ph2_dataset_artifacts/d02_source_pack_v1"


class SourcePackCatalogError(RuntimeError):
    """正式来源、sample、raw root 或覆盖清单发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _axes_parts(axes: dict[str, str]) -> tuple[str, ...]:
    """把完整组合轴展平为规范 key/value 元组。"""
    return tuple(
        value
        for key in sorted(axes)
        for value in (key, axes[key])
    )


def _snapshot_identity(repo: Path, relative_path: str) -> tuple[Path, str]:
    """返回仓库内 snapshot manifest 路径和当前 SHA-256。"""
    path = (repo / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise SourcePackCatalogError("source snapshot manifest 缺失或逃逸")
    return path, _sha256_path(path)


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
        stage: str,
        substage: str,
        ) -> SourcePackSpec:
    """构造所有正式来源共享的不可变 pack spec。"""
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
        stage,
        substage,
        stage,
    )


def _ud_task(repo: Path) -> SourcePackTask:
    """把已核准 UD sample 编译为 raw Observation 来源 pack task。"""
    manifest_rel = "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json"
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_git_snapshot_manifest(manifest_path)
    sample_rel = "data/ph2/ud_zh_gsdsimp_r2_18_dev_s2_v1.conllu.sample"
    sample = repo / Path(*sample_rel.split("/"))
    sample_sha256 = _sha256_path(sample)
    report = scan_ud_conllu(
        sample,
        relative_path=sample_rel,
        split="dev",
        expected_sha256=sample_sha256,
    )
    if report.anomaly_count != 0 or report.sentence_count != 1:
        raise SourcePackCatalogError("UD source sample parser report 漂移")
    raw_text = sample.read_text(encoding="utf-8")
    axes = {
        "code_switch": "NONE",
        "dialect": "UNASSESSED",
        "domain": "dependency_treebank",
        "era": "r2.18",
        "genre": "annotated_sentence",
        "language": "zh",
        "length": "SHORT",
        "register": "UNASSESSED",
        "script_orthography": "SIMPLIFIED_CHINESE",
        "source": manifest.source_key,
        "source_document_cluster": "dev-s2",
    }
    seed = SourceObservationSeed(
        "ud-dev-s2",
        "held_out",
        "zh",
        "conllu-raw-sentence",
        sample_rel + "#sent_id=dev-s2",
        "sha1:" + manifest.commit_sha1,
        sample_sha256,
        CanonicalJsonObject.from_value({
            "parser_event_sha256": report.event_sha256,
            "relative_path": sample_rel,
            "sentence_count": report.sentence_count,
            "split_upstream": "dev",
            "word_count": report.word_count,
        }),
        CanonicalJsonObject.from_value({
            "conllu_text": raw_text,
            "sent_id": "dev-s2",
        }),
        CanonicalJsonObject.from_value(axes),
        ("sentence", "dev-s2"),
        ("sample", sample_sha256),
        ("sentence", "dev-s2"),
        ("conllu", "sentence"),
        ("word_count", report.word_count),
        _axes_parts(axes),
        "read_only_probe",
        "NONE",
        1,
    )
    spec = _spec(
        source_key=manifest.source_key,
        license_id=manifest.license_id,
        snapshot_id=f"{manifest.tag}-{manifest.commit_sha1}",
        official_url=manifest.repository_url,
        attribution="Universal Dependencies contributors; retain sentence id and r2.18 commit",
        snapshot_relative_path=manifest_rel,
        snapshot_sha256=manifest_sha256,
        adapter_version=manifest.adapter_version,
        parser_version=manifest.parser_version,
        pack_name="UD_ZH_GSDSIMP_R2_18--CC-BY-SA-4.0--source-pack-v1",
        stage="W-02",
        substage="D-02-SOURCE-UD-V1",
    )
    return SourcePackTask(1, spec, (seed,))


def _conceptnet_tasks(repo: Path) -> tuple[SourcePackTask, SourcePackTask]:
    """按 assertion 许可把两个 ConceptNet sample 编成物理独立 task。"""
    manifest_rel = "data/ph2/manifests/conceptnet_5_7_0.raw_snapshot.json"
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_raw_snapshot_manifest(manifest_path)
    samples = (
        (
            3,
            "CC-BY-4.0",
            "data/ph2/conceptnet_5_7_0_cc_by_4_0_zh_v1.csv.sample",
            "CONCEPTNET_5_7_0--CC-BY-4.0--source-pack-v1",
        ),
        (
            4,
            "CC-BY-SA-4.0",
            "data/ph2/conceptnet_5_7_0_cc_by_sa_4_0_zh_v1.csv.sample",
            "CONCEPTNET_5_7_0--CC-BY-SA-4.0--source-pack-v1",
        ),
    )
    tasks: list[SourcePackTask] = []
    for pack_id, license_id, sample_rel, pack_name in samples:
        sample = repo / Path(*sample_rel.split("/"))
        sample_sha256 = _sha256_path(sample)
        assertions = read_conceptnet_sample(sample)
        if (not assertions
                or any(item.license_partition != license_id for item in assertions)):
            raise SourcePackCatalogError("ConceptNet sample 许可分区漂移")
        seeds: list[SourceObservationSeed] = []
        for ordinal, assertion in enumerate(assertions, start=1):
            axes = {
                "code_switch": "NONE",
                "dialect": "UNASSESSED",
                "domain": "commonsense_graph",
                "era": manifest.snapshot_id,
                "genre": "external_assertion",
                "language": "zh",
                "length": "ATOMIC",
                "register": "UNASSESSED",
                "script_orthography": "ZH_CONCEPT_URI",
                "source": manifest.source_key,
                "source_document_cluster": assertion.source_cluster_sha256,
            }
            seeds.append(SourceObservationSeed(
                f"assertion-line-{ordinal}-{assertion.metadata_sha256[:16]}",
                "held_out",
                "zh",
                "conceptnet-assertion",
                sample_rel + f"#line={ordinal}",
                "sha256:" + manifest.raw_sha256,
                sample_sha256,
                CanonicalJsonObject.from_value({
                    "assertion_uri": assertion.assertion_uri,
                    "line_number": ordinal,
                    "raw_snapshot_relative_path": manifest.raw_relative_path,
                    "sample_relative_path": sample_rel,
                }),
                CanonicalJsonObject.from_value(assertion.to_dict()),
                CanonicalJsonObject.from_value(axes),
                ("external_sources", assertion.source_cluster_sha256),
                ("assertion_uri", assertion.assertion_uri),
                ("metadata", assertion.metadata_sha256),
                ("relation", assertion.relation),
                ("endpoint_kinds", assertion.start.kind, assertion.end.kind),
                _axes_parts(axes),
                "read_only_probe",
                "NONE",
                ordinal,
            ))
        spec = _spec(
            source_key=manifest.source_key,
            license_id=license_id,
            snapshot_id=manifest.snapshot_id,
            official_url=manifest.official_url,
            attribution=manifest.attribution,
            snapshot_relative_path=manifest_rel,
            snapshot_sha256=manifest_sha256,
            adapter_version=manifest.adapter_version,
            parser_version=manifest.parser_version,
            pack_name=pack_name,
            stage="W-03",
            substage="D-02-SOURCE-CONCEPTNET-V1",
        )
        tasks.append(SourcePackTask(pack_id, spec, tuple(seeds)))
    return tasks[0], tasks[1]


def _wikidata_task(repo: Path, raw_root: Path) -> SourcePackTask:
    """把 11 个 revision-pinned raw JSON 保真编入统一 source pack。"""
    manifest_rel = "data/ph2/manifests/wikidata_revision_v1.pinned_snapshot.json"
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_wikidata_revision_snapshot(manifest_path)
    seeds: list[SourceObservationSeed] = []
    for ordinal, entity in enumerate(manifest.entities, start=1):
        raw = (raw_root / Path(*entity.raw_relative_path.split("/"))).resolve()
        if not raw.is_relative_to(raw_root) or not raw.is_file():
            raise SourcePackCatalogError("Wikidata source raw 缺失或逃逸")
        if _sha256_path(raw) != entity.raw_sha256:
            raise SourcePackCatalogError("Wikidata source raw SHA-256 漂移")
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
            entity.split,
            "zh",
            "wikidata-entity-json-raw",
            entity.raw_relative_path,
            "sha256:" + entity.raw_sha256,
            entity.raw_sha256,
            CanonicalJsonObject.from_value({
                "cluster_id": entity.cluster_id,
                "purpose_keys": list(entity.purpose_keys),
                "qid": entity.qid,
                "raw_relative_path": entity.raw_relative_path,
                "revision": entity.revision,
                "response_url": entity.http.response_url,
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
            "support" if entity.split == "train" else "read_only_probe",
            "NONE",
            ordinal,
        ))
    spec = _spec(
        source_key=manifest.source_key,
        license_id=manifest.license_id,
        snapshot_id=manifest.snapshot_id,
        official_url="https://www.wikidata.org/wiki/Special:EntityData",
        attribution=manifest.attribution,
        snapshot_relative_path=manifest_rel,
        snapshot_sha256=manifest_sha256,
        adapter_version=manifest.adapter_version,
        parser_version=manifest.parser_version,
        pack_name="WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1",
        stage="W-03",
        substage="D-02-SOURCE-WIKIDATA-V1",
    )
    return SourcePackTask(2, spec, tuple(seeds))


def _mediawiki_task(
        repo: Path,
        raw_root: Path,
        *,
        pack_id: int,
        manifest_rel: str,
        pack_name: str,
        stage: str,
        substage: str) -> SourcePackTask:
    """从正式双遍 snapshot 取四页 bounded raw task。"""
    manifest_path, manifest_sha256 = _snapshot_identity(repo, manifest_rel)
    manifest = read_mediawiki_dump_snapshot(manifest_path)
    seeds = bounded_mediawiki_source_seeds(manifest, raw_root=raw_root, limit=4)
    xml = next(item for item in manifest.raw_files if item.role == "XML")
    spec = _spec(
        source_key=manifest.source_key,
        license_id=manifest.license_id,
        snapshot_id=manifest.snapshot_id,
        official_url=xml.official_url,
        attribution=manifest.attribution_policy,
        snapshot_relative_path=manifest_rel,
        snapshot_sha256=manifest_sha256,
        adapter_version=manifest.adapter_version,
        parser_version=manifest.parser_version,
        pack_name=pack_name,
        stage=stage,
        substage=substage,
    )
    return SourcePackTask(pack_id, spec, seeds)


def build_repository_source_pack_tasks(
        repository_root: str | Path,
        raw_root: str | Path) -> tuple[SourcePackTask, ...]:
    """构建六个合法许可分区 task；CC-CEDICT 只进入 blocker 覆盖账。"""
    repo = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    if not (repo / "src" / "pure_integer_ai").is_dir() or not raw.is_dir():
        raise SourcePackCatalogError("source pack repository/raw root 非法")
    concept_by, concept_by_sa = _conceptnet_tasks(repo)
    tasks = (
        _ud_task(repo),
        _wikidata_task(repo, raw),
        concept_by,
        concept_by_sa,
        _mediawiki_task(
            repo,
            raw,
            pack_id=5,
            manifest_rel=(
                "data/ph2/manifests/zhwiktionary_20260701."
                "multistream_snapshot.json"),
            pack_name=(
                "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1"),
            stage="W-03",
            substage="D-02-SOURCE-ZHWIKTIONARY-V1",
        ),
        _mediawiki_task(
            repo,
            raw,
            pack_id=6,
            manifest_rel=(
                "data/ph2/manifests/zhwikipedia_20260701."
                "multistream_snapshot.json"),
            pack_name=(
                "ZHWIKIPEDIA_20260701--CC-BY-SA-4.0--source-pack-v1"),
            stage="W-08",
            substage="D-02-SOURCE-ZHWIKIPEDIA-V1",
        ),
    )
    if tuple(item.pack_id for item in tasks) != tuple(range(1, 7)):
        raise SourcePackCatalogError("source pack task id 序列漂移")
    return tasks


def compile_repository_source_packs(
        repository_root: str | Path,
        raw_root: str | Path,
        artifact_root: str | Path,
        ) -> tuple[tuple[SourcePackTask, SourcePackBuild], ...]:
    """顺序发布或恢复六个 bounded 正式来源 pack，不启动训练。"""
    tasks = build_repository_source_pack_tasks(repository_root, raw_root)
    return tuple(
        (task, compile_or_resume_source_pack(
            task.spec, task.seeds, artifact_root))
        for task in tasks
    )


def build_repository_source_pack_coverage(
        repository_root: str | Path,
        builds: tuple[tuple[SourcePackTask, SourcePackBuild], ...],
        *,
        artifact_relative_root: str = SOURCE_PACK_ARTIFACT_RELATIVE_ROOT,
        ) -> SourcePackCoverageManifest:
    """汇合六个 pack 和 CC-CEDICT blocker 为完整来源覆盖账。"""
    repo = Path(repository_root).resolve()
    if not builds:
        raise SourcePackCatalogError("source pack builds 不能为空")
    entries: list[SourcePackCoverageEntry] = []
    for task, build in builds:
        if build.manifest.source_key != task.spec.source_key:
            raise SourcePackCatalogError("source pack build/task source 漂移")
        combination = build.bundle.combination_audit.to_value()
        manifest_rel = (
            f"{artifact_relative_root}/packs/{task.spec.pack_name}/manifest.json")
        entries.append(SourcePackCoverageEntry(
            task.spec.source_key,
            task.spec.license_id,
            "PACK_FROZEN",
            task.spec.raw_snapshot_manifest_relative_path,
            task.spec.raw_snapshot_manifest_sha256,
            manifest_rel,
            build.manifest.sha256(),
            build.manifest.record_count,
            build.manifest.splits,
            build.bundle.validation.source_cluster_count,
            combination["combination_cluster_count"],
            "",
            (
                task.spec.raw_snapshot_manifest_relative_path,
                "tests/test_d02_source_pack_compiler.py",
            ),
        ))
    cc_rel = "data/ph2/manifests/cc_cedict_20260725.raw_snapshot.json"
    _, cc_sha256 = _snapshot_identity(repo, cc_rel)
    entries.append(SourcePackCoverageEntry(
        "CC_CEDICT_20260725",
        "UNRESOLVED/BLOCKED",
        "BLOCKED",
        cc_rel,
        cc_sha256,
        "",
        "",
        0,
        (),
        0,
        0,
        "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
        (
            "data/ph2/manifests/cc_cedict_20260725.license_reconciliation_v1.json",
            "tests/test_d02_cc_cedict_license_reconciliation.py",
        ),
    ))
    return SourcePackCoverageManifest(
        1,
        "D02-source-pack-coverage-v1",
        tuple(sorted(
            entries,
            key=lambda item: (item.source_key, item.license_partition),
        )),
    )


__all__ = [
    "SOURCE_PACK_ARTIFACT_RELATIVE_ROOT",
    "SOURCE_PACK_COVERAGE_PATH",
    "SourcePackCatalogError",
    "build_repository_source_pack_coverage",
    "build_repository_source_pack_tasks",
    "compile_repository_source_packs",
]
