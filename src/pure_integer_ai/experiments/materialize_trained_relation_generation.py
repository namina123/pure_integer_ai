"""把 active Core 关系命题物化到发布图 R-01 生成闭包。

本阶段只在训练/发布准备期运行。它从冻结 SQLite 恢复 W-06 命题、图内
RelationSurfaceFrame、当前 Evidence、LanguageBranch 和既有 R-01 profile，随后
通过正式 ``AliasRelationCourseLoader`` 追加 Proposition -> Representation 的
``realizes`` 事实。发布运行时仍只读取结果 SQLite，不读取课程或本阶段输入。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    representation_identity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
    GenerationCandidateRealizationBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    AliasRelationManifestProfile,
    build_alias_relation_manifest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_CLAIM
from pure_integer_ai.experiments.sqlite_training_resume import (
    SQLITE_RESUME_ARTIFACT_KIND,
    _database_fingerprint,
    prepare_sqlite_page_resume,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceFrame,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend


MATERIALIZATION_FORMAT = "TRAINED_RELATION_GENERATION_MATERIALIZATION_V1"
MATERIALIZATION_MANIFEST = "trained_relation_generation_materialization.json"
_REQUIRED_PARENT_FILES = (
    "dialogue_pack_manifest.json",
    "training_cursor.int",
    "training_summary.json",
)


# object-model: exception; interop=trained-relation-generation-materialization-v1
class TrainedRelationGenerationMaterializationError(RuntimeError):
    """父训练图、R-01 profile 或生成物化结果不闭合。"""


# object-model: value; representation=struct; interop=trained-relation-generation-materialization-v1
@dataclass(frozen=True, slots=True)
class TrainedRelationGenerationMaterialization:
    """一个已完成并回读的训练后关系生成增量 run。"""

    root: Path
    run_id: str
    database: Path
    parent_run_id: str
    active_core_count: int
    materialized_realization_count: int
    active_realization_count: int
    alias_manifest_sha256: str
    database_sha256: str


def _canonical_json(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainedRelationGenerationMaterializationError(
            f"{label} 不可回读") from error
    if not isinstance(value, dict):
        raise TrainedRelationGenerationMaterializationError(
            f"{label} 必须是 JSON object")
    return value


def _profile_from_prefix(
        prefix: tuple[int, ...], source, scope,
        ) -> AliasRelationManifestProfile:
    """从训练历史中已恢复的 R-01 namespace 解出原 profile。"""
    if len(prefix) < 5 or prefix[0] != 22020:
        raise TrainedRelationGenerationMaterializationError(
            "R-01 profile namespace 不受支持")
    version_size = prefix[1]
    version_end = 2 + version_size
    if version_size <= 0 or version_end >= len(prefix):
        raise TrainedRelationGenerationMaterializationError(
            "R-01 candidate version 不闭合")
    digest_size = prefix[version_end]
    digest_end = version_end + 1 + digest_size
    if digest_size != 32 or digest_end != len(prefix):
        raise TrainedRelationGenerationMaterializationError(
            "R-01 profile digest 不闭合")
    return AliasRelationManifestProfile(
        tuple(prefix[2:version_end]),
        tuple(prefix[version_end + 1:digest_end]),
        source,
        scope,
        1,
    )


def _claim_surface(
        fact: ActiveRelationSurface,
        frame: RelationSurfaceFrame,
        ) -> str:
    """按图内命题槽位包络提取 claim，不猜测任何语言或标点。"""
    if (frame.proposition != fact.proposition
            or frame.source_hash != fact.source_hash
            or frame.envelope_start < 0
            or frame.envelope_end <= frame.envelope_start
            or frame.envelope_end > len(fact.evidence_surface)):
        raise TrainedRelationGenerationMaterializationError(
            "Core relation frame 与来源表层不闭合")
    surface = fact.evidence_surface[
        frame.envelope_start:frame.envelope_end]
    if not surface.strip():
        raise TrainedRelationGenerationMaterializationError(
            "Core relation frame 产生空 Representation")
    return surface


def _advance_graph_id_pool(backend: SQLiteBackend) -> None:
    """从权威 concept_node 高水位恢复续写 allocator。"""
    spaces = tuple(sorted(
        int(row["space_id"])
        for row in backend.select("space")
        if type(row.get("space_id")) is int and row["space_id"] > 0
    ))
    for space_id in spaces:
        rows = backend.select(
            "concept_node",
            where={"space_id": space_id},
            order_by="local_id",
            descending=True,
            limit=1,
        )
        if rows:
            backend.advance_id_pool(space_id, int(rows[0]["local_id"]))


def _build_materialization_input(database: Path):
    """只读恢复 Core、R-01 profile、branch 和命题 Representation 请求。"""
    with TrainedGenerationConnectorRuntime(database) as connector:
        templates = connector.templates()
        profiles = connector._discover_alias_protocols()
        if len(templates) != 1 or len(profiles) != 1:
            raise TrainedRelationGenerationMaterializationError(
                "发布图必须有唯一 connector 和 R-01 profile")
        branch = templates[0].language_branch
        prefix, source, scope = profiles[0]
        profile = _profile_from_prefix(prefix, source, scope)
        alias = connector.alias_runtime(branch)
        realizes = alias.closure.consumer.lookup_relation(
            alias.selector.protocol.realizes_relation)
        representations = tuple(
            binding.filler
            for fact in realizes
            for binding in fact.proposition.bindings
            if binding.filler.object_kind == OBJECT_REPRESENTATION
        )
        families = {representation_parts(item)[0] for item in representations}
        if len(families) != 1:
            raise TrainedRelationGenerationMaterializationError(
                "发布图 R-01 Representation family 不唯一")
        family = next(iter(families))
    with TrainedRelationGraphRuntime(database) as relation:
        frames = {
            item.proposition: item
            for item in relation.active_surface_frames()
        }
        facts = relation.active_surface_facts()
        if len(frames) != len(facts):
            raise TrainedRelationGenerationMaterializationError(
                "active Core relation/frame 数量不闭合")
        bindings = []
        for fact in facts:
            generation = relation.generation_input(fact.proposition)
            forming = tuple(sorted(
                item.stable_key() for item in generation.evidence))
            surface = _claim_surface(fact, frames[fact.proposition])
            bindings.append(GenerationCandidateRealizationBinding(
                fact.proposition,
                representation_identity(
                    family,
                    tuple(ord(character) for character in surface),
                    owner=branch.owner,
                    versions=branch.versions,
                ),
                RULE_CLAIM,
                forming,
            ))
    request = GenerationCandidateAliasCourseRequest(
        branch, tuple(bindings))
    manifest = build_alias_relation_manifest(profile, request)
    return manifest, len(facts)


def _publish_run_metadata(
        *, staging: Path, target: Path, source: Path,
        parent_database_sha256: str, alias_manifest_sha256: str,
        active_core_count: int, materialized_count: int,
        active_realization_count: int,
        ) -> tuple[str, dict[str, object]]:
    """发布增量 run 自身的 SQLite 和训练来源闭包。"""
    database = staging / "training.sqlite3"
    database_sha = _sha256(database)
    page_count, page_size, schema_sha, counts_sha, counts = (
        _database_fingerprint(database))
    parent_summary = _read_object(
        source / "training_summary.json", label="parent training summary")
    summary = dict(parent_summary)
    summary.update({
        "database": str(target / "training.sqlite3"),
        "resume_from": source.name,
        "run_id": target.name,
        "training_cursor": str(target / "training_cursor.int"),
        "typed_relation_generation": {
            "active_core_count": active_core_count,
            "active_realization_count": active_realization_count,
            "alias_manifest_sha256": alias_manifest_sha256,
            "materialized_realization_count": materialized_count,
            "parent_database_sha256": parent_database_sha256,
        },
    })
    (staging / "training_summary.json").write_bytes(_canonical_json(summary))
    materialization = {
        "active_core_count": active_core_count,
        "active_realization_count": active_realization_count,
        "alias_manifest_sha256": alias_manifest_sha256,
        "database_sha256": database_sha,
        "format": MATERIALIZATION_FORMAT,
        "materialized_realization_count": materialized_count,
        "parent_database_sha256": parent_database_sha256,
        "parent_run_id": source.name,
        "run_id": target.name,
        "schema_version": 1,
    }
    materialization_path = staging / MATERIALIZATION_MANIFEST
    materialization_path.write_bytes(_canonical_json(materialization))
    resume = {
        "artifact_kind": SQLITE_RESUME_ARTIFACT_KIND,
        "database_bytes": database.stat().st_size,
        "database_sha256": database_sha,
        "materialization_manifest_sha256": _sha256(materialization_path),
        "pack_sha256": summary.get("pack_sha256"),
        "page_count": page_count,
        "page_size": page_size,
        "run_id": target.name,
        "schema_sha256": schema_sha,
        "schema_version": 1,
        "status": "PASS",
        "table_counts": [list(item) for item in counts],
        "table_counts_sha256": counts_sha,
    }
    (staging / "sqlite_resume_manifest.json").write_bytes(
        _canonical_json(resume))
    return database_sha, materialization


def materialize_trained_relation_generation(
        *, source_run_root: str | Path, target_run_root: str | Path,
        require_k_drive: bool = True,
        ) -> TrainedRelationGenerationMaterialization:
    """从父训练 run 建立不覆盖源库的 R-01 增量物化 run。"""
    if type(require_k_drive) is not bool:
        raise TypeError("require_k_drive 必须是严格 bool")
    source = Path(source_run_root).resolve()
    target = Path(target_run_root).resolve()
    if (not source.is_dir() or source.parent != target.parent
            or target.exists()):
        raise TrainedRelationGenerationMaterializationError(
            "source/target 必须是同一 campaign 下的既有父 run 和新子 run")
    if require_k_drive and (
            source.drive.upper() != "K:" or target.drive.upper() != "K:"):
        raise TrainedRelationGenerationMaterializationError(
            "关系生成物化 run 必须位于 K 盘")
    for name in _REQUIRED_PARENT_FILES:
        if not (source / name).is_file():
            raise TrainedRelationGenerationMaterializationError(
                f"父训练 run 缺少 {name}")
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise TrainedRelationGenerationMaterializationError(
            "关系生成物化 staging 已存在")
    staging.mkdir()
    try:
        parent = prepare_sqlite_page_resume(
            source, staging / "training.sqlite3",
            require_k_drive=require_k_drive,
        )
        for name in ("dialogue_pack_manifest.json", "training_cursor.int"):
            shutil.copyfile(source / name, staging / name)
        manifest, active_core_count = _build_materialization_input(
            staging / "training.sqlite3")
        backend = SQLiteBackend(str(staging / "training.sqlite3"))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            loaded = AliasRelationCourseLoader(
                manifest, manifest.sha256()).load(context)
            backend.commit()
        finally:
            backend.close()
        with TrainedGenerationConnectorRuntime(
                staging / "training.sqlite3") as connector:
            branch = connector.templates()[0].language_branch
            alias = connector.alias_runtime(branch)
            active_realization_count = len(
                alias.closure.consumer.lookup_relation(
                    alias.selector.protocol.realizes_relation))
        materialized_count = len(manifest.entries)
        if (materialized_count != active_core_count
                or active_realization_count < materialized_count
                or loaded.report.active_count < active_realization_count):
            raise TrainedRelationGenerationMaterializationError(
                "R-01 写后 active 计数不闭合")
        database_sha, _record = _publish_run_metadata(
            staging=staging,
            target=target,
            source=source,
            parent_database_sha256=parent.database_sha256,
            alias_manifest_sha256=manifest.sha256(),
            active_core_count=active_core_count,
            materialized_count=materialized_count,
            active_realization_count=active_realization_count,
        )
        staging.rename(target)
        return TrainedRelationGenerationMaterialization(
            target,
            target.name,
            target / "training.sqlite3",
            source.name,
            active_core_count,
            materialized_count,
            active_realization_count,
            manifest.sha256(),
            database_sha,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_trained_relation_generation_materialization(
        root: str | Path,
        ) -> TrainedRelationGenerationMaterialization:
    """回读并核验已发布增量 run 的关键闭包。"""
    target = Path(root).resolve()
    record = _read_object(
        target / MATERIALIZATION_MANIFEST,
        label="relation generation materialization")
    database = target / "training.sqlite3"
    if (record.get("format") != MATERIALIZATION_FORMAT
            or record.get("schema_version") != 1
            or record.get("run_id") != target.name
            or not database.is_file()
            or record.get("database_sha256") != _sha256(database)):
        raise TrainedRelationGenerationMaterializationError(
            "relation generation materialization 漂移")
    return TrainedRelationGenerationMaterialization(
        target,
        target.name,
        database,
        str(record["parent_run_id"]),
        int(record["active_core_count"]),
        int(record["materialized_realization_count"]),
        int(record["active_realization_count"]),
        str(record["alias_manifest_sha256"]),
        str(record["database_sha256"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="materialize trained Core relation generation routes")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    args = parser.parse_args(argv)
    result = materialize_trained_relation_generation(
        source_run_root=args.source_run,
        target_run_root=args.target_run,
    )
    print(json.dumps({
        "active_core_count": result.active_core_count,
        "active_realization_count": result.active_realization_count,
        "alias_manifest_sha256": result.alias_manifest_sha256,
        "database_sha256": result.database_sha256,
        "materialized_realization_count": result.materialized_realization_count,
        "run_id": result.run_id,
        "status": "PASS",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MATERIALIZATION_FORMAT",
    "MATERIALIZATION_MANIFEST",
    "TrainedRelationGenerationMaterialization",
    "TrainedRelationGenerationMaterializationError",
    "load_trained_relation_generation_materialization",
    "materialize_trained_relation_generation",
]
