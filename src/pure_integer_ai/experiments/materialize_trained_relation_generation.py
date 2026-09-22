"""把 active Core 关系命题物化到发布图 R-01 生成闭包。

本阶段只在训练/发布准备期运行。它从冻结 SQLite 恢复 W-06 命题、图内
RelationSurfaceFrame、当前 Evidence、LanguageBranch 和既有 R-01 profile，随后
通过正式 ``AliasRelationCourseLoader`` 追加角色/语言原子 -> Representation 的
``realizes`` 事实和 S-07 顺序。运行时只读取结果 SQLite，不读取本阶段课程输入。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    representation_identity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseManifest,
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
    _definition_graph, _lifecycle, _surface_protocol,
)
from pure_integer_ai.experiments.relation_generation_structure import (
    RELATION_CONNECTOR_PROFILE, compile_relation_connector,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import install_relation_frame_order
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceFrame,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object_identity import (
    GRAPH_OBJECT_MAX_COMPONENTS,
)
from pure_integer_ai.storage.recovery_package import publish_recovery_package


MATERIALIZATION_FORMAT = "TRAINED_RELATION_GENERATION_MATERIALIZATION_V1"
MATERIALIZATION_MANIFEST = "trained_relation_generation_materialization.json"
_REQUIRED_PARENT_FILES = (
    "cursor.json",
    "dialogue_pack_manifest.json",
    "run.manifest.json",
    "run.manifest.sha256",
    "training_cursor.int",
    "training_summary.json",
)
_MATERIALIZATION_MAX_PROJECTED_COMPONENTS = 32 * 1024 * 1024
# 当前 lifecycle Event 会分别在事件本体、活动 Evidence 和 H-04 前后快照中
# 保存完整 Hypothesis。逐 entry 课程的实物化最大路径需要少于十份等长展开；
# 额外的 candidate/recognition 项和固定余量覆盖长度前缀及 verifier 载荷。
# Hypothesis/Evidence 身份按 proposition 分片写入。生命周期事件引用同一
# Hypothesis，不把完整键复制十次到一个 graph object；预算只按单条对象键和
# 一条 Evidence 记录计算，实际关系仍由图边保持完整关联。
_MATERIALIZATION_EVENT_HYPOTHESIS_MULTIPLIER = 1


# object-model: exception; interop=trained-relation-generation-materialization-v1
class TrainedRelationGenerationMaterializationError(RuntimeError):
    """父训练图、R-01 profile 或生成物化结果不闭合。"""


@dataclass(frozen=True, slots=True)
class MaterializationStructureBudget:
    """首写前对局部候选身份及其 lifecycle 放大上界的整数预算。"""

    entry_count: int
    max_entry_components: int
    max_candidate_components: int
    max_hypothesis_components: int
    max_projected_object_components: int
    projected_total_components: int

    def record(self) -> dict[str, int]:
        """返回可写入 cursor/receipt 的纯整数预算投影。"""
        return {
            "entry_count": self.entry_count,
            "max_entry_components": self.max_entry_components,
            "max_candidate_components": self.max_candidate_components,
            "max_hypothesis_components": self.max_hypothesis_components,
            "max_projected_object_components": (
                self.max_projected_object_components),
            "projected_total_components": self.projected_total_components,
            "single_object_limit": GRAPH_OBJECT_MAX_COMPONENTS,
            "total_limit": _MATERIALIZATION_MAX_PROJECTED_COMPONENTS,
        }


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


def _structure_budget(
        manifest: AliasRelationCourseManifest,
        ) -> MaterializationStructureBudget:
    """在首写前量化局部身份及 lifecycle 的整数放大上界。"""
    if not manifest.entries:
        raise TrainedRelationGenerationMaterializationError(
            "关系生成课程不得为空")
    max_entry = 0
    max_candidate = 0
    max_hypothesis = 0
    max_projected_object = 0
    projected_total = 0
    for entry in manifest.entries:
        entry_size = len(entry.stable_key())
        definition = entry.spec.candidate_definition(
            manifest.relation_protocol)
        candidate_size = len(definition.stable_key())
        hypothesis_size = len(
            definition.hypothesis(manifest.learning_protocol).stable_key())
        recognition_size = sum(
            len(item.stable_key()) for item in entry.recognitions)
        # 每个 proposition 是独立物化分片；Event/Hypothesis/Evidence 通过
        # 图内引用关联，不能把整条 lifecycle 的完整键乘法展开进单一对象。
        projected_object = max(
            entry_size,
            candidate_size,
            hypothesis_size * _MATERIALIZATION_EVENT_HYPOTHESIS_MULTIPLIER,
            recognition_size,
        ) + 4096
        projected_total += (
            entry_size + candidate_size + hypothesis_size
            + projected_object)
        max_entry = max(max_entry, entry_size)
        max_candidate = max(max_candidate, candidate_size)
        max_hypothesis = max(max_hypothesis, hypothesis_size)
        max_projected_object = max(
            max_projected_object, projected_object)
    budget = MaterializationStructureBudget(
        len(manifest.entries),
        max_entry,
        max_candidate,
        max_hypothesis,
        max_projected_object,
        projected_total,
    )
    if (max(max_entry, max_candidate, max_hypothesis,
            max_projected_object)
            > GRAPH_OBJECT_MAX_COMPONENTS):
        raise TrainedRelationGenerationMaterializationError(
            "关系生成局部身份超过单对象预算；疑似复制整批请求")
    if projected_total > _MATERIALIZATION_MAX_PROJECTED_COMPONENTS:
        raise TrainedRelationGenerationMaterializationError(
            "关系生成预计 components 总量超过单批预算；必须拆分课程")
    return budget


def _build_materialization_input(database: Path):
    """只读恢复 Core、R-01 profile 及角色级 connector/语言原子课程。"""
    with TrainedGenerationConnectorRuntime(database) as connector:
        templates = connector.templates()
        profiles = connector._discover_alias_protocols()
        if len({item.language_branch for item in templates}) != 1 or len(profiles) != 1:
            raise TrainedRelationGenerationMaterializationError(
                "发布图必须有唯一 connector 和 R-01 profile")
        branch = templates[0].language_branch
        value_protocol = connector._branches[0].definition_graph.value_protocol
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
        courses = []
        bindings = {}
        for fact in facts:
            generation = relation.generation_input(fact.proposition)
            course = compile_relation_connector(fact, frames[fact.proposition], generation,
                                                branch, family, value_protocol, _surface_protocol(branch))
            value_protocol = course.values
            courses.append(course)
            for binding in course.aliases:
                key = (binding.origin, binding.representation)
                prior = bindings.get(key)
                if prior is not None:
                    binding = replace(binding, forming_evidence_keys=tuple(sorted({
                        *prior.forming_evidence_keys, *binding.forming_evidence_keys})))
                bindings[key] = binding
    request = GenerationCandidateAliasCourseRequest(
        branch, tuple(bindings.values()))
    manifest = build_alias_relation_manifest(profile, request)
    budget = _structure_budget(manifest)
    return (
        manifest,
        len(facts),
        tuple(replace(course, values=value_protocol) for course in courses),
        budget,
    )


def _install_role_course(context, course) -> int:
    """在既有 S-07 生命周期及 connector 理论图中写入一份可恢复角色课程。"""
    lifecycle = _lifecycle(context, course.template.language_branch)
    graph = _definition_graph(context, course.template.language_branch, lifecycle)
    graph.value_protocol = course.values
    count = install_relation_frame_order(course, lifecycle)
    graph.materialize(course.template, scope=course.source.proposition.scope,
                      provenance_kind=RELATION_CONNECTOR_PROFILE, content_version=1,
                      qualifiers=course.qualifiers)
    return count


def _publish_portable_recovery(
        *, staging: Path, target: Path, source: Path,
        ) -> None:
    """从物化后 SQLite 导出与该子 run 同身份的完整 portable recovery。"""
    parent_manifest = _read_object(
        source / "run.manifest.json", label="parent recovery manifest")
    parent_cursor = _read_object(
        source / "cursor.json", label="parent recovery cursor")
    table_order = parent_manifest.get("table_order")
    schema = parent_manifest.get("schema")
    spaces = parent_manifest.get("space_ids")
    if (not isinstance(table_order, list) or not table_order
            or any(type(item) is not str or not item for item in table_order)
            or len(set(table_order)) != len(table_order)
            or not isinstance(schema, dict) or set(schema) != set(table_order)
            or not isinstance(spaces, list) or not spaces
            or any(type(item) is not int or item <= 0 for item in spaces)
            or len(set(spaces)) != len(spaces)):
        raise TrainedRelationGenerationMaterializationError(
            "父 portable recovery schema 非法")
    actual_tables: tuple[str, ...]
    connection = sqlite3.connect(
        f"file:{(staging / 'training.sqlite3').as_posix()}?mode=ro", uri=True)
    try:
        actual_tables = tuple(sorted(
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")))
    finally:
        connection.close()
    if actual_tables != tuple(sorted(table_order)):
        raise TrainedRelationGenerationMaterializationError(
            "物化 SQLite 表集合与父 recovery schema 漂移")

    backend = SQLiteBackend(
        str(staging / "training.sqlite3"), read_only=True)
    recovery_root = staging / ".portable-recovery"
    try:
        for table in table_order:
            spec = schema.get(table)
            if not isinstance(spec, dict):
                raise TrainedRelationGenerationMaterializationError(
                    f"父 recovery 表定义非法: {table}")
            col_types = spec.get("col_types")
            indexes = spec.get("indexes")
            recovery_key = spec.get("recovery_key")
            discipline = spec.get("discipline")
            core = spec.get("core")
            if (not isinstance(col_types, list) or not col_types
                    or any(not isinstance(item, list) or len(item) != 2
                           or type(item[0]) is not str
                           or type(item[1]) is not str for item in col_types)
                    or not isinstance(indexes, list)
                    or any(not isinstance(item, list)
                           or any(type(part) is not str for part in item)
                           for item in indexes)
                    or not isinstance(recovery_key, list)
                    or any(type(item) is not str for item in recovery_key)
                    or type(discipline) is not int or type(core) is not bool):
                raise TrainedRelationGenerationMaterializationError(
                    f"父 recovery 表协议非法: {table}")
            backend.register_table(
                table,
                [(item[0], item[1]) for item in col_types],
                discipline,
                [tuple(item) for item in indexes],
                core=core,
                defer_indexes=True,
                recovery_key=tuple(recovery_key),
            )
        cursor_payload = {
            "base_run_id": source.name,
            "completed": list(parent_cursor.get("completed", [])),
            "non_skippable": list(parent_cursor.get("non_skippable", [])),
            "run_id": target.name,
        }
        if (any(not isinstance(cursor_payload[name], list)
                or any(type(item) is not int
                       for item in cursor_payload[name])
                for name in ("completed", "non_skippable"))):
            raise TrainedRelationGenerationMaterializationError(
                "父 recovery cursor 阶段集非法")
        publish_recovery_package(
            backend,
            str(recovery_root),
            target.name,
            spaces=tuple(sorted(spaces)),
            tables=tuple(table_order),
            include_registered_tables=True,
            require_all_spaces=True,
            cursor_payload=cursor_payload,
        )
    finally:
        backend.close()
    published = recovery_root / target.name
    for path in sorted(published.iterdir()):
        destination = staging / path.name
        if destination.exists():
            raise TrainedRelationGenerationMaterializationError(
                f"portable recovery 文件冲突: {path.name}")
        path.replace(destination)
    published.rmdir()
    recovery_root.rmdir()


def _publish_run_metadata(
        *, staging: Path, target: Path, source: Path,
        parent_database_sha256: str, alias_manifest_sha256: str,
        active_core_count: int, materialized_count: int,
        active_realization_count: int,
        structure_budget: MaterializationStructureBudget,
        ) -> tuple[str, dict[str, object]]:
    """发布增量 run 自身的 SQLite 和训练来源闭包。"""
    database = staging / "training.sqlite3"
    database_sha = _sha256(database)
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
            "alias_request_version": 2,
            "alias_manifest_sha256": alias_manifest_sha256,
            "materialized_realization_count": materialized_count,
            "parent_database_sha256": parent_database_sha256,
            "structure_budget": structure_budget.record(),
        },
    })
    (staging / "training_summary.json").write_bytes(_canonical_json(summary))
    materialization = {
        "active_core_count": active_core_count,
        "active_realization_count": active_realization_count,
        "alias_request_version": 2,
        "alias_manifest_sha256": alias_manifest_sha256,
        "database_sha256": database_sha,
        "format": MATERIALIZATION_FORMAT,
        "materialized_realization_count": materialized_count,
        "parent_database_sha256": parent_database_sha256,
        "parent_run_id": source.name,
        "run_id": target.name,
        "schema_version": 1,
        "structure_budget": structure_budget.record(),
    }
    materialization_path = staging / MATERIALIZATION_MANIFEST
    materialization_path.write_bytes(_canonical_json(materialization))
    return database_sha, materialization


def _publish_sqlite_resume_metadata(staging: Path, target: Path) -> None:
    """在 portable recovery 完整就位后封存 SQLite page-resume 五文件身份。"""
    database = staging / "training.sqlite3"
    summary = _read_object(
        staging / "training_summary.json", label="materialized training summary")
    materialization_path = staging / MATERIALIZATION_MANIFEST
    database_sha = _sha256(database)
    page_count, page_size, schema_sha, counts_sha, counts = (
        _database_fingerprint(database))
    files = {
        "cursor.json": staging / "cursor.json",
        "run.manifest.json": staging / "run.manifest.json",
        "run.manifest.sha256": staging / "run.manifest.sha256",
        "training_cursor.int": staging / "training_cursor.int",
        "training_summary.json": staging / "training_summary.json",
    }
    if any(not path.is_file() for path in files.values()):
        raise TrainedRelationGenerationMaterializationError(
            "物化 run 缺少 SQLite resume 所需 recovery 文件")
    resume = {
        "artifact_kind": SQLITE_RESUME_ARTIFACT_KIND,
        "database_bytes": database.stat().st_size,
        "database_sha256": database_sha,
        "materialization_manifest_sha256": _sha256(materialization_path),
        "file_sha256": {
            name: _sha256(path) for name, path in sorted(files.items())
        },
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
        manifest, active_core_count, courses, budget = _build_materialization_input(
            staging / "training.sqlite3")
        backend = SQLiteBackend(str(staging / "training.sqlite3"))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            for course in courses:
                _install_role_course(context, course)
            loaded = AliasRelationCourseLoader(
                manifest, manifest.sha256()).load(
                    context,
                    isolated_preflight_database=(
                        staging / "relation-generation-preflight.sqlite3"),
                )
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
        if (len(courses) != active_core_count
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
            structure_budget=budget,
        )
        _publish_portable_recovery(
            staging=staging, target=target, source=source)
        _publish_sqlite_resume_metadata(staging, target)
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


def materialize_role_generation_database(source_database: Path, target_root: Path, *,
                                         source_manifest: Path,
                                         resume: bool = False,
                                         in_place: bool = False) -> dict[str, object]:
    """在显式 K 盘模型续接角色生成，逐课程 checkpoint。

    checkpoint 只控制阶段恢复，不参与语义计算。失败保留全部产物；同一 parent SHA
    与课程 manifest 身份才允许 resume，别名/顺序沿已有 append-only owner 物化。
    ``in_place`` 只省略父 SQLite 复制；cursor/receipt 仍写入独立子目录。
    """
    if type(in_place) is not bool:
        raise TypeError("in_place 必须是严格 bool")
    source = source_database.resolve(strict=True)
    ledger_path = source_manifest.resolve(strict=True)
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    if (not isinstance(ledger, dict)
            or ledger.get("format") != "PURE_INTEGER_TRAINED_GRAPH_SOURCE_LEDGER_V1"
            or not ledger.get("sources")
            or any(not item.get("license_ids") or not item.get("sha256")
                   for item in ledger["sources"])):
        raise ValueError("父模型来源/许可清单不完整")
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    root = target_root.resolve()
    if (root.drive.upper() != "K:" or root == Path(root.anchor)
            or root in source.parents or root in ledger_path.parents):
        raise ValueError("角色生成训练必须使用独立的显式 K 盘子目录")
    cursor_path = root / "role_generation_cursor.json"
    if root.exists() and not resume:
        raise ValueError("目标训练目录已存在，必须显式 resume 或选择新目录")
    source_sha = _sha256(source)
    database = source if in_place else root / "training.sqlite3"
    code_files = (Path(__file__), Path(__file__).with_name("relation_generation_structure.py"),
                  Path(__file__).with_name("relation_generation_protocol.py"),
                  Path(__file__).with_name("language_generation_connector.py"),
                  Path(__file__).with_name("ph2_generation_candidate_alias_course.py"),
                  Path(__file__).parents[1] / "cognition/shared/generation_role_binding.py",
                  Path(__file__).parents[1] / "cognition/shared/generation_structure_plan.py",
                  Path(__file__).with_name("trained_generation_connector_runtime.py"),
                  Path(__file__).with_name("ph2_grounded_answer_order.py"))
    code_identity = {path.name: _sha256(path) for path in code_files}
    if resume:
        cursor = _read_object(cursor_path, label="role generation cursor")
        source_identity_matches = (
            cursor.get("source_sha256") == source_sha
            if not in_place else cursor.get("database_path") == str(database))
        if (not source_identity_matches or cursor.get("schema_version") != 2
                or cursor.get("in_place") != int(in_place)
                or cursor.get("source_manifest_sha256") != ledger_sha
                or (cursor.get("stage", 0) > 1
                    and cursor.get("alias_request_version") != 2)
                or (root / "source_manifest.json").read_bytes() != ledger_bytes):
            raise ValueError("角色生成父库身份变化或旧批量身份恢复点不再可续写")
    else:
        root.mkdir(parents=True)
        (root / "source_manifest.json").write_bytes(ledger_bytes)
        cursor = dict(schema_version=2, source_sha256=source_sha, stage=0,
                      source_manifest_sha256=ledger_sha, course_cursor=0,
                      order_count=0, in_place=int(in_place),
                      database_path=str(database),
                      database_bytes_before=source.stat().st_size,
                      database_copy_count=0 if in_place else 1)
        cursor_path.write_bytes(_canonical_json(cursor))
    versions = cursor.setdefault("code_versions", [])
    if not versions or versions[-1] != code_identity:
        versions.append(code_identity)

    def checkpoint() -> None:
        """以同目录原子替换提交完整恢复点，语义阶段保持单调递增。"""
        pending = root / "role_generation_cursor.pending"
        pending.write_bytes(_canonical_json(cursor))
        pending.replace(cursor_path)

    if cursor["stage"] == 0:
        if not in_place:
            origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            copied = sqlite3.connect(str(database))
            try:
                origin.backup(copied)
            finally:
                copied.close()
                origin.close()
        cursor["stage"] = 1
        checkpoint()
    if cursor["stage"] < 3:
        manifest, active_count, courses, budget = _build_materialization_input(
            database)
        previous_sha = cursor.get("manifest_sha256")
        manifest_sha = manifest.sha256()
        if previous_sha not in (None, manifest_sha):
            raise ValueError("角色生成课程 manifest 在恢复时漂移")
        versions = cursor.setdefault("manifest_versions", [])
        if previous_sha and previous_sha not in versions:
            versions.append(previous_sha)
        if manifest_sha not in versions:
            versions.append(manifest_sha)
        cursor.update(manifest_sha256=manifest_sha, active_core_count=active_count,
                      alias_request_version=2,
                      realization_count=len(manifest.entries), connector_count=len(courses),
                      structure_budget=budget.record())
        checkpoint()
        backend = SQLiteBackend(str(database))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            for index, course in enumerate(courses):
                if index < cursor["course_cursor"]:
                    continue
                count = _install_role_course(context, course)
                backend.commit()
                cursor["course_cursor"] = index + 1
                cursor["order_count"] += count
                checkpoint()
            cursor["stage"] = 2
            checkpoint()
            AliasRelationCourseLoader(manifest, manifest_sha).load(context)
            backend.commit()
            cursor["stage"] = 3
            checkpoint()
        finally:
            backend.close()
    if cursor["stage"] < 4:
        import gzip
        with TrainedGenerationConnectorRuntime(database) as connector, TrainedRelationGraphRuntime(database) as relation:
            outputs = []
            for ordinal, fact in enumerate(relation.active_surface_facts(), 1):
                result = connector.generate_relation(relation.generation_input(fact.proposition), fact)
                if len(result.representations) < len(fact.bindings) or not result.surface:
                    raise RuntimeError("角色级生成没有实际完成")
                trace = {"connector": list(result.connector.stable_key()),
                         "representations": [list(item.stable_key()) for item in result.representations],
                         "trace": list(result.trace), "surface": result.surface}
                payload = gzip.compress(_canonical_json(trace), mtime=0)
                name = f"generation-{ordinal:03d}.json.gz"
                (root / name).write_bytes(payload)
                outputs.append(dict(surface=result.surface, slot_count=result.slot_count,
                                    trace=name, sha256=hashlib.sha256(payload).hexdigest()))
        database_bytes = database.stat().st_size
        receipt = dict(
            cursor, outputs=outputs, free_dialogue_complete=0,
            model_sha256=_sha256(database), legacy_claim_replay_used=0,
            database_bytes=database_bytes,
            database_bytes_added=(
                database_bytes - int(cursor["database_bytes_before"])),
            semantic_object_copy_count=0)
        (root / "role_generation_receipt.json").write_bytes(_canonical_json(receipt))
        cursor["stage"] = 4
        checkpoint()
    return _read_object(root / "role_generation_receipt.json", label="role generation receipt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="materialize trained Core relation generation routes")
    parser.add_argument("--source-run")
    parser.add_argument("--target-run")
    parser.add_argument("--source-database", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--in-place", action="store_true",
        help="在 source-database 原地追加；target-root 只保存 cursor/receipt")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.source_database is not None:
        if args.target_root is None or args.source_manifest is None or args.source_run or args.target_run:
            parser.error("source-database 必须配合 source-manifest 和独立 target-root")
        result = materialize_role_generation_database(
            args.source_database, args.target_root,
            source_manifest=args.source_manifest, resume=args.resume,
            in_place=args.in_place)
        summary = result if not args.quiet else {
            key: value for key, value in result.items() if key not in {"outputs", "code_versions"}}
        if args.quiet:
            summary["output_count"] = len(result["outputs"])
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0
    if not args.source_run or not args.target_run or args.resume or args.in_place:
        parser.error("必须指定 source-run/target-run，或 source-database/target-root")
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
