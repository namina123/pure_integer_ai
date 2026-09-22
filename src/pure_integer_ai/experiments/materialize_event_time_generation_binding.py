"""在独立 K 盘模型副本物化 active EVENT_TIME 到既有 connector 的图绑定。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.cognition.understanding.query_structure_adapter import (
    RelationSurfaceStructureAdapter,
)
from pure_integer_ai.experiments.event_time_structure_training_bridge import (
    EventTimeRelationGenerationSeed,
    event_time_input_structure_identity,
    materialize_event_time_relation_generation_bindings,
)
from pure_integer_ai.experiments.materialize_trained_relation_generation import (
    _advance_graph_id_pool,
)
from pure_integer_ai.experiments.relation_capability_route import (
    RelationCapabilityRouteRegistry,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorError,
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_typed_relation_bridge import (
    EVENT_TIME_RELATION_KINDS,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend


_FORMAT = "PURE_INTEGER_EVENT_TIME_GENERATION_BINDING_V1"


def _json_bytes(value: object) -> bytes:
    """返回规范 ASCII JSON 宿主记录。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")


def _sha256(path: Path) -> str:
    """流式计算文件身份，不把模型读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_cursor(path: Path, value: dict[str, object]) -> None:
    """以同目录原子替换保存单调恢复点。"""
    pending = path.with_suffix(".pending")
    pending.write_bytes(_json_bytes(value))
    pending.replace(path)


def _build_seeds(
        source: Path,
        ) -> tuple[EventTimeRelationGenerationSeed, ...]:
    """只读复用 active fact、RoleBinding、Event、Context 和 connector 身份。"""
    with TrainedRelationGraphRuntime(source) as core, \
            TrainedGenerationConnectorRuntime(source) as connector:
        facts = core.active_surface_facts()
        templates = {}
        for fact in facts:
            try:
                templates[fact.proposition.stable_key()] = (
                    connector.generation_template_for(fact))
            except TrainedGenerationConnectorError:
                continue
        routes = RelationCapabilityRouteRegistry(
            facts,
            scope_keys={
                fact.proposition.stable_key(): core.generation_input(
                    fact.proposition).hypothesis.scope.stable_key()
                for fact in facts
            },
            generation_keys=frozenset(templates),
        )
        result = []
        for fact in facts:
            proposition_key = fact.proposition.stable_key()
            route = routes.for_proposition(proposition_key)
            template = templates.get(proposition_key)
            if (route is None or route.relation_kind not in EVENT_TIME_RELATION_KINDS
                    or template is None):
                continue
            generation = core.generation_input(fact.proposition)
            definition = generation.proposition.definition
            structure_key = RelationSurfaceStructureAdapter.from_facts(
                (fact,))[0].relation_structure_key()
            result.append(EventTimeRelationGenerationSeed(
                definition.proposition,
                definition.predicate,
                definition.context,
                template.connector,
                event_time_input_structure_identity(structure_key),
                tuple(item.identity_for(definition.proposition)
                      for item in definition.bindings),
                tuple(item.filler for item in definition.bindings),
                generation.proposition.scope,
            ))
    if not result:
        raise ValueError("父模型没有可复用 connector 的 active EVENT_TIME 事实")
    return tuple(sorted(set(result), key=lambda item: item.stable_key()))


def materialize(
        *,
        source_database: Path,
        source_manifest: Path,
        target_root: Path,
        resume: bool = False,
        in_place: bool = False,
        ) -> dict[str, object]:
    """追加零复制绑定；支持复制父图或显式原地阶段恢复。"""
    if type(in_place) is not bool:
        raise TypeError("in_place 必须是严格 bool")
    source = source_database.resolve(strict=True)
    manifest = source_manifest.resolve(strict=True)
    root = target_root.resolve()
    if (source.drive.upper() != "K:" or root.drive.upper() != "K:"
            or root == Path(root.anchor) or source == root
            or not source.is_file() or not manifest.is_file()):
        raise ValueError("Event/Time binding 必须使用现有 K 盘父模型和独立 K 盘目录")
    cursor_path = root / "event_time_generation_binding_cursor.json"
    source_sha = _sha256(source)
    database = source if in_place else root / "training.sqlite3"
    manifest_bytes = manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    identities = {
        "format": _FORMAT,
        "schema_version": 1,
        "source_database_sha256": source_sha,
        "source_manifest_sha256": manifest_sha,
    }
    if resume:
        if not root.is_dir() or not cursor_path.is_file():
            raise ValueError("Event/Time binding resume root 不完整")
        cursor = json.loads(cursor_path.read_bytes())
        identity_keys = tuple(
            key for key in identities
            if not (in_place and key == "source_database_sha256"))
        if (cursor.get("in_place") != int(in_place)
                or cursor.get("database_path") != str(database)
                or any(cursor.get(key) != identities[key]
                       for key in identity_keys)):
            raise ValueError("Event/Time binding 恢复时父模型身份漂移")
        if (root / "parent_manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("Event/Time binding 恢复时父 manifest 漂移")
    else:
        if root.exists():
            raise ValueError("Event/Time binding target 已存在，须换 run id 或显式 resume")
        root.mkdir(parents=True)
        (root / "parent_manifest.json").write_bytes(manifest_bytes)
        cursor = {
            **identities, "stage": 0, "in_place": int(in_place),
            "database_path": str(database),
            "database_bytes_before": source.stat().st_size,
            "database_copy_count": 0 if in_place else 1,
        }
        _write_cursor(cursor_path, cursor)

    if cursor["stage"] == 0:
        if not in_place:
            pending_database = root / "training.pending.sqlite3"
            origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            target = sqlite3.connect(str(pending_database))
            try:
                origin.backup(target)
            finally:
                target.close()
                origin.close()
            pending_database.replace(database)
        cursor["stage"] = 1
        _write_cursor(cursor_path, cursor)

    if cursor["stage"] == 1:
        seeds = _build_seeds(database)
        backend = SQLiteBackend(str(database))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            report = materialize_event_time_relation_generation_bindings(
                context, seeds)
            backend.commit()
        finally:
            backend.close()
        cursor.update(stage=2, **report)
        _write_cursor(cursor_path, cursor)

    if cursor["stage"] == 2:
        with TrainedGraphQueryBridge(database) as bridge:
            bindings = bridge.event_time_generation_bindings
            if len(bindings) != cursor["generation_binding_count"]:
                raise ValueError("Event/Time generation binding 写后恢复数量漂移")
            if any(item.proposition_key not in bridge.core_fact_index
                   for item in bindings):
                raise ValueError("Event/Time generation binding 未接 active Core")
            binding_keys = tuple(item.stable_key() for item in bindings)
        database_bytes = database.stat().st_size
        receipt = {
            **cursor,
            "stage": 3,
            "model_read_only_after_materialization": 1,
            "source_body_read_for_binding": 0,
            "successor_answer_route": 0,
            "semantic_object_copy_count": 0,
            "binding_keys": [list(item) for item in binding_keys],
            "database_bytes": database_bytes,
            "database_bytes_added": (
                database_bytes - int(cursor["database_bytes_before"])),
            "database_sha256": _sha256(database),
            "free_dialogue_complete": 0,
        }
        (root / "event_time_generation_binding_receipt.json").write_bytes(
            _json_bytes(receipt))
        cursor["stage"] = 3
        _write_cursor(cursor_path, cursor)

    receipt_path = root / "event_time_generation_binding_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("Event/Time generation binding receipt 缺失")
    return json.loads(receipt_path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    """从显式父模型和来源 manifest 构建新的 K 盘模型谱系。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--in-place", action="store_true",
        help="在 source-database 原地追加；target-root 只保存 cursor/receipt")
    args = parser.parse_args(argv)
    receipt = materialize(**vars(args))
    print(json.dumps({key: receipt[key] for key in (
        "stage", "generation_binding_count",
        "generation_binding_statement_count", "reused_object_count",
        "semantic_object_copy_count", "database_bytes", "database_sha256",
        "free_dialogue_complete",
    )}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["materialize"]
