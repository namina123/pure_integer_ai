"""在显式 K 盘候选中增量学习 SQLite 教师知识，不重训父图。"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sqlite3
import time

from pure_integer_ai.experiments.authored_relation_sqlite_course import (
    build_sqlite_relation_adapter, file_sha256,
)
from pure_integer_ai.cognition.understanding.occurrence_index import OccurrenceIndex
from pure_integer_ai.cognition.understanding.span_index import SpanIndex
from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.conversation_typed_relation_bridge import _materialize_relation_spans
from pure_integer_ai.experiments.incremental_relation_model import (
    IncrementalRelationTrainingRuntime, materialize_extended_relation_surfaces,
)
from pure_integer_ai.experiments.integer_relation_generation_bridge import materialize_integer_relation_generation
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.materialize_trained_relation_generation import _advance_graph_id_pool
from pure_integer_ai.experiments.ph2_w06_learning import build_w06_learning_runtime
from pure_integer_ai.experiments.relation_knowledge_checkpoint import model_source_body_count
from pure_integer_ai.experiments.relation_model_inheritance import (
    inherit_parent_model_sidecars, parent_model_identity, validate_inherited_native_graphs,
)
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_surface_artifact import read_relation_surface_artifact
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.source_record import SourceRecordRepository, register_source_record_table


def train(args):
    """一份可写 Core、事务化学习和阶段 cursor；禁止覆盖父库或静默改源恢复。"""
    course, parent = args.course.resolve(strict=True), args.parent_model.resolve(strict=True)
    root = args.run_root.resolve()
    if (root.drive.upper() != "K:" or root == Path(root.anchor)
            or parent.drive.upper() != "K:" or root.is_relative_to(parent)
            or parent.is_relative_to(root) or not root.parent.is_dir()):
        raise ValueError("知识增量必须使用独立显式 K 盘根")
    if root.exists() != bool(args.resume):
        raise ValueError("新运行不能覆盖；已有运行只能显式 resume")
    if shutil.disk_usage(root.parent).free < 10 * 1024**3:
        raise RuntimeError("K 盘不足最低训练余量")
    parent_components = parent_model_identity(parent)
    adapter, source_receipt = build_sqlite_relation_adapter(course, args.prototype_root,
        limit=args.limit, offset=args.offset)
    identity = {"course_sha256": source_receipt["course_sha256"],
        "parent_core_sha256": file_sha256(parent / "training.sqlite3").hex(),
        "parent_surface_sha256": file_sha256(parent / "relation_surfaces.sqlite3").hex(),
        "parent_components": parent_components,
        "limit": args.limit, "selection_offset": args.offset,
        "source_body_storage": "EXTERNAL_TYPED_ARCHIVE_V1",
        "code_sha256": {name: file_sha256(path).hex() for name, path in (
            ("trainer", Path(__file__)),
            ("sqlite_course", Path(build_sqlite_relation_adapter.__code__.co_filename)),
            ("incremental_model", Path(materialize_extended_relation_surfaces.__code__.co_filename)),
            ("generation", Path(materialize_integer_relation_generation.__code__.co_filename)),
            ("language_protocols", Path(install_language_graph_protocols.__code__.co_filename)),
            ("occurrence", Path(OccurrenceIndex.__init__.__code__.co_filename)),
            ("span", Path(SpanIndex.__init__.__code__.co_filename)),
            ("source_record", Path(SourceRecordRepository.__init__.__code__.co_filename)),
            ("model_inheritance", Path(parent_model_identity.__code__.co_filename)),
            ("checkpoint", Path(model_source_body_count.__code__.co_filename)))}}
    if not root.exists():
        root.mkdir()
    cursor = sqlite3.connect(root / "training_cursor.sqlite3")
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS run_identity(name TEXT PRIMARY KEY,value_utf8 BLOB NOT NULL) STRICT;
        CREATE TABLE IF NOT EXISTS stage(stage_id INTEGER PRIMARY KEY,
            core_sha256 BLOB NOT NULL CHECK(length(core_sha256)=32)) STRICT;
        CREATE TABLE IF NOT EXISTS resource_sample(stage_id INTEGER PRIMARY KEY,
            endpoint_utf8 BLOB NOT NULL,artifact_bytes INTEGER NOT NULL,
            free_disk_bytes INTEGER NOT NULL) STRICT;
    """)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    stored = cursor.execute("SELECT value_utf8 FROM run_identity WHERE name='frozen'").fetchone()
    if stored is not None and stored[0] != encoded:
        raise ValueError("恢复时课程、父图或算法身份漂移，须保留旧运行并另建身份")
    cursor.execute("INSERT OR IGNORE INTO run_identity VALUES('frozen',?)", (encoded,))
    cursor.commit()
    model = root / "model"
    model.mkdir(exist_ok=True)
    database, parent_core = model / "training.sqlite3", parent / "training.sqlite3"
    parent_surfaces = parent / "relation_surfaces.sqlite3"
    stage = cursor.execute("SELECT coalesce(max(stage_id),0) FROM stage").fetchone()[0]
    if stage and (not database.is_file() or file_sha256(database) != cursor.execute(
            "SELECT core_sha256 FROM stage WHERE stage_id=?", (stage,)).fetchone()[0]):
        raise ValueError("候选 Core 与最后已提交阶段不同，保留现场并换新身份")
    source_archive = root / "training_sources.sqlite3"

    def checkpoint(number):
        """阶段只在真实 SQLite 提交后单调落盘，保留每次模型身份。"""
        resources = current_process_resource_endpoint().to_value()
        artifact_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        free_bytes = shutil.disk_usage(root).free
        if (free_bytes < 10 * 1024**3 or artifact_bytes > 4 * 1024**3
                or max(resources.get("working_set_bytes", 0),
                       resources.get("process_peak_working_set_bytes", 0)) > 8 * 1024**3):
            raise RuntimeError("知识增量的磁盘或内存规模预算耗尽")
        cursor.execute("INSERT INTO resource_sample VALUES(?,?,?,?)", (
            number, json.dumps(resources, sort_keys=True).encode("ascii"), artifact_bytes, free_bytes))
        cursor.execute("INSERT INTO stage VALUES(?,?)", (number, file_sha256(database)))
        cursor.commit()

    started = time.perf_counter_ns()
    if stage < 1:
        if database.exists():
            raise ValueError("未提交阶段已有候选库，保留现场并换新身份")
        pending = model / "training.pending.sqlite3"
        if pending.exists():
            raise ValueError("未完成复制仍在，保留现场并换新身份")
        with (closing(sqlite3.connect(parent_core.as_uri() + "?mode=ro", uri=True)) as source,
              closing(sqlite3.connect(pending)) as copied):
            source.backup(copied)
        pending.rename(database)
        checkpoint(1)
    if stage < 2:
        backend = SQLiteBackend(str(database))
        archive = SQLiteBackend(str(source_archive))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            _semantic, occurrence, span = dialogue_semantic_protocols()
            register_source_record_table(archive)
            sources = SourceRecordRepository(archive, registry=context.scoped_identity_store.registry)
            install_language_graph_protocols(context, occurrence_protocol=occurrence,
                                             span_protocol=span, source_records=sources)
            context.graph_ontology.enable_physical_statement_projection()
            runtime = build_w06_learning_runtime(backend, adapter, context=context)
            _materialize_relation_spans(context, runtime, adapter)
            archive.commit()
            backend.commit()
        finally:
            archive.close()
            backend.close()
        checkpoint(2)
    pre_generation = root / "pre_generation_surfaces.sqlite3"
    if stage < 3:
        if not pre_generation.exists():
            materialize_extended_relation_surfaces(database, parent_core, parent_surfaces, pre_generation,
                                                 source_archive=source_archive)
        with IncrementalRelationTrainingRuntime(database, parent_core=parent_core,
                parent_surfaces=parent_surfaces, source_archive=source_archive) as extended:
            parent_facts, _frames, _meta = read_relation_surface_artifact(parent_surfaces)
            previous = {item.proposition for item in parent_facts}
            added = tuple(item.proposition.stable_key() for item in extended.active_surface_facts()
                          if item.proposition not in previous)
        if len(added) != source_receipt["record_count"]:
            raise ValueError("新 active 命题数未匹配冻结教师分母")
        digest_parts = tuple(int.from_bytes(bytes.fromhex(source_receipt["course_sha256"])[i:i+4], "big")
                             for i in range(0, 32, 4))
        generated = materialize_integer_relation_generation(database,
            branch_key=(92456, 1, *digest_parts), family_key=(92456, 2, *digest_parts),
            representative_limit=len(added), require_full_active_coverage=True,
            surface_artifact_database=pre_generation, proposition_keys=added,
            isolated_preflight_database=root/'generation-r01-preflight.sqlite3')
        if generated.uncovered_proposition_keys:
            raise ValueError("新增知识缺少实际生成关系")
        (root / "generation_receipt.json").write_text(json.dumps({
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(generated).items()}, sort_keys=True), encoding="ascii")
        checkpoint(3)
    if stage < 4:
        output = model / "relation_surfaces.sqlite3"
        if not output.exists():
            materialize_extended_relation_surfaces(database, parent_core, parent_surfaces, output,
                                                 source_archive=source_archive)
        inherited_components = inherit_parent_model_sidecars(parent, root, parent_components)
        with (TrainedRelationGraphRuntime(database, surface_artifact_database=output) as relation,
              TrainedGenerationConnectorRuntime(database, shared_relation_runtime=relation) as generator):
            inherited_native = validate_inherited_native_graphs(model, relation, generator)
            parent_facts, _frames, _meta = read_relation_surface_artifact(parent_surfaces)
            previous = {item.proposition for item in parent_facts}
            outputs = []
            for fact in relation.active_surface_facts():
                if fact.proposition in previous:
                    continue
                generated = generator.generate_relation(relation.generation_input(fact.proposition), fact)
                if not generated.surface or len(generated.representations) < len(fact.bindings):
                    raise ValueError("新增关系没有经六层图生成输出")
                outputs.append({"surface": generated.surface,
                    "proposition": fact.proposition.stable_key(),
                    "connector": generated.connector.stable_key()})
        body_count = model_source_body_count(database)
        if body_count:
            raise ValueError("训练后模型含来源正文，不能作为模型本体候选")
        receipt = {"format": "PURE_INTEGER_KNOWLEDGE_EXTENSION_TRAINING_V1",
            "source": source_receipt, "identity": identity, "new_generated_relations": len(outputs),
            "outputs": outputs, "model_sha256": file_sha256(database).hex(),
            "parent_core_unchanged": int(file_sha256(parent_core).hex() == identity["parent_core_sha256"]),
            "parent_copy_count": 1, "source_body_restored": 0,
            "inherited_components": inherited_components, "inherited_native": inherited_native,
            "language_schema_version": 3,
            "model_source_body_count": body_count,
            "training_source_archive_sha256": file_sha256(source_archive).hex(),
            "elapsed_ns": time.perf_counter_ns() - started,
            "free_dialogue_complete": 0, "production_authorized": 0}
        (root / "training_receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="ascii")
        checkpoint(4)
    cursor.close()
    receipt = json.loads((root / "training_receipt.json").read_bytes())
    print(json.dumps({key: value for key, value in receipt.items() if key != "outputs"}, sort_keys=True), flush=True)


def main():
    """显式接收课程、父模型、运行根和恢复选项，默认只跑四条代表性规模探针。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
