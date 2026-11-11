"""Persist native composition syntax and licensed R-01 atoms in a fresh run Core."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.alias_relation_course import AliasRelationCourseLoader
from pure_integer_ai.experiments.language_generation_connector_graph import LanguageGenerationConnectorGraph
from pure_integer_ai.experiments.materialize_trained_relation_generation import _advance_graph_id_pool
from pure_integer_ai.experiments.operation_composition_course import PROFILE, OperationCompositionWitness
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import GenerationCandidateAliasCourseRequest
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import build_alias_relation_manifest
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import AliasRelationManifestProfile
from pure_integer_ai.experiments.ph2_grounded_answer_order import install_source_generation_order
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime, _definition_graph, _lifecycle
from pure_integer_ai.experiments.trained_multi_claim_composer import materialize_multi_claim_composition_graph, read_multi_claim_composition_graph
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


TABLE = "native_operation_composition"
PARTS = "native_operation_composition_part"


def _register(backend):
    register_extension_table(backend, TABLE, [("variant_id", TYPE_INT), ("witness_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("variant_id",)], recovery_key=("variant_id",))
    register_extension_table(backend, PARTS,
        [("variant_id", TYPE_INT), ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("variant_id", "part_ordinal")],
        recovery_key=("variant_id", "part_ordinal"))


def materialize_native_operation_composition(course, database, graph_database):
    """Append only syntax/alias/order learning; no output Proposition is added."""
    return materialize_native_operation_compositions((course,), database, graph_database)


def materialize_native_operation_compositions(courses, database, graph_database):
    """Materialize one frozen bundle; never copy Core or restart completed syntax."""
    if (type(courses) is not tuple or not courses
            or len({course.variant.variant_id for course in courses}) != len(courses)):
        raise ValueError("native syntax bundle requires unique complete courses")
    database = Path(database).resolve(strict=True)
    target = Path(graph_database).resolve()
    if target.exists():
        raise FileExistsError("native operation composition requires a new graph identity")
    with TrainedGenerationConnectorRuntime(database) as runtime:
        if any(template.connector == course.template.connector for template in runtime.templates() for course in courses):
            raise ValueError("native operation composition was already materialized")
    alias_manifests = tuple(build_alias_relation_manifest(AliasRelationManifestProfile(
        (PROFILE, 1), course.witness.source_sha256, course.witness.source, course.order.scope, 1),
        GenerationCandidateAliasCourseRequest(course.template.language_branch, course.aliases)) for course in courses)
    if len({course.template.language_branch for course in courses}) != 1:
        raise ValueError("native syntax bundle requires one explicit language branch")
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _advance_graph_id_pool(backend)
        order_count = 0
        for course, alias_manifest in zip(courses, alias_manifests, strict=True):
            IntegerIdentityRegistry(backend).register(IDENTITY_SOURCE_RECORD, course.witness.source.stable_key())
            lifecycle = _lifecycle(context, course.template.language_branch)
            parent = _definition_graph(context, course.template.language_branch, lifecycle)
            graph = LanguageGenerationConnectorGraph(context.graph_ontology, lifecycle.order_graph,
                                                      parent.predicates, course.values)
            order_count += install_source_generation_order(course.order, lifecycle)
            graph.materialize(course.template, scope=course.order.scope, provenance_kind=PROFILE,
                              content_version=1, qualifiers=course.order.qualifiers)
            AliasRelationCourseLoader(alias_manifest, alias_manifest.sha256()).load(context)
        backend.commit()
    finally:
        backend.close()
    materialize_native_operation_composition_index(courses, database, target)
    return {"order_count": order_count, "literal_realization_count": sum(len(item.entries) for item in alias_manifests),
            "new_core_claim_count": 0, "free_dialogue_complete": 0}


def materialize_native_operation_composition_index(courses, database, graph_database):
    """Index already learned native syntax; no Core writes or training repeat."""
    target = Path(graph_database).resolve()
    if target.exists():
        raise FileExistsError("native composition index already exists")
    with TrainedGenerationConnectorRuntime(database) as runtime:
        templates = tuple(template.stable_key() for template in runtime.templates())
        if any(templates.count(course.template.stable_key()) != 1 for course in courses):
            raise ValueError("native composition index requires exact already learned G/S-07 contracts")
    return _write_native_composition_index(tuple((course.variant, course.witness) for course in courses), target)


def extend_native_operation_composition_index(parent_graph, courses, database, graph_database):
    """保留父图全部完整生成合同，仅索引已真实学习的新模板，不重训父语法。"""
    variants, witnesses, _metadata = read_native_operation_composition_graph(parent_graph)
    members = (*zip(variants, witnesses, strict=True),
               *((course.variant, course.witness) for course in courses))
    target = Path(graph_database).resolve()
    if target.drive.upper() != "K:" or target.exists():
        raise ValueError("native composition extension requires a new explicit K graph")
    if not courses or len({variant.variant_id for variant, _witness in members}) != len(members):
        raise ValueError("native composition extension duplicates an inherited variant")
    with TrainedGenerationConnectorRuntime(database) as runtime:
        templates = tuple(template.stable_key() for template in runtime.templates())
        if (any(templates.count(witness.template_key) != 1 for _variant, witness in members)
                or any(course.template.stable_key() != course.witness.template_key for course in courses)):
            raise ValueError("native composition extension lacks exact learned parent or child templates")
    return _write_native_composition_index(tuple(members), target)


def _write_native_composition_index(members, target):
    """按完整variant/witness有序记录保存索引，本函数不创建或执行生成模板。"""
    if any(variant.variant_id != witness.variant_id for variant, witness in members):
        raise ValueError("native composition variant and witness differ")
    materialize_multi_claim_composition_graph(tuple(variant for variant, _witness in members), target)
    backend = SQLiteBackend(str(target))
    try:
        _register(backend)
        for variant, witness in members:
            key = witness.stable_key()
            backend.insert(TABLE, {"variant_id": variant.variant_id, "witness_hash": _hash(key)})
            backend.insert_many(PARTS, [{"variant_id": variant.variant_id, "part_ordinal": index,
                                        "part_value": value} for index, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()
    return {"indexed_native_course_count": len(members), "native_training_reexecution_count": 0,
            "core_write_count": 0, "free_dialogue_complete": 0}


def read_native_operation_composition_graph(database):
    variants, metadata = read_multi_claim_composition_graph(database)
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        rows = backend.select(TABLE, order_by="variant_id")
        parts = backend.select(PARTS, order_by="variant_id, part_ordinal")
        if tuple(row["variant_id"] for row in rows) != tuple(variant.variant_id for variant in variants):
            raise ValueError("native composition witness does not cover every graph variant")
        if any(part["variant_id"] not in {row["variant_id"] for row in rows} for part in parts):
            raise ValueError("native composition graph has orphan witness members")
        witnesses = []
        for row in rows:
            members = tuple(part for part in parts if part["variant_id"] == row["variant_id"])
            if tuple(part["part_ordinal"] for part in members) != tuple(range(len(members))):
                raise ValueError("native composition witness is incomplete")
            key = tuple(part["part_value"] for part in members)
            if _hash(key) != row["witness_hash"]:
                raise ValueError("native composition full witness identity drifted")
            witness = OperationCompositionWitness.from_stable_key(key)
            if witness.variant_id != row["variant_id"]:
                raise ValueError("native composition variant identity drifted")
            witnesses.append(witness)
        return variants, tuple(witnesses), metadata
    finally:
        backend.close()
