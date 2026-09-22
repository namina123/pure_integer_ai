"""Materialize licensed generic response acts into a copied trained graph.

The external integer course is consumed only here. The target SQLite receives
connector ontology, S-07 order, LanguageAtom/R-01 routes, and integer source
identities; it never receives source text and runtime never reopens the course.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_ATOM, OBJECT_REPRESENTATION,
)
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.experiments.alias_relation_course import AliasRelationCourseLoader
from pure_integer_ai.experiments.materialize_trained_relation_generation import (
    _advance_graph_id_pool,
    _canonical_json,
    _profile_from_prefix,
    _read_object,
    _sha256,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    build_alias_relation_manifest,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    install_source_generation_order,
)
from pure_integer_ai.experiments.response_generation_course import (
    compile_generic_response_course,
)
from pure_integer_ai.experiments.response_generation_graph import (
    GENERIC_RESPONSE_GRAPH_ROLES,
    generic_response_context_marker, generic_response_context_role,
    generic_response_context_type, generic_response_graph_role,
    generic_response_graph_type, materialize_response_connector,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
    _ALIAS_BUDGET,
    _definition_graph,
    _generation_protocols,
    _lifecycle,
    _surface_protocol,
)
from pure_integer_ai.experiments.unknown_response_course import load_integer_course
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_RESPONSE_VARIANT, IDENTITY_SOURCE_RECORD,
    IntegerIdentityRegistry, integer_identity_hash,
    register_assertion_identity_tables,
)
from pure_integer_ai.storage.backend import SQLiteBackend


def _source_record_count(database: Path) -> int:
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM source_record").fetchone()[0])
    finally:
        connection.close()


def _reject_existing_semantic_variants(
        database: Path, variant_keys: tuple[tuple[int, ...], ...]) -> None:
    """Fail before creating a run when the same response structure exists."""
    backend = SQLiteBackend(str(database), read_only=True)
    try:
        register_assertion_identity_tables(backend)
        identities = IntegerIdentityRegistry(backend)
        duplicates = tuple(
            key for key in variant_keys
            if identities.find(IDENTITY_RESPONSE_VARIANT, key) is not None
        )
    finally:
        backend.close()
    if duplicates:
        raise ValueError(
            "generic response course repeats semantic variants already "
            "registered in the target graph")


def _recovered_constant_bindings(item, owner, values, surface):
    """Validate legacy and graph-slot generic connector contracts."""
    bindings = item.template.bindings
    roles = tuple(binding for binding in bindings
                  if binding.source == values.role_filler_source)
    constants = tuple(binding for binding in bindings
                      if binding.source == values.constant_source)
    slots = {slot.slot: slot for slot in item.template.slots}
    directives = {entry.slot: entry for entry in item.template.surface}
    contexts = tuple(
        binding for binding in roles
        if binding.role == generic_response_context_role(owner.branch)
    )
    if (item.template.context != (
            generic_response_context_marker(owner.branch),)
            or len(contexts) != 1 or len(constants) < 1
            or len(roles) + len(constants) != len(bindings)):
        raise ValueError("recovered generic connector context shape differs")
    context = contexts[0]
    if (values.ordinal_value(context.ordinal) != 0
            or slots[context.slot].role != context.role
            or slots[context.slot].value_type
            != generic_response_context_type(owner.branch)
            or directives[context.slot].action != surface.silent_action
            or directives[context.slot].surface_prefix_steps
            or context.slot == item.marker_slot):
        raise ValueError("recovered generic connector context differs")
    graph_keys = []
    for binding in roles:
        if binding == context:
            continue
        categories = tuple(
            category for category in sorted(GENERIC_RESPONSE_GRAPH_ROLES)
            if binding.role == generic_response_graph_role(
                owner.branch, category)
        )
        if len(categories) != 1:
            raise ValueError("recovered generic graph role is ambiguous")
        category = categories[0]
        ordinal = values.ordinal_value(binding.ordinal)
        if (ordinal <= 0
                or slots[binding.slot].role != binding.role
                or slots[binding.slot].value_type
                != generic_response_graph_type(owner.branch, category)
                or directives[binding.slot].action != surface.emit_action
                or directives[binding.slot].surface_prefix_steps):
            raise ValueError("recovered generic graph slot differs")
        graph_keys.append((category, ordinal))
    if len(set(graph_keys)) != len(graph_keys):
        raise ValueError("recovered generic graph slots are duplicated")
    if any(binding.constant is None
           or binding.constant.object_kind != OBJECT_LANGUAGE_ATOM
           or directives[binding.slot].action != surface.emit_action
           for binding in constants):
        raise ValueError("recovered generic constant/surface differs")
    return constants


def _compile(database: Path, course, digest: tuple[int, ...]):
    """Compile against the parent's one explicit branch and R-01 family."""
    with TrainedGenerationConnectorRuntime(database) as runtime:
        if len(runtime._branches) != 1:
            raise ValueError("generic response materialization requires one LanguageBranch")
        owner = runtime._branches[0]
        profiles = runtime._discover_alias_protocols()
        if len(profiles) != 1:
            raise ValueError("generic response materialization requires one R-01 profile")
        profile = _profile_from_prefix(*profiles[0])
        alias = runtime.alias_runtime(owner.branch)
        families = {
            representation_parts(binding.filler)[0]
            for fact in alias.closure.consumer.lookup_relation(
                alias.selector.protocol.realizes_relation)
            for binding in fact.proposition.bindings
            if binding.filler.object_kind == OBJECT_REPRESENTATION
        }
        if len(families) != 1:
            raise ValueError("generic response target Representation family is ambiguous")
        values = owner.definition_graph.value_protocol
        protocols = _generation_protocols(runtime.context, owner.branch)
        courses = compile_generic_response_course(
            course,
            digest=digest,
            branch=owner.branch,
            values=values,
            family=next(iter(families)),
            content=protocols.content,
            surface=_surface_protocol(owner.branch),
            variant_identity_hashes=tuple(
                integer_identity_hash(
                    IDENTITY_RESPONSE_VARIANT,
                    course.semantic_variant_key(index),
                )
                for index in range(1, len(course.variants) + 1)
            ),
        )
        request = GenerationCandidateAliasCourseRequest(
            owner.branch,
            tuple(binding for item in courses for binding in item.aliases),
        )
        manifest = build_alias_relation_manifest(profile, request)
    return courses, manifest


def materialize(
        *,
        source_database: Path,
        course_path: Path,
        source_manifest: Path,
        parent_source_manifest: Path,
        target_root: Path,
        resume: bool = False,
        in_place: bool = False,
        ) -> dict:
    """Append trained graph generation objects, optionally to the source DB."""
    if type(in_place) is not bool:
        raise TypeError("in_place must be a strict bool")
    source = source_database.resolve(strict=True)
    course_file = course_path.resolve(strict=True)
    ledger_file = source_manifest.resolve(strict=True)
    parent_ledger_file = parent_source_manifest.resolve(strict=True)
    root = target_root.resolve()
    if (root.drive.upper() != "K:" or root == Path(root.anchor)
            or any(root == item or root in item.parents for item in (
                source, course_file, ledger_file, parent_ledger_file))):
        raise ValueError("generic response materialization requires a distinct K drive root")
    course_payload = course_file.read_bytes()
    ledger_payload = ledger_file.read_bytes()
    parent_ledger_payload = parent_ledger_file.read_bytes()
    course = load_integer_course(course_file, ledger_file)
    semantic_variant_keys = tuple(
        course.semantic_variant_key(index)
        for index in range(1, len(course.variants) + 1)
    )
    if len(set(semantic_variant_keys)) != len(semantic_variant_keys):
        raise ValueError("generic response course repeats a semantic variant")
    database = source if in_place else root / "training.sqlite3"
    identities = {
        "parent_database_sha256": _sha256(source),
        "course_sha256": hashlib.sha256(course_payload).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        "parent_source_manifest_sha256": hashlib.sha256(parent_ledger_payload).hexdigest(),
    }
    cursor_path = root / "generic_response_cursor.json"
    if resume:
        cursor = _read_object(cursor_path, label="generic response cursor")
        identity_keys = tuple(
            key for key in identities
            if not (in_place and key == "parent_database_sha256"))
        if (cursor.get("in_place") != int(in_place)
                or cursor.get("database_path") != str(database)
                or any(cursor.get(key) != identities[key]
                       for key in identity_keys)):
            raise ValueError("generic response resume input identity differs")
    else:
        _reject_existing_semantic_variants(source, semantic_variant_keys)
        if root.exists():
            raise ValueError("generic response target exists; use resume or a new run id")
        root.mkdir(parents=True)
        (root / "source_manifest.json").write_bytes(ledger_payload)
        (root / "parent_source_manifest.json").write_bytes(parent_ledger_payload)
        cursor = {
            "schema_version": 1,
            "stage": 0,
            "connector_cursor": 0,
            "order_count": 0,
            "in_place": int(in_place),
            "database_path": str(database),
            "database_bytes_before": source.stat().st_size,
            "database_copy_count": 0 if in_place else 1,
            **identities,
        }

    def checkpoint() -> None:
        pending = root / "generic_response_cursor.pending"
        pending.write_bytes(_canonical_json(cursor))
        pending.replace(cursor_path)

    checkpoint()
    if cursor["stage"] == 0:
        if not in_place:
            origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            copied = sqlite3.connect(str(database))
            try:
                origin.backup(copied)
            finally:
                copied.close()
                origin.close()
        cursor["source_record_count_before"] = _source_record_count(database)
        cursor["stage"] = 1
        checkpoint()
    if cursor["stage"] < 3:
        courses, alias_manifest = _compile(
            database, course, tuple(hashlib.sha256(course_payload).digest()))
        alias_sha = alias_manifest.sha256()
        if cursor.get("alias_manifest_sha256") not in (None, alias_sha):
            if not (cursor["stage"] == 1
                    and cursor["connector_cursor"] == 0
                    and cursor["order_count"] == 0):
                raise ValueError("generic response compiled identity differs on resume")
        cursor.update(
            alias_manifest_sha256=alias_sha,
            connector_count=len(courses),
            realization_count=len(alias_manifest.entries),
            semantic_variant_count=len(semantic_variant_keys),
            semantic_variant_identity_kind=IDENTITY_RESPONSE_VARIANT,
            semantic_variant_duplicate_count=0,
            semantic_variant_registry_normalized=1,
        )
        checkpoint()
        backend = SQLiteBackend(str(database))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            source_identities = IntegerIdentityRegistry(backend)
            for index, item in enumerate(courses):
                source_identities.register(
                    IDENTITY_RESPONSE_VARIANT,
                    semantic_variant_keys[index],
                )
                if index < cursor["connector_cursor"]:
                    continue
                definition = item.response
                lifecycle = _lifecycle(
                    context, definition.template.language_branch)
                graph = _definition_graph(
                    context, definition.template.language_branch, lifecycle)
                graph.value_protocol = item.values
                source_identities.register(
                    IDENTITY_SOURCE_RECORD, definition.source.stable_key())
                count = install_source_generation_order(item.order, lifecycle)
                materialize_response_connector(graph, definition)
                backend.commit()
                artifact = gzip.compress(
                    _canonical_json(list(definition.stable_key())), mtime=0)
                (root / f"generic-response-{index + 1:03d}.int.json.gz").write_bytes(
                    artifact)
                cursor.update(
                    connector_cursor=index + 1,
                    order_count=cursor["order_count"] + count,
                )
                checkpoint()
            cursor["stage"] = 2
            checkpoint()
            AliasRelationCourseLoader(alias_manifest, alias_sha).load(context)
            backend.commit()
            cursor["stage"] = 3
            checkpoint()
        finally:
            backend.close()
    if cursor["stage"] < 4:
        with TrainedGenerationConnectorRuntime(database) as runtime:
            registry = runtime.context.scoped_identity_store.registry
            recovered_semantic_variants = tuple(
                registry.read_key(
                    IDENTITY_RESPONSE_VARIANT,
                    integer_identity_hash(IDENTITY_RESPONSE_VARIANT, key),
                )
                for key in semantic_variant_keys
            )
            if recovered_semantic_variants != semantic_variant_keys:
                raise ValueError(
                    "generic response semantic variant registry did not recover")
            expected = {
                tuple(json.loads(gzip.decompress(
                    (root / f"generic-response-{index:03d}.int.json.gz").read_bytes())))
                for index in range(1, cursor["connector_count"] + 1)
            }
            recovered = tuple(item for item in runtime.response_connectors()
                              if item.stable_key() in expected)
            if {item.stable_key() for item in recovered} != expected:
                raise ValueError("generic response connectors did not recover from the target graph")
            for item in recovered:
                owner = next(owner for owner in runtime._branches
                             if owner.branch == item.template.language_branch)
                values = owner.definition_graph.value_protocol
                surface = _surface_protocol(owner.branch)
                constants = _recovered_constant_bindings(
                    item, owner, values, surface)
                for binding in constants:
                    proposal = runtime.alias_runtime(owner.branch).preview_surface(
                        binding.constant,
                        owner.branch,
                        budget=_ALIAS_BUDGET,
                        allowed_prefix_steps=(),
                    )
                    if proposal.result.selected is None:
                        raise ValueError("generic LanguageAtom has no recovered R-01 route")
        after = _source_record_count(database)
        if after != cursor["source_record_count_before"]:
            raise ValueError("generic response materialization wrote source raw text")
        database_bytes = database.stat().st_size
        receipt = {
            **cursor,
            "stage": 4,
            "recovered_connector_count": len(recovered),
            "recovered_semantic_variant_count": len(
                recovered_semantic_variants),
            "source_record_count_after": after,
            "source_text_rows_added": 0,
            "model_sha256": _sha256(database),
            "database_bytes": database_bytes,
            "database_bytes_added": (
                database_bytes - int(cursor["database_bytes_before"])),
            "semantic_object_copy_count": 0,
            "free_dialogue_complete": 0,
        }
        (root / "generic_response_receipt.json").write_bytes(
            _canonical_json(receipt))
        cursor["stage"] = 4
        checkpoint()
    return _read_object(
        root / "generic_response_receipt.json",
        label="generic response receipt")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--course", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--parent-source-manifest", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--in-place", action="store_true",
        help="append to source-database; target-root stores only cursor/receipt")
    args = parser.parse_args(argv)
    receipt = materialize(
        source_database=args.source_database,
        course_path=args.course,
        source_manifest=args.source_manifest,
        parent_source_manifest=args.parent_source_manifest,
        target_root=args.target_root,
        resume=args.resume,
        in_place=args.in_place,
    )
    print(json.dumps({key: receipt[key] for key in (
        "stage", "connector_count", "order_count", "realization_count",
        "recovered_connector_count", "source_text_rows_added",
        "database_bytes", "model_sha256", "free_dialogue_complete",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["materialize", "main"]
