"""Bind an integer relation graph-course to the in-graph generation owner.

The W-06 relation owner is intentionally kept separate from this module.  This
module only consumes the already materialized relation graph, then writes the
LanguageBranch, S-07 structure, relation connector and R-01 realization facts
into the same SQLite database.  No course surface or answer table is read by
the recovered runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_integer_w06_learning_runtime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
    GenerationCandidateRealizationBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    AliasRelationManifestProfile,
    build_alias_relation_manifest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_LITERAL
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    install_relation_frame_order,
)
from pure_integer_ai.experiments.relation_generation_protocol import (
    RELATION_CONNECTOR_PROFILE,
)
from pure_integer_ai.experiments.relation_generation_structure import (
    RelationConnectorCourse,
    compile_relation_connector,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    _DIALOGUE_GENERATION_NAMESPACE,
    _definition_graph,
    _surface_protocol,
    _value_protocol,
)
from pure_integer_ai.experiments.typed_dialogue_generation_owner import (
    _build_lifecycle_for_branch,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationGenerationInput,
    ActiveRelationSurface,
    RelationSurfaceFrame,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend


class IntegerRelationGenerationBridgeError(RuntimeError):
    """The integer relation generation contract is not closed."""


@dataclass(frozen=True, slots=True)
class IntegerRelationGenerationMaterialization:
    """Receipt-level identity for one bounded same-database materialization."""

    database: Path
    branch: ObjectIdentity
    connector_count: int
    realization_count: int
    proposition_keys: tuple[tuple[int, ...], ...]
    alias_manifest_sha256: str
    active_proposition_count: int = 0
    uncovered_proposition_keys: tuple[tuple[int, ...], ...] = ()
    selected_proposition_count: int = 0
    selection_explicit: int = 0


def _stable_course_digest(
        courses: tuple[RelationConnectorCourse, ...],
        ) -> bytes:
    """Hash only integer connector identities and graph spans."""
    values = []
    for course in courses:
        values.extend(course.template.stable_key())
        values.extend(course.qualifiers)
        for alias in course.aliases:
            values.extend(alias.stable_key())
    return hashlib.sha256(
        (",".join(str(value) for value in values)).encode("ascii")
    ).digest()


def _materialize_generation_protocols(ctx, branch: ObjectIdentity):
    """Materialize the frozen S-07 and connector predicate identities."""
    # The training owner is the canonical first-writer of the 25 S-07
    # predicates/states.  The read-only runtime's _lifecycle is deliberately
    # not used here because it rejects an unmaterialized generation graph.
    lifecycle = _build_lifecycle_for_branch(ctx, branch)
    ontology = ctx.graph_ontology
    for index in range(1, 22):
        ontology.materialize(concept_identity(
            (*_DIALOGUE_GENERATION_NAMESPACE, 72, index),
            owner=branch.owner,
            versions=branch.versions,
        ))
    return lifecycle, _definition_graph(ctx, branch, lifecycle)


def materialize_integer_relation_generation(
        database: str | Path,
        *,
        branch_key: tuple[int, ...],
        family_key: tuple[int, ...],
        representative_limit: int = 1,
        require_full_active_coverage: bool = False,
        surface_artifact_database: str | Path | None = None,
        proposition_keys: tuple[tuple[int, ...], ...] | None = None,
        isolated_preflight_database: str | Path | None = None,
        ) -> IntegerRelationGenerationMaterialization:
    """Write a bounded relation-generation contract into an existing W-06 DB.

    Legacy bounded runs select one representative per predicate/role shape.
    Full coverage is explicit and rejects an insufficient budget before writes.
    Each compiled connector retains its actual proposition provenance; runtime
    selection still requires that exact graph identity, not a shape fallback.
    可选整数表面图支持无旧正文的候选模型；显式命题子集仅约束本轮新增生成
    对象，必须完整属于 active Core，不能把子集覆盖宣称为整个模型覆盖。
    """
    path = Path(database).resolve()
    if not path.is_file():
        raise ValueError("integer relation database 不存在")
    preflight = (path.parent / (path.stem + ".generation-r01-preflight.sqlite3")
        if isolated_preflight_database is None else Path(isolated_preflight_database).resolve())
    if isolated_preflight_database is not None and (
            preflight.drive.upper() != 'K:' or preflight == path
            or preflight.is_relative_to(path.parent) or not preflight.parent.is_dir()):
        raise ValueError('explicit generation preflight must be outside the K model directory')
    if (not isinstance(branch_key, tuple) or not branch_key
            or any(type(item) is not int or item <= 0 for item in branch_key)):
        raise ValueError("branch_key 必须是正整数 tuple")
    if (not isinstance(family_key, tuple) or not family_key
            or any(type(item) is not int or item <= 0 for item in family_key)):
        raise ValueError("family_key 必须是正整数 tuple")
    if type(representative_limit) is not int or representative_limit <= 0:
        raise ValueError("representative_limit 必须为正整数")
    if type(require_full_active_coverage) is not bool:
        raise ValueError("require_full_active_coverage must be bool")

    # Capture immutable Core/Span inputs before opening the writer.  The
    # generation compiler consumes these identities, not the course source.
    with TrainedRelationGraphRuntime(path,
            surface_artifact_database=surface_artifact_database) as relation:
        active_facts = relation.active_surface_facts()
        facts = active_facts
        if proposition_keys is not None:
            if (not isinstance(proposition_keys, tuple) or not proposition_keys
                    or any(not isinstance(key, tuple) or not key
                           or any(type(value) is not int or value < 0 for value in key)
                           for key in proposition_keys)
                    or len(set(proposition_keys)) != len(proposition_keys)):
                raise ValueError("增量生成必须声明唯一完整命题整数键")
            requested = set(proposition_keys)
            facts = tuple(fact for fact in active_facts if fact.proposition.stable_key() in requested)
            if {fact.proposition.stable_key() for fact in facts} != requested:
                raise IntegerRelationGenerationBridgeError("增量生成包含非 active 命题")
        if require_full_active_coverage and len(facts) > representative_limit:
            raise IntegerRelationGenerationBridgeError(
                "full active generation coverage exceeds the declared budget")
        frames = {
            item.proposition: item
            for item in relation.active_surface_frames()
        }
        selected: list[tuple[
            ActiveRelationSurface, RelationSurfaceFrame,
            ActiveRelationGenerationInput,
        ]] = []
        shapes: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = set()
        for fact in facts:
            shape = (
                fact.predicate.stable_key(),
                tuple(item.role.stable_key() for item in fact.bindings),
            )
            if not require_full_active_coverage and shape in shapes:
                continue
            frame = frames.get(fact.proposition)
            if frame is None:
                raise IntegerRelationGenerationBridgeError(
                    "active relation 缺少 Span frame")
            selected.append((fact, frame, relation.generation_input(
                fact.proposition)))
            shapes.add(shape)
            if len(selected) >= representative_limit:
                break
    if not selected:
        raise IntegerRelationGenerationBridgeError(
            "active relation 没有可物化 generation representative")

    backend = SQLiteBackend(str(path))
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        branch = language_branch_identity(
            branch_key,
            versions=selected[0][2].proposition.definition.source.versions,
        )
        ontology.materialize(branch)
        lifecycle, graph = _materialize_generation_protocols(ctx, branch)
        values = _value_protocol(branch)
        surface = _surface_protocol(branch)
        courses: list[RelationConnectorCourse] = []
        for fact, frame, source in selected:
            course = compile_relation_connector(
                fact,
                frame,
                source,
                branch,
                family_key,
                values,
                surface,
            )
            values = course.values
            # The compiler extends the ordinal protocol from the actual
            # RoleBinding coordinates.  Rebind the graph facade to that
            # frozen protocol before its first connector validation.
            graph = LanguageGenerationConnectorGraph(
                ontology,
                lifecycle.order_graph,
                graph.predicates,
                values,
            )
            install_relation_frame_order(course, lifecycle)
            graph.materialize(
                course.template,
                scope=source.proposition.scope,
                provenance_kind=RELATION_CONNECTOR_PROFILE,
                content_version=1,
                qualifiers=course.qualifiers,
            )
            courses.append(course)

        bindings = {}
        for course in courses:
            for item in course.aliases:
                if item.rule != RULE_LITERAL:
                    raise IntegerRelationGenerationBridgeError(
                        "relation generation alias 必须是 literal route")
                key = (item.origin, item.representation)
                prior = bindings.get(key)
                if prior is None:
                    bindings[key] = item
                    continue
                merged = tuple(sorted(set(
                    (*prior.forming_evidence_keys,
                     *item.forming_evidence_keys))))
                bindings[key] = GenerationCandidateRealizationBinding(
                    item.origin, item.representation, item.rule, merged)

        owner_source = selected[0][2].proposition.definition.source
        profile = AliasRelationManifestProfile(
            (20261008, 1),
            tuple(_stable_course_digest(tuple(courses))),
            owner_source,
            document_scope(owner_source),
            1,
        )
        request = GenerationCandidateAliasCourseRequest(
            branch,
            tuple(bindings.values()),
        )
        manifest = build_alias_relation_manifest(profile, request)
        loaded = AliasRelationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(
            ctx,
            isolated_preflight_database=preflight,
        )
        if loaded.report.active_count != len(manifest.entries):
            raise IntegerRelationGenerationBridgeError(
                "R-01 generation manifest 未完整 active")
        backend.commit()
        return IntegerRelationGenerationMaterialization(
            path,
            branch,
            len(courses),
            len(manifest.entries),
            tuple(course.source.proposition.definition.proposition.stable_key()
                  for course in courses),
            manifest.sha256(),
            len(active_facts),
            tuple(sorted({fact.proposition.stable_key() for fact in facts}
                         - {course.source.proposition.definition.proposition.stable_key()
                            for course in courses})),
            len(facts),
            int(proposition_keys is not None),
        )
    except BaseException:
        # Closing an uncommitted SQLite connection rolls back the bounded
        # materialization; no partial generation contract is published.
        raise
    finally:
        backend.close()


__all__ = [
    "IntegerRelationGenerationBridgeError",
    "IntegerRelationGenerationMaterialization",
    "materialize_integer_relation_generation",
]
