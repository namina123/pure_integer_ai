"""只读消费外部纯整数 heldout response course。

Heldout connectors are compiled with the existing response-generation compiler
and loaded into an in-memory R-01 graph.  The canonical SQLite remains the
owner of Core/Memory/Dialogue protocol state; this module only supplies an
overlay for the current QueryState and never registers heldout identities in
that database.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph, StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph, StructureOrderLifecycleProtocol,
)
from pure_integer_ai.crosscut.determinism.fingerprint import integer_tuple_fingerprint
from pure_integer_ai.experiments.alias_relation_runtime import AliasRelationRuntime
from pure_integer_ai.experiments.alias_relation_course import AliasRelationCourseLoader
from pure_integer_ai.experiments.materialize_generic_response_generation import (
    _profile_from_prefix,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    build_alias_relation_manifest,
)
from pure_integer_ai.experiments.response_generation_course import (
    compile_generic_response_course,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    install_source_generation_order,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.unknown_response_course import load_integer_course
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_RESPONSE_VARIANT,
    integer_identity_hash,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_CORE, SpaceRegistry
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry,
)


class HeldoutResponseRuntimeError(RuntimeError):
    """Heldout course is not a committed, integer-only read-only overlay."""


class _HeldoutStructureOrderGraph(StructureOrderGraph):
    """Read-only S-07 facade with an explicit in-memory identity index."""

    def __init__(self, ontology, predicates):
        super().__init__(ontology, predicates)
        self._overlay_refs = {}

    def remember(self, identity, ref):
        self._overlay_refs[identity] = ref

    def resolve_structure(self, identity):
        return self._overlay_refs.get(identity) or super().resolve_structure(identity)


class _CompositeStructureOrderGraph(StructureOrderGraph):
    """Dispatch canonical and process-local heldout S-07 structures."""

    def __init__(self, canonical, overlay):
        super().__init__(overlay.ontology, overlay.predicates)
        self._canonical = canonical
        self._overlay = overlay

    def resolve_structure(self, identity):
        return (self._overlay.resolve_structure(identity)
                or self._canonical.resolve_structure(identity))

    def read_structure(self, structure):
        if structure.space_id == self._overlay.ontology.space_id:
            return self._overlay.read_structure(structure)
        return self._canonical.read_structure(structure)


class _CompositeStructureOrderLifecycle(StructureOrderLifecycleGraph):
    """Lifecycle facade that preserves canonical and heldout projections."""

    def __init__(self, canonical, overlay, order_graph):
        super().__init__(order_graph, overlay.protocol)
        self._canonical = canonical
        self._overlay = overlay

    def project(self, constraint):
        if constraint.space_id == self._overlay.order_graph.ontology.space_id:
            return self._overlay.project(constraint)
        return self._canonical.project(constraint)

    def active_constraints(self, structure):
        if structure.space_id == self._overlay.order_graph.ontology.space_id:
            return self._overlay.active_constraints(structure)
        return self._canonical.active_constraints(structure)


class _OverlayAliasRelationRuntime(AliasRelationRuntime):
    """Dispatch surface previews by LanguageAtom origin without fallback."""

    def __init__(self, canonical: AliasRelationRuntime,
                 overlay: AliasRelationRuntime,
                 overlay_origins: frozenset[ObjectIdentity]) -> None:
        # GenerationSurfaceRuntime requires an AliasRelationRuntime instance.
        # The base fields preserve the canonical protocol contract; all
        # read-only route discovery is explicitly dispatched below.
        super().__init__(canonical.closure, canonical.selector)
        self._canonical = canonical
        self._overlay = overlay
        self._overlay_origins = overlay_origins

    def preview_surface(self, origin, branch, *, budget,
                        allowed_prefix_steps=None, expected_value=None):
        runtime = (self._overlay if origin in self._overlay_origins
                   else self._canonical)
        return runtime.preview_surface(
            origin, branch, budget=budget,
            allowed_prefix_steps=allowed_prefix_steps,
            expected_value=expected_value,
        )

    def preview_reference(self, origin, *, target_kinds, budget):
        runtime = (self._overlay if origin in self._overlay_origins
                   else self._canonical)
        return runtime.preview_reference(
            origin, target_kinds=target_kinds, budget=budget)


def _source_hash(source) -> int:
    """Derive a stable positive integer source identity without registration."""
    digest = integer_tuple_fingerprint(
        source.stable_key(), domain="heldout.response.source.v1")
    value = int.from_bytes(bytes(digest[2:10]), "big") & ((1 << 63) - 1)
    return value or 1


class HeldoutResponseRuntime:
    """Canonical runtime facade plus an in-memory heldout connector overlay."""

    def __init__(self, canonical, *, course_path: str | Path,
                 manifest_path: str | Path) -> None:
        course_file = Path(course_path).resolve(strict=True)
        manifest_file = Path(manifest_path).resolve(strict=True)
        payload = course_file.read_bytes()
        manifest_payload = manifest_file.read_bytes()
        try:
            metadata = json.loads(manifest_payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HeldoutResponseRuntimeError("heldout manifest is not ASCII JSON") from exc
        if (metadata.get("format") not in {
                    "CONDITIONAL_RESPONSE_COURSE_LEDGER_V1",
                    "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
                }
                or metadata.get("split") != "heldout"
                or metadata.get("free_dialogue_claim") != 0
                or metadata.get("course_sha256") != hashlib.sha256(payload).hexdigest()
                or not metadata.get("license_ids")
                or not metadata.get("rights_basis")
                or type(metadata.get("variant_count")) is not int
                or metadata["variant_count"] <= 0):
            raise HeldoutResponseRuntimeError("heldout manifest commitment or license is invalid")
        self.canonical = canonical
        self.course_path = course_file
        self.manifest_path = manifest_file
        self.course_sha256 = hashlib.sha256(payload).hexdigest()
        self.manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        self.manifest = metadata
        course = load_integer_course(course_file, manifest_file)
        branches = tuple(canonical._branches)
        if len(branches) != 1:
            raise HeldoutResponseRuntimeError("heldout overlay requires one canonical LanguageBranch")
        owner = branches[0]
        profiles = canonical._discover_alias_protocols()
        if len(profiles) != 1:
            raise HeldoutResponseRuntimeError("heldout overlay requires one canonical R-01 profile")
        profile = _profile_from_prefix(*profiles[0])
        values = owner.definition_graph.value_protocol
        # These helpers are module-private by design, but are the same
        # recovery contract used by the canonical runtime.
        from pure_integer_ai.experiments.trained_generation_connector_runtime import (
            _generation_protocols, _surface_protocol,
        )
        protocols = _generation_protocols(canonical.context, owner.branch)
        surface_protocol = _surface_protocol(owner.branch)
        canonical_alias = canonical.alias_runtime(owner.branch)
        # Representation family is read from the canonical R-01 closure, as
        # materialization already does; no text or vocabulary is inspected.
        from pure_integer_ai.cognition.shared.identity import OBJECT_REPRESENTATION
        from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
        families = {
            representation_parts(binding.filler)[0]
            for fact in canonical_alias.closure.consumer.lookup_relation(
                canonical_alias.selector.protocol.realizes_relation)
            for binding in fact.proposition.bindings
            if binding.filler.object_kind == OBJECT_REPRESENTATION
        }
        if len(families) != 1:
            raise HeldoutResponseRuntimeError("canonical Representation family is ambiguous")
        compiled = compile_generic_response_course(
            course,
            digest=tuple(hashlib.sha256(payload).digest()),
            branch=owner.branch,
            values=values,
            family=next(iter(families)),
            content=protocols.content,
            surface=surface_protocol,
            variant_identity_hashes=tuple(
                integer_identity_hash(
                    IDENTITY_RESPONSE_VARIANT,
                    course.semantic_variant_key(index),
                )
                for index in range(1, len(course.variants) + 1)
            ),
        )
        request = GenerationCandidateAliasCourseRequest(
            owner.branch, tuple(binding for item in compiled for binding in item.aliases))
        alias_manifest = build_alias_relation_manifest(profile, request)
        self.alias_manifest_sha256 = alias_manifest.sha256()
        # The overlay backend is process-local only.  It is never copied to or
        # written through the canonical SQLite handle.
        self._overlay_backend = DictBackend()
        try:
            # Build the heldout S-07/H-06 topology with the production course
            # installer, but against a process-local backend.  Protocol
            # identities are copied from the canonical branch; definitions,
            # hypotheses, evidence and lifecycle events are newly derived from
            # the licensed integer course and never enter canonical SQLite.
            overlay_context = make_train_context(self._overlay_backend)
            overlay_registry = SpaceRegistry(self._overlay_backend)
            overlay_space_id = overlay_registry.register(
                SPACE_TYPE_CORE, "heldout-s07")
            # Each GraphObjectRepository registers external-key resolvers in
            # its ScopedIdentityStore.  The canonical context's store is
            # already owned by the first ontology; reusing it would reject
            # the second registration for identity kind 5.  Keep the backend
            # shared (so the heldout Core space remains process-local and
            # addressable), but give this ontology an independent resolver
            # registry and make the context point at that store.
            overlay_scoped = ScopedIdentityStore(self._overlay_backend)
            for index in range(1, len(course.variants) + 1):
                overlay_scoped.registry.register(
                    IDENTITY_RESPONSE_VARIANT,
                    course.semantic_variant_key(index),
                )
            overlay_context.scoped_identity_store = overlay_scoped
            overlay_context.graph_ontology = GraphOntology(
                self._overlay_backend,
                space_id=overlay_space_id,
                space_identity=overlay_registry.identity(overlay_space_id),
                scoped_identities=overlay_scoped,
            )
            canonical_lifecycle = owner.lifecycle
            canonical_order = canonical_lifecycle.order_graph
            order_refs = tuple(
                overlay_context.graph_ontology.materialize(
                    canonical_order.ontology.identity_of(ref))
                for ref in canonical_order.predicates.refs()
            )
            lifecycle_refs = tuple(
                overlay_context.graph_ontology.materialize(
                    canonical_order.ontology.identity_of(ref))
                for ref in canonical_lifecycle.protocol.predicate_refs()
            )
            protocol = canonical_lifecycle.protocol
            for identity in (
                    *protocol.state_identities(),
                    *protocol.kind_identities()):
                overlay_context.graph_ontology.materialize(identity)
            overlay_order = _HeldoutStructureOrderGraph(
                overlay_context.graph_ontology,
                StructureOrderGraphPredicates(*order_refs),
            )
            overlay_lifecycle = StructureOrderLifecycleGraph(
                overlay_order,
                StructureOrderLifecycleProtocol(
                    *lifecycle_refs,
                    *protocol.state_identities(),
                    *protocol.kind_identities(),
                    protocol.event_namespace_key,
                ),
            )
            for item in compiled:
                install_source_generation_order(item.order, overlay_lifecycle)
                structure_ref = overlay_context.graph_ontology.resolve(
                    item.response.template.structure)
                if structure_ref is None:
                    raise HeldoutResponseRuntimeError(
                        "heldout S-07 structure was not materialized")
                overlay_order.remember(item.response.template.structure, structure_ref)
            self._overlay_backend.commit()
            composite_order = _CompositeStructureOrderGraph(
                canonical_order, overlay_order)
            composite_lifecycle = _CompositeStructureOrderLifecycle(
                canonical_lifecycle, overlay_lifecycle, composite_order)
            # Install the compiled aliases through the production R-01 course
            # loader.  This creates real AliasResolutionProposal values with
            # active Evidence/Hypothesis and route discovery; the backend is
            # the process-local heldout DictBackend, never canonical SQLite.
            # Do not replace this with a surface-only map: G-03 requires the
            # typed proposal/discovery contract and must preserve route trace.
            self._overlay_alias = AliasRelationCourseLoader(
                alias_manifest, self.alias_manifest_sha256,
            ).load(overlay_context).alias
        except BaseException:
            self._overlay_backend.close()
            raise
        self._heldout = tuple(item.response for item in compiled)
        self._heldout_origins = frozenset(
            binding.origin for item in compiled for binding in item.aliases)
        self._heldout_sources = frozenset(item.source.stable_key() for item in self._heldout)
        self._composite_alias = _OverlayAliasRelationRuntime(
            canonical_alias, self._overlay_alias, self._heldout_origins)
        self._branches = (replace(owner, lifecycle=composite_lifecycle),)
        self.context = canonical.context
        self.backend = canonical.backend
        self.path = canonical.path

    def close(self) -> None:
        self._overlay_backend.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def response_connectors(self, branch=None):
        canonical = self.canonical.response_connectors(branch)
        heldout = tuple(item for item in self._heldout
                        if branch is None or item.template.language_branch == branch)
        return (*canonical, *heldout)

    def alias_runtime(self, branch):
        if branch != self._branches[0].branch:
            raise HeldoutResponseRuntimeError("heldout branch is not the canonical branch")
        return self._composite_alias

    def source_identity_for(self, source) -> int:
        if source.stable_key() in self._heldout_sources:
            return _source_hash(source)
        value = IntegerIdentityRegistry(self.canonical.backend).find(
            IDENTITY_SOURCE_RECORD, source.stable_key())
        if value is None:
            raise HeldoutResponseRuntimeError("canonical source identity is missing")
        return value

    def is_heldout_connector(self, connector_key) -> bool:
        """Return whether a generated connector key belongs to this overlay."""
        if type(connector_key) is not tuple or any(type(item) is not int for item in connector_key):
            return False
        return any(item.template.connector.stable_key() == connector_key
                   for item in self._heldout)

    def __getattr__(self, name):
        return getattr(self.canonical, name)


__all__ = ["HeldoutResponseRuntime", "HeldoutResponseRuntimeError"]
