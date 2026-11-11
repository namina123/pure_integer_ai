"""Role-level transfer from learned graph connectors, with no result sentence."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import EPISTEMIC_SUPPORTED
from pure_integer_ai.cognition.shared.relation_composition_verifier import VerifiedRelationComposition
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition, BoundRoleBinding
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, RelationSurfaceBinding,
)


def composition_generation_input(generator, verified, core):
    """Use a same-relation train connector and exact premise filler realizations."""
    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        _generation_protocols,
    )
    if (not isinstance(verified, VerifiedRelationComposition)
            or verified.active.snapshot.epistemic_status != EPISTEMIC_SUPPORTED):
        raise ValueError("derived generation requires independent active H-05 evidence")
    pending, claim = verified.pending, verified.claim
    premise_ids = {item.proposition for item in pending.candidate.premises}
    donor_facts = tuple(fact for fact in core.active_surface_facts()
                        if fact.predicate == pending.candidate.relation
                        and fact.proposition in premise_ids)
    # Same learned structure must agree across donors; never select an answer
    # from the nearest surface or from an arbitrary competing realization.
    options = {}
    for fact in donor_facts:
        template = generator.generation_template_for(fact)
        options[template.stable_key()] = template
    if len(options) != 1:
        raise ValueError("derived relation has no unique learned role connector")
    template = next(iter(options.values()))
    protocols = _generation_protocols(generator.context, template.language_branch)
    result = {item.role: item for item in pending.candidate.bindings}
    frames = tuple(frame for frame in core.active_surface_frames()
                   if frame.proposition in {fact.proposition for fact in donor_facts})
    role_orders = {frame.roles for frame in frames}
    if len(role_orders) != 1 or set(next(iter(role_orders))) != set(result):
        raise ValueError("derived result roles differ from the trained frame")
    source = pending.witness.source
    donor = core.generation_input(donor_facts[0].proposition).proposition.definition
    bound = BoundProposition(claim, protocols.content.answer, template.predicate,
        template.proposition_structure, donor.source_anchor, donor.context, (),
        tuple(BoundRoleBinding(item.role, item.filler, item.ordinal)
              for item in pending.candidate.bindings), ())
    surfaces = []
    for role in next(iter(role_orders)):
        item = result[role]
        matched = tuple(binding for fact in core.active_surface_facts()
                        if fact.proposition in premise_ids for binding in fact.bindings
                        if binding.filler == item.filler)
        if not matched or len({binding.surface for binding in matched}) != 1:
            raise ValueError("derived filler has no unique actual premise Span realization")
        span = matched[0]
        surfaces.append(RelationSurfaceBinding(role, item.filler, span.surface,
            span.source_hash, span.start, span.end))
    fact = ActiveRelationSurface(claim, pending.candidate.relation, "", tuple(surfaces),
                                 "", source.source_id, 0, 0)
    return bound, template, protocols, fact, document_scope(source)
