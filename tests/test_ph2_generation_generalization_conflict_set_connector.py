from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_plan import GenerationPlanningRequest
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.generation_surface import GenerationSurfaceProtocol
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector import (
    ConflictSetConnectorCompileRequest,
    ConflictSetConnectorCompileError,
    ConflictSetSentenceBinding,
    compile_conflict_set_connector,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetEvidence,
    build_conflict_set_plan,
)

from tests.test_g02_generation_structure_plan import _request, _selection


def _protocol(seed: int) -> GenerationSurfaceProtocol:
    return GenerationSurfaceProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 10)
    ))


def _plan():
    return build_conflict_set_plan(
        scope_id=41,
        claim_ids=("claim-a", "claim-b"),
        evidence=(
            ConflictSetEvidence("e1", "claim-a", "source-1", 41, 1, 0),
            ConflictSetEvidence("e2", "claim-a", "source-2", 41, 0, 1),
            ConflictSetEvidence("e3", "claim-b", "source-3", 41, 1, 0),
            ConflictSetEvidence("e4", "claim-b", "source-4", 41, 0, 1),
        ),
    )


def test_conflict_set_connector_compiles_ordered_multisentence_templates():
    base, _unused = _request(count=2)
    branch = language_branch_identity((91001, 1))
    request = GenerationPlanningRequest(
        replace(base.goal, target_branch=branch),
        base.candidates,
    )
    plan = _plan()
    compilation = compile_conflict_set_connector(
        ConflictSetConnectorCompileRequest(
            plan,
            tuple(
                ConflictSetSentenceBinding(claim_id, candidate.proposition, text)
                for claim_id, candidate, text in zip(
                    plan.claim_ids,
                    request.candidates,
                    ("第一命题。", "第二命题。"),
                    strict=True,
                )
            ),
            branch,
            _protocol(91002),
            (91004,),
            request.candidates[0].source,
            (91003,),
        )
    )
    assert tuple(item.claim_id for item in compilation.sentences) == plan.claim_ids
    assert len(compilation.connector.registry.templates) == 2
    assert compilation.connector.discourse_declarations is not None
    assert all(
        len(template.slots) == 2
        and len(template.bindings) == 2
        and len(template.surface) == 2
        for template in compilation.connector.registry.templates
    )
    assert all(
        template.surface[0].action == compilation.connector.surface_protocol.silent_action
        and template.surface[1].action == compilation.connector.surface_protocol.emit_action
        for template in compilation.connector.registry.templates
    )
    generated = compilation.generate_surfaces(("A", "B"))
    assert tuple(item.claim_id for item in generated) == plan.claim_ids
    assert tuple(item.source_ids for item in generated) == (
        ("source-1", "source-2"),
        ("source-3", "source-4"),
    )
    assert len(compilation.alias_pairs) == 2
    assert all(pair[1].object_kind == OBJECT_REPRESENTATION
               for pair in compilation.alias_pairs)
    selection, _selector, _content_protocol = _selection(request)
    structure = compilation.connector.structure_planner().plan(selection)
    assert tuple(
        sentence.instance.candidate_key
        for sentence in structure.syntax.sentences
    ) == tuple(candidate.stable_key() for candidate in request.candidates)
    assert tuple(
        sentence.values[1].filler
        for sentence in structure.syntax.sentences
    ) == tuple(
        item.template.bindings[1].constant for item in compilation.sentences
    )


def test_conflict_set_connector_rejects_claim_order_or_binding_drift():
    base, _unused = _request(count=2)
    branch = language_branch_identity((91011, 1))
    plan = _plan()
    with pytest.raises(ConflictSetConnectorCompileError):
        compile_conflict_set_connector(
            ConflictSetConnectorCompileRequest(
                plan,
                tuple(
                    ConflictSetSentenceBinding(claim_id, candidate.proposition, "x")
                    for claim_id, candidate in zip(
                        ("claim-b", "claim-a"), base.candidates, strict=True)
                ),
                branch,
                _protocol(91012),
                (91014,),
                base.candidates[0].source,
                (91013,),
            )
        )
