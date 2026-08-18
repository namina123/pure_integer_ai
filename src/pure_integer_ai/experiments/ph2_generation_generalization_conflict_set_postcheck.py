"""从真实 G-03 ``GenerationSurfacePlan`` 恢复 ``CONFLICT_SET`` projection。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationSentenceInstance,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePlan,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector import (
    ConflictSetConnectorCompilation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    CONFLICT_SET_FAIL,
    CONFLICT_SET_NE,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_surface import (
    ConflictSetGeneratedSentence,
    ConflictSetSurfaceParseResult,
    classify_conflict_set_surface,
)


def _indeterminate(
        compilation: ConflictSetConnectorCompilation,
        sentence_count: int,
        ) -> ConflictSetSurfaceParseResult:
    """实际句实例、slot 或 Representation 无法恢复时返回 capability NE。"""
    del compilation
    return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, sentence_count)


def parse_conflict_set_generation_plan(
        compilation: ConflictSetConnectorCompilation,
        plan: GenerationSurfacePlan | None,
        ) -> ConflictSetSurfaceParseResult:
    """只从实际句实例、claim filler 和 Representation units 进行 G-04 postcheck。"""
    if not isinstance(compilation, ConflictSetConnectorCompilation):
        raise TypeError("conflict postcheck compilation 类型错误")
    if plan is None:
        return _indeterminate(compilation, 0)
    if not isinstance(plan, GenerationSurfacePlan):
        raise TypeError("conflict postcheck plan 类型错误")
    if not plan.preview.complete:
        return _indeterminate(compilation, len(plan.preview.slots))
    syntax = plan.preview.request.structure.syntax
    candidates = {
        item.stable_key(): item
        for item in plan.preview.request.structure.selection.request.candidates
    }
    if plan.preview.request.branch != compilation.language_branch:
        return _indeterminate(compilation, len(syntax.sentences))
    if len(syntax.sentences) != len(compilation.sentences):
        return _indeterminate(compilation, len(syntax.sentences))
    preview_by_address = {
        (item.directive.sentence, item.value.slot): item
        for item in plan.preview.slots
    }
    if len(preview_by_address) != len(plan.preview.slots):
        return _indeterminate(compilation, len(syntax.sentences))
    template_map = {
        item.template.sentence: item for item in compilation.sentences}
    if len(template_map) != len(compilation.sentences):
        return _indeterminate(compilation, len(syntax.sentences))
    actual = []
    used_templates = set()
    candidate_identity_drift = False
    for ordinal, sentence in enumerate(syntax.sentences, start=1):
        instance = sentence.instance
        if not isinstance(instance, GenerationSentenceInstance):
            return _indeterminate(compilation, ordinal)
        compiled = template_map.get(instance.template)
        if compiled is None or instance.template in used_templates:
            return _indeterminate(compilation, ordinal)
        used_templates.add(instance.template)
        candidate = candidates.get(instance.candidate_key)
        if candidate is None:
            return _indeterminate(compilation, ordinal)
        if instance.source != candidate.source or instance.scope != candidate.scope:
            return _indeterminate(compilation, ordinal)
        if candidate.stable_key() != compiled.candidate.stable_key():
            candidate_identity_drift = True
        values = {item.slot: item.filler for item in sentence.values}
        if (values.get(compiled.proposition_slot)
                != candidate.proposition.template
                or values.get(compiled.claim_slot) != compiled.claim_filler):
            return _indeterminate(compilation, ordinal)
        claim_preview = preview_by_address.get((instance, compiled.claim_slot))
        if claim_preview is None or claim_preview.representation is None:
            return _indeterminate(compilation, ordinal)
        try:
            _family, units = representation_parts(claim_preview.representation)
            validate_unicode_scalars(units)
            surface = "".join(chr(unit) for unit in units)
        except (TypeError, ValueError, OverflowError):
            return _indeterminate(compilation, ordinal)
        source_ids_by_ref = {
            item.source: item.source_id for item in compiled.source_bindings}
        if any(source not in source_ids_by_ref
               for source in candidate.citation_sources):
            return _indeterminate(compilation, ordinal)
        source_ids = tuple(sorted(
            source_ids_by_ref[source] for source in candidate.citation_sources
        ))
        actual.append(ConflictSetGeneratedSentence(
            ordinal,
            compiled.claim_id,
            compilation.plan.scope_id,
            source_ids,
            int(candidate.state.support),
            int(candidate.state.refute),
            surface,
            units,
        ))
    if used_templates != set(template_map):
        return _indeterminate(compilation, len(actual))
    parsed = classify_conflict_set_surface(
        compilation.plan,
        tuple(actual),
    )
    if candidate_identity_drift and parsed.projection is not None:
        return ConflictSetSurfaceParseResult(
            CONFLICT_SET_FAIL,
            parsed.projection,
            parsed.sentence_count,
        )
    return parsed


__all__ = ["parse_conflict_set_generation_plan"]
