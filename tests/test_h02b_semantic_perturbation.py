"""H-02B 角色、cue、否定、量化和作用域 typed 扰动对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.language_signal import (
    LanguageSignalInstructionResolution,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicOperatorDefinition,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.logic_perturbation import (
    LogicScopeLayer,
    NegationPerturbationAdapter,
    QuantifierPerturbationAdapter,
    ScopeFlipPerturbationAdapter,
)
from pure_integer_ai.cognition.shared.perturbation import (
    ASSESSMENT_EQUIVALENT,
    PerturbationAssessment,
    PerturbationEngine,
    PerturbationProtocol,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    binder_identity,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.cognition.understanding.semantic_perturbation import (
    CueMisalignmentPerturbationAdapter,
    ResolvedCuePlacement,
    RoleSwapPerturbationAdapter,
    SemanticRoleSlot,
)


def _source(source_id: int = 1) -> SourceRef:
    """构造共享 owner/version 且来源身份互异的测试 SourceRef。"""
    return SourceRef(
        19001,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _definition(
        source: SourceRef,
        key: int,
        bindings: tuple[AtomicRoleBinding, ...],
        *,
        predicate: ObjectIdentity | None = None,
        ) -> AtomicPropositionDefinition:
    """构造共享 predicate/anchor/context 的可竞争 S-02 命题定义。"""
    return AtomicPropositionDefinition(
        proposition_identity(source, (19010, key)),
        predicate or concept_identity((19011, 1)),
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (19012, 1)),
        bindings,
    )


class _UnusedHandler:
    """只满足 R-08 typed operator 协议，H-02B 构造不得执行 handler。"""

    def apply(self, executor, definition, proposition, context):
        """若扰动适配器错误执行逻辑 handler，则立即暴露边界泄漏。"""
        raise AssertionError("H-02B trace 构造不得执行 operator handler")


def _operator_spec(
        source: SourceRef,
        key: int,
        role: ObjectIdentity,
        ) -> LogicOperatorCandidateSpec:
    """构造无固定算子意义、由调用方注入结构/指令/子槽的 R-08 候选。"""
    return LogicOperatorCandidateSpec(
        proposition_identity(source, (19020, key)),
        LogicOperatorDefinition(
            structure_concept_identity((19021, key)),
            minimal_instruction_identity((19022, key)),
            (OperatorSlot(role),),
            _UnusedHandler(),
        ),
        (19023, key),
        (source,),
    )


def _leaf(source: SourceRef, key: int) -> BoundProposition:
    """构造不带逻辑 operator 的共同叶 BoundProposition。"""
    return BoundProposition(
        proposition_identity(source, (19030, key)),
        minimal_instruction_identity((19031, key)),
        concept_identity((19032, key)),
        structure_concept_identity((19033, key)),
        occurrence_identity(
            source, start=key, end=key + 1, ordinal=0),
        context_scope_identity(source, (19034, key)),
        (),
        (),
        (),
    )


def _wrap(
        source: SourceRef,
        key: int,
        child: BoundProposition,
        spec: LogicOperatorCandidateSpec,
        *,
        binders: tuple[ObjectIdentity, ...] = (),
        ) -> BoundProposition:
    """按候选定义的唯一注入槽包裹子命题，形成可审计嵌套层。"""
    slot = spec.definition.slots[0]
    return BoundProposition(
        proposition_identity(source, (19040, key)),
        spec.definition.instruction,
        concept_identity((19041, key)),
        spec.definition.structure,
        occurrence_identity(
            source, start=key, end=key + 1, ordinal=0),
        context_scope_identity(source, (19042, key)),
        binders,
        (BoundRoleBinding(slot.role, child, slot.ordinal),),
        (),
    )


def _perturbation_protocol() -> PerturbationProtocol:
    """注入测试使用的反驳、unknown 和重复诊断键。"""
    return PerturbationProtocol(
        (19050, 1),
        (19050, 2),
        (19050, 3),
    )


def test_role_swap_preserves_full_bindings_and_equivalent_writes_nothing():
    """角色互换只改变注入双槽，且 verifier 判等价时不得写负 Evidence。"""
    source = _source(1)
    scope = document_scope(source)
    first_role = role_identity((19100, 1))
    second_role = role_identity((19100, 2))
    first_filler = entity_identity(source, (19100, 3))
    second_filler = entity_identity(source, (19100, 4))
    original = _definition(source, 1, (
        AtomicRoleBinding(first_role, first_filler),
        AtomicRoleBinding(second_role, second_filler),
    ))
    transformed = _definition(source, 2, (
        AtomicRoleBinding(first_role, second_filler),
        AtomicRoleBinding(second_role, first_filler),
    ))
    adapter = RoleSwapPerturbationAdapter((19100, 5))
    result = adapter.build(
        original,
        transformed,
        swapped_slots=(
            SemanticRoleSlot(first_role),
            SemanticRoleSlot(second_role),
        ),
        scope=scope,
    )

    assert result.trace.source == source
    assert result.trace.scope == scope
    assert result.impact_original == tuple(
        binding.identity_for(original.proposition)
        for binding in original.canonical_bindings()
    )
    assert result.impact_transformed == tuple(
        binding.identity_for(transformed.proposition)
        for binding in transformed.canonical_bindings()
    )
    assert first_filler in result.trace.original
    assert first_filler in result.trace.transformed
    assert result.trace.output_to_input != tuple(
        range(len(result.trace.transformed)))

    hypothesis = HypothesisKey(
        (19100, 6),
        original.proposition.stable_key(),
        (19100, 7),
        scope,
        source,
    )
    ledger = HypothesisLedger()
    ledger.register(hypothesis)
    before = ledger.state_key()
    evidence = PerturbationEngine(
        _perturbation_protocol(), ledger=ledger).evaluate(
            result.trace,
            candidates=(hypothesis,),
            verifier=lambda _trace, _candidates: PerturbationAssessment(
                ASSESSMENT_EQUIVALENT,
                (19100, 8),
            ),
            evidence_source=source,
            timestamp_seq=1,
        )
    assert evidence.evidence_ids == ()
    assert ledger.state_key() == before

    changed_predicate = replace(
        transformed, predicate=concept_identity((19100, 9)))
    with pytest.raises(ValueError, match="predicate"):
        adapter.build(
            original,
            changed_predicate,
            swapped_slots=(
                SemanticRoleSlot(first_role),
                SemanticRoleSlot(second_role),
            ),
            scope=scope,
        )
    with pytest.raises(ValueError, match="一等 Role"):
        SemanticRoleSlot(concept_identity((19100, 10)))


def test_cue_misalignment_uses_unique_u04_instruction_and_target_permutation():
    """cue 错位保留 U-04 唯一指令，只置换来源化语义目标。"""
    source = _source(2)
    scope = document_scope(source)
    first_cue = occurrence_identity(
        source, start=0, end=1, ordinal=0)
    second_cue = occurrence_identity(
        source, start=1, end=2, ordinal=0)
    first_instruction = minimal_instruction_identity((19200, 1))
    second_instruction = minimal_instruction_identity((19200, 2))
    first_target = proposition_identity(source, (19200, 3))
    second_target = proposition_identity(source, (19200, 4))

    def placement(cue, instruction, target):
        """从 U-04 唯一解析构造一个不含 surface 的 cue placement。"""
        return ResolvedCuePlacement(
            cue,
            instruction,
            target,
            LanguageSignalInstructionResolution(
                True, instruction.components),
        )

    original = (
        placement(first_cue, first_instruction, first_target),
        placement(second_cue, second_instruction, second_target),
    )
    transformed = (
        placement(first_cue, first_instruction, second_target),
        placement(second_cue, second_instruction, first_target),
    )
    result = CueMisalignmentPerturbationAdapter((19200, 5)).build(
        original,
        transformed,
        source=source,
        scope=scope,
    )

    assert result.impact_original == (first_target, second_target)
    assert result.impact_transformed == (second_target, first_target)
    assert result.trace.original[1] == first_instruction
    assert result.trace.transformed[1] == first_instruction
    assert result.trace.output_to_input[2] == 5
    assert result.trace.output_to_input[5] == 2

    with pytest.raises(ValueError, match="唯一且无冲突"):
        ResolvedCuePlacement(
            first_cue,
            first_instruction,
            first_target,
            LanguageSignalInstructionResolution(True, None),
        )
    with pytest.raises(ValueError, match="只能置换"):
        CueMisalignmentPerturbationAdapter((19200, 5)).build(
            original,
            (
                transformed[0],
                placement(
                    second_cue,
                    second_instruction,
                    proposition_identity(source, (19200, 99)),
                ),
            ),
            source=source,
            scope=scope,
        )


def test_negation_adapter_requires_r08_candidate_and_exact_wrapped_child():
    """否定维度必须由注入 R-08 candidate 精确包裹原命题，不能只看结构名。"""
    source = _source(3)
    scope = document_scope(source)
    role = role_identity((19300, 1))
    spec = _operator_spec(source, 1, role)
    layer = LogicScopeLayer(spec, spec.definition.slots[0])
    original = _leaf(source, 1)
    transformed = _wrap(source, 2, original, spec)
    result = NegationPerturbationAdapter((19300, 2)).build(
        original,
        transformed,
        layer=layer,
        source=source,
        scope=scope,
    )

    assert result.impact_original == (original.template,)
    assert result.impact_transformed == (
        spec.candidate,
        transformed.template,
    )
    assert spec.candidate in result.trace.transformed
    assert original.template in result.trace.transformed

    wrong_child = _leaf(source, 3)
    with pytest.raises(ValueError, match="完整 original"):
        NegationPerturbationAdapter((19300, 2)).build(
            original,
            _wrap(source, 4, wrong_child, spec),
            layer=layer,
            source=source,
            scope=scope,
        )


def test_quantifier_adapter_changes_operator_but_keeps_binder_and_body():
    """量化变换必须保留 Binder/body，并显式替换两个不同 R-08 operator。"""
    source = _source(4)
    scope = document_scope(source)
    role = role_identity((19400, 1))
    first_spec = _operator_spec(source, 2, role)
    second_spec = _operator_spec(source, 3, role)
    first_layer = LogicScopeLayer(
        first_spec, first_spec.definition.slots[0])
    second_layer = LogicScopeLayer(
        second_spec, second_spec.definition.slots[0])
    body = _leaf(source, 5)
    binder = binder_identity(source, (19400, 2))
    original = _wrap(
        source, 6, body, first_spec, binders=(binder,))
    transformed = _wrap(
        source, 7, body, second_spec, binders=(binder,))
    result = QuantifierPerturbationAdapter((19400, 3)).build(
        original,
        transformed,
        original_layer=first_layer,
        transformed_layer=second_layer,
        source=source,
        scope=scope,
    )

    assert result.impact_original == (
        first_spec.candidate,
        original.template,
    )
    assert result.impact_transformed == (
        second_spec.candidate,
        transformed.template,
    )
    assert binder in result.trace.original
    assert binder in result.trace.transformed
    assert body.template in result.trace.original
    assert body.template in result.trace.transformed

    other_body = _leaf(source, 8)
    with pytest.raises(ValueError, match="完整 body"):
        QuantifierPerturbationAdapter((19400, 3)).build(
            original,
            _wrap(
                source, 9, other_body, second_spec,
                binders=(binder,),
            ),
            original_layer=first_layer,
            transformed_layer=second_layer,
            source=source,
            scope=scope,
        )


def test_scope_flip_reorders_same_candidates_and_changes_semantic_root():
    """作用域翻转只重排同一 operator 集，保留共同叶并改变完整语义根。"""
    source = _source(5)
    scope = document_scope(source)
    first_role = role_identity((19500, 1))
    second_role = role_identity((19500, 2))
    first_spec = _operator_spec(source, 4, first_role)
    second_spec = _operator_spec(source, 5, second_role)
    first_layer = LogicScopeLayer(
        first_spec, first_spec.definition.slots[0])
    second_layer = LogicScopeLayer(
        second_spec, second_spec.definition.slots[0])
    leaf = _leaf(source, 10)
    original_inner = _wrap(source, 11, leaf, second_spec)
    original = _wrap(source, 12, original_inner, first_spec)
    transformed_inner = _wrap(source, 13, leaf, first_spec)
    transformed = _wrap(source, 14, transformed_inner, second_spec)
    result = ScopeFlipPerturbationAdapter((19500, 3)).build(
        original,
        transformed,
        original_path=(first_layer, second_layer),
        transformed_path=(second_layer, first_layer),
        source=source,
        scope=scope,
    )

    assert result.impact_original == (
        original.template,
        first_spec.candidate,
        second_spec.candidate,
    )
    assert result.impact_transformed == (
        transformed.template,
        second_spec.candidate,
        first_spec.candidate,
    )
    assert leaf.template in result.trace.original
    assert leaf.template in result.trace.transformed
    assert original.template != transformed.template
    assert result.stable_key() == ScopeFlipPerturbationAdapter(
        (19500, 3)).build(
            original,
            transformed,
            original_path=(first_layer, second_layer),
            transformed_path=(second_layer, first_layer),
            source=source,
            scope=scope,
        ).stable_key()

    with pytest.raises(ValueError, match="嵌套层序"):
        ScopeFlipPerturbationAdapter((19500, 3)).build(
            original,
            original,
            original_path=(first_layer, second_layer),
            transformed_path=(first_layer, second_layer),
            source=source,
            scope=scope,
        )
