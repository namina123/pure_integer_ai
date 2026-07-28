"""R-05 五类专属证明合同、dispatcher、预算和反例测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_DIRECTION_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    ConditionOperator,
    LogicEvidenceState,
    LogicOperatorDefinition,
    ModalOperator,
    ModalResolution,
    NegationOperator,
    OperatorSlot,
    STATE_CONFLICTED,
    STATE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.modal_primitives import (
    MODAL_KIND_BOX_NECESSITY,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicRoleBinding,
    context_scope_identity,
    event_identity,
    role_identity,
    set_expr_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import BindingEnvironment
from pure_integer_ai.experiments.causal_relation_runtime import (
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.experiments.ph2_typed_proof_family_catalog import (
    MANIFEST_PATH,
    build_typed_proof_family_manifest,
)
from pure_integer_ai.experiments.ph2_typed_proof_family_contract import (
    TypedProofFamilyContractError,
    TypedProofFamilyEvidenceFile,
    read_typed_proof_family_manifest,
    verify_typed_proof_family_files,
    write_typed_proof_family_manifest,
)
from pure_integer_ai.experiments.typed_proof_family_contracts import (
    CAUSAL_DIRECTION_INHIBITING,
    CAUSAL_DIRECTION_PROMOTING,
    CONDITION_AFFIRMING_CONSEQUENT,
    CONDITION_ASSERTION,
    CONDITION_MATERIAL,
    CONDITION_NECESSARY,
    CONDITION_SUFFICIENT,
    MODAL_FRAME_EPISTEMIC,
    MODAL_FRAME_NORMATIVE,
    PROOF_ACCEPTED,
    PROOF_BUDGET_EXHAUSTED,
    PROOF_CONFLICTED,
    PROOF_FAIL_CLOSED,
    PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
    PROOF_FAMILY_CONDITION,
    PROOF_FAMILY_MISMATCH,
    PROOF_FAMILY_MODAL,
    PROOF_FAMILY_NOT,
    PROOF_FAMILY_TEMPORAL,
    PROOF_REJECTED,
    PROOF_UNKNOWN,
    CausalCounterfactualProofCertificate,
    ConditionProofCertificate,
    CounterfactualPair,
    CounterfactualState,
    ModalProofCertificate,
    NotProofCertificate,
    ProofWorkBudget,
    TemporalProofCertificate,
)
from pure_integer_ai.experiments.typed_proof_family_runtime import (
    TRACE_AFFIRMING_CONSEQUENT,
    TRACE_BUDGET_EXHAUSTED,
    TRACE_CAUSAL_WITNESS_NOT_INDEPENDENT,
    TRACE_CONDITION_DIRECTION_MISMATCH,
    TRACE_FAMILY_MISMATCH,
    TRACE_MODAL_CONTEXT_INCOMPLETE,
    TRACE_MODAL_FRAME_MISMATCH,
    TRACE_MODAL_RESOLVER_MISSING,
    TRACE_NOT_STRUCTURE_MISMATCH,
    TRACE_TEMPORAL_CONFLICT,
    TRACE_TEMPORAL_DIRECTION_MISSING,
    TRACE_TEMPORAL_UNKNOWN,
    TypedProofFamilyDispatcher,
)
from pure_integer_ai.experiments.verification_orchestration import (
    MultiVerifierOrchestrator,
)
from pure_integer_ai.storage.backend import DictBackend
from tests.test_r06_event_time import (
    _record as _record_event_time,
    _runtime as _event_time_runtime,
    _source as _event_time_source,
)
from tests.test_r07_causal_relation_runtime import (
    _endpoint_evaluations,
    _fixture as _causal_fixture,
    _form as _form_causal,
    _record_time as _record_causal_time,
    _request as _causal_request,
)
from tests.test_s04_logic_executor import (
    _InjectedAtomResolver,
    _binding_failures,
    _bound,
    _definition,
    _executor,
    _logic_failures,
    _source,
    _template,
)


_BASE = 98100
_ROOT = Path(__file__).resolve().parents[1]
_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)
_U = LogicEvidenceState(False, False)
_B = LogicEvidenceState(True, True)


def _dispatch(certificate, *, limit=1000):
    """用新预算执行一次严格 family dispatch。"""
    return TypedProofFamilyDispatcher().check(
        certificate,
        ProofWorkBudget(limit),
    )


def _temporal_certificate(*, direction=EVENT_TIME_BEFORE):
    """构造一个真实 EventTimeVerifier 一致方向结果和释放函数。"""
    backend = DictBackend()
    relation = concept_identity((_BASE + 1, 2))
    _, facts, verifier = _event_time_runtime(
        backend,
        {relation: EVENT_TIME_BEFORE},
    )
    source = _event_time_source(801)
    scope = document_scope(source)
    first = event_identity(source, (_BASE + 2, 1))
    second = event_identity(source, (_BASE + 2, 2))
    fact = _record_event_time(facts, relation, first, second, scope)
    result = verifier.verify((relation,), scope=scope)
    return (
        TemporalProofCertificate(
            PROOF_FAMILY_TEMPORAL,
            result,
            relation,
            first,
            second,
            direction,
            (fact.assertion_hash,),
            source,
            scope,
        ),
        backend,
    )


def _logic_binary(*, necessary=False):
    """经真实 LogicExecutor 构造有序 CONDITION 根和两个原子求值。"""
    source = _source(802 if not necessary else 803)
    scope = document_scope(source)
    failures = _binding_failures(_BASE + 10)
    logic_failures = _logic_failures(_BASE + 20)
    atom_structure = structure_concept_identity((_BASE + 30, 1))
    condition_structure = structure_concept_identity((_BASE + 30, 2))
    first_role = role_identity((_BASE + 30, 3))
    second_role = role_identity((_BASE + 30, 4))
    condition = _definition(source, 801)
    conditioned = _definition(source, 802)
    ordered = (
        (conditioned, condition)
        if necessary else (condition, conditioned)
    )
    root = _definition(source, 803, (
        AtomicRoleBinding(first_role, ordered[0].proposition),
        AtomicRoleBinding(second_role, ordered[1].proposition),
    ))
    templates = (
        _template(condition, atom_structure),
        _template(conditioned, atom_structure),
        _template(root, condition_structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    states = {
        condition.proposition: (_T, 81),
        conditioned.proposition: (_T, 82),
    }
    resolver = _InjectedAtomResolver(
        lambda proposition, _scope: states.get(proposition.template))
    definition = LogicOperatorDefinition(
        condition_structure,
        minimal_instruction_identity((_BASE + 30, 5)),
        (OperatorSlot(first_role), OperatorSlot(second_role)),
        ConditionOperator(),
    )
    executor = _executor(
        (definition,), resolver, failures, logic_failures, protocol)
    evaluation = executor.evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
    )

    def atom(definition_):
        atom_bound, _, _ = _bound(definition_, templates, failures)
        return executor.evaluate(
            atom_bound,
            source=source,
            scope=scope,
            graph=graph,
            environment=BindingEnvironment(),
        )

    return (
        condition_structure,
        atom(condition),
        atom(conditioned),
        evaluation,
    )


def _logic_unary(handler, child_state, *, modal=False):
    """经真实 LogicExecutor 构造 structural NOT 或 resolver-driven MODAL。"""
    source = _source(804 if not modal else 805)
    scope = document_scope(source)
    failures = _binding_failures(_BASE + 40)
    logic_failures = _logic_failures(_BASE + 50)
    atom_structure = structure_concept_identity((_BASE + 60, 1))
    operator_structure = structure_concept_identity(
        (_BASE + 60, 3 if modal else 2))
    child_role = role_identity((_BASE + 60, 4))
    child = _definition(source, 804)
    root = _definition(source, 805, (
        AtomicRoleBinding(child_role, child.proposition),))
    templates = (
        _template(child, atom_structure),
        _template(root, operator_structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    resolver = _InjectedAtomResolver(
        lambda proposition, _scope: (child_state, 83)
        if proposition.template == child.proposition else None)
    definition = LogicOperatorDefinition(
        operator_structure,
        minimal_instruction_identity((_BASE + 60, 5)),
        (OperatorSlot(child_role),),
        handler,
    )
    executor = _executor(
        (definition,), resolver, failures, logic_failures, protocol)
    child_bound, _, _ = _bound(child, templates, failures)
    child_evaluation = executor.evaluate(
        child_bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
    )
    modal_scope = query_scope(806, parent=scope)

    class _ModalResolver:
        """为 modal 测试注入独立 Evidence 和显式新 scope。"""

        def resolve(self, operator, child_result, context):
            """返回来源不变、Evidence 独立的 provisional modal 结果。"""
            assert operator.structure == operator_structure
            assert child_result.source == source
            assert context.scope == scope
            return ModalResolution(_T, source, modal_scope, (84,), ())

    evaluation = executor.evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
        modal_resolver=_ModalResolver() if modal else None,
    )
    return operator_structure, child_evaluation, evaluation, source


def test_temporal_checker_binds_typed_endpoints_direction_and_family():
    """同一 verifier 结果只证明声明方向，反向和跨 family 均不得通过。"""
    certificate, backend = _temporal_certificate()
    try:
        accepted = _dispatch(certificate)
        reversed_result = _dispatch(replace(
            certificate, direction=EVENT_TIME_AFTER))
        mismatched = _dispatch(replace(
            certificate, declared_family=PROOF_FAMILY_NOT))

        assert accepted.result.status == PROOF_ACCEPTED
        assert reversed_result.result.status == PROOF_REJECTED
        assert reversed_result.result.trace[-2] == (
            TRACE_TEMPORAL_DIRECTION_MISSING)
        assert mismatched.result.status == PROOF_FAMILY_MISMATCH
        assert mismatched.result.trace[-2] == TRACE_FAMILY_MISMATCH
    finally:
        backend.close()


def test_temporal_unknown_and_conflict_remain_distinct():
    """未知 relation 不闭世界化，方向环也不被挑成单一答案。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 70, 1))
        unknown = concept_identity((_BASE + 70, 2))
        _, facts, verifier = _event_time_runtime(
            backend,
            {
                before: EVENT_TIME_BEFORE,
                unknown: EVENT_TIME_DIRECTION_UNKNOWN,
            },
        )
        source = _event_time_source(807)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 71, 1))
        second = event_identity(source, (_BASE + 71, 2))
        unknown_fact = _record_event_time(
            facts, unknown, first, second, scope)
        unknown_result = verifier.verify((unknown,), scope=scope)
        unknown_certificate = TemporalProofCertificate(
            PROOF_FAMILY_TEMPORAL,
            unknown_result,
            unknown,
            first,
            second,
            EVENT_TIME_BEFORE,
            (unknown_fact.assertion_hash,),
            source,
            scope,
        )
        assert _dispatch(unknown_certificate).result.status == PROOF_UNKNOWN
        assert _dispatch(unknown_certificate).result.trace[-2] == (
            TRACE_TEMPORAL_UNKNOWN)

        first_fact = _record_event_time(
            facts, before, first, second, scope)
        _record_event_time(facts, before, second, first, scope)
        conflict_result = verifier.verify((before,), scope=scope)
        conflict_certificate = TemporalProofCertificate(
            PROOF_FAMILY_TEMPORAL,
            conflict_result,
            before,
            first,
            second,
            EVENT_TIME_BEFORE,
            (first_fact.assertion_hash,),
            source,
            scope,
        )
        checked = _dispatch(conflict_certificate)
        assert conflict_result.status == EVENT_TIME_CONFLICTED
        assert checked.result.status == PROOF_CONFLICTED
        assert checked.result.trace[-2] == TRACE_TEMPORAL_CONFLICT
    finally:
        backend.close()


def test_causal_counterfactual_requires_independent_witness_and_pair_direction():
    """promoting 必须同时有 active execution、独立 witness 和显式状态转移。"""
    fixture = _causal_fixture()
    try:
        _form_causal(fixture)
        _record_causal_time(fixture, fixture.before)
        request = _causal_request(fixture, 808, stance=EVIDENCE_SUPPORT)
        MultiVerifierOrchestrator().run(
            request,
            (fixture.runtime.registration(),),
            read_only=False,
        )
        cause, effect = _endpoint_evaluations(fixture)
        execution = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            cause,
            effect,
            use_key=(_BASE + 80, 1),
        ).execution
        witness_scope = document_scope(request.witness.verifier_source)
        baseline = CounterfactualState(
            fixture.cause,
            fixture.effect,
            _F,
            _F,
            request.witness.verifier_source,
            query_scope(1, parent=witness_scope),
            (91, 92),
        )
        intervention = CounterfactualState(
            fixture.cause,
            fixture.effect,
            _T,
            _T,
            request.witness.verifier_source,
            query_scope(2, parent=witness_scope),
            (93, 94),
        )
        certificate = CausalCounterfactualProofCertificate(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            execution,
            request.witness,
            CAUSAL_DIRECTION_PROMOTING,
            CounterfactualPair(baseline, intervention),
        )

        assert _dispatch(certificate).result.status == PROOF_ACCEPTED
        assert _dispatch(replace(
            certificate,
            direction=CAUSAL_DIRECTION_INHIBITING,
        )).result.status == PROOF_REJECTED

        same_source_witness = replace(
            request.witness,
            verifier_source=execution.cause.evaluation.source,
        )
        rejected = _dispatch(replace(
            certificate,
            witness=same_source_witness,
        ))
        assert rejected.result.status == PROOF_FAIL_CLOSED
        assert rejected.result.trace[-2] == (
            TRACE_CAUSAL_WITNESS_NOT_INDEPENDENT)
    finally:
        fixture.close()


def test_causal_inhibiting_uses_explicit_counterfactual_not_prediction_relabel():
    """inhibiting 只接受 effect 由支持转反驳且执行未伪造正向 prediction。"""
    fixture = _causal_fixture()
    try:
        _form_causal(fixture)
        _record_causal_time(fixture, fixture.before)
        request = _causal_request(fixture, 809, stance=EVIDENCE_SUPPORT)
        MultiVerifierOrchestrator().run(
            request,
            (fixture.runtime.registration(),),
            read_only=False,
        )
        cause, effect = _endpoint_evaluations(
            fixture,
            cause=(False, False),
        )
        execution = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            cause,
            effect,
            use_key=(_BASE + 81, 1),
        ).execution
        witness_scope = document_scope(request.witness.verifier_source)
        pair = CounterfactualPair(
            CounterfactualState(
                fixture.cause,
                fixture.effect,
                _F,
                _T,
                request.witness.verifier_source,
                query_scope(3, parent=witness_scope),
                (95, 96),
            ),
            CounterfactualState(
                fixture.cause,
                fixture.effect,
                _T,
                _F,
                request.witness.verifier_source,
                query_scope(4, parent=witness_scope),
                (97, 98),
            ),
        )
        certificate = CausalCounterfactualProofCertificate(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            execution,
            request.witness,
            CAUSAL_DIRECTION_INHIBITING,
            pair,
        )

        assert _dispatch(certificate).result.status == PROOF_ACCEPTED
        assert _dispatch(replace(
            certificate,
            direction=CAUSAL_DIRECTION_PROMOTING,
        )).result.status == PROOF_REJECTED
    finally:
        fixture.close()


def test_condition_distinguishes_kinds_direction_and_affirming_consequent():
    """material、充分和必要保持互异 Evidence/方向，肯定后件稳定拒绝。"""
    structure, condition, conditioned, evaluation = _logic_binary()
    material = ConditionProofCertificate(
        PROOF_FAMILY_CONDITION,
        CONDITION_MATERIAL,
        structure,
        condition,
        conditioned,
        evaluation,
        CONDITION_ASSERTION,
    )
    sufficient = replace(
        material,
        condition_kind=CONDITION_SUFFICIENT,
        kind_evidence_ids=(101,),
    )
    affirming = replace(
        material,
        inference_kind=CONDITION_AFFIRMING_CONSEQUENT,
    )

    assert _dispatch(material).result.status == PROOF_ACCEPTED
    assert _dispatch(sufficient).result.status == PROOF_ACCEPTED
    assert _dispatch(replace(
        sufficient,
        kind_evidence_ids=(),
    )).result.status == PROOF_UNKNOWN
    rejected = _dispatch(affirming)
    assert rejected.result.status == PROOF_REJECTED
    assert rejected.result.trace[-2] == TRACE_AFFIRMING_CONSEQUENT

    necessary_structure, necessary_condition, necessary_conditioned, root = (
        _logic_binary(necessary=True))
    necessary = ConditionProofCertificate(
        PROOF_FAMILY_CONDITION,
        CONDITION_NECESSARY,
        necessary_structure,
        necessary_condition,
        necessary_conditioned,
        root,
        CONDITION_ASSERTION,
        (102,),
    )
    assert _dispatch(necessary).result.status == PROOF_ACCEPTED
    wrong_direction = _dispatch(replace(
        necessary,
        condition_kind=CONDITION_SUFFICIENT,
    ))
    assert wrong_direction.result.status == PROOF_FAIL_CLOSED
    assert wrong_direction.result.trace[-2] == (
        TRACE_CONDITION_DIRECTION_MISMATCH)


def test_structural_not_preserves_open_world_unknown_and_conflict():
    """NOT 只交换 Evidence 位；未知和冲突不被 closed-world 或反义词压成二值。"""
    for state, expected in ((_U, PROOF_UNKNOWN), (_B, PROOF_CONFLICTED)):
        structure, child, evaluation, _ = _logic_unary(
            NegationOperator(), state)
        certificate = NotProofCertificate(
            PROOF_FAMILY_NOT,
            structure,
            child,
            evaluation,
        )
        checked = _dispatch(certificate)
        assert checked.result.status == expected
        assert evaluation.status in {STATE_UNKNOWN, STATE_CONFLICTED}

    structure, child, evaluation, _ = _logic_unary(
        NegationOperator(), _F)
    wrong_structure = structure_concept_identity((_BASE + 90, 1))
    rejected = _dispatch(NotProofCertificate(
        PROOF_FAMILY_NOT,
        wrong_structure,
        child,
        evaluation,
    ))
    assert rejected.result.status == PROOF_FAIL_CLOSED
    assert rejected.result.trace[-2] == TRACE_NOT_STRUCTURE_MISMATCH


def test_modal_requires_matching_frame_complete_context_and_resolver_evidence():
    """modal kind 不能跨 frame，world/domain 或 resolver 缺失时不通过。"""
    structure, child, evaluation, source = _logic_unary(
        ModalOperator(), _T, modal=True)
    certificate = ModalProofCertificate(
        PROOF_FAMILY_MODAL,
        structure,
        child,
        evaluation,
        MODAL_KIND_BOX_NECESSITY,
        MODAL_FRAME_EPISTEMIC,
        (context_scope_identity(source, (_BASE + 100, 1)),),
        (set_expr_identity(source, (_BASE + 100, 2)),),
        True,
        True,
        (84,),
    )

    assert _dispatch(certificate).result.status == PROOF_ACCEPTED
    frame = _dispatch(replace(
        certificate,
        frame=MODAL_FRAME_NORMATIVE,
    ))
    assert frame.result.status == PROOF_FAIL_CLOSED
    assert frame.result.trace[-2] == TRACE_MODAL_FRAME_MISMATCH
    incomplete = _dispatch(replace(
        certificate,
        domains_complete=False,
    ))
    assert incomplete.result.status == PROOF_UNKNOWN
    assert incomplete.result.trace[-2] == TRACE_MODAL_CONTEXT_INCOMPLETE
    missing = _dispatch(replace(
        certificate,
        resolver_evidence_ids=(),
    ))
    assert missing.result.status == PROOF_FAIL_CLOSED
    assert missing.result.trace[-2] == TRACE_MODAL_RESOLVER_MISSING


def test_dispatcher_hard_budget_is_atomic_and_result_is_deterministic():
    """预算不足不消费部分工作，相同输入/预算产生 bit-identical 稳定结果。"""
    certificate, backend = _temporal_certificate()
    try:
        first = _dispatch(certificate)
        second = _dispatch(certificate)
        assert first.result.stable_key() == second.result.stable_key()
        assert first.budget == second.budget

        exhausted = _dispatch(certificate, limit=1)
        assert exhausted.result.status == PROOF_BUDGET_EXHAUSTED
        assert exhausted.result.work_units == 0
        assert exhausted.result.trace[-2] == TRACE_BUDGET_EXHAUSTED
        assert exhausted.budget == ProofWorkBudget(1)
    finally:
        backend.close()


def test_contracts_make_no_free_text_or_definitive_truth_claim():
    """生产 checker 只消费 typed artifact，源码不含 renderer/certificate 词表映射。"""
    import inspect

    from pure_integer_ai.experiments import typed_proof_family_runtime

    source = inspect.getsource(typed_proof_family_runtime)
    assert "surface_map" not in source
    assert "lexicon" not in source
    assert "definitive_truth" not in source
    assert "teacher" not in source


@pytest.fixture(scope="module")
def formal_manifest():
    """按当前仓库内容构建一次确定性 R-05 正式 manifest。"""
    return build_typed_proof_family_manifest(_ROOT)


def test_manifest_round_trip_file_identity_and_zero_execution(
        tmp_path, formal_manifest):
    """R-05 artifact 必须 canonical 回读并逐字节绑定实现和测试。"""
    target = tmp_path / "r05.json"
    assert write_typed_proof_family_manifest(
        formal_manifest, target) == target
    assert read_typed_proof_family_manifest(target) == formal_manifest
    verify_typed_proof_family_files(formal_manifest, repository_root=_ROOT)
    assert set(formal_manifest.requirement_decisions.to_value().values()) == {
        "PASS"}
    assert formal_manifest.execution_state.to_value()["formal_training_runs"] == 0
    assert formal_manifest.execution_state.to_value()["teacher_calls"] == 0


def test_manifest_rejects_fake_counterexample_coverage(formal_manifest):
    """任一 family 反例被删都必须使固定合同构造失败。"""
    coverage = formal_manifest.counterexample_coverage.to_value()
    coverage["dispatcher_family_mismatch_rejected"] = 0
    from pure_integer_ai.experiments.ph2_dataset_contract import (
        CanonicalJsonObject,
    )
    with pytest.raises(
            TypedProofFamilyContractError,
            match="counterexample coverage 漂移"):
        replace(
            formal_manifest,
            counterexample_coverage=CanonicalJsonObject.from_value(coverage),
        )


def test_manifest_rejects_file_identity_drift(formal_manifest):
    """任何 checker、生产依赖或测试尺寸/hash 漂移都必须回验失败。"""
    first = formal_manifest.evidence_files[0]
    drifted = TypedProofFamilyEvidenceFile(
        first.relative_path,
        first.role,
        first.byte_count + 1,
        first.sha256,
    )
    manifest = replace(
        formal_manifest,
        evidence_files=(drifted, *formal_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            TypedProofFamilyContractError,
            match="evidence 文件身份漂移"):
        verify_typed_proof_family_files(manifest, repository_root=_ROOT)


def test_manifest_is_idempotent_but_non_overwritable(
        tmp_path, formal_manifest):
    """同内容可幂等重放，同版本异内容不可覆盖。"""
    target = tmp_path / "r05.json"
    write_typed_proof_family_manifest(formal_manifest, target)
    assert write_typed_proof_family_manifest(
        formal_manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(
            TypedProofFamilyContractError,
            match="已存在且内容不同"):
        write_typed_proof_family_manifest(formal_manifest, target)


def test_stored_manifest_is_current_readable_and_deterministic(formal_manifest):
    """仓内正式 artifact 必须等于当前 builder，重复构建字节完全一致。"""
    stored = read_typed_proof_family_manifest(_ROOT / MANIFEST_PATH)
    rebuilt = build_typed_proof_family_manifest(_ROOT)
    assert stored == formal_manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_typed_proof_family_files(stored, repository_root=_ROOT)
