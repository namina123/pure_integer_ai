"""P2-G 五类 typed certificate 的互斥只读 checker 和严格 dispatcher。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.causal_execution import (
    CAUSAL_EXECUTION_CONFLICTED,
    CAUSAL_EXECUTION_PREDICTED,
    CAUSAL_TEMPORAL_ACCEPTED,
    CAUSAL_TEMPORAL_CONFLICTED,
    CAUSAL_TEMPORAL_UNKNOWN,
)
from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_EMPTY,
    EVENT_TIME_SAME,
    EVENT_TIME_UNKNOWN,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
    STATE_CONFLICTED,
    STATE_PROVISIONAL,
    STATE_REFUTED,
    STATE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.modal_primitives import (
    MODAL_KIND_BOX_NECESSITY,
    MODAL_KIND_BOX_POSSIBILITY,
    MODAL_KIND_DEONTIC_NECESSITY,
    MODAL_KIND_DEONTIC_POSSIBILITY,
)
from pure_integer_ai.experiments.causal_relation_runtime import (
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.experiments.typed_proof_family_contracts import (
    CAUSAL_DIRECTION_INHIBITING,
    CAUSAL_DIRECTION_PROMOTING,
    CONDITION_AFFIRMING_CONSEQUENT,
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
    ModalProofCertificate,
    NotProofCertificate,
    ProofCheckResult,
    ProofWorkBudget,
    TemporalProofCertificate,
)


TRACE_VERSION = 1
TRACE_DISPATCH = 1
TRACE_FAMILY_MISMATCH = 2
TRACE_BUDGET_EXHAUSTED = 3
TRACE_SOURCE_SCOPE_MISMATCH = 4
TRACE_TEMPORAL_CONFLICT = 5
TRACE_TEMPORAL_UNKNOWN = 6
TRACE_TEMPORAL_DIRECTION_MISSING = 7
TRACE_TEMPORAL_PREMISE_MISMATCH = 8
TRACE_CAUSAL_TEMPORAL_UNKNOWN = 9
TRACE_CAUSAL_TEMPORAL_CONFLICT = 10
TRACE_CAUSAL_WITNESS_NOT_INDEPENDENT = 11
TRACE_CAUSAL_ENDPOINT_MISMATCH = 12
TRACE_COUNTERFACTUAL_PAIR_MISMATCH = 13
TRACE_CAUSAL_EXECUTION_MISMATCH = 14
TRACE_CONDITION_DIRECTION_MISMATCH = 15
TRACE_CONDITION_KIND_EVIDENCE_MISSING = 16
TRACE_AFFIRMING_CONSEQUENT = 17
TRACE_NOT_STRUCTURE_MISMATCH = 18
TRACE_MODAL_FRAME_MISMATCH = 19
TRACE_MODAL_CONTEXT_INCOMPLETE = 20
TRACE_MODAL_RESOLVER_MISSING = 21
TRACE_LOGIC_PROVISIONAL = 22
TRACE_LOGIC_REFUTED = 23
TRACE_LOGIC_UNKNOWN = 24
TRACE_LOGIC_CONFLICTED = 25
TRACE_UNREGISTERED_CERTIFICATE = 26


@dataclass(frozen=True)
class ProofDispatchReceipt:
    """一次 dispatcher 调用的 checker 结果和不可超支的新预算。"""

    result: ProofCheckResult
    budget: ProofWorkBudget

    def __post_init__(self) -> None:
        """核验结果与预算均为正式合同类型。"""
        if not isinstance(self.result, ProofCheckResult):
            raise TypeError("proof dispatch result 类型非法")
        if not isinstance(self.budget, ProofWorkBudget):
            raise TypeError("proof dispatch budget 类型非法")


def _logic_status(evaluation: LogicEvaluation) -> tuple[int, int]:
    """把 S-04 四态映射为 proof 结果，不宣称现实 definitive truth。"""
    if evaluation.status == STATE_PROVISIONAL:
        return PROOF_ACCEPTED, TRACE_LOGIC_PROVISIONAL
    if evaluation.status == STATE_REFUTED:
        return PROOF_REJECTED, TRACE_LOGIC_REFUTED
    if evaluation.status == STATE_UNKNOWN:
        return PROOF_UNKNOWN, TRACE_LOGIC_UNKNOWN
    if evaluation.status == STATE_CONFLICTED:
        return PROOF_CONFLICTED, TRACE_LOGIC_CONFLICTED
    return PROOF_FAIL_CLOSED, TRACE_UNREGISTERED_CERTIFICATE


def _logic_terminal(
        structure, child: LogicEvaluation, evaluation: LogicEvaluation,
        ) -> bool:
    """核对单目 operator 最终步骤的结构、前提、来源和 scope。"""
    if (child.source != evaluation.source
            or child.scope != evaluation.scope
            or not evaluation.derivation):
        return False
    step = evaluation.derivation[-1]
    return (
        step.operator == structure
        and step.proposition == evaluation.proposition.template
        and step.premises == (child.proposition.template,)
        and step.source == evaluation.source
        and step.scope == evaluation.scope
    )


def _same_root(endpoint, groups):
    """按 EventTimeVerifier 的确定性同序根规范化端点。"""
    for group in groups:
        if endpoint in group:
            return min(group)
    return endpoint


def _temporal_units(certificate: TemporalProofCertificate) -> int:
    """按实际事实、边、同序组和显式前提计工作单元。"""
    result = certificate.result
    return (
        2
        + len(result.fact_set.facts)
        + len(result.before_edges)
        + sum(len(group) for group in result.same_groups)
        + len(certificate.premise_assertion_hashes)
    )


def _causal_units(
        certificate: CausalCounterfactualProofCertificate,
        ) -> int:
    """按 execution trace、witness 输入和反事实 Evidence 计工作单元。"""
    return (
        4
        + len(certificate.execution.fact.evidence_keys)
        + len(certificate.witness.input_objects)
        + len(certificate.pair.baseline.evidence_ids)
        + len(certificate.pair.intervention.evidence_ids)
    )


def _condition_units(certificate: ConditionProofCertificate) -> int:
    """按三个求值 trace 和类型 Evidence 计工作单元。"""
    return (
        3
        + len(certificate.condition.derivation)
        + len(certificate.conditioned.derivation)
        + len(certificate.evaluation.derivation)
        + len(certificate.kind_evidence_ids)
    )


def _not_units(certificate: NotProofCertificate) -> int:
    """按 child/root derivation 计 structural NOT 工作量。"""
    return 2 + len(certificate.child.derivation) + len(
        certificate.evaluation.derivation)


def _modal_units(certificate: ModalProofCertificate) -> int:
    """按 world/domain、resolver Evidence 和 derivation 计工作量。"""
    return (
        3
        + len(certificate.worlds)
        + len(certificate.domains)
        + len(certificate.resolver_evidence_ids)
        + len(certificate.child.derivation)
        + len(certificate.evaluation.derivation)
    )


def _result(
        family: int, status: int, units: int, reason: int,
        certificate_key: tuple[int, ...],
        ) -> ProofCheckResult:
    """形成统一稳定 trace，禁止 surface 或解释文本进入证书结果。"""
    return ProofCheckResult(
        family,
        status,
        units,
        (TRACE_VERSION, TRACE_DISPATCH, family, status, reason, units),
        certificate_key,
    )


def _check_temporal(
        certificate: TemporalProofCertificate, units: int,
        ) -> ProofCheckResult:
    """核对 explicit Event/Proposition、方向、scope 和原始 assertion。"""
    result = certificate.result
    key = certificate.stable_key()
    if (certificate.source != certificate.scope.source
            or result.fact_set.scope != certificate.scope
            or result.fact_set.scope.source != certificate.source):
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_SOURCE_SCOPE_MISMATCH,
            key,
        )
    if certificate.relation not in result.fact_set.relations:
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_TEMPORAL_PREMISE_MISMATCH,
            key,
        )
    available = {fact.assertion_hash for fact in result.fact_set.facts}
    if not set(certificate.premise_assertion_hashes).issubset(available):
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_TEMPORAL_PREMISE_MISMATCH,
            key,
        )
    if result.status == EVENT_TIME_CONFLICTED:
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_CONFLICTED,
            units,
            TRACE_TEMPORAL_CONFLICT,
            key,
        )
    if (result.status in {EVENT_TIME_EMPTY, EVENT_TIME_UNKNOWN}
            or certificate.relation in result.unknown_relations):
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_UNKNOWN,
            units,
            TRACE_TEMPORAL_UNKNOWN,
            key,
        )
    if result.status != EVENT_TIME_CONSISTENT:
        return _result(
            PROOF_FAMILY_TEMPORAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_UNREGISTERED_CERTIFICATE,
            key,
        )
    first = _same_root(certificate.first, result.same_groups)
    second = _same_root(certificate.second, result.same_groups)
    if certificate.direction == EVENT_TIME_BEFORE:
        supported = (first, second) in result.before_edges
    elif certificate.direction == EVENT_TIME_AFTER:
        supported = (second, first) in result.before_edges
    elif certificate.direction == EVENT_TIME_SAME:
        supported = first == second
    else:
        supported = False
    return _result(
        PROOF_FAMILY_TEMPORAL,
        PROOF_ACCEPTED if supported else PROOF_REJECTED,
        units,
        TRACE_LOGIC_PROVISIONAL if supported
        else TRACE_TEMPORAL_DIRECTION_MISSING,
        key,
    )


def _check_causal(
        certificate: CausalCounterfactualProofCertificate, units: int,
        ) -> ProofCheckResult:
    """核对方向、独立 witness、temporal 约束和 baseline/intervention。"""
    execution = certificate.execution
    pair = certificate.pair
    baseline = pair.baseline
    intervention = pair.intervention
    key = certificate.stable_key()
    cause = execution.cause.endpoint
    effect = execution.effect.endpoint
    if (baseline.cause != cause or baseline.effect != effect
            or intervention.cause != cause or intervention.effect != effect):
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_CAUSAL_ENDPOINT_MISMATCH,
            key,
        )
    witness = certificate.witness
    execution_source = execution.cause.evaluation.source
    if (witness.stance != EVIDENCE_SUPPORT
            or witness.verifier_source == execution_source
            or pair.baseline.source != witness.verifier_source
            or not {cause, effect}.issubset(set(witness.input_objects))):
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_CAUSAL_WITNESS_NOT_INDEPENDENT,
            key,
        )
    temporal = execution.temporal_assessment.status
    if temporal == CAUSAL_TEMPORAL_CONFLICTED:
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_CONFLICTED,
            units,
            TRACE_CAUSAL_TEMPORAL_CONFLICT,
            key,
        )
    if temporal == CAUSAL_TEMPORAL_UNKNOWN:
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_UNKNOWN,
            units,
            TRACE_CAUSAL_TEMPORAL_UNKNOWN,
            key,
        )
    if temporal != CAUSAL_TEMPORAL_ACCEPTED:
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_REJECTED,
            units,
            TRACE_CAUSAL_EXECUTION_MISMATCH,
            key,
        )
    cause_transition = (
        not baseline.cause_state.support
        and intervention.cause_state.support
        and not intervention.cause_state.refute
    )
    if certificate.direction == CAUSAL_DIRECTION_PROMOTING:
        effect_transition = (
            not baseline.effect_state.support
            and intervention.effect_state.support
            and not intervention.effect_state.refute
        )
        execution_matches = (
            execution.status == CAUSAL_EXECUTION_PREDICTED
            and execution.predicted_effect
            and execution.effect_state.support
            and not execution.effect_state.refute
        )
    else:
        effect_transition = (
            baseline.effect_state.support
            and not baseline.effect_state.refute
            and not intervention.effect_state.support
            and intervention.effect_state.refute
        )
        execution_matches = (
            execution.status != CAUSAL_EXECUTION_PREDICTED
            and not execution.predicted_effect
        )
    if not cause_transition or not effect_transition:
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_REJECTED,
            units,
            TRACE_COUNTERFACTUAL_PAIR_MISMATCH,
            key,
        )
    if not execution_matches:
        return _result(
            PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
            PROOF_REJECTED,
            units,
            TRACE_CAUSAL_EXECUTION_MISMATCH,
            key,
        )
    return _result(
        PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
        PROOF_ACCEPTED,
        units,
        TRACE_LOGIC_PROVISIONAL,
        key,
    )


def _condition_state(
        antecedent: LogicEvidenceState,
        consequent: LogicEvidenceState,
        ) -> LogicEvidenceState:
    """独立复算四态 material implication，保留前后件顺序。"""
    return LogicEvidenceState(
        antecedent.refute or consequent.support,
        antecedent.support and consequent.refute,
    )


def _check_condition(
        certificate: ConditionProofCertificate, units: int,
        ) -> ProofCheckResult:
    """区分 material/sufficient/necessary 并拒绝肯定后件。"""
    key = certificate.stable_key()
    if certificate.inference_kind == CONDITION_AFFIRMING_CONSEQUENT:
        return _result(
            PROOF_FAMILY_CONDITION,
            PROOF_REJECTED,
            units,
            TRACE_AFFIRMING_CONSEQUENT,
            key,
        )
    if certificate.condition_kind == CONDITION_MATERIAL:
        if certificate.kind_evidence_ids:
            return _result(
                PROOF_FAMILY_CONDITION,
                PROOF_FAIL_CLOSED,
                units,
                TRACE_CONDITION_KIND_EVIDENCE_MISSING,
                key,
            )
    elif certificate.condition_kind in {
            CONDITION_SUFFICIENT, CONDITION_NECESSARY}:
        if not certificate.kind_evidence_ids:
            return _result(
                PROOF_FAMILY_CONDITION,
                PROOF_UNKNOWN,
                units,
                TRACE_CONDITION_KIND_EVIDENCE_MISSING,
                key,
            )
    condition = certificate.condition
    conditioned = certificate.conditioned
    evaluation = certificate.evaluation
    if (condition.source != conditioned.source
            or condition.source != evaluation.source
            or condition.scope != conditioned.scope
            or condition.scope != evaluation.scope
            or not evaluation.derivation):
        return _result(
            PROOF_FAMILY_CONDITION,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_SOURCE_SCOPE_MISMATCH,
            key,
        )
    if certificate.condition_kind == CONDITION_NECESSARY:
        left, right = conditioned, condition
    else:
        left, right = condition, conditioned
    step = evaluation.derivation[-1]
    expected_evidence = tuple(sorted(set(
        left.evidence_ids + right.evidence_ids)))
    shape_matches = (
        step.operator == certificate.structure
        and step.proposition == evaluation.proposition.template
        and step.premises == (
            left.proposition.template,
            right.proposition.template,
        )
        and step.source == evaluation.source
        and step.scope == evaluation.scope
        and evaluation.state == _condition_state(left.state, right.state)
        and evaluation.evidence_ids == expected_evidence
    )
    if not shape_matches:
        return _result(
            PROOF_FAMILY_CONDITION,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_CONDITION_DIRECTION_MISMATCH,
            key,
        )
    status, reason = _logic_status(evaluation)
    return _result(
        PROOF_FAMILY_CONDITION, status, units, reason, key)


def _check_not(
        certificate: NotProofCertificate, units: int,
        ) -> ProofCheckResult:
    """只接受显式 structural NOT，并原样保留 unknown/conflicted。"""
    key = certificate.stable_key()
    child = certificate.child
    evaluation = certificate.evaluation
    if (not _logic_terminal(certificate.structure, child, evaluation)
            or evaluation.state != child.state.negate()
            or evaluation.evidence_ids != child.evidence_ids
            or evaluation.hypotheses != child.hypotheses):
        return _result(
            PROOF_FAMILY_NOT,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_NOT_STRUCTURE_MISMATCH,
            key,
        )
    status, reason = _logic_status(evaluation)
    return _result(PROOF_FAMILY_NOT, status, units, reason, key)


def _check_modal(
        certificate: ModalProofCertificate, units: int,
        ) -> ProofCheckResult:
    """核对 modal kind/frame、完整 world/domain 和独立 resolver Evidence。"""
    key = certificate.stable_key()
    epistemic = certificate.modal_kind in {
        MODAL_KIND_BOX_NECESSITY,
        MODAL_KIND_BOX_POSSIBILITY,
    }
    normative = certificate.modal_kind in {
        MODAL_KIND_DEONTIC_NECESSITY,
        MODAL_KIND_DEONTIC_POSSIBILITY,
    }
    if ((epistemic and certificate.frame != MODAL_FRAME_EPISTEMIC)
            or (normative and certificate.frame != MODAL_FRAME_NORMATIVE)):
        return _result(
            PROOF_FAMILY_MODAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_MODAL_FRAME_MISMATCH,
            key,
        )
    if not certificate.worlds_complete or not certificate.domains_complete:
        return _result(
            PROOF_FAMILY_MODAL,
            PROOF_UNKNOWN,
            units,
            TRACE_MODAL_CONTEXT_INCOMPLETE,
            key,
        )
    child = certificate.child
    evaluation = certificate.evaluation
    if (child.source != evaluation.source
            or not evaluation.derivation):
        return _result(
            PROOF_FAMILY_MODAL,
            PROOF_FAIL_CLOSED,
            units,
            TRACE_SOURCE_SCOPE_MISMATCH,
            key,
        )
    step = evaluation.derivation[-1]
    expected_evidence = tuple(sorted(set(
        child.evidence_ids + certificate.resolver_evidence_ids)))
    resolver_matches = (
        bool(certificate.resolver_evidence_ids)
        and not set(certificate.resolver_evidence_ids).intersection(
            child.evidence_ids)
        and step.operator == certificate.structure
        and step.proposition == evaluation.proposition.template
        and step.premises == (child.proposition.template,)
        and step.source == evaluation.source
        and step.scope == evaluation.scope
        and evaluation.evidence_ids == expected_evidence
        and step.evidence_ids == expected_evidence
    )
    if not resolver_matches:
        status = (
            PROOF_UNKNOWN
            if evaluation.status == STATE_UNKNOWN
            else PROOF_FAIL_CLOSED
        )
        return _result(
            PROOF_FAMILY_MODAL,
            status,
            units,
            TRACE_MODAL_RESOLVER_MISSING,
            key,
        )
    status, reason = _logic_status(evaluation)
    return _result(PROOF_FAMILY_MODAL, status, units, reason, key)


_CERTIFICATES = {
    TemporalProofCertificate: (
        PROOF_FAMILY_TEMPORAL, _temporal_units, _check_temporal),
    CausalCounterfactualProofCertificate: (
        PROOF_FAMILY_CAUSAL_COUNTERFACTUAL, _causal_units, _check_causal),
    ConditionProofCertificate: (
        PROOF_FAMILY_CONDITION, _condition_units, _check_condition),
    NotProofCertificate: (
        PROOF_FAMILY_NOT, _not_units, _check_not),
    ModalProofCertificate: (
        PROOF_FAMILY_MODAL, _modal_units, _check_modal),
}


class TypedProofFamilyDispatcher:
    """按精确 certificate 类型派发，拒绝跨 family 冒充并共享硬预算。"""

    def check(self, certificate, budget: ProofWorkBudget) -> ProofDispatchReceipt:
        """消费一次 certificate；不足预算或 family 漂移时稳定 fail closed。"""
        if not isinstance(budget, ProofWorkBudget):
            raise TypeError("proof dispatcher budget 类型非法")
        registration = _CERTIFICATES.get(type(certificate))
        if registration is None:
            family = PROOF_FAMILY_TEMPORAL
            units = 1
            reserved = budget.reserve(units)
            if reserved is None:
                result = _result(
                    family,
                    PROOF_BUDGET_EXHAUSTED,
                    0,
                    TRACE_BUDGET_EXHAUSTED,
                    (),
                )
                return ProofDispatchReceipt(result, budget)
            result = _result(
                family,
                PROOF_FAIL_CLOSED,
                units,
                TRACE_UNREGISTERED_CERTIFICATE,
                (),
            )
            return ProofDispatchReceipt(result, reserved)
        family, unit_counter, checker = registration
        units = unit_counter(certificate)
        reserved = budget.reserve(units)
        if reserved is None:
            result = _result(
                family,
                PROOF_BUDGET_EXHAUSTED,
                0,
                TRACE_BUDGET_EXHAUSTED,
                certificate.stable_key(),
            )
            return ProofDispatchReceipt(result, budget)
        if certificate.declared_family != family:
            result = _result(
                family,
                PROOF_FAMILY_MISMATCH,
                units,
                TRACE_FAMILY_MISMATCH,
                certificate.stable_key(),
            )
            return ProofDispatchReceipt(result, reserved)
        result = checker(certificate, units)
        return ProofDispatchReceipt(result, reserved)


__all__ = [
    "ProofDispatchReceipt",
    "TRACE_AFFIRMING_CONSEQUENT",
    "TRACE_BUDGET_EXHAUSTED",
    "TRACE_CAUSAL_ENDPOINT_MISMATCH",
    "TRACE_CAUSAL_EXECUTION_MISMATCH",
    "TRACE_CAUSAL_TEMPORAL_CONFLICT",
    "TRACE_CAUSAL_TEMPORAL_UNKNOWN",
    "TRACE_CAUSAL_WITNESS_NOT_INDEPENDENT",
    "TRACE_CONDITION_DIRECTION_MISMATCH",
    "TRACE_CONDITION_KIND_EVIDENCE_MISSING",
    "TRACE_COUNTERFACTUAL_PAIR_MISMATCH",
    "TRACE_FAMILY_MISMATCH",
    "TRACE_LOGIC_CONFLICTED",
    "TRACE_LOGIC_PROVISIONAL",
    "TRACE_LOGIC_REFUTED",
    "TRACE_LOGIC_UNKNOWN",
    "TRACE_MODAL_CONTEXT_INCOMPLETE",
    "TRACE_MODAL_FRAME_MISMATCH",
    "TRACE_MODAL_RESOLVER_MISSING",
    "TRACE_NOT_STRUCTURE_MISMATCH",
    "TRACE_SOURCE_SCOPE_MISMATCH",
    "TRACE_TEMPORAL_CONFLICT",
    "TRACE_TEMPORAL_DIRECTION_MISSING",
    "TRACE_TEMPORAL_PREMISE_MISMATCH",
    "TRACE_TEMPORAL_UNKNOWN",
    "TypedProofFamilyDispatcher",
]
