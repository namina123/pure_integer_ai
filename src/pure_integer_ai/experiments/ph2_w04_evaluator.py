"""W-04 private evaluator 的四 bearing 与 generation hard conjunct。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_ABLATION_KEYS,
    W04_DIMENSION_KEYS,
    W04_EVALUATION_ORDER,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_contract import (
    W04_GENERATION_ABLATION_KEY,
    W04PrivateCase,
    W04PrivateDimensionResult,
    W04PrivateEvaluationError,
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w04_generation import (
    build_w04_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_READY,
    W04GenerationRequest,
)


@dataclass(frozen=True)
class W04EvaluatorAblation:
    """一次只关闭一个预注册承重维度。"""

    key: str

    def __post_init__(self) -> None:
        if self.key not in (*W04_ABLATION_KEYS, W04_GENERATION_ABLATION_KEY):
            raise W04PrivateEvaluationError("W-04 ablation key 非法")

    @property
    def dimension_key(self) -> str:
        if self.key == W04_GENERATION_ABLATION_KEY:
            return W04_EVALUATION_ORDER[-1]
        return self.key.removesuffix("-ABLATION")


def _generation_ok(learning, *, connected: bool) -> tuple[bool, dict[str, int]]:
    if not connected:
        return False, {"connected": 0, "ready": 0, "support": 0}
    runtime = build_w04_generation_runtime(learning)
    ready = 0
    support = 0
    for candidate in learning.active_candidates():
        choice = runtime.choose(W04GenerationRequest(
            candidate.primitive_registry,
            candidate.primitive_kind,
            candidate.context_text,
            True,
        ))
        if choice.status == W04_GENERATION_READY and choice.options:
            ready += 1
            use = runtime.adopt(choice, choice.options)
            if runtime.verify_use(use).verdict == "SUPPORT":
                support += 1
    passed = bool(learning.active_candidates()) and ready == support == len(
        learning.active_candidates())
    return passed, {"connected": 1, "ready": ready, "support": support}


def _result(
        case: W04PrivateCase,
        passed: bool,
        evidence: dict[str, object],
        ) -> W04PrivateDimensionResult:
    return W04PrivateDimensionResult(
        case.dimension_key,
        "PASS" if passed else "FAIL",
        int(passed),
        1,
        int(not passed),
        0,
        evidence_commitment({
            "challenge_key": list(case.challenge_key),
            "dimension_key": case.dimension_key,
            "evidence": evidence,
            "passed": int(passed),
        }),
    )


def evaluate_w04_learning_runtime(
        learning,
        cases: tuple[W04PrivateCase, ...],
        *,
        ablation: W04EvaluatorAblation | None = None,
        generation_bridge_connected: bool = True,
        ) -> tuple[W04PrivateDimensionResult, ...]:
    """在 candidate 的只读逻辑投影上执行五项 1/1 结果。"""
    if (not isinstance(cases, tuple)
            or tuple(item.dimension_key for item in cases)
            != W04_EVALUATION_ORDER):
        raise W04PrivateEvaluationError("W-04 private case 顺序漂移")
    if ablation is not None and not isinstance(ablation, W04EvaluatorAblation):
        raise TypeError("W-04 ablation 类型非法")
    active = learning.active_candidates()
    registered = learning.registered_candidates()
    superseded = learning.superseded_candidates()
    report = learning.report()
    replacement_pairs = tuple(
        (old, new)
        for old in superseded
        for new in active
        if old.coordinate_key() == new.coordinate_key()
        and (old.surface_form, old.context_text)
        != (new.surface_form, new.context_text)
    )
    surface_competition = any(
        len({item.coordinate_key() for item in registered
             if item.surface_form == candidate.surface_form}) > 1
        for candidate in registered
    )
    primitive_competition = any(
        len({item.surface_form for item in registered
             if item.coordinate_key() == candidate.coordinate_key()}) > 1
        for candidate in registered
    )
    baseline = {
        W04_DIMENSION_KEYS[0]: (
            bool(replacement_pairs)
            and report.superseded_candidate_count >= 1
        ),
        W04_DIMENSION_KEYS[1]: (
            surface_competition
            and primitive_competition
            and report.active_candidate_count == 1
        ),
        W04_DIMENSION_KEYS[2]: (
            report.evidence_application_count >= 4
            and report.account_count >= 4
            and report.active_candidate_count == 1
        ),
        W04_DIMENSION_KEYS[3]: (
            report.candidate_count >= 4
            and report.superseded_candidate_count >= 1
            and report.conflict_candidate_count >= 1
        ),
    }
    generation_pass, generation_evidence = _generation_ok(
        learning,
        connected=generation_bridge_connected
        and (ablation is None or ablation.key != W04_GENERATION_ABLATION_KEY),
    )
    baseline[W04_EVALUATION_ORDER[-1]] = generation_pass
    disabled = None if ablation is None else ablation.dimension_key
    results = []
    for case in cases:
        passed = baseline[case.dimension_key] and case.dimension_key != disabled
        evidence = {
            "active_candidate_count": report.active_candidate_count,
            "candidate_count": report.candidate_count,
            "conflict_candidate_count": report.conflict_candidate_count,
            "evidence_application_count": report.evidence_application_count,
            "generation": generation_evidence,
            "superseded_candidate_count": report.superseded_candidate_count,
            "target_disabled": int(case.dimension_key == disabled),
        }
        results.append(_result(case, passed, evidence))
    return tuple(results)


__all__ = [
    "W04EvaluatorAblation",
    "evaluate_w04_learning_runtime",
]
