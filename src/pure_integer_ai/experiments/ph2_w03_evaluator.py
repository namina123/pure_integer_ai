"""PH2 W-03 五维 private evaluator 的结构探针与正交消融。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_DIMENSION_KEYS,
    W03_GENERATION_HARD_CONJUNCT,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_contract import (
    W03PrivateCase,
    W03PrivateDimensionResult,
    W03PrivateEvaluationError,
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w03_generation import (
    W03_GENERATION_CLARIFY,
    W03_GENERATION_HARD_CASES,
    W03_GENERATION_READY,
    W03_GENERATION_UNKNOWN,
    W03ExpressionConstraints,
    W03GenerationCaseResult,
    W03GenerationRequest,
    build_w03_generation_runtime,
    run_w03_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03UnderstandingRuntime,
)
from pure_integer_ai.experiments.ph2_w03_understanding_contract import (
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_UNKNOWN,
)


@dataclass(frozen=True)
class W03EvaluatorAblation:
    """四个冻结 bearing 之一的 evaluator-side 正交数据切断。"""

    ablation_key: str

    def __post_init__(self) -> None:
        if self.ablation_key not in W03_ABLATION_KEYS:
            raise W03PrivateEvaluationError("W-03 evaluator ablation key 非法")

    @property
    def dimension_key(self) -> str:
        return W03_DIMENSION_KEYS[W03_ABLATION_KEYS.index(self.ablation_key)]


def _consumer_active(runtime: W03UnderstandingRuntime, candidate: Any) -> bool:
    """只以 ActiveSenseConsumer 的精确 atom/context 输出判 active。"""
    return any(
        item.sense == candidate.sense
        for item in runtime.consumer.lookup(
            candidate.anchor.atom,
            context=candidate.context,
        )
    )


def _groups(values: tuple[Any, ...], key_fn) -> tuple[tuple[Any, ...], ...]:
    grouped: dict[object, list[Any]] = {}
    for item in values:
        grouped.setdefault(key_fn(item), []).append(item)
    return tuple(
        tuple(sorted(group, key=lambda item: item.sense.stable_key()))
        for _, group in sorted(
            grouped.items(),
            key=lambda item: repr(item[0]),
        )
    )


def _concept_split_evidence(
        runtime: W03UnderstandingRuntime,
        *,
        disabled: bool,
        ) -> tuple[bool, dict[str, int]]:
    """检查同 atom/context 的多 Sense/Concept 保留且没有物理合并。"""
    candidates = runtime.output.candidates
    competition = tuple(
        group for group in _groups(
            candidates, lambda item: (item.anchor.atom, item.context))
        if len(group) > 1
    )
    separated = tuple(
        group for group in competition
        if len({item.sense for item in group}) == len(group)
        and len({item.concept for item in group}) > 1
    )
    passed = (
        not disabled
        and len(candidates) == 59
        and len({item.sense for item in candidates}) == len(candidates)
        and bool(separated)
    )
    return passed, {
        "candidate_count": len(candidates),
        "competition_group_count": len(competition),
        "separated_group_count": len(separated),
        "unique_sense_count": len({item.sense for item in candidates}),
    }


def _polysemy_evidence(
        runtime: W03UnderstandingRuntime,
        *,
        disabled: bool,
        ) -> tuple[bool, dict[str, int]]:
    """检查 exact context 唯一采用与无证据多候选不取首项同时成立。"""
    candidates = runtime.output.candidates
    contextual_groups = _groups(
        candidates, lambda item: (item.anchor.atom, item.context))
    competition = tuple(group for group in contextual_groups if len(group) > 1)
    unique_count = 0
    ambiguous_count = 0
    illegal_selection_count = 0
    for group in contextual_groups:
        resolution = runtime.resolve(
            group[0].anchor.atom,
            context=group[0].context,
        )
        if resolution.status == W03_UNDERSTANDING_UNIQUE:
            unique_count += 1
        if len(group) > 1 and resolution.status in {
                W03_UNDERSTANDING_AMBIGUOUS,
                W03_UNDERSTANDING_UNKNOWN}:
            ambiguous_count += 1
            if resolution.selected is not None:
                illegal_selection_count += 1
    passed = (
        not disabled
        and unique_count >= 1
        and ambiguous_count >= 1
        and illegal_selection_count == 0
    )
    return passed, {
        "ambiguous_or_unknown_group_count": ambiguous_count,
        "competition_group_count": len(competition),
        "illegal_selection_count": illegal_selection_count,
        "unique_group_count": unique_count,
    }


def _source_conflict_evidence(
        runtime: W03UnderstandingRuntime,
        *,
        disabled: bool,
        ) -> tuple[bool, dict[str, int]]:
    """检查 support/refute 分账后 consumer 不把冲突候选私选为答案。"""
    conflict_count = 0
    illegal_selection_count = 0
    for candidate in runtime.output.candidates:
        stances = {
            item.stance for item in runtime.evidence_accounts(candidate.sense)
        }
        if disabled:
            stances.discard(EVIDENCE_REFUTE)
        if not {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}.issubset(stances):
            continue
        conflict_count += 1
        resolution = runtime.resolve(
            candidate.anchor.atom,
            context=candidate.context,
        )
        if resolution.selected is not None:
            illegal_selection_count += 1
    passed = conflict_count >= 1 and illegal_selection_count == 0
    return passed, {
        "conflict_candidate_count": conflict_count,
        "illegal_selection_count": illegal_selection_count,
    }


def _supersede_pairs(runtime: W03UnderstandingRuntime) -> tuple[tuple[Any, Any], ...]:
    pairs = []
    for new_candidate in runtime.output.candidates:
        old_key = runtime.supersedes_observation(new_candidate.sense)
        if old_key is None:
            continue
        for old_candidate in runtime.candidate_for_observation(old_key):
            pairs.append((old_candidate, new_candidate))
    return tuple(sorted(
        pairs,
        key=lambda item: (
            item[0].sense.stable_key(), item[1].sense.stable_key()),
    ))


def _supersede_evidence(
        runtime: W03UnderstandingRuntime,
        *,
        disabled: bool,
        ) -> tuple[bool, dict[str, int]]:
    """检查同 atom revision 只退出旧 Sense，保留新旧身份和 derived refute。"""
    pairs = () if disabled else _supersede_pairs(runtime)
    valid = []
    for old_candidate, new_candidate in pairs:
        derived_refute = any(
            item.derived_supersede and item.stance == EVIDENCE_REFUTE
            for item in runtime.evidence_accounts(old_candidate.sense)
        )
        if (old_candidate.anchor.atom == new_candidate.anchor.atom
                and old_candidate.sense != new_candidate.sense
                and not _consumer_active(runtime, old_candidate)
                and _consumer_active(runtime, new_candidate)
                and derived_refute):
            valid.append((old_candidate, new_candidate))
    passed = bool(valid)
    return passed, {
        "supersede_pair_count": len(pairs),
        "valid_same_atom_pair_count": len(valid),
    }


def _request(
        candidate: Any,
        case: W03PrivateCase,
        ordinal: int,
        *,
        context: Any = ...,
        ) -> W03GenerationRequest:
    """从 private challenge key 派生 request identity，不携带 expected surface。"""
    selected_context = candidate.context if context is ... else context
    return W03GenerationRequest(
        LosslessIntegerKey((30603, ordinal, *case.challenge_key)),
        candidate.sense,
        candidate.concept,
        selected_context,
        candidate.anchor.branch,
        W03ExpressionConstraints(True, True, 64),
        candidate.source_ref,
        document_scope(candidate.source_ref),
    )


def _generation_evidence(
        runtime: W03UnderstandingRuntime,
        case: W03PrivateCase,
        persisted_outcomes: tuple[dict[str, object], ...],
        *,
        sense_consumer_connected: bool,
        choice_bridge_connected: bool,
        ) -> tuple[bool, dict[str, int]]:
    """执行五类 generation case，并把异常留给基础设施层向上冒泡。"""
    candidates = runtime.output.candidates
    active = tuple(item for item in candidates if _consumer_active(runtime, item))
    active_groups = tuple(
        group for group in _groups(
            active,
            lambda item: (item.concept, item.context, item.anchor.branch),
        )
        if len({item.anchor.extracted.surface for item in group}) >= 2
    )
    all_atom_groups = tuple(
        group for group in _groups(candidates, lambda item: item.anchor.atom)
        if (len({item.sense for item in group}) > 1
            and len({item.concept for item in group}) > 1)
    )
    homograph = next(
        (group for group in all_atom_groups
         if any(_consumer_active(runtime, item) for item in group)),
        (),
    )
    pairs = tuple(
        pair for pair in _supersede_pairs(runtime)
        if pair[0].anchor.atom == pair[1].anchor.atom
    )
    generation = build_w03_generation_runtime(
        runtime,
        sense_consumer_connected=sense_consumer_connected,
        choice_bridge_connected=choice_bridge_connected,
    )
    target_ok = False
    isolation_ok = False
    ambiguity_ok = False
    multiple_ok = False
    withdrawal_ok = False
    if active:
        target = active[0]
        choice = generation.choose(_request(target, case, 1))
        target_ok = (
            choice.status == W03_GENERATION_READY
            and any(item.sense == target.sense for item in choice.options)
            and choice.selected is None
        )
    if homograph:
        target = next(item for item in homograph if _consumer_active(runtime, item))
        exact = generation.choose(_request(target, case, 2))
        missing = generation.choose(_request(target, case, 3, context=None))
        other_senses = {item.sense for item in homograph if item.sense != target.sense}
        isolation_ok = (
            exact.status == W03_GENERATION_READY
            and not other_senses.intersection(item.sense for item in exact.options)
        )
        ambiguity_ok = (
            missing.status == W03_GENERATION_CLARIFY
            and missing.options == () and missing.selected is None
        )
    if active_groups:
        target = active_groups[0][0]
        choice = generation.choose(_request(target, case, 4))
        multiple_ok = (
            choice.status == W03_GENERATION_READY
            and len({item.surface for item in choice.options}) >= 2
            and len({item.sense for item in choice.options}) >= 2
            and choice.selected is None
        )
    if pairs:
        old_candidate, new_candidate = pairs[0]
        old_choice = generation.choose(_request(old_candidate, case, 5))
        new_choice = generation.choose(_request(new_candidate, case, 6))
        by_use: dict[tuple[int, ...], set[str]] = {}
        for item in persisted_outcomes:
            raw_key = item.get("use_key") if isinstance(item, dict) else None
            verdict = item.get("verdict") if isinstance(item, dict) else None
            if (isinstance(raw_key, list)
                    and all(type(value) is int for value in raw_key)
                    and verdict in {"SUPPORT", "REFUTE"}):
                by_use.setdefault(tuple(raw_key), set()).add(str(verdict))
        transition_count = sum(
            {"SUPPORT", "REFUTE"}.issubset(values)
            for values in by_use.values()
        )
        withdrawal_ok = (
            old_choice.status == W03_GENERATION_UNKNOWN
            and old_choice.options == ()
            and new_choice.status == W03_GENERATION_READY
            and transition_count >= 1
        )
    booleans = (
        target_ok, isolation_ok, ambiguity_ok, multiple_ok, withdrawal_ok)
    cases = tuple(
        W03GenerationCaseResult(
            name,
            passed,
            LosslessIntegerKey((30603, 100 + ordinal, *case.challenge_key)),
        )
        for ordinal, (name, passed) in enumerate(
            zip(W03_GENERATION_HARD_CASES, booleans, strict=True), start=1)
    )
    report = run_w03_generation_hard_conjunct(
        cases,
        sense_consumer_connected=sense_consumer_connected,
        choice_bridge_connected=choice_bridge_connected,
    )
    return report.status == "PASS", {
        "active_candidate_count": len(active),
        "hard_case_pass_count": sum(booleans),
        "homograph_group_count": len(all_atom_groups),
        "multi_surface_group_count": len(active_groups),
        "supersede_pair_count": len(pairs),
    }


def _result(
        case: W03PrivateCase,
        passed: bool,
        evidence: dict[str, int],
        ) -> W03PrivateDimensionResult:
    """把一个逻辑判定封装为 1/1、fail=0、NE 不被吞并的结果。"""
    return W03PrivateDimensionResult(
        case.dimension_key,
        "PASS" if passed else "FAIL",
        int(passed),
        1,
        int(not passed),
        0,
        evidence_commitment({
            "case_challenge_key": list(case.challenge_key),
            "dimension_key": case.dimension_key,
            "evidence": evidence,
            "passed": int(passed),
        }),
    )


def evaluate_w03_learning_runtime(
        runtime: W03UnderstandingRuntime,
        cases: tuple[W03PrivateCase, ...],
        *,
        persisted_generation_outcomes: tuple[dict[str, object], ...],
        ablation: W03EvaluatorAblation | None = None,
        sense_consumer_connected: bool = True,
        choice_bridge_connected: bool = True,
        ) -> tuple[W03PrivateDimensionResult, ...]:
    """在只读 candidate runtime 上执行四 bearing 与 generation 硬合取。"""
    if not isinstance(runtime, W03UnderstandingRuntime):
        raise TypeError("W-03 evaluator runtime 类型非法")
    if (not isinstance(cases, tuple)
            or tuple(item.dimension_key for item in cases)
            != (*W03_DIMENSION_KEYS, W03_GENERATION_HARD_CONJUNCT)):
        raise W03PrivateEvaluationError("W-03 private case 顺序漂移")
    if not isinstance(persisted_generation_outcomes, tuple):
        raise TypeError("persisted generation outcomes 必须是 tuple")
    if ablation is not None and not isinstance(ablation, W03EvaluatorAblation):
        raise TypeError("W-03 evaluator ablation 类型非法")
    if type(sense_consumer_connected) is not bool or type(
            choice_bridge_connected) is not bool:
        raise TypeError("generation connection 必须是严格 bool")
    disabled = None if ablation is None else ablation.dimension_key
    checks = (
        _concept_split_evidence(
            runtime, disabled=disabled == W03_DIMENSION_KEYS[0]),
        _polysemy_evidence(
            runtime, disabled=disabled == W03_DIMENSION_KEYS[1]),
        _source_conflict_evidence(
            runtime, disabled=disabled == W03_DIMENSION_KEYS[2]),
        _supersede_evidence(
            runtime, disabled=disabled == W03_DIMENSION_KEYS[3]),
        _generation_evidence(
            runtime,
            cases[4],
            persisted_generation_outcomes,
            sense_consumer_connected=sense_consumer_connected,
            choice_bridge_connected=choice_bridge_connected,
        ),
    )
    return tuple(
        _result(case, passed, evidence)
        for case, (passed, evidence) in zip(cases, checks, strict=True)
    )


__all__ = [
    "W03EvaluatorAblation",
    "evaluate_w03_learning_runtime",
]
