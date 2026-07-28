"""MD-04 结果盲 fixture、真实 MD-03 center 和运行绑定。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    MD_BASELINE_KEYS,
    MD_HARD_INVARIANT_KEYS,
    MD_SAMPLE_GROUP_KEYS,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
    DirectionalWriteBoundary,
)
from pure_integer_ai.experiments.ph2_md04_probe_contract import (
    FORMAT_VERSION,
    MD04_ABLATION_KEYS,
    MD04_PLAN_VERSION,
    MD04_PREREGISTRATION_VERSION,
    MD04_SCALE_FACTORS,
    MD04ProbeContractError,
    MD04ProbePlan,
    ProbeCaseDefinition,
    ProbeCenterRef,
    ProbeMemoryCandidate,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MemoryAttentionCenter,
    MemoryCenterOrigin,
    MemoryDynamicsBoundary,
    zero_execution_state,
)


_PAYLOAD_BY_DIRECTION = {
    "GENERATION": "ANSWER_GENERATION_GOAL",
    "REASONING": "EVIDENCE_OBLIGATION",
    "UNDERSTANDING": "CURRENT_TYPED_INPUT",
}
_DIRECTION_ORDINAL = {
    "GENERATION": 1,
    "REASONING": 2,
    "UNDERSTANDING": 3,
}
_RELATION = StableRecordKey((44020, 1))
_OTHER_RELATION = StableRecordKey((44020, 2))


def _key(*values: int) -> StableRecordKey:
    return StableRecordKey(tuple(values))


def _positive_full_key(values: tuple[int, ...]) -> StableRecordKey:
    """把可含零的完整协议键逐项平移为可逆 StableRecordKey。"""
    if not isinstance(values, tuple) or not values:
        raise MD04ProbeContractError("完整协议键不能为空")
    return StableRecordKey(tuple(value + 1 for value in values))


def _make_center(
        case_ordinal: int,
        center_ordinal: int,
        direction: str,
        ) -> DirectionalMemoryCenter:
    direction_ordinal = _DIRECTION_ORDINAL[direction]
    target = _key(44100, case_ordinal, center_ordinal)
    condition = _key(44110, case_ordinal, center_ordinal)
    boundary = MemoryDynamicsBoundary(
        _key(44120, case_ordinal),
        _key(44121, case_ordinal),
        _key(44122, case_ordinal),
        _key(44123, case_ordinal),
    )
    origin = MemoryCenterOrigin(
        "GOAL" if direction == "GENERATION" else "OPEN_QUESTION",
        _key(44130, case_ordinal, center_ordinal),
        tuple(sorted((target, condition))),
    )
    center = MemoryAttentionCenter(
        _key(44140, case_ordinal, center_ordinal),
        direction,
        "MANDATORY",
        (origin,),
        _key(44150, direction_ordinal),
        target,
        boundary,
        _key(44160, case_ordinal),
        _key(44170, case_ordinal),
        tuple(sorted((target, condition, origin.origin_key))),
        "ACTIVE",
        1,
    )
    return DirectionalMemoryCenter(
        center,
        _key(44180, case_ordinal),
        _PAYLOAD_BY_DIRECTION[direction],
        _key(44190, case_ordinal, center_ordinal),
        (condition,),
        DirectionalWriteBoundary.for_direction(direction),
    )


def _center_ref(center: DirectionalMemoryCenter) -> ProbeCenterRef:
    boundary = center.center.boundary
    return ProbeCenterRef(
        _positive_full_key(center.stable_key()),
        center.center.center_key,
        center.center.direction,
        center.center.target_key,
        boundary.owner_key,
        boundary.scope_key,
        boundary.source_key,
        boundary.version_key,
        center.center.expansion_profile_key,
        center.adoption_condition_keys,
    )


def _candidate(
        case_ordinal: int,
        candidate_ordinal: int,
        center: DirectionalMemoryCenter,
        *,
        placement: str,
        correct_structure: bool = True,
        relation: StableRecordKey = _RELATION,
        stance: str = "SUPPORT",
        authorized: int = 1,
        access_allowed: int = 1,
        grounded: int = 1,
        activation: int = 10,
        recency_rank: int = 10,
        distance: int = 1,
        target: StableRecordKey | None = None,
        dependency_ordinal: int = 1,
        ) -> ProbeMemoryCandidate:
    key_ordinal = candidate_ordinal + (100 if placement == "COLD" else 0)
    candidate_key = _key(44200, case_ordinal, key_ordinal)
    structure = (
        center.adoption_condition_keys[0]
        if correct_structure else _key(44210, case_ordinal, candidate_ordinal))
    channel = {
        "COLD": "L4_SEALED_PAGE",
        "HOT": "L1_WORK_MEMORY",
        "INDEX": "SPECIAL_TYPED_INDEX",
    }[placement]
    return ProbeMemoryCandidate(
        candidate_key,
        center.center.target_key if target is None else target,
        relation,
        _key(44220, case_ordinal, candidate_ordinal),
        _key(44230, case_ordinal, candidate_ordinal),
        structure,
        (_key(44240, case_ordinal, dependency_ordinal),),
        placement,
        channel,
        distance,
        recency_rank,
        activation,
        stance,
        authorized,
        access_allowed,
        grounded,
    )


def _ceiling(*, scanned: int = 8, page_reads: int = 2) -> CanonicalJsonObject:
    return CanonicalJsonObject.from_value({
        "max_agenda_entries": 4,
        "max_candidates": 8,
        "max_cold_bytes": 100000,
        "max_consumptions": 4,
        "max_logic_steps": 64,
        "max_page_reads": page_reads,
        "max_recomputes": 1,
        "max_scanned_objects": scanned,
    })


@dataclass(frozen=True)
class ProbeCaseBinding:
    """把 plan 中完整引用绑定到真实 MD-03 center 实例。"""

    case_key: StableRecordKey
    centers: tuple[DirectionalMemoryCenter, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise MD04ProbeContractError("binding case key 非法")
        if (not isinstance(self.centers, tuple) or not self.centers
                or any(not isinstance(item, DirectionalMemoryCenter)
                       for item in self.centers)):
            raise MD04ProbeContractError("binding centers 非法")


@dataclass(frozen=True)
class MD04FixtureBundle:
    """冻结 plan 和不进入 artifact 的类型化对象绑定。"""

    plan: MD04ProbePlan
    bindings: tuple[ProbeCaseBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MD04ProbePlan):
            raise MD04ProbeContractError("fixture plan 非法")
        if (not isinstance(self.bindings, tuple)
                or any(not isinstance(item, ProbeCaseBinding)
                       for item in self.bindings)):
            raise MD04ProbeContractError("fixture bindings 非法")
        binding_keys = tuple(item.case_key for item in self.bindings)
        if binding_keys != tuple(item.case_key for item in self.plan.cases):
            raise MD04ProbeContractError("fixture binding 未逐 case 对齐")
        for case, binding in zip(self.plan.cases, self.bindings):
            refs = tuple(_center_ref(item) for item in binding.centers)
            if refs != case.center_refs:
                raise MD04ProbeContractError("fixture center 完整身份漂移")

    def centers_for(
            self,
            case_key: StableRecordKey,
            ) -> tuple[DirectionalMemoryCenter, ...]:
        for binding in self.bindings:
            if binding.case_key == case_key:
                return binding.centers
        raise KeyError(case_key)


def _case(
        ordinal: int,
        group: str,
        centers: tuple[DirectionalMemoryCenter, ...],
        *,
        hot: tuple[ProbeMemoryCandidate, ...] = (),
        cold: tuple[ProbeMemoryCandidate, ...] = (),
        cold_admitted: int = 1,
        conflict_scan: int = 0,
        revision_dependency: StableRecordKey | None = None,
        ceiling: CanonicalJsonObject | None = None,
        ) -> tuple[ProbeCaseDefinition, ProbeCaseBinding]:
    case_key = _key(44300, ordinal)
    lower = _key(44200, ordinal, 100)
    upper = _key(44200, ordinal, 199)
    definition = ProbeCaseDefinition(
        case_key,
        group,
        tuple(sorted(_center_ref(item) for item in centers)),
        (_RELATION,),
        tuple(sorted(hot)),
        tuple(sorted(cold)),
        lower,
        upper,
        cold_admitted,
        conflict_scan,
        revision_dependency,
        ceiling or _ceiling(),
    )
    return definition, ProbeCaseBinding(case_key, centers)


def _build_cases() -> tuple[
        tuple[ProbeCaseDefinition, ...], tuple[ProbeCaseBinding, ...]]:
    rows: list[tuple[ProbeCaseDefinition, ProbeCaseBinding]] = []

    access = _make_center(1, 1, "UNDERSTANDING")
    rows.append(_case(
        1, "ACCESS_BUDGET_GROUNDING_DISTINCT", (access,),
        cold=(_candidate(
            1, 1, access, placement="COLD", access_allowed=0),),
    ))

    budget = _make_center(2, 1, "UNDERSTANDING")
    rows.append(_case(
        2, "ACCESS_BUDGET_GROUNDING_DISTINCT", (budget,),
        cold=(
            _candidate(
                2, 1, budget, placement="COLD", correct_structure=False,
                activation=30),
            _candidate(2, 2, budget, placement="COLD", activation=20),
            _candidate(2, 3, budget, placement="COLD", activation=10),
        ),
        ceiling=_ceiling(scanned=1, page_reads=1),
    ))

    grounding = _make_center(3, 1, "UNDERSTANDING")
    rows.append(_case(
        3, "ACCESS_BUDGET_GROUNDING_DISTINCT", (grounding,),
        cold=(_candidate(
            3, 1, grounding, placement="COLD", grounded=0),),
    ))

    cold = _make_center(4, 1, "UNDERSTANDING")
    rows.append(_case(
        4, "COLD_CORRECT_HOT_DISTRACTOR", (cold,),
        hot=(_candidate(
            4, 1, cold, placement="HOT", correct_structure=False,
            activation=100, recency_rank=100),),
        cold=(_candidate(
            4, 2, cold, placement="COLD", activation=20,
            recency_rank=1, distance=8),),
    ))

    exact = _make_center(5, 1, "REASONING")
    rows.append(_case(
        5, "EXACT_MEMORY_STRUCTURE_DISTANCE_HELD_OUT", (exact,),
        hot=(_candidate(
            5, 1, exact, placement="HOT", correct_structure=False,
            activation=80, recency_rank=80),),
        cold=(_candidate(
            5, 2, exact, placement="COLD", activation=40,
            recency_rank=2, distance=12),),
    ))

    generation = _make_center(6, 1, "GENERATION")
    rows.append(_case(
        6, "GENERATION_UNAUTHORIZED_ACTIVATION", (generation,),
        hot=(_candidate(
            6, 1, generation, placement="HOT", authorized=0,
            activation=100, recency_rank=100),),
        cold_admitted=0,
    ))

    conflict = _make_center(7, 1, "UNDERSTANDING")
    rows.append(_case(
        7, "HOT_COLD_CONFLICT", (conflict,),
        hot=(_candidate(
            7, 1, conflict, placement="HOT", stance="SUPPORT",
            activation=80),),
        cold=(_candidate(
            7, 2, conflict, placement="COLD", stance="REFUTE",
            activation=20, distance=6),),
        conflict_scan=1,
    ))

    hot = _make_center(8, 1, "UNDERSTANDING")
    rows.append(_case(
        8, "HOT_CORRECT", (hot,),
        hot=(_candidate(
            8, 1, hot, placement="HOT", activation=60),),
        cold_admitted=0,
    ))

    indexed = _make_center(9, 1, "REASONING")
    rows.append(_case(
        9, "INDEXED_DIRECT_ROUTE", (indexed,),
        hot=(_candidate(
            9, 1, indexed, placement="INDEX", activation=30),),
        cold_admitted=0,
    ))

    revision = _make_center(10, 1, "UNDERSTANDING")
    revision_key = _key(44240, 10, 1)
    rows.append(_case(
        10, "LOCAL_REVISION_BIT_IDENTICAL", (revision,),
        hot=(
            _candidate(
                10, 1, revision, placement="HOT", activation=50,
                dependency_ordinal=1),
            _candidate(
                10, 2, revision, placement="HOT", activation=1,
                target=_key(44999, 10), dependency_ordinal=2),
        ),
        cold_admitted=0,
        revision_dependency=revision_key,
    ))

    multicenters = (
        _make_center(11, 1, "GENERATION"),
        _make_center(11, 2, "REASONING"),
        _make_center(11, 3, "UNDERSTANDING"),
    )
    rows.append(_case(
        11, "MULTICENTER_OBLIGATIONS", multicenters,
        cold=tuple(
            _candidate(
                11, ordinal, center, placement="COLD",
                activation=40 - ordinal, distance=5)
            for ordinal, center in enumerate(multicenters, start=1)),
    ))

    unknown = _make_center(12, 1, "UNDERSTANDING")
    rows.append(_case(
        12, "NO_ANSWER_UNKNOWN", (unknown,),
        cold=(),
        cold_admitted=1,
    ))

    definitions = tuple(item[0] for item in rows)
    bindings = tuple(item[1] for item in rows)
    return definitions, bindings


def build_md04_fixture_bundle(
        *,
        md03_manifest_sha256: str,
        baseline_manifest_sha256: str,
        ) -> MD04FixtureBundle:
    """冻结首次结果前 plan，并绑定同一组真实 MD-03 center。"""
    cases, bindings = _build_cases()
    plan = MD04ProbePlan(
        FORMAT_VERSION,
        MD04_PLAN_VERSION,
        MD04_PREREGISTRATION_VERSION,
        "data/ph2/manifests/md03_directional_center_adapter_v1.json",
        md03_manifest_sha256,
        "data/ph2/manifests/language_capability_baseline_v21.json",
        baseline_manifest_sha256,
        MD_BASELINE_KEYS,
        MD_SAMPLE_GROUP_KEYS,
        MD_HARD_INVARIANT_KEYS,
        MD04_ABLATION_KEYS,
        MD04_SCALE_FACTORS,
        cases,
        CanonicalJsonObject.from_value({
            "ablation_min_degraded_dimensions": 1,
            "challenge_min_improvements": 1,
            "freeze_before_run": 1,
            "hard_zero_required": 1,
            "max_peak_hot_objects": 4,
            "quality_regression_allowed": 0,
            "unrelated_growth_allowed": 0,
            "unrelated_query_cold_read_max": 0,
        }),
        0,
        zero_execution_state(),
    )
    return MD04FixtureBundle(plan, bindings)


__all__ = [
    "MD04FixtureBundle",
    "ProbeCaseBinding",
    "build_md04_fixture_bundle",
]
