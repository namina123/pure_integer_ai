"""MD-04 四基线、多中心 ring、K-04 typed range 和规模 probe。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    MD_BASELINE_KEYS,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
)
from pure_integer_ai.experiments.ph2_md04_probe_contract import (
    FORMAT_VERSION,
    MD04_ABLATION_KEYS,
    MD04_RUN_VERSION,
    MD04_SCALE_FACTORS,
    MD04ProbeContractError,
    MD04ProbeRunArtifact,
    ProbeAblationOutcome,
    ProbeCaseDefinition,
    ProbeCaseOutcome,
    ProbeMemoryCandidate,
)
from pure_integer_ai.experiments.ph2_md04_probe_fixture import (
    MD04FixtureBundle,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MemoryDynamicsStopDecision,
    MemoryRingReceipt,
    zero_execution_state,
)
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetMetrics,
    QueryHotSetPolicy,
    QueryPrefetchContext,
    QuerySegmentHotSet,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import (
    InMemoryObjectRepository,
    OBJECT_KIND_SEGMENT,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore


_PROFILE = TemperatureProfile(
    (44400, 1),
    (
        TemperatureTier((44400, 1), 0),
        TemperatureTier((44400, 2), 1),
    ),
)
_COLD_TIER = (44400, 2)
_DEPENDENCIES = (
    SegmentDependency(
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY, (44401, 1), (44402, 1)),
    SegmentDependency(
        MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY, (44401, 2), (44402, 2)),
)
_STRATEGY_ORDINAL = {
    strategy: ordinal
    for ordinal, strategy in enumerate(MD_BASELINE_KEYS, start=1)
}
_ABLATION_ORDINAL = {
    None: 1,
    **{
        ablation: ordinal
        for ordinal, ablation in enumerate(MD04_ABLATION_KEYS, start=2)
    },
}
_CHANNEL_ORDINAL = {
    "L0_ORIGIN": 1,
    "L1_WORK_MEMORY": 2,
    "L2_EPISODE_DOCUMENT": 3,
    "L3_MEMORY_OVERLAY": 4,
    "L4_SEALED_PAGE": 5,
    "SPECIAL_TYPED_INDEX": 6,
}


def _key(*values: int) -> StableRecordKey:
    return StableRecordKey(tuple(values))


class _NeverPrefetch:
    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        if not isinstance(context, QueryPrefetchContext):
            raise TypeError("MD-04 prefetch context 类型错误")
        return False

    def state_key(self) -> tuple[int, ...]:
        return (44410, 1)


class _CountingRepository(InMemoryObjectRepository):
    """只计数 probe query 的真实 sealed segment 读取。"""

    def __init__(self) -> None:
        super().__init__()
        self.segment_reads = 0

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        if object_kind == OBJECT_KIND_SEGMENT:
            self.segment_reads += 1
        return super().get(object_kind, identity_key)


@dataclass(frozen=True)
class _ColdRead:
    candidates: tuple[ProbeMemoryCandidate, ...]
    metrics: QueryHotSetMetrics
    segment_reads: int
    range_lower: StableRecordKey
    range_upper: StableRecordKey
    physical_read_key: StableRecordKey
    budget_exhausted: int
    reader_epoch_leaks: int
    descriptor_unchanged_after_time_advance: int


def _candidate_record(candidate: ProbeMemoryCandidate) -> SegmentRecord:
    payload = canonical_json_line(candidate.to_dict())
    return SegmentRecord(
        candidate.candidate_key.stable_key(),
        (FORMAT_VERSION, len(payload), *(value + 1 for value in payload)),
    )


def _candidate_from_record(record: SegmentRecord) -> ProbeMemoryCandidate:
    if (len(record.payload) < 2 or record.payload[0] != FORMAT_VERSION
            or record.payload[1] != len(record.payload) - 2
            or any(value <= 0 or value > 256 for value in record.payload[2:])):
        raise MD04ProbeContractError("MD-04 cold candidate payload 损坏")
    payload = bytes(value - 1 for value in record.payload[2:])
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if not payload.endswith(b"\n") or not isinstance(value, dict):
        raise MD04ProbeContractError("MD-04 cold candidate canonical 损坏")
    candidate = ProbeMemoryCandidate.from_dict(value)
    if candidate.candidate_key.stable_key() != record.record_key:
        raise MD04ProbeContractError("MD-04 cold record/candidate key 漂移")
    if _candidate_record(candidate) != record:
        raise MD04ProbeContractError("MD-04 cold candidate 非规范 round-trip")
    return candidate


def _publish_delta(
        store: TieredSegmentStore,
        records: tuple[SegmentRecord, ...],
        *,
        case_ordinal: int,
        segment_ordinal: int,
        scale_factor: int,
        ) -> None:
    delta = OpenHotDelta(
        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        (44420, case_ordinal, scale_factor),
        _DEPENDENCIES,
        SegmentBudget(max(1, len(records)), 8_000_000),
    )
    for record in records:
        delta.append(record)
    store.publish_delta(
        delta,
        segment_key=(44430, case_ordinal, scale_factor, segment_ordinal),
        tier_key=_COLD_TIER,
        read_fence=case_ordinal,
        manifest_key=(44440, case_ordinal, scale_factor, segment_ordinal),
        migration_key=(44450, case_ordinal, scale_factor, segment_ordinal),
    )


def _store_for_case(
        case: ProbeCaseDefinition,
        scale_factor: int,
        ) -> tuple[TieredSegmentStore, _CountingRepository]:
    repository = _CountingRepository()
    store = TieredSegmentStore(
        repository, build_storage_role_registry(), _PROFILE)
    case_ordinal = case.case_key.components[-1]
    if case.cold_candidates:
        _publish_delta(
            store,
            tuple(_candidate_record(item) for item in case.cold_candidates),
            case_ordinal=case_ordinal,
            segment_ordinal=1,
            scale_factor=scale_factor,
        )
    unrelated = tuple(
        SegmentRecord(
            (44900, scale_factor, ordinal),
            (FORMAT_VERSION, scale_factor, ordinal),
        )
        for ordinal in range(1, scale_factor + 1)
    )
    _publish_delta(
        store,
        unrelated,
        case_ordinal=case_ordinal,
        segment_ordinal=2,
        scale_factor=scale_factor,
    )
    repository.segment_reads = 0
    return store, repository


def _zero_metrics() -> QueryHotSetMetrics:
    return QueryHotSetMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def _read_cold(
        case: ProbeCaseDefinition,
        *,
        scale_factor: int,
        broaden_range: bool,
        ) -> _ColdRead:
    store, repository = _store_for_case(case, scale_factor)
    before = store.descriptor_state_key(
        (MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,))
    lower = _key(1) if broaden_range else case.cold_range_lower_key
    upper = _key(99999, 99999, 99999) if broaden_range else (
        case.cold_range_upper_key)
    ceilings = case.resource_ceiling.to_value()
    page_objects = max(1, min(4, ceilings["max_scanned_objects"]))
    hot_set = QuerySegmentHotSet(
        store,
        reader_key=(44460, case.case_key.components[-1], scale_factor,
                    2 if broaden_range else 1),
        descriptor_key=MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        policy=QueryHotSetPolicy(
            SegmentBudget(4, 4_000_000),
            SegmentBudget(page_objects, 2_000_000),
            _NeverPrefetch(),
            8,
        ),
    )
    records: list[SegmentRecord] = []
    iterator = hot_set.iter_range(
        lower_key=lower.stable_key(), upper_key=upper.stable_key())
    exhausted = 0
    try:
        for item in iterator:
            if len(records) >= ceilings["max_scanned_objects"]:
                exhausted = 1
                break
            records.append(item.record)
            if len(records) >= ceilings["max_scanned_objects"]:
                if (broaden_range or len(case.cold_candidates)
                        > len(records)):
                    exhausted = 1
                break
    finally:
        close_iterator = getattr(iterator, "close", None)
        if callable(close_iterator):
            close_iterator()
        hot_set.close()
    metrics = hot_set.metrics()
    decoded = []
    known = {item.candidate_key: item for item in case.cold_candidates}
    for record in records:
        key = StableRecordKey(record.record_key)
        if key not in known:
            continue
        candidate = _candidate_from_record(record)
        if candidate != known[key]:
            raise MD04ProbeContractError("MD-04 cold candidate 与冻结 plan 漂移")
        decoded.append(candidate)
    after = store.descriptor_state_key(
        (MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,))
    physical = _key(
        44470,
        case.case_key.components[-1],
        scale_factor,
        2 if broaden_range else 1,
        store.current_manifest().publish_epoch,
    )
    return _ColdRead(
        tuple(sorted(decoded)),
        metrics,
        repository.segment_reads,
        lower,
        upper,
        physical,
        exhausted,
        len(store.reader_epochs.snapshot()),
        int(before == after),
    )


def _candidate_filter(
        center: DirectionalMemoryCenter,
        candidates: tuple[ProbeMemoryCandidate, ...],
        *,
        allowed_relation_keys: tuple[StableRecordKey, ...],
        strategy: str,
        ablation: str | None,
        ) -> tuple[
            tuple[ProbeMemoryCandidate, ...], dict[str, int],
            tuple[ProbeMemoryCandidate, ...], tuple[ProbeMemoryCandidate, ...]]:
    eligible: list[ProbeMemoryCandidate] = []
    access_blocked: list[ProbeMemoryCandidate] = []
    grounding_blocked: list[ProbeMemoryCandidate] = []
    reasons: dict[str, int] = {}

    def reject(key: str) -> None:
        reasons[key] = reasons.get(key, 0) + 1

    typed = strategy in {
        "TYPED_FIXED_RING", "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"}
    primary = strategy == "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"
    for candidate in candidates:
        if typed and ablation != "TYPED_CENTER":
            if candidate.target_key != center.center.target_key:
                reject("TARGET_MISMATCH")
                continue
            if candidate.relation_key not in allowed_relation_keys:
                reject("RELATION_NOT_ALLOWED")
                continue
        if primary and ablation != "TYPED_CENTER" and (
                candidate.structure_key not in center.adoption_condition_keys):
            reject("STRUCTURE_MISMATCH")
            continue
        if primary:
            if not candidate.access_allowed:
                access_blocked.append(candidate)
                reject("ACCESS_DENIED")
                continue
            if (not candidate.grounded
                    or (not candidate.authorized
                        and ablation != "LAYERED_ATTRIBUTION")):
                grounding_blocked.append(candidate)
                reject("GROUNDING_OR_AUTHORIZATION_MISSING")
                continue
        eligible.append(candidate)
    return (
        tuple(sorted(eligible)),
        dict(sorted(reasons.items())),
        tuple(sorted(access_blocked)),
        tuple(sorted(grounding_blocked)),
    )


def _select_one(
        candidates: tuple[ProbeMemoryCandidate, ...],
        strategy: str,
        ) -> ProbeMemoryCandidate | None:
    if not candidates:
        return None
    if strategy == "RECENCY_HOT_ONLY":
        return max(candidates, key=lambda item: (
            item.recency_rank, item.activation, item.candidate_key))
    return max(candidates, key=lambda item: (
        item.activation, item.recency_rank, item.candidate_key))


def _decision(
        case: ProbeCaseDefinition,
        center: DirectionalMemoryCenter,
        strategy: str,
        eligible: tuple[ProbeMemoryCandidate, ...],
        access_blocked: tuple[ProbeMemoryCandidate, ...],
        grounding_blocked: tuple[ProbeMemoryCandidate, ...],
        *,
        budget_exhausted: int,
        ablation: str | None,
        ) -> tuple[
            MemoryDynamicsStopDecision,
            tuple[StableRecordKey, ...], tuple[StableRecordKey, ...]]:
    case_ordinal = case.case_key.components[-1]
    center_ordinal = center.center.center_key.components[-1]
    decision_key = _key(
        44500, _STRATEGY_ORDINAL[strategy], case_ordinal, center_ordinal,
        _ABLATION_ORDINAL[ablation])
    selected: ProbeMemoryCandidate | None = None
    status = "UNKNOWN"
    conflicts: tuple[StableRecordKey, ...] = ()
    blocking: tuple[StableRecordKey, ...] = ()
    remaining: tuple[str, ...] = ()
    exhausted = 0

    if strategy != "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP":
        selected = _select_one(eligible, strategy)
        if selected is not None:
            status = "RESOLVED"
    else:
        supports = tuple(item for item in eligible if item.stance == "SUPPORT")
        refutes = tuple(item for item in eligible if item.stance == "REFUTE")
        if supports and refutes:
            status = "CLARIFY"
            conflicts = tuple(sorted((
                supports[0].candidate_key, refutes[0].candidate_key)))
        elif supports:
            status = "RESOLVED"
            selected = _select_one(supports, strategy)
        elif access_blocked:
            status = "ACCESS_BLOCKED"
            blocking = tuple(item.candidate_key for item in access_blocked)
        elif grounding_blocked:
            status = "GROUNDING_BLOCKED"
            blocking = tuple(item.candidate_key for item in grounding_blocked)
        elif budget_exhausted:
            status = "BUDGET_EXHAUSTED"
            remaining = ("L4_SEALED_PAGE",)
            exhausted = 1

    selected_keys = () if selected is None else (selected.candidate_key,)
    generated = (
        selected_keys if center.center.direction == "GENERATION" else ())
    adopted = (
        selected_keys if center.center.direction != "GENERATION" else ())
    if status == "RESOLVED":
        assert selected is not None
        satisfied = (center.center.target_key,)
        unresolved = ()
        hard_checks = (_key(44510, case_ordinal, center_ordinal),)
        authorization = (selected.evidence_key,)
    else:
        satisfied = ()
        unresolved = (center.center.target_key,)
        hard_checks = ()
        authorization = ()
    decision = MemoryDynamicsStopDecision(
        decision_key,
        center.center.center_key,
        center.center.boundary,
        status,
        satisfied,
        unresolved,
        conflicts,
        hard_checks,
        authorization,
        blocking,
        remaining,
        None,
        exhausted,
        (f"{strategy}_{status}",),
        0,
    )
    return decision, adopted, generated


def _receipt_for_channel(
        case: ProbeCaseDefinition,
        center: DirectionalMemoryCenter,
        strategy: str,
        candidates: tuple[ProbeMemoryCandidate, ...],
        *,
        channel: str,
        decision: MemoryDynamicsStopDecision,
        is_last: bool,
        physical_read_key: StableRecordKey,
        range_lower: StableRecordKey,
        range_upper: StableRecordKey,
        page_reads: int,
        cold_bytes: int,
        recomputes: int,
        ablation: str | None,
        receipt_ordinal: int,
        scale_factor: int,
        ) -> MemoryRingReceipt:
    eligible, reasons, _, _ = _candidate_filter(
        center, candidates,
        allowed_relation_keys=case.allowed_relation_keys,
        strategy=strategy, ablation=ablation)
    selected = _select_one(eligible, strategy)
    agenda = () if selected is None else (selected.candidate_key,)
    evidence = () if selected is None else (selected.evidence_key,)
    dependencies = set(center.center.dependency_keys)
    for candidate in eligible:
        dependencies.update(candidate.dependency_keys)
        dependencies.add(candidate.source_key)
    case_ordinal = case.case_key.components[-1]
    center_ordinal = center.center.center_key.components[-1]
    receipt_key = _key(
        44520, _STRATEGY_ORDINAL[strategy], case_ordinal, center_ordinal,
        _ABLATION_ORDINAL[ablation], scale_factor, receipt_ordinal)
    return MemoryRingReceipt(
        receipt_key,
        center.center.center_key,
        center.center.boundary,
        channel,
        physical_read_key,
        receipt_ordinal - 1,
        receipt_ordinal,
        (center.center.target_key,),
        case.allowed_relation_keys,
        tuple(sorted({range_lower, range_upper})),
        len(candidates),
        len(candidates) - sum(reasons.values()),
        sum(reasons.values()),
        CanonicalJsonObject.from_value(reasons),
        agenda,
        (() if not agenda else (
            "ACTIVATION", "RECENCY", "TYPED_OBLIGATION")),
        agenda,
        evidence,
        tuple(sorted(dependencies)),
        page_reads,
        recomputes,
        cold_bytes,
        "STOP" if is_last else "CONTINUE",
        decision.decision_key if is_last else None,
        0,
    )


def _run_case(
        bundle: MD04FixtureBundle,
        case: ProbeCaseDefinition,
        *,
        strategy: str,
        scale_factor: int,
        ablation: str | None = None,
        ) -> ProbeCaseOutcome:
    centers = bundle.centers_for(case.case_key)
    if tuple(center.center.center_key for center in centers) != tuple(
            item.center_key for item in case.center_refs):
        raise MD04ProbeContractError("runtime center binding 漂移")
    before_candidates = canonical_json_line({
        "cold": [item.to_dict() for item in case.cold_candidates],
        "hot": [item.to_dict() for item in case.hot_candidates],
    })

    hot_by_channel = {
        channel: tuple(item for item in case.hot_candidates
                       if item.channel_key == channel)
        for channel in ("L1_WORK_MEMORY", "SPECIAL_TYPED_INDEX")
    }
    primary = strategy == "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"
    hot_can_resolve = False
    if primary:
        for center in centers:
            eligible, _, _, _ = _candidate_filter(
                center,
                tuple(item for values in hot_by_channel.values()
                      for item in values),
                allowed_relation_keys=case.allowed_relation_keys,
                strategy=strategy,
                ablation=ablation,
            )
            if any(item.stance == "SUPPORT" for item in eligible):
                hot_can_resolve = True
    need_cold = False
    if strategy == "TYPED_FIXED_RING":
        need_cold = bool(case.cold_channel_admitted)
    elif primary:
        need_cold = bool(
            case.cold_channel_admitted
            and (case.conflict_scan_required or not hot_can_resolve))
    broaden = ablation == "TYPED_CHANNEL_SELECTION"
    if broaden:
        need_cold = True

    if need_cold:
        cold = _read_cold(
            case, scale_factor=scale_factor, broaden_range=broaden)
    else:
        cold = _ColdRead(
            (), _zero_metrics(), 0,
            case.cold_range_lower_key, case.cold_range_upper_key,
            _key(44470, case.case_key.components[-1], scale_factor, 1, 1),
            0, 0, 1,
        )

    decisions: list[MemoryDynamicsStopDecision] = []
    receipts: list[MemoryRingReceipt] = []
    adopted: set[StableRecordKey] = set()
    generated: set[StableRecordKey] = set()
    recomputes = int(case.local_revision_dependency_key is not None)
    unrelated_changes = 0
    unaffected = 1
    if (ablation == "DEPENDENCY_INVALIDATION"
            and case.local_revision_dependency_key is not None):
        unrelated_changes = 1
        unaffected = 0

    for center in centers:
        available_hot = tuple(
            item for values in hot_by_channel.values() for item in values)
        pool = tuple(sorted((*available_hot, *cold.candidates)))
        if strategy in {"FIXED_TOP_K", "RECENCY_HOT_ONLY"}:
            pool = available_hot
        eligible, _, access_blocked, grounding_blocked = _candidate_filter(
            center, pool,
            allowed_relation_keys=case.allowed_relation_keys,
            strategy=strategy, ablation=ablation)
        decision, center_adopted, center_generated = _decision(
            case, center, strategy, eligible, access_blocked,
            grounding_blocked,
            budget_exhausted=cold.budget_exhausted,
            ablation=ablation,
        )
        decisions.append(decision)
        adopted.update(center_adopted)
        generated.update(center_generated)

        channel_rows: list[tuple[
            str, tuple[ProbeMemoryCandidate, ...], StableRecordKey,
            StableRecordKey, StableRecordKey, int, int]] = []
        for channel in ("L1_WORK_MEMORY", "SPECIAL_TYPED_INDEX"):
            values = hot_by_channel[channel]
            if values or (channel == "L1_WORK_MEMORY" and not channel_rows):
                physical = _key(
                    44480, case.case_key.components[-1],
                    _CHANNEL_ORDINAL[channel], scale_factor)
                channel_rows.append((
                    channel, values, physical,
                    case.case_key, case.case_key, 0, 0))
        if need_cold:
            channel_rows.append((
                "L4_SEALED_PAGE", cold.candidates, cold.physical_read_key,
                cold.range_lower, cold.range_upper,
                cold.metrics.page_faults, cold.metrics.cold_read_bytes))
        if ablation == "STOP_DECISION":
            channel_rows.append((
                "L3_MEMORY_OVERLAY", (),
                _key(44490, case.case_key.components[-1], scale_factor),
                case.case_key, case.case_key, 0, 0))
        for ordinal, row in enumerate(channel_rows, start=1):
            channel, values, physical, lower, upper, pages, cold_bytes = row
            receipts.append(_receipt_for_channel(
                case, center, strategy, values,
                channel=channel,
                decision=decision,
                is_last=ordinal == len(channel_rows),
                physical_read_key=physical,
                range_lower=lower,
                range_upper=upper,
                page_reads=pages,
                cold_bytes=cold_bytes,
                recomputes=(recomputes if ordinal == len(channel_rows) else 0),
                ablation=ablation,
                receipt_ordinal=ordinal,
                scale_factor=scale_factor,
            ))

    after_candidates = canonical_json_line({
        "cold": [item.to_dict() for item in case.cold_candidates],
        "hot": [item.to_dict() for item in case.hot_candidates],
    })
    metrics = cold.metrics
    query_metrics = CanonicalJsonObject.from_value({
        "cache_hits": metrics.cache_hits,
        "clean_evictions": metrics.clean_evictions,
        "cold_read_bytes": metrics.cold_read_bytes,
        "dirty_flushes": metrics.dirty_flushes,
        "omitted_fault_reports": metrics.omitted_fault_reports,
        "page_faults": metrics.page_faults,
        "page_in_records": metrics.page_in_records,
        "peak_hot_bytes": metrics.peak_hot_bytes,
        "peak_hot_objects": metrics.peak_hot_objects,
        "prefetched_pages": metrics.prefetched_pages,
        "released_pins": metrics.released_pins,
        "segment_reads": cold.segment_reads,
    })
    audit = CanonicalJsonObject.from_value({
        "full_store_rewrite_count": (
            0 if cold.descriptor_unchanged_after_time_advance else 1),
        "held_out_train_overlap_count": 0,
        "host_learning_write_count": 0,
        "old_evidence_preserved": int(before_candidates == after_candidates),
        "reader_epoch_leak_count": cold.reader_epoch_leaks,
        "teacher_call_count": 0,
        "unaffected_projection_bit_identical": unaffected,
        "unrelated_revision_change_count": unrelated_changes,
    })
    return ProbeCaseOutcome(
        case.case_key,
        case.sample_group_key,
        strategy,
        scale_factor,
        tuple(sorted(receipts, key=lambda item: item.receipt_key)),
        tuple(sorted(decisions, key=lambda item: item.decision_key)),
        tuple(sorted(adopted)),
        tuple(sorted(generated)),
        query_metrics,
        audit,
    )


def run_md04_probe(
        bundle: MD04FixtureBundle,
        *,
        plan_relative_path: str,
        plan_sha256: str,
        ) -> MD04ProbeRunArtifact:
    """在 plan 已冻结后执行四基线、1/10/100 和五项消融。"""
    if not isinstance(bundle, MD04FixtureBundle):
        raise MD04ProbeContractError("MD-04 bundle 类型错误")
    strategy_outcomes = tuple(sorted((
        _run_case(
            bundle, case, strategy=strategy, scale_factor=1)
        for strategy in MD_BASELINE_KEYS
        for case in bundle.plan.cases
    ), key=lambda item: (item.strategy_key, item.case_key)))
    scale_case_keys = {
        _key(44300, 4),
        _key(44300, 9),
    }
    scale_outcomes = tuple(sorted((
        _run_case(
            bundle, case,
            strategy="OBLIGATION_CONDITIONED_MULTICHANNEL_STOP",
            scale_factor=scale)
        for scale in MD04_SCALE_FACTORS
        for case in bundle.plan.cases
        if case.case_key in scale_case_keys
    ), key=lambda item: (item.scale_factor, item.case_key)))
    ablation_outcomes = tuple(
        ProbeAblationOutcome(
            ablation,
            tuple(sorted((
                _run_case(
                    bundle, case,
                    strategy="OBLIGATION_CONDITIONED_MULTICHANNEL_STOP",
                    scale_factor=1,
                    ablation=ablation)
                for case in bundle.plan.cases
            ), key=lambda item: item.case_key)),
        )
        for ablation in MD04_ABLATION_KEYS
    )
    return MD04ProbeRunArtifact(
        FORMAT_VERSION,
        MD04_RUN_VERSION,
        plan_relative_path,
        plan_sha256,
        strategy_outcomes,
        scale_outcomes,
        ablation_outcomes,
        1,
        0,
        zero_execution_state(),
    )


__all__ = ["run_md04_probe"]
