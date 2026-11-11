"""核验独立 support/conflict 课程的完整消费及可生成命题集合。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED, EPISTEMIC_SUPPORTED, LIFECYCLE_ACTIVE,
)


def verify_relation_training_integrity(learning, course) -> dict[str, object]:
    """逐候选核对状态和投影，冲突必须保留但不得进入生成分母。

    此合同仅接受无 revision 的独立 support/conflict 课程；其他课程须使用
    自身的生命周期合同，不能在这里默认为全部 active 或忽略缺失候选。
    """
    records = {record.proposition_key: record for record in course.records}
    if (not records or len(records) != len(course.records)
            or any(record.sample_role_code not in {1, 4}
                   or record.supersedes_key is not None for record in course.records)):
        raise ValueError("training integrity requires distinct independent support/conflict records")
    candidates = {item.proposition.proposition.stable_key(): item
                  for item in learning.registered_candidates()}
    if set(candidates) != set(records):
        raise ValueError("Core candidate identities differ from the complete licensed course")
    active = {item.proposition.proposition.stable_key() for item in learning.active_candidates()}
    expected_active, rows = set(), []
    for key, record in sorted(records.items()):
        candidate = candidates[key]
        current = learning.snapshot_for(candidate.proposition.proposition)
        snapshot = current.snapshot
        supported = record.sample_role_code == 1
        expected_state = EPISTEMIC_SUPPORTED if supported else EPISTEMIC_CONFLICTED
        definition = candidate.proposition
        bindings = tuple(sorted((binding.role.stable_key(), binding.filler.stable_key(), binding.ordinal)
                                for binding in definition.bindings))
        source_bindings = tuple(sorted((tuple(row["role_key"]), tuple(row["filler_key"]), row["ordinal"])
                                      for value in record.role_bindings for row in (value.to_value(),)))
        if (definition.predicate.stable_key() != record.predicate_key or bindings != source_bindings
                or snapshot.lifecycle != LIFECYCLE_ACTIVE or snapshot.epistemic_status != expected_state
                or not snapshot.support_evidence_ids
                or bool(snapshot.refute_evidence_ids) == supported
                or (current.active_fact is not None) != supported or (key in active) != supported):
            raise ValueError("Core candidate definition, Evidence, lifecycle or projection differs from its course")
        if supported:
            expected_active.add(key)
        rows.append({"proposition_key": list(key), "sample_role_code": record.sample_role_code,
                     "lifecycle": snapshot.lifecycle, "epistemic_status": snapshot.epistemic_status,
                     "support_count": len(snapshot.support_evidence_ids),
                     "refute_count": len(snapshot.refute_evidence_ids), "active_projection": int(supported)})
    report = learning.report()
    if (report.candidate_count != len(records) or report.schema_rejection_count
            or report.active_candidate_count != len(expected_active)
            or report.conflict_candidate_count != len(records) - len(expected_active)
            or active != expected_active):
        raise ValueError("Core training report disagrees with the verified complete candidate set")
    return {"status": "PASS_COMPLETE_CANDIDATES_AND_EXPECTED_STATES",
            "candidate_count": len(records), "active_candidate_count": len(expected_active),
            "conflict_candidate_count": len(records) - len(expected_active),
            "active_proposition_keys": [list(key) for key in sorted(expected_active)], "records": rows}
