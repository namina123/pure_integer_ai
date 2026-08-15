"""Derive recovery-v8 family-LOSO learner outputs from the sealed protocol."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_learning_contract import (
    NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


V8_LOSO_VIEW_KIND = "NORMALIZATION_RECOVERY_V8_LOSO_TRAINING_VIEW_V1"
V8_LEARNED_RULE_KIND = "NORMALIZATION_RECOVERY_V8_LEARNED_RULE_V1"
V8_IDENTITY_VETO_RULE_KIND = "NORMALIZATION_RECOVERY_V8_IDENTITY_VETO_RULE_V1"
V8_LEARNING_WORK_KIND = "NORMALIZATION_RECOVERY_V8_LEARNING_WORK_ITEM_V1"

_AUTHORIZED_FILES = (
    "authorized-orthographic-atoms.jsonl",
    "authorized-source-conditioned-lexical-atoms.jsonl",
    "authorized-layout-morphology-obligations.jsonl",
)
_EXACT_CONTROL_FILE = "exact-input-control.jsonl"
_LOSO_FILE = "family-loso-plan.jsonl"
_RULE_FILE_BY_LEDGER = {
    "ORTHOGRAPHIC_ATOM": "orthographic-rules.jsonl",
    "SOURCE_CONDITIONED_LEXICAL_ATOM": (
        "source-conditioned-lexical-rules.jsonl"),
    "LAYOUT_MORPHOLOGY_OBLIGATION": (
        "layout-morphology-obligations.jsonl"),
}


def _sha(value: object) -> str:
    """Return a deterministic SHA-256 identity."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _families(value: object, *, label: str) -> list[str]:
    """Read a sorted unique family list within the frozen three-family set."""
    if (not isinstance(value, list)
            or value != sorted(set(value))
            or any(family not in V8_TRAIN_FAMILIES for family in value)):
        raise BroadQaExternalDataError(f"v8 learner {label} 漂移")
    return value


def _authorization_index(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, dict[str, object]]:
    """Index every authorized record and reject duplicate identities."""
    values = tuple(item for name in _AUTHORIZED_FILES
                   for item in protocol_outputs.get(name, ()))
    result = {}
    for item in values:
        identity = item.get("authorization_id")
        ledger = item.get("ledger_kind")
        if (not isinstance(identity, str) or len(identity) != 64
                or ledger not in _RULE_FILE_BY_LEDGER
                or item.get("execution_allowed") != 0
                or item.get("learned_rule_claimed") != 0
                or identity in result):
            raise BroadQaExternalDataError(
                "v8 learner authorization inventory 漂移")
        _families(item.get("support_families"), label="authorization support")
        result[identity] = item
    if not result:
        raise BroadQaExternalDataError("v8 learner authorization 为空")
    return result


def _plan_index(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        authorizations: dict[str, dict[str, object]],
        ) -> dict[str, tuple[dict[str, object], ...]]:
    """Validate and group exactly three held-out directions per authorization."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for plan in protocol_outputs.get(_LOSO_FILE, ()):
        authorization_id = plan.get("authorization_id")
        held_out = plan.get("held_out_family")
        if (authorization_id not in authorizations
                or held_out not in V8_TRAIN_FAMILIES
                or plan.get("held_out_output_read_count") != 0
                or plan.get("held_out_output_may_influence_rule_construction") != 0):
            raise BroadQaExternalDataError("v8 learner LOSO plan 漂移")
        authorization = authorizations[str(authorization_id)]
        before = _families(
            authorization.get("support_families"), label="LOSO before")
        after = [family for family in before if family != held_out]
        expected_present = int(len(after) >= 2)
        held_out_supports = int(held_out in before)
        expected_behavior = (
            "EXACT" if held_out_supports and expected_present else "UNKNOWN")
        if (plan.get("support_families_before_holdout") != before
                or plan.get("support_families_after_holdout") != after
                or plan.get("expected_rule_present_after_holdout")
                != expected_present
                or plan.get("evaluation_case_expected") != held_out_supports
                or plan.get("expected_behavior") != expected_behavior
                or plan.get("ledger_kind") != authorization["ledger_kind"]):
            raise BroadQaExternalDataError(
                "v8 learner LOSO plan authority 漂移")
        grouped[str(authorization_id)].append(plan)
    result = {}
    for authorization_id, plans in grouped.items():
        ordered = tuple(sorted(plans, key=lambda item: str(item["held_out_family"])))
        if ([item["held_out_family"] for item in ordered]
                != list(V8_TRAIN_FAMILIES)):
            raise BroadQaExternalDataError(
                "v8 learner LOSO direction inventory 漂移")
        result[authorization_id] = ordered
    if set(result) != set(authorizations):
        raise BroadQaExternalDataError("v8 learner LOSO authorization 覆盖漂移")
    return result


def _learned_rule(
        authorization: dict[str, object], plan: dict[str, object],
        ) -> dict[str, object]:
    """Project one surviving direction into a disabled learned rule."""
    ledger = str(authorization["ledger_kind"])
    held_out = str(plan["held_out_family"])
    identity = {
        "authorization_id": authorization["authorization_id"],
        "held_out_family": held_out,
        "ledger_kind": ledger,
    }
    base = {
        "authorization_id": authorization["authorization_id"],
        "candidate_id": authorization["candidate_id"],
        "format_version": 1,
        "held_out_family": held_out,
        "ledger_kind": ledger,
        "production_enabled": 0,
        "record_kind": V8_LEARNED_RULE_KIND,
        "rule_id": _sha(identity),
        "train_support_families": plan["support_families_after_holdout"],
        "train_support_family_count": len(
            plan["support_families_after_holdout"]),
    }
    if ledger == "ORTHOGRAPHIC_ATOM":
        return {
            **base,
            "input_atom": authorization["input_atom"],
            "output_atom": authorization["output_atom"],
        }
    if ledger == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        return {
            **base,
            "input_text": authorization["input_text"],
            "official_source_condition_required": 1,
            "official_source_text": authorization["official_source_text"],
            "output_text": authorization["output_text"],
            "unconditioned_execution_allowed": 0,
        }
    if ledger == "LAYOUT_MORPHOLOGY_OBLIGATION":
        return {
            **base,
            "per_observation_rewrite_allowed": 0,
            "structure_preservation_hard_gate": 1,
            "structure_tokens": authorization["structure_tokens"],
        }
    raise BroadQaExternalDataError("v8 learner rule ledger 非法")


def _work_item(
        *, record_id: str, held_out_family: str, work_kind: str,
        rule_increment: int,
        ) -> dict[str, object]:
    """Build one ordered learning work item and its expected emission bit."""
    identity = {
        "held_out_family": held_out_family,
        "record_id": record_id,
        "work_kind": work_kind,
    }
    return {
        **identity,
        "evidence_increment": 1,
        "format_version": 1,
        "record_kind": V8_LEARNING_WORK_KIND,
        "rule_increment": rule_increment,
        "work_id": _sha(identity),
        "work_ordinal": -1,
    }


def _identity_veto_material(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Project multi-family identity controls into direction-scoped vetoes."""
    rules = []
    work = []
    for item in protocol_outputs.get(_EXACT_CONTROL_FILE, ()):
        outputs = item.get("outputs")
        families = _families(item.get("support_families"), label="identity support")
        if (item.get("candidate_status") != "MULTI_FAMILY_UNIQUE_OUTPUT"
                or not isinstance(outputs, list) or len(outputs) != 1
                or not isinstance(outputs[0], dict)
                or outputs[0].get("output_text") != item.get("input_text")):
            continue
        candidate_id = item.get("candidate_id")
        input_text = item.get("input_text")
        if (not isinstance(candidate_id, str) or len(candidate_id) != 64
                or not isinstance(input_text, str)):
            raise BroadQaExternalDataError("v8 learner identity control 漂移")
        for held_out in V8_TRAIN_FAMILIES:
            after = [family for family in families if family != held_out]
            present = int(len(after) >= 2)
            work.append(_work_item(
                record_id=candidate_id,
                held_out_family=held_out,
                work_kind="IDENTITY_VETO_DIRECTION",
                rule_increment=present,
            ))
            if not present:
                continue
            identity = {
                "candidate_id": candidate_id,
                "held_out_family": held_out,
                "rule_kind": "IDENTITY_VETO",
            }
            rules.append({
                "candidate_id": candidate_id,
                "format_version": 1,
                "held_out_family": held_out,
                "input_text": input_text,
                "output_text": input_text,
                "production_enabled": 0,
                "record_kind": V8_IDENTITY_VETO_RULE_KIND,
                "rule_id": _sha(identity),
                "rule_kind": "IDENTITY_VETO",
                "train_support_families": after,
                "train_support_family_count": len(after),
            })
    return tuple(rules), tuple(work)


def _views(
        rule_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """Summarize the executable inventory of each held-out direction."""
    values = []
    for held_out in V8_TRAIN_FAMILIES:
        counts = {
            name: sum(item["held_out_family"] == held_out for item in records)
            for name, records in rule_outputs.items()
        }
        identity = {"held_out_family": held_out, "rule_counts": counts}
        values.append({
            "format_version": 1,
            "held_out_family": held_out,
            "minimum_distinct_train_family_count": 2,
            "production_enabled": 0,
            "record_kind": V8_LOSO_VIEW_KIND,
            "rule_counts": counts,
            "total_rule_count": sum(counts.values()),
            "view_id": _sha(identity),
        })
    return tuple(values)


def _summary(
        outputs: dict[str, tuple[dict[str, object], ...]],
        work: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """Summarize direction-scoped rule and work emission counts."""
    rules = tuple(item for name, _role, _identity in
                  NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES[1:]
                  for item in outputs[name])
    return {
        "evidence_count": len(work),
        "held_out_family_rule_counts": {
            family: sum(item["held_out_family"] == family for item in rules)
            for family in V8_TRAIN_FAMILIES
        },
        "identity_veto_rule_count": len(outputs["identity-veto-rules.jsonl"]),
        "learned_rule_count": len(rules),
        "learner_work_item_count": len(work),
        "loso_view_count": len(outputs["loso-training-views.jsonl"]),
        "production_enabled": 0,
        "result_record_count": len(rules),
        "rule_ledger_counts": dict(sorted(Counter(
            str(item.get("ledger_kind", item.get("rule_kind")))
            for item in rules).items())),
        "rule_support_family_counts": dict(sorted(Counter(
            str(item["train_support_family_count"])
            for item in rules).items())),
        "work_rule_emission_count": sum(
            int(item["rule_increment"]) for item in work),
    }


def derive_normalization_recovery_v8_learning_outputs(
        *, protocol_manifest: dict[str, object],
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object],
            tuple[dict[str, object], ...], tuple[dict[str, int], ...]]:
    """Derive all direction-scoped rule outputs and ordered learner work."""
    if (protocol_manifest.get("status")
            != "THREE_LEDGER_PROTOCOL_FROZEN_NOT_TRAINED"
            or protocol_manifest.get("training_executed") != 0
            or protocol_manifest.get("vlc_final_read_count") != 0):
        raise BroadQaExternalDataError("v8 learner protocol state 非法")
    authorizations = _authorization_index(protocol_outputs)
    plans = _plan_index(protocol_outputs, authorizations)
    rule_lists: dict[str, list[dict[str, object]]] = {
        name: [] for name in _RULE_FILE_BY_LEDGER.values()}
    work = []
    for authorization_id in sorted(authorizations):
        authorization = authorizations[authorization_id]
        for plan in plans[authorization_id]:
            present = int(plan["expected_rule_present_after_holdout"])
            work.append(_work_item(
                record_id=str(plan["loso_plan_id"]),
                held_out_family=str(plan["held_out_family"]),
                work_kind="AUTHORIZED_CANDIDATE_DIRECTION",
                rule_increment=present,
            ))
            if present:
                name = _RULE_FILE_BY_LEDGER[str(authorization["ledger_kind"])]
                rule_lists[name].append(_learned_rule(authorization, plan))
    identity_rules, identity_work = _identity_veto_material(protocol_outputs)
    work.extend(identity_work)
    rule_outputs = {
        name: tuple(sorted(values, key=lambda item: str(item["rule_id"])))
        for name, values in rule_lists.items()
    }
    rule_outputs["identity-veto-rules.jsonl"] = tuple(sorted(
        identity_rules, key=lambda item: str(item["rule_id"])))
    outputs = {
        "loso-training-views.jsonl": _views(rule_outputs),
        **rule_outputs,
    }
    ordered_work = tuple({**item, "work_ordinal": ordinal}
                         for ordinal, item in enumerate(sorted(work, key=lambda item: (
                             str(item["work_kind"]), str(item["record_id"]),
                             str(item["held_out_family"])))))
    if (not ordered_work or len({item["work_id"] for item in ordered_work})
            != len(ordered_work)):
        raise BroadQaExternalDataError("v8 learner work identity 漂移")
    summary = _summary(outputs, ordered_work)
    if summary["learned_rule_count"] != summary["work_rule_emission_count"]:
        raise BroadQaExternalDataError("v8 learner rule emission 漂移")
    increments = tuple({
        "evidence_increment": int(item["evidence_increment"]),
        "rule_increment": int(item["rule_increment"]),
    } for item in ordered_work)
    return outputs, summary, ordered_work, increments


def normalization_recovery_v8_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """Encode all learner outputs as canonical JSONL payloads."""
    expected = {name for name, _role, _identity in
                NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES}
    if set(outputs) != expected:
        raise BroadQaExternalDataError("v8 learner output inventory 漂移")
    return {name: b"".join(canonical_json_line(item) for item in outputs[name])
            for name in expected}


def normalization_recovery_v8_prefix_counts(
        work: tuple[dict[str, object], ...], context: object,
        processed_item_count: int,
        ) -> tuple[int, int]:
    """Count evidence and emitted rules over one deterministic work prefix."""
    if (not isinstance(context, tuple) or len(context) != len(work)
            or type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError("v8 learner prefix context 漂移")
    prefix = context[:processed_item_count]
    return (
        sum(int(item["evidence_increment"]) for item in prefix),
        sum(int(item["rule_increment"]) for item in prefix),
    )


__all__ = [
    "derive_normalization_recovery_v8_learning_outputs",
    "normalization_recovery_v8_output_payloads",
    "normalization_recovery_v8_prefix_counts",
]
