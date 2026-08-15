"""Derive recovery-v8 three-ledger TRAIN protocol records.

The records are plain, tagged value structures.  They authorize candidates for
later training, but they are not executable rules and do not claim learning.
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


V8_TRAIN_FAMILIES = (
    "KEEPASSXC_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)

V8_AUTHORIZED_ORTHOGRAPHIC_KIND = (
    "NORMALIZATION_RECOVERY_V8_AUTHORIZED_ORTHOGRAPHIC_ATOM_V1")
V8_AUTHORIZED_LEXICAL_KIND = (
    "NORMALIZATION_RECOVERY_V8_AUTHORIZED_SOURCE_CONDITIONED_LEXICAL_ATOM_V1")
V8_AUTHORIZED_STRUCTURE_KIND = (
    "NORMALIZATION_RECOVERY_V8_AUTHORIZED_LAYOUT_MORPHOLOGY_OBLIGATION_V1")
V8_EXACT_CONTROL_KIND = (
    "NORMALIZATION_RECOVERY_V8_EXACT_INPUT_CONTROL_V1")
V8_DEFERRED_CANDIDATE_KIND = (
    "NORMALIZATION_RECOVERY_V8_DEFERRED_CANDIDATE_V1")
V8_DEFERRED_OBSERVATION_KIND = (
    "NORMALIZATION_RECOVERY_V8_DEFERRED_OBSERVATION_V1")
V8_LOSO_PLAN_KIND = "NORMALIZATION_RECOVERY_V8_FAMILY_LOSO_PLAN_V1"
V8_PROTOCOL_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_CENSUS_V1")

V8_TRAINING_RECORD_FILES = (
    ("authorized-orthographic-atoms.jsonl", "AUTHORIZED_ORTHOGRAPHIC_ATOMS"),
    ("authorized-source-conditioned-lexical-atoms.jsonl",
     "AUTHORIZED_SOURCE_CONDITIONED_LEXICAL_ATOMS"),
    ("authorized-layout-morphology-obligations.jsonl",
     "AUTHORIZED_LAYOUT_MORPHOLOGY_OBLIGATIONS"),
    ("exact-input-control.jsonl", "EXACT_INPUT_CONTROL_INDEX"),
    ("deferred-candidates.jsonl", "DEFERRED_CANDIDATES"),
    ("deferred-observations.jsonl", "DEFERRED_OBSERVATIONS"),
    ("family-loso-plan.jsonl", "THREE_DIRECTION_FAMILY_LOSO_PLAN"),
    ("protocol-census.jsonl", "TRAINING_PROTOCOL_CENSUS"),
)

_COVERAGE_FILES = (
    "exact-input-mappings.jsonl",
    "source-conditioned-mappings.jsonl",
    "orthographic-atoms.jsonl",
    "structure-obligations.jsonl",
    "coverage-census.jsonl",
)
_OBSERVATION_FILES = (
    "qbittorrent-observations.jsonl",
    "stellarium-observations.jsonl",
    "keepassxc-observations.jsonl",
)


def _sha256(value: dict[str, object]) -> str:
    """Return a deterministic identity for one tagged value structure."""
    return hashlib.sha256(canonical_json_line(value)).hexdigest()


def _string_list(value: object, *, label: str) -> list[str]:
    """Read a sorted, unique string list or fail closed."""
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or value != sorted(set(value))):
        raise BroadQaExternalDataError(f"v8 TRAIN {label} 漂移")
    return value


def _support(item: dict[str, object]) -> tuple[list[str], int]:
    """Read and validate candidate family support."""
    families = _string_list(item.get("support_families"), label="support families")
    if (any(family not in V8_TRAIN_FAMILIES for family in families)
            or item.get("support_family_count", len(families)) != len(families)):
        raise BroadQaExternalDataError("v8 TRAIN support family count 漂移")
    return families, len(families)


def _outputs(item: dict[str, object]) -> list[dict[str, object]]:
    """Read a nonempty mapping/atom output list."""
    outputs = item.get("outputs")
    if (not isinstance(outputs, list) or not outputs
            or any(not isinstance(output, dict) for output in outputs)):
        raise BroadQaExternalDataError("v8 TRAIN candidate outputs 漂移")
    return outputs


def _authorization_base(
        item: dict[str, object], *, ledger_kind: str, record_kind: str,
        ) -> dict[str, object]:
    """Build common authorization fields without making a learning claim."""
    families, count = _support(item)
    if count < 2:
        raise BroadQaExternalDataError("v8 TRAIN authorization family 不足")
    identity = {
        "candidate_id": item.get("candidate_id"),
        "ledger_kind": ledger_kind,
    }
    if (not isinstance(identity["candidate_id"], str)
            or len(identity["candidate_id"]) != 64):
        raise BroadQaExternalDataError("v8 TRAIN candidate id 漂移")
    return {
        "authorization_id": _sha256(identity),
        "authority_basis": "MULTI_FAMILY_UNIQUE_EVIDENCE",
        "candidate_id": identity["candidate_id"],
        "execution_allowed": 0,
        "format_version": 1,
        "learned_rule_claimed": 0,
        "ledger_kind": ledger_kind,
        "minimum_distinct_source_family_count": 2,
        "record_kind": record_kind,
        "support_families": families,
        "support_family_count": count,
    }


def _authorized_orthographic(
        item: dict[str, object],
        ) -> dict[str, object] | None:
    """Authorize only unique multi-family single-Han atom evidence."""
    if item.get("candidate_kind") != "ORTHOGRAPHIC_ATOM":
        raise BroadQaExternalDataError("v8 TRAIN orthographic kind 漂移")
    if item.get("candidate_status") != "MULTI_FAMILY_UNIQUE_OUTPUT":
        return None
    output = _outputs(item)
    if (len(output) != 1 or not isinstance(item.get("input_atom"), str)
            or not isinstance(output[0].get("output_atom"), str)
            or item["input_atom"] == output[0]["output_atom"]):
        raise BroadQaExternalDataError("v8 TRAIN orthographic unique output 漂移")
    return {
        **_authorization_base(
            item, ledger_kind="ORTHOGRAPHIC_ATOM",
            record_kind=V8_AUTHORIZED_ORTHOGRAPHIC_KIND),
        "family_record_counts": output[0].get("family_record_counts"),
        "input_atom": item["input_atom"],
        "output_atom": output[0]["output_atom"],
    }


def _authorized_lexical(
        item: dict[str, object],
        ) -> dict[str, object] | None:
    """Authorize only changed, source-conditioned multi-family evidence."""
    if item.get("candidate_kind") != "SOURCE_CONDITIONED_MAPPING":
        raise BroadQaExternalDataError("v8 TRAIN lexical kind 漂移")
    if item.get("candidate_status") != "MULTI_FAMILY_UNIQUE_OUTPUT":
        return None
    outputs = _outputs(item)
    input_text = item.get("input_text")
    output_text = outputs[0].get("output_text") if len(outputs) == 1 else None
    if (not isinstance(input_text, str) or not isinstance(output_text, str)
            or not isinstance(item.get("official_source_text"), str)):
        raise BroadQaExternalDataError("v8 TRAIN lexical unique output 漂移")
    if input_text == output_text:
        return None
    return {
        **_authorization_base(
            item, ledger_kind="SOURCE_CONDITIONED_LEXICAL_ATOM",
            record_kind=V8_AUTHORIZED_LEXICAL_KIND),
        "family_record_counts": outputs[0].get("family_record_counts"),
        "input_text": input_text,
        "official_source_condition_required": 1,
        "official_source_text": item["official_source_text"],
        "output_text": output_text,
        "unconditioned_execution_allowed": 0,
    }


def _authorized_structure(
        item: dict[str, object],
        ) -> dict[str, object] | None:
    """Authorize generic preservation obligations, never local rewrites."""
    if item.get("candidate_kind") != "STRUCTURE_OBLIGATION":
        raise BroadQaExternalDataError("v8 TRAIN structure kind 漂移")
    if item.get("candidate_status") != "MULTI_FAMILY_OBSERVED":
        return None
    tokens = _string_list_or_ordered(
        item.get("structure_tokens"), label="structure tokens")
    if not tokens:
        raise BroadQaExternalDataError("v8 TRAIN structure obligation 为空")
    return {
        **_authorization_base(
            item, ledger_kind="LAYOUT_MORPHOLOGY_OBLIGATION",
            record_kind=V8_AUTHORIZED_STRUCTURE_KIND),
        "family_record_counts": item.get("family_record_counts"),
        "per_observation_layout_morphology_defer_required": 1,
        "structure_preservation_hard_gate": 1,
        "structure_tokens": tokens,
    }


def _string_list_or_ordered(value: object, *, label: str) -> list[str]:
    """Read an ordered string list; repeated structure tokens are meaningful."""
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)):
        raise BroadQaExternalDataError(f"v8 TRAIN {label} 漂移")
    return value


def _exact_control(item: dict[str, object]) -> dict[str, object]:
    """Preserve exact mappings as non-authorizing control evidence."""
    if (item.get("candidate_kind") != "EXACT_INPUT_MAPPING"
            or not isinstance(item.get("input_text"), str)):
        raise BroadQaExternalDataError("v8 TRAIN exact control 漂移")
    outputs = _outputs(item)
    families, count = _support(item)
    return {
        "authorization_allowed": 0,
        "candidate_id": item.get("candidate_id"),
        "candidate_status": item.get("candidate_status"),
        "control_role": "EVIDENCE_AND_INTERPRETER_COMPARISON_ONLY",
        "format_version": 1,
        "input_text": item["input_text"],
        "outputs": outputs,
        "record_kind": V8_EXACT_CONTROL_KIND,
        "support_families": families,
        "support_family_count": count,
        "unconditioned_changed_rule_allowed": 0,
    }


def _mapping_is_identity_only(item: dict[str, object]) -> bool:
    """Return whether a unique mapping is stable-copy evidence."""
    outputs = _outputs(item)
    return (len(outputs) == 1
            and isinstance(item.get("input_text"), str)
            and item["input_text"] == outputs[0].get("output_text"))


def _defer_reason(item: dict[str, object]) -> str | None:
    """Return an explicit non-authorization reason for one candidate."""
    kind = item.get("candidate_kind")
    status = item.get("candidate_status")
    if kind == "EXACT_INPUT_MAPPING":
        if status in {"MULTI_FAMILY_CONFLICT", "SINGLE_FAMILY_INTERNAL_CONFLICT"}:
            return "OUTPUT_CONFLICT"
        if status == "SINGLE_FAMILY_UNIQUE_OUTPUT":
            return "SINGLE_FAMILY_AUTHORITY"
        if status == "MULTI_FAMILY_UNIQUE_OUTPUT":
            return ("IDENTITY_ONLY_CHANGED_RULE_VETO"
                    if _mapping_is_identity_only(item)
                    else "EXACT_INPUT_CONTROL_ONLY")
    if kind == "SOURCE_CONDITIONED_MAPPING":
        if status in {"MULTI_FAMILY_CONFLICT", "SINGLE_FAMILY_INTERNAL_CONFLICT"}:
            return "OUTPUT_CONFLICT"
        if status == "SINGLE_FAMILY_UNIQUE_OUTPUT":
            return "SINGLE_FAMILY_AUTHORITY"
        if status == "MULTI_FAMILY_UNIQUE_OUTPUT" and _mapping_is_identity_only(item):
            return "IDENTITY_ONLY_CHANGED_RULE_VETO"
        if status == "MULTI_FAMILY_UNIQUE_OUTPUT":
            return None
    if kind == "ORTHOGRAPHIC_ATOM":
        if status in {"MULTI_FAMILY_CONFLICT", "SINGLE_FAMILY_INTERNAL_CONFLICT"}:
            return "OUTPUT_CONFLICT"
        if status == "SINGLE_FAMILY_UNIQUE_OUTPUT":
            return "SINGLE_FAMILY_AUTHORITY"
        if status == "MULTI_FAMILY_UNIQUE_OUTPUT":
            return None
    if kind == "STRUCTURE_OBLIGATION":
        if status == "SINGLE_FAMILY_OBSERVED":
            return "SINGLE_FAMILY_AUTHORITY"
        if status == "MULTI_FAMILY_OBSERVED":
            return None
    raise BroadQaExternalDataError("v8 TRAIN candidate disposition 漂移")


def _deferred_candidate(
        item: dict[str, object], *, reason: str,
        ) -> dict[str, object]:
    """Build one compact, traceable candidate defer record."""
    families, count = _support(item)
    outputs = item.get("outputs", [])
    if not isinstance(outputs, list):
        raise BroadQaExternalDataError("v8 TRAIN deferred outputs 漂移")
    identity = {"candidate_id": item.get("candidate_id"), "reason": reason}
    if not isinstance(identity["candidate_id"], str):
        raise BroadQaExternalDataError("v8 TRAIN deferred candidate id 漂移")
    return {
        "authorization_allowed": 0,
        "candidate_id": identity["candidate_id"],
        "candidate_kind": item.get("candidate_kind"),
        "candidate_status": item.get("candidate_status"),
        "defer_id": _sha256(identity),
        "defer_reason": reason,
        "format_version": 1,
        "output_variant_count": len(outputs),
        "record_kind": V8_DEFERRED_CANDIDATE_KIND,
        "support_families": families,
        "support_family_count": count,
    }


def _eligible(observation: dict[str, object]) -> bool:
    """Read the frozen v8 observation eligibility bit."""
    eligibility = observation.get("eligibility")
    features = eligibility.get("pair_features") if isinstance(eligibility, dict) else None
    value = features.get("v8_training_eligible") if isinstance(features, dict) else None
    if value not in (0, 1):
        raise BroadQaExternalDataError("v8 TRAIN observation eligibility 漂移")
    return value == 1


def _deferred_observation(
        observation: dict[str, object],
        ) -> dict[str, object] | None:
    """Preserve ineligible and per-observation layout/morphology defers."""
    family = observation.get("source_family")
    if family not in V8_TRAIN_FAMILIES:
        raise BroadQaExternalDataError("v8 TRAIN observation family 漂移")
    left = _string_list_or_ordered(
        observation.get("zh_hant_structure_tokens"), label="zh-Hant structure")
    right = _string_list_or_ordered(
        observation.get("zh_hans_structure_tokens"), label="zh-Hans structure")
    eligibility = observation.get("eligibility")
    exclusion = eligibility.get("exclusion_reasons") if isinstance(eligibility, dict) else None
    if (not isinstance(exclusion, list)
            or any(not isinstance(reason, str) or not reason for reason in exclusion)):
        raise BroadQaExternalDataError("v8 TRAIN exclusion reasons 漂移")
    reasons = []
    if not _eligible(observation):
        reasons.append("V8_TRAINING_INELIGIBLE")
    if left or right:
        reasons.append("PER_OBSERVATION_LAYOUT_MORPHOLOGY_UNRESOLVED")
    if not reasons:
        return None
    observation_id = observation.get("observation_id")
    source_pair_id = observation.get("source_pair_id")
    if (not isinstance(observation_id, str) or len(observation_id) != 64
            or not isinstance(source_pair_id, str) or len(source_pair_id) != 64):
        raise BroadQaExternalDataError("v8 TRAIN deferred observation id 漂移")
    structure_candidate_id = ""
    if left and left == right:
        structure_candidate_id = _sha256({
            "candidate_kind": "STRUCTURE_OBLIGATION", "tokens": left})
    return {
        "authorization_allowed": 0,
        "defer_reasons": reasons,
        "exclusion_reasons": exclusion,
        "format_version": 1,
        "observation_id": observation_id,
        "record_kind": V8_DEFERRED_OBSERVATION_KIND,
        "source_family": family,
        "source_pair_id": source_pair_id,
        "structure_candidate_id": structure_candidate_id,
        "zh_hans_structure_tokens": right,
        "zh_hant_structure_tokens": left,
    }


def _loso_records(
        authorizations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """Freeze three family-held-out directions for every authorization."""
    values = []
    for authorization in authorizations:
        before = _string_list(
            authorization.get("support_families"), label="LOSO support families")
        for held_out in V8_TRAIN_FAMILIES:
            after = [family for family in before if family != held_out]
            held_out_supports = int(held_out in before)
            present = int(len(after) >= 2)
            behavior = "EXACT" if held_out_supports and present else "UNKNOWN"
            reason = (
                "TWO_TRAIN_FAMILIES_RETAIN_AUTHORITY"
                if behavior == "EXACT"
                else "SUPPORT_FAMILY_REMOVAL_DROPS_BELOW_CONSENSUS"
                if held_out_supports
                else "NO_HELD_OUT_SUPPORT_CASE")
            identity = {
                "authorization_id": authorization["authorization_id"],
                "held_out_family": held_out,
            }
            values.append({
                "authorization_id": authorization["authorization_id"],
                "candidate_id": authorization["candidate_id"],
                "evaluation_case_expected": held_out_supports,
                "expected_behavior": behavior,
                "expected_behavior_reason": reason,
                "expected_rule_present_after_holdout": present,
                "format_version": 1,
                "held_out_family": held_out,
                "held_out_output_may_influence_rule_construction": 0,
                "held_out_output_read_count": 0,
                "ledger_kind": authorization["ledger_kind"],
                "loso_plan_id": _sha256(identity),
                "record_kind": V8_LOSO_PLAN_KIND,
                "support_families_after_holdout": after,
                "support_families_before_holdout": before,
            })
    return tuple(sorted(values, key=lambda item: (
        str(item["ledger_kind"]), str(item["candidate_id"]),
        str(item["held_out_family"]))))


def _summary(outputs: dict[str, tuple[dict[str, object], ...]]) -> dict[str, object]:
    """Summarize authorization, defer and LOSO partitions."""
    authorized = tuple(item for name in (
        V8_TRAINING_RECORD_FILES[0][0],
        V8_TRAINING_RECORD_FILES[1][0],
        V8_TRAINING_RECORD_FILES[2][0],
    ) for item in outputs[name])
    deferred = outputs[V8_TRAINING_RECORD_FILES[4][0]]
    deferred_observations = outputs[V8_TRAINING_RECORD_FILES[5][0]]
    loso = outputs[V8_TRAINING_RECORD_FILES[6][0]]
    return {
        "authorization_count": len(authorized),
        "authorization_ledger_counts": dict(sorted(Counter(
            str(item["ledger_kind"]) for item in authorized).items())),
        "authorization_support_family_counts": dict(sorted(Counter(
            str(item["support_family_count"]) for item in authorized).items())),
        "deferred_candidate_count": len(deferred),
        "deferred_candidate_kind_counts": dict(sorted(Counter(
            str(item["candidate_kind"]) for item in deferred).items())),
        "deferred_candidate_reason_counts": dict(sorted(Counter(
            str(item["defer_reason"]) for item in deferred).items())),
        "deferred_observation_count": len(deferred_observations),
        "deferred_observation_reason_counts": dict(sorted(Counter(
            str(reason) for item in deferred_observations
            for reason in item["defer_reasons"]).items())),
        "exact_input_control_count": len(
            outputs[V8_TRAINING_RECORD_FILES[3][0]]),
        "learner_run_count": 0,
        "loso_expected_behavior_counts": dict(sorted(Counter(
            str(item["expected_behavior"]) for item in loso).items())),
        "loso_plan_record_count": len(loso),
        "mastery_claimed": 0,
        "production_enabled": 0,
    }


def derive_normalization_recovery_v8_training_records(
        coverage_outputs: dict[str, tuple[dict[str, object], ...]],
        observation_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """Derive all protocol ledgers, defers, controls and LOSO directions."""
    if (any(name not in coverage_outputs for name in _COVERAGE_FILES)
            or any(name not in observation_outputs for name in _OBSERVATION_FILES)):
        raise BroadQaExternalDataError("v8 TRAIN source inventory 漂移")
    exact_candidates = coverage_outputs[_COVERAGE_FILES[0]]
    lexical_candidates = coverage_outputs[_COVERAGE_FILES[1]]
    atom_candidates = coverage_outputs[_COVERAGE_FILES[2]]
    structure_candidates = coverage_outputs[_COVERAGE_FILES[3]]

    orthographic = tuple(filter(None, (
        _authorized_orthographic(item) for item in atom_candidates)))
    lexical = tuple(filter(None, (
        _authorized_lexical(item) for item in lexical_candidates)))
    structures = tuple(filter(None, (
        _authorized_structure(item) for item in structure_candidates)))
    exact_control = tuple(_exact_control(item) for item in exact_candidates)

    all_candidates = (
        exact_candidates + lexical_candidates + atom_candidates
        + structure_candidates)
    deferred = tuple(_deferred_candidate(item, reason=reason)
                     for item in all_candidates
                     for reason in (_defer_reason(item),) if reason is not None)
    observations = tuple(item for name in _OBSERVATION_FILES
                         for item in observation_outputs[name])
    deferred_observations = tuple(filter(None, (
        _deferred_observation(item) for item in observations)))
    authorizations = orthographic + lexical + structures
    loso = _loso_records(authorizations)

    outputs = {
        V8_TRAINING_RECORD_FILES[0][0]: orthographic,
        V8_TRAINING_RECORD_FILES[1][0]: lexical,
        V8_TRAINING_RECORD_FILES[2][0]: structures,
        V8_TRAINING_RECORD_FILES[3][0]: exact_control,
        V8_TRAINING_RECORD_FILES[4][0]: deferred,
        V8_TRAINING_RECORD_FILES[5][0]: deferred_observations,
        V8_TRAINING_RECORD_FILES[6][0]: loso,
    }
    summary = _summary(outputs)
    outputs[V8_TRAINING_RECORD_FILES[7][0]] = ({
        **summary,
        "format_version": 1,
        "record_kind": V8_PROTOCOL_CENSUS_KIND,
    },)
    return outputs, summary


def summarize_normalization_recovery_v8_training_records(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """Recompute the protocol summary from materialized records."""
    if any(name not in outputs for name, _role in V8_TRAINING_RECORD_FILES):
        raise BroadQaExternalDataError("v8 TRAIN material inventory 漂移")
    summary = _summary(outputs)
    expected_census = ({**summary, "format_version": 1,
                        "record_kind": V8_PROTOCOL_CENSUS_KIND},)
    if outputs[V8_TRAINING_RECORD_FILES[7][0]] != expected_census:
        raise BroadQaExternalDataError("v8 TRAIN census 漂移")
    return summary


__all__ = [
    "V8_TRAIN_FAMILIES",
    "V8_TRAINING_RECORD_FILES",
    "derive_normalization_recovery_v8_training_records",
    "summarize_normalization_recovery_v8_training_records",
]
