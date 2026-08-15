"""Reference and indexed interpreters for disabled recovery-v8 LOSO rules."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)


V8_QUERY_KINDS = (
    "LAYOUT_MORPHOLOGY_OBLIGATION",
    "SOURCE_CONDITIONED_LEXICAL_ATOM",
    "ORTHOGRAPHIC_ATOM",
    "IDENTITY_VETO",
)

_RULE_FILES = {
    "LAYOUT_MORPHOLOGY_OBLIGATION": "layout-morphology-obligations.jsonl",
    "SOURCE_CONDITIONED_LEXICAL_ATOM": (
        "source-conditioned-lexical-rules.jsonl"),
    "ORTHOGRAPHIC_ATOM": "orthographic-rules.jsonl",
    "IDENTITY_VETO": "identity-veto-rules.jsonl",
}


def _tokens(value: object, *, label: str) -> list[str]:
    """Read an ordered structure-token list."""
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)):
        raise BroadQaExternalDataError(f"v8 interpreter {label} 漂移")
    return value


def _query(value: dict[str, object]) -> dict[str, object]:
    """Validate one plain query value structure."""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("v8 interpreter query 非法")
    kind = value.get("query_kind")
    held_out = value.get("held_out_family")
    if kind not in V8_QUERY_KINDS or held_out not in V8_TRAIN_FAMILIES:
        raise BroadQaExternalDataError("v8 interpreter query kind/family 漂移")
    tokens = _tokens(value.get("structure_tokens"), label="query tokens")
    input_text = value.get("input_text")
    source = value.get("official_source_text")
    if not isinstance(input_text, str) or not isinstance(source, str):
        raise BroadQaExternalDataError("v8 interpreter query text 漂移")
    if (kind == "LAYOUT_MORPHOLOGY_OBLIGATION" and not tokens
            or kind == "SOURCE_CONDITIONED_LEXICAL_ATOM"
            and (not input_text or not source)
            or kind == "ORTHOGRAPHIC_ATOM" and len(input_text) != 1
            or kind == "IDENTITY_VETO" and not input_text):
        raise BroadQaExternalDataError("v8 interpreter query payload 漂移")
    return value


def _key(query: dict[str, object]) -> tuple[object, ...]:
    """Build the ledger-specific lookup key."""
    kind = str(query["query_kind"])
    held_out = str(query["held_out_family"])
    if kind == "LAYOUT_MORPHOLOGY_OBLIGATION":
        return held_out, tuple(query["structure_tokens"])
    if kind == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        return held_out, query["official_source_text"], query["input_text"]
    return held_out, query["input_text"]


def _rule_key(kind: str, rule: dict[str, object]) -> tuple[object, ...]:
    """Build and validate one persisted rule key."""
    held_out = rule.get("held_out_family")
    if held_out not in V8_TRAIN_FAMILIES:
        raise BroadQaExternalDataError("v8 interpreter rule family 漂移")
    if kind == "LAYOUT_MORPHOLOGY_OBLIGATION":
        return held_out, tuple(_tokens(rule.get("structure_tokens"), label="rule tokens"))
    if kind == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        source = rule.get("official_source_text")
        input_text = rule.get("input_text")
        if not isinstance(source, str) or not source or not isinstance(input_text, str):
            raise BroadQaExternalDataError("v8 interpreter lexical rule 漂移")
        return held_out, source, input_text
    input_text = rule.get("input_atom" if kind == "ORTHOGRAPHIC_ATOM" else "input_text")
    if not isinstance(input_text, str) or not input_text:
        raise BroadQaExternalDataError("v8 interpreter text rule 漂移")
    return held_out, input_text


def build_normalization_recovery_v8_rule_index(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, dict[tuple[object, ...], dict[str, object]]]:
    """Build immutable-by-contract dictionary indexes and reject collisions."""
    indexes = {}
    for kind, name in _RULE_FILES.items():
        records = outputs.get(name)
        if not isinstance(records, tuple):
            raise BroadQaExternalDataError("v8 interpreter rule inventory 漂移")
        index = {}
        for rule in records:
            if (not isinstance(rule, dict)
                    or not isinstance(rule.get("rule_id"), str)
                    or rule.get("production_enabled") != 0):
                raise BroadQaExternalDataError("v8 interpreter rule schema 漂移")
            key = _rule_key(kind, rule)
            if key in index:
                raise BroadQaExternalDataError("v8 interpreter indexed key 冲突")
            index[key] = rule
        indexes[kind] = index
    return indexes


def _result(
        query: dict[str, object], rule: dict[str, object] | None,
        ) -> dict[str, object]:
    """Project a matched rule or an explicit UNKNOWN result."""
    if rule is None:
        return {
            "behavior": "UNKNOWN",
            "exception_count": 0,
            "matched_rule_id": "",
            "output_structure_tokens": [],
            "output_text": "",
            "reason": "NO_DIRECTION_SCOPED_RULE",
            "structure_mismatch_count": 0,
        }
    kind = str(query["query_kind"])
    output_tokens = (list(rule["structure_tokens"])
                     if kind == "LAYOUT_MORPHOLOGY_OBLIGATION"
                     else list(query["structure_tokens"]))
    output_text = ""
    if kind == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        output_text = str(rule["output_text"])
    elif kind == "ORTHOGRAPHIC_ATOM":
        output_text = str(rule["output_atom"])
    elif kind == "IDENTITY_VETO":
        output_text = str(rule["output_text"])
    return {
        "behavior": "EXACT",
        "exception_count": 0,
        "matched_rule_id": rule["rule_id"],
        "output_structure_tokens": output_tokens,
        "output_text": output_text,
        "reason": "DIRECTION_SCOPED_RULE_MATCH",
        "structure_mismatch_count": int(
            output_tokens != list(query["structure_tokens"])),
    }


def interpret_normalization_recovery_v8_reference(
        outputs: dict[str, tuple[dict[str, object], ...]],
        query: dict[str, object],
        ) -> dict[str, object]:
    """Interpret by a complete linear scan of the selected ledger."""
    query = _query(query)
    kind = str(query["query_kind"])
    records = outputs.get(_RULE_FILES[kind])
    if not isinstance(records, tuple):
        raise BroadQaExternalDataError("v8 reference interpreter inventory 漂移")
    key = _key(query)
    matches = [rule for rule in records if _rule_key(kind, rule) == key]
    if len(matches) > 1:
        raise BroadQaExternalDataError("v8 reference interpreter key 冲突")
    return _result(query, matches[0] if matches else None)


def interpret_normalization_recovery_v8_indexed(
        index: dict[str, dict[tuple[object, ...], dict[str, object]]],
        query: dict[str, object],
        ) -> dict[str, object]:
    """Interpret by the independently built ledger-specific dictionary index."""
    query = _query(query)
    kind = str(query["query_kind"])
    if kind not in index or not isinstance(index[kind], dict):
        raise BroadQaExternalDataError("v8 indexed interpreter inventory 漂移")
    return _result(query, index[kind].get(_key(query)))


__all__ = [
    "V8_QUERY_KINDS",
    "build_normalization_recovery_v8_rule_index",
    "interpret_normalization_recovery_v8_indexed",
    "interpret_normalization_recovery_v8_reference",
]
