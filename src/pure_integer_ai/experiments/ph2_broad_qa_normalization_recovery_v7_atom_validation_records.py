"""Audacity 独立 atom-validation 的标签盲授权与独立评分。

held input 只含 zh-TW、结构和官方 English source。TRAIN family 先形成可选
rewrite 共识，atom guard 冻结后 scorer 才接收 zh-CN label。记录不发布 surface。
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
    localization_structure_layout_for_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_records import (
    atomize_segment,
    derive_localized_atom_route_index,
    stable_lexical_atom_authorized,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources import (
    unimorph_segment_facts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    AUDACITY_SOURCE_FAMILY,
    AUDACITY_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records import (
    derive_external_cross_source_optional_rewrite_proposals,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


AUDACITY_ATOM_VALIDATION_AUTHORIZATION_KIND = (
    "NORMALIZATION_RECOVERY_V7_AUDACITY_ATOM_AUTHORIZATION_V1")
AUDACITY_ATOM_VALIDATION_SCORE_KIND = (
    "NORMALIZATION_RECOVERY_V7_AUDACITY_ATOM_SCORE_V1")
AUDACITY_ATOM_VALIDATION_TARGET_SCOPE = (
    "AUDACITY_EXTERNAL_ATOM_TRANSFER_LOWER_BOUND_V1")


def _sha256(payload: bytes) -> str:
    """返回规范 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从完整 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(value))


def _layout_profile(value: str) -> tuple[int, ...]:
    """保留 segment 中空白与标点的确切整数序。"""
    return tuple(ord(scalar) for scalar in value
                 if scalar != "_" and not scalar.isalnum())


def _held_input_index(
        held_inputs: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """核验 Audacity held input 全量 identity，且不接受 label 字段。"""
    values = {}
    forbidden = {"output_text", "held_output", "zh_hans", "zh_cn"}
    for item in held_inputs:
        pair_id = item.get("pair_id") if isinstance(item, dict) else None
        if (not isinstance(pair_id, str) or len(pair_id) != 64
                or pair_id in values
                or forbidden.intersection(item)
                or item.get("source_family") != AUDACITY_SOURCE_FAMILY
                or item.get("source_policy_scope")
                != AUDACITY_SOURCE_POLICY_SCOPE):
            raise BroadQaExternalDataError(
                "Audacity atom-validation held input boundary 漂移")
        values[pair_id] = item
    if not values:
        raise BroadQaExternalDataError(
            "Audacity atom-validation held input 为空")
    return values


def derive_audacity_atom_validation_authorizations(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        held_inputs: tuple[dict[str, object], ...],
        opencc_routes: dict[str, str],
        morphology_by_form: dict[str, tuple[tuple[str, str], ...]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """只凭 TRAIN state 与 held-visible input 冻结外部 atom 授权。"""
    held_by_pair = _held_input_index(held_inputs)
    proposals, proposal_census = (
        derive_external_cross_source_optional_rewrite_proposals(
            observations=observations,
            fragments=fragments,
            plans=plans,
            held_inputs=held_inputs,
        ))
    proposal_by_pair = {str(item["pair_id"]): item for item in proposals}
    if set(proposal_by_pair) != set(held_by_pair):
        raise BroadQaExternalDataError(
            "Audacity atom-validation proposal denominator 漂移")
    routes, route_census = derive_localized_atom_route_index(observations)
    aggregate = Counter(route_census)
    authorizations = []
    for pair_id in sorted(held_by_pair):
        held_input = held_by_pair[pair_id]
        proposal = proposal_by_pair[pair_id]
        input_text = str(held_input["input_text"])
        proposal_text = str(proposal["proposal_output_text"])
        source = str(held_input["official_source_text"])
        tokens = tuple(str(item) for item in held_input["structure_tokens"])
        reasons = Counter()
        authorized = proposal["proposal_decision"] == (
            "PROPOSED_UNIQUE_MULTI_FAMILY_CONSENSUS")
        if not authorized:
            reasons[str(proposal["proposal_decision"])] += 1
        input_layout = localization_structure_layout_for_tokens(
            input_text, tokens)
        proposal_layout = localization_structure_layout_for_tokens(
            proposal_text, tokens)
        source_layout = localization_structure_layout(source)
        if tuple(source_layout["structure_tokens"]) != tokens:
            authorized = False
            reasons["SOURCE_STRUCTURE_MISMATCH"] += 1
        changed_count = 0
        stable_count = 0
        orthographic_count = 0
        marked_count = 0
        if len(source_layout["segments"]) != len(input_layout["segments"]):
            authorized = False
            reasons["SOURCE_SEGMENT_COUNT_MISMATCH"] += 1
        else:
            for ordinal, (input_segment, proposal_segment) in enumerate(zip(
                    input_layout["segments"], proposal_layout["segments"])):
                if input_segment == proposal_segment:
                    continue
                changed_count += 1
                source_segment = source_layout["segments"][ordinal]
                atoms = tuple(
                    atom for atom in atomize_segment(
                        input_segment, proposal_segment)
                    if not atom[0].startswith("LAYOUT_"))
                segment_ok = bool(atoms)
                segment_stable = 0
                for atom in atoms:
                    if opencc_routes.get(atom[1]) == atom[2]:
                        orthographic_count += 1
                    elif stable_lexical_atom_authorized(
                            atom, held_family=AUDACITY_SOURCE_FAMILY,
                            routes=routes):
                        stable_count += 1
                        segment_stable += 1
                    else:
                        segment_ok = False
                        reasons["UNAUTHORIZED_ATOM"] += 1
                if segment_stable == 0:
                    segment_ok = False
                    reasons["NO_STABLE_LEXICAL_ATOM"] += 1
                marked = tuple(
                    fact for fact in unimorph_segment_facts(
                        source_segment, morphology_by_form)
                    if fact.startswith("UNIMORPH_MARKED:"))
                marked_count += len(marked)
                if marked:
                    segment_ok = False
                    reasons["MARKED_MORPHOLOGY_UNRESOLVED"] += len(marked)
                if (_layout_profile(source_segment)
                        != _layout_profile(proposal_segment)):
                    segment_ok = False
                    reasons["SOURCE_LAYOUT_NOT_PRESERVED"] += 1
                authorized = authorized and segment_ok
        if changed_count == 0:
            authorized = False
            reasons["NO_CHANGED_SEGMENT"] += 1
        identity = {
            "pair_id": pair_id,
            "proposal_id": proposal["proposal_id"],
            "target_scope": AUDACITY_ATOM_VALIDATION_TARGET_SCOPE,
        }
        authorizations.append({
            **identity,
            "authorization_decision": (
                "AUTHORIZED" if authorized else "DEFERRED"),
            "authorization_id": _record_id(identity),
            "changed_segment_count": changed_count,
            "format_version": 1,
            "held_label_read_count": 0,
            "indexed_reference_mismatch_count": proposal[
                "indexed_reference_mismatch_count"],
            "input_text": input_text,
            "marked_morphology_fact_count": marked_count,
            "orthographic_atom_count": orthographic_count,
            "partial_commit_count": proposal["partial_commit_count"],
            "proposal_output_sha256": proposal[
                "proposal_output_sha256"],
            "proposal_output_text": proposal_text,
            "reason_counts": {
                key: reasons[key] for key in sorted(reasons)},
            "record_kind": AUDACITY_ATOM_VALIDATION_AUTHORIZATION_KIND,
            "stable_lexical_atom_count": stable_count,
            "structure_token_mismatch_count": proposal[
                "structure_token_mismatch_count"],
            "surface_published": 0,
        })
        aggregate["authorization_count"] += 1
        aggregate["authorized_count"] += int(authorized)
        aggregate["deferred_count"] += int(not authorized)
        aggregate["changed_segment_count"] += changed_count
        aggregate["stable_lexical_atom_count"] += stable_count
        aggregate["orthographic_atom_count"] += orthographic_count
        aggregate["marked_morphology_fact_count"] += marked_count
        aggregate["indexed_reference_mismatch_count"] += int(proposal[
            "indexed_reference_mismatch_count"])
        aggregate["partial_commit_count"] += int(proposal[
            "partial_commit_count"])
        aggregate["structure_token_mismatch_count"] += int(proposal[
            "structure_token_mismatch_count"])
    return tuple(authorizations), {
        "atom_route_census": {
            key: route_census[key] for key in sorted(route_census)},
        "authorization_count": aggregate["authorization_count"],
        "authorized_count": aggregate["authorized_count"],
        "changed_segment_count": aggregate["changed_segment_count"],
        "deferred_count": aggregate["deferred_count"],
        "held_label_read_count": 0,
        "indexed_reference_mismatch_count": aggregate[
            "indexed_reference_mismatch_count"],
        "marked_morphology_fact_count": aggregate[
            "marked_morphology_fact_count"],
        "orthographic_atom_count": aggregate["orthographic_atom_count"],
        "partial_commit_count": aggregate["partial_commit_count"],
        "proposal": proposal_census,
        "stable_lexical_atom_count": aggregate[
            "stable_lexical_atom_count"],
        "structure_token_mismatch_count": aggregate[
            "structure_token_mismatch_count"],
    }


def score_audacity_atom_validation_authorizations(
        authorizations: tuple[dict[str, object], ...],
        *,
        labels_by_pair: dict[str, tuple[str, str]],
        expected_denominator_count: int,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """授权冻结后读取全部 zh-CN label，并执行 v2 三态结果门。"""
    if (type(expected_denominator_count) is not int
            or expected_denominator_count <= 0):
        raise BroadQaExternalDataError(
            "Audacity atom-validation denominator 非法")
    authorization_by_pair = {}
    for item in authorizations:
        pair_id = item.get("pair_id") if isinstance(item, dict) else None
        if (not isinstance(pair_id, str) or pair_id in authorization_by_pair
                or item.get("held_label_read_count") != 0):
            raise BroadQaExternalDataError(
                "Audacity atom-validation authorization freeze 漂移")
        authorization_by_pair[pair_id] = item
    selection_drift = int(
        set(authorization_by_pair) != set(labels_by_pair)
        or len(authorization_by_pair) != expected_denominator_count)
    if selection_drift:
        raise BroadQaExternalDataError(
            "Audacity atom-validation scoring denominator 漂移")
    outcomes = Counter()
    gates = Counter()
    records = []
    for pair_id in sorted(authorization_by_pair):
        authorization = authorization_by_pair[pair_id]
        label = labels_by_pair[pair_id]
        if (not isinstance(label, tuple) or len(label) != 2
                or any(not isinstance(value, str) for value in label)):
            raise BroadQaExternalDataError(
                "Audacity atom-validation label 非法")
        input_text, output_text = label
        if input_text != authorization["input_text"]:
            raise BroadQaExternalDataError(
                "Audacity atom-validation input/label 漂移")
        proposal = str(authorization["proposal_output_text"])
        authorized = authorization["authorization_decision"] == "AUTHORIZED"
        final = proposal if authorized else input_text
        outcome = (
            "EXACT" if final == output_text
            else "UNKNOWN" if final == input_text else "WRONG")
        changed = int(final != input_text)
        identity_pair = int(input_text == output_text)
        outcomes[outcome] += 1
        gates["authorized_changed_exact_output_count"] += int(
            authorized and changed and outcome == "EXACT")
        gates["identity_exact_output_count"] += int(
            identity_pair and outcome == "EXACT")
        gates["identity_false_change_count"] += int(
            identity_pair and changed)
        gates["indexed_reference_mismatch_count"] += int(authorization[
            "indexed_reference_mismatch_count"])
        gates["partial_commit_count"] += int(authorization[
            "partial_commit_count"])
        gates["structure_token_mismatch_count"] += int(authorization[
            "structure_token_mismatch_count"])
        identity = {
            "authorization_id": authorization["authorization_id"],
            "pair_id": pair_id,
            "target_scope": AUDACITY_ATOM_VALIDATION_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "authorization_decision": authorization[
                "authorization_decision"],
            "changed_output": changed,
            "format_version": 1,
            "held_label_read_count": 1,
            "identity_pair": identity_pair,
            "outcome": outcome,
            "proposal_output_sha256": authorization[
                "proposal_output_sha256"],
            "reason_counts": authorization["reason_counts"],
            "record_kind": AUDACITY_ATOM_VALIDATION_SCORE_KIND,
            "score_id": _record_id(identity),
            "surface_published": 0,
        })
    hard_failure = any((
        outcomes["WRONG"] != 0,
        gates["identity_false_change_count"] != 0,
        gates["indexed_reference_mismatch_count"] != 0,
        gates["partial_commit_count"] != 0,
        gates["structure_token_mismatch_count"] != 0,
        selection_drift != 0,
    ))
    outcome = (
        "FAIL_HARD_CONJUNCT"
        if hard_failure else
        "PASS_NONZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG"
        if gates["authorized_changed_exact_output_count"] >= 1 else
        "NE_ZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG")
    return tuple(records), {
        "authorized_changed_exact_output_count": gates[
            "authorized_changed_exact_output_count"],
        "denominator_count": len(records),
        "exception_count": 0,
        "held_label_read_count": len(records),
        "identity_exact_output_count": gates[
            "identity_exact_output_count"],
        "identity_false_change_count": gates[
            "identity_false_change_count"],
        "identity_only_exact_satisfies_transfer_pass": 0,
        "indexed_reference_mismatch_count": gates[
            "indexed_reference_mismatch_count"],
        "outcome": outcome,
        "outcome_counts": {
            key: outcomes[key] for key in ("EXACT", "UNKNOWN", "WRONG")},
        "partial_commit_count": gates["partial_commit_count"],
        "selection_drift_count": selection_drift,
        "structure_token_mismatch_count": gates[
            "structure_token_mismatch_count"],
    }


__all__ = [
    "AUDACITY_ATOM_VALIDATION_AUTHORIZATION_KIND",
    "AUDACITY_ATOM_VALIDATION_SCORE_KIND",
    "AUDACITY_ATOM_VALIDATION_TARGET_SCOPE",
    "derive_audacity_atom_validation_authorizations",
    "score_audacity_atom_validation_authorizations",
]
