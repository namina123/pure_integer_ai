"""派生 recovery-v7 atom identifiable lower-bound TRAIN feasibility。

proposal 先在无 held outcome 的接口中形成；本模块随后只用可见 source/input、
OpenCC、UniMorph 与非 held localized route 冻结授权，最后单独接收 label 评分。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
    localization_structure_layout_for_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources import (
    unimorph_segment_facts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records import (
    derive_cross_source_transformation_unscored_proposals,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


ATOM_IDENTIFIABILITY_PROPOSAL_KIND = (
    "NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_PROPOSAL_V1")
ATOM_IDENTIFIABILITY_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_CENSUS_V1")
ATOM_IDENTIFIABILITY_TARGET_SCOPE = (
    "CROSS_SOURCE_ATOM_IDENTIFIABLE_LOWER_BOUND_V1")

_STABLE_LEXICAL_FAMILY_MIN = 2
_STABLE_LEXICAL_OCCURRENCE_MIN = 3


def _sha256(payload: bytes) -> str:
    """返回记录、surface 或集合的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从完整 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(value))


def _text_sha256(value: str) -> str:
    """哈希内存文本，不把表面写入记录。"""
    return _sha256(value.encode("utf-8"))


def _layout_profile(value: str) -> tuple[int, ...]:
    """保留 text segment 中空白与标点的确切整数序。"""
    return tuple(ord(scalar) for scalar in value
                 if scalar != "_" and not scalar.isalnum())


def atomize_segment(
        source: str,
        target: str,
        ) -> tuple[tuple[str, str, str], ...]:
    """把 localized segment rewrite 拆成 layout、scalar 与变长原子。"""
    if not isinstance(source, str) or not isinstance(target, str):
        raise BroadQaExternalDataError("atom segment 非字符串")
    values = []
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = source[i1:i2]
        right = target[j1:j2]
        if tag == "replace" and len(left) == len(right):
            for a, b in zip(left, right):
                if a == b:
                    continue
                layout = all(
                    scalar != "_" and not scalar.isalnum()
                    for scalar in (a, b))
                values.append((
                    "LAYOUT_SUBSTITUTION" if layout
                    else "SCALAR_SUBSTITUTION", a, b))
            continue
        joined = left + right
        layout = bool(joined) and all(
            scalar != "_" and not scalar.isalnum() for scalar in joined)
        if layout:
            kind = f"LAYOUT_{tag.upper()}"
        elif tag == "insert":
            kind = "SEMANTIC_INSERTION"
        elif tag == "delete":
            kind = "SEMANTIC_DELETION"
        else:
            kind = "SEMANTIC_REWRITE"
        values.append((kind, left, right))
    return tuple(values)


def _observation_indexes(
        observations: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """核对 observation/plan identity 与 family inventory。"""
    observation_by_id = {}
    for item in observations:
        identity = item.get("observation_id") if isinstance(item, dict) else None
        if (not isinstance(identity, str) or len(identity) != 64
                or identity in observation_by_id
                or item.get("source_family") not in V5_SOURCE_FAMILIES):
            raise BroadQaExternalDataError(
                "atom identifiability observation identity 漂移")
        observation_by_id[identity] = item
    plan_by_id = {}
    for item in plans:
        identity = item.get("observation_id") if isinstance(item, dict) else None
        owner = observation_by_id.get(str(identity))
        if (not isinstance(identity, str) or identity in plan_by_id
                or owner is None
                or item.get("source_family") != owner.get("source_family")):
            raise BroadQaExternalDataError(
                "atom identifiability plan identity 漂移")
        plan_by_id[identity] = item
    return observation_by_id, plan_by_id


def _localized_atom_routes(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[tuple[str, str], dict[str, dict[str, int]]], Counter]:
    """从全部 TRAIN family 派生 localized atom route occurrence。"""
    routes: dict[
        tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
    census = Counter()
    for observation in observations:
        tokens = tuple(str(item) for item in observation["structure_tokens"])
        input_layout = (
            localization_structure_layout_for_tokens(
                str(observation["input_text"]), tokens)
            if tokens else localization_structure_layout(
                str(observation["input_text"])))
        output_layout = (
            localization_structure_layout_for_tokens(
                str(observation["output_text"]), tokens)
            if tokens else localization_structure_layout(
                str(observation["output_text"])))
        family = str(observation["source_family"])
        for input_segment, output_segment in zip(
                input_layout["segments"], output_layout["segments"]):
            atoms = atomize_segment(input_segment, output_segment)
            census["localized_segment_count"] += 1
            census["localized_changed_segment_count"] += int(bool(atoms))
            for kind, source, target in atoms:
                routes[(kind, source)][target][family] += 1
                census["localized_atom_occurrence_count"] += 1
    return routes, census


def _stable_lexical_authorized(
        atom: tuple[str, str, str],
        *,
        held_family: str,
        routes: dict[tuple[str, str], dict[str, dict[str, int]]],
        ) -> bool:
    """判定跨 family、零冲突的多 scalar lexical route。"""
    kind, source, target = atom
    if kind != "SEMANTIC_REWRITE" or len(source) < 2 or not target:
        return False
    outputs = routes.get((kind, source), {})
    selected = {
        output: {
            family: count for family, count in families.items()
            if family != held_family}
        for output, families in outputs.items()}
    selected = {output: families for output, families in selected.items()
                if families}
    if set(selected) != {target}:
        return False
    families = selected[target]
    return (len(families) >= _STABLE_LEXICAL_FAMILY_MIN
            and sum(families.values()) >= _STABLE_LEXICAL_OCCURRENCE_MIN)


def derive_atom_identifiability_authorizations(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        official_source_by_pair: dict[str, str],
        opencc_routes: dict[str, str],
        morphology_by_form: dict[str, tuple[tuple[str, str], ...]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """在不读取 held outcome 的条件下冻结 identifiable lower bound。"""
    observation_by_id, plan_by_id = _observation_indexes(observations, plans)
    routes, route_census = _localized_atom_routes(observations)
    proposals = derive_cross_source_transformation_unscored_proposals(
        observations=observations, fragments=fragments, plans=plans)
    authorizations = []
    aggregate = Counter(route_census)
    for proposal in proposals:
        observation_id = str(proposal["held_out_observation_id"])
        held_family = str(proposal["held_out_source_family"])
        observation = observation_by_id[observation_id]
        source = official_source_by_pair.get(str(proposal["source_pair_id"]))
        tokens = tuple(str(item) for item in observation["structure_tokens"])
        reasons = Counter()
        authorized = source is not None
        required_count = 0
        stable_count = 0
        orthographic_count = 0
        marked_count = 0
        if source is None:
            reasons["OFFICIAL_SOURCE_UNAVAILABLE"] += 1
        else:
            source_layout = localization_structure_layout(source)
            if tuple(source_layout["structure_tokens"]) != tokens:
                authorized = False
                reasons["SOURCE_STRUCTURE_MISMATCH"] += 1
            input_layout = localization_structure_layout_for_tokens(
                str(observation["input_text"]), tokens)
            proposal_layout = localization_structure_layout_for_tokens(
                str(proposal["proposal_output_text"]), tokens)
            for ordinal, plan_segment in enumerate(
                    plan_by_id[observation_id]["segments"]):
                if int(plan_segment["proposal_required"]) != 1:
                    continue
                required_count += 1
                source_segment = source_layout["segments"][ordinal]
                atoms = tuple(
                    atom for atom in atomize_segment(
                        input_layout["segments"][ordinal],
                        proposal_layout["segments"][ordinal])
                    if not atom[0].startswith("LAYOUT_"))
                segment_ok = bool(atoms)
                segment_stable = 0
                for atom in atoms:
                    if opencc_routes.get(atom[1]) == atom[2]:
                        orthographic_count += 1
                    elif _stable_lexical_authorized(
                            atom, held_family=held_family, routes=routes):
                        segment_stable += 1
                        stable_count += 1
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
                        != _layout_profile(
                            proposal_layout["segments"][ordinal])):
                    segment_ok = False
                    reasons["SOURCE_LAYOUT_NOT_PRESERVED"] += 1
                authorized = authorized and segment_ok
        authorized = authorized and required_count > 0
        aggregate["proposal_count"] += 1
        aggregate["required_obligation_count"] += required_count
        aggregate["authorized_proposal_count"] += int(authorized)
        aggregate["deferred_proposal_count"] += int(not authorized)
        aggregate["stable_lexical_atom_count"] += stable_count
        aggregate["orthographic_atom_count"] += orthographic_count
        aggregate["marked_morphology_fact_count"] += marked_count
        identity = {
            "held_out_observation_id": observation_id,
            "held_out_source_family": held_family,
            "proposal_output_sha256": proposal["proposal_output_sha256"],
            "target_scope": ATOM_IDENTIFIABILITY_TARGET_SCOPE,
        }
        authorizations.append({
            **identity,
            "authorization_decision": (
                "AUTHORIZED" if authorized else "DEFERRED"),
            "authorization_id": _record_id(identity),
            "format_version": 1,
            "held_label_reads": 0,
            "orthographic_atom_count": orthographic_count,
            "proposal_output_text": proposal["proposal_output_text"],
            "reason_counts": {
                key: reasons[key] for key in sorted(reasons)},
            "required_obligation_count": required_count,
            "stable_lexical_atom_count": stable_count,
        })
    authorizations.sort(key=lambda item: str(item["authorization_id"]))
    return tuple(authorizations), {
        key: aggregate[key] for key in sorted(aggregate)}


def score_atom_identifiability_authorizations(
        authorizations: tuple[dict[str, object], ...],
        *,
        labels_by_observation: dict[str, tuple[str, str]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """在授权冻结后读取 TRAIN label，形成无 surface 的 proposal records。"""
    outcomes = Counter()
    records = []
    for authorization in authorizations:
        observation_id = str(authorization["held_out_observation_id"])
        label = labels_by_observation.get(observation_id)
        if label is None or len(label) != 2:
            raise BroadQaExternalDataError(
                "atom identifiability held label 缺失")
        input_text, output_text = label
        proposal = str(authorization["proposal_output_text"])
        if authorization["authorization_decision"] == "AUTHORIZED":
            final = proposal
        else:
            final = input_text
        outcome = (
            "EXACT" if final == output_text
            else "UNKNOWN" if final == input_text else "WRONG")
        outcomes[outcome] += 1
        identity = {
            "authorization_id": authorization["authorization_id"],
            "target_scope": ATOM_IDENTIFIABILITY_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "authorization_decision": authorization[
                "authorization_decision"],
            "format_version": 1,
            "held_label_read_count": 1,
            "held_out_source_family": authorization[
                "held_out_source_family"],
            "orthographic_atom_count": authorization[
                "orthographic_atom_count"],
            "outcome": outcome,
            "proposal_id": _record_id(identity),
            "proposal_output_sha256": authorization[
                "proposal_output_sha256"],
            "reason_counts": authorization["reason_counts"],
            "record_kind": ATOM_IDENTIFIABILITY_PROPOSAL_KIND,
            "required_obligation_count": authorization[
                "required_obligation_count"],
            "stable_lexical_atom_count": authorization[
                "stable_lexical_atom_count"],
            "surface_published": 0,
        })
    records.sort(key=lambda item: str(item["proposal_id"]))
    exact = outcomes["EXACT"]
    wrong = outcomes["WRONG"]
    return tuple(records), {
        "capability_claimed": 0,
        "feasibility_outcome": (
            "PASS_NONZERO_EXACT_ZERO_WRONG"
            if exact > 0 and wrong == 0 else
            "NE_ZERO_EXACT_ZERO_WRONG" if wrong == 0 else
            "FAIL_WRONG_AUTHORIZATION"),
        "held_label_read_count": len(authorizations),
        "outcome_counts": {
            key: outcomes[key] for key in ("EXACT", "UNKNOWN", "WRONG")},
        "proposal_count": len(records),
        "runtime_claimed": 0,
    }


def derive_atom_identifiability_feasibility(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        official_source_by_pair: dict[str, str],
        opencc_routes: dict[str, str],
        morphology_by_form: dict[str, tuple[tuple[str, str], ...]],
        ) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], dict[str, object]]:
    """依次冻结授权、独立评分并形成 census record。"""
    authorizations, authorization_census = (
        derive_atom_identifiability_authorizations(
            observations=observations, fragments=fragments, plans=plans,
            official_source_by_pair=official_source_by_pair,
            opencc_routes=opencc_routes,
            morphology_by_form=morphology_by_form))
    labels = {
        str(item["observation_id"]): (
            str(item["input_text"]), str(item["output_text"]))
        for item in observations}
    records, scoring = score_atom_identifiability_authorizations(
        authorizations, labels_by_observation=labels)
    census_identity = {"target_scope": ATOM_IDENTIFIABILITY_TARGET_SCOPE}
    census = ({
        **census_identity,
        "authorization": authorization_census,
        "census_id": _record_id(census_identity),
        "format_version": 1,
        "record_kind": ATOM_IDENTIFIABILITY_CENSUS_KIND,
        "scoring": scoring,
    },)
    return records, census, {
        "authorization": authorization_census,
        "scoring": scoring,
    }


__all__ = [
    "ATOM_IDENTIFIABILITY_CENSUS_KIND",
    "ATOM_IDENTIFIABILITY_PROPOSAL_KIND",
    "ATOM_IDENTIFIABILITY_TARGET_SCOPE",
    "atomize_segment",
    "derive_atom_identifiability_authorizations",
    "derive_atom_identifiability_feasibility",
    "score_atom_identifiability_authorizations",
]
