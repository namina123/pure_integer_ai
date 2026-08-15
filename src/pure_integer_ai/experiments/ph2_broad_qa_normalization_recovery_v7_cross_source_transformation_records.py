"""派生 recovery-v7 cross-source segment transformation feasibility。

每个 TRAIN source family 独立从冻结 edit core 与 structured segment 对齐中
学习短原子 rewrite。held-out 执行只有在至少两个独立 family model 重建出相同
整句、且训练侧 neutral source route 对该 output SHA 提供双来源授权时才提交。
neutral projection 只鉴权，不提供生成 surface；所有评分均在 proposal 冻结后进行。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout_for_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


CROSS_SOURCE_TRANSFORMATION_MODEL_KIND = (
    "NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_MODEL_V1")
CROSS_SOURCE_TRANSFORMATION_STAGE_KIND = (
    "NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_STAGE_V1")
CROSS_SOURCE_TRANSFORMATION_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_LOSO_V1")
CROSS_SOURCE_TRANSFORMATION_TARGET_SCOPE = (
    "CROSS_PRODUCT_ZH_CN_VARIABLE_SEGMENT_TRANSFORMATION_V1")

TRANSFORMATION_ATOM_SCALAR_MAX = 4
_PRE_AUTHORIZATION_STAGE = "INDEPENDENT_FAMILY_TRANSFORMATION_CONSENSUS"
_FINAL_STAGE = "NEUTRAL_SOURCE_AUTHORIZED_TRANSFORMATION"


def _sha256(payload: bytes) -> str:
    """返回规范 identity 或 UTF-8 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回字符串的 UTF-8 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _binary(value: object, *, label: str) -> int:
    """读取严格 JSON 二值。"""
    if type(value) is not int or value not in (0, 1):
        raise BroadQaExternalDataError(
            f"v7 cross-source transformation {label} 非二值")
    return value


def _indexes(
        observations: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
        ]:
    """核验 observation/variable plan identity 与 source family。"""
    observation_by_id = {}
    for item in observations:
        observation_id = item.get("observation_id") \
            if isinstance(item, dict) else None
        if (not isinstance(observation_id, str)
                or len(observation_id) != 64
                or observation_id in observation_by_id
                or item.get("source_family") not in V5_SOURCE_FAMILIES):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation observation identity 漂移")
        observation_by_id[observation_id] = item
    plan_by_id = {}
    for plan in plans:
        observation_id = plan.get("observation_id") \
            if isinstance(plan, dict) else None
        owner = observation_by_id.get(str(observation_id))
        if (not isinstance(observation_id, str)
                or observation_id in plan_by_id
                or owner is None
                or plan.get("source_family") != owner.get("source_family")):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation plan identity 漂移")
        plan_by_id[observation_id] = plan
    if not plan_by_id:
        raise BroadQaExternalDataError(
            "v7 cross-source transformation variable plan 为空")
    return observation_by_id, plan_by_id


def _structured_edit_evidence(
        *,
        observation_by_id: dict[str, dict[str, object]],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[tuple[str, str, str, str], ...]:
    """从 variable plan 所属 segment 派生非空 input edit core。"""
    values = []
    for plan in plans:
        observation = observation_by_id[str(plan["observation_id"])]
        tokens = observation.get("structure_tokens")
        input_text = observation.get("input_text")
        output_text = observation.get("output_text")
        if (not isinstance(tokens, list) or not tokens
                or not isinstance(input_text, str)
                or not isinstance(output_text, str)):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation structured input 漂移")
        input_layout = localization_structure_layout_for_tokens(
            input_text, tuple(tokens))
        output_layout = localization_structure_layout_for_tokens(
            output_text, tuple(tokens))
        for input_segment, output_segment in zip(
                input_layout["segments"], output_layout["segments"]):
            matcher = SequenceMatcher(
                None, input_segment, output_segment, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                source = input_segment[i1:i2]
                target = output_segment[j1:j2]
                if tag != "equal" and source and source != target:
                    values.append((
                        str(observation["source_family"]),
                        source,
                        target,
                        str(observation["observation_id"]),
                    ))
    return tuple(values)


def _fragment_edit_evidence(
        fragments: tuple[dict[str, object], ...],
        ) -> tuple[tuple[str, str, str, str], ...]:
    """读取冻结 EDIT_CORE evidence，拒绝 schema 漂移。"""
    values = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation fragment 非对象")
        if fragment.get("fragment_kind") != "EDIT_CORE":
            continue
        source = fragment.get("input_text")
        target = fragment.get("output_text")
        family = fragment.get("source_family")
        observation_id = fragment.get("observation_id")
        if (family not in V5_SOURCE_FAMILIES
                or not isinstance(source, str) or not source
                or not isinstance(target, str)
                or source == target
                or not isinstance(observation_id, str)
                or len(observation_id) != 64):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation EDIT_CORE 漂移")
        values.append((family, source, target, observation_id))
    if not values:
        raise BroadQaExternalDataError(
            "v7 cross-source transformation EDIT_CORE 为空")
    return tuple(values)


def _equal_evidence(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, dict[str, set[str]]],
            dict[str, set[str]],
        ]:
    """派生同 family 稳定复制 scalar 与 identity-veto substring。"""
    stable: dict[str, dict[str, set[str]]] = {
        family: defaultdict(set) for family in V5_SOURCE_FAMILIES}
    unchanged = {family: set() for family in V5_SOURCE_FAMILIES}
    for observation in observations:
        family = str(observation["source_family"])
        observation_id = str(observation["observation_id"])
        input_text = observation.get("input_text")
        output_text = observation.get("output_text")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation observation surface 漂移")
        matcher = SequenceMatcher(
            None, input_text, output_text, autojunk=False)
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            block = input_text[i1:i2]
            for scalar in block:
                stable[family][scalar].add(observation_id)
            maximum = min(TRANSFORMATION_ATOM_SCALAR_MAX, len(block))
            for length in range(1, maximum + 1):
                for start in range(0, len(block) - length + 1):
                    unchanged[family].add(block[start:start + length])
    return stable, unchanged


def _derive_models(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        observation_by_id: dict[str, dict[str, object]],
        ) -> tuple[
            dict[str, dict[str, object]],
            tuple[dict[str, object], ...],
        ]:
    """按 source family 独立学习短 rewrite 与稳定复制集合。"""
    evidence = _fragment_edit_evidence(fragments) + (
        _structured_edit_evidence(
            observation_by_id=observation_by_id,
            plans=plans,
        ))
    variants: dict[
        str, dict[str, dict[str, set[str]]]] = {
            family: defaultdict(lambda: defaultdict(set))
            for family in V5_SOURCE_FAMILIES}
    for family, source, target, observation_id in evidence:
        if len(source) <= TRANSFORMATION_ATOM_SCALAR_MAX:
            variants[family][source][target].add(observation_id)
    stable, unchanged = _equal_evidence(observations)
    models = {}
    representations = []
    for family in V5_SOURCE_FAMILIES:
        routes = []
        conflict_input_count = 0
        identity_veto_input_count = 0
        for source, outputs in variants[family].items():
            if len(outputs) != 1:
                conflict_input_count += 1
                continue
            if source in unchanged[family]:
                identity_veto_input_count += 1
                continue
            target, observation_ids = next(iter(outputs.items()))
            routes.append({
                "input_text": source,
                "observation_ids": tuple(sorted(observation_ids)),
                "output_text": target,
            })
        routes.sort(key=lambda item: (
            -len(str(item["input_text"])),
            str(item["input_text"]),
            str(item["output_text"]),
        ))
        buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
        for route in routes:
            buckets[ord(str(route["input_text"])[0])].append(route)
        frozen_buckets = {
            key: tuple(value) for key, value in buckets.items()}
        reference_partitions: dict[int, list[dict[str, object]]] = defaultdict(
            list)
        for route in reversed(routes):
            reference_partitions[
                ord(str(route["input_text"])[0]) // 256].append(route)
        stable_scalars = frozenset(stable[family])
        route_commitments = [{
            "input_length": len(str(item["input_text"])),
            "input_sha256": _text_sha256(str(item["input_text"])),
            "output_length": len(str(item["output_text"])),
            "output_sha256": _text_sha256(str(item["output_text"])),
            "support_observation_count": len(item["observation_ids"]),
        } for item in routes]
        identity = {
            "source_family": family,
            "target_policy_scope": CROSS_SOURCE_TRANSFORMATION_TARGET_SCOPE,
        }
        representations.append({
            **identity,
            "atom_scalar_max": TRANSFORMATION_ATOM_SCALAR_MAX,
            "conflict_input_count": conflict_input_count,
            "format_version": 1,
            "identity_veto_input_count": identity_veto_input_count,
            "model_id": _record_id(identity),
            "record_kind": CROSS_SOURCE_TRANSFORMATION_MODEL_KIND,
            "route_count": len(routes),
            "route_identity_set_sha256": _sha256(canonical_json_bytes(
                route_commitments)),
            "stable_copy_scalar_count": len(stable_scalars),
            "stable_copy_scalar_set_sha256": _sha256(
                canonical_json_bytes(sorted(stable_scalars))),
        })
        models[family] = {
            "buckets": frozen_buckets,
            "reference_partitions": {
                key: tuple(value)
                for key, value in reference_partitions.items()},
            "stable_scalars": stable_scalars,
        }
    return models, tuple(representations)


def _routes_at(
        *,
        model: dict[str, object],
        segment: str,
        position: int,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """返回当前位置 indexed 或独立 reference route。"""
    if indexed:
        buckets = model.get("buckets")
        if not isinstance(buckets, dict):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation bucket 漂移")
        return tuple(buckets.get(ord(segment[position]), ()))
    partitions = model.get("reference_partitions")
    if not isinstance(partitions, dict):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation reference partition 漂移")
    routes = partitions.get(ord(segment[position]) // 256, ())
    if not isinstance(routes, tuple):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation reference routes 漂移")
    return tuple(
        route for route in routes
        if str(route["input_text"])[0] == segment[position])


def _execute_family_model(
        *,
        observation: dict[str, object],
        plan: dict[str, object] | None,
        model: dict[str, object],
        indexed: bool,
        ) -> dict[str, object]:
    """在 structure segment 内执行一个 family 独立 transformation model。"""
    if type(indexed) is not bool:
        raise BroadQaExternalDataError(
            "v7 cross-source transformation interpreter 非法")
    input_text = observation.get("input_text")
    tokens = observation.get("structure_tokens")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(tokens, list)
            or any(not isinstance(token, str) or not token
                   for token in tokens)):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation execution input 漂移")
    if plan is None:
        payload = {
            "decision": "UNKNOWN_NO_VARIABLE_PLAN",
            "output_text": input_text,
            "partial_commit_count": 0,
            "rewrite_count": 0,
            "structure_token_mismatch_count": 0,
        }
        return {**payload, "result_sha256": _sha256(
            canonical_json_bytes(payload))}
    representation_eligible = _binary(
        plan.get("representation_eligible"),
        label="representation_eligible")
    if (plan.get("observation_id") != observation.get("observation_id")
            or plan.get("source_family") != observation.get("source_family")
            or plan.get("structure_tokens") != tokens):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation execution plan 漂移")
    if representation_eligible == 0:
        payload = {
            "decision": "UNKNOWN_PLAN_INELIGIBLE",
            "output_text": input_text,
            "partial_commit_count": 0,
            "rewrite_count": 0,
            "structure_token_mismatch_count": 0,
        }
        return {**payload, "result_sha256": _sha256(
            canonical_json_bytes(payload))}
    layout = localization_structure_layout_for_tokens(
        input_text, tuple(tokens))
    segments = plan.get("segments")
    stable_scalars = model.get("stable_scalars")
    if (not isinstance(segments, list)
            or len(segments) != len(layout["segments"])
            or not isinstance(stable_scalars, frozenset)):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation segment ledger 漂移")
    output_segments = []
    rewrite_count = 0
    decision = "CANDIDATE"
    for segment_record, segment in zip(segments, layout["segments"]):
        proposal_required = _binary(
            segment_record.get("proposal_required"),
            label="proposal_required")
        if (segment_record.get("input_length") != len(segment)
                or segment_record.get("input_sha256")
                != _text_sha256(segment)):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation segment/input 漂移")
        values = []
        covered = [0] * len(segment)
        position = 0
        segment_rewrite_count = 0
        while position < len(segment):
            matches = []
            for route in _routes_at(
                    model=model,
                    segment=segment,
                    position=position,
                    indexed=indexed):
                source = str(route["input_text"])
                if segment.startswith(source, position):
                    matches.append(route)
            if matches:
                longest = max(len(str(item["input_text"]))
                              for item in matches)
                longest_matches = [
                    item for item in matches
                    if len(str(item["input_text"])) == longest]
                outputs = {str(item["output_text"])
                           for item in longest_matches}
                if len(outputs) != 1:
                    decision = "UNKNOWN_AMBIGUOUS_LONGEST_ROUTE"
                    break
                route = longest_matches[0]
                source = str(route["input_text"])
                values.append(str(route["output_text"]))
                for cursor in range(position, position + len(source)):
                    covered[cursor] = 1
                position += len(source)
                segment_rewrite_count += 1
                continue
            values.append(segment[position])
            position += 1
        if decision != "CANDIDATE":
            break
        output_segment = "".join(values)
        if proposal_required == 0 and output_segment != segment:
            decision = "UNKNOWN_IDENTITY_SEGMENT_CHANGED"
            break
        if proposal_required == 1 and segment_rewrite_count == 0:
            decision = "UNKNOWN_OPEN_OBLIGATION"
            break
        if proposal_required == 1 and any(
                not covered[index] and scalar not in stable_scalars
                for index, scalar in enumerate(segment)):
            decision = "UNKNOWN_UNCERTIFIED_COPY"
            break
        output_segments.append(output_segment)
        rewrite_count += segment_rewrite_count
    output_text = input_text
    structure_mismatch = 0
    if decision == "CANDIDATE":
        rebuilt = []
        for ordinal, segment in enumerate(output_segments):
            rebuilt.append(segment)
            if ordinal < len(layout["raw_tokens"]):
                rebuilt.append(layout["raw_tokens"][ordinal])
        candidate = "".join(rebuilt)
        try:
            output_layout = localization_structure_layout_for_tokens(
                candidate, tuple(tokens))
            structure_mismatch = int(
                output_layout["raw_tokens"] != layout["raw_tokens"])
        except BroadQaExternalDataError:
            structure_mismatch = 1
        if structure_mismatch:
            decision = "UNKNOWN_STRUCTURE_TOKEN_MISMATCH"
        elif candidate == input_text:
            decision = "UNKNOWN_IDENTITY_PROPOSAL"
        else:
            output_text = candidate
    payload = {
        "decision": decision,
        "output_text": output_text,
        "partial_commit_count": 0,
        "rewrite_count": rewrite_count,
        "structure_token_mismatch_count": structure_mismatch,
    }
    return {**payload, "result_sha256": _sha256(
        canonical_json_bytes(payload))}


def _projection_indexes(
        *,
        held_out_family: str,
        projections: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
    """构造 label-blind held feature 与 TRAIN-only neutral authority。"""
    held_feature = {}
    training_support: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    for item in projections:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation projection 非对象")
        family = item.get("source_family")
        pair_id = item.get("pair_id")
        surface_sha256 = item.get("neutral_surface_sha256")
        if (family not in V5_SOURCE_FAMILIES
                or not isinstance(pair_id, str) or len(pair_id) != 64
                or not isinstance(surface_sha256, str)
                or len(surface_sha256) != 64):
            raise BroadQaExternalDataError(
                "v7 cross-source transformation projection identity 漂移")
        if family == held_out_family:
            previous = held_feature.setdefault(pair_id, surface_sha256)
            if previous != surface_sha256:
                raise BroadQaExternalDataError(
                    "v7 cross-source transformation held feature 冲突")
            continue
        output_sha256 = item.get("output_sha256")
        if not isinstance(output_sha256, str) or len(output_sha256) != 64:
            raise BroadQaExternalDataError(
                "v7 cross-source transformation TRAIN output SHA 漂移")
        training_support[surface_sha256][str(family)].add(output_sha256)
    routes = {}
    ambiguous = 0
    insufficient = 0
    for surface_sha256, family_outputs in training_support.items():
        if len(family_outputs) < 2:
            insufficient += 1
            continue
        if any(len(outputs) != 1 for outputs in family_outputs.values()):
            ambiguous += 1
            continue
        outputs = {next(iter(values))
                   for values in family_outputs.values()}
        if len(outputs) != 1:
            ambiguous += 1
            continue
        routes[surface_sha256] = next(iter(outputs))
    return held_feature, routes, {
        "ambiguous_or_conflict_surface_count": ambiguous,
        "authority_route_count": len(routes),
        "held_feature_count": len(held_feature),
        "insufficient_family_surface_count": insufficient,
    }


def _outcome(observation: dict[str, object], output_text: str) -> str:
    """在 proposal 与 authorization 冻结后计算 TRAIN-only outcome。"""
    if output_text == observation["output_text"]:
        return "EXACT"
    if output_text == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _family_loso(
        *,
        held_out_family: str,
        observations: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        observation_by_id: dict[str, dict[str, object]],
        models: dict[str, dict[str, object]],
        projections: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], dict[str, object], Counter, Counter]:
    """执行一个 held-out family 的独立模型共识与 neutral 鉴权。"""
    held_feature, authority_routes, authority_summary = _projection_indexes(
        held_out_family=held_out_family,
        projections=projections,
    )
    pre_outcomes = Counter()
    final_outcomes = Counter()
    decisions = Counter()
    counters = Counter()
    result_rows = []
    held_plans = tuple(
        plan for plan in plans
        if plan["source_family"] == held_out_family)
    for plan in held_plans:
        observation = observation_by_id[str(plan["observation_id"])]
        family_candidates = {}
        for family in V5_SOURCE_FAMILIES:
            if family == held_out_family:
                continue
            indexed_result = _execute_family_model(
                observation=observation,
                plan=plan,
                model=models[family],
                indexed=True,
            )
            reference_result = _execute_family_model(
                observation=observation,
                plan=plan,
                model=models[family],
                indexed=False,
            )
            counters["indexed_reference_mismatch_count"] += int(
                indexed_result != reference_result)
            counters["partial_commit_count"] += int(
                indexed_result["partial_commit_count"])
            counters["structure_token_execution_mismatch_count"] += int(
                indexed_result["structure_token_mismatch_count"])
            decisions[str(indexed_result["decision"])] += 1
            if indexed_result["decision"] == "CANDIDATE":
                family_candidates[family] = str(
                    indexed_result["output_text"])
        output_support = Counter(family_candidates.values())
        consensus_values = [
            output for output, count in output_support.items()
            if count >= 2]
        consensus = (
            consensus_values[0]
            if len(consensus_values) == 1 and len(output_support) == 1
            else None)
        pre_output = str(observation["input_text"]) \
            if consensus is None else consensus
        pre_outcome = _outcome(observation, pre_output)
        pre_outcomes[pre_outcome] += 1
        pair_id = observation.get("source_pair_id")
        neutral_surface_sha256 = held_feature.get(str(pair_id))
        authority_output_sha256 = authority_routes.get(
            str(neutral_surface_sha256))
        authorized = int(
            consensus is not None
            and authority_output_sha256 == _text_sha256(consensus))
        final_output = consensus if authorized else str(
            observation["input_text"])
        final_outcome = _outcome(observation, final_output)
        final_outcomes[final_outcome] += 1
        counters["neutral_feature_available_count"] += int(
            neutral_surface_sha256 is not None)
        counters["neutral_authority_route_available_count"] += int(
            authority_output_sha256 is not None)
        counters["neutral_authorized_count"] += authorized
        result_rows.append({
            "family_candidate_count": len(family_candidates),
            "family_consensus_available": int(consensus is not None),
            "family_consensus_output_sha256": (
                "" if consensus is None else _text_sha256(consensus)),
            "final_outcome": final_outcome,
            "final_output_sha256": _text_sha256(final_output),
            "held_out_observation_id": observation["observation_id"],
            "neutral_authority_route_available": int(
                authority_output_sha256 is not None),
            "neutral_authorized": authorized,
            "neutral_feature_available": int(
                neutral_surface_sha256 is not None),
            "pre_authorization_outcome": pre_outcome,
        })
    for observation in observations:
        if (observation.get("source_family") != held_out_family
                or observation.get("identity_preservation") != 1
                or not observation.get("structure_tokens")):
            continue
        counters["identity_probe_count"] += 1
        for family in V5_SOURCE_FAMILIES:
            if family == held_out_family:
                continue
            left = _execute_family_model(
                observation=observation, plan=None,
                model=models[family], indexed=True)
            right = _execute_family_model(
                observation=observation, plan=None,
                model=models[family], indexed=False)
            counters["indexed_reference_mismatch_count"] += int(left != right)
            counters["identity_false_change_count"] += int(
                left["output_text"] != observation["input_text"])
            counters["partial_commit_count"] += int(
                left["partial_commit_count"])
            counters["structure_token_execution_mismatch_count"] += int(
                left["structure_token_mismatch_count"])
    result_rows.sort(key=lambda item: str(item[
        "held_out_observation_id"]))
    identity = {
        "held_out_source_family": held_out_family,
        "target_policy_scope": CROSS_SOURCE_TRANSFORMATION_TARGET_SCOPE,
    }
    record = {
        **identity,
        "decision_counts": {
            key: decisions[key] for key in sorted(decisions)},
        "family_consensus_outcome_counts": {
            key: pre_outcomes[key]
            for key in ("EXACT", "UNKNOWN", "WRONG")},
        "format_version": 1,
        "identity_false_change_count": counters[
            "identity_false_change_count"],
        "identity_probe_count": counters["identity_probe_count"],
        "indexed_reference_mismatch_count": counters[
            "indexed_reference_mismatch_count"],
        "loso_id": _record_id(identity),
        "neutral_authority_summary": authority_summary,
        "neutral_authorized_count": counters[
            "neutral_authorized_count"],
        "neutral_authorized_outcome_counts": {
            key: final_outcomes[key]
            for key in ("EXACT", "UNKNOWN", "WRONG")},
        "partial_commit_count": counters["partial_commit_count"],
        "record_kind": CROSS_SOURCE_TRANSFORMATION_LOSO_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "structure_token_execution_mismatch_count": counters[
            "structure_token_execution_mismatch_count"],
        "variable_plan_count": len(held_plans),
    }
    for key in ("EXACT", "UNKNOWN", "WRONG"):
        counters[f"pre_{key.lower()}_count"] = pre_outcomes[key]
        counters[f"final_{key.lower()}_count"] = final_outcomes[key]
    counters["variable_plan_count"] = len(held_plans)
    return record, authority_summary, counters, decisions


def _stage_record(
        *,
        stage: str,
        outcomes: dict[str, int],
        ) -> dict[str, object]:
    """形成一个固定 gate 前后的 aggregate stage 记录。"""
    identity = {
        "stage": stage,
        "target_policy_scope": CROSS_SOURCE_TRANSFORMATION_TARGET_SCOPE,
    }
    wrong = outcomes["WRONG"]
    exact = outcomes["EXACT"]
    return {
        **identity,
        "capability_status": (
            "PASS" if wrong == 0 and exact > 0
            else "NE" if wrong == 0 else "FAIL"),
        "format_version": 1,
        "outcome_counts": outcomes,
        "record_kind": CROSS_SOURCE_TRANSFORMATION_STAGE_KIND,
        "stage_id": _record_id(identity),
    }


def derive_cross_source_transformation_feasibility(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        neutral_projections: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生 family models、两阶段账本、四向 LOSO 与总审计。"""
    observation_by_id, _plan_by_id = _indexes(observations, plans)
    models, model_records = _derive_models(
        observations=observations,
        fragments=fragments,
        plans=plans,
        observation_by_id=observation_by_id,
    )
    loso_records = []
    aggregate = Counter()
    authority_by_family = {}
    decisions = Counter()
    for held_out_family in V5_SOURCE_FAMILIES:
        record, authority, counters, family_decisions = _family_loso(
            held_out_family=held_out_family,
            observations=observations,
            plans=plans,
            observation_by_id=observation_by_id,
            models=models,
            projections=neutral_projections,
        )
        loso_records.append(record)
        authority_by_family[held_out_family] = authority
        aggregate.update(counters)
        decisions.update(family_decisions)
    if aggregate["variable_plan_count"] != len(plans):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation plan denominator 未闭合")
    pre_outcomes = {
        key: aggregate[f"pre_{key.lower()}_count"]
        for key in ("EXACT", "UNKNOWN", "WRONG")}
    final_outcomes = {
        key: aggregate[f"final_{key.lower()}_count"]
        for key in ("EXACT", "UNKNOWN", "WRONG")}
    stage_records = (
        _stage_record(
            stage=_PRE_AUTHORIZATION_STAGE,
            outcomes=pre_outcomes,
        ),
        _stage_record(stage=_FINAL_STAGE, outcomes=final_outcomes),
    )
    facility_pass = int(
        aggregate["indexed_reference_mismatch_count"] == 0
        and aggregate["partial_commit_count"] == 0
        and aggregate["structure_token_execution_mismatch_count"] == 0
        and aggregate["identity_false_change_count"] == 0)
    capability_outcome = (
        "PASS_NONZERO_AUTHORIZED_EXACT"
        if facility_pass and final_outcomes["WRONG"] == 0
        and final_outcomes["EXACT"] > 0
        else "NE_ZERO_AUTHORIZED_EXACT"
        if facility_pass and final_outcomes["WRONG"] == 0
        else "FAIL_HARD_GATE")
    model_records = tuple(sorted(
        model_records, key=lambda item: str(item["source_family"])))
    loso_records = tuple(sorted(
        loso_records,
        key=lambda item: str(item["held_out_source_family"])))
    return model_records, stage_records, loso_records, {
        "atom_scalar_max": TRANSFORMATION_ATOM_SCALAR_MAX,
        "capability_outcome": capability_outcome,
        "decision_counts": {
            key: decisions[key] for key in sorted(decisions)},
        "facility_outcome": "PASS" if facility_pass else "FAIL",
        "family_model_count": len(model_records),
        "family_model_route_counts": {
            str(item["source_family"]): int(item["route_count"])
            for item in model_records},
        "final_outcome_counts": final_outcomes,
        "identity_false_change_count": aggregate[
            "identity_false_change_count"],
        "identity_probe_count": aggregate["identity_probe_count"],
        "indexed_reference_mismatch_count": aggregate[
            "indexed_reference_mismatch_count"],
        "neutral_authority_by_held_family": authority_by_family,
        "neutral_authorized_count": aggregate[
            "neutral_authorized_count"],
        "partial_commit_count": aggregate["partial_commit_count"],
        "pre_authorization_outcome_counts": pre_outcomes,
        "structure_token_execution_mismatch_count": aggregate[
            "structure_token_execution_mismatch_count"],
        "variable_plan_count": aggregate["variable_plan_count"],
    }


__all__ = [
    "CROSS_SOURCE_TRANSFORMATION_LOSO_KIND",
    "CROSS_SOURCE_TRANSFORMATION_MODEL_KIND",
    "CROSS_SOURCE_TRANSFORMATION_STAGE_KIND",
    "CROSS_SOURCE_TRANSFORMATION_TARGET_SCOPE",
    "TRANSFORMATION_ATOM_SCALAR_MAX",
    "derive_cross_source_transformation_feasibility",
]
