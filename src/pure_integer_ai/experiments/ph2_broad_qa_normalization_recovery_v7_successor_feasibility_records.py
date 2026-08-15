"""派生 recovery-v7 三项 TRAIN-only successor 可行性记录。

模块只消费调用方提供的既有四来源 TRAIN records，输出结构、正上下文与
source-policy replay 的不可执行投影。它不打开路径，不读取 VLC/Qt，也不生成
learner rule 或 runtime program。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


VARIABLE_STRUCTURE_PROJECTION_KIND = (
    "NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_PROJECTION_V1")
CONTEXT_SCOPED_LOCAL_PROJECTION_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONTEXT_SCOPED_LOCAL_PROJECTION_V1")
SOURCE_POLICY_REPLAY_PROJECTION_KIND = (
    "NORMALIZATION_RECOVERY_V7_SOURCE_POLICY_REPLAY_PROJECTION_V1")

VARIABLE_STRUCTURE_STATUS = "FEASIBLE_NARROW_IMPLEMENTATION_REQUIRED"
CONTEXT_SCOPED_LOCAL_STATUS = (
    "FEASIBLE_WITH_DEFER_AND_ATOMIC_COMMIT_IMPLEMENTATION_REQUIRED")
SOURCE_POLICY_REPLAY_STATUS = (
    "FEASIBLE_PARTIAL_CONTEXT_OR_SOURCE_IDENTITY_REQUIRED")


def _sha256(payload: bytes) -> str:
    """返回规范值或 TRAIN surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA，避免在 projection 中复制 TRAIN surface。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 feasibility text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从规范 projection identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _strict_int(value: object, *, label: str) -> int:
    """读取非负精确整数，并拒绝 bool。"""
    if type(value) is not int or value < 0:
        raise BroadQaExternalDataError(f"v7 feasibility {label} 非负整数非法")
    return value


def derive_variable_structure_projections(
        observations: Iterable[dict[str, object]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """按非空 structure signature 聚合 variable whole observations。"""
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    observation_count = 0
    for item in observations:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v7 variable structure observation 非对象")
        identity = item.get("identity_preservation")
        equal_length = item.get("equal_length")
        tokens = item.get("structure_tokens")
        if (type(identity) is not int or identity not in (0, 1)
                or type(equal_length) is not int
                or equal_length not in (0, 1)
                or not isinstance(tokens, list)
                or any(not isinstance(token, str) or not token
                       for token in tokens)):
            raise BroadQaExternalDataError(
                "v7 variable structure observation schema 漂移")
        if identity == 1 or equal_length == 1 or not tokens:
            continue
        observation_id = item.get("observation_id")
        source_family = item.get("source_family")
        input_text = item.get("input_text")
        output_text = item.get("output_text")
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or source_family not in V5_SOURCE_FAMILIES
                or not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text):
            raise BroadQaExternalDataError(
                "v7 variable structure observation identity 漂移")
        key = tuple(tokens)
        value = grouped.setdefault(key, {
            "family_counts": Counter(),
            "length_deltas": [],
            "observation_ids": [],
        })
        value["family_counts"][source_family] += 1
        value["length_deltas"].append(len(output_text) - len(input_text))
        value["observation_ids"].append(observation_id)
        observation_count += 1
    values = []
    for tokens, item in sorted(grouped.items()):
        family_counts = item["family_counts"]
        observation_ids = sorted(item["observation_ids"])
        deltas = item["length_deltas"]
        identity = {
            "contract": "VARIABLE_STRUCTURE_TRANSFER",
            "structure_tokens": list(tokens),
        }
        values.append({
            **identity,
            "atomic_whole_commit_required": 1,
            "cross_family_support": int(len(family_counts) >= 2),
            "execution_allowed": 0,
            "format_version": 1,
            "identity_and_conflict_veto_required": 1,
            "length_delta_max": max(deltas),
            "length_delta_min": min(deltas),
            "observation_count": len(observation_ids),
            "observation_ids_sha256": _sha256(canonical_json_bytes(
                observation_ids)),
            "projection_id": _record_id(identity),
            "record_kind": VARIABLE_STRUCTURE_PROJECTION_KIND,
            "segment_obligation_learning_required": 1,
            "source_family_counts": {
                family: family_counts[family]
                for family in sorted(family_counts)},
            "source_families": sorted(family_counts),
            "structure_token_preservation_required": 1,
        })
    if not values or sum(int(item["observation_count"]) for item in values) \
            != observation_count:
        raise BroadQaExternalDataError(
            "v7 variable structure projection 分账未闭合")
    cross = [item for item in values if item["cross_family_support"] == 1]
    return tuple(values), {
        "cross_family_observation_count": sum(
            int(item["observation_count"]) for item in cross),
        "cross_family_signature_count": len(cross),
        "max_source_family_count": max(
            len(item["source_families"]) for item in values),
        "status": VARIABLE_STRUCTURE_STATUS,
        "structure_signature_count": len(values),
        "structured_variable_observation_count": observation_count,
    }


def derive_identity_inputs(
        identity_observations: Iterable[dict[str, object]],
        ) -> frozenset[str]:
    """从 frozen identity bucket 提取 exact-input hard veto 集。"""
    values = set()
    for item in identity_observations:
        input_text = item.get("input_text") if isinstance(item, dict) else None
        output_text = item.get("output_text") if isinstance(item, dict) else None
        if (not isinstance(input_text, str) or not input_text
                or output_text != input_text):
            raise BroadQaExternalDataError(
                "v7 feasibility identity observation 漂移")
        values.add(input_text)
    if not values:
        raise BroadQaExternalDataError("v7 feasibility identity veto 为空")
    return frozenset(values)


def _positive_context_record(
        context_signature: dict[str, object],
        evidence_items: list[dict[str, object]],
        ) -> dict[str, object]:
    """把 SUPPORT context surface 压缩为可审计 hash/长度记录。"""
    signature_id = context_signature.get("context_signature_id")
    left = context_signature.get("left_context")
    right = context_signature.get("right_context")
    left_boundary = context_signature.get("left_boundary")
    right_boundary = context_signature.get("right_boundary")
    if (not isinstance(signature_id, str) or len(signature_id) != 64
            or not isinstance(left, str) or not isinstance(right, str)
            or type(left_boundary) is not int or left_boundary not in (0, 1)
            or type(right_boundary) is not int
            or right_boundary not in (0, 1)):
        raise BroadQaExternalDataError(
            "v7 feasibility positive context schema 漂移")
    families = sorted({str(item["source_family"])
                       for item in evidence_items})
    evidence_ids = sorted(str(item["evidence_id"])
                          for item in evidence_items)
    return {
        "context_signature_id": signature_id,
        "left_boundary": left_boundary,
        "left_context_length": len(left),
        "left_context_sha256": _text_sha256(left),
        "nonempty_surface_context": int(bool(left or right)),
        "right_boundary": right_boundary,
        "right_context_length": len(right),
        "right_context_sha256": _text_sha256(right),
        "source_families": families,
        "support_evidence_ids": evidence_ids,
    }


def derive_context_scoped_local_projections(
        *,
        target_rules: Iterable[dict[str, object]],
        evidence: Iterable[dict[str, object]],
        identity_inputs: frozenset[str],
        conflict_inputs: frozenset[str],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """投影 local rule 的 SUPPORT-derived 正上下文与 hard veto 合同。"""
    rules = {}
    evidence_owner = {}
    for rule in target_rules:
        if not isinstance(rule, dict) or rule.get("fragment_kind") \
                == "WHOLE_INPUT":
            continue
        candidate_id = rule.get("candidate_id")
        positive_ids = rule.get("positive_evidence_ids")
        if (not isinstance(candidate_id, str) or len(candidate_id) != 64
                or candidate_id in rules
                or rule.get("candidate_scope_kind") != "TARGET_CROSS_FAMILY"
                or rule.get("rule_class") not in {"EDIT_CORE", "CONTEXT_HUNK"}
                or not isinstance(positive_ids, list) or not positive_ids
                or any(not isinstance(item, str) or len(item) != 64
                       for item in positive_ids)
                or not isinstance(rule.get("defeater_ids"), list)
                or not isinstance(rule.get("source_families"), list)):
            raise BroadQaExternalDataError(
                "v7 feasibility target local rule 漂移")
        rules[candidate_id] = rule
        for evidence_id in positive_ids:
            if evidence_id in evidence_owner:
                raise BroadQaExternalDataError(
                    "v7 feasibility positive evidence 多 owner")
            evidence_owner[evidence_id] = candidate_id
    if not rules:
        raise BroadQaExternalDataError("v7 feasibility target local rule 为空")
    captured: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in evidence:
        evidence_id = item.get("evidence_id") if isinstance(item, dict) else None
        candidate_id = evidence_owner.get(evidence_id)
        if candidate_id is None:
            continue
        if (item.get("stance") != "SUPPORT"
                or item.get("candidate_id") != candidate_id
                or item.get("source_family") not in V5_SOURCE_FAMILIES
                or not isinstance(item.get("context_signature"), dict)):
            raise BroadQaExternalDataError(
                "v7 feasibility positive evidence 漂移")
        captured[candidate_id].append(item)
    values = []
    counters = Counter()
    for candidate_id, rule in sorted(rules.items()):
        items = captured.get(candidate_id, [])
        expected_ids = sorted(str(item) for item in rule["positive_evidence_ids"])
        actual_ids = sorted(str(item["evidence_id"]) for item in items)
        if expected_ids != actual_ids:
            raise BroadQaExternalDataError(
                "v7 feasibility positive evidence 未闭合")
        contexts_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
        signatures = {}
        for item in items:
            signature = item["context_signature"]
            signature_id = signature.get("context_signature_id")
            if (not isinstance(signature_id, str) or len(signature_id) != 64
                    or (signature_id in signatures
                        and signatures[signature_id] != signature)):
                raise BroadQaExternalDataError(
                    "v7 feasibility context signature identity 漂移")
            signatures[signature_id] = signature
            contexts_by_id[signature_id].append(item)
        contexts = [
            _positive_context_record(signatures[key], contexts_by_id[key])
            for key in sorted(signatures)]
        source_families = sorted({str(item["source_family"])
                                  for item in items})
        input_text = rule.get("input_text")
        output_text = rule.get("output_text")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str)):
            raise BroadQaExternalDataError(
                "v7 feasibility local rule surface 漂移")
        nonempty_count = sum(int(item["nonempty_surface_context"])
                             for item in contexts)
        reasons = []
        if len(source_families) < 2:
            reasons.append("CROSS_FAMILY_SUPPORT_INSUFFICIENT")
        if nonempty_count == 0:
            reasons.append("NO_POSITIVE_SURFACE_CONTEXT")
        if not rule["defeater_ids"]:
            reasons.append("NO_NEGATIVE_DEFEATER")
        identity = {
            "candidate_id": candidate_id,
            "contract": "CONTEXT_SCOPED_LOCAL_TRANSFER",
            "predecessor_rule_id": rule["rule_id"],
        }
        values.append({
            **identity,
            "atomic_whole_commit_required": 1,
            "conflict_veto_required": int(input_text in conflict_inputs),
            "defer_reasons": reasons,
            "defeater_count": len(rule["defeater_ids"]),
            "execution_allowed": 0,
            "format_version": 1,
            "identity_veto_required": int(input_text in identity_inputs),
            "input_length": len(input_text),
            "input_sha256": _text_sha256(input_text),
            "output_length": len(output_text),
            "output_sha256": _text_sha256(output_text),
            "positive_context_count": len(contexts),
            "positive_contexts": contexts,
            "positive_evidence_count": len(items),
            "projection_id": _record_id(identity),
            "record_kind": CONTEXT_SCOPED_LOCAL_PROJECTION_KIND,
            "rule_class": rule["rule_class"],
            "source_families": source_families,
            "status": (
                "REPRESENTATION_FEASIBLE" if not reasons else "DEFERRED"),
        })
        counters["rule_count"] += 1
        counters["support_closed"] += 1
        counters["cross_family_support"] += int(len(source_families) >= 2)
        counters["positive_nonempty_context"] += int(nonempty_count > 0)
        counters["cross_family_and_nonempty_context"] += int(
            len(source_families) >= 2 and nonempty_count > 0)
        counters["multi_context_signature"] += int(len(contexts) >= 2)
        counters["has_defeater"] += int(bool(rule["defeater_ids"]))
        counters["exact_input_identity_veto_needed"] += int(
            input_text in identity_inputs)
        counters["exact_input_conflict_veto_needed"] += int(
            input_text in conflict_inputs)
        counters["representation_feasible"] += int(not reasons)
    return tuple(values), {
        **{key: counters[key] for key in (
            "cross_family_and_nonempty_context",
            "cross_family_support",
            "exact_input_conflict_veto_needed",
            "exact_input_identity_veto_needed",
            "has_defeater",
            "multi_context_signature",
            "positive_nonempty_context",
            "representation_feasible",
            "rule_count",
            "support_closed",
        )},
        "status": CONTEXT_SCOPED_LOCAL_STATUS,
    }


def derive_source_policy_replay_projections(
        conflicts: Iterable[dict[str, object]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            dict[str, object],
            frozenset[str],
        ]:
    """投影显式 family/policy fragment replay，并分离需上下文的冲突。"""
    values = []
    conflict_inputs = set()
    counts = Counter()
    kind_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in conflicts:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError("v7 source replay conflict 非对象")
        input_text = item.get("input_text")
        if not isinstance(input_text, str) or not input_text:
            raise BroadQaExternalDataError(
                "v7 source replay conflict input 漂移")
        conflict_inputs.add(input_text)
        if item.get("conflict_kind") != "TRAIN_OUTPUT_CONFLICT":
            continue
        variants = item.get("variants")
        fragment_kind = item.get("fragment_kind")
        conflict_id = item.get("conflict_id")
        if (not isinstance(variants, list) or len(variants) < 2
                or fragment_kind not in {
                    "CONTEXT_HUNK", "EDIT_CORE", "WHOLE_INPUT"}
                or not isinstance(conflict_id, str) or len(conflict_id) != 64):
            raise BroadQaExternalDataError(
                "v7 source replay TRAIN conflict schema 漂移")
        family_routes: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
        variant_records = []
        for variant in variants:
            output_text = variant.get("output_text") \
                if isinstance(variant, dict) else None
            families = variant.get("source_families") \
                if isinstance(variant, dict) else None
            scopes = variant.get("source_policy_scopes") \
                if isinstance(variant, dict) else None
            fragment_ids = variant.get("fragment_ids") \
                if isinstance(variant, dict) else None
            if (not isinstance(output_text, str)
                    or not isinstance(families, list) or not families
                    or any(family not in V5_SOURCE_FAMILIES
                           for family in families)
                    or not isinstance(scopes, list)
                    or not isinstance(fragment_ids, list)
                    or any(not isinstance(value, str) or len(value) != 64
                           for value in fragment_ids)):
                raise BroadQaExternalDataError(
                    "v7 source replay variant schema 漂移")
            output_sha = _text_sha256(output_text)
            for family in families:
                policy = V5_SOURCE_POLICY_BY_FAMILY[family]
                if policy not in scopes:
                    raise BroadQaExternalDataError(
                        "v7 source replay family/policy 未闭合")
                family_routes[family].add((output_sha, len(output_text), policy))
            variant_records.append({
                "fragment_ids_sha256": _sha256(canonical_json_bytes(
                    sorted(fragment_ids))),
                "output_length": len(output_text),
                "output_sha256": output_sha,
                "source_families": sorted(families),
                "source_policy_scopes": sorted(scopes),
                "support_count": _strict_int(
                    variant.get("support_count"), label="variant support_count"),
            })
        replayable = (
            len(family_routes) >= 2
            and all(len(routes) == 1 for routes in family_routes.values()))
        route_records = []
        for family in sorted(family_routes):
            routes = sorted(family_routes[family])
            route_records.append({
                "route_count": len(routes),
                "routes": [
                    {
                        "output_length": route[1],
                        "output_sha256": route[0],
                        "source_policy_scope": route[2],
                    }
                    for route in routes
                ],
                "source_family": family,
            })
        identity = {
            "conflict_id": conflict_id,
            "contract": "SOURCE_POLICY_REPLAY",
        }
        status = (
            "FRAGMENT_KEY_REPLAYABLE" if replayable
            else "CONTEXT_OR_SOURCE_IDENTITY_REQUIRED")
        values.append({
            **identity,
            "execution_allowed": 0,
            "explicit_source_family_required": 1,
            "explicit_source_policy_scope_required": 1,
            "family_routes": route_records,
            "format_version": 1,
            "fragment_kind": fragment_kind,
            "input_length": len(input_text),
            "input_sha256": _text_sha256(input_text),
            "projection_id": _record_id(identity),
            "record_kind": SOURCE_POLICY_REPLAY_PROJECTION_KIND,
            "status": status,
            "unscoped_execution_allowed": 0,
            "variants": sorted(
                variant_records,
                key=lambda value: (
                    str(value["output_sha256"]),
                    tuple(value["source_families"]),
                ),
            ),
        })
        counts["train_output_conflict_count"] += 1
        counts["variant_total"] += len(variants)
        counts["replayable_conflict_count"] += int(replayable)
        counts["context_or_source_identity_required_count"] += int(
            not replayable)
        kind_counts[fragment_kind]["total"] += 1
        kind_counts[fragment_kind]["replayable"] += int(replayable)
        kind_counts[fragment_kind]["context_or_source_identity_required"] \
            += int(not replayable)
    if not values:
        raise BroadQaExternalDataError(
            "v7 source policy replay TRAIN conflict 为空")
    return tuple(sorted(values, key=lambda value: str(value["projection_id"]))), {
        "by_fragment_kind": {
            kind: {
                name: kind_counts[kind][name]
                for name in (
                    "context_or_source_identity_required",
                    "replayable",
                    "total",
                )
            }
            for kind in sorted(kind_counts)
        },
        "context_or_source_identity_required_count": counts[
            "context_or_source_identity_required_count"],
        "replayable_conflict_count": counts["replayable_conflict_count"],
        "status": SOURCE_POLICY_REPLAY_STATUS,
        "train_output_conflict_count": counts[
            "train_output_conflict_count"],
        "variant_total": counts["variant_total"],
    }, frozenset(conflict_inputs)


__all__ = [
    "CONTEXT_SCOPED_LOCAL_PROJECTION_KIND",
    "CONTEXT_SCOPED_LOCAL_STATUS",
    "SOURCE_POLICY_REPLAY_PROJECTION_KIND",
    "SOURCE_POLICY_REPLAY_STATUS",
    "VARIABLE_STRUCTURE_PROJECTION_KIND",
    "VARIABLE_STRUCTURE_STATUS",
    "derive_context_scoped_local_projections",
    "derive_identity_inputs",
    "derive_source_policy_replay_projections",
    "derive_variable_structure_projections",
]
