"""构造 recovery-v7 source commitment identity replay program。

完整 source identity 只授权已见来源条目的确定重放。family、policy、commitment、
fragment kind 与 input 必须全部相等；任何缺失、多 output、重叠或结构漂移都返回
原输入。该设施不把已见 source replay 冒充 unseen-family 语言迁移。
"""
from __future__ import annotations

from collections import Counter, defaultdict
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


SOURCE_REPLAY_CONFLICT_REPRESENTATION_KIND = (
    "NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_CONFLICT_REPRESENTATION_V1")
SOURCE_REPLAY_TARGET_SCOPE = (
    "SEEN_SOURCE_COMMITMENT_IDENTITY_REPLAY_V1")


def _sha256(payload: bytes) -> str:
    """返回规范 identity、record 或 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 source replay text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 id。"""
    return _sha256(canonical_json_bytes(identity))


def source_commitment_identity_sha256(
        observation: dict[str, object],
        ) -> str:
    """核验 observation 来源域并返回完整 commitment identity SHA。"""
    family = observation.get("source_family")
    policy = observation.get("source_policy_scope")
    commitment = observation.get("source_commitment")
    if (family not in V5_SOURCE_FAMILIES
            or policy != V5_SOURCE_POLICY_BY_FAMILY[family]
            or not isinstance(commitment, dict) or not commitment):
        raise BroadQaExternalDataError(
            "v7 source replay observation commitment 漂移")
    return _sha256(canonical_json_bytes(commitment))


def _observation_index(
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """按 observation id 建立带 source identity 的权威索引。"""
    values = {}
    for observation in observations:
        observation_id = observation.get("observation_id") \
            if isinstance(observation, dict) else None
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or observation_id in values):
            raise BroadQaExternalDataError(
                "v7 source replay observation identity 漂移")
        source_identity = source_commitment_identity_sha256(observation)
        values[observation_id] = {
            "observation": observation,
            "source_commitment_sha256": source_identity,
        }
    if not values:
        raise BroadQaExternalDataError("v7 source replay observations 为空")
    return values


def _fragment_index(
        fragments: tuple[dict[str, object], ...],
        observations: dict[str, dict[str, object]],
        ) -> dict[str, dict[str, object]]:
    """核验 fragment 与 observation 来源域闭合。"""
    values = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id") \
            if isinstance(fragment, dict) else None
        observation_id = fragment.get("observation_id") \
            if isinstance(fragment, dict) else None
        owner = observations.get(str(observation_id))
        if (not isinstance(fragment_id, str) or len(fragment_id) != 64
                or fragment_id in values or owner is None
                or fragment.get("source_family")
                != owner["observation"]["source_family"]
                or fragment.get("source_policy_scope")
                != owner["observation"]["source_policy_scope"]):
            raise BroadQaExternalDataError(
                "v7 source replay fragment/observation 漂移")
        values[fragment_id] = fragment
    if not values:
        raise BroadQaExternalDataError("v7 source replay fragments 为空")
    return values


def _route_identity(
        *,
        observation: dict[str, object],
        source_commitment_sha256: str,
        fragment_kind: str,
        input_text: str,
        ) -> dict[str, object]:
    """形成必须完全相等的 seen-source route identity。"""
    return {
        "fragment_kind": fragment_kind,
        "input_text": input_text,
        "source_commitment_sha256": source_commitment_sha256,
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
        "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
    }


def _conflict_groups(
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """选择且核验 TRAIN output conflict groups。"""
    values = []
    for group in groups:
        if (not isinstance(group, dict)
                or group.get("disposition") != "CONFLICT_DEFER"
                or group.get("authority_basis") != "OUTPUT_CONFLICT"):
            continue
        variants = group.get("output_variants")
        if (group.get("fragment_kind") not in {
                "CONTEXT_HUNK", "EDIT_CORE", "WHOLE_INPUT"}
                or not isinstance(group.get("group_id"), str)
                or len(group["group_id"]) != 64
                or not isinstance(group.get("input_text"), str)
                or not group["input_text"]
                or not isinstance(variants, list) or len(variants) < 2):
            raise BroadQaExternalDataError(
                "v7 source replay conflict group 漂移")
        values.append(group)
    if not values:
        raise BroadQaExternalDataError(
            "v7 source replay conflict groups 为空")
    return tuple(values)


def derive_source_replay_program(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从 TRAIN conflict groups 派生 seen-source route program 与无 surface 表示。"""
    observation_by_id = _observation_index(observations)
    fragment_by_id = _fragment_index(fragments, observation_by_id)
    identity_veto_keys = frozenset(
        (
            str(item["source_family"]),
            str(item["source_policy_scope"]),
            source_commitment_identity_sha256(item),
            str(item["input_text"]),
        )
        for item in observations
        if item.get("identity_preservation") == 1)
    route_outputs: dict[
        tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    route_fragments: dict[
        tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    group_route_keys: dict[str, set[tuple[str, str, str, str, str]]] = (
        defaultdict(set))
    fragment_route_key = {}
    conflict_groups = _conflict_groups(groups)
    for group in conflict_groups:
        fragment_kind = str(group["fragment_kind"])
        input_text = str(group["input_text"])
        for variant in group["output_variants"]:
            output_text = variant.get("output_text") \
                if isinstance(variant, dict) else None
            fragment_ids = variant.get("fragment_ids") \
                if isinstance(variant, dict) else None
            if (not isinstance(output_text, str)
                    or not isinstance(fragment_ids, list)
                    or any(not isinstance(value, str) or len(value) != 64
                           for value in fragment_ids)):
                raise BroadQaExternalDataError(
                    "v7 source replay conflict variant 漂移")
            for fragment_id in fragment_ids:
                fragment = fragment_by_id.get(fragment_id)
                if (fragment is None
                        or fragment.get("fragment_kind") != fragment_kind
                        or fragment.get("input_text") != input_text
                        or fragment.get("output_text") != output_text):
                    raise BroadQaExternalDataError(
                        "v7 source replay group/fragment 未闭合")
                owner = observation_by_id[str(fragment["observation_id"])]
                observation = owner["observation"]
                identity = _route_identity(
                    observation=observation,
                    source_commitment_sha256=owner[
                        "source_commitment_sha256"],
                    fragment_kind=fragment_kind,
                    input_text=input_text,
                )
                key = (
                    str(identity["source_family"]),
                    str(identity["source_policy_scope"]),
                    str(identity["source_commitment_sha256"]),
                    fragment_kind,
                    input_text,
                )
                route_outputs[key].add(output_text)
                route_fragments[key].add(fragment_id)
                group_route_keys[str(group["group_id"])].add(key)
                previous = fragment_route_key.setdefault(fragment_id, key)
                if previous != key:
                    raise BroadQaExternalDataError(
                        "v7 source replay fragment route identity 冲突")
    routes = []
    route_audit = {}
    for key in sorted(route_outputs):
        outputs = sorted(route_outputs[key])
        output_hashes = [_text_sha256(value) for value in outputs]
        route_identity = {
            "fragment_kind": key[3],
            "input_sha256": _text_sha256(key[4]),
            "source_commitment_sha256": key[2],
            "source_family": key[0],
            "source_policy_scope": key[1],
            "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
        }
        route_id = _record_id(route_identity)
        route_audit[key] = {
            **route_identity,
            "fragment_ids_sha256": _sha256(canonical_json_bytes(
                sorted(route_fragments[key]))),
            "output_count": len(outputs),
            "output_routes_sha256": _sha256(canonical_json_bytes(
                output_hashes)),
            "route_id": route_id,
            "support_count": len(route_fragments[key]),
        }
        if len(outputs) == 1:
            routes.append({
                **route_audit[key],
                "input_text": key[4],
                "output_text": outputs[0],
            })
    representations = []
    counters = Counter()
    for group in conflict_groups:
        keys = sorted(group_route_keys[str(group["group_id"])])
        records = [route_audit[key] for key in keys]
        unique_count = sum(item["output_count"] == 1 for item in records)
        ambiguous_count = len(records) - unique_count
        identity = {
            "predecessor_group_id": group["group_id"],
            "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
        }
        representations.append({
            **identity,
            "ambiguous_source_identity_route_count": ambiguous_count,
            "execution_allowed": 0,
            "format_version": 1,
            "fragment_kind": group["fragment_kind"],
            "input_length": len(str(group["input_text"])),
            "input_sha256": _text_sha256(str(group["input_text"])),
            "record_kind": SOURCE_REPLAY_CONFLICT_REPRESENTATION_KIND,
            "representation_id": _record_id(identity),
            "route_count": len(records),
            "routes_sha256": _sha256(canonical_json_bytes(records)),
            "source_identity_defeater_required": 1,
            "status": (
                "ALL_SOURCE_IDENTITY_ROUTES_UNIQUE"
                if ambiguous_count == 0
                else "PARTIAL_AMBIGUOUS_SOURCE_IDENTITY_DEFER"),
            "unique_source_identity_route_count": unique_count,
        })
        counters["conflict_count"] += 1
        counters["all_routes_unique_conflict_count"] += int(
            ambiguous_count == 0)
        counters["ambiguous_conflict_count"] += int(ambiguous_count > 0)
        counters["unique_route_count"] += unique_count
        counters["ambiguous_route_count"] += ambiguous_count
        counters[f"kind:{group['fragment_kind']}:total"] += 1
        counters[f"kind:{group['fragment_kind']}:all_unique"] += int(
            ambiguous_count == 0)
        counters[f"kind:{group['fragment_kind']}:ambiguous"] += int(
            ambiguous_count > 0)
    routes.sort(key=lambda item: str(item["route_id"]))
    buckets: dict[tuple[str, str, str, str], list[dict[str, object]]] = (
        defaultdict(list))
    for route in routes:
        input_text = str(route["input_text"])
        buckets[(
            str(route["source_family"]),
            str(route["source_policy_scope"]),
            str(route["source_commitment_sha256"]),
            input_text[0],
        )].append(route)
    frozen_buckets = {
        key: tuple(sorted(values, key=lambda item: (
            -len(str(item["input_text"])),
            str(item["fragment_kind"]),
            str(item["route_id"]))))
        for key, values in buckets.items()
    }
    reference_by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for route in routes:
        reference_by_family[str(route["source_family"])].append(route)
    self_replay = Counter()
    for fragment_id, key in fragment_route_key.items():
        outputs = route_outputs[key]
        fragment = fragment_by_id[fragment_id]
        if len(outputs) != 1:
            self_replay["UNKNOWN"] += 1
        elif next(iter(outputs)) == fragment["output_text"]:
            self_replay["EXACT"] += 1
        else:
            self_replay["WRONG"] += 1
    representations.sort(key=lambda item: str(item["representation_id"]))
    schema_families: dict[tuple[str, ...], set[str]] = defaultdict(set)
    commitment_families: dict[str, set[str]] = defaultdict(set)
    for item in observation_by_id.values():
        observation = item["observation"]
        schema_families[tuple(sorted(
            str(key) for key in observation["source_commitment"]))].add(
                str(observation["source_family"]))
        commitment_families[item["source_commitment_sha256"]].add(
            str(observation["source_family"]))
    program_payload = {
        "route_count": len(routes),
        "route_ids": [str(item["route_id"]) for item in routes],
        "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
    }
    program = {
        "buckets": frozen_buckets,
        "identity_veto_keys": identity_veto_keys,
        "program_sha256": _sha256(canonical_json_bytes(program_payload)),
        "reference_routes_by_family": {
            family: tuple(values)
            for family, values in sorted(reference_by_family.items())},
        "reference_routes": tuple(routes),
        "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
    }
    return program, tuple(representations), {
        "all_routes_unique_conflict_count": counters[
            "all_routes_unique_conflict_count"],
        "ambiguous_conflict_count": counters["ambiguous_conflict_count"],
        "ambiguous_route_count": counters["ambiguous_route_count"],
        "by_fragment_kind": {
            kind: {
                "all_unique": counters[f"kind:{kind}:all_unique"],
                "ambiguous": counters[f"kind:{kind}:ambiguous"],
                "total": counters[f"kind:{kind}:total"],
            }
            for kind in ("CONTEXT_HUNK", "EDIT_CORE", "WHOLE_INPUT")
        },
        "conflict_count": counters["conflict_count"],
        "exact_commitment_cross_family_count": sum(
            len(families) > 1 for families in commitment_families.values()),
        "seen_source_self_replay_counts": {
            key: self_replay[key] for key in ("EXACT", "UNKNOWN", "WRONG")},
        "source_commitment_schema_count": len(schema_families),
        "source_commitment_schema_cross_family_count": sum(
            len(families) > 1 for families in schema_families.values()),
        "unique_route_count": counters["unique_route_count"],
    }


__all__ = [
    "SOURCE_REPLAY_CONFLICT_REPRESENTATION_KIND",
    "SOURCE_REPLAY_TARGET_SCOPE",
    "derive_source_replay_program",
    "source_commitment_identity_sha256",
]
