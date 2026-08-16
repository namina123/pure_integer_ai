"""派生 recovery-v10 五个 TRAIN family 的重叠与冲突账。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_FIVE_FAMILY_AUDIT_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
    "MIXXX_PROJECT",
    "MUMBLE_PROJECT",
)
V10_FIVE_FAMILY_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_CENSUS_V1")
V10_FIVE_FAMILY_PAIRWISE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_PAIRWISE_OVERLAP_V1")
V10_FIVE_FAMILY_COLLISION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_SOURCE_INPUT_COLLISION_V1")
V10_FIVE_FAMILY_AUDIT_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_AUDIT_CENSUS_V1")

_PAIR_FEATURES = (
    "contains_han_both",
    "equal_length",
    "identity_preservation",
    "single_han_difference",
    "structure_equal",
    "training_eligible",
    "v8_training_eligible",
    "within_scalar_limit",
)


def _sha256(payload: bytes) -> str:
    """返回规范记录或表面identity的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _translation(value: object, *, label: str) -> str:
    """从统一Qt/gettext locale record提取唯一translation。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            f"v10 five-family {label} locale schema 漂移")
    values = [value.get("translation"), value.get("msgstr")]
    strings = [item for item in values if isinstance(item, str)]
    if len(strings) != 1:
        raise BroadQaExternalDataError(
            f"v10 five-family {label} translation schema 漂移")
    return strings[0]


def _features(pair: dict[str, object]) -> dict[str, int]:
    """兼容source-pair与既有Observation的冻结资格事实。"""
    eligibility = pair.get("eligibility")
    source = (
        eligibility.get("pair_features")
        if isinstance(eligibility, dict) else pair)
    if not isinstance(source, dict):
        raise BroadQaExternalDataError(
            "v10 five-family pair features 漂移")
    values = {}
    for name in _PAIR_FEATURES:
        value = source.get(name)
        if type(value) is not int or value not in (0, 1):
            raise BroadQaExternalDataError(
                "v10 five-family pair feature 漂移")
        values[name] = value
    return values


def _raw_blob_sets(manifest: dict[str, object]) -> dict[str, set[str]]:
    """从v8/v10 source-pack manifest提取license与locale blob集合。"""
    values = {"license": set(), "locale": set()}
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v10 five-family source-pack file commitment 漂移")
        role = item.get("role")
        sha = item.get("sha256")
        if role in {"V8_SOURCE_RAW_LICENSE_BLOB", "V10_SOURCE_RAW_LICENSE_BLOB"}:
            label = "license"
        elif role in {"V8_SOURCE_RAW_LOCALE_BLOB", "V10_SOURCE_RAW_LOCALE_BLOB"}:
            label = "locale"
        else:
            continue
        if not isinstance(sha, str) or len(sha) != 64:
            raise BroadQaExternalDataError(
                "v10 five-family raw blob identity 漂移")
        values[label].add(sha)
    if not values["license"] or not values["locale"]:
        raise BroadQaExternalDataError(
            "v10 five-family raw blob inventory 漂移")
    return values


def _index(
        family: str,
        pairs: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """建立单family完整分母、结构特征与精确presence索引。"""
    counts = Counter()
    source_texts = set()
    inputs = set()
    outputs = set()
    input_outputs = set()
    mappings = set()
    full_pairs = set()
    source_input_outputs: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_input_structures: dict[
        tuple[str, str], set[tuple[tuple[str, ...], tuple[str, ...]]]
    ] = defaultdict(set)
    identity_keys = set()
    changed_keys = set()
    for pair in pairs:
        if pair.get("source_family") != family:
            raise BroadQaExternalDataError(
                "v10 five-family pair family 漂移")
        source = pair.get("official_source_text")
        hant_tokens = pair.get("zh_hant_structure_tokens")
        hans_tokens = pair.get("zh_hans_structure_tokens")
        if (not isinstance(source, str)
                or not isinstance(hant_tokens, list)
                or not isinstance(hans_tokens, list)
                or any(not isinstance(item, str)
                       for item in hant_tokens + hans_tokens)):
            raise BroadQaExternalDataError(
                "v10 five-family pair surface schema 漂移")
        input_text = _translation(pair.get("zh_hant"), label="zh_hant")
        output_text = _translation(pair.get("zh_hans"), label="zh_hans")
        features = _features(pair)
        for name, value in features.items():
            counts[name] += value
        key = (source, input_text)
        structures = (tuple(hant_tokens), tuple(hans_tokens))
        source_texts.add(source)
        inputs.add(input_text)
        outputs.add(output_text)
        input_outputs.add((input_text, output_text))
        mappings.add((source, input_text, output_text))
        full_pairs.add((source, input_text, output_text, *structures))
        source_input_outputs[key].add(output_text)
        source_input_structures[key].add(structures)
        if input_text == output_text:
            identity_keys.add(key)
        else:
            changed_keys.add(key)
    if not pairs:
        raise BroadQaExternalDataError(
            "v10 five-family empty family")
    return {
        "changed_keys": changed_keys,
        "counts": counts,
        "full_pairs": full_pairs,
        "identity_keys": identity_keys,
        "input_outputs": input_outputs,
        "inputs": inputs,
        "mappings": mappings,
        "outputs": outputs,
        "pair_count": len(pairs),
        "source_input_outputs": source_input_outputs,
        "source_input_structures": source_input_structures,
        "source_texts": source_texts,
    }


def _family_record(
        family: str,
        *,
        manifest: dict[str, object],
        index: dict[str, object],
        ) -> dict[str, object]:
    """形成单family完整pair、资格、结构和冲突分母。"""
    counts = index["counts"]
    pair_count = int(index["pair_count"])
    eligible = int(counts["v8_training_eligible"])
    identity = int(counts["identity_preservation"])
    source_input_outputs = index["source_input_outputs"]
    return {
        "changed_pair_count": pair_count - identity,
        "contains_han_both_pair_count": int(counts["contains_han_both"]),
        "equal_length_pair_count": int(counts["equal_length"]),
        "format_version": 1,
        "identity_pair_count": identity,
        "pair_record_count": pair_count,
        "record_kind": V10_FIVE_FAMILY_RECORD_KIND,
        "single_han_difference_pair_count": int(
            counts["single_han_difference"]),
        "source_family": family,
        "source_pack_manifest_sha256": manifest["manifest_sha256"],
        "structure_equal_pair_count": int(counts["structure_equal"]),
        "structure_unequal_pair_count": (
            pair_count - int(counts["structure_equal"])),
        "training_eligible_pair_count": eligible,
        "training_excluded_pair_count": pair_count - eligible,
        "unique_exact_mapping_count": len(index["mappings"]),
        "unique_input_count": len(index["inputs"]),
        "unique_input_output_count": len(index["input_outputs"]),
        "unique_official_source_count": len(index["source_texts"]),
        "unique_output_count": len(index["outputs"]),
        "unique_source_input_count": len(source_input_outputs),
        "variable_length_pair_count": pair_count - int(counts["equal_length"]),
        "within_family_source_input_conflict_count": sum(
            len(outputs) > 1 for outputs in source_input_outputs.values()),
        "within_scalar_limit_pair_count": int(counts["within_scalar_limit"]),
    }


def _pairwise_record(
        left_family: str,
        right_family: str,
        *,
        manifests: dict[str, dict[str, object]],
        indexes: dict[str, dict[str, object]],
        blobs: dict[str, dict[str, set[str]]],
        ) -> dict[str, object]:
    """形成两个family的lineage、mapping、结构与冲突重叠账。"""
    left = indexes[left_family]
    right = indexes[right_family]
    shared_keys = set(left["source_input_outputs"]).intersection(
        right["source_input_outputs"])
    full_overlap = len(left["full_pairs"].intersection(right["full_pairs"]))
    smaller = min(len(left["full_pairs"]), len(right["full_pairs"]))
    left_raw = manifests[left_family].get("raw_source")
    right_raw = manifests[right_family].get("raw_source")
    if not isinstance(left_raw, dict) or not isinstance(right_raw, dict):
        raise BroadQaExternalDataError(
            "v10 five-family raw lineage 漂移")
    return {
        "commit_equal": int(left_raw.get("commit") == right_raw.get("commit")),
        "exact_full_subset_copy": int(full_overlap == smaller),
        "exact_mapping_overlap_count": len(
            left["mappings"].intersection(right["mappings"])),
        "format_version": 1,
        "full_pair_semantic_overlap_count": full_overlap,
        "full_pair_semantic_overlap_denominator": smaller,
        "identity_changed_mixed_source_input_count": sum(
            key in left["identity_keys"].union(right["identity_keys"])
            and key in left["changed_keys"].union(right["changed_keys"])
            for key in shared_keys),
        "input_output_mapping_overlap_count": len(
            left["input_outputs"].intersection(right["input_outputs"])),
        "left_family": left_family,
        "left_pair_count": int(left["pair_count"]),
        "license_blob_overlap_count": len(
            blobs[left_family]["license"].intersection(
                blobs[right_family]["license"])),
        "locale_blob_overlap_count": len(
            blobs[left_family]["locale"].intersection(
                blobs[right_family]["locale"])),
        "official_source_text_overlap_count": len(
            left["source_texts"].intersection(right["source_texts"])),
        "record_kind": V10_FIVE_FAMILY_PAIRWISE_RECORD_KIND,
        "repository_equal": int(
            left_raw.get("repository") == right_raw.get("repository")),
        "right_family": right_family,
        "right_pair_count": int(right["pair_count"]),
        "root_tree_equal": int(
            left_raw.get("root_tree") == right_raw.get("root_tree")),
        "source_input_conflicting_output_count": sum(
            left["source_input_outputs"][key]
            != right["source_input_outputs"][key]
            for key in shared_keys),
        "source_input_key_overlap_count": len(shared_keys),
        "source_input_structure_variant_count": sum(
            left["source_input_structures"][key]
            != right["source_input_structures"][key]
            for key in shared_keys),
        "zh_hans_output_overlap_count": len(
            left["outputs"].intersection(right["outputs"])),
        "zh_hant_input_overlap_count": len(
            left["inputs"].intersection(right["inputs"])),
    }


def _set_commitment(values: set[object]) -> str:
    """对family局部output或structure集合形成不公开表面的承诺。"""
    identities = sorted(_sha256(canonical_json_line({"value": value}))
                        for value in values)
    return _sha256(canonical_json_line({"identities": identities}))


def _collision_records(
        indexes: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """形成跨family共享source+input的哈希化冲突ledger。"""
    presence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for family in V10_FIVE_FAMILY_AUDIT_FAMILIES:
        for key in indexes[family]["source_input_outputs"]:
            presence[key].add(family)
    records = []
    for source, input_text in sorted(
            key for key, families in presence.items() if len(families) >= 2):
        key = (source, input_text)
        families = tuple(sorted(presence[key]))
        output_sets = {
            family: indexes[family]["source_input_outputs"][key]
            for family in families
        }
        structure_sets = {
            family: indexes[family]["source_input_structures"][key]
            for family in families
        }
        all_outputs = set().union(*output_sets.values())
        output_conflict = len({frozenset(value)
                               for value in output_sets.values()}) > 1
        identity_present = input_text in all_outputs
        changed_present = any(value != input_text for value in all_outputs)
        structure_variant = len({frozenset(value)
                                 for value in structure_sets.values()}) > 1
        if not (output_conflict or (identity_present and changed_present)
                or structure_variant):
            continue
        record = {
            "changed_output_present": int(changed_present),
            "cross_family_output_conflict": int(output_conflict),
            "families": list(families),
            "family_count": len(families),
            "family_output_set_commitments": {
                family: _set_commitment(output_sets[family])
                for family in families
            },
            "family_structure_set_commitments": {
                family: _set_commitment(structure_sets[family])
                for family in families
            },
            "format_version": 1,
            "identity_output_present": int(identity_present),
            "record_kind": V10_FIVE_FAMILY_COLLISION_RECORD_KIND,
            "source_input_sha256": _sha256(canonical_json_line({
                "input_text": input_text,
                "official_source_text": source,
            })),
            "structure_variant": int(structure_variant),
            "unique_output_count": len(all_outputs),
        }
        records.append({
            **record,
            "collision_record_sha256": _sha256(canonical_json_line(record)),
        })
    return tuple(records)


def _presence_count(
        indexes: dict[str, dict[str, object]],
        key: str,
        ) -> int:
    """计算至少出现在两个独立family中的unique identity数量。"""
    presence = defaultdict(int)
    for family in V10_FIVE_FAMILY_AUDIT_FAMILIES:
        for value in indexes[family][key]:
            presence[value] += 1
    return sum(value >= 2 for value in presence.values())


def derive_normalization_recovery_v10_five_family_audit(
        manifests: dict[str, dict[str, object]],
        pairs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """重算五family完整分母、十组pairwise overlap和collision ledger。"""
    families = set(V10_FIVE_FAMILY_AUDIT_FAMILIES)
    if set(manifests) != families or set(pairs) != families:
        raise BroadQaExternalDataError(
            "v10 five-family predecessor inventory 漂移")
    for family in V10_FIVE_FAMILY_AUDIT_FAMILIES:
        manifest = manifests[family]
        if (manifest.get("source_family") != family
                or manifest.get("source_family_vote_count") != 1
                or not isinstance(manifest.get("manifest_sha256"), str)):
            raise BroadQaExternalDataError(
                "v10 five-family source-pack fields 漂移")
    indexes = {
        family: _index(family, pairs[family])
        for family in V10_FIVE_FAMILY_AUDIT_FAMILIES
    }
    blobs = {
        family: _raw_blob_sets(manifests[family])
        for family in V10_FIVE_FAMILY_AUDIT_FAMILIES
    }
    family_records = tuple(_family_record(
        family,
        manifest=manifests[family],
        index=indexes[family],
    ) for family in V10_FIVE_FAMILY_AUDIT_FAMILIES)
    pairwise = []
    for index, left in enumerate(V10_FIVE_FAMILY_AUDIT_FAMILIES):
        for right in V10_FIVE_FAMILY_AUDIT_FAMILIES[index + 1:]:
            pairwise.append(_pairwise_record(
                left,
                right,
                manifests=manifests,
                indexes=indexes,
                blobs=blobs,
            ))
    pairwise.sort(key=lambda item: (item["left_family"], item["right_family"]))
    collision_records = _collision_records(indexes)
    hard_failures = sum(
        int(item[key])
        for item in pairwise
        for key in (
            "commit_equal",
            "exact_full_subset_copy",
            "locale_blob_overlap_count",
            "repository_equal",
            "root_tree_equal",
        ))
    summary = {
        "changed_pair_count": sum(
            int(item["changed_pair_count"]) for item in family_records),
        "cross_family_output_conflict_key_count": sum(
            int(item["cross_family_output_conflict"])
            for item in collision_records),
        "exact_mapping_shared_identity_count": _presence_count(
            indexes, "mappings"),
        "family_count": len(V10_FIVE_FAMILY_AUDIT_FAMILIES),
        "family_vote_count": len(V10_FIVE_FAMILY_AUDIT_FAMILIES),
        "full_pair_semantic_shared_identity_count": _presence_count(
            indexes, "full_pairs"),
        "hard_independence_failure_count": hard_failures,
        "identity_changed_mixed_key_count": sum(
            int(item["identity_output_present"])
            and int(item["changed_output_present"])
            for item in collision_records),
        "identity_pair_count": sum(
            int(item["identity_pair_count"]) for item in family_records),
        "input_output_shared_identity_count": _presence_count(
            indexes, "input_outputs"),
        "license_blob_pairwise_overlap_count": sum(
            int(item["license_blob_overlap_count"]) for item in pairwise),
        "locale_blob_pairwise_overlap_count": sum(
            int(item["locale_blob_overlap_count"]) for item in pairwise),
        "official_source_shared_identity_count": _presence_count(
            indexes, "source_texts"),
        "pair_record_count": sum(
            int(item["pair_record_count"]) for item in family_records),
        "pair_surface_public_git_count": 0,
        "pairwise_record_count": len(pairwise),
        "source_input_collision_record_count": len(collision_records),
        "source_input_shared_identity_count": _presence_count(
            indexes, "source_input_outputs"),
        "structure_equal_pair_count": sum(
            int(item["structure_equal_pair_count"])
            for item in family_records),
        "structure_unequal_pair_count": sum(
            int(item["structure_unequal_pair_count"])
            for item in family_records),
        "structure_variant_key_count": sum(
            int(item["structure_variant"]) for item in collision_records),
        "training_eligible_pair_count": sum(
            int(item["training_eligible_pair_count"])
            for item in family_records),
        "training_excluded_pair_count": sum(
            int(item["training_excluded_pair_count"])
            for item in family_records),
        "within_family_source_input_conflict_count": sum(
            int(item["within_family_source_input_conflict_count"])
            for item in family_records),
    }
    return family_records, tuple(pairwise), collision_records, summary


__all__ = [
    "V10_FIVE_FAMILY_AUDIT_CENSUS_KIND",
    "V10_FIVE_FAMILY_AUDIT_FAMILIES",
    "derive_normalization_recovery_v10_five_family_audit",
]
