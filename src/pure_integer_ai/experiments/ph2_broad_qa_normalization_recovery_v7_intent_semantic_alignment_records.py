"""派生 recovery-v7 neutral source intent/semantic alignment census。

本模块只消费调用方已严格回读的 TRAIN observations、variable plans、neutral
source rows 与 ConceptNet alias pack。它先按 source structure layout 屏蔽占位符和
标记，再形成离散 alias/POS/structure/punctuation facts，并比较三类跨来源签名。
该 census 不创建 learner、candidate、runtime 或 formal evaluation。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
    localization_structure_token_category,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    CONCEPTNET_ALIAS_EVIDENCE_KIND,
    CONCEPTNET_ALIAS_ROUTE_KIND,
    neutral_source_phrases,
    neutral_source_units,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records import (
    derive_cross_source_transformation_consensus_proposals,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


INTENT_SEMANTIC_FACT_FAMILY_KIND = (
    "NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_FACT_FAMILY_V1")
INTENT_SEMANTIC_FAMILY_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_FAMILY_CENSUS_V1")
INTENT_SEMANTIC_SIGNATURE_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_SIGNATURE_CENSUS_V1")
INTENT_SEMANTIC_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_LOSO_V1")
INTENT_SEMANTIC_TARGET_SCOPE = (
    "CROSS_PRODUCT_NEUTRAL_INTENT_SEMANTIC_ALIGNMENT_V1")

SIGNATURE_ALIAS_LEXICAL = "ALIAS_LEXICAL_STRUCTURE_V1"
SIGNATURE_ALIAS_PUNCTUATION = "ALIAS_LEXICAL_PUNCTUATION_STRUCTURE_V1"
SIGNATURE_POS_ONLY = "POS_ONLY_STRUCTURE_V1"
SIGNATURE_STRUCTURE_ONLY = "STRUCTURE_PUNCTUATION_ONLY_V1"
AUTHORIZATION_SIGNATURE_MODE = SIGNATURE_ALIAS_PUNCTUATION

_SOURCE_ORDER = (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)
_SIGNATURE_MODES = (
    SIGNATURE_ALIAS_LEXICAL,
    SIGNATURE_ALIAS_PUNCTUATION,
    SIGNATURE_POS_ONLY,
    SIGNATURE_STRUCTURE_ONLY,
)
_POS_TAGS = frozenset(("a", "n", "r", "v"))


def _sha256(payload: bytes) -> str:
    """返回规范 identity 或 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _valid_digest(value: object) -> bool:
    """判断值是否为规范小写 SHA-256。"""
    return (isinstance(value, str) and len(value) == 64
            and all(item in "0123456789abcdef" for item in value))


def _text_sha256(value: str) -> str:
    """返回 transient source surface 的 UTF-8 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError(
            "v7 intent semantic source surface 非字符串")
    return _sha256(value.encode("utf-8"))


def _alias_catalog(
        evidence: tuple[dict[str, object], ...],
        routes: tuple[dict[str, object], ...],
        *,
        clean_inventory: set[str],
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, tuple[str, ...]],
            dict[str, object],
        ]:
    """从 structure-filtered phrase inventory 构造 alias 与 POS 目录。"""
    route_by_phrase = {}
    unique_route_count = 0
    for item in routes:
        phrase = item.get("english_surface") \
            if isinstance(item, dict) else None
        route_id = item.get("alias_route_id") \
            if isinstance(item, dict) else None
        unique = item.get("unique_chinese_surface") \
            if isinstance(item, dict) else None
        if (item.get("record_kind") != CONCEPTNET_ALIAS_ROUTE_KIND
                or not isinstance(phrase, str) or not phrase
                or not _valid_digest(route_id)
                or type(unique) is not int or unique not in (0, 1)):
            raise BroadQaExternalDataError(
                "v7 intent semantic alias route 漂移")
        if phrase not in clean_inventory:
            continue
        if phrase in route_by_phrase:
            raise BroadQaExternalDataError(
                "v7 intent semantic alias phrase 重复")
        route_by_phrase[phrase] = {
            "alias_route_id": route_id,
            "unique_chinese_surface": unique,
        }
        unique_route_count += unique
    pos_by_phrase: dict[str, set[str]] = defaultdict(set)
    evidence_count = 0
    for item in evidence:
        phrase = item.get("english_surface") \
            if isinstance(item, dict) else None
        suffix = item.get("english_suffix") \
            if isinstance(item, dict) else None
        if (item.get("record_kind") != CONCEPTNET_ALIAS_EVIDENCE_KIND
                or not isinstance(phrase, str)
                or not isinstance(suffix, list)
                or any(not isinstance(value, str) or not value
                       for value in suffix)):
            raise BroadQaExternalDataError(
                "v7 intent semantic alias evidence 漂移")
        if phrase not in route_by_phrase:
            continue
        evidence_count += 1
        if suffix and suffix[0] in _POS_TAGS:
            pos_by_phrase[phrase].add(str(suffix[0]))
    frozen_pos = {
        phrase: tuple(sorted(pos_by_phrase.get(phrase, ())))
        for phrase in route_by_phrase
    }
    specified = sum(bool(values) for values in frozen_pos.values())
    single = sum(len(values) == 1 for values in frozen_pos.values())
    multi = sum(len(values) > 1 for values in frozen_pos.values())
    specified_unique = sum(
        bool(frozen_pos[phrase])
        and route_by_phrase[phrase]["unique_chinese_surface"] == 1
        for phrase in route_by_phrase)
    return route_by_phrase, frozen_pos, {
        "clean_alias_evidence_count": evidence_count,
        "clean_alias_route_count": len(route_by_phrase),
        "multi_specified_pos_route_count": multi,
        "single_specified_pos_route_count": single,
        "specified_pos_route_count": specified,
        "specified_pos_unique_chinese_route_count": specified_unique,
        "unique_chinese_route_count": unique_route_count,
    }


def _punctuation_codepoints(
        segments: tuple[str, ...],
        ) -> tuple[int, ...]:
    """把 source text 中非词元、非空白 delimiter 保留为整数序。"""
    values = []
    for segment in segments:
        for scalar in segment:
            if scalar == "_" or scalar.isspace() or scalar.isalnum():
                continue
            values.append(ord(scalar))
    return tuple(values)


def _structure_spacing_profile(
        segments: tuple[str, ...],
        ) -> tuple[tuple[int, int], ...]:
    """记录每个 source structure token 两侧是否直接存在空白。"""
    return tuple((
        int(bool(segments[index]) and segments[index][-1].isspace()),
        int(bool(segments[index + 1])
            and segments[index + 1][0].isspace()),
    ) for index in range(len(segments) - 1))


def _matched_alias_facts(
        segments: tuple[str, ...],
        *,
        route_by_phrase: dict[str, dict[str, object]],
        pos_by_phrase: dict[str, tuple[str, ...]],
        ) -> dict[str, object]:
    """在每个 text segment 内最长优先匹配一至四单元 alias。"""
    route_ids = []
    pos_roles = []
    unique_flags = []
    unit_count = 0
    unmatched = 0
    digit_units = 0
    for segment in segments:
        units = neutral_source_units(segment) if segment.strip() else ()
        unit_count += len(units)
        digit_units += sum(
            bool(unit) and all(scalar.isdecimal() for scalar in unit)
            for unit in units)
        cursor = 0
        while cursor < len(units):
            match = None
            for length in range(min(4, len(units) - cursor), 0, -1):
                phrase = " ".join(units[cursor:cursor + length])
                route = route_by_phrase.get(phrase)
                if route is not None:
                    match = (phrase, route, length)
                    break
            if match is None:
                unmatched += 1
                cursor += 1
                continue
            phrase, route, length = match
            route_ids.append(str(route["alias_route_id"]))
            tags = pos_by_phrase[phrase]
            pos_roles.append(tags if tags else ("UNSPECIFIED",))
            unique_flags.append(int(route["unique_chinese_surface"]))
            cursor += length
    return {
        "alias_pos_roles": tuple(pos_roles),
        "alias_route_ids": tuple(route_ids),
        "alias_unique_chinese_flags": tuple(unique_flags),
        "digit_unit_count": digit_units,
        "unit_count": unit_count,
        "unmatched_unit_count": unmatched,
    }


def _pair_material(
        *,
        observations: tuple[dict[str, object], ...],
        neutral_projections: tuple[dict[str, object], ...],
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        alias_evidence: tuple[dict[str, object], ...],
        alias_routes: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, dict[str, object]],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """核验 source/projection/observation，并派生 structure-aware pair facts。"""
    if set(rows_by_family) != set(_SOURCE_ORDER):
        raise BroadQaExternalDataError(
            "v7 intent semantic source family inventory 漂移")
    observation_by_pair = {}
    for item in observations:
        pair_id = item.get("source_pair_id") \
            if isinstance(item, dict) else None
        if (not _valid_digest(pair_id)
                or pair_id in observation_by_pair
                or item.get("source_family") not in _SOURCE_ORDER
                or not isinstance(item.get("output_text"), str)
                or not isinstance(item.get("structure_tokens"), list)):
            raise BroadQaExternalDataError(
                "v7 intent semantic observation/pair 漂移")
        observation_by_pair[str(pair_id)] = item
    projection_by_pair = {}
    for item in neutral_projections:
        pair_id = item.get("pair_id") if isinstance(item, dict) else None
        if (not _valid_digest(pair_id) or pair_id in projection_by_pair
                or item.get("source_family") not in _SOURCE_ORDER
                or not _valid_digest(item.get("neutral_surface_sha256"))
                or not _valid_digest(item.get("output_sha256"))):
            raise BroadQaExternalDataError(
                "v7 intent semantic neutral projection 漂移")
        projection_by_pair[str(pair_id)] = item
    clean_phrase_support: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    row_by_pair = {}
    for family in _SOURCE_ORDER:
        rows = rows_by_family[family]
        if family == THUNDERBIRD_SOURCE_FAMILY and rows:
            raise BroadQaExternalDataError(
                "v7 intent semantic Thunderbird neutral rows 非空")
        for row in rows:
            pair_id = row.get("pair_id") if isinstance(row, dict) else None
            surface = row.get("_neutral_surface") \
                if isinstance(row, dict) else None
            if (not _valid_digest(pair_id) or pair_id in row_by_pair
                    or row.get("source_family") != family
                    or not isinstance(surface, str) or not surface
                    or not _valid_digest(row.get("output_sha256"))):
                raise BroadQaExternalDataError(
                    "v7 intent semantic source row 漂移")
            observation = observation_by_pair.get(str(pair_id))
            projection = projection_by_pair.get(str(pair_id))
            if (observation is None or projection is None
                    or observation.get("source_family") != family
                    or projection.get("source_family") != family
                    or projection.get("neutral_surface_sha256")
                    != _text_sha256(surface)
                    or projection.get("output_sha256")
                    != row.get("output_sha256")
                    or _text_sha256(str(observation["output_text"]))
                    != row.get("output_sha256")):
                raise BroadQaExternalDataError(
                    "v7 intent semantic source/projection 对齐漂移")
            layout = localization_structure_layout(surface)
            phrases = set()
            for segment in layout["segments"]:
                if segment:
                    phrases.update(neutral_source_phrases(segment))
            row_by_pair[str(pair_id)] = row
            for phrase in phrases:
                clean_phrase_support[phrase][family].add(str(pair_id))
    if set(row_by_pair) != set(projection_by_pair):
        raise BroadQaExternalDataError(
            "v7 intent semantic source/projection denominator 漂移")
    route_by_phrase, pos_by_phrase, alias_summary = _alias_catalog(
        alias_evidence,
        alias_routes,
        clean_inventory=set(clean_phrase_support),
    )
    pair_facts = {}
    family_records = []
    for family in _SOURCE_ORDER:
        family_pairs = []
        matched_phrases = {
            phrase for phrase in route_by_phrase
            if family in clean_phrase_support.get(phrase, {})}
        for row in rows_by_family[family]:
            pair_id = str(row["pair_id"])
            surface = str(row["_neutral_surface"])
            observation = observation_by_pair[pair_id]
            layout = localization_structure_layout(surface)
            matched = _matched_alias_facts(
                tuple(layout["segments"]),
                route_by_phrase=route_by_phrase,
                pos_by_phrase=pos_by_phrase,
            )
            source_categories = tuple(
                localization_structure_token_category(token)
                for token in layout["structure_tokens"])
            observation_categories = tuple(
                localization_structure_token_category(str(token))
                for token in observation["structure_tokens"])
            facts = {
                **matched,
                "observation_structure_categories": observation_categories,
                "output_sha256": row["output_sha256"],
                "pair_id": pair_id,
                "punctuation_codepoints": _punctuation_codepoints(
                    tuple(layout["segments"])),
                "source_family": family,
                "source_structure_categories": source_categories,
                "structure_spacing_profile": _structure_spacing_profile(
                    tuple(layout["segments"])),
            }
            pair_facts[pair_id] = facts
            family_pairs.append(facts)
        identity = {
            "source_family": family,
            "target_scope": INTENT_SEMANTIC_TARGET_SCOPE,
        }
        family_records.append({
            **identity,
            "all_alias_specified_pos_pair_count": sum(
                bool(item["alias_route_ids"])
                and all(role != ("UNSPECIFIED",)
                        for role in item["alias_pos_roles"])
                for item in family_pairs),
            "any_unique_chinese_alias_pair_count": sum(
                any(item["alias_unique_chinese_flags"])
                for item in family_pairs),
            "clean_alias_pair_count": sum(
                bool(item["alias_route_ids"]) for item in family_pairs),
            "clean_matched_alias_phrase_count": len(matched_phrases),
            "clean_phrase_inventory_count": sum(
                family in support
                for support in clean_phrase_support.values()),
            "complete_alias_coverage_pair_count": sum(
                bool(item["alias_route_ids"])
                and item["unmatched_unit_count"] == 0
                for item in family_pairs),
            "digit_source_pair_count": sum(
                item["digit_unit_count"] > 0 for item in family_pairs),
            "family_census_id": _record_id(identity),
            "format_version": 1,
            "observation_structure_pair_count": sum(
                bool(item["observation_structure_categories"])
                for item in family_pairs),
            "projected_pair_count": len(family_pairs),
            "punctuated_source_pair_count": sum(
                bool(item["punctuation_codepoints"])
                for item in family_pairs),
            "record_kind": INTENT_SEMANTIC_FAMILY_CENSUS_KIND,
            "source_structure_pair_count": sum(
                bool(item["source_structure_categories"])
                for item in family_pairs),
            "specified_pos_pair_count": sum(
                any(role != ("UNSPECIFIED",)
                    for role in item["alias_pos_roles"])
                for item in family_pairs),
            "zero_alias_pair_count": sum(
                not item["alias_route_ids"] for item in family_pairs),
        })
    return pair_facts, tuple(family_records), {
        **alias_summary,
        "clean_phrase_inventory_count": len(clean_phrase_support),
        "pair_count": len(pair_facts),
        "structure_filtered_phrase_count": len(clean_phrase_support),
    }


def _signature_payload(
        facts: dict[str, object],
        *,
        mode: str,
        ) -> dict[str, object]:
    """按冻结 mode 形成不含 source family/product identity 的事实签名。"""
    common = {
        "digit_unit_count": facts["digit_unit_count"],
        "observation_structure_categories": facts[
            "observation_structure_categories"],
        "source_structure_categories": facts[
            "source_structure_categories"],
        "unit_count": facts["unit_count"],
        "unmatched_unit_count": facts["unmatched_unit_count"],
    }
    if mode in (SIGNATURE_ALIAS_LEXICAL, SIGNATURE_ALIAS_PUNCTUATION):
        common["alias_pos_roles"] = facts["alias_pos_roles"]
        common["alias_route_ids"] = facts["alias_route_ids"]
    elif mode == SIGNATURE_POS_ONLY:
        common["alias_count"] = len(facts["alias_route_ids"])
        common["alias_pos_roles"] = facts["alias_pos_roles"]
    elif mode != SIGNATURE_STRUCTURE_ONLY:
        raise BroadQaExternalDataError(
            "v7 intent semantic signature mode 非法")
    if mode in (SIGNATURE_ALIAS_PUNCTUATION, SIGNATURE_STRUCTURE_ONLY):
        common["punctuation_codepoints"] = facts[
            "punctuation_codepoints"]
        common["structure_spacing_profile"] = facts[
            "structure_spacing_profile"]
    return common


def _signature_census(
        pair_facts: dict[str, dict[str, object]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            dict[str, dict[str, dict[str, set[str]]]],
        ]:
    """统计四种离散签名的跨 family consensus/conflict。"""
    all_groups = {}
    records = []
    for mode in _SIGNATURE_MODES:
        groups: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set))
        pair_counts = Counter()
        for facts in pair_facts.values():
            signature = _record_id(_signature_payload(facts, mode=mode))
            family = str(facts["source_family"])
            groups[signature][family].add(str(facts["output_sha256"]))
            pair_counts[signature] += 1
            facts.setdefault("signatures", {})[mode] = signature
        cross = {
            signature: family_outputs
            for signature, family_outputs in groups.items()
            if len(family_outputs) >= 2
        }
        consensus = []
        conflict = 0
        for signature, family_outputs in cross.items():
            outputs = set()
            unique = True
            for values in family_outputs.values():
                unique = unique and len(values) == 1
                outputs.update(values)
            if unique and len(outputs) == 1:
                consensus.append({
                    "output_sha256": next(iter(outputs)),
                    "signature": signature,
                })
            else:
                conflict += 1
        identity = {
            "signature_mode": mode,
            "target_scope": INTENT_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "cross_family_conflict_count": conflict,
            "cross_family_consensus_count": len(consensus),
            "cross_family_consensus_set_sha256": _sha256(
                canonical_json_bytes(sorted(
                    consensus, key=lambda item: str(item["signature"])))),
            "cross_family_pair_count": sum(
                pair_counts[signature] for signature in cross),
            "cross_family_signature_count": len(cross),
            "format_version": 1,
            "record_kind": INTENT_SEMANTIC_SIGNATURE_CENSUS_KIND,
            "signature_census_id": _record_id(identity),
            "signature_count": len(groups),
        })
        all_groups[mode] = groups
    return tuple(records), all_groups


def _authority_routes(
        groups: dict[str, dict[str, set[str]]],
        *,
        held_out_family: str,
        ) -> dict[str, str]:
    """只从至少两个非 held family 的唯一同 output 形成 authority。"""
    routes = {}
    for signature, family_outputs in groups.items():
        training = {
            family: outputs for family, outputs in family_outputs.items()
            if family != held_out_family
        }
        if (len(training) < 2
                or any(len(outputs) != 1
                       for outputs in training.values())):
            continue
        outputs = {next(iter(values)) for values in training.values()}
        if len(outputs) == 1:
            routes[signature] = next(iter(outputs))
    return routes


def _loso_census(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        pair_facts: dict[str, dict[str, object]],
        signature_groups: dict[str, dict[str, dict[str, set[str]]]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """重建 14 个 frozen proposal，并测试 alias fact authority。"""
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    if (len(observation_by_id) != len(observations)
            or any(not _valid_digest(key) for key in observation_by_id)):
        raise BroadQaExternalDataError(
            "v7 intent semantic observation identity 漂移")
    proposals = derive_cross_source_transformation_consensus_proposals(
        observations=observations,
        fragments=fragments,
        plans=plans,
    )
    records = []
    aggregate = Counter()
    groups = signature_groups[AUTHORIZATION_SIGNATURE_MODE]
    for held_out_family in V5_SOURCE_FAMILIES:
        authority = _authority_routes(
            groups, held_out_family=held_out_family)
        counters = Counter()
        result_rows = []
        for proposal in proposals:
            if proposal["held_out_source_family"] != held_out_family:
                continue
            observation = observation_by_id[str(
                proposal["held_out_observation_id"])]
            consensus = str(proposal["proposal_output_text"])
            pair_id = str(proposal["source_pair_id"])
            facts = pair_facts.get(pair_id)
            signatures = facts.get("signatures") \
                if isinstance(facts, dict) else None
            signature = (
                str(signatures[AUTHORIZATION_SIGNATURE_MODE])
                if isinstance(signatures, dict) else "")
            authority_output = authority.get(signature) if signature else None
            authorized = int(
                authority_output == _text_sha256(consensus))
            final_output = consensus if authorized else str(
                observation["input_text"])
            pre_outcome = str(proposal["pre_authorization_outcome"])
            final_outcome = (
                "EXACT" if final_output == observation["output_text"]
                else "UNKNOWN" if final_output == observation["input_text"]
                else "WRONG")
            counters[f"pre_{pre_outcome.lower()}_count"] += 1
            counters[f"final_{final_outcome.lower()}_count"] += 1
            counters["proposal_count"] += 1
            counters["authority_route_available_count"] += int(
                authority_output is not None)
            counters["authorized_count"] += authorized
            counters["clean_alias_available_count"] += int(
                bool(facts) and bool(facts["alias_route_ids"]))
            counters["complete_alias_coverage_count"] += int(
                bool(facts) and bool(facts["alias_route_ids"])
                and facts["unmatched_unit_count"] == 0)
            counters["specified_pos_available_count"] += int(
                bool(facts) and any(role != ("UNSPECIFIED",)
                    for role in facts["alias_pos_roles"]))
            result_rows.append({
                "authorized": authorized,
                "authority_route_available": int(
                    authority_output is not None),
                "final_outcome": final_outcome,
                "held_out_observation_id": observation["observation_id"],
                "pre_outcome": pre_outcome,
                "proposal_output_sha256": _text_sha256(consensus),
                "signature": signature,
            })
        identity = {
            "held_out_source_family": held_out_family,
            "signature_mode": AUTHORIZATION_SIGNATURE_MODE,
            "target_scope": INTENT_SEMANTIC_TARGET_SCOPE,
        }
        record = {
            **identity,
            "authority_route_available_count": counters[
                "authority_route_available_count"],
            "authority_route_count": len(authority),
            "authorized_count": counters["authorized_count"],
            "clean_alias_available_count": counters[
                "clean_alias_available_count"],
            "complete_alias_coverage_count": counters[
                "complete_alias_coverage_count"],
            "final_outcome_counts": {
                key: counters[f"final_{key.lower()}_count"]
                for key in ("EXACT", "UNKNOWN", "WRONG")},
            "format_version": 1,
            "loso_id": _record_id(identity),
            "pre_outcome_counts": {
                key: counters[f"pre_{key.lower()}_count"]
                for key in ("EXACT", "UNKNOWN", "WRONG")},
            "proposal_count": counters["proposal_count"],
            "record_kind": INTENT_SEMANTIC_LOSO_KIND,
            "result_rows_sha256": _sha256(canonical_json_bytes(sorted(
                result_rows,
                key=lambda item: str(item["held_out_observation_id"])))),
            "specified_pos_available_count": counters[
                "specified_pos_available_count"],
        }
        records.append(record)
        aggregate.update(counters)
    records.sort(key=lambda item: str(item["held_out_source_family"]))
    pre = {
        key: aggregate[f"pre_{key.lower()}_count"]
        for key in ("EXACT", "UNKNOWN", "WRONG")}
    final = {
        key: aggregate[f"final_{key.lower()}_count"]
        for key in ("EXACT", "UNKNOWN", "WRONG")}
    return tuple(records), {
        "authority_route_available_count": aggregate[
            "authority_route_available_count"],
        "authorized_count": aggregate["authorized_count"],
        "clean_alias_available_count": aggregate[
            "clean_alias_available_count"],
        "complete_alias_coverage_count": aggregate[
            "complete_alias_coverage_count"],
        "final_outcome_counts": final,
        "pre_outcome_counts": pre,
        "proposal_count": aggregate["proposal_count"],
        "specified_pos_available_count": aggregate[
            "specified_pos_available_count"],
    }


def _fact_family_records(
        *,
        family_records: tuple[dict[str, object], ...],
        alias_summary: dict[str, object],
        signature_records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """形成 available 与缺来源语义边界的固定 fact-family 账本。"""
    family_totals = Counter()
    for record in family_records:
        for key in (
                "clean_alias_pair_count",
                "digit_source_pair_count",
                "observation_structure_pair_count",
                "punctuated_source_pair_count",
                "source_structure_pair_count",
                "specified_pos_pair_count"):
            family_totals[key] += int(record[key])
    alias_signature = next(
        item for item in signature_records
        if item["signature_mode"] == SIGNATURE_ALIAS_PUNCTUATION)
    facts = (
        ("LEXICAL_ALIAS_IDENTITY", "AVAILABLE_CONFLICTED",
         int(alias_summary["clean_alias_route_count"]),
         "CONCEPTNET_SYNONYM_ROUTE_AMBIGUITY_PRESERVED"),
        ("PART_OF_SPEECH_CANDIDATE", "AVAILABLE_CONFLICTED",
         int(alias_summary["specified_pos_route_count"]),
         "CONCEPTNET_URI_SUFFIX_POS_CANDIDATE"),
        ("NEUTRAL_SOURCE_STRUCTURE_SLOT", "AVAILABLE",
         family_totals["source_structure_pair_count"],
         "NEUTRAL_SOURCE_LAYOUT_PRESERVES_SLOT_CATEGORY_AND_ORDER"),
        ("OBSERVATION_STRUCTURE_SLOT", "AVAILABLE",
         family_totals["observation_structure_pair_count"],
         "OBSERVATION_LEDGER_PRESERVES_SLOT_CATEGORY_AND_ORDER"),
        ("SOURCE_PUNCTUATION_PROFILE", "AVAILABLE",
         family_totals["punctuated_source_pair_count"],
         "SOURCE_DELIMITER_CODEPOINTS_ARE_DETERMINISTIC"),
        ("SOURCE_QUANTITY_MARKER", "AVAILABLE_PARTIAL",
         family_totals["digit_source_pair_count"],
         "SOURCE_DECIMAL_UNITS_DO_NOT_ASSIGN_SEMANTIC_ROLES"),
        ("CROSS_FAMILY_ALIAS_SIGNATURE", "AVAILABLE_CONFLICTED",
         int(alias_signature["cross_family_consensus_count"]),
         "NONZERO_CONSENSUS_COEXISTS_WITH_EXPLICIT_CONFLICTS"),
        ("PREDICATE_ACTION_STATE_ROLE", "NE_SOURCE_NOT_PRESENT", 0,
         "POS_DOES_NOT_IDENTIFY_ACTION_VERSUS_STATE_INTENT"),
        ("ARGUMENT_SEMANTIC_ROLE", "NE_SOURCE_NOT_PRESENT", 0,
         "SLOT_ORDER_DOES_NOT_IDENTIFY_ARGUMENT_ROLE"),
        ("MODALITY_NEGATION", "NE_SOURCE_NOT_PRESENT", 0,
         "ALIAS_AND_STRUCTURE_PACKS_LACK_OPERATOR_FACTS"),
        ("TARGET_SPACING_PUNCTUATION_POLICY", "NE_UNSEEN_FAMILY_POLICY", 0,
         "SOURCE_PUNCTUATION_CANNOT_AUTHORIZE_TARGET_PRODUCT_STYLE"),
    )
    records = []
    for family, outcome, evidence_count, reason_code in facts:
        identity = {
            "fact_family": family,
            "target_scope": INTENT_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "evidence_count": evidence_count,
            "fact_family_id": _record_id(identity),
            "format_version": 1,
            "outcome": outcome,
            "reason_code": reason_code,
            "record_kind": INTENT_SEMANTIC_FACT_FAMILY_KIND,
        })
    return tuple(records)


def derive_intent_semantic_alignment_feasibility(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        neutral_projections: tuple[dict[str, object], ...],
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        alias_evidence: tuple[dict[str, object], ...],
        alias_routes: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生 fact/family/signature/LOSO census 与诚实可行性摘要。"""
    pair_facts, family_records, alias_summary = _pair_material(
        observations=observations,
        neutral_projections=neutral_projections,
        rows_by_family=rows_by_family,
        alias_evidence=alias_evidence,
        alias_routes=alias_routes,
    )
    signature_records, signature_groups = _signature_census(pair_facts)
    loso_records, loso_summary = _loso_census(
        observations=observations,
        fragments=fragments,
        plans=plans,
        pair_facts=pair_facts,
        signature_groups=signature_groups,
    )
    fact_records = _fact_family_records(
        family_records=family_records,
        alias_summary=alias_summary,
        signature_records=signature_records,
    )
    alias_signature = next(
        item for item in signature_records
        if item["signature_mode"] == SIGNATURE_ALIAS_PUNCTUATION)
    capability = (
        "PASS_NONZERO_AUTHORIZED_EXACT"
        if loso_summary["final_outcome_counts"]["WRONG"] == 0
        and loso_summary["final_outcome_counts"]["EXACT"] > 0
        else "NE_ZERO_AUTHORIZED_EXACT"
        if loso_summary["final_outcome_counts"]["WRONG"] == 0
        else "FAIL_HARD_GATE")
    return (
        fact_records,
        tuple(sorted(
            family_records, key=lambda item: str(item["source_family"]))),
        tuple(sorted(
            signature_records,
            key=lambda item: str(item["signature_mode"]))),
        loso_records,
        {
            "alias": alias_summary,
            "authorization_signature_mode": AUTHORIZATION_SIGNATURE_MODE,
            "capability_outcome": capability,
            "facility_outcome": "PASS",
            "loso": loso_summary,
            "representation_outcome": (
                "PARTIAL_NONZERO_ALIAS_FACT_SUPPORT"
                if alias_signature["cross_family_consensus_count"] > 0
                else "NE_ZERO_CROSS_FAMILY_ALIAS_FACT_SUPPORT"),
            "source_family_count": len(_SOURCE_ORDER),
        },
    )


__all__ = [
    "AUTHORIZATION_SIGNATURE_MODE",
    "INTENT_SEMANTIC_FACT_FAMILY_KIND",
    "INTENT_SEMANTIC_FAMILY_CENSUS_KIND",
    "INTENT_SEMANTIC_LOSO_KIND",
    "INTENT_SEMANTIC_SIGNATURE_CENSUS_KIND",
    "INTENT_SEMANTIC_TARGET_SCOPE",
    "SIGNATURE_ALIAS_LEXICAL",
    "SIGNATURE_ALIAS_PUNCTUATION",
    "SIGNATURE_POS_ONLY",
    "SIGNATURE_STRUCTURE_ONLY",
    "derive_intent_semantic_alignment_feasibility",
]
