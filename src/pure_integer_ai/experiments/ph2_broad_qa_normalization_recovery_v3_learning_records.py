"""从冻结 recovery-v3 TRAIN 纯派生 phrase Evidence 与禁用态规则。

本模块只消费物化 observation、fragment、group 和 ordered work。它不读取
source、evaluation、reserve、旧 formal 或 candidate，也不写文件。相同 input
跨 fragment kind 先合并为一个 family；真实负例、上下文冲突和 overlap 顺序均
由 TRAIN 字节确定。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    RECOVERY_V3_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V3_EVIDENCE_KIND = (
    "NORMALIZATION_RECOVERY_V3_PHRASE_EVIDENCE_V1")
NORMALIZATION_RECOVERY_V3_PHRASE_RULE_KIND = (
    "NORMALIZATION_RECOVERY_V3_PHRASE_RULE_V1")
NORMALIZATION_RECOVERY_V3_DEFEATER_KIND = (
    "NORMALIZATION_RECOVERY_V3_CONTEXT_DEFEATER_V1")
NORMALIZATION_RECOVERY_V3_CONFLICT_KIND = (
    "NORMALIZATION_RECOVERY_V3_PHRASE_CONFLICT_V1")
NORMALIZATION_RECOVERY_V3_DEFERRED_KIND = (
    "NORMALIZATION_RECOVERY_V3_DEFERRED_INPUT_FAMILY_V1")
NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_KIND = (
    "NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_ENTRY_V1")

NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES = (
    ("evidence.jsonl", "LEARNED_PHRASE_EVIDENCE", "evidence_id"),
    ("phrase-rules.jsonl", "LEARNED_PHRASE_RULES", "rule_id"),
    ("defeaters.jsonl", "LEARNED_CONTEXT_DEFEATERS", "defeater_id"),
    ("conflict-ledger.jsonl", "LEARNED_CONFLICT_LEDGER", "conflict_id"),
    ("deferred-groups.jsonl", "LEARNED_DEFERRED_GROUPS", "deferred_id"),
    ("overlap-index.jsonl", "LEARNED_OVERLAP_INDEX", "index_id"),
)


def _sha256(payload: bytes) -> str:
    """返回规范记录摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _context_signature(text: str, start: int, end: int) -> dict[str, object]:
    """截取 occurrence 两侧至多四个 scalar，并固定边界语义。"""
    identity = {
        "left_boundary": int(start == 0),
        "left_context": text[max(0, start - 4):start],
        "right_boundary": int(end == len(text)),
        "right_context": text[end:min(len(text), end + 4)],
    }
    return {
        **identity,
        "context_signature_id": _record_id(identity),
    }


def _alignment_boundary_map(
        input_text: str,
        output_text: str,
        ) -> dict[int, set[int]]:
    """建立只含确定边界的 input-to-output 对齐表。"""
    result: dict[int, set[int]] = defaultdict(set)
    matcher = SequenceMatcher(None, input_text, output_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        result[i1].add(j1)
        result[i2].add(j2)
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for offset in range(i2 - i1 + 1):
                result[i1 + offset].add(j1 + offset)
        elif tag == "insert":
            result[i1].add(j2)
    return result


def _observed_output(
        observation: dict[str, object],
        start: int,
        end: int,
        boundaries: dict[int, set[int]],
        ) -> str | None:
    """只在 occurrence 两端均有唯一确定对齐时返回实际输出 span。"""
    input_text = observation.get("input_text")
    output_text = observation.get("output_text")
    if not isinstance(input_text, str) or not isinstance(output_text, str):
        raise BroadQaExternalDataError("v3 learner observation surface 漂移")
    starts = boundaries.get(start, set())
    ends = boundaries.get(end, set())
    if len(starts) != 1 or len(ends) != 1:
        return None
    output_start = next(iter(starts))
    output_end = next(iter(ends))
    if not 0 <= output_start <= output_end <= len(output_text):
        return None
    return output_text[output_start:output_end]


def _occurrences(text: str, phrase: str):
    """按 scalar 起点产生允许重叠的全部 literal occurrence。"""
    start = text.find(phrase)
    while start >= 0:
        yield start, start + len(phrase)
        start = text.find(phrase, start + 1)


def _support_evidence(
        *,
        protocol_manifest_sha256: str,
        fragment: dict[str, object],
        observation: dict[str, object],
        family_id: str,
        ) -> dict[str, object]:
    """把每一条物化 fragment 变成来源与 span 完整绑定的 SUPPORT。"""
    signature = _context_signature(
        str(observation["input_text"]),
        int(fragment["input_start"]),
        int(fragment["input_end"]),
    )
    identity = {
        "family_id": family_id,
        "fragment_id": fragment["fragment_id"],
        "hypothesis_input": fragment["input_text"],
        "hypothesis_output": fragment["output_text"],
        "observation_id": fragment["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "stance": "SUPPORT",
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "context_signature": signature,
        "equal_length": fragment["equal_length"],
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "fragment_kind": fragment["fragment_kind"],
        "input_end": fragment["input_end"],
        "input_start": fragment["input_start"],
        "license_id": fragment["license_id"],
        "output_end": fragment["output_end"],
        "output_start": fragment["output_start"],
        "record_kind": NORMALIZATION_RECOVERY_V3_EVIDENCE_KIND,
        "source_commitment": observation["source_commitment"],
        "source_family": fragment["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": fragment["source_policy_scope"],
    }


def _refute_evidence(
        *,
        protocol_manifest_sha256: str,
        family_id: str,
        input_text: str,
        candidate_output: str,
        observation: dict[str, object],
        start: int,
        end: int,
        observed_output: str,
        signature: dict[str, object],
        ) -> dict[str, object]:
    """记录同一 input occurrence 未产生候选 output 的真实 REFUTE。"""
    reason = (
        "IDENTITY_PRESERVATION"
        if observed_output == input_text else "ALTERNATIVE_OUTPUT")
    identity = {
        "candidate_output": candidate_output,
        "family_id": family_id,
        "hypothesis_input": input_text,
        "observation_id": observation["observation_id"],
        "occurrence_end": end,
        "occurrence_start": start,
        "observed_output": observed_output,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "stance": "REFUTE",
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "context_signature": signature,
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "license_id": observation["license_id"],
        "record_kind": NORMALIZATION_RECOVERY_V3_EVIDENCE_KIND,
        "refute_reason": reason,
        "source_commitment": observation["source_commitment"],
        "source_family": observation["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": observation["source_policy_scope"],
    }


def _input_families(
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """合并跨 fragment kind 的同 input group，并固定最后结算 group。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    group_ordinal = {}
    for ordinal, group in enumerate(groups):
        input_text = group.get("input_text")
        group_id = group.get("group_id")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(group_id, str) or group_id in group_ordinal):
            raise BroadQaExternalDataError("v3 learner phrase group 漂移")
        grouped[input_text].append(group)
        group_ordinal[group_id] = ordinal
    values = []
    for input_text, members in sorted(grouped.items()):
        group_ids = sorted(str(item["group_id"]) for item in members)
        identity = {"input_text": input_text, "group_ids": group_ids}
        values.append({
            **identity,
            "candidate_authority": int(any(
                str(item["disposition"]).endswith("CANDIDATE")
                for item in members)),
            "emission_group_id": max(
                group_ids, key=lambda value: group_ordinal[value]),
            "family_id": _record_id(identity),
            "fragment_kinds": sorted({
                str(item["fragment_kind"]) for item in members}),
            "members": members,
        })
    return tuple(values)


def _family_variants(
        family: dict[str, object],
        fragment_by_id: dict[str, dict[str, object]],
        evidence_by_fragment: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """把 family 内各 group variant 合并为 output 级支持账本。"""
    grouped: dict[str, set[str]] = defaultdict(set)
    for group in family["members"]:
        for variant in group["output_variants"]:
            output_text = str(variant["output_text"])
            grouped[output_text].update(str(value)
                                        for value in variant["fragment_ids"])
    values = []
    for output_text, fragment_ids in sorted(grouped.items()):
        try:
            fragments = [fragment_by_id[value]
                         for value in sorted(fragment_ids)]
            evidence_ids = sorted(str(evidence_by_fragment[value]["evidence_id"])
                                  for value in fragment_ids)
        except KeyError as error:
            raise BroadQaExternalDataError(
                "v3 learner family fragment/Evidence 缺失") from error
        values.append({
            "evidence_ids": evidence_ids,
            "fragment_ids": sorted(fragment_ids),
            "license_ids": sorted({str(item["license_id"])
                                   for item in fragments}),
            "output_text": output_text,
            "source_families": sorted({str(item["source_family"])
                                       for item in fragments}),
            "source_policy_scopes": sorted({
                str(item["source_policy_scope"]) for item in fragments}),
            "support_count": len(fragment_ids),
        })
    return tuple(values)


def _conflict_record(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        conflict_kind: str,
        variants: tuple[dict[str, object], ...],
        support_evidence_ids: list[str],
        refute_evidence_ids: list[str],
        context_signature_ids: list[str],
        ) -> dict[str, object]:
    """保留 output 或 support/refute context 冲突，不做启发式裁决。"""
    identity = {
        "conflict_kind": conflict_kind,
        "family_id": family["family_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "conflict_id": _record_id(identity),
        "context_signature_ids": context_signature_ids,
        "format_version": 1,
        "group_ids": family["group_ids"],
        "input_text": family["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V3_CONFLICT_KIND,
        "refute_evidence_ids": refute_evidence_ids,
        "resolution_state": "DEFERRED_NO_OVERRIDE",
        "support_evidence_ids": support_evidence_ids,
        "unscoped_execution_allowed": 0,
        "variants": list(variants),
    }


def _deferred_record(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        defer_reason: str,
        candidate_output: str,
        evidence_ids: list[str],
        conflict_id: str,
        unaligned_occurrence_count: int,
        ) -> dict[str, object]:
    """形成每个未执行 input family 的明确恢复理由。"""
    identity = {
        "defer_reason": defer_reason,
        "family_id": family["family_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
    }
    return {
        **identity,
        "candidate_output": candidate_output,
        "conflict_id": conflict_id,
        "deferred_id": _record_id(identity),
        "evidence_ids": evidence_ids,
        "format_version": 1,
        "fragment_kinds": family["fragment_kinds"],
        "group_ids": family["group_ids"],
        "input_text": family["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V3_DEFERRED_KIND,
        "unaligned_occurrence_count": unaligned_occurrence_count,
    }


def _candidate_occurrence_evidence(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        candidate_output: str,
        observations: tuple[dict[str, object], ...],
        support_fragments: tuple[dict[str, object], ...],
        alignment_by_observation: dict[str, dict[int, set[int]]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            set[str],
            int,
        ]:
    """扫描全部 TRAIN occurrence，返回 REFUTE、正上下文签名与未对齐数。"""
    input_text = str(family["input_text"])
    support_locations = {
        (str(item["observation_id"]), int(item["input_start"]),
         int(item["input_end"]))
        for item in support_fragments
        if item["output_text"] == candidate_output
    }
    support_signatures = set()
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    for fragment in support_fragments:
        if fragment["output_text"] != candidate_output:
            continue
        observation = observation_by_id[str(fragment["observation_id"])]
        signature = _context_signature(
            str(observation["input_text"]),
            int(fragment["input_start"]),
            int(fragment["input_end"]),
        )
        support_signatures.add(str(signature["context_signature_id"]))
    refutes = []
    unaligned = 0
    for observation in observations:
        surface = str(observation["input_text"])
        for start, end in _occurrences(surface, input_text):
            signature = _context_signature(surface, start, end)
            location = (str(observation["observation_id"]), start, end)
            if location in support_locations:
                support_signatures.add(str(signature["context_signature_id"]))
                continue
            observed_output = _observed_output(
                observation,
                start,
                end,
                alignment_by_observation[str(observation["observation_id"])],
            )
            if observed_output is None:
                unaligned += 1
                continue
            if observed_output == candidate_output:
                support_signatures.add(str(signature["context_signature_id"]))
                continue
            refutes.append(_refute_evidence(
                protocol_manifest_sha256=protocol_manifest_sha256,
                family_id=str(family["family_id"]),
                input_text=input_text,
                candidate_output=candidate_output,
                observation=observation,
                start=start,
                end=end,
                observed_output=observed_output,
                signature=signature,
            ))
    result = tuple(sorted(
        {str(item["evidence_id"]): item for item in refutes}.values(),
        key=lambda item: str(item["evidence_id"]),
    ))
    return result, support_signatures, unaligned


def _eligible_outputs(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        variant: dict[str, object],
        refutes: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从正负 Evidence 形成一条禁用态 phrase rule、defeater 与 index。"""
    rule_identity = {
        "family_id": family["family_id"],
        "input_text": family["input_text"],
        "output_text": variant["output_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
    }
    rule_id = _record_id(rule_identity)
    by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)
    for evidence in refutes:
        signature = evidence["context_signature"]
        by_signature[str(signature["context_signature_id"])].append(evidence)
    defeaters = []
    for signature_id, records in sorted(by_signature.items()):
        signature = records[0]["context_signature"]
        identity = {
            "context_signature_id": signature_id,
            "rule_id": rule_id,
            "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
        }
        defeaters.append({
            **identity,
            "action": "BLOCK_PHRASE_RULE_USE_BACKOFF",
            "defeater_id": _record_id(identity),
            "format_version": 1,
            "left_boundary": signature["left_boundary"],
            "left_context": signature["left_context"],
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_V3_DEFEATER_KIND,
            "refute_evidence_ids": sorted(
                str(item["evidence_id"]) for item in records),
            "right_boundary": signature["right_boundary"],
            "right_context": signature["right_context"],
        })
    defeater_ids = sorted(str(item["defeater_id"]) for item in defeaters)
    rule = {
        **rule_identity,
        "application_scope": {
            "defeater_required": 1,
            "input_match": "EXACT_SCALAR_SUBSEQUENCE",
            "unscoped_execution_allowed": 0,
        },
        "defeater_ids": defeater_ids,
        "equal_length": int(
            len(str(family["input_text"])) == len(str(variant["output_text"]))),
        "format_version": 1,
        "fragment_kinds": family["fragment_kinds"],
        "group_ids": family["group_ids"],
        "license_ids": variant["license_ids"],
        "negative_evidence_ids": sorted(
            str(item["evidence_id"]) for item in refutes),
        "positive_evidence_ids": variant["evidence_ids"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V3_PHRASE_RULE_KIND,
        "rule_id": rule_id,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "source_families": variant["source_families"],
        "source_policy_scopes": variant["source_policy_scopes"],
        "variable_length": int(
            len(str(family["input_text"])) != len(str(variant["output_text"]))),
    }
    first_scalar = ord(str(family["input_text"])[0])
    index_identity = {
        "first_scalar": first_scalar,
        "input_scalar_length": len(str(family["input_text"])),
        "rule_id": rule_id,
    }
    index = {
        **index_identity,
        "candidate_input": family["input_text"],
        "format_version": 1,
        "index_id": _record_id(index_identity),
        "priority_contract": (
            "WHOLE_INPUT_EXACT_THEN_LONGEST_MATCH_THEN_CHARACTER_BACKOFF"),
        "record_kind": NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_KIND,
    }
    return rule, tuple(defeaters), index


def _require_work_alignment(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> None:
    """要求 ordered work 精确覆盖三类物化 TRAIN 记录。"""
    expected = (
        [("PAIR_OBSERVATION_INGEST", "PAIR_OBSERVATION", item["observation_id"])
         for item in observations]
        + [("PHRASE_FRAGMENT_INGEST", "PHRASE_FRAGMENT", item["fragment_id"])
           for item in fragments]
        + [("PHRASE_GROUP_RESOLUTION", "PHRASE_GROUP", item["group_id"])
           for item in groups]
    )
    observed = [(item.get("phase"), item.get("work_kind"), item.get("record_id"))
                for item in work]
    if (observed != expected
            or [item.get("work_ordinal") for item in work]
            != list(range(len(work)))):
        raise BroadQaExternalDataError(
            "v3 learner ordered work/material 漂移")


def _derive_material(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """派生全部输出，并返回每个 emission group 的 checkpoint 增量。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v3 learner protocol manifest")
    if (not observations or not fragments or not groups or not work
            or protocol_manifest.get("target_policy_scope")
            != RECOVERY_V3_TARGET_POLICY_SCOPE):
        raise BroadQaExternalDataError(
            "v3 learner protocol material 为空或 scope 漂移")
    _require_work_alignment(
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
    )
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    alignment_by_observation = {
        str(item["observation_id"]): _alignment_boundary_map(
            str(item["input_text"]), str(item["output_text"]))
        for item in observations
    }
    fragment_by_id = {str(item["fragment_id"]): item for item in fragments}
    if (len(observation_by_id) != len(observations)
            or len(fragment_by_id) != len(fragments)):
        raise BroadQaExternalDataError(
            "v3 learner observation/fragment identity 重复")
    families = _input_families(groups)
    family_by_input = {str(item["input_text"]): item for item in families}
    support_evidence = []
    evidence_by_fragment = {}
    for fragment in fragments:
        family = family_by_input.get(str(fragment["input_text"]))
        observation = observation_by_id.get(str(fragment["observation_id"]))
        if family is None or observation is None:
            raise BroadQaExternalDataError(
                "v3 learner fragment family/observation 缺失")
        evidence = _support_evidence(
            protocol_manifest_sha256=protocol_sha,
            fragment=fragment,
            observation=observation,
            family_id=str(family["family_id"]),
        )
        support_evidence.append(evidence)
        evidence_by_fragment[str(fragment["fragment_id"])] = evidence

    refute_evidence = []
    rules = []
    defeaters = []
    conflicts = []
    deferred = []
    overlap = []
    emission_counts = []
    for family in families:
        variants = _family_variants(
            family, fragment_by_id, evidence_by_fragment)
        support_ids = sorted({
            str(value) for variant in variants
            for value in variant["evidence_ids"]})
        family_refutes: tuple[dict[str, object], ...] = ()
        unaligned = 0
        result_increment = 0
        if len(variants) != 1:
            conflict = _conflict_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                conflict_kind="TRAIN_OUTPUT_CONFLICT",
                variants=variants,
                support_evidence_ids=support_ids,
                refute_evidence_ids=[],
                context_signature_ids=[],
            )
            conflicts.append(conflict)
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                defer_reason="TRAIN_OUTPUT_CONFLICT",
                candidate_output="",
                evidence_ids=support_ids,
                conflict_id=str(conflict["conflict_id"]),
                unaligned_occurrence_count=0,
            ))
            result_increment = 2
        elif family["candidate_authority"] != 1:
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                defer_reason="INSUFFICIENT_POSITIVE_AUTHORITY",
                candidate_output=str(variants[0]["output_text"]),
                evidence_ids=support_ids,
                conflict_id="",
                unaligned_occurrence_count=0,
            ))
            result_increment = 1
        else:
            variant = variants[0]
            support_fragments = tuple(
                fragment_by_id[str(value)] for value in variant["fragment_ids"])
            family_refutes, support_signatures, unaligned = (
                _candidate_occurrence_evidence(
                    protocol_manifest_sha256=protocol_sha,
                    family=family,
                    candidate_output=str(variant["output_text"]),
                    observations=observations,
                    support_fragments=support_fragments,
                    alignment_by_observation=alignment_by_observation,
                ))
            refute_evidence.extend(family_refutes)
            refute_signatures = {
                str(item["context_signature"]["context_signature_id"])
                for item in family_refutes}
            collided = sorted(support_signatures.intersection(
                refute_signatures))
            refute_ids = sorted(str(item["evidence_id"])
                                for item in family_refutes)
            all_evidence_ids = sorted(set(support_ids + refute_ids))
            if collided:
                conflict = _conflict_record(
                    protocol_manifest_sha256=protocol_sha,
                    family=family,
                    conflict_kind="SUPPORT_REFUTE_CONTEXT_SIGNATURE_CONFLICT",
                    variants=variants,
                    support_evidence_ids=support_ids,
                    refute_evidence_ids=refute_ids,
                    context_signature_ids=collided,
                )
                conflicts.append(conflict)
                deferred.append(_deferred_record(
                    protocol_manifest_sha256=protocol_sha,
                    family=family,
                    defer_reason="CONTEXT_SIGNATURE_CONFLICT",
                    candidate_output=str(variant["output_text"]),
                    evidence_ids=all_evidence_ids,
                    conflict_id=str(conflict["conflict_id"]),
                    unaligned_occurrence_count=unaligned,
                ))
                result_increment = 2
            elif not family_refutes:
                deferred.append(_deferred_record(
                    protocol_manifest_sha256=protocol_sha,
                    family=family,
                    defer_reason="NO_REAL_REFUTE_EVIDENCE",
                    candidate_output=str(variant["output_text"]),
                    evidence_ids=support_ids,
                    conflict_id="",
                    unaligned_occurrence_count=unaligned,
                ))
                result_increment = 1
            else:
                rule, family_defeaters, index = _eligible_outputs(
                    protocol_manifest_sha256=protocol_sha,
                    family=family,
                    variant=variant,
                    refutes=family_refutes,
                )
                rules.append(rule)
                defeaters.extend(family_defeaters)
                overlap.append(index)
                result_increment = 2 + len(family_defeaters)
        emission_counts.append({
            "evidence_increment": len(family_refutes),
            "group_id": family["emission_group_id"],
            "result_increment": result_increment,
        })

    raw_outputs = {
        "evidence.jsonl": support_evidence + refute_evidence,
        "phrase-rules.jsonl": rules,
        "defeaters.jsonl": defeaters,
        "conflict-ledger.jsonl": conflicts,
        "deferred-groups.jsonl": deferred,
        "overlap-index.jsonl": overlap,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        if len({item[identity_key] for item in values}) != len(values):
            raise BroadQaExternalDataError(
                f"v3 learner {name} identity 重复")
        outputs[name] = values
    stance_counts = Counter(
        str(item["stance"]) for item in outputs["evidence.jsonl"])
    defer_counts = Counter(
        str(item["defer_reason"]) for item in outputs["deferred-groups.jsonl"])
    conflict_counts = Counter(
        str(item["conflict_kind"]) for item in outputs["conflict-ledger.jsonl"])
    result_count = sum(
        len(values) for name, values in outputs.items()
        if name != "evidence.jsonl")
    summary = {
        "conflict_count": len(outputs["conflict-ledger.jsonl"]),
        "conflict_kind_counts": dict(sorted(conflict_counts.items())),
        "defeater_count": len(outputs["defeaters.jsonl"]),
        "defer_reason_counts": dict(sorted(defer_counts.items())),
        "deferred_family_count": len(outputs["deferred-groups.jsonl"]),
        "equal_length_rule_count": sum(
            item["equal_length"] == 1 for item in outputs["phrase-rules.jsonl"]),
        "evidence_count": len(outputs["evidence.jsonl"]),
        "evidence_stance_counts": {
            "REFUTE": stance_counts["REFUTE"],
            "SUPPORT": stance_counts["SUPPORT"],
        },
        "input_family_count": len(families),
        "overlap_index_count": len(outputs["overlap-index.jsonl"]),
        "phrase_rule_count": len(outputs["phrase-rules.jsonl"]),
        "result_record_count": result_count,
        "variable_length_rule_count": sum(
            item["variable_length"] == 1
            for item in outputs["phrase-rules.jsonl"]),
    }
    expected_evidence = len(fragments) + sum(
        int(item["evidence_increment"]) for item in emission_counts)
    expected_results = sum(
        int(item["result_increment"]) for item in emission_counts)
    if (expected_evidence != summary["evidence_count"]
            or expected_results != result_count
            or len(outputs["phrase-rules.jsonl"])
            != len(outputs["overlap-index.jsonl"])):
        raise BroadQaExternalDataError(
            "v3 learner output/prefix count 漂移")
    return outputs, summary, tuple(emission_counts)


def derive_normalization_recovery_v3_learning_outputs(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """公开纯派生入口；第三项供 checkpoint 前缀计数。"""
    return _derive_material(
        protocol_manifest=protocol_manifest,
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
    )


def normalization_recovery_v3_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """机械计算任意 ordered work 前缀的 Evidence/result 数。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError("v3 learner processed prefix 非法")
    by_group = {str(item["group_id"]): item for item in emission_counts}
    if len(by_group) != len(emission_counts):
        raise BroadQaExternalDataError(
            "v3 learner emission group identity 重复")
    evidence_count = 0
    result_count = 0
    seen_emissions = set()
    for item in work[:processed_item_count]:
        kind = item["work_kind"]
        if kind == "PAIR_OBSERVATION":
            continue
        if kind == "PHRASE_FRAGMENT":
            evidence_count += 1
            continue
        if kind != "PHRASE_GROUP":
            raise BroadQaExternalDataError("v3 learner work kind 非法")
        counts = by_group.get(str(item["record_id"]))
        if counts is None:
            continue
        seen_emissions.add(str(item["record_id"]))
        evidence_count += int(counts["evidence_increment"])
        result_count += int(counts["result_increment"])
    if processed_item_count == len(work) and seen_emissions != set(by_group):
        raise BroadQaExternalDataError(
            "v3 learner emission group 未被 work 覆盖")
    return evidence_count, result_count


def normalization_recovery_v3_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把完整输出转为 manifest-last writer 的规范 JSONL。"""
    expected = {
        name for name, _role, _identity
        in NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES}
    if set(outputs) != expected:
        raise BroadQaExternalDataError("v3 learner output inventory 漂移")
    return {
        name: b"".join(canonical_json_line(item) for item in outputs[name])
        for name, _role, _identity
        in NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES
    }


__all__ = [
    "NORMALIZATION_RECOVERY_V3_CONFLICT_KIND",
    "NORMALIZATION_RECOVERY_V3_DEFEATER_KIND",
    "NORMALIZATION_RECOVERY_V3_DEFERRED_KIND",
    "NORMALIZATION_RECOVERY_V3_EVIDENCE_KIND",
    "NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES",
    "NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_KIND",
    "NORMALIZATION_RECOVERY_V3_PHRASE_RULE_KIND",
    "derive_normalization_recovery_v3_learning_outputs",
    "normalization_recovery_v3_output_payloads",
    "normalization_recovery_v3_prefix_output_counts",
]
