"""Recovery transfer candidate 的 indexed 与线性 reference 执行器。"""
from __future__ import annotations

from bisect import bisect_left

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_records import (
    NormalizationRecoveryCandidateProgram,
    NormalizationRecoveryExecutionResult,
    RECOVERY_TRANSFER_REGION_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    SOURCE_POLICY_SCOPES,
)


def _exact_by_input(values, input_text: str):
    """用二分索引查找按 input_text 排序的结构。"""
    index = bisect_left(values, input_text, key=lambda item: item.input_text)
    if index < len(values) and values[index].input_text == input_text:
        return values[index]
    return None


def _exact_scoped(values, policy: str, input_text: str):
    """用二分索引查找按 source policy/input 排序的结构。"""
    key = (policy, input_text)
    index = bisect_left(
        values, key,
        key=lambda item: (item.source_policy_scope, item.input_text))
    if index < len(values):
        value = values[index]
        if (value.source_policy_scope, value.input_text) == key:
            return value
    return None


def _indexed_target_execution(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        regional_scope: str,
        ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """按冻结 exact-first/character composition 次序执行 target rules。"""
    if regional_scope == RECOVERY_TRANSFER_REGION_SCOPE:
        rule = _exact_by_input(program.regional_rules, input_text)
        if rule is not None:
            return rule.output_text, (rule.rule_id,), ()
    rule = _exact_by_input(program.generic_rules, input_text)
    if rule is not None:
        return rule.output_text, (rule.rule_id,), ()
    output = []
    target_ids = []
    conflict_ids = []
    for character in input_text:
        character_rule = None
        if regional_scope == RECOVERY_TRANSFER_REGION_SCOPE:
            character_rule = _exact_by_input(program.regional_rules, character)
        if character_rule is None:
            character_rule = _exact_by_input(program.generic_rules, character)
        if character_rule is None or character_rule.mapping_kind != "CHARACTER_INPUT":
            output.append(character)
            conflict = _exact_by_input(program.conflicts, character)
            if conflict is not None:
                conflict_ids.append(conflict.conflict_id)
        else:
            output.append(character_rule.output_text)
            target_ids.append(character_rule.rule_id)
    whole_conflict = _exact_by_input(program.conflicts, input_text)
    if whole_conflict is not None:
        conflict_ids.append(whole_conflict.conflict_id)
    return (
        "".join(output),
        tuple(sorted(set(target_ids))),
        tuple(sorted(set(conflict_ids))),
    )


def _indexed_source_execution(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        policy_scope: str,
        ) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """按 exact phrase/replay/character composition 次序执行 source policy。"""
    phrase = _exact_scoped(program.phrase_overrides, policy_scope, input_text)
    replay = _exact_scoped(program.source_replays, policy_scope, input_text)
    if phrase is not None:
        if replay is None or replay.output_text != phrase.output_text:
            raise BroadQaExternalDataError(
                "recovery source exact phrase/replay 运行漂移")
        return (
            phrase.output_text,
            (replay.evidence_id,),
            (phrase.rule_id,),
            replay.conflict_ids,
        )
    if replay is not None:
        return replay.output_text, (replay.evidence_id,), (), replay.conflict_ids
    output = []
    evidence_ids = []
    conflict_ids = []
    for character in input_text:
        character_replay = _exact_scoped(
            program.source_replays, policy_scope, character)
        if character_replay is None or len(character_replay.input_text) != 1:
            output.append(character)
        else:
            output.append(character_replay.output_text)
            evidence_ids.append(character_replay.evidence_id)
            conflict_ids.extend(character_replay.conflict_ids)
    return (
        "".join(output),
        tuple(sorted(set(evidence_ids))),
        (),
        tuple(sorted(set(conflict_ids))),
    )


def _indexed_unscoped_conflict(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        ):
    """按 whole input、再按字符顺序查找首个被阻断冲突。"""
    conflict = _exact_by_input(program.conflicts, input_text)
    if conflict is not None:
        return conflict
    for character in input_text:
        conflict = _exact_by_input(program.conflicts, character)
        if conflict is not None:
            return conflict
    return None


def execute_normalization_recovery_candidate(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> NormalizationRecoveryExecutionResult:
    """按显式 target/source/region scope 执行索引 candidate。"""
    if (not isinstance(program, NormalizationRecoveryCandidateProgram)
            or not isinstance(input_text, str) or not input_text
            or not isinstance(policy_scope, str)
            or not isinstance(regional_scope, str)):
        raise BroadQaExternalDataError(
            "recovery candidate execution 输入非法")
    profile = program.transfer_profile
    target_ids = ()
    source_ids = ()
    phrase_ids = ()
    conflict_ids = ()
    transfer_id = ""
    projection_used = 0
    scope_mismatch = 0
    unscoped_blocked = 0
    output = input_text
    if policy_scope == profile.candidate_target_policy_scope:
        if regional_scope != profile.regional_scope:
            scope_mismatch = 1
        else:
            output, target_ids, conflict_ids = _indexed_target_execution(
                program, input_text, regional_scope)
            transfer_id = profile.sha256()
            projection_used = 1
    elif policy_scope == profile.authority_policy_scope:
        if regional_scope not in {"", profile.regional_scope}:
            scope_mismatch = 1
        else:
            output, target_ids, conflict_ids = _indexed_target_execution(
                program, input_text, regional_scope)
    elif policy_scope in SOURCE_POLICY_SCOPES:
        if regional_scope:
            scope_mismatch = 1
        else:
            output, source_ids, phrase_ids, conflict_ids = (
                _indexed_source_execution(program, input_text, policy_scope))
    else:
        scope_mismatch = 1
        blocked = _indexed_unscoped_conflict(program, input_text)
        if blocked is not None:
            unscoped_blocked = 1
            conflict_ids = (blocked.conflict_id,)
    return NormalizationRecoveryExecutionResult(
        input_text=input_text,
        output_text=output,
        requested_policy_scope=policy_scope,
        regional_scope=regional_scope,
        target_rule_ids=target_ids,
        source_evidence_ids=source_ids,
        phrase_rule_ids=phrase_ids,
        conflict_ids=conflict_ids,
        transfer_profile_id=transfer_id,
        projection_used=projection_used,
        scope_mismatch=scope_mismatch,
        unscoped_conflict_blocked=unscoped_blocked,
        production_enabled=program.production_enabled,
    )


def _linear_target_execution(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        regional_scope: str,
        ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """用线性扫描独立执行 target exact-first 与逐字符组合。"""
    if regional_scope == RECOVERY_TRANSFER_REGION_SCOPE:
        regional = next(
            (item for item in program.regional_rules
             if item.input_text == input_text), None)
        if regional is not None:
            return regional.output_text, (regional.rule_id,), ()
    generic = next(
        (item for item in program.generic_rules
         if item.input_text == input_text), None)
    if generic is not None:
        return generic.output_text, (generic.rule_id,), ()
    output = []
    target_ids = []
    conflict_ids = []
    for character in input_text:
        rule = None
        if regional_scope == RECOVERY_TRANSFER_REGION_SCOPE:
            rule = next(
                (item for item in program.regional_rules
                 if item.input_text == character), None)
        if rule is None:
            rule = next(
                (item for item in program.generic_rules
                 if item.input_text == character), None)
        if rule is None or rule.mapping_kind != "CHARACTER_INPUT":
            output.append(character)
            conflict = next(
                (item for item in program.conflicts
                 if item.input_text == character), None)
            if conflict is not None:
                conflict_ids.append(conflict.conflict_id)
        else:
            output.append(rule.output_text)
            target_ids.append(rule.rule_id)
    whole_conflict = next(
        (item for item in program.conflicts
         if item.input_text == input_text), None)
    if whole_conflict is not None:
        conflict_ids.append(whole_conflict.conflict_id)
    return (
        "".join(output),
        tuple(sorted(set(target_ids))),
        tuple(sorted(set(conflict_ids))),
    )


def _linear_source_execution(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        policy_scope: str,
        ) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """用线性扫描独立执行 source phrase/replay/character composition。"""
    phrase = next(
        (item for item in program.phrase_overrides
         if item.source_policy_scope == policy_scope
         and item.input_text == input_text), None)
    replay = next(
        (item for item in program.source_replays
         if item.source_policy_scope == policy_scope
         and item.input_text == input_text), None)
    if phrase is not None:
        if replay is None or replay.output_text != phrase.output_text:
            raise BroadQaExternalDataError(
                "recovery reference phrase/replay 运行漂移")
        return (
            phrase.output_text, (replay.evidence_id,), (phrase.rule_id,),
            replay.conflict_ids)
    if replay is not None:
        return replay.output_text, (replay.evidence_id,), (), replay.conflict_ids
    output = []
    evidence_ids = []
    conflict_ids = []
    for character in input_text:
        item = next(
            (value for value in program.source_replays
             if value.source_policy_scope == policy_scope
             and value.input_text == character), None)
        if item is None or len(item.input_text) != 1:
            output.append(character)
        else:
            output.append(item.output_text)
            evidence_ids.append(item.evidence_id)
            conflict_ids.extend(item.conflict_ids)
    return (
        "".join(output), tuple(sorted(set(evidence_ids))), (),
        tuple(sorted(set(conflict_ids))))


def _linear_unscoped_conflict(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        ):
    """线性查找 whole input 或任一字符上的首个冲突。"""
    whole = next(
        (item for item in program.conflicts
         if item.input_text == input_text), None)
    if whole is not None:
        return whole
    return next(
        (item for character in input_text for item in program.conflicts
         if item.input_text == character), None)


def reference_normalization_recovery_candidate(
        program: NormalizationRecoveryCandidateProgram,
        input_text: str,
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> NormalizationRecoveryExecutionResult:
    """用线性 reference 独立解释同一 transfer program。"""
    if (not isinstance(program, NormalizationRecoveryCandidateProgram)
            or not isinstance(input_text, str) or not input_text
            or not isinstance(policy_scope, str)
            or not isinstance(regional_scope, str)):
        raise BroadQaExternalDataError(
            "recovery reference execution 输入非法")
    profile = program.transfer_profile
    target_ids = ()
    source_ids = ()
    phrase_ids = ()
    conflict_ids = ()
    transfer_id = ""
    projection_used = 0
    scope_mismatch = 0
    unscoped_blocked = 0
    output = input_text
    if policy_scope == profile.candidate_target_policy_scope:
        if regional_scope != profile.regional_scope:
            scope_mismatch = 1
        else:
            output, target_ids, conflict_ids = _linear_target_execution(
                program, input_text, regional_scope)
            transfer_id = profile.sha256()
            projection_used = 1
    elif policy_scope == profile.authority_policy_scope:
        if regional_scope not in {"", profile.regional_scope}:
            scope_mismatch = 1
        else:
            output, target_ids, conflict_ids = _linear_target_execution(
                program, input_text, regional_scope)
    elif policy_scope in SOURCE_POLICY_SCOPES:
        if regional_scope:
            scope_mismatch = 1
        else:
            output, source_ids, phrase_ids, conflict_ids = (
                _linear_source_execution(program, input_text, policy_scope))
    else:
        scope_mismatch = 1
        blocked = _linear_unscoped_conflict(program, input_text)
        if blocked is not None:
            unscoped_blocked = 1
            conflict_ids = (blocked.conflict_id,)
    return NormalizationRecoveryExecutionResult(
        input_text=input_text,
        output_text=output,
        requested_policy_scope=policy_scope,
        regional_scope=regional_scope,
        target_rule_ids=target_ids,
        source_evidence_ids=source_ids,
        phrase_rule_ids=phrase_ids,
        conflict_ids=conflict_ids,
        transfer_profile_id=transfer_id,
        projection_used=projection_used,
        scope_mismatch=scope_mismatch,
        unscoped_conflict_blocked=unscoped_blocked,
        production_enabled=program.production_enabled,
    )


__all__ = [
    "execute_normalization_recovery_candidate",
    "reference_normalization_recovery_candidate",
]
