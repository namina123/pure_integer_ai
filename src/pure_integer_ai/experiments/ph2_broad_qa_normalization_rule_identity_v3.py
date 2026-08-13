"""normalization v3 规则、竞争组和上下文 defeater 的稳定身份。

本模块只负责把冻结 mapping/trial 坐标投影到共享 ``HypothesisKey`` 与
``ObjectIdentity``。它不读取来源文件、不判断 SUPPORT/REFUTE，也不承担 pack I/O。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    SOURCE_INFERENCE_DIRECTIONS,
)


NORMALIZATION_RULE_HYPOTHESIS_KIND = (817043, 1)
NORMALIZATION_CONTEXT_TRIAL_HYPOTHESIS_KIND = (817043, 2)
NORMALIZATION_CONTEXT_DEFEATER_KIND = 817044
_NORMALIZATION_RULE_COMPETITION_KIND = (817043, 3)
_NORMALIZATION_TRIAL_COMPETITION_KIND = (817043, 4)
_DIRECTION_CODE = {"FORWARD": 1, "REVERSE": 2}


def _sha_identity(value: str, *, label: str) -> int:
    """将完整小写 SHA-256 无损投影为正整数身份分量。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return int(value, 16) + 1


def _codepoint(value: object, *, label: str) -> int:
    """要求候选坐标是 Unicode scalar。"""
    if (type(value) is not int or not 0 <= value <= 0x10FFFF
            or 0xD800 <= value <= 0xDFFF):
        raise BroadQaExternalDataError(f"{label} 不是 Unicode scalar")
    return value


def normalization_rule_hypothesis_key(
        *,
        mapping_candidate_id: str,
        input_codepoint: int,
        output_codepoint: int,
        operator: ObjectIdentity,
        schema: ObjectIdentity,
        direction: str,
        operator_version: int,
        applicability_scope: ScopeIdentity,
        ) -> HypothesisKey:
    """构造直接绑定 mapping 身份和完整规则坐标的候选 hypothesis。"""
    mapping_component = _sha_identity(
        mapping_candidate_id, label="normalization mapping candidate")
    input_value = _codepoint(input_codepoint, label="normalization input")
    output_value = _codepoint(output_codepoint, label="normalization output")
    if (not isinstance(operator, ObjectIdentity)
            or operator.object_kind != OBJECT_MINIMAL_INSTRUCTION
            or not isinstance(schema, ObjectIdentity)
            or schema.object_kind != OBJECT_STRUCTURE_CONCEPT
            or direction not in SOURCE_INFERENCE_DIRECTIONS
            or type(operator_version) is not int
            or operator_version <= 0
            or not isinstance(applicability_scope, ScopeIdentity)
            or applicability_scope.source is None):
        raise BroadQaExternalDataError(
            "normalization rule hypothesis 坐标非法")
    operator_key = operator.stable_key()
    schema_key = schema.stable_key()
    candidate_key = (
        mapping_component,
        input_value,
        output_value,
        operator_version,
        _DIRECTION_CODE[direction],
        len(operator_key),
        *operator_key,
        len(schema_key),
        *schema_key,
    )
    competition_key = (
        *_NORMALIZATION_RULE_COMPETITION_KIND,
        input_value,
    )
    return HypothesisKey(
        NORMALIZATION_RULE_HYPOTHESIS_KIND,
        candidate_key,
        competition_key,
        applicability_scope,
        applicability_scope.source,
    )


def normalization_context_trial_hypothesis_key(
        *,
        trial_id: str,
        rule_hypothesis: HypothesisKey,
        ) -> HypothesisKey:
    """为一个 context trial 构造与一般 mapping 规则分离的 hypothesis。"""
    trial_component = _sha_identity(
        trial_id, label="normalization context trial")
    if not isinstance(rule_hypothesis, HypothesisKey):
        raise TypeError("rule_hypothesis 必须是 HypothesisKey")
    rule_key = rule_hypothesis.stable_key()
    return HypothesisKey(
        NORMALIZATION_CONTEXT_TRIAL_HYPOTHESIS_KIND,
        (trial_component, len(rule_key), *rule_key),
        (*_NORMALIZATION_TRIAL_COMPETITION_KIND, trial_component),
        rule_hypothesis.scope,
        rule_hypothesis.observation,
    )


def normalization_context_defeater(trial_id: str) -> ObjectIdentity:
    """把一个冻结 trial 唯一投影为可审计的 context defeater。"""
    return concept_identity((
        NORMALIZATION_CONTEXT_DEFEATER_KIND,
        1,
        _sha_identity(trial_id, label="normalization context trial"),
    ))


__all__ = [
    "NORMALIZATION_CONTEXT_DEFEATER_KIND",
    "NORMALIZATION_CONTEXT_TRIAL_HYPOTHESIS_KIND",
    "NORMALIZATION_RULE_HYPOTHESIS_KIND",
    "normalization_context_defeater",
    "normalization_context_trial_hypothesis_key",
    "normalization_rule_hypothesis_key",
]
