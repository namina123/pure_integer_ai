"""公开 typed 对话课程的 V-00/H2/floor 装配。

这里的标签只从 authored payload 的显式 expected 字段读取；运行结果永远不会
反向生成标签。该模块不执行评测，只建立可写入 formal_train 的五类 ledger 和
分维协议，便于训练入口与独立评测入口复用同一来源身份。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.conversation_training_pack import (
    DialogueTrainingCase,
    DialogueTrainingPack,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationAssignment,
    EvaluationPlan,
    EvaluationProtocol,
    ProtocolKey,
    make_evaluation_data_identity,
)
from pure_integer_ai.experiments.language_generation_h2 import (
    TypedLanguageH2Case,
    TypedLanguageH2Expectation,
    TypedLanguageH2Protocol,
)
from pure_integer_ai.experiments.language_generation_floor import (
    TypedLanguageFloorProtocol,
    TypedLanguageFloorRequirement,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
)


_NAMESPACE = (21405, 1, 60)
_G04_DIMENSIONS = tuple(
    ProtocolKey((21405, 1, 15, index)) for index in range(1, 7))
_G04_VERIFIERS = tuple(
    ProtocolKey((21405, 1, 15, index)) for index in range(7, 13))
_V00_SPLITS = tuple(
    ProtocolKey((*_NAMESPACE, index)) for index in range(1, 6))
_V00_EVIDENCE = ProtocolKey((*_NAMESPACE, 10))
_V00_EXTERNAL_EVIDENCE = ProtocolKey((*_NAMESPACE, 11))
_V00_PROBE = ProtocolKey((*_NAMESPACE, 12))


@dataclass(frozen=True, slots=True)
class TypedDialogueEvaluationBundle:
    """一次公开 V-00 装配的 corpus、计划和两个分维协议。"""

    corpus: tuple[CollectedItem, ...]
    evaluation_plan: EvaluationPlan
    h2_protocol: TypedLanguageH2Protocol
    floor_protocol: TypedLanguageFloorProtocol
    development_item: CollectedItem
    held_out_item: CollectedItem


def _payload(case: DialogueTrainingCase) -> dict[str, object]:
    if case.typed_payload is None:
        raise ValueError("typed evaluation case 缺少显式 payload")
    value = case.typed_payload.to_value()
    if not isinstance(value, dict):
        raise ValueError("typed evaluation payload 必须是 object")
    return value


def _is_positive_postcheck(case: DialogueTrainingCase) -> bool:
    if case.typed_payload is None:
        return False
    value = _payload(case)
    return (
        case.payload_kind == "GenerationAdoptionPostcheckQuery"
        and case.expected_state == "TRUE"
        and isinstance(value.get("postcheck"), dict)
        and value["postcheck"].get("enabled") == 1
        and value["postcheck"].get("renderer_complete") == 1
    )


def _is_negative(case: DialogueTrainingCase) -> bool:
    if case.typed_payload is None:
        return False
    return case.expected_state in {"FALSE", "UNKNOWN", "CONFLICT"}


def _pick_case(
        cases: Iterable[DialogueTrainingCase],
        *,
        predicate,
        label: str,
        ) -> DialogueTrainingCase:
    matches = tuple(case for case in cases if predicate(case))
    if not matches:
        raise ValueError(f"公开 typed V-00 缺少 {label} case")
    return sorted(matches, key=lambda item: item.case_id)[0]


def _identity(case: DialogueTrainingCase, item: CollectedItem,
              *, split: ProtocolKey, training: bool) -> EvaluationAssignment:
    if item.source_ref is None:
        raise ValueError("V-00 item 必须先分配 SourceRef")
    case_key = (case.case_id, case.source_path, case.source_line)
    data = make_evaluation_data_identity(
        item,
        dedup_cluster=("typed-dialogue-v00", "dedup", case_key),
        provenance_cluster=("typed-dialogue-v00", "provenance", case_key),
    )
    return EvaluationAssignment(
        data,
        split,
        None if training else _V00_PROBE,
        () if training else _G04_DIMENSIONS,
        None if training else CanonicalIdentity.from_value((
            case.payload_kind,
            case.expected_state,
            ({} if case.expected_payload is None
             else case.expected_payload.to_value()),
        )),
    )


def _expectations(case: DialogueTrainingCase) -> tuple[TypedLanguageH2Expectation, ...]:
    """从 authored postcheck/source 字段读取唯一显式分维预期。"""
    value = _payload(case)
    expected_state = case.expected_state
    postcheck = value.get("postcheck")
    if not isinstance(postcheck, dict):
        raise ValueError("typed V-00 case 缺少 postcheck label")
    if expected_state != "TRUE":
        raise ValueError("V-00 H2/floor 只接受 authored TRUE postcheck case")
    source_requirements = postcheck.get("requirements")
    source_applicable = bool(source_requirements)
    source_support = all(
        isinstance(row, dict)
        and row.get("source_match") == 1
        and not row.get("refuted_source_ids")
        for row in source_requirements or ()
    )
    values = (
        (APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),
        (APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),
        (APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),
        (APPLICABILITY_NOT_APPLICABLE, VERDICT_UNKNOWN),
        (APPLICABILITY_APPLICABLE if source_applicable
         else APPLICABILITY_NOT_APPLICABLE,
         VERDICT_SUPPORT if source_support and source_applicable
         else VERDICT_UNKNOWN),
        (APPLICABILITY_NOT_APPLICABLE, VERDICT_UNKNOWN),
    )
    source_ref = case.source_ref
    if source_ref is None:
        raise ValueError("typed H2 case 缺 SourceRef")
    return tuple(
        TypedLanguageH2Expectation(
            dimension,
            verifier,
            applicability,
            verdict,
        )
        for (dimension, verifier), (applicability, verdict)
        in zip(zip(_G04_DIMENSIONS, _G04_VERIFIERS), values)
    )


def build_typed_dialogue_evaluation_bundle(
        pack: DialogueTrainingPack,
        items_by_case: dict[str, CollectedItem],
        ) -> TypedDialogueEvaluationBundle:
    """从公开 pack 装配严格五类 split；不复制或改写原始课程。"""
    if not isinstance(pack, DialogueTrainingPack):
        raise TypeError("pack 类型错误")
    cases = {case.case_id: case for case in pack.cases}
    if set(items_by_case) != set(cases):
        raise ValueError("V-00 items_by_case 必须完整覆盖 pack")
    development = _pick_case(
        cases.values(),
        predicate=lambda case: case.split == "train"
        and _is_positive_postcheck(case),
        label="development",
    )
    held_out = _pick_case(
        cases.values(),
        predicate=lambda case: case.split == "heldout"
        and _is_positive_postcheck(case),
        label="held-out",
    )
    adversarial = _pick_case(
        cases.values(),
        predicate=lambda case: case.split == "negative"
        and _is_negative(case),
        label="adversarial",
    )
    external = _pick_case(
        cases.values(),
        predicate=lambda case: case.split == "negative"
        and _is_negative(case)
        and case.case_id != adversarial.case_id,
        label="external",
    )
    selected_eval = {
        development.case_id: _V00_SPLITS[1],
        held_out.case_id: _V00_SPLITS[2],
        adversarial.case_id: _V00_SPLITS[3],
        external.case_id: _V00_SPLITS[4],
    }
    assignments = []
    corpus = []
    # 训练平面只消费 authored train；其它公开 heldout/negative 记录保留在
    # pack 但不被本次 V-00 corpus 隐式吞入。仅四个显式选择的探针进入评测。
    included = tuple(
        case for case in cases.values()
        if case.split == "train" or case.case_id in selected_eval)
    for case in sorted(included, key=lambda item: item.case_id):
        item = items_by_case[case.case_id]
        split = selected_eval.get(case.case_id, _V00_SPLITS[0])
        training = split == _V00_SPLITS[0]
        assignments.append(_identity(case, item, split=split, training=training))
        corpus.append(item)
    protocol = EvaluationProtocol(
        version=1,
        training_split=_V00_SPLITS[0],
        development_split=_V00_SPLITS[1],
        held_out_split=_V00_SPLITS[2],
        adversarial_split=_V00_SPLITS[3],
        external_split=_V00_SPLITS[4],
        statistical_evidence=_V00_EVIDENCE,
        external_evidence=_V00_EXTERNAL_EVIDENCE,
        required_dimensions=_G04_DIMENSIONS,
        required_adversarial_kinds=(_V00_PROBE,),
    )
    plan = EvaluationPlan(protocol, tuple(assignments))
    development_assignment = next(
        item for item in assignments if item.split == _V00_SPLITS[1])
    held_out_assignment = next(
        item for item in assignments if item.split == _V00_SPLITS[2])
    h2_case = TypedLanguageH2Case(
        development_assignment.identity,
        _expectations(development),
    )
    floor_case = TypedLanguageH2Case(
        held_out_assignment.identity,
        _expectations(held_out),
    )
    h2 = TypedLanguageH2Protocol(1, (h2_case,))
    floor = TypedLanguageFloorProtocol(
        1,
        (floor_case,),
        tuple(TypedLanguageFloorRequirement(
            dimension, verifier, 1000)
            for dimension, verifier in zip(_G04_DIMENSIONS, _G04_VERIFIERS)),
    )
    return TypedDialogueEvaluationBundle(
        tuple(corpus), plan, h2, floor,
        items_by_case[development.case_id], items_by_case[held_out.case_id],
    )


__all__ = [
    "TypedDialogueEvaluationBundle",
    "build_typed_dialogue_evaluation_bundle",
]
