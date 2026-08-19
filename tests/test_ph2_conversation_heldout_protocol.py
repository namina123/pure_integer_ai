"""DLG-05 独立对话 held-out family 的冻结与 label-late 专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_EVENT_REFERENCE,
    AXIS_EXPLICIT_REPEAT,
    AXIS_MEMORY_MISS,
    AXIS_OMISSION,
    AXIS_ORDER,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_ROLLBACK,
    AXIS_SCOPE_DRIFT,
    AXIS_SYNONYM,
    AXIS_UNSEEN_RELATION,
    AXIS_UNSEEN_SOURCE,
    CONTEXT_CARRY,
    CONTEXT_EXPLICIT_REPEAT,
    CONTEXT_FRESH,
    CONTEXT_SCOPE_CHANGE,
    ConversationHeldOutCase,
    ConversationHeldOutLabel,
    ConversationHeldOutLabelSet,
    ConversationHeldOutManifest,
    ConversationHeldOutObservation,
    ConversationHeldOutProtocolError,
    ConversationHeldOutTurn,
    MEMORY_OFF,
    MEMORY_ON,
    RESPONSE_ANSWER,
    RESPONSE_CLARIFY,
    RESPONSE_CONFLICT,
    RESPONSE_UNKNOWN,
    evaluate_label_late,
    run_selection_first,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)


_FAMILY = ProtocolKey((20260819, 5, 1))
_REQUIRED_AXES = (
    AXIS_SYNONYM,
    AXIS_ORDER,
    AXIS_OMISSION,
    AXIS_EXPLICIT_REPEAT,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_EVENT_REFERENCE,
    AXIS_UNSEEN_SOURCE,
    AXIS_UNSEEN_RELATION,
    AXIS_CONFLICT,
    AXIS_MEMORY_MISS,
    AXIS_SCOPE_DRIFT,
    AXIS_ROLLBACK,
)


def _identity(kind: str, value: int) -> CanonicalIdentity:
    """构造只在评测边界存在的完整输入/簇身份。"""
    return CanonicalIdentity.from_value(("dlg05", kind, value))


def _turn(
        case: int,
        ordinal: int,
        *,
        memory_mode,
        context_mode,
        reference_mode,
        rollback_mode=(5, 0),
        ) -> ConversationHeldOutTurn:
    """构造一个不含 label 的 typed turn。"""
    return ConversationHeldOutTurn(
        ProtocolKey((20260819, 5, case, ordinal)),
        ordinal,
        _identity("content", case * 10 + ordinal),
        ProtocolKey((20260819, 51, case)),
        ProtocolKey((20260819, 52, case, ordinal)),
        context_mode,
        memory_mode,
        reference_mode,
        ProtocolKey(rollback_mode),
    )


def _case(case: int, axes, *, modes=(MEMORY_OFF, MEMORY_ON)):
    """构造两回合 case，确保显式覆盖 Memory ON/OFF。"""
    turns = (
        _turn(
            case,
            1,
            memory_mode=modes[0],
            context_mode=CONTEXT_FRESH,
            reference_mode=ProtocolKey((6, 0)),
        ),
        _turn(
            case,
            2,
            memory_mode=modes[1],
            context_mode=(
                CONTEXT_SCOPE_CHANGE if AXIS_SCOPE_DRIFT in axes
                else CONTEXT_CARRY
            ),
            reference_mode=ProtocolKey((6, 2 if AXIS_PROPOSITION_REFERENCE in axes
                                        else 3 if AXIS_EVENT_REFERENCE in axes
                                        else 0)),
            rollback_mode=(5, 1) if AXIS_ROLLBACK in axes else (5, 0),
        ),
    )
    return ConversationHeldOutCase(
        ProtocolKey((20260819, 50, case)),
        _FAMILY,
        tuple(axes),
        _identity("dedup", case),
        _identity("provenance", case),
        turns,
    )


def _manifest() -> ConversationHeldOutManifest:
    """冻结覆盖全部 DLG-05 轴的独立 family。"""
    cases = (
        _case(1, (AXIS_SYNONYM, AXIS_ORDER)),
        _case(2, (AXIS_OMISSION, AXIS_PROPOSITION_REFERENCE)),
        _case(3, (AXIS_EVENT_REFERENCE, AXIS_UNSEEN_SOURCE)),
        _case(4, (AXIS_UNSEEN_RELATION, AXIS_CONFLICT),
              modes=(MEMORY_ON, MEMORY_OFF)),
        _case(5, (AXIS_MEMORY_MISS, AXIS_SCOPE_DRIFT)),
        _case(6, (AXIS_EXPLICIT_REPEAT, AXIS_ROLLBACK)),
    )
    return ConversationHeldOutManifest(
        1,
        _FAMILY,
        (_identity("train-content", 1), _identity("train-content", 2)),
        (_identity("train-dedup", 1),),
        (_identity("train-provenance", 1),),
        cases,
        _REQUIRED_AXES,
        (MEMORY_OFF, MEMORY_ON),
    )


def _labels(manifest: ConversationHeldOutManifest) -> ConversationHeldOutLabelSet:
    """在与 manifest 分离的 owner 中建立 evaluator labels。"""
    labels = []
    for case in manifest.cases:
        if AXIS_CONFLICT in case.axis_keys:
            response = RESPONSE_CONFLICT
            selected = ()
            cited = ()
        elif AXIS_MEMORY_MISS in case.axis_keys:
            response = RESPONSE_UNKNOWN
            selected = ()
            cited = ()
        elif AXIS_SCOPE_DRIFT in case.axis_keys:
            response = RESPONSE_CLARIFY
            selected = ()
            cited = ()
        else:
            response = RESPONSE_ANSWER
            selected = ((7000, case.case_key.components[-1]),)
            cited = ((8000, case.case_key.components[-1]),)
        labels.append(ConversationHeldOutLabel(
            case.case_key,
            response,
            tuple(response for _ in case.turns),
            selected,
            cited,
        ))
    return ConversationHeldOutLabelSet(manifest.stable_key(), tuple(labels))


def _execute(case: ConversationHeldOutCase) -> ConversationHeldOutObservation:
    """模拟 selection-first typed consumer，只读取 case，不读取 labels。"""
    assert not hasattr(case, "expected")
    if AXIS_CONFLICT in case.axis_keys:
        response = RESPONSE_CONFLICT
        selected = ()
        cited = ()
    elif AXIS_MEMORY_MISS in case.axis_keys:
        response = RESPONSE_UNKNOWN
        selected = ()
        cited = ()
    elif AXIS_SCOPE_DRIFT in case.axis_keys:
        response = RESPONSE_CLARIFY
        selected = ()
        cited = ()
    else:
        response = RESPONSE_ANSWER
        selected = ((7000, case.case_key.components[-1]),)
        cited = ((8000, case.case_key.components[-1]),)
    return ConversationHeldOutObservation(
        case.case_key,
        response,
        tuple(response for _ in case.turns),
        selected,
        cited,
        len(case.turns) - 1,
        tuple(turn.stable_key() for turn in case.turns),
    )


def test_dlg05_manifest_is_independent_and_selection_first_label_late():
    """family 覆盖全部轴，执行不见 label，合并后才可评估。"""
    manifest = _manifest()
    labels = _labels(manifest)
    assert not hasattr(manifest, "labels")
    observations = run_selection_first(manifest, _execute)
    result = evaluate_label_late(manifest, labels, observations)
    assert result.complete
    assert result.total == result.passed == 6
    assert result.failed_case_keys == ()
    assert result.memory_off_total > 0
    assert result.memory_on_total > 0
    assert manifest.stable_key() == manifest.stable_key()


def test_dlg05_label_late_rejects_missing_or_mismatched_labels():
    """标签缺失、错挂 manifest 或期望不一致都必须显式失败。"""
    manifest = _manifest()
    labels = _labels(manifest)
    observations = run_selection_first(manifest, _execute)
    with pytest.raises(ConversationHeldOutProtocolError, match="manifest key"):
        evaluate_label_late(
            manifest,
            replace(labels, manifest_key=(999, 1)),
            observations,
        )
    with pytest.raises(ConversationHeldOutProtocolError, match="逐 case"):
        evaluate_label_late(
            manifest,
            replace(labels, labels=labels.labels[:-1]),
            observations,
        )
    wrong = replace(
        labels.labels[0],
        response_act=RESPONSE_UNKNOWN,
    )
    result = evaluate_label_late(
        manifest,
        replace(labels, labels=(wrong, *labels.labels[1:])),
        observations,
    )
    assert not result.complete
    assert result.failed_case_keys == (manifest.cases[0].case_key,)


def test_dlg05_manifest_rejects_train_leakage_and_unordered_turns():
    """content/dedup/provenance 泄漏和回合重排不能静默进入 family。"""
    manifest = _manifest()
    first = manifest.cases[0]
    with pytest.raises(ConversationHeldOutProtocolError, match="dedup"):
        ConversationHeldOutManifest(
            manifest.version,
            manifest.family_key,
            manifest.train_contents,
            (_identity("dedup", 1),),
            manifest.train_provenance_clusters,
            manifest.cases,
            manifest.required_axes,
            manifest.required_memory_modes,
        )
    with pytest.raises(ConversationHeldOutProtocolError, match="turns"):
        replace(first, turns=(first.turns[1], first.turns[0]))
    leaked = replace(
        first.turns[0],
        content=manifest.train_contents[0],
    )
    with pytest.raises(ConversationHeldOutProtocolError, match="content"):
        ConversationHeldOutManifest(
            manifest.version,
            manifest.family_key,
            manifest.train_contents,
            manifest.train_dedup_clusters,
            manifest.train_provenance_clusters,
            (replace(first, turns=(leaked, first.turns[1])), *manifest.cases[1:]),
            manifest.required_axes,
            manifest.required_memory_modes,
        )
