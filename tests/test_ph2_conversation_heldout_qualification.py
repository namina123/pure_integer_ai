"""DLG-05 preflight qualification gate 的合取与 fail-closed 专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_heldout_preflight import (
    audit_dlg05_preflight_axis_inputs,
    build_dlg05_preflight_language_compiler,
    build_dlg05_typed_preflight_catalog,
    build_dlg05_unseen_relation_compiler,
    build_dlg05_unseen_source_compiler,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_MEMORY_MISS,
    ConversationHeldOutObservation,
    RESPONSE_UNKNOWN,
)
from pure_integer_ai.experiments.conversation_heldout_qualification import (
    conversation_heldout_rollback_fault_key,
    ConversationHeldOutQualificationError,
    ConversationHeldOutQualificationReceipt,
    ConversationHeldOutRollbackRecoveryReceipt,
    qualify_dlg05_preflight,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    ConversationHeldOutSelectionReceipt,
)


def _receipt():
    """建立不含 label 的完整资格输入，作为 gate 合同测试 fixture。"""
    catalog = build_dlg05_typed_preflight_catalog()
    manifest = catalog.to_manifest()
    audit = audit_dlg05_preflight_axis_inputs(
        catalog,
        compiler=build_dlg05_preflight_language_compiler(catalog),
        relation_compiler=build_dlg05_unseen_relation_compiler(catalog),
        source_compiler=build_dlg05_unseen_source_compiler(catalog),
    )
    observations = tuple(
        ConversationHeldOutObservation(
            case.case_key,
            RESPONSE_UNKNOWN,
            (RESPONSE_UNKNOWN,) * len(case.turns),
            (), (),
            2,
            (),
            0,
            tuple(case.axis_keys),
        )
        for index, case in enumerate(manifest.cases)
    )
    return catalog, manifest, observations, audit


def _selection(manifest, observations):
    """把无标签 observations 包装为 selection-first 协议 receipt。"""
    return ConversationHeldOutSelectionReceipt(
        manifest.stable_key(), observations)


def _qualification(catalog, manifest, observations, audit):
    """建立完整资格 receipt；所有状态均来自结构化 evidence。"""
    execution = _selection(manifest, observations)
    rollback = ConversationHeldOutRollbackRecoveryReceipt(
        conversation_heldout_rollback_fault_key(RuntimeError("injected")),
        (1,),
        (1,),
        (1,),
        execution,
    )
    return ConversationHeldOutQualificationReceipt(
        manifest.stable_key(),
        catalog.stable_key(),
        execution,
        execution,
        execution,
        execution,
        rollback,
        ((1,), (1,), (1,), (1,), (1,)),
        audit,
    )


def test_dlg05_preflight_qualification_accepts_complete_replays():
    catalog, manifest, observations, audit = _receipt()
    receipt = _qualification(catalog, manifest, observations, audit)
    assert qualify_dlg05_preflight(catalog, manifest, receipt) is receipt
    assert receipt.replay_stable
    assert receipt.storage_stable
    assert all(type(item) is int for item in receipt.stable_key())
    assert all(type(item) is int for item in observations[0].stable_key())


def test_dlg05_preflight_qualification_rejects_missing_axis_and_drift():
    catalog, manifest, observations, audit = _receipt()
    receipt = _qualification(catalog, manifest, observations, audit)
    missing = replace(
        receipt,
        execution=replace(
            receipt.execution,
            observations=tuple(replace(
                item,
                proven_axis_keys=tuple(
                    axis for axis in item.proven_axis_keys
                    if axis not in {AXIS_CONFLICT, AXIS_MEMORY_MISS}),
            ) for item in observations),
        ),
    )
    with pytest.raises(
            ConversationHeldOutQualificationError,
            match="runtime proof"):
        qualify_dlg05_preflight(catalog, manifest, missing)
    drift = replace(
        receipt,
        fresh_execution=replace(
            receipt.fresh_execution,
            observations=tuple(reversed(observations)),
        ),
    )
    with pytest.raises(
            ConversationHeldOutQualificationError,
            match="逐 case 覆盖"):
        qualify_dlg05_preflight(catalog, manifest, drift)
