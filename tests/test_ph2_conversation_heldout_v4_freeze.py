"""DLG-05 v4 freeze typed contract 专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_heldout_v4_freeze import (
    ConversationHeldOutV4FreezeError,
    freeze_v4_bundle,
    verify_v4_freeze,
)
from tests.test_ph2_conversation_heldout_v4_bundle import _bundle_fixture


def test_v4_freeze_recomputes_full_bundle_identity():
    """freeze 必须绑定完整 payload，而非只绑定 case/turn 数量。"""
    fixture, bundle, _turn, _source = _bundle_fixture()
    try:
        freeze = freeze_v4_bundle(bundle)
        verify_v4_freeze(bundle, freeze)
        assert freeze.bundle_payload_size == len(bundle.canonical_payload)
        assert freeze.bundle_payload_sha256 == bundle.payload_sha256
        assert not hasattr(freeze, "labels")
    finally:
        fixture.close()

def test_v4_freeze_rejects_digest_or_count_drift():
    """冻结凭证的摘要或计数被替换时必须停止。"""
    fixture, bundle, _turn, _source = _bundle_fixture()
    try:
        freeze = freeze_v4_bundle(bundle)
        with pytest.raises(ConversationHeldOutV4FreezeError, match="不一致"):
            verify_v4_freeze(
                bundle,
                replace(freeze, turn_count=freeze.turn_count + 1),
            )
        with pytest.raises(ConversationHeldOutV4FreezeError, match="不一致"):
            verify_v4_freeze(
                bundle,
                replace(freeze, bundle_payload_sha256=(1,) * 32),
            )
    finally:
        fixture.close()
