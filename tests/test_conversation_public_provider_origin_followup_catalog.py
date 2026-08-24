"""DLG-RAW-11C 公开课程 adapter 的 source/readback 边界。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_provider_origin_followup_catalog import (
    PublicProviderOriginFollowupCatalogError,
    load_public_provider_origin_followup_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
)


_ROOT = Path(__file__).resolve().parents[1]
_LEXICAL_KEY = b"data/ph2/dlg_raw_public_provider_followup_lexical_v1_a.txt.sample"


def test_catalog_replays_all_source_bound_direction_profiles() -> None:
    """两个 form 为三种 DLG-RAW line framing 产生 24 条方向 profile。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    catalog = load_public_provider_origin_followup_catalog_from_closure(
        closure,
        load_public_proof_sentence_provider_from_root(_ROOT),
    )

    assert len(catalog.forms) == 2
    assert len(catalog.profiles) == 24
    assert {bytes(item.profile_key_u8) for item in catalog.profiles} == {
        b"provider-origin-causal-effect-cold-exact-v1",
        b"provider-origin-causal-effect-cold-exact-v1-lf",
        b"provider-origin-causal-effect-cold-exact-v1-crlf",
        b"provider-origin-causal-effect-rain-exact-v1",
        b"provider-origin-causal-effect-rain-exact-v1-lf",
        b"provider-origin-causal-effect-rain-exact-v1-crlf",
        b"provider-origin-causal-effect-cold-alias-v1",
        b"provider-origin-causal-effect-cold-alias-v1-lf",
        b"provider-origin-causal-effect-cold-alias-v1-crlf",
        b"provider-origin-causal-effect-rain-alias-v1",
        b"provider-origin-causal-effect-rain-alias-v1-lf",
        b"provider-origin-causal-effect-rain-alias-v1-crlf",
        b"provider-origin-causal-effect-cold-implicit-v1",
        b"provider-origin-causal-effect-cold-implicit-v1-lf",
        b"provider-origin-causal-effect-cold-implicit-v1-crlf",
        b"provider-origin-causal-effect-rain-implicit-v1",
        b"provider-origin-causal-effect-rain-implicit-v1-lf",
        b"provider-origin-causal-effect-rain-implicit-v1-crlf",
        b"provider-origin-causal-cause-cold-result-exact-v1",
        b"provider-origin-causal-cause-cold-result-exact-v1-lf",
        b"provider-origin-causal-cause-cold-result-exact-v1-crlf",
        b"provider-origin-causal-cause-rain-result-exact-v1",
        b"provider-origin-causal-cause-rain-result-exact-v1-lf",
        b"provider-origin-causal-cause-rain-result-exact-v1-crlf",
    }
    assert all(item.origin_focus_filler_key != item.target_filler_key
               for item in catalog.profiles)


def test_lexical_source_drift_is_rejected_before_profile_is_usable() -> None:
    """课程声明的 SHA/span 不匹配时不得输出部分 follow-up catalog。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    drifted = build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            record.raw_payload + b"x",
        ) if record.logical_key == _LEXICAL_KEY else record
        for record in closure.records
    ))
    with pytest.raises(PublicProviderOriginFollowupCatalogError, match="SHA-256"):
        load_public_provider_origin_followup_catalog_from_closure(
            drifted,
            load_public_proof_sentence_provider_from_root(_ROOT),
        )
