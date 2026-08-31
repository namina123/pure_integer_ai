"""公开对话训练的跨阶段就绪状态回归。"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.experiments.run_conversation_training import (
    _campaign_completion,
    _resume_completed_stages,
)
from pure_integer_ai.experiments.formal_train import _stage_items
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.cognition.shared.types import MODALITY_LANGUAGE
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


def test_local_stage4_ready_cannot_mask_missing_stage2() -> None:
    """局部 W-09 通过不等于完整训练 campaign 可发布。"""
    cumulative, ready, blockers = _campaign_completion(
        prior_completed_stages=(1, 3),
        local_completed_stages=(4,),
        stage_weaning_ready=True,
        stage_blockers=(),
    )

    assert cumulative == (1, 3, 4)
    assert not ready
    assert blockers == ("CAMPAIGN_STAGE_2_INCOMPLETE",)


def test_resume_lineage_requires_matching_pack_identity(tmp_path) -> None:
    """恢复谱系只可累计同一冻结 pack 的真实 completed stages。"""
    base = tmp_path / "base"
    base.mkdir()
    (base / "training_summary.json").write_text(json.dumps({
        "run_id": "base",
        "pack_sha256": "a" * 64,
        "stages_completed": [1, 2],
        "resume_from": None,
    }), encoding="utf-8")

    assert _resume_completed_stages(
        tmp_path, "base", expected_pack_sha256="a" * 64) == (1, 2)
    with pytest.raises(ValueError, match="pack SHA"):
        _resume_completed_stages(
            tmp_path, "base", expected_pack_sha256="b" * 64)


def test_resume_lineage_allows_explicit_additive_pack_lineage(tmp_path) -> None:
    """新增课程 shard 可复用多代基座，但默认冻结 pack 仍严格校验。"""
    base = tmp_path / "base"
    base.mkdir()
    (base / "training_summary.json").write_text(json.dumps({
        "run_id": "base",
        "pack_sha256": "a" * 64,
        "stages_completed": [1],
        "resume_from": None,
    }), encoding="utf-8")

    assert _resume_completed_stages(
        tmp_path, "base", expected_pack_sha256="b" * 64,
        allow_additive_pack=True) == (1,)

    child = tmp_path / "child"
    child.mkdir()
    (child / "training_summary.json").write_text(json.dumps({
        "run_id": "child",
        "pack_sha256": "c" * 64,
        "stages_completed": [2],
        "resume_from": "base",
    }), encoding="utf-8")
    assert _resume_completed_stages(
        tmp_path, "child", expected_pack_sha256="b" * 64,
        allow_additive_pack=True) == (1, 2)


def test_typed_stage_item_filter_keeps_observe_full_and_reward_typed_only() -> None:
    """性能档只裁 Stage 3+，且资格必须来自显式 typed 附件。"""
    plain = CollectedItem(
        source=0, modality=MODALITY_LANGUAGE, raw_text="普通对话",
        tokens=["普通", "对话"],
    )
    typed = CollectedItem(
        source=0, modality=MODALITY_LANGUAGE, raw_text="课程对话",
        tokens=["课程", "对话"],
        typed_payload=CanonicalJsonObject.from_value({"x": 1}),
        payload_kind="SyntheticTypedCourse",
    )
    corpus = [plain, typed]
    assert _stage_items(
        corpus, 1, 1, typed_language_stage_items_only=True) == corpus
    assert _stage_items(
        corpus, 3, 1, typed_language_stage_items_only=True) == [typed]
    assert _stage_items(corpus, 3, 1) == corpus
