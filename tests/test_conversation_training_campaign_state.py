"""公开对话训练的跨阶段就绪状态回归。"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.experiments.run_conversation_training import (
    _campaign_completion,
    _resume_completed_stages,
)


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
