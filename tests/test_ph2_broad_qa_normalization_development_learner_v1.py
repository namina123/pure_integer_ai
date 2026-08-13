"""normalization development learner 的纯归纳与运行边界测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_QUALIFICATIONS,
    publish_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_development_learner_v1 import (
    run_normalization_development_learner_v1,
    validate_normalization_development_checkpoint_chain_v1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_development_learning_v1 import (
    derive_normalization_development_records_v1,
    normalization_development_output_counts_for_prefix,
    normalization_development_qualification_groups,
    require_normalization_development_records_v1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    read_normalization_training_provenance,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    validate_normalization_rule_records_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    publish_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    initial_source_inference_learning_checkpoint,
)


def _sources(tmp_path: Path):
    """发布小型测试工作区中的真实冻结 OpenCC 来源与 TRAIN_SOURCE。"""
    source_pack = tmp_path / "source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=source_pack,
    )
    protocol = tmp_path / "contrastive-protocol"
    publish_normalization_contrastive_protocol(
        run_root=tmp_path,
        source_pack_dir=source_pack,
        target_dir=protocol,
    )
    return (
        source_pack,
        protocol,
        *read_normalization_training_provenance(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
        ),
    )


def test_development_learning_uses_only_candidates_with_both_qualifications(
        tmp_path: Path,
        ) -> None:
    """只有同一 mapping 同时有来源 SUPPORT/REFUTE 才能形成规则。"""
    (
        source_pack,
        protocol,
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        item_ids,
    ) = _sources(tmp_path)
    accepted, rejected = derive_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    groups = normalization_development_qualification_groups(trials)
    eligible = {
        candidate_id for candidate_id, values in groups.items()
        if set(values) == set(NORMALIZATION_CONTRASTIVE_QUALIFICATIONS)
    }
    assert len(item_ids) == 4395
    assert len(eligible) == len(accepted) == 3
    assert {item.candidate.mapping_candidate_id for item in accepted} == eligible
    assert len(rejected) == 80
    assert sum(len(item.evidence_commitments) for item in accepted) == 10
    assert sum(len(item.evidence_commitments) for item in rejected) == 80
    assert all(item.production_enabled == 0 for item in accepted)
    validate_normalization_rule_records_v3(
        source_pack_dir=source_pack,
        contrastive_protocol_dir=protocol,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )


def test_development_learning_is_order_independent_and_requires_refute(
        tmp_path: Path,
        ) -> None:
    """输入枚举顺序不影响结果，删去全部 REFUTE 后必须失败关闭。"""
    (
        _source_pack,
        _protocol,
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        _item_ids,
    ) = _sources(tmp_path)
    baseline = derive_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    reversed_result = derive_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=tuple(reversed(candidates)),
        trials=tuple(reversed(trials)),
    )
    assert baseline == reversed_result
    with pytest.raises(BroadQaExternalDataError, match="完整确定性输出"):
        require_normalization_development_records_v1(
            source_manifest=source_manifest,
            protocol_manifest=protocol_manifest,
            candidates=candidates,
            trials=trials,
            accepted_rules=baseline[0][1:],
            rejected_trials=baseline[1],
        )
    support_only = tuple(
        item for item in trials
        if item["qualification_kind"] == "SOURCE_REPLAY_SUPPORT")
    with pytest.raises(BroadQaExternalDataError, match="双资格来源不足"):
        derive_normalization_development_records_v1(
            source_manifest=source_manifest,
            protocol_manifest=protocol_manifest,
            candidates=candidates,
            trials=support_only,
        )


def test_development_checkpoint_counts_follow_the_frozen_training_prefix(
        tmp_path: Path,
        ) -> None:
    """候选段结束前无输出，完整 TRAIN 前缀精确对应最终 records。"""
    (
        _source_pack,
        _protocol,
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        item_ids,
    ) = _sources(tmp_path)
    accepted, rejected = derive_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    assert normalization_development_output_counts_for_prefix(
        candidates=candidates,
        trials=trials,
        processed_item_count=len(candidates),
    ) == (0, 0)
    final_counts = normalization_development_output_counts_for_prefix(
        candidates=candidates,
        trials=trials,
        processed_item_count=len(item_ids),
    )
    assert final_counts == (
        sum(len(item.evidence_commitments) for item in accepted + rejected),
        len(accepted) + len(rejected),
    ) == (90, 83)


def test_development_runtime_rejects_non_k_run_root_before_writing(
        tmp_path: Path,
        ) -> None:
    """训练运行器不得把 checkpoint 或 learner 输出回退到 D/临时目录。"""
    target = tmp_path / "development-run"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        run_normalization_development_learner_v1(
            run_root=tmp_path,
            source_pack_dir=tmp_path / "source-pack",
            contrastive_protocol_dir=tmp_path / "protocol",
            run_dir=target,
            run_id=hashlib.sha256(b"non-k-run").hexdigest(),
            mode="fresh",
        )
    assert not target.exists()


def test_development_resume_recomputes_checkpoint_candidate_counts(
        tmp_path: Path,
        ) -> None:
    """合法 checkpoint 链形状也不能伪造已归纳 Evidence/record 计数。"""
    (
        _source_pack,
        _protocol,
        _source_manifest,
        protocol_manifest,
        candidates,
        trials,
        item_ids,
    ) = _sources(tmp_path)
    run_id = hashlib.sha256(b"count-drift-run").hexdigest()
    initial = initial_source_inference_learning_checkpoint(
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        operator_family="NORMALIZATION_EQUIVALENCE",
        training_item_ids=item_ids,
    )
    drifted = advance_source_inference_learning_checkpoint(
        initial,
        training_item_ids=item_ids,
        processed_item_ids=item_ids[:4200],
        evidence_candidate_count=0,
        rule_candidate_count=0,
    )
    chain_path = tmp_path / "count-drift.checkpoints.jsonl"
    append_source_inference_learning_checkpoint(chain_path, initial)
    append_source_inference_learning_checkpoint(chain_path, drifted)
    with pytest.raises(BroadQaExternalDataError, match="prefix/count"):
        validate_normalization_development_checkpoint_chain_v1(
            chain_path=chain_path,
            run_id=run_id,
            protocol_sha=protocol_manifest["manifest_sha256"],
            training_item_ids=item_ids,
            candidates=candidates,
            trials=trials,
        )
