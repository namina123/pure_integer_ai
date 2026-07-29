"""W-01 正式语言 PH2 阶段 0 的入口、可见性和状态语义反例。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w01_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W01ContractError,
    W01RunRequest,
    open_w01_frozen_context,
    validate_w01_request,
)


_REPOSITORY = Path(__file__).resolve().parents[1]


def _context():
    """从公开仓库的正式 D-03 发布物打开唯一 W-01 上下文。"""
    return open_w01_frozen_context(
        _REPOSITORY,
        D03_GLOBAL_MANIFEST_PATH,
    )


def _request(context, **changes):
    """构造一个绑定完整 D-03 identity 的合法 SQLite 请求。"""
    request = W01RunRequest(
        run_id=101,
        parent_run_id=0,
        base_run_id=0,
        stage_key=context.stage_key,
        owner_key="PH2_W01_TRANSACTION_OWNER",
        runner_key="PH2_LANGUAGE_STAGE0",
        d03_context_key=context.stable_key(),
        backend_profile_key=(1, 20260729),
        base_fence_key=(1, 0, 20260729),
        worker_count=1,
        mode="fresh",
        requested_payload_paths=(),
    )
    return replace(request, **changes)


def test_remote_published_d03_is_only_w01_truth_and_has_no_candidate_payload():
    """W-01 只读发布 receipt/global/stage，candidate payload 白名单必须为空。"""
    context = _context()

    assert context.d03_published == 1
    assert context.d03_release_key == "PH2-D03-V1"
    assert context.stage_key == "W-01"
    assert context.stage_ordinal == 1
    assert context.train_pack_keys == ()
    assert context.candidate_allowed_paths == ()
    assert context.future_pack_count == 37
    assert context.held_out_visible_count == 0
    assert context.evaluator_visible_count == 0
    assert context.payload_reads == 0
    assert context.payload_bytes == 0
    assert context.logical_shard_count == 16
    assert context.allowed_worker_counts == (1, 2, 4)
    assert len(context.protocol_inputs) == 16


def test_context_open_never_decodes_future_or_private_payload(monkeypatch):
    """D-03 装配允许读协议 JSON，但任何 gzip payload 打开都应令测试立即失败。"""
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        """拒绝 payload 文件，放行合同和 manifest。"""
        if path.name.endswith(".jsonl.gz"):
            raise AssertionError(f"W-01 打开了禁止 payload: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    context = _context()
    assert context.payload_reads == 0
    assert context.payload_bytes == 0


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"stage_key": "W-02"}, "W-01"),
        ({"owner_key": "OTHER_OWNER"}, "owner"),
        ({"runner_key": "RUN_WEANING_TRAIN"}, "语言"),
        ({"worker_count": 3}, "worker"),
        ({"mode": "legacy"}, "mode"),
        ({"d03_context_key": (9, 9)}, "D-03"),
        ({"backend_profile_key": ()}, "backend"),
        ({"base_fence_key": ()}, "base fence"),
        ({"requested_payload_paths": ("future/train.jsonl.gz",)}, "payload"),
    ),
)
def test_entry_rejects_unknown_stage_owner_arithmetic_future_and_missing_versions(
        changes, match):
    """入口在任何执行前拒绝越级、旧算术路径、未来资料和缺版本。"""
    context = _context()
    with pytest.raises(W01ContractError, match=match):
        validate_w01_request(context, _request(context, **changes))


def test_resume_may_change_worker_scheduling_but_not_execution_identity():
    """worker 数只影响调度；run、D-03、backend 和 base fence 仍是完整身份。"""
    context = _context()
    fresh = validate_w01_request(context, _request(context))
    resumed = validate_w01_request(context, _request(
        context,
        worker_count=4,
        mode="resume",
    ))
    assert fresh.execution_identity_key() == resumed.execution_identity_key()
    assert fresh.scheduling_key() != resumed.scheduling_key()


def test_tampered_d03_receipt_with_published_zero_blocks_w01(tmp_path: Path):
    """overlay 中 superseding receipt 退回零时，即使其他 D-03 文件存在也必须停止。"""
    relative = Path(
        "data/ph2/manifests/d03_v1/ph2_d03_post_publication_receipt_v1.json")
    receipt = json.loads((_REPOSITORY / relative).read_text(encoding="utf-8"))
    receipt["execution_state"]["d03_published"] = 0
    receipt["publication_state"]["d03_published"] = 0
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(W01ContractError, match="d03_published"):
        open_w01_frozen_context(
            tmp_path,
            D03_GLOBAL_MANIFEST_PATH,
            dependency_root=_REPOSITORY,
        )


def test_protocol_verified_is_not_language_mastery_or_w02_start():
    """W-01 成功状态必须保持能力、readiness、teacher 和所有学习写为零。"""
    state = _context().verified_zero_learning_state(protocol_execution_runs=1)

    assert state["W01_PROTOCOL_VERIFIED"] == 1
    assert state["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert state["LANGUAGE_READINESS"] == 0
    assert state["W02_STARTED"] == 0
    assert state["formal_training_runs"] == 0
    assert state["protocol_execution_runs"] == 1
    assert state["teacher_calls"] == 0
    assert state["core_learning_writes"] == 0
    assert state["memory_learning_writes"] == 0
    assert state["companion_writes"] == 0
    assert state["use_learning_writes"] == 0
    assert state["w02_semantic_writes"] == 0
