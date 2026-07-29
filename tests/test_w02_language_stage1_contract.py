"""W-02 正式语言阶段的冻结输入、入口和读前可见性反例。"""
from __future__ import annotations

from dataclasses import replace
import gzip
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w02_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W02ContractError,
    W02PayloadAudit,
    W02PayloadFirewall,
    W02RunRequest,
    open_w02_frozen_context,
    validate_w02_request,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_BASE_REMOTE_COMMIT = "6322ed3d6aedf1a0fceeaffd1990ed5c9015e3f8"


def _context():
    """从公开 D-03/W-01 发布物打开唯一 W-02 冻结上下文。"""
    return open_w02_frozen_context(
        _REPOSITORY,
        D03_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=_BASE_REMOTE_COMMIT,
    )


def _request(context, **changes):
    """构造绑定全部 train-only identity 的合法 W-02 请求。"""
    request = W02RunRequest(
        run_id=2,
        parent_run_id=1,
        base_run_id=1,
        stage_key=context.stage_key,
        owner_key="PH2_W02_TRANSACTION_OWNER",
        runner_key="PH2_LANGUAGE_STAGE1",
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        d03_context_key=context.stable_key(),
        w01_receipt_sha256=context.w01_receipt_sha256,
        backend_profile_key=(1, 20260729),
        base_fence_key=(1, 1, 20260729),
        worker_count=1,
        mode="fresh",
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    return replace(request, **changes)


def test_context_binds_remote_d03_w01_and_exact_w02_visibility_without_gzip(
        monkeypatch):
    """冻结上下文只读 manifest/receipt，不得提前打开任何课程 gzip。"""
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        """把任一 payload gzip 读取变成即时失败。"""
        if path.suffix == ".gz":
            raise AssertionError(f"context 提前读取 payload: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    context = _context()

    assert context.current_remote_commit_sha1 == _BASE_REMOTE_COMMIT
    assert context.d03_published == 1
    assert context.d03_release_key == "PH2-D03-V1"
    assert context.d03_receipt_sha256 == (
        "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef"
    )
    assert context.w01_receipt_sha256 == (
        "48a0da96dd6caf3e42a10a755222ac4a9756cb788498b47a45ab50f260c02f57"
    )
    assert context.w01_status == "W01_PROTOCOL_VERIFIED"
    assert context.w01_run_id == 1
    assert context.w01_cursor_next_stage == "W-02"
    assert context.stage_key == "W-02"
    assert context.stage_ordinal == 2
    assert context.prerequisite_stage_keys == ("W-01",)
    assert context.payload_gets == 0
    assert context.payload_bytes == 0
    assert context.learning_writes == 0

    candidate = tuple(
        item.relative_path for item in context.candidate_payload_bindings)
    teacher = tuple(
        item.relative_path for item in context.teacher_evidence_bindings)
    assert len(candidate) == 4
    assert len(teacher) == 2
    assert all("held_out" not in path and "/owners/" not in path
               for path in candidate)
    assert all("/owners/teacher/" in path for path in teacher)
    assert len(context.evaluator_visible_paths) == 9
    assert len(context.evaluator_private_paths) == 7
    assert set(context.evaluator_private_paths) == (
        set(context.evaluator_visible_paths) - set(candidate))
    assert set(candidate).isdisjoint(context.evaluator_private_paths)
    assert set(teacher).isdisjoint(context.evaluator_private_paths)


def test_context_freezes_pack_identity_thresholds_recovery_and_budget():
    """W-02 合同完整绑定两个 pack、四维合取、恢复与资源预算。"""
    context = _context()

    assert context.train_pack_keys == (
        "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1",
        "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1",
    )
    assert tuple(item.record_count for item in context.pack_bindings) == (48, 66)
    assert tuple(item.license_id for item in context.pack_bindings) == (
        "CC0-1.0", "CC0-1.0")
    assert context.dimension_keys == (
        "W-02-BOUNDARY_WITHDRAWAL",
        "W-02-MULTI_CANDIDATE",
        "W-02-NEW_CONTENT_MORPHOLOGY",
        "W-02-OOV",
    )
    assert context.ablation_keys == tuple(
        f"{item}-ABLATION" for item in context.dimension_keys)
    assert context.aggregation_policy == "ALL_BEARING_DIMENSIONS_MUST_PASS"
    assert all(item.min_pass_numerator == item.min_pass_denominator == 1
               and item.max_fail_count == 0 and item.ne_policy == "BLOCK"
               for item in context.thresholds)
    assert context.allowed_worker_counts == (1, 2, 4)
    assert len(context.failure_point_keys) == 6
    assert context.logical_shard_count == 16
    assert context.resource_budget["max_payload_bytes"] == 134217728


def test_candidate_request_contains_no_evaluator_or_ud_field():
    """candidate 请求只能携带 train/teacher binding，不给私有路径留字段。"""
    context = _context()
    request = validate_w02_request(context, _request(context))

    assert not hasattr(request, "evaluator_paths")
    assert not hasattr(request, "held_out_paths")
    assert not hasattr(request, "ud_paths")
    assert set(request.candidate_payload_paths) == {
        item.relative_path for item in context.candidate_payload_bindings}
    assert set(request.teacher_evidence_paths) == {
        item.relative_path for item in context.teacher_evidence_bindings}


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"stage_key": "W-03"}, "W-02"),
        ({"owner_key": "PH2_W01_TRANSACTION_OWNER"}, "owner"),
        ({"runner_key": "RUN_WEANING_TRAIN"}, "runner"),
        ({"current_remote_commit_sha1": "0" * 40}, "commit"),
        ({"d03_context_key": (1,)}, "D-03"),
        ({"w01_receipt_sha256": "0" * 64}, "W-01"),
        ({"worker_count": 3}, "worker"),
        ({"mode": "skip"}, "mode"),
    ),
)
def test_request_rejects_stage_owner_identity_worker_and_mode_before_payload(
        changes, match):
    """入口身份任一漂移均在 payload get 和学习写之前失败。"""
    context = _context()

    with pytest.raises(W02ContractError, match=match):
        validate_w02_request(context, _request(context, **changes))
    assert context.payload_gets == context.payload_bytes == 0
    assert context.learning_writes == 0


def test_request_rejects_missing_extra_private_and_noncanonical_paths():
    """请求路径必须精确等于 reader 白名单，不能缺失、加私有路径或逃逸。"""
    context = _context()
    request = _request(context)
    private = context.evaluator_private_paths[0]
    variants = (
        replace(request, candidate_payload_paths=request.candidate_payload_paths[:-1]),
        replace(request, candidate_payload_paths=(
            *request.candidate_payload_paths, private)),
        replace(request, teacher_evidence_paths=(
            *request.teacher_evidence_paths, "../private.jsonl.gz")),
    )

    for variant in variants:
        with pytest.raises(W02ContractError, match="path|路径|白名单"):
            validate_w02_request(context, variant)
    assert context.payload_gets == context.payload_bytes == 0
    assert context.learning_writes == 0


def test_request_execution_identity_excludes_scheduling_only_fields():
    """worker/mode 只改变调度键，不改变同 run 的规范执行身份。"""
    context = _context()
    first = validate_w02_request(context, _request(context))
    resumed = validate_w02_request(
        context, _request(context, worker_count=4, mode="resume"))

    assert first.execution_identity_key() == resumed.execution_identity_key()
    assert first.scheduling_key() != resumed.scheduling_key()


def test_payload_firewall_reads_only_exact_train_and_replay_teacher_files():
    """防火墙只交付白名单 train/SourceRef 和录放 Evidence，并诚实计数。"""
    context = _context()
    audit = W02PayloadAudit()
    batch = W02PayloadFirewall.open(
        _REPOSITORY,
        context,
        _request(context),
        audit=audit,
    ).read_training_payload()

    assert len(batch.source_refs) == sum(
        item.record_count for item in context.candidate_payload_bindings
        if item.owner_kind == "source")
    assert len(batch.observations) == sum(
        item.record_count for item in context.candidate_payload_bindings
        if item.owner_kind == "observation")
    assert len(batch.teacher_evidence) == sum(
        item.record_count for item in context.teacher_evidence_bindings)
    assert all(item.w_stage == "W-02" and item.split == "train"
               for item in batch.observations)
    assert all(item.visible_from_stage == "W-02"
               for item in batch.teacher_evidence)
    assert audit.payload_gets == 6
    assert audit.payload_bytes == sum(
        item.transport_size_bytes
        for item in (
            *context.candidate_payload_bindings,
            *context.teacher_evidence_bindings,
        )
    )
    assert audit.teacher_evidence_reads == len(batch.teacher_evidence)
    assert audit.teacher_calls == 0
    assert audit.learning_writes == 0


def test_payload_firewall_rejects_bad_request_before_transport_or_learning():
    """入口漂移在任何 transport 尝试、payload 交付和学习写之前失败。"""
    context = _context()
    audit = W02PayloadAudit()
    bad = _request(context, worker_count=3)

    with pytest.raises(W02ContractError, match="worker"):
        W02PayloadFirewall.open(
            _REPOSITORY, context, bad, audit=audit)

    assert audit.transport_attempts == audit.transport_bytes == 0
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0


def test_payload_firewall_rejects_noncanonical_transport_without_delivery(
        tmp_path):
    """内容相同但 gzip header 漂移也不得交付给 candidate。"""
    context = _context()
    first = context.candidate_payload_bindings[0]
    source = _REPOSITORY / Path(*first.relative_path.split("/"))
    target = tmp_path / Path(*first.relative_path.split("/"))
    target.parent.mkdir(parents=True)
    content = gzip.decompress(source.read_bytes())
    target.write_bytes(gzip.compress(content, mtime=1))
    assert target.read_bytes() != source.read_bytes()

    audit = W02PayloadAudit()
    firewall = W02PayloadFirewall.open(
        tmp_path,
        context,
        _request(context),
        dependency_root=_REPOSITORY,
        audit=audit,
    )
    with pytest.raises(W02ContractError, match="transport|gzip|SHA-256"):
        firewall.read_training_payload()

    assert audit.transport_attempts == 1
    assert audit.transport_bytes == target.stat().st_size
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0
