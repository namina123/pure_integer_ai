"""Normalization recovery learner、checkpoint 与恢复边界测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_learner_runtime as runtime_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_rule_pack as pack_runtime_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_learner as learner_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_rule_pack as rule_pack_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_training_protocol as protocol_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learner import (
    NORMALIZATION_RECOVERY_CHECKPOINT_OPEN,
    read_normalization_recovery_learner,
    run_normalization_recovery_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    normalization_recovery_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_rule_pack import (
    publish_normalization_recovery_rule_pack,
    read_normalization_recovery_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_protocol import (
    read_normalization_recovery_learner_input,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    read_source_inference_learning_chain,
)
from test_ph2_broad_qa_normalization_recovery_training import _publish_protocol


def _prepare(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, dict[str, object]]:
    """发布 synthetic protocol，并只替换共享 runtime 的 K 盘检查。"""
    protocol, _sources, report, _calls = _publish_protocol(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "require_k_run_root",
        lambda value, *, label: Path(value).resolve(),
    )
    monkeypatch.setattr(
        pack_runtime_module,
        "require_k_run_root",
        lambda value, *, label: Path(value).resolve(),
    )
    return protocol, report


def _run_pair(
        tmp_path: Path,
        protocol: Path,
        protocol_sha: str,
        ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    """执行一条 fresh 完整 run 与一条 pause/resume run。"""
    fresh = tmp_path / "fresh-run"
    resumed = tmp_path / "resumed-run"
    fresh_report = run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh,
        run_id=hashlib.sha256(b"fresh-recovery-run").hexdigest(),
        mode="fresh",
        checkpoint_interval=3,
    )
    partial = run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-recovery-run").hexdigest(),
        mode="fresh",
        checkpoint_interval=4,
        stop_after=12,
    )
    assert partial["status"] == NORMALIZATION_RECOVERY_CHECKPOINT_OPEN
    resumed_report = run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-recovery-run").hexdigest(),
        mode="resume",
        checkpoint_interval=7,
    )
    return fresh, resumed, fresh_report, resumed_report


def test_fresh_resume_outputs_equal_and_all_external_reads_stay_zero(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """不同 checkpoint 切分必须产生逐字节相同的禁用态语义输出。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    fresh, resumed, fresh_report, resumed_report = _run_pair(
        tmp_path, protocol, report["manifest_sha256"])
    assert fresh_report["run_id"] != resumed_report["run_id"]
    assert fresh_report["resume_markers"]["record_count"] == 0
    assert resumed_report["resume_markers"]["record_count"] == 1
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    assert fresh_report["summary"]["evidence_count"] == 17
    assert fresh_report["summary"]["result_record_count"] == 15
    for key in (
            "candidate_pack_read_count", "evaluation_payload_read_count",
            "evaluation_protocol_manifest_read_count", "loso_audit_read_count",
            "prior_formal_item_read_count", "reserve_identity_read_count",
            "reserve_payload_read_count", "source_pack_read_count",
            "teacher_api_llm_call_count"):
        assert fresh_report[key] == 0
    fresh_read = read_normalization_recovery_learner(
        fresh,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
    )
    resumed_read = read_normalization_recovery_learner(
        resumed,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
    )
    assert normalization_recovery_output_payloads(fresh_read[1]) == (
        normalization_recovery_output_payloads(resumed_read[1]))
    target = tmp_path / "recovery-pack"
    pack_report = publish_normalization_recovery_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=target,
    )
    pack, pack_outputs = read_normalization_recovery_rule_pack(
        target,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
        expected_pack_manifest_sha256=pack_report["manifest_sha256"],
    )
    assert normalization_recovery_output_payloads(pack_outputs) == (
        normalization_recovery_output_payloads(fresh_read[1]))
    assert pack["runtime_state"] == "LEARNED_PACK_DISABLED"
    assert pack["fresh_resume_output_bytes_equal"] == 1
    assert pack["production_enabled"] == 0
    assert pack["mastery_claimed"] == 0
    assert pack["learner_run_read_count"] == 2
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            fresh_run_dir=fresh,
            resumed_run_dir=resumed,
            target_dir=target,
        )


def test_learner_runs_without_loso_and_never_calls_source_or_evaluation(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """LOSO audit 缺失不影响 learner，source/evaluation reader 不得被调用。"""
    protocol, report = _prepare(tmp_path, monkeypatch)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("recovery learner 不得读取 source/evaluation")

    monkeypatch.setattr(
        protocol_module,
        "read_normalization_recovery_evaluation_manifest_only",
        unexpected_read)
    monkeypatch.setattr(
        protocol_module, "read_normalization_source_pack", unexpected_read)
    monkeypatch.setattr(
        protocol_module, "read_normalization_icu_source_pack", unexpected_read)
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_successor_source_pack",
        unexpected_read)
    (protocol / "train.audit.loso.jsonl").unlink()
    run = tmp_path / "loso-free-run"
    report = run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
        run_dir=run,
        run_id=hashlib.sha256(b"loso-free-run").hexdigest(),
        mode="fresh",
    )
    assert report["loso_audit_read_count"] == 0


def test_resume_rejects_structurally_valid_wrong_prefix_counts(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """链形状合法也不能伪造已形成的 Evidence/result 数。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    run_dir = tmp_path / "drifted-run"
    run_id = hashlib.sha256(b"drifted-recovery-run").hexdigest()
    run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=run_dir,
        run_id=run_id,
        mode="fresh",
        checkpoint_interval=3,
        stop_after=6,
    )
    values = read_normalization_recovery_learner_input(
        protocol, expected_manifest_sha256=protocol_sha)
    work = values[5]
    chain_path = run_dir / "checkpoints.jsonl"
    chain = read_source_inference_learning_chain(chain_path)
    drifted = advance_source_inference_learning_checkpoint(
        chain[-1],
        training_item_ids=tuple(str(item["work_id"]) for item in work),
        processed_item_ids=tuple(str(item["work_id"]) for item in work[:7]),
        evidence_candidate_count=5,
        rule_candidate_count=0,
    )
    append_source_inference_learning_checkpoint(chain_path, drifted)
    with pytest.raises(BroadQaExternalDataError, match="prefix/count"):
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=run_dir,
            run_id=run_id,
            mode="resume",
        )


def test_invalid_stop_is_write_free_and_terminal_finalize_recovers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """非法停止不留 run；终态输出中断后可核验已有字节并继续封口。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    invalid = tmp_path / "invalid-stop"
    with pytest.raises(BroadQaExternalDataError, match="stop_after"):
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=invalid,
            run_id=hashlib.sha256(b"invalid-stop").hexdigest(),
            mode="fresh",
            stop_after=0,
        )
    assert not invalid.exists()

    interrupted = tmp_path / "interrupted-finalize"
    original = runtime_module.write_or_verify
    call_count = 0

    def interrupt_after_first(path: Path, payload: bytes, *, label: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic finalize interruption")
        original(path, payload, label=label)

    monkeypatch.setattr(runtime_module, "write_or_verify", interrupt_after_first)
    with pytest.raises(RuntimeError, match="finalize interruption"):
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=interrupted,
            run_id=hashlib.sha256(b"interrupted-finalize").hexdigest(),
            mode="fresh",
        )
    assert (interrupted / "checkpoints.jsonl").is_file()
    assert not (interrupted / "manifest.json").exists()
    monkeypatch.setattr(runtime_module, "write_or_verify", original)
    recovered = run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=interrupted,
        run_id=hashlib.sha256(b"interrupted-finalize").hexdigest(),
        mode="resume",
    )
    assert recovered["resume_markers"]["record_count"] == 0
    read_normalization_recovery_learner(
        interrupted,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
    )


def test_runtime_rejects_non_k_root_before_writing(tmp_path: Path) -> None:
    """正式 runtime 不得把训练 run 回退到 D 盘或临时目录。"""
    target = tmp_path / "forbidden-run"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            run_dir=target,
            run_id="b" * 64,
            mode="fresh",
        )
    assert not target.exists()


def test_reader_rejects_output_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """任一语义输出追加或修改都必须由 protocol 重派生 reader 拒绝。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    run = tmp_path / "tampered-run"
    run_normalization_recovery_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=report["manifest_sha256"],
        run_dir=run,
        run_id=hashlib.sha256(b"tampered-run").hexdigest(),
        mode="fresh",
    )
    path = run / "generic-rules.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="protocol 派生漂移"):
        read_normalization_recovery_learner(
            run,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
        )


def test_pack_rejects_two_fresh_runs_without_resume_lineage(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """输出相等也不能用第二条 fresh run 冒充 checkpoint resume。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    runs = []
    for name in ("fresh-a", "fresh-b"):
        path = tmp_path / name
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            run_dir=path,
            run_id=hashlib.sha256(name.encode()).hexdigest(),
            mode="fresh",
        )
        runs.append(path)
    with pytest.raises(BroadQaExternalDataError, match="lineage 漂移"):
        publish_normalization_recovery_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            fresh_run_dir=runs[0],
            resumed_run_dir=runs[1],
            target_dir=tmp_path / "invalid-pack",
        )


def test_shared_runtime_rejects_overlapping_artifact_roots(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """learner/pack 不得嵌入 protocol 或其他不可变 artifact 根。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    nested_run = protocol / "nested-run"
    with pytest.raises(BroadQaExternalDataError, match="不得重叠"):
        run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            run_dir=nested_run,
            run_id=hashlib.sha256(b"nested-run").hexdigest(),
            mode="fresh",
        )
    assert not nested_run.exists()

    fresh, resumed, _fresh_report, _resumed_report = _run_pair(
        tmp_path, protocol, report["manifest_sha256"])
    nested_pack = fresh / "nested-pack"
    with pytest.raises(BroadQaExternalDataError, match="混淆"):
        publish_normalization_recovery_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            fresh_run_dir=fresh,
            resumed_run_dir=resumed,
            target_dir=nested_pack,
        )
    assert not nested_pack.exists()


def test_shared_runtime_rejects_adapter_payload_drift_before_write(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """共享层不信任 adapter 声称的 JSONL payload 或 pack builder。"""
    protocol, report = _prepare(tmp_path, monkeypatch)
    original_material = learner_module.normalization_recovery_learning_material

    def drifted_material(protocol_dir: Path, protocol_sha: str):
        material = original_material(protocol_dir, protocol_sha)
        payloads = dict(material["payloads"])
        payloads["generic-rules.jsonl"] = b"{}\n"
        return {**material, "payloads": payloads}

    monkeypatch.setattr(
        learner_module,
        "normalization_recovery_learning_material",
        drifted_material,
    )
    target = tmp_path / "payload-drift-run"
    with pytest.raises(BroadQaExternalDataError, match="payload/record"):
        learner_module.run_normalization_recovery_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            run_dir=target,
            run_id=hashlib.sha256(b"payload-drift-run").hexdigest(),
            mode="fresh",
        )
    assert not target.exists()

    monkeypatch.setattr(
        learner_module,
        "normalization_recovery_learning_material",
        original_material,
    )
    fresh, resumed, _fresh_report, _resumed_report = _run_pair(
        tmp_path, protocol, report["manifest_sha256"])
    original_payloads = rule_pack_module.normalization_recovery_output_payloads

    def drifted_payloads(outputs):
        payloads = original_payloads(outputs)
        return {**payloads, "generic-rules.jsonl": b"{}\n"}

    monkeypatch.setattr(
        rule_pack_module,
        "normalization_recovery_output_payloads",
        drifted_payloads,
    )
    pack = tmp_path / "payload-drift-pack"
    with pytest.raises(BroadQaExternalDataError, match="payload/record"):
        rule_pack_module.publish_normalization_recovery_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=report["manifest_sha256"],
            fresh_run_dir=fresh,
            resumed_run_dir=resumed,
            target_dir=pack,
        )
    assert not pack.exists()
