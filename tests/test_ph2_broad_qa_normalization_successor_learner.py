"""normalization successor learner、恢复链与禁用态 pack 测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_learner as learner_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_rule_pack as pack_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_training_protocol as protocol_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learner import (
    read_normalization_successor_learner,
    run_normalization_successor_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    derive_normalization_successor_learning_outputs,
    normalization_successor_output_payloads,
    normalization_successor_prefix_output_counts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_rule_pack import (
    publish_normalization_successor_rule_pack,
    read_normalization_successor_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_protocol import (
    ICU_TRAIN_SOURCE_MANIFEST_SHA256,
    OPENCC_TRAIN_SOURCE_MANIFEST_SHA256,
    publish_normalization_successor_training_protocol,
    read_normalization_successor_learner_input,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    read_source_inference_learning_chain,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _icu_rule(
        input_text: str,
        expected_output: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造带物理来源承诺的最小 ICU rule。"""
    encoded = f"{ordinal}:{input_text}>{expected_output}\n".encode()
    digest = hashlib.sha256(encoded).hexdigest()
    byte_start = ordinal * 100
    byte_end = byte_start + len(encoded)
    return {
        "byte_end": byte_end,
        "byte_start": byte_start,
        "line_end_ordinal": ordinal,
        "line_start_ordinal": ordinal,
        "physical_lines": [{
            "byte_end": byte_end,
            "byte_start": byte_start,
            "line_ordinal": ordinal,
            "line_sha256": digest,
        }],
        "statement_sha256": digest,
        "t2s_expected_output": expected_output,
        "t2s_input": input_text,
        "t2s_reverse_eligible": 1,
    }


def _publish_protocol(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, dict[str, object]]:
    """发布含共识、冲突、single-source 与 context 的 synthetic protocol。"""
    opencc = tmp_path / "opencc-source"
    icu = tmp_path / "icu-source"
    (opencc / "dictionary").mkdir(parents=True)
    icu.mkdir()
    (opencc / "dictionary" / "TSCharacters.txt").write_bytes(
        "鍾\t钟\n乾\t干\n於\t于\n".encode())
    (opencc / "dictionary" / "TSPhrases.txt").write_bytes(
        "鍾馗\t锺馗\n乾杯\t干杯\n".encode())
    rules = tuple(_icu_rule(*values, ordinal) for ordinal, values in enumerate((
        ("鍾", "钟"),
        ("乾", "干"),
        ("鐘", "钟"),
        ("鍾馗", "钟馗"),
        ("乾杯", "乾杯"),
    ), start=1))
    monkeypatch.setattr(protocol_module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_source_pack",
        lambda path: {"manifest_sha256": OPENCC_TRAIN_SOURCE_MANIFEST_SHA256},
    )
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_icu_source_pack",
        lambda path: (
            {"manifest_sha256": ICU_TRAIN_SOURCE_MANIFEST_SHA256}, (), rules),
    )
    target = tmp_path / "training-protocol"
    report = publish_normalization_successor_training_protocol(
        run_root=tmp_path,
        opencc_source_pack_dir=opencc,
        icu_source_pack_dir=icu,
        target_dir=target,
    )
    monkeypatch.setattr(
        learner_module, "_require_k_run_root", lambda value: Path(value).resolve())
    monkeypatch.setattr(
        pack_module, "_require_k_run_root", lambda value: Path(value).resolve())
    return target, report


def _material(
        protocol: Path,
        protocol_sha: str,
        ):
    """严格读取 synthetic protocol 并派生全部学习输出。"""
    values = read_normalization_successor_learner_input(
        protocol, expected_manifest_sha256=protocol_sha)
    outputs, summary = derive_normalization_successor_learning_outputs(
        protocol_manifest=values[0],
        observations=values[1],
        groups=values[2],
        contexts=values[3],
        work=values[4],
    )
    return values, outputs, summary


def _run_pair(
        tmp_path: Path,
        protocol: Path,
        protocol_sha: str,
        ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    """执行一条 fresh 完整 run 与一条 pause/resume run。"""
    fresh = tmp_path / "fresh-run"
    resumed = tmp_path / "resumed-run"
    fresh_report = run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh,
        run_id=hashlib.sha256(b"fresh-successor-run").hexdigest(),
        mode="fresh",
        checkpoint_interval=3,
    )
    partial = run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-successor-run").hexdigest(),
        mode="fresh",
        checkpoint_interval=2,
        stop_after=5,
    )
    assert partial["status"] == "SUCCESSOR_CHECKPOINT_OPEN"
    resumed_report = run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-successor-run").hexdigest(),
        mode="resume",
        checkpoint_interval=4,
    )
    return fresh, resumed, fresh_report, resumed_report


def test_learning_outputs_keep_policy_conflict_and_exact_context_rule(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """共识、冲突、defer 与 context override 形成不同一等记录。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    _values, outputs, summary = _material(
        protocol, report["manifest_sha256"])
    assert summary == {
        "conflict_ledger_count": 2,
        "consensus_rule_count": 2,
        "context_override_rule_count": 2,
        "context_replay_count": 4,
        "evidence_count": 12,
        "evidence_stance_counts": {"REFUTE": 2, "SUPPORT": 10},
        "identity_consensus_noop_count": 0,
        "result_record_count": 12,
        "single_source_defer_count": 2,
    }
    conflict = next(
        item for item in outputs["conflict-ledger.jsonl"]
        if item["input_text"] == "鍾馗")
    assert conflict["unscoped_application_allowed"] == 0
    assert {item["expected_output"] for item in conflict[
        "source_policy_outputs"]} == {"锺馗", "钟馗"}
    assert all(item["target_policy_scope"]
               == "ZH_HANS_CROSS_SOURCE_CONSENSUS_V1"
               for item in outputs["consensus-rules.jsonl"])
    assert all(item["target_policy_scope"] == ""
               for item in outputs["context-rules.jsonl"])
    assert {item["stance"] for item in outputs["evidence.jsonl"]} == {
        "SUPPORT", "REFUTE"}


def test_prefix_counts_follow_three_frozen_phases(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """checkpoint 计数只由已处理的 observation/group/context 前缀决定。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    values, _outputs, summary = _material(protocol, report["manifest_sha256"])
    observations, groups, contexts, work = values[1:]
    assert normalization_successor_prefix_output_counts(
        work=work, contexts=contexts,
        processed_item_count=len(observations)) == (len(observations), 0)
    assert normalization_successor_prefix_output_counts(
        work=work, contexts=contexts,
        processed_item_count=len(observations) + len(groups)) == (
            len(observations), len(groups))
    assert normalization_successor_prefix_output_counts(
        work=work, contexts=contexts,
        processed_item_count=len(work)) == (
            summary["evidence_count"], summary["result_record_count"])


def test_fresh_resume_runs_publish_disabled_pack(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """两条独立运行语义字节相等后才发布生产禁用 pack。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    fresh, resumed, fresh_report, resumed_report = _run_pair(
        tmp_path, protocol, protocol_sha)
    assert fresh_report["run_id"] != resumed_report["run_id"]
    assert fresh_report["resume_markers"]["record_count"] == 0
    assert resumed_report["resume_markers"]["record_count"] == 1
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    fresh_read = read_normalization_successor_learner(
        fresh,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    resumed_read = read_normalization_successor_learner(
        resumed,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    assert normalization_successor_output_payloads(fresh_read[1]) == (
        normalization_successor_output_payloads(resumed_read[1]))
    target = tmp_path / "successor-pack"
    pack_report = publish_normalization_successor_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=target,
    )
    pack, outputs = read_normalization_successor_rule_pack(
        target,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=pack_report["manifest_sha256"],
    )
    assert outputs
    assert pack["fresh_resume_output_bytes_equal"] == 1
    assert pack["runtime_state"] == "LEARNED_PACK_DISABLED"
    assert pack["production_enabled"] == 0
    assert pack["mastery_claimed"] == 0
    assert pack["evaluation_or_reserve_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_successor_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            fresh_run_dir=fresh,
            resumed_run_dir=resumed,
            target_dir=target,
        )


def test_resume_rejects_structurally_valid_wrong_prefix_counts(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """仅 checkpoint 链形状合法不能伪造已产生的 Evidence/record 数。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    run_dir = tmp_path / "drifted-run"
    run_id = hashlib.sha256(b"drifted-successor-run").hexdigest()
    run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=run_dir,
        run_id=run_id,
        mode="fresh",
        checkpoint_interval=2,
        stop_after=3,
    )
    values = read_normalization_successor_learner_input(
        protocol, expected_manifest_sha256=protocol_sha)
    work = values[4]
    chain_path = run_dir / "checkpoints.jsonl"
    chain = read_source_inference_learning_chain(chain_path)
    drifted = advance_source_inference_learning_checkpoint(
        chain[-1],
        training_item_ids=tuple(str(item["work_id"]) for item in work),
        processed_item_ids=tuple(
            str(item["work_id"]) for item in work[:4]),
        evidence_candidate_count=5,
        rule_candidate_count=0,
    )
    append_source_inference_learning_checkpoint(chain_path, drifted)
    with pytest.raises(BroadQaExternalDataError, match="prefix/count"):
        run_normalization_successor_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=run_dir,
            run_id=run_id,
            mode="resume",
        )


def test_pack_rejects_two_fresh_runs_without_resume_evidence(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """输出相等也不能用第二条 fresh run 冒充 checkpoint resume。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    run_a = tmp_path / "fresh-a"
    run_b = tmp_path / "fresh-b"
    for path, seed in ((run_a, b"fresh-a"), (run_b, b"fresh-b")):
        run_normalization_successor_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=path,
            run_id=hashlib.sha256(seed).hexdigest(),
            mode="fresh",
        )
    with pytest.raises(BroadQaExternalDataError, match="lineage 漂移"):
        publish_normalization_successor_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            fresh_run_dir=run_a,
            resumed_run_dir=run_b,
            target_dir=tmp_path / "invalid-pack",
        )


def test_learner_and_pack_reject_output_or_manifest_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """运行输出、pack material 或外部 manifest identity 漂移均失败关闭。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    fresh, resumed, _fresh_report, _resumed_report = _run_pair(
        tmp_path, protocol, protocol_sha)
    learner_file = fresh / "consensus-rules.jsonl"
    learner_file.write_bytes(learner_file.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="protocol 派生漂移"):
        read_normalization_successor_learner(
            fresh,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
        )
    valid_fresh = tmp_path / "valid-fresh"
    run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=valid_fresh,
        run_id=hashlib.sha256(b"valid-fresh").hexdigest(),
        mode="fresh",
    )
    resumed = tmp_path / "resumed-two"
    run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-two").hexdigest(),
        mode="fresh",
        stop_after=4,
    )
    run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"resumed-two").hexdigest(),
        mode="resume",
    )
    target = tmp_path / "pack-for-tamper"
    pack_report = publish_normalization_successor_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=valid_fresh,
        resumed_run_dir=resumed,
        target_dir=target,
    )
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["mastery_claimed"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest identity"):
        read_normalization_successor_rule_pack(
            target,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            expected_pack_manifest_sha256=pack_report["manifest_sha256"],
        )


def test_runtime_rejects_non_k_root_before_writing(tmp_path: Path) -> None:
    """未被测试替换的正式运行器不得把训练 run 回退到 D/临时目录。"""
    target = tmp_path / "forbidden-run"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        run_normalization_successor_learner(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            run_dir=target,
            run_id="b" * 64,
            mode="fresh",
        )
    assert not target.exists()


def test_invalid_stop_is_write_free_and_terminal_finalize_is_recoverable(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """参数错误不留 run；终态输出中断后可核验既有字节并继续封口。"""
    protocol, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = report["manifest_sha256"]
    invalid = tmp_path / "invalid-stop"
    with pytest.raises(BroadQaExternalDataError, match="stop_after"):
        run_normalization_successor_learner(
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
    original = learner_module._write_or_verify
    call_count = 0

    def interrupt_after_first(path: Path, payload: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic finalize interruption")
        original(path, payload)

    monkeypatch.setattr(learner_module, "_write_or_verify", interrupt_after_first)
    with pytest.raises(RuntimeError, match="finalize interruption"):
        run_normalization_successor_learner(
            run_root=tmp_path,
            protocol_dir=protocol,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=interrupted,
            run_id=hashlib.sha256(b"interrupted-finalize").hexdigest(),
            mode="fresh",
        )
    assert (interrupted / "checkpoints.jsonl").is_file()
    assert not (interrupted / "manifest.json").exists()
    monkeypatch.setattr(learner_module, "_write_or_verify", original)
    report = run_normalization_successor_learner(
        run_root=tmp_path,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=interrupted,
        run_id=hashlib.sha256(b"interrupted-finalize").hexdigest(),
        mode="resume",
    )
    assert report["resume_markers"]["record_count"] == 0
    read_normalization_successor_learner(
        interrupted,
        protocol_dir=protocol,
        expected_protocol_manifest_sha256=protocol_sha,
    )
