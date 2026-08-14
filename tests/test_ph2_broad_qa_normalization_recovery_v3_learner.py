"""Recovery-v3 learner、pack 与 phrase runtime 专项测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_learner_runtime as learner_runtime,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_rule_pack as pack_runtime,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v3_training_protocol as protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_learner import (
    NORMALIZATION_RECOVERY_V3_CHECKPOINT_OPEN,
    read_normalization_recovery_v3_learner,
    run_normalization_recovery_v3_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_learning_records import (
    derive_normalization_recovery_v3_learning_outputs,
    normalization_recovery_v3_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_phrase_runtime import (
    compile_normalization_recovery_v3_phrase_program,
    execute_normalization_recovery_v3_phrase_program,
    reference_normalization_recovery_v3_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_rule_pack import (
    publish_normalization_recovery_v3_rule_pack,
    read_normalization_recovery_v3_rule_pack,
)


def _thunderbird_pair(
        pair_id: str,
        traditional: str,
        simplified: str,
        ) -> dict[str, object]:
    """构造最小 plain Thunderbird pair。"""
    side = {
        "file_sha256": "1" * 64,
        "source_slice_sha256": "2" * 64,
        "surface_text": "",
    }
    return {
        "attribute_id": "",
        "entry_kind": "MESSAGE",
        "message_id": pair_id,
        "pair_id": hashlib.sha256(pair_id.encode()).hexdigest(),
        "plain_pair_eligible": 1,
        "record_kind": "THUNDERBIRD_L10N_PATTERN_PAIR_V1",
        "relative_path": "test.ftl",
        "zh_cn": {**side, "surface_text": simplified},
        "zh_tw": {**side, "surface_text": traditional},
    }


def _godot_pair(
        pair_id: str,
        traditional: str,
        simplified: str,
        ) -> dict[str, object]:
    """构造最小训练可用 Godot PO pair。"""
    digest = hashlib.sha256(pair_id.encode()).hexdigest()
    return {
        "pair_id": digest,
        "record_kind": "GODOT_EDITOR_PO_PAIR_V1",
        "source_identity": {
            "msgctxt": "", "msgid": pair_id, "msgid_plural": ""},
        "training_eligible": 1,
        "zh_hans": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "4" * 64,
            "msgstr": simplified,
            "structure_tokens": [],
        },
        "zh_hant": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "5" * 64,
            "msgstr": traditional,
            "structure_tokens": [],
        },
    }


def _install_sources(
        monkeypatch: pytest.MonkeyPatch,
        *,
        collision: bool = False,
        ) -> None:
    """安装含多长度正例、identity 负例和可选签名冲突的来源。"""
    thunderbird = [
        _thunderbird_pair("short-support", "開啟", "打开"),
        _thunderbird_pair("long-support", "開啟檔案", "打开文件"),
        _thunderbird_pair("short-refute", "不要開啟", "不要開啟"),
        _thunderbird_pair("long-refute", "不要開啟檔案", "不要開啟檔案"),
        _thunderbird_pair("delete-support", "刪除", ""),
        _thunderbird_pair("delete-refute", "不要刪除", "不要刪除"),
    ]
    if collision:
        thunderbird.extend((
            _thunderbird_pair("collision-support", "前開啟後", "前打开後"),
            _thunderbird_pair("collision-refute", "前開啟後", "前開啟後"),
        ))
    godot = [
        _godot_pair("short-godot", "開啟", "打开"),
        _godot_pair("long-godot", "開啟檔案", "打开文件"),
        _godot_pair("delete-godot", "刪除", ""),
    ]
    if collision:
        godot.append(_godot_pair(
            "collision-godot", "前開啟後", "前打开後"))
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_evaluation_commitment",
        lambda *args, **kwargs: {"manifest_sha256": (
            protocol.V3_EVALUATION_COMMITMENT_MANIFEST_SHA256)},
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_thunderbird_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256)}, (),
                      tuple(thunderbird)),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_godot_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.GODOT_SOURCE_PACK_MANIFEST_SHA256)}, (), tuple(godot)),
    )


def _publish_protocol(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        collision: bool = False,
        ) -> tuple[Path, dict[str, object]]:
    """发布 synthetic v3 protocol 并替换测试内 K 盘检查。"""
    _install_sources(monkeypatch, collision=collision)
    monkeypatch.setattr(protocol, "_require_k_root", lambda value: Path(value))
    paths = []
    for name in ("prior", "commitment", "thunderbird", "godot"):
        path = tmp_path / name
        path.mkdir()
        paths.append(path)
    target = tmp_path / "protocol"
    report = protocol.publish_normalization_recovery_v3_training_protocol(
        run_root=tmp_path,
        prior_evaluation_protocol_dir=paths[0],
        evaluation_commitment_dir=paths[1],
        thunderbird_source_pack_dir=paths[2],
        godot_source_pack_dir=paths[3],
        target_dir=target,
    )
    monkeypatch.setattr(
        learner_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    monkeypatch.setattr(
        pack_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    return target, report


def _run_pair(
        tmp_path: Path,
        protocol_dir: Path,
        protocol_sha: str,
        ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    """执行 fresh 完整 run 与 pause/resume run。"""
    fresh = tmp_path / "fresh"
    resumed = tmp_path / "resumed"
    fresh_report = run_normalization_recovery_v3_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh,
        run_id=hashlib.sha256(b"v3-fresh").hexdigest(),
        mode="fresh",
        checkpoint_interval=3,
    )
    partial = run_normalization_recovery_v3_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"v3-resumed").hexdigest(),
        mode="fresh",
        checkpoint_interval=2,
        stop_after=3,
    )
    assert partial["status"] == NORMALIZATION_RECOVERY_V3_CHECKPOINT_OPEN
    resumed_report = run_normalization_recovery_v3_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=hashlib.sha256(b"v3-resumed").hexdigest(),
        mode="resume",
        checkpoint_interval=5,
    )
    return fresh, resumed, fresh_report, resumed_report


def test_v3_fresh_resume_pack_and_runtime_close_all_evidence(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """正负 Evidence、defeater、双运行 pack 和最长匹配必须共同闭合。"""
    protocol_dir, protocol_report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = str(protocol_report["manifest_sha256"])
    fresh, resumed, fresh_report, resumed_report = _run_pair(
        tmp_path, protocol_dir, protocol_sha)
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    assert fresh_report["summary"]["evidence_stance_counts"]["REFUTE"] > 0
    assert fresh_report["summary"]["phrase_rule_count"] >= 2
    for key in (
            "base_rule_pack_read_count", "candidate_pack_read_count",
            "evaluation_commitment_read_count", "evaluation_payload_read_count",
            "prior_formal_item_read_count", "reserve_identity_read_count",
            "reserve_payload_read_count", "source_pack_read_count",
            "teacher_api_llm_call_count"):
        assert fresh_report[key] == 0
    fresh_manifest, fresh_outputs = read_normalization_recovery_v3_learner(
        fresh,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    resumed_manifest, resumed_outputs = read_normalization_recovery_v3_learner(
        resumed,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    assert normalization_recovery_v3_output_payloads(fresh_outputs) == (
        normalization_recovery_v3_output_payloads(resumed_outputs))
    assert fresh_manifest["resume_markers"]["record_count"] == 0
    assert resumed_manifest["resume_markers"]["record_count"] == 1

    pack_dir = tmp_path / "pack"
    pack_report = publish_normalization_recovery_v3_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=pack_dir,
    )
    pack_manifest, pack_outputs = read_normalization_recovery_v3_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
    )
    assert pack_manifest["production_enabled"] == 0
    assert pack_manifest["fresh_resume_output_bytes_equal"] == 1
    program = compile_normalization_recovery_v3_phrase_program(
        rule_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        phrase_rules=pack_outputs["phrase-rules.jsonl"],
        defeaters=pack_outputs["defeaters.jsonl"],
        overlap_index=pack_outputs["overlap-index.jsonl"],
    )
    exact = execute_normalization_recovery_v3_phrase_program(
        program, "開啟檔案")
    assert exact["output_text"] == "打开文件"
    assert exact["steps"][0]["mode"] == "WHOLE_INPUT_EXACT"
    longest = execute_normalization_recovery_v3_phrase_program(
        program, "請開啟檔案")
    assert longest["output_text"] == "請打开文件"
    assert longest["steps"][1]["mode"] == "LONGEST_PHRASE_MATCH"
    blocked = execute_normalization_recovery_v3_phrase_program(
        program, "不要開啟檔案")
    assert blocked["output_text"] == "不要開啟檔案"
    assert any(step["blocked_defeater_ids"] for step in blocked["steps"])
    deleted = execute_normalization_recovery_v3_phrase_program(
        program, "刪除")
    assert deleted["output_text"] == ""
    assert deleted["steps"][0]["mode"] == "WHOLE_INPUT_EXACT"
    assert execute_normalization_recovery_v3_phrase_program(
        program, "請開啟檔案", character_rules={"請": "请"}) == (
        reference_normalization_recovery_v3_phrase_program(
            program, "請開啟檔案", character_rules={"請": "请"}))


def test_support_refute_same_context_signature_defers_family(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """相同 literal context 同时正反时不得生成可执行 phrase rule。"""
    protocol_dir, report = _publish_protocol(
        tmp_path, monkeypatch, collision=True)
    values = protocol.read_normalization_recovery_v3_learner_input(
        protocol_dir,
        expected_manifest_sha256=str(report["manifest_sha256"]),
    )
    outputs, summary, _counts = (
        derive_normalization_recovery_v3_learning_outputs(
            protocol_manifest=values[0],
            observations=values[1],
            fragments=values[2],
            groups=values[3],
            work=values[4],
        ))
    conflicts = [item for item in outputs["conflict-ledger.jsonl"]
                 if item["input_text"] == "開啟"]
    assert conflicts
    assert any(item["conflict_kind"]
               == "SUPPORT_REFUTE_CONTEXT_SIGNATURE_CONFLICT"
               for item in conflicts)
    assert not any(item["input_text"] == "開啟"
                   for item in outputs["phrase-rules.jsonl"])
    assert summary["defer_reason_counts"]["CONTEXT_SIGNATURE_CONFLICT"] >= 1


def test_v3_runtime_rejects_index_and_output_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """overlap index 或 learner JSONL 漂移都必须 fail closed。"""
    protocol_dir, report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = str(report["manifest_sha256"])
    fresh, _resumed, _fresh_report, _resumed_report = _run_pair(
        tmp_path, protocol_dir, protocol_sha)
    _manifest, outputs = read_normalization_recovery_v3_learner(
        fresh,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    index = list(outputs["overlap-index.jsonl"])
    index[0] = {**index[0], "input_scalar_length": 999}
    with pytest.raises(BroadQaExternalDataError, match="overlap index"):
        compile_normalization_recovery_v3_phrase_program(
            rule_pack_manifest_sha256="a" * 64,
            phrase_rules=outputs["phrase-rules.jsonl"],
            defeaters=outputs["defeaters.jsonl"],
            overlap_index=tuple(index),
        )
    path = fresh / "phrase-rules.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="protocol 派生漂移"):
        read_normalization_recovery_v3_learner(
            fresh,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
        )


def test_v3_runtime_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 learner 不得把训练 run 回退到 D 盘或临时目录。"""
    target = tmp_path / "forbidden"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        run_normalization_recovery_v3_learner(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            run_dir=target,
            run_id="b" * 64,
            mode="fresh",
        )
    assert not target.exists()
