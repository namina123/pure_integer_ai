"""来源对齐广域问答 formal 冻结与唯一运行协议测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_formal_protocol import (
    FORMAL_FREEZE_NAME,
    FORMAL_INTENT_NAME,
    publish_formal_algorithm_freeze,
    publish_formal_run_intent,
    verify_formal_prediction_authorization,
    verify_formal_algorithm_freeze,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_QUESTION_KIND,
    predict_joint_retrieval,
    score_joint_retrieval,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha256(path: Path) -> str:
    """返回测试 artifact 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    """写入一份规范单行 JSON 测试 artifact。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_line(value))


def _formal_fixture(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """构造可交叉验证的最小 family、来源和开发门闭包。"""
    root = tmp_path / "run"
    family = root / "family"
    dev = root / "dev"
    population = root / "population"
    family.mkdir(parents=True)
    dev.mkdir()
    population.mkdir()
    files = {
        "candidate_manifest": population / "candidate-manifest.json",
        "census": population / "census.jsonl",
        "census_manifest": population / "census-manifest.json",
        "database": family / "joint-index.sqlite3",
        "dev_aggregate": dev / "dev.aggregate.json",
        "dev_labels": family / "dev.labels.jsonl",
        "dev_questions": family / "dev.questions.jsonl",
        "family_manifest": family / "manifest.json",
        "held_out_labels": family / "held_out.labels.jsonl",
        "held_out_questions": family / "held_out.questions.jsonl",
        "runtime_source_manifest": family / "runtime-source-manifest.json",
        "source_targets": family / "source_targets.jsonl",
        "alias_ledger": family / "source-aliases.jsonl",
        "terminal_selection": family / "terminal-selection.json",
        "predictions": family / "formal.predictions.jsonl",
        "aggregate": family / "formal.aggregate.json",
    }
    for role in (
            "candidate_manifest", "census", "census_manifest",
            "dev_labels", "dev_questions", "held_out_labels",
            "held_out_questions", "source_targets", "alias_ledger",
            "terminal_selection"):
        _write(files[role], {"role": role})
    files["database"].write_bytes(b"integer-index")
    thresholds = {
        "minimum_evidence_hit_ppm": 600_000,
        "minimum_recall_at_20_ppm": 800_000,
        "minimum_top1_source_hit_ppm": 700_000,
        "required_answer_citation_valid_ppm": 1_000_000,
    }
    artifacts = [
        {"role": role, "sha256": _sha256(files[role])}
        for role in (
            "dev_questions", "dev_labels", "held_out_questions",
            "held_out_labels", "source_targets")
    ]
    _write(files["family_manifest"], {
        "artifacts": artifacts,
        "candidate_manifest_sha256": _sha256(files["candidate_manifest"]),
        "census_manifest_sha256": _sha256(files["census_manifest"]),
        "census_sha256": _sha256(files["census"]),
        "status": "FROZEN_NOT_RUN",
        "thresholds": thresholds,
    })
    _write(files["runtime_source_manifest"], {
        "alias_sha256": _sha256(files["alias_ledger"]),
        "source_targets_sha256": _sha256(files["source_targets"]),
        "terminal_selection_sha256": _sha256(files["terminal_selection"]),
    })
    _write(files["dev_aggregate"], {
        "alias_sha256": _sha256(files["alias_ledger"]),
        "database_sha256": _sha256(files["database"]),
        "labels_sha256": _sha256(files["dev_labels"]),
        "questions_sha256": _sha256(files["dev_questions"]),
        "scope": "DEVELOPMENT",
        "status": "PASS",
        "target_selection_sha256": _sha256(files["terminal_selection"]),
        "thresholds": thresholds,
    })
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_broad_qa_formal_protocol."
        "_repository_identity",
        lambda repository_root: {
            "clean": 1, "head": "a" * 40, "origin_master": "a" * 40,
        })
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_broad_qa_formal_protocol."
        "_code_bindings",
        lambda repository_root: [{
            "bytes": 1, "relative_path": "algorithm.py", "sha256": "b" * 64,
        }])
    publish_formal_algorithm_freeze(
        root, family,
        candidate_manifest_path=files["candidate_manifest"],
        census_path=files["census"],
        census_manifest_path=files["census_manifest"],
        dev_aggregate_path=files["dev_aggregate"],
        database_path=files["database"],
        alias_path=files["alias_ledger"],
        terminal_selection_path=files["terminal_selection"],
        runtime_source_manifest_path=files["runtime_source_manifest"],
        predictions_path=files["predictions"],
        aggregate_path=files["aggregate"],
        repository_root=tmp_path)
    files["root"] = root
    files["family"] = family
    files["freeze"] = family / FORMAL_FREEZE_NAME
    files["intent"] = family / FORMAL_INTENT_NAME
    return files


def test_held_out_prediction_without_formal_authorization_is_rejected(
        tmp_path: Path) -> None:
    """普通 predict 不得直接消费任何 held_out questions。"""
    questions = tmp_path / "held-out.jsonl"
    questions.write_bytes(canonical_json_line({
        "format_version": 1,
        "item_id": "a" * 64,
        "license_id": "CC-BY-SA-4.0",
        "question": "谁主持修建都江堰？",
        "record_kind": JOINT_QUESTION_KIND,
        "source_key": "CMRC2018",
        "source_partition": "dev",
        "source_question_id": "q1",
        "source_revision": "r1",
        "split": "held_out",
        "upstream_url": "https://example.test/source",
    }))
    with pytest.raises(
            BroadQaExternalDataError,
            match="split/authorization"):
        predict_joint_retrieval(
            questions, tmp_path / "missing.sqlite3",
            predictions_path=tmp_path / "predictions.jsonl")
    assert not (tmp_path / "predictions.jsonl").exists()


def test_formal_freeze_rejects_bound_artifact_drift(
        tmp_path: Path, monkeypatch) -> None:
    """冻结后任一输入字节变化都必须在正式授权前 fail closed。"""
    files = _formal_fixture(tmp_path, monkeypatch)
    files["candidate_manifest"].write_bytes(b"tampered\n")
    with pytest.raises(BroadQaExternalDataError, match="artifact 漂移"):
        verify_formal_algorithm_freeze(
            files["root"], files["freeze"], repository_root=tmp_path)


def test_formal_intent_is_fixed_and_cannot_be_claimed_twice(
        tmp_path: Path, monkeypatch) -> None:
    """相同 family 即使换输出调用也不得占用第二次 formal run。"""
    files = _formal_fixture(tmp_path, monkeypatch)
    first = publish_formal_run_intent(
        files["root"], files["freeze"], repository_root=tmp_path)
    assert first["status"] == "OUTCOME_PENDING"
    with pytest.raises(BroadQaExternalDataError, match="已占用"):
        publish_formal_run_intent(
            files["root"], files["freeze"], repository_root=tmp_path)


def test_prediction_authorization_does_not_read_held_out_labels(
        tmp_path: Path, monkeypatch) -> None:
    """预测授权只读 questions/算法/索引，不得接触独立 labels。"""
    files = _formal_fixture(tmp_path, monkeypatch)
    publish_formal_run_intent(
        files["root"], files["freeze"], repository_root=tmp_path)
    original = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == files["held_out_labels"].resolve():
            raise AssertionError("prediction authorization read labels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = verify_formal_prediction_authorization(
        files["root"], files["freeze"], files["intent"],
        questions_path=files["held_out_questions"],
        database_path=files["database"],
        predictions_path=files["predictions"], repository_root=tmp_path)
    assert report["status"] == "AUTHORIZED"


def test_formal_score_authorizes_before_reading_labels(
        tmp_path: Path, monkeypatch) -> None:
    """正式 score 必须先过冻结链，不能先解析或探测 labels。"""
    selection = object.__new__(BroadQaTargetSelectionManifest)

    def reject(*args, **kwargs):
        raise BroadQaExternalDataError("authorization-first")

    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_broad_qa_joint_eval."
        "verify_formal_score_authorization", reject)
    with pytest.raises(BroadQaExternalDataError, match="authorization-first"):
        score_joint_retrieval(
            tmp_path / "missing.questions.jsonl",
            tmp_path / "missing.predictions.jsonl",
            tmp_path / "missing.labels.jsonl",
            selection,
            tmp_path / "missing.sqlite3",
            alias_path=tmp_path / "missing.aliases.jsonl",
            aggregate_path=tmp_path / "aggregate.json",
            scope="FORMAL_HELD_OUT",
            formal_run_root=tmp_path,
            formal_freeze_path=tmp_path / FORMAL_FREEZE_NAME,
            formal_intent_path=tmp_path / FORMAL_INTENT_NAME,
            formal_selection_path=tmp_path / "selection.json",
            repository_root=tmp_path)
