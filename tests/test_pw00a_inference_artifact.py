"""PW-00A W09 可装载推理状态 artifact 专项测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.pw00a_inference_artifact import (
    EXPECTED_INFERENCE_SHA256,
    build_pw00a_w09_inference_artifact,
    publish_pw00a_w09_inference_artifact,
    read_pw00a_w09_inference_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact_value():
    """本模块只重建一次公开 train-only 推理状态。"""
    return build_pw00a_w09_inference_artifact(ROOT)


def test_pw00a_inference_matches_sealed_w09_commitment(artifact_value) -> None:
    """299 条公开安全规则必须精确等于 W09 封存状态承诺。"""
    assert artifact_value["inference_state_sha256"] == EXPECTED_INFERENCE_SHA256
    state = artifact_value["inference_state"]
    assert len(state["rules"]) == 299
    assert state["training_record_count"] == 309
    assert artifact_value["reconstruction"] == {
        "candidate_root_reads": 0,
        "evaluator_label_reads": 0,
        "private_root_reads": 0,
        "rule_count": 299,
        "teacher_api_calls": 0,
        "training_evidence_count": 309,
        "training_observation_count": 309,
    }


def test_pw00a_inference_artifact_has_no_sensitive_payload(artifact_value) -> None:
    """公开 artifact 不得携带原文、答案、标签或外部封存根路径。"""
    encoded = canonical_json_bytes(artifact_value)
    for token in (
            b'"expected', b'"label', b'"surface', b'"raw_observation',
            b'"private_path', b'"candidate_root":', b'"rotation_root":'):
        assert token not in encoded


def test_pw00a_inference_publish_round_trip(
        artifact_value,
        tmp_path: Path,
        ) -> None:
    """临时独占发布必须规范回读并恢复同一推理状态。"""
    target = tmp_path / "pw00a-inference.json"
    value, state = publish_pw00a_w09_inference_artifact(ROOT, target=target)
    assert value == artifact_value
    assert state.sha256() == EXPECTED_INFERENCE_SHA256
    payload = target.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert read_pw00a_w09_inference_artifact(ROOT, target)[1] == state
    before = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_pw00a_w09_inference_artifact(ROOT, target=target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_pw00a_inference_reader_rejects_tamper(
        artifact_value,
        tmp_path: Path,
        ) -> None:
    """任一规则状态变化都必须在承诺回验处失败。"""
    changed = dict(artifact_value)
    inference = dict(changed["inference_state"])
    rules = [dict(item) for item in inference["rules"]]
    rules[0]["state"] = "UNKNOWN" if rules[0]["state"] != "UNKNOWN" else "TRUE"
    inference["rules"] = rules
    changed["inference_state"] = inference
    target = tmp_path / "tampered.json"
    target.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises((ValueError, RuntimeError), match="inference"):
        read_pw00a_w09_inference_artifact(ROOT, target)
