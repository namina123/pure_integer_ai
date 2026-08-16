"""覆盖 recovery-v10 TRAIN-only local hypothesis projection。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_contract import (
    LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY,
    execute_normalization_recovery_v10_local_hypothesis_contract,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection import (
    NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT,
    derive_normalization_recovery_v10_local_hypothesis_projection,
)


def _sha(label: str) -> str:
    """返回稳定测试 SHA。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation(
        ordinal: int, input_text: str, output_text: str, source: str,
        family: str,
        ) -> dict[str, object]:
    """构造带结构ledger与来源identity的最小TRAIN Observation。"""
    return {
        "observation_id": _sha(f"observation:{ordinal}"),
        "official_source_text": source,
        "source_family": family,
        "source_identity_sha256": _sha(f"source:{ordinal}"),
        "zh_hans": {"translation": output_text},
        "zh_hans_structure_tokens": list(
            localization_structure_tokens(output_text)),
        "zh_hant": {"translation": input_text},
        "zh_hant_structure_tokens": list(
            localization_structure_tokens(input_text)),
    }


def test_projection_emits_absolute_hypothesis_only_spans() -> None:
    """结构segment中的OpenCC changed scalar必须投影为绝对offset trace。"""
    observations = (
        _observation(
            0, "開啟 %s 檔案", "开启 %s 档案", "Open %s",
            V8_TRAIN_FAMILIES[0]),
    )
    records, summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=observations,
            opencc_routes={"開": "开", "啟": "启", "檔": "档"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        ))
    record = records[0]
    assert record["alignment_algorithm"] == (
        NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT)
    assert record["status"] == "HYPOTHESIS_PROJECTED"
    assert record["authorization_count"] == 0
    assert record["hypothesis_count"] == 2
    assert [(item["input_start"], item["input_end"], item["input_text"],
             item["output_text"]) for item in record["span_hypotheses"]] == [
        (0, 2, "開啟", "开启"),
        (6, 7, "檔", "档"),
    ]
    assert all(item["authorization_kind"]
               == LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY
               for item in record["span_hypotheses"])
    assert summary["hypothesis_count"] == 2
    assert summary["authorization_count"] == 0


def test_projection_trace_cannot_become_answer_without_authorization() -> None:
    """projection结果接入合同后仍必须UNKNOWN且answer无输出。"""
    observation = _observation(
        0, "端點不存在", "端点不存在", "Endpoint does not exist",
        V8_TRAIN_FAMILIES[1])
    records, _summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=(observation,),
            opencc_routes={"點": "点"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        ))
    result = execute_normalization_recovery_v10_local_hypothesis_contract(
        query={
            "input_text": records[0]["input_text"],
            "official_source_text": records[0]["official_source_text"],
            "structure_tokens": records[0]["structure_tokens"],
        },
        span_hypotheses=tuple(records[0]["span_hypotheses"]),
    )
    assert result["behavior"] == "UNKNOWN"
    assert result["output_text"] == ""
    assert result["hypothesis_only_span_count"] == 1
    assert result["partial_commit_count"] == 0


def test_semantic_or_variable_length_opcode_is_rejected_as_hypothesis() -> None:
    """混入非OpenCC改写或变长edit的opcode整段不得形成局部trace。"""
    observations = (
        _observation(
            0, "儲存", "保存", "Save", V8_TRAIN_FAMILIES[0]),
        _observation(
            1, "檔", "档案", "File", V8_TRAIN_FAMILIES[1]),
    )
    records, summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=observations,
            opencc_routes={"儲": "储", "檔": "档"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        ))
    assert all(item["status"] == "NO_SUPPORTED_LOCAL_HYPOTHESIS"
               for item in records)
    assert all(item["hypothesis_count"] == 0 for item in records)
    assert summary["hypothesis_count"] == 0
    assert summary["unsupported_opcode_count"] == 2


def test_structure_mismatch_is_counted_without_surface_hypothesis() -> None:
    """不同placeholder ledger不得跨结构segment对齐或产生span。"""
    observation = _observation(
        0, "開啟 %s", "开启 {0}", "Open placeholder",
        V8_TRAIN_FAMILIES[2])
    records, summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=(observation,),
            opencc_routes={"開": "开", "啟": "启"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        ))
    assert records[0]["status"] == "STRUCTURE_MISMATCH"
    assert records[0]["structure_equal"] == 0
    assert records[0]["hypothesis_count"] == 0
    assert summary["status_counts"] == {"STRUCTURE_MISMATCH": 1}
