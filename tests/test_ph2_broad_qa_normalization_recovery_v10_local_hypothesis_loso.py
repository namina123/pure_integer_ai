"""覆盖 recovery-v10 local hypothesis 三方向family LOSO。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_loso import (
    derive_normalization_recovery_v10_local_hypothesis_loso,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection import (
    derive_normalization_recovery_v10_local_hypothesis_projection,
)


def _sha(label: str) -> str:
    """返回稳定测试 SHA。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation(
        ordinal: int, family: str, input_text: str, output_text: str, *,
        source: str = "Save file",
        ) -> dict[str, object]:
    """构造可投影且可由held-out原文读取的最小Observation。"""
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


def _audit(
        observations: tuple[dict[str, object], ...], *,
        predecessor_source_rules: tuple[dict[str, object], ...] = (),
        ):
    """先投影零授权span，再运行LOSO。"""
    records, _summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=observations,
            opencc_routes={"檔": "档"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        ))
    return derive_normalization_recovery_v10_local_hypothesis_loso(
        observations=observations,
        projection_records=records,
        predecessor_source_rules=predecessor_source_rules,
    )


def test_loso_pass_requires_exact_in_all_three_held_out_families() -> None:
    """同source/context局部变化须在三方向均零错且有EXACT才成为survivor。"""
    observations = tuple(_observation(
        index, family, "儲存檔案", "儲存档案")
        for index, family in enumerate(V8_TRAIN_FAMILIES))
    audit, survivors = _audit(observations)
    assert audit["outcome"] == (
        "PASS_ZERO_WRONG_NOVEL_THREE_DIRECTION_SURVIVOR")
    assert audit["outcomes"] == {"EXACT": 3, "UNKNOWN": 0, "WRONG": 0}
    assert audit["survivor_count"] == 1
    assert audit["novel_survivor_count"] == 1
    assert audit["authorization_rule_count"] == 0
    assert len(survivors) == 1
    assert survivors[0]["input_text"] == "檔"
    assert survivors[0]["output_text"] == "档"
    assert all(item["outcomes"]["EXACT"] == 1
               for item in audit["directions"])


def test_loso_identity_held_out_occurrence_is_wrong_not_missing() -> None:
    """held-out恒等输出必须作为false-change WRONG进入分母。"""
    observations = (
        _observation(0, V8_TRAIN_FAMILIES[0], "儲存檔案", "儲存档案"),
        _observation(1, V8_TRAIN_FAMILIES[1], "儲存檔案", "儲存档案"),
        _observation(2, V8_TRAIN_FAMILIES[2], "儲存檔案", "儲存檔案"),
    )
    audit, survivors = _audit(observations)
    assert audit["outcome"] == "FAIL_WRONG_NONZERO"
    assert audit["outcomes"]["WRONG"] == 1
    assert audit["survivor_count"] == 0
    assert survivors == ()
    held = {item["held_out_family"]: item for item in audit["directions"]}
    assert held[V8_TRAIN_FAMILIES[2]]["outcomes"]["WRONG"] == 1


def test_loso_context_mismatch_stays_ne_instead_of_false_pass() -> None:
    """只有surface变化相同但局部上下文不跨family时不得宣布泛化。"""
    observations = (
        _observation(0, V8_TRAIN_FAMILIES[0], "儲存檔案", "儲存档案"),
        _observation(1, V8_TRAIN_FAMILIES[1], "儲存檔案", "儲存档案"),
        _observation(2, V8_TRAIN_FAMILIES[2], "下載檔案", "下載档案"),
    )
    audit, survivors = _audit(observations)
    assert audit["outcome"] == "NE_INCOMPLETE_FAMILY_COVERAGE"
    assert audit["outcomes"] == {"EXACT": 0, "UNKNOWN": 0, "WRONG": 0}
    assert audit["survivor_count"] == 0
    assert survivors == ()


def test_loso_predecessor_whole_input_slice_is_ne_not_new_capability() -> None:
    """已有whole-input source rule的局部分解不得计为新survivor。"""
    observations = tuple(_observation(
        index, family, "儲存檔案", "儲存档案")
        for index, family in enumerate(V8_TRAIN_FAMILIES))
    predecessor = ({
        "candidate_rule_id": _sha("predecessor-rule"),
        "input_text": "儲存檔案",
        "official_source_text": "Save file",
        "output_text": "儲存档案",
    },)
    audit, survivors = _audit(
        observations, predecessor_source_rules=predecessor)
    assert audit["outcome"] == "NE_PREDECESSOR_ONLY_SURVIVORS"
    assert audit["survivor_count"] == 1
    assert audit["novel_survivor_count"] == 0
    assert survivors[0]["predecessor_covered"] == 1
