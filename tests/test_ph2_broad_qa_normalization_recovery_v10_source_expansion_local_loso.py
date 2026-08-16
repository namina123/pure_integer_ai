"""覆盖 recovery-v10 五 family local projection/LOSO。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_loso import (
    derive_normalization_recovery_v10_source_expansion_local_loso,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_projection import (
    V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES,
    derive_normalization_recovery_v10_source_expansion_local_projection,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha(label: str) -> str:
    """返回稳定测试SHA。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation(
        ordinal: int,
        family: str,
        input_text: str,
        output_text: str,
        *,
        source: str = "Options",
        ) -> dict[str, object]:
    """构造可投影且可读取held-out实际输出的Observation。"""
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


def _projection(observations: tuple[dict[str, object], ...]):
    """运行五family局部投影。"""
    return derive_normalization_recovery_v10_source_expansion_local_projection(
        observations=observations,
        opencc_routes={"選": "选", "項": "项"},
        opencc_source_pack_manifest_sha256=_sha("opencc"),
    )


def _audit(
        observations: tuple[dict[str, object], ...],
        *,
        predecessor: tuple[dict[str, object], ...] = (),
        collisions: tuple[str, ...] = (),
        ):
    """先投影再运行带collision gate的五方向LOSO。"""
    records, _summary = _projection(observations)
    return derive_normalization_recovery_v10_source_expansion_local_loso(
        observations=observations,
        projection_records=records,
        predecessor_source_rules=predecessor,
        collision_source_input_sha256s=collisions,
    )


def test_five_family_loso_pass_requires_novel_zero_wrong_survivor() -> None:
    """五个held-out方向均非零EXACT且零WRONG才形成novel survivor。"""
    observations = tuple(_observation(
        index, family, "選項", "选项")
        for index, family in enumerate(
            V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES))
    records, summary = _projection(observations)
    audit, survivors = (
        derive_normalization_recovery_v10_source_expansion_local_loso(
            observations=observations,
            projection_records=records,
            predecessor_source_rules=(),
            collision_source_input_sha256s=(),
        ))
    assert summary["observation_count"] == 5
    assert summary["hypothesis_count"] == 5
    assert audit["outcome"] == (
        "PASS_ZERO_WRONG_NOVEL_FIVE_DIRECTION_SURVIVOR")
    assert audit["outcomes"] == {"EXACT": 5, "UNKNOWN": 0, "WRONG": 0}
    assert audit["direction_count"] == 5
    assert audit["novel_survivor_count"] == 1
    assert len(survivors) == 1


def test_five_family_loso_identity_held_out_is_wrong() -> None:
    """任一held-out family的恒等输出必须作为false-change WRONG。"""
    observations = tuple(_observation(
        index,
        family,
        "選項",
        "選項" if index == 2 else "选项",
    ) for index, family in enumerate(
        V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES))
    audit, survivors = _audit(observations)
    assert audit["outcome"] == "FAIL_WRONG_NONZERO"
    assert audit["outcomes"]["WRONG"] == 1
    assert survivors == ()


def test_five_family_collision_veto_and_predecessor_are_not_capability() -> None:
    """冻结冲突键不得训练；已有whole-input切片只能收口为NE。"""
    observations = tuple(_observation(
        index, family, "選項", "选项")
        for index, family in enumerate(
            V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES))
    collision = hashlib.sha256(canonical_json_line({
        "input_text": "選項",
        "official_source_text": "Options",
    })).hexdigest()
    audit, survivors = _audit(observations, collisions=(collision,))
    assert audit["outcome"] == "NE_INCOMPLETE_FAMILY_COVERAGE"
    assert audit["collision_matched_count"] == 1
    assert audit["collision_vetoed_projection_record_count"] == 5
    assert survivors == ()

    predecessor = ({
        "candidate_rule_id": _sha("predecessor"),
        "input_text": "選項",
        "official_source_text": "Options",
        "output_text": "选项",
    },)
    audit, survivors = _audit(observations, predecessor=predecessor)
    assert audit["outcome"] == "NE_PREDECESSOR_ONLY_SURVIVORS"
    assert audit["novel_survivor_count"] == 0
    assert survivors[0]["predecessor_covered"] == 1
