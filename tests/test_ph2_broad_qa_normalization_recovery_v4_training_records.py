"""normalization recovery-v4 三来源 TRAIN records 测试。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
    normalization_recovery_v3_pair_observation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    derive_normalization_recovery_v4_fragments,
    derive_normalization_recovery_v4_groups,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)


def _fragment(
        identity: str,
        *,
        input_text: str,
        output_text: str,
        source_family: str,
        source_policy_scope: str,
        kind: str = "CONTEXT_HUNK",
        ) -> dict[str, object]:
    """构造 group 纯测试所需的最小 fragment。"""
    return {
        "fragment_id": identity * 64,
        "fragment_kind": kind,
        "input_text": input_text,
        "output_text": output_text,
        "source_family": source_family,
        "source_policy_scope": source_policy_scope,
    }


def test_group_scope_separates_cross_family_source_only_conflict_and_defer(
        ) -> None:
    """四类 disposition 必须互斥，只有跨来源候选获得 target scope。"""
    fragments = (
        _fragment(
            "a", input_text="必須", output_text="必须",
            source_family=GODOT_SOURCE_FAMILY,
            source_policy_scope=GODOT_SOURCE_POLICY_SCOPE),
        _fragment(
            "b", input_text="必須", output_text="必须",
            source_family=VSCODE_SOURCE_FAMILY,
            source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE),
        _fragment(
            "c", input_text="來源", output_text="来源",
            source_family=VSCODE_SOURCE_FAMILY,
            source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE),
        _fragment(
            "d", input_text="來源", output_text="来源",
            source_family=VSCODE_SOURCE_FAMILY,
            source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE),
        _fragment(
            "e", input_text="內容", output_text="内容",
            source_family=GODOT_SOURCE_FAMILY,
            source_policy_scope=GODOT_SOURCE_POLICY_SCOPE),
        _fragment(
            "f", input_text="內容", output_text="内文",
            source_family=THUNDERBIRD_SOURCE_FAMILY,
            source_policy_scope=THUNDERBIRD_SOURCE_POLICY_SCOPE),
        _fragment(
            "1", input_text="單一", output_text="单一",
            source_family=GODOT_SOURCE_FAMILY,
            source_policy_scope=GODOT_SOURCE_POLICY_SCOPE,
            kind="EDIT_CORE"),
    )
    groups = derive_normalization_recovery_v4_groups(fragments)
    by_input = {str(item["input_text"]): item for item in groups}
    assert by_input["必須"]["disposition"] == (
        "CROSS_FAMILY_CONSENSUS_CANDIDATE")
    assert by_input["必須"]["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
    assert by_input["必須"]["target_policy_scope"] == (
        RECOVERY_V4_TARGET_POLICY_SCOPE)
    assert by_input["來源"]["disposition"] == "SOURCE_SCOPED_CANDIDATE"
    assert by_input["來源"]["candidate_scope_kind"] == "SOURCE_ONLY"
    assert by_input["來源"]["target_policy_scope"] == ""
    assert by_input["內容"]["disposition"] == "CONFLICT_DEFER"
    assert by_input["內容"]["candidate_scope_kind"] == "NONE"
    assert by_input["單一"]["disposition"] == "DEFER_INSUFFICIENT_AUTHORITY"


def test_fragment_derivation_reuses_frozen_v3_schema_deterministically() -> None:
    """v4 不复制对齐算法，并保持同 observation 的 fragment identity 稳定。"""
    observation = normalization_recovery_v3_pair_observation(
        source_family=VSCODE_SOURCE_FAMILY,
        source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
        license_id="MIT",
        source_pack_manifest_sha256="a" * 64,
        source_pair_id="b" * 64,
        input_text="匯入資訊",
        output_text="导入信息",
        structure_tokens=(),
        source_commitment={"json_path_sha256": "c" * 64},
    )
    first = derive_normalization_recovery_v4_fragments((observation,))
    second = derive_normalization_recovery_v4_fragments((observation,))
    assert first == second
    assert {item["fragment_kind"] for item in first} == {
        "CONTEXT_HUNK", "EDIT_CORE"}
    assert all(item["source_family"] == VSCODE_SOURCE_FAMILY
               for item in first)
