"""Normalization recovery TRAIN 记录与协议专项测试。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_training_protocol as protocol_module,
    ph2_broad_qa_normalization_recovery_training_records as records_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    NORMALIZATION_ICU_SOURCE_PACK_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_audit import (
    derive_normalization_recovery_loso,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_protocol import (
    ICU_RECOVERY_SOURCE_MANIFEST_SHA256,
    NORMALIZATION_RECOVERY_AUTHORITY_CONTRACT,
    NORMALIZATION_RECOVERY_LEARNER_CONTRACT,
    OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256,
    RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256,
    SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256,
    publish_normalization_recovery_training_protocol,
    read_normalization_recovery_learner_input,
    read_normalization_recovery_training_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    MEDIAWIKI_CN_SOURCE_POLICY_SCOPE,
    MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    UNIHAN_SOURCE_POLICY_SCOPE,
    derive_normalization_recovery_compositions,
    derive_normalization_recovery_groups,
    derive_normalization_recovery_source_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
    NORMALIZATION_SOURCE_PACK_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_source_pack import (
    MEDIAWIKI_CONVERSION_RECORD_KIND,
    NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
    UNIHAN_VARIANT_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _roster() -> tuple[dict[str, object], ...]:
    """构造五 policy、三 family 的最小冻结 roster。"""
    return derive_normalization_recovery_source_roster(
        opencc_manifest={
            "artifact_kind": NORMALIZATION_SOURCE_PACK_KIND,
            "manifest_sha256": "1" * 64,
            "package_version": "1.1.9",
        },
        icu_manifest={
            "artifact_kind": NORMALIZATION_ICU_SOURCE_PACK_KIND,
            "manifest_sha256": "2" * 64,
            "repository_commit": "icu-test-commit",
        },
        successor_manifest={
            "artifact_kind": NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
            "manifest_sha256": "3" * 64,
        },
    )


def _source_commitment(
        policy: str,
        input_text: str,
        output_text: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造符合各来源物理 schema 的最小承诺。"""
    digest = hashlib.sha256(f"line-{ordinal}\n".encode()).hexdigest()
    byte_start = ordinal * 100
    byte_end = byte_start + 20
    if policy == OPENCC_SOURCE_POLICY_SCOPE:
        relative_path = (
            "dictionary/TSCharacters.txt" if len(input_text) == 1
            else "dictionary/TSPhrases.txt")
        return {
            "byte_end": byte_end,
            "byte_start": byte_start,
            "file_sha256": NORMALIZATION_SOURCE_FILES[relative_path]["sha256"],
            "line_ordinal": ordinal,
            "line_sha256": digest,
            "relative_path": relative_path,
        }
    if policy == ICU_SOURCE_POLICY_SCOPE:
        line = {
            "byte_end": byte_end,
            "byte_start": byte_start,
            "line_ordinal": ordinal,
            "line_sha256": digest,
        }
        return {
            "byte_end": byte_end,
            "byte_start": byte_start,
            "line_end_ordinal": ordinal,
            "line_start_ordinal": ordinal,
            "physical_lines": [line],
            "statement_sha256": digest,
        }
    if policy == UNIHAN_SOURCE_POLICY_SCOPE:
        return {
            "byte_end": byte_end,
            "byte_start": byte_start,
            "line_ordinal": ordinal,
            "line_sha256": digest,
            "property_name": "kSimplifiedVariant",
            "source_codepoint": ord(input_text),
            "source_uplus": f"U+{ord(input_text):04X}",
            "targets": [{
                "codepoint": ord(output_text),
                "source_tags": [],
                "text": output_text,
                "uplus": f"U+{ord(output_text):04X}",
            }],
        }
    return {
        "byte_end": byte_end,
        "byte_start": byte_start,
        "line_ordinal": ordinal,
        "line_sha256": digest,
        "table_name": (
            "ZH_TO_HANS" if policy == MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE
            else "ZH_TO_CN"),
    }


def _observation(
        roster: tuple[dict[str, object], ...],
        policy: str,
        input_text: str,
        output_text: str,
        ordinal: int,
        ) -> dict[str, object]:
    """通过生产构造器建立一个带真实 roster binding 的 observation。"""
    by_policy = {item["source_policy_scope"]: item for item in roster}
    return records_module._observation(
        roster_record=by_policy[policy],
        input_text=input_text,
        expected_output=output_text,
        source_commitment=_source_commitment(
            policy, input_text, output_text, ordinal),
        target_variant_count=1,
        selected_target_variant_ordinal=0,
    )


def _ordered(values: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    """按协议 observation identity 固定输入顺序。"""
    return tuple(sorted(values, key=lambda item: str(item["observation_id"])))


def test_family_votes_regional_authority_and_phrase_composition() -> None:
    """同 family 不重复投票，区域精确 authority 与组合证据分账。"""
    roster = _roster()
    raw = [
        _observation(roster, ICU_SOURCE_POLICY_SCOPE, "甲", "一", 1),
        _observation(roster, UNIHAN_SOURCE_POLICY_SCOPE, "甲", "一", 2),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "乙", "一", 3),
        _observation(roster, MEDIAWIKI_CN_SOURCE_POLICY_SCOPE, "乙", "二", 4),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "丙", "一", 5),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "丙", "一", 6),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "丁", "一", 7),
        _observation(roster, ICU_SOURCE_POLICY_SCOPE, "丁", "一", 8),
        _observation(roster, UNIHAN_SOURCE_POLICY_SCOPE, "丁", "二", 9),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "丁", "一", 10),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "戊", "一", 11),
        _observation(roster, ICU_SOURCE_POLICY_SCOPE, "戊", "二", 12),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "舊", "旧", 13),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "舊", "旧", 14),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "詞", "词", 15),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "詞", "词", 16),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "舊詞", "旧词", 17),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "詞舊", "词故", 18),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "舊未", "旧新", 19),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "未知", "未知改", 20),
    ]
    observations = _ordered(raw)
    groups = derive_normalization_recovery_groups(
        roster=roster, observations=observations)
    by_input = {item["input_text"]: item for item in groups}

    assert by_input["甲"]["generic_resolution_kind"] == "SINGLE_FAMILY_DEFER"
    assert by_input["甲"]["source_family_count"] == 1
    assert by_input["乙"]["target_resolution_kind"] == (
        "REGIONAL_EXACT_AUTHORITY")
    assert by_input["乙"]["target_output"] == "二"
    assert by_input["丙"]["generic_resolution_kind"] == (
        "CROSS_FAMILY_CONSENSUS")
    assert by_input["丁"]["generic_resolution_kind"] == (
        "INTRA_FAMILY_CONFLICT")
    assert by_input["丁"]["target_resolution_kind"] == "NO_TARGET_AUTHORITY"
    assert by_input["戊"]["generic_resolution_kind"] == (
        "SOURCE_FAMILY_CONFLICT")

    compositions = derive_normalization_recovery_compositions(
        observations=observations, groups=groups)
    qualifications = {
        item["input_text"]: item["qualification_kind"]
        for item in compositions
    }
    assert qualifications["舊詞"] == "COMPOSITION_SUPPORT"
    assert qualifications["詞舊"] == "EXPLICIT_OVERRIDE"
    assert qualifications["舊未"] == "PARTIAL_COMPOSITION"
    assert qualifications["未知"] == "NO_COMPOSITION_EVIDENCE"
    partial = next(item for item in compositions if item["input_text"] == "舊未")
    assert partial["base_output"] == "旧未"
    assert partial["covered_positions"] == [0]


def test_loso_separates_exact_wrong_and_unknown() -> None:
    """TRAIN-only LOSO 显式区分可预测、冲突预测与无 authority。"""
    roster = _roster()
    observations = _ordered([
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "甲", "一", 1),
        _observation(roster, ICU_SOURCE_POLICY_SCOPE, "甲", "一", 2),
        _observation(roster, MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, "甲", "一", 3),
        _observation(roster, OPENCC_SOURCE_POLICY_SCOPE, "乙", "一", 4),
        _observation(roster, ICU_SOURCE_POLICY_SCOPE, "乙", "一", 5),
        _observation(roster, MEDIAWIKI_CN_SOURCE_POLICY_SCOPE, "乙", "二", 6),
        _observation(roster, UNIHAN_SOURCE_POLICY_SCOPE, "丙", "一", 7),
    ])
    loso = derive_normalization_recovery_loso(
        roster=roster, observations=observations)
    outcomes = {
        (item["held_out_source_policy_scope"], item["input_text"]):
        item["outcome"] for item in loso
    }
    assert outcomes[(OPENCC_SOURCE_POLICY_SCOPE, "甲")] == "EXACT"
    assert outcomes[(MEDIAWIKI_CN_SOURCE_POLICY_SCOPE, "乙")] == "WRONG"
    assert outcomes[(UNIHAN_SOURCE_POLICY_SCOPE, "丙")] == "UNKNOWN"


def test_observation_validation_rejects_source_commitment_drift() -> None:
    """即便攻击者同步重算 observation id，来源 schema 漂移仍失败关闭。"""
    roster = _roster()
    observation = _observation(
        roster, UNIHAN_SOURCE_POLICY_SCOPE, "甲", "一", 1)
    observation["source_commitment"]["targets"][0]["source_tags"] = ["kIRG_TSource"]
    identity = {key: observation[key] for key in (
        "expected_output", "input_text", "source_commitment",
        "source_pack_manifest_sha256", "source_policy_scope")}
    observation["observation_id"] = hashlib.sha256(
        records_module.canonical_json_bytes(identity)).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="Unihan target commitment"):
        derive_normalization_recovery_groups(
            roster=roster, observations=(observation,))


def _icu_rule(
        input_text: str,
        output_text: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造可被 ICU adapter 接受的单 physical-line rule。"""
    digest = hashlib.sha256(f"icu-{ordinal}\n".encode()).hexdigest()
    byte_start = ordinal * 100
    byte_end = byte_start + 20
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
        "t2s_expected_output": output_text,
        "t2s_input": input_text,
        "t2s_reverse_eligible": 1,
    }


def _unihan_record(
        input_text: str,
        output_text: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造可被 Unihan adapter 接受的单 target 简化边。"""
    digest = hashlib.sha256(f"unihan-{ordinal}\n".encode()).hexdigest()
    return {
        "byte_end": ordinal * 100 + 20,
        "byte_start": ordinal * 100,
        "format_version": 1,
        "line_ordinal": ordinal,
        "line_sha256": digest,
        "property_name": "kSimplifiedVariant",
        "record_kind": UNIHAN_VARIANT_RECORD_KIND,
        "source_codepoint": ord(input_text),
        "source_text": input_text,
        "source_uplus": f"U+{ord(input_text):04X}",
        "t2s_expected_output": output_text,
        "t2s_input": input_text,
        "t2s_unambiguous_eligible": 1,
        "targets": [{
            "codepoint": ord(output_text),
            "source_tags": [],
            "text": output_text,
            "uplus": f"U+{ord(output_text):04X}",
        }],
    }


def _mediawiki_record(
        table_name: str,
        input_text: str,
        output_text: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造可被 MediaWiki adapter 接受的冻结表项。"""
    digest = hashlib.sha256(f"mediawiki-{ordinal}\n".encode()).hexdigest()
    return {
        "byte_end": ordinal * 100 + 20,
        "byte_start": ordinal * 100,
        "expected_output": output_text,
        "format_version": 1,
        "input_scalar_count": len(input_text),
        "input_text": input_text,
        "is_identity": int(input_text == output_text),
        "line_ordinal": ordinal,
        "line_sha256": digest,
        "output_scalar_count": len(output_text),
        "record_kind": MEDIAWIKI_CONVERSION_RECORD_KIND,
        "table_name": table_name,
    }


def _publish_protocol(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, tuple[Path, Path, Path, Path], dict[str, object], list[str]]:
    """用完整 adapter 路径发布 synthetic recovery protocol。"""
    opencc = tmp_path / "opencc-source"
    icu = tmp_path / "icu-source"
    successor = tmp_path / "successor-source"
    evaluation = tmp_path / "evaluation-protocol"
    (opencc / "dictionary").mkdir(parents=True)
    icu.mkdir()
    successor.mkdir()
    evaluation.mkdir()
    (opencc / "dictionary" / "TSCharacters.txt").write_bytes(
        "甲\t一\n舊\t旧\n詞\t词\n".encode())
    (opencc / "dictionary" / "TSPhrases.txt").write_bytes(
        "舊詞\t旧词\n".encode())
    icu_rules = (
        _icu_rule("甲", "一", 1),
        _icu_rule("乙", "一", 2),
    )
    unihan_records = (_unihan_record("甲", "一", 1),)
    mediawiki_records = (
        _mediawiki_record("ZH_TO_HANS", "甲", "一", 1),
        _mediawiki_record("ZH_TO_HANS", "舊", "旧", 2),
        _mediawiki_record("ZH_TO_HANS", "詞", "词", 3),
        _mediawiki_record("ZH_TO_CN", "乙", "二", 4),
    )
    opencc_manifest = {
        "artifact_kind": NORMALIZATION_SOURCE_PACK_KIND,
        "manifest_sha256": OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256,
        "package_version": "1.1.9",
    }
    icu_manifest = {
        "artifact_kind": NORMALIZATION_ICU_SOURCE_PACK_KIND,
        "manifest_sha256": ICU_RECOVERY_SOURCE_MANIFEST_SHA256,
        "repository_commit": "icu-test-commit",
    }
    successor_manifest = {
        "artifact_kind": NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
        "manifest_sha256": SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256,
    }
    calls: list[str] = []

    monkeypatch.setattr(
        protocol_module, "_require_k_root", lambda value: Path(value))

    def read_evaluation(path, *, expected_manifest_sha256):
        calls.append("evaluation-manifest-only")
        assert Path(path) == evaluation
        assert expected_manifest_sha256 == (
            RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256)
        return {
            "artifact_kind": NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
            "status": NORMALIZATION_RECOVERY_EVALUATION_STATUS,
        }

    def read_opencc(path):
        calls.append("opencc")
        assert Path(path) == opencc
        return opencc_manifest

    def read_icu(path):
        calls.append("icu")
        assert Path(path) == icu
        return icu_manifest, (), icu_rules

    def read_successor(path):
        calls.append("successor")
        assert Path(path) == successor
        return successor_manifest, unihan_records, mediawiki_records

    monkeypatch.setattr(
        protocol_module,
        "read_normalization_recovery_evaluation_manifest_only",
        read_evaluation)
    monkeypatch.setattr(
        protocol_module, "read_normalization_source_pack", read_opencc)
    monkeypatch.setattr(
        protocol_module, "read_normalization_icu_source_pack", read_icu)
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_successor_source_pack",
        read_successor)
    target = tmp_path / "recovery-training-protocol"
    report = publish_normalization_recovery_training_protocol(
        run_root=tmp_path,
        opencc_source_pack_dir=opencc,
        icu_source_pack_dir=icu,
        successor_source_pack_dir=successor,
        evaluation_protocol_dir=evaluation,
        target_dir=target,
    )
    return target, (opencc, icu, successor, evaluation), report, calls


def test_protocol_round_trip_freezes_order_partitions_and_zero_reads(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """evaluation freeze 必须先于来源读取，learner contract 保持零泄漏。"""
    target, sources, report, calls = _publish_protocol(tmp_path, monkeypatch)
    assert calls == ["evaluation-manifest-only", "opencc", "icu", "successor"]
    restored = read_normalization_recovery_training_protocol(
        target,
        opencc_source_pack_dir=sources[0],
        icu_source_pack_dir=sources[1],
        successor_source_pack_dir=sources[2],
        evaluation_protocol_dir=sources[3],
    )
    manifest, roster, observations, groups, compositions, loso, work = restored
    assert manifest["manifest_sha256"] == report["manifest_sha256"]
    assert len(roster) == 5
    assert len(work) == len(roster) + len(observations) + len(groups) + len(
        compositions)
    assert len(loso) == len(observations)
    assert manifest["learner_contract"]["authority_contract"] == (
        NORMALIZATION_RECOVERY_AUTHORITY_CONTRACT)
    assert manifest["learner_contract"]["learner_contract"] == (
        NORMALIZATION_RECOVERY_LEARNER_CONTRACT)
    assert len(manifest["learner_contract"]["license_partitions"]) == 4
    assert manifest["evaluation_payload_read_count"] == 0
    assert manifest["reserve_payload_read_count"] == 0
    assert manifest["prior_formal_item_read_count"] == 0
    assert manifest["learner_read_count"] == 0
    assert manifest["teacher_api_llm_call_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_training_protocol(
            run_root=tmp_path,
            opencc_source_pack_dir=sources[0],
            icu_source_pack_dir=sources[1],
            successor_source_pack_dir=sources[2],
            evaluation_protocol_dir=sources[3],
            target_dir=target,
        )


def test_learner_reader_opens_no_source_evaluation_or_loso(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """learner 可在 LOSO 文件不可用时按冻结 manifest 读取 TRAIN。"""
    target, _sources, report, _calls = _publish_protocol(tmp_path, monkeypatch)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("learner 不得读取 source/evaluation")

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
    (target / "train.audit.loso.jsonl").unlink()
    restored = read_normalization_recovery_learner_input(
        target, expected_manifest_sha256=report["manifest_sha256"])
    assert restored[0]["learner_read_count"] == 0


def test_external_manifest_identity_and_auditor_reject_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改物化摘要不能越过外部 manifest，auditor 还会核对 LOSO。"""
    target, sources, report, _calls = _publish_protocol(tmp_path, monkeypatch)
    path = target / "train.opencc.observations.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["expected_output"] += "改"
    lines[0] = canonical_json_line(value)
    path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    file_record = next(
        item for item in manifest["files"]
        if item["relative_path"] == path.name)
    file_record["bytes"] = path.stat().st_size
    file_record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        read_normalization_recovery_learner_input(
            target, expected_manifest_sha256=report["manifest_sha256"])

    second, sources, _report, _calls = _publish_protocol(
        tmp_path / "second", monkeypatch)
    (second / "train.audit.loso.jsonl").unlink()
    with pytest.raises(BroadQaExternalDataError, match="LOSO audit JSONL 不可读"):
        read_normalization_recovery_training_protocol(
            second,
            opencc_source_pack_dir=sources[0],
            icu_source_pack_dir=sources[1],
            successor_source_pack_dir=sources[2],
            evaluation_protocol_dir=sources[3],
        )


def test_k_root_and_source_manifest_identity_fail_closed(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """非 K root 与错误来源 manifest 均不得发布协议。"""
    with pytest.raises(BroadQaExternalDataError, match="必须是 K 盘"):
        protocol_module._require_k_root(tmp_path)
    _target, sources, _report, _calls = _publish_protocol(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement"
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_source_pack",
        lambda path: {
            "artifact_kind": NORMALIZATION_SOURCE_PACK_KIND,
            "manifest_sha256": "f" * 64,
            "package_version": "1.1.9",
        },
    )
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        publish_normalization_recovery_training_protocol(
            run_root=tmp_path,
            opencc_source_pack_dir=sources[0],
            icu_source_pack_dir=sources[1],
            successor_source_pack_dir=sources[2],
            evaluation_protocol_dir=sources[3],
            target_dir=replacement,
        )


def test_official_recovery_training_census_is_frozen() -> None:
    """显式 K 盘 fixture 的五 policy、三 family 与组合库存保持一致。"""
    configured = os.environ.get(
        "PURE_INTEGER_AI_NORMALIZATION_RECOVERY_TRAIN_ROOT")
    if not configured:
        pytest.skip("official recovery training fixture is unavailable")
    root = Path(configured)
    paths = {
        "opencc_source_pack_dir": (
            root / "evaluation" / "normalization-dependency-source-pack-v1"),
        "icu_source_pack_dir": (
            root / "evaluation" / "normalization-icu-source-pack-v1"),
        "successor_source_pack_dir": (
            root / "evaluation" / "normalization-successor-source-pack-v1"),
        "evaluation_protocol_dir": (
            root / "normalization-recovery-evaluation-protocol-v2"),
    }
    if any(not path.is_dir() for path in paths.values()):
        pytest.skip("official recovery training fixture is incomplete")
    roster, observations, groups, compositions, loso, work, summary = (
        protocol_module._derive_from_sources(**paths))
    assert (len(roster), len(observations), len(groups), len(compositions),
            len(loso), len(work)) == (5, 21_853, 11_279, 4_063, 21_853, 37_200)
    assert summary["source_observation_counts"] == {
        "MEDIAWIKI_ZH_TO_CN": 2_152,
        "MEDIAWIKI_ZH_TO_HANS": 4_687,
        "OPENCC_T2S": 4_390,
        "UNICODE_ICU_HANS_HANT": 4_177,
        "UNICODE_UNIHAN_VARIANTS": 6_447,
    }
    assert summary["generic_resolution_counts"] == {
        "CROSS_FAMILY_CONSENSUS": 4_168,
        "INTRA_FAMILY_CONFLICT": 37,
        "NO_GENERIC_AUTHORITY": 2_131,
        "SINGLE_FAMILY_DEFER": 4_856,
        "SOURCE_FAMILY_CONFLICT": 87,
    }
    assert summary["target_resolution_counts"] == {
        "CROSS_FAMILY_CONSENSUS": 4_161,
        "NO_TARGET_AUTHORITY": 4_966,
        "REGIONAL_EXACT_AUTHORITY": 2_152,
    }
    assert summary["composition_qualification_counts"] == {
        "COMPOSITION_SUPPORT": 1_430,
        "EXPLICIT_OVERRIDE": 151,
        "NO_COMPOSITION_EVIDENCE": 1_326,
        "PARTIAL_COMPOSITION": 1_156,
    }
    assert (
        summary["target_rule_character_count"],
        summary["target_rule_phrase_count"],
        summary["target_rule_identity_count"],
    ) == (4_053, 1_777, 483)
