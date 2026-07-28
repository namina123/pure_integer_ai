"""LG-00/J-LG-D03 公开 identity 与 secret 扫描口径 T0。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    inventory_public_files,
)
from pure_integer_ai.experiments.ph2_public_gate_rules import (
    LEGACY_RULES,
    SECRET_RULES,
    build_baseline_public_gate,
)


def _inventory(tmp_path: Path, text: str):
    root = tmp_path / "repo"
    source = root / "src" / "candidate.py"
    source.parent.mkdir(parents=True)
    source.write_text(text, encoding="utf-8")
    return root, inventory_public_files(root, ("src/candidate.py",))


def test_rule_keys_are_exact_sorted_and_stable():
    """最终扫描只使用冻结的两类 identity 和七类 secret 规则。"""
    assert tuple(key for key, _ in LEGACY_RULES) == (
        "LEGACY_REGISTERED_NAME_V1",
        "LEGACY_URN_NAMESPACE_V1",
    )
    assert tuple(key for key, _ in SECRET_RULES) == (
        "AWS_ACCESS_KEY_V1",
        "BEARER_TOKEN_V1",
        "GENERIC_API_KEY_ASSIGNMENT_V1",
        "GITHUB_TOKEN_V1",
        "GOOGLE_API_KEY_V1",
        "LLM_API_KEY_V1",
        "PRIVATE_KEY_HEADER_V1",
    )


def test_clean_text_produces_clear_baseline_gate(tmp_path):
    """无命中、无二进制、无不可读文件时两类状态都 CLEAR。"""
    root, inventory = _inventory(tmp_path, "value = 'public'\n")
    gate = build_baseline_public_gate(root, inventory)
    assert gate.legacy_status == "CLEAR"
    assert gate.secret_status == "CLEAR"
    assert gate.legacy_findings == ()
    assert gate.secret_findings == ()
    assert gate.final_rescan_required == 1
    assert gate.public_release_allowed == 0


def test_legacy_name_and_urn_are_both_detected_without_copying_text(tmp_path):
    """旧名称与旧 namespace 分账，artifact 只保存规则键和行 hash。"""
    old_name = "Zero" + " AI"
    old_urn = "urn:" + "zero" + "-ai:ph2:test"
    root, inventory = _inventory(tmp_path, old_name + "\n" + old_urn + "\n")
    gate = build_baseline_public_gate(root, inventory)
    assert gate.legacy_status == "BLOCKED"
    assert tuple(item.rule_key for item in gate.legacy_findings) == (
        "LEGACY_REGISTERED_NAME_V1",
        "LEGACY_URN_NAMESPACE_V1",
    )
    assert all(len(item.line_sha256) == 64 for item in gate.legacy_findings)


def test_each_secret_family_blocks_independently(tmp_path):
    """七类真实形态都必须被命中，不能只扫某一家 LLM key。"""
    values = (
        "AK" + "IA" + "A" * 16,
        "Bearer " + "b" * 24,
        "api_key=" + "\"" + "c" * 24 + "\"",
        "gh" + "p_" + "d" * 36,
        "AI" + "za" + "e" * 35,
        "s" + "k-proj-" + "f" * 24,
        "-----BEGIN " + "PRIVATE KEY-----",
    )
    root, inventory = _inventory(tmp_path, "\n".join(values) + "\n")
    gate = build_baseline_public_gate(root, inventory)
    assert gate.secret_status == "BLOCKED"
    assert tuple(item.rule_key for item in gate.secret_findings) == tuple(
        key for key, _ in SECRET_RULES)
