"""NL-00 非字面与文化依赖语言的分层范围裁决测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_nonliteral_scope_probe_catalog import (
    NL00_MANIFEST_PATH,
    build_nonliteral_scope_probe_manifest,
)
from pure_integer_ai.experiments.ph2_nonliteral_scope_probe_contract import (
    EXECUTION_STATE,
    LAYER_KEYS,
    NonliteralScopeProbeContractError,
    NonliteralScopeProbeManifest,
    evaluate_nonliteral_scope_probe,
    read_nonliteral_scope_probe_manifest,
    verify_nonliteral_scope_probe_files,
    write_nonliteral_scope_probe_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST_SHA256 = (
    "30d164a626b767066495bc4eb18956d842bf927728b6b678569c55f42ecd3679")


@pytest.fixture(scope="module")
def probe() -> NonliteralScopeProbeManifest:
    return build_nonliteral_scope_probe_manifest(REPOSITORY)


def _decision(probe: NonliteralScopeProbeManifest, layer_key: str):
    return next(item for item in probe.decisions if item.layer_key == layer_key)


def test_five_layers_have_independent_bounded_verdicts(probe):
    """五层必须分开裁决，不能由词汇化习语 PASS 掩盖高阶缺口。"""
    assert tuple(item.layer_key for item in probe.decisions) == LAYER_KEYS
    assert {item.layer_key: item.verdict for item in probe.decisions} == {
        "CONVENTIONAL_IMPLICATURE": "REJECT",
        "CULTURAL_ALLUSION": "NE",
        "IRONY_HUMOR": "REJECT",
        "LEXICALIZED_IDIOM": "PASS",
        "PRODUCTIVE_METAPHOR_METONYMY": "REJECT",
    }
    assert (probe.pass_count, probe.reject_count, probe.ne_count) == (1, 3, 1)
    assert probe.scope_claim_only == 1
    assert probe.capability_learned_claims == 0
    assert probe.runtime_pass_authority == 0
    assert probe.unresolved_decision_keys == ("DISC-08", "DISC-12")
    assert probe.execution_state.to_value() == EXECUTION_STATE
    assert all(value == 0 for value in EXECUTION_STATE.values())


def test_every_layer_lists_representation_source_counterexample_and_wall(probe):
    """每层都必须显式给表示、候选来源、反例、evaluator、接地和墙结论。"""
    for item in probe.decisions:
        assert item.representation_contracts
        assert item.candidate_source_contracts
        assert item.counterexample_contracts
        assert item.invariant_results.to_value()
        assert item.evaluator_state
        assert item.grounding_state
        assert item.scope_decision
        assert item.wall_decision
        assert item.evidence_refs
        assert item.host_learning_writes == 0
    assert {item.relative_path for item in probe.evidence_files} == {
        path for item in probe.decisions for path in item.evidence_refs}


def test_lexicalized_idiom_pass_is_structural_representation_only(probe):
    """习语 PASS 只证明 LC-03 表示可行，不证明 runtime 已学会解释或生成。"""
    item = _decision(probe, "LEXICALIZED_IDIOM")
    assert item.verdict == "PASS"
    assert item.typed_layer_available == 1
    assert item.representation_state == "AVAILABLE_NOT_EXECUTED"
    assert item.evaluator_state == "STRUCTURAL_ONLY"
    assert item.grounding_state == "LANGUAGE_INTERNAL"
    assert set(item.invariant_results.to_value().values()) == {"PASS"}
    assert item.ne_conditions == ()
    assert {
        "CONSTRUCTION_IDENTITY_V1",
        "LEXICALIZATION_STATE_V1",
        "REGISTER_SCOPED_SURFACE_V1",
    } == set(item.representation_contracts)
    assert {
        "LITERAL_TOKEN_SUM_BASELINE",
        "SAME_SURFACE_DIFFERENT_CONSTRUCTION",
        "WHOLE_VS_PARTIAL_LEXICALIZATION",
    } == set(item.counterexample_contracts)
    assert "data/ph2/manifests/lc03_construction_course_v1.json" in (
        item.evidence_refs)


def test_higher_layers_retain_runtime_evaluator_and_grounding_gaps(probe):
    """其余四层不得从通用对象复用升级成深层非字面能力 PASS。"""
    for layer_key in set(LAYER_KEYS) - {"LEXICALIZED_IDIOM"}:
        item = _decision(probe, layer_key)
        assert item.typed_layer_available == 0
        assert item.evaluator_state == "INDEPENDENT_EVALUATOR_ABSENT"
        assert item.ne_conditions
        assert item.verdict in {"REJECT", "NE"}
    culture = _decision(probe, "CULTURAL_ALLUSION")
    assert culture.verdict == "NE"
    assert culture.grounding_state == "EXTERNAL_GROUNDING_NE"
    assert culture.invariant_results.to_value()[
        "CULTURAL_GROUNDING_AUTHORIZED"] == "NE"
    assert "W1_EXTERNAL_GROUNDING_NOT_AVAILABLE" in culture.ne_conditions
    irony = _decision(probe, "IRONY_HUMOR")
    assert irony.invariant_results.to_value()["MIND_READING_FORBIDDEN"] == (
        "PASS")
    assert irony.invariant_results.to_value()[
        "STANCE_EXPECTATION_CONTRAST"] == "REJECT"


def test_worst_invariant_and_host_write_fail_closed():
    """单维 REJECT/NE 和任一宿主写都不能被其他 PASS 平均掉。"""
    reject = CanonicalJsonObject.from_value({
        "A": "PASS", "B": "REJECT", "C": "PASS"})
    unknown = CanonicalJsonObject.from_value({
        "A": "PASS", "B": "NE", "C": "PASS"})
    passed = CanonicalJsonObject.from_value({
        "A": "PASS", "B": "PASS", "C": "PASS"})
    assert evaluate_nonliteral_scope_probe(
        reject, host_learning_writes=0) == "REJECT"
    assert evaluate_nonliteral_scope_probe(
        unknown, host_learning_writes=0) == "NE"
    assert evaluate_nonliteral_scope_probe(
        passed, host_learning_writes=0) == "PASS"
    assert evaluate_nonliteral_scope_probe(
        passed, host_learning_writes=1) == "REJECT"


def test_layer_contract_rejects_missing_counterexample_and_scope_spoof(probe):
    """漏反例、伪文化接地、伪心理读取和非零写必须在构造期失败。"""
    lexical = _decision(probe, "LEXICALIZED_IDIOM")
    with pytest.raises(NonliteralScopeProbeContractError, match="不能为空"):
        replace(lexical, counterexample_contracts=())
    culture = _decision(probe, "CULTURAL_ALLUSION")
    with pytest.raises(NonliteralScopeProbeContractError, match="文化典故"):
        replace(culture, grounding_state="LANGUAGE_INTERNAL")
    irony = _decision(probe, "IRONY_HUMOR")
    with pytest.raises(NonliteralScopeProbeContractError, match="心理真值"):
        replace(
            irony,
            invariant_results=CanonicalJsonObject.from_value({
                "LITERAL_CONTENT_PRESERVED": "PASS",
                "MIND_READING_FORBIDDEN": "REJECT",
                "STANCE_EXPECTATION_CONTRAST": "REJECT",
            }),
        )
    with pytest.raises(NonliteralScopeProbeContractError, match="学习写"):
        replace(irony, host_learning_writes=1)


def test_manifest_rejects_aggregate_scope_and_learned_claims(probe):
    """范围 artifact 不能漏层、删待裁决题或签发 learned/runtime PASS。"""
    with pytest.raises(NonliteralScopeProbeContractError, match="五类"):
        replace(probe, decisions=probe.decisions[:-1])
    with pytest.raises(NonliteralScopeProbeContractError, match="待裁决"):
        replace(probe, unresolved_decision_keys=("DISC-08",))
    with pytest.raises(NonliteralScopeProbeContractError, match="越权"):
        replace(probe, capability_learned_claims=1)
    with pytest.raises(NonliteralScopeProbeContractError, match="越权"):
        replace(probe, runtime_pass_authority=1)


def test_round_trip_nonoverwrite_and_strict_fields(tmp_path, probe):
    """NL-00 manifest 可恢复、幂等且拒绝覆盖和额外字段。"""
    output = tmp_path / "nl00.json"
    write_nonliteral_scope_probe_manifest(probe, output)
    assert read_nonliteral_scope_probe_manifest(output) == probe
    write_nonliteral_scope_probe_manifest(probe, output)
    output.write_bytes(b"{}\n")
    with pytest.raises(NonliteralScopeProbeContractError, match="内容不同"):
        write_nonliteral_scope_probe_manifest(probe, output)
    value = probe.to_dict()
    value["learned"] = 1
    with pytest.raises(NonliteralScopeProbeContractError, match="字段不精确"):
        NonliteralScopeProbeManifest.from_dict(value)


def test_file_verifier_detects_baseline_or_evidence_drift(tmp_path, probe):
    """文件级 verifier 必须回验 baseline 与每个承重 facility。"""
    paths = (
        (probe.baseline_manifest_relative_path, probe.baseline_manifest_sha256),
        *((item.relative_path, item.sha256) for item in probe.evidence_files),
    )
    for relative_path, expected_sha256 in paths:
        source = REPOSITORY / Path(*relative_path.split("/"))
        target = tmp_path / Path(*relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        target.write_bytes(payload)
    verify_nonliteral_scope_probe_files(probe, repository_root=tmp_path)
    first = tmp_path / Path(*probe.evidence_files[0].relative_path.split("/"))
    first.write_bytes(b"damaged\n")
    with pytest.raises(NonliteralScopeProbeContractError, match="evidence"):
        verify_nonliteral_scope_probe_files(probe, repository_root=tmp_path)


def test_formal_nl00_manifest_remains_frozen_when_current_files_evolve():
    """历史 artifact 固定 hash；当前文件漂移必须保留 fail-closed 边界。"""
    stored = read_nonliteral_scope_probe_manifest(REPOSITORY / NL00_MANIFEST_PATH)
    rebuilt = build_nonliteral_scope_probe_manifest(REPOSITORY)
    assert stored.sha256() == FORMAL_MANIFEST_SHA256
    assert hashlib.sha256(
        (REPOSITORY / NL00_MANIFEST_PATH).read_bytes()).hexdigest() == (
            FORMAL_MANIFEST_SHA256)
    assert stored != rebuilt
    with pytest.raises(NonliteralScopeProbeContractError, match="evidence"):
        verify_nonliteral_scope_probe_files(stored, repository_root=REPOSITORY)
