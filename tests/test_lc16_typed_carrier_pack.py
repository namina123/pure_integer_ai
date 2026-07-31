"""LC-16 typed carrier pack 的覆盖、预算和零执行合同测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_catalog import (
    TYPED_CARRIER_PACK_MANIFEST_PATH,
    build_typed_carrier_pack_manifest,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    IN_SCOPE_CARRIER_KEYS,
    PACK_EXECUTION_STATE,
    SAMPLE_KINDS,
    SAMPLE_SPLITS,
    TypedCarrierBudget,
    TypedCarrierPackError,
    TypedCarrierPackManifest,
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
    write_typed_carrier_pack_manifest,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pack():
    return build_typed_carrier_pack_manifest(_ROOT)


def test_pack_freezes_nine_carriers_and_sixty_three_unmaterialized_cases(pack):
    assert pack.artifact_version == ARTIFACT_VERSION
    assert pack.artifact_status == ARTIFACT_STATUS
    assert tuple(item.carrier_key for item in pack.budgets) == (
        IN_SCOPE_CARRIER_KEYS)
    assert len(pack.cases) == 63
    for carrier in IN_SCOPE_CARRIER_KEYS:
        cases = tuple(item for item in pack.cases
                      if item.carrier_key == carrier)
        assert tuple(item.sample_kind for item in cases) == SAMPLE_KINDS
        assert tuple(item.split for item in cases) == tuple(
            SAMPLE_SPLITS[item] for item in SAMPLE_KINDS)
        assert all(item.payload_state == "UNMATERIALIZED" for item in cases)
        assert all(item.runtime_state == "NOT_RUN" for item in cases)
        assert all(item.directions == (
            "GENERATION", "REASONING", "UNDERSTANDING") for item in cases)
    assert pack.execution_state.to_value() == PACK_EXECUTION_STATE


def test_pack_round_trips_and_rechecks_all_evidence(pack, tmp_path):
    target = tmp_path / "typed_carrier_pack.json"
    assert write_typed_carrier_pack_manifest(pack, target) == target
    assert read_typed_carrier_pack_manifest(target) == pack
    verify_typed_carrier_pack_files(pack, repository_root=_ROOT)
    assert write_typed_carrier_pack_manifest(pack, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(TypedCarrierPackError, match="已存在且内容不同"):
        write_typed_carrier_pack_manifest(pack, target)


def test_stored_pack_is_current_and_canonical(pack):
    stored = read_typed_carrier_pack_manifest(
        _ROOT / Path(*TYPED_CARRIER_PACK_MANIFEST_PATH.split("/")))
    rebuilt = build_typed_carrier_pack_manifest(_ROOT)
    assert stored == pack == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_typed_carrier_pack_files(stored, repository_root=_ROOT)


def test_pack_rejects_execution_claims(pack):
    state = pack.execution_state.to_value()
    state["teacher_calls"] = 1
    with pytest.raises(TypedCarrierPackError, match="execution_state"):
        replace(
            pack,
            execution_state=CanonicalJsonObject.from_value(state),
        )


def test_pack_rejects_missing_case_and_cross_split_cluster_overlap(pack):
    with pytest.raises(TypedCarrierPackError, match="完整覆盖七类样本"):
        replace(pack, cases=pack.cases[:-1])

    train = next(item for item in pack.cases if item.split == "train")
    held_out = next(item for item in pack.cases if item.split == "held_out")
    overlapping = replace(
        held_out,
        content_cluster=train.content_cluster,
    )
    cases = tuple(
        sorted(
            (overlapping if item.case_key == held_out.case_key else item
             for item in pack.cases),
            key=lambda item: item.case_key,
        )
    )
    with pytest.raises(TypedCarrierPackError, match="content_cluster 跨 split"):
        replace(pack, cases=cases)


def test_pack_rejects_wall_carrier_and_budget_identity_drift(pack):
    with pytest.raises(TypedCarrierPackError, match="未登记"):
        TypedCarrierBudget(
            "SENSORY_GROUNDING", 7, 4096, 512, 1024, 64, 8)

    budgets = (pack.budgets[1], pack.budgets[0], *pack.budgets[2:])
    with pytest.raises(TypedCarrierPackError, match="精确列出九类 carrier"):
        replace(pack, budgets=budgets)


def test_pack_rejects_extra_manifest_fields_and_duplicate_case_key(pack):
    value = pack.to_dict()
    value["unexpected"] = 1
    with pytest.raises(TypedCarrierPackError, match="字段不精确"):
        TypedCarrierPackManifest.from_dict(value)

    case = pack.cases[0]
    duplicate = replace(case, case_key=pack.cases[1].case_key)
    with pytest.raises(TypedCarrierPackError, match="case_key"):
        replace(pack, cases=(duplicate, *pack.cases[1:]))
