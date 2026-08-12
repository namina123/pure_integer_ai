"""FT26 真实公开来源词义 artifact、查询与数据依赖专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_compiler import (
    _apply_supersedes,
    build_w03_public_sense_artifact,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
    W03PublicSenseContractError,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    PUBLIC_W03_SENSE_ARTIFACT,
    PUBLIC_W03_SENSE_ARTIFACT_SHA256,
    W03PublicSenseRuntime,
    W03PublicSenseRuntimeError,
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_source_catalog import (
    FT26_PUBLIC_SENSE_SELECTION_MANIFEST,
    FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT,
    FT26_WIKIDATA_PACK_NAME,
    FT26_WIKTIONARY_PACK_NAME,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PACK_ROOT = REPOSITORY / FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT / "packs"
WIKTIONARY_PACK = PACK_ROOT / FT26_WIKTIONARY_PACK_NAME
WIKIDATA_PACK = PACK_ROOT / FT26_WIKIDATA_PACK_NAME


@pytest.fixture(scope="module")
def runtime() -> W03PublicSenseRuntime:
    return load_w03_public_sense_artifact()


def _query(
        runtime: W03PublicSenseRuntime,
        surface: str,
        context: str | None = None,
        ):
    return query_w03_public_sense(
        runtime, W03PublicSenseQuery(surface, context))


def test_bundled_artifact_is_compact_source_bound_and_formally_zero(
        runtime) -> None:
    """运行时只携带 43 个 entry projection，不含 raw 或 formal 声明。"""
    assert PUBLIC_W03_SENSE_ARTIFACT.stat().st_size == 47944
    assert runtime.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_SHA256 == (
        "7e0e1ae1b4c7bb334d9581c887f880949c5a43c64ca68aad4a9e05a6206e3792")
    assert len(runtime.artifact.source_packs) == 2
    assert len(runtime.artifact.source_revisions) == 1
    assert len(runtime.artifact.entries) == 43
    assert len(runtime.artifact.aliases) == 1
    assert {item.source_key for item in runtime.artifact.source_packs} == {
        "WIKIDATA_REVISION_V1", "ZHWIKTIONARY_20260701"}
    assert {item.source_ref.license_id for item in runtime.artifact.entries} == {
        "CC0-1.0", "CC-BY-SA-4.0"}
    raw = json.loads(PUBLIC_W03_SENSE_ARTIFACT.read_text(encoding="utf-8"))
    assert (
        raw["experimental"], raw["formal_mastery_claim"],
        raw["w02_runtime_evidenced"], raw["w03_started"],
        raw["mastery"], raw["readiness"],
    ) == (1, 0, 0, 0, 0, 0)
    assert "raw_observation" not in raw["payload"]


def test_raw_terms_cover_alias_ambiguity_conflict_unknown_and_context(
        runtime) -> None:
    """同一通用查询逻辑覆盖 FT26 五类公开状态。"""
    alias = _query(runtime, "首页")
    traditional = _query(runtime, "首頁")
    conflict = _query(runtime, "金星")
    unique_alias = _query(runtime, "鸟类")
    unknown = _query(runtime, "不存在词项")
    contextual = _query(runtime, "金星", "距离太阳第二近的行星")

    assert alias.status == traditional.status == "AMBIGUOUS"
    assert alias.alias_path == ("首页", "首頁")
    assert len(alias.candidates) == len(traditional.candidates) == 3
    assert {item.entry.relation_kind for item in alias.candidates} == {
        "DEFINITION"}

    assert conflict.status == "CONFLICT"
    assert conflict.conflict_kind == "UNRESOLVED_SOURCE_PARTITION"
    assert {item.entry.source_ref.source_key for item in conflict.candidates} == {
        "WIKIDATA_REVISION_V1", "ZHWIKTIONARY_20260701"}
    assert {item.entry.relation_kind for item in conflict.candidates} == {
        "DEFINITION", "LABEL"}

    assert unique_alias.status == "UNIQUE"
    assert len(unique_alias.candidates) == 1
    assert unique_alias.candidates[0].entry.relation_kind == "ALIAS"
    assert unknown.status == "UNKNOWN" and unknown.candidates == ()
    assert contextual.status == "UNIQUE"
    assert len(contextual.candidates) == 1
    assert contextual.candidates[0].entry.definition_text == (
        "距离太阳第二近的行星")

    for result in (alias, traditional, conflict, unique_alias, unknown,
                   contextual):
        assert len(result.trace_commitment_sha256) == 64
        assert result.experimental == 1
        assert result.formal_mastery_claim == result.w03_started == 0
        assert result.sha256() == _query(
            runtime, result.query.surface,
            result.query.context_text).sha256()


def test_source_packs_are_manifest_driven_bounded_and_double_hash_bound() -> None:
    """五页与两实体 pack 均绑定 selection/snapshot/hash，不含全量 raw。"""
    selection_path = REPOSITORY / FT26_PUBLIC_SENSE_SELECTION_MANIFEST
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert [item["selection_kind"] for item in selection["source_slices"]] == [
        "QID", "TITLE"]
    assert [len(item["selections"]) for item in selection["source_slices"]] == [
        2, 5]

    wiktionary = read_source_pack(WIKTIONARY_PACK)
    wikidata = read_source_pack(WIKIDATA_PACK)
    assert (len(wiktionary.sources), len(wiktionary.observations)) == (5, 5)
    assert (len(wikidata.sources), len(wikidata.observations)) == (2, 2)
    assert all(item.split == "train" for item in (
        *wiktionary.observations, *wikidata.observations))
    assert all(item.redistribution_policy == "PUBLIC" for item in (
        *wiktionary.sources, *wikidata.sources))
    for source in (*wiktionary.sources, *wikidata.sources):
        span = source.source_span.to_value()
        assert span["selection_manifest_sha256"] == selection_sha
        assert span["selection_manifest_relative_path"] == (
            FT26_PUBLIC_SENSE_SELECTION_MANIFEST.as_posix())
        assert source.upstream_checksum.startswith(("sha1:", "sha256:"))
        assert len(source.local_sha256) == 64
    assert all(
        "compressed_block_offset" in item.source_span.to_value()
        for item in wiktionary.sources)


def test_compile_is_order_independent_and_pack_removal_changes_capability(
        runtime) -> None:
    """同代码下 pack 加入才出现候选，移除后结果失效。"""
    inputs = (
        (WIKTIONARY_PACK.relative_to(REPOSITORY).as_posix(), WIKTIONARY_PACK),
        (WIKIDATA_PACK.relative_to(REPOSITORY).as_posix(), WIKIDATA_PACK),
    )
    rebuilt = build_w03_public_sense_artifact(REPOSITORY, inputs)
    reversed_build = build_w03_public_sense_artifact(
        REPOSITORY, tuple(reversed(inputs)))
    wiktionary_only = build_w03_public_sense_artifact(
        REPOSITORY, inputs[:1])

    assert rebuilt.payload_value() == runtime.artifact.payload_value()
    assert reversed_build.payload_value() == rebuilt.payload_value()
    assert wiktionary_only.payload_sha256() != rebuilt.payload_sha256()
    reduced = W03PublicSenseRuntime(wiktionary_only, "1" * 64)
    assert _query(runtime, "鸟类").status == "UNIQUE"
    assert _query(reduced, "鸟类").status == "UNKNOWN"
    assert _query(reduced, "首页").sha256() != _query(
        runtime, "首页").sha256()


def test_superseding_revision_hides_old_candidate() -> None:
    """active successor 必须关闭旧候选，查询只返回新 revision。"""
    base = load_w03_public_sense_artifact().artifact
    original = next(
        item for item in base.entries
        if item.surface == "鸟类" and item.language == "zh")
    old = replace(original, active=0)
    successor_source = replace(
        original.source_ref,
        revision_id=original.source_ref.revision_id + "-successor",
        source_identity=original.source_ref.source_identity + "#successor",
        source_commitment_sha256="f" * 64,
    )
    successor = replace(
        original,
        entry_key=(9, 1),
        observation_key=(9, 2),
        definition_text="新的公开修订定义",
        source_ref=successor_source,
        supersedes_entry_keys=(original.entry_key,),
    )
    artifact = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        tuple(sorted((old, successor), key=lambda item: item.entry_key)),
        (),
    )
    runtime = W03PublicSenseRuntime(artifact, "2" * 64)
    result = _query(runtime, "鸟类")
    assert result.status == "UNIQUE"
    assert len(result.candidates) == 1
    assert result.candidates[0].entry.entry_key == successor.entry_key
    assert result.candidates[0].entry.definition_text == "新的公开修订定义"


def test_supersede_chain_is_flattened_to_the_active_successor() -> None:
    """多级 revision 的全部旧候选都必须由最终 active entry 承接。"""
    bundle = read_source_pack(WIKTIONARY_PACK)
    keys = tuple(item.stable_key for item in bundle.observations[:3])
    original = load_w03_public_sense_artifact().artifact.entries[0]
    entries = tuple(
        replace(
            original,
            entry_key=(8, ordinal),
            observation_key=key.stable_key(),
        )
        for ordinal, key in enumerate(keys, start=1)
    )
    observations = (
        SimpleNamespace(stable_key=keys[0], supersedes_key=None),
        SimpleNamespace(stable_key=keys[1], supersedes_key=keys[0]),
        SimpleNamespace(stable_key=keys[2], supersedes_key=keys[1]),
    )

    updated = _apply_supersedes(entries, observations)
    by_key = {item.entry_key: item for item in updated}
    assert by_key[(8, 1)].active == by_key[(8, 2)].active == 0
    assert by_key[(8, 1)].supersedes_entry_keys == ()
    assert by_key[(8, 2)].supersedes_entry_keys == ()
    assert by_key[(8, 3)].active == 1
    assert by_key[(8, 3)].supersedes_entry_keys == ((8, 1), (8, 2))


def test_artifact_tamper_and_incomplete_distribution_fail_closed(
        tmp_path, monkeypatch) -> None:
    """payload/source 漂移与缺文件均不得静默 fallback。"""
    from pure_integer_ai.experiments import ph2_w03_public_sense_runtime as module

    value = json.loads(PUBLIC_W03_SENSE_ARTIFACT.read_text(encoding="utf-8"))
    value["payload"]["entries"][0]["source_ref"]["source_key"] = "DRIFT"
    payload = canonical_json_bytes(value) + b"\n"
    target = tmp_path / "tampered.json"
    target.write_bytes(payload)
    monkeypatch.setattr(
        module, "PUBLIC_W03_SENSE_ARTIFACT_SHA256",
        hashlib.sha256(payload).hexdigest())
    with pytest.raises(W03PublicSenseRuntimeError, match="payload commitment"):
        load_w03_public_sense_artifact(target)
    with pytest.raises(W03PublicSenseRuntimeError, match="缺失或不可读"):
        load_w03_public_sense_artifact(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("surface", 313), ("active", "1"), ("active", True)),
)
def test_artifact_deserialization_rejects_implicit_type_coercion(
        field, invalid) -> None:
    """外部 artifact 不得借 str/int 转换绕过冻结字段类型。"""
    value = json.loads(PUBLIC_W03_SENSE_ARTIFACT.read_text(encoding="utf-8"))
    value["payload"]["entries"][0][field] = invalid
    with pytest.raises(W03PublicSenseContractError):
        W03PublicSenseArtifact.from_payload_value(value["payload"])


def test_distribution_contains_only_compact_artifact_not_source_raw() -> None:
    """wheel data-files 增量只有 compact artifact，不发布 source pack/raw dump。"""
    pyproject = tomllib.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"]
        ["share/pure_integer_ai/data/ph2"])
    assert configured[-2:] == (
        "data/ph2/w03_public_sense_runtime_v1.json",
        "data/ph2/w03_public_sense_runtime_v2.json",
    )
    assert not any(
        "ph2_ft26_dataset_artifacts" in item
        or "ph2_ft30_dataset_artifacts" in item
        or "multistream.xml" in item
        or "pinned_v2" in item
        or "census" in item
        for item in configured)


def test_query_and_compiler_have_no_selected_term_or_qid_dispatch() -> None:
    """source selection 只在 manifest，compiler/runtime 不含词项或 QID 分支。"""
    files = (
        "src/pure_integer_ai/experiments/ph2_w03_public_sense_compiler.py",
        "src/pure_integer_ai/experiments/ph2_w03_public_sense_runtime.py",
        "src/pure_integer_ai/experiments/ph2_w03_public_sense_source_catalog.py",
    )
    combined = "\n".join(
        (REPOSITORY / item).read_text(encoding="utf-8") for item in files)
    for forbidden in (
            "首页", "首頁", "苹果", "蘋果", "金星",
            '"Q313"', '"Q5113"'):
        assert forbidden not in combined
