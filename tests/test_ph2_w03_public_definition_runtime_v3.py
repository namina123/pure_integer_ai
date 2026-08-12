"""FT31 public-definition v3 compiler, census, runtime, and opt-in CLI."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import tomllib

from pure_integer_ai.experiments.ph2_w03_public_definition_compiler_v3 import (
    build_ft31_public_definition_artifact_v3,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    PUBLIC_W03_SENSE_ARTIFACT_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V3,
    PUBLIC_W03_SENSE_ARTIFACT_V3_SHA256,
    W03PublicSenseRuntime,
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.run_ph2_w03_public_sense import main


REPOSITORY = Path(__file__).resolve().parents[1]
SELECTION = REPOSITORY / (
    "data/ph2/manifests/ft31_w03_public_definition_selection_v3.json")
CENSUS = REPOSITORY / (
    "data/ph2/manifests/ft31_w03_public_definition_census_v3.json")
BASE_CENSUS = REPOSITORY / (
    "data/ph2/manifests/ft30_w03_public_definition_census_v2.json")
PACK_RELATIVE = Path(
    "ph2_ft31_dataset_artifacts/public_definition_source_v3/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v3")
PACK = REPOSITORY / PACK_RELATIVE
SELECTION_SHA256 = (
    "16bc5f547b24d863e5f00995c34eb0e81238dccef639b6294e89ffac35e94599")
CENSUS_SHA256 = (
    "4f637b372bc69c6c10e63ddba087eee1bfab108ed6381802323e39319242bdc4")
BASE_SHA256 = (
    "db3200d42004cc1fdcf03cbd5872e37a7437872b688fae79c322eb0eff70946a")
BASE_CENSUS_SHA256 = (
    "4a82626cc0eaeea74dec8a8b2626447ddf1f9fb845cd41118e3e96b67afe604b")
V3_SURFACE = "敗仗"


def _build():
    """从冻结 v2 与 v3 pack 重建同一 artifact/census。"""
    return build_ft31_public_definition_artifact_v3(
        base_artifact_path=(
            REPOSITORY / "data/ph2/w03_public_sense_runtime_v2.json"),
        base_artifact_sha256=BASE_SHA256,
        base_census_path=BASE_CENSUS,
        base_census_sha256=BASE_CENSUS_SHA256,
        expansion_pack_relative_path=PACK_RELATIVE.as_posix(),
        expansion_pack_root=PACK,
        selection_manifest_sha256=SELECTION_SHA256,
    )


def _query(runtime: W03PublicSenseRuntime, surface: str):
    """执行一个无上下文中文 sense 查询。"""
    return query_w03_public_sense(runtime, W03PublicSenseQuery(surface))


def test_v3_compile_census_and_query_are_bit_identical() -> None:
    """重复编译、census 与 query 均保持规范身份。"""
    first = _build()
    second = _build()
    runtime = load_w03_public_sense_artifact(artifact_version="v3")
    assert first.artifact.payload_value() == second.artifact.payload_value()
    assert first.census_value == second.census_value
    assert first.artifact.payload_value() == runtime.artifact.payload_value()
    assert len(runtime.artifact.source_packs) == 4
    assert len(runtime.artifact.source_revisions) == 1
    assert len(runtime.artifact.entries) == 117
    assert len(runtime.artifact.aliases) == 10
    assert runtime.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_V3_SHA256 == (
        "304cc1e6674df856e5c75eefab663e15aab85045f08b603d6650e9ad87a4ff1d")
    assert _query(runtime, V3_SURFACE).status == "UNIQUE"


def test_v3_census_freezes_v2_and_all_selected_pages() -> None:
    """v2 baseline、256 页、失败样例与 renderer 零授权同时冻结。"""
    raw = CENSUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    census = json.loads(raw.decode("utf-8"))
    assert census["base_artifact_sha256"] == BASE_SHA256
    assert census["base_census_sha256"] == BASE_CENSUS_SHA256
    assert census["base_definition_count"] == 21
    assert census["base_definition_rendering_counts"] == {
        "DISPLAY": 16,
        "NO_SOURCE_ANSWER": 2,
        "UNKNOWN_TEMPLATE": 3,
    }
    assert census["base_v2_census_identity"] == {
        "definition_count": 18,
        "eligible_definition_count": 9,
        "page_count": 32,
        "page_status_counts": {
            "ACCEPTED_DEFINITION": 4,
            "NON_CHINESE_DEFINITION": 2,
            "NO_DEFINITION": 23,
            "REDIRECT": 3,
        },
        "render_status_counts": {
            "DISPLAY": 14,
            "UNSUPPORTED_MARKUP": 4,
        },
    }
    assert census["page_count"] == 256
    assert census["definition_count"] == 125
    assert census["eligible_definition_count"] == 65
    assert census["page_status_counts"] == {
        "ACCEPTED_DEFINITION": 41,
        "NON_CHINESE_DEFINITION": 12,
        "NO_DEFINITION": 197,
        "REDIRECT": 6,
    }
    assert census["render_status_counts"] == {
        "DISPLAY": 67,
        "MALFORMED_MARKUP": 1,
        "UNSUPPORTED_MARKUP": 57,
    }
    gate = census["template_evidence_gate"]
    assert gate["renderer_authorized_count"] == 0
    qualified = {
        item["template_name"] for item in gate["templates"]
        if item["frequency_gate_met"] == 1}
    assert qualified == {"place", "zh-div"}
    assert all(item["public_specification_reviewed"] == 0
               and item["renderer_authorized"] == 0
               for item in gate["templates"])
    assert all(
        item["source_ref"]["license_id"] == "CC-BY-SA-4.0"
        and "page_title=" in item["source_ref"]["attribution"]
        and "revision_timestamp=" in item["source_ref"]["attribution"]
        and "contributor=" in item["source_ref"]["attribution"]
        for item in census["definitions"])
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == (
        SELECTION_SHA256)


def test_v3_source_removal_revision_and_supersede_change_capability() -> None:
    """新增能力保持对来源、修订与 active 生命周期的因果依赖。"""
    runtime = load_w03_public_sense_artifact(artifact_version="v3")
    base = runtime.artifact
    original = next(item for item in base.entries
                    if item.surface == V3_SURFACE)
    expansion_pack = next(
        item for item in base.source_packs
        if item.relative_path == PACK_RELATIVE.as_posix())
    reduced = W03PublicSenseArtifact(
        tuple(item for item in base.source_packs if item != expansion_pack),
        base.source_revisions,
        tuple(item for item in base.entries
              if item.source_ref.stable_key != original.source_ref.stable_key),
        tuple(item for item in base.aliases
              if item.source_ref.stable_key != original.source_ref.stable_key),
    )
    assert _query(
        W03PublicSenseRuntime(reduced, "1" * 64), V3_SURFACE).status == (
            "UNKNOWN")

    replacement_source = replace(
        original.source_ref,
        revision_id=original.source_ref.revision_id + "-replacement",
        source_identity=original.source_ref.source_identity + "#replacement",
        source_commitment_sha256="f" * 64,
    )
    replacement = replace(original, source_ref=replacement_source)
    replaced = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        tuple(replacement if item.entry_key == original.entry_key else item
              for item in base.entries),
        base.aliases,
    )
    changed = _query(
        W03PublicSenseRuntime(replaced, "2" * 64), V3_SURFACE)
    assert changed.status == "UNIQUE"
    assert changed.sha256() != _query(runtime, V3_SURFACE).sha256()

    old = replace(original, active=0)
    successor = replace(
        original,
        entry_key=(31, 1),
        observation_key=(31, 2),
        definition_text="公开修订后的来源定义",
        source_ref=replacement_source,
        supersedes_entry_keys=(original.entry_key,),
    )
    superseded = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        tuple(sorted((
            *(item for item in base.entries
              if item.entry_key != original.entry_key),
            old,
            successor,
        ), key=lambda item: item.entry_key)),
        base.aliases,
    )
    result = _query(
        W03PublicSenseRuntime(superseded, "3" * 64), V3_SURFACE)
    assert result.status == "UNIQUE"
    assert result.candidates[0].entry.entry_key == successor.entry_key


def test_v3_is_explicit_and_v1_v2_remain_frozen() -> None:
    """默认继续为 v1，显式 v2 bytes/行为不因 v3 漂移。"""
    default = load_w03_public_sense_artifact()
    v2 = load_w03_public_sense_artifact(artifact_version="v2")
    v3 = load_w03_public_sense_artifact(artifact_version="v3")
    assert default.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_SHA256
    assert v2.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256
    assert _query(default, V3_SURFACE).status == "UNKNOWN"
    assert _query(v2, V3_SURFACE).status == "UNKNOWN"
    assert _query(v3, V3_SURFACE).status == "UNIQUE"

    default_output = StringIO()
    v2_output = StringIO()
    v3_output = StringIO()
    assert main([V3_SURFACE], stdout=default_output) == 0
    assert main([
        V3_SURFACE, "--artifact-version", "v2",
    ], stdout=v2_output) == 0
    assert main([
        V3_SURFACE, "--artifact-version", "v3",
    ], stdout=v3_output) == 0
    assert json.loads(default_output.getvalue())["status"] == "UNKNOWN"
    assert json.loads(v2_output.getvalue())["status"] == "UNKNOWN"
    assert json.loads(v3_output.getvalue())["status"] == "UNIQUE"

    display_output = StringIO()
    assert main([
        "什么是" + V3_SURFACE,
        "--artifact-version", "v3", "--display-definition",
    ], stdout=display_output) == 0
    display = json.loads(display_output.getvalue())
    assert display["status"] == "DISPLAY"
    assert display["display_text"] == "失敗的戰爭或競爭"
    source_ref = display["citations"][0]["source_ref"]
    assert source_ref["license_id"] == "CC-BY-SA-4.0"
    assert "contributor=" in source_ref["attribution"]


def test_distribution_adds_only_the_compact_v3_artifact() -> None:
    """wheel 排除 FT31 selection、census、source pack、index 与 raw。"""
    pyproject = tomllib.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"]
        ["share/pure_integer_ai/data/ph2"])
    assert configured[-3:] == (
        "data/ph2/w03_public_sense_runtime_v1.json",
        "data/ph2/w03_public_sense_runtime_v2.json",
        "data/ph2/w03_public_sense_runtime_v3.json",
    )
    assert PUBLIC_W03_SENSE_ARTIFACT_V3.stat().st_size == 157021
    assert not any(
        "ft31" in item
        or "source_v3" in item
        or "multistream" in item
        or "census" in item
        for item in configured)
