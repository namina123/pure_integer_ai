"""FT33 public-definition v4 compiler, census, runtime, and CLI tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import tomllib

from pure_integer_ai.experiments.ph2_w03_public_definition_compiler_v4 import (
    build_ft33_public_definition_artifact_v4,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    PUBLIC_W03_SENSE_ARTIFACT_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V3_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V4,
    PUBLIC_W03_SENSE_ARTIFACT_V4_SHA256,
    W03PublicSenseRuntime,
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.run_ph2_w03_public_sense import main


REPOSITORY = Path(__file__).resolve().parents[1]
SELECTION = REPOSITORY / (
    "data/ph2/manifests/ft33_w03_public_definition_selection_v4.json")
CENSUS = REPOSITORY / (
    "data/ph2/manifests/ft33_w03_public_definition_census_v4.json")
BASE = REPOSITORY / "data/ph2/w03_public_sense_runtime_v3.json"
BASE_CENSUS = REPOSITORY / (
    "data/ph2/manifests/ft31_w03_public_definition_census_v3.json")
REVIEW = REPOSITORY / (
    "data/ph2/manifests/ft32_public_template_specification_review_v1.json")
PACK_RELATIVE = Path(
    "ph2_ft33_dataset_artifacts/public_definition_source_v4/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v4")
PACK = REPOSITORY / PACK_RELATIVE
SELECTION_SHA256 = (
    "5032e3079eefbc0c6b602d913eccbc7f89f0c8c94466c1e11c8014b770f35092")
CENSUS_SHA256 = (
    "3628324f0b334f7e08116a89202b4447afc947808ac0354dc4681da0ff0b2936")
BASE_SHA256 = (
    "304cc1e6674df856e5c75eefab663e15aab85045f08b603d6650e9ad87a4ff1d")
BASE_CENSUS_SHA256 = (
    "4f637b372bc69c6c10e63ddba087eee1bfab108ed6381802323e39319242bdc4")
REVIEW_SHA256 = (
    "4470e7af54247720bf7137f3088a25deada266c6ccbcf61b8f0b7079e8351455")
V4_SURFACE = "亠"


def _build():
    return build_ft33_public_definition_artifact_v4(
        repository_root=REPOSITORY,
        base_artifact_path=BASE,
        base_artifact_sha256=BASE_SHA256,
        base_census_path=BASE_CENSUS,
        base_census_sha256=BASE_CENSUS_SHA256,
        review_manifest_path=REVIEW,
        review_manifest_relative_path=(
            "data/ph2/manifests/"
            "ft32_public_template_specification_review_v1.json"),
        review_manifest_sha256=REVIEW_SHA256,
        expansion_pack_relative_path=PACK_RELATIVE.as_posix(),
        expansion_pack_root=PACK,
        selection_manifest_sha256=SELECTION_SHA256,
    )


def _query(version: str, surface: str):
    runtime = load_w03_public_sense_artifact(artifact_version=version)
    return query_w03_public_sense(runtime, W03PublicSenseQuery(surface))


def test_v4_compile_census_and_runtime_are_bit_identical() -> None:
    """Repeated compiler output matches the published compact artifact."""
    first = _build()
    second = _build()
    runtime = load_w03_public_sense_artifact(artifact_version="v4")
    assert first.artifact.payload_value() == second.artifact.payload_value()
    assert first.census_value == second.census_value
    assert first.artifact.payload_value() == runtime.artifact.payload_value()
    assert runtime.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_V4_SHA256 == (
        "2e2d16a30eac4ea35f7498ea34ab5ab8e25e1439adc2f68affaf17aefec9d57d")
    assert len(runtime.artifact.source_packs) == 5
    assert len(runtime.artifact.entries) == 305
    assert len(runtime.artifact.aliases) == 18
    assert _query("v4", V4_SURFACE).status == "UNIQUE"


def test_v4_census_freezes_all_pages_and_ft32_zero_authorization() -> None:
    """The 512-page census inherits review decisions without a renderer."""
    raw = CENSUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    census = json.loads(raw.decode("utf-8"))
    assert census["base_artifact_identity"] == {
        "alias_count": 10,
        "artifact_sha256": BASE_SHA256,
        "entry_count": 117,
        "source_pack_count": 4,
        "source_revision_count": 1,
    }
    assert census["base_v3_census_identity"] == {
        "definition_count": 125,
        "eligible_definition_count": 65,
        "page_count": 256,
        "page_status_counts": {
            "ACCEPTED_DEFINITION": 41,
            "NON_CHINESE_DEFINITION": 12,
            "NO_DEFINITION": 197,
            "REDIRECT": 6,
        },
        "render_status_counts": {
            "DISPLAY": 67,
            "MALFORMED_MARKUP": 1,
            "UNSUPPORTED_MARKUP": 57,
        },
    }
    assert census["page_count"] == 512
    assert census["definition_count"] == 299
    assert census["eligible_definition_count"] == 188
    assert census["page_status_counts"] == {
        "ACCEPTED_DEFINITION": 111,
        "NON_CHINESE_DEFINITION": 16,
        "NO_DEFINITION": 377,
        "REDIRECT": 8,
    }
    assert census["render_status_counts"] == {
        "DISPLAY": 147,
        "MALFORMED_MARKUP": 2,
        "UNSUPPORTED_MARKUP": 150,
    }
    inherited = census["inherited_template_review"]
    assert inherited["review_manifest_sha256"] == REVIEW_SHA256
    assert inherited["renderer_authorized_count"] == 0
    assert [(item["template_name"], item["status"])
            for item in inherited["decisions"]] == [
        ("place", "BLOCKED"),
        ("zh-div", "REVIEWED_NOT_AUTHORIZED"),
    ]
    gate = census["template_evidence_gate"]
    assert gate["renderer_authorized_count"] == 0
    reviewed = {
        item["template_name"]: item
        for item in gate["templates"]
        if item["public_specification_reviewed"] == 1
    }
    assert reviewed["place"]["inherited_review_status"] == "BLOCKED"
    assert reviewed["zh-div"]["inherited_review_status"] == (
        "REVIEWED_NOT_AUTHORIZED")
    assert reviewed["place"]["renderer_authorized"] == 0
    assert reviewed["zh-div"]["renderer_authorized"] == 0
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == (
        SELECTION_SHA256)


def test_v4_is_opt_in_and_v1_v2_v3_remain_frozen() -> None:
    """Default and all predecessor bytes/behavior remain unchanged."""
    assert load_w03_public_sense_artifact().artifact_sha256 == (
        PUBLIC_W03_SENSE_ARTIFACT_SHA256)
    assert load_w03_public_sense_artifact(
        artifact_version="v2").artifact_sha256 == (
            PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256)
    assert load_w03_public_sense_artifact(
        artifact_version="v3").artifact_sha256 == (
            PUBLIC_W03_SENSE_ARTIFACT_V3_SHA256)
    assert _query("v1", V4_SURFACE).status == "UNKNOWN"
    assert _query("v2", V4_SURFACE).status == "UNKNOWN"
    assert _query("v3", V4_SURFACE).status == "UNKNOWN"
    assert _query("v4", V4_SURFACE).status == "UNIQUE"

    output = StringIO()
    assert main([
        V4_SURFACE, "--artifact-version", "v4",
    ], stdout=output) == 0
    assert json.loads(output.getvalue())["status"] == "UNIQUE"


def test_v4_source_removal_revision_and_supersede_change_capability() -> None:
    """V4-only capability remains causally bound to source lifecycle."""
    runtime = load_w03_public_sense_artifact(artifact_version="v4")
    base = runtime.artifact
    original = next(item for item in base.entries
                    if item.surface == V4_SURFACE)
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
    assert query_w03_public_sense(
        W03PublicSenseRuntime(reduced, "1" * 64),
        W03PublicSenseQuery(V4_SURFACE),
    ).status == "UNKNOWN"

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
    changed = query_w03_public_sense(
        W03PublicSenseRuntime(replaced, "2" * 64),
        W03PublicSenseQuery(V4_SURFACE),
    )
    assert changed.status == "UNIQUE"
    assert changed.sha256() != _query("v4", V4_SURFACE).sha256()

    old = replace(original, active=0)
    successor = replace(
        original,
        entry_key=(33, 1),
        observation_key=(33, 2),
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
    result = query_w03_public_sense(
        W03PublicSenseRuntime(superseded, "3" * 64),
        W03PublicSenseQuery(V4_SURFACE),
    )
    assert result.status == "UNIQUE"
    assert result.candidates[0].entry.entry_key == successor.entry_key


def test_distribution_adds_only_the_compact_v4_artifact() -> None:
    """Wheel excludes FT33 census, selection, source pack, index, and raw."""
    pyproject = tomllib.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"]
        ["share/pure_integer_ai/data/ph2"])
    assert configured[-4:] == (
        "data/ph2/w03_public_sense_runtime_v1.json",
        "data/ph2/w03_public_sense_runtime_v2.json",
        "data/ph2/w03_public_sense_runtime_v3.json",
        "data/ph2/w03_public_sense_runtime_v4.json",
    )
    assert PUBLIC_W03_SENSE_ARTIFACT_V4.stat().st_size == 418031
    assert not any(
        "ft33" in item or "source_v4" in item or "census" in item
        for item in configured)
