"""FT30 public-definition v2 compiler, census, runtime, and opt-in CLI."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import tomllib

from pure_integer_ai.experiments.ph2_w03_public_definition_compiler_v2 import (
    build_ft30_public_definition_artifact_v2,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    PUBLIC_W03_SENSE_ARTIFACT,
    PUBLIC_W03_SENSE_ARTIFACT_SHA256,
    PUBLIC_W03_SENSE_ARTIFACT_V2,
    PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256,
    W03PublicSenseRuntime,
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.run_ph2_w03_public_sense import main


REPOSITORY = Path(__file__).resolve().parents[1]
SELECTION = REPOSITORY / (
    "data/ph2/manifests/ft30_w03_public_definition_selection_v2.json")
CENSUS = REPOSITORY / (
    "data/ph2/manifests/ft30_w03_public_definition_census_v2.json")
PACK_RELATIVE = Path(
    "ph2_ft30_dataset_artifacts/public_definition_source_v2/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--w03-public-definition-v2")
PACK = REPOSITORY / PACK_RELATIVE
SELECTION_SHA256 = (
    "c9271b1d4481086b7554a87866ff2666d27977d10f597f293795e2b420795c2c")
CENSUS_SHA256 = (
    "4a82626cc0eaeea74dec8a8b2626447ddf1f9fb845cd41118e3e96b67afe604b")
BASE_SHA256 = (
    "7e0e1ae1b4c7bb334d9581c887f880949c5a43c64ca68aad4a9e05a6206e3792")
V2_SURFACE = "蘇維埃社會主義共和國聯盟"


def _build():
    return build_ft30_public_definition_artifact_v2(
        base_artifact_path=PUBLIC_W03_SENSE_ARTIFACT,
        base_artifact_sha256=BASE_SHA256,
        expansion_pack_relative_path=PACK_RELATIVE.as_posix(),
        expansion_pack_root=PACK,
        selection_manifest_sha256=SELECTION_SHA256,
    )


def _query(runtime: W03PublicSenseRuntime, surface: str):
    return query_w03_public_sense(runtime, W03PublicSenseQuery(surface))


def test_v2_compile_census_and_query_are_bit_identical() -> None:
    """Repeated source-pack compilation and query retain exact identities."""
    first = _build()
    second = _build()
    runtime = load_w03_public_sense_artifact(artifact_version="v2")

    assert first.artifact.payload_value() == second.artifact.payload_value()
    assert first.census_value == second.census_value
    assert first.artifact.payload_value() == runtime.artifact.payload_value()
    assert len(runtime.artifact.source_packs) == 3
    assert len(runtime.artifact.source_revisions) == 1
    assert len(runtime.artifact.entries) == 52
    assert len(runtime.artifact.aliases) == 4
    assert runtime.artifact_sha256 == PUBLIC_W03_SENSE_ARTIFACT_V2_SHA256 == (
        "db3200d42004cc1fdcf03cbd5872e37a7437872b688fae79c322eb0eff70946a")
    assert _query(runtime, V2_SURFACE).to_dict() == _query(
        runtime, V2_SURFACE).to_dict()
    assert _query(runtime, V2_SURFACE).status == "UNIQUE"


def test_census_freezes_v1_baseline_and_all_selected_pages() -> None:
    """The 7/3/2 baseline and all 32 non-cherry-picked pages stay visible."""
    raw = CENSUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    census = json.loads(raw.decode("utf-8"))
    assert census["base_artifact_sha256"] == BASE_SHA256
    assert census["base_definition_count"] == 12
    assert census["base_definition_rendering_counts"] == {
        "DISPLAY": 7,
        "NO_SOURCE_ANSWER": 2,
        "UNKNOWN_TEMPLATE": 3,
    }
    assert len(census["base_definition_rendering_baseline"]) == 12
    assert census["page_count"] == 32
    assert census["definition_count"] == 18
    assert census["eligible_definition_count"] == 9
    assert census["page_status_counts"] == {
        "ACCEPTED_DEFINITION": 4,
        "NON_CHINESE_DEFINITION": 2,
        "NO_DEFINITION": 23,
        "REDIRECT": 3,
    }
    assert census["render_status_counts"] == {
        "DISPLAY": 14,
        "UNSUPPORTED_MARKUP": 4,
    }
    assert sum(
        item["failure_code"] == "UNKNOWN_TEMPLATE"
        for item in census["definitions"]
    ) == 4
    assert all(
        item["title"]
        and len(item["observation_key"]) == 2
        and item["source_ref"]["license_id"] == "CC-BY-SA-4.0"
        and "Wiktionary contributors" in item["source_ref"]["attribution"]
        and "page_title=" in item["source_ref"]["attribution"]
        and "revision_timestamp=" in item["source_ref"]["attribution"]
        and "contributor=" in item["source_ref"]["attribution"]
        for item in census["definitions"]
    )
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == (
        SELECTION_SHA256)


def test_expansion_source_removal_revision_and_supersede_change_capability(
        ) -> None:
    """Capability remains causally bound to source, revision, and active state."""
    runtime = load_w03_public_sense_artifact(artifact_version="v2")
    base = runtime.artifact
    original = next(
        item for item in base.entries if item.surface == V2_SURFACE)
    expansion_pack = next(
        item for item in base.source_packs
        if item.relative_path == PACK_RELATIVE.as_posix())
    reduced = W03PublicSenseArtifact(
        tuple(item for item in base.source_packs if item != expansion_pack),
        base.source_revisions,
        tuple(
            item for item in base.entries
            if item.source_ref.stable_key != original.source_ref.stable_key),
        tuple(
            item for item in base.aliases
            if item.source_ref.stable_key != original.source_ref.stable_key),
    )
    assert _query(W03PublicSenseRuntime(reduced, "1" * 64), V2_SURFACE).status == (
        "UNKNOWN")

    replacement_source = replace(
        original.source_ref,
        revision_id=original.source_ref.revision_id + "-replacement",
        source_identity=original.source_ref.source_identity + "#replacement",
        source_commitment_sha256="f" * 64,
    )
    replacement = replace(original, source_ref=replacement_source)
    replaced_entries = tuple(
        replacement if item.entry_key == original.entry_key else item
        for item in base.entries)
    replaced = W03PublicSenseArtifact(
        base.source_packs, base.source_revisions, replaced_entries,
        base.aliases)
    replaced_result = _query(
        W03PublicSenseRuntime(replaced, "2" * 64), V2_SURFACE)
    assert replaced_result.status == "UNIQUE"
    assert replaced_result.sha256() != _query(runtime, V2_SURFACE).sha256()
    assert replaced_result.candidates[0].entry.source_ref == replacement_source

    old = replace(original, active=0)
    successor = replace(
        original,
        entry_key=(30, 1),
        observation_key=(30, 2),
        definition_text="公开修订后的来源定义",
        source_ref=replacement_source,
        supersedes_entry_keys=(original.entry_key,),
    )
    retained = tuple(
        item for item in base.entries if item.entry_key != original.entry_key)
    superseded = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        tuple(sorted((*retained, old, successor), key=lambda item: item.entry_key)),
        base.aliases,
    )
    result = _query(W03PublicSenseRuntime(superseded, "3" * 64), V2_SURFACE)
    assert result.status == "UNIQUE"
    assert len(result.candidates) == 1
    assert result.candidates[0].entry.entry_key == successor.entry_key
    assert result.candidates[0].entry.definition_text == successor.definition_text


def test_v2_is_explicit_and_default_v1_bytes_and_behavior_stay_frozen() -> None:
    """Default API/CLI remain v1; v2 is available only by explicit selection."""
    default = load_w03_public_sense_artifact()
    explicit_v1 = load_w03_public_sense_artifact(artifact_version="v1")
    explicit_v2 = load_w03_public_sense_artifact(artifact_version="v2")
    assert default.artifact_sha256 == explicit_v1.artifact_sha256 == (
        PUBLIC_W03_SENSE_ARTIFACT_SHA256)
    assert PUBLIC_W03_SENSE_ARTIFACT.read_bytes() == (
        REPOSITORY / "data/ph2/w03_public_sense_runtime_v1.json").read_bytes()
    assert _query(default, V2_SURFACE).status == "UNKNOWN"
    assert _query(explicit_v2, V2_SURFACE).status == "UNIQUE"

    default_output = StringIO()
    explicit_v1_output = StringIO()
    explicit_v2_output = StringIO()
    assert main([V2_SURFACE], stdout=default_output) == 0
    assert main([
        V2_SURFACE, "--artifact-version", "v1",
    ], stdout=explicit_v1_output) == 0
    assert main([
        V2_SURFACE, "--artifact-version", "v2",
    ], stdout=explicit_v2_output) == 0
    assert default_output.getvalue() == explicit_v1_output.getvalue()
    assert json.loads(default_output.getvalue())["status"] == "UNKNOWN"
    assert json.loads(explicit_v2_output.getvalue())["status"] == "UNIQUE"


def test_distribution_adds_only_the_compact_v2_artifact() -> None:
    """Wheel excludes FT30 selection, census, source pack, index, and raw XML."""
    pyproject = tomllib.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"]
        ["share/pure_integer_ai/data/ph2"])
    assert configured[-2:] == (
        "data/ph2/w03_public_sense_runtime_v1.json",
        "data/ph2/w03_public_sense_runtime_v2.json",
    )
    assert PUBLIC_W03_SENSE_ARTIFACT_V2.stat().st_size == 63656
    assert not any(
        "ft30" in item
        or "source_v2" in item
        or "multistream" in item
        or "census" in item
        for item in configured)
