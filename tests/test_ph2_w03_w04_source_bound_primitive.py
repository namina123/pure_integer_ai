"""FT27 W03 sense to source-bound W04 primitive bridge tests."""
from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json

import pytest

from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    W03PublicSenseRuntime,
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive import (
    W03W04SourceBoundPrimitiveRuntime,
    project_w03_public_sense_to_w04_primitives,
    query_w03_w04_source_bound_primitive_batch,
    query_w03_w04_source_bound_primitives,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive_contract import (
    W03W04SourceBoundPrimitiveError,
    W04_SOURCE_BOUND_PRIMITIVE_KINDS,
    W04_SOURCE_BOUND_PRIMITIVE_REGISTRY,
)
from pure_integer_ai.experiments.run_ph2_w03_public_sense import main


@pytest.fixture(scope="module")
def sense_runtime() -> W03PublicSenseRuntime:
    return load_w03_public_sense_artifact()


@pytest.fixture(scope="module")
def primitive_runtime(
        sense_runtime: W03PublicSenseRuntime,
        ) -> W03W04SourceBoundPrimitiveRuntime:
    return project_w03_public_sense_to_w04_primitives(sense_runtime)


def test_projection_preserves_every_typed_identity_and_truth_boundary(
        sense_runtime, primitive_runtime) -> None:
    """Every FT26 entry becomes one non-adjudicated source claim."""
    entries = sense_runtime.artifact.entries
    primitives = primitive_runtime.primitives
    assert len(entries) == len(primitives) == 43
    assert primitive_runtime.source_binding_sha256 == (
        "adbd6e9ea8cfb3fc4b0e45c1b66cdbf93721e0500881bbf555410a7f08e84284")
    assert primitive_runtime.projection_sha256 == (
        "3931af38dd8197df0107bd397d9f6fb37d150d51dcf6b8c77848a1e90c59d8a1")
    for entry, primitive in zip(entries, primitives, strict=True):
        assert (
            primitive.entry_key,
            primitive.relation_kind,
            primitive.sense_key,
            primitive.concept_key,
            primitive.observation_key,
            primitive.source_ref.to_dict(),
            primitive.field_roles,
            primitive.active,
            primitive.supersedes_entry_keys,
        ) == (
            entry.entry_key,
            entry.relation_kind,
            entry.sense_key,
            entry.concept_key,
            entry.observation_key,
            entry.source_ref.to_dict(),
            entry.field_roles,
            entry.active,
            entry.supersedes_entry_keys,
        )
        assert primitive.primitive_registry == (
            W04_SOURCE_BOUND_PRIMITIVE_REGISTRY)
        assert primitive.primitive_kind == (
            W04_SOURCE_BOUND_PRIMITIVE_KINDS[entry.relation_kind])
        assert primitive.epistemic_status == "SOURCE_ASSERTED"
        assert primitive.truth_status == "NOT_ADJUDICATED"
        assert primitive.w04_candidate.candidate_key == (
            primitive.primitive_key)
        assert primitive.w04_candidate.source_ref_key == (
            entry.source_ref.stable_key)


def test_projection_and_batch_are_bit_identical(
        sense_runtime, primitive_runtime) -> None:
    """Repeated projection and query over one immutable runtime are stable."""
    rebuilt = project_w03_public_sense_to_w04_primitives(sense_runtime)
    assert rebuilt.to_dict() == primitive_runtime.to_dict()
    queries = (
        W03PublicSenseQuery("首页"),
        W03PublicSenseQuery("金星"),
        W03PublicSenseQuery("鸟类"),
        W03PublicSenseQuery("不存在词项"),
        W03PublicSenseQuery("金星", "距离太阳第二近的行星"),
        W03PublicSenseQuery("金星", "不匹配上下文"),
    )
    first = query_w03_w04_source_bound_primitive_batch(
        primitive_runtime, queries)
    second = query_w03_w04_source_bound_primitive_batch(
        primitive_runtime, queries)
    assert tuple(item.to_dict() for item in first) == tuple(
        item.to_dict() for item in second)
    assert tuple(item.sha256() for item in first) == tuple(
        item.sha256() for item in second)


def test_contract_rejects_asserted_value_relation_mismatch(
        primitive_runtime) -> None:
    """A caller cannot relabel arbitrary text as a typed source claim."""
    original = primitive_runtime.primitives[0]
    candidate = replace(
        original.w04_candidate, context_text="伪造声明对象")
    with pytest.raises(
            W03W04SourceBoundPrimitiveError,
            match="does not match"):
        replace(
            original,
            asserted_value="伪造声明对象",
            w04_candidate=candidate,
        )


def test_w03_non_unique_states_are_never_weakened(
        primitive_runtime) -> None:
    """Ambiguity, conflict, unknown and context behavior remain W03-owned."""
    queries = (
        W03PublicSenseQuery("首页"),
        W03PublicSenseQuery("金星"),
        W03PublicSenseQuery("鸟类"),
        W03PublicSenseQuery("不存在词项"),
        W03PublicSenseQuery("金星", "距离太阳第二近的行星"),
        W03PublicSenseQuery("金星", "不匹配上下文"),
    )
    results = query_w03_w04_source_bound_primitive_batch(
        primitive_runtime, queries)
    assert tuple(item.status for item in results) == (
        "AMBIGUOUS", "CONFLICT", "UNIQUE", "UNKNOWN", "UNIQUE",
        "CLARIFY")
    assert results[0].sense_result.alias_path == ("首页", "首頁")
    assert results[0].clarify_required == 1
    assert results[1].clarify_required == 1
    assert {item.source_ref.source_key for item in results[1].primitives} == {
        "WIKIDATA_REVISION_V1", "ZHWIKTIONARY_20260701"}
    assert results[3].primitives == ()
    assert results[4].primitives[0].relation_kind == "LABEL"
    assert results[4].primitives[0].asserted_value == "金星"
    assert results[4].primitives[0].definition_text == (
        "距离太阳第二近的行星")
    assert results[5].clarify_required == 1
    for result in results:
        assert result.status == result.sense_result.status
        assert result.formal_mastery_claim == 0
        assert result.w03_started == result.w04_started == 0


def test_pack_removal_fails_closed_or_removes_capability(
        sense_runtime) -> None:
    """A dangling pack is rejected; coherent removal removes its queries."""
    base = sense_runtime.artifact
    wiktionary_pack = tuple(
        item for item in base.source_packs
        if item.source_key == "ZHWIKTIONARY_20260701")
    dangling = W03PublicSenseArtifact(
        wiktionary_pack,
        (),
        base.entries,
        base.aliases,
    )
    with pytest.raises(
            W03W04SourceBoundPrimitiveError,
            match="no longer bound"):
        project_w03_public_sense_to_w04_primitives(
            W03PublicSenseRuntime(dangling, "a" * 64))

    retained_entries = tuple(
        item for item in base.entries
        if item.source_ref.source_key == "ZHWIKTIONARY_20260701")
    retained_aliases = tuple(
        item for item in base.aliases
        if item.source_ref.source_key == "ZHWIKTIONARY_20260701")
    reduced = W03PublicSenseArtifact(
        wiktionary_pack, (), retained_entries, retained_aliases)
    projected = project_w03_public_sense_to_w04_primitives(
        W03PublicSenseRuntime(reduced, "b" * 64))
    result = query_w03_w04_source_bound_primitives(
        projected, W03PublicSenseQuery("鸟类"))
    assert result.status == "UNKNOWN"
    assert result.primitives == ()


def test_revision_and_supersede_identity_invalidate_projection(
        sense_runtime, primitive_runtime) -> None:
    """Revision replacement changes identity; old entries become inactive."""
    base = sense_runtime.artifact
    original = next(
        item for item in base.entries
        if item.surface == "鸟类" and item.language == "zh")
    replacement_source = replace(
        original.source_ref,
        revision_id=original.source_ref.revision_id + "-replacement",
        source_identity=original.source_ref.source_identity + "#replacement",
        source_commitment_sha256="f" * 64,
    )
    replacement = replace(original, source_ref=replacement_source)
    replaced_artifact = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        tuple(
            replacement if item.entry_key == original.entry_key else item
            for item in base.entries),
        base.aliases,
    )
    replaced_runtime = project_w03_public_sense_to_w04_primitives(
        W03PublicSenseRuntime(replaced_artifact, "c" * 64))
    prior = next(
        item for item in primitive_runtime.primitives
        if item.entry_key == original.entry_key)
    current = next(
        item for item in replaced_runtime.primitives
        if item.entry_key == original.entry_key)
    assert current.primitive_key != prior.primitive_key
    assert replaced_runtime.projection_sha256 != (
        primitive_runtime.projection_sha256)

    changed_revision = replace(
        base.source_revisions[0], active_sha256="e" * 64)
    revision_artifact = W03PublicSenseArtifact(
        base.source_packs,
        (changed_revision,),
        base.entries,
        base.aliases,
    )
    revision_runtime = project_w03_public_sense_to_w04_primitives(
        W03PublicSenseRuntime(revision_artifact, "e" * 64))
    assert revision_runtime.source_binding_sha256 != (
        primitive_runtime.source_binding_sha256)
    assert revision_runtime.projection_sha256 != (
        primitive_runtime.projection_sha256)

    old = replace(original, entry_key=(8, 1), active=0)
    successor = replace(
        original,
        entry_key=(8, 2),
        observation_key=(8, 3),
        source_ref=replacement_source,
        supersedes_entry_keys=(old.entry_key,),
    )
    superseded_artifact = W03PublicSenseArtifact(
        base.source_packs,
        base.source_revisions,
        (old, successor),
        (),
    )
    superseded_runtime = project_w03_public_sense_to_w04_primitives(
        W03PublicSenseRuntime(superseded_artifact, "d" * 64))
    assert tuple(
        (item.entry_key, item.w04_candidate.active,
         item.w04_candidate.superseded)
        for item in superseded_runtime.primitives
    ) == (((8, 1), 0, 1), ((8, 2), 1, 0))
    result = query_w03_w04_source_bound_primitives(
        superseded_runtime, W03PublicSenseQuery("鸟类"))
    assert result.status == "UNIQUE"
    assert tuple(item.entry_key for item in result.primitives) == ((8, 2),)
    assert result.primitives[0].supersedes_entry_keys == ((8, 1),)


def test_cli_primitive_mode_is_explicit_and_default_json_is_unchanged(
        sense_runtime) -> None:
    """The opt-in projection does not alter the original CLI response."""
    default_output = StringIO()
    assert main(["金星"], stdout=default_output) == 0
    expected = query_w03_public_sense(
        sense_runtime, W03PublicSenseQuery("金星"))
    assert default_output.getvalue() == json.dumps(
        expected.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"

    primitive_output = StringIO()
    assert main(["金星", "--primitive"], stdout=primitive_output) == 0
    value = json.loads(primitive_output.getvalue())
    assert value["status"] == "CONFLICT"
    assert value["clarify_required"] == 1
    assert {item["truth_status"] for item in value["primitives"]} == {
        "NOT_ADJUDICATED"}


def test_ft27_code_contains_no_selected_term_qid_or_answer_dispatch() -> None:
    """Only relation metadata, never selected content, drives projection."""
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    files = (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_source_bound_primitive_contract.py",
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_source_bound_primitive.py",
    )
    combined = "\n".join(
        (repository / item).read_text(encoding="utf-8") for item in files)
    for forbidden in (
            "首页", "首頁", "苹果", "蘋果", "金星",
            '"Q313"', '"Q5113"', "距离太阳第二近的行星"):
        assert forbidden not in combined
