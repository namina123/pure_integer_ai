"""FT28 source-bound proposition and raw definition QA tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json

import pytest

from pure_integer_ai.cognition.shared.identity import SourceRef
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
    project_w03_public_sense_to_w04_primitives,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive_contract import (
    W03W04SourceBoundPrimitiveError,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition import (
    W04W05SourceBoundPropositionRuntime,
    project_w04_primitives_to_w05_source_bound_propositions,
    query_w04_w05_source_bound_proposition_batch,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition_contract import (
    W04W05SourceBoundPropositionError,
    W05_SOURCE_BOUND_PROPOSITION_KINDS,
    source_bound_candidate_projection,
    source_bound_proposition_key,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa import (
    answer_w05_raw_definition_batch,
    answer_w05_raw_definition_question,
    match_w05_raw_definition_question,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05RawDefinitionRequest,
)
from pure_integer_ai.experiments.run_ph2_w03_public_sense import main


@pytest.fixture(scope="module")
def sense_runtime() -> W03PublicSenseRuntime:
    return load_w03_public_sense_artifact()


@pytest.fixture(scope="module")
def proposition_runtime(
        sense_runtime: W03PublicSenseRuntime,
        ) -> W04W05SourceBoundPropositionRuntime:
    primitive = project_w03_public_sense_to_w04_primitives(sense_runtime)
    return project_w04_primitives_to_w05_source_bound_propositions(primitive)


def _artifact_for_entries(base, entries) -> W03PublicSenseArtifact:
    entries = tuple(sorted(entries, key=lambda item: item.entry_key))
    bindings = {
        (
            item.source_ref.source_key,
            item.source_ref.snapshot_id,
            item.source_ref.license_id,
        )
        for item in entries
    }
    packs = tuple(
        item for item in base.source_packs
        if (item.source_key, item.snapshot_id, item.license_id) in bindings
    )
    return W03PublicSenseArtifact(
        packs, base.source_revisions, entries, ())


def _proposition_runtime_for_artifact(
        artifact: W03PublicSenseArtifact,
        marker: str,
        ) -> W04W05SourceBoundPropositionRuntime:
    artifact_sha256 = hashlib.sha256(marker.encode("ascii")).hexdigest()
    sense = W03PublicSenseRuntime(artifact, artifact_sha256)
    primitive = project_w03_public_sense_to_w04_primitives(sense)
    return project_w04_primitives_to_w05_source_bound_propositions(primitive)


def _astronomy_definition(base):
    return next(
        item for item in base.entries
        if item.relation_kind == "DEFINITION"
        and item.definition_text is not None
        and "astronomy" in item.definition_text
    )


def test_projection_preserves_all_ft27_identity_and_complete_candidate(
        proposition_runtime) -> None:
    """All 43 primitives become deterministic non-adjudicated claims."""
    primitive_runtime = proposition_runtime.primitive_runtime
    propositions = proposition_runtime.propositions
    assert len(primitive_runtime.primitives) == len(propositions) == 43
    assert proposition_runtime.proposition_projection_sha256 == (
        "2451062a2a728171146cf8b4d986ce27b2791c10efe0203fa5d88ab7d9001244")
    for primitive, proposition in zip(
            primitive_runtime.primitives, propositions, strict=True):
        assert proposition.primitive == primitive
        assert proposition.proposition_key == (
            source_bound_proposition_key(primitive))
        assert proposition.proposition_kind == (
            W05_SOURCE_BOUND_PROPOSITION_KINDS[primitive.relation_kind])
        assert proposition.epistemic_status == "SOURCE_ASSERTED"
        assert proposition.truth_status == "NOT_ADJUDICATED"
        assert proposition.w05_candidate.to_dict() == (
            source_bound_candidate_projection(
                primitive, proposition.proposition_key).to_dict())
        candidate = proposition.w05_candidate
        restored = SourceRef.from_stable_key(candidate.source_ref_key)
        assert restored.stable_key() == candidate.source_ref_key
        assert candidate.source_record_key == primitive.source_ref.stable_key
        assert candidate.source_key == primitive.source_ref.source_key
        assert candidate.source_commitment == (
            primitive.source_ref.source_commitment_sha256)
        assert candidate.reasoning_status == (
            "AUTHORIZED" if primitive.active else "SUPERSEDED")
        assert candidate.active + candidate.superseded == 1


def test_projection_and_query_are_bit_identical(
        sense_runtime, proposition_runtime) -> None:
    """Repeated build and bounded queries are bit-identical."""
    rebuilt = project_w04_primitives_to_w05_source_bound_propositions(
        project_w03_public_sense_to_w04_primitives(sense_runtime))
    assert rebuilt.to_dict() == proposition_runtime.to_dict()
    queries = (
        W03PublicSenseQuery("首页"),
        W03PublicSenseQuery("金星"),
        W03PublicSenseQuery("鸟类"),
        W03PublicSenseQuery("不存在词项"),
        W03PublicSenseQuery("金星", "距离太阳第二近的行星"),
        W03PublicSenseQuery("金星", "不匹配上下文"),
    )
    first = query_w04_w05_source_bound_proposition_batch(
        proposition_runtime, queries)
    second = query_w04_w05_source_bound_proposition_batch(
        proposition_runtime, queries)
    assert tuple(item.status for item in first) == (
        "AMBIGUOUS", "CONFLICT", "UNIQUE", "UNKNOWN", "UNIQUE",
        "CLARIFY")
    assert tuple(item.to_dict() for item in first) == tuple(
        item.to_dict() for item in second)
    assert tuple(item.sha256() for item in first) == tuple(
        item.sha256() for item in second)
    for item in first:
        assert item.status == item.primitive_result.status
        assert item.formal_mastery_claim == 0
        assert item.w03_started == item.w04_started == item.w05_started == 0


def test_definition_question_boundaries_and_five_decision_states(
        sense_runtime, proposition_runtime) -> None:
    """Punctuation is single-match and every honest result state is covered."""
    definition = _astronomy_definition(sense_runtime.artifact)
    answer_requests = tuple(
        W05RawDefinitionRequest(
            question,
            definition.definition_text,
        )
        for question in (
            "什么是金星", "什么是金星?", "什么是金星？",
            "金星是什么意思", "金星是什么意思？",
        )
    )
    matches = tuple(
        match_w05_raw_definition_question(item)
        for item in answer_requests)
    assert all(len(item) == 1 for item in matches)
    assert tuple(item[0].boundary_mark for item in matches) == (
        "", "?", "？", "", "？")
    answers = answer_w05_raw_definition_batch(
        proposition_runtime, answer_requests)
    assert all(item.status == "ANSWER" for item in answers)
    assert all(
        item.answer_text == definition.definition_text for item in answers)
    assert all(
        len(item.selected_propositions) == 1 for item in answers)

    requests = (
        W05RawDefinitionRequest("什么是首页"),
        W05RawDefinitionRequest("什么是金星"),
        W05RawDefinitionRequest("什么是鸟类"),
        W05RawDefinitionRequest("什么是金星", "不匹配上下文"),
        W05RawDefinitionRequest("什么是金星是什么意思"),
        W05RawDefinitionRequest("请解释金星"),
    )
    results = answer_w05_raw_definition_batch(
        proposition_runtime, requests)
    assert tuple(item.status for item in results) == (
        "AMBIGUOUS", "CONFLICT", "UNKNOWN", "CLARIFY", "CLARIFY",
        "UNKNOWN")
    assert all(item.answer_text is None for item in results)
    assert results[2].proposition_result.status == "UNIQUE"
    assert results[2].definition_candidates == ()
    assert results[4].proposition_result is None
    assert len(results[4].matches) == 2


def test_unique_concept_with_multiple_definitions_never_takes_first(
        sense_runtime) -> None:
    """A UNIQUE concept still requires one unique definition text."""
    base = sense_runtime.artifact
    definitions = tuple(
        item for item in base.entries
        if item.surface == "金星" and item.relation_kind == "DEFINITION")
    first, second = definitions[:2]
    second = replace(second, concept_key=first.concept_key)
    artifact = _artifact_for_entries(base, (first, second))
    runtime = _proposition_runtime_for_artifact(artifact, "m")
    result = answer_w05_raw_definition_question(
        runtime, W05RawDefinitionRequest("什么是金星"))
    assert result.proposition_result.status == "UNIQUE"
    assert len(result.definition_candidates) == 2
    assert result.status == "CLARIFY"
    assert result.answer_text is None
    assert result.selected_propositions == ()


def test_pack_revision_and_supersede_changes_capability_or_identity(
        sense_runtime) -> None:
    """Source deletion and lifecycle changes propagate into raw answers."""
    base = sense_runtime.artifact
    definition = _astronomy_definition(base)
    wikidata_entries = tuple(
        item for item in base.entries
        if item.source_ref.source_key == "WIKIDATA_REVISION_V1")
    no_definition_runtime = _proposition_runtime_for_artifact(
        _artifact_for_entries(base, wikidata_entries), "n")
    removed = answer_w05_raw_definition_question(
        no_definition_runtime, W05RawDefinitionRequest("什么是金星"))
    assert removed.status == "UNKNOWN"
    assert removed.definition_candidates == ()

    wikidata_pack = next(
        item for item in base.source_packs
        if item.source_key == "WIKIDATA_REVISION_V1")
    dangling = W03PublicSenseArtifact(
        (wikidata_pack,), (), (definition,), ())
    with pytest.raises(
            W03W04SourceBoundPrimitiveError,
            match="no longer bound"):
        project_w03_public_sense_to_w04_primitives(
            W03PublicSenseRuntime(dangling, "d" * 64))

    original_artifact = _artifact_for_entries(base, (definition,))
    original_runtime = _proposition_runtime_for_artifact(
        original_artifact, "o")
    original_answer = answer_w05_raw_definition_question(
        original_runtime,
        W05RawDefinitionRequest(
            "什么是金星", definition.definition_text),
    )
    assert original_answer.status == "ANSWER"

    replacement_source = replace(
        definition.source_ref,
        revision_id=definition.source_ref.revision_id + "-replacement",
        source_identity=definition.source_ref.source_identity + "#replacement",
        source_commitment_sha256="f" * 64,
    )
    replacement = replace(definition, source_ref=replacement_source)
    replacement_runtime = _proposition_runtime_for_artifact(
        _artifact_for_entries(base, (replacement,)), "r")
    replacement_answer = answer_w05_raw_definition_question(
        replacement_runtime,
        W05RawDefinitionRequest(
            "什么是金星", replacement.definition_text),
    )
    assert replacement_answer.status == "ANSWER"
    assert replacement_answer.selected_propositions[0].proposition_key != (
        original_answer.selected_propositions[0].proposition_key)
    assert replacement_answer.selected_propositions[0].primitive.source_ref == (
        replacement_source)

    old = replace(definition, entry_key=(8, 1), active=0)
    successor = replace(
        definition,
        entry_key=(8, 2),
        observation_key=(8, 3),
        source_ref=replacement_source,
        supersedes_entry_keys=(old.entry_key,),
    )
    successor_runtime = _proposition_runtime_for_artifact(
        _artifact_for_entries(base, (old, successor)), "s")
    old_proposition, current_proposition = successor_runtime.propositions
    assert old_proposition.w05_candidate.lifecycle_status == "SUPERSEDED"
    assert current_proposition.w05_candidate.lifecycle_status == "ACTIVE"
    assert current_proposition.supersedes_proposition_keys == (
        old_proposition.proposition_key,)
    successor_answer = answer_w05_raw_definition_question(
        successor_runtime,
        W05RawDefinitionRequest(
            "什么是金星", successor.definition_text),
    )
    assert successor_answer.status == "ANSWER"
    assert successor_answer.selected_propositions == (current_proposition,)

    forged_current = replace(
        current_proposition, supersedes_proposition_keys=())
    with pytest.raises(
            W04W05SourceBoundPropositionError,
            match="supersede projection drifted"):
        replace(
            successor_runtime,
            propositions=(old_proposition, forged_current),
        )


def test_contract_rejects_forged_candidate_and_projection_commitment(
        proposition_runtime) -> None:
    """Callers cannot relabel candidate structure or its runtime digest."""
    original = proposition_runtime.propositions[0]
    forged_candidate = replace(
        original.w05_candidate,
        surface=original.w05_candidate.surface + "-forged",
    )
    with pytest.raises(
            W04W05SourceBoundPropositionError,
            match="candidate projection drifted"):
        replace(original, w05_candidate=forged_candidate)
    with pytest.raises(
            W04W05SourceBoundPropositionError,
            match="identity drifted"):
        replace(original, proposition_key=(1, 28, 5, 999))
    with pytest.raises(
            W04W05SourceBoundPropositionError,
            match="projection commitment drifted"):
        replace(
            proposition_runtime,
            proposition_projection_sha256="0" * 64,
        )
    result = query_w04_w05_source_bound_proposition_batch(
        proposition_runtime, (W03PublicSenseQuery("金星"),))[0]
    with pytest.raises(
            W04W05SourceBoundPropositionError,
            match="record commitment drifted"):
        replace(result, record_commitment_sha256="0" * 64)


def test_cli_modes_are_explicit_and_default_json_is_unchanged(
        sense_runtime) -> None:
    """New modes are opt-in and preserve the original sense response bytes."""
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

    proposition_output = StringIO()
    assert main(
        ["金星", "--proposition"], stdout=proposition_output) == 0
    proposition = json.loads(proposition_output.getvalue())
    assert proposition["status"] == "CONFLICT"
    assert {item["truth_status"] for item in proposition["propositions"]} == {
        "NOT_ADJUDICATED"}

    definition = _astronomy_definition(sense_runtime.artifact)
    definition_output = StringIO()
    assert main(
        [
            "什么是金星",
            "--context", definition.definition_text,
            "--definition",
        ],
        stdout=definition_output,
    ) == 0
    answer = json.loads(definition_output.getvalue())
    assert answer["status"] == "ANSWER"
    assert answer["answer_text"] == definition.definition_text


def test_ft28_code_contains_no_selected_term_qid_or_answer_dispatch() -> None:
    """Only general relation and question structures drive FT28 behavior."""
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    files = (
        "src/pure_integer_ai/experiments/"
        "ph2_w04_w05_source_bound_proposition_contract.py",
        "src/pure_integer_ai/experiments/"
        "ph2_w04_w05_source_bound_proposition.py",
        "src/pure_integer_ai/experiments/"
        "ph2_w05_raw_definition_qa_contract.py",
        "src/pure_integer_ai/experiments/ph2_w05_raw_definition_qa.py",
    )
    combined = "\n".join(
        (repository / item).read_text(encoding="utf-8") for item in files)
    for forbidden in (
            "首页", "首頁", "苹果", "蘋果", "金星",
            '"Q313"', '"Q5113"', "距离太阳第二近的行星"):
        assert forbidden not in combined
