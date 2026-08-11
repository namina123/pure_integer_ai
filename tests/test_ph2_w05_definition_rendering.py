"""FT29 source-bound definition rendering and citation tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json

import pytest

from pure_integer_ai.experiments.ph2_mediawiki_inline_ast import (
    MediaWikiInlineLabel,
    MediaWikiInlineLink,
    MediaWikiInlineParseError,
    MediaWikiInlineText,
    project_mediawiki_inline,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseArtifact,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    W03PublicSenseRuntime,
    load_w03_public_sense_artifact,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive import (
    project_w03_public_sense_to_w04_primitives,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition import (
    W04W05SourceBoundPropositionRuntime,
    project_w04_primitives_to_w05_source_bound_propositions,
)
from pure_integer_ai.experiments.ph2_w05_definition_rendering import (
    render_w05_definition_answer,
    render_w05_definition_batch,
)
from pure_integer_ai.experiments.ph2_w05_definition_rendering_contract import (
    W05DefinitionRenderingError,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa import (
    answer_w05_raw_definition_question,
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


def _astronomy_definition(runtime: W03PublicSenseRuntime):
    return next(
        item for item in runtime.artifact.entries
        if item.relation_kind == "DEFINITION"
        and item.definition_text is not None
        and "astronomy" in item.definition_text
    )


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


def _runtime_for_artifact(
        artifact: W03PublicSenseArtifact,
        marker: str,
        ) -> W04W05SourceBoundPropositionRuntime:
    digest = hashlib.sha256(marker.encode("ascii")).hexdigest()
    runtime = W03PublicSenseRuntime(artifact, digest)
    primitive = project_w03_public_sense_to_w04_primitives(runtime)
    return project_w04_primitives_to_w05_source_bound_propositions(primitive)


def _raw_answer(
        runtime: W04W05SourceBoundPropositionRuntime,
        definition,
        ):
    return answer_w05_raw_definition_question(
        runtime,
        W05RawDefinitionRequest(
            "什么是" + definition.surface,
            definition.definition_text,
            definition.language,
        ),
    )


def test_inline_ast_projects_links_labels_whitespace_and_punctuation() -> None:
    source = (
        "{{lb|zh|astronomy|formal}} "
        "[[太陽系|太阳系]]的第二顆[[行星]]，為[[類地行星]]；可見。"
    )
    first = project_mediawiki_inline(source)
    second = project_mediawiki_inline(source)
    assert first.to_dict() == second.to_dict()
    assert first.display_text == (
        "（astronomy、formal） 太阳系的第二顆行星，為類地行星；可見。")
    assert first.document.source_text == source
    assert tuple(type(item) for item in first.document.nodes) == (
        MediaWikiInlineLabel,
        MediaWikiInlineText,
        MediaWikiInlineLink,
        MediaWikiInlineText,
        MediaWikiInlineLink,
        MediaWikiInlineText,
        MediaWikiInlineLink,
        MediaWikiInlineText,
    )
    assert first.document.nodes[2].target == "太陽系"
    assert first.document.nodes[2].display_label == "太阳系"


@pytest.mark.parametrize(("source", "code"), (
    ("{{unknown|zh|x}}", "UNKNOWN_TEMPLATE"),
    ("{{lb|zh|x", "UNBALANCED_TEMPLATE"),
    ("[[x", "UNBALANCED_LINK"),
    ("[[x|y|z]]", "AMBIGUOUS_LINK"),
    (r"\[[x]]", "ILLEGAL_ESCAPE"),
    ("{{lb|zh|[[x]]}}", "NESTED_MARKUP"),
    ("{{{value}}}", "UNSUPPORTED_VARIABLE"),
    ("[[File:x]]", "UNSUPPORTED_LINK_TARGET"),
    ("[[x#section]]", "UNSUPPORTED_LINK_TARGET"),
    ("[https://example.invalid x]", "UNSUPPORTED_INLINE_MARKUP"),
    ("''x''", "UNSUPPORTED_INLINE_MARKUP"),
    ("<!--x-->", "UNSUPPORTED_INLINE_MARKUP"),
))
def test_inline_ast_fails_closed_for_unsupported_or_ambiguous_markup(
        source: str,
        code: str,
        ) -> None:
    with pytest.raises(MediaWikiInlineParseError) as captured:
        project_mediawiki_inline(source)
    assert captured.value.code == code


def test_real_answer_keeps_raw_and_binds_display_to_full_citation_chain(
        sense_runtime,
        proposition_runtime,
        ) -> None:
    definition = _astronomy_definition(sense_runtime)
    answer = _raw_answer(proposition_runtime, definition)
    first = render_w05_definition_answer(answer)
    second = render_w05_definition_batch((answer,))[0]
    assert first.to_dict() == second.to_dict()
    assert first.status == "DISPLAY"
    assert first.raw_source_text == definition.definition_text
    assert first.display_text == (
        "（astronomy） 太陽系的第二顆行星，為類地行星")
    assert first.source_answer_sha256 == (
        "af09b2a0525c3de0473b78ca41cf50509749a6055da860a1c5a25cc10ad9023c")
    assert first.display_projection_sha256 == (
        "aca81c23ac4536b24d5d0a6268e70dfe18422a839385eaa6ef59c14498f4f701")
    assert len(first.citations) == 1
    citation = first.citations[0]
    proposition = answer.selected_propositions[0]
    primitive = proposition.primitive
    assert citation.proposition_key == proposition.proposition_key
    assert citation.primitive_key == primitive.primitive_key
    assert citation.entry_key == primitive.entry_key
    assert citation.sense_key == primitive.sense_key
    assert citation.concept_key == primitive.concept_key
    assert citation.observation_key == primitive.observation_key
    assert citation.source_ref == primitive.source_ref
    assert citation.source_ref.revision_id == definition.source_ref.revision_id
    assert citation.source_ref.license_id == "CC-BY-SA-4.0"
    assert citation.source_answer_trace_commitment_sha256 == (
        answer.trace_commitment_sha256)
    assert citation.proposition_query_record_commitment_sha256 == (
        answer.proposition_result.record_commitment_sha256)


def test_unknown_template_and_absent_answer_preserve_honest_boundaries(
        sense_runtime,
        proposition_runtime,
        ) -> None:
    unknown_template = next(
        item for item in sense_runtime.artifact.entries
        if item.definition_text == "{{w|蘋果公司}}")
    unsupported = render_w05_definition_answer(
        _raw_answer(proposition_runtime, unknown_template))
    assert unsupported.status == "UNSUPPORTED_MARKUP"
    assert unsupported.failure_code == "UNKNOWN_TEMPLATE"
    assert unsupported.raw_source_text == unknown_template.definition_text
    assert unsupported.display_text is None
    assert len(unsupported.citations) == 1

    missing = answer_w05_raw_definition_question(
        proposition_runtime,
        W05RawDefinitionRequest("什么是不存在词项"),
    )
    no_answer = render_w05_definition_answer(missing)
    assert no_answer.status == "NO_SOURCE_ANSWER"
    assert no_answer.source_answer.status == "UNKNOWN"
    assert no_answer.raw_source_text is None
    assert no_answer.display_text is None
    assert no_answer.citations == ()


def test_source_deletion_revision_supersede_and_nonunique_source_propagate(
        sense_runtime,
        ) -> None:
    base = sense_runtime.artifact
    definition = _astronomy_definition(sense_runtime)
    wikidata_entries = tuple(
        item for item in base.entries
        if item.source_ref.source_key == "WIKIDATA_REVISION_V1")
    removed_runtime = _runtime_for_artifact(
        _artifact_for_entries(base, wikidata_entries), "removed")
    removed_answer = answer_w05_raw_definition_question(
        removed_runtime,
        W05RawDefinitionRequest("什么是" + definition.surface),
    )
    assert render_w05_definition_answer(removed_answer).status == (
        "NO_SOURCE_ANSWER")

    replacement_source = replace(
        definition.source_ref,
        revision_id=definition.source_ref.revision_id + "-replacement",
        source_identity=definition.source_ref.source_identity + "#replacement",
        source_commitment_sha256="f" * 64,
    )
    replacement = replace(definition, source_ref=replacement_source)
    replacement_runtime = _runtime_for_artifact(
        _artifact_for_entries(base, (replacement,)), "replacement")
    replacement_display = render_w05_definition_answer(
        _raw_answer(replacement_runtime, replacement))
    assert replacement_display.status == "DISPLAY"
    assert replacement_display.citations[0].source_ref == replacement_source

    old = replace(definition, entry_key=(29, 1), active=0)
    successor = replace(
        definition,
        entry_key=(29, 2),
        observation_key=(29, 3),
        source_ref=replacement_source,
        supersedes_entry_keys=(old.entry_key,),
    )
    successor_runtime = _runtime_for_artifact(
        _artifact_for_entries(base, (old, successor)), "successor")
    successor_display = render_w05_definition_answer(
        _raw_answer(successor_runtime, successor))
    assert successor_display.status == "DISPLAY"
    assert successor_display.citations[0].proposition_key == (
        successor_runtime.propositions[1].proposition_key)

    duplicate_source = replace(
        replacement_source,
        stable_key=(29, 7),
        revision_id=replacement_source.revision_id + "-second",
        source_identity=replacement_source.source_identity + "#second",
        source_commitment_sha256="e" * 64,
    )
    duplicate = replace(
        definition,
        entry_key=(29, 4),
        observation_key=(29, 5),
        sense_key=(29, 6),
        source_ref=duplicate_source,
    )
    duplicate_runtime = _runtime_for_artifact(
        _artifact_for_entries(base, (definition, duplicate)), "duplicate")
    duplicate_display = render_w05_definition_answer(
        _raw_answer(duplicate_runtime, definition))
    assert duplicate_display.source_answer.status == "ANSWER"
    assert len(duplicate_display.source_answer.selected_propositions) == 2
    assert duplicate_display.status == "SOURCE_NOT_UNIQUE"
    assert duplicate_display.raw_source_text == definition.definition_text
    assert duplicate_display.display_text is None
    assert len(duplicate_display.citations) == 2


def test_display_and_citation_tampering_are_rejected(
        sense_runtime,
        proposition_runtime,
        ) -> None:
    answer = _raw_answer(
        proposition_runtime, _astronomy_definition(sense_runtime))
    result = render_w05_definition_answer(answer)
    with pytest.raises(
            W05DefinitionRenderingError,
            match="does not match its source"):
        replace(result, display_text=result.display_text + "x")
    with pytest.raises(
            W05DefinitionRenderingError,
            match="commitment drifted"):
        replace(result, display_projection_sha256="0" * 64)
    with pytest.raises(
            W05DefinitionRenderingError,
            match="citation commitment drifted"):
        replace(result.citations[0], raw_source_sha256="0" * 64)


def test_display_cli_is_opt_in_and_raw_definition_bytes_do_not_change(
        sense_runtime,
        proposition_runtime,
        ) -> None:
    definition = _astronomy_definition(sense_runtime)
    request = W05RawDefinitionRequest(
        "什么是" + definition.surface,
        definition.definition_text,
        definition.language,
    )
    raw = answer_w05_raw_definition_question(proposition_runtime, request)
    raw_output = StringIO()
    assert main([
        request.question_surface,
        "--context", definition.definition_text,
        "--definition",
    ], stdout=raw_output) == 0
    assert raw_output.getvalue() == json.dumps(
        raw.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"

    display_output = StringIO()
    assert main([
        request.question_surface,
        "--context", definition.definition_text,
        "--display-definition",
    ], stdout=display_output) == 0
    display = json.loads(display_output.getvalue())
    assert display["status"] == "DISPLAY"
    assert display["raw_source_text"] == definition.definition_text
    assert display["display_text"] == (
        "（astronomy） 太陽系的第二顆行星，為類地行星")
    assert display["citations"][0]["source_ref"]["license_id"] == (
        "CC-BY-SA-4.0")


def test_ft29_code_contains_no_selected_term_qid_or_answer_dispatch() -> None:
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    files = (
        "src/pure_integer_ai/experiments/ph2_mediawiki_inline_ast.py",
        "src/pure_integer_ai/experiments/"
        "ph2_w05_definition_rendering_contract.py",
        "src/pure_integer_ai/experiments/ph2_w05_definition_rendering.py",
    )
    combined = "\n".join(
        (repository / item).read_text(encoding="utf-8") for item in files)
    for forbidden in (
            "首页", "首頁", "苹果", "蘋果", "金星",
            '"Q313"', '"Q5113"', "距离太阳第二近的行星"):
        assert forbidden not in combined
