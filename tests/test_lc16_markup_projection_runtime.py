"""LC-16 Markdown/HTML 共享投影纵切测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    GLOBAL_OWNER_SCOPE,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_html_carrier_adapter import (
    adapt_html_carrier_record,
)
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    read_html_carrier_records,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_adapter import (
    adapt_markdown_carrier_record,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_contract import (
    read_markdown_carrier_records,
)
from pure_integer_ai.experiments.ph2_markup_projection_runtime import (
    MarkupProjectionRuntime,
    MarkupProjectionSpec,
)
from pure_integer_ai.storage.backend import DictBackend


_ROOT = Path(__file__).resolve().parents[1]


def _source(source_id: int) -> SourceRef:
    return SourceRef(
        16617150, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


def _node_receipt(node) -> dict:
    return parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)


def _find_node(materialization, *, name: str | None = None,
               node_type: str | None = None):
    for index, node in enumerate(materialization.structure_nodes):
        receipt = _node_receipt(node)
        actual_name = receipt.get("name", receipt.get("tag", ""))
        actual_type = receipt.get("node_type", receipt.get("type", ""))
        if (name is None or actual_name == name) and (
                node_type is None or actual_type == node_type):
            return index, node
    raise AssertionError(f"未找到结构节点 name={name!r} type={node_type!r}")


def _html_custom_record(records, index: int, text: str):
    base = records[index]
    return replace(
        base,
        raw_text=text,
        raw_unit_count=len(text),
        raw_utf8_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _markdown_custom_record(records, index: int, text: str):
    base = records[index]
    return replace(
        base,
        raw_text=text,
        raw_unit_count=len(text),
        raw_utf8_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _reveal(materialization, event_key: tuple[int, ...], target,
            *, support: bool) -> RevealedObjectObservation:
    source = materialization.sources[0]
    return RevealedObjectObservation(
        source,
        materialization.scopes[0],
        event_key,
        _source(900 + event_key[-1]),
        (target,) if support else (),
        () if support else (target,),
        (event_key[-1], int(support)),
    )


def _fixtures():
    markdown_records = read_markdown_carrier_records(
        _ROOT / "data/ph2/lc16_markdown_carrier_v1.jsonl.sample")
    html_records = read_html_carrier_records(
        _ROOT / "data/ph2/lc16_html_carrier_v1.jsonl.sample")
    forming_markdown = adapt_markdown_carrier_record(markdown_records[0])
    forming_html = adapt_html_carrier_record(
        _html_custom_record(html_records, 0, "<strong>范围</strong>"))
    heldout_markdown = adapt_markdown_carrier_record(markdown_records[2])
    heldout_html = adapt_html_carrier_record(
        _html_custom_record(html_records, 5, "<strong>范围</strong>"))
    correction_markdown = adapt_markdown_carrier_record(
        _markdown_custom_record(markdown_records, 5, "**范围**\n"))
    return (
        forming_markdown,
        forming_html,
        heldout_markdown,
        heldout_html,
        correction_markdown,
    )


def _spec(markdown, html, *, candidate_key: int, semantic_key: int,
          directions=(PROJECTION_UNDERSTANDING,
                       PROJECTION_REASONING,
                       PROJECTION_GENERATION)):
    _, markdown_node = _find_node(markdown, name="strong", node_type="strong_open")
    _, html_node = _find_node(html, name="strong")
    return MarkupProjectionSpec(
        candidate=concept_identity((16617200, candidate_key)),
        competition_key=(16617201, 1),
        projection_kind=concept_identity((16617202, 1)),
        semantic_object=concept_identity((16617203, semantic_key)),
        directions=directions,
        structure_features=(markdown_node.node_kind, html_node.node_kind),
        forming_sources=(markdown.sources[0], html.sources[0]),
    )


def test_markdown_and_html_share_one_candidate_projection_and_three_consumers():
    forming_markdown, forming_html, markdown, html, _ = _fixtures()
    spec = _spec(
        forming_markdown, forming_html, candidate_key=1, semantic_key=1)
    backend = DictBackend()
    try:
        runtime = MarkupProjectionRuntime(backend)
        md_index, _ = _find_node(markdown, name="strong", node_type="strong_open")
        html_index, _ = _find_node(html, name="strong")
        md_trace = runtime.learn(
            spec,
            markdown,
            node_indices=(md_index,),
            event_key=(16617300, 1),
            revealed=_reveal(markdown, (16617300, 1),
                             spec.semantic_object, support=True),
        )
        html_trace = runtime.learn(
            spec,
            html,
            node_indices=(html_index,),
            event_key=(16617300, 2),
            revealed=_reveal(html, (16617300, 2),
                             spec.semantic_object, support=True),
        )
        assert md_trace.carrier_projection is not None
        assert html_trace.carrier_projection is not None
        assert md_trace.carrier_projection.semantic_object == (
            html_trace.carrier_projection.semantic_object)
        assert len(md_trace.uses) == len(html_trace.uses) == 3
        assert all(item.accepted for item in md_trace.uses + html_trace.uses)
        assert all(item.projection.identity == md_trace.carrier_projection.identity
                   for item in md_trace.uses)
        assert md_trace.uses[0].output == spec.semantic_object
        assert md_trace.uses[1].output == (
            md_trace.carrier_projection.hypothesis.object_identity())
        assert md_trace.uses[2].output == (
            md_trace.carrier_projection.envelope_identity)
        assert md_trace.carrier_projection.from_stable_key(
            md_trace.carrier_projection.stable_key()) == md_trace.carrier_projection
        runtime.graph.ontology.clear_runtime_caches()
        retained = runtime.retained_projection(spec.candidate)
        assert retained.state == runtime.protocol.lifecycle.active_state
        assert retained.candidate.definition.candidate == spec.candidate
    finally:
        backend.close()


def test_forming_sources_cannot_be_reused_as_recognition_evidence():
    forming_markdown, forming_html, _, _, _ = _fixtures()
    spec = _spec(
        forming_markdown, forming_html, candidate_key=30, semantic_key=30)
    backend = DictBackend()
    try:
        runtime = MarkupProjectionRuntime(backend)
        markdown_index, _ = _find_node(
            forming_markdown, name="strong", node_type="strong_open")
        with pytest.raises(EvidenceCandidateError, match="forming observation"):
            runtime.learn(
                spec,
                forming_markdown,
                node_indices=(markdown_index,),
                event_key=(16617300, 30),
                revealed=_reveal(
                    forming_markdown,
                    (16617300, 30),
                    spec.semantic_object,
                    support=True,
                ),
            )
    finally:
        backend.close()


def test_unknown_custom_structure_enters_same_data_only_lifecycle():
    markdown_records = read_markdown_carrier_records(
        _ROOT / "data/ph2/lc16_markdown_carrier_v1.jsonl.sample")
    html_records = read_html_carrier_records(
        _ROOT / "data/ph2/lc16_html_carrier_v1.jsonl.sample")
    forming_markdown = adapt_markdown_carrier_record(markdown_records[3])
    forming_html = adapt_html_carrier_record(_html_custom_record(
        html_records, 3,
        '<future-panel data-mode="new"><slot-x>内容</slot-x></future-panel>',
    ))
    heldout_html = adapt_html_carrier_record(_html_custom_record(
        html_records, 5,
        '<future-panel data-extra="held-out">新内容</future-panel>',
    ))
    md_index, md_node = _find_node(forming_markdown, node_type="inline")
    _, html_node = _find_node(forming_html, name="future-panel")
    heldout_index, _ = _find_node(heldout_html, name="future-panel")
    spec = MarkupProjectionSpec(
        concept_identity((16617200, 2)),
        (16617201, 2),
        concept_identity((16617202, 2)),
        concept_identity((16617203, 2)),
        (PROJECTION_UNDERSTANDING,),
        (md_node.node_kind, html_node.node_kind),
        (forming_markdown.sources[0], forming_html.sources[0]),
    )
    backend = DictBackend()
    try:
        runtime = MarkupProjectionRuntime(backend)
        trace = runtime.learn(
            spec,
            heldout_html,
            node_indices=(heldout_index,),
            event_key=(16617300, 3),
            revealed=_reveal(heldout_html, (16617300, 3),
                             spec.semantic_object, support=True),
        )
        assert trace.carrier_projection is not None
        assert len(trace.uses) == 1
        assert trace.uses[0].direction == PROJECTION_UNDERSTANDING
        assert _node_receipt(
            heldout_html.structure_nodes[heldout_index])["name"] == (
            "future-panel")
        assert _node_receipt(
            forming_markdown.structure_nodes[md_index])["type"] == "inline"
    finally:
        backend.close()


def test_candidate_correction_supersedes_projection_and_stops_consumption():
    forming_markdown, forming_html, markdown, html, correction = _fixtures()
    first = _spec(
        forming_markdown, forming_html, candidate_key=10, semantic_key=10)
    second = _spec(
        forming_markdown, forming_html, candidate_key=11, semantic_key=11)
    backend = DictBackend()
    try:
        runtime = MarkupProjectionRuntime(backend)
        runtime.register(first, timestamp_base=1)
        runtime.register(second, timestamp_base=2)
        md_index, _ = _find_node(markdown, name="strong", node_type="strong_open")
        html_index, _ = _find_node(html, name="strong")
        accepted = runtime.learn(
            first,
            markdown,
            node_indices=(md_index,),
            event_key=(16617300, 10),
            revealed=_reveal(markdown, (16617300, 10),
                             first.semantic_object, support=True),
        )
        assert accepted.carrier_projection is not None
        replacement = runtime.learn(
            second,
            html,
            node_indices=(html_index,),
            event_key=(16617300, 11),
            revealed=_reveal(html, (16617300, 11),
                             second.semantic_object, support=True),
        )
        assert replacement.carrier_projection is not None
        assert runtime.retained_projection(first.candidate).state == (
            runtime.protocol.lifecycle.active_state)
        assert runtime.retained_projection(second.candidate).state == (
            runtime.protocol.lifecycle.active_state)
        failed_outcome = runtime.record_outcome(
            accepted.uses[0],
            accepted=False,
            source=_source(950),
            outcome_key=(16617400, 1),
            trace=(16617400, 2),
        )
        assert failed_outcome.stable_key()
        corrected = runtime.learn(
            first,
            correction,
            node_indices=(_find_node(
                correction, name="strong", node_type="strong_open")[0],),
            event_key=(16617300, 12),
            revealed=runtime.reveal_from_outcome(
                correction,
                event_key=(16617300, 12),
                outcome=failed_outcome,
            ),
            replacement_candidate=second.candidate,
        )
        assert corrected.outcome.verification.stance == EVIDENCE_REFUTE
        assert corrected.carrier_projection is not None
        assert corrected.carrier_projection.lifecycle_state == (
            runtime.protocol.lifecycle.superseded_state)
        assert corrected.uses == ()
    finally:
        backend.close()


def test_directional_consumer_ablation_is_explicit():
    forming_markdown, forming_html, _, html, _ = _fixtures()
    for offset, direction in enumerate((
            PROJECTION_UNDERSTANDING,
            PROJECTION_REASONING,
            PROJECTION_GENERATION,
            ), start=20):
        spec = _spec(
            forming_markdown,
            forming_html,
            candidate_key=offset,
            semantic_key=offset,
            directions=(direction,),
        )
        backend = DictBackend()
        try:
            runtime = MarkupProjectionRuntime(backend)
            html_index, _ = _find_node(html, name="strong")
            trace = runtime.learn(
                spec,
                html,
                node_indices=(html_index,),
                event_key=(16617300, offset),
                revealed=_reveal(
                    html,
                    (16617300, offset),
                    spec.semantic_object,
                    support=True,
                ),
            )
            assert trace.carrier_projection is not None
            assert tuple(item.direction for item in trace.uses) == (direction,)
            assert trace.uses[0].accepted
        finally:
            backend.close()
