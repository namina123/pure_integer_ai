import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    DialogueCitation,
    DialogueTurn,
)
from pure_integer_ai.experiments.run_trained_dialogue_terminal import (
    _nearest_rank_latency_us,
    _protocol_turn_payload,
    _public_protocol_turn_payload,
    _release_bound_artifact_root,
    _write_protocol_payload,
)
from pure_integer_ai.experiments import run_dialogue_protocol


def test_protocol_turn_payload_preserves_status_and_citations():
    turn = DialogueTurn(
        ordinal=3,
        question="问题",
        answer="答案",
        display_answer="答案",
        status="ANSWER",
        source_title="来源",
        source_url="https://example.invalid/source",
        turn_key=tuple(range(32)),
        retrieval_question="检索问题",
        citations=(DialogueCitation("证据", "来源", "https://example.invalid/source"),),
    )
    payload = _protocol_turn_payload(turn)
    assert payload["type"] == "response"
    assert payload["status"] == "ANSWER"
    assert payload["turn_key"] == list(range(32))
    assert payload["citations"] == [{
        "surface": "证据",
        "source_title": "来源",
        "source_url": "https://example.invalid/source",
        "license_id": None,
        "attribution": None,
        "source_ref": None,
    }]


def test_public_protocol_turn_payload_exposes_language_not_internal_status():
    turn = DialogueTurn(
        ordinal=4,
        question="未覆盖的问题？",
        answer=None,
        display_answer=None,
        status="UNKNOWN",
        source_title=None,
        source_url=None,
        turn_key=tuple(range(32)),
    )
    payload = _public_protocol_turn_payload(turn)
    assert payload["type"] == "response"
    assert payload["text"] == "当前公开资料无法确认这个问题。"
    assert "UNKNOWN" not in payload
    assert "CLARIFY" not in payload
    assert "status" not in payload
    assert "answer" not in payload
    assert "display_answer" not in payload
    assert "retrieval_question" not in payload


def test_public_protocol_turn_payload_keeps_answer_language_and_source():
    turn = DialogueTurn(
        ordinal=5,
        question="问题",
        answer="答案",
        display_answer="答案",
        status="ANSWER",
        source_title="来源",
        source_url="https://example.invalid/source",
        turn_key=tuple(range(32)),
    )
    payload = _public_protocol_turn_payload(turn)
    assert payload["text"] == "答案\n来源：来源（https://example.invalid/source）"
    assert payload["source"] == {
        "title": "来源", "url": "https://example.invalid/source",
    }
    assert "status" not in payload


def test_public_protocol_turn_payload_prefers_organized_surface_but_keeps_citations():
    citation = DialogueCitation(
        "原始证据句。", "资料来源", "https://example.invalid/source")
    turn = DialogueTurn(
        ordinal=7,
        question="问题",
        answer="原始证据句。",
        display_answer="组织后的资料句。",
        status="ANSWER",
        source_title="资料来源",
        source_url="https://example.invalid/source",
        turn_key=tuple(range(32)),
        citations=(citation,),
    )
    payload = _public_protocol_turn_payload(turn)
    assert payload["text"] == (
        "组织后的资料句。\n\n来源：资料来源（https://example.invalid/source）")
    assert payload["citations"][0]["surface"] == "原始证据句。"


def test_public_protocol_turn_payload_projects_clarification_as_language():
    turn = DialogueTurn(
        ordinal=6,
        question="有歧义的问题？",
        answer=None,
        display_answer=None,
        status="CLARIFY",
        source_title=None,
        source_url=None,
        turn_key=tuple(range(32)),
    )
    payload = _public_protocol_turn_payload(turn)
    assert payload["text"] == "请补充问题的范围或限定条件。"
    assert "status" not in payload
    assert all(value not in payload["text"] for value in ("UNKNOWN", "CLARIFY"))


def test_protocol_writer_is_utf8_jsonl_and_sorted():
    output = io.BytesIO()
    _write_protocol_payload(output, {"z": "末", "a": 1})
    raw = output.getvalue()
    assert raw.startswith(b"{\"a\":1,\"z\":")
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == {"a": 1, "z": "末"}


def test_latency_percentile_uses_conservative_integer_nearest_rank():
    values = [366, 366, 1785, 1955, 2481, 4024, 9076]
    assert _nearest_rank_latency_us(values, 50) == 1955
    assert _nearest_rank_latency_us(values, 95) == 9076
    assert _nearest_rank_latency_us(values, 100) == 9076


def test_dedicated_protocol_entrypoint_forces_jsonl(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 17

    monkeypatch.setattr(run_dialogue_protocol, "_terminal_main", fake_main)
    assert run_dialogue_protocol.main(["--qa-database", "db"]) == 17
    assert captured["argv"][-2:] == ["--protocol", "jsonl"]


def test_release_embedded_artifact_is_automatic_and_conflict_closed(
        tmp_path: Path) -> None:
    embedded = (tmp_path / "embedded").resolve()
    embedded.mkdir()
    release = SimpleNamespace(dialogue_response_artifact=embedded)
    assert _release_bound_artifact_root(
        release, None, attribute="dialogue_response_artifact",
        label="dialogue_response_artifact_root") == embedded
    assert _release_bound_artifact_root(
        release, embedded, attribute="dialogue_response_artifact",
        label="dialogue_response_artifact_root") == embedded
    with pytest.raises(ValueError, match="不得冲突"):
        _release_bound_artifact_root(
            release, tmp_path / "external",
            attribute="dialogue_response_artifact",
            label="dialogue_response_artifact_root")
