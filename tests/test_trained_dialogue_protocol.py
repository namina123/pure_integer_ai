import io
import json

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    DialogueCitation,
    DialogueTurn,
)
from pure_integer_ai.experiments.run_trained_dialogue_terminal import (
    _nearest_rank_latency_us,
    _protocol_turn_payload,
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
    }]


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
