"""FT22 reusable sparse runtime, warm probe, and CLI coverage."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    build_raw_question_alias_frame_anchor_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    build_raw_question_construction_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch import (
    build_raw_question_sparse_dispatch_index,
    project_sparse_question_dispatch_audit,
    run_sparse_question_dispatch,
    run_sparse_question_dispatch_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    SPARSE_QA_RUNTIME_SHA256,
    assemble_sparse_qa_runtime,
    build_public_sparse_qa_runtime,
    run_sparse_qa_queries,
    run_sparse_qa_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_session import (
    build_sparse_qa_session_probe,
    iter_sparse_qa_jsonl_session,
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    return build_public_sparse_qa_runtime(
        tmp_path_factory.mktemp("ft22_public_runtime"))


def _learned_alias_surface(entry, construction) -> str:
    predicate_ordinal = next(
        ordinal for ordinal, segment in enumerate(construction.segments)
        if segment.kind == "PREDICATE"
    )
    proposition_key = construction.vertical_result.link.proposition_key
    alias = next(
        route.alias_surface for route in entry.alias_bridge.routes
        if route.proposition_key == proposition_key
    )
    return "".join(
        alias if ordinal == predicate_ordinal else segment.surface
        for ordinal, segment in enumerate(construction.segments)
    )


def _build_dispatch(registry):
    return build_raw_question_sparse_dispatch_index(
        build_raw_question_alias_frame_anchor_index(
            build_raw_question_construction_index(
                build_raw_question_feature_index(registry))))


def test_public_runtime_build_is_frozen_and_counted_once(runtime) -> None:
    assert runtime.identity_sha256 == runtime.sha256() == (
        SPARSE_QA_RUNTIME_SHA256)
    assert runtime.build_probe.to_dict() == {
        "alias_frame_anchor_build_count": 1,
        "alias_frame_count": 8,
        "alias_index_row_count": 8,
        "construction_index_build_count": 1,
        "dispatch_entry_count": 2,
        "exact_index_row_count": 8,
        "explicit_construction_count": 8,
        "feature_index_build_count": 1,
        "implicit_construction_count": 8,
        "implicit_index_row_count": 8,
        "learned_alias_route_count": 4,
        "registry_build_count": 1,
        "registry_entry_count": 2,
        "runtime_build_count": 1,
        "sparse_dispatch_build_count": 1,
    }
    assert runtime.frozen_identities.to_dict() == {
        "alias_frame_anchor_sha256": (
            "af4dcb9afb8ee0a96ff5883e057afed9527fd118e21bd5f93dea8cae9583dad2"),
        "construction_index_sha256": (
            "0d4495b612e5d6e54fa47f1066ad48543fd5bbb00ffc08d04a67de2fc2067ab8"),
        "feature_index_sha256": (
            "908786f4da77414a5c7728d01f8c2c1528c23f4787a1efcc068b166f70be095b"),
        "registry_sha256": (
            "5b1067d49baaa1465cce48d120717e63e49eea291e1d92dc13d3bb8df6134244"),
        "sparse_dispatch_sha256": (
            "0ba334b34b0863d588103460a800fa3b2de0256fe788c95ec622bef325d3066c"),
    }
    assert tuple(
        item.public_state_sha256
        for item in runtime.entry_public_state_memo
    ) == (
        "0ef1ebb330a327ee815411615573550be9917fc216fc7b33cfc31cbe1e12a76b",
        "e75bfb439cad7c9fd2dc4375b86ce12f5a1fc8bb308d149cb40e3f0ce9cf6937",
    )


def test_all_learned_query_identities_remain_frozen(runtime) -> None:
    memo = tuple(
        (item.entry_sha256, item.public_state_sha256)
        for item in runtime.entry_public_state_memo
    )
    requests = []
    for row in runtime.dispatch_index.entries:
        entry = row.entry
        requests.extend(
            ("EXACT", RawQuestionRequest(item.question_surface))
            for item in entry.feature_catalog.catalog
        )
        requests.extend(
            ("ALIAS", RawQuestionRequest(_learned_alias_surface(entry, item)))
            for item in entry.feature_catalog.catalog
        )
        requests.extend(
            ("IMPLICIT", RawQuestionRequest(item.question_surface))
            for item in entry.implicit_bundle.catalog
        )
    assert len(requests) == 24
    values = []
    for phase, request in requests:
        direct_record = run_sparse_question_dispatch(
            runtime.dispatch_index,
            request,
            public_state_sha256s=memo,
        )
        direct_audit = project_sparse_question_dispatch_audit(
            runtime.dispatch_index, direct_record)
        values.append({
            "audit_sha256": direct_audit.sha256(),
            "execution_record_sha256": direct_record.sha256(),
            "phase": phase,
            "request_sha256": request.sha256(),
        })
    assert hashlib.sha256(canonical_json_bytes(values)).hexdigest() == (
        "2857eec09479acf33b6fb23fe0129b63735a5ba15b053babebe27abc8ac3b000")


def test_exact_alias_implicit_unknown_and_resolved_source(runtime) -> None:
    entry = runtime.dispatch_index.entries[0].entry
    exact = entry.feature_catalog.catalog[0]
    implicit = entry.implicit_bundle.catalog[0]
    cases = (
        (
            RawQuestionRequest(exact.question_surface),
            "EXACT",
            exact.source_record_key,
        ),
        (
            RawQuestionRequest(_learned_alias_surface(entry, exact)),
            "ALIAS",
            exact.source_record_key,
        ),
        (
            RawQuestionRequest(implicit.question_surface),
            "IMPLICIT",
            implicit.source_record_key,
        ),
    )
    for request, phase, source in cases:
        result = run_sparse_qa_query(runtime, request)
        assert result.status == "ANSWER"
        assert result.answer_surface
        assert result.decisive_phase == phase
        assert result.selected_source_record_key == source
        assert result.audit_result is None
        assert "audit" not in result.to_dict()
        assert result.dispatch_probe.sparse_trace_count < (
            result.dispatch_probe.projected_trace_count)

    unknown = run_sparse_qa_query(
        runtime, RawQuestionRequest("未学习的公开问题？"))
    assert unknown.status == "UNKNOWN"
    assert unknown.answer_surface is None
    assert unknown.selected_source_record_key is None
    assert unknown.dispatch_probe.sparse_trace_count == 0
    assert unknown.dispatch_probe.created_dispatch_trace_count == 0


def test_audit_is_explicit_and_ft16_byte_identical(runtime) -> None:
    construction = (
        runtime.dispatch_index.entries[0].entry.feature_catalog.catalog[0])
    request = RawQuestionRequest(
        construction.question_surface,
        construction.source_record_key,
    )
    short = run_sparse_qa_query(runtime, request)
    audited = run_sparse_qa_query(runtime, request, audit=True)
    expected = run_sparse_question_dispatch_registry_answer(
        runtime.dispatch_index, request)
    assert short.audit_result is None
    assert audited.audit_result is not None
    assert audited.audit_result.to_dict() == expected.to_dict()
    assert audited.audit_result.sha256() == expected.sha256()
    assert "audit" not in short.to_dict()
    assert len(audited.to_dict()["audit"]["traces"]) == 2


def test_default_query_never_projects_complete_audit(
        runtime, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_sparse_qa_runtime as query_runtime,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("default sparse query projected FT16 audit")

    monkeypatch.setattr(
        query_runtime,
        "project_sparse_question_dispatch_audit",
        forbidden,
    )
    result = query_runtime.run_sparse_qa_query(
        runtime,
        RawQuestionRequest("未学习的公开问题？"),
    )
    assert result.status == "UNKNOWN"


def test_warm_repeated_queries_are_bit_identical_without_rebuild(runtime) -> None:
    construction = (
        runtime.dispatch_index.entries[0].entry.feature_catalog.catalog[0])
    request = RawQuestionRequest(construction.question_surface)
    batch = run_sparse_qa_queries(runtime, (request,) * 5)
    assert batch.probe.runtime_build_count == 1
    assert batch.probe.query_count == 5
    assert batch.probe.execution_record_count == 5
    assert batch.probe.result_object_count == 5
    assert batch.probe.audit_projection_count == 0
    assert batch.probe.created_audit_trace_count == 0
    assert batch.probe.bit_identical is True
    assert len(set(batch.probe.result_sha256s)) == 1


def test_conflict_remains_clarify_in_reusable_runtime(runtime) -> None:
    entries = tuple(item.entry for item in runtime.dispatch_index.entries)
    two_entry, three_entry = entries
    shared = two_entry.feature_catalog.catalog[0]
    collision_pattern = replace(
        shared.pattern,
        construction_id=f"{shared.pattern.construction_id}_FT22_COLLISION",
    )
    collision = replace(shared, pattern=collision_pattern)
    collision_catalog = replace(
        two_entry.feature_catalog,
        patterns=tuple(sorted(
            (*two_entry.feature_catalog.patterns, collision_pattern),
            key=lambda item: item.sha256(),
        )),
        catalog=tuple(sorted(
            (shared, collision), key=lambda item: item.sha256())),
    )
    collision_entry = RawQuestionFeatureRegistryEntry(
        collision_catalog,
        two_entry.alias_bridge,
        replace(
            two_entry.implicit_bundle,
            explicit_catalog=collision_catalog,
        ),
    )
    conflict_runtime = assemble_sparse_qa_runtime(_build_dispatch(
        build_raw_question_feature_registry(
            (collision_entry, three_entry))))
    result = run_sparse_qa_query(conflict_runtime, RawQuestionRequest(
        shared.question_surface,
        shared.source_record_key,
    ))
    assert result.status == "CLARIFY"
    assert result.answer_surface is None
    assert result.decisive_phase == "EXACT"
    assert result.selected_source_record_key is None


def test_cli_uses_one_runtime_and_hides_audit_by_default(
        runtime, monkeypatch, capsys) -> None:
    from pure_integer_ai.experiments import (
        run_ph2_w03_w04_w05_sparse_qa as cli,
    )

    monkeypatch.setattr(
        cli,
        "load_or_rebuild_public_sparse_qa_runtime",
        lambda: runtime,
    )
    construction = (
        runtime.dispatch_index.entries[0].entry.feature_catalog.catalog[0])
    assert cli.main([construction.question_surface, "--repeat", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "ANSWER"
    assert payload["result"]["selected_source_record_key"] == list(
        construction.source_record_key)
    assert "audit" not in payload["result"]
    assert payload["probe"]["query_count"] == 3
    assert payload["probe"]["runtime_build_count"] == 1
    assert payload["probe"]["bit_identical"] is True

    source = ",".join(str(item) for item in construction.source_record_key)
    assert cli.main([
        construction.question_surface,
        "--source-ref",
        source,
        "--audit",
    ]) == 0
    audited = json.loads(capsys.readouterr().out)
    assert len(audited["result"]["audit"]["traces"]) == 2
    assert audited["probe"]["audit_projection_count"] == 1
    assert audited["probe"]["created_audit_trace_count"] == 2


def test_jsonl_session_orders_isolates_and_reuses_one_runtime(runtime) -> None:
    entry = runtime.dispatch_index.entries[0].entry
    exact = entry.feature_catalog.catalog[0]
    implicit = entry.implicit_bundle.catalog[0]
    lines = (
        "\ufeff" + json.dumps(
            {"question": exact.question_surface}, ensure_ascii=False),
        "{not-json}",
        json.dumps({
            "question": implicit.question_surface,
            "source_ref": list(implicit.source_record_key),
            "audit": True,
        }, ensure_ascii=False),
        json.dumps({"question": exact.question_surface}, ensure_ascii=False),
        json.dumps({"question": "未学习的公开问题？"}, ensure_ascii=False),
    )
    records = tuple(iter_sparse_qa_jsonl_session(runtime, lines))
    assert tuple(item.line_ordinal for item in records) == tuple(range(5))
    assert tuple(item.kind for item in records) == (
        "RESULT", "ERROR", "RESULT", "RESULT", "RESULT")
    assert records[1].error_code == "INVALID_JSON"
    assert records[2].result is not None
    assert records[2].result.audit_result is not None
    assert records[4].result is not None
    assert records[4].result.status == "UNKNOWN"
    assert records[0].result is not None and records[3].result is not None
    assert records[0].result.sha256() == records[3].result.sha256()

    probe = build_sparse_qa_session_probe(runtime, records)
    assert probe.runtime_build_count == 1
    assert probe.input_line_count == 5
    assert probe.query_count == 4
    assert probe.error_count == 1
    assert probe.audit_projection_count == 1
    assert probe.created_audit_trace_count == 2
    assert len(probe.record_commitment_sha256) == 64
    assert len(probe.result_commitment_sha256) == 64


def test_jsonl_session_rejects_bad_lines_without_querying(runtime) -> None:
    lines = (
        "",
        "[]",
        json.dumps({"question": "合法？", "extra": 1}, ensure_ascii=False),
        json.dumps({"question": " 非规范？"}, ensure_ascii=False),
        json.dumps({"question": "合法？", "source_ref": [True]}),
        json.dumps({"question": "合法？", "audit": 1}),
    )
    records = tuple(iter_sparse_qa_jsonl_session(runtime, lines))
    assert tuple(item.error_code for item in records) == (
        "EMPTY_LINE",
        "INVALID_FIELDS",
        "INVALID_FIELDS",
        "INVALID_QUESTION",
        "INVALID_SOURCE_REF",
        "INVALID_AUDIT",
    )
    probe = build_sparse_qa_session_probe(runtime, records)
    assert probe.query_count == 0
    assert probe.error_count == len(lines)
    assert probe.audit_projection_count == 0
    assert probe.created_sparse_trace_count == 0


def test_jsonl_cli_builds_once_and_emits_final_probe(
        runtime, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        run_ph2_w03_w04_w05_sparse_qa as cli,
    )

    calls = {"build": 0}

    def build_once():
        calls["build"] += 1
        return runtime

    monkeypatch.setattr(
        cli, "load_or_rebuild_public_sparse_qa_runtime", build_once)
    construction = (
        runtime.dispatch_index.entries[0].entry.feature_catalog.catalog[0])
    input_stream = StringIO("\n".join((
        json.dumps(
            {"question": construction.question_surface},
            ensure_ascii=False,
        ),
        "not-json",
        json.dumps(
            {"question": construction.question_surface},
            ensure_ascii=False,
        ),
    )))
    output_stream = StringIO()
    assert cli.main(
        ["--jsonl"], stdin=input_stream, stdout=output_stream) == 0
    values = tuple(
        json.loads(item) for item in output_stream.getvalue().splitlines())
    assert calls == {"build": 1}
    assert tuple(item["kind"] for item in values[:-1]) == (
        "RESULT", "ERROR", "RESULT")
    assert values[-1]["kind"] == "SESSION_PROBE"
    assert values[-1]["probe"]["runtime_build_count"] == 1
    assert values[-1]["probe"]["query_count"] == 2
    assert values[-1]["probe"]["error_count"] == 1
    assert values[0]["result_sha256"] == values[2]["result_sha256"]
    assert len(values[-1]["probe"]["record_commitment_sha256"]) == 64


def test_runtime_contains_no_question_or_answer_dispatch_table() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/pure_integer_ai/experiments/"
            "ph2_w03_w04_w05_sparse_qa_runtime.py",
            "src/pure_integer_ai/experiments/"
            "run_ph2_w03_w04_w05_sparse_qa.py",
            "src/pure_integer_ai/experiments/"
            "ph2_w03_w04_w05_sparse_qa_session.py",
        )
    )
    for value in (
            "暴雨", "寒潮", "山区", "桥面", "河水上涨", "路面结冰"):
        assert value not in sources
    assert "if request.question_surface" not in sources
    assert "bundle_name" not in sources
