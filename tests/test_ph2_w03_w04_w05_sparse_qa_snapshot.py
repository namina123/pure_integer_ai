"""FT24B typed canonical sparse QA snapshot coverage."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch import (
    project_sparse_question_dispatch_audit,
    run_sparse_question_dispatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    SPARSE_QA_RUNTIME_SHA256,
    run_sparse_qa_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_session import (
    build_sparse_qa_session_probe,
    iter_sparse_qa_jsonl_session,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT,
    PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
    PUBLIC_SPARSE_QA_SOURCE_ARTIFACT_SHA256S,
    SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256,
    SparseQARuntimeSnapshotError,
    load_or_rebuild_public_sparse_qa_runtime,
    load_public_sparse_qa_runtime_snapshot,
    write_public_sparse_qa_runtime_snapshot,
)


@pytest.fixture(scope="module")
def snapshot_value():
    raw = PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT.read_bytes()
    return json.loads(raw[:-1].decode("utf-8"))


@pytest.fixture(scope="module")
def runtime():
    return load_public_sparse_qa_runtime_snapshot()


def _write(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return path


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


def test_bundled_snapshot_loads_complete_frozen_runtime(runtime) -> None:
    assert PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT.is_file()
    assert runtime.identity_sha256 == runtime.sha256() == (
        SPARSE_QA_RUNTIME_SHA256)
    assert runtime.frozen_identities.registry_sha256 == (
        "5b1067d49baaa1465cce48d120717e63e49eea291e1d92dc13d3bb8df6134244")
    assert runtime.frozen_identities.feature_index_sha256 == (
        "908786f4da77414a5c7728d01f8c2c1528c23f4787a1efcc068b166f70be095b")
    assert runtime.frozen_identities.construction_index_sha256 == (
        "0d4495b612e5d6e54fa47f1066ad48543fd5bbb00ffc08d04a67de2fc2067ab8")
    assert runtime.frozen_identities.alias_frame_anchor_sha256 == (
        "af4dcb9afb8ee0a96ff5883e057afed9527fd118e21bd5f93dea8cae9583dad2")
    assert runtime.frozen_identities.sparse_dispatch_sha256 == (
        "0ba334b34b0863d588103460a800fa3b2de0256fe788c95ec622bef325d3066c")


def test_snapshot_envelope_inventory_and_idempotent_bytes(
        runtime, snapshot_value, tmp_path) -> None:
    assert snapshot_value["schema_version"] == (
        SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION)
    assert snapshot_value["value_type_inventory_sha256"] == (
        SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256)
    assert len(PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS) == 14
    assert len(PUBLIC_SPARSE_QA_SOURCE_ARTIFACT_SHA256S) == 14
    target = write_public_sparse_qa_runtime_snapshot(
        runtime, tmp_path / "snapshot.json")
    assert target.read_bytes() == PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT.read_bytes()
    assert write_public_sparse_qa_runtime_snapshot(runtime, target) == target


def test_snapshot_preserves_24_query_and_audit_commitment(runtime) -> None:
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
    values = []
    for phase, request in requests:
        record = run_sparse_question_dispatch(
            runtime.dispatch_index,
            request,
            public_state_sha256s=memo,
        )
        audit = project_sparse_question_dispatch_audit(
            runtime.dispatch_index, record)
        values.append({
            "audit_sha256": audit.sha256(),
            "execution_record_sha256": record.sha256(),
            "phase": phase,
            "request_sha256": request.sha256(),
        })
    assert len(values) == 24
    assert hashlib.sha256(canonical_json_bytes(values)).hexdigest() == (
        "2857eec09479acf33b6fb23fe0129b63735a5ba15b053babebe27abc8ac3b000")


def test_snapshot_runtime_preserves_query_and_session_boundaries(runtime) -> None:
    entry = runtime.dispatch_index.entries[0].entry
    exact = entry.feature_catalog.catalog[0]
    implicit = entry.implicit_bundle.catalog[0]
    cases = (
        (RawQuestionRequest(exact.question_surface), "EXACT"),
        (RawQuestionRequest(_learned_alias_surface(entry, exact)), "ALIAS"),
        (RawQuestionRequest(implicit.question_surface), "IMPLICIT"),
    )
    for request, phase in cases:
        result = run_sparse_qa_query(runtime, request, audit=True)
        assert result.status == "ANSWER"
        assert result.decisive_phase == phase
        assert result.audit_result is not None
    assert run_sparse_qa_query(
        runtime, RawQuestionRequest("未学习的公开问题？")).status == "UNKNOWN"

    lines = (
        json.dumps({"question": exact.question_surface}, ensure_ascii=False),
        "not-json",
        json.dumps({"question": implicit.question_surface}, ensure_ascii=False),
    )
    records = tuple(iter_sparse_qa_jsonl_session(runtime, lines))
    assert tuple(item.kind for item in records) == (
        "RESULT", "ERROR", "RESULT")
    probe = build_sparse_qa_session_probe(runtime, records)
    assert probe.runtime_build_count == 1
    assert probe.query_count == 2
    assert probe.error_count == 1


def test_snapshot_rejects_truncation_and_noncanonical_integer(tmp_path) -> None:
    raw = PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT.read_bytes()
    truncated = tmp_path / "truncated.json"
    truncated.write_bytes(raw[:-17])
    with pytest.raises(SparseQARuntimeSnapshotError):
        load_public_sparse_qa_runtime_snapshot(truncated)

    marker = b'"schema_version":1'
    assert marker in raw
    noncanonical = tmp_path / "float-version.json"
    noncanonical.write_bytes(raw.replace(
        marker, b'"schema_version":1.0', 1))
    with pytest.raises(SparseQARuntimeSnapshotError):
        load_public_sparse_qa_runtime_snapshot(noncanonical)


@pytest.mark.parametrize("mutation", (
    "EXTRA_FIELD",
    "WRONG_VERSION",
    "WRONG_VERSION_TYPE",
    "INVALID_SHA",
    "WRONG_PARENT",
    "WRONG_SOURCE",
    "PAYLOAD_TAMPER",
    "UNKNOWN_VALUE_TYPE",
    "MISSING_STRUCT_FIELD",
))
def test_snapshot_rejects_altered_fields_types_and_identities(
        mutation, snapshot_value, tmp_path) -> None:
    value = deepcopy(snapshot_value)
    if mutation == "EXTRA_FIELD":
        value["unexpected"] = 1
    elif mutation == "WRONG_VERSION":
        value["schema_version"] += 1
    elif mutation == "WRONG_VERSION_TYPE":
        value["schema_version"] = str(value["schema_version"])
    elif mutation == "INVALID_SHA":
        value["payload_sha256"] = "0" * 63
    elif mutation == "WRONG_PARENT":
        value["frozen_parent_sha256"]["ft16_registry_sha256"] = "0" * 64
    elif mutation == "WRONG_SOURCE":
        value["source_artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "PAYLOAD_TAMPER":
        value["payload"]["nodes"][0]["type"] = "tampered.Type"
    elif mutation == "UNKNOWN_VALUE_TYPE":
        value["payload"]["nodes"][0]["type"] = "unknown.Value"
        value["payload_sha256"] = hashlib.sha256(canonical_json_bytes(
            value["payload"])).hexdigest()
    elif mutation == "MISSING_STRUCT_FIELD":
        value["payload"]["nodes"][0]["values"].pop()
        value["payload_sha256"] = hashlib.sha256(canonical_json_bytes(
            value["payload"])).hexdigest()
    else:
        raise AssertionError(mutation)
    target = _write(tmp_path / f"{mutation}.json", value)
    with pytest.raises(SparseQARuntimeSnapshotError):
        load_public_sparse_qa_runtime_snapshot(target)


def test_missing_or_invalid_snapshot_explicitly_rebuilds(
        runtime, tmp_path, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_sparse_qa_snapshot as snapshot,
    )

    calls = []

    def rebuild(work_root, *, data_root):
        calls.append((work_root, Path(data_root)))
        return runtime

    monkeypatch.setattr(snapshot, "build_public_sparse_qa_runtime", rebuild)
    work_root = tmp_path / "rebuild"
    restored = load_or_rebuild_public_sparse_qa_runtime(
        tmp_path / "missing.json", work_root=work_root)
    assert restored is runtime
    expected = (work_root, snapshot.REPOSITORY / "data/ph2")
    assert calls == [expected]

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"{}\n")
    restored = load_or_rebuild_public_sparse_qa_runtime(
        invalid, work_root=work_root)
    assert restored is runtime
    assert calls == [expected, expected]


def test_snapshot_rebuild_refuses_incomplete_distribution_resources(
        tmp_path, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_sparse_qa_snapshot as snapshot,
    )

    calls = []

    def rebuild(work_root, *, data_root):
        calls.append((work_root, data_root))
        raise AssertionError("incomplete resources must not rebuild")

    monkeypatch.setattr(snapshot, "build_public_sparse_qa_runtime", rebuild)
    with pytest.raises(
            SparseQARuntimeSnapshotError,
            match="complete frozen rebuild resources are unavailable"):
        load_or_rebuild_public_sparse_qa_runtime(
            repository=tmp_path / "empty-distribution")
    assert calls == []


def test_distribution_data_files_match_runtime_resource_contract() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"]
        ["share/pure_integer_ai/data/ph2"])
    assert configured == (
        "data/ph2/sparse_qa_runtime_snapshot_v1.json",
        *PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
    )


def test_snapshot_implementation_forbids_unsafe_object_cache_formats() -> None:
    source = Path(
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_sparse_qa_snapshot.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("pickle", "marshal", "importlib", "__import__"):
        assert forbidden not in source
