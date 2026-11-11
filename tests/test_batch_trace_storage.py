"""压缩存储接入既有 provenance/冷审，规范 trace 身份保持一致。"""
from copy import deepcopy
import json

import pytest

from pure_integer_ai.experiments.batch_case_storage import (
    build_compact_case_provenance, restore_compact_case_provenance,
)
from pure_integer_ai.experiments.batch_trace_storage import (
    TraceStorageError, load_case_query_trace, load_saved_query_trace,
    read_trace_reference, save_case_trace, sha256_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_batch_case_storage import _captured


def test_compressed_blob_and_legacy_json_restore_identical_provenance(tmp_path):
    captured, original = _captured()
    value = json.loads(original)
    plain, compressed = tmp_path / "plain", tmp_path / "compressed"
    plain.mkdir()
    compressed.mkdir()
    plain_ref, plain_raw = save_case_trace(plain, value)
    ref, raw = save_case_trace(compressed, value, compressed=True)
    assert raw == plain_raw == original
    assert ref["sha256"] == plain_ref["sha256"] == sha256_bytes(original)
    assert ref["stored_byte_count"] < len(original)
    compact = build_compact_case_provenance(captured, trace_bytes=raw,
        trace_ref=ref, trace_root=compressed)
    assert restore_compact_case_provenance(compact, root=compressed) == captured
    assert load_saved_query_trace(plain / "trace.int.json") == value
    assert load_saved_query_trace(compressed / "trace.int.json") == value
    receipt = {"trace_sha256": ref["sha256"], "trace_byte_count": len(raw),
        "trace_stored_byte_count": ref["stored_byte_count"],
        "trace_stored_sha256": ref["stored_sha256"],
        "trace_descriptor_sha256": sha256_bytes((compressed / "trace.int.json").read_bytes())}
    (compressed / "call_receipt.json").write_bytes(canonical_json_bytes(receipt))
    assert load_case_query_trace(compressed / "trace.int.json") == value
    receipt["trace_stored_sha256"] = "0" * 64
    with pytest.raises(TraceStorageError, match="physical"):
        load_saved_query_trace(compressed / "trace.int.json", receipt=receipt)


@pytest.mark.parametrize("mutation", ["blob", "descriptor", "reference", "budget", "projection", "path"])
def test_tampered_compressed_storage_is_rejected(tmp_path, mutation):
    _captured_value, original = _captured()
    ref, raw = save_case_trace(tmp_path, json.loads(original), compressed=True)
    if mutation == "blob":
        path = tmp_path / ref["path"]
        path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    elif mutation == "descriptor":
        path = tmp_path / "trace.int.json"
        value = json.loads(path.read_bytes())
        value["trace_ref"]["sha256"] = "0" * 64
        path.write_bytes(canonical_json_bytes(value))
    elif mutation == "reference":
        ref = deepcopy(ref)
        ref["stored_sha256"] = "0" * 64
    elif mutation == "budget":
        with pytest.raises(TraceStorageError, match="budget"):
            read_trace_reference(tmp_path, ref, max_trace_bytes=len(raw) - 1)
        return
    elif mutation == "projection":
        ref["integer_projection_sha256"] = "0" * 64
    else:
        ref["path"] = "../outside.lz4"
    with pytest.raises(TraceStorageError):
        read_trace_reference(tmp_path, ref)


def test_writer_never_overwrites_an_existing_evidence_blob(tmp_path):
    _capture, original = _captured()
    save_case_trace(tmp_path, json.loads(original), compressed=True)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        save_case_trace(tmp_path, json.loads(original), compressed=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
