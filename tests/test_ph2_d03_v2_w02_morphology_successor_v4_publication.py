"""Public receipt and source-feasibility guards for W-02 V4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes


STAGES = Path("data/ph2/manifests/d03_v2/stages")
PROBE = STAGES / "ph2_d03_v2_w02_morphology_successor_v4_public_probe_v1.json"
RECEIPT = STAGES / "ph2_d03_v2_w02_morphology_successor_v4_artifact_receipt_v1.json"
FEASIBILITY = Path(
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_w02_morphology_successor_v4_r6_source_feasibility_v1.json")


def _read(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json_bytes(value) + b"\n"
    return value


def test_v4_public_probe_freezes_real_dev_shadow_and_zero_test_reads() -> None:
    probe = _read(PROBE)

    assert probe["status"] == "PASS"
    assert probe["test_split_content_reads"] == 0
    assert probe["candidate_v1_v2_v3_mutations"] == 0
    assert probe["dev"]["coverage_basis_points"] == 9818
    assert probe["shadow"]["coverage_basis_points"] == 9817
    assert probe["dev"]["max_candidates_per_token"] == 13
    assert probe["shadow"]["max_candidates_per_token"] == 13
    assert probe["metamorphic"]["language_isolation_candidate_count"] == 0


def test_v4_receipt_binds_all_production_code_and_public_evidence() -> None:
    receipt = _read(RECEIPT)

    assert receipt["status"] == "W02_MORPHOLOGY_SUCCESSOR_V4_PUBLIC_ARTIFACT_FROZEN"
    assert receipt["formal_private_evaluation_runs"] == 0
    assert receipt["w02_runtime_evidenced"] == 0
    assert receipt["w03_started"] == 0
    assert receipt["previous_r5_result"]["status"] == "FAIL"
    assert receipt["previous_r5_result"]["rerun_authorized"] == 0
    for row in receipt["code_files"]:
        path = Path(row["repository_path"])
        raw = path.read_bytes()
        assert len(raw) == row["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    probe_raw = PROBE.read_bytes()
    assert len(probe_raw) == receipt["public_probe"]["manifest_size_bytes"]
    assert hashlib.sha256(probe_raw).hexdigest() == receipt["public_probe"]["manifest_sha256"]
    feasibility_raw = FEASIBILITY.read_bytes()
    assert len(feasibility_raw) == receipt["r6_source_feasibility"]["manifest_size_bytes"]
    assert hashlib.sha256(feasibility_raw).hexdigest() == receipt["r6_source_feasibility"]["manifest_sha256"]


def test_r6_source_remains_conditional_without_tuecl_payload_read() -> None:
    feasibility = _read(FEASIBILITY)

    assert feasibility["formal_r6_authorized"] == 0
    assert feasibility["test_payload_reads"] == 0
    assert feasibility["tuecl_metadata"]["sentence_count"] == 100
    assert feasibility["tuecl_metadata"]["token_count"] == 648
    assert feasibility["tuecl_token_span_path"]["false_sentence_inflation_authorized"] == 0
    assert feasibility["tuecl_token_span_path"]["status"] == (
        "REQUIRES_FRESH_ISOLATED_OWNER_PROOF")
