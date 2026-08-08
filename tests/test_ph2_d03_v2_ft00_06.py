"""Public-only FT00-06 owner, freeze, report and exposure-boundary tests."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_BOUNDARY_PATH,
    V2_EVALUATOR_STAGES,
    V2_STAGE_EVALUATION_POLICIES,
    V2EvaluatorBoundaryError,
    V2EvaluatorResourceBudget,
    V2ReportExposureError,
    build_v2_evaluator_boundary_contract,
    build_v2_private_family_registration,
    publish_v2_evaluator_boundary_contract,
    read_v2_evaluator_boundary_contract,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    assert_v2_blind_family_eligible,
    audit_v2_safe_report,
    authorize_v2_access,
    read_v2_exposure_incidents,
)


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / Path(*V2_EVALUATOR_BOUNDARY_PATH.split("/"))
_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _write_fixture(root: Path, relative: str, payload: bytes = b"fixture") -> tuple[str, int]:
    target = root / Path(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return sha256(payload).hexdigest(), len(payload)


def _roots(tmp_path: Path) -> V2PhysicalRoots:
    directories = []
    for name in ("candidate", "teacher", "dev", "shadow", "private", "incidents"):
        target = tmp_path / name
        target.mkdir()
        directories.append(target)
    return V2PhysicalRoots.from_paths(*directories)


def _registration() -> object:
    return build_v2_private_family_registration(
        "W-02",
        payload_commitment=_A,
        case_commitment=_B,
        label_commitment=_C,
        cluster_commitment=_D,
        candidate_freeze_sha256=_E,
        code_freeze_sha256=_F,
        resource_budget=V2EvaluatorResourceBudget(8, 500, 4096, 16, 64, 1),
    )


def _request(
        *,
        stage: str,
        owner: str,
        split: str,
        kind: str,
        relative: str,
        digest: str,
        size: int,
        purpose: str,
        candidate_freeze: str | None = None,
        code_freeze: str | None = None,
        write_account: V2WriteAccount | None = None,
        ) -> V2AccessRequest:
    return V2AccessRequest(
        stage, owner, split, kind, relative, digest, size, purpose,
        candidate_freeze, code_freeze,
        V2WriteAccount() if write_account is None else write_account,
    )


def test_boundary_manifest_is_canonical_and_payload_free() -> None:
    contract = read_v2_evaluator_boundary_contract(ROOT, BOUNDARY)
    assert contract.status == "EVALUATOR_BOUNDARY_FROZEN"
    assert contract.successor_contract.relative_path.endswith("successor_contract_v1.json")
    assert tuple(item.stage_key for item in contract.stage_policies) == V2_EVALUATOR_STAGES
    assert tuple(len(item.bearing_dimension_keys) for item in contract.stage_policies) == (
        4, 4, 4, 4, 7, 7, 5, 5)
    assert contract.initial_state["private_payload_reads"] == 0
    assert contract.initial_state["formal_private_evaluation_runs"] == 0
    assert "absolute_path" not in str(contract.to_dict()).lower()


def test_boundary_manifest_publish_readback_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "boundary.json"
    publish_v2_evaluator_boundary_contract(ROOT, target)
    assert read_v2_evaluator_boundary_contract(ROOT, target) == (
        build_v2_evaluator_boundary_contract(ROOT))
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Exception):
        publish_v2_evaluator_boundary_contract(ROOT, target)


def test_owner_roots_and_private_freeze_gate(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    contract = build_v2_evaluator_boundary_contract(ROOT)
    digest, size = _write_fixture(
        roots.candidate_train, "observations/train.jsonl.gz", b"candidate")
    permit = authorize_v2_access(
        contract, roots,
        _request(
            stage="W-02", owner="PH2_V2_CANDIDATE", split="train",
            kind="observation", relative="observations/train.jsonl.gz",
            digest=digest, size=size, purpose="TRAIN_INTAKE"),
    )
    assert permit.target_path == roots.candidate_train / "observations/train.jsonl.gz"
    assert "target_path" not in permit.to_safe_dict()

    registration = _registration()
    private_digest, private_size = _write_fixture(
        roots.private_evaluator, "observations/held_out.jsonl.gz", b"held-out-fixture")
    with pytest.raises(V2EvaluatorBoundaryError, match="registration|freeze"):
        authorize_v2_access(
            contract, roots,
            _request(
                stage="W-02", owner="PH2_V2_PRIVATE_EVALUATOR", split="held_out",
                kind="observation", relative="observations/held_out.jsonl.gz",
                digest=private_digest, size=private_size, purpose="PRIVATE_EVALUATION"),
        )
    private_permit = authorize_v2_access(
        contract, roots,
        _request(
            stage="W-02", owner="PH2_V2_PRIVATE_EVALUATOR", split="held_out",
            kind="observation", relative="observations/held_out.jsonl.gz",
            digest=private_digest, size=private_size, purpose="PRIVATE_EVALUATION",
            candidate_freeze=registration.candidate_freeze_sha256,
            code_freeze=registration.code_freeze_sha256),
        registration=registration,
    )
    assert private_permit.owner_key == "PH2_V2_PRIVATE_EVALUATOR"
    with pytest.raises(V2EvaluatorBoundaryError):
        authorize_v2_access(
            contract, roots,
            _request(
                stage="W-02", owner="PH2_V2_CANDIDATE", split="held_out",
                kind="observation", relative="observations/held_out.jsonl.gz",
                digest=private_digest, size=private_size, purpose="TRAIN_INTAKE"),
        )


def test_path_layout_write_and_root_overlap_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(V2EvaluatorBoundaryError, match="overlap"):
        V2PhysicalRoots.from_paths(*(tmp_path for _ in range(6)))
    roots = _roots(tmp_path)
    contract = build_v2_evaluator_boundary_contract(ROOT)
    digest, size = _write_fixture(roots.dev_calibration, "observations/dev.jsonl.gz")
    with pytest.raises(V2EvaluatorBoundaryError):
        authorize_v2_access(
            contract, roots,
            _request(
                stage="W-02", owner="PH2_V2_DEV_CALIBRATOR", split="dev",
                kind="observation", relative="../observations/dev.jsonl.gz",
                digest=digest, size=size, purpose="DEV_CALIBRATION"),
        )
    with pytest.raises(V2EvaluatorBoundaryError, match="write"):
        authorize_v2_access(
            contract, roots,
            _request(
                stage="W-02", owner="PH2_V2_DEV_CALIBRATOR", split="dev",
                kind="observation", relative="observations/dev.jsonl.gz",
                digest=digest, size=size, purpose="DEV_CALIBRATION",
                write_account=V2WriteAccount(host_writes=1)),
        )


def test_safe_report_guard_records_irreversible_exposure(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    registration = _registration()
    safe = {
        "artifact_kind": "PH2_D03_V2_SAFE_AGGREGATE",
        "dimension_results": [{
            "dimension_key": "W-02-V2-OOV",
            "evidence_sha256": _A,
            "fail_count": 0,
            "ne_count": 0,
            "pass_count": 1,
            "required_count": 1,
            "status": "PASS",
        }],
        "family_commitment": registration.family_commitment,
        "status": "PASS",
    }
    validate_v2_safe_report(safe)
    with pytest.raises(V2ReportExposureError, match="PATH_STRING"):
        validate_v2_safe_report({"detail": "nested/private.json"})
    bad = {**safe, "expected_state": "hidden"}
    with pytest.raises(V2ReportExposureError, match="FORBIDDEN_FIELD"):
        audit_v2_safe_report(
            bad, roots, registration, phase="REPORT_BUILD")
    incidents = read_v2_exposure_incidents(roots, registration.family_commitment)
    assert len(incidents) == 1
    assert incidents[0].blind_pass_eligible == 0
    with pytest.raises(V2EvaluatorBoundaryError, match="not blind"):
        assert_v2_blind_family_eligible(roots, registration)


def test_stage_policy_thresholds_are_not_weakenable() -> None:
    assert len(V2_STAGE_EVALUATION_POLICIES) == 8
    for policy in V2_STAGE_EVALUATION_POLICIES:
        assert policy.min_pass_numerator == policy.min_pass_denominator == 1
        assert policy.max_fail_count == 0
        assert policy.ne_policy == "BLOCK"
        assert len(policy.ablation_keys) == len(policy.hard_conjunct_keys)
