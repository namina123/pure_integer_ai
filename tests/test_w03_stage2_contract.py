"""W-03 独立 context、request 与 payload 前防火墙合同。"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import gzip
import hashlib
from pathlib import Path, PurePosixPath

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import D03FileIdentity
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_w03_context import open_w03_frozen_context
from pure_integer_ai.experiments.ph2_w03_continuity import (
    FORMAL_W02_AGGREGATE_PATH,
    FORMAL_W02_ATTESTATION_PATH,
    FORMAL_W02_CANDIDATE_FREEZE_PATH,
    FORMAL_W02_RECEIPT_PATH,
    W03CIJobBinding,
    W03PublicationObservation,
    W03W02ContinuityBinding,
    formal_w03_publication_baseline,
    validate_w03_publication_observation,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03ContractError,
    W03PayloadAudit,
    W03RunRequest,
    validate_w03_request,
)
from pure_integer_ai.experiments.ph2_w03_firewall import W03PayloadFirewall


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPOSITORY.parent / "w02_artifacts"


def _identity(path: str, payload: bytes) -> D03FileIdentity:
    """为纯合同 fixture 构造稳定文件身份。"""
    return D03FileIdentity(path, len(payload), hashlib.sha256(payload).hexdigest())


def _w02_continuity() -> W03W02ContinuityBinding:
    """构造与正式顶层 identity 一致、无需 private payload 的 W-02 绑定。"""
    code = tuple(_identity(f"src/w02_{index}.py", bytes([index])) for index in range(10))
    artifacts = tuple(
        _identity(f"formal_candidate_v2/artifact_{index}.bin", bytes([index]))
        for index in range(11)
    )
    digests = tuple(
        (key, hashlib.sha256(key.encode("ascii")).hexdigest())
        for key in ("artifact", "core", "cursor", "logical", "manifest", "memory", "use")
    )
    return W03W02ContinuityBinding(
        receipt_identity=D03FileIdentity(
            FORMAL_W02_RECEIPT_PATH,
            4297,
            "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb",
        ),
        candidate_freeze_identity=D03FileIdentity(
            FORMAL_W02_CANDIDATE_FREEZE_PATH,
            5959,
            "a280e25a01efc50f4f22ab5060be55bed193e1ca600c74050f0937969844d9ff",
        ),
        candidate_attestation_identity=D03FileIdentity(
            FORMAL_W02_ATTESTATION_PATH,
            7781,
            "ff19923f55fef60c29911708a956648c51930ac8f794ff66e9e5103c1bfb642e",
        ),
        aggregate_identity=D03FileIdentity(
            FORMAL_W02_AGGREGATE_PATH,
            4811,
            "dd03d7873897cda597ba5ab0426d793aef6fade06f71852e505d67d39f9a5a94",
        ),
        historical_publication_head_sha1=(
            "588fa8465c6bb23cf8656d606f1d7d3e49d30056"
        ),
        historical_ci_run_id=30463618992,
        historical_ci_jobs=(
            W03CIJobBinding("Python 3.11 on ubuntu-latest", 90615757653),
            W03CIJobBinding("Python 3.14 on ubuntu-latest", 90615757688),
            W03CIJobBinding("Python 3.14 on windows-latest", 90615757748),
            W03CIJobBinding("Secret scan", 90615757658),
        ),
        candidate_run_id=3,
        formal_training_runs=1,
        capability_code_identities=code,
        host_artifact_identities=artifacts,
        host_digests=digests,
        dimension_pass_counts=(2, 2, 4, 2, 5),
        dimension_statuses=("PASS", "PASS", "PASS", "PASS", "PASS"),
        fail_count=0,
        ne_count=0,
        execution_state=(
            ("LANGUAGE_CAPABILITY_MASTERED", 0),
            ("LANGUAGE_READINESS", 0),
            ("W02_BLOCKED_FAILED", 0),
            ("W02_RUNTIME_EVIDENCED", 1),
            ("W02_STARTED", 1),
            ("W03_STARTED", 0),
        ),
    )


def test_w03_context_binds_both_publication_epochs_without_payload(monkeypatch) -> None:
    """context 同时绑定 W-02 历史证据和 W-03 新 baseline，且零 gzip。"""
    original_read_bytes = Path.read_bytes
    gzip_reads: list[str] = []

    def guarded_read_bytes(path: Path) -> bytes:
        """拒绝 context 构造提前打开任何训练或 evaluator payload。"""
        if path.name.endswith(".jsonl.gz"):
            gzip_reads.append(path.as_posix())
            raise AssertionError("W-03 context 不得读取 gzip payload")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    baseline = formal_w03_publication_baseline()
    context = open_w03_frozen_context(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=baseline.head_sha1,
        w02_continuity=_w02_continuity(),
        publication_baseline=baseline,
        backend_profile_key=(3, 1),
    )

    assert context.stage_key == "W-03"
    assert context.w02_continuity.receipt_identity.sha256 == (
        "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb"
    )
    assert context.publication_baseline.head_sha1 == (
        "8344ef609d46f544079da9a938851c230e0a63db"
    )
    assert context.publication_baseline.ci_run_id == 30508304244
    assert tuple(len(items) for items in (
        context.candidate_payload_bindings,
        context.teacher_evidence_bindings,
        context.evaluator_visible_bindings,
    )) == (12, 6, 29)
    assert context.dimension_keys == (
        "W-03-CONCEPT-SPLIT",
        "W-03-POLYSEMY-COMPETITION",
        "W-03-SOURCE-CONFLICT",
        "W-03-SUPERSEDE",
    )
    assert context.generation_hard_conjunct == (
        "W-03-GENERATION-SENSE-SELECTION-HARD-CONJUNCT"
    )
    assert context.logical_shard_count == 16
    assert len(context.failure_point_keys) == 6
    assert context.resource_budget["max_payload_bytes"] == 201326592
    assert set(context.execution_state.values()) == {0}
    assert context.payload_gets == context.payload_bytes == context.learning_writes == 0
    assert gzip_reads == []


@lru_cache(maxsize=1)
def _formal_context():
    """Verify the immutable outer W-02 base and open the formal W-03 context."""
    baseline = formal_w03_publication_baseline()
    continuity = verify_formal_w02_continuity(REPOSITORY, ARTIFACTS_ROOT)
    return open_w03_frozen_context(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=baseline.head_sha1,
        w02_continuity=continuity,
        publication_baseline=baseline,
        backend_profile_key=(3, 1),
    )


def _request(context, **changes) -> W03RunRequest:
    """Build the one exact run-4 request, then apply a test-only mutation."""
    request = W03RunRequest(
        run_id=context.run_id,
        parent_run_id=context.parent_run_id,
        base_run_id=context.base_run_id,
        stage_key=context.stage_key,
        owner_key=context.owner_key,
        runner_key="PH2_LANGUAGE_STAGE2",
        publication_baseline_key=context.publication_baseline.stable_key(),
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        w02_continuity_key=context.w02_continuity.stable_key(),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    return replace(request, **changes)


def _publication_observation(**changes) -> W03PublicationObservation:
    """Represent an independently observed fully green W03-00C epoch."""
    baseline = formal_w03_publication_baseline()
    observation = W03PublicationObservation(
        local_head_sha1=baseline.head_sha1,
        tracking_head_sha1=baseline.head_sha1,
        remote_head_sha1=baseline.head_sha1,
        ci_run_id=baseline.ci_run_id,
        ci_head_sha1=baseline.head_sha1,
        ci_status="completed",
        ci_conclusion="success",
        ci_jobs=baseline.ci_jobs,
    )
    return replace(observation, **changes)


def test_formal_w02_continuity_reverifies_ten_code_and_eleven_artifacts() -> None:
    """W-03 base is the unchanged W-02 run-3 receipt/candidate/five-way PASS."""
    continuity = verify_formal_w02_continuity(REPOSITORY, ARTIFACTS_ROOT)

    assert continuity.receipt_identity.sha256 == (
        "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb"
    )
    assert continuity.candidate_freeze_identity.sha256 == (
        "a280e25a01efc50f4f22ab5060be55bed193e1ca600c74050f0937969844d9ff"
    )
    assert continuity.candidate_attestation_identity.sha256 == (
        "ff19923f55fef60c29911708a956648c51930ac8f794ff66e9e5103c1bfb642e"
    )
    assert continuity.aggregate_identity.sha256 == (
        "dd03d7873897cda597ba5ab0426d793aef6fade06f71852e505d67d39f9a5a94"
    )
    assert continuity.candidate_run_id == 3
    assert continuity.formal_training_runs == 1
    assert len(continuity.capability_code_identities) == 10
    assert len(continuity.host_artifact_identities) == 11
    assert tuple(key for key, _ in continuity.host_digests) == (
        "artifact", "core", "cursor", "logical", "manifest", "memory", "use")
    assert continuity.dimension_pass_counts == (2, 2, 4, 2, 5)
    assert continuity.dimension_statuses == ("PASS",) * 5
    assert continuity.fail_count == continuity.ne_count == 0
    assert dict(continuity.execution_state)["W03_STARTED"] == 0


def test_formal_context_binds_d03_six_pack_and_supplemental_generation_gate() -> None:
    """The D-03 underscore keys map exactly to canonical W-03 hyphen keys."""
    context = _formal_context()

    assert context.d03_receipt_identity.sha256 == (
        "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef"
    )
    assert context.d03_global_manifest_identity.sha256 == (
        "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6"
    )
    assert context.stage_manifest_identity.sha256 == (
        "b32a3174e9d224fb702bc378e311aa9f1d07f205feca90e80f9a81a17ecfcd3b"
    )
    assert tuple(item.manifest_identity.sha256 for item in context.pack_bindings) == (
        "8c0c04d09a81b6ecc59862b4d9dfcf9d7beefef7c85a051b67e1ecb0d34a4f44",
        "552987b414ee9dc2a90c4d5b3d67c2dcd969727c4ece4430e7bbc79369bd018c",
        "0935bce7b99f9b9df1dd1671939e680acd9b9e0d65ffd630eecfb46bd1d6035c",
        "cf0c397e9b3e2c04aae3cadc53b90158efe241759c7f13a8406fe5b4d6c2b4f1",
        "d3b0a87930b72513f20554c7aab75085a157d108e4eca8e521f5138c5f9a7b46",
        "9c947242afd3c00a0dbe35f5e62ba28d3446475f2223e6bd8cd82437b85098e6",
    )
    assert context.dimension_key_map == (
        ("W-03-CONCEPT_SPLIT", "W-03-CONCEPT-SPLIT"),
        ("W-03-POLYSEMY_COMPETITION", "W-03-POLYSEMY-COMPETITION"),
        ("W-03-SOURCE_CONFLICT", "W-03-SOURCE-CONFLICT"),
        ("W-03-SUPERSEDE", "W-03-SUPERSEDE"),
    )
    assert all(item.min_pass_numerator == item.min_pass_denominator == 1
               and item.max_fail_count == 0 and item.ne_policy == "BLOCK"
               for item in context.d03_thresholds)
    assert context.evaluation_order[-1] == (
        "W-03-GENERATION-SENSE-SELECTION-HARD-CONJUNCT")
    assert context.aggregation_policy == "ALL_W03_BEARINGS_AND_GENERATION_MUST_PASS"
    assert context.run_id == 4
    assert context.parent_run_id == context.base_run_id == 3
    assert context.allowed_worker_counts == (1, 2, 4)
    assert context.logical_shard_count == 16
    assert len(context.failure_point_keys) == 6
    assert context.resource_budget["max_logic_operations"] == 3_000_000
    assert context.execution_state == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W03_STARTED": 0,
        "W04_STARTED": 0,
        "formal_w03_training_runs": 0,
        "teacher_calls": 0,
    }


def test_candidate_request_has_no_private_future_or_label_field() -> None:
    """The typed request offers no field through which evaluator data can enter."""
    request = validate_w03_request(_formal_context(), _request(_formal_context()))

    for field in (
            "dev_paths", "held_out_paths", "evaluator_paths", "evaluator_labels",
            "conceptnet_paths", "ud_paths", "future_stage_paths"):
        assert not hasattr(request, field)
    assert all("/owners/evaluator/" not in path
               for path in request.candidate_payload_paths)
    assert all("/owners/teacher/" in path
               for path in request.teacher_evidence_paths)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"run_id": 5}, "run/parent/base"),
        ({"parent_run_id": 2}, "run/parent/base"),
        ({"base_run_id": 2}, "run/parent/base"),
        ({"stage_key": "W-04"}, "W-03"),
        ({"owner_key": "PH2_W02_TRANSACTION_OWNER"}, "owner"),
        ({"runner_key": "PH2_LANGUAGE_STAGE1"}, "runner"),
        ({"publication_baseline_key": (1,)}, "publication"),
        ({"current_remote_commit_sha1": "0" * 40}, "commit"),
        ({"w02_continuity_key": (1,)}, "W-02"),
        ({"d03_context_key": (1,)}, "D-03"),
        ({"backend_profile_key": (9,)}, "backend"),
        ({"base_fence_key": (9,)}, "fence"),
        ({"worker_count": 3}, "worker"),
        ({"mode": "skip"}, "mode"),
        ({"resource_budget": (("max_workers", 5),)}, "resource"),
    ),
)
def test_request_identity_drift_fails_before_transport(changes, match) -> None:
    """Every run/base/backend/fence/scheduling/resource drift closes at T0."""
    context = _formal_context()
    audit = W03PayloadAudit()

    with pytest.raises(W03ContractError, match=match):
        W03PayloadFirewall.open(
            REPOSITORY,
            context,
            _request(context, **changes),
            publication_observation=_publication_observation(),
            audit=audit,
        )
    assert audit.transport_attempts == audit.transport_bytes == 0
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0


def test_request_rejects_private_future_missing_extra_and_unsafe_paths() -> None:
    """Exact path tuples reject evaluator, future, omissions, extras, and traversal."""
    context = _formal_context()
    request = _request(context)
    private = next(
        item.relative_path for item in context.evaluator_visible_bindings
        if item.owner_kind == "evaluator")
    variants = (
        replace(request, candidate_payload_paths=request.candidate_payload_paths[:-1]),
        replace(request, candidate_payload_paths=(
            *request.candidate_payload_paths, private)),
        replace(request, candidate_payload_paths=(
            *request.candidate_payload_paths,
            "ph2_d03_dataset_artifacts/course_freeze_v1/packs/"
            "AUTHORED_CC0_V1--CC0-1.0--lc04-recursive-parse-v1/"
            "observations/train.jsonl.gz")),
        replace(request, teacher_evidence_paths=(
            *request.teacher_evidence_paths, "../private.jsonl.gz")),
    )
    for variant in variants:
        audit = W03PayloadAudit()
        with pytest.raises(W03ContractError, match="path|whitelist|safe"):
            W03PayloadFirewall.open(
                REPOSITORY,
                context,
                variant,
                publication_observation=_publication_observation(),
                audit=audit,
            )
        assert audit.transport_attempts == audit.payload_gets == audit.payload_bytes == 0
        assert audit.learning_writes == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"local_head_sha1": "0" * 40},
        {"tracking_head_sha1": "0" * 40},
        {"remote_head_sha1": "0" * 40},
        {"ci_run_id": 1},
        {"ci_head_sha1": "0" * 40},
        {"ci_status": "in_progress"},
        {"ci_conclusion": "failure"},
        {"ci_jobs": formal_w03_publication_baseline().ci_jobs[:-1]},
    ),
)
def test_publication_head_remote_ci_drift_fails_before_transport(changes) -> None:
    """Local/tracking/remote HEAD and every CI component are hard gates."""
    context = _formal_context()
    audit = W03PayloadAudit()
    with pytest.raises(W03ContractError, match="HEAD/remote/CI"):
        W03PayloadFirewall.open(
            REPOSITORY,
            context,
            _request(context),
            publication_observation=_publication_observation(**changes),
            audit=audit,
        )
    assert audit.transport_attempts == audit.transport_bytes == 0
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0


def test_request_execution_identity_excludes_worker_and_mode_only() -> None:
    """1/2/4 and fresh/restart/resume alter scheduling, not the run identity."""
    context = _formal_context()
    first = validate_w03_request(context, _request(context))
    resumed = validate_w03_request(
        context, _request(context, worker_count=4, mode="resume"))

    assert first.execution_identity_key() == resumed.execution_identity_key()
    assert first.scheduling_key() != resumed.scheduling_key()


def test_public_train_only_firewall_delivers_atomically_without_writes() -> None:
    """The direct public adapter test decodes only 12+6 authorized train files."""
    context = _formal_context()
    audit = W03PayloadAudit()
    firewall = W03PayloadFirewall.open(
        REPOSITORY,
        context,
        _request(context),
        publication_observation=_publication_observation(),
        audit=audit,
    )
    payload = firewall.read_training_payload()

    assert (len(payload.source_refs), len(payload.observations),
            len(payload.teacher_evidence)) == (83, 40, 40)
    assert all(item.split == "train" and item.w_stage in {"W-02", "W-03"}
               for item in payload.observations)
    assert all(item.visible_from_stage in {"W-02", "W-03"}
               for item in payload.teacher_evidence)
    assert audit.transport_attempts == 18
    assert audit.transport_bytes == 134763
    assert audit.payload_gets == 18
    assert audit.payload_bytes == 134763
    assert audit.teacher_calls == audit.learning_writes == 0
    with pytest.raises(W03ContractError, match="replayed"):
        firewall.read_training_payload()
    assert audit.payload_gets == 18


def test_transport_identity_failure_has_attempt_but_zero_delivery(tmp_path: Path) -> None:
    """A changed gzip header is transport work, never a successful delivery."""
    context = _formal_context()
    first = context.candidate_payload_bindings[0]
    source = REPOSITORY / Path(*PurePosixPath(first.relative_path).parts)
    target = tmp_path / Path(*PurePosixPath(first.relative_path).parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(gzip.decompress(source.read_bytes()), mtime=1))
    assert target.read_bytes() != source.read_bytes()
    audit = W03PayloadAudit()
    firewall = W03PayloadFirewall.open(
        tmp_path,
        context,
        _request(context),
        publication_observation=_publication_observation(),
        dependency_root=REPOSITORY,
        audit=audit,
    )

    with pytest.raises(W03ContractError, match="transport|gzip|SHA-256"):
        firewall.read_training_payload()
    assert audit.transport_attempts == 1
    assert audit.transport_bytes == target.stat().st_size
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0


def test_symlink_substitution_fails_before_transport(monkeypatch) -> None:
    """A symlink at any authorized lexical path is not an authorized file."""
    from pure_integer_ai.experiments import ph2_w03_firewall

    context = _formal_context()
    audit = W03PayloadAudit()
    firewall = W03PayloadFirewall.open(
        REPOSITORY,
        context,
        _request(context),
        publication_observation=_publication_observation(),
        audit=audit,
    )
    monkeypatch.setattr(ph2_w03_firewall, "_has_symlink_component", lambda *_: True)

    with pytest.raises(W03ContractError, match="symlink"):
        firewall.read_training_payload()
    assert audit.transport_attempts == audit.transport_bytes == 0
    assert audit.payload_gets == audit.payload_bytes == 0
    assert audit.learning_writes == 0


def test_w02_receipt_byte_drift_closes_continuity(monkeypatch) -> None:
    """The W-02 prerequisite is rehashed rather than trusted from a cached object."""
    original = Path.read_bytes
    target = (ARTIFACTS_ROOT / FORMAL_W02_RECEIPT_PATH).resolve()

    def drifted(path: Path) -> bytes:
        payload = original(path)
        return payload + b" " if path.resolve() == target else payload

    monkeypatch.setattr(Path, "read_bytes", drifted)
    with pytest.raises(W03ContractError, match="identity drifted"):
        verify_formal_w02_continuity(REPOSITORY, ARTIFACTS_ROOT)


def test_publication_observation_accepts_only_exact_formal_baseline() -> None:
    """The positive publication gate binds the same four job ids as W03-00C."""
    baseline = formal_w03_publication_baseline()
    observed = validate_w03_publication_observation(
        baseline, _publication_observation())

    assert observed.ci_run_id == 30508304244
    assert tuple(item.job_id for item in observed.ci_jobs) == (
        90762736125, 90762736134, 90762736152, 90762736103)
