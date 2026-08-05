"""W09-01 typed contract、34/1/37 registry 与 firewall 专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALL_DIMENSION_KEYS,
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09_DEV_OWNER,
    W09_EVALUATOR_OWNER,
    W09_OWNER_KEY,
    W09ContractError,
    W09HostWriteSnapshot,
    W09PayloadAudit,
    make_w09_request,
    open_w09_frozen_contract,
    validate_w09_request,
)
from pure_integer_ai.experiments.ph2_w09_firewall import (
    W09FailureKind,
    W09FirewallError,
    W09PayloadFirewall,
    W09VisibilityFirewall,
    _assert_exact_record_schema,
    _payload_file,
    _sanitize_candidate_observation,
)
from pure_integer_ai.experiments.ph2_w09_registry import (
    W09_SHARED_TYPED_ENGINE_KEY,
    audit_w09_registry_payload,
    build_w09_registry,
)
from pure_integer_ai.experiments.ph2_w09_types import (
    TeacherExitPhase,
    W09AblationResult,
    W09CloneReceipt,
    W09ConsumerChoice,
    W09ConsumerRequest,
    W09DimensionResult,
    W09DirectionalResult,
    W09ResourceAudit,
    W09ResultState,
    W09RollbackReceipt,
    W09StopDecision,
    W09TypeError,
    W09UseOutcome,
    W09VerifierResult,
    W09WindowIdentity,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _key(value: str) -> tuple[int, ...]:
    return digest_value({"test_key": value})


@pytest.fixture(scope="module")
def context():
    return open_w09_frozen_contract(ROOT)


@pytest.fixture(scope="module")
def training(context):
    firewall = W09PayloadFirewall.open(ROOT, context, make_w09_request(context))
    payload = firewall.read_training_payload()
    report = audit_w09_registry_payload(payload, context)
    return firewall, payload, report


def test_w09_contract_freezes_independent_owner_and_exact_registries(context) -> None:
    assert context.authority_sha256 == (
        "71018eba671c0d1f9182f9013f6838edb0ba3561ce297d2018ea04555b732acc"
    )
    assert context.baseline_public_head_commit_sha1 == (
        "ec55415088b88e1a78e622cc6b7170306bfc17c4"
    )
    assert context.owner_key == W09_OWNER_KEY
    assert len(context.candidate_pack_keys) == 34
    assert len(context.dev_pack_keys) == 1
    assert context.dev_pack_keys == (
        "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1",
    )
    assert len(context.held_out_pack_keys) == 37
    assert context.held_out_pack_keys == context.evaluator_pack_keys
    assert context.future_pack_keys == ()
    assert len(context.candidate_bindings) == 68
    assert len(context.training_material_bindings) == 34
    assert len(context.dev_bindings) == 2
    assert len(context.evaluator_bindings) == 111
    assert len(context.forbidden_bindings) == 10
    assert context.carrier_keys == W09_CARRIER_KEYS
    assert context.consumer_keys == W09_CONSUMER_KEYS
    assert context.dimension_keys == W09_DIMENSION_KEYS
    assert context.ablation_keys == W09_ABLATION_KEYS
    assert dict(context.execution_state)["W09_STARTED"] == 0
    assert dict(context.execution_state)["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert dict(context.execution_state)["LANGUAGE_READINESS"] == 0
    assert len(context.stable_key()) == len(context.base_fence_key) == 32


@pytest.mark.parametrize(
    "mutator,poison",
    [
        (lambda request, _: replace(request, worker_count=3), ""),
        (lambda request, _: replace(request, mode="formal"), ""),
        (
            lambda request, _: replace(
                request,
                owner_key="PH2_W08_TRANSACTION_OWNER",
            ),
            "",
        ),
        (
            lambda request, context: replace(
                request,
                candidate_pack_keys=request.candidate_pack_keys[:-1],
            ),
            "",
        ),
        (
            lambda request, context: replace(
                request,
                candidate_payload_paths=(
                    *request.candidate_payload_paths,
                    context.dev_bindings[0].relative_path,
                ),
            ),
            "",
        ),
        (
            lambda request, _: replace(
                request,
                candidate_payload_paths=(
                    "archive.zip!observations/train.jsonl.gz",
                    *request.candidate_payload_paths[1:],
                ),
            ),
            "archive.zip",
        ),
        (
            lambda request, _: replace(
                request,
                forbidden_payload_paths=("../private/label.jsonl.gz",),
            ),
            "label.jsonl.gz",
        ),
    ],
)
def test_w09_poison_requests_fail_before_transport(context, mutator, poison) -> None:
    request = mutator(make_w09_request(context), context)
    audit = W09PayloadAudit()
    with pytest.raises(W09FirewallError) as caught:
        W09PayloadFirewall.open(ROOT, context, request, audit=audit)
    assert caught.value.report.failure_kind is W09FailureKind.INVALID_REQUEST
    assert audit.transport_attempts == audit.payload_gets == 0
    assert audit.dev_reads == audit.held_out_reads == audit.evaluator_label_reads == 0
    if poison:
        assert poison not in str(caught.value)


def test_w09_train_firewall_delivers_all_registered_packs_without_live_teacher(training) -> None:
    firewall, payload, report = training
    assert report.source_ref_count == 535
    assert report.observation_count == report.training_evidence_count == 309
    assert len(report.pack_counts) == 34
    assert all(count > 0 for _, count in report.pack_counts)
    assert report.expected_or_label_field_count == 0
    assert report.shared_typed_engine_count == 1
    assert firewall.audit.transport_attempts == firewall.audit.payload_gets == 102
    assert firewall.audit.redacted_candidate_fields == 35
    assert firewall.audit.dev_reads == firewall.audit.held_out_reads == 0
    assert firewall.audit.evaluator_label_reads == 0
    assert firewall.audit.teacher_calls == firewall.audit.api_calls == 0
    assert firewall.audit.llm_calls == 0
    assert (
        firewall.audit.core_writes
        == firewall.audit.evidence_writes
        == firewall.audit.use_writes
        == firewall.audit.memory_writes
        == firewall.audit.assessment_writes
        == firewall.audit.clock_writes
        == 0
    )
    assert len(payload.observations) == len(payload.training_evidence)
    with pytest.raises(W09FirewallError) as caught:
        firewall.read_training_payload()
    assert caught.value.report.failure_kind is W09FailureKind.PAYLOAD_REPLAY


def test_w09_dev_and_evaluator_authorization_is_owner_freeze_and_write_isolated(context) -> None:
    audit = W09PayloadAudit()
    visibility = W09VisibilityFirewall(context, audit)
    zero = W09HostWriteSnapshot()
    dev_path = context.dev_bindings[0].relative_path
    evaluator_path = next(
        item.relative_path
        for item in context.evaluator_bindings
        if item.identity.owner_kind == "evaluator"
    )
    assert visibility.authorize_dev(
        dev_path,
        owner_key=W09_DEV_OWNER,
        host_writes=zero,
    ).access_phase == "dev"
    with pytest.raises(W09FirewallError) as caught:
        visibility.authorize_dev(
            dev_path,
            owner_key=W09_DEV_OWNER,
            host_writes=W09HostWriteSnapshot(core_writes=1),
        )
    assert caught.value.report.failure_kind is W09FailureKind.HOST_WRITE_INTENT
    with pytest.raises(W09FirewallError) as caught:
        visibility.authorize_evaluator(
            evaluator_path,
            owner_key=W09_EVALUATOR_OWNER,
            candidate_sealed=0,
            code_frozen=1,
            host_writes=zero,
        )
    assert caught.value.report.failure_kind is W09FailureKind.CANDIDATE_NOT_SEALED
    with pytest.raises(W09FirewallError) as caught:
        visibility.authorize_evaluator(
            evaluator_path,
            owner_key=W09_EVALUATOR_OWNER,
            candidate_sealed=1,
            code_frozen=0,
            host_writes=zero,
        )
    assert caught.value.report.failure_kind is W09FailureKind.CODE_NOT_FROZEN
    with pytest.raises(W09FirewallError) as caught:
        visibility.authorize_evaluator(
            evaluator_path,
            owner_key="PH2_TRAIN_CANDIDATE",
            candidate_sealed=1,
            code_frozen=1,
            host_writes=zero,
        )
    assert caught.value.report.failure_kind is W09FailureKind.OWNER_SPOOF
    assert visibility.authorize_evaluator(
        evaluator_path,
        owner_key=W09_EVALUATOR_OWNER,
        candidate_sealed=1,
        code_frozen=1,
        host_writes=zero,
    ).access_phase == "evaluator"
    assert audit.transport_attempts == audit.payload_gets == 0
    assert audit.dev_reads == audit.held_out_reads == audit.evaluator_label_reads == 0
    assert all(value == 0 for _, value in audit.safe_counts())


def test_w09_safe_failure_hides_poison_path_and_rejects_link(context, tmp_path, monkeypatch) -> None:
    poison = "../surface/expected/label.jsonl.gz"
    audit = W09PayloadAudit()
    visibility = W09VisibilityFirewall(context, audit)
    with pytest.raises(W09FirewallError) as caught:
        visibility.authorize_candidate(poison)
    assert caught.value.report.failure_kind is W09FailureKind.PATH_TRAVERSAL
    assert poison not in str(caught.value)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "payload.jsonl.gz").write_bytes(b"not-read")
    original = Path.is_symlink

    def reports_link(path: Path) -> bool:
        return path.name == "linked" or original(path)

    monkeypatch.setattr(Path, "is_symlink", reports_link)
    with pytest.raises(W09FirewallError) as caught:
        _payload_file(
            tmp_path.resolve(),
            "linked/payload.jsonl.gz",
            audit,
            phase="candidate",
        )
    assert caught.value.report.failure_kind is W09FailureKind.LINK_COMPONENT
    assert audit.transport_attempts == audit.payload_gets == 0


def test_w09_extra_record_field_and_new_label_field_are_rejected(context, training) -> None:
    observation_binding = next(
        item
        for item in context.candidate_bindings
        if item.identity.owner_kind == "observation"
    )
    target = ROOT / observation_binding.relative_path
    artifact_root = target.parents[
        len(Path(observation_binding.identity.relative_path).parts) - 1
    ]
    raw_records = read_record_artifact(artifact_root, observation_binding.identity)
    values = [item.to_dict() for item in raw_records]
    values[0]["unexpected_field"] = 0
    extra_payload = b"".join(canonical_json_line(item) for item in values)
    poisoned_identity = replace(
        observation_binding.identity,
        content_sha256=hashlib.sha256(extra_payload).hexdigest(),
        content_size_bytes=len(extra_payload),
    )
    poisoned_binding = replace(observation_binding, identity=poisoned_identity)
    audit = W09PayloadAudit()
    with pytest.raises(W09FirewallError) as caught:
        _assert_exact_record_schema(raw_records, poisoned_binding, audit)
    assert caught.value.report.failure_kind is W09FailureKind.CONTENT_SCHEMA
    observation = training[1].observations[0]
    value = observation.typed_payload.to_value()
    poison_marker = "candidate_private_value_must_not_escape"
    value["label"] = poison_marker
    poisoned_observation = replace(
        observation,
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    with pytest.raises(W09FirewallError) as caught:
        _sanitize_candidate_observation(poisoned_observation, audit)
    assert caught.value.report.failure_kind is W09FailureKind.LABEL_LEAK
    assert poison_marker not in str(caught.value)


def test_w09_binding_owner_spoof_and_unregistered_pack_fail_closed(context) -> None:
    observation_binding = next(
        item
        for item in context.candidate_bindings
        if item.identity.owner_kind == "observation"
    )
    with pytest.raises(W09ContractError, match="owner/split"):
        replace(observation_binding, access_phase="evaluator")
    request = make_w09_request(context)
    changed = replace(
        request,
        candidate_pack_keys=(*request.candidate_pack_keys[:-1], "UNREGISTERED_PACK"),
    )
    with pytest.raises(W09ContractError):
        validate_w09_request(context, changed)


def test_w09_carrier_registry_uses_one_shared_engine(context) -> None:
    registry = build_w09_registry(context)
    assert len(registry.pack_entries) == 37
    assert sum(item.candidate_train for item in registry.pack_entries) == 34
    assert sum(item.dev_calibration for item in registry.pack_entries) == 1
    assert sum(item.held_out_evaluator for item in registry.pack_entries) == 37
    assert tuple(item.carrier_key for item in registry.carrier_bindings) == W09_CARRIER_KEYS
    assert {item.semantic_engine_key for item in registry.carrier_bindings} == {
        W09_SHARED_TYPED_ENGINE_KEY
    }
    assert len({item.carrier_adapter_key for item in registry.carrier_bindings}) == 9


def _direction(consumer: str) -> W09DirectionalResult:
    request_key = _key(f"{consumer}-request")
    choice_key = _key(f"{consumer}-choice")
    candidate_key = _key(f"{consumer}-candidate")
    use_key = _key(f"{consumer}-use")
    outcome_key = _key(f"{consumer}-outcome")
    return W09DirectionalResult(
        W09ConsumerRequest(consumer, request_key, _sha(f"{consumer}-input")),
        W09ConsumerChoice(consumer, request_key, choice_key, candidate_key),
        W09UseOutcome(
            consumer,
            request_key,
            choice_key,
            candidate_key,
            use_key,
            outcome_key,
            "RESOLVED",
        ),
        W09VerifierResult(
            consumer,
            request_key,
            use_key,
            outcome_key,
            _key(f"{consumer}-verifier"),
            W09ResultState.PASS,
            "NONE",
        ),
    )


def test_w09_first_class_types_keep_three_consumers_and_walls_separate() -> None:
    directions = tuple(_direction(key) for key in W09_CONSUMER_KEYS)
    dimension = W09DimensionResult(
        W09_DIMENSION_KEYS[0],
        1,
        directions,
        W09ResultState.PASS,
    )
    assert dimension.status is W09ResultState.PASS
    with pytest.raises(W09TypeError, match="共享身份"):
        W09DimensionResult(
            W09_DIMENSION_KEYS[0],
            1,
            (directions[0], replace(
                directions[1],
                request=replace(
                    directions[1].request,
                    request_key=directions[0].request.request_key,
                ),
                choice=replace(
                    directions[1].choice,
                    request_key=directions[0].request.request_key,
                ),
                use_outcome=replace(
                    directions[1].use_outcome,
                    request_key=directions[0].request.request_key,
                ),
                verifier=replace(
                    directions[1].verifier,
                    request_key=directions[0].request.request_key,
                ),
            ), directions[2]),
            W09ResultState.PASS,
        )
    resource = W09ResourceAudit.zero()
    window = W09WindowIdentity(
        TeacherExitPhase.ZERO_CALL_WINDOW,
        1,
        _sha("window-input"),
        _sha("candidate"),
        0,
        tuple((key, _sha(f"{key}-output")) for key in W09_CONSUMER_KEYS),
        resource,
        _sha("rollback-audit"),
    )
    assert len(window.stable_key()) == 32
    with pytest.raises(W09TypeError, match="teacher call"):
        replace(window, teacher_call_count=1)
    statuses = tuple(
        (
            key,
            W09ResultState.FAIL
            if key == W09_DIMENSION_KEYS[0]
            else W09ResultState.PASS
            if key in W09_DIMENSION_KEYS
            else W09ResultState.NE,
        )
        for key in W09_ALL_DIMENSION_KEYS
    )
    ablation = W09AblationResult(
        W09_ABLATION_KEYS[0],
        W09_DIMENSION_KEYS[0],
        1,
        statuses,
        W09ResultState.PASS,
    )
    assert ablation.status is W09ResultState.PASS
    wall_statuses = tuple(
        (
            key,
            W09ResultState.PASS
            if key in W09_DIMENSION_KEYS
            else W09ResultState.NE,
        )
        for key in W09_ALL_DIMENSION_KEYS
    )
    wall = W09AblationResult(
        W09_ABLATION_KEYS[-1],
        W09_ALL_DIMENSION_KEYS[-1],
        1,
        wall_statuses,
        W09ResultState.NE,
    )
    assert wall.status is W09ResultState.NE
    assert W09StopDecision("RESOLVED", "NONE", resource, 1).publication_allowed == 1
    base = _sha("rollback-base")
    rollback = W09RollbackReceipt("AFTER_PARTIAL_SHARD", base, _sha("preview"), base, 0)
    assert rollback.restored_identity == base
    clone = W09CloneReceipt(
        _sha("source"),
        _sha("clone"),
        _sha("before"),
        _sha("after"),
        1,
        0,
    )
    assert clone.source_write_count == 0
