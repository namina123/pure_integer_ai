"""W08 external private package 的 synthetic metadata、firewall 与执行顺序。"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_w08_candidate import execute_w08_candidate_once
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
    build_w08_candidate_contract,
    publish_w08_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    build_w08_private_family_documents,
    publish_w08_private_family,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_runtime import (
    W08PrivateEvaluatorRuntimeConfig,
    _validate_roots,
    run_w08_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w08_external_package import (
    W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME,
    W08_EXTERNAL_PRIVATE_PACKAGE_V1,
    W08ExternalPrivateBinding,
    W08ExternalPrivatePackageError,
    build_w08_external_private_manifest,
    read_w08_external_private_binding,
    read_w08_external_private_manifest,
    validate_w08_external_private_package_metadata,
    write_w08_external_private_manifest,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_INFERENCE_PAYLOAD_KINDS,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import W08RuntimeConfig


ROOT = Path(__file__).resolve().parents[1]


def _key(*values: int) -> StableRecordKey:
    return StableRecordKey(values)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def synthetic_records():
    """从 public train contract 运行时构造五类 held-out records。"""
    contract = open_w08_frozen_contract(ROOT)
    payload = W08PayloadFirewall.open(
        ROOT, contract, make_w08_request(contract)
    ).read_training_payload()
    evidence_by_observation = {
        item.observation_key: item for item in payload.teacher_evidence
    }
    observations = []
    labels = []
    seen: set[str] = set()
    for original in payload.observations:
        if original.payload_kind in seen:
            continue
        seen.add(original.payload_kind)
        ordinal = len(observations) + 1
        observation = replace(
            original,
            stable_key=_key(8909, 1, ordinal),
            split="held_out",
            dedup_cluster_key=_key(8909, 2, ordinal),
            content_group_key=_key(8909, 3, ordinal),
            template_group_key=_key(8909, 4, ordinal),
            shape_group_key=_key(8909, 5, ordinal),
            logical_order=ordinal,
        )
        evidence = evidence_by_observation[original.stable_key]
        typed = evidence.typed_evidence.to_value()
        if observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
            expected_state = "TRUE"
            expected_payload = {
                "definitive_truth_authoritative": typed[
                    "definitive_truth_authoritative"
                ],
                "raw_observation_sha256": typed["raw_observation_sha256"],
                "source_binding_required": 1,
            }
        else:
            expected_state = typed.get("expected_state", "TRUE")
            expected_payload = typed.get("expected_payload", typed)
        labels.append(EvaluatorLabelRecord(
            1,
            1,
            1,
            observation.dataset_key,
            _key(8909, 6, ordinal),
            _key(8909, 7, ordinal),
            observation.stable_key,
            _key(8909, 8, ordinal),
            expected_state,
            CanonicalJsonObject.from_value(expected_payload),
            160,
            1,
            "W-08",
            _key(8909, 9),
        ))
        observations.append(observation)
    assert seen == set(W08_INFERENCE_PAYLOAD_KINDS)
    return tuple(observations), tuple(labels)


@pytest.fixture
def external_tmp_path():
    with tempfile.TemporaryDirectory(
        prefix="w08-external-package-test-", dir=ROOT.parent
    ) as value:
        yield Path(value)


@pytest.fixture(scope="module")
def synthetic_candidate():
    with tempfile.TemporaryDirectory(
        prefix="w08-external-candidate-test-", dir=ROOT.parent
    ) as value:
        root = Path(value) / "candidate"
        contract = build_w08_candidate_contract(
            ROOT, current_public_head_commit_sha1=_head()
        )
        _, contract_sha = publish_w08_candidate_contract_freeze(
            ROOT, root, contract
        )
        run_root = root / "host"
        result = execute_w08_candidate_once(
            ROOT,
            root,
            config=W08RuntimeConfig(
                ROOT,
                run_root,
                run_root / "coordinator.sqlite",
                worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
                mode=W08_CANDIDATE_FORMAL_MODE,
            ),
            contract=contract,
            candidate_contract_sha256=contract_sha,
        )
        yield {
            "contract_sha": contract_sha,
            "guard_sha": result[5],
            "host_sha": result[3],
            "root": root,
            "seal_sha": result[7],
        }


def _write_package(root: Path, records):
    observations, labels = records
    payload_root = root / "payload"
    payload_root.mkdir()
    clusters = tuple(item.dedup_cluster_key for item in observations)
    observation_identity = write_record_artifact(
        observations,
        payload_root,
        ArtifactWriteSpec(
            "observation",
            "observation",
            "observations/held_out.jsonl.gz",
            "held_out",
            "CC0-1.0",
            clusters,
        ),
    )
    label_identity = write_record_artifact(
        labels,
        payload_root,
        ArtifactWriteSpec(
            "evaluator_label",
            "evaluator",
            "owners/evaluator/held_out_labels.jsonl.gz",
            "held_out",
            "CC0-1.0",
            clusters,
        ),
    )
    cluster_commitment = evidence_commitment([
        list(item.components) for item in clusters
    ])
    manifest = build_w08_external_private_manifest((
        W08ExternalPrivateBinding(
            "SYNTHETIC-ROTATION-A",
            observation_identity,
            tuple(sorted(W08_INFERENCE_PAYLOAD_KINDS)),
            cluster_commitment,
        ),
        W08ExternalPrivateBinding(
            "SYNTHETIC-ROTATION-A",
            label_identity,
            (),
            cluster_commitment,
        ),
    ))
    manifest_path, manifest_sha = write_w08_external_private_manifest(
        payload_root, manifest
    )
    return payload_root, manifest_path, manifest_sha, manifest


def _publish_external_family(root: Path, candidate, manifest):
    documents = build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256=candidate["contract_sha"],
        candidate_guard_sha256=candidate["guard_sha"],
        candidate_host_sha256=candidate["host_sha"],
        candidate_seal_sha256=candidate["seal_sha"],
        evaluator_public_head_commit_sha1=_head(),
        nonce=(8, 41, 73),
        external_package_manifest=manifest,
    )
    family_root = root / "family"
    _, freeze_sha = publish_w08_private_family(
        family_root,
        documents,
        forbidden_roots=(ROOT, candidate["root"], root / "payload"),
    )
    return family_root, freeze_sha


def test_manifest_freezes_metadata_before_any_payload_read(
    external_tmp_path, synthetic_records, monkeypatch
):
    payload_root, manifest_path, manifest_sha, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    assert manifest.package_version == W08_EXTERNAL_PRIVATE_PACKAGE_V1
    assert manifest.sha256() == manifest_sha
    assert manifest.payload_kind_inventory == tuple(
        sorted(W08_INFERENCE_PAYLOAD_KINDS)
    )
    assert len(manifest.bindings) == 2

    def forbidden_payload_read(*_args, **_kwargs):
        raise AssertionError("metadata phase read payload")

    import pure_integer_ai.experiments.ph2_w08_external_package as package_module

    monkeypatch.setattr(
        package_module, "read_record_artifact", forbidden_payload_read
    )
    loaded = read_w08_external_private_manifest(
        payload_root, manifest_path, expected_sha256=manifest_sha
    )
    validate_w08_external_private_package_metadata(payload_root, loaded)
    documents = build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256="1" * 64,
        candidate_guard_sha256="2" * 64,
        candidate_host_sha256="3" * 64,
        candidate_seal_sha256="4" * 64,
        evaluator_public_head_commit_sha1=_head(),
        external_package_manifest=loaded,
    )
    family_root = external_tmp_path / "metadata-family"
    freeze, _ = publish_w08_private_family(
        family_root,
        documents,
        forbidden_roots=(ROOT, payload_root),
    )
    freeze_value = json.loads(freeze.read_bytes())
    assert freeze_value["private_payload_reads"] == 0
    assert freeze_value["resource_limits"]
    assert freeze_value["external_private_package"] == {
        "manifest_sha256": manifest_sha,
        "package_commitment": manifest.package_commitment,
    }


def test_path_owner_kind_and_duplicate_firewalls(
    external_tmp_path, synthetic_records
):
    _, _, _, manifest = _write_package(external_tmp_path, synthetic_records)
    observation = next(
        item for item in manifest.bindings
        if item.identity.owner_kind == "observation"
    )
    label = next(
        item for item in manifest.bindings
        if item.identity.owner_kind == "evaluator"
    )
    with pytest.raises(Exception):
        replace(observation.identity, relative_path="../escape.jsonl.gz")
    with pytest.raises(Exception):
        replace(observation.identity, relative_path="/absolute/payload.jsonl.gz")
    with pytest.raises(W08ExternalPrivatePackageError, match="owner/split/kind"):
        replace(
            observation,
            identity=replace(observation.identity, split="train"),
        )
    with pytest.raises(W08ExternalPrivatePackageError, match="owner/split/kind"):
        replace(
            observation,
            identity=replace(observation.identity, owner_kind="evaluator"),
        )
    with pytest.raises(W08ExternalPrivatePackageError, match="owner/split/kind"):
        replace(
            observation,
            identity=replace(
                observation.identity, record_kind="evaluator_label"
            ),
        )
    with pytest.raises(W08ExternalPrivatePackageError, match="不得携带"):
        replace(label, payload_kind_inventory=(W08_INFERENCE_PAYLOAD_KINDS[0],))
    with pytest.raises(W08ExternalPrivatePackageError, match="重复"):
        build_w08_external_private_manifest((*manifest.bindings, observation))
    with pytest.raises(W08ExternalPrivatePackageError, match="derived commitment"):
        replace(manifest, case_commitment="f" * 64)


@pytest.mark.parametrize("field", ("transport_sha256", "record_count"))
def test_hash_and_count_drift_fail_closed(
    external_tmp_path, synthetic_records, field
):
    payload_root, _, _, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    observation = next(
        item for item in manifest.bindings
        if item.identity.owner_kind == "observation"
    )
    value = "0" * 64 if field == "transport_sha256" else (
        observation.identity.record_count + 1
    )
    drifted = replace(
        observation,
        identity=replace(observation.identity, **{field: value}),
    )
    with pytest.raises(W08ExternalPrivatePackageError, match="transport/hash/count"):
        read_w08_external_private_binding(payload_root, drifted)


def test_reparse_path_is_rejected(
    external_tmp_path, synthetic_records, monkeypatch
):
    payload_root, _, _, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    observation = next(
        item for item in manifest.bindings
        if item.identity.owner_kind == "observation"
    )
    target_name = Path(observation.identity.relative_path).name
    import pure_integer_ai.experiments.ph2_w08_external_package as package_module

    original = package_module._is_reparse
    monkeypatch.setattr(
        package_module,
        "_is_reparse",
        lambda path: path.name == target_name or original(path),
    )
    with pytest.raises(W08ExternalPrivatePackageError, match="link/reparse"):
        validate_w08_external_private_package_metadata(payload_root, manifest)


def test_unbound_payload_file_is_rejected(
    external_tmp_path, synthetic_records
):
    payload_root, _, _, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    (payload_root / "unbound.bin").write_bytes(b"not-authorized")
    with pytest.raises(W08ExternalPrivatePackageError, match="inventory"):
        validate_w08_external_private_package_metadata(payload_root, manifest)


def test_external_runtime_orders_all_inference_before_label_and_passes(
    external_tmp_path,
    synthetic_records,
    synthetic_candidate,
    monkeypatch,
):
    payload_root, manifest_path, _, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    family_root, freeze_sha = _publish_external_family(
        external_tmp_path, synthetic_candidate, manifest
    )
    import pure_integer_ai.experiments.ph2_w08_evaluator_runtime as runtime_module

    head = _head()
    monkeypatch.setattr(runtime_module, "_git_state", lambda _root: (head, ""))
    inference_count = 0
    label_read_counts = []
    original_infer = runtime_module.W08CandidateInferenceAdapter.infer
    original_read = runtime_module.read_w08_external_private_binding

    def counted_infer(self, *args, **kwargs):
        nonlocal inference_count
        inference_count += 1
        return original_infer(self, *args, **kwargs)

    def ordered_read(root, binding):
        if binding.identity.owner_kind == "evaluator":
            label_read_counts.append(inference_count)
        return original_read(root, binding)

    monkeypatch.setattr(
        runtime_module.W08CandidateInferenceAdapter, "infer", counted_infer
    )
    monkeypatch.setattr(
        runtime_module, "read_w08_external_private_binding", ordered_read
    )
    result = run_w08_private_evaluation_once(
        W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            synthetic_candidate["root"],
            family_root,
            family_root / "execution",
            private_payload_root=payload_root,
            package_manifest=manifest_path,
        ),
        family_freeze_sha256=freeze_sha,
    )
    aggregate = json.loads(result.aggregate_path.read_bytes())
    expected_invocations = len(synthetic_records[0]) * 5 * 6
    assert inference_count == expected_invocations
    assert label_read_counts == [expected_invocations]
    assert result.status == aggregate["status"] == "PASS"
    assert all(item["status"] == "PASS" for item in aggregate["dimension_results"])
    assert all(item["status"] == "PASS" for item in aggregate["ablation_results"])
    assert aggregate["open_generation"]["status"] == "PASS"
    assert aggregate["lc16"]["status"] == "PASS"
    assert aggregate["infrastructure"]["inference_before_label_reads"] == 1
    assert aggregate["infrastructure"]["private_payload_writes"] == 0
    assert result.dump_sha256 is not None
    encoded = json.dumps(aggregate, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "relative_path",
        "typed_payload",
        "expected_payload",
        "raw_observation",
    ):
        assert forbidden not in encoded
    with pytest.raises(RuntimeError, match="不可重跑|aggregate 已存在"):
        run_w08_private_evaluation_once(
            W08PrivateEvaluatorRuntimeConfig(
                ROOT,
                synthetic_candidate["root"],
                family_root,
                family_root / "execution",
                private_payload_root=payload_root,
                package_manifest=manifest_path,
            ),
            family_freeze_sha256=freeze_sha,
        )


def test_external_runtime_detects_payload_write_isolation_drift(
    external_tmp_path,
    synthetic_records,
    synthetic_candidate,
    monkeypatch,
):
    payload_root, manifest_path, _, manifest = _write_package(
        external_tmp_path, synthetic_records
    )
    family_root, freeze_sha = _publish_external_family(
        external_tmp_path, synthetic_candidate, manifest
    )
    import pure_integer_ai.experiments.ph2_w08_evaluator_runtime as runtime_module

    head = _head()
    monkeypatch.setattr(runtime_module, "_git_state", lambda _root: (head, ""))
    original_infer = runtime_module.W08CandidateInferenceAdapter.infer
    touched = False
    observation_binding = next(
        item for item in manifest.bindings
        if item.identity.owner_kind == "observation"
    )
    observation_path = payload_root / Path(
        *observation_binding.identity.relative_path.split("/")
    )

    def touching_infer(self, *args, **kwargs):
        nonlocal touched
        outcome = original_infer(self, *args, **kwargs)
        if not touched:
            stat = observation_path.stat()
            os.utime(
                observation_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000),
            )
            touched = True
        return outcome

    monkeypatch.setattr(
        runtime_module.W08CandidateInferenceAdapter, "infer", touching_infer
    )
    result = run_w08_private_evaluation_once(
        W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            synthetic_candidate["root"],
            family_root,
            family_root / "execution",
            private_payload_root=payload_root,
            package_manifest=manifest_path,
        ),
        family_freeze_sha256=freeze_sha,
    )
    aggregate = json.loads(result.aggregate_path.read_bytes())
    assert result.status == "NE"
    assert aggregate["failure_phase"] == "INTEGRITY"
    assert aggregate["write_counts"]["label_writes"] == 1
    assert aggregate["infrastructure"]["private_payload_writes"] == 1


def test_manifest_path_must_be_inside_payload_root(
    external_tmp_path, synthetic_records
):
    payload_root, _, manifest_sha, _ = _write_package(
        external_tmp_path, synthetic_records
    )
    outside = external_tmp_path / W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME
    outside.write_bytes((payload_root / W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME).read_bytes())
    with pytest.raises(W08ExternalPrivatePackageError, match="越界"):
        read_w08_external_private_manifest(
            payload_root, outside, expected_sha256=manifest_sha
        )


def test_payload_root_must_not_overlap_public_and_legacy_fault_position_remains(
    external_tmp_path,
):
    legacy = W08PrivateEvaluatorRuntimeConfig(
        ROOT,
        external_tmp_path / "candidate",
        external_tmp_path / "family",
        external_tmp_path / "family" / "execution",
        "CANDIDATE_VERIFY",
    )
    assert legacy.fault_phase == "CANDIDATE_VERIFY"
    with pytest.raises(RuntimeError, match="root 未隔离"):
        _validate_roots(W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            external_tmp_path / "candidate",
            external_tmp_path / "family",
            external_tmp_path / "family" / "execution",
            private_payload_root=ROOT / "data",
            package_manifest=ROOT / "data" / (
                W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME
            ),
        ))
