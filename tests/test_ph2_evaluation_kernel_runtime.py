"""Synthetic one-shot tests for the generic evaluation runtime and firewall."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
    V2_STAGE_EVALUATION_POLICIES,
    read_v2_evaluator_boundary_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
    V2WriteAccount,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationThreshold,
    build_evaluation_manifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.identity import (
    evaluation_kernel_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.owner_receipt import (
    EvaluationOwnerBinding,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
    EvaluationPluginOutcome,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.private_io import (
    EvaluationFileIdentity,
    evaluation_file_inventory_sha256,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.preflight import (
    EvaluationPreflightCheck,
    build_formal_ready_receipt,
    build_preflight_layer,
    build_transport_preflight_layer,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationDimensionResult,
    EvaluationKernelContractError,
    EvaluationResultSet,
    EvaluationRunAudit,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    EVALUATION_FAILURE_SEAL,
    EVALUATION_GUARD_AVAILABLE,
    EVALUATION_GUARD_CONSUMED,
    EVALUATION_RUNTIME_RECEIPT,
    preflight_evaluation_family,
    publish_evaluation_family,
    publish_evaluation_family_formal_ready,
    run_evaluation_family_once,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.source_binding import (
    EvaluationSourceBinding,
    EvaluationSourceSlice,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _sha(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _write_private_file(root: Path, relative: str, payload: bytes) -> None:
    target = root / Path(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _files(private_root: Path) -> tuple[EvaluationFileIdentity, ...]:
    rows = (
        ("PRIVATE_SOURCE", "", "source_ref", "source/source_refs.jsonl.gz", b"sources\n", 2),
        ("PRIVATE_HELD_OUT_OBSERVATION", "held_out", "observation",
         "observations/held_out.jsonl.gz", b"observations\n", 2),
        ("PRIVATE_HELD_OUT_LABEL", "held_out", "evaluator_label",
         "evaluator/held_out.labels.jsonl.gz", b"labels\n", 2),
    )
    result = []
    for layout, split, kind, relative, payload, count in rows:
        _write_private_file(private_root, relative, payload)
        result.append(EvaluationFileIdentity(
            layout, split, kind, relative, len(payload), _sha(payload),
            len(payload), _sha(b"content:" + payload), count,
        ))
    return tuple(result)


def _roots(tmp_path: Path) -> tuple[V2PhysicalRoots, tuple[EvaluationFileIdentity, ...]]:
    tmp_path.mkdir(parents=True)
    paths = [tmp_path / name for name in (
        "candidate", "teacher", "dev", "shadow", "private", "ledger")]
    for path in paths:
        path.mkdir()
    files = _files(paths[4])
    return V2PhysicalRoots.from_paths(*paths), files


def _manifest(files: tuple[EvaluationFileIdentity, ...], declaration):
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-03")
    source = EvaluationSourceBinding(
        _sha("source-contract"),
        (EvaluationSourceSlice(
            "SYNTHETIC", "held_out", "cluster-a", 1, 2, 2,
            _sha("source-refs")),),
        _sha("all-source-refs"),
    )
    owner = EvaluationOwnerBinding(
        "PH2_V2_PRIVATE_EVALUATOR", _sha("owner-receipt"),
        _sha("owner-metadata"), evaluation_file_inventory_sha256(files),
        _sha("payload"), _sha("cases"), _sha("labels"), _sha("clusters"),
        2, 2, 2, 2,
    )
    support = tuple(
        key for key in policy.hard_conjunct_keys
        if key not in policy.bearing_dimension_keys
        and key != policy.generation_hard_conjunct_key)
    return build_evaluation_manifest(
        release_key="PH2-D03-V2",
        stage_key="W-03",
        family_key="W03-GENERIC-SYNTHETIC",
        revision_key="R1",
        public_head_sha1=_sha1("public-head"),
        kernel_semantic_sha256=evaluation_kernel_semantic_sha256(REPOSITORY),
        stage_manifest_sha256=_sha("stage"),
        plugin=declaration,
        source_binding=source,
        owner_binding=owner,
        candidate_artifact_sha256=_sha("candidate"),
        resource_budget=V2EvaluatorResourceBudget(
            16, 1_000_000, 1_000_000, 10_000, 10_000, 1),
        consumed_lineage_sha256=_sha("lineage"),
        bearing_dimension_keys=policy.bearing_dimension_keys,
        generation_hard_conjunct_key=policy.generation_hard_conjunct_key,
        support_dimension_keys=support,
        thresholds=tuple(
            EvaluationThreshold(key, 1, 1) for key in policy.hard_conjunct_keys),
    )


def _declaration() -> EvaluationPluginDeclaration:
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-03")
    return EvaluationPluginDeclaration(
        "W03-GENERIC-SYNTHETIC", "V1", "W-03",
        "ph2.w03.generic.synthetic", "evaluate", _sha("plugin"),
        policy.hard_conjunct_keys,
    )


@dataclass(frozen=True, slots=True)
class _SyntheticPlugin:
    declaration: EvaluationPluginDeclaration
    status: str = "PASS"
    raises: int = 0

    def evaluate(self, context, records):
        tuple(records)
        if self.raises:
            raise RuntimeError("synthetic private detail must not be published")
        policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                      if item.stage_key == "W-03")
        support = tuple(
            key for key in policy.hard_conjunct_keys
            if key not in policy.bearing_dimension_keys
            and key != policy.generation_hard_conjunct_key)
        roles = (
            *("BEARING" for _ in policy.bearing_dimension_keys),
            "GENERATION", *("SUPPORT" for _ in support),
        )
        results = []
        for index, (key, role) in enumerate(zip(
                policy.hard_conjunct_keys, roles, strict=True)):
            status = self.status if index == 0 else "PASS"
            counts = {
                "PASS": (1, 0, 0, 0), "FAIL": (0, 1, 0, 0),
                "NE": (0, 0, 1, 0), "BLOCKED": (0, 0, 0, 1),
            }[status]
            results.append(EvaluationDimensionResult(
                key, role, status, 1, *counts, _sha(f"{key}:{status}")))
        return EvaluationPluginOutcome(
            EvaluationResultSet(tuple(results)),
            EvaluationRunAudit(
                "COMPLETE", 2, 2, 6, 6, 128, 512, 3, V2WriteAccount()),
        )


def _loader(context, authorized):
    assert context.run_id == 1
    assert len(authorized) == 3
    return ("source-first", "pair-1", "pair-2")


def _ready(boundary, roots, family, files, plugin):
    manifest, authorized = preflight_evaluation_family(
        boundary, roots, family, files, plugin)
    layers = []
    for layer_key in ("P0", "P1", "P2"):
        layers.append(build_preflight_layer(layer_key, (
            EvaluationPreflightCheck(
                f"{layer_key}_SYNTHETIC_PASS", "PASS", _sha(layer_key)),
        )))
    p3 = build_transport_preflight_layer(manifest, authorized)
    receipt = build_formal_ready_receipt(
        manifest, *layers, p3,
        public_dev_status="PASS", public_shadow_a_status="PASS",
        public_shadow_b_or_metamorphic_status="PASS", family_pushed=1,
        publication_evidence_sha256=_sha("publication"))
    publish_evaluation_family_formal_ready(family, receipt)
    return manifest, authorized


def test_generic_runtime_pass_consumes_guard_and_publishes_receipt(tmp_path) -> None:
    roots, files = _roots(tmp_path / "roots")
    plugin = _SyntheticPlugin(_declaration())
    manifest = _manifest(files, plugin.declaration)
    family = publish_evaluation_family(manifest, tmp_path / "family-pass")
    boundary = read_v2_evaluator_boundary_contract(REPOSITORY)
    preflight_manifest, authorized = _ready(
        boundary, roots, family, files, plugin)
    assert preflight_manifest == manifest
    assert len(authorized) == 3
    publication = run_evaluation_family_once(
        boundary, roots, family, files, plugin, _loader)
    assert publication.aggregate.status == "PASS"
    assert (family / EVALUATION_GUARD_CONSUMED).is_file()
    assert not (family / EVALUATION_GUARD_AVAILABLE).exists()
    assert (family / EVALUATION_RUNTIME_RECEIPT).is_file()
    assert not (family / EVALUATION_FAILURE_SEAL).exists()
    with pytest.raises(EvaluationKernelContractError):
        run_evaluation_family_once(boundary, roots, family, files, plugin, _loader)


@pytest.mark.parametrize("status", ("FAIL", "NE", "BLOCKED"))
def test_generic_runtime_non_pass_seals_without_receipt(tmp_path, status: str) -> None:
    roots, files = _roots(tmp_path / f"roots-{status}")
    plugin = _SyntheticPlugin(_declaration(), status=status)
    family = publish_evaluation_family(
        _manifest(files, plugin.declaration), tmp_path / f"family-{status}")
    boundary = read_v2_evaluator_boundary_contract(REPOSITORY)
    _ready(boundary, roots, family, files, plugin)
    publication = run_evaluation_family_once(
        boundary, roots, family, files, plugin, _loader)
    assert publication.aggregate.status == status
    assert publication.failure_seal is not None
    assert (family / EVALUATION_FAILURE_SEAL).is_file()
    assert not (family / EVALUATION_RUNTIME_RECEIPT).exists()


def test_plugin_exception_becomes_blocked_without_false_zero_audit(tmp_path) -> None:
    roots, files = _roots(tmp_path / "roots-block")
    plugin = _SyntheticPlugin(_declaration(), raises=1)
    family = publish_evaluation_family(
        _manifest(files, plugin.declaration), tmp_path / "family-block")
    boundary = read_v2_evaluator_boundary_contract(REPOSITORY)
    _ready(boundary, roots, family, files, plugin)
    publication = run_evaluation_family_once(
        boundary, roots, family, files, plugin, _loader)
    assert publication.aggregate.status == "BLOCKED"
    assert publication.aggregate.run_audit.audit_state == "BLOCKED_UNAVAILABLE"
    assert publication.aggregate.run_audit.write_account is None


def test_transport_drift_fails_before_guard_consumption(tmp_path) -> None:
    roots, files = _roots(tmp_path / "roots-drift")
    plugin = _SyntheticPlugin(_declaration())
    family = publish_evaluation_family(
        _manifest(files, plugin.declaration), tmp_path / "family-drift")
    private_file = roots.private_evaluator / "observations" / "held_out.jsonl.gz"
    private_file.write_bytes(b"changed\n")
    with pytest.raises(Exception):
        run_evaluation_family_once(
            read_v2_evaluator_boundary_contract(REPOSITORY),
            roots, family, files, plugin, _loader)
    assert (family / EVALUATION_GUARD_AVAILABLE).is_file()
    assert not (family / EVALUATION_GUARD_CONSUMED).exists()


def test_missing_formal_ready_refuses_run_without_consuming_guard(tmp_path) -> None:
    roots, files = _roots(tmp_path / "roots-not-ready")
    plugin = _SyntheticPlugin(_declaration())
    family = publish_evaluation_family(
        _manifest(files, plugin.declaration), tmp_path / "family-not-ready")
    with pytest.raises(EvaluationKernelContractError):
        run_evaluation_family_once(
            read_v2_evaluator_boundary_contract(REPOSITORY),
            roots, family, files, plugin, _loader)
    assert (family / EVALUATION_GUARD_AVAILABLE).is_file()
    assert not (family / EVALUATION_GUARD_CONSUMED).exists()
