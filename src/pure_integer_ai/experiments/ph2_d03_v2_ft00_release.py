"""PH2-D03-V2 FT00-07 public release gate."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_CONTRACT_PATH,
    V2_LOGICAL_SHARD_COUNT,
    V2_RELEASE_KEY,
    V2RunIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_STAGES,
    V2_OWNER_ROOT_POLICIES,
    V2_STAGE_EVALUATION_POLICIES,
    read_v2_evaluator_boundary_contract,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_catalog import (
    read_v2_successor_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2GenericTrainer,
    V2PackRegistry,
    V2TrainPackStream,
)
from pure_integer_ai.experiments.ph2_d03_v2_scale_baseline import (
    read_ft00_05_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_source_adapters import (
    read_v2_source_adapter_audit,
)
from pure_integer_ai.experiments.ph2_d03_v2_streaming import (
    V2LogicalShardPlan,
    V2StreamReader,
    V2StreamingError,
)


FT00_RELEASE_GATE_KIND = "PH2_D03_V2_FT00_RELEASE_GATE"
FT00_RELEASE_GATE_VERSION = 1
FT00_RELEASE_GATE_PATH = (
    "data/ph2/manifests/d03_v2/ph2_d03_v2_ft00_release_gate_v1.json"
)
FT00_CHECK_ORDER = (
    "v2_contract",
    "v2_manifest",
    "generic_trainer",
    "stream_reader",
    "checkpoint_resume",
    "worker_equivalence",
    "p0_p1_profile",
    "evaluator_isolation",
    "old_v1_unchanged",
)
FT00_SOURCE_AUDIT_SHA256 = (
    "4e28fa2e09962ff5046f1ed92051900d007bf90c065d274637570f2a68a5ddbe"
)
FT00_SCALE_REPORT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_ft00_05_scale_baseline_v1.json"
)


class FT00ReleaseGateError(RuntimeError):
    """FT00 public release evidence is incomplete or drifted."""


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise FT00ReleaseGateError(f"{where} is not a lowercase SHA-256")
    return value


def _file_identity(root: Path, relative: str) -> tuple[int, str]:
    path = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT00ReleaseGateError("FT00 gate public artifact is missing")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _evidence(check_key: str, values: object) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "check_key": check_key,
        "values": values,
    })).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT00GateCheck:
    """One public gate check with no path or payload detail."""

    check_key: str
    status: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.check_key not in FT00_CHECK_ORDER or self.status != "PASS":
            raise FT00ReleaseGateError("FT00 gate check is not PASS")
        _sha256(self.evidence_sha256, where="FT00 gate evidence")

    def to_dict(self) -> dict[str, str]:
        return {
            "check_key": self.check_key,
            "evidence_sha256": self.evidence_sha256,
            "status": self.status,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT00ReleaseGateReport:
    """Immutable conjunction proving FT00 infrastructure readiness only."""

    artifact_kind: str
    format_version: int
    release_key: str
    check_order: tuple[str, ...]
    checks: tuple[FT00GateCheck, ...]
    formal_training_runs: int
    formal_private_evaluation_runs: int
    private_payload_reads: int
    candidate_writes: int
    teacher_calls: int
    ft00_complete: int
    next_stage: str
    status: str

    def __post_init__(self) -> None:
        if (self.artifact_kind != FT00_RELEASE_GATE_KIND
                or type(self.format_version) is not int
                or self.format_version != FT00_RELEASE_GATE_VERSION
                or self.release_key != V2_RELEASE_KEY
                or self.check_order != FT00_CHECK_ORDER
                or tuple(item.check_key for item in self.checks) != FT00_CHECK_ORDER
                or any(item.status != "PASS" for item in self.checks)):
            raise FT00ReleaseGateError("FT00 gate checks are incomplete")
        if any(type(value) is not int or value != 0 for value in (
                self.formal_training_runs, self.formal_private_evaluation_runs,
                self.private_payload_reads, self.candidate_writes,
                self.teacher_calls)):
            raise FT00ReleaseGateError("FT00 gate must not run training/evaluator")
        if (type(self.ft00_complete) is not int or self.ft00_complete != 1
                or self.next_stage != "W-02"
                or self.status != "FT00_COMPLETE"):
            raise FT00ReleaseGateError("FT00 gate completion state is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "candidate_writes": self.candidate_writes,
            "check_order": list(self.check_order),
            "checks": [item.to_dict() for item in self.checks],
            "formal_private_evaluation_runs": self.formal_private_evaluation_runs,
            "formal_training_runs": self.formal_training_runs,
            "format_version": self.format_version,
            "ft00_complete": self.ft00_complete,
            "next_stage": self.next_stage,
            "private_payload_reads": self.private_payload_reads,
            "release_key": self.release_key,
            "status": self.status,
            "teacher_calls": self.teacher_calls,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _public_ready_paths(root: Path) -> tuple[str, ...]:
    audit = read_v2_source_adapter_audit(root)
    if audit.sha256() != FT00_SOURCE_AUDIT_SHA256:
        raise FT00ReleaseGateError("FT00 source adapter audit identity drifted")
    paths = tuple(item.v2_manifest_relative_path
                  for item in audit.entries if item.status == "READY")
    if len(paths) != 3 or len(set(paths)) != 3:
        raise FT00ReleaseGateError("FT00 ready v2 pack inventory is incomplete")
    return paths


def _check_v2_contract(root: Path) -> str:
    contract = read_v2_evaluator_boundary_contract(root)
    successor = read_v2_successor_contract(root)
    if contract.successor_contract_sha256 != successor.sha256():
        raise FT00ReleaseGateError("FT00 successor contract file identity drifted")
    return _evidence("v2_contract", {
        "boundary": contract.sha256(),
        "successor": contract.successor_contract_sha256,
    })


def _check_v2_manifest(root: Path) -> str:
    paths = _public_ready_paths(root)
    registry = V2PackRegistry.from_manifest_paths(root, paths)
    snapshot = registry.snapshot()
    training_input_count = (
        snapshot.source_ref_count
        + dict(snapshot.observation_counts).get("train", 0)
        + snapshot.teacher_evidence_count
    )
    if (snapshot.pack_count != 3 or snapshot.total_record_count != 57
            or training_input_count != 33):
        raise FT00ReleaseGateError("FT00 public manifest counts drifted")
    return _evidence("v2_manifest", snapshot.to_dict())


def _check_generic_trainer(root: Path) -> str:
    registry = V2PackRegistry.from_manifest_paths(root, _public_ready_paths(root))
    plan = registry.train_plan("W-03", scale_key="P0")
    streams = tuple(
        V2TrainPackStream(
            entry.pack_key,
            lambda entry=entry: (
                item.to_dict() for item in V2StreamReader(root, entry).iter_records("teacher")
            ),
        )
        for entry in registry.entries if "W-03" in entry.w_stages
    )
    result = V2GenericTrainer().validate_train_streams(plan, streams)
    if (result.source_ref_count != plan.source_ref_count
            or result.observation_count != plan.observation_count
            or result.teacher_evidence_count != plan.teacher_evidence_count
            or result.candidate_writes != 0
            or result.core_writes != 0 or result.teacher_calls != 0):
        raise FT00ReleaseGateError("FT00 generic trainer public replay drifted")
    return _evidence("generic_trainer", {
        "input_commitment": result.input_commitment,
        "plan": plan.to_dict(),
    })


def _check_stream_and_checkpoint(root: Path) -> tuple[str, str]:
    registry = V2PackRegistry.from_manifest_paths(root, _public_ready_paths(root))
    entry = next(item for item in registry.entries if "W-03" in item.w_stages)
    reader = V2StreamReader(root, entry)
    records = tuple(reader.iter_records("candidate"))
    if not records:
        raise FT00ReleaseGateError("FT00 candidate public stream is empty")
    tuple(reader.iter_records("dev", split="dev"))
    try:
        tuple(reader.iter_records("private_evaluator"))
    except V2StreamingError:
        private_blocked = 1
    else:
        private_blocked = 0
    if private_blocked != 1:
        raise FT00ReleaseGateError("FT00 stream reader exposed private view")
    stream_evidence = _evidence("stream_reader", {
        "entry": entry.manifest_sha256,
        "candidate_count": len(records),
        "private_blocked": private_blocked,
    })

    shard_plan = V2LogicalShardPlan()
    selected_shard = shard_plan.shard_for(records[0].stable_key.components)
    selected = tuple(
        item.stable_key.components for item in records
        if shard_plan.shard_for(item.stable_key.components) == selected_shard)
    if not selected:
        raise FT00ReleaseGateError("FT00 checkpoint shard selection is empty")
    run = V2RunIdentity(
        V2_RELEASE_KEY, "W-03", "P0", 1, V2_LOGICAL_SHARD_COUNT,
        entry.manifest_sha256, "",
    )
    checkpoint = reader.checkpoint(
        run, owner_key="PH2_V2_CANDIDATE", shard_index=selected_shard,
        cursor_record_key=selected[0], source_state_sha256=entry.manifest_sha256,
    )
    resumed = tuple(
        item.stable_key.components
        for window in reader.iter_from_checkpoint(
            checkpoint, run, owner_key="PH2_V2_CANDIDATE",
            source_state_sha256=entry.manifest_sha256, window_size=8)
        for item in window.records
    )
    if resumed != selected[1:]:
        raise FT00ReleaseGateError("FT00 checkpoint resume is not canonical")
    checkpoint_evidence = _evidence("checkpoint_resume", {
        "checkpoint": checkpoint.sha256(),
        "selected_count": len(selected),
        "resumed_count": len(resumed),
    })
    return stream_evidence, checkpoint_evidence


def _check_worker_equivalence(root: Path) -> str:
    registry = V2PackRegistry.from_manifest_paths(root, _public_ready_paths(root))
    records = tuple(
        (entry.pack_key, item.stable_key.components, canonical_json_bytes(item.to_dict()))
        for entry in registry.entries
        for item in V2StreamReader(root, entry).iter_records("candidate")
    )
    plan = V2LogicalShardPlan()
    digests = []
    for workers in (1, 2, 4):
        buckets: list[list[tuple[tuple[int, ...], tuple[int, ...], bytes]]] = [
            [] for _ in range(workers)
        ]
        for pack_key, record_key, payload in reversed(records):
            shard = plan.shard_for(record_key)
            buckets[shard % workers].append((pack_key, record_key, payload))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            lanes = tuple(executor.map(
                lambda values: tuple(sorted(values)), reversed(buckets)))
        merged = tuple(sorted(item for lane in lanes for item in lane))
        if len(merged) != len(records):
            raise FT00ReleaseGateError("FT00 worker merge lost records")
        digest = hashlib.sha256()
        for pack_key, record_key, payload in merged:
            digest.update(canonical_json_bytes({
                "pack_key": list(pack_key),
                "record_key": list(record_key),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }))
            digest.update(payload)
        digests.append(digest.hexdigest())
    if len(set(digests)) != 1:
        raise FT00ReleaseGateError("FT00 worker semantic digest drifted")
    return _evidence("worker_equivalence", {
        "semantic_digest": digests[0],
        "worker_counts": [1, 2, 4],
    })


def _check_scale(root: Path) -> str:
    report = read_ft00_05_report(root / Path(*PurePosixPath(FT00_SCALE_REPORT_RELATIVE_PATH).parts))
    if (report.status != "PASS" or report.formal_training_runs != 0
            or report.candidate_writes != 0 or report.teacher_calls != 0):
        raise FT00ReleaseGateError("FT00 P0/P1 profile is not a clean PASS")
    return _evidence("p0_p1_profile", {"report": report.sha256()})


def _check_evaluator_isolation(root: Path) -> str:
    contract = read_v2_evaluator_boundary_contract(root)
    if (tuple(item.owner_key for item in contract.owner_root_policies)
            != tuple(item.owner_key for item in V2_OWNER_ROOT_POLICIES)
            or tuple(item.stage_key for item in contract.stage_policies)
            != V2_EVALUATOR_STAGES
            or contract.initial_state["private_payload_reads"] != 0):
        raise FT00ReleaseGateError("FT00 evaluator isolation policy drifted")
    return _evidence("evaluator_isolation", {
        "boundary": contract.sha256(),
        "stage_policy_count": len(V2_STAGE_EVALUATION_POLICIES),
    })


def _check_old_v1_unchanged(root: Path) -> str:
    boundary = read_v2_evaluator_boundary_contract(root)
    successor = read_v2_successor_contract(root)
    successor_size, successor_digest = _file_identity(root, V2_CONTRACT_PATH)
    receipt_size, receipt_digest = _file_identity(
        root, successor.prior_release_receipt.relative_path)
    if (successor_size != boundary.successor_contract.size_bytes
            or successor_digest != boundary.successor_contract.sha256
            or receipt_size != successor.prior_release_receipt.size_bytes
            or receipt_digest != successor.prior_release_receipt.sha256):
        raise FT00ReleaseGateError("FT00 successor parent file drifted")
    return _evidence("old_v1_unchanged", {
        "parent_receipt": receipt_digest,
        "successor_contract": successor_digest,
    })


def run_ft00_release_gate(repository_root: str | Path) -> FT00ReleaseGateReport:
    """Run all bounded public checks and return the immutable FT00 PASS report."""
    root = Path(repository_root).resolve()
    checks_values = {
        "v2_contract": _check_v2_contract(root),
        "v2_manifest": _check_v2_manifest(root),
        "generic_trainer": _check_generic_trainer(root),
    }
    stream_evidence, checkpoint_evidence = _check_stream_and_checkpoint(root)
    checks_values["stream_reader"] = stream_evidence
    checks_values["checkpoint_resume"] = checkpoint_evidence
    checks_values["worker_equivalence"] = _check_worker_equivalence(root)
    checks_values["p0_p1_profile"] = _check_scale(root)
    checks_values["evaluator_isolation"] = _check_evaluator_isolation(root)
    checks_values["old_v1_unchanged"] = _check_old_v1_unchanged(root)
    checks = tuple(FT00GateCheck(key, "PASS", checks_values[key])
                   for key in FT00_CHECK_ORDER)
    return FT00ReleaseGateReport(
        FT00_RELEASE_GATE_KIND, FT00_RELEASE_GATE_VERSION, V2_RELEASE_KEY,
        FT00_CHECK_ORDER, checks, 0, 0, 0, 0, 0, 1, "W-02", "FT00_COMPLETE",
    )


def write_ft00_release_gate(
        report: FT00ReleaseGateReport,
        path: str | Path,
        ) -> Path:
    """Publish the FT00 gate by exclusive or idempotent canonical write."""
    validate_v2_safe_report(report.to_dict())
    return write_immutable_json(report.to_dict(), path)


def read_ft00_release_gate(path: str | Path) -> FT00ReleaseGateReport:
    """Read and strictly validate one immutable FT00 gate report."""
    target = Path(path)
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise FT00ReleaseGateError("FT00 gate newline is invalid")
    value = read_canonical_object(target)
    if value.get("artifact_kind") != FT00_RELEASE_GATE_KIND:
        raise FT00ReleaseGateError("FT00 gate artifact kind drifted")
    required = set(FT00ReleaseGateReport.__dataclass_fields__)
    if set(value) != required or not isinstance(value["checks"], list):
        raise FT00ReleaseGateError("FT00 gate fields are not exact")
    if not isinstance(value["check_order"], list):
        raise FT00ReleaseGateError("FT00 gate check order is not an array")
    checks = []
    for item in value["checks"]:
        if (not isinstance(item, dict)
                or set(item) != {"check_key", "evidence_sha256", "status"}):
            raise FT00ReleaseGateError("FT00 gate check fields are not exact")
        checks.append(FT00GateCheck(
            str(item["check_key"]), str(item["status"]),
            str(item["evidence_sha256"]),
        ))
    return FT00ReleaseGateReport(
        str(value["artifact_kind"]), value["format_version"],
        str(value["release_key"]), tuple(str(item) for item in value["check_order"]),
        tuple(checks), value["formal_training_runs"],
        value["formal_private_evaluation_runs"], value["private_payload_reads"],
        value["candidate_writes"], value["teacher_calls"], value["ft00_complete"],
        str(value["next_stage"]), str(value["status"]),
    )


__all__ = [
    "FT00_CHECK_ORDER", "FT00_RELEASE_GATE_KIND", "FT00_RELEASE_GATE_PATH",
    "FT00ReleaseGateError", "FT00ReleaseGateReport", "FT00GateCheck",
    "read_ft00_release_gate", "run_ft00_release_gate", "write_ft00_release_gate",
]
