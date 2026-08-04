"""W08-09 唯一 private evaluator runtime、故障 NE 与安全发布。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w08_candidate import (
    W08_CANDIDATE_HOST_FREEZE_NAME,
    W08_CANDIDATE_TERMINAL_SEAL_NAME,
)
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_CONTRACT_FREEZE_NAME,
    W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
)
from pure_integer_ai.experiments.ph2_w08_authority import W08_VISIBLE_PACK_KEYS
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_RESOURCE_BUDGET,
    W08FileBinding,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_evaluator import (
    W08PrivateEvaluationPair,
    assess_w08_orthogonal_ablations,
    assess_w08_private_lc16,
    assess_w08_private_open_generation,
    evaluate_w08_private_pairs,
    snapshot_from_w08_outcome,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    W08_EVALUATOR_FAILURE_PHASES,
    W08_EVALUATOR_PHASES,
    W08_PRIVATE_AGGREGATE_NAME,
    W08_PRIVATE_CASE_NAME,
    W08_PRIVATE_CLUSTER_NAME,
    W08_PRIVATE_DUMP_NAME,
    W08_PRIVATE_FAMILY_FREEZE_NAME,
    W08_PRIVATE_FIRST_RUN_GUARD_NAME,
    W08_PRIVATE_LABEL_NAME,
    W08_PRIVATE_INFERENCE_INTERFACE_VERSION,
    W08_PRIVATE_RECOMMENDATION_NAME,
    W08_PRIVATE_SCHEMA_NAME,
    W08_PRIVATE_SOURCE_NAME,
    W08PrivateEvaluationError,
    evidence_commitment,
    public_safe_w08_aggregate,
    strict_sha256,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    consume_w08_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08VisibilityFirewall
from pure_integer_ai.experiments.ph2_w08_inference import (
    W08CandidateInferenceAdapter,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_INFERENCE_PAYLOAD_KINDS,
    W08CandidateInferenceError,
    W08CandidateInferenceOutcome,
    w08_inference_schema_sha256,
)
from pure_integer_ai.experiments.ph2_w08_runtime import (
    load_w08_candidate_inference_state,
    load_w08_public_dump,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import W08RuntimeConfig


class W08EvaluatorInfrastructureError(W08PrivateEvaluationError):
    """private evaluator 的 host、payload、资源或 owner 隔离失败。"""


class W08EvaluatorInjectedFault(W08EvaluatorInfrastructureError):
    """专项验证使用的预注册 evaluator phase 故障。"""


@dataclass(frozen=True)
class W08PrivateEvaluatorRuntimeConfig:
    repository_root: str | Path
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    fault_phase: str | None = None


@dataclass(frozen=True)
class W08PrivateEvaluatorRunResult:
    status: str
    aggregate_path: Path
    aggregate_sha256: str
    recommendation_path: Path | None
    recommendation_sha256: str | None
    family_freeze_sha256: str
    first_run_guard_sha256: str
    dump_sha256: str | None


@dataclass
class _ReadAudit:
    reads_by_path: dict[str, tuple[int, int]]
    payload_gets: int = 0
    payload_bytes: int = 0
    observation_records: int = 0
    label_records: int = 0
    future_payload_reads: int = 0

    def record(self, relative: str, size: int, records: int, owner: str) -> None:
        count, total = self.reads_by_path.get(relative, (0, 0))
        self.reads_by_path[relative] = (count + 1, total + size)
        self.payload_gets += 1
        self.payload_bytes += size
        if owner == "observation":
            self.observation_records += records
        else:
            self.label_records += records


class _W08SanitizedInferenceFailure(RuntimeError):
    """只携带允许进入 public aggregate 的枚举诊断。"""

    def __init__(self, infrastructure: dict[str, object]) -> None:
        super().__init__("W08 Candidate inference failed")
        self.infrastructure = infrastructure


_W08_FORBIDDEN_REPORT_KEYS = frozenset({
    "accepted_surfaces",
    "expected_payload",
    "expected_state",
    "message",
    "observed_surface",
    "raw_observation",
    "relative_path",
    "surface",
    "text",
    "typed_payload",
})


def _validate_public_safe_aggregate(value: object) -> None:
    """递归拒绝 private 字段；不误杀仅含安全计数的键名。"""
    if isinstance(value, dict):
        if any(key in _W08_FORBIDDEN_REPORT_KEYS for key in value):
            raise W08EvaluatorInfrastructureError("W08 safe aggregate 泄漏 private 字段")
        for item in value.values():
            _validate_public_safe_aggregate(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_public_safe_aggregate(item)


def _infer_observation_family(
    adapter: W08CandidateInferenceAdapter,
    observations: tuple[tuple[str, ObservationRecord], ...],
    *,
    disabled_components: tuple[str, ...] = (),
) -> tuple[W08CandidateInferenceOutcome, ...]:
    """执行一个 inference family；失败时只保留不可逆 schema 诊断。"""
    outcomes: list[W08CandidateInferenceOutcome] = []
    invocation_ordinal = 0
    for _, observation in observations:
        for dimension in W08_DIMENSION_KEYS:
            invocation_ordinal += 1
            schema_sha256 = "0" * 64
            try:
                schema_sha256 = w08_inference_schema_sha256(
                    observation.typed_payload.to_value()
                )
                outcome = adapter.infer(
                    observation,
                    dimension_key=dimension,
                    disabled_components=disabled_components,
                )
            except W08CandidateInferenceError as error:
                failure_kind = error.reason_code
            except Exception:
                failure_kind = "OUTPUT_CONTRACT_REJECTED"
            else:
                outcomes.append(outcome)
                continue
            raise _W08SanitizedInferenceFailure({
                "inference_failure_dimension_key": dimension,
                "inference_failure_invocation_ordinal": invocation_ordinal,
                "inference_failure_kind": failure_kind,
                "inference_failure_payload_kind": (
                    observation.payload_kind
                    if observation.payload_kind in W08_INFERENCE_PAYLOAD_KINDS
                    else "UNREGISTERED"
                ),
                "inference_failure_schema_sha256": schema_sha256,
            }) from None
    return tuple(outcomes)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W08EvaluatorInfrastructureError(f"W08 {label} 无法读取") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W08EvaluatorInfrastructureError(f"W08 {label} 不是 canonical object")
    return value, payload


def _tree_digest(root: Path) -> str:
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        values.append([path.relative_to(root).as_posix(), len(payload), _sha256(payload)])
    return evidence_commitment(values)


def _git_state(repository: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise W08EvaluatorInfrastructureError("W08 无法核验 public Git 状态") from error
    return head, status


def _validate_roots(config: W08PrivateEvaluatorRuntimeConfig) -> tuple[Path, Path, Path, Path]:
    if not isinstance(config, W08PrivateEvaluatorRuntimeConfig):
        raise TypeError("W08 evaluator config 类型非法")
    repository = Path(config.repository_root).resolve()
    candidate = Path(config.candidate_root).resolve()
    family = Path(config.family_root).resolve()
    execution = Path(config.execution_root).resolve()
    roots = (repository, candidate, family)
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise W08EvaluatorInfrastructureError("W08 public/Candidate/private root 未隔离")
    if not execution.is_relative_to(family):
        raise W08EvaluatorInfrastructureError("W08 evaluator execution root 越界")
    if config.fault_phase is not None and config.fault_phase not in W08_EVALUATOR_PHASES:
        raise W08EvaluatorInfrastructureError("W08 evaluator fault phase 未注册")
    return repository, candidate, family, execution


def _enter_phase(config: W08PrivateEvaluatorRuntimeConfig, phase: str) -> None:
    if config.fault_phase == phase:
        raise W08EvaluatorInjectedFault(f"W08 evaluator injected phase: {phase}")


def _candidate_inference_available(host_evidence: object) -> bool:
    if not isinstance(host_evidence, dict):
        return False
    interface = host_evidence.get("private_inference_interface")
    if not isinstance(interface, dict):
        return False
    commitment = interface.get("state_commitment")
    return bool(
        interface.get("version") == W08_PRIVATE_INFERENCE_INTERFACE_VERSION
        and interface.get("executable") == 1
        and interface.get("evaluator_label_inputs") == 0
        and interface.get("per_case_invocation_required") == 1
        and tuple(interface.get("component_keys", ())) == W08_DIMENSION_KEYS
        and type(interface.get("rule_count")) is int
        and interface["rule_count"] > 0
        and isinstance(interface.get("state_key"), list)
        and bool(interface["state_key"])
        and isinstance(commitment, str)
        and len(commitment) == 64
        and all(char in "0123456789abcdef" for char in commitment)
    )


def _candidate_documents(candidate: Path, family: dict[str, Any]) -> dict[str, Any]:
    names = {
        "contract": W08_CANDIDATE_CONTRACT_FREEZE_NAME,
        "guard": W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
        "host": W08_CANDIDATE_HOST_FREEZE_NAME,
        "seal": W08_CANDIDATE_TERMINAL_SEAL_NAME,
    }
    expected = {
        "contract": family["candidate_contract_sha256"],
        "guard": family["candidate_first_run_guard_sha256"],
        "host": family["candidate_host_freeze_sha256"],
        "seal": family["candidate_terminal_seal_sha256"],
    }
    values: dict[str, Any] = {}
    for key, name in names.items():
        value, payload = _read_canonical(candidate / name, label=f"Candidate {key}")
        if _sha256(payload) != strict_sha256(expected[key], label=f"Candidate {key}"):
            raise W08EvaluatorInfrastructureError("W08 Candidate artifact SHA 漂移")
        values[key] = value
    if (
        values["guard"].get("formal_run_count_after") != 1
        or values["host"].get("formal_run_count") != 1
        or values["host"].get("candidate_sealed") != 1
        or values["seal"].get("terminal_state") != "PASS"
        or values["seal"].get("candidate_host_freeze_sha256") != expected["host"]
        or values["host"].get("candidate_contract_sha256") != expected["contract"]
        or values["host"].get("private_inference_interface")
        != values["host"].get("host_evidence", {}).get(
            "private_inference_interface"
        )
        or values["seal"].get("candidate_inference_state_sha256")
        != values["host"].get("private_inference_interface", {}).get(
            "state_commitment"
        )
        or values["seal"].get("candidate_inference_state_key")
        != values["host"].get("private_inference_interface", {}).get("state_key")
    ):
        raise W08EvaluatorInfrastructureError("W08 Candidate 未形成 sealed PASS")
    return values


def _family_documents(family_root: Path, freeze_sha: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    freeze, freeze_bytes = _read_canonical(
        family_root / W08_PRIVATE_FAMILY_FREEZE_NAME,
        label="private family freeze",
    )
    if _sha256(freeze_bytes) != strict_sha256(freeze_sha, label="family freeze"):
        raise W08EvaluatorInfrastructureError("W08 private family freeze SHA 漂移")
    documents: dict[str, bytes] = {}
    expected_names = (
        W08_PRIVATE_SOURCE_NAME,
        W08_PRIVATE_SCHEMA_NAME,
        W08_PRIVATE_CASE_NAME,
        W08_PRIVATE_LABEL_NAME,
        W08_PRIVATE_CLUSTER_NAME,
    )
    identities = freeze.get("documents")
    if not isinstance(identities, list) or tuple(item.get("name") for item in identities) != expected_names:
        raise W08EvaluatorInfrastructureError("W08 private document inventory 漂移")
    for identity in identities:
        path = family_root / identity["name"]
        payload = path.read_bytes()
        if len(payload) != identity["size_bytes"] or _sha256(payload) != identity["sha256"]:
            raise W08EvaluatorInfrastructureError("W08 private document identity 漂移")
        documents[identity["name"]] = payload
    return freeze, documents


def _payload_file(repository: Path, binding: W08FileBinding) -> Path:
    target = (repository / binding.relative_path).resolve()
    if not target.is_file() or target.is_symlink() or not target.is_relative_to(repository):
        raise W08EvaluatorInfrastructureError("W08 private payload path 非普通 repo 文件")
    return target


def _read_binding(repository: Path, binding: W08FileBinding, audit: _ReadAudit) -> tuple[object, ...]:
    target = _payload_file(repository, binding)
    local_parts = Path(binding.identity.relative_path).parts
    artifact_root = target.parents[len(local_parts) - 1]
    try:
        records = read_record_artifact(artifact_root, binding.identity)
    except (DatasetArtifactIOError, OSError) as error:
        raise W08EvaluatorInfrastructureError("W08 private payload transport/SHA 失败") from error
    audit.record(
        binding.relative_path,
        binding.identity.transport_size_bytes,
        len(records),
        binding.identity.owner_kind,
    )
    return records


def _read_private_observations(
    repository: Path,
    context,
    audit: _ReadAudit,
) -> tuple[tuple[str, ObservationRecord], ...]:
    visibility = W08VisibilityFirewall(context, audit)  # type: ignore[arg-type]
    observations: dict[object, tuple[str, ObservationRecord]] = {}
    for binding in _private_bindings(context):
        if binding.identity.owner_kind != "observation":
            continue
        visibility.authorize_evaluator(binding.relative_path, candidate_sealed=1)
        records = _read_binding(repository, binding, audit)
        if any(not isinstance(item, ObservationRecord) for item in records):
            raise W08EvaluatorInfrastructureError("W08 held-out artifact record kind 漂移")
        for item in records:
            if item.stable_key in observations:
                raise W08EvaluatorInfrastructureError("W08 held-out Observation 重复")
            observations[item.stable_key] = (binding.pack_key, item)
    if not observations:
        raise W08EvaluatorInfrastructureError("W08 private Observation family 为空")
    return tuple(
        observations[key] for key in sorted(observations, key=lambda item: item.components)
    )


def _read_private_labels(
    repository: Path,
    context,
    audit: _ReadAudit,
) -> tuple[tuple[object, EvaluatorLabelRecord], ...]:
    visibility = W08VisibilityFirewall(context, audit)  # type: ignore[arg-type]
    labels: dict[object, tuple[str, EvaluatorLabelRecord]] = {}
    for binding in _private_bindings(context):
        if binding.identity.owner_kind != "evaluator":
            continue
        visibility.authorize_evaluator(binding.relative_path, candidate_sealed=1)
        records = _read_binding(repository, binding, audit)
        if any(not isinstance(item, EvaluatorLabelRecord) for item in records):
            raise W08EvaluatorInfrastructureError("W08 label artifact record kind 漂移")
        for item in records:
            if item.observation_key in labels:
                raise W08EvaluatorInfrastructureError("W08 evaluator label 重复")
            labels[item.observation_key] = (binding.pack_key, item)
    if not labels:
        raise W08EvaluatorInfrastructureError("W08 private label family 为空")
    return tuple(labels.items())


def _pair_private_records(
    observations: tuple[tuple[str, ObservationRecord], ...],
    labels: tuple[tuple[object, EvaluatorLabelRecord], ...],
) -> tuple[W08PrivateEvaluationPair, ...]:
    observation_by_key = {item.stable_key: (pack, item) for pack, item in observations}
    label_by_key = {key: value for key, value in labels}
    if set(observation_by_key) != set(label_by_key):
        raise W08EvaluatorInfrastructureError("W08 held-out/label reference 不闭合")
    pairs = []
    for key in sorted(observation_by_key, key=lambda item: item.components):
        pack, observation = observation_by_key[key]
        label_pack, label = label_by_key[key]
        if pack != label_pack:
            raise W08EvaluatorInfrastructureError("W08 held-out/label pack 漂移")
        pairs.append(W08PrivateEvaluationPair(pack, observation, label))
    if not pairs:
        raise W08EvaluatorInfrastructureError("W08 private case family 为空")
    return tuple(pairs)


def _private_bindings(context) -> tuple[W08FileBinding, ...]:
    """只返回索引授权给 W08 的七个可见 pack。"""
    return tuple(
        item
        for item in context.evaluator_bindings
        if item.pack_key in W08_VISIBLE_PACK_KEYS
    )


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W08EvaluatorInfrastructureError("W08 private publication 不可覆盖") from error
    return _sha256(payload)


def _safe_failure(
    family_root: Path,
    *,
    family: dict[str, Any],
    phase: str,
    family_freeze_sha256: str,
    guard_sha256: str,
    write_counts: dict[str, int] | None = None,
    infrastructure: dict[str, object] | None = None,
) -> W08PrivateEvaluatorRunResult:
    writes = write_counts or {
        "candidate_writes": 0,
        "label_writes": 0,
        "public_writes": 0,
    }
    safe_infrastructure = {"fault_ne_protocol": 1, **(infrastructure or {})}
    aggregate = public_safe_w08_aggregate(
        (),
        family_commitment=family["family_key"],
        payload_commitment=family["payload_commitment"],
        case_commitment=family["case_commitment"],
        label_commitment=family["label_commitment"],
        cluster_commitment=family["cluster_commitment"],
        failure_phase=phase,
        formal_run_count=1,
        write_counts=writes,
        infrastructure=safe_infrastructure,
    )
    target = family_root / "publication" / W08_PRIVATE_AGGREGATE_NAME
    sha = _write_exclusive(target, canonical_json_bytes(aggregate))
    return W08PrivateEvaluatorRunResult(
        "NE", target, sha, None, None, family_freeze_sha256, guard_sha256, None
    )


def run_w08_private_evaluation_once(
    config: W08PrivateEvaluatorRuntimeConfig,
    *,
    family_freeze_sha256: str,
) -> W08PrivateEvaluatorRunResult:
    """guard 后先验 Candidate inference；可判时才读取 private payload。"""
    repository, candidate, family_root, execution = _validate_roots(config)
    if (family_root / "publication" / W08_PRIVATE_AGGREGATE_NAME).exists():
        raise W08EvaluatorInfrastructureError("W08 private aggregate 已存在，不可重跑")
    family, _documents = _family_documents(family_root, family_freeze_sha256)
    guard_path, guard_sha = consume_w08_private_first_run_guard(
        family_root,
        family_freeze_sha256=family_freeze_sha256,
    )
    del guard_path
    current_phase = "CANDIDATE_VERIFY"
    try:
        _enter_phase(config, current_phase)
        candidate_before = _tree_digest(candidate)
        public_before = _git_state(repository)
        if public_before[1]:
            raise W08EvaluatorInfrastructureError("W08 evaluator 要求 public worktree clean")
        candidate_docs = _candidate_documents(candidate, family)
        current_phase = "CANDIDATE_DUMP_READBACK"
        _enter_phase(config, current_phase)
        candidate_config = W08RuntimeConfig(
            repository,
            candidate / "host",
            candidate / "host" / "coordinator.sqlite",
            worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
            mode=W08_CANDIDATE_FORMAL_MODE,
        )
        candidate_dump = load_w08_public_dump(candidate_config)
        snapshot = snapshot_from_w08_outcome(candidate_dump)
        host_evidence = candidate_docs["host"].get("host_evidence", {})
        if (
            host_evidence.get("semantic_state_key") != list(snapshot.semantic_state_key)
            or host_evidence.get("dump_manifest_sha256")
            != snapshot.dump_manifest_sha256
        ):
            raise W08EvaluatorInfrastructureError("W08 Candidate host/dump commitment 漂移")
        if not _candidate_inference_available(host_evidence):
            candidate_after = _tree_digest(candidate)
            public_after = _git_state(repository)
            writes = {
                "candidate_writes": int(candidate_after != candidate_before),
                "label_writes": 0,
                "public_writes": int(public_after != public_before),
            }
            if any(writes.values()):
                raise W08EvaluatorInfrastructureError(
                    "W08 private inference preflight owner isolation 失败"
                )
            return _safe_failure(
                family_root,
                family=family,
                phase=current_phase,
                family_freeze_sha256=family_freeze_sha256,
                guard_sha256=guard_sha,
                write_counts=writes,
                infrastructure={
                    "candidate_dump_readback": int(candidate_dump.dump_readback),
                    "candidate_inventory_match": 1,
                    "candidate_private_inference_available": 0,
                    "future_payload_reads": 0,
                    "private_label_record_count": 0,
                    "private_observation_record_count": 0,
                    "private_payload_bytes": 0,
                    "private_payload_gets": 0,
                    "private_read_path_count": 0,
                },
            )
        host_interface = candidate_docs["host"].get("private_inference_interface")
        host_evidence = candidate_docs["host"].get("host_evidence", {})
        if host_interface != host_evidence.get("private_inference_interface"):
            raise W08EvaluatorInfrastructureError("W08 Candidate inference interface cross-reference 漂移")
        inference_state = load_w08_candidate_inference_state(candidate_config)
        if (
            not _candidate_inference_available(host_evidence)
            or inference_state.sha256() != host_interface.get("state_commitment")
            or list(inference_state.state_key) != host_interface.get("state_key")
            or len(inference_state.rules) != host_interface.get("rule_count")
        ):
            raise W08EvaluatorInfrastructureError("W08 Candidate inference state preflight 失败")
        adapter = W08CandidateInferenceAdapter(inference_state)
        context = open_w08_frozen_contract(repository)
        private_bindings = _private_bindings(context)
        current_phase = "PAYLOAD_READ"
        _enter_phase(config, current_phase)
        audit = _ReadAudit({})
        label_metadata_before = {
            item.relative_path: (
                _payload_file(repository, item).stat().st_size,
                _payload_file(repository, item).stat().st_mtime_ns,
            )
            for item in private_bindings
            if item.identity.owner_kind == "evaluator"
        }
        observations = _read_private_observations(repository, context, audit)
        if (
            audit.future_payload_reads
            or audit.observation_records > W08_RESOURCE_BUDGET["max_records"]
            or audit.payload_gets > W08_RESOURCE_BUDGET["max_payload_gets"]
            or audit.payload_bytes > W08_RESOURCE_BUDGET["max_payload_bytes"]
        ):
            raise W08EvaluatorInfrastructureError("W08 private Observation audit/resource 漂移")
        current_phase = "PAYLOAD_PAIR"
        _enter_phase(config, current_phase)
        if audit.label_records:
            raise W08EvaluatorInfrastructureError("W08 private label 在 inference 前已读取")
        current_phase = "BASELINE"
        _enter_phase(config, current_phase)
        baseline_outcomes = _infer_observation_family(adapter, observations)
        outcome_families = []
        for index, ablation in enumerate(W08_ABLATION_KEYS):
            current_phase = W08_EVALUATOR_PHASES[5 + index]
            _enter_phase(config, current_phase)
            rerun = _infer_observation_family(
                adapter,
                observations,
                disabled_components=(W08_DIMENSION_KEYS[index],),
            )
            outcome_families.append((ablation, rerun))
        label_read_before_inference = int(audit.label_records == 0)
        current_phase = "PAYLOAD_PAIR"
        labels = _read_private_labels(repository, context, audit)
        pairs = _pair_private_records(observations, labels)
        if (
            audit.future_payload_reads
            or audit.payload_gets != len(private_bindings)
            or audit.observation_records != audit.label_records
            or audit.payload_gets > W08_RESOURCE_BUDGET["max_payload_gets"]
            or audit.payload_bytes > W08_RESOURCE_BUDGET["max_payload_bytes"]
            or len(pairs) > W08_RESOURCE_BUDGET["max_records"]
        ):
            raise W08EvaluatorInfrastructureError("W08 private payload audit/resource 漂移")
        results = evaluate_w08_private_pairs(
            snapshot,
            pairs,
            case_outcomes=baseline_outcomes,
        )
        ablations = assess_w08_orthogonal_ablations(
            snapshot,
            pairs,
            outcome_families=tuple(outcome_families),
        )
        current_phase = "OPEN_GENERATION"
        _enter_phase(config, current_phase)
        open_generation = assess_w08_private_open_generation(
            snapshot,
            pairs,
            case_outcomes=baseline_outcomes,
        )
        current_phase = "LC16"
        _enter_phase(config, current_phase)
        lc16 = assess_w08_private_lc16(
            snapshot,
            pairs,
            case_outcomes=baseline_outcomes,
        )
        dump_value = {
            "artifact_kind": "PH2_W08_PRIVATE_EVALUATION_DUMP",
            "ablation_invocation_commitments": [
                {
                    "ablation_key": key,
                    "outcome_commitment": evidence_commitment([
                        item.safe_commitment_dict() for item in outcomes
                    ]),
                }
                for key, outcomes in outcome_families
            ],
            "baseline_invocation_commitment": evidence_commitment([
                item.safe_commitment_dict() for item in baseline_outcomes
            ]),
            "dimension_results": [item.to_safe_dict() for item in results],
            "family_commitment": family["family_key"],
            "format_version": 1,
            "inference_before_label_reads": label_read_before_inference,
            "inference_invocation_count": len(baseline_outcomes) + sum(
                len(outcomes) for _, outcomes in outcome_families
            ),
            "lc16": lc16,
            "open_generation": open_generation,
            "private_read_commitment": evidence_commitment([
                [path, count, size]
                for path, (count, size) in sorted(audit.reads_by_path.items())
            ]),
            "private_read_path_count": len(audit.reads_by_path),
        }
        dump_path = execution / W08_PRIVATE_DUMP_NAME
        dump_sha = _write_exclusive(dump_path, canonical_json_bytes(dump_value))
        current_phase = "DUMP_READBACK"
        _enter_phase(config, current_phase)
        dump_readback, dump_bytes = _read_canonical(dump_path, label="private dump")
        if dump_readback != dump_value or _sha256(dump_bytes) != dump_sha:
            raise W08EvaluatorInfrastructureError("W08 private dump readback 漂移")
        current_phase = "INTEGRITY"
        _enter_phase(config, current_phase)
        candidate_after = _tree_digest(candidate)
        public_after = _git_state(repository)
        label_metadata_after = {
            item.relative_path: (
                _payload_file(repository, item).stat().st_size,
                _payload_file(repository, item).stat().st_mtime_ns,
            )
            for item in private_bindings
            if item.identity.owner_kind == "evaluator"
        }
        writes = {
            "candidate_writes": int(candidate_after != candidate_before),
            "label_writes": int(label_metadata_after != label_metadata_before),
            "public_writes": int(public_after != public_before),
        }
        if any(writes.values()):
            raise W08EvaluatorInfrastructureError("W08 private evaluator owner isolation 失败")
        infrastructure = {
            "candidate_dump_readback": int(candidate_dump.dump_readback),
            "candidate_inventory_match": 1,
            "candidate_writes": writes["candidate_writes"],
            "dump_readback": 1,
            "evaluator_label_writes": writes["label_writes"],
            "fault_ne_protocol": 1,
            "future_payload_reads": audit.future_payload_reads,
            "host_learning_writes": 0,
            "memory_learning_writes": 0,
            "private_label_record_count": audit.label_records,
            "private_observation_record_count": audit.observation_records,
            "private_payload_bytes": audit.payload_bytes,
            "private_payload_gets": audit.payload_gets,
            "private_read_path_count": len(audit.reads_by_path),
            "inference_before_label_reads": label_read_before_inference,
            "inference_invocation_count": len(baseline_outcomes) + sum(
                len(outcomes) for _, outcomes in outcome_families
            ),
            "public_repo_writes": writes["public_writes"],
        }
        current_phase = "REPORT_SAFETY"
        _enter_phase(config, current_phase)
        aggregate = public_safe_w08_aggregate(
            results,
            family_commitment=family["family_key"],
            payload_commitment=family["payload_commitment"],
            case_commitment=family["case_commitment"],
            label_commitment=family["label_commitment"],
            cluster_commitment=family["cluster_commitment"],
            failure_phase="NONE",
            formal_run_count=1,
            write_counts=writes,
            ablation_results=ablations,
            open_generation=open_generation,
            lc16=lc16,
            infrastructure=infrastructure,
        )
        _validate_public_safe_aggregate(aggregate)
        encoded = canonical_json_bytes(aggregate)
        aggregate_path = family_root / "publication" / W08_PRIVATE_AGGREGATE_NAME
        aggregate_sha = _write_exclusive(aggregate_path, encoded)
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS":
            recommendation = canonical_json_bytes({
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": "PH2_W08_RUNTIME_RECEIPT_RECOMMENDATION",
                "candidate_host_freeze_sha256": family[
                    "candidate_host_freeze_sha256"
                ],
                "family_freeze_sha256": family_freeze_sha256,
                "formal_run_count": 1,
                "format_version": 1,
                "recommend_runtime_receipt": 1,
            })
            recommendation_path = family_root / "publication" / (
                W08_PRIVATE_RECOMMENDATION_NAME
            )
            recommendation_sha = _write_exclusive(
                recommendation_path, recommendation
            )
        return W08PrivateEvaluatorRunResult(
            str(aggregate["status"]),
            aggregate_path,
            aggregate_sha,
            recommendation_path,
            recommendation_sha,
            family_freeze_sha256,
            guard_sha,
            dump_sha,
        )
    except _W08SanitizedInferenceFailure as error:
        return _safe_failure(
            family_root,
            family=family,
            phase=current_phase,
            family_freeze_sha256=family_freeze_sha256,
            guard_sha256=guard_sha,
            infrastructure=error.infrastructure,
        )
    except Exception:
        return _safe_failure(
            family_root,
            family=family,
            phase=current_phase,
            family_freeze_sha256=family_freeze_sha256,
            guard_sha256=guard_sha,
        )


__all__ = [
    "W08EvaluatorInfrastructureError",
    "W08EvaluatorInjectedFault",
    "W08PrivateEvaluatorRunResult",
    "W08PrivateEvaluatorRuntimeConfig",
    "run_w08_private_evaluation_once",
]
