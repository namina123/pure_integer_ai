"""W09-10 唯一 private evaluator runtime 与永久终态封存。"""
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
    SourceRefRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_DIMENSION_KEYS,
    W09_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09_EVALUATOR_OWNER,
    W09HostWriteSnapshot,
    W09PayloadAudit,
    W09FileBinding,
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_evaluator import (
    W09PrivateEvaluationPair,
    assess_w09_orthogonal_ablations,
    assess_w09_private_j_lc,
    assess_w09_private_open_generation,
    assess_w09_private_resource,
    assess_w09_private_rollback,
    assess_w09_private_v06,
    assess_w09_private_windows,
    evaluate_w09_private_pairs,
    snapshot_from_w09_host_document,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_contract import (
    W09_EVALUATOR_PHASES,
    W09_PRIVATE_AGGREGATE_NAME,
    W09_PRIVATE_CASE_NAME,
    W09_PRIVATE_CLUSTER_NAME,
    W09_PRIVATE_DUMP_NAME,
    W09_PRIVATE_FAMILY_FREEZE_NAME,
    W09_PRIVATE_LABEL_NAME,
    W09_PRIVATE_RECOMMENDATION_NAME,
    W09_PRIVATE_SCHEMA_NAME,
    W09_PRIVATE_SOURCE_NAME,
    W09_PRIVATE_TERMINAL_SEAL_NAME,
    W09PrivateEvaluationError,
    evidence_commitment,
    public_safe_w09_aggregate,
    strict_sha1,
    strict_sha256,
    validate_w09_safe_report,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_family import (
    consume_w09_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w09_firewall import (
    W09PayloadFirewall,
    W09VisibilityFirewall,
)
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09CandidateInferenceAdapter,
    W09InferenceError,
    W09InferenceOutcome,
    compile_w09_inference_state,
    schema_sha256,
)
from pure_integer_ai.experiments.ph2_w09_rotation import (
    W09RotationManifest,
    read_w09_rotation_binding,
    read_w09_rotation_manifest,
    validate_w09_rotation_metadata,
)

W09_CANDIDATE_CONTRACT_NAME = "candidate_contract_freeze.json"
W09_CANDIDATE_GUARD_NAME = "formal_first_run_guard.json"
W09_CANDIDATE_HOST_NAME = "candidate_host_freeze.json"
W09_CANDIDATE_SEAL_NAME = "candidate_terminal_seal.json"


class W09EvaluatorInfrastructureError(W09PrivateEvaluationError):
    """W09 private evaluator 的 transport、owner 或封存设施失败。"""


class W09EvaluatorInjectedFault(W09EvaluatorInfrastructureError):
    """测试专用的预注册相位故障。"""


class _W09SanitizedInferenceFailure(RuntimeError):
    """只携带不可逆枚举诊断的 inference failure。"""

    def __init__(self, infrastructure: dict[str, object]) -> None:
        super().__init__("W09 Candidate inference failed")
        self.infrastructure = infrastructure


@dataclass(frozen=True)
class W09PrivateEvaluatorRuntimeConfig:
    """一次 formal evaluator 的四隔离 root 与故障配置。"""

    repository_root: str | Path
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    rotation_root: str | Path
    fault_phase: str | None = None
    require_clean_public: bool = True


@dataclass(frozen=True)
class W09PrivateEvaluatorRunResult:
    """只返回安全 publication 路径和摘要，不返回 private payload。"""

    status: str
    aggregate_path: Path
    aggregate_sha256: str
    recommendation_path: Path | None
    recommendation_sha256: str | None
    family_freeze_sha256: str
    first_run_guard_sha256: str
    dump_sha256: str | None
    terminal_seal_path: Path
    terminal_seal_sha256: str


@dataclass
class _ReadAudit:
    """内部区分 source/Observation/label 的读取账。"""

    reads_by_path: dict[str, tuple[int, int]]
    payload_gets: int = 0
    payload_bytes: int = 0
    source_records: int = 0
    observation_records: int = 0
    label_records: int = 0
    d03_observation_records: int = 0
    rotation_observation_records: int = 0
    future_payload_reads: int = 0

    def record(self, identity: str, size: int, records: int, owner: str, *, family: str) -> None:
        """累计一个 manifest-bound 文件；路径仅留在私有进程内。"""
        count, total = self.reads_by_path.get(identity, (0, 0))
        self.reads_by_path[identity] = (count + 1, total + size)
        self.payload_gets += 1
        self.payload_bytes += size
        if owner == "source":
            self.source_records += records
        elif owner == "observation":
            self.observation_records += records
            if family == "D03":
                self.d03_observation_records += records
            else:
                self.rotation_observation_records += records
        elif owner == "evaluator":
            self.label_records += records
        else:
            raise W09EvaluatorInfrastructureError("W09 private read owner 未登记")


def _sha256(payload: bytes) -> str:
    """计算 artifact 字节摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """读取并校验 canonical JSON object。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W09EvaluatorInfrastructureError(f"W09 {label} 无法读取") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W09EvaluatorInfrastructureError(f"W09 {label} 不是 canonical object")
    return value, payload


def _tree_digest(root: Path) -> str:
    """对一个 owner root 的文件 inventory 做只读摘要。"""
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        values.append([path.relative_to(root).as_posix(), len(payload), _sha256(payload)])
    return evidence_commitment(values)


def _git_state(repository: Path) -> tuple[str, str]:
    """返回 public HEAD 和 porcelain 状态。"""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=repository, check=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise W09EvaluatorInfrastructureError("W09 无法核验 public Git 状态") from error
    return strict_sha1(head, label="public HEAD"), status


def _validate_roots(config: W09PrivateEvaluatorRuntimeConfig) -> tuple[Path, Path, Path, Path, Path]:
    """要求 public/Candidate/family/rotation 四个 owner 物理隔离。"""
    if not isinstance(config, W09PrivateEvaluatorRuntimeConfig):
        raise TypeError("W09 evaluator config 类型非法")
    roots = tuple(Path(item).resolve() for item in (
        config.repository_root, config.candidate_root, config.family_root,
        config.execution_root, config.rotation_root,
    ))
    repository, candidate, family, execution, rotation = roots
    owners = (repository, candidate, family, rotation)
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left) for index, left in enumerate(owners) for right in owners[index + 1:]):
        raise W09EvaluatorInfrastructureError("W09 public/Candidate/family/rotation root 未隔离")
    if not execution.is_relative_to(family):
        raise W09EvaluatorInfrastructureError("W09 evaluator execution root 越界")
    if config.fault_phase is not None and config.fault_phase not in W09_EVALUATOR_PHASES:
        raise W09EvaluatorInfrastructureError("W09 evaluator fault phase 未登记")
    if not isinstance(config.require_clean_public, bool):
        raise W09EvaluatorInfrastructureError("W09 clean public flag 非法")
    return repository, candidate, family, execution, rotation


def _enter_phase(config: W09PrivateEvaluatorRuntimeConfig, phase: str) -> None:
    """在预注册 phase 注入一次测试故障。"""
    if config.fault_phase == phase:
        raise W09EvaluatorInjectedFault(f"W09 evaluator injected phase: {phase}")


def _family_documents(family_root: Path, freeze_sha256: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """回读 freeze 及其固定五文档 identity。"""
    freeze, freeze_bytes = _read_canonical(family_root / W09_PRIVATE_FAMILY_FREEZE_NAME, label="private family freeze")
    if _sha256(freeze_bytes) != strict_sha256(freeze_sha256, label="family freeze"):
        raise W09EvaluatorInfrastructureError("W09 private family freeze SHA 漂移")
    expected_names = (
        W09_PRIVATE_SOURCE_NAME, W09_PRIVATE_SCHEMA_NAME, W09_PRIVATE_CASE_NAME,
        W09_PRIVATE_LABEL_NAME, W09_PRIVATE_CLUSTER_NAME,
    )
    identities = freeze.get("documents")
    if not isinstance(identities, list) or tuple(item.get("name") for item in identities) != expected_names:
        raise W09EvaluatorInfrastructureError("W09 private document inventory 漂移")
    documents = {}
    for identity in identities:
        payload = (family_root / str(identity["name"])).read_bytes()
        if len(payload) != identity.get("size_bytes") or _sha256(payload) != identity.get("sha256"):
            raise W09EvaluatorInfrastructureError("W09 private document identity 漂移")
        documents[str(identity["name"])] = payload
    return freeze, documents


def _candidate_documents(candidate: Path, family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """验证 Candidate 四文档 SHA 与 sealed PASS 状态。"""
    names = {
        "contract": W09_CANDIDATE_CONTRACT_NAME,
        "guard": W09_CANDIDATE_GUARD_NAME,
        "host": W09_CANDIDATE_HOST_NAME,
        "seal": W09_CANDIDATE_SEAL_NAME,
    }
    expected = {
        "contract": family["candidate_contract_sha256"],
        "guard": family["candidate_first_run_guard_sha256"],
        "host": family["candidate_host_freeze_sha256"],
        "seal": family["candidate_terminal_seal_sha256"],
    }
    values = {}
    for key, name in names.items():
        value, payload = _read_canonical(candidate / name, label=f"Candidate {key}")
        if _sha256(payload) != strict_sha256(expected[key], label=f"Candidate {key}"):
            raise W09EvaluatorInfrastructureError("W09 Candidate artifact SHA 漂移")
        values[key] = value
    state = values["host"].get("execution_state", {})
    if (
        values["contract"].get("formal_run_count") != 0
        or values["guard"].get("formal_run_count_after") != 1
        or values["host"].get("formal_run_count") != 1
        or values["host"].get("candidate_sealed") != 1
        or values["host"].get("public_head_commit_sha1") != family.get("evaluator_public_head_commit_sha1")
        or values["seal"].get("terminal_state") != "PASS"
        or values["seal"].get("candidate_host_freeze_sha256") != expected["host"]
        or state.get("W09_RUNTIME_EVIDENCED") != 1
        or state.get("formal_w09_training_runs") != 1
        or state.get("teacher_calls") != 0
        or state.get("LANGUAGE_CAPABILITY_MASTERED") != 0
        or state.get("LANGUAGE_READINESS") != 0
    ):
        raise W09EvaluatorInfrastructureError("W09 Candidate 未形成 sealed PASS")
    return values


def _payload_file(repository: Path, binding: W09FileBinding) -> Path:
    """解析一个 repository-bound evaluator artifact。"""
    target = (repository / binding.relative_path).resolve()
    if not target.is_file() or target.is_symlink() or not target.is_relative_to(repository):
        raise W09EvaluatorInfrastructureError("W09 private payload path 非普通 repo 文件")
    return target


def _read_d03_binding(repository: Path, binding: W09FileBinding, audit: _ReadAudit) -> tuple[object, ...]:
    """按 D-03 manifest identity 读取一个 owner artifact。"""
    target = _payload_file(repository, binding)
    local_parts = Path(binding.identity.relative_path).parts
    artifact_root = target.parents[len(local_parts) - 1]
    try:
        records = read_record_artifact(artifact_root, binding.identity)
    except (DatasetArtifactIOError, OSError, ValueError) as error:
        raise W09EvaluatorInfrastructureError("W09 D-03 private payload transport 失败") from error
    audit.record(
        evidence_commitment(binding.to_dict()), binding.identity.transport_size_bytes,
        len(records), binding.identity.owner_kind, family="D03",
    )
    return records


def _read_d03_owner(
    repository: Path,
    context: object,
    audit: _ReadAudit,
    owner_kind: str,
) -> tuple[tuple[str, object], ...]:
    """按 owner kind 读取全部 D-03 evaluator binding。"""
    firewall = W09VisibilityFirewall(context, W09PayloadAudit())
    result = []
    for binding in context.evaluator_bindings:
        if binding.identity.owner_kind != owner_kind:
            continue
        firewall.authorize_evaluator(
            binding.relative_path, owner_key=W09_EVALUATOR_OWNER,
            candidate_sealed=1, code_frozen=1, host_writes=W09HostWriteSnapshot(),
        )
        for record in _read_d03_binding(repository, binding, audit):
            result.append((binding.pack_key, record))
    if not result:
        raise W09EvaluatorInfrastructureError("W09 D-03 private owner inventory 为空")
    return tuple(result)


def _read_rotation_owner(
    rotation_root: Path,
    manifest: W09RotationManifest,
    audit: _ReadAudit,
    owner_kind: str,
) -> tuple[tuple[str, object], ...]:
    """按 rotation manifest identity 读取一个 owner 文件。"""
    identity = {
        "source": manifest.source_identity,
        "observation": manifest.observation_identity,
        "evaluator": manifest.label_identity,
    }.get(owner_kind)
    if identity is None:
        raise W09EvaluatorInfrastructureError("W09 rotation owner 未登记")
    records = read_w09_rotation_binding(rotation_root, identity)
    audit.record(
        evidence_commitment(identity.to_dict()), identity.transport_size_bytes,
        len(records), owner_kind, family="ROTATION",
    )
    return tuple(("W09-INDEPENDENT-ROTATION", item) for item in records)


def _observations(values: tuple[tuple[str, object], ...], *, family: str) -> tuple[tuple[str, ObservationRecord, str], ...]:
    """核验并规范排序一个 Observation owner inventory。"""
    if any(not isinstance(item, ObservationRecord) for _, item in values):
        raise W09EvaluatorInfrastructureError("W09 private Observation record kind 漂移")
    records = tuple((pack, item, family) for pack, item in values)
    if len({item.stable_key for _, item, _ in records}) != len(records):
        raise W09EvaluatorInfrastructureError("W09 private Observation 重复")
    return tuple(sorted(records, key=lambda value: value[1].stable_key.components))


def _labels(values: tuple[tuple[str, object], ...]) -> tuple[tuple[str, EvaluatorLabelRecord], ...]:
    """核验并规范排序一个 label owner inventory。"""
    if any(not isinstance(item, EvaluatorLabelRecord) for _, item in values):
        raise W09EvaluatorInfrastructureError("W09 private label record kind 漂移")
    records = tuple((pack, item) for pack, item in values)
    if len({item.observation_key for _, item in records}) != len(records):
        raise W09EvaluatorInfrastructureError("W09 private label 重复")
    return tuple(sorted(records, key=lambda value: value[1].observation_key.components))


def _pair_records(
    observations: tuple[tuple[str, ObservationRecord, str], ...],
    labels: tuple[tuple[str, EvaluatorLabelRecord], ...],
) -> tuple[W09PrivateEvaluationPair, ...]:
    """在 inference 完成后才闭合 Observation/label 引用。"""
    by_observation = {item.stable_key: (pack, item, family) for pack, item, family in observations}
    by_label = {item.observation_key: (pack, item) for pack, item in labels}
    if set(by_observation) != set(by_label):
        raise W09EvaluatorInfrastructureError("W09 private Observation/label 引用不闭合")
    result = []
    for key in sorted(by_observation, key=lambda item: item.components):
        pack, observation, family = by_observation[key]
        label_pack, label = by_label[key]
        if family == "D03" and label_pack != pack:
            raise W09EvaluatorInfrastructureError("W09 D-03 pair pack owner 漂移")
        result.append(W09PrivateEvaluationPair(pack, observation, label, family))
    return tuple(result)


def _infer_observation_family(
    adapter: W09CandidateInferenceAdapter,
    observations: tuple[tuple[str, ObservationRecord, str], ...],
    *,
    disabled_components: tuple[str, ...] = (),
) -> tuple[W09InferenceOutcome, ...]:
    """逐 case 执行五维 inference；失败只保留枚举与 schema commitment。"""
    outcomes = []
    ordinal = 0
    for _, observation, _ in observations:
        for dimension in W09_DIMENSION_KEYS:
            ordinal += 1
            shape = "0" * 64
            try:
                shape = schema_sha256(observation.typed_payload.to_value())
                outcomes.append(adapter.infer(observation, dimension_key=dimension, disabled_components=disabled_components))
            except W09InferenceError:
                raise _W09SanitizedInferenceFailure({
                    "inference_failure_dimension_key": dimension,
                    "inference_failure_invocation_ordinal": ordinal,
                    "inference_failure_kind": "INFERENCE_CONTRACT_REJECTED",
                    "inference_failure_payload_kind": observation.payload_kind,
                    "inference_failure_schema_sha256": shape,
                }) from None
            except Exception:
                raise _W09SanitizedInferenceFailure({
                    "inference_failure_dimension_key": dimension,
                    "inference_failure_invocation_ordinal": ordinal,
                    "inference_failure_kind": "OUTPUT_CONTRACT_REJECTED",
                    "inference_failure_payload_kind": observation.payload_kind,
                    "inference_failure_schema_sha256": shape,
                }) from None
    return tuple(outcomes)


def _write_exclusive(path: Path, payload: bytes) -> str:
    """创建父目录并排他写一个 publication artifact。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W09EvaluatorInfrastructureError("W09 private publication 已存在，不可重跑") from error
    return _sha256(payload)


def _write_terminal_seal(
    family_root: Path,
    *,
    aggregate_sha256: str,
    guard_sha256: str,
    family_freeze_sha256: str,
    status: str,
) -> tuple[Path, str]:
    """排他封存 PASS/FAIL/NE，任何状态都终结 family 运行权。"""
    if status not in {"PASS", "FAIL", "NE"}:
        raise W09EvaluatorInfrastructureError("W09 private terminal status 非法")
    payload = canonical_json_bytes({
        "aggregate_sha256": strict_sha256(aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W09_PRIVATE_TERMINAL_SEAL",
        "family_freeze_sha256": strict_sha256(family_freeze_sha256, label="family freeze"),
        "first_run_guard_sha256": strict_sha256(guard_sha256, label="private guard"),
        "formal_run_count": 1,
        "format_version": 1,
        "terminal_state": status,
    })
    path = family_root / "publication" / W09_PRIVATE_TERMINAL_SEAL_NAME
    return path, _write_exclusive(path, payload)


def _zero_write_counts() -> dict[str, int]:
    """返回 evaluator 对全部受保护 owner 的零写账。"""
    return {key: 0 for key in (
        "candidate_writes", "label_writes", "public_writes", "host_writes",
        "core_writes", "evidence_writes", "use_writes", "memory_writes",
        "assessment_writes", "clock_writes",
    )}


def _safe_failure(
    family_root: Path,
    *,
    family: dict[str, Any],
    family_freeze_sha256: str,
    guard_sha256: str,
    phase: str,
    infrastructure: dict[str, object],
) -> W09PrivateEvaluatorRunResult:
    """把异常转换为不可覆盖、无私有字段的 NE aggregate 和 terminal seal。"""
    aggregate = public_safe_w09_aggregate(
        (),
        family_commitment=family["family_key"],
        payload_commitment=family["payload_commitment"],
        case_commitment=family["case_commitment"],
        label_commitment=family["label_commitment"],
        cluster_commitment=family["cluster_commitment"],
        rotation_package_commitment=family["rotation_package_commitment"],
        failure_phase=phase,
        formal_run_count=1,
        write_counts=_zero_write_counts(),
        infrastructure={"fault_ne_protocol": 1, **infrastructure},
    )
    validate_w09_safe_report(aggregate)
    aggregate_path = family_root / "publication" / W09_PRIVATE_AGGREGATE_NAME
    aggregate_sha = _write_exclusive(aggregate_path, canonical_json_bytes(aggregate))
    seal_path, seal_sha = _write_terminal_seal(
        family_root, aggregate_sha256=aggregate_sha, guard_sha256=guard_sha256,
        family_freeze_sha256=family_freeze_sha256, status="NE",
    )
    return W09PrivateEvaluatorRunResult(
        "NE", aggregate_path, aggregate_sha, None, None,
        family_freeze_sha256, guard_sha256, None, seal_path, seal_sha,
    )


def run_w09_private_evaluation_once(
    config: W09PrivateEvaluatorRuntimeConfig,
    *,
    family_freeze_sha256: str,
) -> W09PrivateEvaluatorRunResult:
    """消费唯一 guard，按 observation-first/label-last 顺序正式运行一次。"""
    repository, candidate, family_root, execution, rotation_root = _validate_roots(config)
    if (family_root / "publication" / W09_PRIVATE_AGGREGATE_NAME).exists():
        raise W09EvaluatorInfrastructureError("W09 private aggregate 已存在，不可重跑")
    family, documents = _family_documents(family_root, family_freeze_sha256)
    if family.get("resource_limits") != dict(sorted(W09_RESOURCE_BUDGET.items())):
        raise W09EvaluatorInfrastructureError("W09 private family resource limits 漂移")
    manifest = read_w09_rotation_manifest(rotation_root, expected_sha256=str(family.get("rotation_manifest_sha256")))
    validate_w09_rotation_metadata(rotation_root, manifest)
    if manifest.package_commitment != family.get("rotation_package_commitment"):
        raise W09EvaluatorInfrastructureError("W09 rotation/family commitment 漂移")
    guard_path, guard_sha = consume_w09_private_first_run_guard(family_root, family_freeze_sha256=family_freeze_sha256)
    del guard_path
    phase = "CANDIDATE_VERIFY"
    audit = _ReadAudit({})
    try:
        _enter_phase(config, phase)
        candidate_before = _tree_digest(candidate)
        public_before = _git_state(repository)
        if public_before[0] != family.get("evaluator_public_head_commit_sha1"):
            raise W09EvaluatorInfrastructureError("W09 evaluator public HEAD 漂移")
        if config.require_clean_public and public_before[1]:
            raise W09EvaluatorInfrastructureError("W09 evaluator 要求 public worktree clean")
        candidate_docs = _candidate_documents(candidate, family)
        phase = "CODE_FREEZE_VERIFY"
        _enter_phase(config, phase)
        context = open_w09_frozen_contract(repository)
        training = W09PayloadFirewall.open(repository, context, make_w09_request(context)).read_training_payload()
        inference_state = compile_w09_inference_state(training)
        adapter = W09CandidateInferenceAdapter(inference_state)
        snapshot = snapshot_from_w09_host_document(candidate_docs["host"], inference_state)
        phase = "FAMILY_METADATA_VERIFY"
        _enter_phase(config, phase)
        source_doc = json.loads(documents[W09_PRIVATE_SOURCE_NAME])
        if (
            source_doc.get("fixed_d03_exposure_eligible") != 0
            or source_doc.get("rotation_exposure_audit_clean") != 1
            or source_doc.get("rotation_package", {}).get("package_commitment") != manifest.package_commitment
        ):
            raise W09EvaluatorInfrastructureError("W09 exposure/package metadata 漂移")
        phase = "OBSERVATION_READ"
        _enter_phase(config, phase)
        _read_d03_owner(repository, context, audit, "source")
        _read_rotation_owner(rotation_root, manifest, audit, "source")
        d03_observations = _observations(_read_d03_owner(repository, context, audit, "observation"), family="D03")
        rotation_observations = _observations(_read_rotation_owner(rotation_root, manifest, audit, "observation"), family="ROTATION")
        observations = tuple(sorted((*d03_observations, *rotation_observations), key=lambda item: item[1].stable_key.components))
        if audit.label_records:
            raise W09EvaluatorInfrastructureError("W09 private label 在 inference 前已读取")
        phase = "INFERENCE_INVENTORY"
        _enter_phase(config, phase)
        baseline_outcomes = _infer_observation_family(adapter, observations)
        ablation_families = []
        for index, ablation in enumerate(W09_ABLATION_KEYS):
            if index < len(W09_DIMENSION_KEYS):
                values = _infer_observation_family(adapter, observations, disabled_components=(W09_DIMENSION_KEYS[index],))
            else:
                values = ()
            ablation_families.append((ablation, values))
        if audit.label_records:
            raise W09EvaluatorInfrastructureError("W09 private label 在全部 inference 前已读取")
        phase = "LABEL_READ"
        _enter_phase(config, phase)
        d03_labels = _labels(_read_d03_owner(repository, context, audit, "evaluator"))
        rotation_labels = _labels(_read_rotation_owner(rotation_root, manifest, audit, "evaluator"))
        pairs = _pair_records(observations, tuple((*d03_labels, *rotation_labels)))
        if audit.observation_records != audit.label_records or audit.future_payload_reads:
            raise W09EvaluatorInfrastructureError("W09 private payload read account 不闭合")
        if audit.payload_gets > W09_RESOURCE_BUDGET["max_payload_gets"] or audit.payload_bytes > W09_RESOURCE_BUDGET["max_payload_bytes"] or len(pairs) > W09_RESOURCE_BUDGET["max_records"]:
            raise W09EvaluatorInfrastructureError("W09 private evaluator resource 超限")
        phase = "BASELINE_EVALUATION"
        _enter_phase(config, phase)
        results = evaluate_w09_private_pairs(snapshot, pairs, case_outcomes=baseline_outcomes)
        rotation_keys = {tuple(item.observation.stable_key.components) for item in pairs if item.family_kind == "ROTATION"}
        rotation_pairs = tuple(item for item in pairs if item.family_kind == "ROTATION")
        rotation_outcomes = tuple(item for item in baseline_outcomes if item.observation_key in rotation_keys)
        rotation_results = evaluate_w09_private_pairs(snapshot, rotation_pairs, case_outcomes=rotation_outcomes)
        phase = "ABLATION_EVALUATION"
        _enter_phase(config, phase)
        ablations = assess_w09_orthogonal_ablations(snapshot, pairs, outcome_families=tuple(ablation_families))
        open_generation = assess_w09_private_open_generation(snapshot, pairs, case_outcomes=baseline_outcomes)
        phase = "ZERO_CALL_WINDOWS"
        _enter_phase(config, phase)
        windows = assess_w09_private_windows(snapshot)
        phase = "J_LC_W09"
        _enter_phase(config, phase)
        j_lc = assess_w09_private_j_lc(snapshot, pairs, case_outcomes=baseline_outcomes)
        phase = "V06_CLONE"
        _enter_phase(config, phase)
        rotation_pass = all(item.status == "PASS" for item in rotation_results)
        v06 = assess_w09_private_v06(
            snapshot,
            independent_probe_count=len(rotation_pairs),
            improved_probe_count=len(rotation_pairs) if rotation_pass else 0,
            isolated_learning_write_count=snapshot.learning_event_count,
        )
        phase = "ROLLBACK"
        _enter_phase(config, phase)
        rollback = assess_w09_private_rollback(snapshot, invalidated_count=3, preserved_count=snapshot.learning_event_count, leaked_write_count=0)
        phase = "RESOURCE"
        _enter_phase(config, phase)
        resource = assess_w09_private_resource(snapshot)
        dump_value = {
            "ablation_invocation_commitments": [
                {"ablation_key": key, "outcome_commitment": evidence_commitment([item.safe_dict() for item in values])}
                for key, values in ablation_families
            ],
            "artifact_kind": "PH2_W09_PRIVATE_EVALUATION_DUMP",
            "baseline_invocation_commitment": evidence_commitment([item.safe_dict() for item in baseline_outcomes]),
            "dimension_results": [item.to_safe_dict() for item in results],
            "family_commitment": family["family_key"],
            "format_version": 1,
            "inference_before_label_reads": 1,
            "inference_invocation_count": len(baseline_outcomes) + sum(len(values) for _, values in ablation_families),
            "j_lc": j_lc,
            "open_generation": open_generation,
            "private_read_commitment": evidence_commitment([[count, size] for _, (count, size) in sorted(audit.reads_by_path.items())]),
            "private_read_path_count": len(audit.reads_by_path),
            "resource": resource,
            "rollback": rollback,
            "v06": v06,
            "windows": windows,
        }
        validate_w09_safe_report(dump_value)
        dump_path = execution / W09_PRIVATE_DUMP_NAME
        dump_sha = _write_exclusive(dump_path, canonical_json_bytes(dump_value))
        phase = "DUMP_READBACK"
        _enter_phase(config, phase)
        dump_readback, dump_bytes = _read_canonical(dump_path, label="private dump")
        if dump_readback != dump_value or _sha256(dump_bytes) != dump_sha:
            raise W09EvaluatorInfrastructureError("W09 private dump readback 漂移")
        phase = "INTEGRITY"
        _enter_phase(config, phase)
        candidate_after = _tree_digest(candidate)
        public_after = _git_state(repository)
        writes = _zero_write_counts()
        writes["candidate_writes"] = int(candidate_after != candidate_before)
        writes["public_writes"] = int(public_after != public_before)
        if any(writes.values()):
            raise W09EvaluatorInfrastructureError("W09 private evaluator owner isolation 失败")
        infrastructure = {
            "candidate_inventory_match": 1,
            "d03_fixed_family_eligible": 0,
            "d03_observation_record_count": audit.d03_observation_records,
            "exposure_incident_102_83_retained": 1,
            "future_payload_reads": audit.future_payload_reads,
            "inference_before_label_reads": 1,
            "inference_invocation_count": dump_value["inference_invocation_count"],
            "label_after_all_inference": 1,
            "private_label_record_count": audit.label_records,
            "private_observation_record_count": audit.observation_records,
            "private_payload_bytes": audit.payload_bytes,
            "private_payload_gets": audit.payload_gets,
            "private_read_path_count": len(audit.reads_by_path),
            "rotation_blind_pass_basis": int(rotation_pass),
            "rotation_exposure_audit_clean": 1,
            "rotation_observation_record_count": audit.rotation_observation_records,
            "source_record_count": audit.source_records,
        }
        phase = "REPORT_SAFETY"
        _enter_phase(config, phase)
        aggregate = public_safe_w09_aggregate(
            results,
            family_commitment=family["family_key"],
            payload_commitment=family["payload_commitment"],
            case_commitment=family["case_commitment"],
            label_commitment=family["label_commitment"],
            cluster_commitment=family["cluster_commitment"],
            rotation_package_commitment=family["rotation_package_commitment"],
            failure_phase="NONE",
            formal_run_count=1,
            write_counts=writes,
            ablation_results=ablations,
            windows=windows,
            open_generation=open_generation,
            j_lc=j_lc,
            v06=v06,
            rollback=rollback,
            resource=resource,
            infrastructure=infrastructure,
        )
        if not rotation_pass:
            raise W09EvaluatorInfrastructureError("W09 rotation blind PASS basis 未闭合")
        validate_w09_safe_report(aggregate)
        aggregate_path = family_root / "publication" / W09_PRIVATE_AGGREGATE_NAME
        aggregate_sha = _write_exclusive(aggregate_path, canonical_json_bytes(aggregate))
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS":
            recommendation = {
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": "PH2_W09_RUNTIME_RECEIPT_RECOMMENDATION",
                "family_freeze_sha256": family_freeze_sha256,
                "format_version": 1,
                "j_lc_w09": "PASS",
                "language_capability_mastered": 1,
                "language_readiness": 0,
                "pre_wean_language_learning_capability_evidenced": 1,
                "status": "PASS",
            }
            validate_w09_safe_report(recommendation)
            recommendation_path = family_root / "publication" / W09_PRIVATE_RECOMMENDATION_NAME
            recommendation_sha = _write_exclusive(recommendation_path, canonical_json_bytes(recommendation))
        seal_path, seal_sha = _write_terminal_seal(
            family_root, aggregate_sha256=aggregate_sha, guard_sha256=guard_sha,
            family_freeze_sha256=family_freeze_sha256, status=str(aggregate["status"]),
        )
        return W09PrivateEvaluatorRunResult(
            str(aggregate["status"]), aggregate_path, aggregate_sha,
            recommendation_path, recommendation_sha, family_freeze_sha256,
            guard_sha, dump_sha, seal_path, seal_sha,
        )
    except Exception as error:
        infrastructure = {
            "future_payload_reads": audit.future_payload_reads,
            "private_label_record_count": audit.label_records,
            "private_observation_record_count": audit.observation_records,
            "private_payload_bytes": audit.payload_bytes,
            "private_payload_gets": audit.payload_gets,
            "private_read_path_count": len(audit.reads_by_path),
        }
        if isinstance(error, _W09SanitizedInferenceFailure):
            infrastructure.update(error.infrastructure)
        return _safe_failure(
            family_root, family=family,
            family_freeze_sha256=family_freeze_sha256,
            guard_sha256=guard_sha, phase=phase,
            infrastructure=infrastructure,
        )


__all__ = [
    "W09EvaluatorInfrastructureError",
    "W09EvaluatorInjectedFault",
    "W09PrivateEvaluatorRunResult",
    "W09PrivateEvaluatorRuntimeConfig",
    "run_w09_private_evaluation_once",
]
