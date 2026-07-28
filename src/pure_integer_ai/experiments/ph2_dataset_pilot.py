"""D-02F 全 record kind 小批资料 pilot、恢复、异常隔离和规范报告。"""
from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.ph2_dataset_contract import (
    JSONL_RECORD_KINDS,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    SPLITS,
    W_STAGES,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
    record_kind,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_pilot_probe import (
    PROBE_INPUT_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_pilot_registry import (
    PILOT_PACK_SPECS,
    PilotPackSpec,
    validate_pilot_registry,
)
from pure_integer_ai.experiments.ph2_dataset_pilot_state import (
    PILOT_CLONE_AUDIT_TABLE,
    PILOT_TABLES,
    finish_pilot_state,
    initialize_pilot_state,
    load_pack_results,
    register_pilot_tables,
    store_pack_result,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetValidationError,
    validate_artifact_manifest,
    validate_dataset_bundle,
    validate_stage_visibility,
)
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.telemetry import collect_backend_telemetry


PILOT_REPORT_NAME = "d02f-pilot-report-v1.json"
PILOT_CONTRACT_VERSION = 1
SUPPORTED_WORKER_COUNTS = (1, 2, 4)
FAULT_CODES = (
    "NONCANONICAL_SAMPLE",
    "BAD_SOURCE",
    "BAD_LICENSE",
    "BAD_SUPERSEDE",
    "BAD_PERTURBATION",
)


class DatasetPilotError(RuntimeError):
    """D-02F 编译、读取、隔离、恢复或规范报告不满足合同。"""


class DatasetPilotInterrupted(DatasetPilotError):
    """测试/恢复探针在 pack 已发布但 cursor 未提交时中断。"""


@dataclass(frozen=True)
class _PackBundle:
    """一个已从 manifest 和 gzip artifact 严格重读的 pack。"""

    spec: PilotPackSpec
    sample_sha256: str
    compiler_sha256: str
    manifest: Any
    sources: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teachers: tuple[TeacherEvidenceRecord, ...]
    evaluators: tuple[EvaluatorLabelRecord, ...]


@dataclass(frozen=True)
class DatasetPilotRunResult:
    """返回规范 report 与不进入规范 hash 的执行维度。"""

    report: CanonicalJsonObject
    report_path: Path
    normative_sha256: str
    contract_sha256: str
    backend_kind: str
    worker_count: int
    resumed_pack_count: int
    published_pack_count: int

    def to_dict(self) -> dict[str, Any]:
        """导出供 CI/进度记录使用的执行与规范摘要。"""
        return {
            "backend_kind": self.backend_kind,
            "contract_sha256": self.contract_sha256,
            "normative_sha256": self.normative_sha256,
            "published_pack_count": self.published_pack_count,
            "report_path": self.report_path.as_posix(),
            "resumed_pack_count": self.resumed_pack_count,
            "worker_count": self.worker_count,
        }


def _sha256_path(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    """计算无浮点规范 JSON 值的 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _input_identity(
        spec: PilotPackSpec,
        repository_root: Path,
        ) -> tuple[str, str]:
    """绑定 sample/probe 与 compiler facade 的当前字节。"""
    compiler = spec.load_compiler()
    source_file = inspect.getsourcefile(compiler)
    if source_file is None:
        raise DatasetPilotError(f"pilot compiler 缺少源码文件: {spec.pack_name}")
    compiler_path = Path(source_file).resolve()
    if not compiler_path.is_file():
        raise DatasetPilotError(f"pilot compiler 源码缺失: {spec.pack_name}")
    sample_path = spec.sample_path(repository_root)
    sample_sha256 = (
        PROBE_INPUT_SHA256 if sample_path is None else _sha256_path(sample_path))
    return sample_sha256, _sha256_path(compiler_path)


def _normalize_faults(
        faults: Mapping[int, str] | None,
        specs: tuple[PilotPackSpec, ...],
        ) -> dict[int, str]:
    """只允许一次精确失败注入，且不得把 synthetic probe 当坏 sample。"""
    result = dict(faults or {})
    if len(result) > 1:
        raise DatasetPilotError("D-02F 一次只允许隔离一个坏输入")
    index = {spec.pack_id: spec for spec in specs}
    for pack_id, code in result.items():
        if type(pack_id) is not int or pack_id not in index:
            raise DatasetPilotError("pilot fault pack_id 不在 registry")
        if code not in FAULT_CODES:
            raise DatasetPilotError("pilot fault code 未注册")
        if index[pack_id].synthetic:
            raise DatasetPilotError("synthetic split probe 不接受 sample fault")
    return result


def _contract_payload(
        repository_root: Path,
        specs: tuple[PilotPackSpec, ...],
        faults: dict[int, str],
        ) -> tuple[dict[str, Any], dict[int, tuple[str, str]]]:
    """形成不含 backend/worker/path 的输入和 compiler 规范合同。"""
    identities: dict[int, tuple[str, str]] = {}
    packs: list[dict[str, Any]] = []
    for spec in specs:
        sample_sha256, compiler_sha256 = _input_identity(
            spec, repository_root)
        identities[spec.pack_id] = (sample_sha256, compiler_sha256)
        packs.append({
            "compiler_name": spec.compiler_name,
            "compiler_sha256": compiler_sha256,
            "module_name": spec.module_name,
            "pack_id": spec.pack_id,
            "pack_name": spec.pack_name,
            "sample_relative_path": spec.sample_relative_path,
            "sample_sha256": sample_sha256,
            "stage": spec.stage,
            "substage": spec.substage,
            "synthetic": 1 if spec.synthetic else 0,
        })
    return ({
        "contract_version": PILOT_CONTRACT_VERSION,
        "faults": [
            {"code": faults[pack_id], "pack_id": pack_id}
            for pack_id in sorted(faults)
        ],
        "packs": packs,
    }, identities)


def _execution_order(
        specs: tuple[PilotPackSpec, ...], worker_count: int,
        ) -> tuple[PilotPackSpec, ...]:
    """让 worker 数真实改变 shard 调度，但不改变最终稳定排序。"""
    shards = tuple(
        tuple(spec for spec in specs if (spec.pack_id - 1) % worker_count == shard)
        for shard in range(worker_count)
    )
    return tuple(spec for shard in shards for spec in shard)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    """独占或核对一个规范 pilot 输入/报告文件。"""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise DatasetPilotError(f"pilot 已有文件内容漂移: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.building-", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        handle.write(payload)
        handle.close()
        os.replace(temporary, path)
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()


def _sample_values(path: Path) -> list[dict[str, Any]]:
    """严格恢复 sample 为独立可变 object，供失败注入。"""
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise DatasetPilotError("pilot fault sample 缺少规范换行")
    values: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise DatasetPilotError("pilot fault sample 行边界非法")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        assert isinstance(value, dict)
        values.append(value)
    return values


def _fault_sample(
        spec: PilotPackSpec,
        source: Path,
        release_root: Path,
        fault_code: str,
        ) -> Path:
    """创建单点坏许可/supersede/扰动或非规范 JSON 输入。"""
    target = (
        release_root / "pilot_inputs"
        / f"{spec.pack_id:02d}-{fault_code.casefold()}.jsonl.sample")
    original = source.read_bytes()
    if fault_code == "BAD_SOURCE":
        return source
    if fault_code == "NONCANONICAL_SAMPLE":
        if not original.startswith(b"{"):
            raise DatasetPilotError("pilot sample 首行不是 JSON object")
        _exclusive_bytes(target, b"{ " + original[1:])
        return target
    values = _sample_values(source)
    if fault_code == "BAD_LICENSE":
        values[0]["license_id"] = "UNKNOWN"
    elif fault_code == "BAD_SUPERSEDE":
        candidates = [
            value for value in values if value.get("sample_role") == "supersede"]
        if not candidates:
            raise DatasetPilotError("fault pack 缺少 supersede seed")
        candidates[0]["supersedes_seed_id"] = candidates[0].get("seed_id")
    elif fault_code == "BAD_PERTURBATION":
        if "perturbation_kind" not in values[0]:
            raise DatasetPilotError("fault pack 缺少 perturbation_kind")
        values[0]["perturbation_kind"] = "UNREGISTERED_PERTURBATION"
    else:
        raise DatasetPilotError("fault sample code 非法")
    payload = b"".join(canonical_json_line(value) for value in values)
    _exclusive_bytes(target, payload)
    return target


def _compile_pack(
        spec: PilotPackSpec,
        sample_path: Path | None,
        release_root: Path,
        ) -> bool:
    """若 pack 尚不存在则调用固定 compiler；返回本次是否新发布。"""
    pack_root = release_root / "packs" / spec.pack_name
    if pack_root.exists():
        if not pack_root.is_dir():
            raise DatasetPilotError("pilot pack 路径不是目录")
        return False
    compiler = spec.load_compiler()
    if spec.synthetic:
        compiler(release_root)
    else:
        if sample_path is None:
            raise DatasetPilotError("authored pilot pack 缺少 sample")
        compiler(sample_path, release_root)
    if not (pack_root / "manifest.json").is_file():
        raise DatasetPilotError("pilot compiler 未发布 manifest")
    return True


def _read_pack(
        spec: PilotPackSpec,
        release_root: Path,
        *,
        sample_sha256: str,
        compiler_sha256: str,
        ) -> _PackBundle:
    """逐文件核验 manifest、双 hash、计数、许可、split 和引用图。"""
    pack_root = release_root / "packs" / spec.pack_name
    manifest_path = pack_root / "manifest.json"
    manifest = read_artifact_manifest(manifest_path)
    if manifest.sha256() != _sha256_path(manifest_path):
        raise DatasetPilotError("pilot manifest hash 与规范字节不一致")
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    for identity in manifest.files:
        records = read_record_artifact(pack_root, identity)
        if len(records) != identity.record_count:
            raise DatasetPilotError("pilot artifact record_count 漂移")
        for record in records:
            if isinstance(record, SourceRefRecord):
                sources.append(record)
            elif isinstance(record, ObservationRecord):
                observations.append(record)
            elif isinstance(record, TeacherEvidenceRecord):
                teachers.append(record)
            elif isinstance(record, EvaluatorLabelRecord):
                evaluators.append(record)
            else:
                raise DatasetPilotError("pilot artifact 含未知 record")
    source_tuple = tuple(sources)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teachers)
    evaluator_tuple = tuple(evaluators)
    if (not source_tuple or not observation_tuple
            or not teacher_tuple or not evaluator_tuple):
        raise DatasetPilotError("pilot pack 未覆盖四类 JSONL record")
    if manifest.record_count != sum((
            len(source_tuple), len(observation_tuple),
            len(teacher_tuple), len(evaluator_tuple))):
        raise DatasetPilotError("pilot manifest 总记录数漂移")
    if any(source.local_sha256 != sample_sha256 for source in source_tuple):
        raise DatasetPilotError("pilot SourceRef 未绑定当前 sample/probe hash")
    if manifest.w_stages != (spec.stage,):
        raise DatasetPilotError("pilot manifest W stage 与 registry 漂移")
    if any(item.substage != spec.substage for item in observation_tuple):
        raise DatasetPilotError("pilot Observation substage 与 registry 漂移")
    validate_artifact_manifest(manifest, source_tuple, observation_tuple)
    validate_dataset_bundle(
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
        source_key=manifest.source_key,
        license_partition=manifest.license_partition,
        public_release=manifest.redistribution_policy == "PUBLIC",
    )
    return _PackBundle(
        spec,
        sample_sha256,
        compiler_sha256,
        manifest,
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
    )


def _records(bundle: _PackBundle) -> tuple[Any, ...]:
    """返回一个 pack 的四类正式记录。"""
    return (
        bundle.sources + bundle.observations
        + bundle.teachers + bundle.evaluators)


def _record_digest(bundle: _PackBundle) -> str:
    """按 record kind/full stable key 形成规范记录聚合摘要。"""
    ordered = sorted(
        _records(bundle),
        key=lambda item: (record_kind(item), item.stable_key.components),
    )
    return _sha256_value([item.to_dict() for item in ordered])


def _pass_result(bundle: _PackBundle) -> dict[str, Any]:
    """形成不含执行顺序、backend、worker 和路径根的 pack 证据。"""
    manifest = bundle.manifest
    return {
        "anomaly_code": None,
        "compiler_sha256": bundle.compiler_sha256,
        "file_count": len(manifest.files),
        "files": [
            {
                "content_sha256": item.content_sha256,
                "owner_kind": item.owner_kind,
                "record_count": item.record_count,
                "record_kind": item.record_kind,
                "relative_path": item.relative_path,
                "source_cluster_keys": [
                    key.to_list() for key in item.source_cluster_keys],
                "split": item.split,
            }
            for item in manifest.files
        ],
        "license_partition": manifest.license_partition,
        "manifest_key": manifest.stable_key.to_list(),
        "manifest_sha256": manifest.content_sha256(),
        "pack_id": bundle.spec.pack_id,
        "pack_name": bundle.spec.pack_name,
        "record_aggregate_sha256": _record_digest(bundle),
        "record_counts": {
            RECORD_EVALUATOR_LABEL: len(bundle.evaluators),
            RECORD_OBSERVATION: len(bundle.observations),
            RECORD_SOURCE_REF: len(bundle.sources),
            RECORD_TEACHER_EVIDENCE: len(bundle.teachers),
        },
        "redistribution_policy": manifest.redistribution_policy,
        "sample_relative_path": bundle.spec.sample_relative_path,
        "sample_sha256": bundle.sample_sha256,
        "source_cluster_keys": [
            key.to_list() for key in manifest.source_cluster_keys],
        "splits": list(manifest.splits),
        "stage": bundle.spec.stage,
        "status": "PASS",
        "substage": bundle.spec.substage,
        "synthetic": 1 if bundle.spec.synthetic else 0,
    }


def _anomaly_result(
        spec: PilotPackSpec,
        fault_code: str,
        *,
        sample_sha256: str,
        compiler_sha256: str,
        error: Exception,
        ) -> dict[str, Any]:
    """只记录稳定错误类型/代码，不把临时绝对路径写入规范报告。"""
    error_type = type(error).__name__
    return {
        "anomaly_code": fault_code,
        "compiler_sha256": compiler_sha256,
        "error_identity_sha256": _sha256_value({
            "anomaly_code": fault_code,
            "error_type": error_type,
            "pack_id": spec.pack_id,
        }),
        "error_type": error_type,
        "file_count": 0,
        "files": [],
        "license_partition": None,
        "manifest_key": None,
        "manifest_sha256": None,
        "pack_id": spec.pack_id,
        "pack_name": spec.pack_name,
        "record_aggregate_sha256": None,
        "record_counts": {kind: 0 for kind in JSONL_RECORD_KINDS},
        "redistribution_policy": None,
        "sample_relative_path": spec.sample_relative_path,
        "sample_sha256": sample_sha256,
        "source_cluster_keys": [],
        "splits": [],
        "stage": spec.stage,
        "status": "ANOMALY",
        "substage": spec.substage,
        "synthetic": 0,
    }


def _execute_pack(
        spec: PilotPackSpec,
        repository_root: Path,
        release_root: Path,
        identity: tuple[str, str],
        fault_code: str | None,
        *,
        interrupt_after_publish_pack_id: int | None,
        ) -> tuple[dict[str, Any], bool]:
    """执行一个 pack；正常错误 fail-closed，注入错误形成单一 anomaly。"""
    original_sample = spec.sample_path(repository_root)
    sample_path = original_sample
    if fault_code is not None:
        if original_sample is None:
            raise DatasetPilotError("synthetic pack 不支持 fault")
        sample_path = _fault_sample(
            spec, original_sample, release_root, fault_code)
    actual_sample_sha256 = (
        PROBE_INPUT_SHA256 if sample_path is None else _sha256_path(sample_path))
    compiler_sha256 = identity[1]

    def run_once() -> tuple[dict[str, Any], bool]:
        published = _compile_pack(spec, sample_path, release_root)
        if (published
                and interrupt_after_publish_pack_id == spec.pack_id):
            raise DatasetPilotInterrupted(
                f"pilot pack {spec.pack_id} 已发布、cursor 尚未提交")
        bundle = _read_pack(
            spec,
            release_root,
            sample_sha256=actual_sample_sha256,
            compiler_sha256=compiler_sha256,
        )
        if fault_code == "BAD_SOURCE":
            damaged = (replace(
                bundle.sources[0], source_key="UNREGISTERED_SOURCE"),
                ) + bundle.sources[1:]
            validate_artifact_manifest(
                bundle.manifest, damaged, bundle.observations)
            raise DatasetPilotError("BAD_SOURCE 未被 manifest 校验拒绝")
        return _pass_result(bundle), published

    if fault_code is None:
        return run_once()
    try:
        result, published = run_once()
    except DatasetPilotInterrupted:
        raise
    except Exception as error:
        return (_anomaly_result(
            spec,
            fault_code,
            sample_sha256=actual_sample_sha256,
            compiler_sha256=compiler_sha256,
            error=error,
        ), (release_root / "packs" / spec.pack_name).is_dir())
    raise DatasetPilotError(
        f"fault {fault_code} 被 compiler/validator 静默接受: {result['pack_name']}")


def _reload_pass_bundles(
        specs: tuple[PilotPackSpec, ...],
        results: dict[int, dict[str, Any]],
        release_root: Path,
        identities: dict[int, tuple[str, str]],
        ) -> tuple[_PackBundle, ...]:
    """从正式 artifact 重读全部 PASS pack，拒绝 backend 结果冒充磁盘事实。"""
    bundles: list[_PackBundle] = []
    for spec in specs:
        result = results[spec.pack_id]
        if result.get("status") != "PASS":
            continue
        bundle = _read_pack(
            spec,
            release_root,
            sample_sha256=str(result["sample_sha256"]),
            compiler_sha256=identities[spec.pack_id][1],
        )
        if _pass_result(bundle) != result:
            raise DatasetPilotError("backend pack result 与重读 artifact 漂移")
        bundles.append(bundle)
    return tuple(bundles)


def _validate_global_keys(bundles: tuple[_PackBundle, ...]) -> None:
    """要求所有 pack 的四类 stable key 在 pilot 聚合层仍全局唯一。"""
    owners: dict[Any, str] = {}
    for bundle in bundles:
        for record in _records(bundle):
            prior = owners.get(record.stable_key)
            if prior is not None:
                raise DatasetPilotError(
                    f"pilot 全局 stable key 碰撞: {prior}/{bundle.spec.pack_name}")
            owners[record.stable_key] = bundle.spec.pack_name


def _stage_visibility_audit(
        bundles: tuple[_PackBundle, ...],
        ) -> tuple[int, int]:
    """验证 18 个合法视图，并对每个非末阶段双向注入未来资料负例。"""
    observations = tuple(
        item for bundle in bundles for item in bundle.observations)
    teachers = tuple(item for bundle in bundles for item in bundle.teachers)
    evaluators = tuple(item for bundle in bundles for item in bundle.evaluators)
    valid_checks = 0
    future_rejections = 0
    for current_stage in W_STAGES:
        current_rank = W_STAGES.index(current_stage)
        train_observations = tuple(
            item for item in observations
            if W_STAGES.index(item.w_stage) <= current_rank
            and item.split == "train")
        train_keys = {item.stable_key for item in train_observations}
        train_teachers = tuple(
            item for item in teachers if item.observation_key in train_keys)
        validate_stage_visibility(
            train_observations,
            train_teachers,
            (),
            current_stage=current_stage,
            view_kind="training",
        )
        valid_checks += 1
        evaluation_observations = tuple(
            item for item in observations
            if W_STAGES.index(item.w_stage) <= current_rank
            and item.split != "train")
        evaluation_keys = {
            item.stable_key for item in evaluation_observations}
        evaluation_labels = tuple(
            item for item in evaluators
            if item.observation_key in evaluation_keys)
        validate_stage_visibility(
            evaluation_observations,
            (),
            evaluation_labels,
            current_stage=current_stage,
            view_kind="evaluation",
        )
        valid_checks += 1
        future_train = next((
            item for item in observations
            if W_STAGES.index(item.w_stage) > current_rank
            and item.split == "train"), None)
        future_evaluation = next((
            item for item in observations
            if W_STAGES.index(item.w_stage) > current_rank
            and item.split != "train"), None)
        for view_kind, current, future, owner_records in (
                ("training", train_observations, future_train, train_teachers),
                ("evaluation", evaluation_observations,
                 future_evaluation, evaluation_labels)):
            if future is None:
                continue
            try:
                validate_stage_visibility(
                    current + (future,),
                    owner_records if view_kind == "training" else (),
                    owner_records if view_kind == "evaluation" else (),
                    current_stage=current_stage,
                    view_kind=view_kind,
                )
            except DatasetValidationError:
                future_rejections += 1
            else:
                raise DatasetPilotError("pilot 未来阶段负例未被拒绝")
    return valid_checks, future_rejections


def _physical_split_audit(
        results: dict[int, dict[str, Any]],
        specs: tuple[PilotPackSpec, ...],
        ) -> dict[str, Any]:
    """直接核对 synthetic probe 的四个 Observation 和三类 owner 文件。"""
    probe = next((spec for spec in specs if spec.synthetic), None)
    if probe is None:
        raise DatasetPilotError("pilot registry 缺少 split probe")
    result = results[probe.pack_id]
    if result.get("status") != "PASS":
        raise DatasetPilotError("split probe 未通过")
    files = result.get("files")
    if not isinstance(files, list):
        raise DatasetPilotError("split probe files 非法")
    observation_paths = {
        item["split"]: item["relative_path"]
        for item in files if item["record_kind"] == RECORD_OBSERVATION
    }
    expected_observations = {
        split: f"observations/{split}.jsonl.gz" for split in SPLITS[:4]}
    if observation_paths != expected_observations:
        raise DatasetPilotError("四 split Observation 未物理隔离")
    teacher_files = [
        item for item in files
        if item["record_kind"] == RECORD_TEACHER_EVIDENCE]
    evaluator_files = [
        item for item in files
        if item["record_kind"] == RECORD_EVALUATOR_LABEL]
    if ([item["split"] for item in teacher_files] != ["train"]
            or {item["split"] for item in evaluator_files}
            != set(SPLITS[1:4])):
        raise DatasetPilotError("teacher/evaluator owner 未按 split 物理隔离")
    cluster_sets = {
        split: tuple(next(
            item["source_cluster_keys"] for item in files
            if item["record_kind"] == RECORD_OBSERVATION
            and item["split"] == split))
        for split in SPLITS[:4]
    }
    flattened = [key for keys in cluster_sets.values() for key in keys]
    if len(flattened) != len({tuple(key) for key in flattened}):
        raise DatasetPilotError("split probe 来源簇跨 split")
    return {
        "observation_paths": observation_paths,
        "owner_file_count": len(teacher_files) + len(evaluator_files),
        "physical_split_isolation": 1,
        "source_cluster_disjoint": 1,
    }


def _readonly_clone_audit(
        backend: StorageBackend,
        bundles: tuple[_PackBundle, ...],
        ) -> tuple[int, str]:
    """核验 held-out/evaluator 纯读，且 V-06 backend clone 写不回宿主。"""
    held_out = tuple(
        item for bundle in bundles for item in bundle.observations
        if item.split != "train")
    evaluator_labels = tuple(
        item for bundle in bundles for item in bundle.evaluators)
    baseline = backend.recovery_state_snapshot()
    digest = _sha256_value({
        "evaluator_labels": [item.to_dict() for item in evaluator_labels],
        "held_out_observations": [item.to_dict() for item in held_out],
    })
    if backend.recovery_state_snapshot() != baseline:
        raise DatasetPilotError("held-out/evaluator 读取改变了 pilot backend")
    cloned = clone_backend(backend)
    try:
        cloned.insert(PILOT_CLONE_AUDIT_TABLE, {
            "run_id": 1,
            "audit_kind": "V06_READ_ONLY_CLONE",
            "record_count": len(held_out) + len(evaluator_labels),
        })
        cloned.commit()
        if cloned.count(
                PILOT_CLONE_AUDIT_TABLE,
                {"run_id": 1, "audit_kind": "V06_READ_ONLY_CLONE"}) != 1:
            raise DatasetPilotError("V-06 clone audit 未写入 clone")
    finally:
        cloned.close()
    if backend.recovery_state_snapshot() != baseline:
        raise DatasetPilotError("V-06 clone 写回了宿主 backend")
    if backend.count(PILOT_CLONE_AUDIT_TABLE, {"run_id": 1}) != 0:
        raise DatasetPilotError("V-06 clone audit 污染宿主")
    return len(held_out) + len(evaluator_labels), digest


def _aggregate_report(
        contract_sha256: str,
        results: dict[int, dict[str, Any]],
        specs: tuple[PilotPackSpec, ...],
        bundles: tuple[_PackBundle, ...],
        *,
        valid_view_checks: int,
        future_rejections: int,
        physical_split: dict[str, Any],
        readonly_count: int,
        readonly_digest: str,
        training_state_write_count: int,
        ) -> dict[str, Any]:
    """形成跨 backend/worker/resume bit-identical 的 pilot 规范报告。"""
    ordered_results = [results[spec.pack_id] for spec in specs]
    pass_results = [
        result for result in ordered_results if result["status"] == "PASS"]
    anomalies = [
        {
            "anomaly_code": result["anomaly_code"],
            "error_identity_sha256": result["error_identity_sha256"],
            "error_type": result["error_type"],
            "pack_id": result["pack_id"],
            "pack_name": result["pack_name"],
        }
        for result in ordered_results if result["status"] == "ANOMALY"
    ]
    totals = {kind: 0 for kind in JSONL_RECORD_KINDS}
    split_counts = {split: 0 for split in SPLITS[:4]}
    source_clusters: set[tuple[int, ...]] = set()
    for bundle in bundles:
        totals[RECORD_SOURCE_REF] += len(bundle.sources)
        totals[RECORD_OBSERVATION] += len(bundle.observations)
        totals[RECORD_TEACHER_EVIDENCE] += len(bundle.teachers)
        totals[RECORD_EVALUATOR_LABEL] += len(bundle.evaluators)
        for observation in bundle.observations:
            if observation.split in split_counts:
                split_counts[observation.split] += 1
        source_clusters.update(
            key.components for key in bundle.manifest.source_cluster_keys)
    if any(totals[kind] <= 0 for kind in JSONL_RECORD_KINDS):
        raise DatasetPilotError("pilot 聚合未覆盖全部 record kind")
    if any(split_counts[split] <= 0 for split in SPLITS[:4]):
        raise DatasetPilotError("pilot 聚合未覆盖四 split")
    if training_state_write_count != 0:
        raise DatasetPilotError("pilot 发生非 pilot/training-state backend 写入")
    pipeline_usable = 1 if not anomalies else 0
    return {
        "all_required_record_kinds_present": 1,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "artifact_kind": "PH2_D02F_DATASET_PILOT",
        "companion_enabled": 0,
        "contract_sha256": contract_sha256,
        "contract_version": PILOT_CONTRACT_VERSION,
        "evaluator_training_state_write_count": 0,
        "execution_dimensions_excluded_from_normative_hash": [
            "backend_kind", "release_root", "resume_history", "worker_count"],
        "formal_training_started": 0,
        "future_stage_leak_rejection_count": future_rejections,
        "held_out_training_state_write_count": 0,
        "mastered": 0,
        "memory_enabled": 0,
        "normative_pack_aggregate_sha256": _sha256_value(pass_results),
        "pack_count": len(ordered_results),
        "packs": ordered_results,
        "physical_split_audit": physical_split,
        "pilot_only_no_mastery": 1,
        "pipeline_usable": pipeline_usable,
        "read_only_record_count": readonly_count,
        "read_only_record_sha256": readonly_digest,
        "record_counts": totals,
        "source_cluster_count": len(source_clusters),
        "split_observation_counts": split_counts,
        "stage_visibility_valid_view_count": valid_view_checks,
        "successful_pack_count": len(pass_results),
        "supported_backends": ["DictBackend", "SQLiteBackend"],
        "supported_worker_counts": list(SUPPORTED_WORKER_COUNTS),
        "teacher_call_count": 0,
        "training_state_write_count": training_state_write_count,
        "v06_clone_host_write_count": 0,
        "v06_clone_training_state_write_count": 0,
    }


def run_dataset_pilot(
        repository_root: str | Path,
        release_root: str | Path,
        backend: StorageBackend,
        *,
        worker_count: int = 1,
        faults: Mapping[int, str] | None = None,
        interrupt_after_publish_pack_id: int | None = None,
        pack_specs: tuple[PilotPackSpec, ...] = PILOT_PACK_SPECS,
        ) -> DatasetPilotRunResult:
    """运行/恢复 D-02F；只编译和校验资料，绝不调用正式训练或形成 mastered。"""
    specs = validate_pilot_registry(pack_specs)
    if worker_count not in SUPPORTED_WORKER_COUNTS:
        raise DatasetPilotError("pilot worker_count 只支持 1/2/4")
    if (interrupt_after_publish_pack_id is not None
            and interrupt_after_publish_pack_id
            not in {spec.pack_id for spec in specs}):
        raise DatasetPilotError("interrupt pack_id 不在 registry")
    repository = Path(repository_root).resolve()
    if not (repository / "src" / "pure_integer_ai").is_dir():
        raise DatasetPilotError("repository_root 不是权威 pure_integer_ai 仓库")
    release = Path(release_root).resolve()
    if release == repository:
        raise DatasetPilotError("pilot release_root 不得等于 Git 仓库根")
    release.mkdir(parents=True, exist_ok=True)
    normalized_faults = _normalize_faults(faults, specs)
    contract_payload, identities = _contract_payload(
        repository, specs, normalized_faults)
    contract_sha256 = _sha256_value(contract_payload)
    release_binding_sha256 = _sha256_value({
        "contract_sha256": contract_sha256,
        "release_root": str(release),
    })
    published_count = 0
    with collect_backend_telemetry() as telemetry:
        register_pilot_tables(backend)
        initialize_pilot_state(
            backend,
            contract_sha256=contract_sha256,
            release_binding_sha256=release_binding_sha256,
        )
        initial_results = load_pack_results(backend)
        unknown = set(initial_results) - {spec.pack_id for spec in specs}
        if unknown:
            raise DatasetPilotError("pilot backend 含 registry 外 pack result")
        resumed_count = len(initial_results)
        for spec in _execution_order(specs, worker_count):
            if spec.pack_id in initial_results:
                continue
            result, published = _execute_pack(
                spec,
                repository,
                release,
                identities[spec.pack_id],
                normalized_faults.get(spec.pack_id),
                interrupt_after_publish_pack_id=interrupt_after_publish_pack_id,
            )
            published_count += 1 if published else 0
            store_pack_result(backend, spec.pack_id, result)
            initial_results[spec.pack_id] = result
        results = load_pack_results(backend)
        expected_ids = {spec.pack_id for spec in specs}
        if set(results) != expected_ids:
            raise DatasetPilotError("pilot 未形成完整 pack result 集")
        bundles = _reload_pass_bundles(
            specs, results, release, identities)
        _validate_global_keys(bundles)
        valid_views, future_rejections = _stage_visibility_audit(bundles)
        physical_split = _physical_split_audit(results, specs)
        readonly_count, readonly_digest = _readonly_clone_audit(
            backend, bundles)
        operation_snapshot = telemetry.operation_snapshot()
        non_pilot_write_count = sum(
            rows
            for (operation, table), (_, rows, _) in operation_snapshot.items()
            if operation in {"insert", "update", "delete"}
            and table not in PILOT_TABLES
        )
        report = _aggregate_report(
            contract_sha256,
            results,
            specs,
            bundles,
            valid_view_checks=valid_views,
            future_rejections=future_rejections,
            physical_split=physical_split,
            readonly_count=readonly_count,
            readonly_digest=readonly_digest,
            training_state_write_count=non_pilot_write_count,
        )
        finish_pilot_state(
            backend, report, expected_pack_count=len(specs))
    report_payload = canonical_json_line(report)
    report_path = release / PILOT_REPORT_NAME
    _exclusive_bytes(report_path, report_payload)
    normative_sha256 = hashlib.sha256(report_payload).hexdigest()
    return DatasetPilotRunResult(
        CanonicalJsonObject.from_value(report),
        report_path,
        normative_sha256,
        contract_sha256,
        type(backend).__name__,
        worker_count,
        resumed_count,
        published_count,
    )


__all__ = [
    "DatasetPilotError",
    "DatasetPilotInterrupted",
    "DatasetPilotRunResult",
    "FAULT_CODES",
    "PILOT_CONTRACT_VERSION",
    "PILOT_REPORT_NAME",
    "SUPPORTED_WORKER_COUNTS",
    "run_dataset_pilot",
]
