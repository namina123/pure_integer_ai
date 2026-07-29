"""冻结 W-02 host 的 V-06 clone-only private evaluator。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_backend_evaluation,
)
from pure_integer_ai.experiments.ph2_authored_morphology_course import (
    PAYLOAD_KIND as MORPHOLOGY_PAYLOAD_KIND,
    validate_morphology_payload,
)
from pure_integer_ai.experiments.ph2_authored_text_fidelity_course import (
    PAYLOAD_KIND as TEXT_FIDELITY_PAYLOAD_KIND,
    validate_text_fidelity_payload,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w02_contract import (
    W02_STAGE_KEY,
    open_w02_frozen_context,
)
from pure_integer_ai.experiments.ph2_w02_evaluation_contract import (
    GENERATION_HARD_CONJUNCT,
    STATUS_FAIL,
    STATUS_NE,
    STATUS_PASS,
    W02_ABLATION_ORDER,
    W02_BEARING_DIMENSIONS,
    W02_EVALUATION_ORDER,
    W02AblationResult,
    W02DimensionResult,
    W02EvaluationError,
    W02PrivateEvaluationReport,
    aggregate_w02_evaluation,
)
from pure_integer_ai.experiments.ph2_w02_learning import (
    GENERATION_UNKNOWN,
    SELECTION_ADOPTED,
    SELECTION_CONFLICT,
    W02MorphologyTarget,
    open_w02_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w02_use import W02UseOutcomeStore
from pure_integer_ai.storage.backend import SQLiteBackend, StorageBackend


_FORMAT_VERSION = 1
_PRIVATE_PATH_COUNT = 7
_UNDERSTANDING_PROBE = "研究生命起源"
_OOV_PROBE = "陌生连续串"
_NEW_CONTENT_TARGET = W02MorphologyTarget(
    "suffix-hua-construction-v1", "纸")
_OOV_TARGET = W02MorphologyTarget(
    "suffix-hua-construction-v1", "未登录词")


@dataclass(frozen=True)
class W02EvaluationConfig:
    """绑定 Git/D-03 和 Git 外正式 candidate host freeze。"""

    repository_root: str | Path
    global_manifest_path: str
    candidate_host_freeze_path: str | Path
    current_remote_commit_sha1: str
    dependency_root: str | Path | None = None


@dataclass
class _PrivateAudit:
    """只在 clone 内累计 private transport 和 record reads。"""

    path_reads: int = 0
    payload_bytes: int = 0
    observation_reads: int = 0
    label_reads: int = 0
    source_reads: int = 0
    ud_observation_reads: int = 0
    label_writes: int = 0


@dataclass(frozen=True)
class _PrivatePayload:
    """clone 生命周期内的 private records；不得进入公开 report。"""

    sources: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    labels: tuple[EvaluatorLabelRecord, ...]


@dataclass(frozen=True)
class _Mode:
    """四个正交机制开关。"""

    withdrawal: bool = True
    multi_candidate: bool = True
    morphology: bool = True
    oov: bool = True


@dataclass(frozen=True)
class _EvaluationPass:
    """一次 baseline/ablation 的分维结果和 clone-only 全证据摘要。"""

    dimensions: tuple[W02DimensionResult, ...]
    generation: W02DimensionResult
    evidence_sha256: str


class _PrivatePermit:
    """只由 isolated evaluator block 创建的 private reader permit。"""

    def __init__(self, marker: object) -> None:
        self.marker = marker


_PERMIT_MARKER = object()


def _digest(value: Any) -> str:
    """返回 canonical SHA-256，不输出输入内容。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    """在一个 root 内解析规范 POSIX 相对路径。"""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise W02EvaluationError("evaluator path 非规范")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root):
        raise W02EvaluationError("evaluator path 逃逸")
    return target


def _overlay(primary: Path, dependency: Path, relative: str) -> Path:
    """按 D-03 overlay 顺序返回唯一现存文件。"""
    first = _safe_path(primary, relative)
    if first.is_file():
        return first
    second = _safe_path(dependency, relative)
    if second.is_file():
        return second
    raise W02EvaluationError(f"冻结文件缺失: {relative}")


def _read_freeze(config: W02EvaluationConfig) -> tuple[dict[str, Any], Path]:
    """读取 canonical self-excluded host freeze 并核公开固定项。"""
    freeze_path = Path(config.candidate_host_freeze_path).resolve()
    try:
        payload = freeze_path.read_bytes()
        value = parse_canonical_json_bytes(payload, require_object=True)
    except (OSError, TypeError, ValueError) as exc:
        raise W02EvaluationError("candidate host freeze 无法规范读取") from exc
    assert isinstance(value, dict)
    if canonical_json_bytes(value) != payload:
        raise W02EvaluationError("candidate host freeze 非 canonical bytes")
    required = {
        "ablation_order", "artifact_inventory", "artifact_kind",
        "artifact_version", "base_fence_key", "code_inventory",
        "d03_thresholds", "evaluation_order", "execution_state",
        "format_version", "host_digests", "owner_write_counts",
        "publication_counts", "remote_commit_sha1", "resource_actual",
        "run_id", "self_excluded", "test_inventory",
    }
    if set(value) != required:
        raise W02EvaluationError("candidate host freeze 字段漂移")
    if (value["artifact_kind"] != "PH2_W02_CANDIDATE_HOST_FREEZE"
            or value["format_version"] != _FORMAT_VERSION
            or value["self_excluded"] != 1
            or value["run_id"] != 2
            or value["remote_commit_sha1"] != config.current_remote_commit_sha1
            or tuple(value["evaluation_order"]) != W02_EVALUATION_ORDER
            or tuple(value["ablation_order"]) != W02_ABLATION_ORDER
            or value["d03_thresholds"] != {
                "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
                "max_fail_count": 0,
                "min_pass_denominator": 1,
                "min_pass_numerator": 1,
                "ne_policy": "BLOCK",
            }):
        raise W02EvaluationError("candidate host preregistration 漂移")
    return value, freeze_path.parent


def _verify_file_inventory(
        root: Path,
        inventory: list[dict[str, Any]],
        ) -> None:
    """逐项核 size/SHA，拒绝 host/code/test 任一漂移。"""
    if not isinstance(inventory, list) or not inventory:
        raise W02EvaluationError("freeze inventory 为空")
    seen: set[str] = set()
    for item in inventory:
        if (not isinstance(item, dict)
                or set(item) not in ({"path", "sha256"},
                                     {"path", "sha256", "size_bytes"})):
            raise W02EvaluationError("freeze inventory identity 非法")
        relative = item["path"]
        if not isinstance(relative, str) or relative in seen:
            raise W02EvaluationError("freeze inventory path 重复或非法")
        seen.add(relative)
        path = _safe_path(root, relative)
        if not path.is_file():
            raise W02EvaluationError(f"freeze inventory 文件缺失: {relative}")
        payload = path.read_bytes()
        if ("size_bytes" in item and len(payload) != item["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != item["sha256"]):
            raise W02EvaluationError(f"freeze inventory 文件漂移: {relative}")


def _verify_freeze_inventories(
        repository_root: Path,
        artifact_root: Path,
        freeze: dict[str, Any],
        ) -> None:
    """核 candidate artifact 和冻结 host code/test。"""
    _verify_file_inventory(artifact_root, freeze["artifact_inventory"])
    _verify_file_inventory(repository_root, freeze["code_inventory"])
    _verify_file_inventory(repository_root, freeze["test_inventory"])


def _projection(backend: StorageBackend, predicate) -> tuple:
    """按 runtime 同源规则投影持久表。"""
    schema = backend.schema_snapshot()
    return tuple(
        (table, tuple(tuple(sorted(row.items()))
                      for row in backend.select(table, where=None)))
        for table in sorted(schema)
        if predicate(table, schema[table])
    )


def _host_digests(
        backend: StorageBackend,
        learning,
        cursor_path: Path,
        ) -> dict[str, str]:
    """回读 Core/Memory/Use/cursor 和不含 transaction 的 host state。"""
    learning.state_key()
    core = _projection(backend, lambda _table, meta: bool(meta["core"]))
    memory = _projection(
        backend, lambda table, _meta: table.startswith("memory_"))
    host = _projection(
        backend, lambda table, _meta: table != "ph2_w02_transaction_event")
    try:
        cursor_value = json.loads(cursor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise W02EvaluationError("formal candidate cursor 损坏") from exc
    return {
        "core": _digest(core),
        "cursor": _digest(cursor_value),
        "logical": _digest(host),
        "memory": _digest(memory),
        "use": _digest(learning.use_outcomes.state_key()),
    }


def _manifest_identities(
        reader: D03ReleaseReader,
        primary: Path,
        dependency: Path,
        private_paths: tuple[str, ...],
        ) -> dict[str, Any]:
    """从 D-03 pack binding 和公开 manifest 取得 7 条 private file identity。"""
    result: dict[str, Any] = {}
    stage = reader.stages[1]
    evaluator_keys = set(stage.data_visibility.evaluator_pack_keys)
    for binding in reader.global_manifest.pack_bindings:
        if binding.pack_key not in evaluator_keys:
            continue
        manifest_path = _overlay(
            primary, dependency, binding.manifest_identity.relative_path)
        raw = manifest_path.read_bytes()
        if (len(raw) != binding.manifest_identity.size_bytes
                or hashlib.sha256(raw).hexdigest()
                != binding.manifest_identity.sha256):
            raise W02EvaluationError("evaluator pack manifest identity 漂移")
        manifest = read_artifact_manifest(manifest_path)
        prefix = PurePosixPath(
            binding.manifest_identity.relative_path).parent
        for identity in manifest.files:
            full = PurePosixPath(prefix, identity.relative_path).as_posix()
            if full in private_paths:
                if full in result:
                    raise W02EvaluationError("private path identity 重复")
                result[full] = identity
    if set(result) != set(private_paths):
        raise W02EvaluationError("7 条 private path 未被 pack manifest 完整绑定")
    return result


def _read_private_payload(
        permit: _PrivatePermit,
        *,
        reader: D03ReleaseReader,
        primary: Path,
        dependency: Path,
        private_paths: tuple[str, ...],
        identities: dict[str, Any],
        audit: _PrivateAudit,
        ) -> _PrivatePayload:
    """只在有效 clone permit 下读取 7 条 private artifact。"""
    if not isinstance(permit, _PrivatePermit) or permit.marker is not _PERMIT_MARKER:
        raise W02EvaluationError("private payload 只能由 isolated clone owner 读取")
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    labels: list[EvaluatorLabelRecord] = []
    for relative in private_paths:
        authorized = reader.require_visible_path(
            W02_STAGE_KEY, "evaluator", relative)
        identity = identities[relative]
        local_parts = Path(identity.relative_path).parts
        artifact_root = authorized.parents[len(local_parts) - 1]
        audit.path_reads += 1
        audit.payload_bytes += authorized.stat().st_size
        try:
            records = read_record_artifact(artifact_root, identity)
        except Exception as exc:
            raise W02EvaluationError(
                f"private artifact identity/readback 失败: {relative}") from exc
        for record in records:
            if isinstance(record, SourceRefRecord):
                sources.append(record)
            elif isinstance(record, ObservationRecord):
                observations.append(record)
            elif isinstance(record, EvaluatorLabelRecord):
                labels.append(record)
            else:
                raise W02EvaluationError("private artifact 混入非 evaluator record")
    audit.source_reads = len(sources)
    audit.observation_reads = len(observations)
    audit.label_reads = len(labels)
    audit.ud_observation_reads = sum(
        item.substage == "D-02-SOURCE-UD-V1" for item in observations)
    if (audit.path_reads != _PRIVATE_PATH_COUNT
            or not observations or len(observations) != len(labels)
            or audit.ud_observation_reads <= 0
            or any(item.split != "held_out" for item in observations)
            or any(item.visible_stage != W02_STAGE_KEY
                   or item.owner_mode != "read_only" for item in labels)):
        raise W02EvaluationError("private Observation/label/UD owner 集不闭合")
    observation_keys = {item.stable_key for item in observations}
    label_keys = {item.observation_key for item in labels}
    if (len(observation_keys) != len(observations)
            or len(label_keys) != len(labels)
            or observation_keys != label_keys):
        raise W02EvaluationError("private Observation/label 不是一一闭合")
    return _PrivatePayload(
        tuple(sources), tuple(observations), tuple(labels))


def _contiguous_oov(result, raw_text: str) -> bool:
    """要求至少一个 lattice candidate 把完整未知串保留为单一 OOV part。"""
    return any(
        len(candidate.parts) == 1
        and candidate.parts[0].surface == raw_text
        and not candidate.parts[0].known_word_form
        for candidate in result.candidates
    )


def _morph_target(value: dict[str, Any]) -> W02MorphologyTarget | None:
    """只从 held-out Observation 构造唯一 target；结构不足时返回 NE 前态。"""
    units = tuple(sorted(
        value["analysis_units"], key=lambda item: (item["start"], item["end"])))
    stems = tuple(item for item in units if item["unit_kind"] == "STEM")
    if len(stems) != 1:
        return None
    components = tuple(
        item["surface"] for item in units
        if item["unit_kind"] == "COMPONENT")
    return W02MorphologyTarget(
        value["construction_key"], stems[0]["surface"], components)


def _text_actual(learning, value: dict[str, Any], mode: _Mode) -> tuple[str, tuple[str, ...]]:
    """从 raw-preserving understand 形成公开四态，不读取 label。"""
    raw_text = value["raw_observation"]["text"]
    result = learning.understand(raw_text)
    if result.raw_text != raw_text:
        return STATUS_FAIL, ()
    family = value["sample_family"]
    if family == "AMBIGUOUS":
        if not mode.multi_candidate:
            return "TRUE", ()
        return ("CONFLICT" if len(result.candidates) >= 2 else "UNKNOWN"), ()
    if family == "UNKNOWN":
        if not mode.oov:
            return "TRUE", ()
        return ("UNKNOWN" if _contiguous_oov(result, raw_text) else "TRUE"), ()
    if family == "NEGATIVE" or (
            family == "GENERATION" and value["information_loss"] == 1):
        return "FALSE", ()
    return "TRUE", ()


def _morph_actual(
        learning,
        value: dict[str, Any],
        mode: _Mode,
        ) -> tuple[str | None, tuple[str, ...]]:
    """运行 typed morphology target；expected surface 从不进入 consumer。"""
    family = value["sample_family"]
    observed = value["observed_surface"]["text"]
    if family == "AMBIGUOUS":
        if not mode.multi_candidate:
            return "TRUE", ()
        understood = learning.understand(observed)
        return ("CONFLICT" if len(understood.candidates) >= 2 else "UNKNOWN"), ()
    if family == "UNKNOWN" and not mode.oov:
        return "TRUE", (observed + "伪",)
    target = _morph_target(value)
    if target is None:
        return None, ()
    generated = learning.generate(
        target, morphology_consumer_enabled=mode.morphology)
    surfaces = generated.surfaces
    if family == "UNKNOWN":
        return ("UNKNOWN" if generated.status == GENERATION_UNKNOWN
                else "TRUE"), surfaces
    if (family == "NEGATIVE"
            or value["baseline_kind"] == "DICTIONARY_REPLAY_ONLY"):
        return ("FALSE" if generated.status == GENERATION_UNKNOWN
                else "TRUE"), surfaces
    return ("TRUE" if surfaces else "UNKNOWN"), surfaces


def _label_match(
        label: EvaluatorLabelRecord,
        actual_state: str | None,
        actual_surfaces: tuple[str, ...],
        *,
        morphology: bool,
        ) -> tuple[bool | None, dict[str, Any]]:
    """在 clone 内比较 private label；仅返回 bool 和不可逆 evidence 输入。"""
    expected = label.expected_payload.to_value()
    if actual_state is None:
        return None, {
            "actual_state": "NE",
            "actual_surface_sha256": _digest(list(actual_surfaces)),
            "expected_sha256": _digest(label.to_dict()),
            "label_key": label.stable_key.to_list(),
            "passed": 0,
        }
    surface_match = True
    if morphology and label.expected_state == "TRUE":
        accepted = expected.get("accepted_surfaces")
        surface_match = (
            isinstance(accepted, list)
            and bool(actual_surfaces)
            and all(item in set(accepted) for item in actual_surfaces)
        )
    passed = actual_state == label.expected_state and surface_match
    return passed, {
        "actual_state": actual_state,
        "actual_surface_sha256": _digest(list(actual_surfaces)),
        "expected_sha256": _digest(label.to_dict()),
        "label_key": label.stable_key.to_list(),
        "passed": int(passed),
    }


def _dimension_result(
        dimension: str,
        cases: list[tuple[bool | None, dict[str, Any]]],
        ) -> W02DimensionResult:
    """把 clone-only case 折成 1/1 public dimension。"""
    if not cases:
        cases.append((None, {"reason": "NO_CASE"}))
    passed = sum(value is True for value, _evidence in cases)
    failed = sum(value is False for value, _evidence in cases)
    ne = sum(value is None for value, _evidence in cases)
    status = STATUS_FAIL if failed else STATUS_NE if ne else STATUS_PASS
    return W02DimensionResult(
        dimension,
        status,
        passed,
        len(cases),
        failed,
        ne,
        _digest([evidence for _value, evidence in cases]),
    )


def _evaluate_mode(
        learning,
        payload: _PrivatePayload,
        mode: _Mode,
        ) -> _EvaluationPass:
    """按固定顺序执行 baseline 或一个正交 ablation。"""
    by_observation = {item.stable_key: item for item in payload.observations}
    label_by_observation = {
        item.observation_key: item for item in payload.labels}
    cases: dict[str, list[tuple[bool | None, dict[str, Any]]]] = {
        key: [] for key in W02_BEARING_DIMENSIONS}
    generation_cases: list[tuple[bool | None, dict[str, Any]]] = []
    all_evidence: list[dict[str, Any]] = []

    learned = learning.candidates()
    withdrawal_ok = mode.withdrawal and any(
        item.lifecycle == "SUPERSEDED" and not item.active for item in learned)
    cases[W02_BEARING_DIMENSIONS[0]].append((withdrawal_ok, {
        "mechanism": "SUPERSEDED_INACTIVE",
        "passed": int(withdrawal_ok),
    }))
    selected = learning.select_understanding(_UNDERSTANDING_PROBE)
    unassessed = learning.select_understanding(
        _UNDERSTANDING_PROBE, outcome_assessment_enabled=False)
    multi_host_ok = (
        mode.multi_candidate
        and selected.status == SELECTION_ADOPTED
        and unassessed.status == SELECTION_CONFLICT
    )
    cases[W02_BEARING_DIMENSIONS[1]].append((multi_host_ok, {
        "mechanism": "OUTCOME_ASSESSMENT_COMPETITION",
        "passed": int(multi_host_ok),
    }))
    generated = learning.generate(
        _NEW_CONTENT_TARGET, morphology_consumer_enabled=mode.morphology)
    morph_host_ok = mode.morphology and "纸化" in generated.surfaces
    cases[W02_BEARING_DIMENSIONS[2]].append((morph_host_ok, {
        "mechanism": "TRAIN_SIDE_NEW_CONTENT",
        "passed": int(morph_host_ok),
    }))
    understood_oov = learning.understand(_OOV_PROBE)
    oov_host_ok = (
        mode.oov
        and _contiguous_oov(understood_oov, _OOV_PROBE)
        and learning.generate(_OOV_TARGET).status == GENERATION_UNKNOWN
    )
    cases[W02_BEARING_DIMENSIONS[3]].append((oov_host_ok, {
        "mechanism": "CONTIGUOUS_OOV_FAIL_CLOSED",
        "passed": int(oov_host_ok),
    }))

    for key in sorted(by_observation):
        observation = by_observation[key]
        label = label_by_observation[key]
        value = observation.typed_payload.to_value()
        evidence: dict[str, Any]
        if observation.payload_kind == TEXT_FIDELITY_PAYLOAD_KIND:
            validate_text_fidelity_payload(observation.typed_payload)
            actual_state, surfaces = _text_actual(learning, value, mode)
            passed, evidence = _label_match(
                label, actual_state, surfaces, morphology=False)
            family = value["sample_family"]
            dimension = (
                W02_BEARING_DIMENSIONS[1] if family == "AMBIGUOUS"
                else W02_BEARING_DIMENSIONS[3] if family == "UNKNOWN"
                else W02_BEARING_DIMENSIONS[0]
            )
            cases[dimension].append((passed, evidence))
        elif observation.payload_kind == MORPHOLOGY_PAYLOAD_KIND:
            validate_morphology_payload(observation.typed_payload)
            actual_state, surfaces = _morph_actual(learning, value, mode)
            passed, evidence = _label_match(
                label, actual_state, surfaces, morphology=True)
            family = value["sample_family"]
            dimension = (
                W02_BEARING_DIMENSIONS[1] if family == "AMBIGUOUS"
                else W02_BEARING_DIMENSIONS[3] if family == "UNKNOWN"
                else W02_BEARING_DIMENSIONS[2]
            )
            cases[dimension].append((passed, evidence))
            if family != "AMBIGUOUS":
                generation_cases.append((passed, evidence))
        elif observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
            raw = value.get("raw_observation")
            expected = label.expected_payload.to_value()
            integrity = (
                isinstance(raw, dict)
                and expected.get("raw_observation_sha256")
                == value.get("raw_observation_sha256")
                and label.expected_state == "TRUE"
            )
            evidence = {
                "actual_state": "TRUE" if integrity else "FALSE",
                "expected_sha256": _digest(label.to_dict()),
                "label_key": label.stable_key.to_list(),
                "passed": int(integrity),
                "source_integrity_auxiliary": 1,
            }
        else:
            raise W02EvaluationError("private Observation payload kind 未登记")
        all_evidence.append(evidence)

    dimensions = tuple(
        _dimension_result(key, cases[key]) for key in W02_BEARING_DIMENSIONS)
    generation = _dimension_result(
        GENERATION_HARD_CONJUNCT, generation_cases)
    return _EvaluationPass(dimensions, generation, _digest(all_evidence))


def _ablation_result(
        key: str,
        baseline: _EvaluationPass,
        ablated: _EvaluationPass,
        ) -> W02AblationResult:
    """形成固定 target 的分维消融矩阵。"""
    position = W02_ABLATION_ORDER.index(key)
    target = W02_BEARING_DIMENSIONS[position]
    statuses = tuple(
        (item.dimension_key, item.status) for item in ablated.dimensions)
    return W02AblationResult(
        key,
        target,
        statuses,
        ablated.generation.status,
        _digest({
            "ablation": key,
            "ablated": ablated.evidence_sha256,
            "baseline": baseline.evidence_sha256,
        }),
    )


def run_w02_private_evaluation(
        config: W02EvaluationConfig,
        ) -> W02PrivateEvaluationReport:
    """在正式 host 的 V-06 clone 中首次/唯一读取 private held-out、labels 和 UD。"""
    if not isinstance(config, W02EvaluationConfig):
        raise TypeError("config 必须是 W02EvaluationConfig")
    repository_root = Path(config.repository_root).resolve()
    dependency_root = (
        repository_root if config.dependency_root is None
        else Path(config.dependency_root).resolve())
    freeze, artifact_root = _read_freeze(config)
    _verify_freeze_inventories(repository_root, artifact_root, freeze)
    context = open_w02_frozen_context(
        repository_root,
        config.global_manifest_path,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        dependency_root=dependency_root,
    )
    if (tuple(context.dimension_keys) != W02_BEARING_DIMENSIONS
            or tuple(context.ablation_keys) != W02_ABLATION_ORDER):
        raise W02EvaluationError("D-03 dimension/ablation 与 preregistration 漂移")
    reader = D03ReleaseReader.open(
        repository_root,
        config.global_manifest_path,
        dependency_root=dependency_root,
        require_publication=True,
    )
    private_paths = context.evaluator_private_paths
    if len(private_paths) != _PRIVATE_PATH_COUNT:
        raise W02EvaluationError("D-03 W-02 private path 数量漂移")
    identities = _manifest_identities(
        reader, repository_root, dependency_root, private_paths)

    backend = SQLiteBackend(str(artifact_root / "candidate.sqlite3"))
    try:
        host_learning = open_w02_learning_runtime(backend, mode="resume")
        cursor_path = artifact_root / "runs" / str(freeze["run_id"]) / "cursor.json"
        before = _host_digests(backend, host_learning, cursor_path)
        expected = {
            key: freeze["host_digests"][key]
            for key in ("core", "cursor", "logical", "memory", "use")
        }
        if before != expected:
            raise W02EvaluationError("formal candidate host digest 与 freeze 漂移")
        audit = _PrivateAudit()
        with isolated_backend_evaluation(
                backend, label="PH2-W02-PRIVATE-EVALUATOR-V1") as isolated:
            eval_backend, eval_teacher = isolated
            if eval_teacher is not None:
                raise W02EvaluationError("W-02 private evaluator 禁止 teacher")
            learning = open_w02_learning_runtime(eval_backend, mode="resume")
            private = _read_private_payload(
                _PrivatePermit(_PERMIT_MARKER),
                reader=reader,
                primary=repository_root,
                dependency=dependency_root,
                private_paths=private_paths,
                identities=identities,
                audit=audit,
            )
            baseline = _evaluate_mode(learning, private, _Mode())
            modes = (
                _Mode(withdrawal=False),
                _Mode(multi_candidate=False),
                _Mode(morphology=False),
                _Mode(oov=False),
            )
            ablated = tuple(
                _evaluate_mode(learning, private, mode) for mode in modes)
            ablations = tuple(
                _ablation_result(key, baseline, result)
                for key, result in zip(
                    W02_ABLATION_ORDER, ablated, strict=True))
            generation_disabled = ablated[2].generation.status
            evidence_sha256 = _digest({
                "ablations": [item.evidence_sha256 for item in ablations],
                "baseline": baseline.evidence_sha256,
                "label_reads": audit.label_reads,
                "observation_reads": audit.observation_reads,
                "private_path_reads": audit.path_reads,
                "ud_observation_reads": audit.ud_observation_reads,
            })
        after = _host_digests(backend, host_learning, cursor_path)
    finally:
        backend.close()
    _verify_freeze_inventories(repository_root, artifact_root, freeze)
    return aggregate_w02_evaluation(
        dimensions=baseline.dimensions,
        generation=baseline.generation,
        ablations=ablations,
        generation_consumer_disabled_status=generation_disabled,
        host_digests_before=before,
        host_digests_after=after,
        private_path_reads=audit.path_reads,
        private_payload_bytes=audit.payload_bytes,
        held_out_observation_reads=audit.observation_reads,
        evaluator_label_reads=audit.label_reads,
        ud_observation_reads=audit.ud_observation_reads,
        evaluator_label_writes=audit.label_writes,
        host_write_count=0,
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "GENERATION_HARD_CONJUNCT",
    "STATUS_FAIL",
    "STATUS_NE",
    "STATUS_PASS",
    "W02_ABLATION_ORDER",
    "W02_BEARING_DIMENSIONS",
    "W02_EVALUATION_ORDER",
    "W02AblationResult",
    "W02DimensionResult",
    "W02EvaluationConfig",
    "W02EvaluationError",
    "W02PrivateEvaluationReport",
    "W02UseOutcomeStore",
    "aggregate_w02_evaluation",
    "run_w02_private_evaluation",
]
