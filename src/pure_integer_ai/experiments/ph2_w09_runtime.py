"""W09-08 public transaction freeze、真实 bounded evidence 与恢复 runtime。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_FAILURE_POINT_KEYS,
    W09_RESOURCE_BUDGET,
    W09_ZERO_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09_OWNER_KEY,
    W09FrozenContract,
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_cumulative import (
    open_w09_cumulative_runtime,
)
from pure_integer_ai.experiments.ph2_w09_dimensional import (
    W09_RETENTION_SCOPE_KEY,
    open_w09_dimensional_runtime,
)
from pure_integer_ai.experiments.ph2_w09_faults import W09InjectedFault, hit_w09_fault
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_resource import (
    W09ResourceStopController,
    W09ResourceUsage,
)
from pure_integer_ai.experiments.ph2_w09_transaction import (
    W09TransactionStore,
)
from pure_integer_ai.experiments.ph2_w09_types import (
    W09ConsumerChoice,
    W09ConsumerRequest,
    W09DirectionalResult,
    W09ResultState,
    W09UseOutcome,
    W09VerifierResult,
)
from pure_integer_ai.experiments.ph2_w09_v06_contract import (
    W09V06HostSnapshot,
    W09V06LearningRecord,
    W09V06Probe,
    W09V06ProbeCriterion,
    W09V06ProbeOwner,
    W09V06Protocol,
    W09V06WindowPlan,
    W09_V06_ABLATION_KEYS,
    w09_v06_commitment,
    w09_v06_key,
)
from pure_integer_ai.experiments.ph2_w09_v06_runtime import (
    W09V06HostCandidate,
    W09V06Runtime,
    record_w09_v06_continual_cells,
)
from pure_integer_ai.experiments.ph2_w09_weaning import (
    W09DevCalibrationOwner,
    W09FrozenTeacherEvidenceSource,
    W09ShadowErrorAudit,
    W09TypedWeaningRuntime,
    W09WeaningError,
    make_w09_typed_weaning_protocol_from_contract,
    w09_commitment,
)
from pure_integer_ai.storage.backend import SQLiteBackend

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import SCOPE_KEYS
from pure_integer_ai.experiments.ph2_w09_runtime_contract import (
    W09LogicalShard,
    W09ResourceNormalization,
    W09RunOutcome,
    W09RuntimeComponentReceipt,
    W09RuntimeConfig,
    W09RuntimeError,
    W09RuntimeEvidence,
    W09_RUNTIME_DUMP_NAME,
    W09_RUNTIME_ABLATION_KEYS,
    W09_RUNTIME_J_LC_STATUS,
    W09_RUNTIME_OWNED_TABLES,
    W09_RUNTIME_STATUS,
)


class _Stage4Report:
    """给 typed weaning 提供稳定、已完成且不携带评测标签的前置报告。"""

    complete = True
    outcomes = ("W09-08-shadow-error",)

    def stable_key(self) -> tuple[int, ...]:
        """返回前置报告的固定整数身份。"""
        return (9, 8, 0)


class _AblationBackend:
    """teacher-zero 消融使用的隔离、不可写 host 快照。"""

    def snapshot(self) -> tuple[int]:
        """返回固定 host 状态。"""
        return (0,)


class _AblationTeacher:
    """只用于验证 live teacher call 会被 zero-call gate 拒绝。"""

    def __init__(self) -> None:
        self.call_count = 0


class _AblationContext:
    """把可观察 teacher 计数与稳定 host 快照组合成隔离上下文。"""

    def __init__(self) -> None:
        self.backend = _AblationBackend()
        self.teacher = _AblationTeacher()


def _teacher_zero_ablation(
    context: W09FrozenContract,
    payload: Any,
    typed_protocol: Any,
    stage4: _Stage4Report,
    window_identity: Any,
) -> tuple[int, ...]:
    """真实触发一次 teacher call，确认连续零调用窗口 fail closed。"""
    runtime = W09TypedWeaningRuntime(
        typed_protocol,
        training_material_source=W09FrozenTeacherEvidenceSource(context, payload),
        dev_calibrator=W09DevCalibrationOwner(
            context,
            (("W09-08-ablation", context.dev_pack_keys[0]),),
        ),
        shadow_auditor=W09ShadowErrorAudit(),
        frozen_contract=context,
    )
    current = _AblationContext()
    runtime.run(current, stage4)

    def live_teacher_call() -> None:
        """向隔离 teacher 计数器追加一次调用。"""
        current.teacher.call_count += 1

    try:
        runtime.execute_zero_call_window(current, window_identity, live_teacher_call)
    except W09WeaningError:
        pass
    else:
        raise W09RuntimeError("W-09 teacher-zero ablation 未拒绝 live call")
    if current.teacher.call_count != 1:
        raise W09RuntimeError("W-09 teacher-zero ablation 未观察到 live call")
    return digest_value({
        "call_count": current.teacher.call_count,
        "rejected": 1,
        "window": window_identity.stable_key(),
    })


def _key(value: object) -> tuple[int, ...]:
    """为公开 bounded fixture 生成域分离的稳定身份。"""
    return w09_v06_key(value)


def _host() -> W09V06HostCandidate:
    """构造覆盖 Core/Memory/Use/Evidence/assessment/cursor/report 的 host。"""
    snapshot = W09V06HostSnapshot(
        _key(("W09-08", "host", "core")),
        _key(("W09-08", "host", "memory")),
        _key(("W09-08", "host", "use")),
        _key(("W09-08", "host", "evidence")),
        _key(("W09-08", "host", "assessment")),
        _key(("W09-08", "host", "cursor")),
        _key(("W09-08", "host", "report")),
        41,
        0,
        0,
        0,
    )
    return W09V06HostCandidate(w09_v06_commitment(("W09-08", "candidate")), snapshot)


def _window(ordinal: int, carriers: tuple[str, ...]) -> W09V06WindowPlan:
    """为一个窗口生成九个 data-only 学习项、错误项和未学习 probe。"""
    probes: list[W09V06Probe] = []
    good_records: list[W09V06LearningRecord] = []
    error_records: list[W09V06LearningRecord] = []
    for carrier in carriers:
        for consumer in W09_CONSUMER_KEYS:
            cell = (ordinal, carrier, consumer)
            family_key = _key(("W09-08", "family", *cell))
            good_candidate = _key(("W09-08", "candidate", "good", *cell))
            bad_candidate = _key(("W09-08", "candidate", "bad", *cell))
            criterion = W09V06ProbeCriterion(_key(("W09-08", "criterion", *cell)), (good_candidate,))
            probes.append(W09V06Probe(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("W09-08", "probe", *cell)),
                _key(("W09-08", "probe-content", *cell)),
                _key(("W09-08", "candidate-baseline", *cell)),
                criterion,
            ))
            good_records.append(W09V06LearningRecord(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("W09-08", "learning-content", "good", *cell)),
                _key(("W09-08", "source", "good", *cell)),
                _key(("W09-08", "evidence", "good", *cell)),
                good_candidate,
                10,
            ))
            error_records.append(W09V06LearningRecord(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("W09-08", "learning-content", "bad", *cell)),
                _key(("W09-08", "source", "bad", *cell)),
                _key(("W09-08", "evidence", "bad", *cell)),
                bad_candidate,
                20,
            ))
    owner = W09V06ProbeOwner(_key(("W09-08", "probe-owner", ordinal)), tuple(probes))
    return W09V06WindowPlan(
        ordinal,
        _key(("W09-08", "threshold", ordinal)),
        5,
        tuple(good_records),
        tuple(error_records),
        owner,
    )


def _protocol(host: W09V06HostCandidate) -> W09V06Protocol:
    """把九 carrier 按 3+3+3 分配到连续、不交叠的三个窗口。"""
    return W09V06Protocol(
        host.candidate_identity,
        host.snapshot().core_state_key,
        (
            _window(1, W09_CARRIER_KEYS[0:3]),
            _window(2, W09_CARRIER_KEYS[3:6]),
            _window(3, W09_CARRIER_KEYS[6:9]),
        ),
    )


def _directional(label: object, consumer: str) -> W09DirectionalResult:
    """为 retention cell 创建不与 V-06 学习 identity 重合的三向结果。"""
    request_key = _key(("W09-08", "retention", label, "request"))
    choice_key = _key(("W09-08", "retention", label, "choice"))
    candidate_key = _key(("W09-08", "retention", label, "candidate"))
    use_key = _key(("W09-08", "retention", label, "use"))
    outcome_key = _key(("W09-08", "retention", label, "outcome"))
    return W09DirectionalResult(
        W09ConsumerRequest(consumer, request_key, w09_v06_commitment(("W09-08", "retention", label, "input"))),
        W09ConsumerChoice(consumer, request_key, choice_key, candidate_key),
        W09UseOutcome(consumer, request_key, choice_key, candidate_key, use_key, outcome_key, "RESOLVED"),
        W09VerifierResult(consumer, request_key, use_key, outcome_key, _key(("W09-08", "retention", label, "verifier")), W09ResultState.PASS, "NONE"),
    )


def _compile_evidence(
    repository: Path,
    context: W09FrozenContract,
    worker_count: int,
    fault_point: str | None,
) -> W09RuntimeEvidence:
    """现场执行 W09 public bounded 组件并编译 metadata-only 证据。"""
    request = make_w09_request(context, worker_count=worker_count)
    firewall = W09PayloadFirewall.open(repository, context, request)
    payload = firewall.read_training_payload()
    cumulative = open_w09_cumulative_runtime(repository, context)
    cumulative.ingest_training_payload(payload)
    for carrier in W09_CARRIER_KEYS:
        for consumer in W09_CONSUMER_KEYS:
            cumulative.consume_directional(
                carrier,
                consumer,
                _directional(("cumulative", carrier, consumer), consumer),
            )
    dimensional = open_w09_dimensional_runtime(repository, cumulative)
    for scope in SCOPE_KEYS:
        if scope == W09_RETENTION_SCOPE_KEY:
            continue
        for carrier in W09_CARRIER_KEYS:
            for consumer in W09_CONSUMER_KEYS:
                dimensional.record_cell(
                    scope,
                    carrier,
                    consumer,
                    _directional((scope, carrier, consumer), consumer),
                )
    host = _host()
    protocol = _protocol(host)
    source = W09FrozenTeacherEvidenceSource(context, payload)
    dev = W09DevCalibrationOwner(context, (("W09-08", context.dev_pack_keys[0]),))
    shadow = W09ShadowErrorAudit()
    typed_protocol = make_w09_typed_weaning_protocol_from_contract(
        context,
        candidate_identity=host.candidate_identity,
        input_commitment=w09_commitment(payload.training_evidence),
        threshold_key=_key(("W09-08", "typed-threshold")),
        window_input_commitments=protocol.window_input_commitments,
    )
    typed = W09TypedWeaningRuntime(
        typed_protocol,
        training_material_source=source,
        dev_calibrator=dev,
        shadow_auditor=shadow,
        frozen_contract=context,
    )
    stage4 = _Stage4Report()
    typed.run(host, stage4)
    v06_runtime = W09V06Runtime(host, protocol)
    v06_report = v06_runtime.run(typed_weaning_runtime=typed)
    typed_report = typed.run(host, stage4)
    if not typed_report.complete:
        raise W09RuntimeError("W-09 typed zero-call windows 未完成")
    record_w09_v06_continual_cells(dimensional, v06_report)
    dimensional_report = dimensional.report()
    dimension_ablation = dimensional.ablate_aggregator(dimensional_report)
    if dimension_ablation.target_status != "FAIL":
        raise W09RuntimeError("W-09 dimensional ablation 未击穿")
    v06_ablations = tuple(v06_runtime.ablate(item) for item in W09_V06_ABLATION_KEYS)
    if any(item.target_status != "FAIL" for item in v06_ablations):
        raise W09RuntimeError("W-09 V06 ablation 未击穿")
    controller = W09ResourceStopController(context)
    window_counts = [dict(item.resource_audit.used) for item in v06_report.windows]
    normalized: dict[str, int] = {}
    for key in sorted(W09_RESOURCE_BUDGET):
        values = [item.get(key, 0) for item in window_counts]
        normalized[key] = max(values) if key == "max_workers" else sum(values)
    normalized["max_records"] += len(payload.source_refs) + len(payload.observations) + len(payload.training_evidence)
    normalized["max_payload_bytes"] = firewall.audit.payload_bytes
    normalized["max_payload_gets"] = firewall.audit.payload_gets
    usage = W09ResourceUsage(tuple(sorted(normalized.items())))
    worker_run = controller.run_workers(
        worker_count,
        usage,
        tuple((key, 0) for key in sorted(W09_RESOURCE_BUDGET)),
    )
    if worker_run.stop.decision.stop_state != "RESOLVED":
        raise W09RuntimeError("W-09 resource stop 未 RESOLVED")
    resource_ablation = controller.ablate_controller()
    if resource_ablation.target_status != "FAIL":
        raise W09RuntimeError("W-09 resource ablation 未击穿")
    counts = tuple(sorted(normalized.items()))
    resource = W09ResourceNormalization(counts, worker_run.canonical_result_key)
    rollback_ablation = next(
        item for item in v06_ablations if item.component_key == "ROLLBACK_CONSUMER"
    )
    teacher_ablation_result = _teacher_zero_ablation(
        context,
        payload,
        typed_protocol,
        stage4,
        v06_report.windows[0].window_identity,
    )
    learning_events = tuple(sorted(item.stable_key() for item in v06_report.cells))
    hit_w09_fault("BEFORE_FIRST_SHARD", fault_point)
    shards = _logical_shards(learning_events, fault_point=fault_point)
    component = lambda name, result: W09RuntimeComponentReceipt(
        name,
        W09_OWNER_KEY,
        digest_value({"component": name, "result": list(result)}),
        digest_value({"component": name, "receipt": list(result), "owner": W09_OWNER_KEY}),
        W09_RUNTIME_STATUS,
    )
    rollback_result = digest_value({
        "rollback": [item.rollback_audit_sha256 for item in v06_report.windows],
    })
    dimension_results = (
        dimensional_report.stable_key(),
        resource.stable_key(),
        rollback_result,
        typed_report.stable_key(),
        v06_report.stable_key(),
    )
    dimensions = tuple(
        component(name, result)
        for name, result in zip(W09_DIMENSION_KEYS, dimension_results)
    )
    ablation_results = (
        digest_value(vars(dimension_ablation)),
        digest_value(vars(resource_ablation)),
        digest_value(vars(rollback_ablation)),
        teacher_ablation_result,
        digest_value({
            "ablations": [
                vars(item)
                for item in v06_ablations
                if item.component_key != "ROLLBACK_CONSUMER"
            ],
        }),
    )
    ablations = tuple(
        component(name, result)
        for name, result in zip(W09_RUNTIME_ABLATION_KEYS, ablation_results)
    )
    windows = tuple(
        component(f"WINDOW-{item.window_ordinal}", item.stable_key())
        for item in v06_report.windows
    )
    j_lc = component("J-LC-W09", dimensional_report.stable_key())
    clone = component("V-06-CLONE", v06_report.stable_key())
    rollback = component("ROLLBACK-AUDIT", rollback_result)
    return W09RuntimeEvidence(
        tuple(host.snapshot().stable_key()),
        dimensions,
        ablations,
        windows,
        j_lc,
        clone,
        rollback,
        resource,
        shards,
        learning_events,
        firewall.audit.payload_gets,
        firewall.audit.payload_bytes,
        0,
        0,
        0,
        0,
    )


def _logical_shards(
    event_keys: tuple[tuple[int, ...], ...],
    *,
    fault_point: str | None,
) -> tuple[W09LogicalShard, ...]:
    """把真实学习事件按固定 hash 分到十六个逻辑 shard。"""
    buckets: list[list[tuple[int, ...]]] = [[] for _ in range(16)]
    halfway = max(1, len(event_keys) // 2)
    for ordinal, key in enumerate(event_keys, start=1):
        index = digest_value({"event": list(key), "shard_count": 16})[0] % 16
        buckets[index].append(key)
        if ordinal == halfway:
            hit_w09_fault("AFTER_PARTIAL_SHARD", fault_point)
    return tuple(
        W09LogicalShard(
            index,
            tuple(sorted(values)),
            digest_value({"events": [list(item) for item in sorted(values)], "shard": index}),
        )
        for index, values in enumerate(buckets)
    )


def _run_directory(config: W09RuntimeConfig) -> Path:
    """返回隔离的 W09 run directory。"""
    return Path(config.run_root).resolve() / f"w09_run_{config.run_id:020d}"


def _manifest_path(config: W09RuntimeConfig) -> Path:
    """返回 W09 metadata dump 路径。"""
    return _run_directory(config) / W09_RUNTIME_DUMP_NAME


def _validate_config(config: W09RuntimeConfig) -> None:
    """在打开任何 payload 前校验 run、worker、故障点和隔离路径。"""
    if not isinstance(config, W09RuntimeConfig):
        raise TypeError("config 必须是 W09RuntimeConfig")
    if type(config.run_id) is not int or config.run_id <= 0 or config.run_id == config.parent_run_id:
        raise W09RuntimeError("W-09 run_id 必须是新的正整数")
    if config.parent_run_id != 9 or config.base_run_id != 9:
        raise W09RuntimeError("W-09 parent/base 必须接 W-08 run 9")
    if config.worker_count not in (1, 2, 4):
        raise W09RuntimeError("W-09 worker_count 必须是 1/2/4")
    if config.mode not in {"fresh", "restart", "resume"}:
        raise W09RuntimeError("W-09 mode 必须是 fresh/restart/resume")
    if config.fault_point is not None and config.fault_point not in W09_FAILURE_POINT_KEYS:
        raise W09RuntimeError("W-09 fault point 未注册")
    repository = Path(config.repository_root).resolve()
    run_root = Path(config.run_root).resolve()
    sqlite_path = Path(config.sqlite_path).resolve()
    if run_root == repository:
        raise W09RuntimeError("W-09 run root 不得是公开 Git 根")
    try:
        sqlite_path.relative_to(run_root)
    except ValueError as error:
        raise W09RuntimeError("W-09 coordinator 必须位于 run root 内") from error


def _dump_payload(
    request: Any,
    context: W09FrozenContract,
    tx: W09TransactionStore,
    commit_payload: dict[str, Any],
) -> dict[str, Any]:
    """生成不含 run/worker/mode 的规范 public dump。"""
    events = tx.events()
    if len(events) not in {4, 5}:
        raise W09RuntimeError("W-09 manifest 前事务必须停在 cursor 或 published")
    evidence = W09RuntimeEvidence.from_dict(commit_payload["evidence"])
    if tuple(commit_payload.get("evidence_semantic_key", ())) != evidence.semantic_key():
        raise W09RuntimeError("W-09 commit evidence semantic identity 漂移")
    return {
        "artifact_kind": "PH2_W09_PUBLIC_RUNTIME_DUMP",
        "base_fence_key": list(request.base_fence_key),
        "commit": commit_payload,
        "execution_state": dict(W09_ZERO_EXECUTION_STATE),
        "format_version": 1,
        "j_lc_w09_state": W09_RUNTIME_J_LC_STATUS,
        "owner_key": context.owner_key,
        "owned_tables": list(W09_RUNTIME_OWNED_TABLES),
        "status": W09_RUNTIME_STATUS,
        "transaction": [item.payload for item in events[:4]],
        "transaction_event_count": 5,
    }


def _write_dump(config: W09RuntimeConfig, payload: dict[str, Any]) -> str:
    """排他写入或逐字复用 metadata dump，拒绝 identity 漂移。"""
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        if target.read_bytes() != encoded:
            raise W09RuntimeError("W-09 dump manifest 已存在但 identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _parse_dump(config: W09RuntimeConfig) -> tuple[dict[str, Any], str]:
    """只读解析 canonical dump 并核验公共状态。"""
    path = _manifest_path(config)
    if not path.is_file():
        raise W09RuntimeError("W-09 dump manifest 缺失")
    try:
        value = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    except DatasetContractError as error:
        raise W09RuntimeError("W-09 dump identity/state 损坏") from error
    if not isinstance(value, dict):
        raise W09RuntimeError("W-09 dump 根对象非法")
    if (
        value.get("artifact_kind") != "PH2_W09_PUBLIC_RUNTIME_DUMP"
        or value.get("format_version") != 1
        or value.get("base_fence_key") is None
        or value.get("execution_state") != dict(W09_ZERO_EXECUTION_STATE)
        or value.get("owner_key") != W09_OWNER_KEY
        or tuple(value.get("owned_tables", ())) != W09_RUNTIME_OWNED_TABLES
        or value.get("status") != W09_RUNTIME_STATUS
        or value.get("j_lc_w09_state") != W09_RUNTIME_J_LC_STATUS
        or value.get("transaction_event_count") != 5
    ):
        raise W09RuntimeError("W-09 dump identity/state 漂移")
    commit = value.get("commit")
    if not isinstance(commit, dict):
        raise W09RuntimeError("W-09 dump commit 缺失")
    evidence = W09RuntimeEvidence.from_dict(commit.get("evidence"))
    if commit.get("evidence_semantic_key") != list(evidence.semantic_key()):
        raise W09RuntimeError("W-09 dump evidence digest 漂移")
    if canonical_json_bytes(value) != path.read_bytes():
        raise W09RuntimeError("W-09 dump 不是 canonical JSON")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome_from_dump(
    config: W09RuntimeConfig,
    request: Any,
    *,
    payload_gets_this_call: int,
    payload_bytes_this_call: int,
    dump_readback: bool,
) -> W09RunOutcome:
    """从 metadata dump 构造公开 outcome，禁止二次 train payload 读取。"""
    payload, dump_sha = _parse_dump(config)
    commit = payload["commit"]
    evidence = W09RuntimeEvidence.from_dict(commit["evidence"])
    return W09RunOutcome(
        evidence.semantic_key(),
        digest_value(payload["transaction"] + [
            {"dump_manifest_sha256": dump_sha},
        ]),
        request.scheduling_key(),
        dump_sha,
        evidence,
        5,
        tuple(sorted(W09_ZERO_EXECUTION_STATE.items())),
        0,
        0,
        0,
        payload_gets_this_call,
        payload_bytes_this_call,
        dump_readback,
    )


def _finish_committed(
    config: W09RuntimeConfig,
    request: Any,
    context: W09FrozenContract,
    tx: W09TransactionStore,
    *,
    payload_gets_this_call: int = 0,
    payload_bytes_this_call: int = 0,
) -> W09RunOutcome:
    """从 commit 继续幂等追加 cursor、manifest 和 published。"""
    events = tx.events()
    if len(events) < 3:
        raise W09RuntimeError("W-09 commit 尚未形成")
    commit_payload = events[2].payload
    evidence = W09RuntimeEvidence.from_dict(commit_payload["evidence"])
    if len(events) == 3:
        tx.cursor({
            "completed_shards": list(range(16)),
            "cursor_version": "PH2-D03-CURSOR-V1",
            "learning_event_count": len(evidence.learning_event_keys),
        })
    dump = _dump_payload(request, context, tx, commit_payload)
    dump_sha = _write_dump(config, dump)
    if len(tx.events()) == 4:
        tx.published({"dump_manifest_sha256": dump_sha})
    hit_w09_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
    return _outcome_from_dump(
        config,
        request,
        payload_gets_this_call=payload_gets_this_call,
        payload_bytes_this_call=payload_bytes_this_call,
        dump_readback=False,
    )


def run_w09_public_transaction(config: W09RuntimeConfig) -> W09RunOutcome:
    """执行 W09-08 public bounded transaction，不消费 formal/private guard。"""
    _validate_config(config)
    repository = Path(config.repository_root).resolve()
    _run_directory(config).mkdir(parents=True, exist_ok=True)
    sqlite_path = Path(config.sqlite_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    coordinator = SQLiteBackend(str(sqlite_path))
    try:
        context = open_w09_frozen_contract(repository)
        request = make_w09_request(context, worker_count=config.worker_count, mode=config.mode)
        tx = W09TransactionStore(
            coordinator,
            run_id=config.run_id,
            owner_key=W09_OWNER_KEY,
            execution_identity_key=request.execution_identity_key(),
        )
        events = tx.events()
        if config.mode == "fresh" and events:
            raise W09RuntimeError("fresh mode 要求不存在既有 W-09 transaction")
        if len(events) >= 3:
            return _finish_committed(config, request, context, tx)
        evidence = _compile_evidence(
            repository,
            context,
            config.worker_count,
            config.fault_point,
        )
        tx.begin({
            "base_fence_key": list(request.base_fence_key),
            "owner_key": context.owner_key,
            "request_key": list(request.execution_identity_key()),
        })
        hit_w09_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        tx.preview({
            "logical_shard_count": 16,
            "merge_barrier_key": "PH2-D03-STABLE-MERGE-BARRIER-V1",
            "shards": [item.to_dict() for item in evidence.logical_shards],
        })
        hit_w09_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        tx.commit({
            "evidence": evidence.to_dict(),
            "evidence_semantic_key": list(evidence.semantic_key()),
            "resource_normalization": evidence.resource_normalization.to_dict(),
        })
        hit_w09_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        return _finish_committed(
            config,
            request,
            context,
            tx,
            payload_gets_this_call=evidence.payload_gets,
            payload_bytes_this_call=evidence.payload_bytes,
        )
    finally:
        coordinator.close()


def load_w09_public_dump(config: W09RuntimeConfig) -> W09RunOutcome:
    """只读回读 W09 metadata dump，payload/学习计数保持本调用为零。"""
    _validate_config(config)
    context = open_w09_frozen_contract(Path(config.repository_root).resolve())
    request = make_w09_request(context, worker_count=config.worker_count, mode=config.mode)
    return _outcome_from_dump(
        config,
        request,
        payload_gets_this_call=0,
        payload_bytes_this_call=0,
        dump_readback=True,
    )


__all__ = [
    "W09InjectedFault",
    "W09_RUNTIME_DUMP_NAME",
    "load_w09_public_dump",
    "run_w09_public_transaction",
]
