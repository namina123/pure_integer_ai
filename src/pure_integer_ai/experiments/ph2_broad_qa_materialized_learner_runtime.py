"""共享的 K 盘 materialized TRAIN deterministic learner runtime。

本模块只负责路径边界、append-only checkpoint、resume marker、输出封口与严格
回读。领域协议读取、记录派生、前缀计数和 manifest 固定零字段均由调用方注入。
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE,
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    initial_source_inference_learning_checkpoint,
    read_source_inference_learning_chain,
    source_inference_learning_prefix_sha256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


CHECKPOINT_FILE = "checkpoints.jsonl"
RESUME_MARKER_FILE = "resume-markers.jsonl"
MANIFEST_FILE = "manifest.json"
_RESERVED_ARTIFACT_FILES = {
    CHECKPOINT_FILE, MANIFEST_FILE, RESUME_MARKER_FILE}

MaterialLoader = Callable[[Path, str], dict[str, object]]
PrefixCounter = Callable[[tuple[dict[str, object], ...], object, int], tuple[int, int]]


def _sha256(payload: bytes) -> str:
    """返回文件、记录或规范结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def require_k_run_root(value: str | Path, *, label: str) -> Path:
    """要求显式 learner 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(f"{label} run root 必须是 K 盘目录")
    return root


def within_run_root(
        root: Path,
        value: str | Path,
        *,
        label: str,
        ) -> Path:
    """解析输入输出路径并拒绝逃出显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    """判断两个已解析 artifact 根是否相同或互相包含。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_output_file_roles(
        output_file_roles: object,
        *,
        label: str,
        ) -> tuple[tuple[str, str, str], ...]:
    """核验语义输出只能使用互异的单层普通文件名。"""
    if (not isinstance(output_file_roles, tuple) or not output_file_roles
            or any(type(item) is not tuple or len(item) != 3
                   for item in output_file_roles)):
        raise BroadQaExternalDataError(
            f"{label} output file roles schema 漂移")
    names = []
    for name, role, identity_key in output_file_roles:
        if (not all(isinstance(value, str) and value
                    for value in (name, role, identity_key))
                or name in _RESERVED_ARTIFACT_FILES):
            raise BroadQaExternalDataError(
                f"{label} output file role 非法")
        path = Path(name)
        if (path.is_absolute() or path.drive or path.name != name
                or name in {".", ".."}):
            raise BroadQaExternalDataError(
                f"{label} output path 非法")
        names.append(name)
    if len(set(names)) != len(names):
        raise BroadQaExternalDataError(
            f"{label} output path 重复")
    return output_file_roles


def _validate_output_material(
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        label: str,
        ) -> None:
    """要求记录 identity、规范 JSONL 与 adapter payload 逐字节一致。"""
    expected_names = {name for name, _role, _identity in output_file_roles}
    if (not isinstance(outputs, dict) or not isinstance(payloads, dict)
            or set(outputs) != expected_names
            or set(payloads) != expected_names):
        raise BroadQaExternalDataError(
            f"{label} output inventory 漂移")
    for name, _role, identity_key in output_file_roles:
        values = outputs[name]
        payload = payloads[name]
        if not isinstance(values, tuple) or not isinstance(payload, bytes):
            raise BroadQaExternalDataError(
                f"{label} output type 漂移")
        identities = []
        encoded = []
        for value in values:
            if (not isinstance(value, dict)
                    or not isinstance(value.get(identity_key), str)
                    or not value[identity_key]):
                raise BroadQaExternalDataError(
                    f"{label} {name} record identity 漂移")
            identities.append(value[identity_key])
            encoded.append(canonical_json_line(value))
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"{label} {name} record identity 重复")
        if b"".join(encoded) != payload:
            raise BroadQaExternalDataError(
                f"{label} {name} payload/record 漂移")


def _prefix_counts(
        prefix_counter: PrefixCounter,
        work: tuple[dict[str, object], ...],
        prefix_context: object,
        processed_item_count: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """调用领域前缀计数器并拒绝 bool、负数或非二元结果。"""
    counts = prefix_counter(work, prefix_context, processed_item_count)
    if (type(counts) is not tuple or len(counts) != 2
            or any(type(value) is not int or value < 0 for value in counts)):
        raise BroadQaExternalDataError(
            f"{label} prefix counter 结果非法")
    return counts


def _validate_material(
        material: dict[str, object],
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        protocol_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            dict[str, bytes],
            object,
        ]:
    """核验领域 adapter 返回的 material schema 与协议 identity。"""
    roles = _validate_output_file_roles(output_file_roles, label=(
        "materialized learner"))
    expected_fields = {
        "manifest", "outputs", "payloads", "prefix_context", "summary", "work"}
    if not isinstance(material, dict) or set(material) != expected_fields:
        raise BroadQaExternalDataError(
            "materialized learner adapter schema 漂移")
    manifest = material["manifest"]
    work = material["work"]
    outputs = material["outputs"]
    summary = material["summary"]
    payloads = material["payloads"]
    if (not isinstance(manifest, dict)
            or manifest.get("manifest_sha256") != protocol_manifest_sha256
            or not isinstance(work, tuple) or not work
            or not isinstance(outputs, dict)
            or not isinstance(summary, dict)
            or not isinstance(payloads, dict)):
        raise BroadQaExternalDataError(
            "materialized learner adapter material 漂移")
    work_ids = [item.get("work_id") if isinstance(item, dict) else None
                for item in work]
    if (any(not isinstance(value, str) or not value for value in work_ids)
            or len(set(work_ids)) != len(work_ids)):
        raise BroadQaExternalDataError(
            "materialized learner work identity 漂移")
    _validate_output_material(
        output_file_roles=roles,
        outputs=outputs,
        payloads=payloads,
        label="materialized learner",
    )
    return (
        manifest, work, outputs, summary, payloads,
        material["prefix_context"],
    )


def _output_files(
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        ) -> list[dict[str, object]]:
    """构造全部语义输出的文件承诺。"""
    return [{
        "bytes": len(payloads[name]),
        "record_count": len(outputs[name]),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payloads[name]),
    } for name, role, _identity in output_file_roles]


def _semantic_result_sha256(
        *,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> str:
    """形成独立于 run identity 和 checkpoint 切分的语义结果摘要。"""
    return _sha256(canonical_json_bytes({
        "files": files,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "summary": summary,
    }))


def _target_count(
        *,
        total: int,
        cursor: int,
        stop_after: int | None,
        label: str,
        ) -> int:
    """在任何运行写入前核验本次调用的逻辑停止点。"""
    if stop_after is None:
        return total
    if (type(stop_after) is not int
            or not cursor < stop_after < total):
        raise BroadQaExternalDataError(
            f"{label} stop_after 必须推进且早于完成")
    return stop_after


def write_or_verify(path: Path, payload: bytes, *, label: str) -> None:
    """finalize 时缺失则独占写，崩溃恢复时只接受相同既有字节。"""
    if path.exists():
        try:
            stored = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"{label} finalize 文件不可读: {path.name}") from error
        if stored != payload:
            raise BroadQaExternalDataError(
                f"{label} finalize 文件漂移: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def validate_materialized_checkpoint_chain(
        *,
        chain_path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        operator_family: str,
        work: tuple[dict[str, object], ...],
        prefix_context: object,
        prefix_counter: PrefixCounter,
        require_complete: bool,
        label: str,
        ):
    """逐 revision 重算冻结 work 前缀、输出计数和完整运行身份。"""
    if type(require_complete) is not bool:
        raise BroadQaExternalDataError(
            f"{label} checkpoint require_complete 非 bool")
    chain = read_source_inference_learning_chain(chain_path)
    item_ids = tuple(str(item["work_id"]) for item in work)
    order_sha = source_inference_learning_prefix_sha256(item_ids)
    for checkpoint in chain:
        expected_prefix = source_inference_learning_prefix_sha256(
            item_ids[:checkpoint.processed_item_count])
        expected_counts = _prefix_counts(
            prefix_counter, work, prefix_context,
            checkpoint.processed_item_count, label=label)
        if (checkpoint.run_id != run_id
                or checkpoint.protocol_manifest_sha256
                != protocol_manifest_sha256
                or checkpoint.operator_family != operator_family
                or checkpoint.training_item_count != len(item_ids)
                or checkpoint.training_item_order_sha256 != order_sha
                or checkpoint.processed_item_prefix_sha256 != expected_prefix
                or (checkpoint.evidence_candidate_count,
                    checkpoint.rule_candidate_count) != expected_counts):
            raise BroadQaExternalDataError(
                f"{label} checkpoint identity/prefix/count 漂移")
    if (require_complete
            and chain[-1].status
            != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE):
        raise BroadQaExternalDataError(
            f"{label} checkpoint 未完成完整 TRAIN")
    return chain


def _read_resume_markers(
        *,
        path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        chain,
        marker_kind: str,
        marker_format_version: int,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """严格回读 append-only resume marker 链并绑定非终态 checkpoint。"""
    if not path.exists():
        return ()
    try:
        payload = path.read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise BroadQaExternalDataError(
                f"{label} resume marker 链为空或截断")
        lines = payload.splitlines(keepends=True)
        values = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"{label} resume marker 链不可读") from error
    checkpoint_by_sha = {item.sha256(): item for item in chain}
    previous_sha = ""
    previous_cursor = -1
    expected_fields = {
        "format_version", "marker_id", "marker_ordinal",
        "previous_marker_sha256", "protocol_manifest_sha256", "record_kind",
        "resumed_from_checkpoint_sha256", "resumed_from_cursor", "run_id",
    }
    for ordinal, (value, line) in enumerate(zip(values, lines)):
        if (not isinstance(value, dict) or set(value) != expected_fields
                or canonical_json_line(value) != line
                or type(value["format_version"]) is not int
                or value["format_version"] != marker_format_version
                or value["record_kind"] != marker_kind
                or type(value["marker_ordinal"]) is not int
                or value["marker_ordinal"] != ordinal
                or value["previous_marker_sha256"] != previous_sha
                or value["run_id"] != run_id
                or value["protocol_manifest_sha256"]
                != protocol_manifest_sha256):
            raise BroadQaExternalDataError(
                f"{label} resume marker schema/chain 漂移")
        checkpoint = checkpoint_by_sha.get(
            value["resumed_from_checkpoint_sha256"])
        if (checkpoint is None
                or checkpoint.status
                == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
                or type(value["resumed_from_cursor"]) is not int
                or value["resumed_from_cursor"]
                != checkpoint.processed_item_count
                or value["resumed_from_cursor"] <= previous_cursor):
            raise BroadQaExternalDataError(
                f"{label} resume marker checkpoint/cursor 漂移")
        identity = {key: value[key] for key in (
            "marker_ordinal", "previous_marker_sha256",
            "protocol_manifest_sha256", "resumed_from_checkpoint_sha256",
            "resumed_from_cursor", "run_id")}
        if value["marker_id"] != _sha256(canonical_json_bytes(identity)):
            raise BroadQaExternalDataError(
                f"{label} resume marker identity 漂移")
        previous_sha = _sha256(canonical_json_line(value))
        previous_cursor = value["resumed_from_cursor"]
    return values


def _append_resume_marker(
        *,
        path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        chain,
        marker_kind: str,
        marker_format_version: int,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """在 resume 调用前追加当前非终态 checkpoint 的确定性恢复标记。"""
    existing = _read_resume_markers(
        path=path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        chain=chain,
        marker_kind=marker_kind,
        marker_format_version=marker_format_version,
        label=label,
    )
    checkpoint = chain[-1]
    if checkpoint.status == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE:
        raise BroadQaExternalDataError(
            f"{label} complete checkpoint 不得 resume")
    if (existing and existing[-1]["resumed_from_checkpoint_sha256"]
            == checkpoint.sha256()):
        return existing
    previous_sha = (
        "" if not existing else _sha256(canonical_json_line(existing[-1])))
    identity = {
        "marker_ordinal": len(existing),
        "previous_marker_sha256": previous_sha,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "resumed_from_checkpoint_sha256": checkpoint.sha256(),
        "resumed_from_cursor": checkpoint.processed_item_count,
        "run_id": run_id,
    }
    marker = {
        **identity,
        "format_version": marker_format_version,
        "marker_id": _sha256(canonical_json_bytes(identity)),
        "record_kind": marker_kind,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab" if path.exists() else "xb") as handle:
        handle.write(canonical_json_line(marker))
    return existing + (marker,)


def _resume_marker_commitment(
        path: Path,
        markers: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """返回可缺省物理文件的恢复边界承诺。"""
    payload = b"" if not path.exists() else path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": len(markers),
        "relative_path": RESUME_MARKER_FILE,
        "sha256": _sha256(payload),
    }


def _run_manifest(
        *,
        artifact_kind: str,
        status: str,
        format_version: int,
        run_id: str,
        protocol_manifest_sha256: str,
        checkpoint_payload: bytes,
        checkpoint_terminal_sha256: str,
        resume_commitment: dict[str, object],
        files: list[dict[str, object]],
        summary: dict[str, object],
        work_item_count: int,
        fixed_manifest_fields: dict[str, object],
        ) -> dict[str, object]:
    """构造完整、生产禁用且由调用方声明零读取边界的 manifest。"""
    base = {
        "artifact_kind": artifact_kind,
        "checkpoint_chain_bytes": len(checkpoint_payload),
        "checkpoint_chain_sha256": _sha256(checkpoint_payload),
        "checkpoint_terminal_sha256": checkpoint_terminal_sha256,
        "files": files,
        "format_version": format_version,
        "fresh_resume_output_byte_equivalence_required": 1,
        "mastery_claimed": 0,
        "ordered_work_item_count": work_item_count,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "resume_markers": resume_commitment,
        "run_id": run_id,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": _semantic_result_sha256(
            protocol_manifest_sha256=protocol_manifest_sha256,
            files=files,
            summary=summary,
        ),
        "status": status,
        "summary": summary,
    }
    if set(base).intersection(fixed_manifest_fields):
        raise BroadQaExternalDataError(
            "materialized learner fixed manifest field 冲突")
    return {**base, **fixed_manifest_fields}


def run_materialized_learner(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        run_dir: str | Path,
        run_id: str,
        mode: str,
        checkpoint_interval: int,
        stop_after: int | None,
        label: str,
        artifact_kind: str,
        status: str,
        checkpoint_open_status: str,
        resume_marker_kind: str,
        format_version: int,
        marker_format_version: int,
        operator_family: str,
        output_file_roles: tuple[tuple[str, str, str], ...],
        material_loader: MaterialLoader,
        prefix_counter: PrefixCounter,
        fixed_manifest_fields: dict[str, object],
        ) -> dict[str, object]:
    """在 K 盘 fresh/resume 完整扫描 TRAIN，并以 manifest-last 封口。"""
    root = require_k_run_root(run_root, label=label)
    protocol_root = within_run_root(root, protocol_dir, label="protocol_dir")
    target = within_run_root(root, run_dir, label="run_dir")
    if _paths_overlap(protocol_root, target):
        raise BroadQaExternalDataError(
            f"{label} protocol/run_dir 不得重叠")
    run_sha = _sha_value(run_id, label=f"{label} run_id")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label=f"{label} protocol manifest")
    if mode not in {"fresh", "resume"}:
        raise BroadQaExternalDataError(
            f"{label} mode 必须是 fresh/resume")
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise BroadQaExternalDataError(
            f"{label} checkpoint interval 非法")
    chain_path = target / CHECKPOINT_FILE
    marker_path = target / RESUME_MARKER_FILE
    manifest_path = target / MANIFEST_FILE
    if mode == "fresh" and target.exists():
        raise BroadQaExternalDataError(f"{label} fresh target 已存在")
    if mode == "resume" and (not target.is_dir() or manifest_path.exists()
                             or not chain_path.is_file()):
        raise BroadQaExternalDataError(f"{label} resume 状态非法")
    material = _validate_material(
        material_loader(protocol_root, protocol_sha),
        output_file_roles=output_file_roles,
        protocol_manifest_sha256=protocol_sha,
    )
    _manifest, work, outputs, summary, payloads, prefix_context = material
    item_ids = tuple(str(item["work_id"]) for item in work)
    output_paths = [target / name for name, _role, _identity in output_file_roles]
    total = len(item_ids)
    if mode == "fresh":
        target_count = _target_count(
            total=total, cursor=0, stop_after=stop_after, label=label)
        target.mkdir(parents=True)
        initial = initial_source_inference_learning_checkpoint(
            run_id=run_sha,
            protocol_manifest_sha256=protocol_sha,
            operator_family=operator_family,
            training_item_ids=item_ids,
        )
        append_source_inference_learning_checkpoint(chain_path, initial)
        chain = (initial,)
        markers = ()
    else:
        chain = validate_materialized_checkpoint_chain(
            chain_path=chain_path,
            run_id=run_sha,
            protocol_manifest_sha256=protocol_sha,
            operator_family=operator_family,
            work=work,
            prefix_context=prefix_context,
            prefix_counter=prefix_counter,
            require_complete=False,
            label=label,
        )
        cursor = chain[-1].processed_item_count
        target_count = _target_count(
            total=total, cursor=cursor, stop_after=stop_after, label=label)
        if chain[-1].status == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE:
            markers = _read_resume_markers(
                path=marker_path,
                run_id=run_sha,
                protocol_manifest_sha256=protocol_sha,
                chain=chain,
                marker_kind=resume_marker_kind,
                marker_format_version=marker_format_version,
                label=label,
            )
        else:
            if any(path.exists() for path in output_paths):
                raise BroadQaExternalDataError(
                    f"{label} 非终态提前出现输出")
            markers = _append_resume_marker(
                path=marker_path,
                run_id=run_sha,
                protocol_manifest_sha256=protocol_sha,
                chain=chain,
                marker_kind=resume_marker_kind,
                marker_format_version=marker_format_version,
                label=label,
            )

    cursor = chain[-1].processed_item_count
    checkpoint = chain[-1]
    while cursor < target_count:
        next_cursor = min(cursor + checkpoint_interval, target_count)
        evidence_count, result_count = _prefix_counts(
            prefix_counter, work, prefix_context, next_cursor, label=label)
        checkpoint = advance_source_inference_learning_checkpoint(
            checkpoint,
            training_item_ids=item_ids,
            processed_item_ids=item_ids[:next_cursor],
            evidence_candidate_count=evidence_count,
            rule_candidate_count=result_count,
            complete=next_cursor == total,
        )
        append_source_inference_learning_checkpoint(chain_path, checkpoint)
        cursor = next_cursor
    if cursor < total:
        return {
            "checkpoint_chain_sha256": _sha256(chain_path.read_bytes()),
            "processed_item_count": cursor,
            "resume_marker_count": len(markers),
            "run_id": run_sha,
            "status": checkpoint_open_status,
            "training_item_count": total,
        }

    terminal_counts = _prefix_counts(
        prefix_counter, work, prefix_context, total, label=label)
    if terminal_counts != (
            summary.get("evidence_count"), summary.get("result_record_count")):
        raise BroadQaExternalDataError(
            f"{label} checkpoint/output count 漂移")
    for name, _role, _identity in output_file_roles:
        write_or_verify(target / name, payloads[name], label=label)
    chain = validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        work=work,
        prefix_context=prefix_context,
        prefix_counter=prefix_counter,
        require_complete=True,
        label=label,
    )
    markers = _read_resume_markers(
        path=marker_path,
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        chain=chain,
        marker_kind=resume_marker_kind,
        marker_format_version=marker_format_version,
        label=label,
    )
    files = _output_files(
        output_file_roles=output_file_roles,
        outputs=outputs,
        payloads=payloads,
    )
    manifest = _run_manifest(
        artifact_kind=artifact_kind,
        status=status,
        format_version=format_version,
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        checkpoint_payload=chain_path.read_bytes(),
        checkpoint_terminal_sha256=chain[-1].sha256(),
        resume_commitment=_resume_marker_commitment(marker_path, markers),
        files=files,
        summary=summary,
        work_item_count=total,
        fixed_manifest_fields=fixed_manifest_fields,
    )
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_materialized_learner(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        label: str,
        artifact_kind: str,
        status: str,
        resume_marker_kind: str,
        format_version: int,
        marker_format_version: int,
        operator_family: str,
        output_file_roles: tuple[tuple[str, str, str], ...],
        material_loader: MaterialLoader,
        prefix_counter: PrefixCounter,
        fixed_manifest_fields: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """从 protocol 重派生并严格回读完整 materialized learner artifact。"""
    root = Path(run_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    if _paths_overlap(protocol_root, root):
        raise BroadQaExternalDataError(
            f"{label} protocol/run_dir 不得重叠")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label=f"{label} expected protocol manifest")
    material = _validate_material(
        material_loader(protocol_root, protocol_sha),
        output_file_roles=output_file_roles,
        protocol_manifest_sha256=protocol_sha,
    )
    _protocol_manifest, work, outputs, summary, payloads, prefix_context = material
    manifest_path = root / MANIFEST_FILE
    chain_path = root / CHECKPOINT_FILE
    marker_path = root / RESUME_MARKER_FILE
    try:
        encoded_manifest = manifest_path.read_bytes()
        manifest = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} manifest 不可读") from error
    if (not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded_manifest):
        raise BroadQaExternalDataError(f"{label} manifest 非规范")
    run_id = _sha_value(
        manifest.get("run_id"), label=f"{label} manifest run_id")
    for name, _role, _identity in output_file_roles:
        try:
            stored = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"{label} {name} 不可读") from error
        if stored != payloads[name]:
            raise BroadQaExternalDataError(
                f"{label} {name} 与 protocol 派生漂移")
    chain = validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        work=work,
        prefix_context=prefix_context,
        prefix_counter=prefix_counter,
        require_complete=True,
        label=label,
    )
    markers = _read_resume_markers(
        path=marker_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        chain=chain,
        marker_kind=resume_marker_kind,
        marker_format_version=marker_format_version,
        label=label,
    )
    expected = _run_manifest(
        artifact_kind=artifact_kind,
        status=status,
        format_version=format_version,
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        checkpoint_payload=chain_path.read_bytes(),
        checkpoint_terminal_sha256=chain[-1].sha256(),
        resume_commitment=_resume_marker_commitment(marker_path, markers),
        files=_output_files(
            output_file_roles=output_file_roles,
            outputs=outputs,
            payloads=payloads,
        ),
        summary=summary,
        work_item_count=len(work),
        fixed_manifest_fields=fixed_manifest_fields,
    )
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(f"{label} manifest 漂移")
    return (
        {**manifest, "manifest_sha256": _sha256(encoded_manifest)},
        outputs,
    )


__all__ = [
    "CHECKPOINT_FILE",
    "MANIFEST_FILE",
    "RESUME_MARKER_FILE",
    "read_materialized_learner",
    "require_k_run_root",
    "run_materialized_learner",
    "validate_materialized_checkpoint_chain",
    "within_run_root",
    "write_or_verify",
]
