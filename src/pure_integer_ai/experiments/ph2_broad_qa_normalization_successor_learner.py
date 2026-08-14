"""normalization successor 的 K 盘 deterministic learner 与恢复 reader。

learner 只读取物化 TRAIN protocol 和调用方提供的冻结 manifest SHA；它不打开
source/evaluation/reserve，不调用教师或 LLM，也不启用生产 normalization。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES,
    derive_normalization_successor_learning_outputs,
    normalization_successor_output_payloads,
    normalization_successor_prefix_output_counts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_protocol import (
    read_normalization_successor_learner_input,
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


NORMALIZATION_SUCCESSOR_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_LEARNER_V1")
NORMALIZATION_SUCCESSOR_LEARNER_STATUS = (
    "DEVELOPMENT_COMPLETE_PACK_DISABLED")
NORMALIZATION_SUCCESSOR_CHECKPOINT_OPEN = "SUCCESSOR_CHECKPOINT_OPEN"
NORMALIZATION_SUCCESSOR_RESUME_MARKER_KIND = (
    "NORMALIZATION_SUCCESSOR_RESUME_MARKER_V1")
CHECKPOINT_FILE = "checkpoints.jsonl"
RESUME_MARKER_FILE = "resume-markers.jsonl"
MANIFEST_FILE = "manifest.json"


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


def _require_k_run_root(value: str | Path) -> Path:
    """要求 learner 显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization successor run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出路径并拒绝逃出显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _protocol_material(
        *,
        protocol_dir: Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            dict[str, bytes],
        ]:
    """只从 protocol 读取 material，并纯派生唯一学习输出。"""
    values = read_normalization_successor_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest, observations, groups, contexts, work = values
    outputs, summary = derive_normalization_successor_learning_outputs(
        protocol_manifest=manifest,
        observations=observations,
        groups=groups,
        contexts=contexts,
        work=work,
    )
    payloads = normalization_successor_output_payloads(outputs)
    return (*values, outputs, summary, payloads)


def _output_files(
        *,
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        ) -> list[dict[str, object]]:
    """构造六份语义输出的文件承诺。"""
    return [{
        "bytes": len(payloads[name]),
        "record_count": len(outputs[name]),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payloads[name]),
    } for name, role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES]


def _semantic_result_sha256(
        *,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> str:
    """形成独立于 run identity 和 checkpoint 切分的语义结果摘要。"""
    identity = {
        "files": files,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "summary": summary,
    }
    return _sha256(canonical_json_bytes(identity))


def _checkpoint_counts(
        *,
        work: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """把 successor 前缀输出计数映射到通用 checkpoint 双计数。"""
    return normalization_successor_prefix_output_counts(
        work=work,
        contexts=contexts,
        processed_item_count=processed_item_count,
    )


def _target_count(
        *,
        total: int,
        cursor: int,
        stop_after: int | None,
        ) -> int:
    """在任何运行写入前核验本次调用的逻辑停止点。"""
    if stop_after is None:
        return total
    if (type(stop_after) is not int
            or not cursor < stop_after < total):
        raise BroadQaExternalDataError(
            "successor learner stop_after 必须推进且早于完成")
    return stop_after


def _write_or_verify(path: Path, payload: bytes) -> None:
    """finalize 时缺失则独占写，崩溃恢复时只接受相同既有字节。"""
    if path.exists():
        try:
            stored = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"successor learner finalize 文件不可读: {path.name}") from error
        if stored != payload:
            raise BroadQaExternalDataError(
                f"successor learner finalize 文件漂移: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def validate_normalization_successor_checkpoint_chain(
        *,
        chain_path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        work: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        require_complete: bool,
        ):
    """逐 revision 重算冻结 work 前缀、输出计数和完整运行身份。"""
    if type(require_complete) is not bool:
        raise BroadQaExternalDataError(
            "successor checkpoint require_complete 非 bool")
    chain = read_source_inference_learning_chain(chain_path)
    item_ids = tuple(str(item["work_id"]) for item in work)
    order_sha = source_inference_learning_prefix_sha256(item_ids)
    for checkpoint in chain:
        expected_prefix = source_inference_learning_prefix_sha256(
            item_ids[:checkpoint.processed_item_count])
        expected_counts = _checkpoint_counts(
            work=work,
            contexts=contexts,
            processed_item_count=checkpoint.processed_item_count,
        )
        if (checkpoint.run_id != run_id
                or checkpoint.protocol_manifest_sha256
                != protocol_manifest_sha256
                or checkpoint.operator_family
                != NORMALIZATION_CONTRASTIVE_FAMILY
                or checkpoint.training_item_count != len(item_ids)
                or checkpoint.training_item_order_sha256 != order_sha
                or checkpoint.processed_item_prefix_sha256 != expected_prefix
                or (checkpoint.evidence_candidate_count,
                    checkpoint.rule_candidate_count) != expected_counts):
            raise BroadQaExternalDataError(
                "successor checkpoint identity/prefix/count 漂移")
    if (require_complete
            and chain[-1].status
            != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE):
        raise BroadQaExternalDataError(
            "successor checkpoint 未完成完整 TRAIN")
    return chain


def _read_resume_markers(
        *,
        path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        chain,
        ) -> tuple[dict[str, object], ...]:
    """严格回读 append-only resume marker 链并绑定非终态 checkpoint。"""
    if not path.exists():
        return ()
    try:
        payload = path.read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise BroadQaExternalDataError(
                "successor resume marker 链为空或截断")
        values = tuple(json.loads(line) for line in payload.splitlines())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "successor resume marker 链不可读") from error
    checkpoint_by_sha = {item.sha256(): item for item in chain}
    previous_sha = ""
    previous_cursor = -1
    expected_fields = {
        "format_version", "marker_id", "marker_ordinal",
        "previous_marker_sha256", "protocol_manifest_sha256", "record_kind",
        "resumed_from_checkpoint_sha256", "resumed_from_cursor", "run_id",
    }
    for ordinal, value in enumerate(values):
        if (not isinstance(value, dict) or set(value) != expected_fields
                or canonical_json_line(value) != payload.splitlines(
                    keepends=True)[ordinal]
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or value["record_kind"]
                != NORMALIZATION_SUCCESSOR_RESUME_MARKER_KIND
                or type(value["marker_ordinal"]) is not int
                or value["marker_ordinal"] != ordinal
                or value["previous_marker_sha256"] != previous_sha
                or value["run_id"] != run_id
                or value["protocol_manifest_sha256"]
                != protocol_manifest_sha256):
            raise BroadQaExternalDataError(
                "successor resume marker schema/chain 漂移")
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
                "successor resume marker checkpoint/cursor 漂移")
        identity = {
            key: value[key] for key in (
                "marker_ordinal", "previous_marker_sha256",
                "protocol_manifest_sha256", "resumed_from_checkpoint_sha256",
                "resumed_from_cursor", "run_id")
        }
        if value["marker_id"] != _sha256(canonical_json_bytes(identity)):
            raise BroadQaExternalDataError(
                "successor resume marker identity 漂移")
        previous_sha = _sha256(canonical_json_line(value))
        previous_cursor = value["resumed_from_cursor"]
    return values


def _append_resume_marker(
        *,
        path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        chain,
        ) -> tuple[dict[str, object], ...]:
    """在 resume 调用前追加当前非终态 checkpoint 的确定性恢复标记。"""
    existing = _read_resume_markers(
        path=path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        chain=chain,
    )
    checkpoint = chain[-1]
    if checkpoint.status == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE:
        raise BroadQaExternalDataError(
            "successor complete checkpoint 不得 resume")
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
        "format_version": 1,
        "marker_id": _sha256(canonical_json_bytes(identity)),
        "record_kind": NORMALIZATION_SUCCESSOR_RESUME_MARKER_KIND,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if path.exists() else "xb"
    with path.open(mode) as handle:
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
        run_id: str,
        protocol_manifest_sha256: str,
        checkpoint_payload: bytes,
        checkpoint_terminal_sha256: str,
        resume_commitment: dict[str, object],
        files: list[dict[str, object]],
        summary: dict[str, object],
        work_item_count: int,
        ) -> dict[str, object]:
    """构造完整、生产禁用且不含评测消费的 learner manifest。"""
    return {
        "artifact_kind": NORMALIZATION_SUCCESSOR_LEARNER_KIND,
        "checkpoint_chain_bytes": len(checkpoint_payload),
        "checkpoint_chain_sha256": _sha256(checkpoint_payload),
        "checkpoint_terminal_sha256": checkpoint_terminal_sha256,
        "evaluation_or_reserve_read_count": 0,
        "failed_icu_v2_artifact_read_count": 0,
        "files": files,
        "format_version": 1,
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
        "source_pack_read_count": 0,
        "status": NORMALIZATION_SUCCESSOR_LEARNER_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "training_protocol_read_count": 1,
    }


def run_normalization_successor_learner(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        run_dir: str | Path,
        run_id: str,
        mode: str,
        checkpoint_interval: int = 512,
        stop_after: int | None = None,
        ) -> dict[str, object]:
    """在 K 盘 fresh/resume 完整扫描 TRAIN，并以 manifest-last 封口。"""
    root = _require_k_run_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    target = _within(root, run_dir, label="run_dir")
    run_sha = _sha_value(run_id, label="successor learner run_id")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="successor learner protocol manifest")
    if mode not in {"fresh", "resume"}:
        raise BroadQaExternalDataError(
            "successor learner mode 必须是 fresh/resume")
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise BroadQaExternalDataError(
            "successor learner checkpoint interval 非法")
    (
        manifest, _observations, _groups, contexts, work,
        outputs, summary, payloads,
    ) = _protocol_material(
        protocol_dir=protocol_root,
        expected_manifest_sha256=protocol_sha,
    )
    if manifest["manifest_sha256"] != protocol_sha:
        raise BroadQaExternalDataError(
            "successor learner protocol identity 漂移")
    item_ids = tuple(str(item["work_id"]) for item in work)
    chain_path = target / CHECKPOINT_FILE
    marker_path = target / RESUME_MARKER_FILE
    manifest_path = target / MANIFEST_FILE
    output_paths = [
        target / name for name, _role, _identity
        in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES]
    total = len(item_ids)
    if mode == "fresh":
        if target.exists():
            raise BroadQaExternalDataError(
                "successor learner fresh target 已存在")
        target_count = _target_count(
            total=total, cursor=0, stop_after=stop_after)
        target.mkdir(parents=True)
        initial = initial_source_inference_learning_checkpoint(
            run_id=run_sha,
            protocol_manifest_sha256=protocol_sha,
            operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
            training_item_ids=item_ids,
        )
        append_source_inference_learning_checkpoint(chain_path, initial)
        chain = (initial,)
        markers = ()
    else:
        if (not target.is_dir() or manifest_path.exists()
                or not chain_path.is_file()):
            raise BroadQaExternalDataError(
                "successor learner resume 状态非法")
        chain = validate_normalization_successor_checkpoint_chain(
            chain_path=chain_path,
            run_id=run_sha,
            protocol_manifest_sha256=protocol_sha,
            work=work,
            contexts=contexts,
            require_complete=False,
        )
        cursor = chain[-1].processed_item_count
        target_count = _target_count(
            total=total, cursor=cursor, stop_after=stop_after)
        if (chain[-1].status
                == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE):
            markers = _read_resume_markers(
                path=marker_path,
                run_id=run_sha,
                protocol_manifest_sha256=protocol_sha,
                chain=chain,
            )
        else:
            if any(path.exists() for path in output_paths):
                raise BroadQaExternalDataError(
                    "successor learner 非终态提前出现输出")
            markers = _append_resume_marker(
                path=marker_path,
                run_id=run_sha,
                protocol_manifest_sha256=protocol_sha,
                chain=chain,
            )

    cursor = chain[-1].processed_item_count
    checkpoint = chain[-1]
    while cursor < target_count:
        next_cursor = min(cursor + checkpoint_interval, target_count)
        evidence_count, result_count = _checkpoint_counts(
            work=work,
            contexts=contexts,
            processed_item_count=next_cursor,
        )
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
            "status": NORMALIZATION_SUCCESSOR_CHECKPOINT_OPEN,
            "training_item_count": total,
        }

    terminal_counts = _checkpoint_counts(
        work=work, contexts=contexts, processed_item_count=total)
    if terminal_counts != (
            summary["evidence_count"], summary["result_record_count"]):
        raise BroadQaExternalDataError(
            "successor learner checkpoint/output count 漂移")
    for name, _role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        _write_or_verify(target / name, payloads[name])
    chain = validate_normalization_successor_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        work=work,
        contexts=contexts,
        require_complete=True,
    )
    markers = _read_resume_markers(
        path=marker_path,
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        chain=chain,
    )
    files = _output_files(outputs=outputs, payloads=payloads)
    run_manifest = _run_manifest(
        run_id=run_sha,
        protocol_manifest_sha256=protocol_sha,
        checkpoint_payload=chain_path.read_bytes(),
        checkpoint_terminal_sha256=chain[-1].sha256(),
        resume_commitment=_resume_marker_commitment(marker_path, markers),
        files=files,
        summary=summary,
        work_item_count=total,
    )
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(run_manifest))
    return {
        **run_manifest,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def read_normalization_successor_learner(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 protocol 重派生并严格回读完整 successor learner artifact。"""
    root = Path(run_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="successor learner expected protocol manifest")
    (
        _protocol_manifest, _observations, _groups, contexts, work,
        outputs, summary, payloads,
    ) = _protocol_material(
        protocol_dir=protocol_root,
        expected_manifest_sha256=protocol_sha,
    )
    manifest_path = root / MANIFEST_FILE
    chain_path = root / CHECKPOINT_FILE
    marker_path = root / RESUME_MARKER_FILE
    try:
        encoded_manifest = manifest_path.read_bytes()
        manifest = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "successor learner manifest 不可读") from error
    if (not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded_manifest):
        raise BroadQaExternalDataError(
            "successor learner manifest 非规范")
    run_id = _sha_value(
        manifest.get("run_id"), label="successor learner manifest run_id")
    for name, _role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        try:
            stored_payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"successor learner {name} 不可读") from error
        if stored_payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"successor learner {name} 与 protocol 派生漂移")
    chain = validate_normalization_successor_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        work=work,
        contexts=contexts,
        require_complete=True,
    )
    markers = _read_resume_markers(
        path=marker_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        chain=chain,
    )
    expected = _run_manifest(
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        checkpoint_payload=chain_path.read_bytes(),
        checkpoint_terminal_sha256=chain[-1].sha256(),
        resume_commitment=_resume_marker_commitment(marker_path, markers),
        files=_output_files(outputs=outputs, payloads=payloads),
        summary=summary,
        work_item_count=len(work),
    )
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(
            "successor learner manifest 漂移")
    return (
        {**manifest, "manifest_sha256": _sha256(encoded_manifest)},
        outputs,
    )


__all__ = [
    "NORMALIZATION_SUCCESSOR_CHECKPOINT_OPEN",
    "NORMALIZATION_SUCCESSOR_LEARNER_KIND",
    "NORMALIZATION_SUCCESSOR_LEARNER_STATUS",
    "read_normalization_successor_learner",
    "run_normalization_successor_learner",
    "validate_normalization_successor_checkpoint_chain",
]
