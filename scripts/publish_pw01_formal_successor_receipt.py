"""从封存 PW-00A 数据库执行唯一 PW-01 successor 并发布公开 receipt。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any, Callable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.pw01_formal_successor import (
    assemble_successor_context,
    run_fresh_successor_evidence,
    run_restart_successor_evidence,
    stable_key_sha256,
    validate_pw00a_base_events,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from scripts.publish_pw00a_formal_start_receipt import (
    RECEIPT_PATH as PW00A_RECEIPT_PATH,
    read_formal_start_receipt,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PW01_FORMAL_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "PW01-FORMAL-SUCCESSOR-20260807-A"
STATUS = "PW01_FORMAL_SUCCESSOR_EVIDENCED"
RECEIPT_PATH = "data/ph2/manifests/pw01_formal_successor_receipt_v1.json"
RUNNER_PATH = "scripts/publish_pw01_formal_successor_receipt.py"
RUNTIME_PATH = "src/pure_integer_ai/experiments/pw01_formal_successor.py"
IMPLEMENTATION_COMMIT = "bddcf6a7fc8c06789fd2e329e2de85a4b0705e8a"
RUN_ID = 2026080702
PUBLISH_EPOCH = 2


BaseReceiptReader = Callable[..., dict[str, Any]]


def _sha256(payload: bytes) -> str:
    """返回字节的小写十六进制 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    """返回文件的字节数和 SHA-256。"""
    payload = path.read_bytes()
    return {"sha256": _sha256(payload), "size_bytes": len(payload)}


def _binding(root: Path, relative_path: str, status: str) -> dict[str, Any]:
    """形成不含绝对路径的公开文件承诺。"""
    return {
        **_identity(root / Path(*relative_path.split("/"))),
        "relative_path": relative_path,
        "status": status,
    }


def _exclusive_copy(source: Path, target: Path) -> None:
    """把封存 base 逐字复制到新文件，拒绝覆盖任何既有 successor。"""
    if target.exists():
        raise RuntimeError("PW-01 successor database 已存在，禁止覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def execute_formal_successor(
        repository_root: str | Path,
        base_database_path: str | Path,
        successor_database_path: str | Path,
        *,
        base_receipt_reader: BaseReceiptReader | None = None,
        ) -> dict[str, Any]:
    """复制 base，执行正式因果纵切、真重启，并返回公开安全结果。"""
    root = Path(repository_root).resolve()
    base_database = Path(base_database_path).resolve()
    successor_database = Path(successor_database_path).resolve()
    if base_database == successor_database:
        raise RuntimeError("PW-01 successor 不得原地修改 PW-00A base")
    reader = base_receipt_reader or read_formal_start_receipt
    base_receipt = reader(
        root,
        root / Path(*PW00A_RECEIPT_PATH.split("/")),
        database_path=base_database,
    )
    base_identity = _identity(base_database)
    if base_identity != {
            "sha256": base_receipt["runtime_database"]["sha256"],
            "size_bytes": base_receipt["runtime_database"]["size_bytes"]}:
        raise RuntimeError("PW-01 base identity 与 PW-00A receipt 漂移")
    _exclusive_copy(base_database, successor_database)
    if _identity(successor_database) != base_identity:
        raise RuntimeError("PW-01 successor 初始复制与 base 不一致")

    first_backend = SQLiteBackend(str(successor_database))
    fresh = None
    fresh_manifest_sha256 = None
    projection_records = None
    try:
        validate_pw00a_base_events(first_backend, base_receipt)
        ctx, source, projection, manifest, runtime = assemble_successor_context(
            first_backend,
            root,
            run_id=RUN_ID,
            publish_epoch=PUBLISH_EPOCH,
        )
        fresh_manifest_sha256 = stable_key_sha256(manifest.stable_key())
        projection_records = projection.record_count
        fresh = run_fresh_successor_evidence(
            ctx,
            source,
            runtime,
            root,
            run_id=RUN_ID,
            publish_epoch=PUBLISH_EPOCH,
        )
    finally:
        first_backend.close()
    before_restart_identity = _identity(successor_database)

    second_backend = SQLiteBackend(str(successor_database))
    try:
        validate_pw00a_base_events(second_backend, base_receipt)
        restart = run_restart_successor_evidence(
            second_backend,
            root,
            fresh,
            run_id=RUN_ID,
            publish_epoch=PUBLISH_EPOCH,
        )
    finally:
        second_backend.close()
    final_identity = _identity(successor_database)
    if _identity(base_database) != base_identity:
        raise RuntimeError("PW-01 successor 运行改变了封存 PW-00A base")
    if (fresh["after_answer_sha256"] != restart["restart_answer_sha256"]
            or fresh["core_state_sha256"] == ""):
        raise RuntimeError("PW-01 successor fresh/restart 语义证据未闭合")

    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "base_database": {**base_identity, "status": "PW00A_BASE_UNCHANGED"},
        "base_receipt": _binding(
            root, PW00A_RECEIPT_PATH, "PW00A_FORMAL_START_BASE"),
        "before_restart_database": {
            **before_restart_identity,
            "status": "PW01_FRESH_CLOSED",
        },
        "formal_evidence": {
            **fresh,
            **restart,
            "fresh_manifest_sha256": fresh_manifest_sha256,
            "fresh_projection_record_count": projection_records,
        },
        "format_version": FORMAT_VERSION,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "readiness_transition": {
            "PW00A_STARTED": 1,
            "PW01_COMPLETE": 0,
            "PW01_CONTROLLED_READING_EVIDENCED": 1,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "run_id": RUN_ID,
        "publish_epoch": PUBLISH_EPOCH,
        "runner": _binding(root, RUNNER_PATH, "PW01_FORMAL_SUCCESSOR_OWNER"),
        "runtime": _binding(root, RUNTIME_PATH, "PW01_FORMAL_SUCCESSOR_RUNTIME"),
        "runtime_boundaries": {
            "base_database_unchanged": 1,
            "core_unchanged": 1,
            "cross_space_use_evidenced": 1,
            "exact_source_ablation_evidenced": 1,
            "private_reads": 0,
            "restart_evidenced": 1,
            "teacher_calls": 0,
            "user_acl_evidenced": 1,
        },
        "status": STATUS,
        "successor_database": {
            **final_identity,
            "status": "GIT_EXTERNAL_SEALED",
        },
    }


def _canonical_object(payload: bytes) -> dict[str, Any]:
    """严格解析单换行 canonical JSON object。"""
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("PW-01 successor receipt newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("PW-01 successor receipt 非 canonical bytes")
    return value


def _validate(value: dict[str, Any], root: Path) -> None:
    """核验 successor 固定身份、依赖、状态位和证据字段。"""
    if set(value) != {
            "artifact_kind", "artifact_version", "base_database",
            "base_receipt", "before_restart_database", "formal_evidence",
            "format_version", "implementation_commit", "publish_epoch",
            "readiness_transition", "receipt_relative_path",
            "receipt_self_excluded", "run_id", "runner", "runtime",
            "runtime_boundaries", "status", "successor_database"}:
        raise ValueError("PW-01 successor receipt 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["implementation_commit"] != IMPLEMENTATION_COMMIT
            or value["publish_epoch"] != PUBLISH_EPOCH
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["run_id"] != RUN_ID
            or value["status"] != STATUS):
        raise ValueError("PW-01 successor 固定身份漂移")
    if value["readiness_transition"] != {
            "PW00A_STARTED": 1,
            "PW01_COMPLETE": 0,
            "PW01_CONTROLLED_READING_EVIDENCED": 1}:
        raise ValueError("PW-01 successor readiness 漂移")
    for field, path, status in (
            ("base_receipt", PW00A_RECEIPT_PATH, "PW00A_FORMAL_START_BASE"),
            ("runner", RUNNER_PATH, "PW01_FORMAL_SUCCESSOR_OWNER"),
            ("runtime", RUNTIME_PATH, "PW01_FORMAL_SUCCESSOR_RUNTIME")):
        if value[field] != _binding(root, path, status):
            raise ValueError(f"PW-01 successor {field} binding 漂移")
    if value["runtime_boundaries"] != {
            "base_database_unchanged": 1,
            "core_unchanged": 1,
            "cross_space_use_evidenced": 1,
            "exact_source_ablation_evidenced": 1,
            "private_reads": 0,
            "restart_evidenced": 1,
            "teacher_calls": 0,
            "user_acl_evidenced": 1}:
        raise ValueError("PW-01 successor runtime boundary 漂移")
    evidence = value["formal_evidence"]
    if (not isinstance(evidence, dict)
            or evidence.get("before_complete") != 0
            or evidence.get("exact_ablation_complete") != 0
            or evidence.get("other_user_visible") != 0
            or evidence.get("same_user_other_session_visible") != 1
            or evidence.get("session_interaction_leaked") != 0
            or evidence.get("restart_complete") != 1
            or evidence.get("fresh_projection_record_count") != 3
            or evidence.get("projection_record_count") != 3
            or evidence.get("after_answer_sha256")
            != evidence.get("restart_answer_sha256")):
        raise ValueError("PW-01 successor formal evidence 未闭合")
    for field in (
            "base_database", "before_restart_database", "successor_database"):
        identity = value[field]
        if (not isinstance(identity, dict)
                or set(identity) != {"sha256", "size_bytes", "status"}
                or not isinstance(identity["sha256"], str)
                or len(identity["sha256"]) != 64
                or type(identity["size_bytes"]) is not int
                or identity["size_bytes"] <= 0):
            raise ValueError(f"PW-01 successor {field} identity 非法")


def read_formal_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *,
        base_database_path: str | Path | None = None,
        successor_database_path: str | Path | None = None,
        ) -> dict[str, Any]:
    """严格回读公开 receipt，并可核验两个 Git 外数据库原件。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    value = _canonical_object(target.read_bytes())
    _validate(value, root)
    for supplied, field in (
            (base_database_path, "base_database"),
            (successor_database_path, "successor_database")):
        if supplied is not None and _identity(Path(supplied).resolve()) != {
                "sha256": value[field]["sha256"],
                "size_bytes": value[field]["size_bytes"]}:
            raise ValueError(f"PW-01 successor {field} 原件漂移")
    return value


def run_and_publish(
        repository_root: str | Path,
        base_database_path: str | Path,
        successor_database_path: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """排他执行正式 successor，并以 receipt 原子形成唯一公开可见点。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("PW-01 successor receipt 已存在，禁止重跑")
    value = execute_formal_successor(
        root, base_database_path, successor_database_path)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("PW-01 successor receipt 已存在，禁止覆盖") from error
    restored = read_formal_successor_receipt(
        root,
        destination,
        base_database_path=base_database_path,
        successor_database_path=successor_database_path,
    )
    if restored != value:
        raise ValueError("PW-01 successor receipt 回读漂移")
    return restored


def _main() -> int:
    """解析仓库、base 和新 successor 数据库后执行一次。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--base-database", required=True)
    parser.add_argument("--successor-database", required=True)
    args = parser.parse_args()
    value = run_and_publish(
        args.repository_root,
        args.base_database,
        args.successor_database,
    )
    print(value["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
