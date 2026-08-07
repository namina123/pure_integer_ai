"""执行唯一 PW-00A fresh/restart 正式装载并发布公开安全 receipt。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.formal_post_weaning import (
    FormalPostWeaningLoadRequest,
    PW00A_START_PUBLISHED,
    PW00A_START_RESUMED,
)
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.cognition.shared.types import WEANING_POST
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _PostWeaningParser,
    _post_weaning_source,
    _restore_runtime,
    prepare_facility_context,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.pw00a_authority import (
    RECEIPT_PATH as AUTHORITY_PATH,
    read_pw00a_formal_load_authority,
)
from pure_integer_ai.experiments.pw00a_formal_runtime import PW00AFormalRuntime
from pure_integer_ai.experiments.pw00a_formal_transaction import (
    PW00A_EVENT_PREPARED,
    PW00A_EVENT_PUBLISHED,
    PW00AFormalEventStore,
)
from pure_integer_ai.experiments.pw00a_inference_artifact import (
    ARTIFACT_PATH as INFERENCE_PATH,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PW00A_FORMAL_START_RECEIPT"
ARTIFACT_VERSION = "PW00A-FORMAL-START-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/pw00a_formal_start_receipt_v1.json"
RUNNER_PATH = "scripts/publish_pw00a_formal_start_receipt.py"
STATUS = "PW00A_FORMAL_RUNTIME_STARTED"
RUN_ID = 2026080701
PUBLISH_EPOCH = 1


def _sha256(payload: bytes) -> str:
    """返回字节的十六进制 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    """返回一个文件的字节数和 SHA-256。"""
    payload = path.read_bytes()
    return {"sha256": _sha256(payload), "size_bytes": len(payload)}


def _binding(root: Path, relative_path: str, status: str) -> dict[str, Any]:
    """形成不含绝对路径的公开文件承诺。"""
    return {
        **_identity(root / Path(*relative_path.split("/"))),
        "relative_path": relative_path,
        "status": status,
    }


def _request(root: Path, manifest: Any) -> FormalPostWeaningLoadRequest:
    """从生产设施 owner 和已发布依赖形成唯一正式装载请求。"""
    authority = _identity(root / AUTHORITY_PATH)
    inference = _identity(root / INFERENCE_PATH)
    return FormalPostWeaningLoadRequest(
        RUN_ID,
        PUBLISH_EPOCH,
        manifest.runtime_owner,
        tuple(bytes.fromhex(authority["sha256"])),
        tuple(bytes.fromhex(inference["sha256"])),
        manifest.routes,
        manifest.probe,
        manifest.budget,
        (20260807, 1, 1),
    )


def _event_summary(store: PW00AFormalEventStore) -> list[dict[str, Any]]:
    """提取不含 payload 内容的两条正式事件承诺。"""
    events = store.events(RUN_ID)
    if tuple(item.event_kind for item in events) != (
            PW00A_EVENT_PREPARED, PW00A_EVENT_PUBLISHED):
        raise RuntimeError("PW00A formal event 序列不完整")
    return [
        {
            "event_kind": item.event_kind,
            "event_seq": item.event_seq,
            "manifest_sha256": item.manifest_sha256,
            "payload_sha256": item.payload_sha256,
            "publish_epoch": item.publish_epoch,
            "run_id": item.run_id,
        }
        for item in events
    ]


def execute_formal_start(
        repository_root: str | Path,
        database_path: str | Path,
        ) -> dict[str, Any]:
    """在新 SQLite 上执行 fresh、一次阅读和真重启 resume。"""
    root = Path(repository_root).resolve()
    database = Path(database_path).resolve()
    if database.exists():
        raise RuntimeError("PW00A formal database 已存在，禁止重跑")
    authority = read_pw00a_formal_load_authority(root)

    first_backend = SQLiteBackend(str(database))
    request = None
    projection_key = None
    fresh_key = None
    intake_key = None
    try:
        ctx = make_train_context(first_backend, companion=True)
        prepare_facility_context(ctx)
        request = _request(root, ctx.f01_manifest)
        runtime = PW00AFormalRuntime.start(
            ctx,
            request,
            repository_root=root,
        )
        if (runtime.startup_report.status != PW00A_START_PUBLISHED
                or ctx.weaning_phase != WEANING_POST):
            raise RuntimeError("PW00A fresh 没有进入正式状态")
        source = _post_weaning_source(801)
        intake = runtime.run_intake(PostWeaningIntakeRequest(
            ctx.f01_routes.reading,
            source,
            "PW00A 正式阅读边界来源",
            "CC0-1.0",
            2026080701,
            parser=_PostWeaningParser(source, 81),
            trace=(20260807, 1, 2),
        ))
        if not intake.report.core_unchanged or not intake.report.query_closed:
            raise RuntimeError("PW00A formal intake 边界未闭合")
        fresh_key = runtime.startup_report.stable_key()
        intake_key = intake.report.stable_key()
        projection_key = ctx.f01_projection.stable_key()
    finally:
        first_backend.close()

    second_backend = SQLiteBackend(str(database))
    try:
        ctx, _, dry_runtime = _restore_runtime(second_backend, projection_key)
        resumed = PW00AFormalRuntime.start(
            ctx,
            request,
            repository_root=root,
        )
        if (resumed.startup_report.status != PW00A_START_RESUMED
                or ctx.weaning_phase != WEANING_POST
                or second_backend.owner_write_protection_state()
                != (ctx.core_space.space_id,)):
            raise RuntimeError("PW00A restart 没有恢复正式保护状态")
        resume_key = resumed.startup_report.stable_key()
        events = _event_summary(PW00AFormalEventStore(second_backend))
        if dry_runtime.reports():
            raise RuntimeError("PW00A restart 意外执行 dry-run 操作")
    finally:
        second_backend.close()

    database_identity = _identity(database)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "authority": _binding(root, AUTHORITY_PATH, "FORMAL_LOAD_AUTHORITY"),
        "formal_events": events,
        "format_version": FORMAT_VERSION,
        "fresh_startup_key": list(fresh_key),
        "implementation_commit": authority["head_commit"],
        "inference_artifact": _binding(
            root, INFERENCE_PATH, "LOADABLE_INFERENCE_STATE"),
        "intake_report_key": list(intake_key),
        "readiness_transition": {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS": 1,
            "PW00A_STARTED": 1,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "resume_startup_key": list(resume_key),
        "runner": _binding(root, RUNNER_PATH, "FORMAL_START_OWNER"),
        "runtime_boundaries": {
            "authority_runtime_git_calls": 0,
            "candidate_root_reads": 0,
            "core_owner_write_protected": 1,
            "core_unchanged_after_intake": 1,
            "evaluator_label_reads": 0,
            "formal_event_count": 2,
            "inference_rule_count": 299,
            "private_root_reads": 0,
            "restart_resume_evidenced": 1,
            "teacher_api_calls": 0,
        },
        "runtime_database": {
            **database_identity,
            "status": "GIT_EXTERNAL_SEALED",
        },
        "status": STATUS,
    }


def _canonical_object(payload: bytes) -> dict[str, Any]:
    """严格解析单换行 canonical JSON object。"""
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("PW00A start receipt newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("PW00A start receipt 非 canonical bytes")
    return value


def _validate(value: dict[str, Any], root: Path) -> None:
    """核验正式启动状态、公开依赖、事件序列和运行边界。"""
    if set(value) != {
            "artifact_kind", "artifact_version", "authority",
            "formal_events", "format_version", "fresh_startup_key",
            "implementation_commit", "inference_artifact",
            "intake_report_key", "readiness_transition",
            "receipt_relative_path", "receipt_self_excluded",
            "resume_startup_key", "runner", "runtime_boundaries",
            "runtime_database", "status"}:
        raise ValueError("PW00A start receipt 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["status"] != STATUS):
        raise ValueError("PW00A start receipt 固定身份漂移")
    if value["readiness_transition"] != {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS": 1,
            "PW00A_STARTED": 1}:
        raise ValueError("PW00A start receipt readiness 漂移")
    authority = read_pw00a_formal_load_authority(root)
    if value["implementation_commit"] != authority["head_commit"]:
        raise ValueError("PW00A start implementation commit 漂移")
    for field, path, status in (
            ("authority", AUTHORITY_PATH, "FORMAL_LOAD_AUTHORITY"),
            ("inference_artifact", INFERENCE_PATH, "LOADABLE_INFERENCE_STATE"),
            ("runner", RUNNER_PATH, "FORMAL_START_OWNER")):
        if value[field] != _binding(root, path, status):
            raise ValueError(f"PW00A start {field} binding 漂移")
    events = value["formal_events"]
    if (not isinstance(events, list) or len(events) != 2
            or tuple(item.get("event_kind") for item in events) != (
                PW00A_EVENT_PREPARED, PW00A_EVENT_PUBLISHED)
            or any(item.get("run_id") != RUN_ID
                   or item.get("publish_epoch") != PUBLISH_EPOCH
                   for item in events)):
        raise ValueError("PW00A start formal event 漂移")
    for field in (
            "fresh_startup_key", "intake_report_key", "resume_startup_key"):
        key = value[field]
        if (not isinstance(key, list) or not key
                or any(type(item) is not int for item in key)):
            raise ValueError(f"PW00A start {field} 非整数键")
    if value["runtime_boundaries"] != {
            "authority_runtime_git_calls": 0,
            "candidate_root_reads": 0,
            "core_owner_write_protected": 1,
            "core_unchanged_after_intake": 1,
            "evaluator_label_reads": 0,
            "formal_event_count": 2,
            "inference_rule_count": 299,
            "private_root_reads": 0,
            "restart_resume_evidenced": 1,
            "teacher_api_calls": 0}:
        raise ValueError("PW00A start runtime boundary 漂移")
    database = value["runtime_database"]
    if (not isinstance(database, dict)
            or set(database) != {"sha256", "size_bytes", "status"}
            or database["status"] != "GIT_EXTERNAL_SEALED"
            or type(database["size_bytes"]) is not int
            or database["size_bytes"] <= 0
            or not isinstance(database["sha256"], str)
            or len(database["sha256"]) != 64):
        raise ValueError("PW00A start database identity 非法")


def read_formal_start_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *,
        database_path: str | Path | None = None,
        ) -> dict[str, Any]:
    """严格回读公开 receipt，并可核验 Git 外 SQLite 原件。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    value = _canonical_object(target.read_bytes())
    _validate(value, root)
    if (database_path is not None
            and _identity(Path(database_path).resolve()) != {
                "sha256": value["runtime_database"]["sha256"],
                "size_bytes": value["runtime_database"]["size_bytes"]}):
        raise ValueError("PW00A start Git 外 database 漂移")
    return value


def run_and_publish(
        repository_root: str | Path,
        database_path: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """排他执行唯一正式运行并发布 receipt。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("PW00A formal start receipt 已存在，禁止重跑")
    value = execute_formal_start(root, database_path)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("PW00A formal start receipt 已存在，禁止覆盖") from error
    restored = read_formal_start_receipt(
        root, destination, database_path=database_path)
    if restored != value:
        raise ValueError("PW00A formal start receipt 回读漂移")
    return restored


def _main() -> int:
    """解析显式仓库和 Git 外数据库位置后执行一次。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()
    value = run_and_publish(args.repository_root, args.database)
    print(value["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
