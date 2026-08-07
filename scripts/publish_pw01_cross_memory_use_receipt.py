"""发布 PW-01 跨 Memory Use 维护索引的正式 successor 证据。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any

from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_OBSERVATION,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _ACCESS,
    _close_outer_lifecycle,
    _refresh_projection,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
    post_weaning_component_state_key,
)
from pure_integer_ai.experiments.pw01_formal_successor import (
    assemble_successor_context,
    stable_key_sha256,
)
from pure_integer_ai.experiments.pw01_controlled_reading import (
    build_pw01_question_dialogue,
    pw01_source,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_MEMORY_OBJECT
from pure_integer_ai.storage.backend import SQLiteBackend

try:
    from scripts.publish_pw01_formal_successor_receipt import (
        RECEIPT_PATH as BASE_RECEIPT_PATH,
        read_formal_successor_receipt,
    )
except ModuleNotFoundError:
    from publish_pw01_formal_successor_receipt import (
        RECEIPT_PATH as BASE_RECEIPT_PATH,
        read_formal_successor_receipt,
    )


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PW01_CROSS_MEMORY_USE_RECEIPT"
ARTIFACT_VERSION = "PW01-CROSS-MEMORY-USE-20260807-A"
STATUS = "PW01_CROSS_MEMORY_USE_MAINTENANCE_EVIDENCED"
RECEIPT_PATH = "data/ph2/manifests/pw01_cross_memory_use_receipt_v1.json"
RUNNER_PATH = "scripts/publish_pw01_cross_memory_use_receipt.py"
MECHANISM_COMMIT = "feee2676ffaa732a253ff933901938d3daf168ca"
RUN_ID = 2026080703
PUBLISH_EPOCH = 3
SOURCE_PATHS = (
    "src/pure_integer_ai/storage/cross_memory_use.py",
    "src/pure_integer_ai/experiments/cross_memory_use_runtime.py",
    "src/pure_integer_ai/experiments/memory_use_runtime.py",
    "src/pure_integer_ai/experiments/memory_maintenance_runtime.py",
    "src/pure_integer_ai/experiments/post_weaning_runtime.py",
)


def _sha256(payload: bytes) -> str:
    """返回字节的小写 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    """返回文件字节数和 SHA-256。"""
    payload = path.read_bytes()
    return {"sha256": _sha256(payload), "size_bytes": len(payload)}


def _binding(root: Path, relative_path: str, status: str) -> dict[str, Any]:
    """形成不含绝对路径的固定文件承诺。"""
    return {
        **_identity(root / Path(*relative_path.split("/"))),
        "relative_path": relative_path,
        "status": status,
    }


def _exclusive_copy(source: Path, target: Path) -> None:
    """排他复制封存 base，禁止原地或覆盖维护 successor。"""
    if source == target:
        raise RuntimeError("cross Memory Use successor 不得原地修改 base")
    if target.exists():
        raise RuntimeError("cross Memory Use successor database 已存在")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _evidence(ctx: Any) -> dict[str, Any]:
    """读取唯一桥行并验证 ACL、原 Use 审计和 read timeline 分离。"""
    records = ctx.cross_memory_use_runtime.repository.all_records()
    if len(records) != 1:
        raise RuntimeError("cross Memory Use 正式 base 没有唯一桥事实")
    record = records[0]
    target_key = ctx.scoped_identity_store.registry.read_key(
        IDENTITY_MEMORY_OBJECT, record.target_object_hash)
    target_ref = MemoryObjectRef.from_stable_key(target_key)
    same_user_other_session = MemoryAccessContext(1, 2, 4)
    other_user = MemoryAccessContext(1, 9, 4)
    if ctx.cross_memory_use_runtime.uses_for(
            target_ref, access=_ACCESS) != records:
        raise RuntimeError("cross Memory Use 当前 session 查询漂移")
    if ctx.cross_memory_use_runtime.uses_for(
            target_ref, access=same_user_other_session) != records:
        raise RuntimeError("cross Memory Use 目标级事实没有按 user 保持")
    if ctx.cross_memory_use_runtime.uses_for(
            target_ref, access=other_user):
        raise RuntimeError("cross Memory Use 目标级事实泄漏到其他用户")
    if ctx.cross_memory_use_runtime.audit_use(
            record, access=_ACCESS) is None:
        raise RuntimeError("cross Memory Use 原 session 无法审计 Use")
    if ctx.cross_memory_use_runtime.audit_use(
            record, access=same_user_other_session) is not None:
        raise RuntimeError("cross Memory Use payload 泄漏到其他 session")
    aggregate = ctx.memory_read_aggregates.read(target_ref, access=_ACCESS)
    if aggregate is None or aggregate.last_used_seq != 0:
        raise RuntimeError("cross Memory Use 混入 read aggregate timeline")
    if ctx.cross_memory_use_runtime.recover(access=_ACCESS) != 0:
        raise RuntimeError("cross Memory Use 二次 recover 非幂等")
    return {
        "bridge_record_count": 1,
        "bridge_record_sha256": stable_key_sha256(record.stable_key()),
        "component_state_sha256": stable_key_sha256(
            post_weaning_component_state_key(ctx)),
        "core_state_sha256": bytes(CoreCanonicalStateReader(ctx).read()).hex(),
        "interaction_timeline_seq": record.source_timeline_seq,
        "other_session_payload_visible": 0,
        "other_user_fact_visible": 0,
        "read_last_used_seq": aggregate.last_used_seq,
        "same_user_other_session_fact_visible": 1,
        "target_ref_sha256": stable_key_sha256(target_ref.stable_key()),
    }


def _commit_learned_question(ctx: Any, source: Any, runtime: Any) -> str:
    """重放已学会 held-out，要求完成并把 Use+桥行提交到 successor。"""
    observations = ctx.memory_interact_events.query(access=_ACCESS)
    observation = next(
        item for item in observations
        if item.event.object_ref.object_kind == MEMORY_OBJECT_OBSERVATION)
    _close_outer_lifecycle(ctx)
    _refresh_projection(ctx)
    fixture, dialogue = build_pw01_question_dialogue(ctx, source, observation)
    try:
        operation = runtime.run_question(dialogue, fixture.request)
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)
    if (not operation.result.question.complete
            or {item.trace.source for item in operation.result.sources}
            != {pw01_source(parser_version=1)}):
        raise RuntimeError("cross Memory Use successor 无法复答已学会目标")
    return stable_key_sha256(operation.result.question.stable_key())


def execute_cross_memory_use_successor(
        repository_root: str | Path,
        base_database_path: str | Path,
        successor_database_path: str | Path,
        ) -> dict[str, Any]:
    """从正式 PW-01 base 补桥、真重启并返回公开安全证据。"""
    root = Path(repository_root).resolve()
    base_database = Path(base_database_path).resolve()
    successor_database = Path(successor_database_path).resolve()
    base_receipt = read_formal_successor_receipt(
        root,
        base_database_path=None,
        successor_database_path=base_database,
    )
    base_identity = _identity(base_database)
    if base_identity != {
            "sha256": base_receipt["successor_database"]["sha256"],
            "size_bytes": base_receipt["successor_database"]["size_bytes"]}:
        raise RuntimeError("cross Memory Use base 与正式 receipt 漂移")
    _exclusive_copy(base_database, successor_database)
    if _identity(successor_database) != base_identity:
        raise RuntimeError("cross Memory Use 初始复制与 base 不一致")

    first_backend = SQLiteBackend(str(successor_database))
    try:
        first, source, _, _, runtime = assemble_successor_context(
            first_backend,
            root,
            run_id=RUN_ID,
            publish_epoch=PUBLISH_EPOCH,
        )
        answer_sha256 = _commit_learned_question(first, source, runtime)
        fresh = _evidence(first)
        fresh["committed_answer_sha256"] = answer_sha256
    finally:
        first_backend.close()
    before_restart_identity = _identity(successor_database)

    second_backend = SQLiteBackend(str(successor_database))
    try:
        resumed, _, _, _, _ = assemble_successor_context(
            second_backend,
            root,
            run_id=RUN_ID,
            publish_epoch=PUBLISH_EPOCH,
        )
        restart = _evidence(resumed)
        restart["committed_answer_sha256"] = fresh["committed_answer_sha256"]
    finally:
        second_backend.close()
    fresh_component = fresh.pop("component_state_sha256")
    restart_component = restart.pop("component_state_sha256")
    if fresh != restart:
        raise RuntimeError("cross Memory Use restart 证据漂移")
    fresh["fresh_component_state_sha256"] = fresh_component
    fresh["restart_component_state_sha256"] = restart_component
    final_identity = _identity(successor_database)
    if _identity(base_database) != base_identity:
        raise RuntimeError("cross Memory Use successor 改变了正式 base")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "base_database": {**base_identity, "status": "PW01_BASE_UNCHANGED"},
        "base_receipt": _binding(
            root, BASE_RECEIPT_PATH, "PW01_FORMAL_SUCCESSOR_BASE"),
        "before_restart_database": {
            **before_restart_identity,
            "status": "CROSS_MEMORY_USE_FRESH_CLOSED",
        },
        "evidence": fresh,
        "format_version": FORMAT_VERSION,
        "mechanism_commit": MECHANISM_COMMIT,
        "publish_epoch": PUBLISH_EPOCH,
        "readiness_transition": {
            "PW01_COMPLETE": 0,
            "PW01_CONTROLLED_READING_EVIDENCED": 1,
            "PW01_CROSS_MEMORY_USE_MAINTENANCE_EVIDENCED": 1,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "run_id": RUN_ID,
        "runner": _binding(root, RUNNER_PATH, "CROSS_MEMORY_USE_OWNER"),
        "source_bindings": [
            _binding(root, path, "CROSS_MEMORY_USE_MECHANISM")
            for path in SOURCE_PATHS
        ],
        "status": STATUS,
        "successor_database": {
            **final_identity,
            "status": "GIT_EXTERNAL_SEALED",
        },
    }


def _canonical_object(payload: bytes) -> dict[str, Any]:
    """严格解析单换行 canonical JSON object。"""
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("cross Memory Use receipt newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("cross Memory Use receipt 非 canonical bytes")
    return value


def _validate(value: dict[str, Any], root: Path) -> None:
    """核验固定身份、全部 source leaf、状态位和维护证据。"""
    if set(value) != {
            "artifact_kind", "artifact_version", "base_database",
            "base_receipt", "before_restart_database", "evidence",
            "format_version", "mechanism_commit", "publish_epoch",
            "readiness_transition", "receipt_relative_path",
            "receipt_self_excluded", "run_id", "runner", "source_bindings",
            "status", "successor_database"}:
        raise ValueError("cross Memory Use receipt 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["mechanism_commit"] != MECHANISM_COMMIT
            or value["publish_epoch"] != PUBLISH_EPOCH
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["run_id"] != RUN_ID
            or value["status"] != STATUS):
        raise ValueError("cross Memory Use 固定身份漂移")
    if value["readiness_transition"] != {
            "PW01_COMPLETE": 0,
            "PW01_CONTROLLED_READING_EVIDENCED": 1,
            "PW01_CROSS_MEMORY_USE_MAINTENANCE_EVIDENCED": 1}:
        raise ValueError("cross Memory Use readiness 漂移")
    if value["base_receipt"] != _binding(
            root, BASE_RECEIPT_PATH, "PW01_FORMAL_SUCCESSOR_BASE"):
        raise ValueError("cross Memory Use base receipt 漂移")
    if value["runner"] != _binding(
            root, RUNNER_PATH, "CROSS_MEMORY_USE_OWNER"):
        raise ValueError("cross Memory Use runner 漂移")
    if value["source_bindings"] != [
            _binding(root, path, "CROSS_MEMORY_USE_MECHANISM")
            for path in SOURCE_PATHS]:
        raise ValueError("cross Memory Use source binding 漂移")
    evidence = value["evidence"]
    if (not isinstance(evidence, dict)
            or evidence.get("bridge_record_count") != 1
            or evidence.get("other_session_payload_visible") != 0
            or evidence.get("other_user_fact_visible") != 0
            or evidence.get("read_last_used_seq") != 0
            or evidence.get("same_user_other_session_fact_visible") != 1
            or type(evidence.get("interaction_timeline_seq")) is not int
            or evidence["interaction_timeline_seq"] <= 0):
        raise ValueError("cross Memory Use evidence 未闭合")
    for field in (
            "base_database", "before_restart_database", "successor_database"):
        identity = value[field]
        if (not isinstance(identity, dict)
                or set(identity) != {"sha256", "size_bytes", "status"}
                or not isinstance(identity["sha256"], str)
                or len(identity["sha256"]) != 64
                or type(identity["size_bytes"]) is not int
                or identity["size_bytes"] <= 0):
            raise ValueError(f"cross Memory Use {field} identity 非法")


def read_cross_memory_use_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *,
        base_database_path: str | Path | None = None,
        successor_database_path: str | Path | None = None,
        ) -> dict[str, Any]:
    """严格回读 receipt，并可同时核验两个 Git 外数据库。"""
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
            raise ValueError(f"cross Memory Use {field} 原件漂移")
    return value


def run_and_publish(
        repository_root: str | Path,
        base_database_path: str | Path,
        successor_database_path: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """排他运行维护 successor，并以 receipt 形成唯一公开可见点。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("cross Memory Use receipt 已存在，禁止重跑")
    value = execute_cross_memory_use_successor(
        root, base_database_path, successor_database_path)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("cross Memory Use receipt 已存在，禁止覆盖") from error
    restored = read_cross_memory_use_receipt(
        root,
        destination,
        base_database_path=base_database_path,
        successor_database_path=successor_database_path,
    )
    if restored != value:
        raise ValueError("cross Memory Use receipt 回读漂移")
    return restored


def _main() -> int:
    """解析仓库、正式 base 与新 successor 数据库后执行一次。"""
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
