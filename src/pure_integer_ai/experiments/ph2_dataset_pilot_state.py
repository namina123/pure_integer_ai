"""D-02F pilot 的 Dict/SQLite 共用持久 cursor、结果和 clone 审计表。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


PILOT_STATE_TABLE = "ph2_dataset_pilot_state"
PILOT_RESULT_TABLE = "ph2_dataset_pilot_result"
PILOT_CLONE_AUDIT_TABLE = "ph2_dataset_pilot_clone_audit"
PILOT_TABLES = frozenset({
    PILOT_STATE_TABLE,
    PILOT_RESULT_TABLE,
    PILOT_CLONE_AUDIT_TABLE,
})
RUN_ID = 1


class DatasetPilotStateError(RuntimeError):
    """pilot cursor、backend 恢复状态或规范结果发生漂移。"""


@dataclass(frozen=True)
class PilotStoredState:
    """从 backend 恢复的单 run 状态。"""

    contract_sha256: str
    release_binding_sha256: str
    completed_count: int
    complete: bool
    report_json: str


def register_pilot_tables(backend: StorageBackend) -> None:
    """幂等注册三个非 Core、非训练的 pilot 扩展表。"""
    register_extension_table(
        backend,
        PILOT_STATE_TABLE,
        [
            ("run_id", TYPE_INT),
            ("contract_sha256", TYPE_TEXT),
            ("release_binding_sha256", TYPE_TEXT),
            ("completed_count", TYPE_INT),
            ("complete", TYPE_INT),
            ("report_json", TYPE_TEXT),
        ],
        indexes=[("run_id",)],
        recovery_key=("run_id",),
    )
    register_extension_table(
        backend,
        PILOT_RESULT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("pack_id", TYPE_INT),
            ("payload_json", TYPE_TEXT),
        ],
        indexes=[("run_id", "pack_id")],
        recovery_key=("run_id", "pack_id"),
    )
    register_extension_table(
        backend,
        PILOT_CLONE_AUDIT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("audit_kind", TYPE_TEXT),
            ("record_count", TYPE_INT),
        ],
        indexes=[("run_id", "audit_kind")],
        recovery_key=("run_id", "audit_kind"),
    )


def _sha256(value: Any, *, where: str) -> str:
    """校验 state 中的 SHA-256 字符串。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise DatasetPilotStateError(f"{where} 必须是小写 SHA-256")
    return value


def _decode_payload(text: Any, *, where: str) -> dict[str, Any]:
    """恢复 backend TEXT 中保存的规范 JSON object。"""
    if not isinstance(text, str) or not text:
        raise DatasetPilotStateError(f"{where} 不能为空")
    try:
        payload = text.encode("utf-8")
        value = parse_canonical_json_bytes(payload, require_object=True)
    except Exception as error:
        raise DatasetPilotStateError(f"{where} 不是规范 JSON") from error
    assert isinstance(value, dict)
    return value


def load_pilot_state(backend: StorageBackend) -> PilotStoredState | None:
    """读取唯一 pilot state；没有行时返回 None。"""
    rows = backend.select(PILOT_STATE_TABLE, {"run_id": RUN_ID})
    if not rows:
        return None
    if len(rows) != 1:
        raise DatasetPilotStateError("pilot state run_id 重复")
    row = rows[0]
    completed = row.get("completed_count")
    complete = row.get("complete")
    report_json = row.get("report_json")
    if type(completed) is not int or completed < 0:
        raise DatasetPilotStateError("pilot completed_count 非法")
    if complete not in {0, 1}:
        raise DatasetPilotStateError("pilot complete 非 0/1")
    if not isinstance(report_json, str):
        raise DatasetPilotStateError("pilot report_json 类型错误")
    if complete and not report_json:
        raise DatasetPilotStateError("已完成 pilot 缺少 report_json")
    if not complete and report_json:
        raise DatasetPilotStateError("未完成 pilot 不得提前保存 report_json")
    return PilotStoredState(
        _sha256(row.get("contract_sha256"), where="contract_sha256"),
        _sha256(
            row.get("release_binding_sha256"),
            where="release_binding_sha256",
        ),
        completed,
        bool(complete),
        report_json,
    )


def initialize_pilot_state(
        backend: StorageBackend,
        *,
        contract_sha256: str,
        release_binding_sha256: str,
        ) -> PilotStoredState:
    """初始化或核对不可漂移的 run contract/release 绑定。"""
    contract = _sha256(contract_sha256, where="contract_sha256")
    release = _sha256(
        release_binding_sha256, where="release_binding_sha256")
    state = load_pilot_state(backend)
    if state is None:
        backend.insert(PILOT_STATE_TABLE, {
            "run_id": RUN_ID,
            "contract_sha256": contract,
            "release_binding_sha256": release,
            "completed_count": 0,
            "complete": 0,
            "report_json": "",
        })
        backend.commit()
        state = load_pilot_state(backend)
        assert state is not None
        return state
    if (state.contract_sha256 != contract
            or state.release_binding_sha256 != release):
        raise DatasetPilotStateError(
            "pilot resume contract 或 release_root 漂移")
    return state


def load_pack_results(
        backend: StorageBackend) -> dict[int, dict[str, Any]]:
    """按 pack_id 恢复已提交的规范结果并拒绝重复。"""
    rows = backend.select(
        PILOT_RESULT_TABLE,
        {"run_id": RUN_ID},
        order_by="pack_id",
    )
    results: dict[int, dict[str, Any]] = {}
    for row in rows:
        pack_id = row.get("pack_id")
        if type(pack_id) is not int or pack_id <= 0:
            raise DatasetPilotStateError("pilot result pack_id 非法")
        if pack_id in results:
            raise DatasetPilotStateError("pilot result pack_id 重复")
        results[pack_id] = _decode_payload(
            row.get("payload_json"), where=f"pack[{pack_id}]")
    state = load_pilot_state(backend)
    if state is None or state.completed_count != len(results):
        raise DatasetPilotStateError("pilot cursor 与已提交 pack 数不一致")
    return results


def store_pack_result(
        backend: StorageBackend,
        pack_id: int,
        payload: dict[str, Any],
        ) -> None:
    """幂等拒重复地提交一个 pack 结果并推进 cursor。"""
    if type(pack_id) is not int or pack_id <= 0:
        raise DatasetPilotStateError("store pack_id 非法")
    if not isinstance(payload, dict):
        raise DatasetPilotStateError("store payload 必须是 dict")
    if backend.count(
            PILOT_RESULT_TABLE,
            {"run_id": RUN_ID, "pack_id": pack_id}):
        raise DatasetPilotStateError("pilot pack result 禁止覆盖")
    payload_json = canonical_json_bytes(payload).decode("utf-8")
    backend.insert(PILOT_RESULT_TABLE, {
        "run_id": RUN_ID,
        "pack_id": pack_id,
        "payload_json": payload_json,
    })
    state = load_pilot_state(backend)
    if state is None or state.complete:
        raise DatasetPilotStateError("pilot state 不可推进")
    updated = backend.update(
        PILOT_STATE_TABLE,
        {"run_id": RUN_ID},
        {"completed_count": state.completed_count + 1},
    )
    if updated != 1:
        raise DatasetPilotStateError("pilot cursor 推进失败")
    backend.commit()


def finish_pilot_state(
        backend: StorageBackend,
        report: dict[str, Any],
        *,
        expected_pack_count: int,
        ) -> None:
    """只有全部 pack 结果已提交时才发布规范 report。"""
    if type(expected_pack_count) is not int or expected_pack_count <= 0:
        raise DatasetPilotStateError("expected_pack_count 非法")
    state = load_pilot_state(backend)
    if state is None:
        raise DatasetPilotStateError("pilot state 缺失")
    if state.complete:
        stored = _decode_payload(state.report_json, where="stored report")
        if stored != report:
            raise DatasetPilotStateError("已完成 pilot report 漂移")
        return
    if state.completed_count != expected_pack_count:
        raise DatasetPilotStateError("pilot pack 尚未全部提交")
    report_json = canonical_json_bytes(report).decode("utf-8")
    updated = backend.update(
        PILOT_STATE_TABLE,
        {"run_id": RUN_ID},
        {"complete": 1, "report_json": report_json},
    )
    if updated != 1:
        raise DatasetPilotStateError("pilot report 发布失败")
    backend.commit()


def load_finished_report(backend: StorageBackend) -> dict[str, Any] | None:
    """恢复已完成 report；未完成时返回 None。"""
    state = load_pilot_state(backend)
    if state is None or not state.complete:
        return None
    return _decode_payload(state.report_json, where="finished report")


__all__ = [
    "DatasetPilotStateError",
    "PILOT_CLONE_AUDIT_TABLE",
    "PILOT_RESULT_TABLE",
    "PILOT_STATE_TABLE",
    "PILOT_TABLES",
    "PilotStoredState",
    "finish_pilot_state",
    "initialize_pilot_state",
    "load_finished_report",
    "load_pack_results",
    "load_pilot_state",
    "register_pilot_tables",
    "store_pack_result",
]
