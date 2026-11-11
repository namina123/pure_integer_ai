"""关系 case 的宿主硬墙钟上限：单次被动 event wait，不轮询训练状态。"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Iterator

from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json


@contextmanager
def passive_relation_case_budget(directory: Path, *, max_wall_seconds: int) -> Iterator[None]:
    """超时保存当前进程端点并退出 124；未提交事务只留原库供独立恢复。"""
    directory = directory.resolve()
    if directory.drive.upper() != "K:" or directory.parent == Path("K:/"):
        raise ValueError("case budget requires a named explicit K directory")
    if type(max_wall_seconds) is not int or max_wall_seconds <= 0:
        raise ValueError("case wall budget must be positive strict integer seconds")
    directory.mkdir(exist_ok=False)
    started = current_process_resource_endpoint().to_value()
    write_preflight_json(directory, "started.json", {
        "max_wall_seconds": max_wall_seconds, "resource_endpoint": started,
        "wait_mode": "ONE_PASSIVE_EVENT_WAIT", "process_poll_count": 0,
        "restart_authorized": 0, "free_dialogue_complete": 0,
    })
    completed = Event()
    completion_lock = Lock()

    def wait_once() -> None:
        """只阻塞一次；宿主超时是外部停止事件，不给核心增加语义时钟。"""
        if completed.wait(max_wall_seconds):
            return
        with completion_lock:
            if completed.is_set():
                return
            try:
                write_preflight_json(directory, "hard_budget_stop.json", {
                    "reason": "PASSIVE_CASE_WALL_BUDGET_EXPIRED",
                    "max_wall_seconds": max_wall_seconds,
                    "resource_endpoint": current_process_resource_endpoint().to_value(),
                    "process_poll_count": 0, "exit_code": 124,
                    "restart_authorized": 0, "free_dialogue_complete": 0,
                })
            finally:
                os._exit(124)

    watcher = Thread(target=wait_once, name="relation-case-passive-budget", daemon=True)
    watcher.start()
    original = None
    try:
        yield
    except BaseException as error:
        original = error
        raise
    finally:
        with completion_lock:
            completed.set()
        watcher.join()
        try:
            write_preflight_json(directory, "completion.json", {
                "max_wall_seconds": max_wall_seconds,
                "status": "PASS" if original is None else "FAIL",
                "failure_type": None if original is None else type(original).__name__,
                "resource_endpoint": current_process_resource_endpoint().to_value(),
                "process_poll_count": 0, "restart_authorized": 0,
                "free_dialogue_complete": 0,
            })
        except BaseException as collection_error:
            if original is None:
                raise
            original.add_note("Case budget completion failed: " + type(collection_error).__name__)


__all__ = ["passive_relation_case_budget"]
