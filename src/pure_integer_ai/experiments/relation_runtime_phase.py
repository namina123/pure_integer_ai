"""记录真实关系调用点的独立阶段流；计时不参与任何图语义或能力裁决。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator

from pure_integer_ai.experiments.batch_phase_journal import DurableBatchPhaseRecorder
from pure_integer_ai.experiments.batch_phase_evidence import BatchPhaseRecorder
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json
from pure_integer_ai.experiments.relation_resource_budget import resource_checkpoint


CASE_PHASES = (
    "materialization", "memory_append", "memory_recover", "query", "core",
    "response_plan", "generation", "delivery", "memory_commit",
)
_CURRENT: ContextVar[RelationRuntimePhases | None] = ContextVar(
    "relation_runtime_diagnostic_phases", default=None)


# object-model: resource_owner; representation=runtime; interop=diagnostic-phase-streams-v1
class RelationRuntimePhases:
    """每个实际调用点拥有一个边界流，嵌套区间明确重叠且不得求和。"""

    __slots__ = ("directory", "context", "required", "streams", "report")

    def __init__(self, directory: Path, *, run_identity: str, attempt_key: str,
                 case_ordinal: int, required_phases: tuple[str, ...]) -> None:
        """独占新 K 诊断目录并冻结必需阶段；拒绝覆盖任何旧尝试。"""
        self.directory = directory.resolve()
        if (self.directory.drive.upper() != "K:"
                or self.directory.parent == Path("K:/")):
            raise ValueError("runtime phase evidence requires a named explicit K root")
        if (any(type(value) is not str or not value.strip() for value in (run_identity, attempt_key))
                or type(case_ordinal) is not int or case_ordinal < 0):
            raise ValueError("runtime phase attempt identity is invalid")
        if type(required_phases) is not tuple or not required_phases:
            raise ValueError("runtime phase requirements must be a nonempty tuple")
        BatchPhaseRecorder(required_phases=required_phases)
        self.context = dict(run_identity=run_identity, attempt_key=attempt_key,
                            case_ordinal=case_ordinal)
        self.required = required_phases
        self.streams: dict[tuple[str, str], DurableBatchPhaseRecorder] = {}
        self.report: dict[str, object] | None = None
        self.directory.mkdir(exist_ok=False)
        write_preflight_json(self.directory, "identity.json", {
            **self.context, "required_phases": list(required_phases),
            "interval_accounting": "OVERLAPPING_CALL_SITE_STREAMS_DO_NOT_SUM",
            "language_qualification": "UNKNOWN_DIAGNOSTIC_ONLY",
            "free_dialogue_complete": 0,
        })

    @contextmanager
    def phase(self, name: str, call_site: str) -> Iterator[None]:
        """在实际调用前持久 BEGIN，成功/失败均保留同调用点 END。"""
        key = name, call_site
        if key not in self.streams:
            number = len(self.streams) + 1
            recorder = DurableBatchPhaseRecorder(
                self.directory / f"{number:03d}-{name}", **self.context,
                required_phases=(name,))
            self.streams[key] = recorder
            write_preflight_json(self.directory, f"bindings/{number:03d}.json", {
                "phase": name, "call_site": call_site,
                "stream": recorder.directory.name,
                "interval_accounting": "OVERLAPPING_CALL_SITE_STREAMS_DO_NOT_SUM",
            })
        with self.streams[key].phase(name):
            yield

    def finalize(self) -> dict[str, object]:
        """只汇总实际出现的边界；未运行的调用点不会被补成通过。"""
        rows = []
        for (name, site), recorder in self.streams.items():
            rows.append({"phase": name, "call_site": site, **recorder.report()})
        observed = {row["phase"] for row in rows}
        missing = sorted(set(self.required) - observed)
        status = "PASS" if not missing and all(
            row["journal"]["status"] == "PASS" for row in rows) else "FAIL"
        self.report = {
            **self.context, "status": status,
            "status_scope": "OBSERVED_CALL_SITE_BOUNDARIES_ONLY",
            "required_phases": list(self.required), "missing_phases": missing,
            "case_phase_presence": {name: int(name in observed)
                                    for name in dict.fromkeys((*CASE_PHASES, *self.required))},
            "positive_chain_phase_coverage": int(set(CASE_PHASES) <= observed),
            "streams": rows,
            "interval_accounting": "OVERLAPPING_CALL_SITE_STREAMS_DO_NOT_SUM",
            "phase_peak_qualification": "UNKNOWN_CUMULATIVE_PROCESS_PEAK_ONLY",
            "language_qualification": "UNKNOWN_TIMING_IS_NOT_BEHAVIOR",
            "free_dialogue_complete": 0,
        }
        write_preflight_json(self.directory, "receipt.json", self.report)
        return self.report


@contextmanager
def capture_relation_phases(directory: Path, *, run_identity: str, attempt_key: str,
                            case_ordinal: int,
                            required_phases: tuple[str, ...]) -> Iterator[RelationRuntimePhases]:
    """仅在显式诊断作用域接计时流；失败不掩盖业务异常、不自动恢复训练。"""
    phases = RelationRuntimePhases(
        directory, run_identity=run_identity, attempt_key=attempt_key,
        case_ordinal=case_ordinal, required_phases=required_phases)
    token = _CURRENT.set(phases)
    original = None
    try:
        yield phases
    except BaseException as error:
        original = error
        raise
    finally:
        _CURRENT.reset(token)
        try:
            phases.finalize()
        except BaseException as collection_error:
            if original is None:
                raise
            original.add_note("Runtime phase finalization failed: " + type(collection_error).__name__)


@contextmanager
def relation_phase(name: str, call_site: str) -> Iterator[None]:
    """无诊断 owner 时直接执行；记录器不能选择或改变查询结果。"""
    phases = _CURRENT.get()
    resource_checkpoint("begin:" + name)
    if phases is None:
        yield
    else:
        with phases.phase(name, call_site):
            yield
    resource_checkpoint("end:" + name)


def observed_relation_phase(name: str) -> Callable:
    """为明确生产方法绑定调用点身份，不包装宿主时钟为核心状态。"""
    def decorate(operation: Callable) -> Callable:
        """保留原方法签名与文档，诊断关闭时直接调用原函数。"""
        site = operation.__module__ + "." + operation.__qualname__

        @wraps(operation)
        def measured(*args, **kwargs):
            """只为当前诊断 owner 写阶段边界，原返回值与异常直接传递。"""
            with relation_phase(name, site):
                return operation(*args, **kwargs)
        return measured
    return decorate


__all__ = ["CASE_PHASES", "capture_relation_phases", "observed_relation_phase", "relation_phase"]
