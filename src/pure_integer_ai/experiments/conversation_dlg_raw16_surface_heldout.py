"""DLG-RAW-16 G2：独立 held-out 表层泛化诊断。

这是一个反 theater 的诊断器，不是训练器，也不是正式评测器。它接收独立
构造的 typed ``SurfaceSelectionRequest``，先检查 semantic-shape 与 source /
family 是否没有出现在 G0 course，再调用只读 G1 selector。当前 exact-record
selector 对未见 shape 预期返回 ``NO_MATCH``；该结果明确标记为
``SURFACE_REPLAY_ONLY``，绝不宣称 PASS、TRAINED、MASTERED 或 readiness。

所有报告字段均可降解为有限字符串 scalar、UTF-8 u8 和整数 record。这里不
读取文件、不写 runtime state、不连接默认 terminal、不调用 LLM/网络。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    SurfaceOrganizationRecord,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SURFACE_NO_MATCH,
    SURFACE_SELECTED,
    SurfaceOrganizationSelector,
    SurfaceSelectionRequest,
    SurfaceSelectionResult,
)


DLG_RAW16_G2_PROTOCOL_V1 = 1
G2_HELD_OUT_NOT_READY = "HELD_OUT_NOT_READY"
G2_SURFACE_REPLAY_ONLY = "SURFACE_REPLAY_ONLY"
G2_CONTRACT_INVALID = "CONTRACT_INVALID"
G2_UNEXPECTED_MATCH = "UNEXPECTED_MATCH"
G2_OBSERVATION_ERROR = "OBSERVATION_ERROR"
G2_STATUSES = (
    G2_HELD_OUT_NOT_READY,
    G2_SURFACE_REPLAY_ONLY,
    G2_CONTRACT_INVALID,
    G2_UNEXPECTED_MATCH,
    G2_OBSERVATION_ERROR,
)
_G2_TRACE_DOMAIN = "pure_integer_ai.dlg_raw16.g2.heldout.v1"


class SurfaceHeldOutDiagnosticError(ValueError):
    """G2 held-out case、隔离键或报告不满足诊断合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceHeldOutDiagnosticError(
            f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise SurfaceHeldOutDiagnosticError(f"{where} 含非 Unicode scalar")
    return value


def _strict_int(value: Any, where: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int or (value < 0 if nonnegative else value <= 0):
        raise SurfaceHeldOutDiagnosticError(
            f"{where} 必须是{('非负' if nonnegative else '正')}严格整数")
    return value


def _scalars(value: str, where: str) -> tuple[int, ...]:
    _text(value, where)
    return tuple(ord(item) for item in value)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    return (len(values), *values)


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    return _pack(_scalars(value, where))


def _pack_text_tuple(values: tuple[str, ...], where: str) -> tuple[int, ...]:
    result = [len(values)]
    for index, value in enumerate(values):
        result.extend(_pack_text(value, f"{where}[{index}].text"))
    return tuple(result)


def semantic_shape_key_for_record(record: SurfaceOrganizationRecord) -> tuple[str, ...]:
    """从 G0 record 得出可读、可迁移的 semantic-shape 键。

    该键只描述 act、命题 kind、槽位角色和 required 位，不含 sample/source
    文本；G2 caller 可据此声明一个真正未见的组合，而不是只改编号。
    """
    if not isinstance(record, SurfaceOrganizationRecord):
        raise TypeError("semantic shape record 类型错误")
    result = ["act", record.dialogue_act, "kind", record.proposition_kind,
              "roles"]
    result.extend(item.role for item in record.clause_slots)
    result.append("required")
    result.extend(str(item.required) for item in record.clause_slots)
    return tuple(result)


def _request_shape_key(request: SurfaceSelectionRequest) -> tuple[str, ...]:
    """从 typed request 得出与 G0 shape 相容的保守键。

    request 只有 slot ids，没有 role labels；因此 G2 caller 必须显式提供
    ``semantic_shape_key``，本函数仅用于报告 request 的基础 act/kind 轨迹，
    不猜测槽位语义。
    """
    return ("act", request.dialogue_act, "kind", request.semantic.kind,
            "slot_count", str(len(request.ordered_clause_slots)))


@dataclass(frozen=True, slots=True)
class SurfaceHeldOutCase:
    """独立 held-out typed plan；不携带 accepted surface 或 evaluator label。"""

    case_id: str
    source_namespace: str
    semantic_shape_key: tuple[str, ...]
    request: SurfaceSelectionRequest

    def __post_init__(self) -> None:
        _text(self.case_id, "heldout.case_id")
        _text(self.source_namespace, "heldout.source_namespace")
        if (not isinstance(self.semantic_shape_key, tuple)
                or not self.semantic_shape_key
                or any(not isinstance(item, str) or not item
                       for item in self.semantic_shape_key)):
            raise SurfaceHeldOutDiagnosticError(
                "heldout.semantic_shape_key 必须是非空字符串 tuple")
        if not isinstance(self.request, SurfaceSelectionRequest):
            raise TypeError("heldout.request 类型错误")
        # Held-out identity must be explicit; a caller cannot accidentally
        # turn a broad query into a training-record lookup by omitting keys.
        for name in ("source_id", "context_id", "family_id", "template_family"):
            if getattr(self.request, name) is None:
                raise SurfaceHeldOutDiagnosticError(
                    f"heldout.request.{name} 必须显式绑定独立 identity")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_G2_PROTOCOL_V1]
        result.extend(_pack_text(self.case_id, "heldout.case_id"))
        result.extend(_pack_text(self.source_namespace,
                                 "heldout.source_namespace"))
        result.extend(_pack_text_tuple(self.semantic_shape_key,
                                       "heldout.semantic_shape_key"))
        result.extend(_pack(self.request.canonical_record()))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceHeldOutCaseObservation:
    """单个 held-out 运行的无标签观察。"""

    case_id: str
    shape_unseen: int
    observed_status_code: int
    result_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.case_id, "observation.case_id")
        if self.shape_unseen not in (0, 1):
            raise SurfaceHeldOutDiagnosticError(
                "observation.shape_unseen 必须是 0/1")
        if (not isinstance(self.result_trace, tuple)
                or not self.result_trace
                or any(type(item) is not int or item < 0
                       for item in self.result_trace)):
            raise SurfaceHeldOutDiagnosticError(
                "observation.result_trace 必须是非空非负整数 tuple")

    def canonical_record(self) -> tuple[int, ...]:
        result = [
            DLG_RAW16_G2_PROTOCOL_V1,
            self.shape_unseen,
            self.observed_status_code,
        ]
        result.extend(_pack_text(self.case_id, "observation.case_id"))
        result.extend(_pack(self.result_trace))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceHeldOutDiagnosticReport:
    """G2 诚实收口：当前 exact selector 只能证明 replay-only。"""

    status: str
    total_cases: int
    shape_unseen_cases: int
    no_match_cases: int
    unexpected_match_cases: int
    observations: tuple[SurfaceHeldOutCaseObservation, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in G2_STATUSES:
            raise SurfaceHeldOutDiagnosticError("G2 report status 未注册")
        for name in ("total_cases", "shape_unseen_cases", "no_match_cases",
                     "unexpected_match_cases"):
            _strict_int(getattr(self, name), f"report.{name}", nonnegative=True)
        if (not isinstance(self.observations, tuple)
                or len(self.observations) != self.total_cases
                or len({item.case_id for item in self.observations})
                != len(self.observations)):
            raise SurfaceHeldOutDiagnosticError("G2 report observations 不完整")
        if (self.shape_unseen_cases
                != sum(item.shape_unseen for item in self.observations)):
            raise SurfaceHeldOutDiagnosticError("G2 shape_unseen 分母漂移")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int or item < 0 for item in self.trace)):
            raise SurfaceHeldOutDiagnosticError("G2 report trace 非法")
        if self.status == G2_SURFACE_REPLAY_ONLY:
            if (self.total_cases == 0
                    or self.shape_unseen_cases != self.total_cases
                    or self.no_match_cases != self.total_cases
                    or self.unexpected_match_cases != 0):
                raise SurfaceHeldOutDiagnosticError(
                    "SURFACE_REPLAY_ONLY 计数不闭合")

    @property
    def ready(self) -> int:
        """G2 未通过；该诊断永不把 replay-only 变成 readiness。"""
        return 0

    @property
    def pass_(self) -> int:
        """保留显式 0，避免把拒绝未见输入误报为泛化 PASS。"""
        return 0

    def canonical_record(self) -> tuple[int, ...]:
        result = [
            DLG_RAW16_G2_PROTOCOL_V1,
            self.total_cases,
            self.shape_unseen_cases,
            self.no_match_cases,
            self.unexpected_match_cases,
        ]
        result.extend(_pack_text(self.status, "report.status"))
        result.append(len(self.observations))
        for item in self.observations:
            result.extend(_pack(item.canonical_record()))
        result.extend(_pack(self.trace))
        return tuple(result)


def _record_exact_signature(record: SurfaceOrganizationRecord) -> tuple[Any, ...]:
    return (
        record.dialogue_act,
        record.proposition_id,
        record.proposition_kind,
        record.proposition_subject,
        record.proposition_predicate,
        record.proposition_object,
        tuple(item.slot_id for item in record.clause_slots),
        record.register,
        record.required_proposition_ids,
        record.forbidden_proposition_ids,
        record.source_id,
        record.context_id,
        record.family_id,
        record.template_family,
    )


def _validate_case_is_independent(
        case: SurfaceHeldOutCase,
        selector: SurfaceOrganizationSelector,
        train_shapes: set[tuple[str, ...]],
        train_signatures: set[tuple[Any, ...]],
        train_sources: set[str],
        ) -> int:
    if case.semantic_shape_key in train_shapes:
        raise SurfaceHeldOutDiagnosticError(
            f"heldout {case.case_id} semantic-shape 泄漏到 course")
    request = case.request
    if request.source_id in train_sources:
        raise SurfaceHeldOutDiagnosticError(
            f"heldout {case.case_id} source_id 泄漏到 course")
    signature = (
        request.dialogue_act,
        request.semantic.proposition_id,
        request.semantic.kind,
        request.semantic.subject,
        request.semantic.predicate,
        request.semantic.object,
        request.ordered_clause_slots,
        request.register,
        request.required_proposition_ids,
        request.forbidden_proposition_ids,
        request.source_id,
        request.context_id,
        request.family_id,
        request.template_family,
    )
    if signature in train_signatures:
        raise SurfaceHeldOutDiagnosticError(
            f"heldout {case.case_id} exact request 重放 course")
    # A selector's own record set is authoritative for this check; the local
    # variable makes the boundary explicit and guards accidental future use of
    # a different runtime catalog.
    if not isinstance(selector, SurfaceOrganizationSelector):
        raise TypeError("G2 selector 类型错误")
    return int(case.semantic_shape_key not in train_shapes)


def run_surface_organization_g2_diagnostic(
        selector: SurfaceOrganizationSelector,
        cases: Iterable[SurfaceHeldOutCase],
        ) -> SurfaceHeldOutDiagnosticReport:
    """运行独立 G2 preflight + exact-selector diagnostic。

    ``SURFACE_REPLAY_ONLY`` 是当前预期结果：所有未见 shape 都被 exact
    selector 正确拒绝为 ``NO_MATCH``，因此证明了隔离，却没有证明泛化。
    """
    if not isinstance(selector, SurfaceOrganizationSelector):
        raise TypeError("G2 selector 类型错误")
    case_tuple = tuple(cases)
    if any(not isinstance(item, SurfaceHeldOutCase) for item in case_tuple):
        raise TypeError("G2 cases 含非法 heldout case")
    if not case_tuple:
        trace = integer_tuple_fingerprint(
            (DLG_RAW16_G2_PROTOCOL_V1, 0), domain=_G2_TRACE_DOMAIN)
        return SurfaceHeldOutDiagnosticReport(
            G2_HELD_OUT_NOT_READY, 0, 0, 0, 0, (), trace)
    if len({item.case_id for item in case_tuple}) != len(case_tuple):
        raise SurfaceHeldOutDiagnosticError("G2 case_id 必须唯一")
    records = selector.records
    train_shapes = {semantic_shape_key_for_record(item) for item in records}
    train_signatures = {_record_exact_signature(item) for item in records}
    train_sources = {item.source_id for item in records}
    observations: list[SurfaceHeldOutCaseObservation] = []
    shape_unseen = 0
    no_match = 0
    unexpected = 0
    try:
        for case in case_tuple:
            unseen = _validate_case_is_independent(
                case, selector, train_shapes, train_signatures, train_sources)
            result = selector.select(case.request)
            if not isinstance(result, SurfaceSelectionResult):
                raise SurfaceHeldOutDiagnosticError(
                    f"heldout {case.case_id} selector result 类型错误")
            if unseen:
                shape_unseen += 1
            if result.status_code == SURFACE_NO_MATCH:
                no_match += 1
            else:
                unexpected += 1
            observations.append(SurfaceHeldOutCaseObservation(
                case.case_id, unseen, result.status_code, result.trace))
    except SurfaceHeldOutDiagnosticError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise SurfaceHeldOutDiagnosticError("G2 heldout selector observation 失败") from error
    if shape_unseen == len(case_tuple) and no_match == len(case_tuple):
        status = G2_SURFACE_REPLAY_ONLY
    elif unexpected:
        status = G2_UNEXPECTED_MATCH
    else:
        status = G2_OBSERVATION_ERROR
    trace_values = [DLG_RAW16_G2_PROTOCOL_V1, len(case_tuple), shape_unseen,
                    no_match, unexpected]
    for item in observations:
        trace_values.extend(item.canonical_record())
    trace = integer_tuple_fingerprint(tuple(trace_values), domain=_G2_TRACE_DOMAIN)
    return SurfaceHeldOutDiagnosticReport(
        status,
        len(case_tuple),
        shape_unseen,
        no_match,
        unexpected,
        tuple(observations),
        trace,
    )


__all__ = [
    "DLG_RAW16_G2_PROTOCOL_V1",
    "G2_CONTRACT_INVALID",
    "G2_HELD_OUT_NOT_READY",
    "G2_OBSERVATION_ERROR",
    "G2_STATUSES",
    "G2_SURFACE_REPLAY_ONLY",
    "G2_UNEXPECTED_MATCH",
    "SurfaceHeldOutCase",
    "SurfaceHeldOutCaseObservation",
    "SurfaceHeldOutDiagnosticError",
    "SurfaceHeldOutDiagnosticReport",
    "run_surface_organization_g2_diagnostic",
    "semantic_shape_key_for_record",
]
