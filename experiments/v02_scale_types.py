"""V-02 规模曲线使用的严格整数预算和语言协议配置。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


V02_SCHEMA_VERSION = 1
_ONE_SECOND_NS = 1_000_000_000
_ONE_GIB = 1_073_741_824


def _strict_positive(value: int, *, where: str) -> int:
    """校验严格正整数并拒绝 bool。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须是严格正整数")
    return value


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验严格非负整数并拒绝 bool。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须是严格非负整数")
    return value


def _integer_key(value: Any, *, where: str,
                 optional: bool = False) -> tuple[int, ...] | None:
    """把 JSON 整数列表恢复为开放键，缺省字段可显式返回 None。"""
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} 必须是非空整数序列")
    key = tuple(value)
    assert_int(*key, _where=where)
    if any(type(item) is not int for item in key):
        raise ValueError(f"{where} 必须使用严格整数")
    return key


def _validate_protocol_key(value: tuple[int, ...] | None, *,
                           where: str) -> None:
    """校验直接构造的协议开放键，防止绕过 JSON 入口注入宽类型。"""
    if value is None:
        return
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")


def _reject_unknown_fields(value: dict[str, Any], *, where: str,
                           allowed: frozenset[str]) -> None:
    """拒绝协议 JSON 拼写错误，避免未知语义字段被静默忽略。"""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{where} 含未知字段: {sorted(unknown)}")


@dataclass(frozen=True, order=True)
class LaneBudget:
    """单个规模点的总耗时、查询、增长、候选和工作集上限。"""

    n: int
    max_total_elapsed_ns: int
    max_query_calls_per_item: int
    max_growth_rows_per_item: int
    max_candidates_per_item: int
    max_peak_working_set_bytes: int

    def __post_init__(self) -> None:
        _strict_positive(self.n, where="LaneBudget.n")
        for name, value in (
                ("max_total_elapsed_ns", self.max_total_elapsed_ns),
                ("max_query_calls_per_item", self.max_query_calls_per_item),
                ("max_growth_rows_per_item", self.max_growth_rows_per_item),
                ("max_candidates_per_item", self.max_candidates_per_item),
                ("max_peak_working_set_bytes",
                 self.max_peak_working_set_bytes)):
            _strict_nonnegative(value, where=f"LaneBudget.{name}")

    def to_json(self) -> dict[str, int]:
        """导出可写入预注册清单的纯整数对象。"""
        return {
            "n": self.n,
            "max_total_elapsed_ns": self.max_total_elapsed_ns,
            "max_query_calls_per_item": self.max_query_calls_per_item,
            "max_growth_rows_per_item": self.max_growth_rows_per_item,
            "max_candidates_per_item": self.max_candidates_per_item,
            "max_peak_working_set_bytes": self.max_peak_working_set_bytes,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LaneBudget":
        """从预算 JSON 恢复一个规模点。"""
        return cls(
            value["n"],
            value["max_total_elapsed_ns"],
            value["max_query_calls_per_item"],
            value["max_growth_rows_per_item"],
            value["max_candidates_per_item"],
            value["max_peak_working_set_bytes"],
        )


@dataclass(frozen=True)
class V02Budget:
    """V-02 跑前冻结的固定成本、曲线和归一化增长预算。"""

    provider_cold_max_elapsed_ns: int
    provider_warm_max_elapsed_ns: int
    evaluation_clone_max_elapsed_ns: int
    dump_max_elapsed_ns: int
    max_normalized_growth_permille: int
    observe: tuple[LaneBudget, ...]
    curriculum: tuple[LaneBudget, ...]

    def __post_init__(self) -> None:
        for name, value in (
                ("provider_cold_max_elapsed_ns",
                 self.provider_cold_max_elapsed_ns),
                ("provider_warm_max_elapsed_ns",
                 self.provider_warm_max_elapsed_ns),
                ("evaluation_clone_max_elapsed_ns",
                 self.evaluation_clone_max_elapsed_ns),
                ("dump_max_elapsed_ns", self.dump_max_elapsed_ns),
                ("max_normalized_growth_permille",
                 self.max_normalized_growth_permille)):
            _strict_nonnegative(value, where=f"V02Budget.{name}")
        for lane_name, points in (
                ("observe", self.observe),
                ("curriculum", self.curriculum)):
            ns = [point.n for point in points]
            if not ns or ns != sorted(set(ns)):
                raise ValueError(f"V02Budget.{lane_name} 规模必须严格递增且唯一")

    def lane_budget(self, lane: str, n: int) -> LaneBudget:
        """读取指定 lane 和规模的唯一预算，缺失时拒绝事后补阈值。"""
        points = self.observe if lane == "observe" else (
            self.curriculum if lane == "curriculum" else ())
        matches = [point for point in points if point.n == n]
        if len(matches) != 1:
            raise KeyError(f"预算未预注册: lane={lane!r}, n={n}")
        return matches[0]

    def to_json(self) -> dict[str, Any]:
        """导出稳定预算对象。"""
        return {
            "provider_cold_max_elapsed_ns":
                self.provider_cold_max_elapsed_ns,
            "provider_warm_max_elapsed_ns":
                self.provider_warm_max_elapsed_ns,
            "evaluation_clone_max_elapsed_ns":
                self.evaluation_clone_max_elapsed_ns,
            "dump_max_elapsed_ns": self.dump_max_elapsed_ns,
            "max_normalized_growth_permille":
                self.max_normalized_growth_permille,
            "observe": [point.to_json() for point in self.observe],
            "curriculum": [point.to_json() for point in self.curriculum],
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "V02Budget":
        """从预注册预算文件恢复预算。"""
        if not isinstance(value, dict):
            raise TypeError("V02Budget JSON 根必须是对象")
        return cls(
            value["provider_cold_max_elapsed_ns"],
            value["provider_warm_max_elapsed_ns"],
            value["evaluation_clone_max_elapsed_ns"],
            value["dump_max_elapsed_ns"],
            value["max_normalized_growth_permille"],
            tuple(LaneBudget.from_json(item) for item in value["observe"]),
            tuple(LaneBudget.from_json(item)
                  for item in value["curriculum"]),
        )


def default_v02_budget(scales: tuple[int, ...]) -> V02Budget:
    """按施工前固定公式生成首轮预算，不读取任何运行结果。"""
    normalized = tuple(sorted(set(scales)))
    if not normalized:
        raise ValueError("V-02 规模列表不能为空")
    for n in normalized:
        _strict_positive(n, where="default_v02_budget.scale")

    observe = tuple(
        LaneBudget(
            n=n,
            max_total_elapsed_ns=(60 + n * 4 // 5) * _ONE_SECOND_NS,
            max_query_calls_per_item=30_000,
            max_growth_rows_per_item=5_000,
            max_candidates_per_item=256,
            max_peak_working_set_bytes=8 * _ONE_GIB,
        )
        for n in normalized
    )
    curriculum = tuple(
        LaneBudget(
            n=n,
            max_total_elapsed_ns=(120 + n * 4 // 5) * _ONE_SECOND_NS,
            max_query_calls_per_item=35_000,
            max_growth_rows_per_item=5_000,
            max_candidates_per_item=256,
            max_peak_working_set_bytes=8 * _ONE_GIB,
        )
        for n in normalized
    )
    return V02Budget(
        provider_cold_max_elapsed_ns=120 * _ONE_SECOND_NS,
        provider_warm_max_elapsed_ns=90 * _ONE_SECOND_NS,
        evaluation_clone_max_elapsed_ns=45 * _ONE_SECOND_NS,
        dump_max_elapsed_ns=60 * _ONE_SECOND_NS,
        max_normalized_growth_permille=2_000,
        observe=observe,
        curriculum=curriculum,
    )


@dataclass(frozen=True)
class LanguageProtocolSpec:
    """由独立 JSON 注入的分词、occurrence、Span 和句界开放键。"""

    segmentation_hypothesis_kind_key: tuple[int, ...] | None = None
    segmentation_lexical_reason_key: tuple[int, ...] | None = None
    segmentation_oov_reason_key: tuple[int, ...] | None = None
    segmentation_candidate_limit: int = 0
    occurrence_candidate_relation_key: tuple[int, ...] | None = None
    occurrence_speaker_relation_key: tuple[int, ...] | None = None
    occurrence_order_relation_key: tuple[int, ...] | None = None
    span_structure_relation_key: tuple[int, ...] | None = None
    span_constituent_relation_key: tuple[int, ...] | None = None
    span_occurrence_relation_key: tuple[int, ...] | None = None
    span_candidate_relation_key: tuple[int, ...] | None = None
    span_document_structure_key: tuple[int, ...] | None = None
    span_part_structure_key: tuple[int, ...] | None = None
    span_candidate_shape_namespace_key: tuple[int, ...] | None = None
    span_atomic_structure_key: tuple[int, ...] | None = None
    boundary_hypothesis_kind_key: tuple[int, ...] | None = None
    boundary_document_structure_key: tuple[int, ...] | None = None
    boundary_candidate_structure_key: tuple[int, ...] | None = None
    boundary_anchor_structure_key: tuple[int, ...] | None = None
    boundary_candidate_shape_namespace_key: tuple[int, ...] | None = None
    boundary_selection_relation_key: tuple[int, ...] | None = None
    boundary_withdrawal_relation_key: tuple[int, ...] | None = None
    boundary_selection_clock_kind: int = 0

    def __post_init__(self) -> None:
        for name in (
                "segmentation_hypothesis_kind_key",
                "segmentation_lexical_reason_key",
                "segmentation_oov_reason_key",
                "occurrence_candidate_relation_key",
                "occurrence_speaker_relation_key",
                "occurrence_order_relation_key",
                "span_structure_relation_key",
                "span_constituent_relation_key",
                "span_occurrence_relation_key",
                "span_candidate_relation_key",
                "span_document_structure_key",
                "span_part_structure_key",
                "span_candidate_shape_namespace_key",
                "span_atomic_structure_key",
                "boundary_hypothesis_kind_key",
                "boundary_document_structure_key",
                "boundary_candidate_structure_key",
                "boundary_anchor_structure_key",
                "boundary_candidate_shape_namespace_key",
                "boundary_selection_relation_key",
                "boundary_withdrawal_relation_key"):
            _validate_protocol_key(
                getattr(self, name),
                where=f"LanguageProtocolSpec.{name}",
            )
        segmentation_keys = (
            self.segmentation_hypothesis_kind_key,
            self.segmentation_lexical_reason_key,
            self.segmentation_oov_reason_key,
        )
        if any(key is not None for key in segmentation_keys):
            if not all(key is not None for key in segmentation_keys):
                raise ValueError("分词协议三个开放键必须同时提供")
            _strict_positive(
                self.segmentation_candidate_limit,
                where="LanguageProtocolSpec.segmentation_candidate_limit",
            )
            if self.segmentation_candidate_limit < 3:
                raise ValueError("分词候选预算必须至少为 3")
        elif self.segmentation_candidate_limit != 0:
            raise ValueError("未配置分词键时候选预算必须为 0")

        if (self.occurrence_speaker_relation_key is not None
                and self.occurrence_candidate_relation_key is None):
            raise ValueError("occurrence speaker 协议必须同时配置候选关系")

        span_required_keys = (
            self.span_structure_relation_key,
            self.span_constituent_relation_key,
            self.span_occurrence_relation_key,
            self.span_candidate_relation_key,
            self.span_document_structure_key,
            self.span_part_structure_key,
            self.span_candidate_shape_namespace_key,
        )
        span_keys = span_required_keys + (self.span_atomic_structure_key,)
        if any(key is not None for key in span_keys):
            if not all(key is not None for key in span_required_keys):
                raise ValueError("Span 协议基础关系和结构键必须同时提供")
            if self.occurrence_candidate_relation_key is None:
                raise ValueError("Span 协议必须同时配置 occurrence 协议")
            if self.segmentation_hypothesis_kind_key is None:
                raise ValueError("Span 协议必须同时配置分词协议")
        if (self.occurrence_order_relation_key is not None
                and self.occurrence_candidate_relation_key is None):
            raise ValueError("occurrence 顺序协议必须同时配置 occurrence 协议")

        boundary_keys = (
            self.boundary_hypothesis_kind_key,
            self.boundary_document_structure_key,
            self.boundary_candidate_structure_key,
            self.boundary_anchor_structure_key,
            self.boundary_candidate_shape_namespace_key,
            self.boundary_selection_relation_key,
            self.boundary_withdrawal_relation_key,
        )
        if any(key is not None for key in boundary_keys):
            if not all(key is not None for key in boundary_keys):
                raise ValueError("句界协议全部开放键必须同时提供")
            if not self.span_enabled:
                raise ValueError("句界协议必须同时配置 L-04 Span 协议")
            _strict_positive(
                self.boundary_selection_clock_kind,
                where="LanguageProtocolSpec.boundary_selection_clock_kind",
            )
        elif self.boundary_selection_clock_kind != 0:
            raise ValueError("未配置句界键时选择时钟 kind 必须为 0")

    @property
    def segmentation_enabled(self) -> bool:
        """返回是否完整配置分词候选协议。"""
        return self.segmentation_hypothesis_kind_key is not None

    @property
    def occurrence_enabled(self) -> bool:
        """返回是否配置 occurrence 图关系协议。"""
        return self.occurrence_candidate_relation_key is not None

    @property
    def span_enabled(self) -> bool:
        """返回是否完整配置分词 Span 协议。"""
        return self.span_structure_relation_key is not None

    @property
    def boundary_enabled(self) -> bool:
        """返回是否完整配置 U-03 句界协议。"""
        return self.boundary_hypothesis_kind_key is not None

    def to_json(self) -> dict[str, Any]:
        """导出与输入 JSON 同构的协议对象。"""
        def key(value: tuple[int, ...] | None) -> list[int] | None:
            return None if value is None else list(value)

        return {
            "segmentation": None if not self.segmentation_enabled else {
                "hypothesis_kind_key": key(
                    self.segmentation_hypothesis_kind_key),
                "lexical_match_reason_key": key(
                    self.segmentation_lexical_reason_key),
                "oov_reason_key": key(self.segmentation_oov_reason_key),
                "candidate_limit": self.segmentation_candidate_limit,
            },
            "occurrence": None if not self.occurrence_enabled else {
                "candidate_relation_key": key(
                    self.occurrence_candidate_relation_key),
                "speaker_relation_key": key(
                    self.occurrence_speaker_relation_key),
            },
            "occurrence_order": (
                None if self.occurrence_order_relation_key is None else {
                    "relation_key": key(self.occurrence_order_relation_key),
                }
            ),
            "span": None if not self.span_enabled else {
                "structure_relation_key": key(
                    self.span_structure_relation_key),
                "constituent_relation_key": key(
                    self.span_constituent_relation_key),
                "occurrence_relation_key": key(
                    self.span_occurrence_relation_key),
                "candidate_relation_key": key(
                    self.span_candidate_relation_key),
                "document_structure_key": key(
                    self.span_document_structure_key),
                "part_structure_key": key(self.span_part_structure_key),
                "candidate_shape_namespace_key": key(
                    self.span_candidate_shape_namespace_key),
                "atomic_structure_key": key(
                    self.span_atomic_structure_key),
            },
            "boundary": None if not self.boundary_enabled else {
                "hypothesis_kind_key": key(
                    self.boundary_hypothesis_kind_key),
                "document_structure_key": key(
                    self.boundary_document_structure_key),
                "candidate_structure_key": key(
                    self.boundary_candidate_structure_key),
                "anchor_structure_key": key(
                    self.boundary_anchor_structure_key),
                "candidate_shape_namespace_key": key(
                    self.boundary_candidate_shape_namespace_key),
                "selection_relation_key": key(
                    self.boundary_selection_relation_key),
                "withdrawal_relation_key": key(
                    self.boundary_withdrawal_relation_key),
                "selection_clock_kind": self.boundary_selection_clock_kind,
            },
        }

    @classmethod
    def from_json(cls, value: dict[str, Any] | None
                  ) -> "LanguageProtocolSpec":
        """从独立协议 JSON 恢复开放键，不生成任何宿主内置语义键。"""
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("语言协议 JSON 根必须是对象")
        _reject_unknown_fields(
            value,
            where="protocol",
            allowed=frozenset({
                "segmentation", "occurrence", "occurrence_order", "span",
                "boundary",
            }),
        )
        segmentation = value.get("segmentation")
        occurrence = value.get("occurrence")
        order = value.get("occurrence_order")
        span = value.get("span")
        boundary = value.get("boundary")
        for name, section in (
                ("segmentation", segmentation),
                ("occurrence", occurrence),
                ("occurrence_order", order),
                ("span", span),
                ("boundary", boundary)):
            if section is not None and not isinstance(section, dict):
                raise TypeError(f"语言协议 {name} 必须是对象或 null")
        if segmentation is not None:
            _reject_unknown_fields(
                segmentation,
                where="protocol.segmentation",
                allowed=frozenset({
                    "hypothesis_kind_key", "lexical_match_reason_key",
                    "oov_reason_key", "candidate_limit",
                }),
            )
        if occurrence is not None:
            _reject_unknown_fields(
                occurrence,
                where="protocol.occurrence",
                allowed=frozenset({
                    "candidate_relation_key", "speaker_relation_key",
                }),
            )
        if order is not None:
            _reject_unknown_fields(
                order,
                where="protocol.occurrence_order",
                allowed=frozenset({"relation_key"}),
            )
        if span is not None:
            _reject_unknown_fields(
                span,
                where="protocol.span",
                allowed=frozenset({
                    "structure_relation_key", "constituent_relation_key",
                    "occurrence_relation_key", "candidate_relation_key",
                    "document_structure_key", "part_structure_key",
                    "candidate_shape_namespace_key", "atomic_structure_key",
                }),
            )
        if boundary is not None:
            _reject_unknown_fields(
                boundary,
                where="protocol.boundary",
                allowed=frozenset({
                    "hypothesis_kind_key", "document_structure_key",
                    "candidate_structure_key", "anchor_structure_key",
                    "candidate_shape_namespace_key", "selection_relation_key",
                    "withdrawal_relation_key", "selection_clock_kind",
                }),
            )

        return cls(
            segmentation_hypothesis_kind_key=(
                None if segmentation is None else _integer_key(
                    segmentation.get("hypothesis_kind_key"),
                    where="protocol.segmentation.hypothesis_kind_key")),
            segmentation_lexical_reason_key=(
                None if segmentation is None else _integer_key(
                    segmentation.get("lexical_match_reason_key"),
                    where="protocol.segmentation.lexical_match_reason_key")),
            segmentation_oov_reason_key=(
                None if segmentation is None else _integer_key(
                    segmentation.get("oov_reason_key"),
                    where="protocol.segmentation.oov_reason_key")),
            segmentation_candidate_limit=(
                0 if segmentation is None else segmentation["candidate_limit"]),
            occurrence_candidate_relation_key=(
                None if occurrence is None else _integer_key(
                    occurrence.get("candidate_relation_key"),
                    where="protocol.occurrence.candidate_relation_key")),
            occurrence_speaker_relation_key=(
                None if occurrence is None else _integer_key(
                    occurrence.get("speaker_relation_key"),
                    where="protocol.occurrence.speaker_relation_key",
                    optional=True)),
            occurrence_order_relation_key=(
                None if order is None else _integer_key(
                    order.get("relation_key"),
                    where="protocol.occurrence_order.relation_key")),
            span_structure_relation_key=(
                None if span is None else _integer_key(
                    span.get("structure_relation_key"),
                    where="protocol.span.structure_relation_key")),
            span_constituent_relation_key=(
                None if span is None else _integer_key(
                    span.get("constituent_relation_key"),
                    where="protocol.span.constituent_relation_key")),
            span_occurrence_relation_key=(
                None if span is None else _integer_key(
                    span.get("occurrence_relation_key"),
                    where="protocol.span.occurrence_relation_key")),
            span_candidate_relation_key=(
                None if span is None else _integer_key(
                    span.get("candidate_relation_key"),
                    where="protocol.span.candidate_relation_key")),
            span_document_structure_key=(
                None if span is None else _integer_key(
                    span.get("document_structure_key"),
                    where="protocol.span.document_structure_key")),
            span_part_structure_key=(
                None if span is None else _integer_key(
                    span.get("part_structure_key"),
                    where="protocol.span.part_structure_key")),
            span_candidate_shape_namespace_key=(
                None if span is None else _integer_key(
                    span.get("candidate_shape_namespace_key"),
                    where="protocol.span.candidate_shape_namespace_key")),
            span_atomic_structure_key=(
                None if span is None else _integer_key(
                    span.get("atomic_structure_key"),
                    where="protocol.span.atomic_structure_key",
                    optional=True)),
            boundary_hypothesis_kind_key=(
                None if boundary is None else _integer_key(
                    boundary.get("hypothesis_kind_key"),
                    where="protocol.boundary.hypothesis_kind_key")),
            boundary_document_structure_key=(
                None if boundary is None else _integer_key(
                    boundary.get("document_structure_key"),
                    where="protocol.boundary.document_structure_key")),
            boundary_candidate_structure_key=(
                None if boundary is None else _integer_key(
                    boundary.get("candidate_structure_key"),
                    where="protocol.boundary.candidate_structure_key")),
            boundary_anchor_structure_key=(
                None if boundary is None else _integer_key(
                    boundary.get("anchor_structure_key"),
                    where="protocol.boundary.anchor_structure_key")),
            boundary_candidate_shape_namespace_key=(
                None if boundary is None else _integer_key(
                    boundary.get("candidate_shape_namespace_key"),
                    where=("protocol.boundary."
                           "candidate_shape_namespace_key"))),
            boundary_selection_relation_key=(
                None if boundary is None else _integer_key(
                    boundary.get("selection_relation_key"),
                    where="protocol.boundary.selection_relation_key")),
            boundary_withdrawal_relation_key=(
                None if boundary is None else _integer_key(
                    boundary.get("withdrawal_relation_key"),
                    where="protocol.boundary.withdrawal_relation_key")),
            boundary_selection_clock_kind=(
                0 if boundary is None else boundary["selection_clock_kind"]),
        )


def evaluate_lane_budget(
        budget: LaneBudget,
        *,
        elapsed_ns: int,
        query_calls: int,
        growth_rows: int,
        candidate_count: int,
        peak_working_set_bytes: int,
        ) -> dict[str, Any]:
    """按预注册阈值逐项判定，保留所有失败而不压成单一均分。"""
    for name, value in (
            ("elapsed_ns", elapsed_ns),
            ("query_calls", query_calls),
            ("growth_rows", growth_rows),
            ("candidate_count", candidate_count),
            ("peak_working_set_bytes", peak_working_set_bytes)):
        _strict_nonnegative(value, where=f"evaluate_lane_budget.{name}")
    n = budget.n
    query_per_item = (query_calls + n - 1) // n
    growth_per_item = (growth_rows + n - 1) // n
    candidates_per_item = (candidate_count + n - 1) // n
    checks = {
        "elapsed": elapsed_ns <= budget.max_total_elapsed_ns,
        "query_calls_per_item": (
            query_per_item <= budget.max_query_calls_per_item),
        "growth_rows_per_item": (
            growth_per_item <= budget.max_growth_rows_per_item),
        "candidates_per_item": (
            candidates_per_item <= budget.max_candidates_per_item),
        "peak_working_set_bytes": (
            peak_working_set_bytes
            <= budget.max_peak_working_set_bytes),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "elapsed_ns": elapsed_ns,
            "query_calls_per_item": query_per_item,
            "growth_rows_per_item": growth_per_item,
            "candidates_per_item": candidates_per_item,
            "peak_working_set_bytes": peak_working_set_bytes,
        },
        "budget": budget.to_json(),
    }


__all__ = [
    "LanguageProtocolSpec",
    "LaneBudget",
    "V02Budget",
    "V02_SCHEMA_VERSION",
    "default_v02_budget",
    "evaluate_lane_budget",
]
