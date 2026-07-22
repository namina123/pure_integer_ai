"""V-02 受控规模曲线、跑前预算和 K 盘可恢复 benchmark runner。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable

from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    read_curriculum_manifest,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.corpus_identity import (
    assign_corpus_source_refs,
)
from pure_integer_ai.experiments.data_manifest import read_manifest
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.language_course_intake import (
    build_word_form_providers,
)
from pure_integer_ai.experiments.language_observation import (
    _apply_word_form_providers,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.train_execution import (
    backend_operation_delta,
    capture_execution_snapshot,
    item_candidate_counts,
    table_growth_delta,
)
from pure_integer_ai.experiments.train_gate_profile import (
    push_production_training_gates,
    reset_production_training_gates,
)
from pure_integer_ai.experiments.v02_run_store import (
    HostMonotonicClock,
    HostProcessMemory,
    V02RunStore,
    canonical_json_bytes,
    sha256_path,
)
from pure_integer_ai.experiments.v02_scale_types import (
    LanguageProtocolSpec,
    V02Budget,
    V02_SCHEMA_VERSION,
    default_v02_budget,
    evaluate_lane_budget,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    collect_backend_telemetry,
    record_candidate_count,
    suppress_backend_telemetry,
    telemetry_scope,
)
from pure_integer_ai.training.cursor import dump_run
from pure_integer_ai.training.stages import STAGE1_SKELETON


ClockSource = Callable[[], int]
MemorySource = Callable[[], dict[str, int]]
V02_HOTSPOT_EVENTS = (
    "hotspot.abstract_mark",
    "hotspot.ancestor",
    "hotspot.hub",
    "hotspot.nearest",
    "hotspot.normalize",
    "hotspot.pronoun",
    "hotspot.skeleton_read",
)


@dataclass(frozen=True)
class RuntimeLanguageProtocols:
    """把外部协议配置转换后的运行时对象集中传给两个 benchmark lane。"""

    segmentation: Any = None
    occurrence: Any = None
    occurrence_order: Any = None
    span: Any = None
    boundary: Any = None


@dataclass(frozen=True)
class V02ScaleConfig:
    """一次 V-02 曲线的路径、规模、输入身份和执行 lane。"""

    output_root: str
    run_id: str
    course_root: str
    source_manifest_path: str
    corpus_path: str
    runtime_language: int
    source_kind: int
    domain: int
    visible_splits: tuple[int, ...]
    source_namespace: str | int
    scales: tuple[int, ...] = (20, 50, 100, 300)
    protocol_spec: LanguageProtocolSpec = LanguageProtocolSpec()
    budget: V02Budget | None = None
    join_whitespace: bool = False
    run_provider: bool = True
    run_observe: bool = True
    run_curriculum: bool = True
    measure_evaluation_clone: bool = True
    measure_dump: bool = True
    stop_after_scale: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
                ("runtime_language", self.runtime_language),
                ("source_kind", self.source_kind),
                ("domain", self.domain)):
            assert_int(value, _where=f"V02ScaleConfig.{name}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"V02ScaleConfig.{name} 必须为严格正整数")
        if type(self.source_namespace) is int:
            if self.source_namespace <= 0:
                raise ValueError("V-02 source_namespace 整数必须为正数")
        elif type(self.source_namespace) is str:
            if not self.source_namespace.strip():
                raise ValueError("V-02 source_namespace 字符串不能为空")
        else:
            raise TypeError("V-02 source_namespace 必须是 str 或严格正整数")
        if not isinstance(self.visible_splits, tuple) or not self.visible_splits:
            raise ValueError("V-02 visible_splits 必须是非空 tuple")
        assert_int(*self.visible_splits, _where="V02ScaleConfig.visible_splits")
        scales = tuple(sorted(set(self.scales)))
        if not scales or scales != self.scales:
            raise ValueError("V-02 scales 必须严格递增且唯一")
        assert_int(*scales, _where="V02ScaleConfig.scales")
        if any(type(value) is not int or value <= 0 for value in scales):
            raise ValueError("V-02 scales 必须使用严格正整数")
        for name in (
                "join_whitespace", "run_provider", "run_observe",
                "run_curriculum", "measure_evaluation_clone", "measure_dump"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"V02ScaleConfig.{name} 必须是 bool")
        if not any((self.run_provider, self.run_observe, self.run_curriculum)):
            raise ValueError("V-02 至少启用一个执行 lane")
        if self.budget is not None:
            for lane in ("observe", "curriculum"):
                for n in scales:
                    self.budget.lane_budget(lane, n)
        if self.stop_after_scale is not None:
            assert_int(
                self.stop_after_scale,
                _where="V02ScaleConfig.stop_after_scale",
            )
            if type(self.stop_after_scale) is not int \
                    or self.stop_after_scale not in scales:
                raise ValueError(
                    "V-02 stop_after_scale 必须是预注册规模点或 None")

    def resolved_budget(self) -> V02Budget:
        """返回调用方预算或施工前固定默认预算。"""
        return self.budget or default_v02_budget(self.scales)

    def execution_scales(self) -> tuple[int, ...]:
        """返回本次调用允许施工的已预注册规模前缀。"""
        if self.stop_after_scale is None:
            return self.scales
        return tuple(n for n in self.scales if n <= self.stop_after_scale)


def _read_json(path: str | Path) -> Any:
    """严格读取 UTF-8 JSON。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_protocols(spec: LanguageProtocolSpec
                       ) -> RuntimeLanguageProtocols:
    """只从显式协议配置构造运行时对象，不在代码中生成领域键。"""
    segmentation = None
    occurrence = None
    occurrence_order = None
    span = None
    boundary = None
    if spec.segmentation_enabled:
        from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
            SegmentationProtocol,
        )
        segmentation = SegmentationProtocol(
            spec.segmentation_hypothesis_kind_key,
            spec.segmentation_lexical_reason_key,
            spec.segmentation_oov_reason_key,
            spec.segmentation_candidate_limit,
        )
    if spec.occurrence_enabled:
        from pure_integer_ai.cognition.understanding.occurrence_index import (
            OccurrenceProtocol,
        )
        occurrence = OccurrenceProtocol(
            spec.occurrence_candidate_relation_key,
            spec.occurrence_speaker_relation_key,
        )
    if spec.occurrence_order_relation_key is not None:
        from pure_integer_ai.cognition.understanding.occurrence_order import (
            OccurrenceOrderProtocol,
        )
        occurrence_order = OccurrenceOrderProtocol(
            spec.occurrence_order_relation_key)
    if spec.span_enabled:
        from pure_integer_ai.cognition.understanding.segmentation_span import (
            SegmentationSpanProtocol,
        )
        from pure_integer_ai.cognition.understanding.span_index import SpanProtocol
        span = SegmentationSpanProtocol(
            span_protocol=SpanProtocol(
                spec.span_structure_relation_key,
                spec.span_constituent_relation_key,
                spec.span_occurrence_relation_key,
                spec.span_candidate_relation_key,
            ),
            document_structure_key=spec.span_document_structure_key,
            part_structure_key=spec.span_part_structure_key,
            candidate_shape_namespace_key=(
                spec.span_candidate_shape_namespace_key),
            atomic_structure_key=spec.span_atomic_structure_key,
        )
    if spec.boundary_enabled:
        from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
            BoundaryHypothesisProtocol,
        )
        from pure_integer_ai.cognition.understanding.boundary_span import (
            BoundarySpanProtocol,
        )
        boundary = BoundarySpanProtocol(
            hypothesis_protocol=BoundaryHypothesisProtocol(
                spec.boundary_hypothesis_kind_key),
            span_protocol=span.span_protocol,
            document_structure_key=spec.boundary_document_structure_key,
            candidate_structure_key=spec.boundary_candidate_structure_key,
            anchor_structure_key=spec.boundary_anchor_structure_key,
            candidate_shape_namespace_key=(
                spec.boundary_candidate_shape_namespace_key),
            selection_relation_key=spec.boundary_selection_relation_key,
            withdrawal_relation_key=spec.boundary_withdrawal_relation_key,
            selection_clock_kind=spec.boundary_selection_clock_kind,
        )
    return RuntimeLanguageProtocols(
        segmentation,
        occurrence,
        occurrence_order,
        span,
        boundary,
    )


def load_benchmark_corpus(
        path: str | Path,
        *,
        limit: int,
        source_kind: int,
        runtime_language: int,
        domain: int,
        join_whitespace: bool,
        ) -> list[CollectedItem]:
    """按空段落读取实验输入，纯注释块跳过，是否去分词空白由配置决定。"""
    assert_int(
        limit, source_kind, runtime_language, domain,
        _where="load_benchmark_corpus",
    )
    if limit <= 0:
        raise ValueError("benchmark corpus limit 必须为正")
    text = Path(path).read_text(encoding="utf-8")
    items: list[CollectedItem] = []
    for block in text.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or all(line.startswith("#") for line in lines):
            continue
        raw_text = " ".join(lines)
        tokens = raw_text.split()
        if not tokens:
            continue
        if join_whitespace:
            raw_text = "".join(tokens)
        items.append(CollectedItem(
            tokens=tokens,
            raw_text=raw_text,
            collect_type=COLLECT_PRECEDES,
            source=source_kind,
            lang=runtime_language,
            domain=domain,
        ))
        if len(items) >= limit:
            break
    if len(items) < limit:
        raise ValueError(
            f"benchmark corpus 只有 {len(items)} 项，少于请求规模 {limit}")
    return items


def _backend_digest(backend: DictBackend) -> str:
    """对 canonical 后端快照计算稳定摘要，供优化前后语义比对。"""
    payload = canonical_json_bytes(backend.snapshot())
    return hashlib.sha256(payload).hexdigest()


def _query_calls(operations: list[dict[str, Any]]) -> int:
    """汇总 select/count 调用数，写操作不混入每 item 查询预算。"""
    return sum(
        row["calls"] for row in operations
        if row["operation"] in {"select", "count"}
    )


def _positive_growth_rows(growth: list[dict[str, Any]]) -> int:
    """汇总所有表的正增长，保留逐表明细供对象增长审计。"""
    return sum(max(row["growth"], 0) for row in growth)


def _edge_growth(growth: list[dict[str, Any]]) -> int:
    """返回通用 edge 表增长；缺表时为零。"""
    return sum(row["growth"] for row in growth if row["table"] == "edge")


def _telemetry_query_calls(payload: dict[str, Any] | None) -> int:
    """从完整 V-01 报告读取 select/count 总调用数。"""
    if payload is None:
        return 0
    return sum(
        row["calls"] for row in payload["operation_totals"]
        if row["operation"] in {"select", "count"}
    )


def _top_operations(payload: dict[str, Any] | None,
                    *, limit: int = 20) -> list[dict[str, Any]]:
    """按调用数、行数和稳定名称选出主要 table/operation 热点。"""
    if payload is None:
        return []
    rows = sorted(
        payload["table_operations"],
        key=lambda row: (
            -row["calls"], -row["rows"],
            row["operation"], row["table"]),
    )
    return rows[:limit]


def _top_queries(payload: dict[str, Any] | None,
                 *, limit: int = 20) -> list[dict[str, Any]]:
    """按查询 scope 的后端调用总数列出主要函数或阶段归因。"""
    if payload is None:
        return []
    ranked = []
    for row in payload["by_dimension"]["query"]:
        calls = sum(operation["calls"] for operation in row["operations"])
        touched_rows = sum(operation["rows"] for operation in row["operations"])
        events = sum(event["count"] for event in row.get("events", []))
        ranked.append({
            "query": row["value"],
            "calls": calls,
            "rows": touched_rows,
            "candidates": row["candidates"],
            "events": events,
        })
    return sorted(
        ranked,
        key=lambda row: (
            -row["calls"], -row["events"], -row["rows"],
            repr(row["query"]),
        ),
    )[:limit]


def _hotspot_review(payload: dict[str, Any] | None) -> dict[str, Any]:
    """按开放诊断事件汇总七类历史热点，并显式列出未覆盖项。"""
    observed = {} if payload is None else {
        row["kind"]: row["count"]
        for row in payload.get("event_totals", [])
    }
    rows = [
        {
            "event": event,
            "calls": observed.get(event, 0),
            "observed": observed.get(event, 0) > 0,
        }
        for event in V02_HOTSPOT_EVENTS
    ]
    return {
        "events": rows,
        "observed": [row["event"] for row in rows if row["observed"]],
        "missing": [row["event"] for row in rows if not row["observed"]],
        "all_observed": all(row["observed"] for row in rows),
    }


def _merge_hotspot_reviews(
        reviews: list[dict[str, Any]],
        ) -> dict[str, Any]:
    """合并多个 fresh lane 的热点调用数，不把零覆盖误写为性能通过。"""
    totals = {event: 0 for event in V02_HOTSPOT_EVENTS}
    for review in reviews:
        for row in review.get("events", []):
            if row.get("event") in totals:
                totals[row["event"]] += row.get("calls", 0)
    return _hotspot_review({
        "event_totals": [
            {"kind": event, "count": count}
            for event, count in totals.items()
        ],
    })


def _current_memory_bytes(memory_source: MemorySource) -> int:
    """读取当前工作集，缺失字段时 fail closed 为零。"""
    sample = memory_source()
    value = sample.get("current_working_set_bytes", 0)
    assert_int(value, _where="v02.current_memory")
    return value if value >= 0 else 0


def _implementation_manifest() -> dict[str, str]:
    """记录 active 包全部 Python 实现摘要，避免优化比较漏掉间接依赖。"""
    root = Path(__file__).resolve().parents[1]
    relative_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts and "_archive" not in path.parts
    )
    return {
        relative_path: sha256_path(root / relative_path)
        for relative_path in relative_paths
    }


def _input_manifest(config: V02ScaleConfig) -> dict[str, Any]:
    """读取轻量 manifest 身份和文件摘要，不提前扫描课程 artifact。"""
    course_root = Path(config.course_root).resolve()
    course_manifest_path = course_root / "manifest.json"
    course = read_curriculum_manifest(course_manifest_path)
    source = read_manifest(config.source_manifest_path)
    protocol_payload = config.protocol_spec.to_json()
    return {
        "course_root": str(course_root),
        "course_manifest_path": str(course_manifest_path.resolve()),
        "course_manifest_file_sha256": sha256_path(course_manifest_path),
        "course_manifest_sha256": course.sha256(),
        "course_artifacts": [item.to_dict() for item in course.artifacts],
        "source_manifest_path": str(
            Path(config.source_manifest_path).resolve()),
        "source_manifest_file_sha256": sha256_path(
            config.source_manifest_path),
        "source_manifest_sha256": source.sha256(),
        "corpus_path": str(Path(config.corpus_path).resolve()),
        "corpus_sha256": sha256_path(config.corpus_path),
        "protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol_payload)).hexdigest(),
    }


def _preregistration(config: V02ScaleConfig) -> dict[str, Any]:
    """构造必须在任何 measured workload 前写入的冻结清单。"""
    budget = config.resolved_budget()
    return {
        "schema_version": V02_SCHEMA_VERSION,
        "run_id": config.run_id,
        "inputs": _input_manifest(config),
        "config": {
            "runtime_language": config.runtime_language,
            "source_kind": config.source_kind,
            "domain": config.domain,
            "visible_splits": list(config.visible_splits),
            "source_namespace": config.source_namespace,
            "scales": list(config.scales),
            "join_whitespace": config.join_whitespace,
            "run_provider": config.run_provider,
            "run_observe": config.run_observe,
            "run_curriculum": config.run_curriculum,
            "measure_evaluation_clone": config.measure_evaluation_clone,
            "measure_dump": config.measure_dump,
            "protocol": config.protocol_spec.to_json(),
        },
        "budget": budget.to_json(),
        "implementation": _implementation_manifest(),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "backend": "DictBackend",
        },
    }


def _build_provider_context(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        ) -> tuple[Any, Any, Any]:
    """装配一次 provider 上下文，不附加 benchmark 自身的摘要与遥测成本。"""
    ctx = make_train_context(DictBackend())
    registry, report = build_word_form_providers(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        course_root=config.course_root,
        source_manifest_path=config.source_manifest_path,
        runtime_language=config.runtime_language,
        visible_splits=config.visible_splits,
        segmentation_protocol=protocols.segmentation,
    )
    return ctx, registry, report


def _measure_provider_build(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        *,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        caller: str,
        ) -> tuple[Any, Any, Any, dict[str, Any]]:
    """在预先装配的空上下文上计量一次完整 provider 核验和目录构建。"""
    ctx = make_train_context(DictBackend())
    collector = BackendTelemetryCollector()
    memory_before = memory_source()
    with collect_backend_telemetry(collector):
        before = capture_execution_snapshot(
            ctx.backend,
            working_set_source=lambda: _current_memory_bytes(memory_source),
        )
        started_ns = clock_ns()
        with telemetry_scope(caller=caller, query="provider_assembly"):
            registry, report = build_word_form_providers(
                backend=ctx.backend,
                concept_index=ctx.concept_index,
                ontology=ctx.graph_ontology,
                course_root=config.course_root,
                source_manifest_path=config.source_manifest_path,
                runtime_language=config.runtime_language,
                visible_splits=config.visible_splits,
                segmentation_protocol=protocols.segmentation,
            )
        elapsed_ns = clock_ns() - started_ns
        after = capture_execution_snapshot(
            ctx.backend,
            working_set_source=lambda: _current_memory_bytes(memory_source),
        )
    operations = backend_operation_delta(before.operations, after.operations)
    growth = table_growth_delta(before.table_sizes, after.table_sizes)
    telemetry = collector.to_json()
    payload = {
        "elapsed_ns": elapsed_ns,
        "memory_before": memory_before,
        "memory_after": memory_source(),
        "backend_operations": operations,
        "table_growth": growth,
        "query_calls": _query_calls(operations),
        "growth_rows": _positive_growth_rows(growth),
        "backend_sha256": _backend_digest(ctx.backend),
        "course_report": asdict(report),
        "telemetry": telemetry,
        "top_operations": _top_operations(telemetry),
        "top_queries": _top_queries(telemetry),
    }
    return ctx, registry, report, payload


def _provider_phase(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        *,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        need_context: bool,
        ) -> tuple[Any, Any, Any]:
    """首次运行测冷/热装配；恢复时只重建 direct lane 所需上下文。"""
    result_path = store.run_root / "provider.json"
    budget = config.resolved_budget()
    if result_path.is_file():
        if not need_context:
            return None, None, None
        return _build_provider_context(config, protocols)

    cold_ctx, cold_registry, cold_report, cold = _measure_provider_build(
        config,
        protocols,
        clock_ns=clock_ns,
        memory_source=memory_source,
        caller="benchmark:provider_cold",
    )
    warm_ctx, _warm_registry, _warm_report, warm = _measure_provider_build(
        config,
        protocols,
        clock_ns=clock_ns,
        memory_source=memory_source,
        caller="benchmark:provider_warm",
    )
    canonical_equal = cold["backend_sha256"] == warm["backend_sha256"]
    payload = {
        "cold": cold,
        "warm": warm,
        "canonical_equal": canonical_equal,
        "budget": {
            "cold_passed": (
                cold["elapsed_ns"]
                <= budget.provider_cold_max_elapsed_ns),
            "warm_passed": (
                warm["elapsed_ns"]
                <= budget.provider_warm_max_elapsed_ns),
            "provider_cold_max_elapsed_ns":
                budget.provider_cold_max_elapsed_ns,
            "provider_warm_max_elapsed_ns":
                budget.provider_warm_max_elapsed_ns,
        },
    }
    store.write_named_result("provider.json", payload)
    close = getattr(warm_ctx.backend, "close", None)
    if callable(close):
        close()
    del warm_ctx, _warm_registry, _warm_report
    gc.collect()
    if need_context:
        return cold_ctx, cold_registry, cold_report
    close = getattr(cold_ctx.backend, "close", None)
    if callable(close):
        close()
    return None, None, None


def _observe_setup(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        *,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        ) -> tuple[Any, Any, Any]:
    """为 direct observe 准备上下文；未启用 provider lane 时不伪造测量结果。"""
    if config.run_provider:
        return _provider_phase(
            config,
            protocols,
            store,
            clock_ns=clock_ns,
            memory_source=memory_source,
            need_context=True,
        )
    return _build_provider_context(config, protocols)


def _measure_clone(ctx: Any, *, label: str, clock_ns: ClockSource,
                   memory_source: MemorySource) -> dict[str, Any]:
    """计量一次完整 evaluation clone，并核验退出后宿主摘要不变。"""
    before_digest = _backend_digest(ctx.backend)
    memory_before = memory_source()
    started_ns = clock_ns()
    with isolated_evaluation(ctx, label=label):
        pass
    elapsed_ns = clock_ns() - started_ns
    after_digest = _backend_digest(ctx.backend)
    return {
        "elapsed_ns": elapsed_ns,
        "memory_before": memory_before,
        "memory_after": memory_source(),
        "host_state_unchanged": before_digest == after_digest,
    }


def _measure_dump(ctx: Any, store: V02RunStore, *, n: int,
                  clock_ns: ClockSource,
                  memory_source: MemorySource) -> dict[str, Any]:
    """把 direct observe 当前状态 dump 到独立 K 盘目录并计量成本。"""
    run_id = f"observe_n{n:06d}"
    memory_before = memory_source()
    started_ns = clock_ns()
    dumped_spaces = dump_run(
        ctx.backend,
        str(store.dumps_root),
        run_id,
        spaces=[ctx.space_id],
    )
    elapsed_ns = clock_ns() - started_ns
    run_root = store.dumps_root / run_id
    files = []
    for path in sorted(run_root.iterdir()):
        if path.is_file():
            files.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            })
    return {
        "elapsed_ns": elapsed_ns,
        "memory_before": memory_before,
        "memory_after": memory_source(),
        "dumped_spaces": dumped_spaces,
        "files": files,
    }


def _run_observe_curve(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        ctx: Any,
        registry: Any,
        course_report: Any,
        *,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        ) -> None:
    """在同一已装配 provider/context 上累计处理 item，隔离单次 observe 扩展性。"""
    items = load_benchmark_corpus(
        config.corpus_path,
        limit=max(config.execution_scales()),
        source_kind=config.source_kind,
        runtime_language=config.runtime_language,
        domain=config.domain,
        join_whitespace=config.join_whitespace,
    )
    assign_corpus_source_refs(
        items,
        source_namespace=config.source_namespace,
    )
    ctx.word_form_providers = registry
    ctx.word_form_course_report = course_report
    install_started_ns = clock_ns()
    install_language_graph_protocols(
        ctx,
        occurrence_protocol=protocols.occurrence,
        occurrence_order_protocol=protocols.occurrence_order,
        span_protocol=protocols.span,
        boundary_protocol=protocols.boundary,
    )
    protocol_install_elapsed_ns = clock_ns() - install_started_ns

    collector = BackendTelemetryCollector()
    runner = DefaultRoundRunner()
    targets = set(config.execution_scales())
    largest_target = max(config.execution_scales())
    candidate_count = 0
    retokenized_items = 0
    active_elapsed_ns = 0
    gate_token = push_production_training_gates()
    try:
        with collect_backend_telemetry(collector):
            baseline = capture_execution_snapshot(
                ctx.backend,
                working_set_source=lambda: _current_memory_bytes(memory_source),
            )
            lane_peak_working_set_bytes = baseline.working_set_bytes
            for item_index, item in enumerate(items, start=1):
                if item.source_ref is None:
                    raise ValueError("direct observe item 缺 SourceRef")
                scope = document_scope(item.source_ref)
                item_started_ns = clock_ns()
                with telemetry_scope(
                        caller="benchmark:direct_observe",
                        query="direct_observe_item",
                        source_key=item.source_ref.stable_key(),
                        scope_key=scope.stable_key(),
                        stage=STAGE1_SKELETON,
                        round_id=0,
                        item_index=item_index - 1):
                    retokenized_items += _apply_word_form_providers(
                        [item], registry, commit_evidence=True)
                    for kind, count in item_candidate_counts(item).items():
                        record_candidate_count(kind, count)
                        candidate_count += count
                    runner.run_round(
                        ctx,
                        item,
                        STAGE1_SKELETON,
                        item_index - 1,
                    )
                active_elapsed_ns += clock_ns() - item_started_ns
                if item_index not in targets:
                    continue

                if store.has_point("observe", item_index):
                    continue

                point_snapshot = capture_execution_snapshot(
                    ctx.backend,
                    working_set_source=lambda: _current_memory_bytes(
                        memory_source),
                )
                lane_peak_working_set_bytes = max(
                    lane_peak_working_set_bytes,
                    point_snapshot.working_set_bytes,
                )
                elapsed_ns = active_elapsed_ns
                operations = backend_operation_delta(
                    baseline.operations, point_snapshot.operations)
                growth = table_growth_delta(
                    baseline.table_sizes, point_snapshot.table_sizes)
                memory = memory_source()
                clone = None
                dump = None
                if item_index == largest_target:
                    with suppress_backend_telemetry():
                        if config.measure_evaluation_clone:
                            clone = _measure_clone(
                                ctx,
                                label=f"v02-observe-{item_index}",
                                clock_ns=clock_ns,
                                memory_source=memory_source,
                            )
                        if config.measure_dump:
                            dump = _measure_dump(
                                ctx,
                                store,
                                n=item_index,
                                clock_ns=clock_ns,
                                memory_source=memory_source,
                            )
                query_calls = _query_calls(operations)
                growth_rows = _positive_growth_rows(growth)
                peak_bytes = lane_peak_working_set_bytes
                budget = evaluate_lane_budget(
                    config.resolved_budget().lane_budget(
                        "observe", item_index),
                    elapsed_ns=elapsed_ns,
                    query_calls=query_calls,
                    growth_rows=growth_rows,
                    candidate_count=candidate_count,
                    peak_working_set_bytes=peak_bytes,
                )
                fixed_budget = config.resolved_budget()
                if clone is not None:
                    clone["budget_passed"] = (
                        clone["elapsed_ns"]
                        <= fixed_budget.evaluation_clone_max_elapsed_ns)
                    clone["max_elapsed_ns"] = (
                        fixed_budget.evaluation_clone_max_elapsed_ns)
                if dump is not None:
                    dump["budget_passed"] = (
                        dump["elapsed_ns"]
                        <= fixed_budget.dump_max_elapsed_ns)
                    dump["max_elapsed_ns"] = fixed_budget.dump_max_elapsed_ns
                point = {
                    "lane": "observe",
                    "measurement": "cumulative_single_context",
                    "n": item_index,
                    "elapsed_ns": elapsed_ns,
                    "query_calls": query_calls,
                    "candidate_count": candidate_count,
                    "retokenized_items": retokenized_items,
                    "growth_rows": growth_rows,
                    "edge_growth": _edge_growth(growth),
                    "backend_operations": operations,
                    "table_growth": growth,
                    "memory": memory,
                    "peak_working_set_bytes": peak_bytes,
                    "protocol_install_elapsed_ns":
                        protocol_install_elapsed_ns,
                    "evaluation_clone": clone,
                    "dump": dump,
                    "backend_sha256": _backend_digest(ctx.backend),
                    "budget": budget,
                }
                store.write_point("observe", item_index, point)
    finally:
        reset_production_training_gates(gate_token)

    telemetry = collector.to_json()
    store.write_named_result("observe_telemetry.json", {
        "telemetry": telemetry,
        "top_operations": _top_operations(telemetry),
        "top_queries": _top_queries(telemetry),
        "hotspot_review": _hotspot_review(telemetry),
    })


def _formal_config(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        *,
        n: int,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        ) -> FormalTrainConfig:
    """构造只跑第一观察阶段的正式课程编排配置。"""
    return FormalTrainConfig(
        run_dir=str(store.artifacts_root / "curriculum"),
        run_id=f"curriculum_n{n:06d}",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        persist_graph_dump=False,
        telemetry_enabled=True,
        telemetry_clock_ns=clock_ns,
        telemetry_working_set_bytes=lambda: _current_memory_bytes(
            memory_source),
        language_course_root=config.course_root,
        language_source_manifest_path=config.source_manifest_path,
        language_course_runtime_language=config.runtime_language,
        language_course_visible_splits=config.visible_splits,
        language_segmentation_protocol=protocols.segmentation,
        language_occurrence_protocol=protocols.occurrence,
        language_occurrence_order_protocol=protocols.occurrence_order,
        language_span_protocol=protocols.span,
        language_boundary_protocol=protocols.boundary,
    )


def _evaluation_costs(payload: dict[str, Any] | None) -> dict[str, Any]:
    """从 V-01 caller 桶汇总 formal_train 内部 evaluation 调用成本。"""
    if payload is None:
        return {"calls": 0, "rows": 0, "callers": []}
    callers = []
    total_calls = 0
    total_rows = 0
    for row in payload["by_dimension"]["caller"]:
        if not isinstance(row["value"], str) \
                or not row["value"].startswith("evaluation:"):
            continue
        calls = sum(operation["calls"] for operation in row["operations"])
        rows = sum(operation["rows"] for operation in row["operations"])
        total_calls += calls
        total_rows += rows
        callers.append({
            "caller": row["value"],
            "calls": calls,
            "rows": rows,
        })
    return {
        "calls": total_calls,
        "rows": total_rows,
        "callers": sorted(callers, key=lambda row: row["caller"]),
    }


def _run_curriculum_point(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        *,
        n: int,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        ) -> dict[str, Any]:
    """在 fresh backend 上运行一个完整 formal curriculum 规模点。"""
    items = load_benchmark_corpus(
        config.corpus_path,
        limit=n,
        source_kind=config.source_kind,
        runtime_language=config.runtime_language,
        domain=config.domain,
        join_whitespace=config.join_whitespace,
    )
    backend = DictBackend()
    memory_before = memory_source()
    started_ns = clock_ns()
    result = formal_train(
        _formal_config(
            config,
            protocols,
            store,
            n=n,
            clock_ns=clock_ns,
            memory_source=memory_source,
        ),
        items,
        backend=backend,
    )
    elapsed_ns = clock_ns() - started_ns
    telemetry = result.execution.backend_telemetry
    query_calls = _telemetry_query_calls(telemetry)
    growth = result.execution.run_table_growth
    growth_rows = _positive_growth_rows(growth)
    candidate_count = sum(
        sum(item_candidate_counts(item).values()) for item in items)
    memory_after = memory_source()
    peak_bytes = max(
        result.execution.peak_working_set_bytes,
        memory_before.get("current_working_set_bytes", 0),
        memory_after.get("current_working_set_bytes", 0),
    )
    budget = evaluate_lane_budget(
        config.resolved_budget().lane_budget("curriculum", n),
        elapsed_ns=elapsed_ns,
        query_calls=query_calls,
        growth_rows=growth_rows,
        candidate_count=candidate_count,
        peak_working_set_bytes=peak_bytes,
    )
    execution_path = (
        store.artifacts_root / "curriculum"
        / f"curriculum_n{n:06d}" / "execution.json"
    )
    point = {
        "lane": "curriculum",
        "measurement": "fresh_formal_train_with_provider_assembly",
        "n": n,
        "elapsed_ns": elapsed_ns,
        "query_calls": query_calls,
        "candidate_count": candidate_count,
        "growth_rows": growth_rows,
        "edge_growth": _edge_growth(growth),
        "table_growth": growth,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "peak_working_set_bytes": peak_bytes,
        "execution": {
            "bootstrap_elapsed_ns": result.execution.bootstrap_elapsed_ns,
            "discovery_elapsed_ns": result.execution.discovery_elapsed_ns,
            "stage_loop_elapsed_ns": result.execution.stage_loop_elapsed_ns,
            "finalize_elapsed_ns": result.execution.finalize_elapsed_ns,
            "total_elapsed_ns": result.execution.total_elapsed_ns,
            "stages": [stage.to_json() for stage in result.execution.stages],
            "phases": [phase.to_json() for phase in result.execution.phases],
        },
        "execution_path": str(execution_path.resolve()),
        "evaluation_costs": _evaluation_costs(telemetry),
        "top_operations": _top_operations(telemetry),
        "top_queries": _top_queries(telemetry),
        "hotspot_review": _hotspot_review(telemetry),
        "backend_sha256": _backend_digest(backend),
        "budget": budget,
    }
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    return point


def _run_curriculum_curve(
        config: V02ScaleConfig,
        protocols: RuntimeLanguageProtocols,
        store: V02RunStore,
        *,
        clock_ns: ClockSource,
        memory_source: MemorySource,
        ) -> None:
    """按规模独立运行 fresh formal_train，已完成点直接跳过。"""
    for n in config.execution_scales():
        if store.has_point("curriculum", n):
            continue
        point = _run_curriculum_point(
            config,
            protocols,
            store,
            n=n,
            clock_ns=clock_ns,
            memory_source=memory_source,
        )
        store.write_point("curriculum", n, point)
        _write_summary(config, store)
        gc.collect()


def _normalized_growth(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """导出每项规模指标，并比较相邻规模的单位耗时增长。"""
    out = []
    previous = None
    for point in sorted(points, key=lambda item: item["n"]):
        per_item_ns = (point["elapsed_ns"] + point["n"] - 1) // point["n"]
        ratio = None
        if previous is not None:
            ratio = per_item_ns * 1000 // max(previous, 1)
        out.append({
            "n": point["n"],
            "elapsed_ns_per_item": per_item_ns,
            "query_calls_per_item": (
                point["query_calls"] + point["n"] - 1) // point["n"],
            "candidate_count_per_item": (
                point["candidate_count"] + point["n"] - 1) // point["n"],
            "growth_rows_per_item": (
                point["growth_rows"] + point["n"] - 1) // point["n"],
            "edge_growth_per_item": (
                point["edge_growth"] + point["n"] - 1) // point["n"],
            "current_working_set_bytes": (
                point.get("memory", point.get("memory_after", {})).get(
                    "current_working_set_bytes", 0)),
            "peak_working_set_bytes": point.get(
                "peak_working_set_bytes", 0),
            "growth_permille_from_previous": ratio,
        })
        previous = per_item_ns
    return out


def _read_points(store: V02RunStore, lane: str,
                 scales: tuple[int, ...]) -> list[dict[str, Any]]:
    """读取一个 lane 已完成的所有预注册规模点。"""
    return [
        store.read_point(lane, n)
        for n in scales if store.has_point(lane, n)
    ]


def _write_summary(config: V02ScaleConfig, store: V02RunStore) -> None:
    """汇总当前已完成曲线、预算失败和严格恢复点。"""
    budget = config.resolved_budget()
    observe = _read_points(store, "observe", config.scales)
    curriculum = _read_points(store, "curriculum", config.scales)
    observe_growth = _normalized_growth(observe)
    curriculum_growth = _normalized_growth(curriculum)
    growth_checks = {
        "observe": (not config.run_observe) or all(
            row["growth_permille_from_previous"] is None
            or row["growth_permille_from_previous"]
            <= budget.max_normalized_growth_permille
            for row in observe_growth
        ),
        "curriculum": (not config.run_curriculum) or all(
            row["growth_permille_from_previous"] is None
            or row["growth_permille_from_previous"]
            <= budget.max_normalized_growth_permille
            for row in curriculum_growth
        ),
    }
    point_passes = [
        point["budget"]["passed"]
        for point in observe + curriculum
    ]
    clone_passes = [
        point["evaluation_clone"]["budget_passed"]
        for point in observe
        if point.get("evaluation_clone") is not None
    ]
    dump_passes = [
        point["dump"]["budget_passed"]
        for point in observe if point.get("dump") is not None
    ]
    provider_path = store.run_root / "provider.json"
    provider = _read_json(provider_path) if provider_path.is_file() else None
    observe_telemetry_path = store.run_root / "observe_telemetry.json"
    observe_telemetry = (
        _read_json(observe_telemetry_path)
        if observe_telemetry_path.is_file() else None
    )
    hotspot_reviews = [
        point["hotspot_review"]
        for point in curriculum if "hotspot_review" in point
    ]
    if observe_telemetry is not None:
        hotspot_reviews.append(observe_telemetry["hotspot_review"])
    provider_passes = [] if provider is None else [
        provider["budget"]["cold_passed"],
        provider["budget"]["warm_passed"],
        provider["canonical_equal"],
    ]
    expected = {
        "provider": config.run_provider,
        "observe": list(config.scales) if config.run_observe else [],
        "curriculum": list(config.scales) if config.run_curriculum else [],
    }
    completed = {
        "provider": provider is not None,
        "observe": [point["n"] for point in observe],
        "curriculum": [point["n"] for point in curriculum],
    }
    all_expected_complete = (
        completed["provider"] == expected["provider"]
        and all(
        completed[lane] == expected[lane]
        for lane in ("observe", "curriculum")
        )
    )
    all_checks = (
        point_passes + clone_passes + dump_passes + provider_passes
        + list(growth_checks.values())
    )
    store.write_summary({
        "schema_version": V02_SCHEMA_VERSION,
        "run_id": config.run_id,
        "expected": expected,
        "completed": completed,
        "all_expected_complete": all_expected_complete,
        "budget_passed": all_expected_complete and all(all_checks),
        "normalized_growth": {
            "max_permille": budget.max_normalized_growth_permille,
            "checks": growth_checks,
            "observe": observe_growth,
            "curriculum": curriculum_growth,
        },
        "provider": provider,
        "hotspot_review": _merge_hotspot_reviews(hotspot_reviews),
        "points": {
            "observe": observe,
            "curriculum": curriculum,
        },
    })


def run_v02_scale_curve(
        config: V02ScaleConfig,
        *,
        clock_ns: ClockSource | None = None,
        memory_source: MemorySource | None = None,
        ) -> Path:
    """先原子预注册，再按 provider、observe、curriculum 顺序执行可恢复曲线。"""
    if not isinstance(config, V02ScaleConfig):
        raise TypeError("config 必须是 V02ScaleConfig")
    clock = clock_ns or HostMonotonicClock()
    memory = memory_source or HostProcessMemory()
    store = V02RunStore(config.output_root, config.run_id)
    store.preregister(_preregistration(config))
    _write_summary(config, store)

    protocols = _runtime_protocols(config.protocol_spec)
    observe_complete = all(
        store.has_point("observe", n) for n in config.execution_scales())
    need_context = config.run_observe and not observe_complete
    ctx = registry = course_report = None
    if need_context:
        ctx, registry, course_report = _observe_setup(
            config,
            protocols,
            store,
            clock_ns=clock,
            memory_source=memory,
        )
        _write_summary(config, store)
    elif config.run_provider:
        ctx, registry, course_report = _provider_phase(
            config,
            protocols,
            store,
            clock_ns=clock,
            memory_source=memory,
            need_context=need_context,
        )
        _write_summary(config, store)
    if need_context:
        _run_observe_curve(
            config,
            protocols,
            store,
            ctx,
            registry,
            course_report,
            clock_ns=clock,
            memory_source=memory,
        )
        _write_summary(config, store)
        close = getattr(ctx.backend, "close", None)
        if callable(close):
            close()
        del ctx, registry, course_report
        gc.collect()
    if config.run_curriculum:
        _run_curriculum_curve(
            config,
            protocols,
            store,
            clock_ns=clock,
            memory_source=memory,
        )
    _write_summary(config, store)
    return store.summary_path


def _parse_integer_tuple(value: str) -> tuple[int, ...]:
    """把逗号分隔 CLI 参数解析为严格整数 tuple。"""
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",")
                       if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔整数") from error
    if not parsed:
        raise argparse.ArgumentTypeError("整数列表不能为空")
    return parsed


def _config_from_args(args: argparse.Namespace) -> V02ScaleConfig:
    """把 CLI 参数和可选 JSON 文件收敛为严格配置。"""
    protocol = LanguageProtocolSpec.from_json(
        None if args.protocol_json is None else _read_json(args.protocol_json))
    budget = None if args.budget_json is None else V02Budget.from_json(
        _read_json(args.budget_json))
    lanes = set(args.lanes.split(","))
    unknown = lanes - {"provider", "observe", "curriculum"}
    if unknown:
        raise ValueError(f"未知 V-02 lane: {sorted(unknown)}")
    return V02ScaleConfig(
        output_root=args.output_root,
        run_id=args.run_id,
        course_root=args.course_root,
        source_manifest_path=args.source_manifest,
        corpus_path=args.corpus,
        runtime_language=args.runtime_language,
        source_kind=args.source_kind,
        domain=args.domain,
        visible_splits=args.visible_splits,
        source_namespace=(
            args.source_namespace
            if args.source_namespace is not None
            else sha256_path(args.corpus)
        ),
        scales=args.scales,
        protocol_spec=protocol,
        budget=budget,
        join_whitespace=args.join_whitespace,
        run_provider="provider" in lanes,
        run_observe="observe" in lanes,
        run_curriculum="curriculum" in lanes,
        measure_evaluation_clone=not args.skip_evaluation_clone,
        measure_dump=not args.skip_dump,
        stop_after_scale=args.stop_after_scale,
    )


def _main() -> None:
    """V-02 命令行入口，只打印最终摘要路径。"""
    parser = argparse.ArgumentParser(
        description="运行 V-02 provider/observe/curriculum 受控规模曲线")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--course-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--runtime-language", required=True, type=int)
    parser.add_argument("--source-kind", required=True, type=int)
    parser.add_argument("--domain", required=True, type=int)
    parser.add_argument(
        "--visible-splits", type=_parse_integer_tuple, default=(1,))
    parser.add_argument("--source-namespace")
    parser.add_argument(
        "--scales", type=_parse_integer_tuple, default=(20, 50, 100, 300))
    parser.add_argument("--protocol-json")
    parser.add_argument("--budget-json")
    parser.add_argument(
        "--lanes", default="provider,observe,curriculum")
    parser.add_argument("--join-whitespace", action="store_true")
    parser.add_argument("--skip-evaluation-clone", action="store_true")
    parser.add_argument("--skip-dump", action="store_true")
    parser.add_argument("--stop-after-scale", type=int)
    args = parser.parse_args()
    summary = run_v02_scale_curve(_config_from_args(args))
    print(summary)


if __name__ == "__main__":
    _main()


__all__ = [
    "RuntimeLanguageProtocols",
    "V02_HOTSPOT_EVENTS",
    "V02ScaleConfig",
    "load_benchmark_corpus",
    "run_v02_scale_curve",
]
