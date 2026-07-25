"""正式语言观察所需 occurrence、顺序、Span 和句界协议装配。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.train_context import TrainContext


def install_language_graph_protocols(
        ctx: TrainContext,
        *,
        occurrence_protocol: Any = None,
        occurrence_order_protocol: Any = None,
        span_protocol: Any = None,
        boundary_protocol: Any = None,
        prediction_protocol: Any = None,
        ) -> None:
    """按依赖顺序安装语言图协议，并拒绝缺少前置设施的部分配置。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")

    if occurrence_protocol is not None:
        from pure_integer_ai.cognition.understanding.occurrence_index import (
            OccurrenceIndex,
            OccurrenceProtocol,
        )
        if not isinstance(occurrence_protocol, OccurrenceProtocol):
            raise TypeError("language_occurrence_protocol 必须是 OccurrenceProtocol")
        ctx.occurrence_index = OccurrenceIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            occurrence_protocol,
        )

    if occurrence_order_protocol is not None:
        from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
        from pure_integer_ai.cognition.understanding.occurrence_order import (
            OccurrenceOrderProtocol,
            OccurrenceOrderReader,
            OccurrenceOrderWriter,
        )
        if ctx.occurrence_index is None:
            raise ValueError("L-06 occurrence 顺序协议必须同时启用 L-03 occurrence")
        if not isinstance(
                occurrence_order_protocol, OccurrenceOrderProtocol):
            raise TypeError(
                "language_occurrence_order_protocol 必须是 OccurrenceOrderProtocol")
        order_facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        ctx.occurrence_order_reader = OccurrenceOrderReader(
            order_facts,
            ctx.occurrence_index,
            occurrence_order_protocol,
        )
        ctx.occurrence_order_writer = OccurrenceOrderWriter(
            order_facts,
            occurrence_order_protocol,
        )

    if span_protocol is not None:
        from pure_integer_ai.cognition.understanding.segmentation_span import (
            SegmentationSpanMaterializer,
            SegmentationSpanProtocol,
        )
        from pure_integer_ai.cognition.understanding.span_index import SpanIndex
        if ctx.occurrence_index is None:
            raise ValueError("L-04 Span 协议必须同时启用 L-03 occurrence")
        if not isinstance(span_protocol, SegmentationSpanProtocol):
            raise TypeError(
                "language_span_protocol 必须是 SegmentationSpanProtocol")
        ctx.span_index = SpanIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            span_protocol.span_protocol,
            ctx.occurrence_index,
        )
        ctx.segmentation_span_materializer = SegmentationSpanMaterializer(
            ctx.span_index,
            span_protocol,
        )
        if ctx.word_form_providers is not None:
            ctx.word_form_providers.install_segmentation_span_materializer(
                ctx.segmentation_span_materializer)

    if boundary_protocol is not None:
        from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
            BoundaryHypothesisEngine,
        )
        from pure_integer_ai.cognition.understanding.boundary_span import (
            BoundarySpanMaterializer,
            BoundarySpanProtocol,
        )
        if ctx.span_index is None or ctx.occurrence_index is None:
            raise ValueError(
                "U-03 句界协议必须同时启用 L-03 occurrence 和 L-04 Span")
        if not isinstance(boundary_protocol, BoundarySpanProtocol):
            raise TypeError(
                "language_boundary_protocol 必须是 BoundarySpanProtocol")
        ctx.boundary_hypothesis_engine = BoundaryHypothesisEngine(
            boundary_protocol.hypothesis_protocol)
        ctx.boundary_span_materializer = BoundarySpanMaterializer(
            ctx.span_index,
            boundary_protocol,
        )

    if prediction_protocol is not None:
        from pure_integer_ai.cognition.understanding.language_prediction import (
            LanguagePredictionProtocol,
            LanguagePredictionRuntime,
        )
        if ctx.occurrence_index is None:
            raise ValueError("H-01 语言预测协议必须同时启用 L-03 occurrence")
        if not isinstance(prediction_protocol, LanguagePredictionProtocol):
            raise TypeError(
                "language_prediction_protocol 必须是 LanguagePredictionProtocol")
        ctx.language_prediction_runtime = LanguagePredictionRuntime(
            ctx.graph_ontology,
            ctx.occurrence_index,
            prediction_protocol,
        )


__all__ = ["install_language_graph_protocols"]
