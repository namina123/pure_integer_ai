"""公开 Runtime 资料使用的 typed S-02/S-03 协议装配器。

该入口只装配与公开对话训练共用的整数协议，不读取训练产物，也不把 Runtime
资料自动 promotion 到 Core。资料是否形成课程仍由调用方显式携带 typed payload
决定；没有 payload 时 semantic mapper 保持无 lesson。
"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationProtocol,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.language_course_intake import (
    LanguageCourseIntakeReport,
    build_word_form_providers,
)
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    SPLIT_TRAIN,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    install_language_semantic_course_runtime,
)
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
    dialogue_semantic_query_protocol,
)
from pure_integer_ai.experiments.train_context import TrainContext


def install_runtime_dialogue_semantic_course(ctx: TrainContext):
    """在专用 Runtime 上下文装配 occurrence、Span 和 typed semantic course。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    semantic, occurrence, span = dialogue_semantic_protocols()
    install_language_graph_protocols(
        ctx,
        occurrence_protocol=occurrence,
        span_protocol=span,
    )
    return install_language_semantic_course_runtime(
        ctx,
        semantic,
        dialogue_semantic_query_protocol(),
    )


def install_runtime_dialogue_language_course(
        ctx: TrainContext,
        *,
        course_root: str | Path,
        source_manifest_path: str | Path,
        runtime_language: int = LANG_ZH,
        visible_splits: tuple[int, ...] = (SPLIT_TRAIN,),
        segmentation_protocol: SegmentationProtocol | None = None,
        ) -> LanguageCourseIntakeReport:
    """装配真实 D-01 词形课程，再装配 Runtime typed 语义课程。

    ``course_root`` 与 ``source_manifest_path`` 是调用方提供的只读 source
    connection；本函数不复制课程、不写 Core，也不把 Runtime 资料提升为
    learned fact。课程目录和 manifest 必须位于 K: 训练/运行数据盘，避免把
    大词表或 SQLite 产物落回工程盘。
    """
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    root = Path(course_root).resolve()
    manifest = Path(source_manifest_path).resolve()
    if root.drive.upper() != "K:" or manifest.drive.upper() != "K:":
        raise ValueError("Runtime 词形课程与 source manifest 必须位于 K: 盘")
    providers, report = build_word_form_providers(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        course_root=root,
        source_manifest_path=manifest,
        runtime_language=runtime_language,
        visible_splits=visible_splits,
        segmentation_protocol=segmentation_protocol,
    )
    ctx.word_form_providers = providers
    ctx.word_form_course_report = report
    install_runtime_dialogue_semantic_course(ctx)
    return report


__all__ = [
    "install_runtime_dialogue_language_course",
    "install_runtime_dialogue_semantic_course",
]
