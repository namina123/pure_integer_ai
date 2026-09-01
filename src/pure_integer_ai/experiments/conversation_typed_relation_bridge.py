"""把公开 authored relation pack 接入正式对话训练图。\n\n+JSONL 仍只是课程交换格式；本模块在训练启动时把已声明的 W-06 relation
records 适配为共享 H-05/R-00 owner，并将 owner 绑定到 formal_train 当前的
TrainContext。发布运行时不读取这里的课程文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06TypedAdapterOutput,
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_w06_span_graph_protocol import (
    w06_span_anchor_predicate,
    w06_span_endpoint_predicate,
)


def _materialize_relation_spans(context, runtime) -> None:
    """把 relation 来源原文的端点/锚点落入共享 Span 图。

    Authored JSONL 只在训练构造阶段存在；发布图保留来源、区间和角色绑定，
    使后续查询可以从整数图恢复关系而不回读课程表面。
    """
    span_index = getattr(context, "span_index", None)
    if span_index is None:
        raise RuntimeError("W-06 typed relation 必须先安装 occurrence/span 协议")
    ontology = context.graph_ontology
    endpoint_predicate = ontology.materialize(w06_span_endpoint_predicate())
    anchor_predicate = ontology.materialize(w06_span_anchor_predicate())
    for candidate in runtime.registered_candidates():
        source = candidate.source_ref
        scope = document_scope(source)
        raw_text = candidate.surface
        proposition_ref = ontology.materialize(candidate.proposition.proposition)
        bindings = tuple(candidate.proposition.canonical_bindings())
        endpoint_by_identity = {
            endpoint.identity: endpoint for endpoint in candidate.endpoints
        }
        if (len(bindings) != len(candidate.endpoints)
                or len(endpoint_by_identity) != len(candidate.endpoints)
                or {binding.filler for binding in bindings}
                != set(endpoint_by_identity)):
            raise RuntimeError("W-06 endpoint 与 RoleBinding 数量不一致")
        # The relation anchor and each endpoint are independent source spans.
        anchor = candidate.proposition.source_anchor
        source_key = source.stable_key()
        if anchor.components[:len(source_key)] != source_key:
            raise RuntimeError("W-06 anchor 与 relation 来源不一致")
        anchor_payload = anchor.components[len(source_key):]
        if len(anchor_payload) != 3:
            raise RuntimeError("W-06 anchor occurrence identity 布局非法")
        anchor_start, anchor_end, anchor_ordinal = anchor_payload
        anchor_ref = span_index.ensure_ref(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((anchor_start, anchor_end),),
            ordinal=anchor_ordinal,
        )
        ontology.relate(
            anchor_predicate,
            anchor_ref,
            proposition_ref,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
        )
        for binding in bindings:
            endpoint = endpoint_by_identity[binding.filler]
            endpoint_ref = span_index.ensure_ref(
                source=source,
                raw_text=raw_text,
                scope=scope,
                members=((endpoint.start, endpoint.end),),
                ordinal=endpoint.ordinal,
            )
            binding_ref = ontology.materialize(
                binding.identity_for(candidate.proposition.proposition))
            ontology.relate(
                endpoint_predicate,
                endpoint_ref,
                binding_ref,
                scope=scope,
                provenance_kind=source.source_kind,
                content_version=source.versions.parser.value,
            )


_COURSE_BUILDERS = {
    "authored_relation_alias_refers_w06_seed_v2.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_alias_refers_w06_course",
        "compile_authored_alias_refers_w06_course",
    ),
    "authored_relation_subset_member_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_subset_member_course",
        "compile_authored_subset_member_course",
    ),
    "authored_relation_property_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_property_course",
        "compile_authored_property_course",
    ),
    "authored_relation_mereology_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_mereology_course",
        "compile_authored_mereology_course",
    ),
    "authored_relation_similar_antonym_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_semantic_pair_course",
        "compile_authored_semantic_pair_course",
    ),
    "authored_relation_precedes_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_precedes_course",
        "compile_authored_precedes_course",
    ),
    "authored_relation_causes_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_causes_course",
        "compile_authored_causes_course",
    ),
}


def _course_builder(path: Path):
    """按已登记文件名加载对应 authored parser/compiler。"""
    import importlib

    entry = _COURSE_BUILDERS.get(path.name)
    if entry is None:
        return None
    module = importlib.import_module(entry[0])
    return getattr(module, entry[1])


def build_authored_w06_adapter(
        course_paths: Iterable[str | Path],
        pack_root: str | Path,
        ) -> W06TypedAdapterOutput:
    """编译公开 relation samples 并形成统一 W-06 typed adapter output。"""
    paths = tuple(sorted(Path(item).resolve() for item in course_paths))
    selected = tuple(path for path in paths if _course_builder(path) is not None)
    if not selected:
        raise ValueError("未找到已登记 authored relation sample")
    output_root = Path(pack_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_refs = []
    observations = []
    teacher_evidence = []
    for ordinal, path in enumerate(selected, start=1):
        builder = _course_builder(path)
        assert builder is not None
        build = builder(path, output_root / f"pack-{ordinal:02d}")
        manifest = read_artifact_manifest(build.pack_root / "manifest.json")
        for identity in manifest.files:
            records = read_record_artifact(build.pack_root, identity)
            if identity.record_kind == RECORD_SOURCE_REF:
                source_refs.extend(records)
            elif identity.record_kind == RECORD_OBSERVATION:
                observations.extend(
                    item for item in records if item.split == "train")
            elif identity.record_kind == RECORD_TEACHER_EVIDENCE:
                teacher_evidence.extend(records)
    return adapt_w06_training_payload(W06TrainingPayload(
        tuple(source_refs), tuple(observations), tuple(teacher_evidence)))


def build_authored_w06_learning_runtime(
        backend,
        context,
        course_paths: Iterable[str | Path],
        pack_root: str | Path,
        ) -> W06RelationLearningRuntime:
    """在指定正式 TrainContext 上消费 authored relation，返回共享 owner。"""
    adapter = build_authored_w06_adapter(course_paths, pack_root)
    # W-06 is the publication-facing relation slice: retain a physical
    # graph_statement index for independent readers while preserving the
    # assertion record as its canonical source of truth.
    context.graph_ontology.enable_physical_statement_projection()
    runtime = build_w06_learning_runtime(backend, adapter, context=context)
    _materialize_relation_spans(context, runtime)
    return runtime


__all__ = [
    "build_authored_w06_adapter",
    "build_authored_w06_learning_runtime",
]
