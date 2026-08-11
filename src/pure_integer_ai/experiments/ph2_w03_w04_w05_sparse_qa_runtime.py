"""Build once and repeatedly query the public FT22 sparse QA runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    QUESTION_ALIAS_FRAME_ANCHOR_SHA256,
    build_raw_question_alias_frame_anchor_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    QUESTION_CONSTRUCTION_INDEX_SHA256,
    build_raw_question_construction_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    raw_question_feature_catalog,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_composition import (
    build_three_role_question_feature_composition,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    QUESTION_FEATURE_INDEX_SHA256,
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    QUESTION_FEATURE_REGISTRY_SHA256,
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer import (
    public_w03_w04_w05_state_sha256,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch import (
    QUESTION_SPARSE_DISPATCH_SHA256,
    build_raw_question_sparse_dispatch_index,
    project_sparse_question_dispatch_audit,
    run_sparse_question_dispatch,
    sparse_question_dispatch_probe,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch_contract import (
    RawQuestionSparseDispatchIndex,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    build_learned_predicate_alias_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionAnswerResult,
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    build_implicit_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    build_three_role_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SPARSE_QA_RUNTIME_EXPRESSION_BOUNDARY,
    SPARSE_QA_RUNTIME_SHA256,
    SparseQAFrozenIdentities,
    SparseQAEntryPublicStateMemo,
    SparseQAQueryBatch,
    SparseQAQueryProbe,
    SparseQAResult,
    SparseQARuntime,
    SparseQARuntimeBuildProbe,
    W03W04W05SparseQARuntimeError,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    build_w03_w04_w05_vertical_generalization_overlay,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    build_w03_w04_w05_three_role_vertical_overlay,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PUBLIC_DATA = REPOSITORY / "data/ph2"


def _compile_overlay(
        root: Path,
        semantic_sample: str,
        primitive_map_sample: str,
        atomic_sample: str,
        builder,
        ):
    base = compile_authored_semantic_primitive_bridge_course(
        PUBLIC_DATA / semantic_sample,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        PUBLIC_DATA / primitive_map_sample,
        PUBLIC_DATA / atomic_sample,
        root / "donor",
    )
    return builder(base, donor)


def _build_public_registry(work_root: Path):
    two_overlay = _compile_overlay(
        work_root / "two_role",
        "authored_semantic_primitive_bridge_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_generalization_v1.jsonl.sample",
        build_w03_w04_w05_vertical_generalization_overlay,
    )
    two_explicit = build_raw_question_generalization(
        two_overlay,
        PUBLIC_DATA / (
            "authored_vertical_question_cause_generalization_v1.jsonl.sample"),
        PUBLIC_DATA / (
            "authored_vertical_question_effect_generalization_v1.jsonl.sample"),
    )
    two_catalog = raw_question_feature_catalog(two_explicit)
    two_entry = RawQuestionFeatureRegistryEntry(
        two_catalog,
        build_learned_predicate_alias_bridge(two_catalog),
        build_implicit_question_bundle(
            two_catalog,
            PUBLIC_DATA / (
                "authored_vertical_question_implicit_reason_v1.jsonl.sample"),
            PUBLIC_DATA / (
                "authored_vertical_question_implicit_result_v1.jsonl.sample"),
        ),
    )

    three_overlay = _compile_overlay(
        work_root / "three_role",
        "authored_semantic_primitive_bridge_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample",
        build_w03_w04_w05_three_role_vertical_overlay,
    )
    three_explicit = build_three_role_question_bundle(
        three_overlay,
        PUBLIC_DATA / (
            "authored_vertical_question_three_role_actor_v1.jsonl.sample"),
        PUBLIC_DATA / (
            "authored_vertical_question_three_role_location_v1.jsonl.sample"),
    )
    three_composition = build_three_role_question_feature_composition(
        three_explicit,
        PUBLIC_DATA / (
            "authored_vertical_question_three_role_implicit_actor_v1.jsonl.sample"),
        PUBLIC_DATA / (
            "authored_vertical_question_three_role_implicit_location_v1.jsonl.sample"),
    )
    three_entry = RawQuestionFeatureRegistryEntry(
        three_composition.feature_catalog,
        three_composition.alias_bridge,
        three_composition.implicit_bundle,
    )
    return build_raw_question_feature_registry(
        (two_entry, three_entry),
        expected_identity_sha256=QUESTION_FEATURE_REGISTRY_SHA256,
    )


def assemble_sparse_qa_runtime(
        dispatch_index: RawQuestionSparseDispatchIndex,
        *,
        expected_identity_sha256: str | None = None,
        ) -> SparseQARuntime:
    """Wrap one immutable FT20 index without rebuilding any learned layer."""
    if not isinstance(dispatch_index, RawQuestionSparseDispatchIndex):
        raise TypeError("FT22 assembly input is invalid")
    anchor = dispatch_index.anchor_index
    construction = anchor.construction_index
    feature = construction.feature_index
    registry = feature.registry
    identities = SparseQAFrozenIdentities(
        registry.identity_sha256,
        feature.identity_sha256,
        construction.identity_sha256,
        anchor.identity_sha256,
        dispatch_index.identity_sha256,
    )
    probe = SparseQARuntimeBuildProbe(
        1,
        1,
        1,
        1,
        1,
        1,
        len(registry.entries),
        sum(len(item.feature_catalog.catalog) for item in registry.entries),
        sum(len(item.implicit_bundle.catalog) for item in registry.entries),
        sum(len(item.alias_bridge.routes) for item in registry.entries),
        len(construction.exact_rows),
        len(construction.alias_rows),
        len(construction.alias_frame_rows),
        len(construction.implicit_rows),
        len(dispatch_index.entries),
    )
    memo = tuple(sorted(
        (
            SparseQAEntryPublicStateMemo(
                entry.sha256(),
                public_w03_w04_w05_state_sha256(
                    entry.feature_catalog.w03_batch,
                    entry.feature_catalog.w04_batch,
                    entry.feature_catalog.w05_batch,
                ),
            )
            for entry in registry.entries
        ),
        key=lambda item: item.entry_sha256,
    ))
    payload = {
        "build_probe": probe.to_dict(),
        "experimental": 1,
        "formal_mastery_claim": 0,
        "frozen_identities": identities.to_dict(),
        "runtime_boundary": [
            {"capability": key, "status": status}
            for key, status in SPARSE_QA_RUNTIME_EXPRESSION_BOUNDARY
        ],
        "w03_started": 0,
        "w04_started": 0,
        "w05_started": 0,
    }
    identity = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    value = SparseQARuntime(
        dispatch_index,
        identities,
        probe,
        identity,
        memo,
    )
    if (expected_identity_sha256 is not None
            and value.identity_sha256 != expected_identity_sha256):
        raise W03W04W05SparseQARuntimeError(
            "FT22 runtime commitment drifted")
    return value


def _build_public_sparse_qa_runtime(work_root: Path) -> SparseQARuntime:
    registry = _build_public_registry(work_root)
    feature = build_raw_question_feature_index(
        registry,
        expected_identity_sha256=QUESTION_FEATURE_INDEX_SHA256,
    )
    construction = build_raw_question_construction_index(
        feature,
        expected_identity_sha256=QUESTION_CONSTRUCTION_INDEX_SHA256,
    )
    anchor = build_raw_question_alias_frame_anchor_index(
        construction,
        expected_identity_sha256=QUESTION_ALIAS_FRAME_ANCHOR_SHA256,
    )
    dispatch = build_raw_question_sparse_dispatch_index(
        anchor,
        expected_identity_sha256=QUESTION_SPARSE_DISPATCH_SHA256,
    )
    return assemble_sparse_qa_runtime(
        dispatch,
        expected_identity_sha256=SPARSE_QA_RUNTIME_SHA256,
    )


def build_public_sparse_qa_runtime(
        work_root: str | Path | None = None,
        ) -> SparseQARuntime:
    """Build the frozen public runtime once; temporary artifacts stay outside Git."""
    if work_root is None:
        with TemporaryDirectory(prefix="pure_integer_ai_ft22_") as temporary:
            return _build_public_sparse_qa_runtime(Path(temporary))
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    return _build_public_sparse_qa_runtime(root)


def _selected_raw_result(record) -> RawQuestionAnswerResult | None:
    decision = record.decision
    if decision.status != "ANSWER" or decision.selected_entry_sha256 is None:
        return None
    trace = next(
        item for item in record.traces
        if item.entry_sha256 == decision.selected_entry_sha256
    )
    if decision.decisive_phase == "EXACT":
        return trace.exact_result
    if decision.decisive_phase == "ALIAS" and trace.alias_result is not None:
        return trace.alias_result.normalized_result
    if (decision.decisive_phase == "IMPLICIT"
            and trace.implicit_result is not None):
        return trace.implicit_result.implicit_result
    raise W03W04W05SparseQARuntimeError(
        "FT22 selected sparse decision lacks its typed result")


def run_sparse_qa_query(
        runtime: SparseQARuntime,
        request: RawQuestionRequest,
        *,
        audit: bool = False,
        ) -> SparseQAResult:
    """Run one FT20 hot query and project complete FT16 traces only on request."""
    if (not isinstance(runtime, SparseQARuntime)
            or not isinstance(request, RawQuestionRequest)
            or type(audit) is not bool):
        raise TypeError("FT22 sparse QA query inputs are invalid")
    record = run_sparse_question_dispatch(
        runtime.dispatch_index,
        request,
        public_state_sha256s=tuple(
            (item.entry_sha256, item.public_state_sha256)
            for item in runtime.entry_public_state_memo
        ),
    )
    raw_result = _selected_raw_result(record)
    source = None
    if raw_result is not None:
        if raw_result.selected_construction is None:
            raise W03W04W05SparseQARuntimeError(
                "FT22 ANSWER lacks a selected learned construction")
        source = raw_result.selected_construction.source_record_key
    audit_result = (
        project_sparse_question_dispatch_audit(runtime.dispatch_index, record)
        if audit else None
    )
    decision = record.decision
    return SparseQAResult(
        runtime.identity_sha256,
        runtime.frozen_identities,
        request,
        decision.status,
        decision.answer_surface,
        decision.decisive_phase,
        decision.selected_entry_sha256,
        source,
        record.sha256(),
        sparse_question_dispatch_probe(runtime.dispatch_index, record),
        audit_result,
    )


def run_sparse_qa_queries(
        runtime: SparseQARuntime,
        requests: tuple[RawQuestionRequest, ...],
        *,
        audit: bool = False,
        ) -> SparseQAQueryBatch:
    """Reuse one runtime for an ordered batch and prove deterministic counts."""
    if (not isinstance(runtime, SparseQARuntime)
            or not isinstance(requests, tuple) or not requests
            or any(not isinstance(item, RawQuestionRequest)
                   for item in requests)
            or type(audit) is not bool):
        raise TypeError("FT22 sparse QA batch inputs are invalid")
    results = tuple(
        run_sparse_qa_query(runtime, request, audit=audit)
        for request in requests
    )
    result_sha256s = tuple(item.sha256() for item in results)
    probe = SparseQAQueryProbe(
        runtime.build_probe.runtime_build_count,
        len(results),
        len(results),
        len(results),
        sum(item.audit_result is not None for item in results),
        sum(item.dispatch_probe.sparse_trace_count for item in results),
        sum(
            0 if item.audit_result is None else len(item.audit_result.traces)
            for item in results
        ),
        len({item.request.sha256() for item in results}),
        len(set(result_sha256s)),
        result_sha256s,
    )
    return SparseQAQueryBatch(results, probe)


__all__ = [
    "PUBLIC_DATA",
    "REPOSITORY",
    "SPARSE_QA_RUNTIME_SHA256",
    "SparseQAFrozenIdentities",
    "SparseQAEntryPublicStateMemo",
    "SparseQAQueryBatch",
    "SparseQAQueryProbe",
    "SparseQAResult",
    "SparseQARuntime",
    "SparseQARuntimeBuildProbe",
    "W03W04W05SparseQARuntimeError",
    "assemble_sparse_qa_runtime",
    "build_public_sparse_qa_runtime",
    "run_sparse_qa_queries",
    "run_sparse_qa_query",
]
