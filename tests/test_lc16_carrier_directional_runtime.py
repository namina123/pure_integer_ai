"""LC-16 九载体方向专属 consumer/evaluator 测试。"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
    ArtifactSemanticProjection,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.experiments.ph2_carrier_directional_contract import (
    GENERATION_ADOPT,
    GENERATION_DEFER,
    GENERATION_NEEDS_NARROWING,
    GENERATION_READY,
    GENERATION_REFUSED,
    GENERATION_REJECT,
    OUTCOME_REFUTE,
    OUTCOME_SUPPORT,
    REASONING_ACCEPT,
    REASONING_REFUTED,
    REASONING_REJECT,
    REASONING_SUPPORTED,
    REASONING_UNRESOLVED,
    UNDERSTANDING_ADOPT,
    UNDERSTANDING_REJECT,
    UNDERSTANDING_REJECTED,
    UNDERSTANDING_UNDERSTOOD,
    CarrierDirectionalContractError,
    DirectionalProjectionContext,
    GenerationRequest,
    GenerationUse,
    ReasoningRequest,
    UnderstandingRequest,
    UnderstandingUse,
)
from pure_integer_ai.experiments import (
    ph2_carrier_directional_catalog as directional_catalog,
    ph2_carrier_directional_runtime as directional_runtime,
)
from pure_integer_ai.experiments.ph2_carrier_directional_catalog import (
    CARRIER_DIRECTIONAL_MANIFEST_PATH,
    PARENT_RUNTIME_SHA256,
    build_carrier_directional_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_directional_manifest_contract import (
    DEPENDENCY_ROLES,
    DIRECTIONAL_SCOPE,
    EVIDENCE_ROLES,
    EXECUTION_STATE,
    CarrierDirectionalManifest,
    CarrierDirectionalManifestContractError,
    read_carrier_directional_manifest,
    verify_carrier_directional_files,
    write_carrier_directional_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_directional_evaluator import (
    GenerationEvaluator,
    ReasoningEvaluator,
    UnderstandingEvaluator,
)
from pure_integer_ai.experiments.ph2_carrier_directional_runtime import (
    CarrierDirectionalRuntimeError,
    GenerationConsumer,
    ReasoningConsumer,
    UnderstandingConsumer,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionMapper,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_catalog import (
    PARENT_PACK_PATH,
    build_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime import (
    CarrierProjectionRuntime,
    CarrierProjectionSpec,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_adapter import (
    adapt_document_container_carrier_record,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_contract import (
    read_document_container_carrier_records,
)
from pure_integer_ai.experiments.ph2_html_carrier_adapter import (
    adapt_html_carrier_record,
)
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    read_html_carrier_records,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_adapter import (
    adapt_markdown_carrier_record,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_contract import (
    read_markdown_carrier_records,
)
from pure_integer_ai.experiments.ph2_math_notation_carrier_adapter import (
    adapt_math_notation_carrier_record,
)
from pure_integer_ai.experiments.ph2_math_notation_carrier_contract import (
    read_math_notation_carrier_records,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_adapter import (
    adapt_plain_text_carrier_record,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_contract import (
    read_plain_text_carrier_records,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_adapter import (
    adapt_reference_link_embed_carrier_record,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    read_reference_link_embed_carrier_records,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_adapter import (
    adapt_source_code_carrier_record,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_contract import (
    read_source_code_carrier_records,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_adapter import (
    adapt_table_grid_carrier_record,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_contract import (
    read_table_grid_carrier_records,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_adapter import (
    adapt_transcribed_ocr_asr_carrier_record,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    read_transcribed_ocr_asr_carrier_records,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    read_typed_carrier_pack_manifest,
)
from pure_integer_ai.storage.backend import DictBackend


_ROOT = Path(__file__).resolve().parents[1]
_DIRECTIONS = (
    PROJECTION_UNDERSTANDING,
    PROJECTION_REASONING,
    PROJECTION_GENERATION,
)
_CARRIERS = (
    ("DOCUMENT_CONTAINER", "data/ph2/lc16_document_container_carrier_v1.jsonl.sample",
     read_document_container_carrier_records, adapt_document_container_carrier_record),
    ("HTML", "data/ph2/lc16_html_carrier_v1.jsonl.sample",
     read_html_carrier_records, adapt_html_carrier_record),
    ("MARKDOWN", "data/ph2/lc16_markdown_carrier_v1.jsonl.sample",
     read_markdown_carrier_records, adapt_markdown_carrier_record),
    ("MATH_NOTATION", "data/ph2/lc16_math_notation_carrier_v1.jsonl.sample",
     read_math_notation_carrier_records, adapt_math_notation_carrier_record),
    ("PLAIN_TEXT", "data/ph2/lc16_plain_text_carrier_v1.jsonl.sample",
     read_plain_text_carrier_records, adapt_plain_text_carrier_record),
    ("REFERENCE_LINK_EMBED", "data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample",
     read_reference_link_embed_carrier_records,
     adapt_reference_link_embed_carrier_record),
    ("SOURCE_CODE", "data/ph2/lc16_source_code_carrier_v1.jsonl.sample",
     read_source_code_carrier_records, adapt_source_code_carrier_record),
    ("TABLE_GRID", "data/ph2/lc16_table_grid_carrier_v1.jsonl.sample",
     read_table_grid_carrier_records, adapt_table_grid_carrier_record),
    ("TRANSCRIBED_OCR_ASR", "data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample",
     read_transcribed_ocr_asr_carrier_records,
     adapt_transcribed_ocr_asr_carrier_record),
)


def _source(source_id: int) -> SourceRef:
    return SourceRef(
        16618100, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


def _reveal(mapped, target, *, support: bool, source_id: int):
    return RevealedObjectObservation(
        mapped.source,
        mapped.scope,
        mapped.input_key,
        _source(source_id),
        (target,) if support else (),
        () if support else (target,),
        (source_id, int(support)),
    )


def _spec(
        mapped, *, candidate_key: int, semantic_key: int,
        directions: tuple[int, ...] = _DIRECTIONS,
        competition_key: tuple[int, ...] | None = None,
        ) -> CarrierProjectionSpec:
    return CarrierProjectionSpec(
        concept_identity((16618110, candidate_key)),
        ((16618111, candidate_key) if competition_key is None
         else competition_key),
        concept_identity((16618112, 1)),
        concept_identity((16618113, semantic_key)),
        directions,
        mapped.feature_identities,
        (_source(100 + candidate_key), _source(200 + candidate_key)),
    )


def _mapped_inputs():
    parent = read_typed_carrier_pack_manifest(_ROOT / PARENT_PACK_PATH)
    manifest = build_carrier_projection_mapper_manifest(_ROOT)
    rules = {item.carrier_key: item for item in manifest.rules}
    mapper = CarrierProjectionMapper(parent)
    result = {}
    for index, (carrier_key, path, reader, adapter) in enumerate(
            _CARRIERS, start=1):
        materialization = adapter(reader(_ROOT / path)[0])
        result[carrier_key] = mapper.map(
            carrier_key,
            materialization,
            rules[carrier_key],
            item_indices=(0,),
            input_key=(16618120, index),
        )
    return result


def _html_inputs():
    parent = read_typed_carrier_pack_manifest(_ROOT / PARENT_PACK_PATH)
    manifest = build_carrier_projection_mapper_manifest(_ROOT)
    rule = next(item for item in manifest.rules if item.carrier_key == "HTML")
    records = read_html_carrier_records(
        _ROOT / "data/ph2/lc16_html_carrier_v1.jsonl.sample")
    mapper = CarrierProjectionMapper(parent)
    return tuple(
        mapper.map(
            "HTML",
            adapt_html_carrier_record(records[index]),
            rule,
            item_indices=(0,),
            input_key=(16618130, index + 1),
        )
        for index in range(3)
    )


def _learn_context(runtime, mapped, spec, *, source_id: int, support: bool = True):
    trace = runtime.learn(
        spec,
        mapped,
        revealed=_reveal(
            mapped, spec.semantic_object,
            support=support, source_id=source_id),
    )
    assert trace.projection is not None
    return DirectionalProjectionContext(trace.projection, mapped)


def test_three_directional_chains_execute_all_nine_carriers():
    backend = DictBackend()
    try:
        projection_runtime = CarrierProjectionRuntime(backend)
        for index, (carrier_key, mapped) in enumerate(
                _mapped_inputs().items(), start=1):
            spec = _spec(mapped, candidate_key=index, semantic_key=index)
            context = _learn_context(
                projection_runtime, mapped, spec, source_id=300 + index)

            understanding = UnderstandingConsumer()
            u_result = understanding.consume(
                UnderstandingRequest(context, (16618140, index, 1)))
            assert u_result.status == UNDERSTANDING_UNDERSTOOD
            assert u_result.semantic_object == spec.semantic_object
            assert u_result.subject_identities == context.subject_identities
            assert u_result.feature_identities == mapped.feature_identities
            u_use = understanding.use(u_result, UNDERSTANDING_ADOPT)
            assert UnderstandingEvaluator().evaluate(
                u_use, context).verdict == OUTCOME_SUPPORT

            reasoning = ReasoningConsumer()
            r_result = reasoning.consume(ReasoningRequest(
                context, spec.semantic_object, (16618140, index, 2)))
            assert r_result.conclusion == REASONING_SUPPORTED
            assert spec.semantic_object in r_result.premise_identities
            r_use = reasoning.use(r_result, REASONING_ACCEPT)
            assert ReasoningEvaluator().evaluate(
                r_use, context).verdict == OUTCOME_SUPPORT

            generation = GenerationConsumer()
            g_result = generation.consume(GenerationRequest(
                context,
                spec.semantic_object,
                len(mapped.envelope.raw_units) + 1,
                (16618140, index, 3),
            ))
            assert g_result.status == GENERATION_READY, carrier_key
            assert g_result.surface_units == mapped.envelope.raw_units
            assert g_result.surface_units
            g_use = generation.use(g_result, GENERATION_ADOPT)
            assert GenerationEvaluator().evaluate(
                g_use, context).verdict == OUTCOME_SUPPORT
    finally:
        backend.close()


def test_direction_ablation_fails_before_consumer_execution():
    mapped = _mapped_inputs()["MARKDOWN"]
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        spec = _spec(
            mapped,
            candidate_key=20,
            semantic_key=20,
            directions=(PROJECTION_UNDERSTANDING,),
        )
        context = _learn_context(runtime, mapped, spec, source_id=400)
        UnderstandingRequest(context, (16618141, 1))
        with pytest.raises(
                CarrierDirectionalContractError, match="未授权当前方向"):
            ReasoningRequest(context, spec.semantic_object, (16618141, 2))
        with pytest.raises(
                CarrierDirectionalContractError, match="未授权当前方向"):
            GenerationRequest(
                context, spec.semantic_object, 1000, (16618141, 3))
    finally:
        backend.close()


def test_reasoning_unknown_claim_and_generation_budget_are_not_passed_through():
    mapped = _mapped_inputs()["SOURCE_CODE"]
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        spec = _spec(mapped, candidate_key=21, semantic_key=21)
        context = _learn_context(runtime, mapped, spec, source_id=401)

        reasoning = ReasoningConsumer()
        unknown = reasoning.consume(ReasoningRequest(
            context,
            concept_identity((16618199, 1)),
            (16618142, 1),
        ))
        assert unknown.conclusion == REASONING_UNRESOLVED

        generation = GenerationConsumer()
        narrow = generation.consume(GenerationRequest(
            context, spec.semantic_object, 1, (16618142, 2)))
        assert narrow.status == GENERATION_NEEDS_NARROWING
        assert narrow.surface_units == ()
        use = generation.use(narrow, GENERATION_DEFER)
        assert GenerationEvaluator().evaluate(
            use, context).verdict == OUTCOME_SUPPORT
    finally:
        backend.close()


def test_superseded_refute_is_consumed_and_retained_without_false_adoption():
    first_input, second_input, correction_input = _html_inputs()
    competition = (16618150, 1)
    first = _spec(
        first_input,
        candidate_key=30,
        semantic_key=30,
        competition_key=competition,
    )
    second = _spec(
        second_input,
        candidate_key=31,
        semantic_key=31,
        competition_key=competition,
    )
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        runtime.register(first, timestamp_base=1)
        runtime.register(second, timestamp_base=3)
        runtime.learn(
            first,
            first_input,
            revealed=_reveal(
                first_input, first.semantic_object,
                support=True, source_id=500),
        )
        runtime.learn(
            second,
            second_input,
            revealed=_reveal(
                second_input, second.semantic_object,
                support=True, source_id=501),
        )
        corrected = runtime.learn(
            first,
            correction_input,
            revealed=_reveal(
                correction_input, first.semantic_object,
                support=False, source_id=502),
            replacement_candidate=second.candidate,
        )
        assert corrected.projection is not None
        runtime.graph.ontology.clear_runtime_caches()
        retained = runtime.retained_projection(first.candidate)
        assert retained.state == runtime.protocol.lifecycle.superseded_state
        projection = ArtifactSemanticProjection.from_stable_key(
            corrected.projection.stable_key())
        context = DirectionalProjectionContext(projection, correction_input)

        understanding = UnderstandingConsumer()
        u_result = understanding.consume(
            UnderstandingRequest(context, (16618143, 1)))
        assert u_result.status == UNDERSTANDING_REJECTED
        rejected = understanding.use(u_result, UNDERSTANDING_REJECT)
        assert UnderstandingEvaluator().evaluate(
            rejected, context).verdict == OUTCOME_SUPPORT
        adopted = UnderstandingUse(
            u_result, UNDERSTANDING_ADOPT, (16618144, 1))
        assert UnderstandingEvaluator().evaluate(
            adopted, context).verdict == OUTCOME_REFUTE

        reasoning = ReasoningConsumer()
        r_result = reasoning.consume(ReasoningRequest(
            context, first.semantic_object, (16618143, 2)))
        assert r_result.conclusion == REASONING_REFUTED
        r_use = reasoning.use(r_result, REASONING_REJECT)
        assert ReasoningEvaluator().evaluate(
            r_use, context).verdict == OUTCOME_SUPPORT

        generation = GenerationConsumer()
        g_result = generation.consume(GenerationRequest(
            context, first.semantic_object, 10000, (16618143, 3)))
        assert g_result.status == GENERATION_REFUSED
        assert g_result.surface_units == ()
        g_use = generation.use(g_result, GENERATION_REJECT)
        assert GenerationEvaluator().evaluate(
            g_use, context).verdict == OUTCOME_SUPPORT
    finally:
        backend.close()


def test_generation_request_has_no_surface_template_or_label_channel():
    assert tuple(item.name for item in fields(GenerationRequest)) == (
        "context",
        "goal",
        "max_surface_units",
        "request_key",
    )
    forbidden = {
        "expected_surface",
        "answer_template",
        "complete_answer",
        "evaluator_label",
        "expected_verdict",
    }
    assert forbidden.isdisjoint(item.name for item in fields(GenerationRequest))


def test_generation_evaluator_rejects_surface_tampering():
    mapped = _mapped_inputs()["PLAIN_TEXT"]
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        spec = _spec(mapped, candidate_key=40, semantic_key=40)
        context = _learn_context(runtime, mapped, spec, source_id=600)
        consumer = GenerationConsumer()
        result = consumer.consume(GenerationRequest(
            context, spec.semantic_object, 10000, (16618145, 1)))
        tampered = replace(
            result,
            surface_units=(*result.surface_units[:-1], result.surface_units[-1] + 1),
        )
        use = GenerationUse(tampered, GENERATION_ADOPT, (16618145, 2))
        assert GenerationEvaluator().evaluate(
            use, context).verdict == OUTCOME_REFUTE
    finally:
        backend.close()


def test_evaluators_do_not_reuse_consumer_result_builders(monkeypatch):
    mapped = _mapped_inputs()["TABLE_GRID"]
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        spec = _spec(mapped, candidate_key=41, semantic_key=41)
        context = _learn_context(runtime, mapped, spec, source_id=601)

        understanding = UnderstandingConsumer()
        u_result = understanding.consume(
            UnderstandingRequest(context, (16618146, 1)))
        u_use = understanding.use(u_result, UNDERSTANDING_ADOPT)
        reasoning = ReasoningConsumer()
        r_result = reasoning.consume(ReasoningRequest(
            context, spec.semantic_object, (16618146, 2)))
        r_use = reasoning.use(r_result, REASONING_ACCEPT)
        generation = GenerationConsumer()
        g_result = generation.consume(GenerationRequest(
            context, spec.semantic_object, 10000, (16618146, 3)))
        g_use = generation.use(g_result, GENERATION_ADOPT)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("evaluator 不得调用 consumer result builder")

        monkeypatch.setattr(directional_runtime, "_understanding_result", forbidden)
        monkeypatch.setattr(directional_runtime, "_reasoning_result", forbidden)
        monkeypatch.setattr(directional_runtime, "_generation_result", forbidden)
        assert UnderstandingEvaluator().evaluate(
            u_use, context).verdict == OUTCOME_SUPPORT
        assert ReasoningEvaluator().evaluate(
            r_use, context).verdict == OUTCOME_SUPPORT
        assert GenerationEvaluator().evaluate(
            g_use, context).verdict == OUTCOME_SUPPORT
    finally:
        backend.close()


def test_each_direction_rejects_duplicate_request_and_result_use():
    mapped = _mapped_inputs()["DOCUMENT_CONTAINER"]
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        spec = _spec(mapped, candidate_key=42, semantic_key=42)
        context = _learn_context(runtime, mapped, spec, source_id=602)

        understanding = UnderstandingConsumer()
        u_request = UnderstandingRequest(context, (16618147, 1))
        u_result = understanding.consume(u_request)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复"):
            understanding.consume(u_request)
        understanding.use(u_result, UNDERSTANDING_ADOPT)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复 Use"):
            understanding.use(u_result, UNDERSTANDING_ADOPT)

        reasoning = ReasoningConsumer()
        r_request = ReasoningRequest(
            context, spec.semantic_object, (16618147, 2))
        r_result = reasoning.consume(r_request)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复"):
            reasoning.consume(r_request)
        reasoning.use(r_result, REASONING_ACCEPT)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复 Use"):
            reasoning.use(r_result, REASONING_ACCEPT)

        generation = GenerationConsumer()
        g_request = GenerationRequest(
            context, spec.semantic_object, 10000, (16618147, 3))
        g_result = generation.consume(g_request)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复"):
            generation.consume(g_request)
        generation.use(g_result, GENERATION_ADOPT)
        with pytest.raises(CarrierDirectionalRuntimeError, match="重复 Use"):
            generation.use(g_result, GENERATION_ADOPT)
    finally:
        backend.close()


def test_directional_manifest_freezes_parent_scope_and_zero_capability_state():
    manifest = read_carrier_directional_manifest(
        _ROOT / CARRIER_DIRECTIONAL_MANIFEST_PATH)
    assert manifest.parent_runtime_sha256 == PARENT_RUNTIME_SHA256
    assert tuple(item.role for item in manifest.dependencies) == DEPENDENCY_ROLES
    assert tuple(item.role for item in manifest.evidence_files) == EVIDENCE_ROLES
    assert manifest.directional_scope.to_value() == DIRECTIONAL_SCOPE
    assert manifest.execution_state.to_value() == EXECUTION_STATE


def test_directional_manifest_round_trip_and_no_overwrite(tmp_path):
    manifest = read_carrier_directional_manifest(
        _ROOT / CARRIER_DIRECTIONAL_MANIFEST_PATH)
    target = tmp_path / "directional.json"
    assert write_carrier_directional_manifest(manifest, target) == target
    assert read_carrier_directional_manifest(target) == manifest
    assert write_carrier_directional_manifest(manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(
            CarrierDirectionalManifestContractError, match="内容不同"):
        write_carrier_directional_manifest(manifest, target)


def test_stored_directional_manifest_remains_frozen_when_parent_evolves():
    """历史 directional manifest 固定；parent 漂移必须阻断重建与回验。"""
    stored = read_carrier_directional_manifest(
        _ROOT / CARRIER_DIRECTIONAL_MANIFEST_PATH)
    with pytest.raises(directional_catalog.CarrierDirectionalCatalogError,
                       match="无法严格回验"):
        build_carrier_directional_manifest(_ROOT)
    with pytest.raises(CarrierDirectionalManifestContractError, match="身份漂移"):
        verify_carrier_directional_files(stored, repository_root=_ROOT)


def test_directional_manifest_fails_closed_on_parent_or_unknown_field(
        monkeypatch,
        ):
    manifest = read_carrier_directional_manifest(
        _ROOT / CARRIER_DIRECTIONAL_MANIFEST_PATH)
    monkeypatch.setattr(directional_catalog, "PARENT_RUNTIME_SHA256", "0" * 64)
    with pytest.raises(
            directional_catalog.CarrierDirectionalCatalogError,
            match="parent runtime"):
        build_carrier_directional_manifest(_ROOT)
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(
            CarrierDirectionalManifestContractError, match="字段不精确"):
        CarrierDirectionalManifest.from_dict(value)
