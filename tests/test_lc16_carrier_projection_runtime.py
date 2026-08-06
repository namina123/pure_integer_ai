"""LC-16 九类 carrier-neutral 候选投影运行时测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionMapper,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_catalog import (
    PARENT_PACK_PATH,
    build_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments import (
    ph2_carrier_projection_runtime_catalog as runtime_catalog,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime import (
    CarrierProjectionRuntime,
    CarrierProjectionRuntimeError,
    CarrierProjectionSpec,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_catalog import (
    CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH,
    PARENT_MAPPER_SHA256,
    build_carrier_projection_runtime_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    DEPENDENCY_ROLES,
    EVIDENCE_ROLES,
    EXECUTION_STATE,
    RUNTIME_SCOPE,
    CarrierProjectionRuntimeContractError,
    CarrierProjectionRuntimeManifest,
    read_carrier_projection_runtime_manifest,
    verify_carrier_projection_runtime_files,
    write_carrier_projection_runtime_manifest,
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
     read_reference_link_embed_carrier_records, adapt_reference_link_embed_carrier_record),
    ("SOURCE_CODE", "data/ph2/lc16_source_code_carrier_v1.jsonl.sample",
     read_source_code_carrier_records, adapt_source_code_carrier_record),
    ("TABLE_GRID", "data/ph2/lc16_table_grid_carrier_v1.jsonl.sample",
     read_table_grid_carrier_records, adapt_table_grid_carrier_record),
    ("TRANSCRIBED_OCR_ASR", "data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample",
     read_transcribed_ocr_asr_carrier_records, adapt_transcribed_ocr_asr_carrier_record),
)
_DIRECTIONS = (
    PROJECTION_UNDERSTANDING,
    PROJECTION_REASONING,
    PROJECTION_GENERATION,
)


def _source(source_id: int) -> SourceRef:
    return SourceRef(
        16617900, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


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


def _spec(mapped, *, candidate_key: int, semantic_key: int,
          competition_key: tuple[int, ...] | None = None):
    return CarrierProjectionSpec(
        concept_identity((16617910, candidate_key)),
        ((16617911, candidate_key) if competition_key is None
         else competition_key),
        concept_identity((16617912, 1)),
        concept_identity((16617913, semantic_key)),
        _DIRECTIONS,
        mapped.feature_identities,
        (_source(100 + candidate_key), _source(200 + candidate_key)),
    )


def _mapper_inputs():
    parent = read_typed_carrier_pack_manifest(_ROOT / PARENT_PACK_PATH)
    manifest = build_carrier_projection_mapper_manifest(_ROOT)
    mapper = CarrierProjectionMapper(parent)
    rules = {item.carrier_key: item for item in manifest.rules}
    result = {}
    for index, (carrier_key, path, reader, adapter) in enumerate(
            _CARRIERS, start=1):
        materialization = adapter(reader(_ROOT / path)[0])
        result[carrier_key] = mapper.map(
            carrier_key,
            materialization,
            rules[carrier_key],
            item_indices=(0,),
            input_key=(16617920, index),
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
            input_key=(16617930, index + 1),
        )
        for index in range(3)
    )


def test_one_runtime_projects_all_nine_carriers_without_fake_plain_text_node():
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        for index, (carrier_key, mapped) in enumerate(
                _mapper_inputs().items(), start=1):
            spec = _spec(
                mapped, candidate_key=index, semantic_key=index)
            trace = runtime.learn(
                spec,
                mapped,
                revealed=_reveal(
                    mapped, spec.semantic_object,
                    support=True, source_id=300 + index),
            )
            assert trace.projection is not None
            assert trace.projection.lifecycle_state == (
                runtime.protocol.lifecycle.active_state)
            assert trace.outcome.prediction.visible_inputs == mapped.visible_inputs
            assert set(mapped.feature_identities) <= set(
                trace.outcome.prediction.visible_inputs)
            assert trace.outcome.evidence.evidence_id in {
                item.evidence_id for item in trace.projection.evidence
            }
            assert trace.projection.anchor_identities == mapped.anchor_identities
            assert trace.projection.structure_node_identities == (
                mapped.structure_node_identities)
            if carrier_key == "PLAIN_TEXT":
                assert trace.projection.anchor_identities
                assert trace.projection.structure_node_identities == ()
    finally:
        backend.close()


def test_feature_binding_mismatch_fails_before_any_candidate_write():
    mapped = _mapper_inputs()["HTML"]
    spec = replace(
        _spec(mapped, candidate_key=20, semantic_key=20),
        feature_identities=(concept_identity((16617999, 1)),),
    )
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        before = runtime.learning.state_key()
        with pytest.raises(
                CarrierProjectionRuntimeError, match="不精确对齐"):
            runtime.learn(
                spec,
                mapped,
                revealed=_reveal(
                    mapped, spec.semantic_object,
                    support=True, source_id=400),
            )
        assert runtime.learning.state_key() == before
        assert runtime.learning.report().candidate_count == 0
    finally:
        backend.close()


def test_reveal_route_mismatch_fails_before_registration():
    mapped = _mapper_inputs()["MARKDOWN"]
    spec = _spec(mapped, candidate_key=21, semantic_key=21)
    reveal = replace(
        _reveal(mapped, spec.semantic_object, support=True, source_id=401),
        event_key=(16617998, 1),
    )
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        before = runtime.learning.state_key()
        with pytest.raises(CarrierProjectionRuntimeError, match="精确绑定"):
            runtime.learn(spec, mapped, revealed=reveal)
        assert runtime.learning.state_key() == before
    finally:
        backend.close()


def test_unknown_replacement_fails_before_registration():
    mapped = _mapper_inputs()["HTML"]
    spec = _spec(mapped, candidate_key=22, semantic_key=22)
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        before = runtime.learning.state_key()
        with pytest.raises(KeyError, match="replacement candidate"):
            runtime.learn(
                spec,
                mapped,
                revealed=_reveal(
                    mapped, spec.semantic_object,
                    support=False, source_id=402),
                replacement_candidate=concept_identity((16617999, 2)),
            )
        assert runtime.learning.state_key() == before
        assert runtime.learning.report().candidate_count == 0
    finally:
        backend.close()


def test_refute_correction_supersedes_and_retains_competing_candidate():
    first_input, second_input, correction_input = _html_inputs()
    competition = (16617940, 1)
    first = _spec(
        first_input, candidate_key=30, semantic_key=30,
        competition_key=competition)
    second = _spec(
        second_input, candidate_key=31, semantic_key=31,
        competition_key=competition)
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        runtime.register(first, timestamp_base=1)
        runtime.register(second, timestamp_base=3)
        accepted = runtime.learn(
            first,
            first_input,
            revealed=_reveal(
                first_input, first.semantic_object,
                support=True, source_id=500),
        )
        replacement = runtime.learn(
            second,
            second_input,
            revealed=_reveal(
                second_input, second.semantic_object,
                support=True, source_id=501),
        )
        assert accepted.projection is not None
        assert replacement.projection is not None
        corrected = runtime.learn(
            first,
            correction_input,
            revealed=_reveal(
                correction_input, first.semantic_object,
                support=False, source_id=502),
            replacement_candidate=second.candidate,
        )
        assert corrected.outcome.verification.stance == EVIDENCE_REFUTE
        assert corrected.projection is not None
        assert corrected.projection.lifecycle_state == (
            runtime.protocol.lifecycle.superseded_state)
        assert corrected.outcome.evidence.evidence_id in {
            item.evidence_id for item in corrected.projection.evidence
        }
        assert len(corrected.projection.evidence) > 1
        runtime.graph.ontology.clear_runtime_caches()
        assert runtime.retained_projection(first.candidate).state == (
            runtime.protocol.lifecycle.superseded_state)
        assert runtime.retained_projection(second.candidate).state == (
            runtime.protocol.lifecycle.active_state)
    finally:
        backend.close()


def test_runtime_manifest_freezes_parent_core_and_zero_capability_state():
    manifest = build_carrier_projection_runtime_manifest(_ROOT)
    assert manifest.parent_mapper_sha256 == PARENT_MAPPER_SHA256
    assert tuple(item.role for item in manifest.dependencies) == DEPENDENCY_ROLES
    assert tuple(item.role for item in manifest.evidence_files) == EVIDENCE_ROLES
    assert manifest.runtime_scope.to_value() == RUNTIME_SCOPE
    assert manifest.execution_state.to_value() == EXECUTION_STATE


def test_runtime_manifest_round_trip_and_no_overwrite(tmp_path):
    manifest = build_carrier_projection_runtime_manifest(_ROOT)
    target = tmp_path / "runtime.json"
    assert write_carrier_projection_runtime_manifest(manifest, target) == target
    assert read_carrier_projection_runtime_manifest(target) == manifest
    assert write_carrier_projection_runtime_manifest(manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(CarrierProjectionRuntimeContractError, match="内容不同"):
        write_carrier_projection_runtime_manifest(manifest, target)


def test_stored_runtime_manifest_remains_frozen_when_files_evolve():
    """历史 runtime manifest 固定；当前文件漂移必须 fail closed。"""
    stored = read_carrier_projection_runtime_manifest(
        _ROOT / CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH)
    rebuilt = build_carrier_projection_runtime_manifest(_ROOT)
    assert stored != rebuilt
    with pytest.raises(CarrierProjectionRuntimeContractError, match="身份漂移"):
        verify_carrier_projection_runtime_files(stored, repository_root=_ROOT)


def test_runtime_manifest_fails_closed_on_parent_or_unknown_field(monkeypatch):
    manifest = build_carrier_projection_runtime_manifest(_ROOT)
    monkeypatch.setattr(runtime_catalog, "PARENT_MAPPER_SHA256", "0" * 64)
    with pytest.raises(
            runtime_catalog.CarrierProjectionRuntimeCatalogError,
            match="parent mapper"):
        build_carrier_projection_runtime_manifest(_ROOT)
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(
            CarrierProjectionRuntimeContractError, match="字段不精确"):
        CarrierProjectionRuntimeManifest.from_dict(value)
