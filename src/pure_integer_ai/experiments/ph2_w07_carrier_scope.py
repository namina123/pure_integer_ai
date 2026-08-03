"""W-07 typed logic 在 LC16 九载体上的共享 scope 投影。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
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
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
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
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_LC16_CARRIER_KEYS,
    W05_LC16_DIRECTIONS,
    W05_LC16_SCOPE_KEY,
    digest_value,
)
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER


_NAMESPACE = 70740
_DIRECTION_VALUES = {
    "GENERATION": PROJECTION_GENERATION,
    "REASONING": PROJECTION_REASONING,
    "UNDERSTANDING": PROJECTION_UNDERSTANDING,
}
_CARRIER_SPECS: tuple[
    tuple[str, str, Callable[[Path], Any], Callable[[Any], Any]], ...
] = (
    (
        "DOCUMENT_CONTAINER",
        "data/ph2/lc16_document_container_carrier_v1.jsonl.sample",
        read_document_container_carrier_records,
        adapt_document_container_carrier_record,
    ),
    (
        "HTML",
        "data/ph2/lc16_html_carrier_v1.jsonl.sample",
        read_html_carrier_records,
        adapt_html_carrier_record,
    ),
    (
        "MARKDOWN",
        "data/ph2/lc16_markdown_carrier_v1.jsonl.sample",
        read_markdown_carrier_records,
        adapt_markdown_carrier_record,
    ),
    (
        "MATH_NOTATION",
        "data/ph2/lc16_math_notation_carrier_v1.jsonl.sample",
        read_math_notation_carrier_records,
        adapt_math_notation_carrier_record,
    ),
    (
        "PLAIN_TEXT",
        "data/ph2/lc16_plain_text_carrier_v1.jsonl.sample",
        read_plain_text_carrier_records,
        adapt_plain_text_carrier_record,
    ),
    (
        "REFERENCE_LINK_EMBED",
        "data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample",
        read_reference_link_embed_carrier_records,
        adapt_reference_link_embed_carrier_record,
    ),
    (
        "SOURCE_CODE",
        "data/ph2/lc16_source_code_carrier_v1.jsonl.sample",
        read_source_code_carrier_records,
        adapt_source_code_carrier_record,
    ),
    (
        "TABLE_GRID",
        "data/ph2/lc16_table_grid_carrier_v1.jsonl.sample",
        read_table_grid_carrier_records,
        adapt_table_grid_carrier_record,
    ),
    (
        "TRANSCRIBED_OCR_ASR",
        "data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample",
        read_transcribed_ocr_asr_carrier_records,
        adapt_transcribed_ocr_asr_carrier_record,
    ),
)


class W07CarrierScopeError(RuntimeError):
    """九载体、七逻辑或 U/R/G scope 投影未闭合。"""


@dataclass(frozen=True)
class W07LogicDirectionArtifact:
    direction: str
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        if self.direction not in W05_LC16_DIRECTIONS:
            raise W07CarrierScopeError("logic direction 未注册")
        if any(not isinstance(value, tuple) or not value
               or any(type(item) is not int for item in value)
               for value in (self.use_key, self.outcome_key)):
            raise W07CarrierScopeError("logic Use/outcome key 非法")
        if self.status != "SUPPORT":
            raise W07CarrierScopeError("logic direction outcome 未通过")

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "outcome_key": list(self.outcome_key),
            "status": self.status,
            "use_key": list(self.use_key),
        }


@dataclass(frozen=True)
class W07LogicProjectionTarget:
    substage: str
    proposition: ObjectIdentity
    structure_tree_key: tuple[int, ...]
    role_tree_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    evidence_source: SourceRef
    state_key: tuple[int, int]
    direction_artifacts: tuple[W07LogicDirectionArtifact, ...]

    def __post_init__(self) -> None:
        if self.substage not in W07_SUBSTAGE_ORDER:
            raise W07CarrierScopeError("logic substage 未注册")
        if not isinstance(self.proposition, ObjectIdentity):
            raise W07CarrierScopeError("logic target proposition 非法")
        if not isinstance(self.source, SourceRef) or not isinstance(
                self.evidence_source, SourceRef):
            raise W07CarrierScopeError("logic target source 非法")
        if self.source == self.evidence_source:
            raise W07CarrierScopeError("logic forming source 不独立")
        if not isinstance(self.scope, ScopeIdentity):
            raise W07CarrierScopeError("logic target scope 非法")
        if self.state_key not in {(1, 0), (0, 1), (0, 0), (1, 1)}:
            raise W07CarrierScopeError("logic target 四态非法")
        if tuple(item.direction for item in self.direction_artifacts) != (
                W05_LC16_DIRECTIONS):
            raise W07CarrierScopeError("logic target U/R/G 不完整")

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_artifacts": [
                item.to_dict() for item in self.direction_artifacts],
            "evidence_source": list(self.evidence_source.stable_key()),
            "proposition": list(self.proposition.stable_key()),
            "role_tree_key": list(self.role_tree_key),
            "scope": list(self.scope.stable_key()),
            "source": list(self.source.stable_key()),
            "state_key": list(self.state_key),
            "structure_tree_key": list(self.structure_tree_key),
            "substage": self.substage,
        }


@dataclass(frozen=True)
class W07CarrierLogicProjection:
    substage: str
    carrier_candidate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    projection_key: tuple[int, ...]
    retained_projection_equal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_candidate_key": list(self.carrier_candidate_key),
            "projection_key": list(self.projection_key),
            "proposition_key": list(self.proposition_key),
            "retained_projection_equal": int(self.retained_projection_equal),
            "substage": self.substage,
        }


@dataclass(frozen=True)
class W07CarrierLogicCell:
    carrier_key: str
    substage: str
    direction: str
    scope_key: str
    projection_direction: int
    use_commitment: tuple[int, ...]
    outcome_commitment: tuple[int, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "direction": self.direction,
            "outcome_commitment": list(self.outcome_commitment),
            "projection_direction": self.projection_direction,
            "scope_key": self.scope_key,
            "status": self.status,
            "substage": self.substage,
            "use_commitment": list(self.use_commitment),
        }


@dataclass(frozen=True)
class W07CarrierScopeRecord:
    carrier_key: str
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    envelope_key: tuple[int, ...]
    feature_keys: tuple[tuple[int, ...], ...]
    projections: tuple[W07CarrierLogicProjection, ...]
    logic_cells: tuple[W07CarrierLogicCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "envelope_key": list(self.envelope_key),
            "feature_keys": [list(item) for item in self.feature_keys],
            "logic_cells": [item.to_dict() for item in self.logic_cells],
            "projections": [item.to_dict() for item in self.projections],
            "scope_key": list(self.scope_key),
            "source_key": list(self.source_key),
        }


@dataclass(frozen=True)
class W07CarrierScopeReport:
    records: tuple[W07CarrierScopeRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        if tuple(item.carrier_key for item in self.records) != (
                W05_LC16_CARRIER_KEYS):
            raise W07CarrierScopeError("九载体缺失或顺序漂移")
        projections = tuple(
            item for record in self.records for item in record.projections)
        cells = tuple(item for record in self.records for item in record.logic_cells)
        expected_projections = {
            (carrier, substage)
            for carrier in W05_LC16_CARRIER_KEYS
            for substage in W07_SUBSTAGE_ORDER
        }
        expected_cells = {
            (carrier, substage, direction)
            for carrier in W05_LC16_CARRIER_KEYS
            for substage in W07_SUBSTAGE_ORDER
            for direction in W05_LC16_DIRECTIONS
        }
        if ({(record.carrier_key, item.substage)
             for record in self.records for item in record.projections}
                != expected_projections
                or len(projections) != len(expected_projections)
                or not all(item.retained_projection_equal
                           for item in projections)):
            raise W07CarrierScopeError("carrier logic projection 覆盖不完整")
        if ({(item.carrier_key, item.substage, item.direction)
             for item in cells} != expected_cells
                or len(cells) != len(expected_cells)
                or any(item.status != "SUPPORT" for item in cells)):
            raise W07CarrierScopeError("carrier logic U/R/G cell 覆盖不完整")
        actual = hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()
        if self.digest != actual:
            raise W07CarrierScopeError("carrier scope report digest 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_count": len(self.records),
            "carrier_projection_count": len(self.records),
            "logic_cell_count": sum(
                len(item.logic_cells) for item in self.records),
            "logic_projection_binding_count": sum(
                len(item.projections) for item in self.records),
            "records": [item.to_dict() for item in self.records],
            "scope_key": W05_LC16_SCOPE_KEY,
        }


def _public_file(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W07CarrierScopeError(f"公开 LC16 文件缺失: {relative}")
    return target


def build_w07_carrier_scope_report(
        repository_root: str | Path,
        backend,
        targets: tuple[W07LogicProjectionTarget, ...],
        ) -> W07CarrierScopeReport:
    """用同一 carrier runtime 投影七类 typed logic，不建立九套解释器。"""
    if tuple(item.substage for item in targets) != W07_SUBSTAGE_ORDER:
        raise W07CarrierScopeError("七逻辑 target 顺序或覆盖漂移")
    root = Path(repository_root).resolve()
    parent = read_typed_carrier_pack_manifest(_public_file(root, PARENT_PACK_PATH))
    mapper_manifest = build_carrier_projection_mapper_manifest(root)
    rules = {item.carrier_key: item for item in mapper_manifest.rules}
    if tuple(item[0] for item in _CARRIER_SPECS) != W05_LC16_CARRIER_KEYS:
        raise W07CarrierScopeError("carrier adapter registry 顺序漂移")
    mapper = CarrierProjectionMapper(parent)
    runtime = CarrierProjectionRuntime(backend)
    projection_kind = concept_identity((_NAMESPACE, 100))
    aggregate_target = concept_identity((
        _NAMESPACE, 101, *digest_value([
            item.to_dict() for item in targets])))
    forming_sources = (targets[0].source, targets[0].evidence_source)
    if len(set(forming_sources)) != 2:
        raise W07CarrierScopeError("logic bundle forming source 不独立")
    directions = tuple(sorted(_DIRECTION_VALUES.values()))
    pending = []
    before_cache = {}
    for carrier_ordinal, (carrier_key, relative, reader, adapter) in enumerate(
            _CARRIER_SPECS, start=1):
        records = reader(_public_file(root, relative))
        if not records:
            raise W07CarrierScopeError(f"{carrier_key} sample 为空")
        mapped = mapper.map(
            carrier_key,
            adapter(records[0]),
            rules[carrier_key],
            item_indices=(0,),
            input_key=(_NAMESPACE, 200, carrier_ordinal),
        )
        carrier_candidate = concept_identity((
            _NAMESPACE, 300, carrier_ordinal))
        spec = CarrierProjectionSpec(
            carrier_candidate,
            (_NAMESPACE, 400, carrier_ordinal),
            projection_kind,
            aggregate_target,
            directions,
            mapped.feature_identities,
            forming_sources,
        )
        trace = runtime.learn(
            spec,
            mapped,
            revealed=RevealedObjectObservation(
                mapped.source,
                mapped.scope,
                mapped.input_key,
                targets[0].evidence_source,
                (aggregate_target,),
                (),
                (_NAMESPACE, 500, carrier_ordinal),
            ),
        )
        if trace.projection is None:
            raise W07CarrierScopeError(
                "carrier logic projection 未进入 active lifecycle")
        before_cache[carrier_candidate] = runtime.retained_projection(
            carrier_candidate)
        pending.append((
            mapped, carrier_candidate, trace.projection, targets))
    runtime.graph.ontology.clear_runtime_caches()

    output = []
    for mapped, candidate, projection, bound_targets in pending:
        projections = tuple(W07CarrierLogicProjection(
            target.substage,
            candidate.stable_key(),
            target.proposition.stable_key(),
            projection.stable_key(),
            runtime.retained_projection(candidate) == before_cache[candidate],
        ) for target in bound_targets)
        cells = tuple(W07CarrierLogicCell(
            mapped.carrier_key,
            target.substage,
            artifact.direction,
            W05_LC16_SCOPE_KEY,
            _DIRECTION_VALUES[artifact.direction],
            digest_value(list(artifact.use_key)),
            digest_value(list(artifact.outcome_key)),
            artifact.status,
        ) for target in bound_targets
            for artifact in target.direction_artifacts)
        output.append(W07CarrierScopeRecord(
            mapped.carrier_key,
            mapped.source.stable_key(),
            mapped.scope.stable_key(),
            mapped.envelope.identity.stable_key(),
            tuple(item.stable_key() for item in mapped.feature_identities),
            projections,
            cells,
        ))
    ordered = tuple(sorted(
        output,
        key=lambda item: W05_LC16_CARRIER_KEYS.index(item.carrier_key),
    ))
    value = {
        "carrier_count": len(ordered),
        "carrier_projection_count": len(ordered),
        "logic_cell_count": sum(len(item.logic_cells) for item in ordered),
        "logic_projection_binding_count": sum(
            len(item.projections) for item in ordered),
        "records": [item.to_dict() for item in ordered],
        "scope_key": W05_LC16_SCOPE_KEY,
    }
    return W07CarrierScopeReport(
        ordered,
        hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )


__all__ = [
    "W07CarrierLogicCell",
    "W07CarrierLogicProjection",
    "W07CarrierScopeError",
    "W07CarrierScopeRecord",
    "W07CarrierScopeReport",
    "W07LogicDirectionArtifact",
    "W07LogicProjectionTarget",
    "build_w07_carrier_scope_report",
]
