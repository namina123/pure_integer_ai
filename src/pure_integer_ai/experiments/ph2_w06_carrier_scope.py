"""W-06 typed relation 在 LC16 九载体上的 carrier-neutral scope 投影。"""
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
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import concept_identity
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
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
)


_NAMESPACE = 50640
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


class W06CarrierScopeError(RuntimeError):
    """九载体输入、共享 relation 或 U/R/G scope 投影未闭合。"""


@dataclass(frozen=True)
class W06CarrierDirectionCell:
    """一个 carrier 上一个方向对同一 relation Use/outcome 的投影。"""

    carrier_key: str
    direction: str
    scope_key: str
    projection_direction: int
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        if self.carrier_key not in W05_LC16_CARRIER_KEYS:
            raise W06CarrierScopeError("未知 LC16 carrier")
        if (self.direction not in W05_LC16_DIRECTIONS
                or self.projection_direction != _DIRECTION_VALUES[self.direction]):
            raise W06CarrierScopeError("LC16 direction identity 漂移")
        if self.scope_key != W05_LC16_SCOPE_KEY:
            raise W06CarrierScopeError("LC16 scope key 漂移")
        for name in ("use_key", "outcome_key"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int for item in value)):
                raise W06CarrierScopeError(f"{name} 非法")
        if not isinstance(self.status, str) or not self.status:
            raise W06CarrierScopeError("direction status 非法")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "direction": self.direction,
            "outcome_key": list(self.outcome_key),
            "projection_direction": self.projection_direction,
            "scope_key": self.scope_key,
            "status": self.status,
            "use_key": list(self.use_key),
        }


@dataclass(frozen=True)
class W06CarrierScopeRecord:
    """一个 carrier-local input 到共享 typed relation 结构的回执。"""

    carrier_key: str
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    envelope_key: tuple[int, ...]
    feature_keys: tuple[tuple[int, ...], ...]
    carrier_candidate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    endpoint_keys: tuple[tuple[int, ...], ...]
    role_binding_keys: tuple[tuple[int, ...], ...]
    proposition_context_key: tuple[int, ...]
    projection_key: tuple[int, ...]
    retained_projection_equal: bool
    direction_cells: tuple[W06CarrierDirectionCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_candidate_key": list(self.carrier_candidate_key),
            "carrier_key": self.carrier_key,
            "direction_cells": [item.to_dict() for item in self.direction_cells],
            "endpoint_keys": [list(item) for item in self.endpoint_keys],
            "envelope_key": list(self.envelope_key),
            "feature_keys": [list(item) for item in self.feature_keys],
            "predicate_key": list(self.predicate_key),
            "projection_key": list(self.projection_key),
            "proposition_context_key": list(self.proposition_context_key),
            "proposition_key": list(self.proposition_key),
            "retained_projection_equal": int(self.retained_projection_equal),
            "role_binding_keys": [list(item) for item in self.role_binding_keys],
            "scope_key": list(self.scope_key),
            "source_key": list(self.source_key),
        }


@dataclass(frozen=True)
class W06CarrierScopeReport:
    """九 carrier、27 direction cell 和 cache readback 的统一报告。"""

    records: tuple[W06CarrierScopeRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        if tuple(item.carrier_key for item in self.records) != (
                W05_LC16_CARRIER_KEYS):
            raise W06CarrierScopeError("九载体缺失或顺序漂移")
        cells = tuple(
            cell for item in self.records for cell in item.direction_cells)
        expected = {
            (carrier, direction)
            for carrier in W05_LC16_CARRIER_KEYS
            for direction in W05_LC16_DIRECTIONS
        }
        if (len(cells) != 27
                or {(item.carrier_key, item.direction) for item in cells}
                != expected):
            raise W06CarrierScopeError("relation scope cell 覆盖不完整")
        if not all(item.retained_projection_equal for item in self.records):
            raise W06CarrierScopeError("carrier projection cache readback 漂移")
        actual = hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()
        if self.digest != actual:
            raise W06CarrierScopeError("carrier scope report digest 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_count": len(self.records),
            "direction_cell_count": sum(
                len(item.direction_cells) for item in self.records),
            "records": [item.to_dict() for item in self.records],
            "scope_key": W05_LC16_SCOPE_KEY,
        }


def _public_file(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W06CarrierScopeError(f"公开 LC16 文件缺失: {relative}")
    return target


def _evidence_source(
        learning: W06RelationLearningRuntime,
        candidate: W06RelationCandidate,
        ):
    for application in learning.applications():
        for account in application.accounts:
            if (account.candidate == candidate.proposition.proposition
                    and account.stance == EVIDENCE_SUPPORT
                    and account.observation_source != candidate.source_ref):
                return account.observation_source
    raise W06CarrierScopeError("目标 relation 缺独立 support Evidence source")


def _direction_cells(
        carrier_key: str,
        artifacts: dict[str, tuple[tuple[int, ...], tuple[int, ...], str]],
        ) -> tuple[W06CarrierDirectionCell, ...]:
    if set(artifacts) != set(W05_LC16_DIRECTIONS):
        raise W06CarrierScopeError("U/R/G artifact 集不完整")
    return tuple(
        W06CarrierDirectionCell(
            carrier_key,
            direction,
            W05_LC16_SCOPE_KEY,
            _DIRECTION_VALUES[direction],
            artifacts[direction][0],
            artifacts[direction][1],
            artifacts[direction][2],
        )
        for direction in W05_LC16_DIRECTIONS
    )


def build_w06_carrier_scope_report(
        repository_root: str | Path,
        backend,
        learning: W06RelationLearningRuntime,
        candidate: W06RelationCandidate,
        direction_artifacts: dict[
            str, tuple[tuple[int, ...], tuple[int, ...], str]],
        ) -> W06CarrierScopeReport:
    """用一个共享 runtime 将同一 typed relation 投影到九载体。"""
    if not isinstance(learning, W06RelationLearningRuntime):
        raise TypeError("learning 必须是 W06RelationLearningRuntime")
    if not isinstance(candidate, W06RelationCandidate):
        raise TypeError("candidate 必须是 W06RelationCandidate")
    root = Path(repository_root).resolve()
    parent = read_typed_carrier_pack_manifest(
        _public_file(root, PARENT_PACK_PATH))
    mapper_manifest = build_carrier_projection_mapper_manifest(root)
    rules = {item.carrier_key: item for item in mapper_manifest.rules}
    if tuple(key for key, *_ in _CARRIER_SPECS) != W05_LC16_CARRIER_KEYS:
        raise W06CarrierScopeError("carrier adapter registry 顺序漂移")
    mapper = CarrierProjectionMapper(parent)
    runtime = CarrierProjectionRuntime(backend)
    evidence_source = _evidence_source(learning, candidate)
    projection_kind = concept_identity((_NAMESPACE, 100))
    directions = tuple(sorted(_DIRECTION_VALUES.values()))
    pending = []
    before_cache = {}
    for ordinal, (carrier_key, relative, reader, adapter) in enumerate(
            _CARRIER_SPECS, start=1):
        records = reader(_public_file(root, relative))
        if not records:
            raise W06CarrierScopeError(f"{carrier_key} sample 为空")
        mapped = mapper.map(
            carrier_key,
            adapter(records[0]),
            rules[carrier_key],
            item_indices=(0,),
            input_key=(_NAMESPACE, 200, ordinal),
        )
        carrier_candidate = concept_identity((_NAMESPACE, 300, ordinal))
        spec = CarrierProjectionSpec(
            carrier_candidate,
            (_NAMESPACE, 400, ordinal),
            projection_kind,
            candidate.proposition.proposition,
            directions,
            mapped.feature_identities,
            (candidate.source_ref, evidence_source),
        )
        trace = runtime.learn(
            spec,
            mapped,
            revealed=RevealedObjectObservation(
                mapped.source,
                mapped.scope,
                mapped.input_key,
                evidence_source,
                (candidate.proposition.proposition,),
                (),
                (_NAMESPACE, 500, ordinal),
            ),
        )
        if trace.projection is None:
            raise W06CarrierScopeError("carrier projection 未进入 active lifecycle")
        before_cache[carrier_candidate] = runtime.retained_projection(
            carrier_candidate)
        pending.append((mapped, carrier_candidate, trace.projection))
    runtime.graph.ontology.clear_runtime_caches()

    role_keys = tuple(
        item.identity_for(candidate.proposition.proposition).stable_key()
        for item in candidate.proposition.canonical_bindings()
    )
    output = []
    for mapped, carrier_candidate, projection in pending:
        output.append(W06CarrierScopeRecord(
            mapped.carrier_key,
            mapped.source.stable_key(),
            mapped.scope.stable_key(),
            mapped.envelope.identity.stable_key(),
            tuple(item.stable_key() for item in mapped.feature_identities),
            carrier_candidate.stable_key(),
            candidate.proposition.proposition.stable_key(),
            candidate.proposition.predicate.stable_key(),
            tuple(item.identity.stable_key() for item in candidate.endpoints),
            role_keys,
            candidate.proposition.context.stable_key(),
            projection.stable_key(),
            runtime.retained_projection(carrier_candidate)
            == before_cache[carrier_candidate],
            _direction_cells(mapped.carrier_key, direction_artifacts),
        ))
    ordered = tuple(sorted(
        output,
        key=lambda item: W05_LC16_CARRIER_KEYS.index(item.carrier_key),
    ))
    value = {
        "carrier_count": len(ordered),
        "direction_cell_count": sum(
            len(item.direction_cells) for item in ordered),
        "records": [item.to_dict() for item in ordered],
        "scope_key": W05_LC16_SCOPE_KEY,
    }
    return W06CarrierScopeReport(
        ordered,
        hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )


__all__ = [
    "W06CarrierDirectionCell",
    "W06CarrierScopeError",
    "W06CarrierScopeRecord",
    "W06CarrierScopeReport",
    "build_w06_carrier_scope_report",
]
