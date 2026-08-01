"""W-05 在 LC16 九载体上的 carrier-neutral Role/Proposition scope 投影。"""
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
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05AtomicPropositionCandidate,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_LC16_CARRIER_KEYS,
    W05_LC16_DIRECTIONS,
    W05_LC16_SCOPE_KEY,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
)


_NAMESPACE = 50540
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


class W05CarrierScopeError(RuntimeError):
    """九载体输入、共享 Proposition 或 U/R/G 投影未精确闭合。"""


@dataclass(frozen=True)
class W05CarrierDirectionCell:
    """一个 carrier 上一个方向对同一 W-05 Use/outcome 的投影。"""

    carrier_key: str
    direction: str
    scope_key: str
    projection_direction: int
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        if self.carrier_key not in W05_LC16_CARRIER_KEYS:
            raise W05CarrierScopeError("未知 LC16 carrier")
        if (self.direction not in W05_LC16_DIRECTIONS
                or self.projection_direction != _DIRECTION_VALUES[self.direction]):
            raise W05CarrierScopeError("LC16 direction identity 漂移")
        if self.scope_key != W05_LC16_SCOPE_KEY:
            raise W05CarrierScopeError("LC16 scope key 漂移")
        for name in ("use_key", "outcome_key"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int for item in value)):
                raise W05CarrierScopeError(f"{name} 非法")
        if not isinstance(self.status, str) or not self.status:
            raise W05CarrierScopeError("direction status 非法")

    def to_dict(self) -> dict[str, Any]:
        """返回可规范持久化的方向 cell。"""
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
class W05CarrierScopeRecord:
    """一个 carrier-local input 到共享 W-05 Proposition 结构的回执。"""

    carrier_key: str
    case_key: tuple[int, ...]
    input_key: tuple[int, ...]
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    envelope_key: tuple[int, ...]
    anchor_keys: tuple[tuple[int, ...], ...]
    structure_node_keys: tuple[tuple[int, ...], ...]
    feature_keys: tuple[tuple[int, ...], ...]
    carrier_candidate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    occurrence_keys: tuple[tuple[int, ...], ...]
    role_binding_keys: tuple[tuple[int, ...], ...]
    proposition_context_key: tuple[int, ...]
    projection_key: tuple[int, ...]
    retained_projection_equal: bool
    direction_cells: tuple[W05CarrierDirectionCell, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回不含 carrier-specific 语义捷径的规范记录。"""
        return {
            "anchor_keys": [list(item) for item in self.anchor_keys],
            "carrier_candidate_key": list(self.carrier_candidate_key),
            "carrier_key": self.carrier_key,
            "case_key": list(self.case_key),
            "direction_cells": [item.to_dict() for item in self.direction_cells],
            "envelope_key": list(self.envelope_key),
            "feature_keys": [list(item) for item in self.feature_keys],
            "input_key": list(self.input_key),
            "occurrence_keys": [list(item) for item in self.occurrence_keys],
            "projection_key": list(self.projection_key),
            "proposition_context_key": list(self.proposition_context_key),
            "proposition_key": list(self.proposition_key),
            "retained_projection_equal": int(self.retained_projection_equal),
            "role_binding_keys": [list(item) for item in self.role_binding_keys],
            "scope_key": list(self.scope_key),
            "source_key": list(self.source_key),
            "structure_node_keys": [
                list(item) for item in self.structure_node_keys],
        }


@dataclass(frozen=True)
class W05CarrierScopeReport:
    """九 carrier、27 direction cell 和 cache readback 的统一报告。"""

    records: tuple[W05CarrierScopeRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        if tuple(item.carrier_key for item in self.records) != W05_LC16_CARRIER_KEYS:
            raise W05CarrierScopeError("九载体缺失或顺序漂移")
        cells = tuple(cell for item in self.records for cell in item.direction_cells)
        if len(cells) != 27:
            raise W05CarrierScopeError("ROLE_PROPOSITION_SCOPE 必须有 27 cells")
        expected = {
            (carrier, direction)
            for carrier in W05_LC16_CARRIER_KEYS
            for direction in W05_LC16_DIRECTIONS
        }
        if {(item.carrier_key, item.direction) for item in cells} != expected:
            raise W05CarrierScopeError("ROLE_PROPOSITION_SCOPE cell 覆盖不完整")
        if not all(item.retained_projection_equal for item in self.records):
            raise W05CarrierScopeError("carrier projection cache readback 漂移")
        actual = hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()
        if self.digest != actual:
            raise W05CarrierScopeError("carrier scope report digest 漂移")

    def to_dict(self) -> dict[str, Any]:
        """返回九载体统一 projection 的规范值。"""
        return {
            "carrier_count": len(self.records),
            "direction_cell_count": sum(
                len(item.direction_cells) for item in self.records),
            "records": [item.to_dict() for item in self.records],
            "scope_key": W05_LC16_SCOPE_KEY,
        }


def _overlay_file(primary: Path, dependency: Path, relative: str) -> Path:
    """从 candidate root 或冻结依赖根解析一个公开 LC16 文件。"""
    for root in (primary, dependency):
        target = (root / Path(relative)).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise W05CarrierScopeError(f"公开 LC16 文件缺失: {relative}")


def _evidence_source(
        learning: W05AtomicPropositionLearningRuntime,
        candidate: W05AtomicPropositionCandidate,
        ):
    """返回支持目标 Proposition 的既有独立 Evidence SourceRef。"""
    for application in learning.applications():
        for account in application.accounts:
            if (account.candidate == candidate.candidate
                    and account.stance == EVIDENCE_SUPPORT
                    and account.observation_source != candidate.source_ref):
                return account.observation_source
    raise W05CarrierScopeError("目标 Proposition 缺独立 support Evidence source")


def _direction_cells(
        carrier_key: str,
        direction_artifacts: dict[
            str, tuple[tuple[int, ...], tuple[int, ...], str]],
        ) -> tuple[W05CarrierDirectionCell, ...]:
    """把同一 U/R/G exact Use/outcome 投影到一个 carrier。"""
    if set(direction_artifacts) != set(W05_LC16_DIRECTIONS):
        raise W05CarrierScopeError("U/R/G artifact 集不完整")
    return tuple(
        W05CarrierDirectionCell(
            carrier_key,
            direction,
            W05_LC16_SCOPE_KEY,
            _DIRECTION_VALUES[direction],
            direction_artifacts[direction][0],
            direction_artifacts[direction][1],
            direction_artifacts[direction][2],
        )
        for direction in W05_LC16_DIRECTIONS
    )


def build_w05_carrier_scope_report(
        repository_root: str | Path,
        backend,
        learning: W05AtomicPropositionLearningRuntime,
        candidate: W05AtomicPropositionCandidate,
        direction_artifacts: dict[
            str, tuple[tuple[int, ...], tuple[int, ...], str]],
        *,
        dependency_root: str | Path | None = None,
        ) -> W05CarrierScopeReport:
    """用一个共享 runtime 将同一 W-05 Proposition 投影到九载体。"""
    if not isinstance(learning, W05AtomicPropositionLearningRuntime):
        raise TypeError("learning 必须是 W05AtomicPropositionLearningRuntime")
    if not isinstance(candidate, W05AtomicPropositionCandidate):
        raise TypeError("candidate 必须是 W05AtomicPropositionCandidate")
    primary = Path(repository_root).resolve()
    dependency = (
        primary if dependency_root is None else Path(dependency_root).resolve()
    )
    parent_path = _overlay_file(primary, dependency, PARENT_PACK_PATH)
    catalog_root = primary if (primary / PARENT_PACK_PATH).is_file() else dependency
    parent = read_typed_carrier_pack_manifest(parent_path)
    mapper_manifest = build_carrier_projection_mapper_manifest(catalog_root)
    rules = {item.carrier_key: item for item in mapper_manifest.rules}
    if tuple(key for key, *_ in _CARRIER_SPECS) != W05_LC16_CARRIER_KEYS:
        raise W05CarrierScopeError("carrier adapter registry 顺序漂移")
    mapper = CarrierProjectionMapper(parent)
    runtime = CarrierProjectionRuntime(backend)
    evidence_source = _evidence_source(learning, candidate)
    projection_kind = concept_identity((_NAMESPACE, 100))
    directions = tuple(sorted(_DIRECTION_VALUES.values()))
    pending: list[tuple[Any, Any, Any, Any]] = []
    before_cache = {}
    for ordinal, (carrier_key, relative, reader, adapter) in enumerate(
            _CARRIER_SPECS, start=1):
        records = reader(_overlay_file(primary, dependency, relative))
        if not records:
            raise W05CarrierScopeError(f"{carrier_key} sample 为空")
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
            candidate.candidate,
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
                (candidate.candidate,),
                (),
                (_NAMESPACE, 500, ordinal),
            ),
        )
        if trace.projection is None:
            raise W05CarrierScopeError("carrier projection 未进入 active lifecycle")
        retained = runtime.retained_projection(carrier_candidate)
        before_cache[carrier_candidate] = retained
        pending.append((mapped, carrier_candidate, trace.projection, spec))
    runtime.graph.ontology.clear_runtime_caches()

    records = []
    for mapped, carrier_candidate, projection, _spec in pending:
        retained_equal = (
            runtime.retained_projection(carrier_candidate)
            == before_cache[carrier_candidate]
        )
        records.append(W05CarrierScopeRecord(
            mapped.carrier_key,
            mapped.case_key.stable_key(),
            mapped.input_key,
            mapped.source.stable_key(),
            mapped.scope.stable_key(),
            mapped.envelope.identity.stable_key(),
            tuple(item.stable_key() for item in mapped.anchor_identities),
            tuple(item.stable_key()
                  for item in mapped.structure_node_identities),
            tuple(item.stable_key() for item in mapped.feature_identities),
            carrier_candidate.stable_key(),
            candidate.candidate.stable_key(),
            tuple(item.stable_key() for item in candidate.occurrence_order),
            tuple(item.stable_key() for item in sorted(
                candidate.role_binding_identities(), key=lambda value: value.stable_key())),
            candidate.proposition_definition.context.stable_key(),
            projection.stable_key(),
            retained_equal,
            _direction_cells(mapped.carrier_key, direction_artifacts),
        ))
    ordered = tuple(sorted(records, key=lambda item: W05_LC16_CARRIER_KEYS.index(
        item.carrier_key)))
    value = {
        "carrier_count": len(ordered),
        "direction_cell_count": sum(len(item.direction_cells) for item in ordered),
        "records": [item.to_dict() for item in ordered],
        "scope_key": W05_LC16_SCOPE_KEY,
    }
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return W05CarrierScopeReport(ordered, digest)


__all__ = [
    "W05CarrierDirectionCell",
    "W05CarrierScopeError",
    "W05CarrierScopeRecord",
    "W05CarrierScopeReport",
    "build_w05_carrier_scope_report",
]
