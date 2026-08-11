"""FT28 从 FT27 primitive 到来源绑定 W-05 proposition 的确定性投影。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive import (
    W03W04SourceBoundPrimitiveRuntime,
    query_w03_w04_source_bound_primitives,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive_contract import (
    W04SourceBoundPrimitive,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition_contract import (
    W04W05SourceBoundPropositionError,
    W04W05SourceBoundPropositionQueryResult,
    W05_SOURCE_BOUND_PROPOSITION_KINDS,
    W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
    W05SourceBoundProposition,
    source_bound_candidate_projection,
    source_bound_proposition_key,
    source_bound_query_record_commitment,
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _propositions(
        primitives: tuple[W04SourceBoundPrimitive, ...],
        ) -> tuple[W05SourceBoundProposition, ...]:
    keys = {
        item.entry_key: source_bound_proposition_key(item)
        for item in primitives}
    output = []
    for primitive in primitives:
        proposition_key = keys[primitive.entry_key]
        try:
            supersedes = tuple(sorted(
                keys[item] for item in primitive.supersedes_entry_keys))
        except KeyError as error:
            raise W04W05SourceBoundPropositionError(
                "FT28 proposition supersede predecessor is missing") from error
        output.append(W05SourceBoundProposition(
            proposition_key,
            primitive,
            W05_SOURCE_BOUND_PROPOSITION_REGISTRY,
            W05_SOURCE_BOUND_PROPOSITION_KINDS[primitive.relation_kind],
            supersedes,
            source_bound_candidate_projection(primitive, proposition_key),
        ))
    return tuple(output)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04W05SourceBoundPropositionRuntime:
    """不可变 FT27 runtime 与全部 active/superseded W05 propositions。"""

    primitive_runtime: W03W04SourceBoundPrimitiveRuntime
    propositions: tuple[W05SourceBoundProposition, ...]
    proposition_projection_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(
                self.primitive_runtime,
                W03W04SourceBoundPrimitiveRuntime,
                ):
            raise TypeError("FT28 primitive runtime type is invalid")
        if (not isinstance(self.propositions, tuple)
                or not self.propositions
                or any(not isinstance(item, W05SourceBoundProposition)
                       for item in self.propositions)):
            raise W04W05SourceBoundPropositionError(
                "FT28 proposition inventory is invalid")
        expected = tuple(
            item.primitive_key
            for item in self.primitive_runtime.primitives)
        actual = tuple(
            item.primitive.primitive_key for item in self.propositions)
        if (actual != expected
                or len(set(item.proposition_key for item in self.propositions))
                != len(self.propositions)):
            raise W04W05SourceBoundPropositionError(
                "FT28 proposition inventory lost primitive identity")
        keys = {
            item.primitive.entry_key: item.proposition_key
            for item in self.propositions}
        for item in self.propositions:
            try:
                expected_supersedes = tuple(sorted(
                    keys[key]
                    for key in item.primitive.supersedes_entry_keys))
            except KeyError as error:
                raise W04W05SourceBoundPropositionError(
                    "FT28 proposition supersede predecessor is missing"
                ) from error
            if item.supersedes_proposition_keys != expected_supersedes:
                raise W04W05SourceBoundPropositionError(
                    "FT28 proposition supersede projection drifted")
        if (not isinstance(self.proposition_projection_sha256, str)
                or len(self.proposition_projection_sha256) != 64):
            raise W04W05SourceBoundPropositionError(
                "FT28 proposition projection commitment drifted")
        expected_projection = _sha({
            "primitive_projection_sha256": (
                self.primitive_runtime.projection_sha256),
            "projection_version": 1,
            "propositions": [
                item.to_dict() for item in self.propositions],
            "source_binding_sha256": (
                self.primitive_runtime.source_binding_sha256),
        })
        if self.proposition_projection_sha256 != expected_projection:
            raise W04W05SourceBoundPropositionError(
                "FT28 proposition projection commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "experimental": 1,
            "formal_mastery_claim": 0,
            "primitive_projection_sha256": (
                self.primitive_runtime.projection_sha256),
            "proposition_projection_sha256": (
                self.proposition_projection_sha256),
            "propositions": [item.to_dict() for item in self.propositions],
            "source_binding_sha256": (
                self.primitive_runtime.source_binding_sha256),
            "w03_started": 0,
            "w04_started": 0,
            "w05_started": 0,
        }


def project_w04_primitives_to_w05_source_bound_propositions(
        runtime: W03W04SourceBoundPrimitiveRuntime,
        ) -> W04W05SourceBoundPropositionRuntime:
    """把每个 FT27 primitive 一对一结构化为 source-claim proposition。"""
    if not isinstance(runtime, W03W04SourceBoundPrimitiveRuntime):
        raise TypeError("FT28 projection requires an FT27 primitive runtime")
    propositions = _propositions(runtime.primitives)
    projection = _sha({
        "primitive_projection_sha256": runtime.projection_sha256,
        "projection_version": 1,
        "propositions": [item.to_dict() for item in propositions],
        "source_binding_sha256": runtime.source_binding_sha256,
    })
    return W04W05SourceBoundPropositionRuntime(
        runtime, propositions, projection)


def query_w04_w05_source_bound_propositions(
        runtime: W04W05SourceBoundPropositionRuntime,
        query: W03PublicSenseQuery,
        ) -> W04W05SourceBoundPropositionQueryResult:
    """查询来源 propositions，绝不弱化 FT27 的非唯一状态。"""
    if (not isinstance(runtime, W04W05SourceBoundPropositionRuntime)
            or not isinstance(query, W03PublicSenseQuery)):
        raise TypeError("FT28 proposition query inputs are invalid")
    primitive_result = query_w03_w04_source_bound_primitives(
        runtime.primitive_runtime, query)
    by_primitive = {
        item.primitive.primitive_key: item for item in runtime.propositions}
    try:
        propositions = tuple(
            by_primitive[item.primitive_key]
            for item in primitive_result.primitives)
    except KeyError as error:
        raise W04W05SourceBoundPropositionError(
            "FT28 projection is missing an FT27 candidate") from error
    record_commitment = source_bound_query_record_commitment(
        query,
        primitive_result,
        propositions,
        runtime.proposition_projection_sha256,
    )
    return W04W05SourceBoundPropositionQueryResult(
        query,
        primitive_result,
        primitive_result.status,
        propositions,
        primitive_result.source_revisions,
        runtime.primitive_runtime.projection_sha256,
        runtime.proposition_projection_sha256,
        record_commitment,
        int(primitive_result.status in {
            "AMBIGUOUS", "CONFLICT", "CLARIFY"}),
    )


def query_w04_w05_source_bound_proposition_batch(
        runtime: W04W05SourceBoundPropositionRuntime,
        queries: tuple[W03PublicSenseQuery, ...],
        ) -> tuple[W04W05SourceBoundPropositionQueryResult, ...]:
    """在同一不可变 proposition runtime 上查询有界批次。"""
    if (not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03PublicSenseQuery)
                   for item in queries)):
        raise TypeError("FT28 proposition query batch is invalid")
    return tuple(
        query_w04_w05_source_bound_propositions(runtime, item)
        for item in queries)


__all__ = [
    "W04W05SourceBoundPropositionRuntime",
    "project_w04_primitives_to_w05_source_bound_propositions",
    "query_w04_w05_source_bound_proposition_batch",
    "query_w04_w05_source_bound_propositions",
]
