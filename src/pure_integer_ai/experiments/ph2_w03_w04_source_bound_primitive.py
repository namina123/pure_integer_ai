"""FT27 从 FT26 artifact 到 W-04 来源声明的确定性投影。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseEntry,
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    W03PublicSenseRuntime,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive_contract import (
    W03W04SourceBoundPrimitiveError,
    W03W04SourceBoundPrimitiveQueryResult,
    W04_SOURCE_BOUND_PRIMITIVE_KINDS,
    W04_SOURCE_BOUND_PRIMITIVE_REGISTRY,
    W04SourceBoundPrimitive,
    asserted_value,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicCandidateProjection,
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity_key(value: object) -> tuple[int, ...]:
    """把一个规范身份编码为稳定的非零整数键。"""
    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    values = tuple(
        int.from_bytes(digest[offset:offset + 8], "big")
        & ((1 << 63) - 1)
        for offset in range(0, len(digest), 8)
    )
    return (1, 27, 4, *(item or 1 for item in values))


def _primitive(entry: W03PublicSenseEntry) -> W04SourceBoundPrimitive:
    """投影一个 entry，但不把其中的文本解释为最终事实。"""
    kind = W04_SOURCE_BOUND_PRIMITIVE_KINDS.get(entry.relation_kind)
    if kind is None:
        raise W03W04SourceBoundPrimitiveError(
            "FT27 encountered an unsupported W03 relation")
    value = asserted_value(entry)
    primitive_key = _identity_key({
        "entry": entry.to_dict(),
        "primitive_kind": kind,
        "primitive_registry": W04_SOURCE_BOUND_PRIMITIVE_REGISTRY,
        "projection_version": 1,
    })
    candidate = W04V2PublicCandidateProjection(
        entry.surface,
        value,
        W04_SOURCE_BOUND_PRIMITIVE_REGISTRY,
        kind,
        primitive_key,
        entry.source_ref.stable_key,
        entry.source_ref.source_key,
        entry.source_ref.source_commitment_sha256,
        entry.source_ref.license_id,
        entry.active,
        int(entry.active == 0),
    )
    return W04SourceBoundPrimitive(
        primitive_key,
        entry.entry_key,
        entry.relation_kind,
        entry.surface,
        entry.canonical_surface,
        entry.language,
        value,
        entry.definition_text,
        entry.sense_key,
        entry.concept_key,
        entry.observation_key,
        entry.source_ref,
        entry.field_roles,
        entry.active,
        entry.supersedes_entry_keys,
        candidate,
        primitive_kind=kind,
    )


def _validate_source_pack_bindings(runtime: W03PublicSenseRuntime) -> None:
    """拒绝 compact pack 已被删除的 entry 或 redirect alias。"""
    packs = {
        (item.source_key, item.snapshot_id, item.license_id)
        for item in runtime.artifact.source_packs
    }
    source_refs = (
        *(item.source_ref for item in runtime.artifact.entries),
        *(item.source_ref for item in runtime.artifact.aliases),
    )
    if any(
            (item.source_key, item.snapshot_id, item.license_id) not in packs
            for item in source_refs):
        raise W03W04SourceBoundPrimitiveError(
            "FT27 source entry is no longer bound to a compact source pack")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04SourceBoundPrimitiveRuntime:
    """不可变 FT26 artifact 及全部 active/superseded FT27 声明。"""

    sense_runtime: W03PublicSenseRuntime
    primitives: tuple[W04SourceBoundPrimitive, ...]
    source_binding_sha256: str
    projection_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sense_runtime, W03PublicSenseRuntime):
            raise TypeError("FT27 sense runtime type is invalid")
        if (not isinstance(self.primitives, tuple) or not self.primitives
                or any(not isinstance(item, W04SourceBoundPrimitive)
                       for item in self.primitives)):
            raise W03W04SourceBoundPrimitiveError(
                "FT27 primitive inventory is invalid")
        entry_keys = tuple(item.entry_key for item in self.primitives)
        expected = tuple(
            item.entry_key for item in self.sense_runtime.artifact.entries)
        if (entry_keys != expected
                or len(set(item.primitive_key for item in self.primitives))
                != len(self.primitives)):
            raise W03W04SourceBoundPrimitiveError(
                "FT27 primitive inventory lost W03 identity")
        for name in ("source_binding_sha256", "projection_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise W03W04SourceBoundPrimitiveError(
                    f"FT27 {name} drifted")

    def to_dict(self) -> dict[str, object]:
        """导出不含 raw 来源材料的可观测投影。"""
        return {
            "artifact_sha256": self.sense_runtime.artifact_sha256,
            "experimental": 1,
            "formal_mastery_claim": 0,
            "primitives": [item.to_dict() for item in self.primitives],
            "projection_sha256": self.projection_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "source_revisions": [
                item.to_dict()
                for item in self.sense_runtime.artifact.source_revisions],
            "w03_started": 0,
            "w04_started": 0,
        }


def project_w03_public_sense_to_w04_primitives(
        runtime: W03PublicSenseRuntime,
        ) -> W03W04SourceBoundPrimitiveRuntime:
    """只从 FT26 构造逐位一致的 W-04 来源声明投影。"""
    if not isinstance(runtime, W03PublicSenseRuntime):
        raise TypeError("FT27 projection requires W03PublicSenseRuntime")
    _validate_source_pack_bindings(runtime)
    primitives = tuple(_primitive(item) for item in runtime.artifact.entries)
    source_value = {
        "source_packs": [
            item.to_dict() for item in runtime.artifact.source_packs],
        "source_revisions": [
            item.to_dict() for item in runtime.artifact.source_revisions],
    }
    source_binding = _sha(source_value)
    projection = _sha({
        "artifact_sha256": runtime.artifact_sha256,
        "primitives": [item.to_dict() for item in primitives],
        "projection_version": 1,
        "source_binding_sha256": source_binding,
    })
    return W03W04SourceBoundPrimitiveRuntime(
        runtime, primitives, source_binding, projection)


def query_w03_w04_source_bound_primitives(
        runtime: W03W04SourceBoundPrimitiveRuntime,
        query: W03PublicSenseQuery,
        ) -> W03W04SourceBoundPrimitiveQueryResult:
    """保留 W03 非唯一状态，同时公开匹配的 W04 来源声明。"""
    if (not isinstance(runtime, W03W04SourceBoundPrimitiveRuntime)
            or not isinstance(query, W03PublicSenseQuery)):
        raise TypeError("FT27 primitive query inputs are invalid")
    sense = query_w03_public_sense(runtime.sense_runtime, query)
    by_entry = {item.entry_key: item for item in runtime.primitives}
    try:
        primitives = tuple(
            by_entry[item.entry.entry_key] for item in sense.candidates)
    except KeyError as error:
        raise W03W04SourceBoundPrimitiveError(
            "FT27 projection is missing a W03 candidate") from error
    return W03W04SourceBoundPrimitiveQueryResult(
        sense,
        sense.status,
        primitives,
        runtime.sense_runtime.artifact.source_revisions,
        runtime.source_binding_sha256,
        runtime.projection_sha256,
        int(sense.status in {"AMBIGUOUS", "CONFLICT", "CLARIFY"}),
    )


def query_w03_w04_source_bound_primitive_batch(
        runtime: W03W04SourceBoundPrimitiveRuntime,
        queries: tuple[W03PublicSenseQuery, ...],
        ) -> tuple[W03W04SourceBoundPrimitiveQueryResult, ...]:
    """在同一不可变 FT27 投影上查询一个有界批次。"""
    if (not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03PublicSenseQuery)
                   for item in queries)):
        raise TypeError("FT27 primitive query batch is invalid")
    return tuple(
        query_w03_w04_source_bound_primitives(runtime, item)
        for item in queries)


__all__ = [
    "W03W04SourceBoundPrimitiveRuntime",
    "project_w03_public_sense_to_w04_primitives",
    "query_w03_w04_source_bound_primitive_batch",
    "query_w03_w04_source_bound_primitives",
]
