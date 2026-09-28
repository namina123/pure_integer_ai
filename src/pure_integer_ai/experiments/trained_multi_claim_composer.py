"""Read-only integer graph runtime for multi-claim response composition.

The graph stores only ordered literal chunks and claim/role coordinates.  A
runtime binds those coordinates to the current Core ``ActiveRelationSurface``
objects; it never stores or replays a complete answer sentence.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    concept_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    STRUCTURE_CLOSED,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GraphRelationGeneration,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_FORMAT = "PURE_INTEGER_MULTI_CLAIM_COMPOSITION_GRAPH_V1"
GRAPH_KIND = 91640
META_TABLE = "multi_claim_composition_graph"
VARIANT_TABLE = "multi_claim_composition_variant"
PART_TABLE = "multi_claim_composition_part"
IDENTITY_PART_TABLE = "multi_claim_composition_identity_part"

PART_LITERAL = 1
PART_CLAIM_ROLE = 2


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    digest = hashlib.sha256(bytes(payload)).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return value or 1


def _key(value, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    value = tuple(value)
    if ((not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label} 必须是非负整数 tuple")
    return value


@dataclass(frozen=True, slots=True)
class MultiClaimCompositionVariant:
    """One learned ordered composition frame."""

    variant_id: int
    claim_count: int
    parts: tuple[tuple, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.variant_id) is not int or self.variant_id <= 0:
            raise ValueError("variant_id 必须为正整数")
        if type(self.claim_count) is not int or self.claim_count < 2:
            raise ValueError("multi-claim variant 至少需要两个 claim")
        if type(self.parts) is not tuple or not self.parts:
            raise ValueError("composition parts 不能为空")
        for part in self.parts:
            if (not isinstance(part, tuple) or not part
                    or part[0] not in {PART_LITERAL, PART_CLAIM_ROLE}):
                raise ValueError("composition part kind 未注册")
            if part[0] == PART_LITERAL:
                if (len(part) != 2 or not part[1]
                        or any(type(value) is not int or value < 0
                               for value in part[1])):
                    raise ValueError("literal part 必须是非空整数 tuple")
            elif (len(part) != 3
                  or type(part[1]) is not int or not 1 <= part[1] <= self.claim_count
                  or type(part[2]) is not int or part[2] <= 0):
                raise ValueError("claim role part 坐标非法")
        _key(self.source_identity, label="source_identity")
        _key(self.license_identity, label="license_identity")

    def stable_key(self) -> tuple[int, ...]:
        values = [GRAPH_KIND, self.variant_id, self.claim_count, len(self.parts)]
        for part in self.parts:
            values.extend((part[0], len(part[1]), *part[1])
                          if part[0] == PART_LITERAL
                          else (part[0], part[1], part[2]))
        values.extend((len(self.source_identity), *self.source_identity,
                       len(self.license_identity), *self.license_identity))
        return tuple(values)


def _register_tables(backend) -> None:
    register_extension_table(
        backend, META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("format_version", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("variant_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)], recovery_key=("graph_id",),
    )
    register_extension_table(
        backend, VARIANT_TABLE,
        [("variant_id", TYPE_INT), ("variant_hash", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT),
         ("claim_count", TYPE_INT), ("part_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("variant_id",)], recovery_key=("variant_id",),
    )
    register_extension_table(
        backend, PART_TABLE,
        [("variant_id", TYPE_INT), ("part_ordinal", TYPE_INT),
         ("part_kind", TYPE_INT), ("claim_ordinal", TYPE_INT),
         ("binding_ordinal", TYPE_INT), ("value_ordinal", TYPE_INT),
         ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("variant_id", "part_ordinal")],
        recovery_key=("variant_id", "part_ordinal", "value_ordinal"),
    )
    register_extension_table(
        backend, IDENTITY_PART_TABLE,
        [("graph_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id", "field_code")],
        recovery_key=("graph_id", "field_code", "part_ordinal"),
    )


def materialize_multi_claim_composition_graph(
        variants: tuple[MultiClaimCompositionVariant, ...],
        database: str | Path,
        ) -> dict[str, object]:
    """Materialize a new sidecar graph without opening canonical SQLite."""
    if type(variants) is not tuple or not variants:
        raise ValueError("variants 必须是非空 tuple")
    if any(not isinstance(item, MultiClaimCompositionVariant) for item in variants):
        raise TypeError("composition variant 类型错误")
    variants = tuple(sorted(variants, key=lambda item: item.variant_id))
    if tuple(item.variant_id for item in variants) != tuple(
            range(1, len(variants) + 1)):
        raise ValueError("variant_id 必须从 1 连续")
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError(target)
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    source = variants[0].source_identity
    license_key = variants[0].license_identity
    if any(item.source_identity != source or item.license_identity != license_key
           for item in variants):
        raise ValueError("composition graph source/license 不一致")
    graph_id = _key_hash((GRAPH_KIND,), source, license_key)
    try:
        _register_tables(backend)
        backend.insert(META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "format_version": 1, "source_hash": _key_hash(source),
            "license_hash": _key_hash(license_key),
            "variant_count": len(variants),
        })
        backend.insert_many(IDENTITY_PART_TABLE, [
            {"graph_id": graph_id, "field_code": field_code,
             "part_ordinal": ordinal, "part_value": value}
            for field_code, key in ((1, source), (2, license_key))
            for ordinal, value in enumerate(key)
        ])
        for variant in variants:
            variant_hash = _key_hash(variant.stable_key())
            backend.insert(VARIANT_TABLE, {
                "variant_id": variant.variant_id,
                "variant_hash": variant_hash,
                "source_hash": _key_hash(source),
                "license_hash": _key_hash(license_key),
                "claim_count": variant.claim_count,
                "part_count": len(variant.parts),
            })
            rows = []
            for ordinal, part in enumerate(variant.parts):
                if part[0] == PART_LITERAL:
                    rows.extend({
                        "variant_id": variant.variant_id,
                        "part_ordinal": ordinal,
                        "part_kind": PART_LITERAL,
                        "claim_ordinal": 0,
                        "binding_ordinal": 0,
                        "value_ordinal": index,
                        "part_value": value,
                    } for index, value in enumerate(part[1]))
                else:
                    rows.append({
                        "variant_id": variant.variant_id,
                        "part_ordinal": ordinal,
                        "part_kind": PART_CLAIM_ROLE,
                        "claim_ordinal": part[1],
                        "binding_ordinal": part[2],
                        "value_ordinal": 0,
                        "part_value": 0,
                    })
            backend.insert_many(PART_TABLE, rows)
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT,
        "graph_id": graph_id,
        "variant_count": len(variants),
        "source_identity": list(source),
        "license_identity": list(license_key),
    }


def read_multi_claim_composition_graph(
        database: str | Path,
        ) -> tuple[tuple[MultiClaimCompositionVariant, ...], dict[str, object]]:
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register_tables(backend)
        metas = backend.select(META_TABLE)
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise ValueError("composition graph metadata 不唯一")
        rows = backend.select(VARIANT_TABLE, order_by="variant_id")
        if len(rows) != metas[0]["variant_count"]:
            raise ValueError("composition graph variant count 漂移")
        meta = metas[0]
        identity_rows = backend.select(
            IDENTITY_PART_TABLE, where={"graph_id": meta["graph_id"]},
            order_by="field_code,part_ordinal")
        identities = {}
        for item in identity_rows:
            identities.setdefault(item["field_code"], []).append(item["part_value"])
        source_identity = _key(identities.get(1, ()), label="source_identity")
        license_identity = _key(identities.get(2, ()), label="license_identity")
        if (meta["source_hash"] != _key_hash(source_identity)
                or meta["license_hash"] != _key_hash(license_identity)):
            raise ValueError("composition graph identity hash 漂移")
        result = []
        for row in rows:
            part_rows = backend.select(PART_TABLE, where={"variant_id": row["variant_id"]},
                                       order_by="part_ordinal,value_ordinal")
            groups = {}
            for item in part_rows:
                groups.setdefault(item["part_ordinal"], []).append(item)
            if tuple(groups) != tuple(range(row["part_count"])):
                raise ValueError("composition part ordinal 不连续")
            parts = []
            for ordinal in range(row["part_count"]):
                items = groups[ordinal]
                kind = items[0]["part_kind"]
                if any(item["part_kind"] != kind for item in items):
                    raise ValueError("composition part kind 漂移")
                if kind == PART_LITERAL:
                    if any(item["value_ordinal"] != index for index, item in enumerate(items)):
                        raise ValueError("literal part value ordinal 不连续")
                    parts.append((PART_LITERAL, tuple(item["part_value"] for item in items)))
                elif kind == PART_CLAIM_ROLE:
                    if len(items) != 1 or items[0]["value_ordinal"] != 0:
                        raise ValueError("claim role part 不唯一")
                    parts.append((PART_CLAIM_ROLE, items[0]["claim_ordinal"],
                                  items[0]["binding_ordinal"]))
                else:
                    raise ValueError("composition part kind 未注册")
            variant = MultiClaimCompositionVariant(
                int(row["variant_id"]), int(row["claim_count"]), tuple(parts),
                source_identity, license_identity)
            if row["variant_hash"] != _key_hash(variant.stable_key()):
                raise ValueError("composition variant hash 漂移")
            result.append(variant)
        metadata = {
            "format": GRAPH_FORMAT,
            "graph_id": int(metas[0]["graph_id"]),
            "variant_count": int(metas[0]["variant_count"]),
            "source_identity": source_identity,
            "license_identity": license_identity,
        }
        return tuple(result), metadata
    finally:
        backend.close()


class TrainedMultiClaimComposer:
    """Delegate ordinary relation generation and add one trained composer."""

    def __init__(self, graph_database: str | Path, relation_generator) -> None:
        self.variants, self.metadata = read_multi_claim_composition_graph(graph_database)
        if not self.variants or relation_generator is None:
            raise ValueError("multi-claim composer graph/runtime 不完整")
        self._relation_generator = relation_generator

    def __getattr__(self, name):
        return getattr(self._relation_generator, name)

    def generate_relation(self, source, fact):
        return self._relation_generator.generate_relation(source, fact)

    def generate_composition(self, components, state, *, variant_id=None):
        """Execute a learned part order against current Core role surfaces.

        A claim-count-only lookup remains valid for the legacy single-variant
        sidecar.  When a sidecar contains several surface variants, callers
        must select the integer variant explicitly; no ordering guess is made.
        """
        if (type(components) is not tuple or len(components) < 2
                or not getattr(state, "evidence_closed", 0)
                or getattr(state, "conflict_open", 1)
                or getattr(state, "required_slots_open", 1)):
            raise ValueError("composition state is not closed")
        if variant_id is not None and (
                type(variant_id) is not int or variant_id <= 0):
            raise ValueError("composition variant_id 必须为正整数")
        matches = tuple(item for item in self.variants
                        if item.claim_count == len(components)
                        and (variant_id is None or item.variant_id == variant_id))
        if len(matches) != 1:
            raise ValueError("composition variant is not unique")
        variant = matches[0]
        units = []
        for part in variant.parts:
            if part[0] == PART_LITERAL:
                units.extend(part[1])
                continue
            claim_index, binding_index = part[1] - 1, part[2] - 1
            if claim_index >= len(components):
                raise ValueError("composition claim coordinate out of range")
            fact = components[claim_index][1]
            bindings = getattr(fact, "bindings", ())
            if binding_index >= len(bindings):
                raise ValueError("composition role coordinate out of range")
            surface = bindings[binding_index].surface
            if type(surface) is not str or not surface:
                raise ValueError("composition role lacks trained surface")
            units.extend(map(ord, surface))
        connector = structure_concept_identity((GRAPH_KIND, variant.variant_id, 1))
        frame = structure_concept_identity((GRAPH_KIND, variant.variant_id, 2))
        return GraphRelationGeneration(
            "".join(map(chr, units)), frame, _key_hash(variant.stable_key()),
            sum(len(item[1].bindings) for item in components), connector,
            (concept_identity((GRAPH_KIND, variant.variant_id, 3)),),
            (GRAPH_KIND, variant.variant_id, len(units)),
        )


__all__ = [
    "GRAPH_FORMAT", "GRAPH_KIND", "PART_LITERAL", "PART_CLAIM_ROLE",
    "MultiClaimCompositionVariant", "TrainedMultiClaimComposer",
    "materialize_multi_claim_composition_graph",
    "read_multi_claim_composition_graph",
]
