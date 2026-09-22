"""Integer-only bridge from carrier projections to an existing semantic graph.

The carrier course is an input/reference artifact, not a language course.  This
module keeps the complete carrier and Evidence identities in an append-only
extension graph and, when used by formal training, adds only projection-to-
existing-object edges.  It never creates a missing semantic object and never
copies source text into Core.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.artifact_envelope import (
    ArtifactSemanticProjection,
)
from pure_integer_ai.cognition.shared.graph_ontology import relation_concept_identity
from pure_integer_ai.experiments.source_wide_artifact_projection_course import (
    TrainedFactCarrierRecord,
    read_trained_fact_carrier_course,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.backend import register_extension_table
from pure_integer_ai.experiments.train_context import make_train_context


FORMAT_VERSION = 1
BRIDGE_RECORD_KIND = 16619941
BRIDGE_COURSE_KIND = 16619942
BRIDGE_GRAPH_KIND = 16619943
BRIDGE_STATE_BOUND = 1
BRIDGE_STATE_UNRESOLVED = 2
BRIDGE_RELATION_PROJECTION_SEMANTIC = 16619940
BRIDGE_RELATION_PROJECTION_PROPOSITION = 16619939
BRIDGE_RELATION_PROJECTION_HYPOTHESIS = 16619938
BRIDGE_TABLE = "artifact_semantic_binding"
BRIDGE_PART_TABLE = "artifact_semantic_binding_part"


class ArtifactSemanticBridgeError(RuntimeError):
    """The bridge course or its target graph is not closed."""


def _integer_tree(value: object, *, where: str) -> None:
    if type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _integer_tree(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactSemanticBridgeError(f"{where} field key 非文本")
            _integer_tree(item, where=f"{where}.{key}")
        return
    raise ArtifactSemanticBridgeError(f"{where} 含非整数 payload")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    value = int.from_bytes(hashlib.sha256(bytes(payload)).digest()[:8], "big")
    # SQLite INTEGER is signed 64-bit; keep a deterministic positive key.
    value &= (1 << 63) - 1
    return value or 1


def _key_tuple(value: object, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ArtifactSemanticBridgeError(f"{where} 必须是整数列表")
    if any(type(item) is not int for item in value):
        raise ArtifactSemanticBridgeError(f"{where} 必须使用严格整数")
    return tuple(value)


def _strict_int(value: object, *, where: str) -> int:
    if type(value) is not int:
        raise ArtifactSemanticBridgeError(f"{where} 必须是严格整数")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactSemanticBinding:
    """One carrier projection and references to an already learned claim."""

    carrier_code: int
    fact_ordinal: int
    projection_ordinal: int
    projection_key: tuple[int, ...]
    semantic_object_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    directions: tuple[int, ...]
    binding_state: int = BRIDGE_STATE_BOUND

    def __post_init__(self) -> None:
        fields = (
            ("carrier_code", self.carrier_code),
            ("fact_ordinal", self.fact_ordinal),
            ("projection_ordinal", self.projection_ordinal),
            ("binding_state", self.binding_state),
        )
        if any(type(value) is not int or value <= 0 for _name, value in fields):
            raise ValueError("ArtifactSemanticBinding ordinal/state 非法")
        if self.binding_state not in {BRIDGE_STATE_BOUND, BRIDGE_STATE_UNRESOLVED}:
            raise ValueError("ArtifactSemanticBinding state 未注册")
        for name, value in (
                ("projection_key", self.projection_key),
                ("semantic_object_key", self.semantic_object_key),
                ("proposition_key", self.proposition_key),
                ("source_ref", self.source_ref),
                ("scope_key", self.scope_key),
                ("directions", self.directions)):
            if not isinstance(value, tuple) or not value or any(type(item) is not int for item in value):
                raise ValueError(f"{name} 必须是非空整数 tuple")
        if not isinstance(self.evidence_keys, tuple) or not self.evidence_keys:
            raise ValueError("evidence_keys 不得为空")
        if any(not isinstance(item, tuple) or not item or any(type(value) is not int for value in item)
               for item in self.evidence_keys):
            raise ValueError("evidence_keys 必须是整数 tuple 集")

    def natural_key(self) -> tuple[int, ...]:
        return (self.carrier_code, self.fact_ordinal, self.projection_ordinal,
                *self.projection_key, *self.semantic_object_key,
                *self.proposition_key)

    def to_dict(self) -> dict[str, object]:
        value = {
            "binding_state": self.binding_state,
            "carrier_code": self.carrier_code,
            "directions": list(self.directions),
            "evidence_keys": [list(item) for item in self.evidence_keys],
            "fact_ordinal": self.fact_ordinal,
            "format_version": FORMAT_VERSION,
            "natural_key": list(self.natural_key()),
            "projection_key": list(self.projection_key),
            "projection_ordinal": self.projection_ordinal,
            "proposition_key": list(self.proposition_key),
            "record_kind": BRIDGE_RECORD_KIND,
            "scope_key": list(self.scope_key),
            "semantic_object_key": list(self.semantic_object_key),
            "source_ref": list(self.source_ref),
        }
        _integer_tree(value, where="ArtifactSemanticBinding")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ArtifactSemanticBinding":
        if (value.get("record_kind") != BRIDGE_RECORD_KIND
                or value.get("format_version") != FORMAT_VERSION):
            raise ArtifactSemanticBridgeError("bridge record kind/version 漂移")
        evidence = value.get("evidence_keys")
        if not isinstance(evidence, list):
            raise ArtifactSemanticBridgeError("bridge evidence_keys 缺失")
        result = cls(
            _strict_int(value.get("carrier_code"), where="carrier_code"),
            _strict_int(value.get("fact_ordinal"), where="fact_ordinal"),
            _strict_int(value.get("projection_ordinal"), where="projection_ordinal"),
            _key_tuple(value.get("projection_key"), where="projection_key"),
            _key_tuple(value.get("semantic_object_key"), where="semantic_object_key"),
            _key_tuple(value.get("proposition_key"), where="proposition_key"),
            _key_tuple(value.get("source_ref"), where="source_ref"),
            _key_tuple(value.get("scope_key"), where="scope_key"),
            tuple(_key_tuple(item, where="evidence_key") for item in evidence),
            _key_tuple(value.get("directions"), where="directions"),
            _strict_int(value.get("binding_state", BRIDGE_STATE_BOUND), where="binding_state"),
        )
        _integer_tree(value, where="bridge record")
        if tuple(value.get("natural_key", ())) != result.natural_key():
            raise ArtifactSemanticBridgeError("bridge natural_key 漂移")
        return result


def read_artifact_semantic_bridge_course(path: str | Path) -> tuple[ArtifactSemanticBinding, ...]:
    result = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line:
            result.append(ArtifactSemanticBinding.from_dict(json.loads(line)))
    if not result:
        raise ArtifactSemanticBridgeError("bridge course 为空")
    keys = tuple(item.natural_key() for item in result)
    if len(set(keys)) != len(keys):
        raise ArtifactSemanticBridgeError("bridge course natural key 重复")
    return tuple(sorted(result, key=lambda item: item.natural_key()))


def _binding_from_projection(record: TrainedFactCarrierRecord,
                             projection: ArtifactSemanticProjection,
                             ordinal: int) -> ArtifactSemanticBinding:
    metadata = record.metadata
    proposition = _key_tuple(metadata.get("proposition"), where="metadata.proposition")
    source = record.artifact.envelope.source.stable_key()
    scope = record.artifact.envelope.scope.stable_key()
    evidence = tuple(item.stable_key() for item in projection.evidence)
    return ArtifactSemanticBinding(
        record.carrier_code, record.fact_ordinal, ordinal,
        projection.identity.stable_key(), projection.semantic_object.stable_key(),
        proposition, source, scope, evidence,
        tuple(projection.directions),
    )


def _evidence_event_signature(item: EvidenceRecord) -> tuple[object, ...]:
    """Compare Evidence events without treating the remapped Hypothesis as source identity."""
    return (
        item.evidence_id, item.stance, item.reason_key, item.source.stable_key(),
        item.timestamp_seq, item.payload, item.supersedes_evidence_id,
    )


def compile_artifact_semantic_bridge_course(
        carrier_course: str | Path,
        model_database: str | Path,
        run_root: str | Path,
        ) -> dict[str, object]:
    """Compile carrier projections into references to active model objects."""
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root.exists():
        raise ArtifactSemanticBridgeError("bridge run root 必须是新的 K 盘目录")
    root.mkdir(parents=True)
    records = read_trained_fact_carrier_course(carrier_course)
    bindings: list[ArtifactSemanticBinding] = []
    seen: set[tuple[int, ...]] = set()
    with TrainedGraphQueryBridge(Path(model_database).resolve()) as bridge:
        facts = {item.proposition.stable_key(): item for item in bridge.facts}
        for record in records:
            proposition_key = _key_tuple(record.metadata.get("proposition"), where="proposition")
            fact = facts.get(proposition_key)
            if fact is None:
                raise ArtifactSemanticBridgeError("carrier 引用的 proposition 不在 active Core")
            active_signatures = {
                _evidence_event_signature(item)
                for item in bridge.core_runtime.evidence_history(fact.proposition)
            }
            allowed = {fact.predicate.stable_key(), *(item.filler.stable_key() for item in fact.bindings)}
            for ordinal, projection in enumerate(record.artifact.projections, 1):
                if projection.semantic_object.stable_key() not in allowed:
                    raise ArtifactSemanticBridgeError("projection semantic object 不是事实 predicate/filler")
                if projection.envelope_identity != record.artifact.envelope.identity:
                    raise ArtifactSemanticBridgeError("projection envelope identity 漂移")
                if not projection.evidence:
                    raise ArtifactSemanticBridgeError("projection 缺少 Evidence")
                projection_signatures = {
                    _evidence_event_signature(item) for item in projection.evidence
                }
                if not projection_signatures <= active_signatures:
                    raise ArtifactSemanticBridgeError(
                        "projection Evidence 不属于 active Core 历史")
                binding = _binding_from_projection(record, projection, ordinal)
                if binding.natural_key() in seen:
                    raise ArtifactSemanticBridgeError("bridge semantic reference 重复")
                seen.add(binding.natural_key())
                bindings.append(binding)
    course_path = root / "artifact_semantic_bridge.course.jsonl"
    with course_path.open("x", encoding="ascii", newline="\n") as stream:
        for binding in sorted(bindings, key=lambda item: item.natural_key()):
            stream.write(_canonical_bytes(binding.to_dict()).decode("ascii") + "\n")
    manifest = {
        "bridge_course_kind": BRIDGE_COURSE_KIND,
        "binding_count": len(bindings),
        "carrier_codes": sorted({item.carrier_code for item in bindings}),
        "course_sha256": list(hashlib.sha256(course_path.read_bytes()).digest()[:16]),
        "format_version": FORMAT_VERSION,
        "model_sha256": list(hashlib.sha256(Path(model_database).read_bytes()).digest()[:16]),
        "projection_reference_only": 1,
        "semantic_object_copy_count": 0,
    }
    _integer_tree(manifest, where="bridge manifest")
    manifest_path = root / "artifact_semantic_bridge.manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
    graph = materialize_bridge_graph(tuple(bindings), root / "artifact_semantic_bridge_graph.sqlite3")
    receipt = {
        "course": str(course_path),
        "manifest": str(manifest_path),
        "graph": graph,
        "binding_count": len(bindings),
        "semantic_object_copy_count": 0,
        "free_dialogue_complete": 0,
    }
    (root / "receipt.json").write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def _register_bridge_tables(backend) -> None:
    register_extension_table(
        backend, BRIDGE_TABLE,
        [("binding_id", TYPE_INT), ("carrier_code", TYPE_INT),
         ("fact_ordinal", TYPE_INT), ("projection_ordinal", TYPE_INT),
         ("binding_state", TYPE_INT), ("projection_hash", TYPE_INT),
         ("semantic_hash", TYPE_INT), ("proposition_hash", TYPE_INT),
         ("source_hash", TYPE_INT), ("evidence_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("binding_id",), ("semantic_hash",), ("proposition_hash",)],
        recovery_key=("binding_id",),
    )
    register_extension_table(
        backend, BRIDGE_PART_TABLE,
        [("binding_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("binding_id", "field_code", "part_ordinal")],
        recovery_key=("binding_id", "field_code", "part_ordinal"),
    )


def _part_rows(binding_id: int, field_code: int, key: Iterable[int]):
    return [{"binding_id": binding_id, "field_code": field_code,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(key)]


def materialize_bridge_graph(bindings: tuple[ArtifactSemanticBinding, ...],
                             database: str | Path) -> dict[str, int]:
    backend = SQLiteBackend(str(Path(database)), performance_mode="bulk")
    try:
        _register_bridge_tables(backend)
        for binding in bindings:
            binding_id = _key_hash(binding.projection_key, binding.semantic_object_key,
                                   binding.proposition_key)
            row = {
                "binding_id": binding_id,
                "carrier_code": binding.carrier_code,
                "fact_ordinal": binding.fact_ordinal,
                "projection_ordinal": binding.projection_ordinal,
                "binding_state": binding.binding_state,
                "projection_hash": _key_hash(binding.projection_key),
                "semantic_hash": _key_hash(binding.semantic_object_key),
                "proposition_hash": _key_hash(binding.proposition_key),
                "source_hash": _key_hash(binding.source_ref),
                "evidence_count": len(binding.evidence_keys),
            }
            existing = backend.select(BRIDGE_TABLE, where={"binding_id": binding_id})
            if existing and existing[0] != row:
                raise ArtifactSemanticBridgeError("bridge binding hash collision")
            if not existing:
                backend.insert(BRIDGE_TABLE, row)
                parts = [
                    (1, binding.projection_key), (2, binding.semantic_object_key),
                    (3, binding.proposition_key), (4, binding.source_ref),
                    (5, binding.scope_key), (7, binding.directions),
                ]
                for ordinal, evidence in enumerate(binding.evidence_keys):
                    parts.append((100 + ordinal, evidence))
                for field_code, key in parts:
                    backend.insert_many(BRIDGE_PART_TABLE,
                                        _part_rows(binding_id, field_code, key))
        backend.commit()
        with sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True) as conn:
            count = int(conn.execute(f"SELECT COUNT(*) FROM {BRIDGE_TABLE}").fetchone()[0])
            part_count = int(conn.execute(f"SELECT COUNT(*) FROM {BRIDGE_PART_TABLE}").fetchone()[0])
            duplicate = int(conn.execute(
                f"SELECT COUNT(*) FROM (SELECT binding_id,field_code,part_ordinal,COUNT(*) n FROM {BRIDGE_PART_TABLE} GROUP BY binding_id,field_code,part_ordinal HAVING n>1)"
            ).fetchone()[0])
        if duplicate:
            raise ArtifactSemanticBridgeError("bridge part natural key 重复")
        return {"binding_count": count, "part_count": part_count,
                "duplicate_part_groups": duplicate}
    finally:
        backend.close()


class ArtifactSemanticTrainingRuntime:
    """Attach bridge references to one formal TrainContext without copying targets."""

    def __init__(self, context, course_paths: tuple[str | Path, ...]):
        self.context = context
        self.course_paths = tuple(Path(item).resolve() for item in course_paths)
        self.bound_count = 0
        self.unresolved_count = 0
        self.reference_count = 0
        # Formal training installs the bridge before the language stages so
        # the course is part of the same TrainContext, but target semantic
        # identities may only exist after those stages materialize them.  Keep
        # consumption explicit and idempotent so a caller can defer it until
        # the authoritative graph is ready without duplicating rows.
        self._consumed_binding_ids: set[int] = set()
        _register_bridge_tables(context.backend)
        self._semantic_predicate = context.graph_ontology.materialize(
            relation_concept_identity((BRIDGE_RELATION_PROJECTION_SEMANTIC,)))
        self._proposition_predicate = context.graph_ontology.materialize(
            relation_concept_identity((BRIDGE_RELATION_PROJECTION_PROPOSITION,)))
        self._hypothesis_predicate = context.graph_ontology.materialize(
            relation_concept_identity((BRIDGE_RELATION_PROJECTION_HYPOTHESIS,)))

    def consume(self) -> dict[str, int]:
        for path in self.course_paths:
            for binding in read_artifact_semantic_bridge_course(path):
                self._consume_binding(binding)
        return self.report()

    def _consume_binding(self, binding: ArtifactSemanticBinding) -> None:
        binding_id = _key_hash(binding.projection_key, binding.semantic_object_key,
                               binding.proposition_key)
        if binding_id in self._consumed_binding_ids:
            return
        semantic = ObjectIdentity.from_stable_key(binding.semantic_object_key)
        proposition = ObjectIdentity.from_stable_key(binding.proposition_key)
        source = SourceRef.from_stable_key(binding.source_ref)
        scope = ScopeIdentity.from_stable_key(binding.scope_key)
        semantic_ref = self.context.graph_ontology.resolve(semantic)
        proposition_ref = self.context.graph_ontology.resolve(proposition)
        state = BRIDGE_STATE_BOUND if semantic_ref is not None and proposition_ref is not None else BRIDGE_STATE_UNRESOLVED
        if state == BRIDGE_STATE_BOUND:
            projection = self.context.graph_ontology.materialize(
                ObjectIdentity.from_stable_key(binding.projection_key))
            self.context.graph_ontology.relate(
                self._semantic_predicate, projection, semantic_ref,
                scope=scope, provenance_kind=EPI_STRUCTURED,
                content_version=FORMAT_VERSION)
            self.context.graph_ontology.relate(
                self._proposition_predicate, projection, proposition_ref,
                scope=scope, provenance_kind=EPI_STRUCTURED,
                content_version=FORMAT_VERSION)
            hypothesis_keys = set()
            for evidence_key in binding.evidence_keys:
                evidence = EvidenceRecord.from_stable_key(evidence_key)
                hypothesis_keys.add(evidence.hypothesis.object_identity())
            for hypothesis in sorted(hypothesis_keys, key=lambda item: item.stable_key()):
                hypothesis_ref = self.context.graph_ontology.materialize(hypothesis)
                self.context.graph_ontology.relate(
                    self._hypothesis_predicate, projection, hypothesis_ref,
                    scope=scope, provenance_kind=EPI_STRUCTURED,
                    content_version=FORMAT_VERSION)
            self.bound_count += 1
        else:
            self.unresolved_count += 1
        row = {
            "binding_id": binding_id, "carrier_code": binding.carrier_code,
            "fact_ordinal": binding.fact_ordinal,
            "projection_ordinal": binding.projection_ordinal,
            "binding_state": state,
            "projection_hash": _key_hash(binding.projection_key),
            "semantic_hash": _key_hash(binding.semantic_object_key),
            "proposition_hash": _key_hash(binding.proposition_key),
            "source_hash": _key_hash(source.stable_key()),
            "evidence_count": len(binding.evidence_keys),
        }
        existing = self.context.backend.select(BRIDGE_TABLE, where={"binding_id": binding_id})
        if existing:
            if existing[0] != row:
                raise ArtifactSemanticBridgeError("formal bridge binding 状态漂移")
        else:
            self.context.backend.insert(BRIDGE_TABLE, row)
            for field_code, key in (
                    (1, binding.projection_key), (2, binding.semantic_object_key),
                    (3, binding.proposition_key), (4, binding.source_ref),
                    (5, binding.scope_key), (7, binding.directions)):
                self.context.backend.insert_many(
                    BRIDGE_PART_TABLE, _part_rows(binding_id, field_code, key))
            for ordinal, evidence_key in enumerate(binding.evidence_keys):
                self.context.backend.insert_many(
                    BRIDGE_PART_TABLE,
                    _part_rows(binding_id, 100 + ordinal, evidence_key))
        self.reference_count += 1
        self._consumed_binding_ids.add(binding_id)

    def report(self) -> dict[str, int]:
        return {"course_count": len(self.course_paths),
                "reference_count": self.reference_count,
                "bound_count": self.bound_count,
                "unresolved_count": self.unresolved_count,
                "semantic_object_copy_count": 0}


def build_artifact_semantic_training_runtime(
        context, course_paths: Iterable[str | Path], *, consume: bool = True):
    runtime = ArtifactSemanticTrainingRuntime(context, tuple(course_paths))
    if consume:
        runtime.consume()
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-course", required=True)
    parser.add_argument("--model-database", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    result = compile_artifact_semantic_bridge_course(
        args.carrier_course, args.model_database, args.run_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


__all__ = [
    "ArtifactSemanticBinding", "ArtifactSemanticBridgeError",
    "ArtifactSemanticTrainingRuntime", "build_artifact_semantic_training_runtime",
    "compile_artifact_semantic_bridge_course", "main",
    "materialize_bridge_graph", "read_artifact_semantic_bridge_course",
]
