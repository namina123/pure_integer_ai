"""Typed canonical FT24B snapshot for the public sparse QA runtime."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments import ph2_dataset_core
from pure_integer_ai.experiments import ph2_dataset_owner_records
from pure_integer_ai.experiments import ph2_dataset_records
from pure_integer_ai.experiments import ph2_evaluation_public_source
from pure_integer_ai.experiments import ph2_w03_v2_public_query
from pure_integer_ai.experiments import ph2_w03_w04_public_bridge_contract
from pure_integer_ai.experiments import ph2_w03_w04_w05_question_feature_catalog
from pure_integer_ai.experiments import ph2_w03_w04_w05_question_feature_registry_contract
from pure_integer_ai.experiments import ph2_w03_w04_w05_raw_question_alias_contract
from pure_integer_ai.experiments import ph2_w03_w04_w05_raw_question_contract
from pure_integer_ai.experiments import ph2_w03_w04_w05_raw_question_implicit
from pure_integer_ai.experiments import ph2_w03_w04_w05_vertical_contract
from pure_integer_ai.experiments import ph2_w04_v2_public_query_contract
from pure_integer_ai.experiments import ph2_w04_w05_public_bridge_contract
from pure_integer_ai.experiments import ph2_w05_v2_public_query_contract
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel import source_binding
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    QUESTION_ALIAS_FRAME_ANCHOR_SHA256,
    build_raw_question_alias_frame_anchor_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    QUESTION_CONSTRUCTION_INDEX_SHA256,
    build_raw_question_construction_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    QUESTION_FEATURE_INDEX_SHA256,
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    QUESTION_FEATURE_REGISTRY_SHA256,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    RawQuestionFeatureRegistry,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch import (
    QUESTION_SPARSE_DISPATCH_SHA256,
    build_raw_question_sparse_dispatch_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    SPARSE_QA_RUNTIME_SHA256,
    SparseQARuntime,
    assemble_sparse_qa_runtime,
    build_public_sparse_qa_runtime,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT = (
    REPOSITORY / "data/ph2/sparse_qa_runtime_snapshot_v1.json")
SPARSE_QA_RUNTIME_SNAPSHOT_FORMAT = (
    "PURE_INTEGER_AI_PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT")
SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION = 1
SPARSE_QA_RUNTIME_SNAPSHOT_MAX_BYTES = 2 * 1024 * 1024
SPARSE_QA_RUNTIME_SNAPSHOT_MAX_NODES = 10000
SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256 = (
    "bb41b198c4d1f77051f2c72dcf89a7f6d8cb97c396313e4d9156006591bb0e60")
SPARSE_QA_RUNTIME_SNAPSHOT_BOUNDARY = (
    ("payload_root", "FT16_PUBLIC_REGISTRY_ONLY"),
    ("derived_indexes", "FT17_FT20_REBUILT_AND_VERIFIED"),
    ("runtime", "FT22_REASSEMBLED_AND_VERIFIED"),
    ("private_formal_data", "FORBIDDEN"),
    ("backend_lifecycle_owner", "FORBIDDEN"),
    ("python_object_cache", "FORBIDDEN"),
    ("partial_load", "FORBIDDEN"),
    ("invalid_snapshot", "EXPLICIT_FULL_REBUILD_ONLY"),
)
PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS = tuple(sorted((
    "data/ph2/authored_primitive_atomic_bridge_map_generalization_v1.jsonl.sample",
    "data/ph2/authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample",
    "data/ph2/authored_primitive_atomic_bridge_seed_generalization_v1.jsonl.sample",
    "data/ph2/authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample",
    "data/ph2/authored_semantic_primitive_bridge_generalization_v1.jsonl.sample",
    "data/ph2/authored_semantic_primitive_bridge_three_role_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_cause_generalization_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_effect_generalization_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_implicit_reason_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_implicit_result_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_three_role_actor_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_three_role_implicit_actor_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_three_role_implicit_location_v1.jsonl.sample",
    "data/ph2/authored_vertical_question_three_role_location_v1.jsonl.sample",
)))


# object-model: exception
class SparseQARuntimeSnapshotError(ValueError):
    """The typed snapshot is missing, noncanonical, or identity-incompatible."""


_VALUE_TYPES = (
    ph2_dataset_core.CanonicalJsonObject,
    ph2_dataset_core.StableRecordKey,
    ph2_dataset_owner_records.TeacherEvidenceRecord,
    ph2_dataset_records.ObservationRecord,
    ph2_dataset_records.SourceRefRecord,
    source_binding.EvaluationSourceBinding,
    source_binding.EvaluationSourceSlice,
    ph2_evaluation_public_source.EvaluationPublicBatch,
    ph2_evaluation_public_source.EvaluationPublicObservationEvidencePair,
    ph2_evaluation_public_source.EvaluationPublicSourceRecord,
    ph2_w03_v2_public_query.W03V2PublicCandidateProjection,
    ph2_w03_v2_public_query.W03V2PublicQuery,
    ph2_w03_v2_public_query.W03V2PublicQueryResult,
    ph2_w03_w04_public_bridge_contract.W03W04PublicBridgeLink,
    ph2_w03_w04_public_bridge_contract.W03W04PublicBridgeQuery,
    ph2_w03_w04_public_bridge_contract.W03W04PublicBridgeResult,
    ph2_w03_w04_w05_question_feature_catalog.RawQuestionFeatureCatalog,
    ph2_w03_w04_w05_question_feature_registry_contract.RawQuestionFeatureRegistry,
    ph2_w03_w04_w05_question_feature_registry_contract.RawQuestionFeatureRegistryEntry,
    ph2_w03_w04_w05_raw_question_alias_contract.LearnedPredicateAliasBridge,
    ph2_w03_w04_w05_raw_question_alias_contract.LearnedPredicateAliasRoute,
    ph2_w03_w04_w05_raw_question_contract.RawQuestionConstruction,
    ph2_w03_w04_w05_raw_question_contract.RawQuestionConstructionSegment,
    ph2_w03_w04_w05_raw_question_contract.RawQuestionPattern,
    ph2_w03_w04_w05_raw_question_contract.RawQuestionPatternSegment,
    ph2_w03_w04_w05_raw_question_implicit.W03W04W05ImplicitQuestionBundle,
    ph2_w03_w04_w05_vertical_contract.W03W04W05VerticalLink,
    ph2_w03_w04_w05_vertical_contract.W03W04W05VerticalQuery,
    ph2_w03_w04_w05_vertical_contract.W03W04W05VerticalResult,
    ph2_w04_v2_public_query_contract.W04V2PublicCandidateProjection,
    ph2_w04_v2_public_query_contract.W04V2PublicGenerationProjection,
    ph2_w04_v2_public_query_contract.W04V2PublicQuery,
    ph2_w04_v2_public_query_contract.W04V2PublicQueryResult,
    ph2_w04_w05_public_bridge_contract.W04W05PublicBridgeLink,
    ph2_w04_w05_public_bridge_contract.W04W05PublicBridgeQuery,
    ph2_w04_w05_public_bridge_contract.W04W05PublicBridgeResult,
    ph2_w05_v2_public_query_contract.W05V2PublicCandidateProjection,
    ph2_w05_v2_public_query_contract.W05V2PublicGenerationProjection,
    ph2_w05_v2_public_query_contract.W05V2PublicOccurrenceProjection,
    ph2_w05_v2_public_query_contract.W05V2PublicQuery,
    ph2_w05_v2_public_query_contract.W05V2PublicQueryResult,
    ph2_w05_v2_public_query_contract.W05V2PublicRoleBindingProjection,
)


def _type_name(value_type: type[object]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


if len(set(_VALUE_TYPES)) != len(_VALUE_TYPES):
    raise RuntimeError("FT24B value type allowlist contains duplicates")
_VALUE_TYPES_BY_NAME = {
    _type_name(value_type): value_type for value_type in _VALUE_TYPES}
_VALUE_TYPE_NAMES = tuple(sorted(_VALUE_TYPES_BY_NAME))
if hashlib.sha256(canonical_json_bytes(list(_VALUE_TYPE_NAMES))).hexdigest() != (
        SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256):
    raise RuntimeError("FT24B value type inventory commitment drifted")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise SparseQARuntimeSnapshotError(
            f"{where} is not a canonical SHA-256")
    return value


def _exact_object(
        value: object,
        keys: tuple[str, ...],
        *,
        where: str,
        ) -> dict[str, object]:
    if (not isinstance(value, dict)
            or set(value) != set(keys)
            or any(not isinstance(key, str) for key in value)):
        raise SparseQARuntimeSnapshotError(
            f"{where} fields are not canonical")
    return value


def _boundary_value() -> list[dict[str, str]]:
    return [
        {"capability": capability, "status": status}
        for capability, status in SPARSE_QA_RUNTIME_SNAPSHOT_BOUNDARY
    ]


def _parent_identities() -> dict[str, str]:
    return {
        "ft16_registry_sha256": QUESTION_FEATURE_REGISTRY_SHA256,
        "ft17_feature_index_sha256": QUESTION_FEATURE_INDEX_SHA256,
        "ft18_construction_index_sha256": QUESTION_CONSTRUCTION_INDEX_SHA256,
        "ft19_alias_frame_anchor_sha256": (
            QUESTION_ALIAS_FRAME_ANCHOR_SHA256),
        "ft20_sparse_dispatch_sha256": QUESTION_SPARSE_DISPATCH_SHA256,
        "ft21_reused_construction_index_sha256": (
            QUESTION_CONSTRUCTION_INDEX_SHA256),
        "ft22_runtime_sha256": SPARSE_QA_RUNTIME_SHA256,
    }


def public_sparse_qa_source_artifact_identities(
        repository: str | Path = REPOSITORY,
        ) -> list[dict[str, str]]:
    """Hash every public authored source consumed by the frozen registry."""
    root = Path(repository).resolve()
    values = []
    for relative in PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SparseQARuntimeSnapshotError(
                "FT24B source artifact escaped the repository") from error
        if not path.is_file():
            raise SparseQARuntimeSnapshotError(
                f"FT24B source artifact is missing: {relative}")
        values.append({
            "path": relative,
            "sha256": _sha256_path(path),
        })
    return values


def _encode_value_graph(root: RawQuestionFeatureRegistry) -> dict[str, object]:
    nodes: list[dict[str, object] | None] = []
    positions: dict[int, int] = {}

    def encode(value: object) -> object:
        if value is None or isinstance(value, str) or type(value) is int:
            return value
        if isinstance(value, bool):
            raise SparseQARuntimeSnapshotError(
                "FT24B registry contains a non-integer bool")
        if not (is_dataclass(value) or isinstance(value, (tuple, bytes))):
            raise SparseQARuntimeSnapshotError(
                f"FT24B registry contains unsupported {type(value).__name__}")
        identity = id(value)
        if identity in positions:
            return {"ref": positions[identity]}
        ordinal = len(nodes)
        positions[identity] = ordinal
        nodes.append(None)
        if isinstance(value, tuple):
            node = {
                "items": [encode(item) for item in value],
                "kind": "TUPLE",
            }
        elif isinstance(value, bytes):
            node = {"hex": value.hex(), "kind": "BYTES"}
        else:
            value_type = type(value)
            if value_type not in _VALUE_TYPES:
                raise SparseQARuntimeSnapshotError(
                    f"FT24B value type is not allowed: {_type_name(value_type)}")
            params = getattr(value_type, "__dataclass_params__", None)
            value_fields = fields(value)
            if (params is None or not params.frozen
                    or any(not field.init for field in value_fields)):
                raise SparseQARuntimeSnapshotError(
                    f"FT24B value type is not constructor-restorable: "
                    f"{_type_name(value_type)}")
            node = {
                "kind": "STRUCT",
                "type": _type_name(value_type),
                "values": [
                    encode(getattr(value, field.name))
                    for field in value_fields
                ],
            }
        nodes[ordinal] = node
        return {"ref": ordinal}

    root_ref = encode(root)
    if any(node is None for node in nodes):
        raise SparseQARuntimeSnapshotError(
            "FT24B value graph contains an incomplete node")
    return {
        "nodes": nodes,
        "root": root_ref,
    }


def _decode_value_graph(payload: object) -> RawQuestionFeatureRegistry:
    graph = _exact_object(payload, ("nodes", "root"), where="FT24B payload")
    nodes = graph["nodes"]
    if (not isinstance(nodes, list) or not nodes
            or len(nodes) > SPARSE_QA_RUNTIME_SNAPSHOT_MAX_NODES):
        raise SparseQARuntimeSnapshotError(
            "FT24B node table is empty or exceeds its frozen boundary")
    unresolved = object()
    cache: list[object] = [unresolved] * len(nodes)
    visiting: set[int] = set()

    def decode_atom(value: object) -> object:
        if value is None or isinstance(value, str) or type(value) is int:
            return value
        if isinstance(value, bool):
            raise SparseQARuntimeSnapshotError(
                "FT24B payload contains a non-integer bool")
        reference = _exact_object(value, ("ref",), where="FT24B reference")
        ordinal = reference["ref"]
        if type(ordinal) is not int or not 0 <= ordinal < len(nodes):
            raise SparseQARuntimeSnapshotError(
                "FT24B node reference is invalid")
        return decode_node(ordinal)

    def decode_node(ordinal: int) -> object:
        if cache[ordinal] is not unresolved:
            return cache[ordinal]
        if ordinal in visiting:
            raise SparseQARuntimeSnapshotError(
                "FT24B value graph contains a cycle")
        visiting.add(ordinal)
        node = nodes[ordinal]
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise SparseQARuntimeSnapshotError("FT24B node is invalid")
        kind = node["kind"]
        if kind == "TUPLE":
            value_node = _exact_object(
                node, ("items", "kind"), where="FT24B tuple node")
            items = value_node["items"]
            if not isinstance(items, list):
                raise SparseQARuntimeSnapshotError(
                    "FT24B tuple items are invalid")
            restored: object = tuple(decode_atom(item) for item in items)
        elif kind == "BYTES":
            value_node = _exact_object(
                node, ("hex", "kind"), where="FT24B bytes node")
            hexadecimal = value_node["hex"]
            if (not isinstance(hexadecimal, str)
                    or len(hexadecimal) % 2
                    or any(item not in "0123456789abcdef"
                           for item in hexadecimal)):
                raise SparseQARuntimeSnapshotError(
                    "FT24B bytes node is not canonical hexadecimal")
            restored = bytes.fromhex(hexadecimal)
        elif kind == "STRUCT":
            value_node = _exact_object(
                node,
                ("kind", "type", "values"),
                where="FT24B struct node",
            )
            type_name = value_node["type"]
            values = value_node["values"]
            if (not isinstance(type_name, str)
                    or type_name not in _VALUE_TYPES_BY_NAME
                    or not isinstance(values, list)):
                raise SparseQARuntimeSnapshotError(
                    "FT24B struct type or values are invalid")
            value_type = _VALUE_TYPES_BY_NAME[type_name]
            value_fields = fields(value_type)
            if len(values) != len(value_fields):
                raise SparseQARuntimeSnapshotError(
                    "FT24B struct field count drifted")
            restored_values = tuple(decode_atom(item) for item in values)
            try:
                restored = value_type(*restored_values)
            except Exception as error:
                raise SparseQARuntimeSnapshotError(
                    f"FT24B struct restoration failed: {type_name}") from error
        else:
            raise SparseQARuntimeSnapshotError(
                "FT24B node kind is not allowed")
        visiting.remove(ordinal)
        cache[ordinal] = restored
        return restored

    restored = decode_atom(graph["root"])
    if any(value is unresolved for value in cache):
        raise SparseQARuntimeSnapshotError(
            "FT24B node table contains unreachable values")
    if not isinstance(restored, RawQuestionFeatureRegistry):
        raise SparseQARuntimeSnapshotError(
            "FT24B payload root is not the public registry")
    return restored


def _runtime_from_registry(
        registry: RawQuestionFeatureRegistry,
        ) -> SparseQARuntime:
    if registry.identity_sha256 != QUESTION_FEATURE_REGISTRY_SHA256:
        raise SparseQARuntimeSnapshotError(
            "FT24B registry parent identity drifted")
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


def public_sparse_qa_runtime_snapshot_value(
        runtime: SparseQARuntime,
        *,
        repository: str | Path = REPOSITORY,
        ) -> dict[str, object]:
    """Build the canonical envelope without serializing derived indexes."""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("FT24B snapshot input is not a sparse QA runtime")
    if runtime.identity_sha256 != SPARSE_QA_RUNTIME_SHA256:
        raise SparseQARuntimeSnapshotError(
            "FT24B runtime identity is not the frozen public runtime")
    registry = (
        runtime.dispatch_index.anchor_index.construction_index.feature_index.registry)
    payload = _encode_value_graph(registry)
    return {
        "boundary_flags": _boundary_value(),
        "format": SPARSE_QA_RUNTIME_SNAPSHOT_FORMAT,
        "frozen_parent_sha256": _parent_identities(),
        "payload": payload,
        "payload_sha256": _sha256_bytes(canonical_json_bytes(payload)),
        "runtime_identity_sha256": runtime.identity_sha256,
        "schema_version": SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION,
        "source_artifacts": public_sparse_qa_source_artifact_identities(
            repository),
        "value_type_inventory_sha256": (
            SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256),
    }


def write_public_sparse_qa_runtime_snapshot(
        runtime: SparseQARuntime,
        path: str | Path = PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT,
        *,
        repository: str | Path = REPOSITORY,
        ) -> Path:
    """Publish one immutable snapshot or accept an identical existing file."""
    target = Path(path)
    value = public_sparse_qa_runtime_snapshot_value(
        runtime, repository=repository)
    payload = canonical_json_bytes(value) + b"\n"
    if len(payload) > SPARSE_QA_RUNTIME_SNAPSHOT_MAX_BYTES:
        raise SparseQARuntimeSnapshotError(
            "FT24B snapshot exceeds its frozen size boundary")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise SparseQARuntimeSnapshotError(
                "FT24B snapshot already exists with different bytes")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise SparseQARuntimeSnapshotError(
            "FT24B snapshot could not be published") from error
    return target


def load_public_sparse_qa_runtime_snapshot(
        path: str | Path = PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT,
        *,
        repository: str | Path = REPOSITORY,
        ) -> SparseQARuntime:
    """Fail closed unless the complete typed snapshot and all parents match."""
    try:
        source = Path(path)
        if (not source.is_file()
                or not 1 < source.stat().st_size
                <= SPARSE_QA_RUNTIME_SNAPSHOT_MAX_BYTES):
            raise SparseQARuntimeSnapshotError(
                "FT24B snapshot file boundary is invalid")
        raw = source.read_bytes()
        if (not raw.endswith(b"\n") or raw.endswith(b"\n\n")
                or len(raw) <= 1):
            raise SparseQARuntimeSnapshotError(
                "FT24B snapshot newline boundary drifted")
        value = ph2_dataset_core.parse_canonical_json_bytes(
            raw[:-1], require_object=True)
        envelope = _exact_object(
            value,
            (
                "boundary_flags",
                "format",
                "frozen_parent_sha256",
                "payload",
                "payload_sha256",
                "runtime_identity_sha256",
                "schema_version",
                "source_artifacts",
                "value_type_inventory_sha256",
            ),
            where="FT24B envelope",
        )
        if (type(envelope["schema_version"]) is not int
                or envelope["schema_version"]
                != SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION):
            raise SparseQARuntimeSnapshotError(
                "FT24B schema version is incompatible")
        if envelope["format"] != SPARSE_QA_RUNTIME_SNAPSHOT_FORMAT:
            raise SparseQARuntimeSnapshotError(
                "FT24B snapshot format is incompatible")
        if envelope["boundary_flags"] != _boundary_value():
            raise SparseQARuntimeSnapshotError(
                "FT24B snapshot boundary flags drifted")
        if envelope["frozen_parent_sha256"] != _parent_identities():
            raise SparseQARuntimeSnapshotError(
                "FT24B frozen parent identities drifted")
        if envelope["runtime_identity_sha256"] != SPARSE_QA_RUNTIME_SHA256:
            raise SparseQARuntimeSnapshotError(
                "FT24B runtime identity drifted")
        if (envelope["value_type_inventory_sha256"]
                != SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256):
            raise SparseQARuntimeSnapshotError(
                "FT24B value type inventory drifted")
        if (envelope["source_artifacts"]
                != public_sparse_qa_source_artifact_identities(repository)):
            raise SparseQARuntimeSnapshotError(
                "FT24B public source artifact identities drifted")
        payload = envelope["payload"]
        payload_sha256 = _sha256(
            envelope["payload_sha256"], where="FT24B payload")
        if _sha256_bytes(canonical_json_bytes(payload)) != payload_sha256:
            raise SparseQARuntimeSnapshotError(
                "FT24B payload commitment drifted")
        registry = _decode_value_graph(payload)
        if _encode_value_graph(registry) != payload:
            raise SparseQARuntimeSnapshotError(
                "FT24B value graph encoding drifted")
        runtime = _runtime_from_registry(registry)
    except SparseQARuntimeSnapshotError:
        raise
    except Exception as error:
        raise SparseQARuntimeSnapshotError(
            "FT24B snapshot could not be restored") from error
    if runtime.identity_sha256 != envelope["runtime_identity_sha256"]:
        raise SparseQARuntimeSnapshotError(
            "FT24B restored runtime identity drifted")
    return runtime


def load_or_rebuild_public_sparse_qa_runtime(
        path: str | Path = PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT,
        *,
        repository: str | Path = REPOSITORY,
        work_root: str | Path | None = None,
        ) -> SparseQARuntime:
    """Load the complete snapshot or explicitly perform one clean rebuild."""
    try:
        return load_public_sparse_qa_runtime_snapshot(
            path, repository=repository)
    except SparseQARuntimeSnapshotError:
        return build_public_sparse_qa_runtime(work_root)


__all__ = [
    "PUBLIC_SPARSE_QA_RUNTIME_SNAPSHOT",
    "PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS",
    "SPARSE_QA_RUNTIME_SNAPSHOT_BOUNDARY",
    "SPARSE_QA_RUNTIME_SNAPSHOT_FORMAT",
    "SPARSE_QA_RUNTIME_SNAPSHOT_MAX_BYTES",
    "SPARSE_QA_RUNTIME_SNAPSHOT_MAX_NODES",
    "SPARSE_QA_RUNTIME_SNAPSHOT_SCHEMA_VERSION",
    "SPARSE_QA_RUNTIME_SNAPSHOT_VALUE_TYPES_SHA256",
    "SparseQARuntimeSnapshotError",
    "load_or_rebuild_public_sparse_qa_runtime",
    "load_public_sparse_qa_runtime_snapshot",
    "public_sparse_qa_runtime_snapshot_value",
    "public_sparse_qa_source_artifact_identities",
    "write_public_sparse_qa_runtime_snapshot",
]
