"""PH2-D03-V2 的严格记录 schema、载体结构和跨记录隔离校验。"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactManifest,
    DatasetContractError,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    RECORD_ARTIFACT_MANIFEST,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    SPLITS,
    W_STAGES,
    _sha256,
)


V2_FORMAT_VERSION = 2
V2_SCHEMA_VERSION = 2
V2_COURSE_VERSION = 2
V2_RECORD_KINDS = (
    RECORD_SOURCE_REF,
    RECORD_OBSERVATION,
    RECORD_TEACHER_EVIDENCE,
    RECORD_EVALUATOR_LABEL,
    RECORD_ARTIFACT_MANIFEST,
)
V2_OBSERVATION_STAGES = W_STAGES[1:]
V2_CARRIER_KINDS = (
    "plain_text",
    "markdown",
    "html",
    "source_code",
    "mathematics",
    "table",
    "document_container",
    "quotation_embedding",
    "ocr_asr_transcript",
)
V2_PUBLIC_LICENSES = (
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
)
V2_SOURCE_KEYS = (
    "AUTHORED_CC0",
    "CONCEPTNET_5_7_0",
    "UD_ZH_GSDSIMP_R2_18",
    "WIKIDATA_REVISION_V1",
    "ZHWIKIPEDIA_20260701",
    "ZHWIKTIONARY_20260701",
)
V2_SOURCE_LICENSES = {
    "AUTHORED_CC0": ("CC0-1.0",),
    "CONCEPTNET_5_7_0": ("CC-BY-4.0", "CC-BY-SA-4.0"),
    "UD_ZH_GSDSIMP_R2_18": ("CC-BY-SA-4.0",),
    "WIKIDATA_REVISION_V1": ("CC0-1.0",),
    "ZHWIKIPEDIA_20260701": ("CC-BY-SA-4.0",),
    "ZHWIKTIONARY_20260701": ("CC-BY-SA-4.0",),
}


SOURCE_REF_FIELDS = (
    "artifact_key", "attribution", "course_version", "dataset_key",
    "format_version", "license_id", "local_sha256", "official_url",
    "parser_version", "record_kind", "record_ordinal", "redistribution_policy",
    "revision_id", "schema_version", "snapshot_id", "source_cluster_key",
    "source_identity", "source_key", "source_span", "stable_key",
    "upstream_checksum",
)
OBSERVATION_FIELDS = (
    "artifact_key", "content_group_key", "course_version", "dataset_key",
    "dedup_cluster_key", "epistemic_role", "format_version", "language",
    "license_partition", "logical_order", "payload_kind", "perturbation_kind",
    "prerequisite_keys", "record_kind", "representation", "sample_role",
    "schema_version", "shape_group_key", "source_ref_key", "split",
    "stable_key", "substage", "supersedes_key", "template_group_key",
    "typed_payload", "w_stage",
)
TEACHER_EVIDENCE_FIELDS = (
    "artifact_key", "course_version", "dataset_key", "evidence_kind",
    "format_version", "observation_key", "owner_key", "record_kind",
    "schema_version", "source_ref_key", "stable_key", "typed_evidence",
    "visible_from_stage", "withdrawal_level",
)
EVALUATOR_LABEL_FIELDS = (
    "artifact_key", "budget_units", "course_version", "dataset_key",
    "dimension_key", "evaluator_version", "expected_payload", "expected_state",
    "format_version", "observation_key", "owner_key", "owner_mode",
    "record_kind", "schema_version", "stable_key", "visible_stage",
)
ARTIFACT_MANIFEST_FIELDS = (
    "adapter_version", "artifact_version", "course_version", "dataset_key",
    "earliest_invalidated_stage", "files", "format_version", "generator_version",
    "license_partition", "parser_version", "prerequisite_manifest_keys", "record_count",
    "record_kind", "redistribution_policy", "schema_version", "source_cluster_keys",
    "source_key", "splits", "stable_key", "w_stages",
)
ARTIFACT_FILE_FIELDS = (
    "content_sha256", "content_size_bytes", "first_record_key", "last_record_key",
    "license_partition", "owner_kind", "record_count", "record_kind",
    "relative_path", "source_cluster_keys", "split", "transport_sha256",
    "transport_size_bytes",
)
_RECORD_FIELDS = {
    RECORD_SOURCE_REF: SOURCE_REF_FIELDS,
    RECORD_OBSERVATION: OBSERVATION_FIELDS,
    RECORD_TEACHER_EVIDENCE: TEACHER_EVIDENCE_FIELDS,
    RECORD_EVALUATOR_LABEL: EVALUATOR_LABEL_FIELDS,
    RECORD_ARTIFACT_MANIFEST: ARTIFACT_MANIFEST_FIELDS,
}


def _exact(value: Any, fields: tuple[str, ...], *, where: str) -> dict[str, Any]:
    """要求 object 的字段集合与 v2 schema 完全相等。"""
    if not isinstance(value, dict) or set(value) != set(fields):
        raise DatasetContractError(f"{where} 字段不精确")
    return value


def _key(value: Any, *, where: str) -> tuple[int, ...]:
    """解析一个稳定整数键并返回其不可变表示。"""
    return StableRecordKey.from_value(value, where=where).components


def _keys(value: Any, *, where: str, allow_empty: bool = False) -> tuple[tuple[int, ...], ...]:
    """解析去重稳定整数键列表。"""
    if not isinstance(value, list):
        raise DatasetContractError(f"{where} 必须是数组")
    result = tuple(_key(item, where=f"{where}[]") for item in value)
    if not allow_empty and not result:
        raise DatasetContractError(f"{where} 不能为空")
    if len(result) != len(set(result)):
        raise DatasetContractError(f"{where} 不得重复")
    return result


def _validate_source_span(value: Any, *, source_key: str) -> None:
    """冻结文档/实体分簇和来源定位，不允许退化成只有文本的 source ref。"""
    raw = _exact(value, (
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    ), where="v2 source_span")
    if raw["locator_kind"] not in ("page", "entity", "sentence", "file", "record"):
        raise DatasetContractError("v2 source_span locator_kind 非法")
    if not isinstance(raw["locator_value"], str) or not raw["locator_value"]:
        raise DatasetContractError("v2 source_span locator_value 非法")
    if (type(raw["span_start"]) is not int or type(raw["span_end"]) is not int
            or raw["span_start"] < 0 or raw["span_end"] < raw["span_start"]):
        raise DatasetContractError("v2 source_span span 非法")
    _key(raw["document_cluster_key"], where="v2 document_cluster_key")
    _key(raw["entity_graph_cluster_key"], where="v2 entity_graph_cluster_key")
    if source_key not in V2_SOURCE_KEYS:
        raise DatasetContractError("v2 source_key 不在 successor allowlist")


def _validate_carrier(value: Any, *, representation: str) -> None:
    """校验保留载体节点/边/span 的最小结构，避免 Markdown/HTML 被展平。"""
    raw = _exact(value, ("carrier", "language_payload"), where="v2 typed_payload")
    carrier = _exact(raw["carrier"], (
        "carrier_kind", "edges", "nodes", "raw_text_sha256", "root_node_keys",
    ), where="v2 carrier")
    kind = carrier["carrier_kind"]
    if kind not in V2_CARRIER_KINDS or representation != kind:
        raise DatasetContractError("v2 carrier kind/representation 不一致")
    _sha256(carrier["raw_text_sha256"], where="v2 carrier raw_text_sha256")
    node_values = carrier["nodes"]
    if not isinstance(node_values, list) or not node_values:
        raise DatasetContractError("v2 carrier nodes 不能为空")
    node_keys: list[tuple[int, ...]] = []
    parents: dict[tuple[int, ...], tuple[int, ...] | None] = {}
    for node in node_values:
        item = _exact(node, (
            "attributes", "node_key", "node_kind", "parent_node_key",
            "span_end", "span_start",
        ), where="v2 carrier node")
        key = _key(item["node_key"], where="v2 carrier node_key")
        if node_keys and key <= node_keys[-1]:
            raise DatasetContractError("v2 carrier nodes 必须按 node_key 排序")
        node_keys.append(key)
        if not isinstance(item["node_kind"], str) or not item["node_kind"]:
            raise DatasetContractError("v2 carrier node_kind 非法")
        if not isinstance(item["attributes"], dict):
            raise DatasetContractError("v2 carrier node attributes 必须是 object")
        if (type(item["span_start"]) is not int or type(item["span_end"]) is not int
                or item["span_start"] < 0 or item["span_end"] < item["span_start"]):
            raise DatasetContractError("v2 carrier node span 非法")
        parent = None if item["parent_node_key"] is None else _key(
            item["parent_node_key"], where="v2 carrier parent_node_key")
        if parent == key:
            raise DatasetContractError("v2 carrier node 不得以自身为 parent")
        parents[key] = parent
    known = set(node_keys)
    roots = _keys(carrier["root_node_keys"], where="v2 carrier root_node_keys")
    if any(key not in known for key in roots):
        raise DatasetContractError("v2 carrier root 不在 nodes")
    if {key for key, parent in parents.items() if parent is None} != set(roots):
        raise DatasetContractError("v2 carrier roots 与 parent 关系不一致")
    for key, parent in parents.items():
        seen: set[tuple[int, ...]] = set()
        current = parent
        while current is not None:
            if current in seen:
                raise DatasetContractError("v2 carrier parent 图成环")
            seen.add(current)
            if current not in known:
                raise DatasetContractError("v2 carrier parent 不在 nodes")
            current = parents[current]
    edges = carrier["edges"]
    if not isinstance(edges, list):
        raise DatasetContractError("v2 carrier edges 必须是数组")
    edge_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for edge in edges:
        item = _exact(edge, (
            "attributes", "edge_kind", "source_node_key", "target_node_key",
        ), where="v2 carrier edge")
        if (not isinstance(item["attributes"], dict)
                or not isinstance(item["edge_kind"], str)
                or not item["edge_kind"]):
            raise DatasetContractError("v2 carrier edge 字段非法")
        source = _key(item["source_node_key"], where="v2 edge source")
        target = _key(item["target_node_key"], where="v2 edge target")
        if source not in known or target not in known or source == target:
            raise DatasetContractError("v2 carrier edge endpoint 非法")
        edge_key = (item["edge_kind"], source, target)
        if edge_keys and edge_key <= edge_keys[-1]:
            raise DatasetContractError("v2 carrier edges 必须稳定排序且不重复")
        edge_keys.append(edge_key)
    if not isinstance(raw["language_payload"], dict):
        raise DatasetContractError("v2 language_payload 必须是 object")


def _validate_versions(value: dict[str, Any], *, where: str) -> None:
    """所有 v2 记录使用统一整数 format/schema/course 版本。"""
    if (value["format_version"] != V2_FORMAT_VERSION
            or value["schema_version"] != V2_SCHEMA_VERSION
            or value["course_version"] != V2_COURSE_VERSION):
        raise DatasetContractError(f"{where} v2 version 漂移")


def validate_v2_record(value: dict[str, Any]) -> object:
    """严格恢复一个 v2 record，拒绝 v1 字段、未知字段和载体展平。"""
    if not isinstance(value, dict):
        raise DatasetContractError("v2 record 根必须是 object")
    kind = value.get("record_kind")
    fields = _RECORD_FIELDS.get(kind)
    if fields is None:
        raise DatasetContractError("v2 record_kind 未知")
    raw = _exact(value, fields, where=f"v2 {kind}")
    _validate_versions(raw, where=kind)
    record = record_from_dict(raw)
    if kind == RECORD_SOURCE_REF:
        assert isinstance(record, SourceRefRecord)
        if record.license_id not in V2_PUBLIC_LICENSES:
            raise DatasetContractError("v2 source license 必须是可公开许可")
        if record.source_key not in V2_SOURCE_KEYS:
            raise DatasetContractError("v2 source 不在 allowlist")
        if (record.license_id not in V2_SOURCE_LICENSES[record.source_key]
                or record.redistribution_policy != "PUBLIC"):
            raise DatasetContractError("v2 source license/redistribution 与 allowlist 漂移")
        _validate_source_span(record.source_span.to_value(), source_key=record.source_key)
    elif kind == RECORD_OBSERVATION:
        assert isinstance(record, ObservationRecord)
        if record.w_stage not in V2_OBSERVATION_STAGES:
            raise DatasetContractError("v2 observation 不得使用 W-01 或未知阶段")
        if record.split not in SPLITS:
            raise DatasetContractError("v2 observation split 非法")
        _validate_carrier(record.typed_payload.to_value(), representation=record.representation)
    elif kind == RECORD_TEACHER_EVIDENCE:
        assert isinstance(record, TeacherEvidenceRecord)
        if record.visible_from_stage not in V2_OBSERVATION_STAGES:
            raise DatasetContractError("v2 teacher Evidence stage 非法")
    elif kind == RECORD_EVALUATOR_LABEL:
        assert isinstance(record, EvaluatorLabelRecord)
        if record.visible_stage not in V2_OBSERVATION_STAGES:
            raise DatasetContractError("v2 evaluator stage 非法")
    else:
        assert isinstance(record, ArtifactManifest)
        if record.w_stages[0] == "W-01":
            raise DatasetContractError("v2 manifest 不得绑定 W-01")
        if (record.source_key not in V2_SOURCE_KEYS
                or record.license_partition not in V2_SOURCE_LICENSES[record.source_key]
                or record.redistribution_policy != "PUBLIC"):
            raise DatasetContractError("v2 manifest source/license 与 allowlist 漂移")
        if (raw["splits"] != list(record.splits)
                or raw["w_stages"] != list(record.w_stages)
                or raw["source_cluster_keys"] != [
                    key.to_list() for key in record.source_cluster_keys]
                or raw["prerequisite_manifest_keys"] != [
                    key.to_list() for key in record.prerequisite_manifest_keys]):
            raise DatasetContractError("v2 manifest 有序字段不是规范顺序")
        _validate_manifest_files(raw["files"], record)
    return record


def _validate_manifest_files(value: Any, manifest: ArtifactManifest) -> None:
    """校验 pack 的物理 owner、split 路径和非空边界。"""
    if not isinstance(value, list) or not value:
        raise DatasetContractError("v2 pack manifest files 不能为空")
    if manifest.record_count <= 0:
        raise DatasetContractError("v2 pack 不得是空 pack")
    kinds = set()
    source_count = 0
    observation_count = 0
    paths: set[str] = set()
    file_splits: set[str] = set()
    manifest_clusters = {key.components for key in manifest.source_cluster_keys}
    raw_paths = [item.get("relative_path") if isinstance(item, dict) else None
                 for item in value]
    if (any(not isinstance(path, str) for path in raw_paths)
            or raw_paths != sorted(raw_paths)):
        raise DatasetContractError("v2 pack files 必须按路径排序")
    for file_value in value:
        item = _exact(file_value, ARTIFACT_FILE_FIELDS, where="v2 ArtifactFileIdentity")
        kind = item["record_kind"]
        owner = item["owner_kind"]
        path = item["relative_path"]
        if path in paths:
            raise DatasetContractError("v2 pack 文件路径重复")
        paths.add(path)
        kinds.add(kind)
        parts = PurePosixPath(path).parts
        split = item["split"]
        if split is not None:
            file_splits.add(split)
        file_clusters = set(_keys(
            item["source_cluster_keys"], where="v2 file source clusters"))
        if item["source_cluster_keys"] != sorted(item["source_cluster_keys"]):
            raise DatasetContractError("v2 file source clusters 必须排序")
        if not file_clusters.issubset(manifest_clusters):
            raise DatasetContractError("v2 file source cluster 不在 manifest")
        if kind == RECORD_SOURCE_REF:
            if (owner != "source" or split is not None
                    or parts != ("source_refs.jsonl.gz",)):
                raise DatasetContractError("v2 source ref owner/path 漂移")
            source_count += item["record_count"]
        elif kind == RECORD_OBSERVATION:
            if (owner != "observation" or split not in SPLITS
                    or len(parts) != 2 or parts[0] != "observations"
                    or PurePosixPath(path).name != f"{split}.jsonl.gz"):
                raise DatasetContractError("v2 observation owner/path 漂移")
            observation_count += item["record_count"]
        elif kind == RECORD_TEACHER_EVIDENCE:
            if (owner != "teacher" or split != "train"
                    or parts != ("owners", "teacher", "train.evidence.jsonl.gz")):
                raise DatasetContractError("v2 teacher owner/path 漂移")
        elif kind == RECORD_EVALUATOR_LABEL:
            if (owner != "evaluator" or split not in SPLITS[1:]
                    or parts != (
                        "owners", "evaluator", f"{split}.labels.jsonl.gz")):
                raise DatasetContractError("v2 evaluator owner/path 漂移")
        else:
            raise DatasetContractError("v2 ArtifactFileIdentity record_kind 非法")
    if kinds != set(V2_RECORD_KINDS[:-1]):
        raise DatasetContractError("v2 pack 必须覆盖四类 record owner")
    if source_count <= 0 or observation_count <= 0:
        raise DatasetContractError("v2 pack source/observation 不得为空")
    if file_splits != set(manifest.splits):
        raise DatasetContractError("v2 pack 文件 split 与 manifest 不一致")


def validate_v2_record_set(
        values: list[dict[str, Any]],
        *,
        teacher_owner_key: tuple[int, ...],
        evaluator_owner_key: tuple[int, ...],
        ) -> tuple[object, ...]:
    """跨记录验证引用、split 盲区、owner 归属和 cluster 不跨 split。"""
    records = tuple(validate_v2_record(value) for value in values)
    sources = {record.stable_key.components: record for record in records
               if isinstance(record, SourceRefRecord)}
    observations = {record.stable_key.components: record for record in records
                    if isinstance(record, ObservationRecord)}
    if not sources or not observations:
        raise DatasetContractError("v2 record set 必须包含 source_ref 和 observation")
    stable_keys = [record.stable_key.components for record in records
                   if hasattr(record, "stable_key")]
    if len(stable_keys) != len(set(stable_keys)):
        raise DatasetContractError("v2 record stable_key 重复")
    clusters: dict[tuple[str, tuple[int, ...]], str] = {}
    for observation in observations.values():
        if observation.source_ref_key.components not in sources:
            raise DatasetContractError("v2 observation source_ref 引用缺失")
        source = sources[observation.source_ref_key.components]
        if observation.license_partition != source.license_id:
            raise DatasetContractError("v2 observation/source license 漂移")
        span = source.source_span.to_value()
        cluster_values = (
            ("source", source.source_cluster_key.components),
            ("document", tuple(span["document_cluster_key"])),
            ("entity", tuple(span["entity_graph_cluster_key"])),
            ("dedup", observation.dedup_cluster_key.components),
            ("content", observation.content_group_key.components),
            ("template", observation.template_group_key.components),
            ("shape", observation.shape_group_key.components),
        )
        for dimension, key in cluster_values:
            previous = clusters.setdefault((dimension, key), observation.split)
            if previous != observation.split:
                raise DatasetContractError("v2 cluster 不得跨 split")
    teacher_records = [record for record in records if isinstance(record, TeacherEvidenceRecord)]
    for teacher in teacher_records:
        if teacher.owner_key.components != teacher_owner_key:
            raise DatasetContractError("v2 teacher owner 漂移")
        observation = observations.get(teacher.observation_key.components)
        if observation is None or observation.split != "train":
            raise DatasetContractError("v2 TeacherEvidence 只能绑定 train observation")
        if teacher.source_ref_key != observation.source_ref_key:
            raise DatasetContractError("v2 TeacherEvidence source ref 漂移")
    for label in (record for record in records if isinstance(record, EvaluatorLabelRecord)):
        if label.owner_key.components != evaluator_owner_key:
            raise DatasetContractError("v2 evaluator owner 漂移")
        observation = observations.get(label.observation_key.components)
        if observation is None or observation.split == "train":
            raise DatasetContractError("v2 evaluator label 不得绑定 train observation")
    return records


def record_schema_bindings() -> tuple[dict[str, Any], ...]:
    """返回 manifest 要冻结的五类 schema 字段身份。"""
    return tuple({
        "course_version": V2_COURSE_VERSION,
        "field_keys": list(fields),
        "format_version": V2_FORMAT_VERSION,
        "record_kind": kind,
        "schema_version": V2_SCHEMA_VERSION,
        "strict_unknown_fields": 1,
    } for kind, fields in _RECORD_FIELDS.items())


__all__ = [
    "ARTIFACT_FILE_FIELDS",
    "ARTIFACT_MANIFEST_FIELDS",
    "EVALUATOR_LABEL_FIELDS",
    "OBSERVATION_FIELDS",
    "record_schema_bindings",
    "SOURCE_REF_FIELDS",
    "TEACHER_EVIDENCE_FIELDS",
    "V2_CARRIER_KINDS",
    "V2_COURSE_VERSION",
    "V2_FORMAT_VERSION",
    "V2_OBSERVATION_STAGES",
    "V2_PUBLIC_LICENSES",
    "V2_RECORD_KINDS",
    "V2_SCHEMA_VERSION",
    "V2_SOURCE_KEYS",
    "V2_SOURCE_LICENSES",
    "record_schema_bindings",
    "validate_v2_record",
    "validate_v2_record_set",
]
