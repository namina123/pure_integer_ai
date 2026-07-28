"""P3-Ia 中文自由文本层级、中心与长期召回原创课程。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_capability_course_contract import (
    COURSE_EXECUTION_STATE,
    COURSE_INVARIANTS,
    COURSE_SPLIT_AXES,
    CapabilityCourseManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    DatasetContractError,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1"
STAGE = "W-09"
SUBSTAGE = "FREE_TEXT_HIERARCHY_RECALL"
PAYLOAD_KIND = "FreeTextHierarchyRecallObservationV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "P3-Ia-free-text-hierarchy-recall-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_p3ia_dataset_artifacts/d02_language_courses_v1")
SAMPLE_RELATIVE_PATH = (
    "data/ph2/authored_free_text_hierarchy_recall_course_seed_v1.jsonl.sample")

PHENOMENA = (
    "ACL_NEIGHBOR",
    "AMBIGUITY",
    "CLARIFY",
    "COLD_PAGE_IN",
    "CONFLICT",
    "CROSS_PARAGRAPH_DEPENDENCY",
    "DESCRIPTION_REPLACEMENT",
    "ELLIPSIS",
    "MULTI_CENTER",
    "PARAGRAPH_MOVE",
    "QUESTION_RESTRUCTURE",
    "REFERENCE",
    "REVISION_LOCALITY",
    "SCOPE_NEIGHBOR",
    "SOURCE_NEIGHBOR",
    "SYNONYM_PARAPHRASE",
    "UNKNOWN",
)
EVALUATOR_DIMENSIONS = (
    "CENTER_FORMATION",
    "CITATION_EXACTNESS",
    "HIERARCHY_FORMATION",
    "LABEL_ISOLATION",
    "QA_CONSUMER",
    "RECALL_SELECTION",
    "REVISION_LOCALITY",
)
ABLATION_KEYS = (
    "REMOVE_CENTER_FORMER",
    "REMOVE_COLD_PAGE_IN",
    "REMOVE_HIERARCHY_FORMER",
    "REMOVE_PARAPHRASE_EVIDENCE",
)
PAYLOAD_KEYS = (
    "allowed_history",
    "document",
    "phenomena",
    "query",
    "recall_budget",
    "recall_index",
    "sample_family",
    "selection_state",
    "split_identity",
)
_BANNED_CANDIDATE_KEYS = frozenset({
    "answer", "boundary", "center", "citation", "evaluator", "expected",
    "label", "paragraph", "proposition", "span",
})
_SEED_FIELDS = {
    "allowed_history", "document", "evaluation_dimension", "expected_payload",
    "expected_state", "family", "label_owner", "license_id", "logical_order",
    "perturbation_kind", "phenomena", "query", "recall_budget",
    "recall_index", "sample_family", "sample_role", "seed_id",
    "split", "split_identity", "supersedes_seed_id", "template_family",
}


class AuthoredFreeTextHierarchyRecallCourseError(RuntimeError):
    """自由文本层级召回课程字段、split 或私有标签边界不成立。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求课程文本无首尾空白，默认还要求非空。"""
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise AuthoredFreeTextHierarchyRecallCourseError(
            f"{where} 必须是规范文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求课程序号和预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredFreeTextHierarchyRecallCourseError(
            f"{where} 必须是正严格整数")
    return value


def _object(value: Any, *, where: str) -> CanonicalJsonObject:
    """把规范对象收紧为不可变 CanonicalJsonObject。"""
    if not isinstance(value, dict):
        raise AuthoredFreeTextHierarchyRecallCourseError(f"{where} 必须是对象")
    try:
        return CanonicalJsonObject.from_value(value)
    except DatasetContractError as error:
        raise AuthoredFreeTextHierarchyRecallCourseError(
            f"{where} 不是规范整数/文本对象") from error


def _objects(value: Any, *, where: str) -> tuple[CanonicalJsonObject, ...]:
    """读取一组规范对象并拒绝空数组。"""
    if not isinstance(value, list) or not value:
        raise AuthoredFreeTextHierarchyRecallCourseError(f"{where} 必须是非空数组")
    return tuple(_object(item, where=where) for item in value)


def _strings(
        value: Any, *, where: str, allowed: tuple[str, ...] | None = None,
        ) -> tuple[str, ...]:
    """要求文本数组稳定排序去重，并可限制到冻结枚举。"""
    if not isinstance(value, list):
        raise AuthoredFreeTextHierarchyRecallCourseError(f"{where} 必须是数组")
    result = tuple(_text(item, where=where) for item in value)
    if not result or result != tuple(sorted(set(result))):
        raise AuthoredFreeTextHierarchyRecallCourseError(
            f"{where} 必须非空、排序且去重")
    if allowed is not None and any(item not in allowed for item in result):
        raise AuthoredFreeTextHierarchyRecallCourseError(f"{where} 含未登记值")
    return result


def _candidate_key_audit(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """递归拒绝候选投影出现私有正确答案字段。"""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in _BANNED_CANDIDATE_KEYS):
                joined = ".".join((*path, str(key)))
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    f"candidate payload 泄漏私有字段: {joined}")
            _candidate_key_audit(child, path=(*path, str(key)))
    elif isinstance(value, list):
        for ordinal, child in enumerate(value):
            _candidate_key_audit(child, path=(*path, str(ordinal)))


@dataclass(frozen=True)
class AuthoredFreeTextHierarchyRecallSeed:
    """一个 physically split 的 raw 文档、query、历史 Evidence 与私有结果。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_family: str
    sample_role: str
    document: CanonicalJsonObject
    query: CanonicalJsonObject
    allowed_history: tuple[CanonicalJsonObject, ...]
    recall_index: tuple[CanonicalJsonObject, ...]
    recall_budget: CanonicalJsonObject
    phenomena: tuple[str, ...]
    split_identity: CanonicalJsonObject
    expected_state: str
    expected_payload: CanonicalJsonObject
    evaluation_dimension: str
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int
    license_id: str

    def __post_init__(self) -> None:
        """核对 owner、split、cluster、预算和候选/私有投影隔离。"""
        for name in (
                "seed_id", "family", "template_family", "sample_role",
                "perturbation_kind"):
            _text(getattr(self, name), where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredFreeTextHierarchyRecallCourseError("label_owner 非法")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredFreeTextHierarchyRecallCourseError("sample_family 未登记")
        if not all(isinstance(item, CanonicalJsonObject) for item in (
                self.document, self.query, self.recall_budget,
                self.split_identity, self.expected_payload)):
            raise TypeError("课程对象字段必须是 CanonicalJsonObject")
        if (not self.allowed_history or not self.recall_index
                or any(not isinstance(item, CanonicalJsonObject)
                       for item in (*self.allowed_history, *self.recall_index))):
            raise TypeError("allowed_history/recall_index 必须是非空规范对象 tuple")
        if self.phenomena != tuple(sorted(set(self.phenomena))):
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "phenomena 必须排序去重")
        if any(item not in PHENOMENA for item in self.phenomena):
            raise AuthoredFreeTextHierarchyRecallCourseError("phenomena 未登记")
        if self.expected_state not in {"CONFLICT", "FALSE", "TRUE", "UNKNOWN"}:
            raise AuthoredFreeTextHierarchyRecallCourseError("expected_state 非四态")
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _positive(self.logical_order, where="logical_order")
        if self.license_id != LICENSE_ID:
            raise AuthoredFreeTextHierarchyRecallCourseError("课程许可必须 CC0-1.0")
        document = self.document.to_value()
        query = self.query.to_value()
        if set(document) != {"raw_sha256", "raw_text", "source_ref_key"}:
            raise AuthoredFreeTextHierarchyRecallCourseError("document 字段集合非法")
        if set(query) != {"raw_sha256", "raw_text", "source_ref_key"}:
            raise AuthoredFreeTextHierarchyRecallCourseError("query 字段集合非法")
        for name, item in (("document", document), ("query", query)):
            raw_text = _text(item["raw_text"], where=f"{name}.raw_text")
            actual = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if item["raw_sha256"] != actual:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    f"{name} raw SHA 漂移")
            source_key = item["source_ref_key"]
            if (not isinstance(source_key, list) or len(source_key) != 11
                    or any(type(value) is not int for value in source_key)):
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    f"{name} SourceRef key 非法")
        budget = self.recall_budget.to_value()
        if set(budget) != {
                "max_index_gets", "max_results", "max_segment_payload_bytes",
                "max_segment_payload_gets"}:
            raise AuthoredFreeTextHierarchyRecallCourseError("recall budget 字段非法")
        for name, value in budget.items():
            _positive(value, where=f"recall_budget.{name}")
        evidence_ids = []
        for item in self.allowed_history:
            raw = item.to_value()
            if set(raw) != {
                    "evidence_id", "feature_key", "owner_kind", "surface"}:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "allowed_history 字段集合非法")
            evidence_ids.append(_positive(
                raw["evidence_id"], where="allowed_history.evidence_id"))
            _positive(raw["feature_key"], where="allowed_history.feature_key")
            if raw["owner_kind"] != "HISTORY":
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "allowed_history owner 必须 HISTORY")
            _text(raw["surface"], where="allowed_history.surface")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "allowed_history Evidence identity 重复")
        record_keys = []
        for item in self.recall_index:
            raw = item.to_value()
            if set(raw) != {
                    "feature_keys", "owner_key", "record_key", "scope_key",
                    "source_ref_key"}:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "recall_index 字段集合非法")
            for name in ("feature_keys", "record_key", "scope_key"):
                values = raw[name]
                if (not isinstance(values, list) or not values
                        or any(type(value) is not int or value <= 0
                               for value in values)):
                    raise AuthoredFreeTextHierarchyRecallCourseError(
                        f"recall_index.{name} 非法")
            for name in ("owner_key", "source_ref_key"):
                values = raw[name]
                if (not isinstance(values, list) or not values
                        or any(type(value) is not int or value < 0
                               for value in values)):
                    raise AuthoredFreeTextHierarchyRecallCourseError(
                        f"recall_index.{name} 非法")
            if len(raw["source_ref_key"]) != 11:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "recall_index SourceRef key 长度非法")
            if raw["owner_key"] != raw["source_ref_key"][3:7]:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "recall_index owner/source 漂移")
            if raw["feature_keys"] != sorted(set(raw["feature_keys"])):
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "recall_index feature keys 必须排序去重")
            record_keys.append(tuple(raw["record_key"]))
        if len(set(record_keys)) != len(record_keys):
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "recall_index record key 重复")
        split_identity = self.split_identity.to_value()
        expected_clusters = {
            "document_cluster", "paraphrase_cluster", "source_cluster",
            "structure_cluster", "template_cluster"}
        if set(split_identity) != expected_clusters:
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "split identity 未列全五类 cluster")
        if any(not isinstance(value, str) or not value
               for value in split_identity.values()):
            raise AuthoredFreeTextHierarchyRecallCourseError("cluster identity 非法")
        _candidate_key_audit(self.observation_payload().to_value())

    def observation_payload(self) -> CanonicalJsonObject:
        """只投影 candidate 可读原文、来源、历史 Evidence、index 和预算。"""
        return CanonicalJsonObject.from_value({
            "allowed_history": [item.to_value() for item in self.allowed_history],
            "document": self.document.to_value(),
            "phenomena": list(self.phenomena),
            "query": self.query.to_value(),
            "recall_budget": self.recall_budget.to_value(),
            "recall_index": [item.to_value() for item in self.recall_index],
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "split_identity": self.split_identity.to_value(),
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """形成共用发布器需要的候选投影和私有 owner payload。"""
        clusters = self.split_identity.to_value()
        document = self.document.to_value()
        query = self.query.to_value()
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            PAYLOAD_KIND,
            self.observation_payload(),
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (self.seed_id, clusters["document_cluster"]),
            (document["raw_sha256"], query["raw_sha256"]),
            (
                clusters["structure_cluster"],
                clusters["paraphrase_cluster"],
                len(self.allowed_history),
                len(self.recall_index),
            ),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredFreeTextHierarchyRecallSeed:
    """从精确字段规范对象恢复一个课程 seed。"""
    if set(value) != _SEED_FIELDS:
        raise AuthoredFreeTextHierarchyRecallCourseError("seed 字段集合非法")
    return AuthoredFreeTextHierarchyRecallSeed(
        _text(value["seed_id"], where="seed_id"),
        _text(value["family"], where="family"),
        _text(value["template_family"], where="template_family"),
        _text(value["label_owner"], where="label_owner"),
        _text(value["split"], where="split"),
        _text(value["sample_family"], where="sample_family"),
        _text(value["sample_role"], where="sample_role"),
        _object(value["document"], where="document"),
        _object(value["query"], where="query"),
        _objects(value["allowed_history"], where="allowed_history"),
        _objects(value["recall_index"], where="recall_index"),
        _object(value["recall_budget"], where="recall_budget"),
        _strings(value["phenomena"], where="phenomena", allowed=PHENOMENA),
        _object(value["split_identity"], where="split_identity"),
        _text(value["expected_state"], where="expected_state"),
        _object(value["expected_payload"], where="expected_payload"),
        _text(value["evaluation_dimension"], where="evaluation_dimension"),
        _text(value["perturbation_kind"], where="perturbation_kind"),
        _text(value["supersedes_seed_id"], where="supersedes_seed_id",
              allow_empty=True),
        _positive(value["logical_order"], where="logical_order"),
        _text(value["license_id"], where="license_id"),
    )


def read_authored_free_text_hierarchy_recall_seeds(
        path: str | Path,
        ) -> tuple[AuthoredFreeTextHierarchyRecallSeed, ...]:
    """严格读取课程并审计 owner、cluster、覆盖面和修正链。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredFreeTextHierarchyRecallCourseError("课程 sample 无法读取") from error
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredFreeTextHierarchyRecallCourseError("课程 sample 换行非法")
    seeds = []
    for line in payload.splitlines(keepends=True):
        try:
            raw = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredFreeTextHierarchyRecallCourseError(
                "课程 sample 不是规范 JSONL") from error
        if not isinstance(raw, dict) or canonical_json_line(raw) != line:
            raise AuthoredFreeTextHierarchyRecallCourseError("课程行非规范 round-trip")
        seeds.append(_seed_from_value(raw))
    identifiers = tuple(item.seed_id for item in seeds)
    orders = tuple(item.logical_order for item in seeds)
    if len(set(identifiers)) != len(identifiers):
        raise AuthoredFreeTextHierarchyRecallCourseError("seed_id 重复")
    if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
        raise AuthoredFreeTextHierarchyRecallCourseError("logical_order 未严格递增")
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    if not teacher or not evaluator:
        raise AuthoredFreeTextHierarchyRecallCourseError("课程必须物理分离双 owner")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredFreeTextHierarchyRecallCourseError(
            "teacher/evaluator family 或 template 泄漏")
    cluster_splits: dict[tuple[str, str], str] = {}
    for seed in seeds:
        for name, value in seed.split_identity.to_value().items():
            key = name, value
            prior = cluster_splits.setdefault(key, seed.split)
            if prior != seed.split:
                raise AuthoredFreeTextHierarchyRecallCourseError(
                    "source/document/template/structure/paraphrase cluster 跨 split")
    if set(item.sample_family for item in seeds) != set(SAMPLE_FAMILIES):
        raise AuthoredFreeTextHierarchyRecallCourseError("课程未列全七类 sample family")
    covered = {phenomenon for item in seeds for phenomenon in item.phenomena}
    if covered != set(PHENOMENA):
        raise AuthoredFreeTextHierarchyRecallCourseError("课程未列全 R-01.3 现象")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredFreeTextHierarchyRecallCourseError("独立 evaluator 维度未列全")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        prior = index.get(seed.supersedes_seed_id)
        if (prior is None or prior.logical_order >= seed.logical_order
                or prior.family != seed.family or prior.split != seed.split
                or "REVISION_LOCALITY" not in seed.phenomena):
            raise AuthoredFreeTextHierarchyRecallCourseError("revision 链非法")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    """返回 P3-Ia pack 的冻结来源、版本和评估预算。"""
    return AuthoredCourseSpec(
        SOURCE_KEY, LICENSE_ID, COURSE_VERSION, ARTIFACT_VERSION,
        ADAPTER_VERSION, GENERATOR_VERSION, PARSER_VERSION, PACK_NAME,
        STAGE, SUBSTAGE, "authored-free-text-hierarchy-recall-seed-v1",
        "urn:pure-integer-ai:ph2:authored-free-text-hierarchy-recall-v1",
        "Pure Integer AI PH2 authored free text hierarchy recall seed",
        "FREE_TEXT_HIERARCHY_RECALL_LABEL",
        "p3ia-free-text-hierarchy-recall",
        512,
    )


def compile_authored_free_text_hierarchy_recall_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    """把规范 seed 编译为 source/observation/teacher/evaluator 分账 pack。"""
    seeds = read_authored_free_text_hierarchy_recall_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_free_text_hierarchy_recall_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    """建立不可冒充训练或 mastered 事实的 COURSE_FROZEN manifest。"""
    sample = Path(sample_path)
    seeds = read_authored_free_text_hierarchy_recall_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-07", "LC-13", "LC-15"),
        (
            "DISCOURSE_INFORMATION_STRUCTURE",
            "MEMORY_DYNAMICS",
            "REFERENCE_DISCOURSE_REVISION",
            "TYPED_LEARNING_OBJECTIVES",
        ),
        STAGE,
        SUBSTAGE,
        SOURCE_KEY,
        LICENSE_ID,
        f"data/ph2/{sample.name}",
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        len(seeds),
        SAMPLE_FAMILIES,
        COURSE_SPLIT_AXES,
        tuple(sorted({item.family for item in teacher})),
        tuple(sorted({item.family for item in evaluator})),
        tuple(sorted({item.template_family for item in teacher})),
        tuple(sorted({item.template_family for item in evaluator})),
        PAYLOAD_KIND,
        PAYLOAD_KEYS,
        (
            "CONTROLLED_PERTURBATION",
            "CROSS_CONTEXT_CONSISTENCY",
            "GENERATION_ADOPTION",
            "GENERATION_FAILURE",
            "NEXT_DISCOURSE_UNIT",
            "ORDER_RECOVERY",
        ),
        EVALUATOR_DIMENSIONS,
        (
            "CROSS_LANGUAGE_OR_CODE_SWITCH_REQUESTED",
            "DEFINITIVE_REALITY_TRUTH_REQUESTED",
            "HELD_OUT_LABEL_UNAVAILABLE",
            "TRAINING_OR_MASTERED_REQUESTED",
        ),
        (
            "DUMP_RESUME_EXACT_SOURCE_RECALL",
            "REVISION_DEPENDENCY_LOCAL_REBUILD",
            "SOURCE_WITHDRAWAL_LOCAL_INVALIDATION",
        ),
        (
            "DOCUMENT_X_STRUCTURE_X_PARAPHRASE",
            "SOURCE_X_SCOPE_X_ACL",
        ),
        (
            "EXACT_SURFACE_ONLY",
            "NO_CENTER_FORMER",
            "NO_HIERARCHY_FORMER",
            "NO_PAGE_IN",
        ),
        ABLATION_KEYS,
        f"{artifact_relative_root}/packs/{PACK_NAME}/manifest.json",
        build.manifest.sha256(),
        build.manifest.record_count,
        build.manifest.splits,
        CanonicalJsonObject.from_value(COURSE_INVARIANTS),
        CanonicalJsonObject.from_value(COURSE_EXECUTION_STATE),
    )


def _source_key(ordinal: int, *, query: bool = False) -> list[int]:
    """为默认 authored seed 形成固定 SourceRef stable key。"""
    return [
        322 if query else 321,
        81000 + ordinal,
        ordinal,
        191,
        71,
        0,
        3,
        1,
        1,
        1,
        1,
    ]


def _default_seed_value(
        ordinal: int,
        *,
        owner: str,
        sample_family: str,
        document_text: str,
        query_text: str,
        answer_text: str,
        phenomena: tuple[str, ...],
        dimension: str,
        status: str = "TRUE",
        center_count: int = 1,
        supersedes: str = "",
        family_override: str = "",
        ) -> dict[str, Any]:
    """只在 authoring 侧构造完整默认行；候选发布投影会删除私有结果。"""
    split = "train" if owner == "teacher" else "held_out"
    prefix = "tr" if owner == "teacher" else "ho"
    family = family_override or f"{prefix}_family_{ordinal:02d}"
    source_key = _source_key(ordinal)
    query_source = _source_key(ordinal, query=True)
    query_features = [70000 + ordinal, 71000 + ordinal]
    record_keys = [[73000 + ordinal, index] for index in range(1, center_count + 1)]
    answer_start = document_text.index(answer_text) if answer_text else 0
    answer_end = answer_start + len(answer_text)
    lines = []
    cursor = 0
    for raw_line in document_text.splitlines(keepends=True):
        surface = raw_line.rstrip("\r\n")
        if surface:
            lines.append({"end": cursor + len(surface), "start": cursor})
        cursor += len(raw_line)
    expected = {
        "answer_surface": answer_text,
        "center_record_keys": record_keys,
        "citation": {
            "end": answer_end,
            "record_key": record_keys[0],
            "source_ref_key": source_key,
            "start": answer_start,
        },
        "hierarchy_ranges": lines,
        "invalidated_keys": (
            [[74000 + ordinal, 1], [74000 + ordinal, 2], [74000 + ordinal, 3]]
            if "REVISION_LOCALITY" in phenomena else []),
        "preserved_keys": (
            [[75000 + ordinal, 1]]
            if "REVISION_LOCALITY" in phenomena else []),
        "proposition_record_key": record_keys[0],
        "required_stop_reason": (
            "CLARIFY" if "CLARIFY" in phenomena or center_count > 1
            else "UNKNOWN" if status == "UNKNOWN" else "RESOLVED"),
    }
    history = [
        {
            "evidence_id": 76000 + ordinal * 10 + index,
            "feature_key": feature,
            "owner_kind": "HISTORY",
            "surface": surface,
        }
        for index, (surface, feature) in enumerate((
            (query_text[:max(1, len(query_text) // 2)], query_features[0]),
            (query_text[max(1, len(query_text) // 2):], query_features[1]),
        ), start=1)
        if surface
    ]
    recall_index = [
        {
            "feature_keys": query_features,
            "owner_key": source_key[3:7],
            "record_key": record_key,
            "scope_key": [77000 + ordinal, index],
            "source_ref_key": source_key,
        }
        for index, record_key in enumerate(record_keys, start=1)
    ]
    neighbor_ordinal = center_count + 1
    if "SOURCE_NEIGHBOR" in phenomena:
        wrong_source = list(source_key)
        wrong_source[1] += 100000
        recall_index.append({
            "feature_keys": [query_features[0], 71900 + ordinal],
            "owner_key": wrong_source[3:7],
            "record_key": [73500 + ordinal, 1],
            "scope_key": [77500 + ordinal, neighbor_ordinal],
            "source_ref_key": wrong_source,
        })
        neighbor_ordinal += 1
    if "SCOPE_NEIGHBOR" in phenomena:
        recall_index.append({
            "feature_keys": [query_features[0], 71950 + ordinal],
            "owner_key": source_key[3:7],
            "record_key": [73500 + ordinal, 2],
            "scope_key": [77550 + ordinal, neighbor_ordinal],
            "source_ref_key": source_key,
        })
        neighbor_ordinal += 1
    if "ACL_NEIGHBOR" in phenomena:
        wrong_acl_source = list(source_key)
        wrong_acl_source[4] += 1
        recall_index.append({
            "feature_keys": query_features,
            "owner_key": wrong_acl_source[3:7],
            "record_key": [73500 + ordinal, 3],
            "scope_key": [77600 + ordinal, neighbor_ordinal],
            "source_ref_key": wrong_acl_source,
        })
    return {
        "allowed_history": history,
        "document": {
            "raw_sha256": hashlib.sha256(document_text.encode("utf-8")).hexdigest(),
            "raw_text": document_text,
            "source_ref_key": source_key,
        },
        "evaluation_dimension": dimension,
        "expected_payload": expected,
        "expected_state": status,
        "family": family,
        "label_owner": owner,
        "license_id": LICENSE_ID,
        "logical_order": ordinal,
        "perturbation_kind": phenomena[0],
        "phenomena": list(sorted(phenomena)),
        "query": {
            "raw_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "raw_text": query_text,
            "source_ref_key": query_source,
        },
        "recall_budget": {
            "max_index_gets": 8,
            "max_results": 4,
            "max_segment_payload_bytes": 65536,
            "max_segment_payload_gets": 4,
        },
        "recall_index": recall_index,
        "sample_family": sample_family,
        "sample_role": {
            "AMBIGUOUS": "anomaly",
            "GENERATION": "support",
            "NEGATIVE": "refute",
            "POSITIVE": "support",
            "RETENTION": "read_only_probe",
            "REVISION": "supersede",
            "UNKNOWN": "anomaly",
        }[sample_family],
        "seed_id": f"p3ia_{prefix}_{ordinal:02d}",
        "split": split,
        "split_identity": {
            "document_cluster": f"{prefix}_document_{ordinal:02d}",
            "paraphrase_cluster": f"{prefix}_paraphrase_{ordinal:02d}",
            "source_cluster": f"{prefix}_source_{ordinal:02d}",
            "structure_cluster": f"{prefix}_structure_{ordinal:02d}",
            "template_cluster": f"{prefix}_template_{ordinal:02d}",
        },
        "supersedes_seed_id": supersedes,
        "template_family": f"{prefix}_template_family_{ordinal:02d}",
    }


def build_default_free_text_hierarchy_recall_seed_values(
        ) -> tuple[dict[str, Any], ...]:
    """返回覆盖 R-01.3 全现象、双 owner 和七类样本的确定 authored 行。"""
    specs = (
        ("teacher", "POSITIVE", "航标记录\n初始情况\n玄衡站的备用端口设在北川。\n更新\n后来它迁到了南岭。", "玄衡站备用接口后来在哪儿？", "南岭", ("SYNONYM_PARAPHRASE", "REFERENCE", "CROSS_PARAGRAPH_DEPENDENCY"), "HIERARCHY_FORMATION", "TRUE", 1),
        ("teacher", "GENERATION", "施工简报\n第一段\n青岚库的归档点位于东塔。\n第二段\n该资料存放处现已改至西台。", "青岚库如今把资料放在哪？", "西台", ("DESCRIPTION_REPLACEMENT", "QUESTION_RESTRUCTURE"), "QA_CONSUMER", "TRUE", 1),
        ("teacher", "REVISION", "维护日志\n旧记\n苍梧节点从东门校验。\n新记\n现改由西门完成。", "苍梧节点现在呢？", "西门", ("ELLIPSIS", "REVISION_LOCALITY"), "REVISION_LOCALITY", "TRUE", 1),
        ("teacher", "RETENTION", "冷存档\n条目\n景曜计划的验收柜在三号库。", "景曜计划去哪验收？", "三号库", ("COLD_PAGE_IN",), "CITATION_EXACTNESS", "TRUE", 1),
        ("teacher", "AMBIGUOUS", "联席记录\n甲项\n两套方案都可使用北台。\n乙项\n两套方案也都可使用南台。", "联席方案该选哪个台？", "北台", ("AMBIGUITY", "MULTI_CENTER", "CLARIFY"), "CENTER_FORMATION", "UNKNOWN", 2),
        ("teacher", "NEGATIVE", "隔离记录\n甲域\n同名晨星在外域指向北库。\n乙域\n本域晨星指向南库。", "本域晨星指向哪里？", "南库", ("SOURCE_NEIGHBOR", "SCOPE_NEIGHBOR"), "RECALL_SELECTION", "TRUE", 1),
        ("teacher", "UNKNOWN", "权限说明\n公开条目\n公开记录没有给出密钥位置。", "密钥存放在哪里？", "", ("ACL_NEIGHBOR", "UNKNOWN"), "LABEL_ISOLATION", "UNKNOWN", 1),
        ("teacher", "POSITIVE", "调序纪要\n结论\n最终采用云台。\n背景\n此前曾考虑河台。", "最后采用哪里？", "云台", ("PARAGRAPH_MOVE",), "HIERARCHY_FORMATION", "TRUE", 1),
        ("evaluator", "POSITIVE", "值守簿\n上篇\n霁月组的备用席原在松台。\n下篇\n随后这一候补位置换到竹台。", "霁月组后来把备席挪去何处？", "竹台", ("SYNONYM_PARAPHRASE", "DESCRIPTION_REPLACEMENT"), "HIERARCHY_FORMATION", "TRUE", 1),
        ("evaluator", "GENERATION", "运维单\n先记\n若水服务从南桥核验。\n后记\n现在改在北桥。", "若水服务现在从哪儿核验？", "北桥", ("QUESTION_RESTRUCTURE", "CROSS_PARAGRAPH_DEPENDENCY"), "QA_CONSUMER", "TRUE", 1),
        ("evaluator", "RETENTION", "封存册\n唯一条目\n扶光工程的复核室为七号间。", "扶光工程在哪间复核？", "七号间", ("COLD_PAGE_IN",), "CITATION_EXACTNESS", "TRUE", 1),
        ("evaluator", "AMBIGUOUS", "会商录\n方案甲\n可转至东港。\n方案乙\n也可转至西港。", "接下来转去哪里？", "东港", ("AMBIGUITY", "MULTI_CENTER", "CLARIFY"), "CENTER_FORMATION", "UNKNOWN", 2),
        ("evaluator", "NEGATIVE", "租户目录\n公开域\n同名启明属于甲租户。\n当前域\n本次问题属于乙租户。", "当前启明的归属在哪里？", "乙租户", ("ACL_NEIGHBOR", "SOURCE_NEIGHBOR", "SCOPE_NEIGHBOR"), "RECALL_SELECTION", "TRUE", 1),
        ("evaluator", "UNKNOWN", "调查单\n观察\n记录只说明任务已启动。", "任务由谁批准？", "", ("UNKNOWN",), "LABEL_ISOLATION", "UNKNOWN", 1),
        ("evaluator", "REVISION", "更正单\n原记\n照临库位于甲区。\n更正\n应为乙区，其余记录不变。", "照临库最终在哪区？", "乙区", ("REFERENCE", "REVISION_LOCALITY"), "REVISION_LOCALITY", "TRUE", 1),
        ("evaluator", "NEGATIVE", "冲突报表\n来源一\n归档点为石台。\n来源二\n归档点不是石台。", "归档点是否为石台？", "石台", ("CONFLICT",), "CENTER_FORMATION", "CONFLICT", 2),
    )
    values = []
    for ordinal, spec in enumerate(specs, start=1):
        values.append(_default_seed_value(
            ordinal,
            owner=spec[0],
            sample_family=spec[1],
            document_text=spec[2],
            query_text=spec[3],
            answer_text=spec[4],
            phenomena=spec[5],
            dimension=spec[6],
            status=spec[7],
            center_count=spec[8],
        ))
    train_revision = _default_seed_value(
        17,
        owner="teacher",
        sample_family="REVISION",
        document_text=(
            "维护日志\n旧记\n苍梧节点从东门校验。\n"
            "新记\n现改由西门完成。\n补充\n仅校验地点受此更正。"),
        query_text="苍梧节点目前从哪里校验？",
        answer_text="西门",
        phenomena=("REVISION_LOCALITY",),
        dimension="REVISION_LOCALITY",
        supersedes="p3ia_tr_03",
        family_override="tr_family_03",
    )
    train_revision["split_identity"]["source_cluster"] = "tr_source_03"
    held_out_revision = _default_seed_value(
        18,
        owner="evaluator",
        sample_family="REVISION",
        document_text=(
            "更正单\n原记\n照临库位于甲区。\n"
            "更正\n应为乙区，其余记录不变。\n补记\n本次只修正库区。"),
        query_text="照临库最后处于哪一区？",
        answer_text="乙区",
        phenomena=("REVISION_LOCALITY",),
        dimension="REVISION_LOCALITY",
        supersedes="p3ia_ho_15",
        family_override="ho_family_15",
    )
    held_out_revision["split_identity"]["source_cluster"] = "ho_source_15"
    values.extend((train_revision, held_out_revision))
    return tuple(values)


def default_free_text_hierarchy_recall_sample_bytes() -> bytes:
    """返回可跨 PYTHONHASHSEED 重建的规范 JSONL 字节。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_free_text_hierarchy_recall_seed_values()
    )


__all__ = [
    "ABLATION_KEYS",
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "PACK_NAME",
    "PAYLOAD_KEYS",
    "PHENOMENA",
    "SAMPLE_RELATIVE_PATH",
    "AuthoredFreeTextHierarchyRecallCourseError",
    "AuthoredFreeTextHierarchyRecallSeed",
    "build_default_free_text_hierarchy_recall_seed_values",
    "build_free_text_hierarchy_recall_course_manifest",
    "compile_authored_free_text_hierarchy_recall_course",
    "default_free_text_hierarchy_recall_sample_bytes",
    "read_authored_free_text_hierarchy_recall_seeds",
]
