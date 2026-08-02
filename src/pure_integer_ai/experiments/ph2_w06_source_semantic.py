"""W-06 七类关系的来源独立性与语义 profile 防火墙。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    DIRECTION_SYMMETRIC,
    RELATION_ANTONYM,
    RELATION_CAUSES,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    RELATION_HAS_PART,
    RELATION_MEMBER,
    RELATION_PART_OF,
    RELATION_PROPERTY,
    RELATION_PURE_ALIAS,
    RELATION_REFERS,
    RELATION_SIMILAR,
    RELATION_SUBSET,
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
    ROLE_ANTONYM_LEFT,
    ROLE_ANTONYM_RIGHT,
    ROLE_CAUSE,
    ROLE_EFFECT,
    ROLE_EVENT_AFTER_OBJECT,
    ROLE_EVENT_AFTER_SUBJECT,
    ROLE_EVENT_BEFORE_OBJECT,
    ROLE_EVENT_BEFORE_SUBJECT,
    ROLE_EVENT_SAME_OBJECT,
    ROLE_EVENT_SAME_SUBJECT,
    ROLE_EVENT_UNKNOWN_OBJECT,
    ROLE_EVENT_UNKNOWN_SUBJECT,
    ROLE_HAS_PART_PART,
    ROLE_HAS_PART_WHOLE,
    ROLE_MEMBER_ELEMENT,
    ROLE_MEMBER_SET,
    ROLE_PART_OF_PART,
    ROLE_PART_OF_WHOLE,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_INTENSITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_VALUE,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
    ROLE_SIMILAR_LEFT,
    ROLE_SIMILAR_RIGHT,
    ROLE_SUBSET_CHILD,
    ROLE_SUBSET_PARENT,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)


W06_RELATION_SUBSTAGE_ORDER = (
    "PURE_ALIAS_REFERS",
    "SUBSET_MEMBER",
    "PROPERTY",
    "MEREOLOGY",
    "SIMILAR_ANTONYM",
    "PRECEDES",
    "CAUSES",
)
W06_GENERATION_HARD_CONJUNCT = (
    "W-06-GENERATION-RELATION-STRUCTURE-HARD-CONJUNCT"
)


class W06SourceSemanticError(RuntimeError):
    """W-06 来源隔离、关系方向或语义 profile 不满足冻结合同。"""


@dataclass(frozen=True)
class W06RelationProfile:
    """声明一个 relation family 的 stage、方向、Role 端点和闭包策略。"""

    substage_key: str
    relation_kind: int
    directionality: int
    role_object_kinds: tuple[tuple[int, frozenset[int]], ...]
    closure_policy: str

    def __post_init__(self) -> None:
        if self.substage_key not in W06_RELATION_SUBSTAGE_ORDER:
            raise W06SourceSemanticError("W-06 relation substage 未注册")
        if self.directionality not in {DIRECTION_FORWARD, DIRECTION_SYMMETRIC}:
            raise W06SourceSemanticError("W-06 relation 方向未注册")
        if not self.role_object_kinds:
            raise W06SourceSemanticError("W-06 relation Role profile 不能为空")
        if not self.closure_policy:
            raise W06SourceSemanticError("W-06 relation 闭包策略不能为空")

    def allowed_for_role(self, role_kind: int) -> frozenset[int]:
        """返回指定冻结 Role 允许的一等对象类型。"""
        for current, allowed in self.role_object_kinds:
            if current == role_kind:
                return allowed
        raise W06SourceSemanticError("W-06 relation 使用了未注册 Role")


_STABLE_REFERENCE_KINDS = frozenset({OBJECT_CONCEPT, OBJECT_ENTITY})
_EVENT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})
_PROPERTY_KINDS = frozenset({OBJECT_CONCEPT, OBJECT_ENTITY})

W06_RELATION_PROFILES = {
    "PURE_ALIAS": W06RelationProfile(
        "PURE_ALIAS_REFERS",
        RELATION_PURE_ALIAS,
        DIRECTION_SYMMETRIC,
        (
            (ROLE_ALIAS_LEFT, _STABLE_REFERENCE_KINDS),
            (ROLE_ALIAS_RIGHT, _STABLE_REFERENCE_KINDS),
        ),
        "PURE_ALIAS_ONLY_TRANSITIVE",
    ),
    "REFERS": W06RelationProfile(
        "PURE_ALIAS_REFERS",
        RELATION_REFERS,
        DIRECTION_FORWARD,
        (
            (ROLE_REFERS_FROM, _STABLE_REFERENCE_KINDS),
            (ROLE_REFERS_TO, _STABLE_REFERENCE_KINDS),
        ),
        "NO_OCCURRENCE_OR_DISCOURSE_CLOSURE",
    ),
    "SUBSET": W06RelationProfile(
        "SUBSET_MEMBER",
        RELATION_SUBSET,
        DIRECTION_FORWARD,
        (
            (ROLE_SUBSET_CHILD, frozenset({OBJECT_SET_EXPR})),
            (ROLE_SUBSET_PARENT, frozenset({OBJECT_SET_EXPR})),
        ),
        "SUBSET_ONLY_TRANSITIVE",
    ),
    "MEMBER": W06RelationProfile(
        "SUBSET_MEMBER",
        RELATION_MEMBER,
        DIRECTION_FORWARD,
        (
            (ROLE_MEMBER_ELEMENT, frozenset({OBJECT_ENTITY, OBJECT_SET_EXPR})),
            (ROLE_MEMBER_SET, frozenset({OBJECT_SET_EXPR})),
        ),
        "NO_MEMBER_TRANSITIVE_CLOSURE",
    ),
    "PROPERTY": W06RelationProfile(
        "PROPERTY",
        RELATION_PROPERTY,
        DIRECTION_FORWARD,
        tuple(
            (role, _PROPERTY_KINDS)
            for role in (
                ROLE_PROPERTY_SUBJECT,
                ROLE_PROPERTY_ATTRIBUTE,
                ROLE_PROPERTY_VALUE,
                ROLE_PROPERTY_POLARITY,
                ROLE_PROPERTY_MODALITY,
                ROLE_PROPERTY_INTENSITY,
            )
        ),
        "DIRECT_ACTIVE_FACTS_ONLY",
    ),
    "PART_OF": W06RelationProfile(
        "MEREOLOGY",
        RELATION_PART_OF,
        DIRECTION_FORWARD,
        (
            (ROLE_PART_OF_PART, frozenset({OBJECT_ENTITY})),
            (ROLE_PART_OF_WHOLE, frozenset({OBJECT_ENTITY})),
        ),
        "EXPLICIT_INVERSE_ONLY",
    ),
    "HAS_PART": W06RelationProfile(
        "MEREOLOGY",
        RELATION_HAS_PART,
        DIRECTION_FORWARD,
        (
            (ROLE_HAS_PART_PART, frozenset({OBJECT_ENTITY})),
            (ROLE_HAS_PART_WHOLE, frozenset({OBJECT_ENTITY})),
        ),
        "EXPLICIT_INVERSE_ONLY",
    ),
    "SIMILAR": W06RelationProfile(
        "SIMILAR_ANTONYM",
        RELATION_SIMILAR,
        DIRECTION_SYMMETRIC,
        (
            (ROLE_SIMILAR_LEFT, frozenset({OBJECT_CONCEPT})),
            (ROLE_SIMILAR_RIGHT, frozenset({OBJECT_CONCEPT})),
        ),
        "SYMMETRIC_NON_TRANSITIVE",
    ),
    "ANTONYM": W06RelationProfile(
        "SIMILAR_ANTONYM",
        RELATION_ANTONYM,
        DIRECTION_SYMMETRIC,
        (
            (ROLE_ANTONYM_LEFT, frozenset({OBJECT_CONCEPT})),
            (ROLE_ANTONYM_RIGHT, frozenset({OBJECT_CONCEPT})),
        ),
        "SYMMETRIC_IRREFLEXIVE_NON_TRANSITIVE",
    ),
    "EVENT_BEFORE": W06RelationProfile(
        "PRECEDES",
        RELATION_EVENT_BEFORE,
        DIRECTION_FORWARD,
        (
            (ROLE_EVENT_BEFORE_SUBJECT, _EVENT_KINDS),
            (ROLE_EVENT_BEFORE_OBJECT, _EVENT_KINDS),
        ),
        "EXPLICIT_PRECEDES_RULES_ONLY",
    ),
    "EVENT_AFTER": W06RelationProfile(
        "PRECEDES",
        RELATION_EVENT_AFTER,
        DIRECTION_FORWARD,
        (
            (ROLE_EVENT_AFTER_SUBJECT, _EVENT_KINDS),
            (ROLE_EVENT_AFTER_OBJECT, _EVENT_KINDS),
        ),
        "EXPLICIT_PRECEDES_RULES_ONLY",
    ),
    "EVENT_SAME": W06RelationProfile(
        "PRECEDES",
        RELATION_EVENT_SAME,
        DIRECTION_FORWARD,
        (
            (ROLE_EVENT_SAME_SUBJECT, _EVENT_KINDS),
            (ROLE_EVENT_SAME_OBJECT, _EVENT_KINDS),
        ),
        "SAME_STATE_ONLY",
    ),
    "EVENT_UNKNOWN": W06RelationProfile(
        "PRECEDES",
        RELATION_EVENT_UNKNOWN,
        DIRECTION_FORWARD,
        (
            (ROLE_EVENT_UNKNOWN_SUBJECT, _EVENT_KINDS),
            (ROLE_EVENT_UNKNOWN_OBJECT, _EVENT_KINDS),
        ),
        "UNKNOWN_STATE_NO_CLOSURE",
    ),
    "CAUSES": W06RelationProfile(
        "CAUSES",
        RELATION_CAUSES,
        DIRECTION_FORWARD,
        (
            (ROLE_CAUSE, _EVENT_KINDS),
            (ROLE_EFFECT, _EVENT_KINDS),
        ),
        "CAUSES_ONLY_EXPLICIT_RULES",
    ),
}


@dataclass(frozen=True)
class W06SourceIsolationReport:
    """记录同一公开来源键下物理互斥的训练和评测来源证据。"""

    pack_key: str
    source_key: str
    train_observation_count: int
    held_out_observation_count: int
    teacher_evidence_count: int
    evaluator_label_count: int
    train_cluster_keys: tuple[tuple[int, ...], ...]
    held_out_cluster_keys: tuple[tuple[int, ...], ...]
    teacher_owner_keys: tuple[tuple[int, ...], ...]
    evaluator_owner_keys: tuple[tuple[int, ...], ...]
    teacher_families: tuple[str, ...]
    evaluator_families: tuple[str, ...]
    teacher_templates: tuple[str, ...]
    evaluator_templates: tuple[str, ...]

    def __post_init__(self) -> None:
        if min(
                self.train_observation_count,
                self.held_out_observation_count,
                self.teacher_evidence_count,
                self.evaluator_label_count) <= 0:
            raise W06SourceSemanticError("W-06 train/evaluator 分账不得为空")
        if set(self.train_cluster_keys) & set(self.held_out_cluster_keys):
            raise W06SourceSemanticError("W-06 train/held-out 来源簇泄漏")
        if set(self.teacher_owner_keys) & set(self.evaluator_owner_keys):
            raise W06SourceSemanticError("W-06 teacher/evaluator owner 泄漏")
        if set(self.teacher_families) & set(self.evaluator_families):
            raise W06SourceSemanticError("W-06 teacher/evaluator seed family 泄漏")
        if set(self.teacher_templates) & set(self.evaluator_templates):
            raise W06SourceSemanticError("W-06 teacher/evaluator template family 泄漏")


def validate_w06_relation_seed(seed: AuthoredRelationSeed) -> W06RelationProfile:
    """按冻结 relation/Role/端点合同拒绝 cue、类型和篇章指代偷渡。"""
    if not isinstance(seed, AuthoredRelationSeed):
        raise TypeError("W-06 semantic firewall 需要 AuthoredRelationSeed")
    profile = W06_RELATION_PROFILES.get(seed.relation_family)
    if profile is None:
        raise W06SourceSemanticError("W-06 relation family 未注册")
    if (seed.relation_kind != profile.relation_kind
            or seed.directionality != profile.directionality):
        raise W06SourceSemanticError("W-06 relation kind 或方向漂移")
    if seed.relation_family == "REFERS" and any(
            item.object_kind == OBJECT_OCCURRENCE for item in seed.endpoints):
        raise W06SourceSemanticError(
            "W-06 REFERS 禁止 occurrence-bound 篇章指代")
    endpoints = {item.endpoint_id: item for item in seed.endpoints}
    if len(endpoints) != len(seed.endpoints):
        raise W06SourceSemanticError("W-06 relation endpoint id 重复")
    seen_roles = set()
    role_type_violations = 0
    for binding in seed.bindings:
        allowed = profile.allowed_for_role(binding.role_kind)
        endpoint = endpoints.get(binding.endpoint_id)
        if endpoint is None:
            raise W06SourceSemanticError("W-06 relation Role 引用了未知端点")
        if endpoint.object_kind not in allowed:
            role_type_violations += 1
        if (endpoint.object_kind not in allowed
                and not (
                    seed.sample_role == "refute"
                    and seed.perturbation_kind == "TYPE_MISMATCH"
                )):
            raise W06SourceSemanticError("W-06 relation Role 端点类型越界")
        seen_roles.add(binding.role_kind)
    if seen_roles != {role for role, _ in profile.role_object_kinds}:
        raise W06SourceSemanticError("W-06 relation Role 集合不完整")
    if (seed.perturbation_kind == "TYPE_MISMATCH"
            and role_type_violations == 0):
        raise W06SourceSemanticError("W-06 TYPE_MISMATCH 负例没有类型破坏")
    if seed.relation_family == "PURE_ALIAS" and len({
            item.object_kind for item in seed.endpoints}) != 1:
        raise W06SourceSemanticError("W-06 PURE_ALIAS 两端必须同型")
    return profile


def _records(pack_root: Path, record_kind: str) -> tuple[object, ...]:
    """按 manifest 读取指定 kind 的全部不可变记录。"""
    manifest = read_artifact_manifest(pack_root / "manifest.json")
    result = []
    for identity in manifest.files:
        if identity.record_kind == record_kind:
            result.extend(read_record_artifact(pack_root, identity))
    return tuple(result)


def audit_w06_authored_source_isolation(
        pack_root: str | Path,
        seeds: tuple[AuthoredRelationSeed, ...],
        ) -> W06SourceIsolationReport:
    """证明原创 train/evaluator 由 cluster、owner、family 和 template 四重隔离。"""
    root = Path(pack_root)
    manifest = read_artifact_manifest(root / "manifest.json")
    if manifest.w_stages != ("W-06",):
        raise W06SourceSemanticError("关系 pack 未绑定 W-06")
    sources = _records(root, RECORD_SOURCE_REF)
    observations = _records(root, RECORD_OBSERVATION)
    teachers = _records(root, RECORD_TEACHER_EVIDENCE)
    evaluators = _records(root, RECORD_EVALUATOR_LABEL)
    if (not all(isinstance(item, SourceRefRecord) for item in sources)
            or not all(isinstance(item, ObservationRecord) for item in observations)
            or not all(isinstance(item, TeacherEvidenceRecord) for item in teachers)
            or not all(isinstance(item, EvaluatorLabelRecord) for item in evaluators)):
        raise W06SourceSemanticError("关系 pack 记录类型漂移")
    source_index = {item.stable_key: item for item in sources}
    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    teacher_seeds = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator_seeds = tuple(item for item in seeds if item.label_owner == "evaluator")
    return W06SourceIsolationReport(
        str(manifest.stable_key.components),
        manifest.source_key,
        len(train),
        len(held_out),
        len(teachers),
        len(evaluators),
        tuple(sorted({
            source_index[item.source_ref_key].source_cluster_key.components
            for item in train
        })),
        tuple(sorted({
            source_index[item.source_ref_key].source_cluster_key.components
            for item in held_out
        })),
        tuple(sorted({item.owner_key.components for item in teachers})),
        tuple(sorted({item.owner_key.components for item in evaluators})),
        tuple(sorted({item.family for item in teacher_seeds})),
        tuple(sorted({item.family for item in evaluator_seeds})),
        tuple(sorted({item.template_family for item in teacher_seeds})),
        tuple(sorted({item.template_family for item in evaluator_seeds})),
    )


__all__ = [
    "W06_GENERATION_HARD_CONJUNCT",
    "W06_RELATION_PROFILES",
    "W06_RELATION_SUBSTAGE_ORDER",
    "W06RelationProfile",
    "W06SourceIsolationReport",
    "W06SourceSemanticError",
    "audit_w06_authored_source_isolation",
    "validate_w06_relation_seed",
]
