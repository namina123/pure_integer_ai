"""W-06 七个子阶段的来源、证据、负例与 consumer 公开注册表。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
    W06_RELATION_SUBSTAGE_ORDER,
)


W06_CONSUMER_KEYS = ("UNDERSTANDING", "REASONING", "GENERATION")
W06_REQUIRED_SAMPLE_ROLES = ("support", "refute", "conflict", "supersede")
W06_AUTHORED_SOURCE = "AUTHORED_CC0_V1"
W06_WIKIDATA_SOURCE = "WIKIDATA_REVISION_V1"
W06_CONCEPTNET_SOURCE = "CONCEPTNET_5_7_0"
W06_WIKTIONARY_SOURCE = "ZHWIKTIONARY_20260701"
W06_TEMPORAL_QUALIFIER_KEYS = frozenset({"P580", "P582", "P585", "P1545"})


class W06RegistryError(RuntimeError):
    """W-06 relation、来源、证据、负例或外部映射未进入冻结注册表。"""


@dataclass(frozen=True)
class W06RelationRegistryEntry:
    """冻结一个 W-06 子阶段的关系族、来源、证据和三向 consumer 合同。"""

    substage_key: str
    relation_families: tuple[str, ...]
    forming_source_keys: tuple[str, ...]
    independent_oracle_source_keys: tuple[str, ...]
    teacher_evidence_kinds: tuple[str, ...]
    required_perturbations: tuple[str, ...]
    consumer_keys: tuple[str, ...] = W06_CONSUMER_KEYS
    teacher_withdrawal_level: int = 0
    minimum_independent_oracle_count: int = 1

    def __post_init__(self) -> None:
        if self.substage_key not in W06_RELATION_SUBSTAGE_ORDER:
            raise W06RegistryError("W-06 registry substage 未冻结")
        if (not self.relation_families
                or len(self.relation_families) != len(set(self.relation_families))):
            raise W06RegistryError("W-06 registry relation family 重复或为空")
        for family in self.relation_families:
            profile = W06_RELATION_PROFILES.get(family)
            if profile is None or profile.substage_key != self.substage_key:
                raise W06RegistryError("W-06 registry relation profile 不一致")
        if (not self.forming_source_keys
                or not self.independent_oracle_source_keys
                or not self.teacher_evidence_kinds
                or not self.required_perturbations):
            raise W06RegistryError("W-06 registry 来源、证据或负例合同为空")
        if self.consumer_keys != W06_CONSUMER_KEYS:
            raise W06RegistryError("W-06 registry 必须绑定 U/R/G 三向 consumer")
        if (self.teacher_withdrawal_level != 0
                or self.minimum_independent_oracle_count != 1):
            raise W06RegistryError("W-06 registry 证据等级合同漂移")


W06_RELATION_REGISTRY = {
    "PURE_ALIAS_REFERS": W06RelationRegistryEntry(
        "PURE_ALIAS_REFERS",
        ("PURE_ALIAS", "REFERS"),
        (W06_AUTHORED_SOURCE, W06_WIKIDATA_SOURCE),
        (W06_WIKTIONARY_SOURCE, W06_CONCEPTNET_SOURCE, W06_AUTHORED_SOURCE),
        # 早期 authored pack 已发布的 evidence kind 与稳定 registry 名称
        # 都是同一关系 owner 的公开命名；在 registry 中显式登记两者，
        # 不把已存在的来源记录改写成另一种语义。
        (
            "ALIAS_REFERS_STABLE_RELATION_LABEL",
            "ALIAS_REFERS_RELATION_LABEL",
        ),
        (
            "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
            "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
        ),
    ),
    "SUBSET_MEMBER": W06RelationRegistryEntry(
        "SUBSET_MEMBER",
        ("SUBSET", "MEMBER"),
        (W06_AUTHORED_SOURCE, W06_WIKIDATA_SOURCE),
        (W06_CONCEPTNET_SOURCE, W06_AUTHORED_SOURCE),
        ("SUBSET_MEMBER_RELATION_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
            "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
            "TYPE_MISMATCH",
        ),
    ),
    "PROPERTY": W06RelationRegistryEntry(
        "PROPERTY",
        ("PROPERTY",),
        (W06_AUTHORED_SOURCE, W06_WIKIDATA_SOURCE),
        (W06_CONCEPTNET_SOURCE, W06_AUTHORED_SOURCE),
        ("PROPERTY_RELATION_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "PSEUDO_RELATION",
            "CONFLICT_SOURCE", "PARSER_REVISION", "ROLE_MISMATCH",
            "VALUE_REPLACEMENT", "INTENSITY_REPLACEMENT",
        ),
    ),
    "MEREOLOGY": W06RelationRegistryEntry(
        "MEREOLOGY",
        ("PART_OF", "HAS_PART"),
        (W06_AUTHORED_SOURCE, W06_WIKIDATA_SOURCE),
        (W06_CONCEPTNET_SOURCE, W06_AUTHORED_SOURCE),
        ("MEREOLOGY_RELATION_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
            "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
            "INVERSE_RELATION", "RELATION_CONFUSION",
        ),
    ),
    "SIMILAR_ANTONYM": W06RelationRegistryEntry(
        "SIMILAR_ANTONYM",
        ("SIMILAR", "ANTONYM"),
        (W06_AUTHORED_SOURCE, W06_CONCEPTNET_SOURCE),
        (W06_WIKTIONARY_SOURCE, W06_AUTHORED_SOURCE),
        ("SEMANTIC_PAIR_RELATION_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "PSEUDO_RELATION",
            "CONFLICT_SOURCE", "PARSER_REVISION", "PAIR_REVERSAL",
            "ALIAS_CONFUSION", "RELATION_CONFUSION",
        ),
    ),
    "PRECEDES": W06RelationRegistryEntry(
        "PRECEDES",
        ("EVENT_BEFORE", "EVENT_AFTER", "EVENT_SAME", "EVENT_UNKNOWN"),
        (W06_AUTHORED_SOURCE, W06_WIKIDATA_SOURCE),
        (W06_AUTHORED_SOURCE,),
        ("PRECEDES_EVENT_TIME_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
            "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
            "UNKNOWN_DIRECTION", "OCCURRENCE_ORDER_CONFUSION",
            "STRUCTURE_ORDER_CONFUSION",
        ),
    ),
    "CAUSES": W06RelationRegistryEntry(
        "CAUSES",
        ("CAUSES",),
        (W06_AUTHORED_SOURCE, W06_CONCEPTNET_SOURCE),
        (W06_WIKIDATA_SOURCE, W06_AUTHORED_SOURCE),
        ("CAUSES_INDEPENDENT_EVIDENCE_LABEL",),
        (
            "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
            "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
            "TEMPORAL_ONLY", "CORRELATION_CONFUSION",
            "CONFOUNDING_CONFUSION", "COUNTERFACTUAL_OVERCLAIM",
        ),
    ),
}

_ENTRY_BY_RELATION = {
    family: entry
    for entry in W06_RELATION_REGISTRY.values()
    for family in entry.relation_families
}

W06_WIKIDATA_PROPERTY_RELATIONS = {
    "P31": frozenset({"MEMBER"}),
    "P279": frozenset({"SUBSET"}),
    "P361": frozenset({"PART_OF"}),
    "P527": frozenset({"HAS_PART"}),
    "P155": frozenset({"EVENT_BEFORE", "EVENT_AFTER"}),
    "P156": frozenset({"EVENT_BEFORE", "EVENT_AFTER"}),
    "P17": frozenset({"PROPERTY"}),
    "P571": frozenset({"PROPERTY"}),
    "P2048": frozenset({"PROPERTY"}),
    "P828": frozenset({"CAUSES"}),
    "P1542": frozenset({"CAUSES"}),
}
W06_CONCEPTNET_RELATION_FAMILIES = {
    "Synonym": frozenset({"PURE_ALIAS", "SIMILAR"}),
    "Antonym": frozenset({"ANTONYM"}),
    "IsA": frozenset({"SUBSET", "MEMBER"}),
    "PartOf": frozenset({"PART_OF"}),
    "HasProperty": frozenset({"PROPERTY"}),
    "Causes": frozenset({"CAUSES"}),
}


@dataclass(frozen=True)
class W06RegistryAuditReport:
    """汇总 train payload 中七子阶段的注册表覆盖，不写入学习状态。"""

    observation_count: int
    teacher_evidence_count: int
    substage_counts: tuple[tuple[str, int], ...]
    relation_counts: tuple[tuple[str, int], ...]
    source_keys: tuple[str, ...]


def _entry_for_relation(relation_family: str) -> W06RelationRegistryEntry:
    """返回 relation family 的唯一注册项，未知 relation 一律拒绝。"""
    entry = _ENTRY_BY_RELATION.get(relation_family)
    if entry is None:
        raise W06RegistryError("W-06 relation family 未注册")
    return entry


def validate_w06_external_relation(
        source_key: str,
        relation_family: str,
        external_relation_key: str,
        *,
        qualifier_keys: tuple[str, ...] = (),
        ) -> W06RelationRegistryEntry:
    """验证 Wikidata/ConceptNet 映射，并对外部 PRECEDES 强制时间 qualifier。"""
    entry = _entry_for_relation(relation_family)
    if source_key not in entry.forming_source_keys:
        raise W06RegistryError("W-06 relation 来源未获 forming 授权")
    if source_key == W06_WIKIDATA_SOURCE:
        allowed = W06_WIKIDATA_PROPERTY_RELATIONS.get(external_relation_key)
        if allowed is None or relation_family not in allowed:
            raise W06RegistryError("W-06 Wikidata property 未注册或关系错配")
        if entry.substage_key == "PRECEDES":
            if (not qualifier_keys
                    or any(item not in W06_TEMPORAL_QUALIFIER_KEYS
                           for item in qualifier_keys)):
                raise W06RegistryError("W-06 PRECEDES 缺少已注册时间 qualifier")
    elif source_key == W06_CONCEPTNET_SOURCE:
        allowed = W06_CONCEPTNET_RELATION_FAMILIES.get(external_relation_key)
        if allowed is None or relation_family not in allowed:
            raise W06RegistryError("W-06 ConceptNet relation 未注册或关系错配")
        if qualifier_keys:
            raise W06RegistryError("W-06 ConceptNet relation 不接受 Wikidata qualifier")
    else:
        raise W06RegistryError("W-06 外部 relation validator 不接受该来源")
    return entry


def _validate_training_record(
        source: SourceRefRecord,
        observation: ObservationRecord,
        teacher: TeacherEvidenceRecord,
        ) -> tuple[W06RelationRegistryEntry, str]:
    """逐条验证 W-06 train Observation 与 teacher Evidence 的来源和 typed 合同。"""
    if (observation.w_stage != "W-06"
            or observation.source_ref_key != source.stable_key
            or teacher.source_ref_key != source.stable_key
            or teacher.observation_key != observation.stable_key):
        raise W06RegistryError("W-06 registry record 引用未闭合")
    payload = observation.typed_payload.to_value()
    relation_family = payload.get("relation_family")
    if not isinstance(relation_family, str):
        raise W06RegistryError("W-06 typed payload 缺 relation family")
    entry = _entry_for_relation(relation_family)
    if (observation.substage != entry.substage_key
            or source.source_key not in entry.forming_source_keys
            or observation.payload_kind != "TypedRelationQuery"
            or observation.epistemic_role != "forming"
            or observation.sample_role not in W06_REQUIRED_SAMPLE_ROLES
            or observation.perturbation_kind not in entry.required_perturbations):
        raise W06RegistryError("W-06 train Observation 未匹配 registry")
    if (teacher.evidence_kind not in entry.teacher_evidence_kinds
            or teacher.visible_from_stage != "W-06"
            or teacher.withdrawal_level != entry.teacher_withdrawal_level):
        raise W06RegistryError("W-06 teacher Evidence 等级或种类未注册")
    return entry, relation_family


def audit_w06_registry_payload(payload: W06TrainingPayload) -> W06RegistryAuditReport:
    """审核七子阶段 train 覆盖，要求每条 W-06 observation 恰有一条 Evidence。"""
    if not isinstance(payload, W06TrainingPayload):
        raise TypeError("W-06 registry audit 需要 W06TrainingPayload")
    sources = {item.stable_key: item for item in payload.source_refs}
    observations = {
        item.stable_key: item for item in payload.observations
        if item.w_stage == "W-06"
    }
    teachers: dict[object, list[TeacherEvidenceRecord]] = defaultdict(list)
    for item in payload.teacher_evidence:
        if item.observation_key in observations:
            teachers[item.observation_key].append(item)
    if not observations or set(teachers) != set(observations):
        raise W06RegistryError("W-06 registry observation/Evidence inventory 不闭合")

    substage_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    source_keys = set()
    roles: dict[str, set[str]] = defaultdict(set)
    perturbations: dict[str, set[str]] = defaultdict(set)
    for key, observation in observations.items():
        evidence = teachers[key]
        if len(evidence) != 1:
            raise W06RegistryError("W-06 observation 必须恰有一条 teacher Evidence")
        source = sources.get(observation.source_ref_key)
        if source is None:
            raise W06RegistryError("W-06 registry 缺 SourceRef")
        entry, relation_family = _validate_training_record(
            source, observation, evidence[0])
        substage_counts[entry.substage_key] += 1
        relation_counts[relation_family] += 1
        source_keys.add(source.source_key)
        roles[entry.substage_key].add(observation.sample_role)
        perturbations[entry.substage_key].add(observation.perturbation_kind)

    if set(substage_counts) != set(W06_RELATION_SUBSTAGE_ORDER):
        raise W06RegistryError("W-06 registry 子阶段覆盖漂移")
    for substage, entry in W06_RELATION_REGISTRY.items():
        if (roles[substage] != set(W06_REQUIRED_SAMPLE_ROLES)
                or not set(entry.relation_families).issubset(relation_counts)
                or not perturbations[substage].issubset(
                    set(entry.required_perturbations))):
            raise W06RegistryError("W-06 registry 关系、负例或 lifecycle 覆盖不完整")
    return W06RegistryAuditReport(
        len(observations),
        sum(len(items) for items in teachers.values()),
        tuple((item, substage_counts[item])
              for item in W06_RELATION_SUBSTAGE_ORDER),
        tuple(sorted(relation_counts.items())),
        tuple(sorted(source_keys)),
    )


__all__ = [
    "W06_CONCEPTNET_RELATION_FAMILIES",
    "W06_CONSUMER_KEYS",
    "W06_RELATION_REGISTRY",
    "W06_REQUIRED_SAMPLE_ROLES",
    "W06_TEMPORAL_QUALIFIER_KEYS",
    "W06_WIKIDATA_PROPERTY_RELATIONS",
    "W06RegistryAuditReport",
    "W06RegistryError",
    "W06RelationRegistryEntry",
    "audit_w06_registry_payload",
    "validate_w06_external_relation",
]
