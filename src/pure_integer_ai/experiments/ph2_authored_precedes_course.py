"""D-02C.6 PRECEDES 事件时序 typed 极小 pack。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    REQUEST_EVENT_TIME_VERIFICATION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_EVENT_AFTER_OBJECT,
    ROLE_EVENT_AFTER_SUBJECT,
    ROLE_EVENT_BEFORE_OBJECT,
    ROLE_EVENT_BEFORE_SUBJECT,
    ROLE_EVENT_SAME_OBJECT,
    ROLE_EVENT_SAME_SUBJECT,
    ROLE_EVENT_UNKNOWN_OBJECT,
    ROLE_EVENT_UNKNOWN_SUBJECT,
    SCHEMA_EVENT_AFTER,
    SCHEMA_EVENT_BEFORE,
    SCHEMA_EVENT_SAME,
    SCHEMA_EVENT_UNKNOWN,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--precedes-v1"
STAGE = "W-06"
SUBSTAGE = "PRECEDES"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "UNKNOWN_DIRECTION",
    "OCCURRENCE_ORDER_CONFUSION",
    "STRUCTURE_ORDER_CONFUSION",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "EVENT_BEFORE": (
        RELATION_EVENT_BEFORE,
        SCHEMA_EVENT_BEFORE,
        ROLE_EVENT_BEFORE_SUBJECT,
        ROLE_EVENT_BEFORE_OBJECT,
        EVENT_TIME_BEFORE,
    ),
    "EVENT_AFTER": (
        RELATION_EVENT_AFTER,
        SCHEMA_EVENT_AFTER,
        ROLE_EVENT_AFTER_SUBJECT,
        ROLE_EVENT_AFTER_OBJECT,
        EVENT_TIME_AFTER,
    ),
    "EVENT_SAME": (
        RELATION_EVENT_SAME,
        SCHEMA_EVENT_SAME,
        ROLE_EVENT_SAME_SUBJECT,
        ROLE_EVENT_SAME_OBJECT,
        EVENT_TIME_SAME,
    ),
    "EVENT_UNKNOWN": (
        RELATION_EVENT_UNKNOWN,
        SCHEMA_EVENT_UNKNOWN,
        ROLE_EVENT_UNKNOWN_SUBJECT,
        ROLE_EVENT_UNKNOWN_OBJECT,
        EVENT_TIME_DIRECTION_UNKNOWN,
    ),
}
_ALLOWED_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})
_SPEC = AuthoredCourseSpec(
    SOURCE_KEY,
    LICENSE_ID,
    1,
    1,
    1,
    1,
    1,
    PACK_NAME,
    STAGE,
    SUBSTAGE,
    "authored-precedes-seed-v1",
    "urn:pure-integer-ai:ph2:authored-precedes-v1",
    "Pure Integer AI PH2 authored typed event time seed",
    "PRECEDES_EVENT_TIME_LABEL",
    "precedes",
    100,
)


def _identity_key(identity) -> list[int]:
    """把一等对象或 scope 身份投影为规范严格整数列表。"""
    return list(identity.stable_key())


def _protocol_payload(seed: AuthoredRelationSeed, value: dict) -> dict:
    """输出 relation 方向 resolver、独立 dimension 和严格时序层边界。"""
    profile = _PROFILE[seed.relation_family]
    relation = authored_relation_identity(profile[0])
    resolved = ResolvedEventTimeRelation(
        relation,
        profile[4],
        (20660, profile[0], profile[4]),
    )
    proposition = ObjectIdentity.from_stable_key(tuple(
        value["candidate_definition"]["proposition_key"]))
    scope = document_scope(semantic_source(proposition))
    return {
        "causes_effect": 0,
        "detail_key": list(resolved.detail_key),
        "dimension_key": [20661, 1],
        "direction": resolved.direction,
        "hypothesis_kind_key": [20662, 1],
        "object_role_key": _identity_key(
            authored_relation_role_identity(profile[3])),
        "occurrence_order_consumed": 0,
        "relation_key": _identity_key(resolved.relation),
        "scope_key": _identity_key(scope),
        "structure_order_consumed": 0,
        "subject_role_key": _identity_key(
            authored_relation_role_identity(profile[2])),
        "verifier_key": [20661, 2],
    }


def _compile_precedes_seed(seed: AuthoredRelationSeed) -> AuthoredCompiledSeed:
    """把 event-time resolver 和独立 verification protocol 附加到 payload。"""
    base = compile_relation_seed(seed)
    value = base.observation_payload.to_value()
    protocol = _protocol_payload(seed, value)
    value["event_time_protocol"] = protocol
    payload = CanonicalJsonObject.from_value(value)
    return replace(
        base,
        observation_payload=payload,
        dedup_parts=(seed.surface, value),
        content_parts=(
            seed.surface,
            value["candidate_definition"],
            protocol,
        ),
        shape_parts=(
            *base.shape_parts,
            "event_time_protocol_v1",
            protocol["direction"],
        ),
    )


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对 event-time relation、Role、端点类型和 verification request。"""
    profile = _PROFILE.get(seed.relation_family)
    if profile is None:
        raise AuthoredRelationCourseError("event-time relation family 未注册")
    relation, schema, subject_role, object_role = profile[:4]
    if (seed.relation_kind != relation
            or seed.schema_kind != schema
            or seed.directionality != DIRECTION_FORWARD):
        raise AuthoredRelationCourseError("event-time relation profile 坐标漂移")
    if len(seed.endpoints) != 2 or len(seed.bindings) != 2:
        raise AuthoredRelationCourseError("event-time 必须恰有两个 endpoint")
    binding_by_role = {item.role_kind: item for item in seed.bindings}
    if set(binding_by_role) != {subject_role, object_role}:
        raise AuthoredRelationCourseError(
            "event-time subject/object Role profile 漂移")
    if any(frozenset(item.allowed_object_kinds) != _ALLOWED_ENDPOINT_KINDS
           for item in seed.bindings):
        raise AuthoredRelationCourseError("event-time Role slot 类型漂移")
    if any(item.object_kind not in _ALLOWED_ENDPOINT_KINDS
           for item in seed.endpoints):
        raise AuthoredRelationCourseError(
            "event-time endpoint 必须是 Event 或 Proposition")
    request = seed.consumer_request
    if request.request_kind != REQUEST_EVENT_TIME_VERIFICATION:
        raise AuthoredRelationCourseError(
            "event-time consumer request kind 漂移")
    declared_subject = binding_by_role[subject_role].endpoint_id
    declared_object = binding_by_role[object_role].endpoint_id
    if seed.perturbation_kind == "DIRECTION_REVERSAL":
        if (request.origin_endpoint_id != declared_object
                or request.object_endpoint_id != declared_subject):
            raise AuthoredRelationCourseError(
                "event-time direction reversal 未只交换 subject/object")
    elif (request.origin_endpoint_id != declared_subject
          or request.object_endpoint_id != declared_object):
        raise AuthoredRelationCourseError(
            "event-time query subject/object 锚漂移")


def read_authored_precedes_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 PRECEDES JSONL，并核对方向覆盖、owner 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("precedes sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("precedes sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"precedes 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"precedes 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"precedes 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("precedes seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("precedes logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError("precedes supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "precedes supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError(
                "precedes supersede 必须是 parser revision")
    teacher_families = {
        item.family for item in seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in seeds if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in seeds if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in seeds
        if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredRelationCourseError(
            "precedes teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError("precedes 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredRelationCourseError("precedes 缺少必需反向破坏")
    if {item.relation_family for item in seeds} != set(_PROFILE):
        raise AuthoredRelationCourseError(
            "event before/after/same/unknown 必须全部覆盖")
    return tuple(seeds)


def compile_authored_precedes_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.6 typed PRECEDES/event-time 极小 pack。"""
    seeds = read_authored_precedes_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(_compile_precedes_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError("precedes pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_precedes_course",
    "read_authored_precedes_seeds",
]
