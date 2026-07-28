"""D-02C.7 CAUSES 独立证据与时间约束 typed 极小 pack。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalEndpointProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalVerificationProtocol,
)
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
    authored_relation_rule_identity,
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_CAUSES,
    REQUEST_CAUSAL_VERIFICATION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_CAUSE,
    ROLE_EFFECT,
    SCHEMA_CAUSES,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.verification_orchestration import ProtocolKey


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--causes-v1"
STAGE = "W-06"
SUBSTAGE = "CAUSES"
CAUSAL_EXECUTION_RULE_KIND = 5
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "TEMPORAL_ONLY",
    "CORRELATION_CONFUSION",
    "CONFOUNDING_CONFUSION",
    "COUNTERFACTUAL_OVERCLAIM",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
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
    "authored-causes-seed-v1",
    "urn:pure-integer-ai:ph2:authored-causes-v1",
    "Pure Integer AI PH2 authored typed causal seed",
    "CAUSES_INDEPENDENT_EVIDENCE_LABEL",
    "causes",
    100,
)


def _identity_key(identity) -> list[int]:
    """把一等对象、scope 或开放协议键投影为严格整数列表。"""
    return list(identity.stable_key())


def _protocol_payload(seed: AuthoredRelationSeed, value: dict) -> dict:
    """输出 cause/effect、独立 witness、时序必要条件和反事实边界。"""
    endpoint_protocol = CausalEndpointProtocol(
        authored_relation_identity(RELATION_CAUSES),
        authored_relation_role_identity(ROLE_CAUSE),
        authored_relation_role_identity(ROLE_EFFECT),
        authored_relation_rule_identity(CAUSAL_EXECUTION_RULE_KIND),
    )
    verification = CausalVerificationProtocol(
        ProtocolKey((20670, 1)),
        ProtocolKey((20670, 2)),
        ProtocolKey((20670, 3)),
    )
    proposition = ObjectIdentity.from_stable_key(tuple(
        value["candidate_definition"]["proposition_key"]))
    scope = document_scope(semantic_source(proposition))
    return {
        "causal_implies_event_time_fact": 0,
        "counterfactual_verdict_claimed": 0,
        "dimension_key": _identity_key(verification.dimension),
        "effect_role_key": _identity_key(endpoint_protocol.effect_role),
        "evidence_target_kind_key": _identity_key(
            verification.evidence_target_kind),
        "execution_instruction_key": _identity_key(
            endpoint_protocol.execution_instruction),
        "forming_source_reusable_as_witness": 0,
        "independent_witness_required": 1,
        "occurrence_order_consumed": 0,
        "precedence_implies_causation": 0,
        "relation_key": _identity_key(endpoint_protocol.relation),
        "scope_key": _identity_key(scope),
        "structure_order_consumed": 0,
        "temporal_support_sufficient": 0,
        "verifier_key": _identity_key(verification.verifier),
        "cause_role_key": _identity_key(endpoint_protocol.cause_role),
    }


def _compile_causes_seed(seed: AuthoredRelationSeed) -> AuthoredCompiledSeed:
    """把 causal endpoint 与独立核验协议附加到 typed payload。"""
    base = compile_relation_seed(seed)
    value = base.observation_payload.to_value()
    protocol = _protocol_payload(seed, value)
    value["causal_protocol"] = protocol
    payload = CanonicalJsonObject.from_value(value)
    return replace(
        base,
        observation_payload=payload,
        dedup_parts=(seed.surface, value),
        content_parts=(
            seed.surface,
            value["candidate_definition"],
            value["consumer_request"],
            protocol,
        ),
        shape_parts=(
            *base.shape_parts,
            "causal_independent_evidence_v1",
            seed.perturbation_kind,
        ),
    )


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对 CAUSES profile、Role、端点类型和独立 causal consumer。"""
    if (seed.relation_family != "CAUSES"
            or seed.relation_kind != RELATION_CAUSES
            or seed.schema_kind != SCHEMA_CAUSES
            or seed.directionality != DIRECTION_FORWARD):
        raise AuthoredRelationCourseError("causal relation profile 坐标漂移")
    if len(seed.endpoints) != 2 or len(seed.bindings) != 2:
        raise AuthoredRelationCourseError("causal 必须恰有两个 endpoint")
    binding_by_role = {item.role_kind: item for item in seed.bindings}
    if set(binding_by_role) != {ROLE_CAUSE, ROLE_EFFECT}:
        raise AuthoredRelationCourseError("causal cause/effect Role profile 漂移")
    if any(frozenset(item.allowed_object_kinds) != _ALLOWED_ENDPOINT_KINDS
           for item in seed.bindings):
        raise AuthoredRelationCourseError("causal Role slot 类型漂移")
    if any(item.object_kind not in _ALLOWED_ENDPOINT_KINDS
           for item in seed.endpoints):
        raise AuthoredRelationCourseError(
            "causal endpoint 必须是 Event 或 Proposition")
    request = seed.consumer_request
    if request.request_kind != REQUEST_CAUSAL_VERIFICATION:
        raise AuthoredRelationCourseError("causal consumer request kind 漂移")
    declared_cause = binding_by_role[ROLE_CAUSE].endpoint_id
    declared_effect = binding_by_role[ROLE_EFFECT].endpoint_id
    if seed.perturbation_kind == "DIRECTION_REVERSAL":
        if (request.origin_endpoint_id != declared_effect
                or request.effect_endpoint_id != declared_cause):
            raise AuthoredRelationCourseError(
                "causal direction reversal 未只交换 cause/effect")
    elif (request.origin_endpoint_id != declared_cause
          or request.effect_endpoint_id != declared_effect):
        raise AuthoredRelationCourseError("causal query cause/effect 锚漂移")


def read_authored_causes_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 CAUSES JSONL，并核对独立 owner、扰动和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("causes sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("causes sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"causes 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"causes 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"causes 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("causes seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("causes logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError("causes supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "causes supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError(
                "causes supersede 必须是 parser revision")
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
            "causes teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError("causes 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredRelationCourseError("causes 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_causes_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.7 typed CAUSES 极小 pack。"""
    seeds = read_authored_causes_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(_compile_causes_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError("causes pack 发布失败") from error


__all__ = [
    "CAUSAL_EXECUTION_RULE_KIND",
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_causes_course",
    "read_authored_causes_seeds",
]
