"""把 D-02D modal seed 编译为 unary bound operator 和 resolver payload。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvidenceState,
    ModalResolution,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_GENERATION,
    SCOPE_QUERY,
    ScopeIdentity,
    generation_scope,
    query_scope,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    compile_logic_seed,
)
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    AuthoredModalSeed,
    RESOLVER_RESOLVED,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)


def _resolution_scope(seed, input_scope: ScopeIdentity) -> ScopeIdentity:
    """按显式 scope kind 构造同 source 的 modal 输出边界。"""
    if seed.scope_kind == SCOPE_DOCUMENT:
        return input_scope
    if seed.scope_kind == SCOPE_QUERY:
        return query_scope(seed.scope_local_id, parent=input_scope)
    if seed.scope_kind == SCOPE_GENERATION:
        return generation_scope(seed.scope_local_id, parent=input_scope)
    raise ValueError("modal resolution scope kind 尚无 compiler handler")


def compile_modal_seed(seed: AuthoredModalSeed) -> AuthoredCompiledSeed:
    """生成 modal definition、child、候选、scope 和 resolver 执行请求。"""
    if not isinstance(seed, AuthoredModalSeed):
        raise TypeError("compile_modal_seed 需要 AuthoredModalSeed")
    base = compile_logic_seed(seed.logic)
    payload = base.observation_payload.to_value()
    source = SourceRef.from_stable_key(tuple(
        payload["candidate_spec"]["forming_source_keys"][0]))
    input_scope = ScopeIdentity.from_stable_key(tuple(
        payload["consumer_request"]["scope_key"]))
    resolver = seed.resolver
    output_scope = None
    if resolver.status == RESOLVER_RESOLVED:
        output_scope = _resolution_scope(resolver, input_scope)
        resolution = ModalResolution(
            LogicEvidenceState(
                bool(resolver.resolution_support),
                bool(resolver.resolution_refute),
            ),
            source,
            output_scope,
            resolver.evidence_ids,
        )
        resolution_state = {
            "refute": int(resolution.state.refute),
            "support": int(resolution.state.support),
        }
        evidence_ids = list(resolution.evidence_ids)
    else:
        resolution_state = {"refute": 0, "support": 0}
        evidence_ids = []
    payload["child_evaluated_without_resolver"] = 0
    payload["child_state_is_modal_result"] = 0
    payload["consumer_request"]["budget"]["max_resolver_calls"] = (
        seed.max_resolver_calls)
    payload["modal_resolution_plan"] = {
        "evidence_ids": evidence_ids,
        "input_scope_key": list(input_scope.stable_key()),
        "output_scope_key": (
            None if output_scope is None else list(output_scope.stable_key())),
        "resolution_state": resolution_state,
        "source_key": list(source.stable_key()),
        "source_unchanged": 1,
        "status": resolver.status,
    }
    payload["query_kind"] = "typed_modal_candidate"
    payload["resolver_required"] = 1
    typed_payload = CanonicalJsonObject.from_value(payload)
    return AuthoredCompiledSeed(
        base.seed_id,
        base.family,
        base.template_family,
        base.label_owner,
        base.split,
        base.sample_role,
        "ModalExecutionQuery",
        typed_payload,
        base.expected_state,
        base.expected_payload,
        base.perturbation_kind,
        base.supersedes_seed_id,
        base.logical_order,
        (base.seed_id, payload),
        (
            payload["surface"],
            payload["bound_root"],
            payload["consumer_request"],
            payload["modal_resolution_plan"],
        ),
        (
            "typed_modal_v1",
            seed.logic.operator_kind,
            resolver.status,
            resolver.scope_kind,
            seed.logic.perturbation_kind,
        ),
    )


__all__ = ["compile_modal_seed"]
